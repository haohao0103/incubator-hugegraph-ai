# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Schema evolution: turn discarded extraction into versioned schema candidates.

During schema-guided extraction, any entity/relation whose label is not in the
current schema is dropped.  This module collects those discarded items into a
:class:`SchemaEvolutionDraft` (a list of candidate vertex/edge labels with
sample properties), persists drafts in a versioned store for human review, and
applies an accepted draft back onto the schema.

This closes the "evolution loop" for GraphRAG schema design:

1. run an incremental extraction;
2. collect labels the schema did not cover;
3. review the resulting draft (human-in-the-loop);
4. ``apply_schema_draft`` merges accepted labels into a new schema version.
"""

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log

_MAX_SAMPLE_VALUES = 3


@dataclass
class SchemaEvolutionDraft:
    """A reviewable set of schema additions inferred from discarded items."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    new_vertex_labels: List[Dict[str, Any]] = field(default_factory=list)
    new_edge_labels: List[Dict[str, Any]] = field(default_factory=list)
    total_discarded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaEvolutionDraft":
        return cls(
            created_at=data.get("created_at", ""),
            new_vertex_labels=list(data.get("new_vertex_labels", [])),
            new_edge_labels=list(data.get("new_edge_labels", [])),
            total_discarded=int(data.get("total_discarded", 0)),
        )


def _add_sample(example_properties: Dict[str, List[str]], prop: str, value: Any) -> None:
    """Record a non-empty sample value for a property (capped, deduplicated)."""
    if value is None or value == "":
        return
    text = str(value)
    samples = example_properties.setdefault(prop, [])
    if text not in samples and len(samples) < _MAX_SAMPLE_VALUES:
        samples.append(text)


def collect_schema_candidates(
    schema: Dict[str, Any],
    discarded_items: List[Dict[str, Any]],
) -> SchemaEvolutionDraft:
    """Collect candidate schema additions from discarded extraction items.

    Args:
        schema: Current schema dict (``vertexlabels``/``edgelabels``).
        discarded_items: Raw items whose label was not in the schema, e.g.
            ``{"type": "vertex", "label": "Skill", "properties": {...}}`` or
            ``{"type": "edge", "label": "has_skill", "outVLabel": ..., "inVLabel": ...}``.

    Returns:
        A :class:`SchemaEvolutionDraft` grouping candidates by label with sample
        properties, ready for human review.
    """
    existing_vertex = {vl["name"] for vl in schema.get("vertexlabels", [])}
    existing_edge = {el["name"] for el in schema.get("edgelabels", [])}

    vertex_candidates: Dict[str, Dict[str, Any]] = {}
    edge_candidates: Dict[str, Dict[str, Any]] = {}

    for item in discarded_items:
        item_type = item.get("type")
        label = item.get("label")
        if not label:
            continue
        props = item.get("properties") or {}
        if item_type == "vertex":
            if label in existing_vertex:
                continue
            candidate = vertex_candidates.setdefault(
                label, {"name": label, "sample_count": 0, "example_properties": {}}
            )
            candidate["sample_count"] += 1
            for prop, value in props.items():
                _add_sample(candidate["example_properties"], prop, value)
        elif item_type == "edge":
            if label in existing_edge:
                continue
            candidate = edge_candidates.setdefault(
                label,
                {
                    "name": label,
                    "source_label": item.get("outVLabel") or item.get("source_label", ""),
                    "target_label": item.get("inVLabel") or item.get("target_label", ""),
                    "sample_count": 0,
                    "example_properties": {},
                },
            )
            candidate["sample_count"] += 1
            for prop, value in props.items():
                _add_sample(candidate["example_properties"], prop, value)

    draft = SchemaEvolutionDraft(
        total_discarded=len(discarded_items),
        new_vertex_labels=list(vertex_candidates.values()),
        new_edge_labels=list(edge_candidates.values()),
    )
    log.info(
        "Schema evolution draft: %d new vertex labels, %d new edge labels from %d discarded items",
        len(draft.new_vertex_labels),
        len(draft.new_edge_labels),
        draft.total_discarded,
    )
    return draft


def apply_schema_draft(schema: Dict[str, Any], draft: SchemaEvolutionDraft) -> Dict[str, Any]:
    """Merge an accepted draft into a new schema dict (does not mutate input).

    Heuristics (for human review before commit): vertex labels get their sample
    property names as properties, ``name`` as primary key when present (else the
    first property), and all properties are declared as ``TEXT``/``SINGLE``.
    Edge labels carry the majority ``source_label``/``target_label`` recorded
    during collection.
    """
    new_schema = copy.deepcopy(schema)
    new_schema.setdefault("vertexlabels", [])
    new_schema.setdefault("edgelabels", [])
    new_schema.setdefault("propertykeys", [])

    existing_vertex = {vl["name"] for vl in new_schema["vertexlabels"]}
    existing_edge = {el["name"] for el in new_schema["edgelabels"]}
    existing_props = {pk["name"] for pk in new_schema["propertykeys"]}

    def declare_props(prop_names: List[str]) -> None:
        for prop in prop_names:
            if prop not in existing_props:
                new_schema["propertykeys"].append({"name": prop, "data_type": "TEXT", "cardinality": "SINGLE"})
                existing_props.add(prop)

    for vl in draft.new_vertex_labels:
        if vl["name"] in existing_vertex:
            continue
        props = list(vl.get("example_properties", {}).keys())
        declare_props(props)
        new_schema["vertexlabels"].append(
            {
                "name": vl["name"],
                "properties": props,
                "primary_keys": ["name"] if "name" in props else props[:1],
                "nullable_keys": [],
            }
        )

    for el in draft.new_edge_labels:
        if el["name"] in existing_edge:
            continue
        props = list(el.get("example_properties", {}).keys())
        declare_props(props)
        new_schema["edgelabels"].append(
            {
                "name": el["name"],
                "source_label": el.get("source_label", ""),
                "target_label": el.get("target_label", ""),
                "properties": props,
            }
        )

    return new_schema


class SchemaEvolutionStore:
    """Append-only, JSON-file-backed store of schema evolution drafts.

    Each record is ``{"version": int, "draft": {...}}``; ``record`` appends the
    next version and rewrites the file, so callers can keep a versioned audit
    trail of schema changes.
    """

    def __init__(self, path: str):
        self.path = path

    def _read(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def record(self, draft: SchemaEvolutionDraft) -> Dict[str, Any]:
        records = self._read()
        version = (max((r.get("version", 0) for r in records), default=0)) + 1
        record = {"version": version, "draft": draft.to_dict()}
        records.append(record)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return record

    def list(self) -> List[Dict[str, Any]]:
        return self._read()

    def latest(self) -> Optional[Dict[str, Any]]:
        records = self._read()
        return records[-1] if records else None
