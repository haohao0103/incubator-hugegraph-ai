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
"""Explicit KG schema definition (generalized from neo4j-graphrag-python's GraphSchema).

A declarative schema tells the LLM extractor exactly what it is allowed to
produce, and lets the pipeline validate extraction output before it hits the
graph:

  * :class:`PropertyType`   -- a property name with a HugeGraph data type.
  * :class:`NodeType`       -- a vertex label with its properties; the
    ``additional_properties`` toggle controls whether unknown properties in
    extracted entities are kept (lenient) or dropped (strict).
  * :class:`RelationshipType` -- an edge label with source/target labels.
  * :class:`GraphSchema`    -- the full schema; ``to_dict()`` emits the dict
    consumed by ``SchemaManager.ensure_schema`` (propertykeys/vertexlabels/
    edgelabels), and ``validate_extraction()`` splits extracted
    entities/relationships into valid and invalid buckets with reasons.

This fills the gap between hugegraph-ai's fixed ``KG_SCHEMA`` (library-table
graphs) and the lenient LightRAG-style free extraction: a production-grade
"explicit schema + extraction validation" tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.hugegraph_op.schema_manager import (
    ID_STRATEGY_CUSTOMIZE_STRING,
)
from hugegraph_llm.utils.log import log

# HugeGraph property data types (subset of PropertyDataType that makes sense
# for LLM-extracted values).
SUPPORTED_DATA_TYPES = ("TEXT", "INT", "LONG", "DOUBLE", "BOOLEAN", "DATE", "UUID")

_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _validate_label(label: str, kind: str) -> None:
    """Validate a HugeGraph label name (letters first, alnum + underscore)."""
    if not label or not re.match(r"[A-Za-z_][A-Za-z0-9_]*$", label):
        raise ValueError(f"Invalid {kind} label {label!r}: must start with a "
                         "letter and contain only alphanumerics/underscores")


def _sanitize_value(data_type: str, value: Any) -> Any:
    """Coerce/validate a value against a HugeGraph data type.

    Returns the coerced value when convertible, else raises ``ValueError``.
    """
    if data_type == "TEXT":
        if isinstance(value, str):
            return value
        return str(value)
    if data_type in ("INT", "LONG"):
        if isinstance(value, bool):
            raise ValueError(f"expected int, got bool: {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            return int(value)
        raise ValueError(f"expected int for {data_type}, got {value!r}")
    if data_type == "DOUBLE":
        if isinstance(value, bool):
            raise ValueError(f"expected number, got bool: {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"expected number, got {value!r}") from exc
        raise ValueError(f"expected number, got {value!r}")
    if data_type == "BOOLEAN":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"expected bool, got {value!r}")
    if data_type == "DATE":
        if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            raise ValueError(f"expected yyyy-MM-dd date, got {value!r}")
        return value
    if data_type == "UUID":
        return str(value)
    raise ValueError(f"unsupported data type {data_type!r}")


@dataclass
class PropertyType:
    """A property on a node or relationship."""

    name: str
    data_type: str = "TEXT"
    cardinality: str = "SINGLE"  # SINGLE | LIST | SET
    description: str = ""
    required: bool = False

    def __post_init__(self) -> None:
        if self.data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"unsupported property type {self.data_type!r}; "
                f"supported: {SUPPORTED_DATA_TYPES}"
            )
        if self.cardinality not in ("SINGLE", "LIST", "SET"):
            raise ValueError(f"unsupported cardinality {self.cardinality!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "cardinality": self.cardinality,
            "description": self.description,
        }


@dataclass
class NodeType:
    """A possible vertex label with its properties."""

    label: str
    description: str = ""
    properties: List[PropertyType] = field(default_factory=list)
    additional_properties: bool = True
    id_strategy: str = ID_STRATEGY_CUSTOMIZE_STRING
    primary_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_label(self.label, "vertex")
        names = [p.name for p in self.properties]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate properties on node {self.label!r}")

    def property_names(self) -> Set[str]:
        return {p.name for p in self.properties}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.label,
            "properties": [p.name for p in self.properties],
            "id_strategy": self.id_strategy,
            "primary_keys": list(self.primary_keys),
            "nullable_keys": [p.name for p in self.properties if not p.required],
        }


@dataclass
class RelationshipType:
    """A possible edge label between two vertex labels."""

    label: str
    source_label: str
    target_label: str
    description: str = ""
    properties: List[PropertyType] = field(default_factory=list)
    additional_properties: bool = True

    def __post_init__(self) -> None:
        _validate_label(self.label, "edge")
        _validate_label(self.source_label, "source vertex")
        _validate_label(self.target_label, "target vertex")

    def property_names(self) -> Set[str]:
        return {p.name for p in self.properties}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.label,
            "source_label": self.source_label,
            "target_label": self.target_label,
            "properties": [p.name for p in self.properties],
            "nullable_keys": [p.name for p in self.properties if not p.required],
        }


@dataclass
class ExtractionValidation:
    """Result of validating extraction output against a GraphSchema."""

    valid_entities: List[Dict[str, Any]] = field(default_factory=list)
    invalid_entities: List[Dict[str, Any]] = field(default_factory=list)
    valid_relationships: List[Dict[str, Any]] = field(default_factory=list)
    invalid_relationships: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            len(self.valid_entities) + len(self.invalid_entities)
            + len(self.valid_relationships) + len(self.invalid_relationships)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid_entities": self.valid_entities,
            "invalid_entities": self.invalid_entities,
            "valid_relationships": self.valid_relationships,
            "invalid_relationships": self.invalid_relationships,
        }


class GraphSchema:
    """The full explicit schema: nodes + relationships."""

    def __init__(
        self,
        nodes: Optional[List[NodeType]] = None,
        relationships: Optional[List[RelationshipType]] = None,
    ) -> None:
        self.nodes: List[NodeType] = list(nodes or [])
        self.relationships: List[RelationshipType] = list(relationships or [])
        # validate unique labels
        node_labels = [n.label for n in self.nodes]
        rel_labels = [r.label for r in self.relationships]
        if len(node_labels) != len(set(node_labels)):
            raise ValueError(f"duplicate node labels: {node_labels}")
        if len(rel_labels) != len(set(rel_labels)):
            raise ValueError(f"duplicate relationship labels: {rel_labels}")

    def node(self, label: str) -> Optional[NodeType]:
        return next((n for n in self.nodes if n.label == label), None)

    def relationship(self, label: str) -> Optional[RelationshipType]:
        return next((r for r in self.relationships if r.label == label), None)

    def to_dict(self) -> Dict[str, Any]:
        """Emit the schema dict consumed by ``SchemaManager.ensure_schema``."""
        propertykeys: List[Dict[str, Any]] = []
        seen_props: Set[Tuple[str, str]] = set()
        for node in self.nodes:
            for prop in node.properties:
                key = (prop.name, prop.cardinality)
                if key in seen_props:
                    continue
                seen_props.add(key)
                propertykeys.append({
                    "name": prop.name,
                    "data_type": prop.data_type,
                    "cardinality": prop.cardinality,
                })
        for rel in self.relationships:
            for prop in rel.properties:
                key = (prop.name, prop.cardinality)
                if key in seen_props:
                    continue
                seen_props.add(key)
                propertykeys.append({
                    "name": prop.name,
                    "data_type": prop.data_type,
                    "cardinality": prop.cardinality,
                })
        return {
            "propertykeys": propertykeys,
            "vertexlabels": [n.to_dict() for n in self.nodes],
            "edgelabels": [r.to_dict() for r in self.relationships],
        }

    def validate_extraction(
        self,
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> ExtractionValidation:
        """Split extracted items into valid / invalid buckets with reasons.

        Entities: ``{"label": str, "properties": {k: v}}`` (an ``id`` key is
        allowed and preserved). Relationships: ``{"label", "source", "target",
        "properties": {k: v}}``. Invalid items keep an extra ``_reason`` key.
        """
        result = ExtractionValidation()
        for entity in entities or []:
            reason = self._validate_node(entity)
            bucket = result.valid_entities if reason is None else result.invalid_entities
            item = dict(entity)
            if reason is not None:
                item["_reason"] = reason
            bucket.append(item)
        for rel in relationships or []:
            reason = self._validate_relationship(rel)
            bucket = result.valid_relationships if reason is None else result.invalid_relationships
            item = dict(rel)
            if reason is not None:
                item["_reason"] = reason
            bucket.append(item)
        return result

    # -- internals ------------------------------------------------------------

    def _validate_node(self, entity: Dict[str, Any]) -> Optional[str]:
        label = entity.get("label")
        node_type = self.node(label) if isinstance(label, str) else None
        if node_type is None:
            return f"unknown vertex label {label!r}"
        props = entity.get("properties") or {}
        reason = self._validate_properties(node_type, props, entity)
        if reason is not None:
            return reason
        # strict mode: drop unknown properties from the item
        if not node_type.additional_properties:
            known = node_type.property_names()
            entity["properties"] = {k: v for k, v in props.items() if k in known}
        return None

    def _validate_relationship(self, rel: Dict[str, Any]) -> Optional[str]:
        label = rel.get("label")
        rel_type = self.relationship(label) if isinstance(label, str) else None
        if rel_type is None:
            return f"unknown relationship label {label!r}"
        if not rel.get("source") or not rel.get("target"):
            return "missing source/target"
        props = rel.get("properties") or {}
        reason = self._validate_properties(rel_type, props, rel)
        if reason is not None:
            return reason
        if not rel_type.additional_properties:
            known = rel_type.property_names()
            rel["properties"] = {k: v for k, v in props.items() if k in known}
        return None

    def _validate_properties(
        self, type_, props: Dict[str, Any], item: Dict[str, Any]
    ) -> Optional[str]:
        known = type_.property_names()
        for prop in type_.properties:
            if prop.name not in props:
                if prop.required:
                    return f"missing required property {prop.name!r}"
                continue
            try:
                self._coerce_property(prop, props)
            except ValueError as exc:
                return str(exc)
        if not type_.additional_properties:
            for key in props:
                if key in known or key == "id":
                    continue
                log.debug("dropping unknown property %r", key)
        return None

    def _coerce_property(self, prop: PropertyType, props: Dict[str, Any]) -> None:
        """Coerce/validate one property value in place (raises on mismatch)."""
        value = props[prop.name]
        if prop.cardinality in ("LIST", "SET"):
            if not isinstance(value, list):
                raise ValueError(
                    f"property {prop.name!r} expects a list, got {value!r}"
                )
            props[prop.name] = [
                _sanitize_value(prop.data_type, item) for item in value
            ]
        else:
            props[prop.name] = _sanitize_value(prop.data_type, value)
