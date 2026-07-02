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

"""AutoSchemaKG: single-document → LLM-inferred schema draft → human review → HugeGraph schema commit.

This module provides a minimal runnable version of automatic schema construction for
HugeGraph-backed GraphRAG.  It is intentionally decoupled from the existing
``SchemaBuilder`` / EDC pipeline so that it can be used as a standalone operator in
index flows or UI demos.

Design goals
------------
* One document in, one schema draft out.
* LLM output is normalized to a HugeGraph-compatible schema JSON.
* A review callback lets humans approve, reject, or modify the draft before commit.
* Commits happen only when explicitly allowed and after review passes.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PropertyKeyDef:
    """Definition of a HugeGraph property key."""
    name: str
    data_type: str = "text"
    cardinality: str = "single"

    def to_dict(self) -> Dict[str, Any]:
        # HugeGraph Commit2Graph expects uppercase data_type / cardinality values.
        return {"name": self.name, "data_type": self.data_type.upper(), "cardinality": self.cardinality.upper()}


@dataclass
class VertexLabelDef:
    """Definition of a HugeGraph vertex label."""
    name: str
    properties: List[str] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    nullable_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "properties": list(self.properties),
            "primary_keys": list(self.primary_keys),
            "nullable_keys": list(self.nullable_keys),
        }


@dataclass
class EdgeLabelDef:
    """Definition of a HugeGraph edge label."""
    name: str
    source_label: str = ""
    target_label: str = ""
    properties: List[str] = field(default_factory=list)
    nullable_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_label": self.source_label,
            "target_label": self.target_label,
            "properties": list(self.properties),
            "nullable_keys": list(self.nullable_keys),
        }


@dataclass
class SchemaDraft:
    """Human-readable + machine-readable schema draft produced by AutoSchemaKG."""
    property_keys: List[PropertyKeyDef] = field(default_factory=list)
    vertex_labels: List[VertexLabelDef] = field(default_factory=list)
    edge_labels: List[EdgeLabelDef] = field(default_factory=list)
    source_document: str = ""
    raw_llm_response: str = ""

    def to_schema_dict(self) -> Dict[str, Any]:
        """Return a schema dict compatible with ``Commit2Graph.init_schema_if_need``."""
        return {
            "propertykeys": [p.to_dict() for p in self.property_keys],
            "vertexlabels": [v.to_dict() for v in self.vertex_labels],
            "edgelabels": [e.to_dict() for e in self.edge_labels],
        }

    def to_human_readable(self) -> str:
        """Render a Markdown summary suitable for UI review."""
        lines: List[str] = [
            "## AutoSchemaKG Draft",
            "",
            f"**Source document length:** {len(self.source_document)} chars",
            "",
        ]

        lines.append("### Property Keys")
        if not self.property_keys:
            lines.append("_None inferred._")
        else:
            lines.append("| Name | Data Type | Cardinality |")
            lines.append("|------|-----------|-------------|")
            for pk in self.property_keys:
                lines.append(f"| {pk.name} | {pk.data_type} | {pk.cardinality} |")
        lines.append("")

        lines.append("### Vertex Labels")
        if not self.vertex_labels:
            lines.append("_None inferred._")
        else:
            lines.append("| Label | Properties | Primary Keys | Nullable Keys |")
            lines.append("|-------|------------|--------------|---------------|")
            for vl in self.vertex_labels:
                props = ", ".join(vl.properties) or "-"
                pks = ", ".join(vl.primary_keys) or "-"
                nks = ", ".join(vl.nullable_keys) or "-"
                lines.append(f"| {vl.name} | {props} | {pks} | {nks} |")
        lines.append("")

        lines.append("### Edge Labels")
        if not self.edge_labels:
            lines.append("_None inferred._")
        else:
            lines.append("| Label | Source → Target | Properties |")
            lines.append("|-------|-----------------|------------|")
            for el in self.edge_labels:
                props = ", ".join(el.properties) or "-"
                lines.append(f"| {el.name} | {el.source_label} → {el.target_label} | {props} |")
        lines.append("")

        lines.append("### Warnings")
        warnings = self._validation_warnings()
        if not warnings:
            lines.append("_No obvious issues detected._")
        else:
            for warning in warnings:
                lines.append(f"- {warning}")
        lines.append("")
        return "\n".join(lines)

    def _validation_warnings(self) -> List[str]:
        """Lightweight sanity checks surfaced to the reviewer."""
        warnings: List[str] = []
        pk_names = {pk.name for pk in self.property_keys}
        for vl in self.vertex_labels:
            for prop in vl.properties:
                if prop not in pk_names:
                    warnings.append(f"Vertex label '{vl.name}' uses undefined property '{prop}'")
            for pk in vl.primary_keys:
                if pk not in vl.properties:
                    warnings.append(f"Vertex label '{vl.name}' primary key '{pk}' not in properties")
        for el in self.edge_labels:
            if el.source_label not in {v.name for v in self.vertex_labels}:
                warnings.append(f"Edge label '{el.name}' source '{el.source_label}' not in vertex labels")
            if el.target_label not in {v.name for v in self.vertex_labels}:
                warnings.append(f"Edge label '{el.name}' target '{el.target_label}' not in vertex labels")
            for prop in el.properties:
                if prop not in pk_names:
                    warnings.append(f"Edge label '{el.name}' uses undefined property '{prop}'")
        return warnings


@dataclass
class SchemaReviewResult:
    """Result of the human review step."""
    approved: bool = False
    modified_schema: Optional[SchemaDraft] = None
    reason: str = ""

    def effective_schema(self, original: SchemaDraft) -> SchemaDraft:
        """Return the schema to commit: the modified one if provided, otherwise the original."""
        return self.modified_schema if self.modified_schema is not None and self.approved else original


@dataclass
class AutoSchemaKGResult:
    """Final output of the AutoSchemaKG operator."""
    draft: SchemaDraft
    review: SchemaReviewResult
    committed: bool = False
    commit_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "draft": self.draft.to_schema_dict(),
            "human_readable": self.draft.to_human_readable(),
            "review": {
                "approved": self.review.approved,
                "reason": self.review.reason,
            },
            "committed": self.committed,
            "commit_error": self.commit_error,
        }


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

AutoSchemaKGReviewCallback = Callable[[SchemaDraft], SchemaReviewResult]


class AutoSchemaKGOperator:
    """Generate, review, and optionally commit a HugeGraph schema from a single document.

    Args:
        llm: LLM instance. If ``None``, uses ``LLMs().get_extract_llm()``.
        schema_commit_client: Object with an ``init_schema_if_need(schema_dict)`` method.
            If provided and review passes, the schema is committed to HugeGraph.
        review_callback: Callable that receives a ``SchemaDraft`` and returns a
            ``SchemaReviewResult``. If ``None``, the draft is auto-approved when
            ``allow_commit`` is ``True``.
        allow_commit: Whether to write the approved schema to HugeGraph. Even when
            ``True``, commit only happens if ``review_callback`` approves (or the
            default auto-approval is in effect).
        instructions: Extra guidance added to the LLM prompt (e.g., domain hints).
    """

    DEFAULT_SCHEMA_PROMPT = """You are an expert graph schema designer for Apache HugeGraph.

Given the document below, infer a concise property graph schema that best models the
entities and relationships described.  Prefer common sense labels and properties.

Return ONLY a valid JSON object with exactly these top-level keys:
- "propertykeys": list of property key definitions
- "vertexlabels": list of vertex label definitions
- "edgelabels": list of edge label definitions

Each property key must have:
  {{"name": "<name>", "data_type": "<text|int|long|double|date>", "cardinality": "<single|list|set>"}}

Each vertex label must have:
  {{"name": "<Label>", "properties": ["prop1", ...], "primary_keys": ["prop1"], "nullable_keys": []}}

Each edge label must have:
  {{"name": "<REL>", "source_label": "<SourceLabel>", "target_label": "<TargetLabel>", "properties": [], "nullable_keys": []}}

Rules:
1. Primary keys must be SINGLE cardinality text properties.
2. Every property referenced by a vertex or edge label must be defined in propertykeys.
3. Edge labels must connect existing vertex labels.
4. Keep the schema minimal but sufficient to represent the document.

Document:
{document}

Extra instructions:
{instructions}

JSON:"""

    _VALID_DATA_TYPES = {"text", "int", "long", "double", "date", "boolean"}
    _VALID_CARDINALITIES = {"single", "list", "set"}

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        schema_commit_client: Optional[Any] = None,
        review_callback: Optional[AutoSchemaKGReviewCallback] = None,
        allow_commit: bool = True,
        instructions: str = "",
    ):
        self.llm = llm or LLMs().get_extract_llm()
        self.schema_commit_client = schema_commit_client
        self.review_callback = review_callback
        self.allow_commit = allow_commit
        self.instructions = instructions or "No extra instructions."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, document: str, context: Optional[Dict[str, Any]] = None) -> AutoSchemaKGResult:
        """Generate a schema draft from ``document``, optionally review it, and commit.

        Args:
            document: Input text document.
            context: Optional dictionary for downstream compatibility; updated in place.

        Returns:
            AutoSchemaKGResult containing draft, review decision, and commit status.
        """
        if context is None:
            context = {}

        if not isinstance(document, str):
            raise ValueError("document must be a string")
        if not document.strip():
            raise ValueError("document must not be empty")

        raw_response = self._generate_schema(document)
        draft = self._parse_and_normalize(raw_response, document)

        review = self._review(draft)
        if not review.approved:
            log.warning("AutoSchemaKG draft rejected by reviewer: %s", review.reason)
            return AutoSchemaKGResult(draft=draft, review=review, committed=False)

        committed, commit_error = self._commit_if_allowed(review.effective_schema(draft))
        return AutoSchemaKGResult(draft=draft, review=review, committed=committed, commit_error=commit_error)

    # ------------------------------------------------------------------
    # LLM generation and parsing
    # ------------------------------------------------------------------

    def _generate_schema(self, document: str) -> str:
        prompt = self.DEFAULT_SCHEMA_PROMPT.format(
            document=document.strip(),
            instructions=self.instructions,
        )
        try:
            response = self.llm.generate(prompt=prompt)
        except Exception as e:
            log.error("LLM schema generation failed: %s", e)
            raise RuntimeError(f"LLM schema generation failed: {e}") from e
        if not response or not response.strip():
            raise RuntimeError("LLM returned empty schema response")
        return response.strip()

    def _parse_and_normalize(self, raw_response: str, document: str) -> SchemaDraft:
        json_text = self._extract_json(raw_response)
        try:
            raw_schema = json.loads(json_text)
        except json.JSONDecodeError as e:
            log.error("Failed to parse LLM schema as JSON: %s", json_text)
            raise RuntimeError("Invalid JSON schema from LLM") from e

        return self._normalize_schema(raw_schema, document, raw_response)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a Markdown code block or return the trimmed text."""
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If the whole response looks like JSON, use it as is
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()
        return text.strip()

    def _normalize_schema(
        self,
        raw_schema: Dict[str, Any],
        document: str,
        raw_response: str,
    ) -> SchemaDraft:
        """Convert raw LLM JSON into a validated SchemaDraft."""
        property_keys = self._normalize_property_keys(raw_schema.get("propertykeys", []))
        vertex_labels = self._normalize_vertex_labels(raw_schema.get("vertexlabels", []))
        edge_labels = self._normalize_edge_labels(raw_schema.get("edgelabels", []))

        # Ensure primary key properties exist for each vertex label
        for vl in vertex_labels:
            for pk in vl.primary_keys:
                if pk not in {p.name for p in property_keys}:
                    property_keys.append(PropertyKeyDef(name=pk, data_type="text", cardinality="single"))

        # Ensure edge properties exist
        for el in edge_labels:
            for prop in el.properties:
                if prop not in {p.name for p in property_keys}:
                    property_keys.append(PropertyKeyDef(name=prop, data_type="text", cardinality="single"))

        return SchemaDraft(
            property_keys=property_keys,
            vertex_labels=vertex_labels,
            edge_labels=edge_labels,
            source_document=document,
            raw_llm_response=raw_response,
        )

    def _normalize_property_keys(self, raw_items: List[Any]) -> List[PropertyKeyDef]:
        normalized: List[PropertyKeyDef] = []
        seen: set = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            data_type = str(item.get("data_type", item.get("dataType", "text"))).lower().strip()
            if data_type not in self._VALID_DATA_TYPES:
                data_type = "text"
            cardinality = str(item.get("cardinality", "single")).lower().strip()
            if cardinality not in self._VALID_CARDINALITIES:
                cardinality = "single"
            normalized.append(PropertyKeyDef(name=name, data_type=data_type, cardinality=cardinality))
        return normalized

    def _normalize_vertex_labels(self, raw_items: List[Any]) -> List[VertexLabelDef]:
        normalized: List[VertexLabelDef] = []
        seen: set = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            properties = self._as_string_list(item.get("properties", []))
            primary_keys = self._as_string_list(item.get("primary_keys", item.get("primaryKeys", [])))
            nullable_keys = self._as_string_list(item.get("nullable_keys", item.get("nullableKeys", [])))
            # If no primary key is specified, prefer the first text property named "name"
            if not primary_keys and properties:
                name_prop = next((p for p in properties if p.lower() == "name"), properties[0])
                primary_keys = [name_prop]
            normalized.append(VertexLabelDef(
                name=name,
                properties=properties,
                primary_keys=primary_keys,
                nullable_keys=nullable_keys,
            ))
        return normalized

    def _normalize_edge_labels(self, raw_items: List[Any]) -> List[EdgeLabelDef]:
        normalized: List[EdgeLabelDef] = []
        seen: set = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            source_label = str(item.get("source_label", item.get("sourceLabel", ""))).strip()
            target_label = str(item.get("target_label", item.get("targetLabel", ""))).strip()
            properties = self._as_string_list(item.get("properties", []))
            nullable_keys = self._as_string_list(item.get("nullable_keys", item.get("nullableKeys", [])))
            normalized.append(EdgeLabelDef(
                name=name,
                source_label=source_label,
                target_label=target_label,
                properties=properties,
                nullable_keys=nullable_keys,
            ))
        return normalized

    @staticmethod
    def _as_string_list(value: Any) -> List[str]:
        """Coerce a value to a list of non-empty strings."""
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    # ------------------------------------------------------------------
    # Review and commit
    # ------------------------------------------------------------------

    def _review(self, draft: SchemaDraft) -> SchemaReviewResult:
        if self.review_callback is not None:
            return self.review_callback(draft)
        # Auto-approve by default when no callback is provided
        return SchemaReviewResult(approved=True, reason="Auto-approved (no review callback)")

    def _commit_if_allowed(self, schema: SchemaDraft) -> Tuple[bool, str]:
        if not self.allow_commit:
            return False, "Commit disabled (allow_commit=False)"
        if self.schema_commit_client is None:
            return False, "No schema_commit_client provided"
        try:
            self.schema_commit_client.init_schema_if_need(schema.to_schema_dict())
            log.info("AutoSchemaKG committed schema with %d vertex labels and %d edge labels",
                     len(schema.vertex_labels), len(schema.edge_labels))
            return True, ""
        except Exception as e:  # pylint: disable=broad-except
            log.error("Failed to commit AutoSchemaKG schema: %s", e)
            return False, str(e)


# ---------------------------------------------------------------------------
# Schema merge / diff / conflict detection
# ---------------------------------------------------------------------------

@dataclass
class SchemaConflict:
    """A single conflict detected between schema drafts or within one draft."""

    conflict_type: str  # e.g., "duplicate_property", "type_mismatch", "primary_key_mismatch", "edge_endpoint_missing"
    name: str  # affected label or property name
    details: str
    source_drafts: List[int] = field(default_factory=list)  # indices into the draft list; [-1] for intra-draft

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_type": self.conflict_type,
            "name": self.name,
            "details": self.details,
            "source_drafts": self.source_drafts,
        }


@dataclass
class SchemaDiff:
    """Diff between two schema drafts (base -> target)."""

    added_property_keys: List[PropertyKeyDef] = field(default_factory=list)
    removed_property_keys: List[PropertyKeyDef] = field(default_factory=list)
    modified_property_keys: List[Dict[str, Any]] = field(default_factory=list)
    added_vertex_labels: List[VertexLabelDef] = field(default_factory=list)
    removed_vertex_labels: List[VertexLabelDef] = field(default_factory=list)
    modified_vertex_labels: List[Dict[str, Any]] = field(default_factory=list)
    added_edge_labels: List[EdgeLabelDef] = field(default_factory=list)
    removed_edge_labels: List[EdgeLabelDef] = field(default_factory=list)
    modified_edge_labels: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added_property_keys": [p.to_dict() for p in self.added_property_keys],
            "removed_property_keys": [p.to_dict() for p in self.removed_property_keys],
            "modified_property_keys": self.modified_property_keys,
            "added_vertex_labels": [v.to_dict() for v in self.added_vertex_labels],
            "removed_vertex_labels": [v.to_dict() for v in self.removed_vertex_labels],
            "modified_vertex_labels": self.modified_vertex_labels,
            "added_edge_labels": [e.to_dict() for e in self.added_edge_labels],
            "removed_edge_labels": [e.to_dict() for e in self.removed_edge_labels],
            "modified_edge_labels": self.modified_edge_labels,
        }


class SchemaDiffCalculator:
    """Calculate structural differences between two schema drafts."""

    def diff(self, base: SchemaDraft, target: SchemaDraft) -> SchemaDiff:
        """Return the diff from ``base`` to ``target``."""
        base_pk = {pk.name: pk for pk in base.property_keys}
        target_pk = {pk.name: pk for pk in target.property_keys}

        added_pks = [target_pk[name] for name in target_pk if name not in base_pk]
        removed_pks = [base_pk[name] for name in base_pk if name not in target_pk]
        modified_pks = []
        for name in base_pk:
            if name in target_pk and base_pk[name] != target_pk[name]:
                modified_pks.append({
                    "name": name,
                    "base": base_pk[name].to_dict(),
                    "target": target_pk[name].to_dict(),
                })

        base_vl = {vl.name: vl for vl in base.vertex_labels}
        target_vl = {vl.name: vl for vl in target.vertex_labels}
        added_vls = [target_vl[name] for name in target_vl if name not in base_vl]
        removed_vls = [base_vl[name] for name in base_vl if name not in target_vl]
        modified_vls = []
        for name in base_vl:
            if name in target_vl and base_vl[name] != target_vl[name]:
                modified_vls.append({
                    "name": name,
                    "base": base_vl[name].to_dict(),
                    "target": target_vl[name].to_dict(),
                })

        base_el = {el.name: el for el in base.edge_labels}
        target_el = {el.name: el for el in target.edge_labels}
        added_els = [target_el[name] for name in target_el if name not in base_el]
        removed_els = [base_el[name] for name in base_el if name not in target_el]
        modified_els = []
        for name in base_el:
            if name in target_el and base_el[name] != target_el[name]:
                modified_els.append({
                    "name": name,
                    "base": base_el[name].to_dict(),
                    "target": target_el[name].to_dict(),
                })

        return SchemaDiff(
            added_property_keys=added_pks,
            removed_property_keys=removed_pks,
            modified_property_keys=modified_pks,
            added_vertex_labels=added_vls,
            removed_vertex_labels=removed_vls,
            modified_vertex_labels=modified_vls,
            added_edge_labels=added_els,
            removed_edge_labels=removed_els,
            modified_edge_labels=modified_els,
        )


class SchemaConflictDetector:
    """Detect conflicts within or across schema drafts."""

    def detect_intra_draft(self, draft: SchemaDraft) -> List[SchemaConflict]:
        """Detect conflicts inside a single draft."""
        conflicts: List[SchemaConflict] = []
        pk_names = {pk.name: pk for pk in draft.property_keys}
        vl_names = {vl.name: vl for vl in draft.vertex_labels}

        # Duplicate property keys with different definitions
        seen_pk: Dict[str, PropertyKeyDef] = {}
        for pk in draft.property_keys:
            if pk.name in seen_pk and pk != seen_pk[pk.name]:
                conflicts.append(SchemaConflict(
                    conflict_type="duplicate_property",
                    name=pk.name,
                    details=f"Property '{pk.name}' has conflicting definitions.",
                    source_drafts=[-1],
                ))
            seen_pk[pk.name] = pk

        # Duplicate vertex labels with different definitions
        seen_vl: Dict[str, VertexLabelDef] = {}
        for vl in draft.vertex_labels:
            if vl.name in seen_vl and vl != seen_vl[vl.name]:
                conflicts.append(SchemaConflict(
                    conflict_type="duplicate_vertex_label",
                    name=vl.name,
                    details=f"Vertex label '{vl.name}' has conflicting definitions.",
                    source_drafts=[-1],
                ))
            seen_vl[vl.name] = vl

        # Duplicate edge labels with different endpoints
        seen_el: Dict[str, EdgeLabelDef] = {}
        for el in draft.edge_labels:
            if el.name in seen_el and el != seen_el[el.name]:
                conflicts.append(SchemaConflict(
                    conflict_type="duplicate_edge_label",
                    name=el.name,
                    details=(
                        f"Edge label '{el.name}' has conflicting source/target labels: "
                        f"{seen_el[el.name].source_label}→{seen_el[el.name].target_label} vs "
                        f"{el.source_label}→{el.target_label}."
                    ),
                    source_drafts=[-1],
                ))
            seen_el[el.name] = el

        # Primary key properties missing or wrong type
        for vl in draft.vertex_labels:
            for pk in vl.primary_keys:
                if pk not in pk_names:
                    conflicts.append(SchemaConflict(
                        conflict_type="undefined_primary_key",
                        name=pk,
                        details=f"Vertex label '{vl.name}' primary key '{pk}' is not defined as a property key.",
                        source_drafts=[-1],
                    ))
                elif pk_names[pk].cardinality != "single":
                    conflicts.append(SchemaConflict(
                        conflict_type="non_single_primary_key",
                        name=pk,
                        details=f"Primary key '{pk}' must be single cardinality, got '{pk_names[pk].cardinality}'.",
                        source_drafts=[-1],
                    ))

        # Edge endpoints missing
        for el in draft.edge_labels:
            if el.source_label not in vl_names:
                conflicts.append(SchemaConflict(
                    conflict_type="edge_source_missing",
                    name=el.source_label,
                    details=f"Edge label '{el.name}' source '{el.source_label}' is not a vertex label.",
                    source_drafts=[-1],
                ))
            if el.target_label not in vl_names:
                conflicts.append(SchemaConflict(
                    conflict_type="edge_target_missing",
                    name=el.target_label,
                    details=f"Edge label '{el.name}' target '{el.target_label}' is not a vertex label.",
                    source_drafts=[-1],
                ))
            for prop in el.properties:
                if prop not in pk_names:
                    conflicts.append(SchemaConflict(
                        conflict_type="undefined_edge_property",
                        name=prop,
                        details=f"Edge label '{el.name}' uses undefined property '{prop}'.",
                        source_drafts=[-1],
                    ))

        return conflicts

    def detect_cross_draft(self, drafts: List[SchemaDraft]) -> List[SchemaConflict]:
        """Detect conflicts across multiple drafts, e.g., type mismatches on shared labels."""
        conflicts: List[SchemaConflict] = []
        if len(drafts) < 2:
            return conflicts

        # Property key conflicts across drafts
        pk_groups: Dict[str, List[Tuple[int, PropertyKeyDef]]] = {}
        for idx, draft in enumerate(drafts):
            for pk in draft.property_keys:
                pk_groups.setdefault(pk.name, []).append((idx, pk))
        for name, entries in pk_groups.items():
            definitions = {json.dumps(e[1].to_dict(), sort_keys=True) for e in entries}
            if len(definitions) > 1:
                conflict_indices = sorted({e[0] for e in entries})
                details = "; ".join(
                    f"draft {idx}: {e[1].to_dict()}" for idx, e in enumerate(entries) if e[0] in conflict_indices
                )
                conflicts.append(SchemaConflict(
                    conflict_type="cross_draft_property_type_mismatch",
                    name=name,
                    details=f"Property key '{name}' has inconsistent definitions across drafts: {details}.",
                    source_drafts=conflict_indices,
                ))

        # Vertex label conflicts across drafts (same name, different properties/primary keys)
        vl_groups: Dict[str, List[Tuple[int, VertexLabelDef]]] = {}
        for idx, draft in enumerate(drafts):
            for vl in draft.vertex_labels:
                vl_groups.setdefault(vl.name, []).append((idx, vl))
        for name, entries in vl_groups.items():
            definitions = {json.dumps(e[1].to_dict(), sort_keys=True) for e in entries}
            if len(definitions) > 1:
                conflict_indices = sorted({e[0] for e in entries})
                conflicts.append(SchemaConflict(
                    conflict_type="cross_draft_vertex_label_mismatch",
                    name=name,
                    details=f"Vertex label '{name}' has inconsistent definitions across drafts.",
                    source_drafts=conflict_indices,
                ))

        # Edge label conflicts across drafts (same name, different endpoints)
        el_groups: Dict[str, List[Tuple[int, EdgeLabelDef]]] = {}
        for idx, draft in enumerate(drafts):
            for el in draft.edge_labels:
                el_groups.setdefault(el.name, []).append((idx, el))
        for name, entries in el_groups.items():
            endpoints = {(e[1].source_label, e[1].target_label) for e in entries}
            if len(endpoints) > 1:
                conflict_indices = sorted({e[0] for e in entries})
                conflicts.append(SchemaConflict(
                    conflict_type="cross_draft_edge_endpoint_mismatch",
                    name=name,
                    details=f"Edge label '{name}' connects different endpoints across drafts.",
                    source_drafts=conflict_indices,
                ))

        return conflicts

    def detect(self, drafts: List[SchemaDraft]) -> List[SchemaConflict]:
        """Detect both intra-draft and cross-draft conflicts."""
        conflicts: List[SchemaConflict] = []
        for draft in drafts:
            conflicts.extend(self.detect_intra_draft(draft))
        conflicts.extend(self.detect_cross_draft(drafts))
        return conflicts


class SchemaMerger:
    """Merge multiple schema drafts into a single unified schema draft.

    Merge strategy:
    - Property keys: union by name; on type/cardinality conflict, keep the first definition and flag it.
    - Vertex labels: union by name; merge properties and primary keys; on conflict, keep the union.
    - Edge labels: union by name; on endpoint conflict, keep the first definition and flag it.
    """

    def __init__(self) -> None:
        self.conflict_detector = SchemaConflictDetector()
        self.diff_calculator = SchemaDiffCalculator()

    def merge(self, drafts: List[SchemaDraft]) -> Tuple[SchemaDraft, List[SchemaConflict]]:
        """Merge ``drafts`` into a unified ``SchemaDraft`` and return conflicts."""
        if not drafts:
            return SchemaDraft(), []
        if len(drafts) == 1:
            return drafts[0], self.conflict_detector.detect_intra_draft(drafts[0])

        conflicts = self.conflict_detector.detect(drafts)

        merged_property_keys: List[PropertyKeyDef] = []
        pk_map: Dict[str, PropertyKeyDef] = {}
        for draft in drafts:
            for pk in draft.property_keys:
                if pk.name not in pk_map:
                    pk_map[pk.name] = pk
                    merged_property_keys.append(pk)

        merged_vertex_labels: List[VertexLabelDef] = []
        vl_map: Dict[str, VertexLabelDef] = {}
        for draft in drafts:
            for vl in draft.vertex_labels:
                if vl.name not in vl_map:
                    vl_map[vl.name] = VertexLabelDef(
                        name=vl.name,
                        properties=list(vl.properties),
                        primary_keys=list(vl.primary_keys),
                        nullable_keys=list(vl.nullable_keys),
                    )
                else:
                    existing = vl_map[vl.name]
                    existing.properties = list(dict.fromkeys(existing.properties + vl.properties))
                    existing.primary_keys = list(dict.fromkeys(existing.primary_keys + vl.primary_keys))
                    existing.nullable_keys = list(dict.fromkeys(existing.nullable_keys + vl.nullable_keys))

        for vl in vl_map.values():
            # Ensure primary keys are still present in the merged properties
            for pk in vl.primary_keys:
                if pk not in vl.properties:
                    vl.properties.append(pk)
            merged_vertex_labels.append(vl)

        merged_edge_labels: List[EdgeLabelDef] = []
        el_map: Dict[str, EdgeLabelDef] = {}
        for draft in drafts:
            for el in draft.edge_labels:
                if el.name not in el_map:
                    el_map[el.name] = EdgeLabelDef(
                        name=el.name,
                        source_label=el.source_label,
                        target_label=el.target_label,
                        properties=list(el.properties),
                        nullable_keys=list(el.nullable_keys),
                    )
                else:
                    existing = el_map[el.name]
                    existing.properties = list(dict.fromkeys(existing.properties + el.properties))
                    existing.nullable_keys = list(dict.fromkeys(existing.nullable_keys + el.nullable_keys))

        merged_edge_labels = list(el_map.values())

        # Merge source documents for traceability
        source_documents = []
        for draft in drafts:
            if draft.source_document:
                source_documents.append(draft.source_document)
        merged_source = "\n\n".join(source_documents)

        merged = SchemaDraft(
            property_keys=merged_property_keys,
            vertex_labels=merged_vertex_labels,
            edge_labels=merged_edge_labels,
            source_document=merged_source[:2000],  # keep a preview for UI
        )

        # Re-normalize to ensure merged properties have keys
        existing_pk_names = {pk.name for pk in merged.property_keys}
        for vl in merged.vertex_labels:
            for prop in vl.properties:
                if prop not in existing_pk_names:
                    merged.property_keys.append(PropertyKeyDef(name=prop, data_type="text", cardinality="single"))
                    existing_pk_names.add(prop)
        for el in merged.edge_labels:
            for prop in el.properties:
                if prop not in existing_pk_names:
                    merged.property_keys.append(PropertyKeyDef(name=prop, data_type="text", cardinality="single"))
                    existing_pk_names.add(prop)

        return merged, conflicts


# ---------------------------------------------------------------------------
# Batch / multi-document schema inference
# ---------------------------------------------------------------------------

@dataclass
class BatchAutoSchemaKGResult:
    """Result of inferring schema from multiple documents."""

    merged_draft: SchemaDraft
    per_document_results: List[AutoSchemaKGResult]
    conflicts: List[SchemaConflict]
    diff_from_first: Optional[SchemaDiff] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_schema": self.merged_draft.to_schema_dict(),
            "human_readable": self.merged_draft.to_human_readable(),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "document_count": len(self.per_document_results),
            "diff_from_first": self.diff_from_first.to_dict() if self.diff_from_first else None,
        }


class BatchAutoSchemaKGOperator:
    """Infer schema from multiple documents and merge them into one unified draft.

    This operator is designed for enterprise-scale use cases where the full corpus is
    too large to feed into a single LLM call. It samples documents, infers a schema
    per document, merges the results, and reports conflicts for human review.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        schema_commit_client: Optional[Any] = None,
        review_callback: Optional[AutoSchemaKGReviewCallback] = None,
        allow_commit: bool = True,
        instructions: str = "",
        max_workers: int = 1,
    ):
        self.llm = llm
        self.schema_commit_client = schema_commit_client
        self.review_callback = review_callback
        self.allow_commit = allow_commit
        self.instructions = instructions
        self.max_workers = max_workers
        self._single_operator = AutoSchemaKGOperator(
            llm=self.llm,
            schema_commit_client=None,  # commit happens at batch level after merge
            review_callback=None,
            allow_commit=False,
            instructions=self.instructions,
        )
        self._merger = SchemaMerger()

    def run(
        self,
        documents: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> BatchAutoSchemaKGResult:
        """Infer a merged schema from ``documents``.

        Args:
            documents: List of text documents. Empty strings are skipped.
            context: Optional dictionary for downstream compatibility.

        Returns:
            BatchAutoSchemaKGResult containing the merged draft, per-document results,
            detected conflicts, and a diff from the first document's draft.
        """
        if context is None:
            context = {}

        valid_documents = [doc for doc in documents if isinstance(doc, str) and doc.strip()]
        if not valid_documents:
            raise ValueError("At least one non-empty document is required")

        per_document_results: List[AutoSchemaKGResult] = []
        for doc in valid_documents:
            try:
                result = self._single_operator.run(doc, context=context)
                per_document_results.append(result)
            except Exception as e:  # pylint: disable=broad-except
                log.error("AutoSchemaKG failed for one document: %s", e)
                # Placeholder result with empty draft so the batch can continue
                per_document_results.append(
                    AutoSchemaKGResult(
                        draft=SchemaDraft(source_document=doc),
                        review=SchemaReviewResult(approved=False, reason=f"Failed: {e}"),
                    )
                )

        drafts = [r.draft for r in per_document_results if r.draft.property_keys or r.draft.vertex_labels]
        merged_draft, conflicts = self._merger.merge(drafts)

        diff = None
        if len(drafts) >= 2:
            diff = self._merger.diff_calculator.diff(drafts[0], merged_draft)

        # Optional review + commit at the merged level
        review = self._review_merged(merged_draft)
        committed = False
        commit_error = ""
        if review.approved:
            committed, commit_error = self._commit_if_allowed(merged_draft)

        if not committed and commit_error:
            log.warning("Batch AutoSchemaKG commit skipped or failed: %s", commit_error)

        return BatchAutoSchemaKGResult(
            merged_draft=merged_draft,
            per_document_results=per_document_results,
            conflicts=conflicts,
            diff_from_first=diff,
        )

    def _review_merged(self, draft: SchemaDraft) -> SchemaReviewResult:
        if self.review_callback is not None:
            return self.review_callback(draft)
        return SchemaReviewResult(approved=True, reason="Auto-approved (no review callback)")

    def _commit_if_allowed(self, schema: SchemaDraft) -> Tuple[bool, str]:
        if not self.allow_commit:
            return False, "Commit disabled (allow_commit=False)"
        if self.schema_commit_client is None:
            return False, "No schema_commit_client provided"
        try:
            self.schema_commit_client.init_schema_if_need(schema.to_schema_dict())
            log.info(
                "Batch AutoSchemaKG committed schema with %d vertex labels and %d edge labels",
                len(schema.vertex_labels),
                len(schema.edge_labels),
            )
            return True, ""
        except Exception as e:  # pylint: disable=broad-except
            log.error("Failed to commit batch AutoSchemaKG schema: %s", e)
            return False, str(e)


# ---------------------------------------------------------------------------
# Multimodal input support
# ---------------------------------------------------------------------------

def multimodal_result_to_document(
    pdf_extraction_result: Optional[Dict[str, Any]] = None,
    vlm_descriptions: Optional[List[Dict[str, Any]]] = None,
    text_blocks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Convert multimodal extraction results into a single text document.

    The output is suitable as the ``document`` argument of ``AutoSchemaKGOperator``.
    It preserves page-level text, headings, and VLM-generated image/table descriptions
    so that the schema inference can cover both textual and visual content.
    """
    lines: List[str] = []

    # Page text blocks (if a full pdf_extraction_result is provided)
    if pdf_extraction_result:
        pages = pdf_extraction_result.get("pages", [])
        for page in pages:
            page_num = page.get("page_num", 0)
            lines.append(f"--- Page {page_num + 1} ---")
            for tb in page.get("text_blocks", []):
                text = str(tb.get("text", "")).strip()
                if not text:
                    continue
                if tb.get("is_heading"):
                    lines.append(f"# {text}")
                else:
                    lines.append(text)
            lines.append("")

    # Standalone text blocks (e.g. from UnifiedDocumentParser)
    if text_blocks:
        lines.append("--- Text Blocks ---")
        for tb in text_blocks:
            text = str(tb.get("text", tb.get("content", ""))).strip()
            if not text:
                continue
            if tb.get("is_heading"):
                lines.append(f"# {text}")
            else:
                lines.append(text)
        lines.append("")

    # VLM image / chart / table descriptions
    if vlm_descriptions:
        lines.append("--- Image and Chart Descriptions ---")
        for desc in vlm_descriptions:
            image_id = desc.get("image_id", "unknown")
            lines.append(f"Image {image_id}:")
            if desc.get("caption"):
                lines.append(f"  Caption: {desc['caption']}")
            if desc.get("detailed_description"):
                lines.append(f"  Description: {desc['detailed_description']}")
            if desc.get("chart_type"):
                lines.append(f"  Type: {desc['chart_type']}")
            if desc.get("key_insights"):
                lines.append(f"  Key insights: {', '.join(str(x) for x in desc['key_insights'])}")
            if desc.get("related_keywords"):
                lines.append(f"  Keywords: {', '.join(str(x) for x in desc['related_keywords'])}")
            if desc.get("object_labels"):
                lines.append(f"  Objects: {', '.join(str(x) for x in desc['object_labels'])}")
            lines.append("")

    return "\n".join(lines).strip()


class MultimodalAutoSchemaKGOperator(AutoSchemaKGOperator):
    """AutoSchemaKG variant that accepts multimodal extraction results.

    Inputs may include:
    * ``pdf_extraction_result``: output of ``PDFImageExtractor`` / ``MultimodalExtractNode``
    * ``vlm_descriptions``: output of ``VLMDescribeNode``
    * ``text_blocks``: plain text blocks from a document parser

    The operator concatenates these sources into a single document string and
    then runs the standard AutoSchemaKG pipeline.
    """

    def run(
        self,
        document: Optional[str] = None,
        pdf_extraction_result: Optional[Dict[str, Any]] = None,
        vlm_descriptions: Optional[List[Dict[str, Any]]] = None,
        text_blocks: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AutoSchemaKGResult:
        """Infer a schema from text and/or multimodal inputs.

        At least one of ``document``, ``pdf_extraction_result``, ``vlm_descriptions``,
        or ``text_blocks`` must be provided.
        """
        if document is None:
            document = multimodal_result_to_document(
                pdf_extraction_result=pdf_extraction_result,
                vlm_descriptions=vlm_descriptions,
                text_blocks=text_blocks,
            )
        else:
            # Still append multimodal context if a primary document is provided.
            multimodal_doc = multimodal_result_to_document(
                pdf_extraction_result=pdf_extraction_result,
                vlm_descriptions=vlm_descriptions,
                text_blocks=text_blocks,
            )
            if multimodal_doc:
                document = f"{document}\n\n{multimodal_doc}".strip()

        if not document:
            raise ValueError("No document or multimodal input provided")
        return super().run(document, context=context)
