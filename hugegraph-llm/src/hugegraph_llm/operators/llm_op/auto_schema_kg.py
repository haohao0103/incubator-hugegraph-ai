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
        return {"name": self.name, "data_type": self.data_type, "cardinality": self.cardinality}


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
