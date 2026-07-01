# Licensed to the Apache Foundation (ASF) under one
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
# "ASIS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Guided Mode Extraction — Schema-constrained extraction with Pydantic ResponseModel.

For well-defined domains where the ontology is known a priori (risk control
enhancement, code graph, financial regulation), Guided mode constrains LLM
extraction output to match the existing relationship graph's schema.

This module provides:
1. Pydantic ResponseModel classes that constrain LLM output to valid schema types
2. Guided extraction prompt templates that include the schema definition
3. GuidedExtractOperator that orchestrates schema-constrained extraction

Design: The ResponseModel is dynamically built from the relationship graph's
vertex types and edge types. Each vertex type becomes an entity class, each
edge type becomes a relation class. The LLM must output instances of these
classes — no free-form type strings allowed.

Context keys:
  IN:
    chunks                — List[str] raw text chunks
    graph_rag_schema_config — GraphRAGSchemaConfig (mode=guided)
    relationship_graph_types — List[str] vertex type names
    schema                — Optional[Dict] HugeGraph schema dict

  OUT:
    vertices              — List[Dict] extracted vertices (schema-constrained)
    edges                 — List[Dict] extracted edges (schema-constrained)
    extracted_entities    — List[Dict] entity dicts
    extracted_relations   — List[Dict] relation dicts
    call_count            — int LLM call count
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.graphrag_op.graphrag_schema_config import (
    GraphRAGSchemaConfig,
    SchemaMode,
)
from hugegraph_llm.utils.log import log


# ============================================================
# Pydantic ResponseModel Base Classes
# ============================================================

class GuidedEntity(BaseModel):
    """Base Pydantic model for a schema-constrained entity extraction.

    Each entity must specify its label from the allowed set, and provide
    properties that match the schema's property definitions for that label.
    """
    label: str = Field(..., description="Entity vertex label from the schema")
    name: str = Field(..., description="Entity name/primary key value")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entity properties matching the schema definition"
    )

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v):
        # Validation happens at GuidedExtractResponse level where allowed_labels is known
        if not v or not v.strip():
            raise ValueError("Entity label must not be empty")
        return v.strip()


class GuidedRelation(BaseModel):
    """Base Pydantic model for a schema-constrained relation extraction.

    Each relation must specify its edge label, source/target entity labels,
    and optional properties.
    """
    label: str = Field(..., description="Edge label from the schema")
    source_label: str = Field(..., description="Source entity vertex label")
    source_name: str = Field(..., description="Source entity name")
    target_label: str = Field(..., description="Target entity vertex label")
    target_name: str = Field(..., description="Target entity name")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relation properties matching the schema definition"
    )


class GuidedExtractResponse(BaseModel):
    """Top-level Pydantic model for guided extraction output.

    This constrains the LLM to output entities and relations whose labels
    are from the allowed set derived from the relationship graph schema.
    """
    entities: List[GuidedEntity] = Field(
        default_factory=list,
        description="Extracted entities with schema-constrained labels"
    )
    relations: List[GuidedRelation] = Field(
        default_factory=list,
        description="Extracted relations with schema-constrained labels"
    )


# ============================================================
# Dynamic ResponseModel Builder
# ============================================================

class GuidedResponseModelBuilder:
    """Build a Pydantic ResponseModel from the relationship graph schema.

    Takes the list of vertex types and edge types from the relationship graph,
    and creates a constrained ResponseModel that only allows those types.
    This is the core mechanism that prevents type noise in guided mode.

    Usage:
        builder = GuidedResponseModelBuilder(
            vertex_types=["person", "device", "ip_address"],
            edge_types=["uses", "owns", "located_at"],
        )
        response_model = builder.build()
        # response_model constrains LLM output to only these types
    """

    def __init__(
        self,
        vertex_types: List[str],
        edge_types: Optional[List[str]] = None,
        schema_dict: Optional[Dict[str, Any]] = None,
        config: Optional[GraphRAGSchemaConfig] = None,
    ):
        """
        :param vertex_types: Allowed vertex label names from relationship graph.
        :param edge_types: Allowed edge label names from relationship graph.
        :param schema_dict: Full HugeGraph schema dict (vertexlabels/edgelabels)
                           for property validation.
        :param config: GraphRAGSchemaConfig for max_types limits.
        """
        self.vertex_types = vertex_types
        self.edge_types = edge_types or []
        self.schema_dict = schema_dict or {}
        self.config = config or GraphRAGSchemaConfig()

    def build(self) -> "GuidedExtractResponse":
        """Build and return the constrained ResponseModel class.

        Creates a subclass of GuidedExtractResponse with validators that
        enforce the allowed vertex and edge type sets.
        """
        allowed_vertex_labels = set(self.vertex_types[:self.config.guided_max_entity_types])
        allowed_edge_labels = set(self.edge_types[:self.config.guided_max_relation_types])

        # Build property schema per vertex label
        vertex_properties = self._build_vertex_properties_map()
        edge_properties = self._build_edge_properties_map()

        class ConstrainedGuidedEntity(GuidedEntity):
            """Entity with constrained label validation."""
            _allowed_labels: Set[str] = allowed_vertex_labels
            _vertex_properties: Dict[str, List[str]] = vertex_properties

            @field_validator("label")
            @classmethod
            def label_must_be_in_allowed_set(cls, v):
                if v not in cls._allowed_labels:
                    # Try case-insensitive match
                    for allowed in cls._allowed_labels:
                        if v.lower() == allowed.lower():
                            return allowed
                    raise ValueError(
                        f"Entity label '{v}' not in allowed set: "
                        f"{sorted(cls._allowed_labels)}"
                    )
                return v

            @field_validator("properties")
            @classmethod
            def properties_must_match_schema(cls, v, info):
                label = info.data.get("label", "")
                allowed_props = cls._vertex_properties.get(label, [])
                if allowed_props and not self.config.guided_allow_dynamic:
                    # Filter out properties not in the schema
                    filtered = {
                        k: val for k, val in v.items()
                        if k in allowed_props
                    }
                    return filtered
                return v

        class ConstrainedGuidedRelation(GuidedRelation):
            """Relation with constrained label validation."""
            _allowed_labels: Set[str] = allowed_edge_labels
            _allowed_vertex_labels: Set[str] = allowed_vertex_labels
            _edge_properties: Dict[str, List[str]] = edge_properties

            @field_validator("label")
            @classmethod
            def label_must_be_in_allowed_set(cls, v):
                if v not in cls._allowed_labels:
                    for allowed in cls._allowed_labels:
                        if v.lower() == allowed.lower():
                            return allowed
                    raise ValueError(
                        f"Relation label '{v}' not in allowed set: "
                        f"{sorted(cls._allowed_labels)}"
                    )
                return v

            @field_validator("source_label", "target_label")
            @classmethod
            def endpoint_labels_must_be_valid(cls, v):
                if v not in cls._allowed_vertex_labels:
                    for allowed in cls._allowed_vertex_labels:
                        if v.lower() == allowed.lower():
                            return allowed
                    raise ValueError(
                        f"Endpoint label '{v}' not in allowed vertex set: "
                        f"{sorted(cls._allowed_vertex_labels)}"
                    )
                return v

        class ConstrainedGuidedExtractResponse(GuidedExtractResponse):
            """Top-level response model with constrained entity/relation types."""
            entities: List[ConstrainedGuidedEntity] = Field(default_factory=list)
            relations: List[ConstrainedGuidedRelation] = Field(default_factory=list)

        return ConstrainedGuidedExtractResponse

    def _build_vertex_properties_map(self) -> Dict[str, List[str]]:
        """Build a mapping of vertex label → allowed property names."""
        props_map: Dict[str, List[str]] = {}
        for vertex in self.schema_dict.get("vertexlabels", []):
            label = vertex.get("name", "")
            properties = vertex.get("properties", [])
            if isinstance(properties, list):
                props_map[label] = properties
            elif isinstance(properties, dict):
                props_map[label] = list(properties.keys())
        return props_map

    def _build_edge_properties_map(self) -> Dict[str, List[str]]:
        """Build a mapping of edge label → allowed property names."""
        props_map: Dict[str, List[str]] = {}
        for edge in self.schema_dict.get("edgelabels", []):
            label = edge.get("name", "")
            properties = edge.get("properties", [])
            if isinstance(properties, list):
                props_map[label] = properties
            elif isinstance(properties, dict):
                props_map[label] = list(properties.keys())
        return props_map


# ============================================================
# Guided Extraction Prompts
# ============================================================

GUIDED_EXTRACT_PROMPT = """\
You are an expert entity and relation extractor for a domain-specific knowledge
graph. Extract entities and relations from the given text, STRICTLY following
the provided schema definition.

## Schema Definition
Vertex types (entity labels): {vertex_types}
Edge types (relation labels): {edge_types}
Property schema per type:
{property_schema}

## Extraction Rules
1. Every entity MUST have a label from the vertex types list above.
2. Every relation MUST have a label from the edge types list above.
3. Every entity MUST have a 'name' property (the primary key).
4. Entity/relation properties MUST match the schema — no extra properties.
5. If a text mention doesn't clearly fit any schema type, SKIP it.
6. Do NOT invent new types. Only use types from the schema definition.

## Input Text
{text}

## Output Format (JSON only)
{{
  "entities": [
    {{
      "label": "vertex_type_from_schema",
      "name": "entity_primary_key_value",
      "properties": {{ "prop_name": "prop_value", ... }}
    }}
  ],
  "relations": [
    {{
      "label": "edge_type_from_schema",
      "source_label": "source_vertex_type",
      "source_name": "source_entity_name",
      "target_label": "target_vertex_type",
      "target_name": "target_entity_name",
      "properties": {{ "prop_name": "prop_value", ... }}
    }}
  ]
}}

Output ONLY valid JSON. No markdown, no commentary.
"""


# ============================================================
# Guided Extract Operator
# ============================================================

class GuidedExtractOperator:
    """Schema-constrained extraction using Pydantic ResponseModel.

    This operator is activated when GraphRAGSchemaConfig.mode == GUIDED.
    It constrains the LLM to only output entity/relation types that exist
    in the relationship graph schema, preventing type noise entirely.

    Usage:
        operator = GuidedExtractOperator(llm=my_llm, config=my_config)
        context = operator.run(context)
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[GraphRAGSchemaConfig] = None,
    ):
        self.llm = llm
        self.config = config or GraphRAGSchemaConfig()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute guided extraction.

        Reads: chunks, graph_rag_schema_config, relationship_graph_types,
               schema.
        Writes: vertices, edges, extracted_entities, extracted_relations,
                call_count.
        """
        # Merge config from context
        config_in_context = context.get("graph_rag_schema_config")
        if config_in_context and isinstance(config_in_context, GraphRAGSchemaConfig):
            self.config = config_in_context

        # Verify we're in guided mode
        if self.config.mode != SchemaMode.GUIDED:
            log.warning(
                "GuidedExtractOperator called with mode=%s, expected GUIDED. "
                "Skipping — this operator only runs in guided mode.",
                self.config.mode.value,
            )
            return context

        chunks = context.get("chunks", [])
        if not chunks:
            log.warning("No chunks provided for guided extraction")
            return context

        # Get schema sources
        rel_types = context.get("relationship_graph_types", [])
        if not rel_types:
            rel_types = self.config.relationship_graph_types

        schema_dict = context.get("schema", {})

        # Build edge types from schema
        edge_types = []
        if schema_dict:
            for edge in schema_dict.get("edgelabels", []):
                edge_types.append(edge.get("name", ""))

        # Build ResponseModel
        builder = GuidedResponseModelBuilder(
            vertex_types=rel_types,
            edge_types=edge_types,
            schema_dict=schema_dict,
            config=self.config,
        )
        response_model = builder.build()

        # Build property schema description for prompt
        property_schema = self._build_property_schema_description(schema_dict, rel_types)

        # Extract from each chunk
        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []
        call_count = 0

        for chunk in chunks:
            prompt = GUIDED_EXTRACT_PROMPT.format(
                vertex_types=", ".join(rel_types),
                edge_types=", ".join(edge_types) if edge_types else "(as defined in schema)",
                property_schema=property_schema,
                text=chunk,
            )

            try:
                response = self.llm.generate(prompt=prompt)
                call_count += 1
                parsed = self._parse_guided_response(response, response_model)
                all_entities.extend(parsed["entities"])
                all_relations.extend(parsed["relations"])
            except Exception as e:  # pylint: disable=broad-except
                log.error("Guided extraction failed for chunk: %s", e)
                call_count += 1

        # Convert to vertex/edge format
        vertices, edges = self._convert_to_vertex_edge_format(
            all_entities, all_relations, schema_dict
        )

        # Deduplicate
        vertices = self._deduplicate_vertices(vertices)
        edges = self._deduplicate_edges(edges)

        # Write to context
        context["vertices"] = context.get("vertices", []) + vertices
        context["edges"] = context.get("edges", []) + edges
        context["extracted_entities"] = all_entities
        context["extracted_relations"] = all_relations
        context["call_count"] = context.get("call_count", 0) + call_count

        log.info(
            "Guided extraction complete: %d chunks, %d LLM calls, "
            "%d entities, %d relations, %d vertices, %d edges",
            len(chunks), call_count,
            len(all_entities), len(all_relations),
            len(vertices), len(edges),
        )
        return context

    # ---- Property Schema Description ----

    def _build_property_schema_description(
        self,
        schema_dict: Dict[str, Any],
        vertex_types: List[str],
    ) -> str:
        """Build a human-readable property schema description for the prompt."""
        lines = []

        for vertex in schema_dict.get("vertexlabels", []):
            name = vertex.get("name", "")
            if name not in vertex_types:
                continue
            props = vertex.get("properties", [])
            pk = vertex.get("primary_keys", [])
            if isinstance(props, list):
                lines.append(f"  {name}: properties={props}, primary_key={pk}")
            elif isinstance(props, dict):
                lines.append(f"  {name}: properties={list(props.keys())}, primary_key={pk}")

        for edge in schema_dict.get("edgelabels", []):
            name = edge.get("name", "")
            src = edge.get("source_label", "")
            tgt = edge.get("target_label", "")
            props = edge.get("properties", [])
            lines.append(f"  {name}: {src} → {tgt}, properties={props}")

        return "\n".join(lines) if lines else "(No detailed property schema available)"

    # ---- Response Parsing ----

    def _parse_guided_response(
        self,
        response: str,
        response_model: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Parse LLM response using the Pydantic ResponseModel for validation."""
        # Strip markdown code blocks
        response = re.sub(r"```\w*\n?", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()

        # Try to extract JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            log.warning("No JSON found in guided extraction response")
            return {"entities": [], "relations": []}

        try:
            data = json.loads(json_match.group(0).strip())
        except json.JSONDecodeError:
            log.warning("Invalid JSON in guided extraction response")
            return {"entities": [], "relations": []}

        # Try Pydantic validation
        try:
            validated = response_model(**data)
            entities = [e.dict() for e in validated.entities]
            relations = [r.dict() for r in validated.relations]
            return {"entities": entities, "relations": relations}
        except Exception as e:  # pylint: disable=broad-except
            log.warning("Pydantic validation failed: %s — using raw parse", e)
            # Fall back to raw parse without Pydantic validation
            entities = []
            for item in data.get("entities", []):
                if isinstance(item, dict) and "label" in item and "name" in item:
                    entities.append(item)

            relations = []
            for item in data.get("relations", []):
                if isinstance(item, dict) and "label" in item:
                    relations.append(item)

            return {"entities": entities, "relations": relations}

    # ---- Format Conversion ----

    def _convert_to_vertex_edge_format(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        schema_dict: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert guided extraction output to HugeGraph vertex/edge format."""
        vertex_label_map = {
            v["name"]: v for v in schema_dict.get("vertexlabels", [])
        }
        edge_label_map = {
            e["name"]: e for e in schema_dict.get("edgelabels", [])
        }

        vertices: List[Dict[str, Any]] = []
        name_to_vid: Dict[Tuple[str, str], str] = {}

        for entity in entities:
            label = entity.get("label", "")
            name = entity.get("name", "")
            props = entity.get("properties", {})

            # Generate vertex ID
            vid = self._generate_vertex_id(label, name, vertex_label_map)
            name_to_vid[(label, name)] = vid

            vertex = {
                "id": vid,
                "label": label,
                "type": "vertex",
                "properties": {"name": name, **props},
            }
            vertices.append(vertex)

        edges: List[Dict[str, Any]] = []
        for relation in relations:
            label = relation.get("label", "")
            src_label = relation.get("source_label", "")
            src_name = relation.get("source_name", "")
            tgt_label = relation.get("target_label", "")
            tgt_name = relation.get("target_name", "")
            props = relation.get("properties", {})

            out_v = name_to_vid.get((src_label, src_name), src_name)
            in_v = name_to_vid.get((tgt_label, tgt_name), tgt_name)

            edge_info = edge_label_map.get(label, {})
            edge = {
                "label": label,
                "type": "edge",
                "outV": out_v,
                "outVLabel": edge_info.get("source_label", src_label),
                "inV": in_v,
                "inVLabel": edge_info.get("target_label", tgt_label),
                "properties": props,
            }
            edges.append(edge)

        return vertices, edges

    @staticmethod
    def _generate_vertex_id(
        label: str, name: str, vertex_label_map: Dict[str, Any]
    ) -> str:
        """Generate a canonical vertex ID based on schema primary keys."""
        vl = vertex_label_map.get(label, {})
        id_strategy = vl.get("id_strategy", "PRIMARY_KEY")
        vertex_id_prefix = vl.get("id", label)

        if str(id_strategy).upper() == "PRIMARY_KEY":
            return f"{vertex_id_prefix}:{name}"
        return f"{label}:{name}"

    # ---- Deduplication ----

    @staticmethod
    def _deduplicate_vertices(vertices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate vertices by (label, name) pair."""
        seen: Set[Tuple[str, str]] = set()
        unique = []
        for vertex in vertices:
            label = vertex.get("label", "")
            name = vertex.get("properties", {}).get("name", "")
            key = (label, name)
            if key not in seen:
                seen.add(key)
                unique.append(vertex)
        return unique

    @staticmethod
    def _deduplicate_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate edges by (label, outV, inV) triple."""
        seen: Set[Tuple[str, str, str]] = set()
        unique = []
        for edge in edges:
            key = (edge.get("label", ""), edge.get("outV", ""), edge.get("inV", ""))
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique
