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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
EDC Define Operator — Generate semantic definitions for new entity/relation types.

Phase 2 of the EDC (Extract → Define → Canonicalize) pipeline.

For each entity/relation type discovered in the Extract phase that is NOT
already in the known_type_registry, this operator calls the LLM to produce:
- A human-readable description of what this type represents
- A list of expected properties (name, type, cardinality, description)
- Example instances to clarify the type's semantics
- Parent type suggestions (for hierarchical classification)

These definitions serve two purposes:
1. They enrich the type with enough semantic information for the Canonicalize
   phase to compute meaningful embeddings for similarity matching.
2. They accumulate in the known_type_registry so that subsequent runs skip
   the Define phase for already-known types (near-zero extra LLM calls).

Trigger policy is controlled by GraphRAGSchemaConfig.define_trigger_policy:
- NEW_TYPES_ONLY (default): Only define types not in known_type_registry.
- ALWAYS: Re-define all types every run (expensive, for bootstrapping).
- THRESHOLD: Define only when new-type ratio exceeds a threshold.

Context keys:
  IN:
    extracted_entities   — List[Dict] entities from Extract phase
    extracted_relations  — List[Dict] relations from Extract phase
    raw_types            — List[str] raw LLM-extracted type strings
    graph_rag_schema_config — GraphRAGSchemaConfig instance
    known_type_registry  — Dict[str, Dict] cached type definitions

  OUT:
    known_type_registry  — Updated registry with newly defined types
    type_definitions     — Dict[str, Dict] definitions generated this run
    define_call_count    — int number of LLM calls made
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.graphrag_op.graphrag_schema_config import (
    DefineTriggerPolicy,
    GraphRAGSchemaConfig,
    SchemaMode,
)
from hugegraph_llm.utils.log import log


# ============================================================
# Define Prompt Templates
# ============================================================

DEFINE_ENTITY_TYPE_PROMPT = """\
You are a knowledge graph schema architect. Given an entity type extracted from
text by an LLM, produce a precise semantic definition that describes what this
type represents in a domain context.

## Entity Type to Define
Type name: {type_name}
Source text samples where this type appeared:
{examples}

## Output Format (JSON only)
{{
  "description": "A clear, precise description of what '{type_name}' entities represent.",
  "properties": [
    {{
      "name": "property_name",
      "type": "string|integer|float|boolean|date|datetime|list<string>",
      "cardinality": "single|optional|multi",
      "required": true|false,
      "description": "What this property represents"
    }}
  ],
  "parent_types": ["list of broader types this could belong to, e.g. 'Entity'"],
  "distinguishing_features": "What makes this type different from similar types"
}}

## Rules
1. The description must be specific enough to differentiate from similar types
   (e.g. "A technology company founded after 2000" vs just "Company").
2. Properties should include at minimum: name (required), description (optional).
3. Maximum {max_properties} properties per type.
4. parent_types should suggest hierarchical classification paths.
5. Output ONLY valid JSON, no markdown, no commentary.
"""

DEFINE_RELATION_TYPE_PROMPT = """\
You are a knowledge graph schema architect. Given a relation/edge type extracted
from text, produce a precise semantic definition.

## Relation Type to Define
Type name: {type_name}
Source text samples where this type appeared:
{examples}

## Output Format (JSON only)
{{
  "description": "A clear description of what the '{type_name}' relation represents.",
  "source_types": ["list of entity types that can be the source"],
  "target_types": ["list of entity types that can be the target"],
  "properties": [
    {{
      "name": "property_name",
      "type": "string|integer|float|boolean|date",
      "cardinality": "single|optional|multi",
      "required": true|false,
      "description": "What this property represents"
    }}
  ],
  "symmetric": false,
  "distinguishing_features": "What makes this relation different from similar ones"
}}

## Rules
1. The description must be precise (e.g. "Founded by" vs just "Related to").
2. source_types and target_types should reference entity types from the graph.
3. Maximum {max_properties} properties per relation type.
4. Output ONLY valid JSON, no markdown, no commentary.
"""

BATCH_DEFINE_PROMPT = """\
You are a knowledge graph schema architect. Define the following entity and
relation types that were newly discovered in text extraction. For each type,
provide a semantic definition with properties.

## Types to Define
{types_list}

## Existing Known Types (for reference, do NOT redefine these)
{known_types}

## Output Format (JSON only)
{{
  "definitions": [
    {{
      "type_name": "...",
      "type_category": "entity|relation",
      "description": "A precise semantic description",
      "properties": [
        {{ "name": "...", "type": "string|integer|float|...", "cardinality": "single|optional|multi", "required": true|false, "description": "..." }}
      ],
      "parent_types": ["..."],
      "distinguishing_features": "..."
    }}
  ]
}}

## Rules
1. Each description must differentiate from similar types.
2. Entity types must include at minimum: name (required, single), description (optional).
3. Relation types must include source_types and target_types.
4. Maximum {max_properties} properties per type.
5. Output ONLY valid JSON.
"""


# ============================================================
# Define Operator
# ============================================================

class KGSchemaDefineOperator:
    """EDC Define phase: generate semantic definitions for new types.

    Usage:
        operator = KGSchemaDefineOperator(llm=my_llm)
        context = operator.run(context)
        # context now has updated known_type_registry and type_definitions
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[GraphRAGSchemaConfig] = None,
    ):
        self.llm = llm
        self.config = config or GraphRAGSchemaConfig()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Define phase.

        Reads: extracted_entities, extracted_relations, raw_types,
               graph_rag_schema_config, known_type_registry from context.
        Writes: known_type_registry, type_definitions, define_call_count.
        """
        # Merge config from context
        config_in_context = context.get("graph_rag_schema_config")
        if config_in_context and isinstance(config_in_context, GraphRAGSchemaConfig):
            self.config = config_in_context

        # Skip Define entirely in Guided mode (schema is pre-defined)
        if self.config.mode == SchemaMode.GUIDED:
            log.info("Schema mode is GUIDED — skipping Define phase")
            context["type_definitions"] = {}
            context["define_call_count"] = 0
            return context

        # Collect all type names from extraction results
        entity_types, relation_types = self._collect_type_names(context)
        all_types = set(entity_types) | set(relation_types)

        # Determine which types need definition
        registry = self._get_registry(context)
        new_types = self._filter_new_types(all_types, registry)

        if not new_types:
            log.info("No new types to define (all %d types already in registry)", len(all_types))
            context["type_definitions"] = {}
            context["define_call_count"] = 0
            return context

        log.info(
            "Define phase: %d new types out of %d total (registry has %d)",
            len(new_types), len(all_types), len(registry),
        )

        # Check trigger policy
        if not self._should_trigger_define(new_types, all_types):
            log.info("Define trigger policy '%s' — skipping definition this run",
                     self.config.define_trigger_policy.value)
            context["type_definitions"] = {}
            context["define_call_count"] = 0
            return context

        # Apply manual overrides: types with human definitions skip LLM call
        definitions: Dict[str, Dict[str, Any]] = {}
        override_count = 0
        if self.config.allow_manual_override and self.config.manual_type_definitions:
            remaining_new = set()
            for type_name in new_types:
                if type_name in self.config.manual_type_definitions:
                    definitions[type_name] = self.config.manual_type_definitions[type_name]
                    registry[type_name] = definitions[type_name]
                    override_count += 1
                    log.info("Manual override applied for type '%s' (skipped LLM Define)", type_name)
                else:
                    remaining_new.add(type_name)
            new_types = remaining_new

        if not new_types:
            log.info(
                "All %d new types covered by manual overrides — no LLM calls needed",
                override_count,
            )
            context["known_type_registry"] = registry
            context["type_definitions"] = definitions
            context["define_call_count"] = 0
            self.config.known_type_registry = registry
            context["graph_rag_schema_config"] = self.config
            return context

        # Collect example text for each new type (those not covered by override)
        type_examples = self._collect_type_examples(new_types, context)

        # Generate definitions via LLM for remaining new types
        llm_definitions, call_count = self._generate_definitions(new_types, type_examples, registry)

        # Merge LLM definitions with any manual override definitions
        definitions.update(llm_definitions)

        # Update registry
        for type_name, type_def in definitions.items():
            registry[type_name] = type_def

        # Write results to context
        context["known_type_registry"] = registry
        context["type_definitions"] = definitions
        context["define_call_count"] = call_count
        # Also update config's registry
        self.config.known_type_registry = registry
        context["graph_rag_schema_config"] = self.config

        log.info(
            "Define phase complete: %d types defined, %d LLM calls, registry now has %d types",
            len(definitions), call_count, len(registry),
        )
        return context

    # ---- Type Collection ----

    def _collect_type_names(
        self, context: Dict[str, Any]
    ) -> Tuple[Set[str], Set[str]]:
        """Collect all entity and relation type names from extraction results."""
        entity_types: Set[str] = set()
        relation_types: Set[str] = set()

        # From extracted_entities (HybridExtractor format)
        for entity in context.get("extracted_entities", []):
            type_name = entity.get("type", entity.get("entity_type", ""))
            if type_name:
                entity_types.add(type_name)

        # From vertices (PropertyGraphExtract format)
        for vertex in context.get("vertices", []):
            label = vertex.get("label", "")
            if label:
                entity_types.add(label)

        # From raw_types (EDC post-processing format)
        for raw_type in context.get("raw_types", []):
            if raw_type:
                entity_types.add(raw_type)

        # From triples (InfoExtract format)
        # triples are (subject, predicate, object) — predicate could be relation type
        for triple in context.get("triples", []):
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                predicate = triple[1]
                if predicate:
                    relation_types.add(predicate)

        # From edges (PropertyGraphExtract format)
        for edge in context.get("edges", []):
            label = edge.get("label", "")
            if label:
                relation_types.add(label)

        # From extracted_relations (HybridExtractor format)
        for relation in context.get("extracted_relations", []):
            predicate = relation.get("predicate", "")
            if predicate:
                relation_types.add(predicate)

        return entity_types, relation_types

    def _get_registry(self, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get known_type_registry from context or config."""
        registry = context.get("known_type_registry", {})
        if not registry:
            registry = self.config.known_type_registry
        return registry

    def _filter_new_types(
        self, all_types: Set[str], registry: Dict[str, Dict[str, Any]]
    ) -> Set[str]:
        """Filter types that are NOT already in the known_type_registry."""
        return {t for t in all_types if t not in registry}

    # ---- Trigger Policy ----

    def _should_trigger_define(
        self, new_types: Set[str], all_types: Set[str]
    ) -> bool:
        """Check if the Define phase should be triggered based on policy."""
        policy = self.config.define_trigger_policy

        if policy == DefineTriggerPolicy.ALWAYS:
            return True

        if policy == DefineTriggerPolicy.NEW_TYPES_ONLY:
            return len(new_types) > 0

        if policy == DefineTriggerPolicy.THRESHOLD:
            if not all_types:
                return False
            ratio = len(new_types) / len(all_types)
            return ratio >= self.config.define_threshold_ratio

        return True  # Default: always trigger if there are new types

    # ---- Example Collection ----

    def _collect_type_examples(
        self, new_types: Set[str], context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Collect example text snippets for each new type from chunks."""
        examples: Dict[str, List[str]] = {t: [] for t in new_types}
        chunks = context.get("chunks", [])

        # Limit examples per type (avoid bloating prompts)
        max_examples = 3

        # From extracted_entities
        for entity in context.get("extracted_entities", []):
            type_name = entity.get("type", entity.get("entity_type", ""))
            if type_name in new_types:
                name = entity.get("name", "")
                if name and len(examples[type_name]) < max_examples:
                    examples[type_name].append(f"Entity: {name} (type: {type_name})")

        # From vertices
        for vertex in context.get("vertices", []):
            label = vertex.get("label", "")
            if label in new_types:
                props = vertex.get("properties", {})
                name = props.get("name", "")
                if name and len(examples[label]) < max_examples:
                    examples[label].append(f"Vertex: {name} (label: {label})")

        # From raw text chunks — find mentions
        for chunk in chunks[:10]:  # Sample first 10 chunks
            chunk_lower = chunk.lower()
            for type_name in new_types:
                if type_name.lower() in chunk_lower and len(examples[type_name]) < max_examples:
                    # Extract a snippet around the mention
                    idx = chunk_lower.find(type_name.lower())
                    start = max(0, idx - 50)
                    end = min(len(chunk), idx + len(type_name) + 50)
                    snippet = chunk[start:end].strip()
                    examples[type_name].append(f"Text snippet: ...{snippet}...")

        return examples

    # ---- LLM Definition Generation ----

    def _generate_definitions(
        self,
        new_types: Set[str],
        type_examples: Dict[str, List[str]],
        registry: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], int]:
        """Generate semantic definitions for new types via LLM.

        For small batches (≤5 types), define individually for precision.
        For larger batches, use batch prompt for efficiency.
        """
        definitions: Dict[str, Dict[str, Any]] = {}
        call_count = 0

        # Decide: batch vs individual
        if len(new_types) <= 5:
            # Individual definitions — more precise
            for type_name in new_types:
                is_relation = self._is_relation_type(type_name)
                examples_str = "\n".join(type_examples.get(type_name, []))
                if not examples_str:
                    examples_str = f"(Type '{type_name}' appeared in extracted data)"

                if is_relation:
                    prompt = DEFINE_RELATION_TYPE_PROMPT.format(
                        type_name=type_name,
                        examples=examples_str,
                        max_properties=self.config.define_max_properties,
                    )
                else:
                    prompt = DEFINE_ENTITY_TYPE_PROMPT.format(
                        type_name=type_name,
                        examples=examples_str,
                        max_properties=self.config.define_max_properties,
                    )

                try:
                    response = self.llm.generate(prompt=prompt)
                    call_count += 1
                    parsed = self._parse_definition_response(response, type_name, is_relation)
                    if parsed:
                        definitions[type_name] = parsed
                    else:
                        # Fallback: minimal definition
                        definitions[type_name] = self._minimal_definition(type_name, is_relation)
                except Exception as e:  # pylint: disable=broad-except
                    log.warning("LLM Define failed for type '%s': %s", type_name, e)
                    definitions[type_name] = self._minimal_definition(type_name, is_relation)
                    call_count += 1
        else:
            # Batch definition — more efficient
            types_list = self._format_types_list(new_types, type_examples)
            known_types_str = "\n".join(
                f"  - {name}: {defn.get('description', 'N/A')}"
                for name, defn in registry.items()
            )[:2000]  # Limit known types in prompt to avoid overflow

            prompt = BATCH_DEFINE_PROMPT.format(
                types_list=types_list,
                known_types=known_types_str or "(No known types yet)",
                max_properties=self.config.define_max_properties,
            )

            try:
                response = self.llm.generate(prompt=prompt)
                call_count += 1
                definitions = self._parse_batch_definition_response(response, new_types)
            except Exception as e:  # pylint: disable=broad-except
                log.warning("LLM batch Define failed: %s", e)
                call_count += 1
                # Fallback: minimal definitions for all new types
                for type_name in new_types:
                    is_relation = self._is_relation_type(type_name)
                    definitions[type_name] = self._minimal_definition(type_name, is_relation)

        return definitions, call_count

    def _is_relation_type(self, type_name: str) -> bool:
        """Heuristic: determine if a type name is likely a relation/edge type.

        Relation types tend to be verbs or verb phrases (created_by, works_at,
        located_in), while entity types tend to be nouns (person, company, city).
        """
        # Common relation type patterns
        relation_patterns = [
            r"_by$", r"_at$", r"_in$", r"_to$", r"_from$", r"_with$",
            r"_of$", r"_for$", r"_on$", r"_into$", r"_between$",
            r"^has_", r"^is_", r"^belongs_to", r"^relates_to",
            r"^created", r"^founded", r"^located", r"^works",
            r"^manages", r"^owns", r"^employs", r"^connects",
        ]
        for pattern in relation_patterns:
            if re.search(pattern, type_name.lower()):
                return True

        # Chinese relation patterns
        cn_relation_suffixes = ["关系", "关联", "连接", "属于", "创建", "拥有", "管理"]
        for suffix in cn_relation_suffixes:
            if type_name.endswith(suffix):
                return True

        # Single-word verb-like types
        verb_like = {
            "relates_to", "belongs_to", "has_chunk", "contains",
            "describes", "references", "manages", "owns",
            "employs", "creates", "connects", "links",
        }
        if type_name.lower() in verb_like:
            return True

        return False

    @staticmethod
    def _minimal_definition(
        type_name: str, is_relation: bool
    ) -> Dict[str, Any]:
        """Generate a minimal definition as fallback when LLM fails."""
        if is_relation:
            return {
                "type_category": "relation",
                "description": f"Relation type '{type_name}' (auto-generated minimal definition)",
                "source_types": ["Entity"],
                "target_types": ["Entity"],
                "properties": [],
                "parent_types": [],
            }
        return {
            "type_category": "entity",
            "description": f"Entity type '{type_name}' (auto-generated minimal definition)",
            "properties": [
                {"name": "name", "type": "string", "cardinality": "single",
                 "required": True, "description": "Entity name"},
            ],
            "parent_types": ["Entity"],
        }

    def _format_types_list(
        self,
        new_types: Set[str],
        type_examples: Dict[str, List[str]],
    ) -> str:
        """Format new types and their examples for the batch prompt."""
        lines = []
        for type_name in sorted(new_types):
            is_relation = self._is_relation_type(type_name)
            category = "relation" if is_relation else "entity"
            examples = type_examples.get(type_name, [])
            examples_str = "; ".join(examples[:2]) if examples else "(appeared in extracted data)"
            lines.append(
                f"  - {type_name} ({category}): {examples_str}"
            )
        return "\n".join(lines)

    # ---- Response Parsing ----

    def _parse_definition_response(
        self, response: str, type_name: str, is_relation: bool
    ) -> Optional[Dict[str, Any]]:
        """Parse a single type definition from LLM response."""
        # Strip markdown code blocks
        response = re.sub(r"```\w*\n?", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()

        # Try to extract JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            log.warning("No JSON found in Define response for type '%s'", type_name)
            return None

        try:
            data = json.loads(json_match.group(0).strip())
        except json.JSONDecodeError:
            log.warning("Invalid JSON in Define response for type '%s'", type_name)
            return None

        # Validate and normalize
        definition = {
            "type_category": "relation" if is_relation else "entity",
            "description": data.get("description", ""),
            "properties": data.get("properties", []),
            "parent_types": data.get("parent_types", []),
        }

        if is_relation:
            definition["source_types"] = data.get("source_types", ["Entity"])
            definition["target_types"] = data.get("target_types", ["Entity"])
            definition["symmetric"] = data.get("symmetric", False)

        # Ensure description is non-empty
        if not definition["description"]:
            definition["description"] = f"Type '{type_name}' extracted from text"

        return definition

    def _parse_batch_definition_response(
        self, response: str, expected_types: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse a batch definition response from LLM."""
        # Strip markdown code blocks
        response = re.sub(r"```\w*\n?", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()

        # Try to extract JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            log.warning("No JSON found in batch Define response")
            return {}

        try:
            data = json.loads(json_match.group(0).strip())
        except json.JSONDecodeError:
            log.warning("Invalid JSON in batch Define response")
            return {}

        definitions_list = data.get("definitions", [])
        if not definitions_list:
            # Try flat format: each key is a type name
            if isinstance(data, dict) and not "definitions" in data:
                definitions_list = [
                    {"type_name": k, **v} for k, v in data.items()
                    if isinstance(v, dict)
                ]

        definitions: Dict[str, Dict[str, Any]] = {}
        for item in definitions_list:
            if not isinstance(item, dict):
                continue
            type_name = item.get("type_name", "")
            if not type_name:
                continue

            is_relation = item.get("type_category", "") == "relation"
            if not is_relation:
                is_relation = self._is_relation_type(type_name)

            definition = {
                "type_category": "relation" if is_relation else "entity",
                "description": item.get("description", ""),
                "properties": item.get("properties", []),
                "parent_types": item.get("parent_types", []),
            }

            if is_relation:
                definition["source_types"] = item.get("source_types", ["Entity"])
                definition["target_types"] = item.get("target_types", ["Entity"])

            if not definition["description"]:
                definition["description"] = f"Type '{type_name}' extracted from text"

            definitions[type_name] = definition

        # Fill in any expected types that weren't in the response
        for type_name in expected_types:
            if type_name not in definitions:
                is_relation = self._is_relation_type(type_name)
                definitions[type_name] = self._minimal_definition(type_name, is_relation)
                log.warning(
                    "Type '%s' not found in batch Define response, using minimal definition",
                    type_name,
                )

        return definitions
