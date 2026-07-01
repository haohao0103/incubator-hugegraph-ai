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
EDC Canonicalize Operator — Deduplicate/merge synonym types within the
KG's own evolving schema via embedding similarity.

Phase 3 of the EDC (Extract → Define → Canonicalize) pipeline.

After the Extract phase freely generates entity/relation types and the
Define phase enriches them with semantic definitions, Canonicalize maps
each LLM-generated type to the closest known type in the KG's own type
registry. This prevents type explosion — when the LLM calls the same
concept by different names across runs (嫌疑人/嫌疑犯/suspect),
Canonicalize merges them into one unified type, keeping the KG schema
compact and consistent.

Three canonicalization strategies (controlled by config):
- EMBEDDING_SIM (default): Compute embedding similarity between type
  name + description and pre-computed known type embeddings from the
  KG's type registry. Threshold-controlled: high similarity → forced
  mapping, moderate → suggested mapping, low → keep original type name.
- EXACT_MATCH: Only case-insensitive string matching. Zero embedding cost.
- LLM_CLASSIFY: LLM classifies each type into one of the existing known
  types in the KG's type registry. Highest accuracy, highest cost.

Design decision: Pre-computed embeddings stored in the KG's type registry,
not online queries. Known type count is limited (typically ~50 in a domain
KG), so one-time initialization suffices.

Context keys:
  IN:
    raw_types               — List[str] raw LLM-extracted type strings
    type_definitions        — Dict[str, Dict] semantic definitions from Define
    graph_rag_schema_config — GraphRAGSchemaConfig instance
    known_vertex_types      — List[str] known vertex types from the KG's type registry
    known_type_embeddings   — Dict[str, List[float]] pre-computed embeddings

  OUT:
    canonicalized_types     — Dict[str, str] raw_type → canonical_type mapping
    canonicalize_details    — Dict[str, Dict] per-type canonicalization details
    canonicalize_suggestions — Dict[str, str] suggested but not forced mappings
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.graphrag_op.graphrag_schema_config import (
    CanonicalizeStrategy,
    GraphRAGSchemaConfig,
    SchemaMode,
)
from hugegraph_llm.utils.log import log


# ============================================================
# LLM Classification Prompt
# ============================================================

LLM_CLASSIFY_PROMPT = """\
You are a knowledge graph schema classifier. Given a list of new entity/relation
types discovered by LLM extraction, classify each one into the most appropriate
existing KG vertex type from the type registry, or mark it as "NEW" if no
existing type is a good match.

## New Types to Classify
{new_types}

## Existing KG Vertex Types (from type registry)
{existing_types}

## Type Definitions (for context)
{type_definitions}

## Output Format (JSON only)
{{
  "classifications": [
    {{
      "new_type": "type_name_from_new_types",
      "classified_as": "exact_existing_type_name_or_NEW",
      "confidence": 0.0_to_1.0,
      "reason": "Brief explanation of why this mapping was chosen"
    }}
  ]
}}

## Rules
1. classified_as must be an EXACT string from existing_types list, or "NEW".
2. Only classify as an existing type if the semantic meaning clearly matches.
3. If uncertain, prefer "NEW" over a weak match.
4. Output ONLY valid JSON.
"""


# ============================================================
# Canonicalize Operator
# ============================================================

class KGSchemaCanonicalizeOperator:
    """EDC Canonicalize phase: deduplicate/merge synonym types within the KG's
    own type registry.

    Prevents type explosion where the LLM calls the same concept by different
    names across runs (嫌疑人/嫌疑犯/suspect → unified "suspect" type).

    Usage:
        operator = KGSchemaCanonicalizeOperator(llm=my_llm, embed_func=my_embed)
        context = operator.run(context)
        # context now has canonicalized_types mapping
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embed_func: Optional[Any] = None,
        config: Optional[GraphRAGSchemaConfig] = None,
    ):
        """
        :param llm: LLM instance (required for LLM_CLASSIFY strategy).
        :param embed_func: Embedding function that takes a string and returns
                           a List[float] vector. Required for EMBEDDING_SIM
                           strategy unless pre-computed embeddings are provided.
        :param config: GraphRAGSchemaConfig instance.
        """
        self.llm = llm
        self.embed_func = embed_func
        self.config = config or GraphRAGSchemaConfig()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Canonicalize phase.

        Reads: raw_types, type_definitions, graph_rag_schema_config,
               known_vertex_types, known_type_embeddings.
        Writes: canonicalized_types, canonicalize_details,
                canonicalize_suggestions.
        """
        # Merge config from context
        config_in_context = context.get("graph_rag_schema_config")
        if config_in_context and isinstance(config_in_context, GraphRAGSchemaConfig):
            self.config = config_in_context

        # Skip Canonicalize in Guided mode (types are already constrained)
        if self.config.mode == SchemaMode.GUIDED:
            log.info("Schema mode is GUIDED — skipping Canonicalize phase")
            context["canonicalized_types"] = {}
            context["canonicalize_details"] = {}
            context["canonicalize_suggestions"] = {}
            return context

        # Get raw types to canonicalize
        raw_types = set(context.get("raw_types", []))

        # Also collect type names from extraction results if raw_types is empty
        if not raw_types:
            raw_types = self._collect_types_from_extraction(context)

        if not raw_types:
            log.info("No types to canonicalize")
            context["canonicalized_types"] = {}
            context["canonicalize_details"] = {}
            context["canonicalize_suggestions"] = {}
            return context

        # Get known types from the KG's type registry
        known_types = self._get_known_types(context)

        if not known_types:
            log.warning(
                "No known_vertex_types provided — Canonicalize cannot "
                "deduplicate types. All types will remain as-is."
            )
            # No alignment possible: each type maps to itself
            canonicalized = {t: t for t in raw_types}
            context["canonicalized_types"] = canonicalized
            context["canonicalize_details"] = {
                t: {"strategy": "no_target_types", "confidence": 0.0}
                for t in raw_types
            }
            context["canonicalize_suggestions"] = {}
            return context

        # Execute canonicalization based on strategy
        strategy = self.config.canonicalize_strategy

        if strategy == CanonicalizeStrategy.EXACT_MATCH:
            canonicalized, details, suggestions = self._exact_match(raw_types, known_types)
        elif strategy == CanonicalizeStrategy.EMBEDDING_SIM:
            canonicalized, details, suggestions = self._embedding_similarity(
                raw_types, known_types, context
            )
        elif strategy == CanonicalizeStrategy.LLM_CLASSIFY:
            canonicalized, details, suggestions = self._llm_classify(
                raw_types, known_types, context
            )
        else:
            log.warning("Unknown canonicalize strategy '%s', falling back to EXACT_MATCH",
                        strategy.value)
            canonicalized, details, suggestions = self._exact_match(raw_types, known_types)

        # Write results to context
        context["canonicalized_types"] = canonicalized
        context["canonicalize_details"] = details
        context["canonicalize_suggestions"] = suggestions

        # Update entities/vertices with canonicalized labels if preserve_raw_type is True
        if self.config.preserve_raw_type:
            self._apply_canonicalization_to_entities(context, canonicalized)

        log.info(
            "Canonicalize phase complete: %d types processed, %d mapped, "
            "%d suggested, %d unchanged. Strategy: %s",
            len(raw_types),
            sum(1 for k, v in canonicalized.items() if k != v),
            len(suggestions),
            sum(1 for k, v in canonicalized.items() if k == v),
            strategy.value,
        )
        return context

    # ---- Type Collection ----

    def _collect_types_from_extraction(
        self, context: Dict[str, Any]
    ) -> Set[str]:
        """Collect type names from extraction results when raw_types is empty."""
        types: Set[str] = set()

        for entity in context.get("extracted_entities", []):
            type_name = entity.get("type", entity.get("entity_type", ""))
            if type_name:
                types.add(type_name)

        for vertex in context.get("vertices", []):
            label = vertex.get("label", "")
            if label:
                types.add(label)

        return types

    def _get_known_types(
        self, context: Dict[str, Any]
    ) -> List[str]:
        """Get known vertex type names from the KG's type registry (context or config)."""
        types = context.get("known_vertex_types", [])
        if not types:
            types = self.config.known_vertex_types
        return types

    # ---- Exact Match Strategy ----

    def _exact_match(
        self,
        raw_types: Set[str],
        known_types: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, str]]:
        """Case-insensitive exact string matching.

        Types that match exactly are mapped. Types that don't match any
        known type remain as-is.
        """
        # Build lowercase lookup
        known_types_lower = {t.lower(): t for t in known_types}

        canonicalized: Dict[str, str] = {}
        details: Dict[str, Dict] = {}
        suggestions: Dict[str, str] = {}

        for raw_type in raw_types:
            matched = known_types_lower.get(raw_type.lower())
            if matched:
                canonicalized[raw_type] = matched
                details[raw_type] = {
                    "strategy": "exact_match",
                    "confidence": 1.0,
                    "matched_to": matched,
                }
            else:
                canonicalized[raw_type] = raw_type  # Keep as-is
                details[raw_type] = {
                    "strategy": "exact_match",
                    "confidence": 0.0,
                    "matched_to": None,
                    "reason": "No exact match found",
                }

        return canonicalized, details, suggestions

    # ---- Embedding Similarity Strategy ----

    def _embedding_similarity(
        self,
        raw_types: Set[str],
        known_types: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, str]]:
        """Vector embedding similarity between type descriptions and
        pre-computed known type embeddings from the KG's type registry.

        Steps:
        1. Get pre-computed embeddings for known types
        2. Compute embeddings for each raw type (name + description)
        3. Compute cosine similarity between each raw type and each known type
        4. Apply threshold: similarity >= threshold → forced mapping,
           similarity >= suggest_threshold → suggested mapping,
           below both → keep original type
        """
        # Get pre-computed embeddings
        known_embeddings = self._get_known_embeddings(context)
        if not known_embeddings:
            log.warning(
                "No known_type_embeddings available for "
                "EMBEDDING_SIM strategy. Falling back to EXACT_MATCH."
            )
            return self._exact_match(raw_types, known_types)

        # Compute embeddings for raw types
        type_definitions = context.get("type_definitions", {})
        raw_embeddings = self._compute_raw_type_embeddings(raw_types, type_definitions)

        if not raw_embeddings:
            log.warning("Failed to compute embeddings for raw types, falling back to EXACT_MATCH")
            return self._exact_match(raw_types, known_types)

        # Compute cosine similarity matrix
        threshold = self.config.canonicalize_similarity_threshold
        suggest_threshold = self.config.canonicalize_suggest_threshold

        canonicalized: Dict[str, str] = {}
        details: Dict[str, Dict] = {}
        suggestions: Dict[str, str] = {}

        for raw_type in raw_types:
            raw_emb = raw_embeddings.get(raw_type)
            if raw_emb is None:
                canonicalized[raw_type] = raw_type
                details[raw_type] = {
                    "strategy": "embedding_sim",
                    "confidence": 0.0,
                    "reason": "No embedding computed for this type",
                }
                continue

            # Find best matching known type
            best_match: Optional[str] = None
            best_sim: float = 0.0
            all_sims: Dict[str, float] = {}

            for known_type, known_emb in known_embeddings.items():
                sim = self._cosine_similarity(raw_emb, known_emb)
                all_sims[known_type] = sim
                if sim > best_sim:
                    best_sim = sim
                    best_match = known_type

            if best_match and best_sim >= threshold:
                # High confidence match → forced mapping
                canonicalized[raw_type] = best_match
                details[raw_type] = {
                    "strategy": "embedding_sim",
                    "confidence": best_sim,
                    "matched_to": best_match,
                    "all_similarities": all_sims,
                }
            elif best_match and best_sim >= suggest_threshold:
                # Moderate confidence → suggested mapping (not forced)
                canonicalized[raw_type] = raw_type  # Keep original
                suggestions[raw_type] = best_match
                details[raw_type] = {
                    "strategy": "embedding_sim",
                    "confidence": best_sim,
                    "suggested_match": best_match,
                    "all_similarities": all_sims,
                    "reason": f"Similarity {best_sim:.3f} between suggest ({suggest_threshold}) "
                              f"and force ({threshold}) thresholds",
                }
            else:
                # Low confidence → keep original type
                canonicalized[raw_type] = raw_type
                details[raw_type] = {
                    "strategy": "embedding_sim",
                    "confidence": best_sim,
                    "all_similarities": all_sims,
                    "reason": f"Best similarity {best_sim:.3f} below suggest threshold "
                              f"{suggest_threshold}",
                }

        return canonicalized, details, suggestions

    def _get_known_embeddings(
        self, context: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Get pre-computed embeddings for known types in the KG's type registry.

        Priority:
        1. From context (known_type_embeddings)
        2. From config (known_type_embeddings)
        3. Compute at runtime using embed_func (expensive, last resort)
        """
        embeddings = context.get("known_type_embeddings", {})
        if not embeddings:
            embeddings = self.config.known_type_embeddings

        if embeddings:
            return embeddings

        # Last resort: compute embeddings at runtime
        if self.embed_func:
            known_types = self._get_known_types(context)
            log.info("Computing known type embeddings at runtime (%d types)",
                     len(known_types))
            embeddings = {}
            for type_name in known_types:
                try:
                    embeddings[type_name] = self.embed_func(type_name)
                except Exception as e:  # pylint: disable=broad-except
                    log.warning("Failed to compute embedding for type '%s': %s", type_name, e)
            return embeddings

        return {}

    def _compute_raw_type_embeddings(
        self,
        raw_types: Set[str],
        type_definitions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        """Compute embeddings for each raw type using name + description.

        The embedding source is: type_name + description from Define phase.
        This provides richer semantic information than just the type name alone.
        """
        if not self.embed_func:
            log.warning("No embed_func available — cannot compute raw type embeddings")
            return {}

        embeddings: Dict[str, List[float]] = {}
        for type_name in raw_types:
            # Build embedding text: type name + description
            definition = type_definitions.get(type_name, {})
            description = definition.get("description", "")
            embed_text = f"{type_name}: {description}" if description else type_name

            try:
                embeddings[type_name] = self.embed_func(embed_text)
            except Exception as e:  # pylint: disable=broad-except
                log.warning("Failed to compute embedding for raw type '%s': %s", type_name, e)

        return embeddings

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ---- LLM Classification Strategy ----

    def _llm_classify(
        self,
        raw_types: Set[str],
        known_types: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, str]]:
        """LLM classifies each raw type into an existing known type from the
        KG's type registry.

        Most accurate but most expensive strategy.
        """
        if self.llm is None:
            log.warning("No LLM available for LLM_CLASSIFY strategy, falling back to EXACT_MATCH")
            return self._exact_match(raw_types, known_types)

        type_definitions = context.get("type_definitions", {})

        # Format new types with definitions
        new_types_str = "\n".join(
            f"  - {t}: {type_definitions.get(t, {}).get('description', 'N/A')}"
            for t in sorted(raw_types)
        )
        existing_types_str = "\n".join(f"  - {t}" for t in known_types)

        # Include type definitions for context
        defs_str = "\n".join(
            f"  {t}: {defn.get('description', 'N/A')[:100]}"
            for t, defn in type_definitions.items()
        )[:2000]

        prompt = LLM_CLASSIFY_PROMPT.format(
            new_types=new_types_str,
            existing_types=existing_types_str,
            type_definitions=defs_str or "(No definitions available)",
        )

        try:
            response = self.llm.generate(prompt=prompt)
            classifications = self._parse_llm_classify_response(response, raw_types, known_types)
        except Exception as e:  # pylint: disable=broad-except
            log.warning("LLM_CLASSIFY failed: %s, falling back to EXACT_MATCH", e)
            return self._exact_match(raw_types, known_types)

        canonicalized: Dict[str, str] = {}
        details: Dict[str, Dict] = {}
        suggestions: Dict[str, str] = {}

        for raw_type in raw_types:
            classification = classifications.get(raw_type)
            if classification:
                classified_as = classification.get("classified_as", "NEW")
                confidence = classification.get("confidence", 0.0)
                reason = classification.get("reason", "")

                if classified_as == "NEW" or classified_as not in known_types:
                    # LLM decided this is a genuinely new type
                    canonicalized[raw_type] = raw_type
                    details[raw_type] = {
                        "strategy": "llm_classify",
                        "confidence": confidence,
                        "classified_as": "NEW",
                        "reason": reason,
                    }
                elif confidence >= self.config.canonicalize_similarity_threshold:
                    # High confidence LLM classification → forced mapping
                    canonicalized[raw_type] = classified_as
                    details[raw_type] = {
                        "strategy": "llm_classify",
                        "confidence": confidence,
                        "matched_to": classified_as,
                        "reason": reason,
                    }
                elif confidence >= self.config.canonicalize_suggest_threshold:
                    # Moderate confidence → suggested mapping
                    canonicalized[raw_type] = raw_type
                    suggestions[raw_type] = classified_as
                    details[raw_type] = {
                        "strategy": "llm_classify",
                        "confidence": confidence,
                        "suggested_match": classified_as,
                        "reason": reason,
                    }
                else:
                    # Low confidence → keep original
                    canonicalized[raw_type] = raw_type
                    details[raw_type] = {
                        "strategy": "llm_classify",
                        "confidence": confidence,
                        "reason": reason,
                    }
            else:
                # No classification found for this type
                canonicalized[raw_type] = raw_type
                details[raw_type] = {
                    "strategy": "llm_classify",
                    "confidence": 0.0,
                    "reason": "Not found in LLM response",
                }

        return canonicalized, details, suggestions

    def _parse_llm_classify_response(
        self,
        response: str,
        expected_types: Set[str],
        known_types: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Parse LLM classification response."""
        import json
        import re

        # Strip markdown code blocks
        response = re.sub(r"```\w*\n?", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            log.warning("No JSON found in LLM_CLASSIFY response")
            return {}

        try:
            data = json.loads(json_match.group(0).strip())
        except json.JSONDecodeError:
            log.warning("Invalid JSON in LLM_CLASSIFY response")
            return {}

        classifications_list = data.get("classifications", [])
        if not classifications_list and isinstance(data, dict):
            # Try flat format
            classifications_list = [
                {"new_type": k, **v} for k, v in data.items() if isinstance(v, dict)
            ]

        classifications: Dict[str, Dict[str, Any]] = {}
        for item in classifications_list:
            if not isinstance(item, dict):
                continue
            new_type = item.get("new_type", "")
            if new_type in expected_types:
                classifications[new_type] = {
                    "classified_as": item.get("classified_as", "NEW"),
                    "confidence": float(item.get("confidence", 0.0)),
                    "reason": item.get("reason", ""),
                }

        return classifications

    # ---- Apply Canonicalization to Entities ----

    def _apply_canonicalization_to_entities(
        self,
        context: Dict[str, Any],
        canonicalized: Dict[str, str],
    ) -> None:
        """Apply canonicalization mapping to entities and vertices in context.

        When preserve_raw_type=True, this adds a 'canonical_label' field to
        each entity/vertex alongside the original 'label' (or 'type') field.
        The original field is preserved for storage key normalization,
        while canonical_label is used for EDC pipeline alignment.
        """
        if not canonicalized:
            return

        # Apply to extracted_entities (HybridExtractor format)
        for entity in context.get("extracted_entities", []):
            raw_type = entity.get("type", entity.get("entity_type", ""))
            if raw_type in canonicalized:
                canonical_type = canonicalized[raw_type]
                if raw_type != canonical_type:
                    entity["raw_type"] = raw_type
                    entity["type"] = canonical_type
                    if "entity_type" in entity:
                        entity["entity_type"] = canonical_type

        # Apply to vertices (PropertyGraphExtract format)
        for vertex in context.get("vertices", []):
            raw_label = vertex.get("label", "")
            if raw_label in canonicalized:
                canonical_label = canonicalized[raw_label]
                if raw_label != canonical_label:
                    vertex["raw_label"] = raw_label
                    vertex["label"] = canonical_label

        # Apply to edges — update label if relation type was canonicalized
        for edge in context.get("edges", []):
            raw_label = edge.get("label", "")
            if raw_label in canonicalized:
                canonical_label = canonicalized[raw_label]
                if raw_label != canonical_label:
                    edge["raw_label"] = raw_label
                    edge["label"] = canonical_label
