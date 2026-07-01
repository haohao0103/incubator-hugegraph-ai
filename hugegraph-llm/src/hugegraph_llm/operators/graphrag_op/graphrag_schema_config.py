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

"""
GraphRAG Schema Configuration — EDC + Guided mode orchestration.

Implements the schema strategy decided for HugeGraph-AI:
- **Evolving (default)**: EDC three-phase pipeline (Extract → Define → Canonicalize)
  for the KG's own schema evolution. LLM freely extracts entities/relations,
  Define generates semantic definitions for new types, then Canonicalize
  deduplicates/merges synonym types within the KG's own evolving schema
  (e.g., 嫌疑人/嫌疑犯/suspect all map to one unified type).
- **Guided (optional enhancement)**: Schema-constrained extraction with
  Pydantic ResponseModel. Useful for well-defined domains (risk control,
  code graph) where the KG ontology is known a priori.

NOT supported as standalone: Schema-free (no canonicalization) because
it produces type noise that leads to type explosion in the KG's evolving
schema — the same concept gets many names across runs.

Reference: EDC Framework (EMNLP 2024), OpenSPG concept/type separation.

Context keys used by this module:
  IN:
    chunks                — List[str] raw text chunks
    schema                — Optional[Dict] HugeGraph schema dict (vertexlabels/edgelabels)
    graph_rag_schema_mode — Optional[str] "evolving" | "guided" (default: evolving)
    known_type_registry   — Optional[Dict] cached type definitions from prior runs
    known_vertex_types    — Optional[List[str]] known vertex types from the KG's type registry

  OUT:
    graph_rag_schema_mode — Confirmed schema mode
    graph_rag_schema_config — This config instance
    known_type_registry   — Updated registry (with any newly defined types)
    raw_types             — List[str] raw LLM-extracted type strings (before canonicalize)
    canonicalized_types   — Dict[str,str] raw_type → canonical_type mapping
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from hugegraph_llm.utils.log import log


# ============================================================
# Schema Mode Enum
# ============================================================

class SchemaMode(str, Enum):
    """GraphRAG schema extraction mode.

    EVOLVING: EDC three-phase pipeline (default).
        Extract → Define (for new types) → Canonicalize (deduplicate/merge
        synonym types within the KG's own evolving schema). This is the EDC
        pipeline for KG's own schema evolution — preventing type explosion
        where the LLM calls the same concept by different names across runs.

    GUIDED: Schema-constrained extraction.
        LLM output is constrained by Pydantic ResponseModel for well-defined
        domains where the KG ontology is known a priori (risk control,
        code graph, financial regulation).

    NOTE: Schema-free (no canonicalization) is NOT a standalone mode.
    It is the intermediate state within EVOLVING before Canonicalize phase.
    """
    EVOLVING = "evolving"
    GUIDED = "guided"


# ============================================================
# Canonicalize Strategy Enum
# ============================================================

class CanonicalizeStrategy(str, Enum):
    """How to deduplicate/merge synonym types within the KG's own type registry.

    The Canonicalize phase prevents type explosion: when the LLM calls the
    same concept by different names across runs (嫌疑人/嫌疑犯/suspect),
    Canonicalize merges them into one unified type using known types in the
    KG's own type registry as the reference set.

    EMBEDDING_SIM: Vector embedding similarity between type name/description
        and pre-computed known type embeddings from the KG's type registry.
        Requires: known_type_embeddings in context or config.

    EXACT_MATCH: Only exact string match (case-insensitive) is accepted.
        New types that don't match any known type are kept as-is.
        Lowest cost, but lowest deduplication quality.

    LLM_CLASSIFY: LLM classifies each new type into one of the existing
        known types in the KG's type registry. Highest accuracy, highest cost.
    """
    EMBEDDING_SIM = "embedding_sim"
    EXACT_MATCH = "exact_match"
    LLM_CLASSIFY = "llm_classify"


# ============================================================
# Define Trigger Policy
# ============================================================

class DefineTriggerPolicy(str, Enum):
    """When to trigger the Define phase (LLM-generated semantic definitions).

    NEW_TYPES_ONLY: Only trigger Define for types not in known_type_registry.
        First run = heavy LLM calls; stable runs = near-zero extra calls.
        This is the recommended policy for production use.

    ALWAYS: Always generate definitions for all types, even known ones.
        Useful for bootstrapping or when the registry is empty.

    THRESHOLD: Only trigger Define when the ratio of new types exceeds
        a configurable threshold (e.g., >10% of total types are new).
    """
    NEW_TYPES_ONLY = "new_types_only"
    ALWAYS = "always"
    THRESHOLD = "threshold"


# ============================================================
# Configuration Dataclass
# ============================================================

@dataclass
class GraphRAGSchemaConfig:
    """Complete configuration for EDC + Guided schema pipeline.

    This is the single source of truth for all schema-related settings.
    It controls which mode to use, how Define and Canonicalize operate,
    and what thresholds/parameters to apply.

    Usage:
        config = GraphRAGSchemaConfig(mode=SchemaMode.EVOLVING)
        context["graph_rag_schema_config"] = config
        # Downstream operators read config from context
    """

    # --- Mode Selection ---
    mode: SchemaMode = SchemaMode.EVOLVING
    canonicalize_strategy: CanonicalizeStrategy = CanonicalizeStrategy.EMBEDDING_SIM
    define_trigger_policy: DefineTriggerPolicy = DefineTriggerPolicy.NEW_TYPES_ONLY

    # --- Embedding Similarity Thresholds ---
    # Types with similarity >= this threshold are considered a match
    canonicalize_similarity_threshold: float = 0.85
    # Types with similarity >= this but < threshold get a "suggested" mapping
    # (not forced, but recorded for human review)
    canonicalize_suggest_threshold: float = 0.70

    # --- Define Phase Parameters ---
    # Maximum properties to define per entity type
    define_max_properties: int = 10
    # Include example instances in the definition prompt (helps LLM)
    define_include_examples: bool = True

    # --- Guided Mode Parameters ---
    # Maximum entity types allowed in guided ResponseModel
    guided_max_entity_types: int = 20
    # Maximum relation types allowed in guided ResponseModel
    guided_max_relation_types: int = 30
    # Whether to allow dynamic labels in guided mode (strict vs permissive)
    guided_allow_dynamic: bool = False

    # --- Threshold Policy Parameters ---
    # New type ratio threshold for THRESHOLD trigger policy
    define_threshold_ratio: float = 0.10

    # --- Known Type Registry ---
    # Pre-loaded type definitions from prior runs or manual configuration.
    # Dict mapping: type_name → {description, properties, parent_types}
    known_type_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Known Vertex Types ---
    # Known vertex type names from the KG's own type registry (previously
    # extracted types from prior runs). Used by Canonicalize to deduplicate
    # synonym types within the KG's evolving schema.
    known_vertex_types: List[str] = field(default_factory=list)

    # --- Known Type Embeddings ---
    # Pre-computed embedding vectors for each known type in the KG's registry.
    # Dict mapping: type_name → embedding_vector (List[float])
    # Loaded once at initialization; enables EMBEDDING_SIM canonicalize.
    known_type_embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # --- Embedding Model for Canonicalize ---
    # Name of the embedding model to use for computing type embedding similarity.
    # If None, defaults to the system's configured embedding model.
    embedding_model_name: Optional[str] = None

    # --- LLM Role for Define ---
    # Which LLM role to use for the Define phase (defaults to "extractor")
    define_llm_role: str = "extractor"

    # --- Human Override ---
    # Whether human-provided definitions override LLM-generated ones
    allow_manual_override: bool = True
    # Human-provided type definitions that pre-seed the registry.
    # Dict mapping: type_name → {description, properties, parent_types, ...}
    # When allow_manual_override=True and this contains a definition for a new
    # type, the Define phase skips the LLM call and uses this definition instead.
    manual_type_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Canonicalize mappings with confidence below this threshold require human
    # approval. Suggested mappings are already recorded; this adds a gate for
    # what MUST be explicitly approved before being applied.
    require_human_approval_below: float = 0.70

    # --- Misc ---
    # Whether to preserve raw_type alongside canonicalized type
    # (dual output: normalized key for storage + raw_type for EDC pipeline)
    preserve_raw_type: bool = True
    # Whether to log detailed canonicalization decisions
    verbose_logging: bool = False

    def validate(self) -> List[str]:
        """Validate config consistency, return list of issues (empty = valid)."""
        issues = []

        if self.mode == SchemaMode.GUIDED:
            # Guided mode requires known_vertex_types for ResponseModel
            if not self.known_vertex_types:
                issues.append(
                    "Guided mode requires known_vertex_types to build "
                    "Pydantic ResponseModel. Provide the known vertex types "
                    "from the KG's type registry."
                )

        if self.canonicalize_strategy == CanonicalizeStrategy.EMBEDDING_SIM:
            if not self.known_type_embeddings and not self.known_vertex_types:
                issues.append(
                    "EMBEDDING_SIM canonicalize strategy requires either "
                    "known_type_embeddings (pre-computed) or "
                    "known_vertex_types (to compute embeddings at runtime)."
                )

        if self.define_trigger_policy == DefineTriggerPolicy.THRESHOLD:
            if self.define_threshold_ratio <= 0 or self.define_threshold_ratio > 1:
                issues.append(
                    f"define_threshold_ratio must be in (0, 1], got {self.define_threshold_ratio}"
                )

        if self.canonicalize_similarity_threshold < self.canonicalize_suggest_threshold:
            issues.append(
                f"canonicalize_similarity_threshold ({self.canonicalize_similarity_threshold}) "
                f"should be >= canonicalize_suggest_threshold ({self.canonicalize_suggest_threshold})"
            )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for context passing."""
        return {
            "mode": self.mode.value,
            "canonicalize_strategy": self.canonicalize_strategy.value,
            "define_trigger_policy": self.define_trigger_policy.value,
            "canonicalize_similarity_threshold": self.canonicalize_similarity_threshold,
            "canonicalize_suggest_threshold": self.canonicalize_suggest_threshold,
            "define_max_properties": self.define_max_properties,
            "define_include_examples": self.define_include_examples,
            "guided_max_entity_types": self.guided_max_entity_types,
            "guided_max_relation_types": self.guided_max_relation_types,
            "guided_allow_dynamic": self.guided_allow_dynamic,
            "define_threshold_ratio": self.define_threshold_ratio,
            "preserve_raw_type": self.preserve_raw_type,
            "verbose_logging": self.verbose_logging,
            "define_llm_role": self.define_llm_role,
            "embedding_model_name": self.embedding_model_name,
            # Do NOT serialize large dicts (registry, embeddings) by default
            "known_type_registry_count": len(self.known_type_registry),
            "known_vertex_types_count": len(self.known_vertex_types),
            "known_type_embeddings_count": len(self.known_type_embeddings),
            "allow_manual_override": self.allow_manual_override,
            "manual_type_definitions_count": len(self.manual_type_definitions),
            "require_human_approval_below": self.require_human_approval_below,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRAGSchemaConfig":
        """Deserialize config from dict."""
        config = cls()
        if "mode" in data:
            config.mode = SchemaMode(data["mode"])
        if "canonicalize_strategy" in data:
            config.canonicalize_strategy = CanonicalizeStrategy(data["canonicalize_strategy"])
        if "define_trigger_policy" in data:
            config.define_trigger_policy = DefineTriggerPolicy(data["define_trigger_policy"])
        for key in [
            "canonicalize_similarity_threshold", "canonicalize_suggest_threshold",
            "define_max_properties", "define_include_examples",
            "guided_max_entity_types", "guided_max_relation_types",
            "guided_allow_dynamic", "define_threshold_ratio",
            "preserve_raw_type", "verbose_logging",
            "define_llm_role", "embedding_model_name",
            "allow_manual_override", "require_human_approval_below",
        ]:
            if key in data:
                setattr(config, key, data[key])
        # Large dicts — only deserialize if explicitly provided
        if "known_type_registry" in data:
            config.known_type_registry = data["known_type_registry"]
        if "known_vertex_types" in data:
            config.known_vertex_types = data["known_vertex_types"]
        if "known_type_embeddings" in data:
            config.known_type_embeddings = data["known_type_embeddings"]
        if "manual_type_definitions" in data:
            config.manual_type_definitions = data["manual_type_definitions"]
        return config

    def merge_from_context(self, context: Dict[str, Any]) -> "GraphRAGSchemaConfig":
        """Update this config from values found in the pipeline context dict.

        Context may override config settings set at construction time.
        Returns self (mutated) for convenience.
        """
        if "graph_rag_schema_mode" in context:
            mode_str = context["graph_rag_schema_mode"]
            try:
                self.mode = SchemaMode(mode_str)
            except ValueError:
                log.warning("Unknown schema mode '%s', keeping %s", mode_str, self.mode.value)

        if "known_type_registry" in context:
            registry = context["known_type_registry"]
            if isinstance(registry, dict):
                # Merge: new definitions augment existing ones
                for type_name, type_def in registry.items():
                    if type_name not in self.known_type_registry:
                        self.known_type_registry[type_name] = type_def

        if "known_vertex_types" in context:
            types = context["known_vertex_types"]
            if isinstance(types, list):
                self.known_vertex_types = types

        if "known_type_embeddings" in context:
            embeddings = context["known_type_embeddings"]
            if isinstance(embeddings, dict):
                self.known_type_embeddings = embeddings

        if "manual_type_definitions" in context:
            defs = context["manual_type_definitions"]
            if isinstance(defs, dict):
                # Merge: manual definitions augment existing ones
                for type_name, type_def in defs.items():
                    if type_name not in self.manual_type_definitions:
                        self.manual_type_definitions[type_name] = type_def

        return self

    # ---- Operator Protocol ----

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute config initialization via operator protocol.

        Reads context overrides, validates config, writes config back to context.
        This is the first operator in the EDC/Guided pipeline.
        """
        # Merge overrides from context
        self.merge_from_context(context)

        # Validate
        issues = self.validate()
        if issues:
            for issue in issues:
                log.warning("GraphRAGSchemaConfig issue: %s", issue)
            # For non-critical issues, continue; for critical ones, abort
            critical = [i for i in issues if "requires" in i.lower()]
            if critical:
                log.error("Critical config issues, EDC pipeline may fail: %s", critical)

        # Write config back to context
        context["graph_rag_schema_config"] = self
        context["graph_rag_schema_mode"] = self.mode.value

        # Initialize known_type_registry in context if not already present
        if "known_type_registry" not in context:
            context["known_type_registry"] = self.known_type_registry

        log.info(
            "GraphRAGSchemaConfig initialized: mode=%s, canonicalize=%s, "
            "define_trigger=%s, known_types=%d, known_vertex_types=%d",
            self.mode.value,
            self.canonicalize_strategy.value,
            self.define_trigger_policy.value,
            len(self.known_type_registry),
            len(self.known_vertex_types),
        )
        return context
