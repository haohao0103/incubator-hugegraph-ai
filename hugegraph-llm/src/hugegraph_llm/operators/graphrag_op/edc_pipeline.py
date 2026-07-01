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
EDC Pipeline Orchestrator — Coordinates the Extract → Define → Canonicalize
three-phase pipeline for evolving schema mode, and routes to Guided mode
when configured.

This is the top-level entry point for the GraphRAG schema-aware extraction
system. It reads the schema mode from config and orchestrates the appropriate
pipeline:

- EVOLVING mode: Extract (existing extractor) → Define (KGSchemaDefineOperator)
  → Canonicalize (KGSchemaCanonicalizeOperator)
- GUIDED mode: GuidedExtractOperator (schema-constrained extraction)

The orchestrator also adds the EDC post-processing step to the Extract phase
that preserves raw_type alongside normalized labels, enabling the downstream
Define and Canonicalize phases to operate on the original LLM output.

Context keys (full pipeline):
  IN:
    chunks                — List[str] raw text chunks
    schema                — Optional[Dict] HugeGraph schema dict
    graph_rag_schema_mode — Optional[str] "evolving" | "guided"
    graph_rag_schema_config — Optional[GraphRAGSchemaConfig]
    known_type_registry   — Optional[Dict] cached type definitions
    known_vertex_types    — Optional[List[str]] known vertex types from KG registry
    known_type_embeddings — Optional[Dict] pre-computed embeddings

  OUT (evolving mode):
    raw_types             — List[str] raw LLM-extracted type strings
    type_definitions      — Dict[str, Dict] semantic definitions from Define
    canonicalized_types   — Dict[str, str] raw_type → canonical_type mapping
    canonicalize_details  — Dict[str, Dict] per-type details
    known_type_registry   — Updated registry
    vertices, edges       — Extracted graph elements (canonicalized labels)

  OUT (guided mode):
    vertices, edges       — Schema-constrained graph elements
    extracted_entities, extracted_relations — Raw extraction output
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.graphrag_op.graphrag_schema_config import (
    GraphRAGSchemaConfig,
    SchemaMode,
)
from hugegraph_llm.operators.graphrag_op.kg_schema_canonicalize import (
    KGSchemaCanonicalizeOperator,
)
from hugegraph_llm.operators.graphrag_op.kg_schema_define import (
    KGSchemaDefineOperator,
)
from hugegraph_llm.operators.graphrag_op.guided_extract import (
    GuidedExtractOperator,
)
from hugegraph_llm.utils.log import log


class EDCPipelineOrchestrator:
    """Coordinates the EDC three-phase pipeline or routes to Guided mode.

    Usage:
        orchestrator = EDCPipelineOrchestrator(
            llm=my_llm,
            embed_func=my_embed_func,
            config=my_config,
        )
        context = orchestrator.run(context)
    """

    def __init__(
        self,
        llm: BaseLLM,
        embed_func: Optional[Any] = None,
        config: Optional[GraphRAGSchemaConfig] = None,
    ):
        """
        :param llm: LLM instance for Define and Guided extraction.
        :param embed_func: Embedding function for Canonicalize (EMBEDDING_SIM).
        :param config: GraphRAGSchemaConfig instance.
        """
        self.llm = llm
        self.embed_func = embed_func
        self.config = config or GraphRAGSchemaConfig()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full schema-aware extraction pipeline.

        Steps:
        1. Initialize config from context (GraphRAGSchemaConfig.run)
        2. Route based on mode:
           - EVOLVING: Extract → add raw_types → Define → Canonicalize
           - GUIDED: GuidedExtractOperator (schema-constrained)
        3. Return enriched context
        """
        # Phase 0: Config initialization
        config_op = self.config  # Use config as operator (it has run method)
        context = config_op.run(context)

        mode = self.config.mode

        log.info("EDC Pipeline starting: mode=%s", mode.value)

        if mode == SchemaMode.GUIDED:
            return self._run_guided(context)
        elif mode == SchemaMode.EVOLVING:
            return self._run_evolving(context)
        else:
            log.error("Unknown schema mode: %s", mode.value)
            return context

    # ---- Evolving Mode Pipeline ----

    def _run_evolving(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the EDC three-phase pipeline for evolving mode.

        1. Extract: Use existing extractor (PropertyGraphExtract or HybridExtractor)
           The Extract phase is assumed to have already been run by the pipeline.
           We add a post-processing step that preserves raw_type.

        2. Define: KGSchemaDefineOperator generates semantic definitions
           for new types not in known_type_registry.

        3. Canonicalize: KGSchemaCanonicalizeOperator deduplicates/merges
           synonym types within the KG's own type registry.
        """
        # Phase 1: Extract post-processing — collect raw types
        context = self._extract_post_process(context)

        # Phase 2: Define
        define_op = KGSchemaDefineOperator(llm=self.llm, config=self.config)
        context = define_op.run(context)

        # Phase 3: Canonicalize
        canonicalize_op = KGSchemaCanonicalizeOperator(
            llm=self.llm,
            embed_func=self.embed_func,
            config=self.config,
        )
        context = canonicalize_op.run(context)

        log.info(
            "EDC Evolving pipeline complete: raw_types=%d, defined=%d, "
            "canonicalized=%d, suggestions=%d",
            len(context.get("raw_types", [])),
            len(context.get("type_definitions", {})),
            len(context.get("canonicalized_types", {})),
            len(context.get("canonicalize_suggestions", {})),
        )
        return context

    def _extract_post_process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extraction results to preserve raw_type for EDC pipeline.

        This is the additive step that DOES NOT replace existing normalization
        (lowercase+strip for storage key). Instead, it adds a parallel output:
        - Normalized label/type: used as storage key (existing behavior)
        - raw_type: preserved for EDC pipeline (Define + Canonicalize phases)

        The dual output ensures backward compatibility with existing graph
        storage while enabling the EDC pipeline to work on original LLM output.
        """
        raw_types: List[str] = []

        # From extracted_entities (HybridExtractor format)
        for entity in context.get("extracted_entities", []):
            raw_type = entity.get("type", entity.get("entity_type", ""))
            if raw_type:
                raw_types.append(raw_type)
                # Add raw_type field alongside normalized type
                if "raw_type" not in entity:
                    entity["raw_type"] = raw_type

        # From vertices (PropertyGraphExtract format)
        for vertex in context.get("vertices", []):
            label = vertex.get("label", "")
            if label:
                raw_types.append(label)
                # Add raw_label field alongside normalized label
                if "raw_label" not in vertex:
                    vertex["raw_label"] = label

        # From triples (InfoExtract format)
        for triple in context.get("triples", []):
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                predicate = triple[1]
                if predicate:
                    raw_types.append(predicate)

        # From edges
        for edge in context.get("edges", []):
            label = edge.get("label", "")
            if label:
                raw_types.append(label)

        # Deduplicate raw_types (preserve all unique type names)
        context["raw_types"] = list(set(raw_types))

        log.info(
            "Extract post-processing: %d unique raw types collected from "
            "%d entities, %d vertices, %d edges",
            len(context["raw_types"]),
            len(context.get("extracted_entities", [])),
            len(context.get("vertices", [])),
            len(context.get("edges", [])),
        )
        return context

    # ---- Guided Mode Pipeline ----

    def _run_guided(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schema-constrained extraction for guided mode."""
        guided_op = GuidedExtractOperator(llm=self.llm, config=self.config)
        context = guided_op.run(context)

        log.info(
            "Guided extraction complete: %d vertices, %d edges",
            len(context.get("vertices", [])),
            len(context.get("edges", [])),
        )
        return context
