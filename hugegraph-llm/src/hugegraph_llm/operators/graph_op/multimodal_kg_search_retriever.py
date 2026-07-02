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

"""Multimodal KG Search Retriever: fuse MultiModalRetriever with KGSearchRetriever.

Bridges the existing four-channel MultiModalRetriever (vector + BM25 + vision + graph)
with the N-hop KGSearchRetriever so that:

- A text query is handled by KGSearchRetriever as before.
- An image (or image + text) query first runs through the multimodal search layer
  to discover image-matched entities/chunks.
- Those image-matched entities are injected as external seeds into KGSearchRetriever
  for cross-modal graph propagation.
- The final result combines text-based KG results with image-derived context and
  graph-discovered neighbors.

This module is intentionally thin: it does not duplicate the heavy vision/BM25/vector
logic that lives in MultiModalRetriever; it only orchestrates the fusion between the
multimodal search layer and the KG search layer.

Design references:
    - LightRAG: multimodal context integration via entity->chunk source_id edges
    - MS-GraphRAG: claim extraction across modalities (images, tables, text)
    - HippoRAG2: multi-modal entity propagation for multi-hop reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.graph_op.kg_search_retriever import (
    KGSearchResult,
    KGSearchRetriever,
    ScoredChunk,
    ScoredEntity,
)
from hugegraph_llm.operators.llm_op.query_rewrite import QueryRewriteResult
from hugegraph_llm.utils.log import log


logger = logging.getLogger(__name__)


@dataclass
class MultimodalQuery:
    """Input for multimodal KG search retrieval."""

    text: str = ""
    image_base64: str = ""  # Base64 encoded image data (optional)

    # "auto" detects image presence; "text" forces text-only; "image" forces image-aware
    query_type: str = "auto"

    @property
    def is_text_only(self) -> bool:
        return self.query_type == "text" or (not self.image_base64 and self.query_type == "auto")

    @property
    def has_image(self) -> bool:
        return bool(self.image_base64)


@dataclass
class MultimodalItem:
    """Single item from the multimodal search layer."""

    id: str
    label: str = ""
    score: float = 0.0
    source_type: str = "text"  # text / image / graph / mixed
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.source_type in ("image", "mixed")

    @property
    def is_text(self) -> bool:
        return self.source_type in ("text", "mixed")


@dataclass
class MultimodalSearchResult:
    """Lightweight result from the multimodal search layer.

    Mirrors the key fields of MultiModalRetriever's output so that either the real
    MultiModalRetriever or a mock/test function can be injected.
    """

    query: MultimodalQuery
    query_mode: str = "text"  # text / image / mixed
    results: List[MultimodalItem] = field(default_factory=list)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0

    def image_entity_ids(self) -> List[str]:
        """Return IDs of items that can serve as graph seeds for image content."""
        return [r.id for r in self.results if r.is_image]

    def text_chunk_ids(self) -> List[str]:
        """Return IDs of text items."""
        return [r.id for r in self.results if r.is_text]

    def top_k(self, k: int = 10) -> List[MultimodalItem]:
        return sorted(self.results, key=lambda x: x.score, reverse=True)[:k]


@dataclass
class MultimodalKGSearchResult:
    """Combined result of text KG search and image-driven cross-modal KG search."""

    # Pure text KG search result (same query, no image seeds)
    text_result: KGSearchResult = field(default_factory=KGSearchResult)

    # Result from KG search with image seeds injected (cross-modal propagation)
    cross_modal_result: KGSearchResult = field(default_factory=KGSearchResult)

    # Raw multimodal search result (e.g., from MultiModalRetriever)
    multimodal_search_result: Optional[MultimodalSearchResult] = None

    # Image-derived entity IDs that were used as cross-modal seeds
    cross_modal_seeds: List[str] = field(default_factory=list)

    # Combined deduplicated entities and chunks from both text and cross-modal paths
    combined_entities: List[ScoredEntity] = field(default_factory=list)
    combined_chunks: List[ScoredChunk] = field(default_factory=list)

    # Provenance for debugging
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_result": self.text_result.to_dict(),
            "cross_modal_result": self.cross_modal_result.to_dict(),
            "multimodal_search_result": {
                "query_mode": self.multimodal_search_result.query_mode if self.multimodal_search_result else "",
                "source_distribution": self.multimodal_search_result.source_distribution if self.multimodal_search_result else {},
                "latency_ms": self.multimodal_search_result.latency_ms if self.multimodal_search_result else 0,
            },
            "cross_modal_seeds": self.cross_modal_seeds,
            "combined_entities": [e.entity_id for e in self.combined_entities],
            "combined_chunks": [c.chunk_id for c in self.combined_chunks],
            "provenance": self.provenance,
        }


MultimodalSearchFunc = Callable[[MultimodalQuery], MultimodalSearchResult]
"""Type alias for the multimodal search function."""


class MultimodalKGSearchRetriever:
    """Fuse multimodal search with KG search retrieval.

    Usage::

        def mock_multimodal_search(query: MultimodalQuery) -> MultimodalSearchResult:
            return MultimodalSearchResult(
                query=query,
                query_mode="mixed",
                results=[
                    MultimodalItem(id="desc_1", label="ImageDescription", source_type="image"),
                ],
            )

        kg_retriever = KGSearchRetriever(...)
        mm_retriever = MultimodalKGSearchRetriever(
            kg_retriever=kg_retriever,
            multimodal_search_func=mock_multimodal_search,
        )

        result = mm_retriever.retrieve(
            MultimodalQuery(text="revenue trend", image_base64=...)
        )

    The retrieval pipeline:
        1. If query is text-only, run KGSearchRetriever normally.
        2. If image is present, run the multimodal search function to discover
           image-matched items (ImageDescription, charts, etc.).
        3. Extract cross-modal seeds from image-matched items.
        4. Run KGSearchRetriever again with the image seeds as external seeds.
        5. Merge and deduplicate the two result sets.

    Args:
        kg_retriever: A configured KGSearchRetriever instance.
        multimodal_search_func: Function that takes a MultimodalQuery and returns
            a MultimodalSearchResult. If None, image queries are treated as text-only.
    """

    def __init__(
        self,
        kg_retriever: KGSearchRetriever,
        multimodal_search_func: Optional[MultimodalSearchFunc] = None,
    ) -> None:
        """Initialize the multimodal KG search retriever."""
        self.kg_retriever = kg_retriever
        self.multimodal_search_func = multimodal_search_func

    def retrieve(
        self,
        query: MultimodalQuery,
        rewrite: Optional[QueryRewriteResult] = None,
    ) -> MultimodalKGSearchResult:
        """Execute multimodal KG search retrieval.

        Args:
            query: MultimodalQuery with optional text and image.
            rewrite: Optional query rewrite result.

        Returns:
            MultimodalKGSearchResult combining text and cross-modal KG results.
        """
        if not query.text and not query.image_base64:
            return MultimodalKGSearchResult()

        # 1. Always run text-based KG search as the baseline
        text_result = self.kg_retriever.retrieve(query.text, rewrite)

        # 2. If no image or no multimodal search function, return text-only result
        if query.is_text_only or self.multimodal_search_func is None:
            return MultimodalKGSearchResult(
                text_result=text_result,
                cross_modal_result=text_result,
                combined_entities=text_result.entities,
                combined_chunks=text_result.chunks,
                provenance={
                    "query_type": "text",
                    "text_entity_count": len(text_result.entities),
                    "text_chunk_count": len(text_result.chunks),
                },
            )

        # 3. Run multimodal search to discover image-matched entities/chunks
        multimodal_result = self.multimodal_search_func(query)

        # 4. Extract cross-modal seeds from image-derived items
        cross_modal_seeds = multimodal_result.image_entity_ids()
        log.info(
            "[MultimodalKGSearch] query_mode=%s image_results=%d seeds=%d",
            multimodal_result.query_mode,
            len(multimodal_result.results),
            len(cross_modal_seeds),
        )

        # 5. Run KG search with image seeds injected
        cross_modal_result = self._retrieve_with_seeds(
            query.text, rewrite, cross_modal_seeds
        )

        # 6. Merge and deduplicate results
        combined_entities = self._merge_entities(
            text_result.entities, cross_modal_result.entities
        )
        combined_chunks = self._merge_chunks(
            text_result.chunks, cross_modal_result.chunks
        )

        return MultimodalKGSearchResult(
            text_result=text_result,
            cross_modal_result=cross_modal_result,
            multimodal_search_result=multimodal_result,
            cross_modal_seeds=cross_modal_seeds,
            combined_entities=combined_entities,
            combined_chunks=combined_chunks,
            provenance={
                "query_type": multimodal_result.query_mode,
                "text_entity_count": len(text_result.entities),
                "text_chunk_count": len(text_result.chunks),
                "cross_modal_entity_count": len(cross_modal_result.entities),
                "cross_modal_chunk_count": len(cross_modal_result.chunks),
                "cross_modal_seeds": cross_modal_seeds,
                "multimodal_latency_ms": multimodal_result.latency_ms,
            },
        )

    def search_by_image(
        self,
        image_base64: str,
        text_query: str = "",
        rewrite: Optional[QueryRewriteResult] = None,
    ) -> MultimodalKGSearchResult:
        """Convenience method for image-first retrieval.

        Args:
            image_base64: Base64 encoded image data.
            text_query: Optional text query to combine with the image.
            rewrite: Optional query rewrite result.

        Returns:
            MultimodalKGSearchResult.
        """
        query = MultimodalQuery(
            text=text_query,
            image_base64=image_base64,
            query_type="image",
        )
        return self.retrieve(query, rewrite)

    def _retrieve_with_seeds(
        self,
        text_query: str,
        rewrite: Optional[QueryRewriteResult],
        seed_entity_ids: List[str],
    ) -> KGSearchResult:
        """Run KGSearchRetriever with temporary external seed entity IDs.

        This avoids mutating the shared kg_retriever instance by creating a
        lightweight copy with the same dependencies and the additional seeds.
        """
        seeded_retriever = KGSearchRetriever(
            router=self.kg_retriever._router,
            graph_traversal_func=self.kg_retriever._graph_traversal,
            entity_score_func=self.kg_retriever._entity_score,
            entity_ranker=self.kg_retriever._entity_ranker,
            community_search_func=self.kg_retriever._community_search,
            chunk_lookup_func=self.kg_retriever._chunk_lookup,
            external_seed_entity_ids=seed_entity_ids,
            config=self.kg_retriever.config,
        )
        return seeded_retriever.retrieve(text_query, rewrite)

    @staticmethod
    def _merge_entities(
        text_entities: List[ScoredEntity],
        cross_modal_entities: List[ScoredEntity],
    ) -> List[ScoredEntity]:
        """Deduplicate entities and keep the higher score across text and cross-modal results."""
        merged: Dict[str, ScoredEntity] = {}
        for entity in text_entities + cross_modal_entities:
            existing = merged.get(entity.entity_id)
            if existing is None or entity.score > existing.score:
                merged[entity.entity_id] = entity
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)

    @staticmethod
    def _merge_chunks(
        text_chunks: List[ScoredChunk],
        cross_modal_chunks: List[ScoredChunk],
    ) -> List[ScoredChunk]:
        """Deduplicate chunks and keep the higher score across text and cross-modal results."""
        merged: Dict[str, ScoredChunk] = {}
        for chunk in text_chunks + cross_modal_chunks:
            existing = merged.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                merged[chunk.chunk_id] = chunk
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def multimodal_kg_search(
    query_text: str,
    image_base64: str = "",
    kg_retriever: Optional[KGSearchRetriever] = None,
    multimodal_search_func: Optional[MultimodalSearchFunc] = None,
) -> MultimodalKGSearchResult:
    """One-shot multimodal KG search.

    Args:
        query_text: Text query.
        image_base64: Optional base64 image data.
        kg_retriever: Configured KGSearchRetriever.
        multimodal_search_func: Multimodal search function.

    Returns:
        MultimodalKGSearchResult.
    """
    if kg_retriever is None or multimodal_search_func is None:
        logger.warning("multimodal_kg_search requires both kg_retriever and multimodal_search_func")
        return MultimodalKGSearchResult()

    retriever = MultimodalKGSearchRetriever(
        kg_retriever=kg_retriever,
        multimodal_search_func=multimodal_search_func,
    )
    return retriever.retrieve(MultimodalQuery(text=query_text, image_base64=image_base64))
