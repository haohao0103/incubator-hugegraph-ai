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

"""KG Search Retriever: multi-hop, scored knowledge-graph retrieval.

Combines RAGFlow v0.26.0's KGSearchRetrieval concepts (QueryRewrite, N-hop,
scoring, community search) with LightRAG's local/global retrieval modes and
HippoRAG2's PageRank-based entity ranking. The retriever is backend-agnostic:
all graph operations are injected as callables, so it can work with HugeGraph,
NetworkX, or any other graph store.

Design references:
    - RAGFlow v0.26.0: KGSearchRetrieval (N-hop, scoring, query_rewrite, community)
    - LightRAG: local/global hybrid retrieval modes
    - HippoRAG2: PageRank-style entity ranking for multi-hop reasoning
    - MS-GraphRAG: entity/relationship ranking attributes
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.graph_op.query_mode_router import (
    QueryMode,
    QueryModeRouter,
)
from hugegraph_llm.operators.llm_op.query_rewrite import QueryRewriteResult
from hugegraph_llm.utils.log import log


logger = logging.getLogger(__name__)


@dataclass
class KGSearchConfig:
    """Configuration for KG Search Retriever."""

    # Sub-query handling
    max_sub_queries: int = 4
    sub_query_timeout: Optional[float] = None  # not used by default, reserved

    # N-hop traversal
    max_depth: int = 2
    max_fanout: int = 20

    # Scoring weights
    entity_rank_weight: float = 0.3
    vector_similarity_weight: float = 0.3
    frequency_weight: float = 0.2
    community_weight: float = 0.2

    # Community search
    top_communities: int = 3

    # Final ranking
    top_k: int = 10
    rrf_k: int = 60

    # Mode override for underlying router (None = detect automatically)
    mode: Optional[QueryMode] = None


@dataclass
class ScoredEntity:
    """Entity with retrieval score and provenance."""

    entity_id: str
    name: str = ""
    score: float = 0.0
    depth: int = 0
    source_query: str = ""
    rank_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """Chunk with retrieval score and provenance."""

    chunk_id: str
    text: str = ""
    score: float = 0.0
    source_queries: List[str] = field(default_factory=list)
    source_entities: List[str] = field(default_factory=list)
    rank_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class KGSearchResult:
    """Result of KG search retrieval."""

    chunks: List[ScoredChunk] = field(default_factory=list)
    entities: List[ScoredEntity] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": c.score,
                    "source_queries": c.source_queries,
                    "source_entities": c.source_entities,
                    "rank_factors": c.rank_factors,
                }
                for c in self.chunks
            ],
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "score": e.score,
                    "depth": e.depth,
                    "source_query": e.source_query,
                    "rank_factors": e.rank_factors,
                }
                for e in self.entities
            ],
            "communities": self.communities,
            "provenance": self.provenance,
        }

    @property
    def chunk_texts(self) -> List[str]:
        return [c.text for c in self.chunks]


class KGSearchRetriever:
    """Multi-hop scored knowledge-graph retriever.

    The retriever is fully backend-agnostic: all graph operations are injected
    as callables. This makes it testable with mocks and adaptable to different
    graph stores (HugeGraph, NetworkX, etc.).

    Usage::

        retriever = KGSearchRetriever(
            router=QueryModeRouter(...),
            graph_traversal_func=traverse_hugegraph,
            entity_score_func=page_rank_score,
            community_search_func=search_communities,
        )
        result = retriever.run({
            "query": "...",
            "query_rewrite": QueryRewriteResult(...),
        })
    """

    def __init__(
        self,
        router: Optional[QueryModeRouter] = None,
        graph_traversal_func: Optional[
            Callable[[str, int, int], List[Tuple[str, int, str]]]
        ] = None,
        entity_score_func: Optional[Callable[[str], float]] = None,
        community_search_func: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        chunk_lookup_func: Optional[Callable[[str], str]] = None,
        config: Optional[KGSearchConfig] = None,
    ) -> None:
        """Initialize KGSearchRetriever.

        Args:
            router: QueryModeRouter for local/global/hybrid retrieval per sub-query.
            graph_traversal_func: ``f(entity_id, max_depth, max_fanout) -> [(neighbor_id, depth, edge_type), ...]``
            entity_score_func: ``f(entity_id) -> float`` returning a precomputed rank/score.
            community_search_func: ``f(query_text, top_k) -> [community_dict, ...]``
            chunk_lookup_func: ``f(chunk_id) -> chunk_text``
            config: KGSearchConfig.
        """
        self._router = router
        self._graph_traversal = graph_traversal_func
        self._entity_score = entity_score_func
        self._community_search = community_search_func
        self._chunk_lookup = chunk_lookup_func
        self.config = config or KGSearchConfig()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Operator protocol: execute KG search retrieval.

        Reads from context:
            query: Original user question string.
            query_rewrite: Optional QueryRewriteResult dataclass.

        Writes to context:
            kg_search_result: KGSearchResult dataclass.
        """
        query = context.get("query", "")
        rewrite = context.get("query_rewrite")
        result = self.retrieve(query, rewrite)
        context["kg_search_result"] = result
        return context

    def retrieve(
        self,
        query: str,
        rewrite: Optional[QueryRewriteResult] = None,
    ) -> KGSearchResult:
        """Execute multi-hop KG search retrieval.

        Args:
            query: Original user question.
            rewrite: Optional query rewrite result. If None, the query is treated
                     as a single sub-query.

        Returns:
            KGSearchResult with scored chunks, entities, and communities.
        """
        if not query:
            return KGSearchResult()

        # Determine executable sub-queries
        if rewrite is None:
            executable_queries = [query]
        else:
            executable_queries = rewrite.executable_queries[: self.config.max_sub_queries]
            if not executable_queries:
                executable_queries = [query]

        provenance: Dict[str, Any] = {
            "original_query": query,
            "num_sub_queries": len(executable_queries),
            "sub_queries": executable_queries,
        }

        all_scored_chunks: Dict[str, ScoredChunk] = {}
        all_scored_entities: Dict[str, ScoredEntity] = {}
        all_communities: List[Dict[str, Any]] = []

        for sub_query in executable_queries:
            sub_result = self._retrieve_single(sub_query)
            self._merge_chunks(all_scored_chunks, sub_result.chunks, sub_query)
            self._merge_entities(all_scored_entities, sub_result.entities, sub_query)
            all_communities.extend(sub_result.communities)

        # Final ranking
        ranked_entities = self._rank_entities(list(all_scored_entities.values()))
        ranked_chunks = self._rank_chunks(list(all_scored_chunks.values()))
        ranked_communities = self._rank_communities(all_communities)

        provenance["entity_count"] = len(ranked_entities)
        provenance["chunk_count"] = len(ranked_chunks)
        provenance["community_count"] = len(ranked_communities)

        return KGSearchResult(
            chunks=ranked_chunks,
            entities=ranked_entities,
            communities=ranked_communities,
            provenance=provenance,
        )

    def _retrieve_single(self, sub_query: str) -> KGSearchResult:
        """Retrieve for a single sub-query using the router and graph traversal."""
        chunks: List[ScoredChunk] = []
        entities: List[ScoredEntity] = []
        communities: List[Dict[str, Any]] = []

        # 1. Router-based retrieval (local/global/hybrid/mix)
        if self._router is not None:
            try:
                mode = self.config.mode
                route_result = self._router.route(sub_query, mode=mode)
                router_chunks = route_result.chunks
                provenance_mode = route_result.provenance.get("mode", "unknown")
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Router failed for sub-query '%s': %s", sub_query, e)
                router_chunks = []
                provenance_mode = "router_error"
        else:
            router_chunks = []
            provenance_mode = "no_router"

        for text in router_chunks:
            if self._chunk_lookup is not None:
                cid = text
                resolved_text = self._chunk_lookup(cid) or text
            else:
                cid = text
                resolved_text = text
            chunks.append(
                ScoredChunk(
                    chunk_id=cid,
                    text=resolved_text,
                    score=0.5,  # base router score
                    source_queries=[sub_query],
                    source_entities=[],
                    rank_factors={"router": 0.5, "mode": provenance_mode},
                )
            )

        # 2. Extract seed entities from router result or use sub-query as seed
        seed_entity_ids = self._extract_seed_entities(sub_query, router_chunks)

        # 3. N-hop graph traversal + scoring
        if self._graph_traversal is not None:
            for seed_id in seed_entity_ids:
                neighbors = self._graph_traversal(
                    seed_id, self.config.max_depth, self.config.max_fanout
                )
                for neighbor_id, depth, edge_type in neighbors:
                    # Avoid duplicate seeds
                    if neighbor_id == seed_id and depth == 0:
                        continue
                    entity_score = self._compute_entity_score(
                        neighbor_id, depth, sub_query, edge_type
                    )
                    entities.append(
                        ScoredEntity(
                            entity_id=neighbor_id,
                            name=neighbor_id,
                            score=entity_score,
                            depth=depth,
                            source_query=sub_query,
                            rank_factors={
                                "base_rank": entity_score,
                                "depth": depth,
                                "edge_type": edge_type,
                            },
                        )
                    )

        # 4. Community search
        if self._community_search is not None:
            communities = self._community_search(sub_query, self.config.top_communities)

        return KGSearchResult(chunks=chunks, entities=entities, communities=communities)

    def _extract_seed_entities(self, sub_query: str, router_chunks: List[str]) -> List[str]:
        """Extract seed entity IDs from a sub-query or router chunks.

        If no router is available, the sub-query itself is used as a seed.
        """
        seeds: List[str] = []
        # If router returned chunks, treat chunk texts as pseudo-entity seeds for traversal
        if router_chunks:
            for chunk in router_chunks[: self.config.max_fanout]:
                seed = chunk.strip()[:100]  # truncated chunk as seed
                if seed:
                    seeds.append(seed)
        # Always include the sub-query as a fallback seed
        if not seeds and sub_query:
            seeds.append(sub_query)
        return list(dict.fromkeys(seeds))[: self.config.max_fanout]

    def _compute_entity_score(
        self,
        entity_id: str,
        depth: int,
        sub_query: str,
        edge_type: str = "",
    ) -> float:
        """Compute a composite score for a traversed entity.

        Combines:
        - Precomputed entity rank (e.g., PageRank)
        - Depth decay (closer to seed = higher score)
        - Edge specificity bonus (typed edges score higher)
        """
        # Base rank
        base_rank = 0.5
        if self._entity_score is not None:
            try:
                base_rank = max(0.0, min(1.0, self._entity_score(entity_id)))
            except Exception:  # pylint: disable=broad-except
                logger.debug("Entity score function failed for %s", entity_id)

        # Depth decay: closer to seed is better
        depth_decay = 1.0 / (1.0 + depth)

        # Edge type bonus: typed edges are more informative
        edge_bonus = 0.1 if edge_type else 0.0

        return base_rank * self.config.entity_rank_weight + \
            depth_decay * self.config.vector_similarity_weight + \
            edge_bonus * self.config.frequency_weight

    def _merge_chunks(
        self,
        acc: Dict[str, ScoredChunk],
        chunks: List[ScoredChunk],
        sub_query: str,
    ) -> None:
        """Merge chunks from a sub-query into the accumulator, deduplicating."""
        for chunk in chunks:
            key = chunk.chunk_id
            if key in acc:
                existing = acc[key]
                existing.score = max(existing.score, chunk.score)
                if sub_query not in existing.source_queries:
                    existing.source_queries.append(sub_query)
                existing.rank_factors.update(chunk.rank_factors)
            else:
                acc[key] = chunk

    def _merge_entities(
        self,
        acc: Dict[str, ScoredEntity],
        entities: List[ScoredEntity],
        sub_query: str,
    ) -> None:
        """Merge entities from a sub-query into the accumulator, deduplicating."""
        for entity in entities:
            key = entity.entity_id
            if key in acc:
                existing = acc[key]
                existing.score = max(existing.score, entity.score)
                existing.depth = min(existing.depth, entity.depth)
                existing.rank_factors.update(entity.rank_factors)
            else:
                acc[key] = entity

    def _rank_entities(self, entities: List[ScoredEntity]) -> List[ScoredEntity]:
        """Sort entities by score descending."""
        return sorted(entities, key=lambda e: e.score, reverse=True)

    def _rank_chunks(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        """Sort chunks by score descending and keep top-k."""
        # Resolve text if chunk_lookup is provided
        if self._chunk_lookup is not None:
            for chunk in chunks:
                if not chunk.text:
                    try:
                        chunk.text = self._chunk_lookup(chunk.chunk_id)
                    except Exception:  # pylint: disable=broad-except
                        logger.debug("Chunk lookup failed for %s", chunk.chunk_id)

        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[: self.config.top_k]

    def _rank_communities(self, communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and rank communities."""
        seen: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for comm in communities:
            cid = comm.get("id") or comm.get("community_id") or str(comm)
            if cid not in seen:
                seen.add(cid)
                unique.append(comm)
        # Sort by score if available
        unique.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return unique[: self.config.top_communities]
