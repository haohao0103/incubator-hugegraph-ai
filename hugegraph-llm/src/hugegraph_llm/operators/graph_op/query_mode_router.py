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

"""Five-mode query router for GraphRAG retrieval — LightRAG-style dispatch.

Inspired by LightRAG's ``_perform_kg_search`` (operate.py lines 4315-4522),
this router dispatches a user query to one of five retrieval modes:

- **naive**:  Pure vector search → top-k chunks.
- **local**:  ll_keywords → entity VDB → graph k_neighbor → associated chunks.
- **global**: hl_keywords → relation/edge VDB → associated entities + chunks.
- **hybrid**: Round-Robin merge of local + global via RRF.
- **mix**:    Round-Robin merge of hybrid + naive via RRF.

All search functions are injected for testability — the router has no direct
database connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion
from hugegraph_llm.operators.llm_op.dual_keyword_extract import DualKeywords

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class QueryMode(Enum):
    """Retrieval mode for GraphRAG queries."""

    NAIVE = "naive"       # Pure vector search
    LOCAL = "local"       # Entity-centric (ll_keywords → entity VDB → graph)
    GLOBAL = "global"     # Relation-centric (hl_keywords → relation VDB)
    HYBRID = "hybrid"     # RRF merge of local + global
    MIX = "mix"           # RRF merge of hybrid + naive


@dataclass
class QueryModeConfig:
    """Configuration for the query mode router."""

    default_mode: QueryMode = QueryMode.HYBRID
    top_k: int = 10                # Final retrieval limit
    rrf_k: int = 60                # RRF fusion constant
    graph_max_depth: int = 2       # k_neighbor traversal depth
    vector_search_top_k: int = 20  # Per-channel vector search candidates


@dataclass
class QueryRouteResult:
    """Result of a routed query — retrieved chunks plus provenance."""

    mode: QueryMode
    chunks: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heuristic query mode detection
# ---------------------------------------------------------------------------


def detect_query_mode(query: str) -> QueryMode:
    """Heuristic: choose a query mode based on query characteristics.

    Rules (mirrors LightRAG's default behaviour):
    - Very short queries (<15 chars) → NAIVE (too vague for graph traversal)
    - Queries with specific entity names (capitalized words) → LOCAL
    - Queries about themes/concepts (long, abstract words) → GLOBAL
    - Most queries → HYBRID (best default, LightRAG's recommendation)
    """
    if not query:
        return QueryMode.NAIVE

    # Very short queries: likely simple lookups, no graph benefit
    if len(query) < 15:
        return QueryMode.NAIVE

    # Check for capitalized words (likely proper nouns / specific entities)
    words = query.split()
    capitalized = [w for w in words if w[0].isupper() and len(w) > 2 and w.lower() not in {
        "what", "who", "how", "why", "when", "where", "which", "the", "this", "that",
    }]

    # Queries with multiple specific entities → LOCAL
    if len(capitalized) >= 2:
        return QueryMode.LOCAL

    # Long, abstract queries (>60 chars) → GLOBAL
    if len(query) > 60 and len(capitalized) == 0:
        return QueryMode.GLOBAL

    # Default → HYBRID (LightRAG's recommended default)
    return QueryMode.HYBRID


# ---------------------------------------------------------------------------
# Core router
# ---------------------------------------------------------------------------


class QueryModeRouter:
    """Route queries to appropriate retrieval mode using injected search funcs.

    All search functions are Callable — no direct DB connections.  This makes
    the router fully testable with mocks and interchangeable backends.

    Parameters
    ----------
    config : QueryModeConfig, optional
        Router configuration (defaults are sensible).
    vector_search_func : callable
        ``f(query_text, top_k) -> [chunk_id, ...]``
    entity_search_func : callable
        ``f(ll_keywords_str, top_k) -> [entity_id, ...]``
    relation_search_func : callable
        ``f(hl_keywords_str, top_k) -> [relation_id, ...]``
    graph_neighbor_func : callable
        ``f(entity_id, max_depth) -> [chunk_id, ...]``
    chunk_lookup_func : callable, optional
        ``f(chunk_id) -> chunk_text`` — resolves IDs to text.
        If not provided, IDs are used directly as text (vector_search_func
        may return text directly).
    keyword_extract_func : callable, optional
        ``f(query) -> DualKeywords`` — dual keyword extractor.
        If not provided, heuristic extraction is used.
    """

    def __init__(
        self,
        config: Optional[QueryModeConfig] = None,
        vector_search_func: Optional[Callable[[str, int], List[str]]] = None,
        entity_search_func: Optional[Callable[[str, int], List[str]]] = None,
        relation_search_func: Optional[Callable[[str, int], List[str]]] = None,
        graph_neighbor_func: Optional[Callable[[str, int], List[str]]] = None,
        chunk_lookup_func: Optional[Callable[[str], str]] = None,
        keyword_extract_func: Optional[Callable[[str], DualKeywords]] = None,
    ) -> None:
        self.config = config or QueryModeConfig()
        self._vector_search = vector_search_func
        self._entity_search = entity_search_func
        self._relation_search = relation_search_func
        self._graph_neighbor = graph_neighbor_func
        self._chunk_lookup = chunk_lookup_func
        self._keyword_extract = keyword_extract_func

    # -- public entry point -------------------------------------------------

    def route(
        self,
        query: str,
        mode: Optional[QueryMode] = None,
    ) -> QueryRouteResult:
        """Route *query* through the specified (or detected) retrieval mode.

        Args:
            query: User question string.
            mode: Explicit mode override.  If ``None``, ``detect_query_mode``
                  is used.

        Returns:
            ``QueryRouteResult`` with retrieved chunks and provenance.
        """
        if mode is None:
            mode = detect_query_mode(query)

        provenance: Dict[str, Any] = {"query": query, "mode": mode.value}

        # Extract keywords for modes that need them
        keywords: Optional[DualKeywords] = None
        if mode in (QueryMode.LOCAL, QueryMode.GLOBAL, QueryMode.HYBRID, QueryMode.MIX):
            keywords = self._extract_keywords(query)
            provenance["hl_keywords"] = keywords.hl_keywords if keywords else []
            provenance["ll_keywords"] = keywords.ll_keywords if keywords else []

        # Dispatch to mode handler
        if mode == QueryMode.NAIVE:
            chunks = self._route_naive(query, provenance)
        elif mode == QueryMode.LOCAL:
            chunks = self._route_local(query, keywords, provenance)
        elif mode == QueryMode.GLOBAL:
            chunks = self._route_global(query, keywords, provenance)
        elif mode == QueryMode.HYBRID:
            chunks = self._route_hybrid(query, keywords, provenance)
        elif mode == QueryMode.MIX:
            chunks = self._route_mix(query, keywords, provenance)
        else:
            # Fallback to naive
            chunks = self._route_naive(query, provenance)

        return QueryRouteResult(mode=mode, chunks=chunks, provenance=provenance)

    # -- keyword extraction -------------------------------------------------

    def _extract_keywords(self, query: str) -> Optional[DualKeywords]:
        """Extract dual keywords from query."""
        if self._keyword_extract:
            return self._keyword_extract(query)
        # Heuristic fallback
        from hugegraph_llm.operators.llm_op.dual_keyword_extract import DualKeywordExtract, DualKeywordConfig
        extractor = DualKeywordExtract(llm=None, config=DualKeywordConfig())
        return extractor.extract(query)

    # -- chunk ID → text resolution ----------------------------------------

    def _resolve_chunks(self, chunk_ids: List[str]) -> List[str]:
        """Resolve chunk IDs to text using chunk_lookup_func if available."""
        if self._chunk_lookup:
            return [self._chunk_lookup(id_) for id_ in chunk_ids]
        # If no lookup, assume IDs are already text
        return chunk_ids

    # -- mode handlers -----------------------------------------------------

    def _route_naive(self, query: str, provenance: Dict) -> List[str]:
        """NAIVE: pure vector search → top-k chunks."""
        if not self._vector_search:
            logger.warning("[QueryRouter] No vector_search_func; returning empty")
            return []

        chunk_ids = self._vector_search(query, self.config.vector_search_top_k)
        provenance["naive_candidates"] = len(chunk_ids)
        chunks = self._resolve_chunks(chunk_ids)
        return chunks[:self.config.top_k]

    def _route_local(self, query: str, keywords: Optional[DualKeywords], provenance: Dict) -> List[str]:
        """LOCAL: ll_keywords → entity VDB → graph k_neighbor → chunks."""
        if not keywords or not keywords.ll_keywords:
            # No low-level keywords → fallback to naive
            provenance["local_fallback"] = "no_ll_keywords"
            return self._route_naive(query, provenance)

        if not self._entity_search or not self._graph_neighbor:
            logger.warning("[QueryRouter] LOCAL mode requires entity_search + graph_neighbor funcs")
            return self._route_naive(query, provenance)

        # Step 1: ll_keywords → entity VDB search
        entity_ids = self._entity_search(
            keywords.ll_str, self.config.vector_search_top_k
        )
        provenance["local_entity_ids"] = entity_ids[:5]  # first 5 for traceability

        # Step 2: each entity → graph k_neighbor → associated chunks
        chunk_ids: List[str] = []
        for eid in entity_ids[:self.config.top_k]:
            neighbor_chunks = self._graph_neighbor(eid, self.config.graph_max_depth)
            chunk_ids.extend(neighbor_chunks)

        # Deduplicate preserving order
        chunk_ids = list(dict.fromkeys(chunk_ids))
        provenance["local_chunk_candidates"] = len(chunk_ids)

        chunks = self._resolve_chunks(chunk_ids)
        return chunks[:self.config.top_k]

    def _route_global(self, query: str, keywords: Optional[DualKeywords], provenance: Dict) -> List[str]:
        """GLOBAL: hl_keywords → relation VDB → associated entities + chunks."""
        if not keywords or not keywords.hl_keywords:
            provenance["global_fallback"] = "no_hl_keywords"
            return self._route_naive(query, provenance)

        if not self._relation_search:
            logger.warning("[QueryRouter] GLOBAL mode requires relation_search func")
            return self._route_naive(query, provenance)

        # Step 1: hl_keywords → relation VDB search
        relation_ids = self._relation_search(
            keywords.hl_str, self.config.vector_search_top_k
        )
        provenance["global_relation_ids"] = relation_ids[:5]

        # Step 2: if graph_neighbor available, expand from relation endpoints
        chunk_ids: List[str] = []
        if self._graph_neighbor:
            for rid in relation_ids[:self.config.top_k]:
                neighbor_chunks = self._graph_neighbor(rid, self.config.graph_max_depth)
                chunk_ids.extend(neighbor_chunks)
        else:
            # Without graph traversal, use relation IDs as chunk IDs directly
            chunk_ids = list(relation_ids)

        chunk_ids = list(dict.fromkeys(chunk_ids))
        provenance["global_chunk_candidates"] = len(chunk_ids)

        chunks = self._resolve_chunks(chunk_ids)
        return chunks[:self.config.top_k]

    def _route_hybrid(self, query: str, keywords: Optional[DualKeywords], provenance: Dict) -> List[str]:
        """HYBRID: Round-Robin merge of LOCAL + GLOBAL via RRF."""
        local_chunks = self._route_local(query, keywords, provenance)
        global_chunks = self._route_global(query, keywords, provenance)

        if not local_chunks and not global_chunks:
            return []
        if not local_chunks:
            return global_chunks
        if not global_chunks:
            return local_chunks

        # RRF fuse: local channel + global channel
        rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        fused = rrf.fuse([
            ("local", local_chunks),
            ("global", global_chunks),
        ])
        provenance["hybrid_fused_count"] = len(fused.items)
        return fused.top_k(self.config.top_k)

    def _route_mix(self, query: str, keywords: Optional[DualKeywords], provenance: Dict) -> List[str]:
        """MIX: Round-Robin merge of HYBRID + NAIVE via RRF."""
        # Get hybrid results (local + global fused)
        hybrid_chunks = self._route_hybrid(query, keywords, provenance)
        # Get naive results (pure vector)
        naive_chunks = self._route_naive(query, provenance)

        if not hybrid_chunks and not naive_chunks:
            return []
        if not hybrid_chunks:
            return naive_chunks
        if not naive_chunks:
            return hybrid_chunks

        # RRF fuse: hybrid channel + naive channel
        rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        fused = rrf.fuse([
            ("hybrid", hybrid_chunks),
            ("naive", naive_chunks),
        ])
        provenance["mix_fused_count"] = len(fused.items)
        return fused.top_k(self.config.top_k)
