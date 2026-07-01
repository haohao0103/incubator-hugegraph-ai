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

"""Tests for query_mode_router — 5-mode retrieval dispatch."""

import pytest
from unittest.mock import MagicMock

from hugegraph_llm.operators.graph_op.query_mode_router import (
    QueryMode,
    QueryModeConfig,
    QueryModeRouter,
    QueryRouteResult,
    detect_query_mode,
)
from hugegraph_llm.operators.llm_op.dual_keyword_extract import DualKeywords


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_router(
    vector_results=None,
    entity_results=None,
    relation_results=None,
    neighbor_results=None,
    keywords=None,
    config=None,
):
    """Create a router with mock search functions."""
    if vector_results is None:
        vector_results = ["chunk_v1", "chunk_v2", "chunk_v3"]
    if entity_results is None:
        entity_results = ["entity_A", "entity_B"]
    if relation_results is None:
        relation_results = ["relation_X", "relation_Y"]
    if neighbor_results is None:
        neighbor_results = ["chunk_n1", "chunk_n2", "chunk_n3"]
    if keywords is None:
        keywords = DualKeywords(
            hl_keywords=["treatment", "management"],
            ll_keywords=["diabetes", "insulin"],
        )

    vector_mock = MagicMock(return_value=vector_results)
    entity_mock = MagicMock(return_value=entity_results)
    relation_mock = MagicMock(return_value=relation_results)
    neighbor_mock = MagicMock(return_value=neighbor_results)
    kw_mock = MagicMock(return_value=keywords)

    router = QueryModeRouter(
        config=config or QueryModeConfig(),
        vector_search_func=vector_mock,
        entity_search_func=entity_mock,
        relation_search_func=relation_mock,
        graph_neighbor_func=neighbor_mock,
        keyword_extract_func=kw_mock,
    )
    return router, vector_mock, entity_mock, relation_mock, neighbor_mock, kw_mock


# ---------------------------------------------------------------------------
# detect_query_mode — heuristic
# ---------------------------------------------------------------------------


class TestDetectQueryMode:
    """Heuristic mode detection based on query characteristics."""

    def test_empty_query(self):
        assert detect_query_mode("") == QueryMode.NAIVE

    def test_very_short_query(self):
        assert detect_query_mode("diabetes") == QueryMode.NAIVE

    def test_short_query(self):
        assert detect_query_mode("What is AI?") == QueryMode.NAIVE

    def test_entity_rich_query(self):
        """Multiple capitalized words → LOCAL."""
        assert detect_query_mode("How does BigQuery integrate with TensorFlow?") == QueryMode.LOCAL

    def test_long_abstract_query(self):
        """Long, no specific entities → GLOBAL."""
        result = detect_query_mode(
            "What are the fundamental principles underlying modern approaches "
            "to sustainable development and environmental conservation?"
        )
        assert result == QueryMode.GLOBAL

    def test_default_hybrid(self):
        """Most normal queries → HYBRID."""
        result = detect_query_mode("What is the treatment for diabetes?")
        assert result == QueryMode.HYBRID

    def test_single_capitalized(self):
        """One capitalized word is not enough for LOCAL → HYBRID."""
        result = detect_query_mode("How does Python handle memory?")
        assert result == QueryMode.HYBRID


# ---------------------------------------------------------------------------
# NAIVE mode
# ---------------------------------------------------------------------------


class TestNaiveMode:
    """Pure vector search mode."""

    def test_naive_returns_vector_results(self):
        router, v_mock, *_ = _make_router()
        result = router.route("test query", mode=QueryMode.NAIVE)
        assert result.mode == QueryMode.NAIVE
        assert len(result.chunks) == 3
        v_mock.assert_called_once_with("test query", 20)

    def test_naive_top_k_limit(self):
        router, v_mock, *_ = _make_router(
            vector_results=["c1", "c2", "c3", "c4", "c5", "c6"],
            config=QueryModeConfig(top_k=3, vector_search_top_k=10),
        )
        result = router.route("q", mode=QueryMode.NAIVE)
        assert len(result.chunks) == 3

    def test_naive_no_vector_func(self):
        router = QueryModeRouter(config=QueryModeConfig())
        result = router.route("q", mode=QueryMode.NAIVE)
        assert result.chunks == []

    def test_naive_provenance(self):
        router, *_ = _make_router()
        result = router.route("q", mode=QueryMode.NAIVE)
        assert result.provenance["mode"] == "naive"
        assert result.provenance["query"] == "q"


# ---------------------------------------------------------------------------
# LOCAL mode
# ---------------------------------------------------------------------------


class TestLocalMode:
    """Entity-centric retrieval: ll_keywords → entity VDB → graph."""

    def test_local_with_keywords(self):
        router, v_mock, e_mock, r_mock, n_mock, kw_mock = _make_router()
        result = router.route("What causes diabetes?", mode=QueryMode.LOCAL)
        assert result.mode == QueryMode.LOCAL
        # Should have called entity_search and graph_neighbor
        e_mock.assert_called_once()
        # graph_neighbor called for each entity
        assert n_mock.call_count == 2

    def test_local_no_keywords_fallback(self):
        """No ll_keywords → fallback to naive."""
        router, v_mock, e_mock, r_mock, n_mock, kw_mock = _make_router(
            keywords=DualKeywords(hl_keywords=["treatment"], ll_keywords=[]),
        )
        result = router.route("abstract query", mode=QueryMode.LOCAL)
        # Should fallback to naive since no ll_keywords
        assert "local_fallback" in result.provenance

    def test_local_no_entity_func(self):
        """Missing entity_search → fallback to naive."""
        router, v_mock, _, _, _, kw_mock = _make_router()
        router._entity_search = None
        result = router.route("q", mode=QueryMode.LOCAL)
        assert len(result.chunks) > 0  # naive fallback

    def test_local_provenance(self):
        router, *_ = _make_router()
        result = router.route("q", mode=QueryMode.LOCAL)
        assert "ll_keywords" in result.provenance
        assert "local_entity_ids" in result.provenance


# ---------------------------------------------------------------------------
# GLOBAL mode
# ---------------------------------------------------------------------------


class TestGlobalMode:
    """Relation-centric retrieval: hl_keywords → relation VDB."""

    def test_global_with_keywords(self):
        router, *_ = _make_router()
        result = router.route("treatment approaches", mode=QueryMode.GLOBAL)
        assert result.mode == QueryMode.GLOBAL
        assert len(result.chunks) > 0

    def test_global_no_keywords_fallback(self):
        router, *_ = _make_router(
            keywords=DualKeywords(hl_keywords=[], ll_keywords=["diabetes"]),
        )
        result = router.route("specific entity", mode=QueryMode.GLOBAL)
        assert "global_fallback" in result.provenance

    def test_global_no_relation_func(self):
        router, *_ = _make_router()
        router._relation_search = None
        result = router.route("q", mode=QueryMode.GLOBAL)
        # Fallback to naive
        assert len(result.chunks) > 0

    def test_global_without_graph_neighbor(self):
        """Without graph traversal, relation IDs used as chunk IDs."""
        router, *_ = _make_router()
        router._graph_neighbor = None
        result = router.route("q", mode=QueryMode.GLOBAL)
        assert len(result.chunks) > 0


# ---------------------------------------------------------------------------
# HYBRID mode
# ---------------------------------------------------------------------------


class TestHybridMode:
    """RRF merge of local + global channels."""

    def test_hybrid_merges_local_and_global(self):
        router, *_ = _make_router()
        result = router.route("What is diabetes treatment?", mode=QueryMode.HYBRID)
        assert result.mode == QueryMode.HYBRID
        assert len(result.chunks) > 0
        assert "hybrid_fused_count" in result.provenance

    def test_hybrid_only_local(self):
        """If global returns nothing, use local only."""
        router, *_ = _make_router(relation_results=[])
        result = router.route("q", mode=QueryMode.HYBRID)
        assert len(result.chunks) > 0

    def test_hybrid_only_global(self):
        """If local returns nothing, use global only."""
        # entity_results=[] → LOCAL has no entities → fallback to naive
        # But GLOBAL still works with relation_search + graph_neighbor
        # Need neighbor_results to be non-empty for global path
        def selective_neighbor(entity_id, depth):
            # Return chunks for relation endpoints but not for empty entities
            if entity_id.startswith("relation_") or entity_id.startswith("entity_"):
                return ["chunk_g1", "chunk_g2"]
            return []

        router, v_mock, e_mock, r_mock, n_mock, kw_mock = _make_router(
            entity_results=[],  # LOCAL: no entities found
        )
        router._graph_neighbor = selective_neighbor
        result = router.route("q", mode=QueryMode.HYBRID)
        assert len(result.chunks) > 0

    def test_hybrid_both_empty(self):
        router = QueryModeRouter(config=QueryModeConfig())
        result = router.route("q", mode=QueryMode.HYBRID)
        assert result.chunks == []

    def test_hybrid_top_k_limit(self):
        router, *_ = _make_router(
            config=QueryModeConfig(top_k=3),
        )
        result = router.route("q", mode=QueryMode.HYBRID)
        assert len(result.chunks) <= 3


# ---------------------------------------------------------------------------
# MIX mode
# ---------------------------------------------------------------------------


class TestMixMode:
    """RRF merge of hybrid + naive channels."""

    def test_mix_merges_hybrid_and_naive(self):
        router, *_ = _make_router()
        result = router.route("diabetes treatment options", mode=QueryMode.MIX)
        assert result.mode == QueryMode.MIX
        assert len(result.chunks) > 0
        assert "mix_fused_count" in result.provenance

    def test_mix_only_naive(self):
        router, *_ = _make_router(entity_results=[], neighbor_results=[], relation_results=[])
        result = router.route("q", mode=QueryMode.MIX)
        # Hybrid returns nothing → mix uses naive only
        assert len(result.chunks) > 0

    def test_mix_both_empty(self):
        router = QueryModeRouter(config=QueryModeConfig())
        result = router.route("q", mode=QueryMode.MIX)
        assert result.chunks == []


# ---------------------------------------------------------------------------
# Auto mode detection
# ---------------------------------------------------------------------------


class TestAutoModeDetection:
    """Route without explicit mode — uses detect_query_mode."""

    def test_auto_mode_short_query(self):
        router, *_ = _make_router()
        result = router.route("diabetes")  # short → NAIVE
        assert result.mode == QueryMode.NAIVE

    def test_auto_mode_normal_query(self):
        router, *_ = _make_router()
        result = router.route("What is the treatment for diabetes?")  # → HYBRID
        assert result.mode == QueryMode.HYBRID

    def test_explicit_mode_override(self):
        router, *_ = _make_router()
        result = router.route("short q", mode=QueryMode.LOCAL)  # explicit override
        assert result.mode == QueryMode.LOCAL


# ---------------------------------------------------------------------------
# Chunk lookup
# ---------------------------------------------------------------------------


class TestChunkLookup:
    """Chunk ID → text resolution."""

    def test_with_chunk_lookup_func(self):
        lookup = MagicMock(side_effect=lambda id_: f"text_of_{id_}")
        router, *_ = _make_router()
        router._chunk_lookup = lookup
        result = router.route("q", mode=QueryMode.NAIVE)
        assert all("text_of_" in c for c in result.chunks)

    def test_without_chunk_lookup_func(self):
        router, *_ = _make_router()
        result = router.route("q", mode=QueryMode.NAIVE)
        # IDs used directly as text
        assert result.chunks[0] == "chunk_v1"


# ---------------------------------------------------------------------------
# QueryModeConfig
# ---------------------------------------------------------------------------


class TestQueryModeConfig:
    """Configuration defaults and customization."""

    def test_defaults(self):
        config = QueryModeConfig()
        assert config.default_mode == QueryMode.HYBRID
        assert config.top_k == 10
        assert config.rrf_k == 60
        assert config.graph_max_depth == 2
        assert config.vector_search_top_k == 20

    def test_custom(self):
        config = QueryModeConfig(top_k=5, rrf_k=30, graph_max_depth=3)
        assert config.top_k == 5
        assert config.rrf_k == 30
