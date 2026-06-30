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

"""Tests for cascade_propagation.py — Entity→Relation→Chunk three-layer scoring."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hugegraph_llm.operators.graph_op.cascade_propagation import (
    CascadeConfig,
    CascadeMatrixBuilder,
    CascadePropagation,
    CascadeResult,
    RankingConfig,
    apply_threshold_ranking,
    apply_topk_ranking,
    csr_from_indices_list,
)


# ── Test csr_from_indices_list ───────────────────────────────────


class TestCSRFromIndicesList:
    """Test sparse matrix construction from index lists."""

    def test_simple_binary_matrix(self):
        """Basic 3x3 binary matrix."""
        data = [[0, 1], [1, 2], [0, 2]]
        mat = csr_from_indices_list(data, shape=(3, 3))
        assert mat.shape == (3, 3)
        assert mat.nnz == 6  # 2+2+2 entries
        assert mat[0, 0] == 1
        assert mat[0, 1] == 1
        assert mat[1, 1] == 1
        assert mat[1, 2] == 1
        assert mat[2, 0] == 1
        assert mat[2, 2] == 1

    def test_empty_data(self):
        """Empty index list produces zero matrix."""
        data = []
        mat = csr_from_indices_list(data, shape=(0, 5))
        assert mat.shape == (0, 5)
        assert mat.nnz == 0

    def test_empty_rows(self):
        """Rows with empty lists produce zero entries."""
        data = [[], [0], []]
        mat = csr_from_indices_list(data, shape=(3, 5))
        assert mat.shape == (3, 5)
        assert mat.nnz == 1
        assert mat[1, 0] == 1

    def test_single_row_single_col(self):
        """Minimal 1x1 matrix."""
        data = [[0]]
        mat = csr_from_indices_list(data, shape=(1, 1))
        assert mat[0, 0] == 1

    def test_large_sparse_matrix(self):
        """Large sparse matrix with many empty rows."""
        data = [[5]] + [[] for _ in range(99)]
        mat = csr_from_indices_list(data, shape=(100, 200))
        assert mat.shape == (100, 200)
        assert mat.nnz == 1
        assert mat[0, 5] == 1


# ── Test ranking policies ────────────────────────────────────────


class TestThresholdRanking:
    """Test apply_threshold_ranking."""

    def test_threshold_filters_low_scores(self):
        """Scores below threshold are zeroed."""
        scores = csr_matrix([[0.001, 0.01, 0.1, 0.5]])
        result = apply_threshold_ranking(scores, threshold=0.01, max_count=100)
        assert result.nnz == 3  # 0.001 filtered out
        assert result[0, 0] == 0  # Below threshold
        assert result[0, 1] == 0.01
        assert result[0, 2] == 0.1
        assert result[0, 3] == 0.5

    def test_max_count_limits_entries(self):
        """Only top max_count entries survive."""
        scores = csr_matrix([[0.1, 0.2, 0.3, 0.4, 0.5]])
        result = apply_threshold_ranking(scores, threshold=0.0, max_count=3)
        assert result.nnz == 3
        # Top 3 scores kept: 0.3, 0.4, 0.5
        assert result[0, 2] == 0.3
        assert result[0, 3] == 0.4
        assert result[0, 4] == 0.5

    def test_all_below_threshold(self):
        """All scores below threshold → empty matrix."""
        scores = csr_matrix([[0.001, 0.002, 0.003]])
        result = apply_threshold_ranking(scores, threshold=0.01, max_count=100)
        assert result.nnz == 0

    def test_empty_matrix(self):
        """Empty input produces empty output."""
        scores = csr_matrix((1, 5))
        result = apply_threshold_ranking(scores, threshold=0.01, max_count=10)
        assert result.nnz == 0


class TestTopKRanking:
    """Test apply_topk_ranking."""

    def test_topk_keeps_best(self):
        """Only top-k entries survive."""
        scores = csr_matrix([[0.1, 0.5, 0.3, 0.8, 0.2]])
        result = apply_topk_ranking(scores, top_k=2)
        assert result.nnz == 2
        assert result[0, 1] == 0.5  # 2nd highest
        assert result[0, 3] == 0.8  # highest

    def test_topk_all_kept_when_small(self):
        """If entries <= top_k, all kept."""
        scores = csr_matrix([[0.1, 0.2]])
        result = apply_topk_ranking(scores, top_k=5)
        assert result.nnz == 2

    def test_topk_exact_equal(self):
        """top_k = nnz keeps all."""
        scores = csr_matrix([[0.1, 0.2, 0.3]])
        result = apply_topk_ranking(scores, top_k=3)
        assert result.nnz == 3

    def test_topk_one(self):
        """top_k=1 keeps only the highest."""
        scores = csr_matrix([[0.1, 0.9, 0.5]])
        result = apply_topk_ranking(scores, top_k=1)
        assert result.nnz == 1
        assert result[0, 1] == 0.9


# ── Test CascadePropagation ──────────────────────────────────────


class TestCascadePropagationBasic:
    """Test basic cascade propagation pipeline."""

    def _make_simple_data(self):
        """Create simple test data for cascade."""
        # 3 entities, 3 relations, 3 chunks
        entity_index_map = {"e0": 0, "e1": 1, "e2": 2}
        relation_index_map = {"r0": 0, "r1": 1, "r2": 2}
        chunk_index_map = {"c0": 0, "c1": 1, "c2": 2}

        # e2r: e0→r0,r1; e1→r1,r2; e2→r0,r2
        e2r = csr_from_indices_list([[0, 1], [1, 2], [0, 2]], shape=(3, 3))

        # r2c: r0→c0; r1→c1; r2→c2
        r2c = csr_from_indices_list([[0], [1], [2]], shape=(3, 3))

        return entity_index_map, relation_index_map, chunk_index_map, e2r, r2c

    def _make_vector_search_fn(self, results):
        """Create a mock vector search function."""
        def fn(query, top_k=20, threshold=0.5):
            return results
        return fn

    def test_full_cascade_chain(self):
        """Test complete Entity→Relation→Chunk propagation."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()

        vector_fn = self._make_vector_search_fn([
            ("e0", 0.8), ("e1", 0.6),
        ])

        cascade = CascadePropagation(config=CascadeConfig(
            ranking=RankingConfig(
                entity_threshold=0.001,
                entity_max_count=100,
                relation_top_k=100,
                chunk_top_k=100,
            ),
        ))

        result = cascade.retrieve(
            query="test query",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
            relation_index_map=rel_map,
            chunk_index_map=chunk_map,
        )

        assert len(result.seed_entities) == 2
        assert "e0" in result.entity_scores
        assert "e1" in result.entity_scores
        # Relation scores should propagate from entities
        assert len(result.relation_scores) > 0
        # Chunk scores should propagate from relations
        assert len(result.chunk_scores) > 0
        assert result.stats["duration_ms"] > 0

    def test_empty_seed_entities(self):
        """No seed entities → empty result."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()
        vector_fn = self._make_vector_search_fn([])  # Empty results

        cascade = CascadePropagation()
        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
        )

        assert result.seed_entities == []
        assert len(result.entity_scores) == 0
        assert len(result.chunk_scores) == 0

    def test_bm25_optional_enhancement(self):
        """BM25 enabled → RRF fusion with cascade results."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()
        vector_fn = self._make_vector_search_fn([
            ("e0", 0.8), ("e1", 0.6),
        ])
        bm25_fn = lambda q, k: ["c0", "c3"]  # BM25 returns different chunks

        cascade = CascadePropagation(config=CascadeConfig(
            bm25_enabled=True,
            bm25_top_k=10,
            ranking=RankingConfig(entity_threshold=0.001, relation_top_k=100, chunk_top_k=100),
        ))

        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
            relation_index_map=rel_map,
            chunk_index_map=chunk_map,
            bm25_search_fn=bm25_fn,
        )

        assert result.bm25_results == ["c0", "c3"]
        # Final chunks should include both cascade and BM25 results
        assert len(result.chunk_scores) > 0

    def test_bm25_disabled(self):
        """BM25 disabled (default) → no BM25 results."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()
        vector_fn = self._make_vector_search_fn([("e0", 0.8)])

        cascade = CascadePropagation()  # bm25_enabled=False by default
        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
        )

        assert result.bm25_results == []
        assert result.stats["bm25_enabled"] == False

    def test_no_e2r_matrix(self):
        """No e2r matrix → skip Entity→Relation propagation."""
        entity_map, _, _, _, _ = self._make_simple_data()
        vector_fn = self._make_vector_search_fn([("e0", 0.8)])

        cascade = CascadePropagation(config=CascadeConfig(
            ranking=RankingConfig(entity_threshold=0.001, entity_max_count=100),
        ))
        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            entity_index_map=entity_map,
        )

        assert len(result.entity_scores) > 0
        assert len(result.relation_scores) == 0  # Skipped
        assert len(result.chunk_scores) == 0  # Also skipped

    def test_cascade_result_dataclass(self):
        """CascadeResult can be created and serialized."""
        result = CascadeResult(
            entity_scores={"e0": 0.5},
            relation_scores={"r0": 0.3},
            chunk_scores={"c0": 0.2},
            seed_entities=["e0"],
        )
        assert result.entity_scores["e0"] == 0.5


# ── Test CascadeMatrixBuilder ─────────────────────────────────────


class TestCascadeMatrixBuilderLocal:
    """Test CascadeMatrixBuilder.build_from_local (no HugeGraph needed)."""

    def test_simple_local_build(self):
        """Build matrices from local data."""
        entities = ["e0", "e1", "e2"]
        relations = [
            {"source": "e0", "target": "e1", "label": "r0"},
            {"source": "e1", "target": "e2", "label": "r1"},
            {"source": "e0", "target": "e2", "label": "r2"},
        ]
        chunks = ["c0", "c1", "c2"]
        chunk_texts = {
            "c0": "text about e0 and e1",
            "c1": "text about e1 and e2",
            "c2": "text about e0 and e2",
        }

        builder = CascadeMatrixBuilder()
        e2r, r2c, e_map, r_map, c_map = builder.build_from_local(
            entities, relations, chunks, chunk_texts,
        )

        # e2r should map each entity to its participating relations
        assert e2r.shape == (3, 3)
        assert e_map["e0"] == 0
        assert e_map["e1"] == 1
        assert e_map["e2"] == 2

        # e0 participates in r0 and r2
        row_e0 = e2r[0, :].toarray().flatten()
        assert row_e0[0] == 1  # r0
        assert row_e0[2] == 1  # r2

        # r2c should map each relation to its chunks
        assert r2c.shape == (3, 3)

    def test_empty_entities(self):
        """Empty entity list produces zero matrices."""
        builder = CascadeMatrixBuilder()
        e2r, r2c, e_map, r_map, c_map = builder.build_from_local(
            [], [], [],
        )
        assert e2r.shape == (0, 0)
        assert r2c.shape == (0, 0)
        assert len(e_map) == 0

    def test_no_chunk_texts(self):
        """No chunk texts → r2c may be sparse."""
        entities = ["e0", "e1"]
        relations = [{"source": "e0", "target": "e1", "label": "r0"}]
        chunks = ["c0"]
        # No chunk_texts → r2c will only map via source_id property

        builder = CascadeMatrixBuilder()
        e2r, r2c, e_map, r_map, c_map = builder.build_from_local(
            entities, relations, chunks,
        )

        assert e2r.shape == (2, 1)
        # e0 → r0, e1 → r0
        assert e2r[0, 0] == 1
        assert e2r[1, 0] == 1

    def test_entity_with_no_relations(self):
        """Entity not in any relation → zero row in e2r."""
        entities = ["e0", "e1", "e_lonely"]
        relations = [{"source": "e0", "target": "e1", "label": "r0"}]
        chunks = ["c0"]

        builder = CascadeMatrixBuilder()
        e2r, r2c, e_map, r_map, c_map = builder.build_from_local(
            entities, relations, chunks,
        )

        assert e2r.shape == (3, 1)
        # e_lonely (index 2) has zero row
        assert e2r[2, 0] == 0


# ── Test sparse_to_score_dict ─────────────────────────────────────


class TestSparseToScoreDict:
    """Test CascadePropagation._sparse_to_score_dict."""

    def test_with_index_map(self):
        """Convert sparse scores to dict using index map."""
        scores = csr_matrix([[0.5, 0.0, 0.3]])
        index_map = {"a": 0, "b": 1, "c": 2}
        result = CascadePropagation._sparse_to_score_dict(scores, index_map)
        assert result["a"] == 0.5
        assert result["c"] == 0.3
        # b has score 0, should not appear
        assert "b" not in result

    def test_without_index_map(self):
        """Convert sparse scores to dict using integer indices."""
        scores = csr_matrix([[0.5, 0.3]])
        result = CascadePropagation._sparse_to_score_dict(scores, None)
        assert result["0"] == 0.5
        assert result["1"] == 0.3

    def test_empty_matrix(self):
        """Empty matrix → empty dict."""
        scores = csr_matrix((1, 0))
        result = CascadePropagation._sparse_to_score_dict(scores, None)
        assert result == {}


# ── Test CascadeConfig ────────────────────────────────────────────


class TestCascadeConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Default config has expected values."""
        config = CascadeConfig()
        assert config.ppr_alpha == 0.85
        assert config.ppr_epsilon == 1e-6
        assert config.vector_top_k == 20
        assert config.vector_threshold == 0.5
        assert config.bm25_enabled == False
        assert config.bm25_weight == 0.3
        assert config.ranking.entity_threshold == 0.005
        assert config.ranking.relation_top_k == 64
        assert config.ranking.chunk_top_k == 8

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = CascadeConfig(
            ppr_alpha=0.15,
            bm25_enabled=True,
            ranking=RankingConfig(entity_threshold=0.01, chunk_top_k=10),
        )
        assert config.ppr_alpha == 0.15
        assert config.bm25_enabled == True
        assert config.ranking.entity_threshold == 0.01
        assert config.ranking.chunk_top_k == 10


# ── Test PPR via HugeGraph ────────────────────────────────────────


class TestPPRViaHugeGraph:
    """Test cascade with mock PPRRetriever (covers _ppr_via_hugegraph)."""

    def _make_simple_data(self):
        entity_index_map = {"e0": 0, "e1": 1, "e2": 2}
        relation_index_map = {"r0": 0, "r1": 1, "r2": 2}
        chunk_index_map = {"c0": 0, "c1": 1, "c2": 2}
        e2r = csr_from_indices_list([[0, 1], [1, 2], [0, 2]], shape=(3, 3))
        r2c = csr_from_indices_list([[0], [1], [2]], shape=(3, 3))
        return entity_index_map, relation_index_map, chunk_index_map, e2r, r2c

    def _make_mock_ppr_retriever(self, results):
        """Create a mock PPRRetriever that returns predefined PPR scores."""
        class MockPPRRetriever:
            def search(self, source_id, max_depth=2, alpha=0.85, epsilon=1e-6, top_k=20):
                return results.get(source_id, [])

        return MockPPRRetriever()

    def _make_mock_graph_client(self):
        """Create a mock graph client (non-None to trigger PPR path)."""
        class MockClient:
            pass
        return MockClient()

    def test_ppr_via_hugegraph_propagation(self):
        """PPR via HugeGraph propagates importance beyond seed entities."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()

        # PPR returns additional entity scores beyond seed
        ppr_retriever = self._make_mock_ppr_retriever({
            "e0": [
                {"node_id": "e1", "ppr_score": 0.3},
                {"node_id": "e2", "ppr_score": 0.1},
            ],
        })
        graph_client = self._make_mock_graph_client()

        vector_fn = lambda q, top_k=20, threshold=0.5: [("e0", 0.8)]

        cascade = CascadePropagation(config=CascadeConfig(
            ranking=RankingConfig(entity_threshold=0.001, entity_max_count=100,
                                  relation_top_k=100, chunk_top_k=100),
        ))

        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
            relation_index_map=rel_map,
            chunk_index_map=chunk_map,
            ppr_retriever=ppr_retriever,
            graph_client=graph_client,
        )

        # PPR should have propagated scores to e1 and e2
        assert len(result.entity_scores) >= 2
        # Seed entity "e0" should have highest score
        assert result.entity_scores.get("e0", 0) > 0

    def test_ppr_failure_fallback_to_seed(self):
        """PPR failure → falls back to seed scores directly."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()

        # PPR retriever that raises exception
        class ErrorPPRRetriever:
            def search(self, **kwargs):
                raise RuntimeError("PPR service unavailable")

        vector_fn = lambda q, top_k=20, threshold=0.5: [("e0", 0.8)]

        cascade = CascadePropagation(config=CascadeConfig(
            ranking=RankingConfig(entity_threshold=0.001, entity_max_count=100,
                                  relation_top_k=100, chunk_top_k=100),
        ))

        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
            ppr_retriever=ErrorPPRRetriever(),
            graph_client=self._make_mock_graph_client(),
        )

        # Should still have results from seed scores (PPR failure logged)
        assert len(result.entity_scores) >= 1

    def test_no_ppr_retriever_uses_seed_scores(self):
        """No PPR retriever → seed scores used directly."""
        entity_map, rel_map, chunk_map, e2r, r2c = self._make_simple_data()
        vector_fn = lambda q, top_k=20, threshold=0.5: [("e0", 0.8)]

        cascade = CascadePropagation()
        result = cascade.retrieve(
            query="test",
            vector_search_fn=vector_fn,
            e2r_matrix=e2r,
            r2c_matrix=r2c,
            entity_index_map=entity_map,
        )

        assert "e0" in result.entity_scores


# ── Test CascadeMatrixBuilder from HugeGraph ──────────────────────


class TestCascadeMatrixBuilderFromHugeGraph:
    """Test CascadeMatrixBuilder.build_from_hugegraph with mock client."""

    def _make_mock_client_for_build(self):
        """Mock PyHugeClient that returns vertices and edges."""
        class MockClient:
            def getVertexByCondition(self, label="", limit=10000):
                if label == "Entity":
                    return [{"id": "e0"}, {"id": "e1"}, {"id": "e2"}]
                if label == "Chunk":
                    return [
                        {"id": "c0", "properties": {"text": "about e0"}},
                        {"id": "c1", "properties": {"text": "about e1"}},
                    ]
                return []

            def getEdgeByCondition(self, edge_label="", limit=10000):
                if edge_label == "relation":
                    return [
                        {"source": "e0", "target": "e1", "label": "rel_r0",
                         "properties": {"source_id": "c0"}},
                        {"source": "e1", "target": "e2", "label": "rel_r1",
                         "properties": {"source_id": "c1"}},
                    ]
                return []

        return MockClient()

    def test_build_from_hugegraph_with_mock_client(self):
        """Build cascade matrices from mock HugeGraph client."""
        client = self._make_mock_client_for_build()
        builder = CascadeMatrixBuilder(graph_client=client)

        e2r, r2c, e_map, r_map, c_map = builder.build(
            entity_label="Entity",
            relation_edge_labels=["relation"],
            chunk_label="Chunk",
            graph_name="hugegraph",
        )

        # Should have entities, relations, chunks
        assert len(e_map) >= 2
        assert e2r.shape[0] >= 2
        assert e2r.nnz > 0

    def test_build_from_hugegraph_no_client(self):
        """No client → empty matrices."""
        builder = CascadeMatrixBuilder(graph_client=None)
        e2r, r2c, e_map, r_map, c_map = builder.build(
            entity_label="Entity",
        )
        assert e2r.shape == (0, 0)
        assert len(e_map) == 0

    def test_client_exception_returns_empty(self):
        """Client raises exception → empty result."""
        class ErrorClient:
            def getVertexByCondition(self, **kwargs):
                raise RuntimeError("Connection failed")

            def getEdgeByCondition(self, **kwargs):
                raise RuntimeError("Connection failed")

        builder = CascadeMatrixBuilder(graph_client=ErrorClient())
        e2r, r2c, e_map, r_map, c_map = builder.build(
            entity_label="Entity",
        )
        assert len(e_map) == 0

    def test_fetch_all_vertices_helper(self):
        """_fetch_all_vertices returns vertex IDs."""
        client = self._make_mock_client_for_build()
        builder = CascadeMatrixBuilder(graph_client=client)
        result = builder._fetch_all_vertices("Entity", "hugegraph")
        assert len(result) >= 2

    def test_fetch_all_edges_helper(self):
        """_fetch_all_edges returns edge dicts."""
        client = self._make_mock_client_for_build()
        builder = CascadeMatrixBuilder(graph_client=client)
        result = builder._fetch_all_edges(["relation"], "hugegraph")
        assert len(result) >= 1

    def test_get_chunk_texts_helper(self):
        """_get_chunk_texts returns chunk_id → text dict."""
        client = self._make_mock_client_for_build()
        builder = CascadeMatrixBuilder(graph_client=client)
        result = builder._get_chunk_texts("Chunk", "hugegraph")
        assert len(result) >= 1

    def test_fetch_helpers_no_client(self):
        """Helper methods return empty when no client."""
        builder = CascadeMatrixBuilder(graph_client=None)
        assert builder._fetch_all_vertices("X", "g") == []
        assert builder._fetch_all_edges(["X"], "g") == []
        assert builder._get_chunk_texts("X", "g") == {}


# ── Test CascadeConfig additional fields ──────────────────────────


class TestCascadeConfigAdditional:
    """Test extended CascadeConfig fields."""

    def test_ppr_config(self):
        """PPR-specific config fields."""
        config = CascadeConfig(ppr_alpha=0.5, ppr_epsilon=1e-4, ppr_max_depth=3)
        assert config.ppr_alpha == 0.5
        assert config.ppr_epsilon == 1e-4
        assert config.ppr_max_depth == 3

    def test_bm25_weight_config(self):
        """BM25 weight config."""
        config = CascadeConfig(bm25_enabled=True, bm25_weight=0.5, bm25_top_k=30)
        assert config.bm25_weight == 0.5
        assert config.bm25_top_k == 30
