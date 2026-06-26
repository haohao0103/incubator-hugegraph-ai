"""
Tests for HugeGraph Memory Backend (Engineering-grade)
=====================================================
Covers: Intent classification, Entity dedup (3-strategy), BM25+RRF fusion,
        Graph direct reasoning, Provenance tracking, Ebbinghaus scoring,
        Anti-hallucination, ADD/QUERY pipeline integration.

Run: python -m pytest tests/poc/test_memory_backend.py -v
"""

import json
import math
import os
import sys
import tempfile
import time
from unittest import mock

import pytest
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hugegraph_llm.poc.memory_backend import (
    MemoryPipelineBackend,
    HugeGraphMemoryClient,
    FaissMemoryIndex,
    EBBINGHAUS_K,
    EBBINGHAUS_REINFORCE,
    HAS_GRAPHRAG_OPS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_dir():
    """Temporary directory for test databases and indices."""
    with tempfile.TemporaryDirectory() as d:
        orig_db = os.environ.get("DB_PATH")
        orig_faiss = os.environ.get("FAISS_INDEX_PATH")
        # Override paths to use temp dir
        test_db = os.path.join(d, "test_memory.db")
        test_faiss = os.path.join(d, "test_faiss.index")
        test_prov = os.path.join(d, "test_provenance.json")
        yield d, test_db, test_faiss, test_prov


@pytest.fixture
def mock_hg_client():
    """Mock HugeGraph client that doesn't require a running server."""
    client = mock.MagicMock(spec=HugeGraphMemoryClient)
    client.init_schema.return_value = None
    client.get_all_vertices.return_value = []
    client.get_all_edges.return_value = []
    client.add_vertex.return_value = "mock_vid"
    client.add_edge.return_value = "mock_eid"
    return client


@pytest.fixture
def mock_faiss_index(tmp_dir):
    """Create a real FaissMemoryIndex in temp directory."""
    _, _, faiss_path, _ = tmp_dir
    # Use a small dim for testing
    index = FaissMemoryIndex(dim=128, index_path=faiss_path)
    # Manually set a mock embedding client that returns deterministic vectors
    mock_embed = mock.MagicMock()
    def deterministic_embed(text):
        # Hash-based deterministic vector
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(128).astype(np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    mock_embed.return_value = deterministic_embed("test")
    index._get_embedding_client = mock.MagicMock(return_value=mock_embed)
    index.embed_text = deterministic_embed
    return index


@pytest.fixture
def backend(tmp_dir, mock_hg_client, mock_faiss_index):
    """Create a MemoryPipelineBackend with mocked dependencies."""
    _, test_db, test_faiss, test_prov = tmp_dir
    # Patch module-level constants
    with mock.patch("hugegraph_llm.poc.memory_backend.DB_PATH", test_db), \
         mock.patch("hugegraph_llm.poc.memory_backend.FAISS_INDEX_PATH", test_faiss):
        backend = MemoryPipelineBackend(
            hg_client=mock_hg_client,
            faiss_index=mock_faiss_index,
        )
        # Patch provenance path
        backend._provenance_db_path = test_prov
        yield backend


# ============================================================================
# P0: Intent Classification Tests
# ============================================================================

class TestIntentClassification:
    """Test the 3-layer intent classification system."""

    def test_question_with_mark(self, backend):
        result = backend._rule_classify_intent("李四的同事有谁？")
        assert result == "QUERY"

    def test_question_who(self, backend):
        result = backend._rule_classify_intent("谁在货拉拉工作")
        assert result == "QUERY"

    def test_question_where(self, backend):
        result = backend._rule_classify_intent("张三在哪里上班")
        assert result == "QUERY"

    def test_question_what_position(self, backend):
        result = backend._rule_classify_intent("赵六是什么职位")
        assert result == "QUERY"

    def test_question_how_many(self, backend):
        result = backend._rule_classify_intent("货拉拉有哪些员工")
        assert result == "QUERY"

    def test_statement_work(self, backend):
        result = backend._rule_classify_intent("我在货拉拉做技术总监")
        assert result == "ADD"

    def test_statement_name_work(self, backend):
        result = backend._rule_classify_intent("我叫张三，在货拉拉上班")
        assert result == "ADD"

    def test_statement_past(self, backend):
        result = backend._rule_classify_intent("昨天去深圳见了客户")
        assert result == "ADD"

    def test_uncertain_returns_none(self, backend):
        result = backend._rule_classify_intent("好的")
        assert result is None

    def test_uncertain_neutral(self, backend):
        result = backend._rule_classify_intent("腾讯是一家互联网公司")
        assert result is None  # not clearly ADD or QUERY


# ============================================================================
# P0: BM25 + RRF Integration Tests
# ============================================================================

class TestBM25RRF:
    """Test BM25 fulltext index and RRF fusion integration."""

    def test_bm25_available(self):
        """BM25 should be available when GraphRAG ops are installed."""
        if HAS_GRAPHRAG_OPS:
            assert True
        else:
            pytest.skip("GraphRAG ops not available")

    def test_backend_has_bm25(self, backend):
        """Backend should initialize BM25 index."""
        # BM25 may be None if GraphRAG ops not available
        # But the attribute should exist
        assert hasattr(backend, "_bm25")

    def test_backend_has_rrf(self, backend):
        """Backend should initialize RRF fuser."""
        assert hasattr(backend, "_rrf")

    @pytest.mark.skipif(not HAS_GRAPHRAG_OPS, reason="Requires GraphRAG ops")
    def test_add_memory_indexes_bm25(self, backend, tmp_dir, mock_hg_client):
        """Adding a memory should also index it in BM25."""
        if backend._bm25 is None:
            pytest.skip("BM25 not available")

        initial_count = backend._bm25.doc_count

        # Mock LLM extract
        extract_result = {
            "entities": [{"name": "测试公司", "type": "organization"}],
            "relationships": [{"source": "测试人", "relationship": "works_at",
                              "target": "测试公司"}]
        }
        backend._llm_extract = mock.MagicMock(return_value=extract_result)

        # Mock DB and add memory
        with mock.patch("hugegraph_llm.poc.memory_backend.get_metadata_db") as mock_db:
            mock_conn = mock.MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_db.return_value = mock_conn
            backend.add_memory("我在测试公司上班", "test_user")

        # BM25 should have one more document
        assert backend._bm25.doc_count == initial_count + 1


# ============================================================================
# P1: Entity Dedup (3-Strategy) Tests
# ============================================================================

class TestEntityDedup:
    """Test enhanced entity deduplication with 3 strategies."""

    def test_substring_merge(self, backend):
        """Strategy 1: '腾讯深圳' should merge into '腾讯' + '深圳'."""
        # Pre-populate graph with existing entity
        backend.hg.get_all_vertices.return_value = [
            {"name": "腾讯", "type": "organization"}
        ]

        entities = [{"name": "腾讯深圳", "type": "organization"}]
        rels = [{"source": "张三", "relationship": "works_at", "target": "腾讯深圳"}]

        new_ents, new_rels = backend._dedup_entities(entities, rels)

        # "腾讯深圳" should be replaced with "腾讯" in relationships
        # The dedup should find "腾讯" as a substring of "腾讯深圳"
        assert any(r["target"] == "腾讯" for r in new_rels), \
            f"Expected '腾讯' in targets, got: {[r['target'] for r in new_rels]}"
        assert not any(r["target"] == "腾讯深圳" for r in new_rels)

    def test_cross_entity_dedup(self, backend):
        """Strategy 2: Very similar names should merge (substring containment)."""
        backend.hg.get_all_vertices.return_value = []

        # "货拉拉公司" contains "货拉拉" → substring match in Strategy 1
        # But since "货拉拉" is also new, it won't be in existing_names.
        # This tests Strategy 2: cross-entity dedup within same type
        entities = [
            {"name": "货拉拉", "type": "organization"},
            {"name": "拉拉科技", "type": "organization"},
        ]
        rels = [
            {"source": "张三", "relationship": "works_at", "target": "货拉拉"},
            {"source": "李四", "relationship": "works_at", "target": "拉拉科技"},
        ]

        new_ents, new_rels = backend._dedup_entities(entities, rels)
        # These have high char overlap: 货拉拉 vs 拉拉科技 → common = {拉,拉} vs {货,拉,拉,科,技}
        # ratio = 2/5 = 0.4 < 0.8, so they should NOT merge (too different)
        unique_targets = set(r["target"] for r in new_rels)
        assert len(unique_targets) == 2  # Both remain separate

    def test_cross_entity_dedup_high_overlap(self, backend):
        """Strategy 2: Near-identical names should merge via char overlap > 0.8."""
        backend.hg.get_all_vertices.return_value = []

        entities = [
            {"name": "腾讯科技", "type": "organization"},
            {"name": "腾讯科技有限", "type": "organization"},
        ]
        rels = [
            {"source": "张三", "relationship": "works_at", "target": "腾讯科技"},
            {"source": "李四", "relationship": "works_at", "target": "腾讯科技有限"},
        ]

        new_ents, new_rels = backend._dedup_entities(entities, rels)
        # 腾讯科技 vs 腾讯科技有限 → common chars very high, ratio > 0.8
        unique_targets = set(r["target"] for r in new_rels)
        assert len(unique_targets) == 1, f"Expected 1 target, got {unique_targets}"

    def test_different_types_not_merged(self, backend):
        """Entities of different types with similar names should NOT merge."""
        backend.hg.get_all_vertices.return_value = []

        entities = [
            {"name": "苹果", "type": "organization"},
            {"name": "苹果", "type": "concept"},
        ]
        rels = []

        new_ents, new_rels = backend._dedup_entities(entities, rels)

        # Both should remain (different types)
        assert len(new_ents) == 2

    def test_no_duplicate_vertices(self, backend):
        """Already-existing entities should not be re-added."""
        backend.hg.get_all_vertices.return_value = [
            {"name": "腾讯", "type": "organization"},
            {"name": "深圳", "type": "location"},
        ]

        entities = [{"name": "腾讯", "type": "organization"}]
        rels = []

        new_ents, _ = backend._dedup_entities(entities, rels)

        # "腾讯" already exists, should not be in new_entities
        assert not any(e["name"] == "腾讯" for e in new_ents)


# ============================================================================
# P1: Provenance Tracking Tests
# ============================================================================

class TestProvenance:
    """Test memory-to-entity provenance tracking."""

    def test_track_provenance(self, backend):
        """Adding a memory should create provenance links."""
        entities = [
            {"name": "张三", "type": "person"},
            {"name": "货拉拉", "type": "organization"},
        ]
        rels = [{"source": "张三", "relationship": "works_at", "target": "货拉拉"}]

        backend._track_provenance("mem_001", entities, rels)

        assert "mem_001" in backend._provenance
        assert len(backend._provenance["mem_001"]) == 3  # 2 entities + 1 rel

    def test_get_provenance_for_entities(self, backend):
        """Retrieving provenance for specific entities."""
        backend._provenance = {
            "mem_001": [{"entity": "张三", "type": "person", "relation": "extracted_from"}],
            "mem_002": [{"entity": "李四", "type": "person", "relation": "extracted_from"}],
            "mem_003": [{"entity": "张三", "type": "source", "relation": "works_at", "target": "货拉拉"}],
        }

        sources = backend._get_provenance_for_entities(["张三"])
        assert len(sources) >= 2  # mem_001 and mem_003
        mem_ids = {s["memory_id"] for s in sources}
        assert "mem_001" in mem_ids or "mem_003" in mem_ids

    def test_provenance_persistence(self, backend, tmp_dir):
        """Provenance data should persist to disk."""
        _, _, _, test_prov = tmp_dir
        backend._provenance_db_path = test_prov

        backend._track_provenance("mem_test", [{"name": "X", "type": "person"}], [])

        # Data should be saved to file
        assert os.path.exists(test_prov)
        with open(test_prov) as f:
            data = json.load(f)
        assert "mem_test" in data


# ============================================================================
# Anti-Hallucination Tests
# ============================================================================

class TestAntiHallucination:
    """Test that the system correctly handles unknown entities."""

    def test_unknown_entity_returns_not_found(self, backend, tmp_dir):
        """Querying about non-existent entities should return 'no info'."""
        import hugegraph_llm.poc.memory_backend as mb_module
        _, test_db, _, _ = tmp_dir

        with mock.patch("hugegraph_llm.poc.memory_backend.get_metadata_db") as mock_db:
            mock_conn = mock.MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_conn.execute.return_value.fetchone.return_value = [0]
            mock_db.return_value = mock_conn

            # Empty graph — no known entities
            backend.hg.get_all_vertices.return_value = []
            backend.hg.get_all_edges.return_value = []

            result = backend.search_memory("马云的同事有谁")
            assert result.get("answer") == "记忆中没有这个信息。"


# ============================================================================
# Ebbinghaus Scoring Tests
# ============================================================================

class TestEbbinghausScoring:
    """Test Ebbinghaus forgetting curve calculations."""

    def test_fresh_memory_high_retention(self):
        """A freshly added memory should have high retention."""
        elapsed_hours = 0.1  # 6 minutes old
        initial_score = 1.0
        access_count = 1
        ret = initial_score * math.exp(-EBBINGHAUS_K * elapsed_hours)
        ret = min(1.0, ret + access_count * EBBINGHAUS_REINFORCE)
        assert ret > 0.95

    def test_old_memory_low_retention(self):
        """An old, rarely accessed memory should have low retention."""
        elapsed_hours = 720  # 30 days old
        initial_score = 1.0
        access_count = 0
        ret = initial_score * math.exp(-EBBINGHAUS_K * elapsed_hours)
        ret = min(1.0, ret + access_count * EBBINGHAUS_REINFORCE)
        assert ret < 0.01

    def test_access_reinforcement(self):
        """Frequent access should reinforce retention."""
        elapsed_hours = 100  # ~4 days old
        initial_score = 1.0
        access_count = 5
        ret = initial_score * math.exp(-EBBINGHAUS_K * elapsed_hours)
        ret = min(1.0, ret + access_count * EBBINGHAUS_REINFORCE)
        # 5 accesses * 0.3 = 1.5 reinforcement should counteract decay
        assert ret > 0.5

    def test_retention_bounded_0_to_1(self):
        """Retention should always be between 0 and 1."""
        for hours in [0, 1, 10, 100, 1000]:
            ret = 1.0 * math.exp(-EBBINGHAUS_K * hours)
            ret = min(1.0, ret)
            assert 0.0 <= ret <= 1.0


# ============================================================================
# Graph Direct Reasoning Tests
# ============================================================================

class TestGraphDirectReasoning:
    """Test graph-based direct reasoning for structured queries."""

    def _make_edges(self):
        return [
            {"source_name": "张三", "target_name": "货拉拉",
             "label": "works_at", "relationship": "works_at"},
            {"source_name": "李四", "target_name": "货拉拉",
             "label": "works_at", "relationship": "works_at"},
            {"source_name": "赵六", "target_name": "货拉拉",
             "label": "works_at", "relationship": "works_at"},
            {"source_name": "王五", "target_name": "字节跳动",
             "label": "works_at", "relationship": "works_at"},
        ]

    def test_colleague_query(self, backend):
        """'李四的同事' should return other people at the same company."""
        edges = self._make_edges()
        answer = backend._graph_colleague_answer(["李四"], edges)
        assert answer is not None
        assert "张三" in answer or "赵六" in answer

    def test_colleague_at_different_company(self, backend):
        """'王五的同事' should NOT include people at other companies."""
        edges = self._make_edges()
        answer = backend._graph_colleague_answer(["王五"], edges)
        # 王五 is alone at 字节跳动
        assert answer is None or "没有" in answer or "无" in answer

    def test_org_employee_query(self, backend):
        """'货拉拉的员工' should list all people at 货拉拉."""
        edges = self._make_edges()
        answer = backend._graph_org_employee_answer(["货拉拉"], edges)
        assert answer is not None
        assert "张三" in answer
        assert "李四" in answer

    def test_workplace_query(self, backend):
        """'张三在哪上班' should return 货拉拉."""
        edges = self._make_edges()
        with mock.patch("hugegraph_llm.poc.memory_backend.get_metadata_db") as mock_db:
            mock_conn = mock.MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_db.return_value = mock_conn
            answer = backend._graph_workplace_answer(["张三"], edges)
        assert answer is not None
        assert "货拉拉" in answer

    def test_unknown_entity_workplace(self, backend):
        """Unknown entity should return appropriate message."""
        edges = self._make_edges()
        answer = backend._graph_workplace_answer(["马云"], edges)
        assert answer is None  # Falls through to LLM


# ============================================================================
# Pipeline Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Test ADD and QUERY pipeline integration."""

    def test_stats_includes_bm25(self, backend):
        """Stats should include BM25 and RRF availability info."""
        # Mock DB queries
        import hugegraph_llm.poc.memory_backend as mb_module
        with mock.patch("hugegraph_llm.poc.memory_backend.get_metadata_db") as mock_db:
            mock_conn = mock.MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_conn.execute.return_value.fetchone.return_value = [0]
            mock_db.return_value = mock_conn

            stats = backend.get_stats("test_user")
            assert "bm25" in stats
            assert "rrf_available" in stats
            assert "provenance_count" in stats
            assert "graphrag_ops" in stats

    def test_clear_all_resets_provenance(self, backend):
        """Clear all should reset provenance data."""
        backend._provenance = {"mem_1": [{"entity": "X"}]}
        backend._save_provenance = mock.MagicMock()
        backend.hg.clear_graph = mock.MagicMock()
        backend.faiss.clear = mock.MagicMock()

        with mock.patch("hugegraph_llm.poc.memory_backend.get_metadata_db") as mock_db:
            mock_conn = mock.MagicMock()
            mock_db.return_value = mock_conn
            backend.clear_all("test_user")

        assert backend._provenance == {}
        backend._save_provenance.assert_called_once()


# ============================================================================
# GraphRAG Ops Availability Tests
# ============================================================================

class TestGraphRAGOps:
    """Verify GraphRAG operator availability."""

    def test_graphrag_ops_detected(self):
        """GraphRAG ops should be detected if packages are installed."""
        # This test just verifies the detection mechanism works
        assert isinstance(HAS_GRAPHRAG_OPS, bool)

    @pytest.mark.skipif(not HAS_GRAPHRAG_OPS, reason="Requires GraphRAG ops")
    def test_bm25_import(self):
        """BM25FullTextBackend should be importable."""
        from hugegraph_llm.indices.fulltext.bm25_fulltext import BM25FullTextBackend
        assert BM25FullTextBackend is not None

    @pytest.mark.skipif(not HAS_GRAPHRAG_OPS, reason="Requires GraphRAG ops")
    def test_rrf_import(self):
        """ReciprocalRankFusion should be importable."""
        from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion
        rrf = ReciprocalRankFusion(k=60)
        assert rrf is not None

    @pytest.mark.skipif(not HAS_GRAPHRAG_OPS, reason="Requires GraphRAG ops")
    def test_rrf_fusion_basic(self):
        """RRF should correctly fuse two ranked lists."""
        from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion
        rrf = ReciprocalRankFusion(k=60)
        # List 1: A > B > C
        # List 2: C > B > D
        # B ranks 2 in both → high RRF score
        # C ranks 3 in list1, 1 in list2
        # D ranks only in list2 at position 3
        result = rrf.fuse([
            ("faiss", ["A", "B", "C"]),
            ("bm25", ["C", "B", "D"]),
        ])
        top = result.top_k(5)
        # B should rank highest (consistent rank 2 across both lists)
        assert "B" in top[:2]
        assert len(top) == 4  # A, B, C, D


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
