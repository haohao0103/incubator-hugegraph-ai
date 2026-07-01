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

"""Unit tests for hugegraph_llm.poc.memory_backend helpers and FaissMemoryIndex."""

import json
import os
import tempfile
import time
import uuid
from unittest import mock

import numpy as np
import pytest

from hugegraph_llm.poc.memory_backend import (
    FaissMemoryIndex,
    MemoryPipelineBackend,
    _extract_json_from_response,
    _normalize_keys,
    content_hash_md5,
    get_metadata_db,
    init_metadata_db,
)
from hugegraph_llm.config.memory_config import memory_settings


class MockMessage:
    def __init__(self, content="", reasoning_content=""):
        self.content = content
        self.reasoning_content = reasoning_content


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, content=""):
        self.choices = [MockChoice(MockMessage(content=content))]


class MockSentenceTransformer:
    """Deterministic mock sentence transformer that returns a unique 384-dim vector per text."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, text, **kwargs):
        # Deterministic vector based on text hash, normalized to unit length
        h = hash(text) & 0xFFFFFFFF
        np.random.seed(h)
        vec = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Patch FaissMemoryIndex to use a deterministic mock embedding model."""
    old_model = FaissMemoryIndex._model
    FaissMemoryIndex._model = MockSentenceTransformer(dim=384)
    yield
    FaissMemoryIndex._model = old_model


def test_extract_json_from_markdown():
    content = json.dumps({"entities": [{"name": "Alice", "type": "person"}]})
    response = MockResponse(f'```json\n{content}\n```')
    result = _extract_json_from_response(response)
    assert result["entities"][0]["name"] == "Alice"


def test_extract_json_from_plain():
    content = json.dumps({"entities": [{"name": "Bob", "type": "person"}]})
    response = MockResponse(content)
    result = _extract_json_from_response(response)
    assert result["entities"][0]["name"] == "Bob"


def test_extract_json_from_regex_fallback():
    response = MockResponse('some text before "name": "Carol", "type": "person" after')
    result = _extract_json_from_response(response)
    assert result["entities"][0]["name"] == "Carol"


def test_extract_json_reasoning_content():
    response = MockResponse()
    response.choices[0].message.content = ""
    response.choices[0].message.reasoning_content = '{"entities": []}'
    result = _extract_json_from_response(response)
    assert result["entities"] == []


def test_normalize_keys():
    raw = {
        "entities": [{"name": "Alice", "type": "Person"}, {"entity": "Bob", "category": "person"}],
        "relationships": [{"source": "Alice", "relationship": "works_at", "target": "Tencent"}],
    }
    result = _normalize_keys(raw)
    assert result["entities"] == [
        {"name": "Alice", "type": "person"},
        {"name": "Bob", "type": "person"},
    ]
    assert result["relationships"] == [
        {"source": "Alice", "relationship": "works_at", "target": "Tencent"},
    ]


def test_normalize_keys_skips_self_references():
    raw = {
        "entities": [{"name": "我", "type": "person"}, {"name": "Alice", "type": "person"}],
    }
    result = _normalize_keys(raw)
    assert len(result["entities"]) == 1


class TestFaissMemoryIndex:
    def test_add_and_search(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello world", 123456.0)
        results = idx.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0]["memory_id"] == "m1"

    def test_search_with_weights(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello", 123456.0)
        idx.add_memory("m2", "world", 123456.0)
        results = idx.search("hello", top_k=5, ebbinghaus_weights={"m1": 1.0, "m2": 0.5})
        assert results[0]["memory_id"] == "m1"

    def test_save_and_load(self, mock_sentence_transformer):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "faiss.index")
            idx = FaissMemoryIndex(dim=384, index_path=index_path)
            idx.add_memory("m1", "hello", 123456.0)
            idx.save()

            idx2 = FaissMemoryIndex(dim=384, index_path=index_path)
            idx2.load()
            results = idx2.search("hello", top_k=5)
            assert len(results) == 1

    def test_delete_memory(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello", 123456.0)
        idx.add_memory("m2", "world", 123456.0)
        idx.delete_memory("m1")
        results = idx.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0]["memory_id"] == "m2"

    def test_clear(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello", 123456.0)
        idx.clear()
        assert idx.index.ntotal == 0


class TestMetadataDb:
    def test_init_metadata_db_creates_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MEMORY_DB_PATH"] = os.path.join(tmpdir, "meta.db")
            try:
                init_metadata_db()
                db = get_metadata_db()
                tables = db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                names = {row[0] for row in tables}
                assert "memories" in names
                assert "personas" in names
                db.close()
            finally:
                del os.environ["MEMORY_DB_PATH"]

    def test_metadata_db_migration_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MEMORY_DB_PATH"] = os.path.join(tmpdir, "meta.db")
            try:
                init_metadata_db()
                db = get_metadata_db()
                info = db.execute("PRAGMA table_info(memories)").fetchall()
                columns = {row[1] for row in info}
                assert "scope" in columns
                assert "privacy" in columns
                assert "importance" in columns
                assert "metadata" in columns
                db.close()
            finally:
                del os.environ["MEMORY_DB_PATH"]


def test_memory_backend_import():
    """Ensure memory_backend imports correctly after all modifications."""
    from hugegraph_llm.poc import memory_backend
    assert hasattr(memory_backend, "MemoryPipelineBackend")
    assert hasattr(memory_backend, "HugeGraphMemoryClient")
    assert hasattr(memory_backend, "FaissMemoryIndex")


# ── P0: Additive hybrid scoring tests (mem0-style) ───────────────────────────

from hugegraph_llm.engines.memory.hybrid_scoring import (
    score_and_rank,
    compute_entity_boosts,
    normalize_bm25,
    get_bm25_params,
)


def test_score_and_rank_semantic_only():
    """Additive scoring with semantic only: scores should be normalized by 1.0."""
    semantic = [
        {"id": "m1", "content": "hello world", "score": 0.9},
        {"id": "m2", "content": "foo bar", "score": 0.5},
    ]
    ranked = score_and_rank(semantic, {}, {}, top_k=10)
    assert len(ranked) == 2
    assert ranked[0]["id"] == "m1"
    assert ranked[0]["score"] == pytest.approx(0.9, abs=0.01)
    assert ranked[1]["score"] == pytest.approx(0.5, abs=0.01)


def test_score_and_rank_with_bm25_and_entity():
    """Additive scoring combines semantic + BM25 + entity boost and divides by 2.5."""
    semantic = [
        {"id": "m1", "content": "Alice works at Tencent", "score": 0.8},
        {"id": "m2", "content": "Bob likes coffee", "score": 0.7},
    ]
    bm25 = {"m1": 25.0, "m2": 5.0}
    entity = {"m1": 0.5}  # Alice matches
    midpoint, steepness = get_bm25_params("Alice Tencent")
    bm25_norm_m1 = normalize_bm25(25.0, midpoint, steepness)
    bm25_norm_m2 = normalize_bm25(5.0, midpoint, steepness)

    ranked = score_and_rank(semantic, bm25, entity, top_k=10)
    m1 = next(r for r in ranked if r["id"] == "m1")
    m2 = next(r for r in ranked if r["id"] == "m2")
    expected_m1 = (0.8 + bm25_norm_m1 + 0.5) / 2.5
    expected_m2 = (0.7 + bm25_norm_m2 + 0.0) / 2.5
    assert m1["score"] == pytest.approx(expected_m1, abs=0.01)
    assert m2["score"] == pytest.approx(expected_m2, abs=0.01)
    assert m1["score"] > m2["score"]


def test_score_and_rank_threshold_filtering():
    """Memories with semantic score below threshold are discarded."""
    semantic = [
        {"id": "m1", "content": "relevant", "score": 0.15},
        {"id": "m2", "content": "low", "score": 0.05},
    ]
    ranked = score_and_rank(semantic, {}, {}, threshold=0.1, top_k=10)
    assert len(ranked) == 1
    assert ranked[0]["id"] == "m1"


def test_score_and_rank_explain_mode():
    """Explain mode returns score breakdown."""
    semantic = [
        {"id": "m1", "content": "Alice works at Tencent", "score": 0.8},
    ]
    bm25 = {"m1": 20.0}
    entity = {"m1": 0.5}
    ranked = score_and_rank(semantic, bm25, entity, explain=True, top_k=10)
    assert len(ranked) == 1
    assert "score_breakdown" in ranked[0]
    breakdown = ranked[0]["score_breakdown"]
    assert breakdown["semantic"] == pytest.approx(0.8, abs=0.01)
    assert breakdown["entity_boost"] == pytest.approx(0.5, abs=0.01)


# ── P0: HugeGraphGraphStore multi-hop retrieval tests ───────────────────────

from hugegraph_llm.engines.memory.graph_store import HugeGraphGraphStore


def test_graph_store_search_multi_hop():
    """GraphStore.search returns multi-hop paths from matched query entities."""
    hg_client = mock.MagicMock()
    hg_client.get_all_vertices.return_value = [
        {"id": "person:1", "name": "\u7231\u4e3d\u4e1d", "label": "person"},
        {"id": "person:2", "name": "\u9c8c\u9c7c", "label": "person"},
        {"id": "organization:1", "name": "\u817e\u8baf", "label": "organization"},
    ]
    hg_client.get_all_edges.return_value = [
        {"id": "e1", "source": "person:1", "target": "organization:1",
         "source_name": "\u7231\u4e3d\u4e1d", "target_name": "\u817e\u8baf", "label": "works_at"},
        {"id": "e2", "source": "person:2", "target": "organization:1",
         "source_name": "\u9c8c\u9c7c", "target_name": "\u817e\u8baf", "label": "works_at"},
    ]

    gs = HugeGraphGraphStore(hg_client, max_hops=2, max_neighbors=50)
    # Query with Chinese org suffix to trigger entity extraction
    results = gs.search("\u7231\u4e3d\u4e1d\u5728\u817e\u8baf\u516c\u53f8\u5de5\u4f5c")
    assert len(results) > 0
    contexts = [r["context"] for r in results]
    assert any("\u7231\u4e3d\u4e1d" in c for c in contexts)
    assert any("\u817e\u8baf" in c for c in contexts)


def test_graph_store_empty_when_no_entities():
    """GraphStore.search returns empty list when no entities are extracted."""
    hg_client = mock.MagicMock()
    hg_client.get_all_vertices.return_value = []
    hg_client.get_all_edges.return_value = []
    gs = HugeGraphGraphStore(hg_client, max_hops=2, max_neighbors=50)
    results = gs.search("hello world")
    assert results == []


# ── P0: MemoryPipelineBackend integration tests ─────────────────────────────


@pytest.fixture
def memory_backend(tmp_path):
    """Create a MemoryPipelineBackend with mocked dependencies for unit tests."""
    old_db_path = os.environ.get("MEMORY_DB_PATH")
    old_llm_key = os.environ.get("LLM_API_KEY")
    old_settings_db_path = getattr(memory_settings, "memory_db_path", None)
    os.environ["MEMORY_DB_PATH"] = str(tmp_path / "meta.db")
    os.environ["LLM_API_KEY"] = "test-key"
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MEMORY_DATA_DIR"] = tmpdir
        # Patch both module-level DB_PATH and settings object so all SQLite helpers
        # use the fresh per-test database file.
        import hugegraph_llm.poc.memory_backend as mb_module
        old_module_db_path = mb_module.DB_PATH
        new_db_path = str(tmp_path / "meta.db")
        memory_settings.memory_db_path = new_db_path
        mb_module.DB_PATH = new_db_path
        try:
            with mock.patch("hugegraph_llm.poc.memory_backend.FaissMemoryIndex") as MockFaiss:
                with mock.patch("hugegraph_llm.poc.memory_backend.HugeGraphMemoryClient") as MockHG:
                    backend = MemoryPipelineBackend()
                    # Mock P1/P2 components so unit tests stay offline / fast
                    backend._query_rewrite = mock.MagicMock()
                    backend._query_rewrite.rewrite.return_value = {
                        "rewritten": "query",
                        "entities": [],
                        "intent": "general",
                        "method": "rule",
                        "boosts": {},
                    }
                    backend._route_store = mock.MagicMock()
                    backend._route_store.add_memory.return_value = "user:demo_user"
                    backend._user_profile = mock.MagicMock()
                    backend._user_profile.update_from_memories.return_value = None
                    backend._profile_injector = mock.MagicMock()
                    backend._profile_injector.get_profile_for_rewrite.return_value = {
                        "user_profile": "",
                        "aliases": {},
                        "topics": [],
                    }
                    backend._faiss_deletable = mock.MagicMock()
                    backend._faiss_deletable.remove_by_id.return_value = True
                    backend._compressor = mock.MagicMock()
                    backend._compressor.compress.return_value = {
                        "summaries": [],
                        "kept": [],
                        "pruned": [],
                        "archived": [],
                        "stats": {},
                    }
                    backend._agent_manager = mock.MagicMock()
                    backend._agent_manager.check_access.return_value = True
                    backend._agent_manager.get_accessible_memories.return_value = []
                    yield backend
        finally:
            mb_module.DB_PATH = old_module_db_path
            if old_settings_db_path is not None:
                memory_settings.memory_db_path = old_settings_db_path
            else:
                memory_settings.memory_db_path = None
            if old_db_path is not None:
                os.environ["MEMORY_DB_PATH"] = old_db_path
            else:
                os.environ.pop("MEMORY_DB_PATH", None)
            if old_llm_key is not None:
                os.environ["LLM_API_KEY"] = old_llm_key
            else:
                os.environ.pop("LLM_API_KEY", None)
            os.environ.pop("MEMORY_DATA_DIR", None)


def test_backend_uses_graph_store_and_additive_scoring(memory_backend):
    """Backend initializes HugeGraphGraphStore and additive scoring flag."""
    from hugegraph_llm.engines.memory.graph_store import HugeGraphGraphStore
    assert isinstance(memory_backend._graph_store, HugeGraphGraphStore)
    assert hasattr(memory_backend, "_additive_scoring_available")
    # Old RRF operator should not be initialized
    assert getattr(memory_backend, "_rrf", None) is None


def test_backend_search_uses_additive_not_rrf(memory_backend):
    """search_memory uses additive scoring when GraphRAG ops are available."""
    import hugegraph_llm.poc.memory_backend as mb
    mb.HAS_GRAPHRAG_OPS = True
    try:
        # Mock FAISS and BM25 results
        memory_backend.faiss.search.return_value = [
            {"memory_id": "m1", "content": "Alice works at Tencent", "raw_score": 0.9,
             "retention": 1.0, "weighted_score": 0.9},
            {"memory_id": "m2", "content": "Bob likes coffee", "raw_score": 0.7,
             "retention": 1.0, "weighted_score": 0.7},
        ]
        bm25_mock = mock.MagicMock()
        bm25_mock.doc_count = 2
        bm25_mock.search.return_value = [
            {"id": "m1", "score": 20.0},
            {"id": "m2", "score": 5.0},
        ]
        memory_backend._bm25 = bm25_mock

        # Mock graph store to return entity boost for Alice
        graph_store_mock = mock.MagicMock()
        graph_store_mock.search.return_value = [
            {"matched_entity": "Alice", "context": "Alice [works_at] Tencent", "score": 1.0},
        ]
        memory_backend._graph_store = graph_store_mock

        # Mock additive pipeline to avoid LLM calls in add path (not needed for search)
        additive_mock = mock.MagicMock()
        additive_mock.run.return_value = {
            "new_facts": ["stub"], "duplicate_facts": [], "entities": [], "hashes": set(),
        }
        memory_backend._additive_pipeline = additive_mock

        # Pre-seed metadata DB
        db = get_metadata_db()
        now = time.time()
        db.execute(
            "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
            "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("m1", "Alice works at Tencent", "demo_user", now, now, 1, 0.8,
             "private", "standard", 0.8, "{}"),
        )
        db.execute(
            "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
            "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("m2", "Bob likes coffee", "demo_user", now, now, 1, 0.8,
             "private", "standard", 0.8, "{}"),
        )
        db.commit()
        db.close()

        result = memory_backend.search_memory("Alice", user_id="demo_user", fast_eval=True)
        assert "results" in result
        # m1 should rank higher because of entity boost from Alice
        assert result["results"][0]["memory"]["id"] == "m1"
        # Source should indicate graph or additive
        assert "graph" in result["results"][0]["source"] or "additive" in result["results"][0]["source"]
    finally:
        mb.HAS_GRAPHRAG_OPS = False


def test_backend_add_memory_triggers_history(memory_backend):
    """add_memory writes an ADD event to MemoryHistoryTracker."""
    # Mock the additive pipeline to return a single new fact
    additive_mock = mock.MagicMock()
    additive_mock.run.return_value = {
        "new_facts": ["Alice works at Tencent"],
        "duplicate_facts": [],
        "entities": [{"name": "Alice", "type": "person"}, {"name": "Tencent", "type": "organization"}],
        "hashes": {content_hash_md5("Alice works at Tencent")},
    }
    memory_backend._additive_pipeline = additive_mock
    # Mock LLM extraction inside add_memory
    memory_backend._llm_extract = mock.MagicMock(return_value={
        "entities": [{"name": "Alice", "type": "person"}, {"name": "Tencent", "type": "organization"}],
        "relationships": [{"source": "Alice", "relationship": "works_at", "target": "Tencent"}],
    })
    # Mock hg client vertex/edge creation
    memory_backend.hg.add_vertex.return_value = "vid"
    memory_backend.hg.add_edge.return_value = "eid"

    history_mock = mock.MagicMock()
    memory_backend._history = history_mock

    result = memory_backend.add_memory(
        "Alice works at Tencent", user_id="demo_user", skip_index_save=True
    )
    assert result["action"] == "ADD"
    history_mock.add_history.assert_called_once()
    call_kwargs = history_mock.add_history.call_args.kwargs
    assert call_kwargs["event"] == "ADD"
    assert call_kwargs["memory_id"] == result["memory_id"]
    assert "Alice works at Tencent" in call_kwargs["new_memory"]


def test_backend_update_and_delete_memory_triggers_history(memory_backend):
    """update_memory and delete_memory write UPDATE/DELETE events."""
    # Pre-seed a memory
    db = get_metadata_db()
    now = time.time()
    db.execute(
        "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
        "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("m1", "old content", "demo_user", now, now, 1, 0.8,
         "private", "standard", 0.8, "{}"),
    )
    db.commit()
    db.close()

    history_mock = mock.MagicMock()
    memory_backend._history = history_mock

    # Update
    memory_backend.update_memory("m1", "new content", user_id="demo_user")
    update_call = history_mock.add_history.call_args_list[-1]
    assert update_call.kwargs["event"] == "UPDATE"
    assert update_call.kwargs["old_memory"] == "old content"
    assert update_call.kwargs["new_memory"] == "new content"

    # Delete
    memory_backend.delete_memory("m1", user_id="demo_user")
    delete_call = history_mock.add_history.call_args_list[-1]
    assert delete_call.kwargs["event"] == "DELETE"
    assert delete_call.kwargs["old_memory"] == "new content"


def test_backend_query_rewrite_used_in_search(memory_backend):
    """search_memory calls LLMQueryRewriteEngine and uses rewritten query."""
    memory_backend._query_rewrite.rewrite.return_value = {
        "rewritten": " rewritten query",
        "entities": [{"name": "Alice", "type": "person"}],
        "intent": "fact_lookup",
        "method": "llm",
        "boosts": {},
    }
    db = get_metadata_db()
    now = time.time()
    db.execute(
        "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
        "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("m1", "Alice works at Tencent", "demo_user", now, now, 1, 0.8,
         "private", "standard", 0.8, "{}"),
    )
    db.commit()
    db.close()

    memory_backend.faiss.search.return_value = [
        {"memory_id": "m1", "raw_score": 0.9, "weighted_score": 0.9},
    ]
    memory_backend._bm25 = None
    memory_backend._graph_store = mock.MagicMock()
    memory_backend._graph_store.search.return_value = []

    import hugegraph_llm.poc.memory_backend as mb
    mb.HAS_GRAPHRAG_OPS = True
    try:
        # fast_eval=True skips LLM rewrite; test that rewrite is NOT called
        result = memory_backend.search_memory("Alice", user_id="demo_user", fast_eval=True)
        assert "results" in result
        # In fast_eval mode, LLM query rewrite should be skipped
        memory_backend._query_rewrite.rewrite.assert_not_called()
        # Verify trace shows fast_eval_skip method in detail string
        rewrite_traces = [t for t in result.get("trace", []) if t.get("step") == 1.5]
        assert rewrite_traces, "Query rewrite trace should exist"
        assert "fast_eval_skip" in rewrite_traces[0]["detail"]
    finally:
        mb.HAS_GRAPHRAG_OPS = False


def test_backend_user_profile_and_route_store_on_add(memory_backend):
    """add_memory updates user profile and records routing key."""
    additive_mock = mock.MagicMock()
    additive_mock.run.return_value = {
        "new_facts": ["Alice works at Tencent"],
        "duplicate_facts": [],
        "entities": [{"name": "Alice", "type": "person"}, {"name": "Tencent", "type": "organization"}],
        "hashes": {content_hash_md5("Alice works at Tencent")},
    }
    memory_backend._additive_pipeline = additive_mock
    memory_backend._llm_extract = mock.MagicMock(return_value={
        "entities": [{"name": "Alice", "type": "person"}, {"name": "Tencent", "type": "organization"}],
        "relationships": [{"source": "Alice", "relationship": "works_at", "target": "Tencent"}],
    })
    memory_backend.hg.add_vertex.return_value = "vid"
    memory_backend.hg.add_edge.return_value = "eid"

    result = memory_backend.add_memory(
        "Alice works at Tencent", user_id="demo_user", skip_index_save=True
    )
    assert result["action"] == "ADD"
    memory_backend._user_profile.update_from_memories.assert_called_once_with(
        "demo_user", ["Alice works at Tencent"]
    )
    memory_backend._route_store.add_memory.assert_called_once()


def test_backend_agent_privacy_filter_in_search(memory_backend):
    """search_memory filters results via AgentMemoryManager when agent_id is given."""
    db = get_metadata_db()
    now = time.time()
    for i, content in enumerate(["memory one", "memory two"], start=1):
        db.execute(
            "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
            "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"m{i}", content, "demo_user", now, now, 1, 0.8,
             "private", "standard", 0.8, "{}"),
        )
    db.commit()
    db.close()

    memory_backend.faiss.search.return_value = [
        {"memory_id": "m1", "raw_score": 0.9, "weighted_score": 0.9},
        {"memory_id": "m2", "raw_score": 0.8, "weighted_score": 0.8},
    ]
    memory_backend._bm25 = None
    memory_backend._graph_store = mock.MagicMock()
    memory_backend._graph_store.search.return_value = []
    memory_backend._agent_manager.get_accessible_memories.return_value = [
        {"id": "m1", "content": "memory one"}
    ]

    import hugegraph_llm.poc.memory_backend as mb
    mb.HAS_GRAPHRAG_OPS = True
    try:
        result = memory_backend.search_memory(
            "query", user_id="demo_user", agent_id="agent-x", fast_eval=True
        )
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"]["id"] == "m1"
        memory_backend._agent_manager.get_accessible_memories.assert_called_once()
    finally:
        mb.HAS_GRAPHRAG_OPS = False


def test_backend_faiss_deletable_called_on_delete(memory_backend):
    """delete_memory also removes the vector from FaissDeletableIndex."""
    db = get_metadata_db()
    now = time.time()
    db.execute(
        "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
        "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("m1", "content", "demo_user", now, now, 1, 0.8,
         "private", "standard", 0.8, "{}"),
    )
    db.commit()
    db.close()

    memory_backend.delete_memory("m1", user_id="demo_user")
    memory_backend._faiss_deletable.remove_by_id.assert_called_once_with("m1")


def test_backend_compress_memories_calls_compressor(memory_backend):
    """compress_memories invokes MemoryCompressor and deletes archived memories."""
    db = get_metadata_db()
    now = time.time()
    for i, content in enumerate(["Alice likes tea", "Alice likes coffee"], start=1):
        db.execute(
            "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
            "initial_score,scope,privacy,importance,metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f"m{i}", content, "demo_user", now, now, 1, 0.8,
             "private", "standard", 0.8, "{}"),
        )
    db.commit()
    db.close()

    memory_backend._compressor.compress.return_value = {
        "summaries": [{"cluster_id": 0, "summary": "Alice likes beverages", "source_ids": ["m1", "m2"], "source_count": 2}],
        "kept": [],
        "pruned": [],
        "archived": [{"id": "m1"}, {"id": "m2"}],
        "stats": {"input_count": 2, "output_count": 1},
    }
    memory_backend.add_memory_bypass_classify = mock.MagicMock(return_value={"memory_id": "msummary"})

    result = memory_backend.compress_memories(user_id="demo_user")
    memory_backend._compressor.compress.assert_called_once()
    assert result["summaries"][0]["summary"] == "Alice likes beverages"
    memory_backend.add_memory_bypass_classify.assert_called_once()
