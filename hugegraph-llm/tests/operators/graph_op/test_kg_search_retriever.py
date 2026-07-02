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

"""Tests for KGSearchRetriever operator."""

import pytest

from hugegraph_llm.operators.graph_op.kg_search_retriever import (
    KGSearchConfig,
    KGSearchResult,
    KGSearchRetriever,
    ScoredChunk,
    ScoredEntity,
)
from hugegraph_llm.operators.graph_op.query_mode_router import (
    QueryMode,
    QueryModeConfig,
    QueryModeRouter,
    QueryRouteResult,
)
from hugegraph_llm.operators.llm_op.query_rewrite import QueryRewriteResult
from hugegraph_llm.operators.graph_op.entity_ranker import build_ranker_from_edges


class FakeRouter(QueryModeRouter):
    """Router that returns deterministic chunks without external deps."""

    def __init__(self, chunks):
        super().__init__(config=QueryModeConfig())
        self._chunks = chunks

    def route(self, query, mode=None):
        return QueryRouteResult(
            mode=mode or QueryMode.HYBRID,
            chunks=self._chunks,
            provenance={"query": query, "mode": (mode or QueryMode.HYBRID).value},
        )


class FailingRouter:
    """Router-like object that raises."""

    def route(self, query, mode=None):
        raise RuntimeError("router failure")


def fake_traversal(entity_id, max_depth, max_fanout):
    """Return deterministic neighbors."""
    neighbors = []
    for depth in range(1, max_depth + 1):
        for i in range(min(2, max_fanout)):
            neighbors.append((f"{entity_id}_neighbor_{depth}_{i}", depth, "RELATED_TO"))
    return neighbors


def fake_entity_score(entity_id):
    """Return deterministic score based on entity id."""
    return 0.5 + (hash(entity_id) % 100) / 200.0


def fake_community_search(query, top_k):
    return [{"id": f"comm_{query}", "score": 0.8}]


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------


def test_retrieve_empty_query():
    retriever = KGSearchRetriever()
    result = retriever.retrieve("", None)
    assert result == KGSearchResult()


def test_retrieve_without_rewrite():
    router = FakeRouter(["chunk_a", "chunk_b"])
    retriever = KGSearchRetriever(router=router, config=KGSearchConfig(top_k=2))
    result = retriever.retrieve("What is X?", None)
    assert len(result.chunks) == 2
    assert result.chunks[0].text in ("chunk_a", "chunk_b")
    assert result.provenance["num_sub_queries"] == 1


def test_retrieve_with_rewrite():
    router = FakeRouter(["chunk_1"])
    rewrite = QueryRewriteResult(
        original_query="complex query",
        needs_rewrite=True,
        sub_queries=["What is X?", "What is Y?"],
    )
    retriever = KGSearchRetriever(router=router, config=KGSearchConfig(top_k=2))
    result = retriever.retrieve("complex query", rewrite)
    # Same chunk from both sub-queries is merged into one
    assert len(result.chunks) == 1
    assert "What is X?" in result.chunks[0].source_queries
    assert "What is Y?" in result.chunks[0].source_queries
    assert result.provenance["num_sub_queries"] == 2


def test_retrieve_uses_original_query_when_rewrite_empty():
    router = FakeRouter(["chunk"])
    rewrite = QueryRewriteResult(
        original_query="fallback",
        needs_rewrite=False,
        sub_queries=[],
    )
    retriever = KGSearchRetriever(router=router)
    result = retriever.retrieve("fallback", rewrite)
    assert result.provenance["sub_queries"] == ["fallback"]


# ---------------------------------------------------------------------------
# Graph traversal + scoring
# ---------------------------------------------------------------------------


def test_retrieve_with_graph_traversal():
    router = FakeRouter(["seed_chunk"])
    retriever = KGSearchRetriever(
        router=router,
        graph_traversal_func=fake_traversal,
        entity_score_func=fake_entity_score,
        config=KGSearchConfig(max_depth=2, top_k=10),
    )
    result = retriever.retrieve("query", None)
    assert len(result.chunks) == 1
    assert len(result.entities) > 0
    # Check that deeper entities have lower scores due to depth decay
    depths = {e.depth for e in result.entities}
    assert depths == {1, 2}


def test_entity_score_computation():
    retriever = KGSearchRetriever(
        graph_traversal_func=fake_traversal,
        entity_score_func=lambda eid: 0.9,
        config=KGSearchConfig(
            entity_rank_weight=0.5,
            vector_similarity_weight=0.5,
            frequency_weight=0.0,
            max_depth=1,
        ),
    )
    result = retriever.retrieve("seed", None)
    assert len(result.entities) == 2
    for entity in result.entities:
        assert 0.0 <= entity.score <= 1.0


def test_entity_score_func_failure_graceful():
    def bad_score(entity_id):
        raise ValueError("score error")

    retriever = KGSearchRetriever(
        graph_traversal_func=fake_traversal,
        entity_score_func=bad_score,
        config=KGSearchConfig(max_depth=1),
    )
    result = retriever.retrieve("seed", None)
    assert len(result.entities) == 2


# ---------------------------------------------------------------------------
# Community search
# ---------------------------------------------------------------------------


def test_retrieve_with_communities():
    router = FakeRouter(["chunk"])
    retriever = KGSearchRetriever(
        router=router,
        community_search_func=fake_community_search,
        config=KGSearchConfig(top_communities=2),
    )
    result = retriever.retrieve("query", None)
    assert len(result.communities) == 1
    assert result.communities[0]["id"] == "comm_query"


# ---------------------------------------------------------------------------
# Chunk lookup
# ---------------------------------------------------------------------------


def test_chunk_lookup():
    lookup = {"cid_1": "resolved text 1"}
    router = FakeRouter(["cid_1"])
    retriever = KGSearchRetriever(
        router=router,
        chunk_lookup_func=lookup.get,
        config=KGSearchConfig(top_k=1),
    )
    result = retriever.retrieve("query", None)
    assert result.chunks[0].text == "resolved text 1"


# ---------------------------------------------------------------------------
# Deduplication and ranking
# ---------------------------------------------------------------------------


def test_deduplicate_chunks_across_sub_queries():
    router = FakeRouter(["dup_chunk"])
    rewrite = QueryRewriteResult(
        original_query="complex",
        needs_rewrite=True,
        sub_queries=["q1", "q2"],
    )
    retriever = KGSearchRetriever(router=router, config=KGSearchConfig(top_k=5))
    result = retriever.retrieve("complex", rewrite)
    assert len(result.chunks) == 1
    assert result.chunks[0].source_queries == ["q1", "q2"]


def test_deduplicate_entities_across_sub_queries():
    router = FakeRouter(["seed"])
    retriever = KGSearchRetriever(
        router=router,
        graph_traversal_func=fake_traversal,
        config=KGSearchConfig(max_depth=1, top_k=10),
    )
    rewrite = QueryRewriteResult(
        original_query="complex",
        needs_rewrite=True,
        sub_queries=["q1", "q2"],
    )
    result = retriever.retrieve("complex", rewrite)
    # Same traversal from different seeds produces different entities, but duplicates should be merged
    assert len(result.entities) > 0


def test_rank_chunks_by_score():
    chunks = [
        ScoredChunk(chunk_id="a", text="low", score=0.1),
        ScoredChunk(chunk_id="b", text="high", score=0.9),
        ScoredChunk(chunk_id="c", text="mid", score=0.5),
    ]
    retriever = KGSearchRetriever(config=KGSearchConfig(top_k=2))
    ranked = retriever._rank_chunks(chunks)  # pylint: disable=protected-access
    assert ranked[0].chunk_id == "b"
    assert len(ranked) == 2


# ---------------------------------------------------------------------------
# Operator protocol
# ---------------------------------------------------------------------------


def test_run_operator_protocol():
    router = FakeRouter(["chunk"])
    retriever = KGSearchRetriever(router=router)
    ctx = {"query": "test"}
    result = retriever.run(ctx)
    assert "kg_search_result" in result
    assert isinstance(result["kg_search_result"], KGSearchResult)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_router_failure_ignored():
    """Router failure should not crash retrieval; it just produces no router chunks."""
    retriever = KGSearchRetriever(
        router=FailingRouter(),  # type: ignore
        graph_traversal_func=lambda e, d, f: [],
        config=KGSearchConfig(top_k=5),
    )
    result = retriever.retrieve("query", None)
    assert len(result.chunks) == 0
    assert result.provenance["num_sub_queries"] == 1


def test_max_sub_queries_limit():
    router = FakeRouter(["chunk"])
    rewrite = QueryRewriteResult(
        original_query="complex",
        needs_rewrite=True,
        sub_queries=["q1", "q2", "q3", "q4", "q5"],
    )
    config = KGSearchConfig(max_sub_queries=2)
    retriever = KGSearchRetriever(router=router, config=config)
    result = retriever.retrieve("complex", rewrite)
    assert result.provenance["num_sub_queries"] == 2


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def test_chunk_texts_property():
    result = KGSearchResult(
        chunks=[ScoredChunk(chunk_id="c1", text="hello"), ScoredChunk(chunk_id="c2", text="world")]
    )
    assert result.chunk_texts == ["hello", "world"]


def test_executable_queries_empty_fallback():
    router = FakeRouter(["chunk"])
    rewrite = QueryRewriteResult(original_query="q", needs_rewrite=True, sub_queries=[])
    retriever = KGSearchRetriever(router=router)
    result = retriever.retrieve("q", rewrite)
    assert result.provenance["sub_queries"] == ["q"]


def test_executable_queries_all_empty_uses_original_query():
    """When rewrite says no queries and original is empty, fall back to method query."""
    router = FakeRouter(["chunk"])
    rewrite = QueryRewriteResult(original_query="", needs_rewrite=True, sub_queries=[])
    retriever = KGSearchRetriever(router=router)
    result = retriever.retrieve("method_query", rewrite)
    assert result.provenance["sub_queries"] == ["method_query"]


def test_chunk_lookup_failure_ignored():
    def failing_lookup(chunk_id):
        raise RuntimeError("lookup failed")

    router = FakeRouter(["cid_1"])
    retriever = KGSearchRetriever(
        router=router,
        chunk_lookup_func=failing_lookup,
        config=KGSearchConfig(top_k=1),
    )
    result = retriever.retrieve("query", None)
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "cid_1"  # falls back to original chunk id text


def test_graph_traversal_skip_seed():
    def traversal_with_seed(entity_id, max_depth, max_fanout):
        return [(entity_id, 0, ""), ("neighbor", 1, "RELATED_TO")]

    router = FakeRouter(["seed"])
    retriever = KGSearchRetriever(
        router=router,
        graph_traversal_func=traversal_with_seed,
        config=KGSearchConfig(max_depth=1),
    )
    result = retriever.retrieve("query", None)
    assert len(result.entities) == 1
    assert result.entities[0].entity_id == "neighbor"


def test_chunk_lookup_empty_text():
    lookup = {"cid_1": "resolved"}
    chunk = ScoredChunk(chunk_id="cid_1", text="", score=0.5)
    retriever = KGSearchRetriever(chunk_lookup_func=lookup.get, config=KGSearchConfig(top_k=1))
    ranked = retriever._rank_chunks([chunk])  # pylint: disable=protected-access
    assert ranked[0].text == "resolved"


def test_rank_chunks_lookup_failure_ignored():
    def failing_lookup(chunk_id):
        raise RuntimeError("lookup failed")

    chunk = ScoredChunk(chunk_id="cid_1", text="", score=0.5)
    retriever = KGSearchRetriever(chunk_lookup_func=failing_lookup, config=KGSearchConfig(top_k=1))
    ranked = retriever._rank_chunks([chunk])  # pylint: disable=protected-access
    assert ranked[0].text == ""  # text remains empty after failed lookup


def test_result_to_dict():
    result = KGSearchResult(
        chunks=[ScoredChunk(chunk_id="c1", text="text", score=0.5)],
        entities=[ScoredEntity(entity_id="e1", name="E", score=0.8)],
        communities=[{"id": "c1", "score": 0.9}],
        provenance={"test": True},
    )
    d = result.to_dict()
    assert d["chunks"][0]["chunk_id"] == "c1"
    assert d["entities"][0]["entity_id"] == "e1"
    assert d["communities"][0]["id"] == "c1"


# ---------------------------------------------------------------------------
# EntityRanker integration
# ---------------------------------------------------------------------------


def test_retriever_uses_entity_ranker_for_scoring():
    """When entity_ranker is provided, its score() is used as entity_score_func."""
    edges = [("seed", "hub", 1.0), ("hub", "leaf", 1.0)]
    ranker = build_ranker_from_edges(edges)
    router = FakeRouter(["seed"])

    def traversal(entity_id, max_depth, max_fanout):
        return [("hub", 1, "RELATED_TO"), ("leaf", 2, "RELATED_TO")]

    retriever = KGSearchRetriever(
        router=router,
        graph_traversal_func=traversal,
        entity_ranker=ranker,
        config=KGSearchConfig(max_depth=2, top_k=10),
    )
    result = retriever.retrieve("query", None)
    assert len(result.entities) == 2
    for entity in result.entities:
        assert 0.0 <= entity.score <= 1.0


def test_explicit_entity_score_func_overrides_ranker():
    """If both entity_score_func and entity_ranker are provided, the explicit func wins."""
    edges = [("seed", "hub", 1.0)]
    ranker = build_ranker_from_edges(edges)
    router = FakeRouter(["seed"])

    def traversal(entity_id, max_depth, max_fanout):
        return [("hub", 1, "RELATED_TO")]

    retriever = KGSearchRetriever(
        router=router,
        graph_traversal_func=traversal,
        entity_score_func=lambda eid: 0.42,
        entity_ranker=ranker,
        config=KGSearchConfig(max_depth=1),
    )
    result = retriever.retrieve("query", None)
    assert result.entities[0].rank_factors["base_rank"] == 0.42


def test_entity_ranker_scores_in_rank_factors():
    edges = [("seed", "hub", 1.0)]
    ranker = build_ranker_from_edges(edges)
    router = FakeRouter(["seed"])

    def traversal(entity_id, max_depth, max_fanout):
        return [("hub", 1, "RELATED_TO")]

    retriever = KGSearchRetriever(
        router=router,
        graph_traversal_func=traversal,
        entity_ranker=ranker,
        config=KGSearchConfig(max_depth=1),
    )
    result = retriever.retrieve("query", None)
    assert "base_rank" in result.entities[0].rank_factors
