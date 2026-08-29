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

import unittest
from unittest.mock import MagicMock

import pytest

from hugegraph_llm.operators.graph_op.kg_retriever_base import RetrieverResult
from hugegraph_llm.operators.graph_op.kg_search_retriever import (
    KGSearchConfig,
    KGSearchResult,
    KGSearchRetriever,
    ScoredChunk,
)
from hugegraph_llm.operators.llm_op.query_rewrite import QueryRewriteResult

pytestmark = [pytest.mark.unit]


class _FakeRouteResult:
    def __init__(self, chunks=None):
        self.chunks = chunks or ["c1", "c2"]
        self.provenance = {"mode": "local"}


def _make_retriever(**overrides):
    router = MagicMock()
    router.route.return_value = _FakeRouteResult()
    defaults = dict(
        router=router,
        graph_traversal_func=lambda eid, depth, fanout: [("n1", 1, "knows")],
        entity_score_func=lambda eid: 0.8,
        community_search_func=lambda q, k: [],
        chunk_lookup_func=lambda cid: f"text-of-{cid}",
    )
    defaults.update(overrides)
    return KGSearchRetriever(**defaults)


class TestKGSearchRetrieverRetrieve(unittest.TestCase):
    def test_retrieve_with_router(self):
        r = _make_retriever()
        result = r.retrieve("who knows Tom?")
        self.assertIsInstance(result, KGSearchResult)
        self.assertGreaterEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "text-of-c1")
        self.assertEqual(result.provenance["original_query"], "who knows Tom?")
        self.assertEqual(result.provenance["num_sub_queries"], 1)

    def test_retrieve_empty_query(self):
        r = _make_retriever()
        result = r.retrieve("")
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.entities, [])
        self.assertEqual(result.provenance, {})

    def test_retrieve_with_rewrite(self):
        r = _make_retriever()
        rewrite = MagicMock(spec=QueryRewriteResult)
        rewrite.executable_queries = ["q1", "q2"]
        rewrite.has_rewritten = True
        rewrite.rewritten_query = "q"
        result = r.retrieve("orig", rewrite)
        self.assertEqual(result.provenance["num_sub_queries"], 2)

    def test_retrieve_rewrite_empty_queries_falls_back_to_query(self):
        r = _make_retriever()
        rewrite = MagicMock(spec=QueryRewriteResult)
        rewrite.executable_queries = []
        result = r.retrieve("orig", rewrite)
        self.assertEqual(result.provenance["num_sub_queries"], 1)

    def test_router_failure_is_tolerated(self):
        router = MagicMock()
        router.route.side_effect = RuntimeError("router down")
        r = _make_retriever(router=router)
        result = r.retrieve("q")
        self.assertEqual(result.provenance["sub_queries"], ["q"])

    def test_no_router_uses_subquery_seed(self):
        r = _make_retriever(router=None)
        result = r.retrieve("q")
        # sub-query becomes the seed; traversal yields n1
        self.assertGreaterEqual(len(result.entities), 1)

    def test_external_seeds_are_appended(self):
        r = _make_retriever(external_seed_entity_ids=["ext1"])
        result = r.retrieve("q")
        self.assertGreaterEqual(len(result.entities), 1)


class TestKGSearchRetrieverSearchProtocol(unittest.TestCase):
    """KGRetriever unified entry: search() -> RetrieverResult."""

    def test_search_returns_retriever_result(self):
        r = _make_retriever()
        result = r.search("who knows Tom?")
        self.assertIsInstance(result, RetrieverResult)
        self.assertGreaterEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.content, "text-of-c1")
        self.assertEqual(item.metadata["chunk_id"], "c1")
        self.assertEqual(item.score, 0.5)
        self.assertEqual(result.metadata["__retriever"], "KGSearchRetriever")

    def test_search_forward_rewrite_kwarg(self):
        r = _make_retriever()
        rewrite = MagicMock(spec=QueryRewriteResult)
        rewrite.executable_queries = ["q1"]
        result = r.search("orig", rewrite=rewrite)
        self.assertEqual(result.metadata["num_sub_queries"], 1)

    def test_search_empty_query_is_empty(self):
        r = _make_retriever()
        self.assertTrue(r.search("").is_empty)

    def test_get_search_results_delegates_to_retrieve(self):
        r = _make_retriever()
        raw = r.get_search_results("q")
        self.assertIsInstance(raw, KGSearchResult)
        self.assertEqual(raw.provenance["original_query"], "q")

    def test_kg_chunk_formatter(self):
        chunk = ScoredChunk(chunk_id="x", text="hello", score=0.7)
        item = KGSearchRetriever._kg_chunk_formatter(chunk)
        self.assertEqual(item.content, "hello")
        self.assertEqual(item.metadata["chunk_id"], "x")
        self.assertEqual(item.score, 0.7)

    def test_kg_search_result_to_dict_and_chunk_texts(self):
        result = KGSearchResult(
            chunks=[ScoredChunk(chunk_id="c1", text="t1", score=0.9)],
            entities=[],
        )
        d = result.to_dict()
        self.assertEqual(d["chunks"][0]["chunk_id"], "c1")
        self.assertEqual(result.chunk_texts, ["t1"])

    def test_run_operator_protocol(self):
        r = _make_retriever()
        ctx = {"query": "q", "query_rewrite": None}
        out = r.run(ctx)
        self.assertIsInstance(out["kg_search_result"], KGSearchResult)

    def test_entity_ranker_fallback(self):
        ranker = MagicMock()
        ranker.score.return_value = 0.9
        r = _make_retriever(entity_score_func=None, entity_ranker=ranker)
        self.assertIs(r._entity_score, ranker.score)

    def test_chunk_lookup_failure_tolerated(self):
        def boom(cid):
            raise RuntimeError("lookup down")

        r = _make_retriever(chunk_lookup_func=boom)
        result = r.retrieve("q")
        # chunk text falls back to the raw chunk id
        self.assertEqual(result.chunks[0].text, "c1")

    def test_no_chunk_lookup_uses_raw_text(self):
        r = _make_retriever(chunk_lookup_func=None)
        result = r.retrieve("q")
        self.assertEqual(result.chunks[0].text, "c1")

    def test_no_graph_traversal_no_communities(self):
        r = _make_retriever(
            graph_traversal_func=None,
            community_search_func=None,
        )
        result = r.retrieve("q")
        self.assertEqual(result.entities, [])
        self.assertEqual(result.communities, [])

    def test_self_loop_neighbor_skipped(self):
        r = _make_retriever(
            graph_traversal_func=lambda eid, depth, fanout: [("q", 0, "self")],
            router=None,
        )
        # seed is the sub-query "q"; self-loop (q,0) is skipped
        result = r.retrieve("q")
        self.assertEqual(result.entities, [])

    def test_entity_score_failure_tolerated(self):
        def boom(eid):
            raise RuntimeError("score down")

        r = _make_retriever(
            entity_score_func=boom,
            router=None,
            graph_traversal_func=lambda eid, depth, fanout: [("n1", 1, "knows")],
        )
        result = r.retrieve("q")
        self.assertEqual(result.entities[0].entity_id, "n1")

    def test_no_entity_score_uses_base_rank(self):
        r = _make_retriever(
            entity_score_func=None,
            router=None,
            graph_traversal_func=lambda eid, depth, fanout: [("n1", 1, "knows")],
        )
        result = r.retrieve("q")
        self.assertGreaterEqual(result.entities[0].score, 0.0)

    def test_external_seed_dedup_with_router_seed(self):
        router = MagicMock()
        router.route.return_value = _FakeRouteResult(chunks=["c1"])
        r = _make_retriever(
            router=router,
            external_seed_entity_ids=["c1", "ext2"],
            graph_traversal_func=lambda eid, depth, fanout: [],
            community_search_func=None,
        )
        result = r.retrieve("q")
        self.assertEqual(result.entities, [])

    def test_empty_router_chunk_seed_skipped(self):
        router = MagicMock()
        router.route.return_value = _FakeRouteResult(chunks=[""])
        r = _make_retriever(
            router=router,
            graph_traversal_func=lambda eid, depth, fanout: [],
            community_search_func=None,
            chunk_lookup_func=None,
        )
        result = r.retrieve("q")
        # empty chunk skipped as seed; sub-query used as fallback seed
        self.assertEqual(result.chunks[0].chunk_id, "")

    def test_merge_chunks_accumulates_source_queries(self):
        router = MagicMock()
        router.route.return_value = _FakeRouteResult(chunks=["c1"])
        r = _make_retriever(router=router)
        rewrite = MagicMock(spec=QueryRewriteResult)
        rewrite.executable_queries = ["q1", "q2"]
        result = r.retrieve("orig", rewrite)
        self.assertEqual(result.provenance["num_sub_queries"], 2)

    def test_rank_chunks_fills_missing_text_via_lookup(self):
        router = MagicMock()
        router.route.return_value = _FakeRouteResult(chunks=["c1"])
        lookup = MagicMock(side_effect=lambda cid: f"resolved-{cid}")
        r = _make_retriever(router=router, chunk_lookup_func=lookup)
        result = r.retrieve("q")
        self.assertEqual(result.chunks[0].text, "resolved-c1")

    def test_rank_chunks_lookup_failure_tolerated(self):
        router = MagicMock()
        router.route.return_value = _FakeRouteResult(chunks=["c1"])

        def boom(cid):
            raise RuntimeError("boom")

        r = _make_retriever(router=router, chunk_lookup_func=boom)
        result = r.retrieve("q")
        self.assertEqual(result.chunks[0].text, "c1")

    def test_merge_chunks_same_subquery_source_dedup(self):
        # same sub-query merging the same chunk twice: source_queries stays unique
        router = MagicMock()
        router.route.return_value = _FakeRouteResult(chunks=["c1"])
        r = _make_retriever(router=router)
        rewrite = MagicMock(spec=QueryRewriteResult)
        rewrite.executable_queries = ["q1", "q1"]
        result = r.retrieve("orig", rewrite)
        self.assertEqual(result.chunks[0].source_queries, ["q1"])

    def test_rank_chunks_empty_text_lookup_failure_tolerated(self):
        def boom(cid):
            raise RuntimeError("lookup down")

        r = _make_retriever(chunk_lookup_func=boom)
        chunk = ScoredChunk(chunk_id="x", text="", score=0.5)
        ranked = r._rank_chunks([chunk])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].text, "")

    def test_rank_communities_dedupes(self):
        r = _make_retriever(community_search_func=lambda q, k: [
            {"id": "c1"}, {"id": "c1"}, {"community_id": "c2"},
        ])
        result = r.retrieve("q")
        ids = [c.get("id") or c.get("community_id") for c in result.communities]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
