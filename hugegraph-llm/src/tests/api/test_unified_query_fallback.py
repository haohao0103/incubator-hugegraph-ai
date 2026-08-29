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
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from hugegraph_llm.api.models.unified_requests import UnifiedQueryRequest
from hugegraph_llm.api.unified_query_api import (
    _apply_fallback,
    _graphrag,
    _text2gremlin,
    unified_query,
    unified_query_http_api,
)

pytestmark = [pytest.mark.unit]


class TestApplyFallback(unittest.TestCase):
    def test_fallback_overrides_empty_answer(self):
        from hugegraph_llm.api.models.unified_requests import UnifiedQueryResponse

        resp = UnifiedQueryResponse(answer="", route="precise")
        out = _apply_fallback(resp, "no data")
        self.assertEqual(out.answer, "no data")

    def test_fallback_keeps_non_empty_answer(self):
        from hugegraph_llm.api.models.unified_requests import UnifiedQueryResponse

        resp = UnifiedQueryResponse(answer="real", route="precise")
        out = _apply_fallback(resp, "no data")
        self.assertEqual(out.answer, "real")

    def test_no_fallback_keeps_empty(self):
        from hugegraph_llm.api.models.unified_requests import UnifiedQueryResponse

        resp = UnifiedQueryResponse(answer="", route="precise")
        out = _apply_fallback(resp, None)
        self.assertEqual(out.answer, "")


class TestUnifiedQueryFallback(unittest.TestCase):
    @patch("hugegraph_llm.api.unified_query_api._text2gremlin")
    def test_precise_empty_answer_uses_fallback(self, mock_t2g):
        mock_t2g.return_value = {"raw_execution_result": "", "template_execution_result": "", "match_result": []}
        req = UnifiedQueryRequest(question="q", mode="precise", response_fallback="no data found")
        resp = unified_query(req)
        self.assertEqual(resp.answer, "no data found")
        self.assertEqual(resp.route, "precise")

    @patch("hugegraph_llm.api.unified_query_api._text2gremlin")
    def test_precise_real_answer_kept(self, mock_t2g):
        mock_t2g.return_value = {"raw_execution_result": "real answer", "match_result": []}
        req = UnifiedQueryRequest(question="q", mode="precise", response_fallback="no data found")
        resp = unified_query(req)
        self.assertEqual(resp.answer, "real answer")

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_hybrid_empty_answer_uses_fallback(self, mock_rag):
        mock_rag.return_value = {"graph_vector_answer": "", "vector_only_answer": ""}
        req = UnifiedQueryRequest(question="q", mode="hybrid", response_fallback="fallback-hybrid")
        resp = unified_query(req)
        self.assertEqual(resp.answer, "fallback-hybrid")
        self.assertEqual(resp.route, "graphrag")

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_semantic_empty_answer_uses_fallback(self, mock_rag):
        mock_rag.return_value = {"graph_vector_answer": "", "vector_only_answer": ""}
        req = UnifiedQueryRequest(question="q", mode="semantic", response_fallback="fb-sem")
        resp = unified_query(req)
        self.assertEqual(resp.answer, "fb-sem")
        self.assertEqual(resp.route, "semantic")

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_hybrid_real_answer_kept(self, mock_rag):
        mock_rag.return_value = {"graph_vector_answer": "real rag", "vector_only_answer": ""}
        req = UnifiedQueryRequest(question="q", mode="hybrid", response_fallback="fb")
        resp = unified_query(req)
        self.assertEqual(resp.answer, "real rag")

    @patch("hugegraph_llm.api.unified_query_api._text2gremlin")
    def test_auto_match_routes_to_precise_with_fallback(self, mock_t2g):
        mock_t2g.return_value = {
            "raw_execution_result": "",
            "match_result": [{"id": "Table:order"}],
        }
        req = UnifiedQueryRequest(question="q", mode="auto", response_fallback="fb-auto")
        resp = unified_query(req)
        self.assertEqual(resp.route, "precise")
        self.assertEqual(resp.answer, "fb-auto")

    @patch("hugegraph_llm.api.unified_query_api._text2gremlin")
    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_auto_no_match_falls_to_hybrid(self, mock_rag, mock_t2g):
        mock_t2g.return_value = {"match_result": []}
        mock_rag.return_value = {"graph_vector_answer": "", "vector_only_answer": ""}
        req = UnifiedQueryRequest(question="q", mode="auto", response_fallback="fb-auto-hybrid")
        resp = unified_query(req)
        self.assertEqual(resp.route, "graphrag")
        self.assertEqual(resp.answer, "fb-auto-hybrid")

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_retriever_config_overrides_top_k(self, mock_rag):
        mock_rag.return_value = {"graph_vector_answer": "a"}
        req = UnifiedQueryRequest(
            question="q", mode="hybrid", retriever_config={"top_k": 42}
        )
        unified_query(req)
        args, _ = mock_rag.call_args
        self.assertEqual(args[1], 42)

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_retriever_config_overrides_vector_only_and_graph_search(self, mock_rag):
        mock_rag.return_value = {"graph_vector_answer": "a"}
        req = UnifiedQueryRequest(
            question="q",
            mode="semantic",
            retriever_config={"vector_only": False, "graph_search": True},
        )
        unified_query(req)
        _, kwargs = mock_rag.call_args
        self.assertFalse(kwargs["vector_only"])
        self.assertTrue(kwargs["graph_search"])

    def test_empty_question_raises(self):
        with self.assertRaises(HTTPException) as cm:
            unified_query(UnifiedQueryRequest(question="  "))
        self.assertEqual(cm.exception.status_code, 400)


class TestQueryHelpers(unittest.TestCase):
    @patch("hugegraph_llm.api.unified_query_api.SchedulerSingleton")
    def test_text2gremlin_schedules_flow(self, mock_singleton):
        mock_scheduler = MagicMock()
        mock_scheduler.schedule_flow.return_value = {"ok": True}
        mock_singleton.get_instance.return_value = mock_scheduler

        out = _text2gremlin("what is order?")

        self.assertEqual(out, {"ok": True})
        args, _ = mock_scheduler.schedule_flow.call_args
        self.assertEqual(args[1], "what is order?")
        self.assertIn("raw_execution_result", args[5])

    @patch("hugegraph_llm.api.unified_query_api.SchedulerSingleton")
    def test_graphrag_schedules_rag_flow(self, mock_singleton):
        mock_scheduler = MagicMock()
        mock_scheduler.schedule_flow.return_value = {"graph_vector_answer": "a"}
        mock_singleton.get_instance.return_value = mock_scheduler

        out = _graphrag("q", 7, graph_search=True, vector_only=False)

        self.assertEqual(out, {"graph_vector_answer": "a"})
        _, kwargs = mock_scheduler.schedule_flow.call_args
        self.assertEqual(kwargs["query"], "q")
        self.assertEqual(kwargs["topk_return_results"], 7)
        self.assertTrue(kwargs["graph_search"])
        self.assertFalse(kwargs["vector_only_answer"])


class TestUnifiedQueryHttpApi(unittest.TestCase):
    def test_router_registers_endpoint(self):
        router = MagicMock()
        decorator = MagicMock()
        router.post.return_value = decorator
        unified_query_http_api(router)
        router.post.assert_called_once()
        path = router.post.call_args.args[0]
        self.assertEqual(path, "/api/v1/query")
        decorator.assert_called_once()

    @patch("hugegraph_llm.api.unified_query_api.unified_query")
    def test_handler_success(self, mock_uq):
        router = MagicMock()
        decorator = MagicMock()
        router.post.return_value = decorator
        unified_query_http_api(router)
        handler = decorator.call_args.args[0]

        mock_uq.return_value = UnifiedQueryRequest(
            question="q", mode="precise"
        )  # placeholder; response is a Pydantic model
        from hugegraph_llm.api.models.unified_requests import UnifiedQueryResponse

        mock_uq.return_value = UnifiedQueryResponse(answer="ok")
        resp = handler(UnifiedQueryRequest(question="q"))
        self.assertEqual(resp.answer, "ok")
        mock_uq.assert_called_once()

    @patch("hugegraph_llm.api.unified_query_api.unified_query")
    def test_handler_rethrows_http_exception(self, mock_uq):
        router = MagicMock()
        decorator = MagicMock()
        router.post.return_value = decorator
        unified_query_http_api(router)
        handler = decorator.call_args.args[0]

        mock_uq.side_effect = HTTPException(status_code=400, detail="bad")
        with self.assertRaises(HTTPException):
            handler(UnifiedQueryRequest(question="q"))

    @patch("hugegraph_llm.api.unified_query_api.unified_query")
    def test_handler_wraps_unknown_exception(self, mock_uq):
        router = MagicMock()
        decorator = MagicMock()
        router.post.return_value = decorator
        unified_query_http_api(router)
        handler = decorator.call_args.args[0]

        mock_uq.side_effect = RuntimeError("boom")
        with self.assertRaises(HTTPException) as cm:
            handler(UnifiedQueryRequest(question="q"))
        self.assertEqual(cm.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
