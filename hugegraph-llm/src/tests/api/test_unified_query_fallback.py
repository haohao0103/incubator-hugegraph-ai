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
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from hugegraph_llm.api.models.unified_requests import UnifiedQueryRequest
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine
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

    @patch("hugegraph_llm.api.unified_query_api._text2gremlin")
    def test_precise_stages_contract(self, mock_t2g):
        mock_t2g.return_value = {
            "raw_execution_result": "real answer",
            "template_gremlin": "g.V().hasLabel('Table')",
            "raw_gremlin": "raw-gremlin",
            "match_result": [{"id": "Table:order"}],
            "template_execution_result": ["row1"],
        }
        resp = unified_query(UnifiedQueryRequest(question="how many orders?", mode="precise"))
        stages = resp.stages
        self.assertEqual([st.stage for st in stages], ["text2gremlin", "graph_execution"])
        # text2gremlin stage carries question input + generated gremlin output
        self.assertEqual(stages[0].input["question"], "how many orders?")
        self.assertEqual(stages[0].output["raw_gremlin"], "raw-gremlin")
        self.assertEqual(stages[0].output["match_result"], [{"id": "Table:order"}])
        # graph_execution carries the queried data
        self.assertEqual(stages[1].output["raw_execution_result"], "real answer")
        # raw stays for backward compat
        self.assertEqual(resp.raw["template_gremlin"], "g.V().hasLabel('Table')")

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_hybrid_stages_contract(self, mock_rag):
        mock_rag.return_value = {
            "graph_vector_answer": "real rag",
            "graph_only_answer": "graph-part",
            "vector_only_answer": "vec-part",
            "query_intent": "count",
            "retrieval_level": "dual",
        }
        resp = unified_query(UnifiedQueryRequest(question="q", mode="hybrid"))
        stages = resp.stages
        # hybrid = graph_execution + vector_recall
        self.assertEqual([st.stage for st in stages], ["graph_execution", "vector_recall"])
        self.assertEqual(stages[0].output["graph_only_answer"], "graph-part")
        self.assertEqual(stages[1].output["top_k"], 5)
        self.assertEqual(stages[1].output["retrieval_level"], "dual")

    @patch("hugegraph_llm.api.unified_query_api._graphrag")
    def test_semantic_stages_contract_vector_only(self, mock_rag):
        mock_rag.return_value = {"vector_only_answer": "vec", "retrieval_level": "single"}
        resp = unified_query(UnifiedQueryRequest(question="q", mode="semantic"))
        # semantic = vector_recall only (no graph_execution)
        self.assertEqual([st.stage for st in resp.stages], ["vector_recall"])
        self.assertEqual(resp.stages[0].output["vector_only_answer"], "vec")

    def test_query_stage_builder_and_model(self):
        from hugegraph_llm.api.models.unified_requests import QueryStageBuilder

        stage = QueryStageBuilder.make("reasoning", output={"rules": ["r1"]}, input={"q": "x"})
        self.assertEqual(stage.stage, "reasoning")
        self.assertEqual(stage.input, {"q": "x"})
        self.assertEqual(stage.output, {"rules": ["r1"]})
        # model round-trip
        d = stage.model_dump()
        self.assertEqual(d["stage"], "reasoning")

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


def _sample_graph() -> Dict[str, Any]:
    return {
        "vertices": {
            "Table": [
                {"name": "order", "comment": "订单表"},
                {"name": "payment", "comment": "支付表"},
            ],
            "Field": [
                {"name": "order.amount"},
                {"name": "order.city"},
                {"name": "payment.amount"},
            ],
            "Metric": [
                {"name": "order_total", "formula": "SUM(order.amount)",
                 "definition": "订单总额"},
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.amount"),
                ("order", "order.city"),
                ("payment", "payment.amount"),
            ],
            "computedFrom": [("order_total", "order")],
            "computedFromField": [("order_total", "order.amount")],
            "dependsOn": [],
        },
    }


def _fake_llms():
    class _FakeLLM:
        def generate(self, prompt):
            return "```sql\nSELECT SUM(order.amount) FROM order\n```"

    class _FakeLLMs:
        def get_text2gql_llm(self):
            return _FakeLLM()

    return _FakeLLMs


class TestUnifiedQueryNl2Sql(unittest.TestCase):
    """Wiring of ``mode='nl2sql'`` through unified_query (P0 + P1 pipeline)."""

    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    @patch("hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline.KgNL2SQLPipeline")
    def test_nl2sql_routes_with_domain_and_client(self, mock_pipe_cls, mock_client):
        from hugegraph_llm.api.models.unified_requests import UnifiedQueryResponse

        mock_client.return_value = object()
        mock_instance = MagicMock()
        mock_instance.run.return_value = UnifiedQueryResponse(
            answer="SELECT SUM(order.amount) FROM order", route="nl2sql"
        )
        mock_pipe_cls.return_value = mock_instance

        req = UnifiedQueryRequest(question="订单总额", mode="nl2sql", domain="finance")
        resp = unified_query(req)

        self.assertEqual(resp.answer, "SELECT SUM(order.amount) FROM order")
        self.assertEqual(resp.route, "nl2sql")
        mock_pipe_cls.assert_called_once()
        _, kwargs = mock_pipe_cls.call_args
        self.assertEqual(kwargs["question"], "订单总额")
        self.assertEqual(kwargs["domain"], "finance")
        self.assertIs(kwargs["client"], mock_client.return_value)
        mock_instance.run.assert_called_once_with()

    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    @patch("hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline.KgNL2SQLPipeline")
    def test_nl2sql_empty_answer_uses_fallback(self, mock_pipe_cls, mock_client):
        from hugegraph_llm.api.models.unified_requests import UnifiedQueryResponse

        mock_client.return_value = object()
        mock_instance = MagicMock()
        mock_instance.run.return_value = UnifiedQueryResponse(
            answer="", route="nl2sql"
        )
        mock_pipe_cls.return_value = mock_instance

        req = UnifiedQueryRequest(
            question="q", mode="nl2sql", response_fallback="no sql available"
        )
        resp = unified_query(req)
        self.assertEqual(resp.answer, "no sql available")

    @patch("hugegraph_llm.models.llms.init_llm.LLMs", _fake_llms())
    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_nl2sql_real_pipeline_through_api(self, mock_client, _r):
        # real pipeline (no KgNL2SQLPipeline patch) but offline: mocked graph
        # server + mocked glm-5.3 generation
        mock_client.return_value = object()
        req = UnifiedQueryRequest(question="订单总额", mode="nl2sql")
        resp = unified_query(req)
        self.assertEqual(resp.route, "nl2sql")
        self.assertEqual(resp.answer, "SELECT SUM(order.amount) FROM order")
        stage_names = [s.stage for s in resp.stages]
        self.assertEqual(
            stage_names[:5],
            ["linking", "sql_generation", "sql_validation", "sql_voting", "lineage"],
        )
        gen = next(s for s in resp.stages if s.stage == "sql_generation")
        self.assertEqual(gen.output["source"], "llm")


class TestUnifiedQuerySchemaMode(unittest.TestCase):
    """mode='schema': schema retrieval + no-evidence refusal."""

    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_schema_no_evidence_refuses(self, mock_client, _load):
        mock_client.return_value = object()
        req = UnifiedQueryRequest(question="风控引擎实时决策表在哪里", mode="schema")
        resp = unified_query(req)
        self.assertTrue(resp.no_evidence)
        self.assertEqual(resp.route, "schema")
        self.assertEqual(resp.answer, "未找到相关元数据，可能需加工")
        self.assertTrue(resp.subgraph.get("no_evidence"))

    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_schema_no_evidence_custom_fallback_wins(self, mock_client, _load):
        mock_client.return_value = object()
        req = UnifiedQueryRequest(
            question="风控引擎实时决策表在哪里",
            mode="schema",
            response_fallback="自定义拒答：未找到，需加工",
        )
        resp = unified_query(req)
        self.assertTrue(resp.no_evidence)
        self.assertEqual(resp.answer, "自定义拒答：未找到，需加工")

    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_schema_linked_returns_evidence(self, mock_client, _load):
        mock_client.return_value = object()
        req = UnifiedQueryRequest(question="订单总额", mode="schema")
        resp = unified_query(req)
        self.assertFalse(resp.no_evidence)
        self.assertEqual(resp.route, "schema")
        self.assertIn("order_total", resp.subgraph.get("metrics", []))
        stage_names = [s.stage for s in resp.stages]
        self.assertIn("query_understanding", stage_names)
        self.assertIn("schema_retrieval", stage_names)
        # intent trace flows through raw
        self.assertIn("expanded_terms", resp.raw["intent"])

    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_schema_importance_weight_and_table_hit(self, mock_client, _load):
        mock_client.return_value = object()
        req = UnifiedQueryRequest(
            question="订单表",
            mode="schema",
            retriever_config={"importance_weight": 0.5},
        )
        resp = unified_query(req)
        self.assertFalse(resp.no_evidence)
        self.assertIn("order", resp.subgraph.get("tables", []))
        self.assertIn("订单表", resp.answer)

    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_schema_intent_weight_override(self, mock_client, _load):
        mock_client.return_value = object()
        # explicit intent_weight in retriever_config overrides the default
        req = UnifiedQueryRequest(
            question="订单总额是多少",
            mode="schema",
            retriever_config={"intent_weight": 0.0},
        )
        resp = unified_query(req)
        self.assertFalse(resp.no_evidence)
        self.assertIn("order_total", resp.subgraph.get("metrics", []))
        self.assertNotIn("intent_weight", resp.raw)  # raw only carries intent trace

    @patch.object(
        KgRuleEngine,
        "load_graph",
        return_value={
            "vertices": {
                "Table": [],
                "Field": [],
                "Metric": [{"name": "m1"}],  # no formula/definition
            },
            "edges": {
                "hasColumn": [], "computedFrom": [], "computedFromField": [],
                "dependsOn": [],
            },
        },
    )
    @patch("hugegraph_llm.utils.hugegraph_utils.get_hg_client")
    def test_schema_empty_fields_and_evidence_branches(self, mock_client, _load):
        mock_client.return_value = object()
        req = UnifiedQueryRequest(question="m1", mode="schema")
        resp = unified_query(req)
        self.assertFalse(resp.no_evidence)
        self.assertIn("m1", resp.subgraph.get("metrics", []))
        self.assertEqual(resp.subgraph.get("fields"), [])
        self.assertEqual(resp.raw.get("evidence"), [])


if __name__ == "__main__":
    unittest.main()
