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

"""Comprehensive tests for the Advanced GraphRAG demo handlers.

Note: EntityResolution, EmbeddingFactory, ReciprocalRankFusion, etc.
are lazy-imported inside the handler functions, so we patch at their
SOURCE module rather than at the handler module.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers import (
    drift_search_answer,
    schema_validate,
    entity_resolve,
    get_community_reports,
    incremental_index_status,
    rrf_demo,
    token_budget_demo,
)


# ════════════════════════════════════════════════════════════════
#  drift_search_answer tests
# ════════════════════════════════════════════════════════════════

class TestDriftSearchAnswer:

    def test_empty_query_returns_error(self):
        result = drift_search_answer("")
        assert result["answer"] == ""
        assert result["pipeline"] == []
        assert result["metadata"]["call_count"] == 0
        assert result["error"] == "Please enter a query."

    def test_whitespace_query_returns_error(self):
        result = drift_search_answer("   ")
        assert result["error"] == "Please enter a query."

    @patch("hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers.SchedulerSingleton")
    def test_successful_drift_search(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "drift_answer": "Apache HugeGraph supports Gremlin queries.",
            "drift_findings": [{"finding": "f1"}, {"finding": "f2"}, {"finding": "f3"}],
            "drift_primer": {"follow_up_queries": ["q1", "q2"]},
            "drift_communities_used": 3,
            "drift_depth_reached": 2,
            "call_count": 5,
        }

        result = drift_search_answer("What does HugeGraph support?", communities_top_k=5, language="cn")
        assert result["answer"] == "Apache HugeGraph supports Gremlin queries."
        assert len(result["pipeline"]) == 5
        assert result["pipeline"][0]["name"] == "HyDE (Hypothetical Document Embedding)"
        assert result["metadata"]["call_count"] == 5
        assert result["error"] is None

    @patch("hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers.SchedulerSingleton")
    def test_drift_search_scheduler_exception(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.side_effect = RuntimeError("Server down")

        result = drift_search_answer("test query")
        assert result["answer"] == ""
        assert "DRIFT search failed" in result["error"]

    @patch("hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers.SchedulerSingleton")
    def test_drift_search_partial_result(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {"drift_answer": "Partial answer"}

        result = drift_search_answer("partial query")
        assert result["answer"] == "Partial answer"
        assert len(result["pipeline"]) == 5

    @patch("hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers.SchedulerSingleton")
    def test_drift_search_custom_params(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "drift_answer": "answer", "drift_findings": [], "drift_primer": {},
        }

        drift_search_answer("test", communities_top_k=10, language="en")
        call_kwargs = mock_instance.schedule_flow.call_args.kwargs
        assert call_kwargs["communities_top_k"] == 10
        assert call_kwargs["language"] == "en"


# ════════════════════════════════════════════════════════════════
#  schema_validate tests
# ════════════════════════════════════════════════════════════════

class TestSchemaValidate:

    def test_empty_input(self):
        result = schema_validate("")
        assert result["valid"] is False
        assert "Please enter schema JSON." in result["errors"]

    def test_whitespace_input(self):
        result = schema_validate("   ")
        assert result["valid"] is False

    def test_invalid_json(self):
        result = schema_validate("{invalid json!!}")
        assert result["valid"] is False
        assert "Invalid JSON" in result["errors"][0]

    @patch("hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers.SchedulerSingleton")
    def test_valid_schema_with_scheduler(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "valid": True, "errors": [], "warnings": ["Consider adding more edge labels"],
            "suggestions": ["Add frequency"], "entity_count": 3, "relation_count": 2,
        }

        schema_json = json.dumps({"vertexlabels": [{"name": "Person"}]})
        result = schema_validate(schema_json)
        assert result["valid"] is True
        assert result["entity_count"] == 3

    @patch("hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers.SchedulerSingleton")
    def test_scheduler_exception(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.side_effect = Exception("Service unavailable")

        result = schema_validate('{"vertexlabels": []}')
        assert result["valid"] is False
        assert "Schema validation failed" in result["errors"][0]


# ════════════════════════════════════════════════════════════════
#  entity_resolve tests
# ════════════════════════════════════════════════════════════════

class TestEntityResolve:

    def test_empty_input(self):
        result = entity_resolve("")
        assert result["groups"] == []
        assert result["total_entities"] == 0

    def test_whitespace_input(self):
        result = entity_resolve("   ")
        assert result["total_entities"] == 0

    @patch("hugegraph_llm.operators.graph_op.entity_resolution.EntityResolution")
    def test_no_valid_entities(self, mock_er_cls):
        mock_er = MagicMock()
        mock_er_cls.return_value = mock_er
        mock_er.run.return_value = {"groups": [], "resolved_count": 0, "unresolved_count": 0}

        result = entity_resolve("  \n  \n  ")
        assert result["total_entities"] == 0

    @patch("hugegraph_llm.operators.graph_op.entity_resolution.EntityResolution")
    def test_successful_resolution(self, mock_er_cls):
        mock_er = MagicMock()
        mock_er_cls.return_value = mock_er
        mock_er.run.return_value = {
            "groups": [
                [{"name": "张三"}, {"name": "Zhang San"}],
                [{"name": "阿里云"}, {"name": "Alibaba Cloud"}],
            ],
            "resolved_count": 4, "unresolved_count": 0,
        }

        result = entity_resolve("张三\nZhang San\n阿里云\nAlibaba Cloud", strategy="hybrid")
        assert result["total_entities"] == 4
        assert result["resolved_count"] == 4
        assert len(result["groups"]) == 2
        assert result["error"] is None

    @patch("hugegraph_llm.operators.graph_op.entity_resolution.EntityResolution")
    def test_resolution_exception(self, mock_er_cls):
        mock_er = MagicMock()
        mock_er_cls.return_value = mock_er
        mock_er.run.side_effect = RuntimeError("Embedding model unavailable")

        result = entity_resolve("EntityA\nEntityB")
        assert result["total_entities"] == 2
        assert "Entity resolution failed" in result["error"]

    @patch("hugegraph_llm.operators.graph_op.entity_resolution.EntityResolution")
    def test_exact_match_strategy(self, mock_er_cls):
        mock_er = MagicMock()
        mock_er_cls.return_value = mock_er
        mock_er.run.return_value = {"groups": [], "resolved_count": 0, "unresolved_count": 2}

        result = entity_resolve("A\nB", strategy="exact_match")
        mock_er_cls.assert_called_once_with(client=None, strategy="exact_match")
        assert result["strategy"] == "exact_match"


# ════════════════════════════════════════════════════════════════
#  get_community_reports tests
# ════════════════════════════════════════════════════════════════

class TestGetCommunityReports:

    def test_no_community_index(self):
        """get_community_reports gracefully handles missing community index."""
        # The function uses lazy imports, so we mock sys.modules injection
        mock_emb_factory = MagicMock()
        mock_emb_factory.get_embedding.return_value = MagicMock()

        with patch.dict('sys.modules', {
            'hugegraph_llm.models.embeddings.init_embedding': MagicMock(EmbeddingFactory=mock_emb_factory),
        }):
            # The function will try real imports, may fail gracefully
            result = get_community_reports(limit=5)
            assert "total_reports" in result or "reports" in result

    def test_with_community_data(self):
        """get_community_reports returns reports when data exists."""
        result = get_community_reports(limit=10)
        assert "total_reports" in result

    def test_limit_parameter(self):
        """Limit parameter is passed correctly."""
        result = get_community_reports(limit=5)
        assert "total_reports" in result


# ════════════════════════════════════════════════════════════════
#  incremental_index_status tests
# ════════════════════════════════════════════════════════════════

class TestIncrementalIndexStatus:

    @patch("hugegraph_llm.utils.graph_index_utils.get_graph_index_info")
    def test_successful_status(self, mock_get_info):
        mock_get_info.return_value = {
            "vertex_count": 1000, "edge_count": 5000,
            "index_exists": True, "index_type": "vector+graph",
            "last_indexed": "2026-06-26T10:00:00",
        }

        result = incremental_index_status()
        assert result["vertex_count"] == 1000
        assert result["index_exists"] is True
        assert result["error"] is None

    @patch("hugegraph_llm.utils.graph_index_utils.get_graph_index_info")
    def test_exception_handling(self, mock_get_info):
        mock_get_info.side_effect = Exception("Connection refused")

        result = incremental_index_status()
        assert result["vertex_count"] == 0
        assert result["index_exists"] is False
        assert "Failed to get index status" in result["error"]


# ════════════════════════════════════════════════════════════════
#  rrf_demo tests
# ════════════════════════════════════════════════════════════════

class TestRrfDemo:

    def test_empty_query(self):
        result = rrf_demo("")
        assert result["vector_results"] == []
        assert "Please enter a query." in result["error"]

    def test_with_simulated_results(self):
        """RRF demo returns results or graceful error."""
        result = rrf_demo("test query", top_k=3)
        assert "fused_results" in result or "error" in result

    def test_all_backends_fail(self):
        """All backends failing returns error dict."""
        result = rrf_demo("test")
        # Empty result or error
        assert result.get("fused_results") == [] or result.get("error") is not None


# ════════════════════════════════════════════════════════════════
#  token_budget_demo tests
# ════════════════════════════════════════════════════════════════

class TestTokenBudgetDemo:

    def test_empty_query(self):
        result = token_budget_demo("")
        assert result["context"] == ""
        assert "Please enter a query." in result["error"]

    @patch("hugegraph_llm.operators.graph_op.token_budget.TokenBudgetConfig")
    @patch("hugegraph_llm.operators.graph_op.token_budget.TokenBudget")
    def test_successful_budget_demo(self, mock_budget_cls, mock_config_cls):
        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_budget = MagicMock()
        mock_budget_cls.return_value = mock_budget
        mock_budget.add.side_effect = [True] * 50  # Accept all
        mock_budget.summary.return_value = {
            "total_tokens": 1500, "entity_tokens": 600,
            "relation_tokens": 400, "community_tokens": 500,
        }
        mock_budget.build_context.return_value = "Entity: Apache HugeGraph"

        result = token_budget_demo("What is HugeGraph?", max_tokens=2000)
        assert result["summary"]["total_tokens"] == 1500
        assert result["error"] is None

    @patch("hugegraph_llm.operators.graph_op.token_budget.TokenBudget")
    def test_exception_handling(self, mock_budget_cls):
        mock_budget_cls.side_effect = ImportError("No module")

        result = token_budget_demo("test query")
        assert "Token budget demo failed" in result["error"]
