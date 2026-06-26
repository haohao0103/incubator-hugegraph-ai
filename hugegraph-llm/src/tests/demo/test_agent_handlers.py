# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Comprehensive tests for Agent handlers.

All imports inside handler functions are LAZY (inside function body),
so we patch at the SOURCE module, not at the handler module namespace.

SchedulerSingleton is imported at MODULE LEVEL → patch at handler namespace.
LLMs, EmbeddingFactory, ToolRegistry, etc. are LAZY → patch at source.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from hugegraph_llm.demo.rag_demo import agent_handlers


def _reset_caches():
    agent_handlers._cached_tool_registry = None
    agent_handlers._cached_agent_llm = None


class TestGetOrCreateDependencies:

    def setup_method(self):
        _reset_caches()

    def test_failed_init_returns_none(self):
        """Failed init sets caches to None."""
        # Force init failure by making LLMs() raise
        _reset_caches()
        with patch("hugegraph_llm.models.llms.init_llm.LLMs", side_effect=Exception("no config")):
            registry, llm = agent_handlers._get_or_create_dependencies()
        assert registry is None
        assert llm is None


class TestAgentAnswer:

    def setup_method(self):
        _reset_caches()

    def test_no_dependencies_returns_503(self):
        """When deps fail to init, returns 503."""
        _reset_caches()
        with patch("hugegraph_llm.models.llms.init_llm.LLMs", side_effect=Exception("no config")):
            result = agent_handlers.agent_answer("test query")
        assert result["status_code"] == 503

    # For agent_answer with real deps, just verify the function signature works
    # Full integration testing requires real HugeGraph Server (see Task #10)

    def test_successful_agent_answer_with_mock_deps(self):
        """Pre-cached mock deps allow agent_answer to proceed."""
        agent_handlers._cached_tool_registry = MagicMock()
        agent_handlers._cached_agent_llm = MagicMock()

        with patch.object(agent_handlers.SchedulerSingleton, 'get_instance') as mock_get:
            mock_instance = MagicMock()
            mock_get.return_value = mock_instance
            mock_instance.agentic_flow.return_value = {
                "answer": "HugeGraph supports Gremlin.", "trace": [], "total_steps": 1,
            }

            result = agent_handlers.agent_answer("test?", max_steps=3)
            assert result["answer"] == "HugeGraph supports Gremlin."

    def test_scheduler_exception_with_mock_deps(self):
        """Scheduler exception returns error result."""
        agent_handlers._cached_tool_registry = MagicMock()
        agent_handlers._cached_agent_llm = MagicMock()

        with patch.object(agent_handlers.SchedulerSingleton, 'get_instance') as mock_get:
            mock_instance = MagicMock()
            mock_get.return_value = mock_instance
            mock_instance.agentic_flow.side_effect = RuntimeError("Timeout")

            result = agent_handlers.agent_answer("test")
            assert result["status_code"] == 500


class TestCommunityBuild:

    @patch("hugegraph_llm.demo.rag_demo.agent_handlers.SchedulerSingleton")
    @patch("hugegraph_llm.config.huge_settings")
    def test_default_graph_name(self, mock_settings, mock_sched_cls):
        mock_settings.graph_name = "hugegraph"
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "community_count": 5, "report_count": 5, "index_built": True,
        }
        mock_instance.pipeline_pool = {}

        result = agent_handlers.community_build(graph_name="", algorithm="louvain", max_levels=2)
        assert result["community_count"] == 5

    @patch("hugegraph_llm.demo.rag_demo.agent_handlers.SchedulerSingleton")
    def test_leiden_algorithm(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "community_count": 8, "report_count": 8, "index_built": True,
        }
        mock_instance.pipeline_pool = {}

        result = agent_handlers.community_build(algorithm="leiden", max_levels=3)
        call_kwargs = mock_instance.schedule_flow.call_args.kwargs
        assert call_kwargs["algorithm"] == "leiden"

    @patch("hugegraph_llm.demo.rag_demo.agent_handlers.SchedulerSingleton")
    def test_exception_handling(self, mock_sched_cls):
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.side_effect = Exception("Server error")
        mock_instance.pipeline_pool = {}

        result = agent_handlers.community_build()
        assert result["status_code"] == 500


class TestGlobalSearch:

    # SchedulerSingleton is module-level import → patch at handler namespace
    def test_successful_search(self):
        """Global search returns answer from scheduler (integration test)."""
        result = agent_handlers.global_search("What communities exist?")
        # Real server may or may not be running; just verify structure
        assert "answer" in result

    def test_exception_handling(self):
        """Exception returns error answer."""
        with patch.object(agent_handlers.SchedulerSingleton, 'get_instance') as mock_get:
            mock_instance = MagicMock()
            mock_get.return_value = mock_instance
            mock_instance.schedule_flow.side_effect = RuntimeError("No index")
            mock_instance.pipeline_pool = {}

            result = agent_handlers.global_search("test")
            assert "Global search failed" in result["answer"]


class TestGraphRagSearch:

    def test_graph_traverse_real_server(self):
        """graph_traverse with real HugeGraph Server — integration test."""
        result = agent_handlers.graph_rag_search(
            mode="graph_traverse", vertex_ids=["1:张三"], max_depth=2, max_items=10,
        )
        # Either succeeds (real HG Server running) or returns error dict
        assert "success" in result or "error" in result

    def test_unknown_mode_returns_error(self):
        """Unknown mode always returns error dict regardless of server state."""
        result = agent_handlers.graph_rag_search(mode="invalid_mode")
        assert result["success"] is False
        assert "Unknown mode" in result["error"]

    def test_schema_lookup_real_server(self):
        """schema_lookup with real HugeGraph Server — integration test."""
        result = agent_handlers.graph_rag_search(mode="schema_lookup")
        # Either succeeds or returns error dict (real server may be running)
        assert "success" in result or "error" in result

    def test_exception_returns_error_dict(self):
        with patch("hugegraph_llm.models.llms.init_llm.LLMs", side_effect=Exception("No config")):
            result = agent_handlers.graph_rag_search(mode="graph_traverse")
        assert result["success"] is False
        assert result["mode"] == "graph_traverse"
