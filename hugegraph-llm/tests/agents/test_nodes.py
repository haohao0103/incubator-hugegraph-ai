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

"""Tests for agent and graph pipeline nodes."""

from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.nodes.agent_node.agent_loop_node import AgentLoopNode
from hugegraph_llm.nodes.agent_node.query_classifier_node import QueryClassifierNode
from hugegraph_llm.nodes.agent_node.tool_execution_node import ToolExecutionNode
from hugegraph_llm.nodes.graph_node.community_nodes import (
    BuildCommunityIndexNode,
    CommunityDetectNode,
    CommunityIndexQueryNode,
    CommunityReportNode,
    GlobalSearchNode,
)
from hugegraph_llm.agents.tool_registry import Tool, ToolRegistry


# ── AgentLoopNode Tests ─────────────────────────────────────────


class TestAgentLoopNode:
    """Tests for the AgentLoopNode pipeline node."""

    def _make_mock_llm(self, responses=None):
        """Create a mock LLM with optional sequence of responses."""
        llm = MagicMock()
        if responses:
            llm.generate.side_effect = responses
        else:
            llm.generate.return_value = "Final Answer: Test response."
        return llm

    def _make_registry(self):
        """Create a ToolRegistry with test tools."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="keyword_extract",
                description="Extract keywords.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda **kw: {"keywords": [kw["query"]]},
            )
        )
        return registry

    def test_node_init_creates_agent(self):
        """Test that node_init creates a ReActAgent."""
        node = AgentLoopNode(
            tool_registry=self._make_registry(),
            llm=self._make_mock_llm(),
            max_steps=5,
            verbose=True,
        )
        node.node_init()
        assert node._agent is not None
        assert node._max_steps == 5
        assert node._verbose is True

    def test_operator_schedule_with_query(self):
        """Test operator_schedule processes a query and returns results."""
        registry = self._make_registry()
        llm = self._make_mock_llm([
            "Thought: I need to find keywords.\n"
            "Action: keyword_extract\n"
            'Action Input: {"query": "Who is test entity?"}',
        ])

        node = AgentLoopNode(
            tool_registry=registry,
            llm=llm,
            max_steps=1,
        )
        node.node_init()

        data = {"query": "Who is test entity?"}
        result = node.operator_schedule(data)

        assert "agent_answer" in result
        assert "agent_trace" in result
        assert "agent_total_steps" in result
        assert result["agent_total_steps"] >= 0

    def test_operator_schedule_with_empty_query(self):
        """Test operator_schedule handles empty query gracefully."""
        node = AgentLoopNode(
            tool_registry=self._make_registry(),
            llm=self._make_mock_llm(),
        )
        node.node_init()

        data = {"query": ""}
        result = node.operator_schedule(data)

        assert result["agent_answer"] == ""
        assert result["agent_trace"] == []
        assert result["agent_total_steps"] == 0

    def test_operator_schedule_with_no_query_key(self):
        """Test operator_schedule when query key is missing."""
        node = AgentLoopNode(
            tool_registry=self._make_registry(),
            llm=self._make_mock_llm(),
        )
        node.node_init()

        data = {}
        result = node.operator_schedule(data)

        assert result["agent_answer"] == ""
        assert result["agent_trace"] == []
        assert result["agent_total_steps"] == 0

    def test_operator_schedule_simple_query_route(self):
        """Test that a simple query is routed without multi-step loop."""
        registry = self._make_registry()
        llm = self._make_mock_llm()

        node = AgentLoopNode(
            tool_registry=registry,
            llm=llm,
            max_steps=10,
        )
        node.node_init()

        data = {"query": "Who is Sarah?"}
        result = node.operator_schedule(data)

        # Simple query should result in is_simple_query=True, empty answer
        assert "agent_answer" in result
        assert result.get("agent_is_simple_query", False) is True

    def test_operator_schedule_handles_agent_exception(self):
        """Test that agent exceptions are caught and reported."""
        registry = self._make_registry()
        llm = self._make_mock_llm()
        llm.generate.side_effect = RuntimeError("Agent crash")

        node = AgentLoopNode(
            tool_registry=registry,
            llm=llm,
            max_steps=5,
        )
        node.node_init()

        data = {"query": "Compare X and Y"}
        result = node.operator_schedule(data)

        assert "agent_error" in result
        assert "Agent crash" in result["agent_error"]
        assert result["agent_trace"] == []


# ── QueryClassifierNode Tests ────────────────────────────────────


class TestQueryClassifierNode:
    """Tests for the QueryClassifierNode pipeline node."""

    def test_classify_simple_query(self):
        """Test that a simple query is classified correctly."""
        node = QueryClassifierNode(llm=None)
        data = {"query": "Who is Sarah?"}
        result = node.operator_schedule(data)

        assert result["is_complex"] is False
        assert result["route_target"] == "fast_graph_only"

    def test_classify_complex_query(self):
        """Test that a complex query is classified correctly."""
        node = QueryClassifierNode(llm=None)
        data = {"query": "Compare entity A and entity B"}
        result = node.operator_schedule(data)

        assert result["is_complex"] is True
        assert result["route_target"] == "agent"

    def test_classify_chinese_complex_query(self):
        """Test classification of Chinese complex query."""
        node = QueryClassifierNode(llm=None)
        data = {"query": "分析 X 和 Y 之间的关系"}
        result = node.operator_schedule(data)

        assert result["is_complex"] is True
        assert result["route_target"] == "agent"

    def test_classify_with_llm(self):
        """Test classification with LLM fallback."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "complex"

        node = QueryClassifierNode(llm=mock_llm)
        data = {"query": "Tell me about entity X"}
        result = node.operator_schedule(data)

        # The LLM says complex, so it should route to agent
        assert result["is_complex"] is True
        assert result["route_target"] == "agent"

    def test_classify_empty_query(self):
        """Test classification of empty query."""
        node = QueryClassifierNode(llm=None)
        data = {"query": ""}
        result = node.operator_schedule(data)

        assert result["is_complex"] is False
        assert result["route_target"] == "fast_graph_only"

    def test_node_init_noop(self):
        """Test that node_init is a no-op."""
        node = QueryClassifierNode()
        node.node_init()
        # Should not raise


# ── ToolExecutionNode Tests ─────────────────────────────────────


class TestToolExecutionNode:
    """Tests for the ToolExecutionNode pipeline node."""

    def _make_registry(self):
        """Create a ToolRegistry with a test tool."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="echo",
                description="Echoes the input.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=lambda **kw: {"echo": kw.get("text", "")},
            )
        )
        return registry

    def test_execute_tool_by_name_in_init(self):
        """Test executing a tool specified in __init__."""
        registry = self._make_registry()
        node = ToolExecutionNode(
            tool_registry=registry,
            tool_name="echo",
        )
        data = {
            "tool_name": "echo",
            "tool_params": {"text": "hello"},
        }
        result = node.operator_schedule(data)

        assert result["tool_success"] is True
        assert result["tool_result"]["echo"] == "hello"

    def test_execute_tool_by_name_in_data(self):
        """Test executing a tool specified in the data dict."""
        registry = self._make_registry()
        node = ToolExecutionNode(
            tool_registry=registry,
            tool_name=None,
        )
        data = {
            "tool_name": "echo",
            "tool_params": {"text": "world"},
        }
        result = node.operator_schedule(data)

        assert result["tool_success"] is True
        assert result["tool_result"]["echo"] == "world"

    def test_execute_no_tool_name(self):
        """Test when no tool name is provided."""
        registry = self._make_registry()
        node = ToolExecutionNode(
            tool_registry=registry,
            tool_name=None,
        )
        data = {"tool_params": {}}
        result = node.operator_schedule(data)

        assert result["tool_success"] is False
        assert result["tool_result"] == {}

    def test_execute_unknown_tool(self):
        """Test executing an unknown tool returns error."""
        registry = self._make_registry()
        node = ToolExecutionNode(
            tool_registry=registry,
            tool_name="nonexistent",
        )
        data = {
            "tool_name": "nonexistent",
            "tool_params": {},
        }
        result = node.operator_schedule(data)

        assert result["tool_success"] is False
        assert "error" in result["tool_result"]

    def test_node_init_noop(self):
        """Test that node_init is a no-op."""
        node = ToolExecutionNode(tool_registry=MagicMock())
        node.node_init()
        # Should not raise


# ── Community Nodes Tests ────────────────────────────────────────


class TestCommunityDetectNode:
    """Tests for the CommunityDetectNode."""

    @patch("hugegraph_llm.nodes.graph_node.community_nodes.CommunityDetect")
    def test_node_init_with_defaults(self, mock_detect_cls):
        """Test node initialization with default parameters."""
        node = CommunityDetectNode()
        node.node_init()
        assert node._algorithm == "leiden"
        assert node._max_levels == 2
        mock_detect_cls.assert_called_once()

    @patch("hugegraph_llm.nodes.graph_node.community_nodes.CommunityDetect")
    def test_node_init_with_custom_params(self, mock_detect_cls):
        """Test node initialization with custom algorithm and levels."""
        node = CommunityDetectNode(
            client=MagicMock(),
            algorithm="louvain",
            max_levels=3,
        )
        node.node_init()
        assert node._algorithm == "louvain"
        assert node._max_levels == 3
        mock_detect_cls.assert_called_once()

    @patch("hugegraph_llm.nodes.graph_node.community_nodes.CommunityDetect")
    def test_operator_schedule_delegates(self, mock_detect_cls):
        """Test that operator_schedule delegates to the CommunityDetect operator."""
        mock_detector = MagicMock()
        mock_detector.run.return_value = {
            "communities": [{"id": "C0", "vertices": ["1:A", "2:B"]}],
            "community_count": 1,
            "engine_used": "networkx",
        }
        mock_detect_cls.return_value = mock_detector

        node = CommunityDetectNode(
            algorithm="louvain",
            max_levels=1,
        )
        node.node_init()

        data = {"vertices": [], "edges": []}
        result = node.operator_schedule(data)

        assert "communities" in result
        assert "community_count" in result
        assert "engine_used" in result
        mock_detector.run.assert_called_once()


class TestCommunityReportNode:
    """Tests for the CommunityReportNode."""

    def test_node_init_creates_reporter(self):
        """Test that node_init creates a CommunityReportGenerate instance."""
        mock_llm = MagicMock()
        node = CommunityReportNode(llm=mock_llm)
        node.node_init()
        assert node._reporter is not None

    def test_operator_schedule_empty_communities(self):
        """Test handling of empty communities."""
        node = CommunityReportNode()
        node.node_init()
        data = {"communities": []}
        result = node.operator_schedule(data)
        assert result["community_reports"] == []

    def test_operator_schedule_with_communities(self):
        """Test generation of fallback reports for communities."""
        node = CommunityReportNode()
        node.node_init()
        data = {
            "communities": [
                {
                    "id": "L0_C0", "level": 0, "size": 3,
                    "vertex_details": [
                        {"id": "A", "label": "person", "props": {"name": "Alice"}},
                    ],
                    "edge_details": [],
                    "density": 0.5,
                }
            ]
        }
        result = node.operator_schedule(data)
        assert len(result["community_reports"]) == 1


class TestBuildCommunityIndexNode:
    """Tests for the BuildCommunityIndexNode."""

    def _make_mock_embedding(self):
        """Create a mock embedding model."""
        emb = MagicMock()
        emb.get_embedding_dim.return_value = 128
        emb.get_texts_embeddings.return_value = [[0.1] * 128]
        return emb

    def _make_mock_vector_index(self):
        """Create a mock vector index class."""
        index = MagicMock()
        index_cls = MagicMock(return_value=index)
        index_cls.from_name = MagicMock(return_value=index)
        return index_cls

    def test_node_init(self):
        """Test node initialization."""
        node = BuildCommunityIndexNode(
            vector_index_cls=MagicMock(),
            embedding=MagicMock(),
        )
        node.node_init()
        assert node._builder is not None

    def test_operator_schedule_empty_reports(self):
        """Test handling when no community reports exist."""
        node = BuildCommunityIndexNode(
            vector_index_cls=self._make_mock_vector_index(),
            embedding=self._make_mock_embedding(),
        )
        node.node_init()
        data = {"community_reports": []}
        result = node.operator_schedule(data)

        assert result["community_index_built"] is False
        assert result["community_index_count"] == 0


class TestCommunityIndexQueryNode:
    """Tests for the CommunityIndexQueryNode."""

    def test_node_init(self):
        """Test node initialization."""
        node = CommunityIndexQueryNode(
            vector_index_cls=MagicMock(),
            embedding=MagicMock(),
            top_k=5,
        )
        node.node_init()
        assert node._querier is not None
        assert node._top_k == 5

    def test_operator_schedule_empty_query(self):
        """Test handling of empty query."""
        mock_emb = MagicMock()
        mock_emb.get_embedding_dim.return_value = 128
        mock_emb.get_texts_embeddings.return_value = [[0.1] * 128]

        node = CommunityIndexQueryNode(
            vector_index_cls=MagicMock(),
            embedding=mock_emb,
        )
        node.node_init()
        data = {"query": ""}
        result = node.operator_schedule(data)

        assert result["community_matches"] == []


class TestGlobalSearchNode:
    """Tests for the GlobalSearchNode."""

    def test_node_init(self):
        """Test node initialization creates a GlobalSearch instance."""
        mock_llm = MagicMock()
        node = GlobalSearchNode(llm=mock_llm)
        node.node_init()
        assert node._searcher is not None

    def test_operator_schedule_empty_reports(self):
        """Test global search with no reports."""
        node = GlobalSearchNode()
        node.node_init()
        data = {"query": "test", "community_reports": []}
        result = node.operator_schedule(data)

        assert "community reports have not been generated" in result["global_answer"].lower()

    def test_operator_schedule_with_reports(self):
        """Test global search with community reports."""
        node = GlobalSearchNode()
        node.node_init()
        data = {
            "query": "What is the structure?",
            "community_reports": [
                {
                    "community_id": "C0", "title": "Group A",
                    "summary": "A test group.",
                    "key_entities": ["E1"],
                    "relationship_patterns": ["p1"],
                    "importance_score": 5.0,
                }
            ],
        }
        result = node.operator_schedule(data)

        # Without LLM, should get a fallback message
        assert result["global_answer"] != ""
        assert "map_findings" in result
        assert "communities_used" in result
