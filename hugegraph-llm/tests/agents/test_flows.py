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

"""Tests for graphrag pipeline flows."""

from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.agent_flow import AgentFlow
from hugegraph_llm.flows.community_flow import (
    CommunityDetectionFlow,
    GlobalSearchFlow,
)
from hugegraph_llm.flows.provenance_flow import (
    ProvenanceAwareKGFlow,
    _ProvenanceLinkNode,
)
from hugegraph_llm.agents.tool_registry import ToolRegistry
from hugegraph_llm.operators.hugegraph_op.provenance_manager import ProvenanceManager
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


# ── FlowName Enum Tests ─────────────────────────────────────────


class TestFlowName:
    """Tests for the FlowName enum with new graphrag values."""

    def test_agent_flow_name(self):
        """Test that AGENT flow name is defined."""
        assert FlowName.AGENT == "agent"
        assert FlowName.AGENT.value == "agent"

    def test_community_detect_flow_name(self):
        """Test that COMMUNITY_DETECT flow name is defined."""
        assert FlowName.COMMUNITY_DETECT == "community_detect"
        assert FlowName.COMMUNITY_DETECT.value == "community_detect"

    def test_global_search_flow_name(self):
        """Test that GLOBAL_SEARCH flow name is defined."""
        assert FlowName.GLOBAL_SEARCH == "global_search"
        assert FlowName.GLOBAL_SEARCH.value == "global_search"

    def test_provenance_flow_name(self):
        """Test that PROVENANCE_KG_BUILD flow name is defined."""
        assert FlowName.PROVENANCE_KG_BUILD == "provenance_kg_build"
        assert FlowName.PROVENANCE_KG_BUILD.value == "provenance_kg_build"

    def test_all_new_flow_names_exist(self):
        """Test all new graphrag flow names are present."""
        new_names = {"agent", "community_detect", "global_search", "provenance_kg_build"}
        for name in new_names:
            assert hasattr(FlowName, name.upper() if "_" in name else name.upper())

    def test_legacy_flow_names_still_exist(self):
        """Test that legacy flow names are still present."""
        assert FlowName.RAG_GRAPH_ONLY == "rag_graph_only"
        assert FlowName.TEXT2GREMLIN == "text2gremlin"
        assert FlowName.GRAPH_EXTRACT == "graph_extract"


# ── AgentFlow Tests ─────────────────────────────────────────────


class TestAgentFlow:
    """Tests for the AgentFlow pipeline."""

    def _make_mock_llm(self):
        """Create a mock BaseLLM."""
        llm = MagicMock()
        llm.generate.return_value = "Final Answer: test answer."
        return llm

    def test_init_with_defaults(self):
        """Test AgentFlow initialization with default values."""
        flow = AgentFlow(max_steps=10)
        assert flow._max_steps == 10
        assert flow._tool_registry is None
        assert flow._llm is None

    def test_init_with_dependencies(self):
        """Test AgentFlow with all dependencies provided."""
        registry = MagicMock(spec=ToolRegistry)
        llm = self._make_mock_llm()
        flow = AgentFlow(
            tool_registry=registry,
            llm=llm,
            max_steps=5,
        )
        assert flow._tool_registry is registry
        assert flow._llm is llm
        assert flow._max_steps == 5

    def test_prepare_sets_query(self):
        """Test that prepare() populates WkFlowInput correctly."""
        flow = AgentFlow()
        input_obj = WkFlowInput()
        flow.prepare(input_obj, query="Who is test?", max_steps=3)

        assert input_obj.query == "Who is test?"
        assert input_obj.max_deep == 3

    def test_prepare_default_max_steps(self):
        """Test that prepare uses default max_steps when not provided."""
        flow = AgentFlow(max_steps=7)
        input_obj = WkFlowInput()
        flow.prepare(input_obj, query="test query")

        assert input_obj.max_deep == 7

    def test_build_flow_missing_tool_registry(self):
        """Test that build_flow raises when ToolRegistry is missing."""
        llm = self._make_mock_llm()
        flow = AgentFlow(llm=llm, max_steps=5)
        with pytest.raises(ValueError, match="ToolRegistry"):
            flow.build_flow()

    def test_build_flow_missing_llm(self):
        """Test that build_flow raises when LLM is missing."""
        registry = MagicMock(spec=ToolRegistry)
        flow = AgentFlow(tool_registry=registry, max_steps=5)
        with pytest.raises(ValueError, match="LLM"):
            flow.build_flow()

    def test_set_tool_registry(self):
        """Test that set_tool_registry updates internal state."""
        flow = AgentFlow()
        registry = MagicMock(spec=ToolRegistry)
        flow.set_tool_registry(registry)
        assert flow._tool_registry is registry

    def test_set_llm(self):
        """Test that set_llm updates internal state."""
        flow = AgentFlow()
        llm = self._make_mock_llm()
        flow.set_llm(llm)
        assert flow._llm is llm

    def test_set_max_steps(self):
        """Test that set_max_steps updates the step limit."""
        flow = AgentFlow(max_steps=10)
        flow.set_max_steps(20)
        assert flow._max_steps == 20

    @patch("hugegraph_llm.flows.agent_flow.GPipeline")
    @patch("hugegraph_llm.flows.agent_flow.AgentLoopNode")
    def test_post_deal_no_pipeline(self, _mock_node, _mock_pipeline):
        """Test post_deal with no pipeline returns error."""
        flow = AgentFlow()
        result = flow.post_deal(pipeline=None)
        assert "error" in result
        assert result["error"] == "No pipeline provided"

    @patch("hugegraph_llm.flows.agent_flow.GPipeline")
    @patch("hugegraph_llm.flows.agent_flow.AgentLoopNode")
    def test_post_deal_with_pipeline(self, _mock_node, _mock_pipeline):
        """Test post_deal extracts results from pipeline state."""
        flow = AgentFlow()

        mock_pipeline = MagicMock()
        mock_state = MagicMock()
        mock_state.to_json.return_value = {
            "agent_answer": "Test answer.",
            "agent_trace": [],
            "agent_total_steps": 0,
            "agent_is_simple_query": True,
        }
        mock_pipeline.getGParamWithNoEmpty.return_value = mock_state

        result = flow.post_deal(pipeline=mock_pipeline)
        assert result["status_code"] == 200
        assert result["answer"] == "Test answer."
        assert result["total_steps"] == 0
        assert result["is_simple_query"] is True

    @patch("hugegraph_llm.flows.agent_flow.GPipeline")
    @patch("hugegraph_llm.flows.agent_flow.AgentLoopNode")
    def test_post_deal_no_state(self, _mock_node, _mock_pipeline):
        """Test post_deal when no workflow state is found."""
        flow = AgentFlow()

        mock_pipeline = MagicMock()
        mock_pipeline.getGParamWithNoEmpty.return_value = None

        result = flow.post_deal(pipeline=mock_pipeline)
        assert "error" in result
        assert "No workflow state" in result["error"]


# ── CommunityDetectionFlow Tests ────────────────────────────────


class TestCommunityDetectionFlow:
    """Tests for the CommunityDetectionFlow."""

    def test_init_defaults(self):
        """Test default initialization."""
        flow = CommunityDetectionFlow()
        assert flow._algorithm == "leiden"
        assert flow._max_levels == 2
        assert flow._client is None
        assert flow._llm is None

    def test_init_with_dependencies(self):
        """Test initialization with all dependencies."""
        mock_client = MagicMock()
        mock_llm = MagicMock()
        flow = CommunityDetectionFlow(
            client=mock_client,
            llm=mock_llm,
            algorithm="louvain",
            max_levels=3,
        )
        assert flow._algorithm == "louvain"
        assert flow._max_levels == 3
        assert flow._client is mock_client
        assert flow._llm is mock_llm

    def test_prepare_sets_graph_name(self):
        """Test that prepare sets the graph name."""
        flow = CommunityDetectionFlow()
        input_obj = WkFlowInput()
        flow.prepare(input_obj, graph_name="test_graph")
        assert input_obj.graph_name == "test_graph"

    def test_setters_update_dependencies(self):
        """Test that lazy dependency setters work."""
        flow = CommunityDetectionFlow()
        mock_llm = MagicMock()
        mock_emb = MagicMock()
        mock_index = MagicMock()

        flow.set_llm(mock_llm)
        flow.set_embedding(mock_emb)
        flow.set_vector_index_cls(mock_index)

        assert flow._llm is mock_llm
        assert flow._embedding is mock_emb
        assert flow._vector_index_cls is mock_index

    @patch("hugegraph_llm.flows.community_flow.GPipeline")
    def test_build_flow_creates_dag(self, mock_gpipeline_cls):
        """Test that build_flow constructs the expected DAG structure."""
        mock_pipeline = MagicMock()
        mock_gpipeline_cls.return_value = mock_pipeline

        flow = CommunityDetectionFlow()
        pipeline = flow.build_flow()

        # Verify pipeline was initialized
        mock_pipeline.init.assert_called_once()
        # 3 nodes in the community detection DAG
        assert mock_pipeline.registerGElement.call_count == 3

    @patch("hugegraph_llm.flows.community_flow.GPipeline")
    def test_post_deal_no_pipeline(self, _mock):
        """Test post_deal with no pipeline returns error."""
        flow = CommunityDetectionFlow()
        result = flow.post_deal(pipeline=None)
        assert "error" in result

    @patch("hugegraph_llm.flows.community_flow.GPipeline")
    def test_post_deal_returns_community_count(self, _mock):
        """Test post_deal extracts community metrics."""
        flow = CommunityDetectionFlow()

        mock_pipeline = MagicMock()
        mock_state = MagicMock()
        mock_state.to_json.return_value = {
            "community_count": 5,
            "community_reports": [{"id": "C0"}, {"id": "C1"}],
            "community_index_built": True,
        }
        mock_pipeline.getGParamWithNoEmpty.return_value = mock_state

        result = flow.post_deal(pipeline=mock_pipeline)

        assert result["status_code"] == 200
        assert result["community_count"] == 5
        assert result["report_count"] == 2
        assert result["index_built"] is True


# ── GlobalSearchFlow Tests ──────────────────────────────────────


class TestGlobalSearchFlow:
    """Tests for the GlobalSearchFlow."""

    def test_init_defaults(self):
        """Test default initialization."""
        flow = GlobalSearchFlow()
        assert flow._llm is None
        assert flow._embedding is None
        assert flow._vector_index_cls is None

    def test_init_with_dependencies(self):
        """Test initialization with all dependencies."""
        mock_llm = MagicMock()
        mock_emb = MagicMock()
        mock_index = MagicMock()
        flow = GlobalSearchFlow(
            llm=mock_llm,
            embedding=mock_emb,
            vector_index_cls=mock_index,
        )
        assert flow._llm is mock_llm
        assert flow._embedding is mock_emb
        assert flow._vector_index_cls is mock_index

    def test_prepare_sets_query(self):
        """Test that prepare sets the user query."""
        flow = GlobalSearchFlow()
        input_obj = WkFlowInput()
        flow.prepare(input_obj, query="What are the main themes?")
        assert input_obj.query == "What are the main themes?"

    def test_setters_update_dependencies(self):
        """Test that lazy dependency setters work."""
        flow = GlobalSearchFlow()
        mock_llm = MagicMock()
        flow.set_llm(mock_llm)
        assert flow._llm is mock_llm

        mock_emb = MagicMock()
        flow.set_embedding(mock_emb)
        assert flow._embedding is mock_emb

        mock_index = MagicMock()
        flow.set_vector_index_cls(mock_index)
        assert flow._vector_index_cls is mock_index

    @patch("hugegraph_llm.flows.community_flow.GPipeline")
    def test_build_flow_creates_dag(self, mock_gpipeline_cls):
        """Test that build_flow constructs the expected DAG."""
        mock_pipeline = MagicMock()
        mock_gpipeline_cls.return_value = mock_pipeline

        flow = GlobalSearchFlow()
        pipeline = flow.build_flow()

        mock_pipeline.init.assert_called_once()
        # 2 nodes in the global search DAG
        assert mock_pipeline.registerGElement.call_count == 2

    @patch("hugegraph_llm.flows.community_flow.GPipeline")
    def test_post_deal_returns_answer(self, _mock):
        """Test post_deal extracts the global answer."""
        flow = GlobalSearchFlow()

        mock_pipeline = MagicMock()
        mock_state = MagicMock()
        mock_state.to_json.return_value = {
            "global_answer": "The main themes are X and Y.",
            "map_findings": [{"finding": "X is important"}, {"finding": "Y matters"}],
            "communities_used": 3,
        }
        mock_pipeline.getGParamWithNoEmpty.return_value = mock_state

        result = flow.post_deal(pipeline=mock_pipeline)

        assert result["status_code"] == 200
        assert result["answer"] == "The main themes are X and Y."
        assert result["communities_used"] == 3
        assert len(result["map_findings"]) == 2

    @patch("hugegraph_llm.flows.community_flow.GPipeline")
    def test_post_deal_no_pipeline(self, _mock):
        """Test post_deal with no pipeline."""
        flow = GlobalSearchFlow()
        result = flow.post_deal(pipeline=None)
        assert "error" in result


# ── ProvenanceAwareKGFlow Tests ─────────────────────────────────


class TestProvenanceAwareKGFlow:
    """Tests for the ProvenanceAwareKGFlow."""

    def test_init_creates_provenance_manager(self):
        """Test that init creates a ProvenanceManager when client is provided."""
        mock_client = MagicMock()
        flow = ProvenanceAwareKGFlow(client=mock_client)
        assert flow._pm is not None
        assert isinstance(flow._pm, ProvenanceManager)

    def test_init_uses_provided_manager(self):
        """Test that init uses an externally provided ProvenanceManager."""
        mock_pm = MagicMock(spec=ProvenanceManager)
        flow = ProvenanceAwareKGFlow(provenance_manager=mock_pm)
        assert flow._pm is mock_pm

    def test_prepare_sets_provenance_fields(self):
        """Test that prepare sets document metadata."""
        mock_pm = MagicMock(spec=ProvenanceManager)
        flow = ProvenanceAwareKGFlow(provenance_manager=mock_pm)
        input_obj = WkFlowInput()
        flow.prepare(
            input_obj,
            texts=["text chunk 1", "text chunk 2"],
            doc_name="report.pdf",
            doc_source="/data/report.pdf",
            language="EN",
        )

        assert input_obj.texts == ["text chunk 1", "text chunk 2"]
        assert input_obj.language == "EN"
        assert input_obj.data_json["doc_name"] == "report.pdf"
        assert input_obj.data_json["doc_source"] == "/data/report.pdf"

    @patch("hugegraph_llm.nodes.llm_node.extract_info.ExtractInfoNode", MagicMock(), create=True)
    @patch("hugegraph_llm.nodes.document_node.chunk_split.ChunkSplitNode", MagicMock(), create=True)
    @patch("hugegraph_llm.nodes.hugegraph_node.commit_to_hugegraph.Commit2GraphNode", MagicMock(), create=True)
    @patch("hugegraph_llm.flows.provenance_flow.GPipeline")
    def test_build_flow_creates_dag(self, mock_gpipeline):
        """Test that build_flow constructs the 4-node DAG."""
        mock_pipeline = MagicMock()
        mock_gpipeline.return_value = mock_pipeline

        mock_pm = MagicMock(spec=ProvenanceManager)
        flow = ProvenanceAwareKGFlow(provenance_manager=mock_pm)
        pipeline = flow.build_flow()

        mock_pipeline.init.assert_called_once()
        # 4 nodes (chunk + extract + provenance + commit) in the DAG
        assert mock_pipeline.registerGElement.call_count == 4

    @patch("hugegraph_llm.flows.provenance_flow.GPipeline")
    def test_post_deal_returns_metrics(self, _mock):
        """Test post_deal extracts construction metrics."""
        mock_pm = MagicMock(spec=ProvenanceManager)
        flow = ProvenanceAwareKGFlow(provenance_manager=mock_pm)

        mock_pipeline = MagicMock()
        mock_state = MagicMock()
        mock_state.to_json.return_value = {
            "vertices": [{"id": "1:A"}, {"id": "2:B"}],
            "edges": [{"label": "knows", "outV": "1:A", "inV": "2:B"}],
            "provenance_link_count": 3,
        }
        mock_pipeline.getGParamWithNoEmpty.return_value = mock_state

        result = flow.post_deal(pipeline=mock_pipeline)

        assert result["status_code"] == 200
        assert result["vertex_count"] == 2
        assert result["edge_count"] == 1
        assert result["provenance_links"] == 3

    @patch("hugegraph_llm.flows.provenance_flow.GPipeline")
    def test_post_deal_no_pipeline(self, _mock):
        """Test post_deal with no pipeline."""
        flow = ProvenanceAwareKGFlow()
        result = flow.post_deal(pipeline=None)
        assert "error" in result


# ── _ProvenanceLinkNode Tests ───────────────────────────────────


class TestProvenanceLinkNode:
    """Tests for the internal _ProvenanceLinkNode."""

    def test_init_calls_provenance_manager(self):
        """Test that init() initializes the provenance schema."""
        mock_pm = MagicMock(spec=ProvenanceManager)
        node = _ProvenanceLinkNode(mock_pm)
        node.init()
        mock_pm.init_schema.assert_called_once()

    def test_init_without_pm(self):
        """Test init when no provenance manager is set."""
        node = _ProvenanceLinkNode(None)
        node.init()
        # Should not raise

    def test_run_no_data(self):
        """Test run with empty data."""
        mock_pm = MagicMock(spec=ProvenanceManager)
        node = _ProvenanceLinkNode(mock_pm)
        state = WkFlowState()
        result = node.run(state)
        assert result["provenance_link_count"] == 0

    def test_run_with_vertices_and_chunks(self):
        """Test run with actual vertices and chunks links them."""
        mock_pm = MagicMock(spec=ProvenanceManager)
        mock_pm.create_document.return_value = "DOC:test.pdf"
        mock_pm.create_chunk.return_value = "CHUNK:abc123"
        mock_pm.link_entity_to_chunk.return_value = True

        node = _ProvenanceLinkNode(mock_pm)
        state = WkFlowState()
        state.vertices = [
            {"id": "1:Sarah", "label": "person", "properties": {"name": "Sarah"}},
            {"id": "2:Lila", "label": "person", "properties": {"name": "Lila"}},
        ]
        state.chunks = ["Sarah is 30 years old.", "Lila lives in Chicago."]
        state.data_json = {"doc_name": "test.pdf", "doc_source": "/data/test.pdf"}

        result = node.run(state)
        assert "provenance_link_count" in result
        assert "doc_id" in result
        assert result["doc_id"] == "DOC:test.pdf"
        mock_pm.create_document.assert_called_once()

    def test_run_without_pm(self):
        """Test run when provenance manager is None."""
        node = _ProvenanceLinkNode(None)
        state = WkFlowState()
        state.vertices = [{"id": "1:A", "properties": {"name": "A"}}]
        result = node.run(state)
        assert result["provenance_link_count"] == 0
