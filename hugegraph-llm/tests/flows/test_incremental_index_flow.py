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

"""Tests for IncrementalIndexFlow and checkpoint-aware nodes."""

import tempfile
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.config import llm_settings
from hugegraph_llm.flows.checkpoint import IncrementalCheckpointManager
from hugegraph_llm.flows.incremental_index_flow import (
    AffectedCommunityDetectNode,
    CheckpointingNode,
    IncrementalCommunityReportNode,
    IncrementalIndexFlow,
    IncrementalVectorAddNode,
)
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class DummyNode(BaseNode):
    """Mock BaseNode that returns a deterministic result."""

    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def __init__(self, result: Optional[Dict[str, Any]] = None, raise_on_run: bool = False):
        super().__init__()
        self.result = result
        self.raise_on_run = raise_on_run
        self.call_count = 0

    def operator_schedule(self, data_json: Optional[Dict[str, Any]]):
        self.call_count += 1
        if self.raise_on_run:
            raise RuntimeError("simulated failure")
        return self.result


def test_checkpointing_node_skips_completed_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        saved_state = {"chunks": ["a"], "call_count": 1}
        manager.save_stage("chunk_split", "completed", saved_state)

        inner = DummyNode(result={"chunks": ["b"]})
        node = CheckpointingNode(manager, "chunk_split", inner)
        node.context = WkFlowState()
        node.wk_input = WkFlowInput()

        result = node.operator_schedule({})
        assert result == saved_state
        assert inner.call_count == 0


def test_checkpointing_node_runs_pending_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        inner = DummyNode(result={"chunks": ["a"]})
        node = CheckpointingNode(manager, "chunk_split", inner)
        node.context = WkFlowState()
        node.wk_input = WkFlowInput()

        result = node.operator_schedule({})
        assert result == {"chunks": ["a"]}
        assert inner.call_count == 1
        assert manager.get_stage_status("chunk_split") == "completed"
        assert manager.load_stage_state("chunk_split") == {"chunks": ["a"]}


def test_checkpointing_node_saves_failed_stage():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        inner = DummyNode(raise_on_run=True)
        node = CheckpointingNode(manager, "chunk_split", inner)
        node.context = WkFlowState()
        node.wk_input = WkFlowInput()

        with pytest.raises(RuntimeError, match="simulated failure"):
            node.operator_schedule({})
        assert manager.get_stage_status("chunk_split") == "failed"
        assert manager.get_stage_error("chunk_split") == "simulated failure"


def test_checkpointing_node_running_state_resumable():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "running", {"chunks": ["old"]})
        inner = DummyNode(result={"chunks": ["new"]})
        node = CheckpointingNode(manager, "chunk_split", inner)
        node.context = WkFlowState()
        node.wk_input = WkFlowInput()

        result = node.operator_schedule({"chunks": ["input"]})
        assert result == {"chunks": ["new"]}
        assert manager.get_stage_status("chunk_split") == "completed"


def test_affected_community_detect_node_no_vertices():
    node = AffectedCommunityDetectNode(client=None, hop=1)
    result = node.operator_schedule({})
    assert result == {"affected_community_ids": [], "affected_community_count": 0}


def test_affected_community_detect_node_with_vertices():
    with patch(
        "hugegraph_llm.operators.graph_op.incremental_utils.find_affected_communities"
    ) as mock_find:
        mock_find.return_value = {"c1", "c2"}
        node = AffectedCommunityDetectNode(client=MagicMock(), hop=1)
        result = node.operator_schedule({"vertex_ids": ["v1", "v2"]})
        assert sorted(result["affected_community_ids"]) == ["c1", "c2"]
        assert result["affected_community_count"] == 2
        mock_find.assert_called_once()


def test_incremental_vector_add_node_no_chunks():
    node = IncrementalVectorAddNode(vector_index_cls=None, embedding=None)
    result = node.operator_schedule({})
    assert result == {"vectors_added": 0}


def test_incremental_vector_add_node_with_chunks():
    vector_index = MagicMock()
    embedding = MagicMock()
    embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]
    embedding.get_embeddings_parallel.return_value = [[0.1, 0.2], [0.3, 0.4]]
    vector_index_cls = MagicMock()
    vector_index_cls.from_name.return_value = vector_index

    node = IncrementalVectorAddNode(
        vector_index_cls=vector_index_cls, embedding=embedding
    )
    chunks = [{"content": "hello"}, {"content": "world"}]
    with patch("hugegraph_llm.config.huge_settings") as mock_settings:
        mock_settings.graph_name = "test_graph"
        result = node.operator_schedule({"chunks": chunks})

    assert result == {"vectors_added": 2}
    vector_index.add.assert_called_once()
    vector_index.save_index_by_name.assert_called_once_with("test_graph", "chunks")


def test_incremental_vector_add_node_missing_embedding():
    node = IncrementalVectorAddNode(vector_index_cls=MagicMock(), embedding=None)
    result = node.operator_schedule({"chunks": [{"content": "a"}]})
    assert result == {"vectors_added": 0}


def test_incremental_community_report_node_no_communities():
    node = IncrementalCommunityReportNode(
        client=None, llm=None, vector_index_cls=None, embedding=None
    )
    result = node.operator_schedule({"affected_community_ids": []})
    assert result == {"community_reports_updated": 0}


def test_incremental_community_report_node_with_communities():
    with patch(
        "hugegraph_llm.operators.graph_op.incremental_utils.get_community_vertices"
    ) as mock_vertices, patch(
        "hugegraph_llm.operators.graph_op.incremental_utils.get_community_edges"
    ) as mock_edges, patch(
        "hugegraph_llm.operators.llm_op.community_report.CommunityReportGenerate"
    ) as mock_report_gen:
        mock_vertices.return_value = {"c1": [{"id": "v1"}, {"id": "v2"}]}
        mock_edges.return_value = []
        mock_report_gen_instance = MagicMock()
        mock_report_gen_instance.run.return_value = {
            "community_reports": [{"title": "R1", "summary": "S", "key_entities": ["E"]}]
        }
        mock_report_gen.return_value = mock_report_gen_instance

        node = IncrementalCommunityReportNode(
            client=MagicMock(), llm=MagicMock(), vector_index_cls=None, embedding=None
        )
        result = node.operator_schedule({"affected_community_ids": ["c1"]})
        assert result == {"community_reports_updated": 1}
        mock_report_gen_instance.run.assert_called_once()


def test_incremental_index_flow_builds_without_checkpoint():
    flow = IncrementalIndexFlow()
    pipeline = flow.build_flow(texts=["hello world"])
    assert pipeline is not None


def test_incremental_index_flow_builds_with_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        flow = IncrementalIndexFlow(checkpoint_dir=tmpdir, job_id="j1")
        pipeline = flow.build_flow(texts=["hello world"])
        assert pipeline is not None
        assert flow._checkpoint_manager is not None
        assert flow._checkpoint_manager.get_resume_stage() == "auto_schema_kg"


def test_incremental_index_flow_post_deal_no_pipeline():
    flow = IncrementalIndexFlow()
    result = flow.post_deal(pipeline=None)
    assert result["status_code"] == 500
    assert "error" in result


def test_incremental_index_flow_prepare_sets_defaults():
    flow = IncrementalIndexFlow()
    prepared = WkFlowInput()
    flow.prepare(prepared, texts=["hello"])
    assert prepared.text == ["hello"]
    assert prepared.language == llm_settings.language
    assert prepared.split_type == "document"
    assert prepared.extract_type == "property_graph"


def test_incremental_index_flow_prepare_uses_kwargs():
    flow = IncrementalIndexFlow()
    prepared = WkFlowInput()
    flow.prepare(
        prepared,
        texts=["hello"],
        language="CN",
        split_type="paragraph",
        extract_type="triples",
        example_prompt="custom prompt",
    )
    assert prepared.language == "CN"
    assert prepared.split_type == "paragraph"
    assert prepared.extract_type == "triples"
    assert prepared.example_prompt == "custom prompt"


def test_incremental_index_flow_setters():
    flow = IncrementalIndexFlow()
    flow.set_client("client")
    flow.set_llm("llm")
    flow.set_embedding("emb")
    flow.set_vector_index_cls("cls")
    assert flow._client == "client"
    assert flow._llm == "llm"
    assert flow._embedding == "emb"
    assert flow._vector_index_cls == "cls"


def test_checkpointing_node_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        inner = DummyNode(result={"chunks": ["a"]})
        node = CheckpointingNode(manager, "chunk_split", inner)
        node.context = WkFlowState()
        node.wk_input = WkFlowInput()

        status = node.node_init()
        assert status.isOK()
        assert inner.context is node.context
        assert inner.wk_input is node.wk_input


def test_checkpointing_node_completed_without_saved_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncrementalCheckpointManager(checkpoint_dir=tmpdir, job_id="j1")
        manager.save_stage("chunk_split", "completed", None)
        inner = DummyNode(result={"chunks": ["a"]})
        node = CheckpointingNode(manager, "chunk_split", inner)
        node.context = WkFlowState()
        node.wk_input = WkFlowInput()

        result = node.operator_schedule({})
        assert result == {"chunks": ["a"]}
        assert inner.call_count == 1


def test_incremental_index_flow_run_sync_failure():
    flow = IncrementalIndexFlow()
    with patch.object(flow, "build_flow") as mock_build:
        pipeline = MagicMock()
        status = MagicMock()
        status.isErr.return_value = True
        status.getInfo.return_value = "pipeline error"
        pipeline.process.return_value = status
        mock_build.return_value = pipeline

        result = flow.run_sync(texts=["hello"])
        assert result["status_code"] == 500
        assert result["error"] == "pipeline error"


def test_incremental_index_flow_post_deal_success():
    flow = IncrementalIndexFlow()
    pipeline = MagicMock()
    state = WkFlowState()
    state.vertices_added = 10
    state.edges_added = 5
    state.resolution_result = {"merged_count": 2}
    state.affected_community_count = 1
    state.vectors_added = 3
    state.community_reports_updated = 1
    pipeline.getGParamWithNoEmpty.return_value = state

    result = flow.post_deal(pipeline)
    assert result["status_code"] == 200
    assert result["vertices_added"] == 10
    assert result["edges_added"] == 5
    assert result["entities_merged"] == 2
    assert result["affected_communities"] == 1
    assert result["vectors_added"] == 3
    assert result["community_reports_updated"] == 1


def test_incremental_index_flow_post_deal_with_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        flow = IncrementalIndexFlow(checkpoint_dir=tmpdir, job_id="j1")
        pipeline = MagicMock()
        state = WkFlowState()
        pipeline.getGParamWithNoEmpty.return_value = state

        result = flow.post_deal(pipeline)
        assert "checkpoint" in result
        assert result["checkpoint"]["job_id"] == "j1"


def test_incremental_index_flow_run_with_checkpoint_no_manager():
    flow = IncrementalIndexFlow()
    with patch.object(flow, "run_sync") as mock_run_sync:
        mock_run_sync.return_value = {"status_code": 200}
        result = flow.run_with_checkpoint(texts=["hello"])
        assert result == {"status_code": 200}
        mock_run_sync.assert_called_once_with(texts=["hello"])


def test_incremental_index_flow_set_checkpoint_dir():
    flow = IncrementalIndexFlow()
    with tempfile.TemporaryDirectory() as tmpdir:
        flow.set_checkpoint_dir(tmpdir)
        assert flow._checkpoint_dir == tmpdir
        assert flow._checkpoint_manager is not None


def test_incremental_index_flow_set_job_id_with_checkpoint():
    flow = IncrementalIndexFlow()
    with tempfile.TemporaryDirectory() as tmpdir:
        flow.set_checkpoint_dir(tmpdir)
        flow.set_job_id("new_id")
        assert flow._job_id == "new_id"
        assert flow._checkpoint_manager.job_id == "new_id"


def test_incremental_index_flow_set_job_id_without_checkpoint():
    flow = IncrementalIndexFlow()
    flow.set_job_id("only_id")
    assert flow._job_id == "only_id"
    assert flow._checkpoint_manager is None


def test_incremental_community_report_update_index():
    with patch(
        "hugegraph_llm.operators.graph_op.incremental_utils.get_community_vertices"
    ) as mock_vertices, patch(
        "hugegraph_llm.operators.graph_op.incremental_utils.get_community_edges"
    ) as mock_edges, patch(
        "hugegraph_llm.operators.llm_op.community_report.CommunityReportGenerate"
    ) as mock_report_gen, patch(
        "hugegraph_llm.config.huge_settings"
    ) as mock_settings:
        mock_vertices.return_value = {"c1": [{"id": "v1"}]}
        mock_edges.return_value = []
        mock_settings.graph_name = "test_graph"

        vector_index = MagicMock()
        vector_index_cls = MagicMock()
        vector_index_cls.from_name.return_value = vector_index
        embedding = MagicMock()
        embedding.get_text_embedding.return_value = [0.1, 0.2]
        embedding.get_embeddings_parallel.return_value = [[0.1, 0.2]]

        mock_report_gen_instance = MagicMock()
        mock_report_gen_instance.run.return_value = {
            "community_reports": [
                {"title": "R1", "summary": "S", "key_entities": ["E1"]}
            ]
        }
        mock_report_gen.return_value = mock_report_gen_instance

        node = IncrementalCommunityReportNode(
            client=MagicMock(),
            llm=MagicMock(),
            vector_index_cls=vector_index_cls,
            embedding=embedding,
        )
        result = node.operator_schedule({"affected_community_ids": ["c1"]})
        assert result == {"community_reports_updated": 1}
        vector_index.add.assert_called_once()
        vector_index.save_index_by_name.assert_called_once_with("test_graph", "communities")

