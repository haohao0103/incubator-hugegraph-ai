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

"""Incremental indexing flow with checkpoint/resume support.

Processes new documents incrementally without full graph reconstruction:
1. ChunkSplit: Split new documents into chunks
2. ExtractInfo: LLM entity/relation extraction from chunks
3. EntityResolution: Merge new entities with existing ones
4. CommitNew: Append new vertices/edges to HugeGraph
5. DetectAffectedCommunities: Find communities impacted by new vertices
6. IncrementalCommunityReport: Regenerate reports for affected communities only
7. IncrementalVectorAdd: Add new chunk vectors to existing vector index

Checkpointing:
    When checkpoint_dir is provided, the flow persists the WkFlowState after
    each completed stage.  If a run is interrupted, the next run with the same
    job_id resumes from the last completed stage.  This design mirrors
    LightRAG's DocStatusStorage and MS-GraphRAG's PipelineRunContext.

Usage:
    flow = IncrementalIndexFlow(
        client=hugegraph_client,
        llm=extract_llm,
        embedding=emb,
        vector_index_cls=faiss_cls,
        checkpoint_dir="/tmp/hg_checkpoints",
        job_id="my_job_001",
    )
    result = flow.run_with_checkpoint(texts=["New document content..."])
"""

from typing import Any, Dict, List, Optional

from pycgraph import CStatus, GPipeline

from hugegraph_llm.config import llm_settings, prompt
from hugegraph_llm.flows.checkpoint import (
    INCREMENTAL_STAGES,
    IncrementalCheckpointManager,
)
from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class IncrementalIndexFlow(BaseFlow):
    """Incremental indexing flow with optional checkpoint/resume.

    Unlike full-rebuild flows, this only processes new content and updates the
    affected portions of the index.  When checkpoint_dir is provided, the flow
    can resume from the last successful stage after a failure or interruption.
    """

    def __init__(
        self,
        client: Any = None,
        llm: Any = None,
        embedding: Any = None,
        vector_index_cls: Any = None,
        entity_resolution_strategy: str = "hybrid",
        entity_resolution_threshold: float = 0.85,
        community_hop: int = 1,
        checkpoint_dir: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        self._client = client
        self._llm = llm
        self._embedding = embedding
        self._vector_index_cls = vector_index_cls
        self._entity_resolution_strategy = entity_resolution_strategy
        self._entity_resolution_threshold = entity_resolution_threshold
        self._community_hop = community_hop
        self._checkpoint_dir = checkpoint_dir
        self._job_id = job_id
        self._checkpoint_manager: Optional[IncrementalCheckpointManager] = None
        if checkpoint_dir:
            self._checkpoint_manager = IncrementalCheckpointManager(
                checkpoint_dir=checkpoint_dir,
                job_id=job_id,
                stages=INCREMENTAL_STAGES,
            )

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        """Prepare the flow input with new document texts and options."""
        texts = kwargs.get("texts", [])
        prepared_input.text = texts if isinstance(texts, list) else [texts]
        prepared_input.graph_name = kwargs.get("graph_name", "")
        prepared_input.language = kwargs.get("language", llm_settings.language)
        prepared_input.split_type = kwargs.get("split_type", "document")
        prepared_input.extract_type = kwargs.get("extract_type", "property_graph")
        prepared_input.example_prompt = kwargs.get(
            "example_prompt", prompt.extract_graph_prompt
        )
        prepared_input.schema = kwargs.get("schema", None)
        prepared_input.data_json = kwargs.get("data_json", None)

    def build_flow(self, **kwargs):
        """Build the incremental indexing DAG pipeline.

        The pipeline processes new documents through extraction, entity
        resolution, graph commit, affected community detection, and
        incremental index updates.
        """
        from hugegraph_llm.nodes.document_node.chunk_split import ChunkSplitNode
        from hugegraph_llm.nodes.hugegraph_node.commit_to_hugegraph import Commit2GraphNode
        from hugegraph_llm.nodes.hugegraph_node.entity_resolution_node import EntityResolutionNode
        from hugegraph_llm.nodes.llm_node.extract_info import ExtractNode

        pipeline = GPipeline()

        prepared_input = WkFlowInput()
        self.prepare(prepared_input, **kwargs)
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Node 1: ChunkSplit
        chunk_node = ChunkSplitNode()

        # Node 2: ExtractInfo
        extract_node = ExtractNode()

        # Node 3: EntityResolution
        resolution_node = EntityResolutionNode()

        # Node 4: CommitToHugegraph
        commit_node = Commit2GraphNode()

        # Node 5: AffectedCommunityDetect
        affected_node = AffectedCommunityDetectNode(
            client=self._client,
            hop=self._community_hop,
        )

        # Node 6: IncrementalCommunityReport
        report_node = IncrementalCommunityReportNode(
            client=self._client,
            llm=self._llm,
            vector_index_cls=self._vector_index_cls,
            embedding=self._embedding,
        )

        # Node 7: IncrementalVectorAdd
        vector_node = IncrementalVectorAddNode(
            vector_index_cls=self._vector_index_cls,
            embedding=self._embedding,
        )

        # Wrap with checkpoint-aware nodes if checkpointing is enabled.
        if self._checkpoint_manager is not None:
            chunk_node = CheckpointingNode(
                self._checkpoint_manager, "chunk_split", chunk_node
            )
            extract_node = CheckpointingNode(
                self._checkpoint_manager, "extract_info", extract_node
            )
            resolution_node = CheckpointingNode(
                self._checkpoint_manager, "entity_resolution", resolution_node
            )
            commit_node = CheckpointingNode(
                self._checkpoint_manager, "commit_new", commit_node
            )
            affected_node = CheckpointingNode(
                self._checkpoint_manager, "affected_detect", affected_node
            )
            report_node = CheckpointingNode(
                self._checkpoint_manager, "incremental_report", report_node
            )
            vector_node = CheckpointingNode(
                self._checkpoint_manager, "incremental_vector", vector_node
            )

        # Register DAG: linear chain with vector add parallel to report.
        pipeline.registerGElement(chunk_node, set(), "chunk_split")
        pipeline.registerGElement(extract_node, {chunk_node}, "extract_info")
        pipeline.registerGElement(resolution_node, {extract_node}, "entity_resolution")
        pipeline.registerGElement(commit_node, {resolution_node}, "commit_new")
        pipeline.registerGElement(affected_node, {commit_node}, "affected_detect")
        pipeline.registerGElement(report_node, {affected_node}, "incremental_report")
        pipeline.registerGElement(vector_node, {commit_node}, "incremental_vector")

        pipeline.init()
        return pipeline

    def post_deal(self, pipeline=None, **kwargs) -> Dict[str, Any]:
        """Extract results from the completed pipeline."""
        if pipeline is None:
            return {"error": "No pipeline provided", "status_code": 500}

        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        state_json = state.to_json() if state else {}

        result = {
            "status_code": 200,
            "message": "Incremental indexing completed",
            "vertices_added": state_json.get("vertices_added", 0),
            "edges_added": state_json.get("edges_added", 0),
            "entities_merged": state_json.get("resolution_result", {}).get("merged_count", 0),
            "affected_communities": state_json.get("affected_community_count", 0),
            "vectors_added": state_json.get("vectors_added", 0),
            "community_reports_updated": state_json.get("community_reports_updated", 0),
        }
        if self._checkpoint_manager is not None:
            result["checkpoint"] = self._checkpoint_manager.get_summary()
        return result

    def run_sync(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Synchronous entry point for incremental indexing (without explicit resume)."""
        pipeline = self.build_flow(texts=texts, **kwargs)
        status = pipeline.process()
        if status.isErr():
            log.error("IncrementalIndexFlow failed: %s", status.getInfo())
            return {"error": status.getInfo(), "status_code": 500}
        return self.post_deal(pipeline)

    def run_with_checkpoint(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Synchronous entry point that returns checkpoint summary.

        This is functionally identical to run_sync when checkpointing is
        enabled; the checkpoint manager embedded in the flow nodes handles
        resume automatically.  If no checkpoint_dir was configured, this
        falls back to run_sync with a warning.
        """
        if self._checkpoint_manager is None:
            log.warning(
                "run_with_checkpoint called without checkpoint_dir; "
                "running without checkpoint/resume support."
            )
        return self.run_sync(texts=texts, **kwargs)

    # Lazy dependency setters

    def set_client(self, client) -> None:
        self._client = client

    def set_llm(self, llm) -> None:
        self._llm = llm

    def set_embedding(self, embedding) -> None:
        self._embedding = embedding

    def set_vector_index_cls(self, cls) -> None:
        self._vector_index_cls = cls

    def set_checkpoint_dir(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_manager = IncrementalCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            job_id=self._job_id,
            stages=INCREMENTAL_STAGES,
        )

    def set_job_id(self, job_id: str) -> None:
        self._job_id = job_id
        if self._checkpoint_dir:
            self._checkpoint_manager = IncrementalCheckpointManager(
                checkpoint_dir=self._checkpoint_dir,
                job_id=job_id,
                stages=INCREMENTAL_STAGES,
            )


class CheckpointingNode(BaseNode):
    """Wraps a real BaseNode and persists checkpoint state after each stage.

    On entry, if the wrapped stage has already been completed, the saved
    WkFlowState snapshot is returned and the inner node is skipped.  This makes
    the pipeline idempotent and resumable.
    """

    def __init__(
        self,
        checkpoint_manager: IncrementalCheckpointManager,
        stage_name: str,
        inner_node: BaseNode,
    ):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager
        self._stage_name = stage_name
        self._inner = inner_node

    def node_init(self):
        self._inner.context = self.context
        self._inner.wk_input = self.wk_input
        return self._inner.node_init()

    def operator_schedule(self, data_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        status = self._checkpoint_manager.get_stage_status(self._stage_name)
        if status == "completed":
            saved_state = self._checkpoint_manager.load_stage_state(self._stage_name)
            if saved_state is not None:
                log.info(
                    "Checkpoint resume: skipping completed stage '%s'", self._stage_name
                )
                return saved_state

        self._checkpoint_manager.save_stage(
            self._stage_name, "running", state_dict=data_json
        )
        try:
            result = self._inner.operator_schedule(data_json)
            if result is not None and isinstance(result, dict):
                self._checkpoint_manager.save_stage(
                    self._stage_name, "completed", state_dict=result
                )
            return result
        except Exception as exc:  # noqa: BLE001
            self._checkpoint_manager.save_stage(
                self._stage_name, "failed", state_dict=data_json, error=str(exc)
            )
            raise


class AffectedCommunityDetectNode(BaseNode):
    """Detect communities affected by newly added vertices.

    Uses Gremlin to find neighbors of new vertices and collects their
    community_id assignments.
    """

    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def __init__(self, client: Any, hop: int = 1):
        super().__init__()
        self._client = client
        self._hop = hop

    def operator_schedule(self, data_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        from hugegraph_llm.operators.graph_op.incremental_utils import find_affected_communities

        new_vids = (data_json or {}).get("vertex_ids", [])
        if not new_vids:
            log.info("No new vertex IDs found for affected community detection")
            return {
                "affected_community_ids": [],
                "affected_community_count": 0,
            }

        affected = find_affected_communities(self._client, new_vids, hop=self._hop)
        log.info(
            "Affected community detection: %d communities affected by %d new vertices",
            len(affected), len(new_vids),
        )
        return {
            "affected_community_ids": list(affected),
            "affected_community_count": len(affected),
        }


class IncrementalCommunityReportNode(BaseNode):
    """Regenerate community reports only for affected communities.

    Fetches vertices/edges for affected communities, regenerates LLM
    summaries, and updates the community vector index.
    """

    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def __init__(
        self,
        client: Any,
        llm: Any,
        vector_index_cls: Any = None,
        embedding: Any = None,
    ):
        super().__init__()
        self._client = client
        self._llm = llm
        self._vector_index_cls = vector_index_cls
        self._embedding = embedding

    def operator_schedule(self, data_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        from hugegraph_llm.operators.graph_op.incremental_utils import (
            get_community_edges,
            get_community_vertices,
        )
        from hugegraph_llm.operators.llm_op.community_report import CommunityReportGenerate

        affected_ids = (data_json or {}).get("affected_community_ids", [])
        if not affected_ids:
            log.info("No affected communities to rebuild")
            return {"community_reports_updated": 0}

        community_vertices = get_community_vertices(self._client, set(affected_ids))
        all_vids = set()
        for vids in community_vertices.values():
            all_vids.update(v.get("id") for v in vids if v.get("id"))
        community_edges = get_community_edges(self._client, all_vids)

        context = {
            "communities": [
                {
                    "id": cid,
                    "level": 0,
                    "vertices": [v["id"] for v in verts],
                    "size": len(verts),
                    "vertex_details": verts,
                }
                for cid, verts in community_vertices.items()
            ],
        }

        report_generator = CommunityReportGenerate(llm=self._llm)
        context = report_generator.run(context)

        reports = context.get("community_reports", [])
        if reports and self._vector_index_cls and self._embedding:
            self._update_community_index(reports)

        log.info("Incremental community rebuild: %d reports regenerated", len(reports))
        return {"community_reports_updated": len(reports)}

    def _update_community_index(self, reports: List[Dict]) -> None:
        """Update the community vector index with new reports."""
        from hugegraph_llm.config import huge_settings

        graph_name = huge_settings.graph_name
        test_vec = self._embedding.get_text_embedding("test")
        vector_index = self._vector_index_cls.from_name(
            len(test_vec), graph_name, "communities",
        )

        texts = []
        for report in reports:
            title = report.get("title", "")
            summary = report.get("summary", "")
            key_entities = report.get("key_entities", [])
            texts.append(
                f"Title: {title}\nSummary: {summary}\nKey Entities: {', '.join(key_entities)}"
            )

        if texts:
            embeddings = self._embedding.get_embeddings_parallel(texts)
            vector_index.add(embeddings, reports)
            vector_index.save_index_by_name(graph_name, "communities")
            log.info("Updated community vector index with %d reports", len(texts))


class IncrementalVectorAddNode(BaseNode):
    """Incrementally add new chunk vectors to the existing vector index."""

    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def __init__(
        self,
        vector_index_cls: Any = None,
        embedding: Any = None,
    ):
        super().__init__()
        self._vector_index_cls = vector_index_cls
        self._embedding = embedding

    def operator_schedule(self, data_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        from hugegraph_llm.config import huge_settings

        chunks = (data_json or {}).get("chunks", [])
        if not chunks:
            log.info("No chunks to add to vector index")
            return {"vectors_added": 0}

        if not self._vector_index_cls or not self._embedding:
            log.warning("Vector index or embedding not configured; skipping incremental vector add")
            return {"vectors_added": 0}

        graph_name = huge_settings.graph_name
        test_vec = self._embedding.get_text_embedding("test")
        vector_index = self._vector_index_cls.from_name(len(test_vec), graph_name, "chunks")

        texts = [c.get("content", c.get("text", str(c))) for c in chunks]
        embeddings = self._embedding.get_embeddings_parallel(texts)
        vector_index.add(embeddings, chunks)
        vector_index.save_index_by_name(graph_name, "chunks")

        log.info("Incremental vector add: %d vectors added to index", len(chunks))
        return {"vectors_added": len(chunks)}
