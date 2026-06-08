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

"""Incremental indexing flow for HugeGraph-AI GraphRAG.

Processes new documents incrementally without full graph reconstruction:
1. ChunkSplit: Split new documents into chunks
2. ExtractInfo: LLM entity/relation extraction from chunks
3. EntityResolution: Merge new entities with existing ones (Sprint 1)
4. CommitNew: Append new vertices/edges to HugeGraph
5. DetectAffectedCommunities: Find communities impacted by new vertices
6. IncrementalCommunityReport: Regenerate reports for affected communities only
7. IncrementalVectorAdd: Add new chunk vectors to existing FAISS index
8. IncrementalCommunityIndex: Update community vector index for affected communities

Usage:
    flow = IncrementalIndexFlow(
        client=hugegraph_client,
        llm=extract_llm,
        embedding=emb,
        vector_index_cls=faiss_cls,
    )
    result = flow.run(texts=["New document content..."])
"""

from typing import Any, Dict, List, Optional

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.operators.graph_op.incremental_utils import (
    clear_stale_community_assignments,
    find_affected_communities,
    get_community_edges,
    get_community_vertices,
    persist_community_assignments,
)
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class IncrementalIndexFlow(BaseFlow):
    """Incremental indexing flow: new documents → partial KG + vector + community update.

    Unlike BuildVectorIndexFlow + GraphExtractFlow + CommunityDetectionFlow
    which rebuild everything, this flow only processes the new content and
    updates the affected portions of the index.

    DAG:
        ChunkSplit → ExtractInfo → EntityResolution → CommitNew →
        DetectAffected → RebuildAffectedReports → IncrementalVectorAdd →
        IncrementalCommunityIndexUpdate
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
    ):
        self._client = client
        self._llm = llm
        self._embedding = embedding
        self._vector_index_cls = vector_index_cls
        self._entity_resolution_strategy = entity_resolution_strategy
        self._entity_resolution_threshold = entity_resolution_threshold
        self._community_hop = community_hop

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        """Prepare the flow input with new document texts.

        Args:
            prepared_input: Flow input container.
            **kwargs: Must include 'texts' (list of new document strings).
        """
        texts = kwargs.get("texts", [])
        prepared_input.text = texts if isinstance(texts, list) else [texts]
        prepared_input.graph_name = kwargs.get("graph_name", "")

    def build_flow(self, **kwargs):
        """Build the incremental indexing DAG pipeline.

        The pipeline processes new documents through extraction, entity
        resolution, graph commit, affected community detection, and
        incremental index updates.
        """
        from hugegraph_llm.nodes.hugegraph_node.commit_to_hugegraph import CommitToHugegraphNode
        from hugegraph_llm.nodes.hugegraph_node.entity_resolution_node import EntityResolutionNode

        pipeline = GPipeline()

        prepared_input = WkFlowInput()
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # ── Node 1: ChunkSplit (reuse existing) ──
        from hugegraph_llm.nodes.document_node.chunk_split_node import ChunkSplitNode
        chunk_node = ChunkSplitNode(embedding=self._embedding)

        # ── Node 2: ExtractInfo (reuse existing) ──
        from hugegraph_llm.nodes.graph_node.extract_nodes import ExtractNode
        extract_node = ExtractNode(llm=self._llm)

        # ── Node 3: EntityResolution (Sprint 1 product) ──
        resolution_node = EntityResolutionNode(
            client=self._client,
            llm=self._llm,
            embedding=self._embedding,
            strategy=self._entity_resolution_strategy,
            threshold=self._entity_resolution_threshold,
        )

        # ── Node 4: CommitToHugegraph (reuse existing, append-only) ──
        commit_node = CommitToHugegraphNode(client=self._client)

        # ── Node 5: AffectedCommunityDetect (new) ──
        affected_node = AffectedCommunityDetectNode(
            client=self._client,
            hop=self._community_hop,
        )

        # ── Node 6: IncrementalCommunityReport (new) ──
        report_node = IncrementalCommunityReportNode(
            client=self._client,
            llm=self._llm,
            vector_index_cls=self._vector_index_cls,
            embedding=self._embedding,
        )

        # ── Node 7: IncrementalVectorAdd (new) ──
        vector_node = IncrementalVectorAddNode(
            vector_index_cls=self._vector_index_cls,
            embedding=self._embedding,
        )

        # Register DAG: linear chain
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
            return {"error": "No pipeline provided"}

        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        state_json = state.to_json() if state else {}

        return {
            "status_code": 200,
            "message": "Incremental indexing completed",
            "vertices_added": state_json.get("vertices_added", 0),
            "edges_added": state_json.get("edges_added", 0),
            "entities_merged": state_json.get("resolution_result", {}).get("merged_count", 0),
            "affected_communities": state_json.get("affected_community_count", 0),
            "vectors_added": state_json.get("vectors_added", 0),
            "community_reports_updated": state_json.get("community_reports_updated", 0),
        }

    def run_sync(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Synchronous entry point for incremental indexing.

        Args:
            texts: List of new document strings to index.
            **kwargs: Additional options (graph_name, etc.)

        Returns:
            Dict with indexing results.
        """
        return self.build_flow(texts=texts, **kwargs)

    # ── Lazy dependency setters ─────────────────────────────

    def set_client(self, client) -> None:
        self._client = client

    def set_llm(self, llm) -> None:
        self._llm = llm

    def set_embedding(self, embedding) -> None:
        self._embedding = embedding

    def set_vector_index_cls(self, cls) -> None:
        self._vector_index_cls = cls


# ── New DAG Nodes for Incremental Pipeline ─────────────────────


class AffectedCommunityDetectNode:
    """Detect communities affected by newly added vertices.

    Uses Gremlin to find 1-hop neighbors of new vertices and
    collects their community_id assignments.
    """

    def __init__(self, client: Any, hop: int = 1):
        self._client = client
        self._hop = hop

    def operator_schedule(self) -> Any:
        """Return a callable that will be scheduled in the DAG."""
        return self._run

    def _run(self, wkflow_state: WkFlowState) -> Any:
        from hugegraph_llm.operators.graph_op.incremental_utils import find_affected_communities

        # Get new vertex IDs from resolution result
        new_vids = wkflow_state.vertex_ids or []
        if not new_vids:
            log.info("No new vertex IDs found for affected community detection")
            wkflow_state.affected_community_ids = []
            wkflow_state.affected_community_count = 0
            return wkflow_state

        affected = find_affected_communities(self._client, new_vids, hop=self._hop)
        wkflow_state.affected_community_ids = list(affected)
        wkflow_state.affected_community_count = len(affected)

        log.info(
            "Affected community detection: %d communities affected by %d new vertices",
            len(affected), len(new_vids),
        )
        return wkflow_state


class IncrementalCommunityReportNode:
    """Regenerate community reports only for affected communities.

    Fetches vertices/edges for affected communities, regenerates
    LLM summaries, and updates the community vector index.
    """

    def __init__(
        self,
        client: Any,
        llm: Any,
        vector_index_cls: Any = None,
        embedding: Any = None,
    ):
        self._client = client
        self._llm = llm
        self._vector_index_cls = vector_index_cls
        self._embedding = embedding

    def operator_schedule(self) -> Any:
        return self._run

    def _run(self, wkflow_state: WkFlowState) -> Any:
        from hugegraph_llm.operators.llm_op.community_report import CommunityReportGenerate

        affected_ids = wkflow_state.affected_community_ids or []
        if not affected_ids:
            log.info("No affected communities to rebuild")
            wkflow_state.community_reports_updated = 0
            return wkflow_state

        # Step 1: Fetch affected community vertices
        community_vertices = get_community_vertices(self._client, set(affected_ids))

        # Step 2: Fetch edges within affected communities
        all_vids = set()
        for vids in community_vertices.values():
            all_vids.update(v.get("id") for v in vids if v.get("id"))
        community_edges = get_community_edges(self._client, all_vids)

        # Step 3: Rebuild community reports for affected communities only
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
        wkflow_state.community_reports_updated = len(reports)

        # Step 4: Update community vector index (only affected communities)
        if reports and self._vector_index_cls and self._embedding:
            self._update_community_index(reports)

        log.info(
            "Incremental community rebuild: %d reports regenerated",
            len(reports),
        )
        return wkflow_state

    def _update_community_index(self, reports: List[Dict]) -> None:
        """Update the community vector index with new reports."""
        from hugegraph_llm.config import huge_settings

        graph_name = huge_settings.graph_name
        vector_index = self._vector_index_cls.from_name(
            self._embedding.get_text_embedding("test").__len__() if self._embedding else 1024,
            graph_name, "communities",
        )

        texts = []
        for report in reports:
            title = report.get("title", "")
            summary = report.get("summary", "")
            key_entities = report.get("key_entities", [])
            texts.append(f"Title: {title}\nSummary: {summary}\nKey Entities: {', '.join(key_entities)}")

        if texts:
            embeddings = self._embedding.get_embeddings_parallel(texts)
            vector_index.add(embeddings, reports)
            vector_index.save_index_by_name(graph_name, "communities")
            log.info("Updated community vector index with %d reports", len(texts))


class IncrementalVectorAddNode:
    """Incrementally add new chunk vectors to the existing FAISS index.

    Loads the existing index, appends new vectors, and saves.
    """

    def __init__(
        self,
        vector_index_cls: Any = None,
        embedding: Any = None,
    ):
        self._vector_index_cls = vector_index_cls
        self._embedding = embedding

    def operator_schedule(self) -> Any:
        return self._run

    def _run(self, wkflow_state: WkFlowState) -> Any:
        from hugegraph_llm.config import huge_settings

        chunks = wkflow_state.chunks or []
        if not chunks:
            log.info("No chunks to add to vector index")
            wkflow_state.vectors_added = 0
            return wkflow_state

        if not self._vector_index_cls or not self._embedding:
            log.warning("Vector index or embedding not configured; skipping incremental vector add")
            wkflow_state.vectors_added = 0
            return wkflow_state

        graph_name = huge_settings.graph_name

        # Load existing index (FAISS from_name handles loading existing)
        test_vec = self._embedding.get_text_embedding("test")
        vector_index = self._vector_index_cls.from_name(len(test_vec), graph_name, "chunks")

        # Embed new chunks
        texts = [c.get("content", c.get("text", str(c))) for c in chunks]
        embeddings = self._embedding.get_embeddings_parallel(texts)

        # Add to existing index
        vector_index.add(embeddings, chunks)
        vector_index.save_index_by_name(graph_name, "chunks")

        wkflow_state.vectors_added = len(chunks)
        log.info("Incremental vector add: %d vectors added to index", len(chunks))
        return wkflow_state
