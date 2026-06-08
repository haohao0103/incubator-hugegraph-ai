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

"""Provenance-aware KG construction flow.

Extends the standard GraphExtractFlow to include Document → Chunk → Entity
provenance tracking, enabling answer citations and audit trails.
"""

from typing import Any, Dict

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.operators.hugegraph_op.provenance_manager import ProvenanceManager
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class ProvenanceAwareKGFlow(BaseFlow):
    """Knowledge graph construction with text provenance tracking.

    This flow extends the standard graph extraction pipeline by:
    1. Creating Document and Chunk nodes in HugeGraph
    2. Recording which chunk each entity/relation was extracted from
    3. Building provenance edges (CONTAINS_CHUNK, EXTRACTED_FROM)

    Usage:
        flow = ProvenanceAwareKGFlow(client=hugegraph_client)
        scheduler.schedule_flow(FlowName.PROVENANCE_KG_BUILD,
                                doc_name="report.pdf",
                                texts=["..."])
    """

    def __init__(
        self,
        client: Any = None,
        provenance_manager: ProvenanceManager = None,
    ):
        self._client = client
        self._pm = provenance_manager
        if self._pm is None and client is not None:
            self._pm = ProvenanceManager(client=client)
            self._pm.init_schema()

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        prepared_input.texts = kwargs.get("texts", "")
        prepared_input.language = kwargs.get("language", "EN")
        prepared_input.split_type = kwargs.get("split_type", "paragraph")
        prepared_input.schema = kwargs.get("schema")
        prepared_input.extract_type = kwargs.get("extract_type", "property_graph")
        # Provenance-specific
        prepared_input.data_json = {
            "doc_name": kwargs.get("doc_name", "unknown"),
            "doc_source": kwargs.get("doc_source", ""),
        }

    def build_flow(self, **kwargs):
        """Build the provenance-aware KG construction pipeline.

        DAG:
        [CreateDocument] → [ChunkSplit] → [Extract] → [LinkProvenance] → [Commit2Graph]
        """
        from hugegraph_llm.nodes.document_node.chunk_split import ChunkSplitNode
        from hugegraph_llm.nodes.hugegraph_node.commit_to_hugegraph import (
            Commit2GraphNode,
        )
        from hugegraph_llm.nodes.llm_node.extract_info import ExtractInfoNode

        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Node 1: Chunk Split
        chunk_node = ChunkSplitNode()

        # Node 2: Entity/Relation Extraction
        extract_node = ExtractInfoNode()

        # Node 3: Provenance Linking
        provenance_node = _ProvenanceLinkNode(self._pm)

        # Node 4: Commit to Graph
        commit_node = Commit2GraphNode()

        # Register DAG
        pipeline.registerGElement(chunk_node, set(), "chunk_split")
        pipeline.registerGElement(extract_node, {chunk_node}, "extract")
        pipeline.registerGElement(provenance_node, {extract_node}, "provenance_link")
        pipeline.registerGElement(commit_node, {provenance_node}, "commit")

        pipeline.init()
        return pipeline

    def post_deal(self, pipeline=None, **kwargs) -> Dict[str, Any]:
        if pipeline is None:
            return {"error": "No pipeline provided"}

        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        state_json = state.to_json() if state else {}

        return {
            "status_code": 200,
            "message": "KG construction with provenance completed",
            "vertex_count": len(state_json.get("vertices", [])),
            "edge_count": len(state_json.get("edges", [])),
            "provenance_links": state_json.get("provenance_link_count", 0),
        }


class _ProvenanceLinkNode:
    """Internal node for provenance linking (wraps ProvenanceManager).

    This is a simplified node that doesn't use GPipeline DAG registration
    for the provenance schema setup, since it needs to run before
    Commit2Graph to ensure labels exist.
    """

    def __init__(self, pm: ProvenanceManager):
        self._pm = pm

    def init(self):
        if self._pm:
            self._pm.init_schema()

    def run(self, context: WkFlowState) -> Dict[str, Any]:
        """Link extracted entities to their source chunks."""
        data = context.to_json()
        vertices = data.get("vertices", [])
        chunks = data.get("chunks", [])
        doc_name = data.get("doc_name", "unknown")
        doc_source = data.get("doc_source", "")

        if not self._pm or not vertices:
            data["provenance_link_count"] = 0
            return data

        # Create document node
        doc_id = self._pm.create_document(doc_name, doc_source)

        # Create chunk nodes and link entities
        link_count = 0
        for i, chunk_text in enumerate(chunks):
            chunk_id = self._pm.create_chunk(doc_id, chunk_text, i)

            # Find entities that were potentially extracted from this chunk
            # Match by checking if entity names appear in the chunk text
            chunk_lower = chunk_text.lower()
            for vertex in vertices:
                props = vertex.get("properties", {})
                names = [props.get(k, "") for k in props if "name" in k.lower()]
                for name in names:
                    if name and name.lower() in chunk_lower:
                        if self._pm.link_entity_to_chunk(vertex.get("id", ""), chunk_id):
                            link_count += 1
                            break

        data["provenance_link_count"] = link_count
        data["doc_id"] = doc_id
        log.info("Created %d provenance links for document %s", link_count, doc_name)
        return data
