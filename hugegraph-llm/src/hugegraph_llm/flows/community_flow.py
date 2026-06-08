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

"""Community detection and Global Search flows.

1. CommunityDetectionFlow: Offline pipeline for building communities.
   DAG: FetchGraphData → CommunityDetect → CommunityReportGenerate → BuildCommunityIndex

2. GlobalSearchFlow: Online query pipeline for macro Q&A.
   DAG: CommunityIndexQuery → GlobalSearchMap → GlobalSearchReduce
"""

from typing import Any, Dict

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.graph_node.community_nodes import (
    BuildCommunityIndexNode,
    CommunityDetectNode,
    CommunityIndexQueryNode,
    CommunityReportNode,
    GlobalSearchNode,
)
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class CommunityDetectionFlow(BaseFlow):
    """Offline flow: Community detection → Reports → Index.

    This is a build-time flow that processes an existing knowledge graph,
    detects communities, generates LLM summaries, and indexes them for
    efficient retrieval during Global Search.

    Usage:
        flow = CommunityDetectionFlow(client=hugegraph_client, llm=extract_llm,
                                       embedding=emb, vector_index=faiss_cls)
        scheduler.schedule_flow(FlowName.COMMUNITY_DETECT)
    """

    def __init__(
        self,
        client: Any = None,
        llm: Any = None,
        embedding: Any = None,
        vector_index_cls: Any = None,
        algorithm: str = "leiden",
        max_levels: int = 2,
    ):
        self._client = client
        self._llm = llm
        self._embedding = embedding
        self._vector_index_cls = vector_index_cls
        self._algorithm = algorithm
        self._max_levels = max_levels

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        prepared_input.graph_name = kwargs.get("graph_name", "")

    def build_flow(self, **kwargs):
        pipeline = GPipeline()

        prepared_input = WkFlowInput()
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Node 1: Community Detection
        detect_node = CommunityDetectNode(
            client=self._client,
            algorithm=self._algorithm,
            max_levels=self._max_levels,
        )

        # Node 2: Community Report Generation (depends on detection)
        report_node = CommunityReportNode(llm=self._llm)

        # Node 3: Build Community Index (depends on reports)
        index_node = BuildCommunityIndexNode(
            vector_index_cls=self._vector_index_cls,
            embedding=self._embedding,
        )

        # Register nodes with DAG dependencies
        pipeline.registerGElement(detect_node, set(), "community_detect")
        pipeline.registerGElement(
            report_node, {detect_node}, "community_report"
        )
        pipeline.registerGElement(
            index_node, {report_node}, "community_index"
        )

        pipeline.init()
        return pipeline

    def post_deal(self, pipeline=None, **kwargs) -> Dict[str, Any]:
        if pipeline is None:
            return {"error": "No pipeline provided"}

        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        state_json = state.to_json() if state else {}

        return {
            "status_code": 200,
            "message": "Community detection and indexing completed",
            "community_count": state_json.get("community_count", 0),
            "report_count": len(state_json.get("community_reports", [])),
            "index_built": state_json.get("community_index_built", False),
        }

    # ── Lazy dependency setters ─────────────────────────────

    def set_llm(self, llm) -> None:
        """Set the LLM for community report generation."""
        self._llm = llm

    def set_embedding(self, embedding) -> None:
        """Set the embedding model for community index."""
        self._embedding = embedding

    def set_vector_index_cls(self, cls) -> None:
        """Set the vector store class for community index."""
        self._vector_index_cls = cls


class GlobalSearchFlow(BaseFlow):
    """Online flow: Global Search over community reports.

    This is a query-time flow that:
    1. Queries the community index for communities relevant to the user's question
    2. Runs MAP phase: per-community point-form findings
    3. Runs REDUCE phase: synthesizes a comprehensive answer

    Usage:
        flow = GlobalSearchFlow(llm=chat_llm, embedding=emb, vector_index_cls=faiss_cls)
        scheduler.schedule_flow(FlowName.GLOBAL_SEARCH, query="What are the main themes?")
    """

    def __init__(
        self,
        llm: Any = None,
        embedding: Any = None,
        vector_index_cls: Any = None,
    ):
        self._llm = llm
        self._embedding = embedding
        self._vector_index_cls = vector_index_cls

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        prepared_input.query = kwargs.get("query", "")

    def build_flow(self, **kwargs):
        pipeline = GPipeline()

        prepared_input = WkFlowInput()
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Node 1: Query community index for relevant communities
        query_node = CommunityIndexQueryNode(
            vector_index_cls=self._vector_index_cls,
            embedding=self._embedding,
            top_k=10,
        )

        # Node 2: Global Search (MAP + REDUCE)
        search_node = GlobalSearchNode(llm=self._llm)

        # Register DAG
        pipeline.registerGElement(query_node, set(), "community_query")
        pipeline.registerGElement(
            search_node, {query_node}, "global_search"
        )

        pipeline.init()
        return pipeline

    def post_deal(self, pipeline=None, **kwargs) -> Dict[str, Any]:
        if pipeline is None:
            return {"error": "No pipeline provided"}

        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        state_json = state.to_json() if state else {}

        return {
            "status_code": 200,
            "answer": state_json.get("global_answer", ""),
            "map_findings": state_json.get("map_findings", []),
            "communities_used": state_json.get("communities_used", 0),
        }

    # ── Lazy dependency setters ─────────────────────────────

    def set_llm(self, llm) -> None:
        """Set the LLM for Global Search (MAP + REDUCE)."""
        self._llm = llm

    def set_embedding(self, embedding) -> None:
        """Set the embedding model for community index query."""
        self._embedding = embedding

    def set_vector_index_cls(self, cls) -> None:
        """Set the vector store class for community index."""
        self._vector_index_cls = cls
