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

"""E2E RAG Flow.

DAG: E2ERAGNode (single step)

Unified entry point for build / query / refresh / assess workflows.
The stage is set via prepare().
"""

from typing import Any, Dict

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.graph_node.e2e_rag_node import E2ERAGNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class E2ERAGFlow(BaseFlow):
    """End-to-end RAG flow orchestrator.

    DAG:
        E2ERAGNode (single step)

    Usage:
        # Build
        scheduler.schedule_flow(
            FlowName.E2E_RAG, stage="build",
            documents=[{"content": "...", "id": "doc1"}]
        )

        # Query
        scheduler.schedule_flow(
            FlowName.E2E_RAG, stage="query",
            question="What is HugeGraph?"
        )

        # Refresh
        scheduler.schedule_flow(
            FlowName.E2E_RAG, stage="refresh"
        )
    """

    def __init__(
        self,
        llm=None,
        embedding=None,
        graph_client=None,
        vector_index_cls=None,
    ):
        self._llm = llm
        self._embedding = embedding
        self._graph_client = graph_client
        self._vector_index_cls = vector_index_cls

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        """Set up flow input parameters."""
        stage = kwargs.get("stage", "query")
        data = {"stage": stage}
        if stage == "build":
            data["documents"] = kwargs.get("documents", [])
            data["options"] = kwargs.get("options", {})
        elif stage == "query":
            data["question"] = kwargs.get("question", "")
            data["context"] = kwargs.get("context", None)
            data["mode"] = kwargs.get("mode", "auto")
        elif stage == "refresh":
            data["scope"] = kwargs.get("scope", "stale")
            data["options"] = kwargs.get("options", {})
        prepared_input.data_json = data

    def build_flow(self, **kwargs) -> GPipeline:
        """Build the E2E RAG pipeline DAG."""
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        self.prepare(prepared_input, **kwargs)

        state = WkFlowState()
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(state, "wkflow_state")

        node = E2ERAGNode(
            llm=self._llm,
            embedding=self._embedding,
            graph_client=self._graph_client,
            vector_index_cls=self._vector_index_cls,
        )
        pipeline.registerGElement(node, set(), "e2e_rag")
        return pipeline

    def post_deal(self, pipeline=None, **kwargs) -> Dict[str, Any]:
        """Extract pipeline result from state."""
        if pipeline is None:
            return {"error": "No pipeline provided"}
        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        state_json = state.to_json() if state else {}
        result = state_json.get("e2e_rag_result", state_json)
        return {
            "status_code": 200,
            "result": result,
        }
