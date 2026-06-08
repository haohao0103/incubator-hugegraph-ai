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

"""DRIFT search flow — Dynamic Reasoning and Inference with Flexible Traversal."""

from typing import Optional

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.graph_node.drift_node import DriftSearchNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from pycgraph import CStatus, GPipeline


class DriftFlow(BaseFlow):
    """DRIFT search flow for deep analytical questions.

    Combines the breadth of Global Search with the depth of Local Search
    through a 5-step pipeline:
    1. HyDE: Generate hypothetical answer
    2. Community Match: Find relevant communities
    3. Primer: Initial analysis + follow-up questions
    4. Parallel Local Search: Iterative deep search
    5. Reduce: Synthesize comprehensive answer
    """

    def prepare(
        self,
        prepared_input: WkFlowInput,
        query: str,
        drift_max_depth: Optional[int] = None,
        drift_communities_top_k: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> CStatus:
        """Prepare input parameters for DRIFT search.

        :param prepared_input: The WkFlowInput parameter container.
        :param query: The user's analytical question.
        :param drift_max_depth: Max iteration depth (1-2).
        :param drift_communities_top_k: Number of top communities to match.
        :param language: "en" or "cn".
        """
        prepared_input.query = query
        if drift_max_depth is not None:
            prepared_input.drift_max_depth = drift_max_depth
        if drift_communities_top_k is not None:
            prepared_input.drift_communities_top_k = drift_communities_top_k
        if language is not None:
            prepared_input.language = language

        prepared_input.wk_type = "drift_flow"
        return CStatus()

    def build_flow(self, **kwargs) -> GPipeline:
        """Build the DRIFT search DAG pipeline."""
        pipeline = GPipeline()

        drift_node = DriftSearchNode()
        pipeline.registerGElement(drift_node, set(), "drift")

        return pipeline

    def post_deal(self, pipeline=None, **kwargs) -> dict:
        """Extract DRIFT search results."""
        if pipeline is None:
            return {
                "drift_answer": "",
                "error": "Pipeline not executed",
            }

        result = pipeline.get_final_result()
        if result is None:
            return {
                "drift_answer": "",
                "error": "No result from pipeline",
            }

        return {
            "drift_answer": result.get("drift_answer", ""),
            "drift_findings": result.get("drift_findings", []),
            "drift_communities_used": result.get("drift_communities_used", 0),
            "drift_depth_reached": result.get("drift_depth_reached", 0),
            "drift_primer": result.get("drift_primer", {}),
            "call_count": result.get("call_count", 0),
        }
