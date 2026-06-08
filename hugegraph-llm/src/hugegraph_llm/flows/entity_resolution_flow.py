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

"""Entity Resolution Flow.

DAG: EntityResolutionNode (standalone) or chained after GraphExtractFlow.

Two usage modes:
    1. Standalone: resolves entities already in the graph store.
    2. Chained: resolves newly extracted entities before committing.
"""

import json

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.hugegraph_node.entity_resolution_node import EntityResolutionNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class EntityResolutionFlow(BaseFlow):
    """Flow for entity resolution.

    Standalone mode (resolves existing graph)::

        flow = EntityResolutionFlow()
        pipeline = flow.build_flow()
        pipeline.run()
        result = flow.post_deal(pipeline)

    Chained mode (resolves after extraction)::

        pipeline = GPipeline()
        # ... register schema, chunk_split, extract nodes ...
        resolution_node = EntityResolutionNode()
        pipeline.registerGElement(resolution_node, {extract_node}, "entity_resolution")
    """

    def prepare(
        self,
        prepared_input: WkFlowInput,
        vertex_labels=None,
        strategy="hybrid",
        threshold=None,
        batch_size=None,
        **kwargs,
    ):
        """Prepare input parameters for the resolution flow."""
        if vertex_labels:
            prepared_input.data_json = prepared_input.data_json or {}
            prepared_input.data_json["vertex_labels"] = vertex_labels
        if strategy:
            prepared_input.data_json = prepared_input.data_json or {}
            prepared_input.data_json["resolution_strategy"] = strategy
        if threshold is not None:
            prepared_input.data_json = prepared_input.data_json or {}
            prepared_input.data_json["resolution_threshold"] = threshold
        if batch_size is not None:
            prepared_input.data_json = prepared_input.data_json or {}
            prepared_input.data_json["resolution_batch_size"] = batch_size

    def build_flow(self, **kwargs):
        """Build the entity resolution DAG pipeline."""
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        self.prepare(prepared_input, **kwargs)

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        resolution_node = EntityResolutionNode()
        pipeline.registerGElement(resolution_node, set(), "entity_resolution")

        return pipeline

    def post_deal(self, pipeline=None, **kwargs):
        """Extract and format resolution results."""
        if pipeline is None:
            return json.dumps({"error": "No pipeline provided"}, ensure_ascii=False)

        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        resolution_result = res.get("resolution_result", {})

        log.info(
            "Entity resolution complete: merged_count=%d, edges_migrated=%d, errors=%d",
            resolution_result.get("merged_count", 0),
            resolution_result.get("edges_migrated", 0),
            len(resolution_result.get("errors", [])),
        )

        return json.dumps(resolution_result, ensure_ascii=False, indent=2)
