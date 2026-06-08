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

"""DAG node wrapper for EntityResolution operator."""

from typing import Dict, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.config import huge_settings
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.graph_op.entity_resolution import EntityResolution
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class EntityResolutionNode(BaseNode):
    """DAG node for entity resolution.

    Wraps the EntityResolution operator as a pycgraph GNode
    for use in GPipeline DAG workflows.

    Reads configuration from:
        - context.get("resolution_strategy", "hybrid")
        - context.get("resolution_threshold", 0.85)
        - context.get("resolution_batch_size", 50)
        - context.get("resolution_vertex_labels", None)
    """

    entity_resolution_op: Optional[EntityResolution] = None
    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def node_init(self):
        data_json = self.wk_input.data_json if self.wk_input.data_json else None
        if data_json:
            self.context.assign_from_json(data_json)

        client = PyHugeClient(
            url=huge_settings.graph_url,
            graph=huge_settings.graph_name,
            user=huge_settings.graph_user,
            pwd=huge_settings.graph_pwd,
            graphspace=huge_settings.graph_space,
        )

        self.entity_resolution_op = EntityResolution(
            client=client,
            strategy="hybrid",
            threshold=huge_settings.entity_resolution_threshold,
            batch_size=huge_settings.entity_resolution_batch_size,
        )
        return super().node_init()

    def operator_schedule(self, data_json) -> Optional[Dict]:
        if self.entity_resolution_op is None:
            raise RuntimeError("EntityResolutionNode not properly initialized")

        return self.entity_resolution_op.run(data_json)
