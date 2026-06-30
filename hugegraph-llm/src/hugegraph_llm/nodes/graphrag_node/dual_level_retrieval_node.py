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

"""Dual-level retrieval node (LightRAG-style)."""

from typing import Any, Dict, Optional

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.graphrag_op.dual_level_retrieval import DualLevelRetriever
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class DualLevelRetrievalNode(BaseNode):
    """
    Node for LightRAG-style dual-level retrieval.

    Provides two retrieval granularities without community detection:
    - Low-level: Entity-centric for specific fact questions
    - High-level: Relationship-centric for abstract questions
    """

    retriever: Optional[DualLevelRetriever] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        graph_client = None
        try:
            from pyhugegraph.client import PyHugeClient

            from hugegraph_llm.config import huge_settings

            graph_client = PyHugeClient(
                url=huge_settings.graph_url,
                graph=huge_settings.graph_name,
                user=huge_settings.graph_user,
                pwd=huge_settings.graph_pwd,
                graphspace=huge_settings.graph_space,
            )
        except Exception:  # pylint: disable=broad-except
            pass

        self.retriever = DualLevelRetriever(graph_client=graph_client)
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self.retriever.run(data_json)
