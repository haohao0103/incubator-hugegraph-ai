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

"""Simplified query planner node (LightRAG-style)."""

from typing import Any, Dict, Optional

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.graphrag_op.query_planner import QueryLevel, QueryPlanner
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class QueryPlannerNode(BaseNode):
    """
    Node for LightRAG-style query planning.

    Maps user queries to retrieval level (LOW/HIGH/HYBRID)
    without depending on community detection.
    """

    planner: Optional[QueryPlanner] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        default_level = QueryLevel.HYBRID
        llm = None
        try:
            from hugegraph_llm.config import llm_settings
            from hugegraph_llm.models.llms.init_llm import get_chat_llm

            llm = get_chat_llm(llm_settings)
        except Exception:  # pylint: disable=broad-except
            pass

        self.planner = QueryPlanner(llm=llm, default_level=default_level)
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self.planner.run(data_json)
