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

"""Query Classifier Node for agent workflow routing."""

from typing import Any, Dict

from hugegraph_llm.agents.agent_loop import QueryClassifier as Classifier
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class QueryClassifierNode(BaseNode):
    """Classifies user queries as simple or complex.

    Simple queries are routed to existing fast RAG flows.
    Complex queries enter the ReAct agent loop for multi-step reasoning.

    Writes to context:
        is_complex (bool): Whether the query requires agent reasoning.
        route_target (str): Which flow to route to ("agent" or "fast_graph_only").
    """

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, llm: BaseLLM = None):
        super().__init__()
        self._llm = llm

    def node_init(self):
        pass

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        query = data_json.get("query", "")
        classifier = Classifier()
        is_complex = classifier.classify(query, self._llm)

        if is_complex:
            data_json["is_complex"] = True
            data_json["route_target"] = "agent"
        else:
            data_json["is_complex"] = False
            data_json["route_target"] = "fast_graph_only"

        return data_json
