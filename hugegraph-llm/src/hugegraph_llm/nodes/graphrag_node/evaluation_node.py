# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not with this file except in compliance
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

"""GraphRAG evaluation node."""

from typing import Any, Dict, Optional

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.graphrag_op.evaluation import GraphRAGEvaluator
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class EvaluationNode(BaseNode):
    """
    Node for GraphRAG evaluation.

    Evaluates answer quality across multiple dimensions
    (comprehensiveness, diversity, empowerment, faithfulness, etc.)
    """

    evaluator: Optional[GraphRAGEvaluator] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        llm = None
        try:
            from hugegraph_llm.config import llm_settings
            from hugegraph_llm.models.llms.init_llm import get_chat_llm

            llm = get_chat_llm(llm_settings)
        except Exception:  # pylint: disable=broad-except
            pass

        self.evaluator = GraphRAGEvaluator(llm=llm)
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self.evaluator.run(data_json)
