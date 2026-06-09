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

"""ReAct Agent Loop Node for multi-step graph reasoning."""

from typing import Any, Dict

from hugegraph_llm.agents.agent_loop import ReActAgent
from hugegraph_llm.agents.tool_registry import ToolRegistry
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class AgentLoopNode(BaseNode):
    """Executes the ReAct agent loop for complex queries.

    Takes the user query, runs the ReAct reasoning loop with tool
    execution, and produces a final answer with execution trace.

    Reads from context:
        query (str): The user's question.

    Writes to context:
        agent_answer (str): The final answer from the agent.
        agent_trace (list): List of agent step dicts.
        agent_total_steps (int): Number of steps taken.
    """

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm: BaseLLM,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        super().__init__()
        self._tool_registry = tool_registry
        self._llm = llm
        self._max_steps = max_steps
        self._verbose = verbose

    def node_init(self):
        self._agent = ReActAgent(
            tool_registry=self._tool_registry,
            llm=self._llm,
            max_steps=self._max_steps,
            verbose=self._verbose,
        )

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        query = data_json.get("query", "")
        if not query:
            data_json["agent_answer"] = ""
            data_json["agent_trace"] = []
            data_json["agent_total_steps"] = 0
            return data_json

        try:
            result = self._agent.run(query)
            data_json["agent_answer"] = result.answer
            data_json["agent_trace"] = [s.to_dict() for s in result.trace]
            data_json["agent_total_steps"] = len(result.trace)
            data_json["agent_is_simple_query"] = result.is_simple_query
            log.info(
                "Agent completed in %d steps. Answer length: %d",
                len(result.trace),
                len(result.answer),
            )
        except Exception as e:
            log.error("Agent loop failed: %s", str(e))
            data_json["agent_answer"] = f"Agent execution failed: {str(e)}"
            data_json["agent_trace"] = []
            data_json["agent_total_steps"] = 0
            data_json["agent_error"] = str(e)

        return data_json
