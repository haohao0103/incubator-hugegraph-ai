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

"""Tool Execution Node for individual tool calls within the agent pipeline."""

from typing import Any, Dict

from hugegraph_llm.agents.tool_registry import ToolRegistry
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class ToolExecutionNode(BaseNode):
    """Executes a single tool call from within the pipeline.

    This node is used when the agent flow needs to execute a tool
    directly (not via the ReAct loop), e.g., pre-processing steps
    like schema lookup before entering the agent loop.

    Reads from context via wk_input:
        tool_name (str): Name of the tool to execute.
        tool_params (dict): Parameters for the tool.

    Writes to context:
        tool_result (dict): The result of tool execution.
        tool_success (bool): Whether the execution succeeded.
    """

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, tool_registry: ToolRegistry, tool_name: str = None):
        super().__init__()
        self._tool_registry = tool_registry
        self._tool_name = tool_name  # Can be set at init or passed via wk_input

    def node_init(self):
        pass

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = self._tool_name or data_json.get("tool_name", "")
        tool_params = data_json.get("tool_params", {})

        if not tool_name:
            log.warning("ToolExecutionNode: no tool_name specified")
            data_json["tool_result"] = {}
            data_json["tool_success"] = False
            return data_json

        try:
            result = self._tool_registry.execute(tool_name, **tool_params)
            data_json["tool_result"] = result.get("data", result)
            data_json["tool_success"] = result.get("success", False)
            log.info("Tool '%s' executed successfully.", tool_name)
        except Exception as e:
            log.error("Tool '%s' execution failed: %s", tool_name, str(e))
            data_json["tool_result"] = {"error": str(e)}
            data_json["tool_success"] = False

        return data_json
