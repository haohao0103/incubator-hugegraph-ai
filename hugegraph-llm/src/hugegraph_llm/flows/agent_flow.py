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

"""Agent Flow for multi-step graph reasoning.

This flow integrates the ReAct agent loop into the existing DAG pipeline
architecture. It acts as a bridge between the pycgraph GPipeline system
and the agent orchestration layer.

Flow DAG:
    [SchemaNode] → [AgentLoopNode]
         ↓
    (Schema loaded as context for the agent)
         ↓
    [AgentLoopNode]
         → If simple: routes to fast RAG flow
         → If complex: runs ReAct loop with tools
"""

from typing import Any, Dict

from pycgraph import GPipeline

from hugegraph_llm.agents.tool_registry import ToolRegistry
from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.nodes.agent_node.agent_loop_node import AgentLoopNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class AgentFlow(BaseFlow):
    """Agent-based multi-step reasoning flow.

    This flow wraps the ReAct agent loop in the GPipeline pattern.
    It classifies queries and routes simple ones to fast flows,
    while running the full ReAct loop for complex multi-step queries.

    Input (via WkFlowInput):
        query: The user's natural language question.
        max_steps: Maximum ReAct steps (default: 10).
        stream: Whether to enable streaming output.

    Output (via WkFlowState):
        agent_answer: The final answer.
        agent_trace: List of reasoning steps.
        agent_total_steps: Number of ReAct steps taken.
        agent_is_simple_query: If True, query was routed to fast flow.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry = None,
        llm: BaseLLM = None,
        max_steps: int = 10,
    ):
        """Initialize the agent flow.

        Args:
            tool_registry: ToolRegistry with registered tools.
            llm: LLM instance for agent reasoning.
            max_steps: Maximum ReAct steps per query.
        """
        self._tool_registry = tool_registry
        self._llm = llm
        self._max_steps = max_steps

    def prepare(self, prepared_input: WkFlowInput, **kwargs):
        """Prepare the workflow input for agent execution.

        Args:
            prepared_input: WkFlowInput to populate.
            **kwargs: Can include 'query', 'max_steps', 'tools_filter'.
        """
        prepared_input.query = kwargs.get("query", "")
        prepared_input.stream = kwargs.get("stream", False)
        prepared_input.max_deep = kwargs.get("max_steps", self._max_steps)

    def build_flow(self, **kwargs) -> GPipeline:
        """Build the agent DAG pipeline.

        The pipeline contains a single AgentLoopNode that handles
        query classification and ReAct execution internally.
        """
        pipeline = GPipeline()

        # Create shared state objects
        prepared_input = WkFlowInput()
        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Ensure tool registry and LLM are set
        if self._tool_registry is None:
            raise ValueError("AgentFlow requires a ToolRegistry. Call set_tool_registry().")
        if self._llm is None:
            raise ValueError("AgentFlow requires an LLM. Call set_llm().")

        # Create the agent loop node (no dependencies = runs first)
        agent_node = AgentLoopNode(
            tool_registry=self._tool_registry,
            llm=self._llm,
            max_steps=self._max_steps,
        )
        pipeline.registerGElement(agent_node, set(), "agent_loop")

        pipeline.init()
        return pipeline

    def post_deal(self, pipeline: GPipeline = None, **kwargs) -> Dict[str, Any]:
        """Extract agent results from the pipeline state.

        Args:
            pipeline: The completed GPipeline instance.

        Returns:
            Dict with agent answer, trace, and metadata.
        """
        if pipeline is None:
            return {"error": "No pipeline provided"}

        state: WkFlowState = pipeline.getGParamWithNoEmpty("wkflow_state")
        if state is None:
            return {"error": "No workflow state found"}

        state_json = state.to_json()

        result = {
            "status_code": 200,
            "message": "Agent execution completed",
            "answer": state_json.get("agent_answer", ""),
            "trace": state_json.get("agent_trace", []),
            "total_steps": state_json.get("agent_total_steps", 0),
            "is_simple_query": state_json.get("agent_is_simple_query", False),
        }
        log.info(
            "Agent flow completed. Steps: %d, Answer length: %d",
            result["total_steps"],
            len(result["answer"]),
        )
        return result

    # ── Configuration Setters ─────────────────────────────────

    def set_tool_registry(self, tool_registry: ToolRegistry) -> None:
        """Set the tool registry for the agent flow."""
        self._tool_registry = tool_registry

    def set_llm(self, llm: BaseLLM) -> None:
        """Set the LLM for agent reasoning."""
        self._llm = llm

    def set_max_steps(self, max_steps: int) -> None:
        """Set the maximum ReAct steps."""
        self._max_steps = max_steps
