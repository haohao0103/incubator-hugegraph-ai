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

"""ReAct (Reasoning + Acting) Agent Loop for multi-step graph queries.

This module implements a ReAct-pattern agent that can reason about complex
graph queries, select appropriate tools, execute them, and synthesize results.

The agent uses the existing ToolRegistry to access graph search, vector
search, Text2Gremlin, and other capabilities as LLM-callable tools.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from hugegraph_llm.agents.tool_registry import ToolRegistry
from hugegraph_llm.utils.log import log


# ── Complex query patterns (English + Chinese) ──────────────────

_COMPLEX_QUERY_PATTERNS = [
    # English patterns
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\banaly(?:ze|sis)\b",
    r"\bsummar(?:ize|y)\b",
    r"\brelationship\s+(?:between|among|chain)",
    r"\b(?:impact|effect|influence|cause|lead\s+to)\b",
    r"\bhow\s+(?:does|do|are|is|did|can)\b",
    r"\bwhat\s+(?:is\s+the\s+)?relation",
    r"\bevol(?:ve|ution)\b",
    r"\bconnect(?:ed|ion)\b",
    r"\btrend\b",
    r"\bpattern\b",
    r"\b(?:find|list|show)\s+(?:all|every|entities)\b",
    # Chinese patterns
    r"比较",
    r"对比",
    r"分析",
    r"总结",
    r"关系",
    r"影响",
    r"导致",
    r"原因",
    r"如何",
    r"演变",
    r"连接",
    r"趋势",
    r"模式",
    r"所有.*(?:实体|关系|节点)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_QUERY_PATTERNS]

# ── Data classes ─────────────────────────────────────────────────


@dataclass
class AgentStep:
    """A single step in the ReAct loop."""

    step_num: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    """Result of an agent run."""

    answer: str
    trace: List[AgentStep] = field(default_factory=list)
    is_simple_query: bool = False
    simple_flow_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "trace": [s.to_dict() for s in self.trace],
            "is_simple_query": self.is_simple_query,
            "simple_flow_used": self.simple_flow_used,
            "total_steps": len(self.trace),
        }


# ── Query Classifier ─────────────────────────────────────────────


class QueryClassifier:
    """Classifies queries as simple or complex to decide routing.

    Simple queries are routed to existing fast flows (Raw/Vector/Graph/Hybrid).
    Complex queries enter the ReAct agent loop for multi-step reasoning.

    Uses dual classification:
    1. Code-side regex matching (fast, no API call)
    2. LLM-side classification (for nuanced cases, optional)
    """

    @staticmethod
    def is_complex_regex(query: str) -> bool:
        """Check if the query matches complex patterns using regex."""
        match_count = 0
        for pattern in _COMPILED_PATTERNS:
            if pattern.search(query):
                match_count += 1
        # Need at least 1 complex pattern match
        return match_count >= 1

    @staticmethod
    def is_complex_llm(query: str, llm: Any) -> bool:
        """Use LLM to classify query complexity.

        This is more accurate for nuanced queries but adds latency/cost.
        Only called when regex classification is ambiguous.
        """
        classifier_prompt = (
            "Classify the following query as 'simple' or 'complex'.\n\n"
            "Simple queries ask for specific facts about known entities. "
            "Example: 'Who is Sarah?', 'What is the capital of France?'\n\n"
            "Complex queries require multi-step reasoning, comparison, "
            "analysis, or understanding relationships between entities. "
            "Example: 'Compare the relationships between entity A and B', "
            "'Analyze how X influences Y through the network.'\n\n"
            f"Query: {query}\n\n"
            "Respond with exactly one word: 'simple' or 'complex'."
        )
        try:
            response = llm.generate(prompt=classifier_prompt).strip().lower()
            return "complex" in response
        except Exception as e:
            log.warning("LLM query classification failed: %s, falling back to regex", e)
            return QueryClassifier.is_complex_regex(query)

    @classmethod
    def classify(cls, query: str, llm: Any = None) -> bool:
        """Classify a query as complex (True) or simple (False).

        Uses regex first; only falls back to LLM if available and regex
        result is borderline (few matches).
        """
        is_complex = cls.is_complex_regex(query)

        # If regex says complex, trust it (save API call)
        if is_complex:
            return True

        # If regex says simple but LLM is available, double-check
        if llm is not None:
            return cls.is_complex_llm(query, llm)

        return is_complex


# ── ReAct Agent ──────────────────────────────────────────────────


class ReActAgent:
    """A ReAct (Reasoning + Acting) agent for knowledge graph queries.

    The agent maintains a conversation loop where the LLM:
    1. Reasons about what information is needed (Thought)
    2. Selects and calls a tool (Action + Action Input)
    3. Observes the tool result (Observation)
    4. Repeats until it can provide a Final Answer

    This enables multi-step reasoning like:
    - "Compare the relationships of entity A and entity B"
      → keyword_extract("A and B") → semantic_id_lookup(["A","B"])
      → graph_traverse([A_id, B_id]) → answer_synthesize(context)

    Usage:
        agent = ReActAgent(tool_registry, llm)
        result = await agent.run("Complex multi-step query about the graph")
    """

    # ReAct system prompt template
    SYSTEM_PROMPT_TEMPLATE = """You are an intelligent knowledge graph analysis agent. Your task is to answer user questions by reasoning step-by-step and using available tools to explore the graph.

## Available Tools
{tool_descriptions}

## Response Format
You MUST respond using the following format:

Thought: <reason about what you need to do next>
Action: <the name of the tool to call>
Action Input: <JSON object with tool parameters>

... (repeat Thought/Action/Action Input as needed)

When you have enough information to answer the question, respond with:

Thought: I now have enough information to answer the question.
Final Answer: <your comprehensive answer>

## Important Rules
1. Always start by extracting keywords from the user query using keyword_extract.
2. Use semantic_id_lookup to find vertex IDs for the keywords before doing graph traversals.
3. Use schema_lookup to understand the graph structure before writing complex queries.
4. Use graph_traverse to explore relationships between entities.
5. Use text2gremlin for complex graph queries that need precise traversal patterns.
6. Use vector_search to find relevant document passages for factual questions.
7. Use answer_synthesize ONLY as the final step to produce a natural language answer.
8. NEVER fabricate information. If tools don't return what you need, say so honestly.
9. Each step should make progress toward answering the question.
10. If the same tool returns the same results, stop and synthesize what you have.

## Current Conversation
{conversation_history}
"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm: Any,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        """Initialize the ReAct agent.

        Args:
            tool_registry: ToolRegistry with registered tools.
            llm: LLM instance for reasoning (should support generate() with messages).
            max_steps: Maximum number of ReAct steps before forced stop.
            verbose: If True, log detailed step information.
        """
        self.tool_registry = tool_registry
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose

    # ── Public API ────────────────────────────────────────────

    def run(self, query: str, stream: bool = False) -> AgentResult:
        """Run the agent synchronously.

        Args:
            query: The user's natural language question.
            stream: If True, yield intermediate steps (not yet implemented for sync).

        Returns:
            AgentResult with the final answer and execution trace.
        """
        classifier = QueryClassifier()
        is_complex = classifier.classify(query, self.llm)

        if not is_complex:
            return AgentResult(
                answer="",
                is_simple_query=True,
                simple_flow_used="graph_only",
            )

        steps: List[AgentStep] = []
        messages = self._build_initial_messages(query)

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                log.info("--- Agent Step %d/%d ---", step_num, self.max_steps)

            # Call the LLM
            response = self.llm.generate(messages=messages)

            # Parse the response
            step = self._parse_react_response(response, step_num)
            if self.verbose:
                log.info(
                    "Thought: %s | Action: %s | Input: %s",
                    step.thought[:100],
                    step.action,
                    str(step.action_input)[:100],
                )

            steps.append(step)

            # Check if the agent wants to stop
            if step.action == "FINAL_ANSWER":
                return AgentResult(
                    answer=step.action_input.get("answer", response),
                    trace=steps,
                )

            # Execute the tool
            observation = self._execute_tool(step)
            step.observation = observation

            # Add the exchange to the conversation
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}",
                }
            )

        # Max steps reached - force final answer
        final_prompt = (
            "You have reached the maximum number of steps. "
            "Based on all the observations above, please provide your "
            "Final Answer to the original question."
        )
        messages.append({"role": "user", "content": final_prompt})
        final_response = self.llm.generate(messages=messages)

        return AgentResult(answer=final_response, trace=steps)

    async def arun(self, query: str) -> AgentResult:
        """Run the agent asynchronously.

        Args:
            query: The user's natural language question.

        Returns:
            AgentResult with the final answer and execution trace.
        """
        classifier = QueryClassifier()
        is_complex = classifier.classify(query, self.llm)

        if not is_complex:
            return AgentResult(
                answer="",
                is_simple_query=True,
                simple_flow_used="graph_only",
            )

        steps: List[AgentStep] = []
        messages = self._build_initial_messages(query)

        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                log.info("--- Agent Step %d/%d (async) ---", step_num, self.max_steps)

            response = await self.llm.agenerate(messages=messages)
            step = self._parse_react_response(response, step_num)
            steps.append(step)

            if step.action == "FINAL_ANSWER":
                return AgentResult(
                    answer=step.action_input.get("answer", response),
                    trace=steps,
                )

            observation = self._execute_tool(step)
            step.observation = observation

            messages.append({"role": "assistant", "content": response})
            messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

        final_prompt = (
            "You have reached the maximum number of steps. "
            "Based on all the observations above, please provide your "
            "Final Answer to the original question."
        )
        messages.append({"role": "user", "content": final_prompt})
        final_response = await self.llm.agenerate(messages=messages)

        return AgentResult(answer=final_response, trace=steps)

    async def arun_stream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the agent asynchronously, yielding each step as it completes.

        Args:
            query: The user's natural language question.

        Yields:
            Dict with step information or final result.
        """
        classifier = QueryClassifier()
        is_complex = classifier.classify(query, self.llm)

        if not is_complex:
            yield {
                "type": "routing",
                "is_simple_query": True,
                "simple_flow_used": "graph_only",
                "message": "Query routed to fast graph-only RAG flow.",
            }
            return

        messages = self._build_initial_messages(query)
        steps: List[AgentStep] = []

        for step_num in range(1, self.max_steps + 1):
            response = await self.llm.agenerate(messages=messages)
            step = self._parse_react_response(response, step_num)
            steps.append(step)

            # Yield thought
            yield {
                "type": "thought",
                "step_num": step_num,
                "thought": step.thought,
            }

            if step.action == "FINAL_ANSWER":
                answer = step.action_input.get("answer", response)
                yield {
                    "type": "final_answer",
                    "answer": answer,
                    "trace": [s.to_dict() for s in steps],
                }
                return

            # Yield action
            yield {
                "type": "action",
                "step_num": step_num,
                "action": step.action,
                "action_input": step.action_input,
            }

            # Execute and yield observation
            observation = self._execute_tool(step)
            step.observation = observation
            yield {
                "type": "observation",
                "step_num": step_num,
                "observation": observation,
            }

            messages.append({"role": "assistant", "content": response})
            messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

        # Force final answer
        final_prompt = (
            "You have reached the maximum number of steps. "
            "Please provide your Final Answer based on all observations."
        )
        messages.append({"role": "user", "content": final_prompt})
        final_response = await self.llm.agenerate(messages=messages)
        yield {
            "type": "final_answer",
            "answer": final_response,
            "trace": [s.to_dict() for s in steps],
            "forced": True,
        }

    # ── Internal Helpers ───────────────────────────────────────

    def _build_initial_messages(self, query: str) -> List[Dict[str, Any]]:
        """Build the initial message list for the ReAct loop."""
        tool_descriptions = self._format_tool_descriptions()

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            conversation_history=f"User Query: {query}",
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    def _format_tool_descriptions(self) -> str:
        """Format all registered tools as a readable description."""
        lines = []
        for tool_name in self.tool_registry.get_tool_names():
            tool = self.tool_registry._tools[tool_name]
            # Try to extract parameter descriptions
            params = tool.parameters.get("properties", {})
            param_desc = ", ".join(
                f"{k}: {v.get('description', v.get('type', 'any'))}"
                for k, v in params.items()
            )
            lines.append(f"- **{tool.name}**: {tool.description}")
            if param_desc:
                lines.append(f"  Parameters: {param_desc}")
        return "\n".join(lines)

    def _parse_react_response(self, response: str, step_num: int) -> AgentStep:
        """Parse the LLM response into a structured AgentStep.

        Supports two output formats:
        1. Standard ReAct: Thought/Action/Action Input
        2. Final Answer: Thought/Final Answer
        """
        step = AgentStep(step_num=step_num)

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Check for Final Answer first
        final_match = re.search(
            r"Final Answer:\s*(.+)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            step.action = "FINAL_ANSWER"
            step.action_input = {"answer": final_match.group(1).strip()}
            return step

        # Extract Action
        action_match = re.search(
            r"Action:\s*(\S+)",
            response,
            re.IGNORECASE,
        )
        if action_match:
            step.action = action_match.group(1).strip()

        # Extract Action Input (JSON)
        input_match = re.search(
            r"Action Input:\s*(\{.+?\})",
            response,
            re.DOTALL,
        )
        if input_match:
            try:
                step.action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to extract key=value pairs as fallback
                kv_match = re.search(
                    r"Action Input:\s*(.+)",
                    response,
                    re.DOTALL,
                )
                if kv_match:
                    raw_input = kv_match.group(1).strip()
                    step.action_input = {"raw_input": raw_input}
                    log.warning(
                        "Could not parse Action Input as JSON: %s", raw_input[:200]
                    )

        return step

    def _execute_tool(self, step: AgentStep) -> str:
        """Execute the tool specified in the step and return the observation.

        Returns:
            A string representation of the tool's output.
        """
        tool_name = step.action
        if not tool_name or tool_name == "FINAL_ANSWER":
            return "No action to execute."

        try:
            result = self.tool_registry.execute(tool_name, **step.action_input)

            if not result.get("success", False):
                return f"Error: {result.get('error', 'Unknown error')}"

            data = result.get("data", {})
            return json.dumps(data, ensure_ascii=False, indent=2)

        except ValueError as e:
            return f"Tool not found: {str(e)}"
        except Exception as e:
            log.error("Tool execution failed: %s", str(e))
            return f"Tool execution error: {str(e)}"


# ── Agent Factory ────────────────────────────────────────────────


def create_react_agent(
    tool_registry: ToolRegistry,
    llm: Any,
    max_steps: int = 10,
    verbose: bool = False,
) -> ReActAgent:
    """Create a ready-to-use ReAct agent.

    This is the recommended way to instantiate the agent.

    Args:
        tool_registry: ToolRegistry with all needed tools registered.
        llm: LLM instance for agent reasoning.
        max_steps: Maximum ReAct steps per query.
        verbose: Enable detailed logging.

    Returns:
        Configured ReActAgent instance.
    """
    return ReActAgent(
        tool_registry=tool_registry,
        llm=llm,
        max_steps=max_steps,
        verbose=verbose,
    )
