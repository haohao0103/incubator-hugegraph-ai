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

"""Tests for the ReAct Agent Loop and Query Classifier."""

from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.agents.agent_loop import (
    AgentResult,
    AgentStep,
    QueryClassifier,
    ReActAgent,
    create_react_agent,
)
from hugegraph_llm.agents.tool_registry import Tool, ToolRegistry


class TestQueryClassifier:
    """Unit tests for the Query Classifier."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Who is Sarah?", False),
            ("What is the capital of France?", False),
            ("Find entity X", False),
            ("Tell me about the weather", False),
            ("What is 2 + 2?", False),
            ("Compare entity A and entity B", True),
            ("Analyze the relationship between X and Y", True),
            ("How does A influence B through the network?", True),
            ("Summarize the main themes", True),
            ("对比实体 A 和 B", True),
            ("分析 X 和 Y 之间的关系", True),
            ("如何通过网络影响", True),
            ("总结主要主题", True),
            ("What is the relationship between all entities?", True),
            ("Find all entities connected to both A and B", True),
            ("List all entities connected to X", True),
        ],
    )
    def test_regex_classification(self, query, expected):
        """Test regex-based query classification."""
        result = QueryClassifier.is_complex_regex(query)
        assert result == expected, f"Query '{query}' should be {'complex' if expected else 'simple'}"

    def test_llm_classification_simple(self):
        """Test LLM-based classification for a simple query."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "simple"

        result = QueryClassifier.is_complex_llm("Who is Sarah?", mock_llm)
        assert result is False

    def test_llm_classification_complex(self):
        """Test LLM-based classification for a complex query."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "complex"

        result = QueryClassifier.is_complex_llm("Compare A and B", mock_llm)
        assert result is True

    def test_llm_classification_fallback(self):
        """Test that LLM failure falls back to regex."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")

        # Fallback to regex - this is a complex pattern
        result = QueryClassifier.is_complex_llm("Analyze the trends", mock_llm)
        assert result is True  # "Analyze" and "trends" both match complex patterns

    def test_classify_without_llm(self):
        """Test classify without LLM (regex only)."""
        assert QueryClassifier.classify("Who is Sarah?") is False
        assert QueryClassifier.classify("Compare X and Y") is True

    def test_classify_with_llm_simple(self):
        """Test classify with LLM for potentially borderline case."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "complex"

        # Regex says simple, LLM says complex → should check LLM
        result = QueryClassifier.classify("Tell me about entity X", mock_llm)
        assert result is True  # LLM overrides when regex says simple


class TestAgentStep:
    """Unit tests for the AgentStep dataclass."""

    def test_agent_step_creation(self):
        """Test basic AgentStep creation."""
        step = AgentStep(
            step_num=1,
            thought="I need to find the entity.",
            action="keyword_extract",
            action_input={"query": "test"},
            observation="Found keyword: test",
        )
        assert step.step_num == 1
        assert "keyword_extract" == step.action

    def test_to_dict(self):
        """Test AgentStep serialization."""
        step = AgentStep(
            step_num=2,
            thought="Now I will search.",
            action="vector_search",
            action_input={"query": "test", "top_k": 3},
            observation="Found 3 results.",
        )
        d = step.to_dict()
        assert d["step_num"] == 2
        assert d["action"] == "vector_search"
        assert d["action_input"]["top_k"] == 3
        assert d["observation"] == "Found 3 results."


class TestAgentResult:
    """Unit tests for the AgentResult dataclass."""

    def test_simple_query_result(self):
        """Test AgentResult for a simple query."""
        result = AgentResult(
            answer="",
            is_simple_query=True,
            simple_flow_used="graph_only",
        )
        d = result.to_dict()
        assert d["is_simple_query"] is True
        assert d["simple_flow_used"] == "graph_only"
        assert d["total_steps"] == 0

    def test_complex_query_result(self):
        """Test AgentResult for a complex query with trace."""
        steps = [
            AgentStep(1, "Find entity", "keyword_extract",
                      {"query": "X"}, "Found: X"),
            AgentStep(2, "Look up ID", "semantic_id_lookup",
                      {"keywords": ["X"]}, "Found ID: 1:X"),
            AgentStep(3, "I have enough info.", "FINAL_ANSWER",
                      {"answer": "X is a person."}, ""),
        ]
        result = AgentResult(answer="X is a person.", trace=steps)
        d = result.to_dict()
        assert d["answer"] == "X is a person."
        assert d["total_steps"] == 3
        assert len(d["trace"]) == 3


class TestReActAgent:
    """Tests for the ReAct agent loop."""

    def _make_mock_llm(self, responses):
        """Create a mock LLM that returns a sequence of responses."""
        llm = MagicMock()
        llm.generate.side_effect = responses
        return llm

    def _make_basic_registry(self):
        """Create a ToolRegistry with basic tools for testing."""
        registry = ToolRegistry()

        registry.register(
            Tool(
                name="keyword_extract",
                description="Extract keywords from query.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda **kw: {"keywords": [kw["query"]]},
            )
        )
        registry.register(
            Tool(
                name="answer_synthesize",
                description="Generate final answer.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["query", "context"],
                },
                handler=lambda **kw: {"answer": f"Answer to: {kw['query']}"},
            )
        )
        return registry

    def test_parse_react_response_with_action(self):
        """Test parsing a standard ReAct response with an action."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        response = (
            "Thought: I need to extract keywords first.\n"
            "Action: keyword_extract\n"
            "Action Input: {\"query\": \"Who is Sarah?\"}"
        )

        step = agent._parse_react_response(response, 1)
        assert step.step_num == 1
        assert "extract keywords" in step.thought.lower()
        assert step.action == "keyword_extract"
        assert step.action_input == {"query": "Who is Sarah?"}

    def test_parse_react_response_with_final_answer(self):
        """Test parsing a response with Final Answer."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        response = (
            "Thought: I now have enough information.\n"
            "Final Answer: Sarah is a 30-year-old software engineer."
        )

        step = agent._parse_react_response(response, 3)
        assert step.action == "FINAL_ANSWER"
        assert "Sarah" in step.action_input["answer"]

    def test_parse_react_response_invalid_json_input(self):
        """Test parsing with invalid JSON in Action Input."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        response = (
            "Thought: Let me search.\n"
            "Action: vector_search\n"
            "Action Input: query = test query text"
        )

        step = agent._parse_react_response(response, 1)
        assert step.action == "vector_search"
        assert "raw_input" in step.action_input

    def test_execute_tool_success(self):
        """Test executing a tool successfully."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        step = AgentStep(
            step_num=1,
            thought="Extract keywords.",
            action="keyword_extract",
            action_input={"query": "test"},
        )

        observation = agent._execute_tool(step)
        assert "test" in observation

    def test_execute_tool_unknown(self):
        """Test executing an unknown tool."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        step = AgentStep(
            step_num=1,
            thought="Try unknown tool.",
            action="nonexistent_tool",
            action_input={},
        )

        observation = agent._execute_tool(step)
        assert "Unknown tool" in observation

    def test_build_initial_messages(self):
        """Test that initial messages contain tool descriptions and query."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        messages = agent._build_initial_messages("Who is Sarah?")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "keyword_extract" in messages[0]["content"]
        assert "answer_synthesize" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Who is Sarah?"

    def test_format_tool_descriptions(self):
        """Test that tool descriptions are formatted correctly."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        desc = agent._format_tool_descriptions()
        assert "keyword_extract" in desc
        assert "answer_synthesize" in desc

    def test_run_simple_query(self):
        """Test that simple queries are routed without running the agent loop."""
        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=MagicMock(),
        )

        result = agent.run("Who is Sarah?")
        assert result.is_simple_query is True
        assert result.simple_flow_used == "graph_only"
        assert len(result.trace) == 0

    def test_run_complex_query(self):
        """Test a multi-step ReAct execution for a complex query."""
        llm = self._make_mock_llm([
            # Step 1: Extract keywords
            "Thought: I need to find keywords.\n"
            "Action: keyword_extract\n"
            'Action Input: {"query": "Compare X and Y"}',
            # Step 2: Final answer
            "Thought: I have extracted the keywords. Now I can answer.\n"
            "Final Answer: X and Y are related entities.",
        ])

        agent = ReActAgent(
            tool_registry=self._make_basic_registry(),
            llm=llm,
        )

        result = agent.run("Compare the entities X and Y")
        assert len(result.trace) == 2
        assert "X and Y" in result.answer


class TestCreateReactAgent:
    """Tests for the create_react_agent factory."""

    def test_creates_agent(self):
        """Test that the factory creates a properly configured agent."""
        registry = ToolRegistry()
        llm = MagicMock()

        agent = create_react_agent(
            tool_registry=registry,
            llm=llm,
            max_steps=5,
            verbose=True,
        )

        assert isinstance(agent, ReActAgent)
        assert agent.max_steps == 5
        assert agent.verbose is True
