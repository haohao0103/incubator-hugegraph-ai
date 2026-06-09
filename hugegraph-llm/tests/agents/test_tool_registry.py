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

"""Tests for the Agent Tool Registry."""

import pytest

from hugegraph_llm.agents.tool_registry import (
    Tool,
    ToolRegistry,
    create_default_tool_registry,
)


class TestTool:
    """Unit tests for the Tool dataclass."""

    def test_tool_creation(self):
        """Test basic Tool creation with required fields."""
        handler = lambda **kw: {"result": "ok"}
        tool = Tool(
            name="test_tool",
            description="A test tool for unit testing.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query text."}
                },
                "required": ["query"],
            },
            handler=handler,
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool for unit testing."
        assert tool.requires_hugegraph is False
        assert tool.requires_vector_index is False

    def test_get_openai_function_definition(self):
        """Test that tool definitions match OpenAI function-calling format."""
        tool = Tool(
            name="search",
            description="Search for information.",
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query."}
                },
                "required": ["q"],
            },
            handler=lambda **kw: {"results": []},
        )

        definition = tool.get_openai_function_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "search"
        assert definition["function"]["description"] == "Search for information."
        assert "q" in definition["function"]["parameters"]["properties"]
        assert "q" in definition["function"]["parameters"]["required"]

    def test_execute_success(self):
        """Test successful tool execution."""
        handler = lambda **kw: {"data": kw.get("x", 0) * 2}
        tool = Tool(
            name="double",
            description="Doubles the input.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=handler,
        )

        result = tool.execute(x=5)
        assert result["success"] is True
        assert result["data"]["data"] == 10

    def test_execute_failure(self):
        """Test that tool execution failures are captured."""
        def failing_handler(**kw):
            raise ValueError("Something went wrong")

        tool = Tool(
            name="failing",
            description="Always fails.",
            parameters={"type": "object", "properties": {}},
            handler=failing_handler,
        )

        result = tool.execute()
        assert result["success"] is False
        assert "Something went wrong" in result["error"]

    def test_tool_with_hugegraph_and_vector_flags(self):
        """Test that dependency flags are correctly stored."""
        tool = Tool(
            name="graph_tool",
            description="A tool requiring HugeGraph.",
            parameters={"type": "object", "properties": {}},
            handler=lambda **kw: {},
            requires_hugegraph=True,
            requires_vector_index=True,
        )
        assert tool.requires_hugegraph is True
        assert tool.requires_vector_index is True


class TestToolRegistry:
    """Unit tests for the ToolRegistry."""

    def test_register_and_retrieve(self):
        """Test registering a tool and retrieving it."""
        registry = ToolRegistry()
        tool = Tool(
            name="echo",
            description="Echoes input.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda **kw: {"echo": kw.get("text", "")},
        )

        registry.register(tool)
        assert "echo" in registry.get_tool_names()

    def test_register_overwrite_warns(self):
        """Test that registering the same name overwrites the previous tool."""
        registry = ToolRegistry()
        tool1 = Tool(
            name="same",
            description="First version.",
            parameters={"type": "object", "properties": {}},
            handler=lambda **kw: {"v": 1},
        )
        tool2 = Tool(
            name="same",
            description="Second version.",
            parameters={"type": "object", "properties": {}},
            handler=lambda **kw: {"v": 2},
        )

        registry.register(tool1)
        registry.register(tool2)
        assert len(registry.get_tool_names()) == 1

    def test_unregister(self):
        """Test removing a tool from the registry."""
        registry = ToolRegistry()
        tool = Tool(
            name="temp",
            description="Temporary tool.",
            parameters={"type": "object", "properties": {}},
            handler=lambda **kw: {},
        )

        registry.register(tool)
        assert "temp" in registry.get_tool_names()

        registry.unregister("temp")
        assert "temp" not in registry.get_tool_names()

    def test_get_tool_definitions(self):
        """Test that get_tool_definitions returns correct OpenAI format."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="tool_a",
                description="Tool A description.",
                parameters={"type": "object", "properties": {}},
                handler=lambda **kw: {},
            )
        )
        registry.register(
            Tool(
                name="tool_b",
                description="Tool B description.",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
                handler=lambda **kw: {},
            )
        )

        definitions = registry.get_tool_definitions()
        assert len(definitions) == 2
        assert all(d["type"] == "function" for d in definitions)
        names = [d["function"]["name"] for d in definitions]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_get_tool_definitions_filtered(self):
        """Test filtering tool definitions by name."""
        registry = ToolRegistry()
        for name in ["a", "b", "c"]:
            registry.register(
                Tool(
                    name=name,
                    description=f"Tool {name}.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda **kw: {},
                )
            )

        filtered = registry.get_tool_definitions(tool_names=["a", "c"])
        assert len(filtered) == 2
        names = [d["function"]["name"] for d in filtered]
        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_execute_known_tool(self):
        """Test executing a registered tool."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="add",
                description="Adds two numbers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
                handler=lambda **kw: {"sum": kw["a"] + kw["b"]},
            )
        )

        result = registry.execute("add", a=3, b=4)
        assert result["success"] is True
        assert result["data"]["sum"] == 7

    def test_execute_unknown_tool_raises(self):
        """Test that executing an unregistered tool raises ValueError."""
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            registry.execute("nonexistent")

    def test_get_tool_names(self):
        """Test that get_tool_names returns all registered names."""
        registry = ToolRegistry()
        for name in ["x", "y", "z"]:
            registry.register(
                Tool(
                    name=name,
                    description=f"Tool {name}.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda **kw: {},
                )
            )

        names = registry.get_tool_names()
        assert sorted(names) == ["x", "y", "z"]


class TestCreateDefaultToolRegistry:
    """Tests for the convenience factory function."""

    def test_creates_registry_with_no_args(self):
        """Test that create_default_tool_registry works without any arguments."""
        registry = create_default_tool_registry()
        assert isinstance(registry, ToolRegistry)
        # Without LLM/embedding/client, tools are still registered but may
        # return errors when executed due to missing dependencies
        assert len(registry.get_tool_names()) == 7

    def test_default_tools_registered(self):
        """Test that all 7 default tools are registered."""
        registry = create_default_tool_registry()
        expected_tools = [
            "keyword_extract",
            "vector_search",
            "graph_traverse",
            "text2gremlin",
            "semantic_id_lookup",
            "answer_synthesize",
            "schema_lookup",
        ]
        names = registry.get_tool_names()
        for expected in expected_tools:
            assert expected in names, f"Missing tool: {expected}"

    def test_tool_definitions_are_valid_openai_format(self):
        """Test that all default tool definitions are valid OpenAI format."""
        registry = create_default_tool_registry()
        definitions = registry.get_tool_definitions()

        for defn in definitions:
            assert defn["type"] == "function"
            func = defn["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]
