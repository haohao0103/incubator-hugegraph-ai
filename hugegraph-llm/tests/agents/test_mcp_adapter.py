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

"""Tests for the MCP (Model Context Protocol) Adapter."""

from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.agents.mcp_adapter import (
    MCPAdapter,
    MCPServerConfig,
    MCPToolInfo,
    create_mcp_adapter,
)
from hugegraph_llm.agents.tool_registry import ToolRegistry


class TestMCPServerConfig:
    """Tests for MCP server configuration."""

    def test_default_transport(self):
        """Test that default transport is 'sse'."""
        config = MCPServerConfig(name="test-server")
        assert config.transport == "sse"
        assert config.name == "test-server"

    def test_stdio_config(self):
        """Test stdio transport configuration."""
        config = MCPServerConfig(
            name="stdio-server",
            transport="stdio",
            command="python",
            args=["-m", "my_mcp_server"],
        )
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["-m", "my_mcp_server"]

    def test_sse_config(self):
        """Test SSE transport configuration."""
        config = MCPServerConfig(
            name="sse-server",
            transport="sse",
            url="http://localhost:9000/sse",
        )
        assert config.transport == "sse"
        assert config.url == "http://localhost:9000/sse"


class TestMCPToolInfo:
    """Tests for MCP tool info dataclass."""

    def test_tool_info_creation(self):
        """Test basic MCPToolInfo creation."""
        info = MCPToolInfo(
            name="external_search",
            description="Search external data.",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            server_name="ext-server",
        )
        assert info.name == "external_search"
        assert info.server_name == "ext-server"


class TestMCPAdapter:
    """Tests for the MCP Adapter."""

    def test_add_and_remove_server(self):
        """Test adding and removing MCP server configs."""
        adapter = MCPAdapter()
        server = MCPServerConfig(name="test", url="http://localhost/sse")

        adapter.add_server(server)
        assert len(adapter.get_servers()) == 1

        adapter.remove_server("test")
        assert len(adapter.get_servers()) == 0

    def test_add_multiple_servers(self):
        """Test adding multiple MCP servers."""
        adapter = MCPAdapter()
        adapter.add_server(MCPServerConfig(name="s1", url="http://localhost/sse"))
        adapter.add_server(MCPServerConfig(name="s2", url="http://other/sse"))

        servers = adapter.get_servers()
        assert len(servers) == 2
        names = [s.name for s in servers]
        assert "s1" in names
        assert "s2" in names

    @pytest.mark.asyncio
    async def test_discover_tools_sse(self):
        """Test discovering tools from an SSE MCP server."""
        adapter = MCPAdapter()
        adapter.add_server(
            MCPServerConfig(name="test-server", url="http://localhost:9000/sse")
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "tools": [
                    {
                        "name": "weather",
                        "description": "Get weather data.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"],
                        },
                    },
                    {
                        "name": "search",
                        "description": "Search the web.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = (
                mock_response
            )
            tools = await adapter.discover_tools()

        assert len(tools) == 2
        assert tools[0].name == "weather"
        assert tools[0].server_name == "test-server"
        assert tools[1].name == "search"

    @pytest.mark.asyncio
    async def test_discover_tools_error_handling(self):
        """Test that tool discovery errors are handled gracefully."""
        adapter = MCPAdapter()
        adapter.add_server(
            MCPServerConfig(name="broken", url="http://invalid/sse")
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = (
                ConnectionError("Connection refused")
            )
            tools = await adapter.discover_tools()

        assert tools == []  # Error handled, empty result

    def test_register_with_tool_registry(self):
        """Test registering discovered tools with ToolRegistry."""
        adapter = MCPAdapter()
        # Manually set discovered tools (simulating discovery)
        adapter._discovered_tools = [
            MCPToolInfo(
                name="tool1",
                description="External tool 1.",
                input_schema={"type": "object", "properties": {}},
                server_name="srv",
            ),
            MCPToolInfo(
                name="tool2",
                description="External tool 2.",
                input_schema={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
                server_name="srv",
            ),
        ]

        registry = ToolRegistry()
        count = adapter.register_with(registry)

        assert count == 2
        names = registry.get_tool_names()
        assert "mcp:srv:tool1" in names
        assert "mcp:srv:tool2" in names

        # Verify the MCP prefix naming convention
        for name in names:
            assert name.startswith("mcp:")

    def test_unsupported_transport(self):
        """Test that unsupported transport raises ValueError."""
        adapter = MCPAdapter()
        import asyncio

        async def test():
            return await adapter._list_tools_from_server(
                MCPServerConfig(name="x", transport="unknown")
            )

        with pytest.raises(ValueError, match="Unsupported"):
            asyncio.run(test())


class TestCreateMCPAdapter:
    """Tests for the create_mcp_adapter factory."""

    def test_creates_empty_adapter(self):
        """Test creating an adapter with no servers."""
        adapter = create_mcp_adapter()
        assert isinstance(adapter, MCPAdapter)
        assert len(adapter.get_servers()) == 0

    def test_creates_adapter_with_servers(self):
        """Test creating an adapter with pre-configured servers."""
        configs = [
            MCPServerConfig(name="s1", url="http://a/sse"),
            MCPServerConfig(name="s2", url="http://b/sse"),
        ]
        adapter = create_mcp_adapter(server_configs=configs)
        assert len(adapter.get_servers()) == 2
        names = [s.name for s in adapter.get_servers()]
        assert "s1" in names
        assert "s2" in names
