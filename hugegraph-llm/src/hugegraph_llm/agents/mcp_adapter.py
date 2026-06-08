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

"""MCP (Model Context Protocol) adapter for external tool integration.

This module provides an adapter that connects external MCP-compatible
servers to the HugeGraph agent ToolRegistry, allowing the ReAct agent
to use tools hosted by MCP servers alongside built-in tools.

Reference: https://modelcontextprotocol.io/
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.agents.tool_registry import Tool, ToolRegistry
from hugegraph_llm.utils.log import log


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server.

    MCP servers can use different transports:
    - stdio: Launch as a subprocess
    - sse: Connect via Server-Sent Events (HTTP)

    Attributes:
        name: Human-readable name for this MCP server.
        transport: Transport protocol ("stdio" or "sse").
        command: For stdio transport, the command to run (e.g., "python").
        args: Arguments for the command.
        url: For SSE transport, the server URL.
    """

    name: str
    transport: str = "sse"  # "stdio" or "sse"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None


@dataclass
class MCPToolInfo:
    """Information about a tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


class MCPAdapter:
    """Adapter for connecting MCP-compatible external tools.

    Discovers tools from MCP servers and converts them to the
    internal Tool format so they can be used by the ReAct agent.

    Usage:
        adapter = MCPAdapter()
        adapter.add_server(MCPServerConfig(
            name="my-server",
            transport="sse",
            url="http://localhost:9000/sse",
        ))
        # Discover tools from all connected servers
        tools = await adapter.discover_tools()
        # Register discovered tools in the ToolRegistry
        adapter.register_with(tool_registry)
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._discovered_tools: List[MCPToolInfo] = []

    # ── Server Management ──────────────────────────────────────

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration.

        The connection is lazily established when discover_tools()
        or call_tool() is invoked.
        """
        self._servers[config.name] = config
        log.info("Added MCP server: %s (transport=%s)", config.name, config.transport)

    def remove_server(self, name: str) -> None:
        """Remove an MCP server configuration."""
        self._servers.pop(name, None)
        log.info("Removed MCP server: %s", name)

    def get_servers(self) -> List[MCPServerConfig]:
        """Return all configured MCP servers."""
        return list(self._servers.values())

    # ── Tool Discovery ─────────────────────────────────────────

    async def discover_tools(self) -> List[MCPToolInfo]:
        """Discover tools from all connected MCP servers.

        For each server, calls the MCP tools/list endpoint (or equivalent)
        to retrieve available tool definitions.

        Returns:
            List of MCPToolInfo objects describing available external tools.
        """
        self._discovered_tools = []

        for server_name, config in self._servers.items():
            try:
                tools = await self._list_tools_from_server(config)
                for tool_def in tools:
                    self._discovered_tools.append(
                        MCPToolInfo(
                            name=tool_def.get("name", "unknown"),
                            description=tool_def.get("description", ""),
                            input_schema=tool_def.get("inputSchema", {}),
                            server_name=server_name,
                        )
                    )
                log.info(
                    "Discovered %d tools from MCP server %s",
                    len(tools),
                    server_name,
                )
            except Exception as e:
                log.error(
                    "Failed to discover tools from MCP server %s: %s",
                    server_name,
                    str(e),
                )

        return self._discovered_tools

    async def _list_tools_from_server(self, config: MCPServerConfig) -> List[Dict]:
        """List tools from a single MCP server based on its transport.

        This implements a minimal MCP client for the tools/list method.
        For production use, consider using the official mcp Python SDK.
        """
        if config.transport == "sse" and config.url:
            return await self._list_tools_sse(config)
        elif config.transport == "stdio":
            return await self._list_tools_stdio(config)
        else:
            raise ValueError(f"Unsupported MCP transport: {config.transport}")

    async def _list_tools_sse(self, config: MCPServerConfig) -> List[Dict]:
        """List tools from an SSE-based MCP server.

        Uses a simple HTTP POST to the /tools/list endpoint as a
        lightweight alternative to full SSE protocol.
        """
        import httpx

        base_url = config.url.rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Send MCP JSON-RPC request for tools/list
            response = await client.post(
                f"{base_url}/tools/list",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("result", {}).get("tools", [])

    async def _list_tools_stdio(self, config: MCPServerConfig) -> List[Dict]:
        """List tools from a stdio-based MCP server.

        Launches the command as a subprocess and communicates
        via stdin/stdout JSON-RPC.
        """
        import asyncio

        if not config.command:
            raise ValueError("stdio transport requires a command")

        proc = await asyncio.create_subprocess_exec(
            config.command,
            *(config.args or []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
            request_bytes = (json.dumps(request) + "\n").encode()

            stdout, stderr = await asyncio.wait_for(
                asyncio.gather(
                    self._stdio_communicate(proc, request_bytes),
                    proc.stderr.read(),
                ),
                timeout=30.0,
            )

            if stderr:
                log.warning("MCP stdio stderr: %s", stderr.decode())

            response = json.loads(stdout.decode())
            return response.get("result", {}).get("tools", [])

        finally:
            if proc.returncode is None:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)

    @staticmethod
    async def _stdio_communicate(proc, request_bytes: bytes) -> bytes:
        """Write request to stdin and read response from stdout."""
        proc.stdin.write(request_bytes)
        await proc.stdin.drain()
        line = await proc.stdout.readline()
        return line

    # ── Tool Call ──────────────────────────────────────────────

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on a specific MCP server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Tool parameters.

        Returns:
            Tool execution result dict.

        Raises:
            ValueError: If the server is not configured.
        """
        if server_name not in self._servers:
            raise ValueError(f"MCP server not configured: {server_name}")

        config = self._servers[server_name]

        if config.transport == "sse" and config.url:
            return await self._call_tool_sse(config, tool_name, arguments)
        elif config.transport == "stdio":
            return await self._call_tool_stdio(config, tool_name, arguments)
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    async def _call_tool_sse(
        self, config: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via SSE transport."""
        import httpx

        base_url = config.url.rstrip("/")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/tools/call",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("result", {})

    async def _call_tool_stdio(
        self, config: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via stdio transport."""
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            config.command,
            *(config.args or []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        try:
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            request_bytes = (json.dumps(request) + "\n").encode()

            stdout, _ = await asyncio.wait_for(
                asyncio.gather(
                    self._stdio_communicate(proc, request_bytes),
                    proc.stderr.read(),
                ),
                timeout=60.0,
            )

            response = json.loads(stdout.decode())
            return response.get("result", {})

        finally:
            if proc.returncode is None:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)

    # ── Registration with ToolRegistry ──────────────────────────

    def register_with(self, tool_registry: ToolRegistry) -> int:
        """Register all discovered MCP tools with the given ToolRegistry.

        Each MCP tool gets an "mcp:" prefix to distinguish it from
        built-in tools.

        Args:
            tool_registry: The ToolRegistry to register with.

        Returns:
            Number of tools successfully registered.
        """
        count = 0
        for tool_info in self._discovered_tools:
            # Skip tools that conflict with built-in names
            prefixed_name = f"mcp:{tool_info.server_name}:{tool_info.name}"

            mcp_tool = Tool(
                name=prefixed_name,
                description=f"[MCP:{tool_info.server_name}] {tool_info.description}",
                parameters=tool_info.input_schema,
                handler=self._make_mcp_tool_handler(
                    tool_info.server_name, tool_info.name
                ),
            )
            tool_registry.register(mcp_tool)
            count += 1

        log.info("Registered %d MCP tools in ToolRegistry", count)
        return count

    def _make_mcp_tool_handler(self, server_name: str, tool_name: str):
        """Create a handler that delegates to the MCP server."""

        async def handler(**kwargs) -> Dict[str, Any]:
            result = await self.call_tool(server_name, tool_name, kwargs)
            return result

        # Return a sync wrapper since Tool handlers are synchronous
        def sync_handler(**kwargs) -> Dict[str, Any]:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # We're in an async context, use run_coroutine_threadsafe
                # but that's complex; for now, create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(
                            self.call_tool(server_name, tool_name, kwargs)
                        )
                    )
                    return future.result(timeout=60)
            else:
                return asyncio.run(
                    self.call_tool(server_name, tool_name, kwargs)
                )

        return sync_handler


# ── Convenience Factory ────────────────────────────────────────


def create_mcp_adapter(
    server_configs: Optional[List[MCPServerConfig]] = None,
) -> MCPAdapter:
    """Create an MCP adapter with optional server configurations.

    Args:
        server_configs: Optional list of MCP server configs to add.

    Returns:
        Configured MCPAdapter instance.
    """
    adapter = MCPAdapter()
    if server_configs:
        for config in server_configs:
            adapter.add_server(config)
    return adapter
