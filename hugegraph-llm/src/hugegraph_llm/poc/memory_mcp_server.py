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

"""
HugeGraph-AI-Memory MCP Server
==============================

Exposes the memory pipeline backend as MCP tools:
    - add_memory
    - search_memory
    - forget_memory
    - get_persona
    - update_persona

Usage:
    # stdio (for Claude Desktop / Cursor / Windsurf)
    python memory_mcp_server.py

    # SSE (for web clients, PowerMem-style)
    python memory_mcp_server.py --transport sse --port 8848

Configuration is loaded from `.env` via hugegraph_llm.config.memory_config.
"""

import argparse
import json
import os
import sys
from typing import Optional

# Ensure src is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.poc.memory_backend import MemoryPipelineBackend, HugeGraphMemoryClient
from hugegraph_llm.utils.log import log


def _create_backend() -> MemoryPipelineBackend:
    """Lazily create and initialize the memory backend."""
    hg_client = HugeGraphMemoryClient(
        url=memory_settings.hugegraph_url,
        user=memory_settings.hugegraph_user,
        pwd=memory_settings.hugegraph_pwd,
        graph=memory_settings.hugegraph_graph,
    )
    hg_client.init_schema()

    backend = MemoryPipelineBackend(hg_client=hg_client)
    # Override LLM settings loaded from env
    backend.llm_base_url = memory_settings.llm_base_url
    backend.llm_model = memory_settings.llm_model
    backend.llm_api_key = memory_settings.llm_api_key or ""
    if not backend.llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")
    return backend


_BACKEND: Optional[MemoryPipelineBackend] = None


def _get_backend() -> MemoryPipelineBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _create_backend()
    return _BACKEND


def _to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# MCP Server (FastMCP)
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "mcp Python SDK is required. Install: pip install mcp"
    ) from exc

mcp = FastMCP(memory_settings.mcp_server_name)


@mcp.tool(description="Add a memory for a user/agent. Auto-extracts entities and relations, writes to HugeGraph.")
def add_memory(content: str, user_id: str = "demo_user") -> str:
    """
    Add a new memory. If the input looks like a question, it is automatically
    routed to search_memory instead.

    Args:
        content: The memory text to store.
        user_id: Scope identifier (user or agent id).
    """
    try:
        backend = _get_backend()
        result = backend.add_memory(content=content, user_id=user_id)
        return _to_json(result)
    except Exception as e:
        log.error("add_memory failed: %s", e)
        return _to_json({"error": str(e), "tool": "add_memory"})


@mcp.tool(description="Search memories for a user/agent using vector + BM25 + graph RRF fusion.")
def search_memory(query: str, user_id: str = "demo_user", top_k: int = 5) -> str:
    """
    Search memories through the 3-channel RRF pipeline.

    Args:
        query: The query text.
        user_id: Scope identifier (user or agent id).
        top_k: Number of top results to return.
    """
    try:
        backend = _get_backend()
        result = backend.search_memory(query=query, user_id=user_id, top_k=top_k)
        return _to_json(result)
    except Exception as e:
        log.error("search_memory failed: %s", e)
        return _to_json({"error": str(e), "tool": "search_memory"})


@mcp.tool(description="Delete all memories for a user/agent from SQLite, FAISS and BM25 indexes. Graph vertices/edges are retained for provenance.")
def forget_memory(user_id: str = "demo_user") -> str:
    """
    Forget all vector/fulltext memories for a scope. The graph structure in
    HugeGraph is kept for historical provenance.

    Args:
        user_id: Scope identifier (user or agent id).
    """
    try:
        backend = _get_backend()
        backend.forget_user(user_id=user_id)
        return _to_json({"status": "ok", "action": "forgot_user", "user_id": user_id})
    except Exception as e:
        log.error("forget_memory failed: %s", e)
        return _to_json({"error": str(e), "tool": "forget_memory"})


@mcp.tool(description="Get the persona summary for a user/agent.")
def get_persona(user_id: str = "demo_user") -> str:
    """
    Retrieve the L3 persona / user profile for a scope.

    Args:
        user_id: Scope identifier (user or agent id).
    """
    try:
        backend = _get_backend()
        persona = backend.get_persona(user_id=user_id)
        return _to_json(persona)
    except Exception as e:
        log.error("get_persona failed: %s", e)
        return _to_json({"error": str(e), "tool": "get_persona"})


@mcp.tool(description="Update the persona summary for a user/agent.")
def update_persona(user_id: str = "demo_user", summary: str = "") -> str:
    """
    Update the L3 persona / user profile for a scope.

    Args:
        user_id: Scope identifier (user or agent id).
        summary: New persona summary text.
    """
    try:
        backend = _get_backend()
        backend.update_persona(user_id=user_id, summary=summary)
        return _to_json({"status": "ok", "action": "updated_persona", "user_id": user_id})
    except Exception as e:
        log.error("update_persona failed: %s", e)
        return _to_json({"error": str(e), "tool": "update_persona"})


@mcp.tool(description="Distill atomic memories into Experience + Skill layers for a user/agent.")
def distill_memories(user_id: str = "demo_user", threshold: int = 5) -> str:
    """
    Run Experience + Skill distillation on all memories of a scope.

    Args:
        user_id: Scope identifier (user or agent id).
        threshold: Minimum number of memories required to trigger distillation.
    """
    try:
        backend = _get_backend()
        result = backend.distill_user_memories(user_id=user_id, threshold=threshold)
        return _to_json(result)
    except Exception as e:
        log.error("distill_memories failed: %s", e)
        return _to_json({"error": str(e), "tool": "distill_memories"})


@mcp.tool(description="Get distilled experiences for a user/agent.")
def get_experiences(query: str = "", user_id: str = "demo_user", top_k: int = 5) -> str:
    """
    Retrieve Experience layer entries for a scope.

    Args:
        query: Optional filter query.
        user_id: Scope identifier.
        top_k: Number of results.
    """
    try:
        backend = _get_backend()
        return _to_json(backend.get_experiences(query=query, user_id=user_id, top_k=top_k))
    except Exception as e:
        log.error("get_experiences failed: %s", e)
        return _to_json({"error": str(e), "tool": "get_experiences"})


@mcp.tool(description="Get distilled skills for a user/agent.")
def get_skills(query: str = "", user_id: str = "demo_user", top_k: int = 5) -> str:
    """
    Retrieve Skill layer entries for a scope.

    Args:
        query: Optional filter query.
        user_id: Scope identifier.
        top_k: Number of results.
    """
    try:
        backend = _get_backend()
        return _to_json(backend.get_skills(query=query, user_id=user_id, top_k=top_k))
    except Exception as e:
        log.error("get_skills failed: %s", e)
        return _to_json({"error": str(e), "tool": "get_skills"})


@mcp.tool(description="Get a memory by id.")
def get_memory_by_id(memory_id: str, user_id: str = "demo_user") -> str:
    """Get a single memory by id."""
    try:
        backend = _get_backend()
        result = backend.get_memory_by_id(memory_id)
        if result is None:
            return _to_json({"error": "NOT_FOUND", "memory_id": memory_id})
        return _to_json(result)
    except Exception as e:
        log.error("get_memory_by_id failed: %s", e)
        return _to_json({"error": str(e), "tool": "get_memory_by_id"})


@mcp.tool(description="List all memories for a user/agent.")
def list_memories(user_id: str = "demo_user") -> str:
    """List memories for a scope."""
    try:
        backend = _get_backend()
        return _to_json(backend.list_memories(user_id=user_id))
    except Exception as e:
        log.error("list_memories failed: %s", e)
        return _to_json({"error": str(e), "tool": "list_memories"})


@mcp.tool(description="Update a memory by id.")
def update_memory(memory_id: str, content: str, user_id: str = "demo_user") -> str:
    """Update a memory's content."""
    try:
        backend = _get_backend()
        return _to_json(backend.update_memory(memory_id=memory_id, content=content, user_id=user_id))
    except Exception as e:
        log.error("update_memory failed: %s", e)
        return _to_json({"error": str(e), "tool": "update_memory"})


@mcp.tool(description="Delete a memory by id.")
def delete_memory(memory_id: str, user_id: str = "demo_user") -> str:
    """Delete a memory by id."""
    try:
        backend = _get_backend()
        return _to_json(backend.delete_memory(memory_id=memory_id, user_id=user_id))
    except Exception as e:
        log.error("delete_memory failed: %s", e)
        return _to_json({"error": str(e), "tool": "delete_memory"})


@mcp.tool(description="Add a procedural skill for a user/agent.")
def add_skill(content: str, user_id: str = "demo_user") -> str:
    """Add a skill/procedural memory."""
    try:
        backend = _get_backend()
        return _to_json(backend.add_skill(content=content, user_id=user_id))
    except Exception as e:
        log.error("add_skill failed: %s", e)
        return _to_json({"error": str(e), "tool": "add_skill"})


@mcp.tool(description="Search skills for a user/agent.")
def search_skills(query: str, user_id: str = "demo_user", top_k: int = 5) -> str:
    """Search skills for a scope."""
    try:
        backend = _get_backend()
        return _to_json(backend.search_skills(query=query, user_id=user_id, top_k=top_k))
    except Exception as e:
        log.error("search_skills failed: %s", e)
        return _to_json({"error": str(e), "tool": "search_skills"})


@mcp.tool(description="Get the user profile / persona for a user/agent.")
def get_user_profile(user_id: str = "demo_user") -> str:
    """Get the L3 persona / user profile."""
    try:
        backend = _get_backend()
        return _to_json(backend.get_user_profile(user_id=user_id))
    except Exception as e:
        log.error("get_user_profile failed: %s", e)
        return _to_json({"error": str(e), "tool": "get_user_profile"})


@mcp.tool(description="Update the user profile / persona for a user/agent.")
def update_user_profile(user_id: str = "demo_user", summary: str = "") -> str:
    """Update the L3 persona / user profile."""
    try:
        backend = _get_backend()
        return _to_json(backend.update_user_profile(user_id=user_id, summary=summary))
    except Exception as e:
        log.error("update_user_profile failed: %s", e)
        return _to_json({"error": str(e), "tool": "update_user_profile"})


def main():
    parser = argparse.ArgumentParser(description="HugeGraph-AI-Memory MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="MCP transport type (stdio for Claude Desktop, sse for web clients).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=memory_settings.mcp_server_port,
        help="Port for SSE transport.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport.",
    )
    args = parser.parse_args()

    log.info(
        "Starting HugeGraph-AI-Memory MCP Server (transport=%s, graph=%s)",
        args.transport,
        memory_settings.hugegraph_graph,
    )

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
