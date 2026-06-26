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
Handler functions that wire the Agent, Community Detection, and Global Search
flows into the FastAPI endpoint layer.

Each function follows the same pattern:
1. Initialize/cache dependencies (LLM, ToolRegistry, ProvenanceManager, etc.)
2. Delegate to the Scheduler via the correct FlowName
3. Return results in API-compatible dict format
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.utils.log import log

# ── Cached dependencies ──────────────────────────────────────

_cached_tool_registry = None
_cached_agent_llm = None


def _get_or_create_dependencies():
    """Lazily initialize the ToolRegistry and Agent LLM.

    This avoids creating heavy objects at import time and
    ensures they're created only when actually needed.
    """
    global _cached_tool_registry, _cached_agent_llm

    if _cached_tool_registry is None:
        from hugegraph_llm.agents.tool_registry import create_default_tool_registry
        from hugegraph_llm.models.llms.init_llm import LLMs

        try:
            llms = LLMs()
            agent_llm = llms.get_agent_llm()
            embedding = None  # Will be resolved when tools are used
            from hugegraph_llm.models.embeddings.init_embedding import Embeddings
            embedding = Embeddings().get_embedding()

            _cached_agent_llm = agent_llm
            _cached_tool_registry = create_default_tool_registry(
                llm=agent_llm,
                embedding=embedding,
                # client will be resolved per-request via scheduler flows
            )

            log.info("Agent dependencies initialized: %d tools", len(_cached_tool_registry.get_tool_names()))
        except Exception as e:
            log.warning("Failed to initialize agent deps: %s. /agent endpoint will not be available.", e)
            _cached_tool_registry = None
            _cached_agent_llm = None

    return _cached_tool_registry, _cached_agent_llm


# ── Agent handler ────────────────────────────────────────────

def agent_answer(
    query: str,
    max_steps: int = 10,
    tools_filter: Optional[List[str]] = None,
    stream: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute agent-based multi-step reasoning for a query.

    Routes simple queries to the fast graph-only RAG flow,
    complex queries to the ReAct agent loop.

    Args:
        query: Natural language query.
        max_steps: Max ReAct reasoning steps.
        tools_filter: Optional list of tool names to enable.
        stream: Enable streaming (not yet wired).
        verbose: Enable detailed logging.

    Returns:
        Dict with answer, trace, and metadata.
    """
    tool_registry, agent_llm = _get_or_create_dependencies()

    if tool_registry is None or agent_llm is None:
        return {
            "answer": "Agent is not available. Please check LLM configuration.",
            "trace": [],
            "total_steps": 0,
            "status_code": 503,
        }

    scheduler = SchedulerSingleton.get_instance()

    try:
        result = scheduler.agentic_flow(
            tool_registry=tool_registry,
            llm=agent_llm,
            query=query,
            max_steps=max_steps,
            stream=stream,
        )
        log.info("Agent completed for query: '%s'", query[:80])
        return result
    except Exception as e:
        log.error("Agent execution failed: %s", e)
        return {
            "answer": f"Agent execution failed: {str(e)}",
            "trace": [],
            "total_steps": 0,
            "status_code": 500,
            "error": str(e),
        }


# ── Community build handler ──────────────────────────────────

def community_build(
    graph_name: str = "",
    algorithm: str = "louvain",
    max_levels: int = 2,
) -> Dict[str, Any]:
    """Build community detection index for the knowledge graph.

    Triggers: Community detection → LLM report generation → Vector index

    Args:
        graph_name: Graph to analyze (uses config default if empty).
        algorithm: "louvain", "wcc", or "label_propagation".
        max_levels: Hierarchical levels for community decomposition.

    Returns:
        Dict with community_count, report_count, index_built status.
    """
    scheduler = SchedulerSingleton.get_instance()

    try:
        from hugegraph_llm.config import huge_settings

        # Configure the community flow before running
        flow_entry = scheduler.pipeline_pool.get(FlowName.COMMUNITY_DETECT, {})
        if "flow" in flow_entry:
            flow = flow_entry["flow"]
            from hugegraph_llm.models.llms.init_llm import LLMs
            from hugegraph_llm.utils.vector_index_utils import get_vector_index_class
            from hugegraph_llm.config.index_config import IndexConfig
            from hugegraph_llm.models.embeddings.init_embedding import Embeddings

            try:
                llms = LLMs()
                flow.set_llm(llms.get_extract_llm()) if hasattr(flow, 'set_llm') else None
                flow.set_embedding(Embeddings().get_embedding()) if hasattr(flow, 'set_embedding') else None
                flow.set_vector_index_cls(get_vector_index_class(IndexConfig().cur_vector_index)) if hasattr(flow, 'set_vector_index_cls') else None
            except Exception as e:
                log.warning("Could not auto-configure community flow: %s", e)

        result = scheduler.schedule_flow(
            FlowName.COMMUNITY_DETECT,
            graph_name=graph_name or huge_settings.graph_name,
            algorithm=algorithm,
            max_levels=max_levels,
        )
        return result
    except Exception as e:
        log.error("Community build failed: %s", e)
        return {
            "status_code": 500,
            "message": f"Community detection failed: {str(e)}",
            "community_count": 0,
            "report_count": 0,
            "index_built": False,
        }


# ── Global search handler ────────────────────────────────────

def global_search(query: str) -> Dict[str, Any]:
    """Execute macro-level Global Search over community reports.

    Performs MapReduce: match query to community summaries, generate
    per-community point-form findings, synthesize final answer.

    Args:
        query: Broad thematic question about the entire knowledge graph.

    Returns:
        Dict with answer, communities_used, map_findings.
    """
    scheduler = SchedulerSingleton.get_instance()

    try:
        from hugegraph_llm.models.llms.init_llm import LLMs
        from hugegraph_llm.utils.vector_index_utils import get_vector_index_class
        from hugegraph_llm.config.index_config import IndexConfig
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings

        # Configure the global search flow
        flow_entry = scheduler.pipeline_pool.get(FlowName.GLOBAL_SEARCH, {})
        if "flow" in flow_entry:
            flow = flow_entry["flow"]
            try:
                llms = LLMs()
                flow.set_llm(llms.get_chat_llm()) if hasattr(flow, 'set_llm') else None
                flow.set_embedding(Embeddings().get_embedding()) if hasattr(flow, 'set_embedding') else None
                flow.set_vector_index_cls(get_vector_index_class(IndexConfig().cur_vector_index)) if hasattr(flow, 'set_vector_index_cls') else None
            except Exception as e:
                log.warning("Could not auto-configure global search flow: %s", e)

        result = scheduler.schedule_flow(
            FlowName.GLOBAL_SEARCH,
            query=query,
        )
        return result
    except Exception as e:
        log.error("Global search failed: %s", e)
        return {
            "answer": f"Global search failed: {str(e)}",
            "communities_used": 0,
            "map_findings": [],
        }


# ── Graph RAG search handler ──────────────────────────────────

def graph_rag_search(
    mode: str,
    query: Optional[str] = None,
    vertex_ids: Optional[List[str]] = None,
    max_depth: int = 2,
    max_items: int = 10,
    keywords: Optional[List[str]] = None,
    gremlin_example_num: int = 3,
) -> Dict[str, Any]:
    """Execute a direct graph RAG search operation.

    Dispatches to the appropriate ToolRegistry handler based on mode:
    - graph_traverse: k-hop subgraph traversal from vertex IDs.
    - semantic_id_lookup: map keywords to vertex IDs via embedding similarity.
    - text2gremlin: NL-to-Gremlin query generation and execution.
    - schema_lookup: retrieve graph vertex/edge labels and relations.

    Args:
        mode: One of "graph_traverse", "semantic_id_lookup",
              "text2gremlin", "schema_lookup".
        query: Natural language query (for text2gremlin).
        vertex_ids: List of vertex IDs (for graph_traverse).
        max_depth: Traversal depth (for graph_traverse).
        max_items: Max paths to return (for graph_traverse).
        keywords: Keywords to look up (for semantic_id_lookup).
        gremlin_example_num: Number of Gremlin examples (for text2gremlin).

    Returns:
        Dict with operation-specific results.
    """
    try:
        from hugegraph_llm.agents.tool_registry import ToolRegistry
        from hugegraph_llm.models.llms.init_llm import LLMs
        from hugegraph_llm.utils.hugegraph_utils import get_hg_client
        from hugegraph_llm.utils.vector_index_utils import get_vector_index_class
        from hugegraph_llm.config.index_config import IndexConfig
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings

        # Initialize dependencies
        llms = LLMs()
        agent_llm = llms.get_agent_llm()
        embedding = Embeddings().get_embedding()
        vector_index_cls = get_vector_index_class(IndexConfig().cur_vector_index)

        # Create a fresh ToolRegistry and register default tools
        registry = ToolRegistry()
        registry.register_default_tools(
            llm=agent_llm,
            embedding=embedding,
            client=get_hg_client(),
            vector_index_cls=vector_index_cls,
        )

        # Dispatch to the appropriate tool
        if mode == "graph_traverse":
            result = registry.execute(
                "graph_traverse",
                vertex_ids=vertex_ids or [],
                max_depth=max_depth,
                max_items=max_items,
            )
        elif mode == "semantic_id_lookup":
            result = registry.execute(
                "semantic_id_lookup",
                keywords=keywords or [],
            )
        elif mode == "text2gremlin":
            result = registry.execute(
                "text2gremlin",
                query=query or "",
            )
        elif mode == "schema_lookup":
            result = registry.execute("schema_lookup")
        else:
            return {
                "success": False,
                "error": f"Unknown mode: {mode}. "
                         f"Supported modes: graph_traverse, semantic_id_lookup, "
                         f"text2gremlin, schema_lookup.",
            }

        if result.get("success") is False:
            log.error("Graph RAG search '%s' failed: %s", mode, result.get("error"))
        else:
            log.info("Graph RAG search '%s' succeeded", mode)

        return result

    except Exception as e:
        log.error("Graph RAG search failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "mode": mode,
        }
