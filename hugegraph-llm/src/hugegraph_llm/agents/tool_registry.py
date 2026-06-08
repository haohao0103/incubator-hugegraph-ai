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

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from hugegraph_llm.utils.log import log


@dataclass
class Tool:
    """Represents a callable tool that can be invoked by an LLM agent.

    Each tool wraps an existing Operator or query function and exposes it
    with OpenAI-compatible function-calling schema.
    """

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for the tool's parameters
    handler: Callable[..., Dict[str, Any]]  # The actual function to execute
    requires_hugegraph: bool = False  # Whether this tool needs HugeGraph connection
    requires_vector_index: bool = False  # Whether this tool needs a vector index

    def get_openai_function_definition(self) -> Dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        try:
            result = self.handler(**kwargs)
            return {"success": True, "data": result}
        except Exception as e:
            log.error("Tool %s execution failed: %s", self.name, str(e))
            return {"success": False, "error": str(e)}


class ToolRegistry:
    """Registry for managing and executing LLM-agent tools.

    Wraps existing hugegraph-llm Operators as tools callable by an agent LLM.
    Tools are registered at initialization time and can be filtered for
    specific agent configurations.

    Usage:
        registry = ToolRegistry()
        # Register tools individually or use defaults
        registry.register_default_tools(
            llm=chat_llm, embedding=embedding, client=hugegraph_client
        )
        # Get OpenAI-compatible tool definitions
        definitions = registry.get_tool_definitions()
        # Execute a tool
        result = registry.execute("vector_search", query="Who is Sarah?")
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    # ── Registration ──────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        """Register a single tool. Overwrites if name already exists."""
        if tool.name in self._tools:
            log.warning("Tool '%s' is being overwritten in registry.", tool.name)
        self._tools[tool.name] = tool
        log.debug("Registered tool: %s", tool.name)

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(tool_name, None)

    def register_default_tools(
        self,
        llm: Any = None,
        embedding: Any = None,
        client: Any = None,
        vector_index_cls: Any = None,
        gremlin_generator: Any = None,
    ) -> None:
        """Register the 7 default tools that wrap existing Operators.

        Args:
            llm: The chat LLM instance (for keyword extraction, answer synthesis).
            embedding: The embedding model instance (for vector search).
            client: HugeGraph PyHugeClient instance (for graph operations).
            vector_index_cls: The vector store class (FAISS/Milvus/Qdrant).
            gremlin_generator: Optional pre-configured GremlinGenerateSynthesize.
        """
        # 1. keyword_extract
        self.register(
            Tool(
                name="keyword_extract",
                description=(
                    "Extract key search keywords from the user's query. "
                    "Returns a list of keywords with importance scores. "
                    "Use this to identify what entities/concepts to search for."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text to extract keywords from.",
                        },
                        "max_keywords": {
                            "type": "integer",
                            "description": "Maximum number of keywords to extract (default: 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._make_keyword_extract_handler(llm),
            )
        )

        # 2. vector_search
        self.register(
            Tool(
                name="vector_search",
                description=(
                    "Search the document vector index for text chunks semantically "
                    "similar to the query. Returns relevant document passages. "
                    "Use this for fact-finding and retrieving specific information."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query text.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 3).",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._make_vector_search_handler(embedding, vector_index_cls),
                requires_vector_index=True,
            )
        )

        # 3. graph_traverse
        self.register(
            Tool(
                name="graph_traverse",
                description=(
                    "Traverse the knowledge graph starting from given vertex IDs. "
                    "Performs k-neighbor subgraph traversal to discover connected "
                    "entities and relationships. Use this to explore how entities "
                    "are related in the graph."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "vertex_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of vertex IDs to start traversal from.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum traversal depth / number of hops (default: 2).",
                            "default": 2,
                        },
                        "max_items": {
                            "type": "integer",
                            "description": "Maximum number of paths to return (default: 10).",
                            "default": 10,
                        },
                    },
                    "required": ["vertex_ids"],
                },
                handler=self._make_graph_traverse_handler(client),
                requires_hugegraph=True,
            )
        )

        # 4. text2gremlin
        self.register(
            Tool(
                name="text2gremlin",
                description=(
                    "Convert a natural language question into a Gremlin graph query "
                    "and execute it against HugeGraph. Use this for complex graph "
                    "queries that require precise traversal patterns. "
                    "Results are the raw graph query output."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the graph query.",
                        },
                    },
                    "required": ["query"],
                },
                handler=self._make_text2gremlin_handler(client, gremlin_generator, llm),
                requires_hugegraph=True,
            )
        )

        # 5. semantic_id_lookup
        self.register(
            Tool(
                name="semantic_id_lookup",
                description=(
                    "Map keywords to specific vertex IDs in the graph using semantic "
                    "similarity search. Returns matched vertex IDs that can be used "
                    "with graph_traverse. Use this to find graph nodes matching "
                    "a concept or entity name."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords to look up in the graph.",
                        },
                    },
                    "required": ["keywords"],
                },
                handler=self._make_semantic_id_handler(client, embedding, vector_index_cls),
                requires_hugegraph=True,
                requires_vector_index=True,
            )
        )

        # 6. answer_synthesize
        self.register(
            Tool(
                name="answer_synthesize",
                description=(
                    "Synthesize a final natural language answer from retrieved context. "
                    "Use this as the LAST step after gathering sufficient context "
                    "from other tools. The generated answer will include references "
                    "to the provided context."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The original user question.",
                        },
                        "context": {
                            "type": "string",
                            "description": "All gathered context information to base the answer on.",
                        },
                    },
                    "required": ["query", "context"],
                },
                handler=self._make_answer_synthesize_handler(llm),
            )
        )

        # 7. schema_lookup
        self.register(
            Tool(
                name="schema_lookup",
                description=(
                    "Retrieve the graph schema including vertex labels, edge labels, "
                    "and their properties. Use this to understand the graph structure "
                    "before writing Gremlin queries or planning traversals."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self._make_schema_lookup_handler(client),
                requires_hugegraph=True,
            )
        )

    # ── Tool Definition Export ────────────────────────────────────

    def get_tool_definitions(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return tool definitions in OpenAI function-calling format.

        Args:
            tool_names: Optional list of tool names to include. If None, returns all.

        Returns:
            List of OpenAI-compatible tool definition dicts.
        """
        if tool_names is None:
            tools = list(self._tools.values())
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        return [t.get_openai_function_definition() for t in tools]

    def get_tool_names(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    # ── Execution ─────────────────────────────────────────────────

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a registered tool by name.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Parameters to pass to the tool handler.

        Returns:
            Dict with 'success' (bool) and either 'data' or 'error'.

        Raises:
            ValueError: If the tool is not registered.
        """
        if tool_name not in self._tools:
            available = ", ".join(self._tools.keys())
            raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {available}")
        tool = self._tools[tool_name]
        log.info("Executing tool: %s with params: %s", tool_name, str(kwargs))
        return tool.execute(**kwargs)

    # ── Tool Handler Factories ────────────────────────────────────

    def _make_keyword_extract_handler(self, llm: Any) -> Callable:
        """Create a handler for keyword extraction."""

        def handler(query: str, max_keywords: int = 5) -> Dict[str, Any]:
            from hugegraph_llm.config import prompt
            from hugegraph_llm.operators.llm_op.keyword_extract import KeywordExtract

            extractor = KeywordExtract(
                text=query,
                llm=llm,
                max_keywords=max_keywords,
                extract_template=prompt.keywords_extract_prompt,
            )
            context = {"query": query}
            result = extractor.run(context)
            return {
                "keywords": result.get("keywords", []),
                "query": query,
            }

        return handler

    def _make_vector_search_handler(self, embedding: Any, vector_index_cls: Any) -> Callable:
        """Create a handler for vector index search."""

        def handler(query: str, top_k: int = 3) -> Dict[str, Any]:
            from hugegraph_llm.config import huge_settings
            from hugegraph_llm.operators.index_op.vector_index_query import VectorIndexQuery

            if embedding is None or vector_index_cls is None:
                return {"error": "Vector index is not configured. Please build the index first."}

            searcher = VectorIndexQuery(
                vector_index=vector_index_cls,
                embedding=embedding,
                topk=top_k,
            )
            context = {"query": query}
            result = searcher.run(context)
            return {
                "results": result.get("vector_result", []),
                "query": query,
            }

        return handler

    def _make_graph_traverse_handler(self, client: Any) -> Callable:
        """Create a handler for subgraph traversal."""

        def handler(
            vertex_ids: List[str], max_depth: int = 2, max_items: int = 10
        ) -> Dict[str, Any]:
            if client is None:
                return {"error": "HugeGraph client is not configured."}

            from hugegraph_llm.config import huge_settings

            # Build the Gremlin neighbor traversal query
            keywords_str = ", ".join(f"'{vid}'" for vid in vertex_ids)
            edge_limit = huge_settings.edge_limit_pre_label * 10  # generous default

            gremlin_query = f"""\
g.V({keywords_str})
.repeat(
   bothE().limit({edge_limit}).otherV().dedup()
).times({max_depth}).emit()
.simplePath()
.path()
.by(project('label', 'id', 'props')
   .by(label())
   .by(id())
   .by(valueMap().by(unfold()))
)
.by(project('label', 'inV', 'outV', 'props')
   .by(label())
   .by(inV().id())
   .by(outV().id())
   .by(valueMap().by(unfold()))
)
.limit({max_items})
.toList()
"""
            try:
                result = client.gremlin().exec(gremlin=gremlin_query)
                data = result.get("data", []) if isinstance(result, dict) else []
            except Exception as e:
                return {"error": f"Graph traversal failed: {str(e)}", "paths": []}

            # Format paths into readable strings
            formatted_paths = []
            for path in data:
                path_str = self._format_path(path)
                if path_str:
                    formatted_paths.append(path_str)

            return {
                "paths": formatted_paths,
                "path_count": len(formatted_paths),
                "vertex_ids": vertex_ids,
                "max_depth": max_depth,
            }

        return handler

    @staticmethod
    def _format_path(path: list) -> str:
        """Format a Gremlin path result into a readable string."""
        if not path:
            return ""
        parts = []
        for item in path:
            if isinstance(item, dict):
                label = item.get("label", "")
                props = item.get("props", {})
                vid = item.get("id", "")
                if "inV" in item:
                    # This is an edge
                    parts.append(f"-[{label}]->")
                else:
                    # This is a vertex
                    prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
                    parts.append(f"({label}:{vid}{{{prop_str}}})")
        return " ".join(parts)

    def _make_text2gremlin_handler(
        self, client: Any, gremlin_generator: Any, llm: Any
    ) -> Callable:
        """Create a handler for Text2Gremlin query generation and execution."""

        def handler(query: str) -> Dict[str, Any]:
            if client is None:
                return {"error": "HugeGraph client is not configured."}

            from hugegraph_llm.operators.llm_op.gremlin_generate import (
                GremlinGenerateSynthesize,
            )

            if gremlin_generator is not None:
                generator = gremlin_generator
            else:
                generator = GremlinGenerateSynthesize(
                    llm=llm,
                    num_gremlin_generate_example=3,
                    language="EN",
                )

            context = {
                "query": query,
                "match_result": [],
                "schema": self._get_schema_str(client),
            }
            try:
                result = generator.run(context)
                gremlin = result.get("result", "")
            except Exception as e:
                return {"error": f"Gremlin generation failed: {str(e)}", "gremlin": ""}

            # Execute the generated Gremlin
            if gremlin and "g.V().limit(0)" not in gremlin:
                try:
                    exec_result = client.gremlin().exec(gremlin=gremlin)
                    data = exec_result.get("data", []) if isinstance(exec_result, dict) else []
                except Exception as e:
                    return {
                        "gremlin": gremlin,
                        "error": f"Gremlin execution failed: {str(e)}",
                        "results": [],
                    }
                return {
                    "gremlin": gremlin,
                    "results": data,
                    "result_count": len(data) if isinstance(data, list) else 0,
                }

            return {
                "gremlin": gremlin or "",
                "results": [],
                "note": "Query was classified as too complex for Text2Gremlin.",
            }

        return handler

    def _make_semantic_id_handler(
        self, client: Any, embedding: Any, vector_index_cls: Any
    ) -> Callable:
        """Create a handler for semantic vertex ID lookup."""

        def handler(keywords: List[str]) -> Dict[str, Any]:
            if client is None:
                return {"error": "HugeGraph client is not configured."}

            from hugegraph_llm.config import huge_settings
            from hugegraph_llm.operators.index_op.semantic_id_query import SemanticIdQuery

            searcher = SemanticIdQuery(
                client=client,
                embedding=embedding,
                vector_index=vector_index_cls,
                by="keywords",
                topk_per_keyword=huge_settings.topk_per_keyword,
            )
            context = {"query": " ".join(keywords), "keywords": keywords}
            result = searcher.run(context)
            return {
                "matched_vids": result.get("match_vids", []),
                "keywords": keywords,
            }

        return handler

    def _make_answer_synthesize_handler(self, llm: Any) -> Callable:
        """Create a handler for answer synthesis."""

        def handler(query: str, context: str) -> Dict[str, Any]:
            from hugegraph_llm.config import prompt

            answer_prompt_tpl = prompt.answer_prompt
            prompt_text = answer_prompt_tpl.format(
                context_str=context, query_str=query
            )
            try:
                answer = llm.generate(prompt=prompt_text)
                return {"answer": answer, "query": query}
            except Exception as e:
                return {"error": f"Answer generation failed: {str(e)}", "answer": ""}

        return handler

    def _make_schema_lookup_handler(self, client: Any) -> Callable:
        """Create a handler for graph schema retrieval."""

        def handler() -> Dict[str, Any]:
            if client is None:
                return {"error": "HugeGraph client is not configured."}

            try:
                schema = client.schema()
                vertex_labels = schema.getVertexLabels()
                edge_labels = schema.getEdgeLabels()
                relations = schema.getRelations()

                schema_str = "Vertex Labels:\n"
                for vl in vertex_labels:
                    props = vl.get("properties", [])
                    schema_str += f"  - {vl['name']} (properties: {props})\n"

                schema_str += "\nEdge Labels:\n"
                for el in edge_labels:
                    props = el.get("properties", [])
                    schema_str += f"  - {el['name']} (properties: {props})\n"

                if relations:
                    schema_str += "\nRelations:\n"
                    for rel in relations:
                        schema_str += (
                            f"  - ({rel['source_label']})-[{rel['name']}]"
                            f"->({rel['target_label']})\n"
                        )

                return {
                    "schema_text": schema_str,
                    "vertex_labels": [vl["name"] for vl in vertex_labels],
                    "edge_labels": [el["name"] for el in edge_labels],
                    "relations": relations,
                }
            except Exception as e:
                return {"error": f"Schema lookup failed: {str(e)}"}

        return handler

    @staticmethod
    def _get_schema_str(client: Any) -> str:
        """Helper to get a compact schema string from HugeGraph."""
        try:
            schema = client.schema()
            vertex_labels = schema.getVertexLabels()
            edge_labels = schema.getEdgeLabels()
            parts = ["Vertex Labels:"]
            for vl in vertex_labels:
                parts.append(f"  - {vl['name']}: {vl.get('properties', [])}")
            parts.append("Edge Labels:")
            for el in edge_labels:
                parts.append(f"  - {el['name']}: {el.get('properties', [])}")
            return "\n".join(parts)
        except Exception:
            return "Schema unavailable"


# ── Convenience factory ────────────────────────────────────────

def create_default_tool_registry(
    llm: Any = None,
    embedding: Any = None,
    client: Any = None,
    vector_index_cls: Any = None,
    gremlin_generator: Any = None,
) -> ToolRegistry:
    """Create a ToolRegistry with all 7 default tools registered.

    This is the recommended way to create a ready-to-use registry
    for the ReAct agent.

    Args:
        llm: Chat LLM instance (for keyword extraction, answer synthesis, Text2Gremlin).
        embedding: Embedding model instance (for vector search, semantic ID lookup).
        client: HugeGraph PyHugeClient instance.
        vector_index_cls: Vector store class (FAISS/Milvus/Qdrant).
        gremlin_generator: Optional pre-configured Gremlin generate synthesizer.

    Returns:
        A ToolRegistry with all default tools registered.
    """
    registry = ToolRegistry()
    registry.register_default_tools(
        llm=llm,
        embedding=embedding,
        client=client,
        vector_index_cls=vector_index_cls,
        gremlin_generator=gremlin_generator,
    )
    return registry
