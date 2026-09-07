# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP tool layer for the semantic layer (milestone M3).

**Transport-agnostic on purpose.** ``mcp`` is not installed in this
environment and is not declared as a dependency, so importing it here would
make the whole semantic layer unimportable. Tools are therefore plain
callables with JSON Schema input descriptions; mounting them on
``HugeGraphMCPServer`` (or any MCP SDK server) is a thin adapter written
wherever the SDK actually is.

**Capability probing**, following neocarta: rather than registering every
tool and letting the agent discover that half of them fail, the server
inspects what the graph actually offers and registers only what will work.
The chosen context tool is *named after the capability it uses*
(``..._term_hybrid_search``), so the tool list itself tells the agent what
indexes are available -- no separate introspection call needed.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from hugegraph_llm.semantic_layer.context import estimate_tokens
from hugegraph_llm.semantic_layer.join_path import (
    find_join_path,
    render_join_path,
)
from hugegraph_llm.semantic_layer.readers import SemanticGraphReader
from hugegraph_llm.semantic_layer.retrieval import (
    RetrievalConfig,
    SemanticLayerRetriever,
)

__all__ = [
    "ToolSpec",
    "Capabilities",
    "SemanticLayerTools",
    "DEFAULT_MAX_TOKENS",
]

DEFAULT_MAX_TOKENS = 4000


@dataclass
class ToolSpec:
    """One MCP tool: schema plus the callable that answers it."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]
    #: Lower runs first when several variants could satisfy the same need.
    priority: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """MCP-shaped tool description (name/description/inputSchema)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class Capabilities:
    """What the backing graph can actually support right now."""

    has_terms: bool = False
    has_vector: bool = False
    has_bm25: bool = False
    has_metrics: bool = False
    table_count: int = 0
    column_count: int = 0

    def describe(self) -> str:
        parts = [
            f"tables={self.table_count}",
            f"columns={self.column_count}",
            f"business_terms={'yes' if self.has_terms else 'no'}",
            f"vector_index={'yes' if self.has_vector else 'no'}",
            f"lexical_index={'yes' if self.has_bm25 else 'no'}",
        ]
        return ", ".join(parts)


class SemanticLayerTools:
    """Builds the tool set available for a given graph.

    :param reader: supplies the projection.
    :param vector_store: optional external ANN store; without it the vector
        recall path is skipped rather than faked.
    :param embed: embedding function, required for the vector path.
    """

    def __init__(
        self,
        reader: SemanticGraphReader,
        *,
        vector_store: Any = None,
        embed: Optional[Callable[[str], List[float]]] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.reader = reader
        self.config = config or RetrievalConfig()
        self.retriever = SemanticLayerRetriever(
            reader, vector_store=vector_store, embed=embed, config=self.config
        )
        self._vector_store = vector_store
        self._embed = embed
        self._tools: Optional[Dict[str, ToolSpec]] = None

    # -- probing ------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Inspect the graph once and report what retrieval can use."""
        proj = self.reader.projection()
        has_bm25 = False
        try:
            has_bm25 = bool(self.retriever._corpus())
        except Exception:  # noqa: BLE001 - probing must never raise
            has_bm25 = False
        return Capabilities(
            has_terms=bool(proj.terms),
            has_vector=bool(self._vector_store and self._embed),
            has_bm25=has_bm25,
            has_metrics=bool(getattr(proj, "metric_columns", {})),
            table_count=len(proj.tables),
            column_count=len(proj.columns),
        )

    # -- registration -------------------------------------------------------

    def register(self, refresh: bool = False) -> Dict[str, ToolSpec]:
        """Return the tools to expose, keyed by name. Cached."""
        if self._tools is not None and not refresh:
            return self._tools

        caps = self.capabilities()
        tools: Dict[str, ToolSpec] = {}

        # Exactly one context tool: the best variant this graph supports.
        context_tool = self._context_tool(caps)
        tools[context_tool.name] = context_tool

        for spec in (
            self._list_schemas(),
            self._list_tables(),
            self._get_join_path(),
            self._get_table_columns(),
            self._get_full_metadata_schema(),
        ):
            tools[spec.name] = spec

        if caps.has_terms:
            spec = self._search_terms()
            tools[spec.name] = spec

        self._tools = tools
        return tools

    def _context_tool(self, caps: Capabilities) -> ToolSpec:
        """Pick the richest context tool the available indexes support.

        The name encodes the capability so the agent can see -- just from the
        tool list -- whether it is getting business-term or pure lexical
        recall.
        """
        if caps.has_vector and caps.has_terms:
            name = "get_context_by_term_hybrid_search"
            description = (
                "Retrieve the schema slice relevant to a question, using "
                "business-term + vector + lexical recall. Returns tables, "
                "their columns, keys and declared join targets, trimmed to "
                "fit a token budget. Prefer this over enumerating the schema."
            )
        elif caps.has_vector:
            name = "get_context_by_table_hybrid_search"
            description = (
                "Retrieve the schema slice relevant to a question using "
                "vector + lexical recall. Returns tables, columns, keys and "
                "declared join targets within a token budget."
            )
        elif caps.has_bm25:
            name = "get_context_by_table_full_text_search"
            description = (
                "Retrieve the schema slice relevant to a question using "
                "lexical (BM25) recall over table and column names and "
                "comments. No vector index is configured."
            )
        else:
            name = "get_context_by_table_lookup"
            description = (
                "Retrieve schema context by listing tables. Neither a vector "
                "nor a lexical index is available, so questioning is limited "
                "to browsing."
            )
        return ToolSpec(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to retrieve schema for.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": (
                            "Token budget for the returned context "
                            f"(default {DEFAULT_MAX_TOKENS})."
                        ),
                    },
                },
                "required": ["question"],
            },
            handler=self._handle_context,
            priority=1,
            tags=["context"],
        )

    # -- handlers -----------------------------------------------------------

    def _handle_context(
        self, question: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Dict[str, Any]:
        result = self.retriever.retrieve(question, max_tokens=max_tokens)
        summary = result.summary()
        summary["contexts"] = result.to_dicts()
        if not result.tables:
            summary["hint"] = (
                "No table matched. Try a business term, or list_schemas to "
                "browse what is available."
            )
        return summary

    def _list_schemas(self) -> ToolSpec:
        return ToolSpec(
            name="list_schemas",
            description=(
                "List the databases and schemas present in the semantic "
                "layer, with the number of tables in each."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=self._handle_list_schemas,
            tags=["browse"],
        )

    def _handle_list_schemas(self) -> Dict[str, Any]:
        proj = self.reader.projection()
        grouped: Dict[str, Dict[str, int]] = {}
        for table in proj.tables.values():
            database = table.database or "(default)"
            schema = table.schema or "(default)"
            key = f"{database}.{schema}"
            grouped.setdefault(key, {"tables": 0})["tables"] += 1
        return {"schemas": grouped, "capabilities": self.capabilities().describe()}

    def _list_tables(self) -> ToolSpec:
        return ToolSpec(
            name="list_tables_by_schema",
            description=(
                "List the tables of one schema with their row counts and "
                "comments. Optionally filter by a name substring."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "string",
                        "description": "Schema name to list (optional).",
                    },
                    "name_contains": {
                        "type": "string",
                        "description": "Substring filter on table name.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum tables to return (default 50).",
                    },
                },
            },
            handler=self._handle_list_tables,
            tags=["browse"],
        )

    def _handle_list_tables(
        self,
        schema: Optional[str] = None,
        name_contains: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        proj = self.reader.projection()
        rows = []
        for table in sorted(proj.tables.values(), key=lambda t: -t.row_count):
            if schema and table.schema != schema and table.database != schema:
                continue
            if name_contains and name_contains.lower() not in table.name.lower():
                continue
            rows.append({
                "name": table.name,
                "schema": f"{table.database}.{table.schema}",
                "comment": table.comment,
                "row_count": table.row_count,
                "confidence": table.confidence,
            })
            if len(rows) >= limit:
                break
        return {"tables": rows, "returned": len(rows)}

    def _get_join_path(self) -> ToolSpec:
        return ToolSpec(
            name="get_join_path",
            description=(
                "Return the join path between two tables: the ordered column "
                "pairs to join on, and whether each step is a declared "
                "foreign key (proven) or merely inferred. Use this instead of "
                "guessing an ON clause. If no column-level path exists, no "
                "join should be written."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "string", "description": "Left table name."},
                    "right": {"type": "string", "description": "Right table name."},
                },
                "required": ["left", "right"],
            },
            handler=self._handle_join_path,
            tags=["join"],
        )

    def _handle_join_path(self, left: str, right: str) -> Dict[str, Any]:
        proj = self.reader.projection()
        path = find_join_path(proj, left, right)
        payload = path.to_dict()
        payload["render"] = render_join_path(path)
        if path.found and not path.all_proven:
            payload["warning"] = (
                "Path includes inferred joins; confirm before generating SQL."
            )
        return payload

    def _get_table_columns(self) -> ToolSpec:
        return ToolSpec(
            name="get_table_columns",
            description=(
                "Return the columns of one table with types, comments, "
                "key flags and declared join targets."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name."},
                },
                "required": ["table"],
            },
            handler=self._handle_table_columns,
            tags=["browse"],
        )

    def _handle_table_columns(self, table: str) -> Dict[str, Any]:
        proj = self.reader.projection()
        if table not in proj.tables:
            return {"error": f"unknown table: {table}", "columns": []}
        columns = []
        for col in sorted(proj.columns_of(table), key=lambda c: c.name):
            columns.append({
                "name": col.name,
                "data_type": col.data_type,
                "comment": col.comment,
                "key_type": "primary" if col.is_primary_key else (
                    "foreign" if col.is_foreign_key else ""),
                "references": proj.references.get(col.qualified, []),
                "is_time_dimension": col.is_time_dimension,
            })
        return {
            "table": table,
            "description": proj.tables[table].comment,
            "columns": columns,
        }

    def _search_terms(self) -> ToolSpec:
        return ToolSpec(
            name="search_business_terms",
            description=(
                "Search business terms (glossary entries) by name, alias or "
                "description, and see which tables and metrics define them. "
                "Use this when a question uses business vocabulary rather "
                "than table names."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Term to look up."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum terms to return (default 20).",
                    },
                },
                "required": ["query"],
            },
            handler=self._handle_search_terms,
            tags=["glossary"],
        )

    def _handle_search_terms(self, query: str, limit: int = 20) -> Dict[str, Any]:
        proj = self.reader.projection()
        needle = str(query or "").strip().lower()
        hits = []
        for name, row in proj.terms.items():
            haystack = " ".join([name, row.description, " ".join(row.aliases)]).lower()
            if not needle or needle in haystack:
                tables = {c.split(".", 1)[0] for c in proj.term_columns.get(name, [])}
                for table, terms in getattr(proj, "table_terms", {}).items():
                    if name in terms:
                        tables.add(table)
                for metric in getattr(proj, "term_metrics", {}).get(name, []):
                    for column in getattr(proj, "metric_columns", {}).get(metric, []):
                        tables.add(column.split(".", 1)[0])
                hits.append({
                    "term": name,
                    "description": row.description,
                    "aliases": row.aliases,
                    "tables": sorted(tables),
                })
            if len(hits) >= limit:
                break
        return {"terms": hits, "returned": len(hits)}

    def _get_full_metadata_schema(self) -> ToolSpec:
        return ToolSpec(
            name="get_full_metadata_schema",
            description=(
                "Dump the entire semantic layer: every table with all of its "
                "columns. EXPENSIVE -- output is capped and may be truncated. "
                "Prefer the context tool; use this only when the question is "
                "genuinely about the whole model."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "max_tokens": {
                        "type": "integer",
                        "description": "Output cap in tokens (default 8000).",
                    },
                },
            },
            handler=self._handle_full_schema,
            priority=99,
            tags=["expensive"],
        )

    def _handle_full_schema(self, max_tokens: int = 8000) -> Dict[str, Any]:
        """Full dump, budgeted and honest about truncation."""
        proj = self.reader.projection()
        results = []
        used = 0
        truncated = False
        for name in sorted(proj.tables):
            ctx = self.retriever._build_context(proj, name, 0.0, [], True)
            rendered = ctx.render("full")
            cost = estimate_tokens(rendered)
            if used + cost > max_tokens:
                truncated = True
                break
            results.append(ctx.to_dict("full"))
            used += cost
        return {
            "tables": results,
            "returned": len(results),
            "total_tables": len(proj.tables),
            "used_tokens": used,
            "truncated": truncated,
        }

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke a registered tool by name.

        Unknown names raise ``KeyError`` rather than returning an error dict:
        a misnamed tool is a programming error, not a runtime condition the
        agent should have to interpret.
        """
        tools = self.register()
        if name not in tools:
            raise KeyError(f"unregistered tool: {name}")
        return tools[name].handler(**(arguments or {}))

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        """MCP-shaped list of every registered tool."""
        return [spec.to_dict() for spec in self.register().values()]

    def invalidate(self) -> None:
        """Drop caches after the graph is re-ingested."""
        self.retriever.invalidate()
        self._tools = None
