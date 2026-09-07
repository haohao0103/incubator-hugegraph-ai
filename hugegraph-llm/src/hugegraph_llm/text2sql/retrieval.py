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

"""Deterministic retrieval operators for the Text2SQL semantic-layer graph.

The five operators mirror the industry split:

- ``term_resolution``   — normalize a mention to a canonical term (schema linking).
- ``schema_link``       — follow ``maps_to`` / ``maps_to_metric`` edges (graph traversal).
- ``find_join_path``    — BFS over ``joins`` edges (TAG: determinism over vectors).
- ``resolve_metric``    — read a metric's 口径 verbatim (MetricFlow: never recompute).
- ``build_sql_prompt``  — assemble the SQL-generation prompt (Vanna's DDL + docs + SQL few-shot).

All operators are pure functions over the in-memory graph produced by
:func:`hugegraph_llm.text2sql.seed.seed_graph`; in production they map 1:1 to
HugeGraph Gremlin traversals.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.text2sql.model import Filter, Term


class SemanticGraph:
    """In-memory adjacency wrapper over ``{"vertices", "edges"}`` graph data."""

    def __init__(self, graph_data: Dict[str, Any]):
        self.vertices: Dict[str, Dict[str, Any]] = {v["id"]: v for v in graph_data["vertices"]}
        self.edges: List[Dict[str, Any]] = graph_data["edges"]
        self._out: Dict[str, List[Dict[str, Any]]] = {}
        self._in: Dict[str, List[Dict[str, Any]]] = {}
        self._by_label_name: Dict[tuple, Dict[str, Any]] = {}
        for vertex in self.vertices.values():
            self._by_label_name[(vertex["label"], vertex.get("properties", {}).get("name"))] = vertex
        for edge in self.edges:
            self._out.setdefault(edge["outV"], []).append(edge)
            self._in.setdefault(edge["inV"], []).append(edge)

    def vertex(self, vid: str) -> Optional[Dict[str, Any]]:
        return self.vertices.get(vid)

    def by_name(self, label: str, name: str) -> Optional[Dict[str, Any]]:
        """Look up a vertex by its label and ``name`` property (bare name)."""
        return self._by_label_name.get((label, name))

    def out_edges(self, vid: str, label: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self._out.get(vid, []) if label is None or e["label"] == label]

    def in_edges(self, vid: str, label: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self._in.get(vid, []) if label is None or e["label"] == label]

    def adjacent(self, vid: str, label: Optional[str] = None) -> List[tuple]:
        """Return ``(neighbor_vid, edge)`` pairs across both directions."""
        result: List[tuple] = []
        for edge in self._out.get(vid, []):
            if label is None or edge["label"] == label:
                result.append((edge["inV"], edge))
        for edge in self._in.get(vid, []):
            if label is None or edge["label"] == label:
                result.append((edge["outV"], edge))
        return result


# ---------------------------------------------------------------------------
# Operator 1: term resolution
# ---------------------------------------------------------------------------


class TermIndex:
    """Resolve a natural-language mention to a canonical term name.

    Exact alias/name match first, then a substring fallback.  In production the
    substring fallback is replaced by vector recall + LLM disambiguation; the
    canonical term returned here feeds :func:`schema_link`.
    """

    def __init__(self, terms: List[Term]):
        self._name_to_term: Dict[str, str] = {}
        self._alias_to_term: Dict[str, str] = {}
        for term in terms:
            self._name_to_term[term.name.lower()] = term.name
            for alias in term.aliases:
                self._alias_to_term[alias.lower()] = term.name

    def resolve(self, mention: str) -> Optional[str]:
        normalized = mention.strip().lower()
        if not normalized:
            return None
        if normalized in self._alias_to_term:
            return self._alias_to_term[normalized]
        if normalized in self._name_to_term:
            return self._name_to_term[normalized]
        for alias, term_name in self._alias_to_term.items():
            if alias and alias in normalized:
                return term_name
        return None

    def find_in_text(self, text: str) -> List[str]:
        """Return the canonical terms whose name/alias appears in ``text``.

        Longest keys are matched first so a specific alias wins over a short
        generic one; results are deduplicated and deterministic.
        """
        lowered = text.lower()
        keys = sorted(set(self._name_to_term) | set(self._alias_to_term), key=len, reverse=True)
        found: List[str] = []
        seen = set()
        for key in keys:
            if key in lowered:
                term_name = self._alias_to_term.get(key) or self._name_to_term.get(key)
                if term_name not in seen:
                    seen.add(term_name)
                    found.append(term_name)
        return found


# ---------------------------------------------------------------------------
# Operator 2: schema linking
# ---------------------------------------------------------------------------


def schema_link(graph: SemanticGraph, term_name: str) -> Dict[str, List[str]]:
    """Follow ``maps_to`` / ``maps_to_metric`` edges from a resolved term.

    Column results are reported as ``full_name`` (``"table.column"``) and metric
    results as bare names, because those are what SQL generation / downstream
    operators consume — while the edges themselves point at label-scoped ids.
    """
    term = graph.by_name("term", term_name)
    if term is None:
        return {"columns": [], "metrics": []}

    columns: List[str] = []
    for edge in graph.out_edges(term["id"], "maps_to"):
        column = graph.vertex(edge["inV"])
        if column is not None:
            columns.append(column.get("properties", {}).get("full_name", ""))

    metrics: List[str] = []
    for edge in graph.out_edges(term["id"], "maps_to_metric"):
        metric = graph.vertex(edge["inV"])
        if metric is not None:
            metrics.append(metric.get("properties", {}).get("name", ""))

    return {"columns": columns, "metrics": metrics}


# ---------------------------------------------------------------------------
# Operator 3: join path finding
# ---------------------------------------------------------------------------


@dataclass
class JoinStep:
    from_table: str
    to_table: str
    on_condition: str = ""
    join_type: str = "LEFT JOIN"
    fanout_risk: str = "none"


def find_join_path(
    graph: SemanticGraph,
    from_table: str,
    to_table: str,
) -> List[JoinStep]:
    """Return the shortest join path (BFS) between two tables, or ``[]``.

    The ``joins`` edge is treated as undirected for path finding because an
    equality ``on_condition`` is symmetric; ``fanout_risk`` is still reported so
    the SQL builder can emit DISTINCT/dedup hints.
    """
    if from_table == to_table:
        return []

    start = graph.by_name("table", from_table)
    end = graph.by_name("table", to_table)
    if start is None or end is None:
        return []

    start_id = start["id"]
    end_id = end["id"]
    prev: Dict[str, Optional[tuple]] = {start_id: None}
    queue = deque([start_id])
    visited = {start_id}

    while queue:
        current = queue.popleft()
        if current == end_id:
            break
        for neighbor, edge in graph.adjacent(current, "joins"):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            prev[neighbor] = (current, edge)
            queue.append(neighbor)

    if end_id not in prev:
        return []

    steps: List[JoinStep] = []
    current_id = end_id
    while prev[current_id] is not None:
        prev_id, edge = prev[current_id]
        props = edge.get("properties", {})
        prev_vertex = graph.vertex(prev_id)
        curr_vertex = graph.vertex(current_id)
        prev_table = prev_vertex.get("properties", {}).get("name", prev_id) if prev_vertex else prev_id
        curr_table = curr_vertex.get("properties", {}).get("name", current_id) if curr_vertex else current_id
        steps.append(
            JoinStep(
                from_table=prev_table,
                to_table=curr_table,
                on_condition=props.get("on_condition", ""),
                join_type=props.get("join_type", "LEFT JOIN"),
                fanout_risk=props.get("fanout_risk", "none"),
            )
        )
        current_id = prev_id
    steps.reverse()
    return steps


# ---------------------------------------------------------------------------
# Operator 4: metric resolution (read 口径 verbatim)
# ---------------------------------------------------------------------------


@dataclass
class MetricResolution:
    name: str
    agg_func: str = ""
    measure: str = ""
    filters: List[Filter] = field(default_factory=list)
    time_granularity: str = ""
    time_column: str = ""
    dedup: bool = False
    dimensions: List[str] = field(default_factory=list)


def resolve_metric(graph: SemanticGraph, metric_name: str) -> Optional[MetricResolution]:
    """Read a metric's definition verbatim from its vertex (never recompute).

    Filters are structured (``has_filter`` -> ``filter`` vertices), so the 口径
    is auditable: each condition is a ``column/operator/values`` triple that can
    be verified against the ``column`` / ``value`` vertices.
    """
    vertex = graph.by_name("metric", metric_name)
    if vertex is None:
        return None
    props = vertex.get("properties", {})

    filters: List[Filter] = []
    for edge in graph.out_edges(vertex["id"], "has_filter"):
        filter_vertex = graph.vertex(edge["inV"])
        if filter_vertex is None:
            continue
        fp = filter_vertex.get("properties", {})
        filters.append(
            Filter(
                column=fp.get("column", ""),
                operator=fp.get("operator", "IN"),
                values=list(fp.get("values", [])),
            )
        )

    return MetricResolution(
        name=metric_name,
        agg_func=props.get("agg_func", ""),
        measure=props.get("measure", ""),
        filters=filters,
        time_granularity=props.get("time_granularity", ""),
        time_column=props.get("time_column", ""),
        dedup=bool(props.get("dedup", False)),
        dimensions=list(props.get("dimensions", [])),
    )


# ---------------------------------------------------------------------------
# Operator 5: SQL prompt building
# ---------------------------------------------------------------------------


def render_table_ddl(graph: SemanticGraph, table_name: str) -> str:
    """Render a table as DDL from its ``has_column`` columns (Vanna's DDL source)."""
    table = graph.by_name("table", table_name)
    if table is None:
        return f"-- table {table_name} not found"
    description = table.get("properties", {}).get("description", "")
    lines = [f"-- {description}" if description else f"-- table {table_name}", f"CREATE TABLE {table_name} ("]
    columns = []
    for edge in graph.out_edges(table["id"], "has_column"):
        column = graph.vertex(edge["inV"])
        if column is None:
            continue
        props = column.get("properties", {})
        col_name = props.get("name", "")
        comment = f" -- {props['description']}" if props.get("description") else ""
        columns.append(f"  {col_name} {props.get('data_type', 'TEXT')},{comment}")
    lines.append(",\n".join(columns))
    lines.append(");")
    return "\n".join(lines)


def render_value_mappings(graph: SemanticGraph, tables: List[str]) -> str:
    """Render code -> meaning mappings for the columns of the given tables."""
    lines: List[str] = []
    for table_name in tables:
        table = graph.by_name("table", table_name)
        if table is None:
            continue
        for col_edge in graph.out_edges(table["id"], "has_column"):
            column = graph.vertex(col_edge["inV"])
            column_name = column.get("properties", {}).get("full_name", col_edge["inV"]) if column else col_edge["inV"]
            for value_edge in graph.out_edges(col_edge["inV"], "has_value"):
                value = graph.vertex(value_edge["inV"])
                if value is None:
                    continue
                props = value.get("properties", {})
                lines.append(f"-- {column_name}: {props.get('code')} = {props.get('meaning')}")
    return "\n".join(lines)


def render_sql_filter(filters: List[Filter]) -> str:
    """Reconstruct a SQL WHERE clause from structured filters (audit output)."""
    parts: List[str] = []
    for filter_ in filters:
        quoted = ", ".join(f"'{value}'" for value in filter_.values)
        if filter_.operator == "IN":
            parts.append(f"{filter_.column} IN ({quoted})")
        elif filter_.operator == "BETWEEN":
            parts.append(f"{filter_.column} BETWEEN {quoted}")
        elif filter_.operator in ("EQ", "NE"):
            parts.append(f"{filter_.column} {filter_.operator} {quoted}")
        else:
            parts.append(f"{filter_.column} {filter_.operator} {quoted}")
    return " AND ".join(parts)


def render_metric_definitions(graph: SemanticGraph, metric_names: List[str]) -> str:
    """Render metric 口径 definitions (MetricFlow source).

    The WHERE clause is reconstructed from structured ``filter`` vertices, which
    is what makes the 口径 auditable and verifiable.
    """
    lines: List[str] = []
    for name in metric_names:
        metric = resolve_metric(graph, name)
        if metric is None:
            continue
        definition = f"{metric.name} = {metric.agg_func}({metric.measure})"
        where = render_sql_filter(metric.filters)
        if where:
            definition += f" WHERE {where}"
        if metric.time_column and metric.time_granularity:
            definition += f" [by {metric.time_column} @ {metric.time_granularity}]"
        if metric.dedup:
            definition += " [dedup required]"
        lines.append(f"- {definition}")
    return "\n".join(lines)


def build_sql_prompt(
    graph: SemanticGraph,
    question: str,
    tables: List[str],
    metrics: Optional[List[str]] = None,
    join_paths: Optional[List[JoinStep]] = None,
    few_shot_sql: Optional[List[str]] = None,
    db_type: str = "StarRocks",
) -> str:
    """Assemble the SQL-generation prompt from deterministic graph context."""
    metrics = metrics or []
    join_paths = join_paths or []
    few_shot_sql = few_shot_sql or []

    sections: List[str] = [
        f"You are an expert {db_type} SQL engineer. Generate ONLY a valid, executable SQL query.",
        "",
        "# Database schema (DDL)",
    ]
    sections.extend(render_table_ddl(graph, t) for t in tables)

    if metrics:
        sections += [
            "",
            "# Metric definitions (口径 — follow EXACTLY, never recompute)",
            render_metric_definitions(graph, metrics),
        ]

    value_mappings = render_value_mappings(graph, tables)
    if value_mappings:
        sections += ["", "# Value mappings (use for WHERE filters)", value_mappings]

    if join_paths:
        join_lines = [
            f"- {step.from_table} JOIN {step.to_table} ON {step.on_condition} "
            f"({step.join_type}, fanout={step.fanout_risk})"
            for step in join_paths
        ]
        sections += ["", "# Join paths (use these exact ON conditions)", "\n".join(join_lines)]

    if few_shot_sql:
        sections += ["", "# Few-shot examples (question -> verified SQL)", "\n\n".join(few_shot_sql)]

    sections += ["", "# Question", question, "", "SQL:"]
    return "\n".join(sections)
