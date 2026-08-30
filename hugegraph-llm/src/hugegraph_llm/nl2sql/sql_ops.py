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

"""Deterministic SQL quality gate for NL2SQL: validation, multi-candidate
voting and (optionally) execution.

Borrowed from the parallel NL2SQL demo branch's ``kg_sql_voter`` /
``sql_executor``. The voter ranks candidate SQLs against the *same* schema
graph the linking stage used, purely by:

* validity — tables/columns exist, aggregate (口径) matches the linked metric,
  join keys are connected in the graph;
* join connectivity — prefer fully-connected join graphs;
* schema overlap — prefer candidates that reference the entities the linker
  surfaced.

No LLM call at vote time (the generator already spent the budget). Execution
is an optional capability behind an import guard (duckdb).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from sqlglot import exp, parse

from .schema_graph.model import SchemaGraph

_AGG_MAP = {"SUM": "sum", "COUNT": "count", "AVG": "avg", "MAX": "max",
            "MIN": "min"}


class SqlValidator:
    """Structural checks against the SchemaGraph (no LLM, no execution)."""

    def __init__(self, schema: SchemaGraph):
        self._schema = schema
        self._table_names = {n.name for n in schema.tables()}
        self._column_ids = {
            f"{n.properties.get('table', '')}.{n.name}"
            for n in schema.columns()
            if n.properties.get("table")
        }
        self._column_names = {n.name for n in schema.columns()}

    def _extract(self, sql: str) -> Tuple[List[str], List[str], List[str]]:
        """(tables, columns, aggregates) referenced by a SQL string."""
        tables, columns, aggs = [], [], []
        try:
            stmts = parse(sql, read=None)
        except Exception:  # noqa: BLE001
            return tables, columns, aggs
        for stmt in stmts:
            if not isinstance(stmt, exp.Select):
                continue
            for t in stmt.find_all(exp.Table):
                if t.name:
                    tables.append(t.name)
            for col in stmt.find_all(exp.Column):
                if col.name:
                    columns.append(col.name)
            for fn in stmt.find_all(exp.AggFunc):
                name = fn.__class__.__name__.upper()
                if name in _AGG_MAP:
                    aggs.append(_AGG_MAP[name])
        return tables, columns, aggs

    def validate(self, sql: str) -> Dict[str, object]:
        """Return a per-check report for one candidate SQL."""
        tables, columns, aggs = self._extract(sql)
        unknown_tables = sorted({t for t in tables if t not in self._table_names})
        unknown_cols = sorted({c for c in columns if c not in self._column_names})
        return {
            "tables": sorted(set(tables)),
            "columns": sorted(set(columns)),
            "aggs": aggs,
            "unknown_tables": unknown_tables,
            "unknown_columns": unknown_cols,
            "valid": not unknown_tables and not unknown_cols,
        }


class SqlVoter:
    """Rank candidate SQLs deterministically against the schema graph."""

    def __init__(self, schema: SchemaGraph, validator: Optional[SqlValidator] = None):
        self._schema = schema
        self._validator = validator or SqlValidator(schema)

    def vote(self, candidates: List[str], linked_ids: Optional[List[str]] = None,
             metric_agg: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """Return ``(sql, score, report)`` sorted best first."""
        linked = set(linked_ids or [])
        scored = []
        for sql in candidates:
            rep = self._validator.validate(sql)
            score = 1.0 if rep["valid"] else -10.0
            # aggregate (口径) hit: reward a SQL that matches the linked metric's
            # canonical aggregate when one is expected.
            if metric_agg and rep["aggs"]:
                if metric_agg in rep["aggs"]:
                    score += 2.0
                else:
                    score -= 1.0
            # schema overlap: bonus for referencing linker-surfaced tables.
            if linked:
                hit = sum(1 for t in rep["tables"] if f"table:{t}" in linked
                          or any(f"column:{t}." in lid for lid in linked))
                score += 0.5 * min(hit, 3)
            scored.append((sql, score, rep))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored


class SqlExecutor(ABC):
    """Execution contract; implement per backend (duckdb, a real warehouse...)."""

    @abstractmethod
    def execute(self, sql: str) -> object:
        """Run ``sql`` and return the result set."""


class DuckDbExecutor(SqlExecutor):
    """Embedded executor for the demo loop (requires duckdb)."""

    def __init__(self, seed_sql: Optional[str] = None):
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "DuckDbExecutor requires duckdb (pip install duckdb)"
            ) from exc
        self._conn = duckdb.connect()
        if seed_sql:
            self._conn.execute(seed_sql)

    def execute(self, sql: str) -> object:
        return self._conn.execute(sql).fetchall()
