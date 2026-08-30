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

"""
Build the Schema Graph from deterministic warehouse metadata.

Inputs are all *deterministic* — no LLM is involved at build time:
- table / column catalog
- lineage (upstream -> downstream table)
- declared foreign keys (column -> column)
- historical query logs (table co-occurrence)
- business term glossary (term -> column / expression)

Usage:
    builder = SchemaGraphBuilder()
    builder.add_table(Table(name="orders", database="dw", is_fact=True))
    builder.add_column(Column(name="amount", table="dw.orders"))
    builder.add_foreign_key("dw.orders.user_id", "dw.users.id")
    builder.add_query_logs([["dw.orders", "dw.users"], ["dw.orders"]])
    graph = builder.build()
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from hugegraph_llm.utils.log import log

from .model import (
    Column,
    Edge,
    EdgeType,
    SchemaGraph,
    Table,
    Term,
)


class SchemaGraphBuilder:
    """Incrementally assembles a :class:`SchemaGraph`."""

    def __init__(self):
        self._tables: Dict[str, Table] = {}
        self._columns: Dict[str, Column] = {}
        self._terms: Dict[str, Term] = {}
        self._lineage: List[Tuple[str, str]] = []
        self._foreign_keys: List[Tuple[str, str]] = []
        self._term_bindings: List[Tuple[str, str]] = []
        # co-occurrence counts keyed by the unordered table pair
        self._co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)

    # ---- catalog ----

    def add_table(self, table: Table) -> "SchemaGraphBuilder":
        self._tables[table.full_name] = table
        return self

    def add_tables(self, tables: Iterable[Table]) -> "SchemaGraphBuilder":
        for t in tables:
            self.add_table(t)
        return self

    def add_column(self, column: Column) -> "SchemaGraphBuilder":
        self._columns[column.full_name] = column
        return self

    def add_columns(self, columns: Iterable[Column]) -> "SchemaGraphBuilder":
        for c in columns:
            self.add_column(c)
        return self

    # ---- relations ----

    def add_lineage(self, upstream: str, downstream: str) -> "SchemaGraphBuilder":
        """Declare that ``downstream`` is derived from ``upstream``."""
        self._lineage.append((upstream, downstream))
        return self

    def add_foreign_key(self, src_column: str, dst_column: str) -> "SchemaGraphBuilder":
        """Declare a foreign key from ``src_column`` to ``dst_column``."""
        self._foreign_keys.append((src_column, dst_column))
        return self

    def add_query_logs(self, queries: Iterable[Iterable[str]]) -> "SchemaGraphBuilder":
        """Mine table co-occurrence from historical queries.

        Each query is an iterable of table names touched by that query. Every
        unordered pair within a query increments its co-occurrence count.
        """
        for tables in queries:
            names = sorted({t for t in tables})
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    self._co_occurrence[(a, b)] += 1
        return self

    def add_term(self, term: Term) -> "SchemaGraphBuilder":
        self._terms[term.name] = term
        return self

    def bind_term(self, term_name: str, column_name: str) -> "SchemaGraphBuilder":
        """Bind a business term to a physical column."""
        self._term_bindings.append((term_name, column_name))
        return self

    # ---- build ----

    def build(self) -> SchemaGraph:
        """Materialise the graph, dropping any edge whose endpoint is unknown.

        Unknown endpoints are skipped with a warning rather than raising,
        because warehouse metadata is routinely incomplete: a dangling
        lineage reference should degrade one edge, not abort the build.
        """
        graph = SchemaGraph()

        for table in self._tables.values():
            graph.add_node(table.to_node())
        for column in self._columns.values():
            if column.table not in self._tables:
                # Unregistered parent table: drop the column rather than emit a
                # dangling node. Warehouse metadata is routinely incomplete.
                log.warning("column %s references unknown table %s; skipped",
                            column.full_name, column.table)
                continue
            graph.add_node(column.to_node())
        for term in self._terms.values():
            graph.add_node(term.to_node())

        # Column -> Table ownership
        for column in self._columns.values():
            if column.table not in self._tables:
                continue
            source = f"column:{column.full_name}"
            target = f"table:{column.table}"
            if target in graph.nodes:
                graph.add_edge(
                    Edge(source, target, EdgeType.BELONGS_TO)
                )
            else:  # pragma: no cover - guarded above
                log.warning("column %s references unknown table %s",
                            column.full_name, column.table)

        # Foreign keys: the strongest join signal
        for src, dst in self._foreign_keys:
            source, target = f"column:{src}", f"column:{dst}"
            if source in graph.nodes and target in graph.nodes:
                graph.add_edge(Edge(source, target, EdgeType.FOREIGN_KEY))
            else:
                log.warning("foreign key endpoint missing: %s -> %s", src, dst)

        # Lineage between tables
        for upstream, downstream in self._lineage:
            source, target = f"table:{upstream}", f"table:{downstream}"
            if source in graph.nodes and target in graph.nodes:
                graph.add_edge(Edge(source, target, EdgeType.LINEAGE))
            else:
                log.warning("lineage endpoint missing: %s -> %s",
                            upstream, downstream)

        # Co-occurrence between tables (weighted)
        for (a, b), count in self._co_occurrence.items():
            source, target = f"table:{a}", f"table:{b}"
            if source in graph.nodes and target in graph.nodes:
                graph.add_edge(
                    Edge(source, target, EdgeType.CO_OCCUR, weight=float(count))
                )
            else:
                log.warning("co-occurrence endpoint missing: %s <-> %s", a, b)

        # Business term bindings
        for term_name, column_name in self._term_bindings:
            source, target = f"term:{term_name}", f"column:{column_name}"
            if source in graph.nodes and target in graph.nodes:
                graph.add_edge(Edge(source, target, EdgeType.TERM_MAPS))
            else:
                log.warning("term binding missing: %s -> %s",
                            term_name, column_name)

        log.info(
            "schema graph built: %s tables, %s columns, %s terms, %s edges",
            len(graph.tables()), len(graph.columns()),
            len(graph.terms()), len(graph.edges),
        )
        return graph
