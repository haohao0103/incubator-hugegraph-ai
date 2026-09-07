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

"""Reading the semantic layer back out of the graph.

All Gremlin lives in this module. Retrieval logic operates on a plain
:class:`SemanticProjection` instead, which keeps the scoring and pruning
rules unit-testable without a server and makes the reader swappable.

**Why the projection is loaded whole rather than traversed server-side.**
The join projection of a warehouse catalogue is small -- ACME's 33 tables
and 323 columns are ~400 nodes -- so paging it in costs a handful of
requests and makes expansion a plain BFS that can be inspected and tested.
Server-side traversal would win past roughly 10k nodes; if a catalogue grows
to that, ``GremlinSemanticReader.expand`` is the single place to change, and
the documented behaviour (2 hops, no tag edges) is what it must preserve.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.semantic_layer.paging import iter_edges, iter_vertices

__all__ = [
    "TableRow",
    "ColumnRow",
    "TermRow",
    "SemanticProjection",
    "SemanticGraphReader",
    "GremlinSemanticReader",
    "InMemorySemanticReader",
]


@dataclass
class TableRow:
    name: str
    database: str = ""
    schema: str = ""
    comment: str = ""
    row_count: int = 0
    confidence: float = 1.0
    freshness_ts: Optional[int] = None
    source_system: str = ""


@dataclass
class ColumnRow:
    name: str
    table: str
    data_type: str = ""
    comment: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_time_dimension: bool = False
    sample_values: List[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class TermRow:
    name: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class SemanticProjection:
    """The part of the graph retrieval needs, in plain Python."""

    tables: Dict[str, TableRow] = field(default_factory=dict)
    #: ``table name -> vertex id``. Names are the retrieval keys; ids are
    #: needed whenever something must *write* back (feedback edges reference
    #: endpoints by id, and Ossie namespaces its ids per model, so the
    #: ``table:<name>`` convention alone is not enough).
    table_vids: Dict[str, str] = field(default_factory=dict)
    #: keyed by ``table.column``
    columns: Dict[str, ColumnRow] = field(default_factory=dict)
    terms: Dict[str, TermRow] = field(default_factory=dict)
    #: ``table.column -> [table.column, ...]``, join evidence
    references: Dict[str, List[str]] = field(default_factory=dict)
    #: ``(from_col, to_col) -> proven``
    reference_proven: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    #: ``table -> [term, ...]``
    table_terms: Dict[str, List[str]] = field(default_factory=dict)
    #: ``term -> [table.column, ...]``
    term_columns: Dict[str, List[str]] = field(default_factory=dict)
    #: ``metric -> [table.column, ...]``, from ``HAS_EXPRESSION``
    metric_columns: Dict[str, List[str]] = field(default_factory=dict)
    #: ``term -> [metric, ...]``, reverse of ``METRIC_TAGGED_WITH``
    term_metrics: Dict[str, List[str]] = field(default_factory=dict)
    #: ``table -> [table, ...]``
    lineage: Dict[str, List[str]] = field(default_factory=dict)
    co_occur: Dict[str, List[str]] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------

    def columns_of(self, table: str) -> List[ColumnRow]:
        return [c for c in self.columns.values() if c.table == table]

    def tables_adjacent(self, table: str) -> Dict[str, List[str]]:
        """Neighbour tables with the reasons they are reachable.

        **Adjacency is symmetric.** ``orders.customer_id -> customers.id`` is
        stored in one direction only, but a query about customers can join to
        orders just as well; treating it as one-way would make BFS from the
        referenced table find nothing and would silently drop joinable
        tables from the result set.

        Only join-capable edges count: ``TAGGED_WITH`` / ``TERM_MAPS`` carry
        infinite cost (they annotate, they do not join) and are excluded.
        """
        neighbours: Dict[str, List[str]] = {}
        for col in self.columns_of(table):
            for target in self.references.get(col.qualified, []):
                other = target.split(".", 1)[0]
                if other != table:
                    neighbours.setdefault(other, []).append("REFERENCES")
        # Reverse direction: columns elsewhere pointing at this table.
        suffix = f"{table}."
        for source, targets in self.references.items():
            owner = source.split(".", 1)[0]
            if owner == table:
                continue
            if any(t.startswith(suffix) for t in targets):
                neighbours.setdefault(owner, []).append("REFERENCES")

        for edge_map, reason in ((self.lineage, "LINEAGE"), (self.co_occur, "CO_OCCUR")):
            for other in edge_map.get(table, []):
                if other != table:
                    neighbours.setdefault(other, []).append(reason)
            for owner, targets in edge_map.items():
                if owner != table and table in targets:
                    neighbours.setdefault(owner, []).append(reason)
        return neighbours

    @property
    def is_empty(self) -> bool:
        return not self.tables


class SemanticGraphReader(ABC):
    """Supplies a :class:`SemanticProjection`."""

    @abstractmethod
    def projection(self, refresh: bool = False) -> SemanticProjection:
        """Return the projection, cached unless ``refresh``."""

    def invalidate(self) -> None:
        """Drop any cached projection. No-op for uncached readers."""


class InMemorySemanticReader(SemanticGraphReader):
    """Reader over an already-built projection. Used by tests and offline runs."""

    def __init__(self, projection: SemanticProjection) -> None:
        self._projection = projection

    def projection(self, refresh: bool = False) -> SemanticProjection:
        return self._projection


class GremlinSemanticReader(SemanticGraphReader):
    """Builds the projection from a live HugeGraph via Gremlin."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self._cached: Optional[SemanticProjection] = None

    def projection(self, refresh: bool = False) -> SemanticProjection:
        if self._cached is not None and not refresh:
            return self._cached
        self._cached = self._build()
        return self._cached

    def invalidate(self) -> None:
        """Force the next ``projection()`` to re-read from the server.

        Callers that write back to the graph (feedback recording) must call
        this, or their own writes stay invisible to their later reads.
        """
        self._cached = None

    # -- building -----------------------------------------------------------

    def _build(self) -> SemanticProjection:
        proj = SemanticProjection()

        cid_to_qualified: Dict[str, str] = {}
        tid_to_table: Dict[str, str] = {}

        for vid, props in iter_vertices(self.client, VertexLabel.TABLE.value):
            name = str(props.get("name", ""))
            if not name:
                continue
            tid_to_table[vid] = name
            proj.table_vids[name] = vid
            proj.tables[name] = TableRow(
                name=name,
                database=str(props.get("database", "") or ""),
                schema=str(props.get("schema", "") or ""),
                comment=str(props.get("comment", "") or ""),
                row_count=int(props.get("row_count") or 0),
                confidence=float(props.get("confidence", 1.0) or 1.0),
                freshness_ts=props.get("freshness_ts"),
                source_system=str(props.get("source_system", "") or ""),
            )

        for vid, props in iter_vertices(self.client, VertexLabel.COLUMN.value):
            name = str(props.get("name", ""))
            table = str(props.get("table", ""))
            if not name or not table:
                continue
            row = ColumnRow(
                name=name,
                table=table,
                data_type=str(props.get("data_type", "") or ""),
                comment=str(props.get("comment", "") or ""),
                is_primary_key=bool(props.get("is_primary_key", False)),
                is_foreign_key=bool(props.get("is_foreign_key", False)),
                is_time_dimension=bool(props.get("is_time_dimension", False)),
                sample_values=self._as_list(props.get("sample_values")),
                confidence=float(props.get("confidence", 1.0) or 1.0),
            )
            proj.columns[row.qualified] = row
            cid_to_qualified[vid] = row.qualified

        metricid_to_name: Dict[str, str] = {}
        for vid, props in iter_vertices(self.client, VertexLabel.METRIC.value):
            name = str(props.get("name", ""))
            if name:
                metricid_to_name[vid] = name

        termid_to_name: Dict[str, str] = {}
        for vid, props in iter_vertices(self.client, VertexLabel.BUSINESS_TERM.value):
            name = str(props.get("name", ""))
            if not name:
                continue
            termid_to_name[vid] = name
            proj.terms[name] = TermRow(
                name=name,
                description=str(props.get("description", "") or ""),
                aliases=self._as_list(props.get("aliases")),
                confidence=float(props.get("confidence", 1.0) or 1.0),
            )

        for out_v, in_v, props in iter_edges(self.client, EdgeLabel.REFERENCES.value):
            src = cid_to_qualified.get(out_v)
            dst = cid_to_qualified.get(in_v)
            if not src or not dst:
                continue
            proj.references.setdefault(src, []).append(dst)
            proj.reference_proven[(src, dst)] = bool(props.get("proven", False))

        for out_v, in_v, _props in iter_edges(
            self.client, EdgeLabel.TABLE_TAGGED_WITH.value
        ):
            table = tid_to_table.get(out_v)
            term = termid_to_name.get(in_v)
            if table and term:
                proj.table_terms.setdefault(table, []).append(term)

        for out_v, in_v, _props in iter_edges(self.client, EdgeLabel.TERM_MAPS.value):
            term = termid_to_name.get(out_v)
            column = cid_to_qualified.get(in_v)
            if term and column:
                proj.term_columns.setdefault(term, []).append(column)

        for out_v, in_v, _props in iter_edges(
            self.client, EdgeLabel.HAS_EXPRESSION.value
        ):
            metric = metricid_to_name.get(out_v)
            column = cid_to_qualified.get(in_v)
            if metric and column:
                proj.metric_columns.setdefault(metric, []).append(column)

        for out_v, in_v, _props in iter_edges(
            self.client, EdgeLabel.METRIC_TAGGED_WITH.value
        ):
            metric = metricid_to_name.get(out_v)
            term = termid_to_name.get(in_v)
            if metric and term:
                proj.term_metrics.setdefault(term, []).append(metric)

        for label, target in (
            (EdgeLabel.LINEAGE.value, proj.lineage),
            (EdgeLabel.CO_OCCUR.value, proj.co_occur),
        ):
            for out_v, in_v, _props in iter_edges(self.client, label):
                src = tid_to_table.get(out_v)
                dst = tid_to_table.get(in_v)
                if src and dst and src != dst:
                    target.setdefault(src, []).append(dst)

        return proj

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]
