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
Join path discovery over the Schema Graph.

Given the tables a query needs, work out *how to connect them*. This is the
step that most Text2SQL systems leave to the model — and the one where
models fabricate joins.

Two entry points, both independently callable:

``shortest_path(a, b)``
    Cheapest path between two tables (Dijkstra on join cost).

``connect([a, b, c])``
    Approximate Steiner tree spanning several tables (metric-closure MST,
    within 2x of optimal).

Each hop carries its join keys when they can be *proven* from a declared
foreign key, and marks lower confidence when the hop is only supported by
lineage or query co-occurrence. That distinction matters: a proven join can
be emitted directly, an unproven one should be surfaced to the caller for
confirmation rather than silently trusted.

The *search* is delegated to a :class:`~hugegraph_llm.nl2sql.engine.base.\
GraphEngine` (networkx in-process by default, Vermeer's ``weighted_sssp`` when
a cluster engine is injected). Join-key resolution and edge-type attribution
stay local either way, because they read the declared schema rather than
traversing it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hugegraph_llm.utils.log import log

from ..engine.base import GraphEngine
from ..engine.local import LocalEngine
from ..schema_graph.backend import to_join_graph
from ..schema_graph.model import EdgeType, SchemaGraph


@dataclass
class JoinStep:
    """One hop in a join path."""

    left_table: str
    right_table: str
    left_column: str = ""
    right_column: str = ""
    edge_type: str = ""
    cost: float = 0.0
    proven: bool = False

    def to_sql(self) -> str:
        """Render the ON clause, or a marker when keys are unknown."""
        if self.proven and self.left_column and self.right_column:
            left = self.left_column.split(".")[-1]
            right = self.right_column.split(".")[-1]
            return (
                f"{self.left_table}.{left} = {self.right_table}.{right}"
            )
        return f"/* unproven join: {self.left_table} <-> {self.right_table} */"

    def __str__(self) -> str:
        return f"{self.left_table} -> {self.right_table} [{self.edge_type}]"


@dataclass
class JoinPath:
    """A sequence of tables plus the joins connecting them."""

    tables: List[str] = field(default_factory=list)
    steps: List[JoinStep] = field(default_factory=list)
    total_cost: float = 0.0

    @property
    def all_proven(self) -> bool:
        return all(s.proven for s in self.steps)

    def to_join_clauses(self) -> List[str]:
        return [s.to_sql() for s in self.steps]


class JoinPathFinder:
    """Finds join paths between warehouse tables."""

    def __init__(
        self,
        schema: SchemaGraph,
        engine: Optional[GraphEngine] = None,
    ):
        """
        :param schema: Schema Graph from :class:`SchemaGraphBuilder`.
        :param engine: Graph compute engine used for the path search.
                       Defaults to an in-process :class:`LocalEngine`.
        """
        self._schema = schema
        # Kept locally regardless of engine: cost lookup and edge-type
        # attribution are metadata reads over a table-level projection, not
        # traversals, so there is nothing to distribute.
        self._join_graph = to_join_graph(schema)
        self._engine = engine if engine is not None else LocalEngine(schema)
        self._fk_index = self._build_fk_index()

    @property
    def engine(self) -> GraphEngine:
        """The engine actually running the path search."""
        return self._engine

    # ---- public API (individually callable) ----

    def shortest_path(self, source: str, target: str) -> Optional[JoinPath]:
        """Cheapest join path between two tables, or ``None``."""
        node_ids = self._engine.shortest_join_path(
            f"table:{source}", f"table:{target}"
        )
        if not node_ids:
            log.warning("no join path between %s and %s", source, target)
            return None
        return self._to_path(node_ids)

    def connect(self, tables: List[str]) -> Optional[JoinPath]:
        """Span several tables with a single connected subgraph."""
        node_ids = [f"table:{t}" for t in tables]
        path_nodes, cost = self._engine.steiner_join_tree(node_ids)
        if not path_nodes:
            log.warning("no join tree spanning %s", tables)
            return None
        path = self._to_path(path_nodes)
        if path is not None:
            path.total_cost = cost
        return path

    def domains(
        self, resolution: float = 1.0, algorithm: str = "louvain"
    ) -> Dict[str, List[str]]:
        """Group tables into subject domains via community detection.

        Two tables in different domains almost never belong in the same query;
        surfacing that lets a caller either narrow the candidate set before
        linking, or flag a question that appears to span unrelated domains.

        :returns: domain id (as a string) -> sorted table names.
        """
        labels = self._engine.communities(
            resolution=resolution, algorithm=algorithm
        )
        grouped: Dict[str, List[str]] = {}
        for node_id, cid in labels.items():
            grouped.setdefault(str(cid), []).append(node_id.split(":", 1)[-1])
        return {cid: sorted(names) for cid, names in sorted(grouped.items())}

    # ---- internals ----

    def _build_fk_index(self) -> Dict[Tuple[str, str], Tuple[str, str]]:
        """Map (table_a, table_b) -> (column_a, column_b) for declared FKs."""
        index: Dict[Tuple[str, str], Tuple[str, str]] = {}
        table_of = self._column_to_table()
        for edge in self._schema.edges:
            if edge.edge_type != EdgeType.FOREIGN_KEY:
                continue
            src_table = table_of.get(edge.source)
            dst_table = table_of.get(edge.target)
            if not src_table or not dst_table:
                continue
            index[_ordered(src_table, dst_table)] = (edge.source, edge.target)
        return index

    def _column_to_table(self) -> Dict[str, str]:
        table_of: Dict[str, str] = {}
        for edge in self._schema.edges:
            if edge.edge_type == EdgeType.BELONGS_TO:
                table_of[edge.source] = edge.target
        return table_of

    def _to_path(self, node_ids: List[str]) -> JoinPath:
        tables = [n.split(":", 1)[1] for n in node_ids]
        steps: List[JoinStep] = []
        total = 0.0
        for left, right in zip(node_ids, node_ids[1:]):
            cost = self._edge_cost(left, right)
            total += cost
            edge_type = self._edge_type(left, right)
            left_col, right_col = self._fk_index.get(_ordered(left, right), ("", ""))
            proven = bool(left_col and right_col)
            steps.append(
                JoinStep(
                    left_table=left.split(":", 1)[1],
                    right_table=right.split(":", 1)[1],
                    left_column=left_col.split(":", 1)[-1].split(".", 1)[-1]
                    if left_col else "",
                    right_column=right_col.split(":", 1)[-1].split(".", 1)[-1]
                    if right_col else "",
                    edge_type=edge_type,
                    cost=cost,
                    proven=proven,
                )
            )
        return JoinPath(tables=tables, steps=steps, total_cost=total)

    def _edge_cost(self, left: str, right: str) -> float:
        data = self._join_graph.get_edge_data(left, right)
        return float(data.get("weight", 1.0)) if data else 1.0

    def _edge_type(self, left: str, right: str) -> str:
        if _ordered(left, right) in self._fk_index:
            return EdgeType.FOREIGN_KEY.value
        # Distinguish lineage vs co-occurrence for caller confidence.
        for edge in self._schema.edges:
            if edge.edge_type == EdgeType.LINEAGE:
                if _ordered(edge.source, edge.target) == _ordered(left, right):
                    return EdgeType.LINEAGE.value
        return EdgeType.CO_OCCUR.value


def _ordered(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)
