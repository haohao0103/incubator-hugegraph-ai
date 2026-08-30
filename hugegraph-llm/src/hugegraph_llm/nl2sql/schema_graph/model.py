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
Data model for the warehouse Schema Graph.

The graph is the substrate for graph-enhanced Text2SQL. It is deliberately
**not** a knowledge graph in the LLM-extraction sense: every node and edge
comes from deterministic warehouse metadata (catalog, lineage, foreign keys,
query logs), so the mapping stays auditable.

Node types
----------
Table
    A physical table / view in the warehouse.
Column
    A column belonging to a Table.
Term
    A business term (e.g. "营收") that maps to one or more columns or a
    metric expression.

Edge types
----------
BELONGS_TO
    Column -> Table. Structural ownership.
FOREIGN_KEY
    Column -> Column. Declared referential integrity; the strongest join hint.
LINEAGE
    Table -> Table (upstream -> downstream). From the lineage system.
CO_OCCUR
    Table <-> Table. Mined from historical query logs (weighted by frequency).
TERM_MAPS
    Term -> Column. Business vocabulary to physical column binding.

Why both strong and weak edges matter: strong edges (FK, lineage) give
*correct* join paths, while weak edges (co-occurrence) give *empirically
useful* ones — including associations no data engineer ever declared.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    """Vertex kinds in the Schema Graph."""

    TABLE = "table"
    COLUMN = "column"
    TERM = "term"


class EdgeType(str, Enum):
    """Edge kinds in the Schema Graph.

    ``JOIN_WEIGHT`` maps each type to how strongly it implies a valid join.
    """

    BELONGS_TO = "belongs_to"
    FOREIGN_KEY = "foreign_key"
    LINEAGE = "lineage"
    CO_OCCUR = "co_occur"
    TERM_MAPS = "term_maps"


# Stronger edges are cheaper to traverse when searching for join paths.
# A foreign key is a declared, always-valid join; co-occurrence is a hint
# that may or may not be joinable, so it carries a penalty.
EDGE_JOIN_WEIGHT: Dict[EdgeType, float] = {
    EdgeType.FOREIGN_KEY: 1.0,
    EdgeType.LINEAGE: 1.5,
    EdgeType.CO_OCCUR: 3.0,
    EdgeType.BELONGS_TO: 0.5,
    EdgeType.TERM_MAPS: float("inf"),  # not traversable for joins
}


@dataclass(frozen=True)
class Node:
    """A vertex in the Schema Graph.

    ``node_id`` is namespaced by type, e.g. ``table:dw.orders`` or
    ``column:dw.orders.amount``, so that ids stay unique across types.
    """

    node_id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.node_type.value


@dataclass(frozen=True)
class Edge:
    """An edge in the Schema Graph."""

    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def join_cost(self) -> float:
        """Cost of traversing this edge when building a join path.

        Derived from the edge type and weight: a foreign key costs little,
        a weak co-occurrence costs more. ``inf`` for non-joinable edges.
        """
        base = EDGE_JOIN_WEIGHT[self.edge_type]
        if base == float("inf"):
            return float("inf")
        return base / max(self.weight, 1e-9)


@dataclass
class Table:
    """Warehouse table metadata used to build a :class:`Node`."""

    name: str
    database: str = ""
    comment: str = ""
    row_count: int = 0
    is_fact: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.database}.{self.name}" if self.database else self.name

    def to_node(self) -> Node:
        props = {
            "database": self.database,
            "comment": self.comment,
            "row_count": self.row_count,
            "is_fact": self.is_fact,
        }
        props.update(self.properties)
        return Node(
            node_id=f"table:{self.full_name}",
            node_type=NodeType.TABLE,
            name=self.name,
            properties=props,
        )


@dataclass
class Column:
    """Warehouse column metadata used to build a :class:`Node`."""

    name: str
    table: str
    data_type: str = ""
    comment: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.table}.{self.name}"

    def to_node(self) -> Node:
        props = {
            "table": self.table,
            "data_type": self.data_type,
            "comment": self.comment,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
        }
        props.update(self.properties)
        return Node(
            node_id=f"column:{self.full_name}",
            node_type=NodeType.COLUMN,
            name=self.name,
            properties=props,
        )


@dataclass
class Term:
    """A business term bound to physical columns or a metric expression."""

    name: str
    aliases: List[str] = field(default_factory=list)
    expression: str = ""
    comment: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_node(self) -> Node:
        props = {
            "aliases": list(self.aliases),
            "expression": self.expression,
            "comment": self.comment,
        }
        props.update(self.properties)
        return Node(
            node_id=f"term:{self.name}",
            node_type=NodeType.TERM,
            name=self.name,
            properties=props,
        )


@dataclass
class SchemaGraph:
    """An in-memory Schema Graph, independent of any storage backend.

    Kept as plain dicts so it can be materialised into networkx, HugeGraph,
    or anything else by the corresponding backend.
    """

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(
                f"edge endpoint missing: {edge.source} -> {edge.target}"
            )
        self.edges.append(edge)

    def tables(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.TABLE]

    def columns(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.COLUMN]

    def terms(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.TERM]

    def edges_of_type(self, edge_type: EdgeType) -> List[Edge]:
        return [e for e in self.edges if e.edge_type == edge_type]
