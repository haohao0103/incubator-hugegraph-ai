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
Materialise a :class:`SchemaGraph` into a runnable graph for algorithms.

Two projections are provided because the two consumers need different views:

``to_networkx(graph)``
    Full graph (tables + columns + terms) used by Schema Linking, which has
    to reason about *columns* and business *terms*.

``to_join_graph(graph)``
    Table-only projection used by join-path search. Column-level foreign keys
    are folded into table-level edges, because a SQL JOIN connects tables,
    not columns. Edge weight is the join cost, so cheaper paths win.

The join projection is where HugeGraph would take over for very large
catalogs (millions of columns), with the same node/edge vocabulary defined in
``model.py``; the networkx path keeps the module runnable and testable with
no external services.
"""

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from hugegraph_llm.utils.log import log

from .model import EdgeType, NodeType, SchemaGraph


def to_networkx(graph: SchemaGraph) -> nx.Graph:
    """Project the full Schema Graph (tables, columns, terms) into networkx."""
    g = nx.Graph()
    for node_id, node in graph.nodes.items():
        g.add_node(node_id, node_type=node.node_type.value, name=node.name,
                   **node.properties)
    for edge in graph.edges:
        g.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            join_cost=edge.join_cost,
        )
    return g


def to_join_graph(graph: SchemaGraph) -> nx.Graph:
    """Project a table-only graph whose edge weights are join costs.

    Foreign keys are declared column-to-column, so they are folded into
    table-to-table edges: ``dw.orders.user_id -> dw.users.id`` becomes an edge
    between ``dw.orders`` and ``dw.users``.
    """
    g = nx.Graph()
    for node in graph.tables():
        g.add_node(node.node_id, name=node.name, **node.properties)

    best_cost: Dict[Tuple[str, str], float] = {}

    def _table_of(column_id: str) -> Optional[str]:
        for edge in graph.edges:
            if edge.edge_type == EdgeType.BELONGS_TO and edge.source == column_id:
                return edge.target
        return None

    for edge in graph.edges:
        if edge.edge_type == EdgeType.FOREIGN_KEY:
            src_table = _table_of(edge.source)
            dst_table = _table_of(edge.target)
            if src_table is None or dst_table is None or src_table == dst_table:
                continue
            key = _ordered(src_table, dst_table)
            cost = edge.join_cost
            if key not in best_cost or cost < best_cost[key]:
                best_cost[key] = cost
        elif edge.edge_type in (EdgeType.LINEAGE, EdgeType.CO_OCCUR):
            key = _ordered(edge.source, edge.target)
            cost = edge.join_cost
            if key not in best_cost or cost < best_cost[key]:
                best_cost[key] = cost

    for (src, dst), cost in best_cost.items():
        if src in g.nodes and dst in g.nodes and math.isfinite(cost):
            g.add_edge(src, dst, weight=cost)

    log.debug("join graph: %s tables, %s edges", g.number_of_nodes(),
              g.number_of_edges())
    return g


def _ordered(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def shortest_join_path(
    join_graph: nx.Graph, source: str, target: str
) -> Optional[List[str]]:
    """Cheapest table path between two tables (Dijkstra on join cost).

    Returns the list of table node ids, or ``None`` when disconnected.
    """
    if source == target:
        return [source]
    if source not in join_graph or target not in join_graph:
        return None
    try:
        return nx.shortest_path(join_graph, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def steiner_join_tree(
    join_graph: nx.Graph, tables: List[str]
) -> Tuple[List[str], float]:
    """Approximate Steiner tree connecting several tables.

    Exact Steiner tree is NP-hard, so this uses the standard metric-closure +
    MST approximation: build the metric closure over the required terminals,
    take its minimum spanning tree, then expand tree edges back into original
    paths. Guarantees a solution within 2x of optimal.

    Returns ``(path_nodes, total_cost)``; ``path_nodes`` is empty when the
    terminals are not all connected.
    """
    terminals = [t for t in tables if t in join_graph]
    if not terminals:
        return [], 0.0
    if len(terminals) == 1:
        return terminals, 0.0

    # Every terminal must be reachable from the others, otherwise no single
    # connected subgraph spans them. Reachability is checked over the *whole*
    # join graph, not the subgraph induced by the terminals: the entire point
    # of a Steiner tree is that it may route through intermediate tables, so
    # `orders -> users -> cities` is a valid answer for terminals
    # {orders, cities} even though those two share no direct edge.
    component = nx.node_connected_component(join_graph, terminals[0])
    unreachable = set(terminals) - component
    if unreachable:
        log.debug("steiner terminals unreachable from %s: %s",
                  terminals[0], sorted(unreachable))
        return [], 0.0

    # networkx's steiner_tree misbehaves when the graph carries nodes outside the
    # terminals' connected component (e.g. an isolated table with degree 0): it
    # indexes shortest-path results by every node and raises KeyError. Restrict
    # to the component that actually holds the terminals before calling it.
    subg = join_graph.subgraph(component)

    try:
        from networkx.algorithms.approximation import steiner_tree
        tree = steiner_tree(subg, list(terminals), weight="weight")
    except (nx.NetworkXError, nx.NetworkXNoPath, KeyError):
        return [], 0.0

    # Cost is summed over the *expanded, deduplicated* edges of the tree, not
    # over the metric closure. Two reasons: networkx returns original graph
    # edges (carrying "weight", never the "distance" of the closure), and a
    # closure sum double-counts any segment shared by two legs — which would
    # overstate the price of a JOIN that is only executed once.
    nodes: List[str] = []
    seen_nodes = set()
    seen_edges = set()
    total_cost = 0.0
    for u, v in tree.edges():
        path = nx.shortest_path(join_graph, u, v, weight="weight")
        for a, b in zip(path, path[1:]):
            key = _ordered(a, b)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            total_cost += float(join_graph[a][b].get("weight", 1.0))
        for n in path:
            if n not in seen_nodes:
                seen_nodes.add(n)
                nodes.append(n)
    return nodes, total_cost
