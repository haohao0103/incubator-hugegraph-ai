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
In-process graph engine backed by networkx.

This is the default and the reference implementation: no external service, no
cluster to operate, and it is what the test suite pins behaviour against. It
stays viable well past toy scale — the point at which it stops being viable is
Python's per-iteration cost over millions of edges, which is exactly where
:class:`~hugegraph_llm.nl2sql.engine.vermeer.VermeerEngine` takes over.
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx

from hugegraph_llm.utils.log import log

from ..schema_graph.backend import (
    shortest_join_path,
    steiner_join_tree,
    to_join_graph,
    to_networkx,
)
from ..schema_graph.model import SchemaGraph
from .base import EngineCapabilities, GraphEngine, affinity_of


class LocalEngine(GraphEngine):
    """Runs every graph primitive in-process on networkx."""

    def __init__(self, schema: SchemaGraph):
        self._schema = schema
        self._graph = to_networkx(schema)
        self._join_graph = to_join_graph(schema)

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="local",
            distributed=False,
            weighted_ppr=True,
            weighted_paths=True,
            community_algorithms=("louvain",),
        )

    # ---- exposed projections (used by callers that need adjacency) ----

    @property
    def graph(self) -> nx.Graph:
        """Full schema graph projection (tables + columns + terms)."""
        return self._graph

    @property
    def join_graph(self) -> nx.Graph:
        """Table-only projection whose edge weights are join costs."""
        return self._join_graph

    # ---- L1 ----

    def personalized_pagerank(
        self, seeds: Dict[str, float], alpha: float = 0.85
    ) -> Dict[str, float]:
        if self._graph.number_of_nodes() == 0:
            return {}
        personalization = {n: 0.0 for n in self._graph.nodes()}
        for node_id, weight in seeds.items():
            if node_id in personalization:
                personalization[node_id] = weight
        if not any(personalization.values()):
            return {}
        return nx.pagerank(
            self._graph,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
        )

    # ---- L2 ----

    def shortest_join_path(
        self, source: str, target: str
    ) -> Optional[List[str]]:
        return shortest_join_path(self._join_graph, source, target)

    def steiner_join_tree(
        self, terminals: List[str]
    ) -> Tuple[List[str], float]:
        return steiner_join_tree(self._join_graph, terminals)

    # ---- L3 ----

    def communities(
        self, resolution: float = 1.0, algorithm: str = "louvain"
    ) -> Dict[str, int]:
        if algorithm != "louvain":
            raise ValueError(
                f"local engine supports louvain only, got {algorithm!r}"
            )
        if self._join_graph.number_of_nodes() == 0:
            return {}
        affinity = _affinity_graph(self._join_graph)
        groups = nx.community.louvain_communities(
            affinity, weight="weight", resolution=resolution, seed=42
        )
        labels: Dict[str, int] = {}
        for cid, group in enumerate(groups):
            for node in group:
                labels[node] = cid
        log.debug("local louvain: %s communities over %s tables",
                  len(groups), len(labels))
        return labels


def _affinity_graph(join_graph: nx.Graph) -> nx.Graph:
    """Flip join *costs* into modularity *affinities* (see ``affinity_of``)."""
    g = nx.Graph()
    g.add_nodes_from(join_graph.nodes())
    for u, v, data in join_graph.edges(data=True):
        g.add_edge(u, v, weight=affinity_of(data.get("weight", 1.0)))
    return g
