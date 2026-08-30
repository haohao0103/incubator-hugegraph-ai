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
Graph compute engine abstraction for the NL2SQL stack.

Everything the NL2SQL layers need from a graph engine is four primitives:

``personalized_pagerank(seeds, alpha)``
    Relevance propagation over the *full* schema graph (tables + columns +
    terms). Drives Schema Linking.

``shortest_join_path(source, target)``
    Cheapest table-to-table path over the *join projection*. Drives two-table
    join discovery.

``steiner_join_tree(terminals)``
    Connected subgraph spanning several tables. Drives multi-table join
    discovery.

``communities(resolution, algorithm)``
    Table clustering over the join projection — subject-domain partitioning of
    the warehouse. Useful for narrowing the candidate set before linking, and
    for detecting that a question spans unrelated domains.

Two implementations ship:

- :class:`~hugegraph_llm.nl2sql.engine.local.LocalEngine` (default) runs
  everything in-process on networkx. No external service, no setup cost.
- :class:`~hugegraph_llm.nl2sql.engine.vermeer.VermeerEngine` offloads the
  O(V+E) numeric work to a Vermeer cluster (Go, BSP, distributed), for
  catalogs where the Python path stops being viable.

The abstraction exists so that upstream platforms can keep calling the same
``link`` / ``join_path`` / ``connect_tables`` API and swap the engine
underneath. It also makes the *differences* between engines explicit rather
than folkloric — see :class:`EngineCapabilities`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Join costs are "lower is closer" (a declared foreign key costs 1.0, a mined
# co-occurrence costs 3.0). Community detection wants the opposite polarity:
# modularity treats the edge weight as *affinity*, so a tightly coupled pair
# must carry the larger number. Both engines must agree on the conversion or
# their community results are not comparable, hence one function here.
_MIN_COST = 1e-9
_MAX_AFFINITY = 1e9


def affinity_of(cost: float) -> float:
    """Convert a join cost into a modularity-style affinity weight."""
    try:
        c = float(cost)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    if c <= _MIN_COST:
        return _MAX_AFFINITY
    if c == float("inf"):
        return 0.0
    return 1.0 / c


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can and cannot do, stated up front.

    Callers that care about numeric fidelity (for example, an A/B comparison
    between engines) should read this instead of assuming parity.
    """

    #: Engine identifier, e.g. ``"local"`` / ``"vermeer"``.
    name: str
    #: True when work is pushed to an external cluster.
    distributed: bool = False
    #: True when PPR honours schema-graph edge weights. Vermeer's ``ppr``
    #: operator has no edge-weight parameter, so it treats every edge as 1.0;
    #: results match the local engine exactly only on uniform-weight graphs.
    weighted_ppr: bool = True
    #: True when shortest-path / Steiner search honours join costs.
    weighted_paths: bool = True
    #: Community-detection algorithms this engine accepts.
    community_algorithms: Tuple[str, ...] = ("louvain",)

    def supports_community(self, algorithm: str) -> bool:
        return algorithm in self.community_algorithms


class GraphEngine(ABC):
    """Graph primitives required by the NL2SQL layers."""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Declared behaviour of this engine."""

    @property
    def name(self) -> str:
        return self.capabilities.name

    # ---- L1: schema linking ----

    @abstractmethod
    def personalized_pagerank(
        self, seeds: Dict[str, float], alpha: float = 0.85
    ) -> Dict[str, float]:
        """PPR over the full schema graph.

        :param seeds: node id -> teleport weight. Weights need not be
                      normalised; implementations normalise internally so the
                      returned scores sum to ~1.
        :param alpha: damping factor.
        :returns: node id -> score. Empty dict when there is nothing to seed.
        """

    # ---- L2: join paths ----

    @abstractmethod
    def shortest_join_path(
        self, source: str, target: str
    ) -> Optional[List[str]]:
        """Cheapest table path, as join-graph node ids (``table:db.name``).

        :returns: ``[source, ..., target]``, or ``None`` when disconnected.
        """

    @abstractmethod
    def steiner_join_tree(
        self, terminals: List[str]
    ) -> Tuple[List[str], float]:
        """Approximate Steiner tree spanning ``terminals``.

        :returns: ``(node_ids, total_cost)``; node ids empty when the
                  terminals cannot all be connected.
        """

    # ---- L3: subject domains ----

    @abstractmethod
    def communities(
        self, resolution: float = 1.0, algorithm: str = "louvain"
    ) -> Dict[str, int]:
        """Cluster the join graph into subject domains.

        :returns: table node id -> community id.
        """

    # ---- lifecycle ----

    def close(self) -> None:
        """Release any external resources. No-op for in-process engines."""

    def __enter__(self) -> "GraphEngine":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
