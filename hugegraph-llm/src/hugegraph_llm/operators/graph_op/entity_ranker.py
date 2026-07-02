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

"""Entity ranking: global PageRank and Personalized PageRank for KG entities.

Provides backend-agnostic scoring for entities in a knowledge graph. The core
algorithm is the standard PageRank power iteration, extended to support:

- Edge weights (relationship strength / confidence)
- Dangling-node handling (uniform teleport)
- Personalized PageRank with multiple weighted source nodes
- Cached global PageRank for repeated scoring

Design references:
    - HippoRAG2: PageRank-based entity ranking for multi-hop reasoning
    - LightRAG: entity/relationship importance attributes
    - MS-GraphRAG: entity ranking in graph construction and retrieval
    - Fast-GraphRAG: sparse-matrix propagation and ranking policies

The module is intentionally backend-agnostic: all graph data is loaded through
injected callables, so it can be used with HugeGraph, NetworkX, or in-memory
mock graphs for testing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from hugegraph_llm.utils.log import log


logger = logging.getLogger(__name__)


DEFAULT_ALPHA: float = 0.15
DEFAULT_EPSILON: float = 1e-6
DEFAULT_MAX_ITERATIONS: int = 100
DEFAULT_DANGLING_TELEPORT: bool = True


@dataclass
class EntityRankerConfig:
    """Configuration for PageRank / PPR computation."""

    # Damping / restart probability (1 - alpha = continuation probability)
    alpha: float = DEFAULT_ALPHA

    # Convergence threshold for power iteration (L1 norm of score change)
    epsilon: float = DEFAULT_EPSILON

    # Maximum number of power iterations
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    # If True, dangling nodes (no out-neighbors) teleport uniformly instead of
    # losing their probability mass. This is the standard PageRank behavior.
    dangling_teleport: bool = DEFAULT_DANGLING_TELEPORT

    # If True, treat edges as undirected (add reverse edges). Useful for KG
    # where many relationships are semantically symmetric.
    undirected: bool = False

    # Name of edge property used for weight. If the graph loader does not
    # provide weights, all edges have weight 1.0.
    weight_property: str = "weight"

    # Whether to normalize scores to [0, 1] after computation.
    normalize_scores: bool = True

    # Cache TTL for global PageRank in seconds (None = no expiration).
    global_cache_ttl_seconds: Optional[float] = None


@dataclass
class EntityRankerResult:
    """Result of a PageRank or PPR computation."""

    scores: Dict[str, float] = field(default_factory=dict)
    num_iterations: int = 0
    converged: bool = False
    final_delta: float = 0.0
    source_weights: Dict[str, float] = field(default_factory=dict)

    def get_score(self, entity_id: str, default: float = 0.0) -> float:
        """Return score for a single entity, with default fallback."""
        return self.scores.get(entity_id, default)

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k entities sorted by score descending."""
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:k]


Edge = Tuple[str, str, float]
"""Type alias for a weighted edge (source, target, weight)."""


class EntityRanker:
    """Backend-agnostic entity ranker using PageRank / PPR.

    The ranker loads graph edges via an injected callable, builds an adjacency
    structure, and runs power-iteration PageRank. The result can be used as
    a precomputed entity score in ``KGSearchRetriever`` or as a standalone
    ranking signal.

    Usage::

        def load_edges():
            return [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]

        ranker = EntityRanker(edges_loader=load_edges)
        result = ranker.compute_global_pagerank()
        print(result.top_k(3))

        # Use as entity score function in KGSearchRetriever
        retriever = KGSearchRetriever(
            entity_ranker=ranker,
        )

    Args:
        edges_loader: Callable that returns a list of weighted edges
            ``[(source_id, target_id, weight), ...]``. If weights are omitted
            by the caller, the adapter should return 1.0 for each edge.
        config: ``EntityRankerConfig`` with algorithm parameters.
    """

    def __init__(
        self,
        edges_loader: Optional[Callable[[], List[Edge]]] = None,
        config: Optional[EntityRankerConfig] = None,
    ) -> None:
        """Initialize the entity ranker."""
        self._edges_loader = edges_loader
        self.config = config or EntityRankerConfig()

        # Cached global PageRank result (computed once and reused for scoring)
        self._global_pagerank: Optional[EntityRankerResult] = None

        # Adjacency structure built from last edge load. Keys are node IDs.
        self._adjacency: Dict[str, Dict[str, float]] = {}
        self._nodes: Set[str] = set()
        self._out_weights: Dict[str, float] = {}

    def load_graph(self) -> None:
        """Force (re)load the graph structure from the injected loader.

        This is called automatically on first scoring if not already loaded.
        """
        if self._edges_loader is None:
            self._adjacency = {}
            self._nodes = set()
            self._out_weights = {}
            return

        edges = self._edges_loader()
        self._build_adjacency(edges)

    def _build_adjacency(self, edges: List[Edge]) -> None:
        """Build adjacency structure from weighted edges."""
        adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        nodes: Set[str] = set()

        for edge in edges:
            if len(edge) < 2:
                continue
            src = edge[0]
            tgt = edge[1]
            weight = float(edge[2]) if len(edge) >= 3 else 1.0

            if not src or not tgt:
                continue

            nodes.add(src)
            nodes.add(tgt)
            adj[src][tgt] += weight

            if self.config.undirected:
                adj[tgt][src] += weight

        # Ensure every node has an entry even if it has no outgoing edges
        for node in nodes:
            adj.setdefault(node, defaultdict(float))

        self._adjacency = {nid: dict(neighbors) for nid, neighbors in adj.items()}
        self._nodes = nodes
        self._out_weights = {
            nid: sum(neighbors.values()) for nid, neighbors in self._adjacency.items()
        }

    def _ensure_graph_loaded(self) -> None:
        """Load graph if not already loaded."""
        if not self._nodes and self._edges_loader is not None:
            self.load_graph()

    def compute_global_pagerank(
        self,
        force_refresh: bool = False,
    ) -> EntityRankerResult:
        """Compute global PageRank over all entities in the graph.

        The result is cached; subsequent calls return the cached result unless
        ``force_refresh`` is True.

        Args:
            force_refresh: If True, recompute even if cached result exists.

        Returns:
            EntityRankerResult with scores and convergence metadata.
        """
        if not force_refresh and self._global_pagerank is not None:
            return self._global_pagerank

        self._ensure_graph_loaded()
        result = self._power_iteration_pagerank(
            source_weights={},
            alpha=self.config.alpha,
            epsilon=self.config.epsilon,
            max_iterations=self.config.max_iterations,
        )
        self._global_pagerank = result
        return result

    def compute_ppr(
        self,
        source_weights: Dict[str, float],
    ) -> EntityRankerResult:
        """Compute Personalized PageRank from one or more source entities.

        Args:
            source_weights: Mapping from source entity ID to initial weight.
                Weights are normalized internally to form a probability vector.
                If empty, falls back to uniform teleport (global PageRank).

        Returns:
            EntityRankerResult with PPR scores and convergence metadata.
        """
        self._ensure_graph_loaded()
        return self._power_iteration_pagerank(
            source_weights=source_weights,
            alpha=self.config.alpha,
            epsilon=self.config.epsilon,
            max_iterations=self.config.max_iterations,
        )

    def score(self, entity_id: str) -> float:
        """Return the global PageRank score for an entity.

        This method signature matches the ``entity_score_func`` expected by
        ``KGSearchRetriever``.

        Args:
            entity_id: Entity ID to score.

        Returns:
            PageRank score in [0, 1], or 0.0 if the entity is not in the graph.
        """
        global_result = self.compute_global_pagerank()
        return global_result.get_score(entity_id, default=0.0)

    def score_personalized(
        self,
        entity_id: str,
        source_weights: Dict[str, float],
    ) -> float:
        """Return PPR score for an entity relative to given source weights.

        Args:
            entity_id: Entity ID to score.
            source_weights: Mapping from source entity ID to weight.

        Returns:
            PPR score in [0, 1], or 0.0 if not reachable.
        """
        ppr_result = self.compute_ppr(source_weights)
        return ppr_result.get_score(entity_id, default=0.0)

    def _power_iteration_pagerank(
        self,
        source_weights: Dict[str, float],
        alpha: float,
        epsilon: float,
        max_iterations: int,
    ) -> EntityRankerResult:
        """Core power iteration for PageRank / PPR.

        Solves: r = alpha * v + (1 - alpha) * M^T * r

        where:
            - v is the teleport vector (uniform for global PR, personalized for PPR)
            - M is the row-stochastic transition matrix (column-stochastic in M^T)

        Args:
            source_weights: If non-empty, used as personalization vector.
            alpha: Restart probability.
            epsilon: Convergence tolerance.
            max_iterations: Maximum iterations.

        Returns:
            EntityRankerResult with scores and metadata.
        """
        nodes = sorted(self._nodes)
        n = len(nodes)

        if n == 0:
            return EntityRankerResult(
                scores={},
                num_iterations=0,
                converged=True,
                final_delta=0.0,
                source_weights=source_weights,
            )

        node_to_idx = {nid: i for i, nid in enumerate(nodes)}

        # Build teleport vector
        teleport = self._build_teleport_vector(nodes, source_weights)

        # Initialize scores uniformly
        scores = np.full(n, 1.0 / n, dtype=np.float64)

        one_minus_alpha = 1.0 - alpha

        # Precompute out-degree weighted sums for fast transition
        out_sums = np.array(
            [self._out_weights.get(nid, 0.0) for nid in nodes],
            dtype=np.float64,
        )

        converged = False
        final_delta = 0.0
        num_iterations = 0

        for iteration in range(max_iterations):
            new_scores = alpha * teleport.copy()

            # Contribution from each node u: (1 - alpha) * score[u] / out_weight[u]
            # distributed to neighbors v
            for u_idx, u in enumerate(nodes):
                out_sum = out_sums[u_idx]
                if out_sum > 0:
                    contribution = one_minus_alpha * scores[u_idx] / out_sum
                    for v, weight in self._adjacency.get(u, {}).items():
                        v_idx = node_to_idx.get(v)
                        if v_idx is not None:
                            new_scores[v_idx] += contribution * weight
                elif self.config.dangling_teleport:
                    # Dangling node: teleport uniformly (standard PageRank)
                    new_scores += one_minus_alpha * scores[u_idx] / n

            delta = np.sum(np.abs(new_scores - scores))
            scores = new_scores
            num_iterations = iteration + 1
            final_delta = delta

            if delta < epsilon:
                converged = True
                break

        # Normalize scores if requested
        if self.config.normalize_scores:
            max_score = float(np.max(scores)) if n > 0 else 0.0
            if max_score > 0:
                scores = scores / max_score

        score_dict = {
            nid: float(scores[i]) for i, nid in enumerate(nodes) if scores[i] > 1e-14
        }

        return EntityRankerResult(
            scores=score_dict,
            num_iterations=num_iterations,
            converged=converged,
            final_delta=final_delta,
            source_weights=source_weights,
        )

    def _build_teleport_vector(
        self,
        nodes: List[str],
        source_weights: Dict[str, float],
    ) -> np.ndarray:
        """Build the teleport / personalization vector."""
        n = len(nodes)
        teleport = np.full(n, 1.0 / n, dtype=np.float64)

        if not source_weights:
            return teleport

        # Filter to known nodes and normalize
        filtered: Dict[str, float] = {}
        total = 0.0
        for nid, weight in source_weights.items():
            if nid in self._nodes and weight > 0:
                filtered[nid] = weight
                total += weight

        if total <= 0:
            return teleport

        node_to_idx = {nid: i for i, nid in enumerate(nodes)}
        teleport = np.zeros(n, dtype=np.float64)
        for nid, weight in filtered.items():
            idx = node_to_idx[nid]
            teleport[idx] = weight / total

        return teleport

    def get_nodes(self) -> Set[str]:
        """Return the set of nodes in the loaded graph."""
        self._ensure_graph_loaded()
        return set(self._nodes)

    def get_edge_count(self) -> int:
        """Return number of edges in the loaded graph."""
        self._ensure_graph_loaded()
        return sum(len(neighbors) for neighbors in self._adjacency.values())

    def reset_cache(self) -> None:
        """Clear cached global PageRank result."""
        self._global_pagerank = None


class HugeGraphEntityRankerAdapter:
    """Adapter to build an ``EntityRanker`` backed by HugeGraph.

    Loads edges from HugeGraph via a PyHugeClient-compatible client and returns
    weighted edges for the ranker.

    Usage::

        from pyhugegraph.client import PyHugeClient
        from hugegraph_llm.utils.hugegraph_utils import get_hg_client

        client = get_hg_client()
        adapter = HugeGraphEntityRankerAdapter(client)
        ranker = adapter.build_ranker()
        result = ranker.compute_global_pagerank()
    """

    def __init__(
        self,
        graph_client: Any,
        edge_labels: Optional[List[str]] = None,
        weight_property: str = "weight",
        default_weight: float = 1.0,
    ) -> None:
        """Initialize the adapter.

        Args:
            graph_client: PyHugeClient or compatible client.
            edge_labels: Optional list of edge labels to include. If None,
                all edges are fetched (subject to client limit).
            weight_property: Edge property name used for weight. If the
                property is missing, ``default_weight`` is used.
            default_weight: Weight for edges without the weight property.
        """
        self._client = graph_client
        self._edge_labels = edge_labels
        self._weight_property = weight_property
        self._default_weight = default_weight

    def _load_edges(self) -> List[Edge]:
        """Fetch edges from HugeGraph and convert to weighted edge tuples."""
        if self._client is None:
            return []

        edges: List[Dict[str, Any]] = []
        try:
            if self._edge_labels:
                for label in self._edge_labels:
                    fetched = self._client.getEdgeByCondition(
                        edge_label=label, limit=10000
                    )
                    if fetched:
                        edges.extend(fetched)
            else:
                # Attempt to fetch all edges; depends on client capability
                fetched = self._client.getEdgeByCondition(limit=10000)
                if fetched:
                    edges.extend(fetched)
        except Exception as e:
            log.warning("[EntityRanker] Failed to load edges from HugeGraph: %s", e)
            return []

        result: List[Edge] = []
        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if not src or not tgt:
                continue

            props = edge.get("properties", {}) or {}
            weight = self._default_weight
            if self._weight_property in props:
                try:
                    weight = float(props[self._weight_property])
                except (TypeError, ValueError):
                    weight = self._default_weight

            result.append((src, tgt, weight))

        return result

    def build_ranker(
        self,
        config: Optional[EntityRankerConfig] = None,
    ) -> EntityRanker:
        """Build an EntityRanker instance backed by HugeGraph edges."""
        config = config or EntityRankerConfig()
        return EntityRanker(edges_loader=self._load_edges, config=config)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_ranker_from_edges(
    edges: List[Edge],
    config: Optional[EntityRankerConfig] = None,
) -> EntityRanker:
    """Build an EntityRanker from an in-memory edge list.

    Useful for unit tests and offline ad-hoc usage.

    Args:
        edges: List of (source, target, weight) tuples.
        config: Optional EntityRankerConfig.

    Returns:
        Configured EntityRanker with graph already loaded.
    """
    ranker = EntityRanker(config=config)
    ranker._build_adjacency(edges)  # pylint: disable=protected-access
    return ranker
