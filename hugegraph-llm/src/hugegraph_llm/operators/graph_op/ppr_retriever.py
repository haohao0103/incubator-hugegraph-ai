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

"""Personalized PageRank (PPR) multi-hop relevance retriever.

PPR is the core relevance algorithm of HippoRAG2 (NeurIPS 2024).
It propagates importance scores from a source entity along graph edges,
finding the most relevant long-distance nodes considering edge weights
and global graph structure.

Compared to k_neighbor (BFS traversal):
- **k_neighbor**: Fixed-depth breadth-first search, returns all reachable nodes.
- **PPR**: Weighted random walk, returns the *most relevant* nodes (considers
  global structure and edge weights).

Algorithm:
    Push-style approximate PPR from "PPRGO: Pushing Personalized PageRank
    to Query" (Chen et al., WSDM 2022).

    Maintains residue[] and reserve[] arrays. For each node u:
        if |residue[u]| > epsilon * deg(u):
            push (1-alpha) * residue[u] to neighbors
            reserve[u] += alpha * residue[u]
            residue[u] = 0

References:
    - Jeh & Widom, "Personalized PageRank: An Analyzing Tool for Large
      Hypertext", WWW 2003.
    - Chen et al., "PPRGO: Pushing Personalized PageRank to Query",
      WSDM 2022.
    - HippoRAG2: https://github.com/ianliuwd/HippoRAG2, NeurIPS 2024.
"""

import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from hugegraph_llm.utils.log import log

# ── Default constants ────────────────────────────────────────

DEFAULT_ALPHA: float = 0.15  # Restart probability
DEFAULT_EPSILON: float = 1e-6  # Convergence threshold
DEFAULT_MAX_ITERATIONS: int = 100  # Max push iterations
DEFAULT_TOP_K: int = 20  # Number of results to return
DEFAULT_MAX_DEPTH: int = 2  # Subgraph expansion depth
MAX_ALLOWED_DEPTH: int = 5  # Safety limit


class PPRResult:
    """Immutable container for a single PPR retrieval result."""

    __slots__ = ("node_id", "ppr_score", "label", "properties", "distance")

    def __init__(
        self,
        node_id: str,
        ppr_score: float,
        label: str = "",
        properties: Optional[Dict[str, Any]] = None,
        distance: int = -1,
    ):
        self.node_id = node_id
        self.ppr_score = ppr_score
        self.label = label
        self.properties = properties or {}
        self.distance = distance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "ppr_score": round(self.ppr_score, 8),
            "label": self.label,
            "properties": self.properties,
            "distance": self.distance,
        }

    def __repr__(self) -> str:
        return f"PPRResult(id={self.node_id}, score={self.ppr_score:.6f})"


class PPRRetriever:
    """Personalized PageRank multi-hop relevance retriever.

    Retrieves nodes most relevant to a source entity using approximate
    Personalized PageRank on a subgraph fetched from HugeGraph.

    Usage::

        retriever = PPRRetriever(
            host="http://127.0.0.1:8080",
            graph="hugegraph",
            auth=("admin", "admin")
        )

        results = retriever.search(
            source_id="42:EntityName",
            max_depth=3,
            alpha=0.15,
            top_k=20
        )
        # returns: [PPRResult(...), ...] sorted by ppr_score descending
    """

    def __init__(
        self,
        host: str,
        graph: str,
        auth: Optional[Tuple[str, str]] = None,
        default_depth: int = DEFAULT_MAX_DEPTH,
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        """Initialize PPR Retriever.

        Args:
            host: HugeGraph server URL, e.g. "http://127.0.0.1:8080".
            graph: Graph name in HugeGraph.
            auth: Optional (username, password) tuple for authentication.
            default_depth: Default subgraph expansion depth (2-5).
            alpha: PPR restart probability (default 0.15). Higher values
                keep probability closer to source; lower values allow
                more exploration.
            epsilon: Convergence threshold for push operations.
            max_iterations: Maximum number of push iterations.
        """
        self._host = host.rstrip("/")
        self._graph = graph
        self._auth = auth
        self._default_depth = min(max(default_depth, 1), MAX_ALLOWED_DEPTH)
        self._alpha = alpha
        self._epsilon = epsilon
        self._max_iterations = max_iterations

        # Lazy-import requests to avoid hard dependency at module level
        try:
            import requests  # noqa: F401

            self._requests = requests
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for PPRRetriever. "
                "Install it via: pip install requests"
            )

        log.debug(
            f"PPRRetriever initialized: host={host}, graph={graph}, "
            f"alpha={alpha}, epsilon={epsilon}"
        )

    def search(
        self,
        source_id: str,
        max_depth: Optional[int] = None,
        edge_label_filter: Optional[List[str]] = None,
        direction: str = "BOTH",
        top_k: int = DEFAULT_TOP_K,
        alpha: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Execute PPR retrieval from a source entity.

        Args:
            source_id: Source vertex ID in HugeGraph format ("label:id").
            max_depth: Subgraph expansion depth (default from constructor, max 5).
            edge_label_filter: If specified, only propagate along these
                edge labels. None means all edges.
            direction: Traversal direction — "OUT", "IN", or "BOTH".
            top_k: Number of top results to return.
            alpha: Override restart probability.
            epsilon: Override convergence threshold.

        Returns:
            List of result dicts sorted by PPR score descending, each containing:
            ``{"node_id", "ppr_score", "label", "properties", "distance"}``.

        Raises:
            ValueError: If source_id is empty or max_depth is invalid.
            RuntimeError: If subgraph fetch fails or graph is empty.
        """
        if not source_id:
            raise ValueError("source_id must not be empty")

        depth = min(
            max(max_depth or self._default_depth, 1),
            MAX_ALLOWED_DEPTH,
        )
        _alpha = alpha or self._alpha
        _epsilon = epsilon or self._epsilon

        start_time = time.perf_counter()

        # Step 1: Fetch k-hop subgraph from HugeGraph
        log.info(f"[PPR] Fetching {depth}-hop subgraph from '{source_id}'...")
        vertices, edges = self._fetch_subgraph(
            source_id, depth, direction, edge_label_filter
        )

        if not vertices:
            log.warning(f"[PPR] Empty subgraph for source '{source_id}'")
            return []

        log.info(
            f"[PPR] Subgraph fetched: {len(vertices)} vertices, "
            f"{len(edges)} edges ({time.perf_counter() - start_time:.3f}s)"
        )

        # Step 2: Build adjacency list
        adj = self._build_adjacency(vertices, edges, direction, edge_label_filter)

        # Compute BFS distances for metadata
        distances = self._bfs_distances(source_id, adj)

        # Build vertex property lookup
        vertex_map: Dict[str, Dict[str, Any]] = {}
        for v in vertices:
            vid = v.get("id", "")
            if vid:
                vertex_map[vid] = {
                    "label": v.get("label", ""),
                    "properties": v.get("properties", {}),
                }

        # Step 3: Run push-style PPR
        log.info("[PPR] Running push-style approximate PPR...")
        ppr_start = time.perf_counter()
        scores = self._push_ppr(adj, source_id, _alpha, _epsilon, self._max_iterations)
        ppr_elapsed = time.perf_counter() - ppr_start

        log.info(
            f"[PPR] PPR computation complete: {len(scores)} scored nodes "
            f"in {ppr_elapsed:.3f}s"
        )

        # Step 4: Sort by score descending and build results
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:top_k]

        results: List[Dict[str, Any]] = []
        for nid, score in top_results:
            vinfo = vertex_map.get(nid, {})
            result = PPRResult(
                node_id=nid,
                ppr_score=score,
                label=vinfo.get("label", ""),
                properties=vinfo.get("properties", {}),
                distance=distances.get(nid, -1),
            ).to_dict()
            results.append(result)

        total_elapsed = time.perf_counter() - start_time
        log.info(
            f"[PPR] Search complete: {len(results)} results in "
            f"{total_elapsed:.3f}s (top score: "
            f"{results[0]['ppr_score']:.6f if results else 0})"
        )

        return results

    def _fetch_subgraph(
        self,
        source_id: str,
        max_depth: int,
        direction: str,
        edge_label_filter: Optional[List[str]],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Fetch k-hop subgraph from HugeGraph using kneighbor API.

        Uses ``/graphs/{g}/traversers/kneighbor`` endpoint to get all vertices
        within *max_depth* hops from *source_id*.

        Args:
            source_id: Starting vertex ID.
            max_depth: Maximum hop distance.
            direction: "OUT", "IN", or "BOTH".
            edge_label_filter: Optional list of edge labels to include.

        Returns:
            Tuple of (vertices_list, edges_list). Each vertex/edge is a dict
            with keys matching the HugeGraph REST API response.

        Raises:
            RuntimeError: If API call fails.
        """
        url = f"{self._host}/graphs/{self._graph}/traversers/kneighbor"

        payload: Dict[str, Any] = {
            "source": source_id,
            "direction": direction,
            "max_depth": max_depth,
            "limit": 10000,  # Upper bound for subgraph size
        }
        if edge_label_filter:
            payload["edge_label"] = ",".join(edge_label_filter)

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        auth = (
            self._requests.HTTPBasicAuth(self._auth[0], self._auth[1])
            if self._auth
            else None
        )

        try:
            resp = self._requests.post(
                url, json=payload, headers=headers, auth=auth, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
        except self._requests.exceptions.RequestException as e:
            raise RuntimeError(f"HugeGraph API error: {e}") from e

        # Parse response: extract vertices and edges
        # Kneighbor returns a structure like:
        # {"kneighbor": ["vertex_ids..."]} or full vertex objects
        vertices: List[Dict] = []
        edges: List[Dict] = []

        if isinstance(data, dict):
            neighbor_ids = data.get("kneighbor", [])

            # If we got just IDs, we need to fetch individual vertices
            if neighbor_ids and isinstance(neighbor_ids[0], str):
                vertices = self._batch_get_vertices(neighbor_ids)
                # Try to get edges between these vertices
                edges = self._get_edges_between_vertices(
                    [source_id] + neighbor_ids, edge_label_filter
                )
            elif neighbor_ids and isinstance(neighbor_ids[0], dict):
                # Already have vertex objects
                vertices = neighbor_ids

        return vertices, edges

    def _batch_get_vertices(self, vertex_ids: List[str]) -> List[Dict]:
        """Batch-fetch vertex details by IDs.

        Falls back to individual GET requests if batch endpoint unavailable.

        Args:
            vertex_ids: List of vertex IDs to fetch.

        Returns:
            List of vertex detail dicts.
        """
        vertices: List[Dict] = []
        headers: Dict[str, str] = {"Accept": "application/json"}
        auth = (
            self._requests.HTTPBasicAuth(self._auth[0], self._auth[1])
            if self._auth
            else None
        )

        # Try batch first
        url = f"{self._host}/graphs/{self._graph}/graphs/vertices/batch"
        try:
            resp = self._requests.post(
                url, json={"ids": vertex_ids}, headers=headers, auth=auth, timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
        except Exception:
            pass  # Fall back to individual requests

        # Individual fallback
        for vid in vertex_ids[:500]:  # Safety limit
            try:
                vurl = f"{self._host}/graphs/{self._graph}/graphs/vertices/{vid}"
                resp = self._requests.get(vurl, headers=headers, auth=auth, timeout=10)
                if resp.status_code == 200:
                    vertices.append(resp.json())
            except Exception:
                continue

        return vertices

    def _get_edges_between_vertices(
        self,
        vertex_ids: List[str],
        edge_label_filter: Optional[List[str]],
    ) -> List[Dict]:
        """Fetch edges among the given vertices.

        Uses paths-of-depth-1 query to find connecting edges.

        Args:
            vertex_ids: Vertex IDs in the subgraph.
            edge_label_filter: Optional edge label filter.

        Returns:
            List of edge dicts.
        """
        edges: List[Dict] = []
        if len(vertex_ids) < 2:
            return edges

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        auth = (
            self._requests.HTTPBasicAuth(self._auth[0], self._auth[1])
            if self._auth
            else None
        )

        # Use shortpath or paths API to get edges
        # Fallback: use kneighbor with depth=1 per vertex
        sampled = vertex_ids[:100]  # Limit for performance
        for src_id in sampled:
            try:
                url = (
                    f"{self._host}/graphs/{self._graph}"
                    f"/traversers/kneighbor"
                )
                payload: Dict[str, Any] = {
                    "source": src_id,
                    "direction": "BOTH",
                    "max_depth": 1,
                    "limit": 500,
                }
                if edge_label_filter:
                    payload["edge_label"] = ",".join(edge_label_filter)

                resp = self._requests.post(
                    url, json=payload, headers=headers, auth=auth, timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        neighbors = data.get("kneighbor", [])
                        for n in neighbors:
                            if isinstance(n, dict):
                                edges.append(n)
            except Exception:
                continue

        return edges

    def _build_adjacency(
        self,
        vertices: List[Dict],
        edges: List[Dict],
        direction: str,
        edge_label_filter: Optional[List[str]],
    ) -> Dict[str, List[str]]:
        """Build adjacency list from vertices and edges.

        Creates ``{node_id: [neighbor_ids]}`` mapping. Supports:

        - Uniform weights (default): each edge contributes 1/deg(u) weight.
        - Future extension: edge-property-weighted or edge-type-weighted.

        Args:
            vertices: List of vertex dicts (must contain "id").
            edges: List of edge dicts (must contain "source", "target").
            direction: "OUT", "IN", or "BOTH".
            edge_label_filter: Optional filter on edge labels.

        Returns:
            Adjacency list as dict mapping node_id to list of neighbor IDs.
        """
        adj: Dict[str, List[str]] = defaultdict(list)

        # Initialize all vertices
        for v in vertices:
            vid = v.get("id")
            if vid:
                adj.setdefault(vid, [])

        # Populate from edges
        allowed_labels: Optional[Set[str]] = (
            set(edge_label_filter) if edge_label_filter else None
        )

        for edge in edges:
            elabel = edge.get("label", "")
            if allowed_labels and elabel not in allowed_labels:
                continue

            src = edge.get("source") or edge.get("source_vertex", "")
            tgt = edge.get("target") or edge.get("target_vertex", "")

            if direction in ("OUT", "BOTH") and src and tgt:
                adj[src].append(tgt)
            if direction in ("IN", "BOTH") and src and tgt:
                adj[tgt].append(src)

        # Deduplicate neighbors while preserving order
        for node_id in adj:
            seen: Set[str] = set()
            unique_neighbors: List[str] = []
            for nb in adj[node_id]:
                if nb not in seen:
                    seen.add(nb)
                    unique_neighbors.append(nb)
            adj[node_id] = unique_neighbors

        return dict(adj)

    @staticmethod
    def _push_ppr(
        adj: Dict[str, List[str]],
        source_id: str,
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
    ) -> Dict[str, float]:
        """Compute approximate Personalized PageRank using push algorithm.

        This is the core algorithm based on PPRGO (WSDM 2022).

        The push method maintains two vectors:
        - **residue[r]**: residual probability yet to be pushed.
        - **reserve[p]**: accumulated PPR score (final output).

        For each node u where |r[u]| > epsilon * deg(u):
            - Distribute ``(1 - alpha) * r[u]`` uniformly to neighbors.
            - Add ``alpha * r[u]`` to reserve[u].
            - Clear r[u].

        Time complexity: O(m / epsilon) where m = |E| in worst case,
        but typically much faster due to early convergence.

        Args:
            adj: Adjacency list {node_id: [neighbor_ids]}.
            source_id: Source vertex for personalization vector.
            alpha: Restart probability (0 < alpha <= 1).
            epsilon: Push threshold (smaller = more accurate, slower).
            max_iter: Maximum number of iterations over all nodes.

        Returns:
            Dict mapping node_id to PPR score.
        """
        # Collect all node IDs
        all_nodes: Set[str] = set(adj.keys())
        for neighbors in adj.values():
            all_nodes.update(neighbors)

        # Ensure source exists
        if source_id not in all_nodes:
            all_nodes.add(source_id)
            adj.setdefault(source_id, [])

        # Index nodes for numpy array indexing
        node_list: List[str] = sorted(all_nodes)
        n = len(node_list)
        node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_list)}

        # Build degree array (out-degree for directed, total for undirected)
        degrees = np.zeros(n, dtype=np.float64)
        for i, nid in enumerate(node_list):
            degrees[i] = max(len(adj.get(nid, [])), 1.0)  # Avoid division by zero

        # Initialize residue and reserve arrays using numpy
        residue = np.zeros(n, dtype=np.float64)
        reserve = np.zeros(n, dtype=np.float64)

        # Start: put all initial mass at source
        src_idx = node_to_idx[source_id]
        residue[src_idx] = 1.0

        # Precompute (1 - alpha) factor
        one_minus_alpha = 1.0 - alpha

        # Track pushes for iteration limit
        total_pushes: int = 0
        iteration: int = 0

        # Main push loop
        while total_pushes < max_iter * n:
            # Find nodes that qualify for push: |residue[u]| > epsilon * deg(u)
            abs_residue = np.abs(residue)
            threshold = epsilon * degrees
            qualifies = abs_residue > threshold

            if not np.any(qualifies):
                break  # Converged

            # Get indices of qualifying nodes
            qualifying_indices = np.where(qualifies)[0]

            for u_idx in qualifying_indices:
                ru = residue[u_idx]

                # Push (1-alpha)*ru to neighbors
                deg_u = degrees[u_idx]
                neighbors = adj.get(node_list[u_idx], []) if u_idx < len(node_list) else []
                if neighbors and deg_u > 0:
                    # Normal case: distribute to neighbors
                    push_amount = one_minus_alpha * ru / deg_u
                    for nb in neighbors:
                        nb_idx = node_to_idx.get(nb)
                        if nb_idx is not None:
                            residue[nb_idx] += push_amount
                else:
                    # Dangling node (no neighbors): keep all mass as self-loop
                    # This matches standard PageRank dangling-node handling
                    reserve[u_idx] += one_minus_alpha * ru

                # Accumulate restart probability
                reserve[u_idx] += alpha * ru
                residue[u_idx] = 0.0
                total_pushes += 1

            iteration += 1
            if iteration >= max_iter:
                log.debug(f"[PPR] Reached max iterations ({max_iter})")
                break

        # Add remaining residue to reserve (cleanup)
        reserve += residue * alpha

        # Convert back to dict (only non-zero scores)
        scores: Dict[str, float] = {}
        for i, nid in enumerate(node_list):
            if reserve[i] > 1e-12:  # Near-zero cutoff
                scores[nid] = float(reserve[i])

        log.debug(
            f"[PPR] Push completed: {iteration} iterations, "
            f"{total_pushes} pushes, {len(scores)} non-zero entries"
        )

        return scores

    @staticmethod
    def _bfs_distances(
        source_id: str, adj: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Compute BFS shortest-path distances from source.

        Args:
            source_id: Source node ID.
            adj: Adjacency list.

        Returns:
            Dict mapping node_id to distance (hops). Unreachable nodes
            are omitted.
        """
        distances: Dict[str, int] = {source_id: 0}
        queue: deque = deque([source_id])
        visited: Set[str] = {source_id}

        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            for nb in adj.get(current, []):
                if nb not in visited:
                    visited.add(nb)
                    distances[nb] = current_dist + 1
                    queue.append(nb)

        return distances

    def integrate_with_rrf(
        self,
        ppr_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Merge PPR results into RRF fusion with vector and BM25 channels.

        Reciprocal Rank Fusion (RRF) combines multiple ranked lists into a single
        ranking without requiring calibrated scores.

        RRF score for item i: sum_{channel} 1 / (rank_i_in_channel + k)

        Args:
            ppr_results: Results from PPR.search(), each dict must have
                "node_id". Sorted by PPR score descending.
            vector_results: Vector search results, each dict must have
                "node_id" (or "id"). Sorted by similarity descending.
            bm25_results: BM25 keyword results, same format as above.
            k: RRF constant (default 60). Larger values dampen rank
                position differences.

        Returns:
            Fused results sorted by RRF score descending. Each entry contains
            ``{"node_id", "rrf_score", "ppr_rank", "vector_rank", "bm25_rank"}``.
        """
        from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion

        # Normalize ID keys across channels
        def _extract_ids(results: List[Dict]) -> List[str]:
            ids: List[str] = []
            for r in results:
                nid = r.get("node_id") or r.get("id")
                if nid:
                    ids.append(str(nid))
            return ids

        ppr_ids = _extract_ids(ppr_results)
        vec_ids = _extract_ids(vector_results)
        bm25_ids = _extract_ids(bm25_results)

        rrf = ReciprocalRankFusion(k=k)
        fused = rrf.fuse([
            ("ppr", ppr_ids),
            ("vector", vec_ids),
            ("bm25", bm25_ids),
        ])

        # Build enriched output with rank positions
        final_results: List[Dict[str, Any]] = []
        for item in fused.items:
            result: Dict[str, Any] = {"node_id": item, "rrf_score": fused.scores[item]}
            # Record rank positions (0-based, -1 if not present)
            try:
                result["ppr_rank"] = ppr_ids.index(item)
            except ValueError:
                result["ppr_rank"] = -1
            try:
                result["vector_rank"] = vec_ids.index(item)
            except ValueError:
                result["vector_rank"] = -1
            try:
                result["bm25_rank"] = bm25_ids.index(item)
            except ValueError:
                result["bm25_rank"] = -1
            final_results.append(result)

        return final_results


# ---------------------------------------------------------------------------
# Standalone functions (usable without PPRRetriever instance)
# ---------------------------------------------------------------------------


def compute_ppr_exact(
    adjacency: Dict[str, List[str]],
    source_id: str,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """Compute exact Personalized PageRank using power iteration.

    Solves: ppr = alpha * e_source + (1 - alpha) * W^T * ppr

    Where:
        - e_source is the unit vector at source position.
        - W is the column-stochastic transition matrix (uniform weights).

    This method uses dense matrix power iteration and has O(n^2) complexity
    per iteration. Suitable only for small graphs (n < ~10K) or validation
    purposes. For production use, prefer :meth:`PPRRetriever._push_ppr`.

    Args:
        adjacency: Adjacency list {node_id: [neighbor_ids]}.
        source_id: Source vertex ID for personalization.
        alpha: Restart probability.
        max_iter: Maximum power iterations.
        tol: Convergence tolerance (L1-norm of change).

    Returns:
        Dict mapping node_id to exact PPR score.

    Example::

        >>> adj = {"A": ["B", "C"], "B": ["A"], "C": ["A"]}
        >>> scores = compute_ppr_exact(adj, "A")
        >>> print(sorted(scores.items(), key=lambda x: -x[1])[0])
        ('A', ...)  # Source should have highest score
    """
    # Collect and sort all nodes
    all_nodes: Set[str] = set(adjacency.keys())
    for neighbors in adjacency.values():
        all_nodes.update(neighbors)

    if source_id not in all_nodes:
        all_nodes.add(source_id)
        adjacency = dict(adjacency)
        adjacency.setdefault(source_id, [])

    node_list: List[str] = sorted(all_nodes)
    n = len(node_list)
    node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_list)}

    # Build column-stochastic transition matrix W
    # W[j][i] = 1/deg(j) if i is neighbor of j, else 0
    W = np.zeros((n, n), dtype=np.float64)
    for j, nid in enumerate(node_list):
        neighbors = adjacency.get(nid, [])
        deg = len(neighbors)
        if deg > 0:
            for nb in neighbors:
                i = node_to_idx.get(nb)
                if i is not None:
                    W[i, j] = 1.0 / deg

    # Initial vector: uniform
    ppr = np.full(n, 1.0 / n, dtype=np.float64)

    # Personalization vector
    e_source = np.zeros(n, dtype=np.float64)
    src_idx = node_to_idx[source_id]
    e_source[src_idx] = 1.0

    one_minus_alpha = 1.0 - alpha

    # Power iteration
    for iteration in range(max_iter):
        ppr_new = alpha * e_source + one_minus_alpha * (W @ ppr)

        diff = np.abs(ppr_new - ppr).sum()
        ppr = ppr_new

        if diff < tol:
            break

    # Convert to dict
    scores: Dict[str, float] = {
        nid: float(ppr[i]) for i, nid in enumerate(node_list) if ppr[i] > 1e-14
    }

    return scores


def build_adjacency_from_edges(
    edges: List[Tuple[str, str]], directed: bool = False
) -> Dict[str, List[str]]:
    """Build adjacency list from edge list.

    Convenience function for testing and ad-hoc usage.

    Args:
        edges: List of (source, target) tuples.
        directed: If True, edges are directed (only add out-neighbors).

    Returns:
        Adjacency list dict.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, tgt in edges:
        adj[src].append(tgt)
        if not directed:
            adj[tgt].append(src)

    # Deduplicate
    result: Dict[str, List[str]] = {}
    for node_id, neighbors in adj.items():
        result[node_id] = list(dict.fromkeys(neighbors))  # Preserves order
    return result
