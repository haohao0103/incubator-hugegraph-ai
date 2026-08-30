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
Vermeer-backed graph engine for the NL2SQL stack.

Vermeer is HugeGraph's distributed BSP graph-compute engine (Go). This module
maps the four NL2SQL primitives onto Vermeer operators:

============================  ===============================================
NL2SQL primitive              Vermeer operator
============================  ===============================================
``personalized_pagerank``     ``ppr`` — one task per seed, summed (see below)
``shortest_join_path``        ``weighted_sssp`` + local path reconstruction
``steiner_join_tree``         ``weighted_sssp`` x terminals + metric closure
``communities``               ``louvain_weighted`` / ``leiden``
============================  ===============================================

Two graphs are materialised, because the two consumers want different views
and different edge properties:

``<prefix>_schema``
    Tables + columns + terms. Feeds ``ppr``. No edge property: Vermeer's
    ``ppr`` has no edge-weight parameter at all.
``<prefix>_join``
    Table-only projection. Two Float32 edge properties: ``join_cost`` for
    ``weighted_sssp`` (lower is closer) and ``affinity`` for
    ``louvain_weighted`` (higher is closer, see ``base.affinity_of``).

Four Vermeer behaviours drive the design and are easy to get wrong:

1. **Undirected edges must be written twice.** ``load.use_undirected`` is
   accepted as an option, but the reverse-edge insertion in
   ``apps/worker/load_graph_bl.go`` is commented out — the flag only turns on
   out-edge indexing. The Schema Graph is undirected, so every edge is emitted
   in both directions.

2. **Property parsing ignores ``load.delimiter``.** ``PropertyValue.LoadFromString``
   splits on a hard-coded ``,``, so the field delimiter must be something else
   or the first property is parsed as an empty string. Fields are tab-separated
   here; properties within the last field are comma-separated.

3. **``ppr`` takes exactly one source.** Multi-seed PPR is therefore assembled
   from N single-seed runs. In ``personalized_pagerank.go`` the teleport term
   is added only at the source vertex, so the iteration is affine in the
   teleport vector and ``sum(w_i * ppr(seed_i)) == ppr(sum(w_i * seed_i))`` —
   *provided no vertex is dangling*. Dangling mass is redistributed to the
   source alone, which makes the operator source-dependent, so an isolated
   node turns the identity into an approximation. Isolated nodes are counted
   at materialisation time and warned about explicitly rather than silently
   tolerated.

4. **Results need ``output.need_query=1``.** Without it the master frees the
   result set one minute after completion, and local result files land on the
   worker hosts rather than on the caller's filesystem.

Path *reconstruction* stays local: ``weighted_sssp`` returns a distance vector,
not predecessors. The join projection is table-level (thousands of nodes even
for a catalog with millions of columns), so walking it in Python is cheap. The
expensive part — PPR over the full tables+columns+terms graph — is what
actually goes to the cluster.
"""

import math
import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from hugegraph_llm.utils.log import log

from ..schema_graph.backend import to_join_graph
from ..schema_graph.model import SchemaGraph
from .base import EngineCapabilities, GraphEngine, affinity_of
from .vermeer_client import COMPUTE_DONE_STATES, VermeerClient, VermeerError

#: Field separator in the load files. Must NOT be "," — see module docstring.
FIELD_SEP = "\t"
#: Separator *within* the property field, hard-coded in Vermeer.
PROP_SEP = ","

JOIN_COST_PROP = "join_cost"
AFFINITY_PROP = "affinity"

#: ``weighted_sssp`` initialises distances to float32 max; anything at or above
#: this magnitude means "unreachable", not "very far".
_UNREACHABLE = 3.0e38
#: Relative tolerance when matching float32 distances during path rebuild.
_PATH_TOL = 1e-3

_LOUVAIN = "louvain_weighted"
_LEIDEN = "leiden"
#: ``"louvain"`` is accepted as an alias so that a caller can invoke
#: ``communities()`` with the default argument against either engine.
_COMMUNITY_ALIASES = {"louvain": _LOUVAIN, _LOUVAIN: _LOUVAIN, _LEIDEN: _LEIDEN}


class VermeerEngine(GraphEngine):
    """Runs the NL2SQL graph primitives on a Vermeer cluster."""

    def __init__(
        self,
        schema: SchemaGraph,
        client: Optional[VermeerClient] = None,
        data_dir: Optional[str] = None,
        data_hosts: Optional[Sequence[str]] = None,
        graph_prefix: Optional[str] = None,
        compute_parallel: int = 4,
        max_step: int = 100,
        ppr_diff_threshold: float = 1e-5,
        sssp_diff_threshold: float = 1e-4,
        louvain_step: int = 10,
        louvain_threshold: float = 1e-4,
        keep_graphs: bool = False,
    ):
        """
        :param schema: the Schema Graph to materialise.
        :param client: a :class:`VermeerClient`; a default localhost client is
                       created when omitted.
        :param data_dir: directory for the generated load files. Must be
                         readable by every host in ``data_hosts`` — Vermeer
                         workers open the files themselves. Defaults to a
                         temp dir that is removed by :meth:`close`.
        :param data_hosts: worker IPs that should read the load files.
                           Defaults to the IPs the master reports for the
                           client's worker group.
        :param graph_prefix: Vermeer graph-name prefix; defaults to a unique
                             ``nl2sql_<hex>`` so concurrent runs never collide.
        :param compute_parallel: ``compute.parallel``.
        :param max_step: ``compute.max_step``. PPR at damping 0.85 and
                         Bellman-Ford both need well over the Vermeer default
                         of 10 on a non-trivial graph.
        :param keep_graphs: leave the loaded graphs in place on
                            :meth:`close` (useful when debugging).
        """
        self._schema = schema
        self._client = client or VermeerClient()
        self._hosts: Optional[List[str]] = list(data_hosts) if data_hosts else None
        self._prefix = graph_prefix or f"nl2sql_{uuid.uuid4().hex[:12]}"
        self._compute_parallel = max(1, int(compute_parallel))
        self._max_step = max(1, int(max_step))
        self._ppr_diff_threshold = float(ppr_diff_threshold)
        self._sssp_diff_threshold = float(sssp_diff_threshold)
        self._louvain_step = max(1, int(louvain_step))
        self._louvain_threshold = float(louvain_threshold)
        self._keep_graphs = bool(keep_graphs)

        self._owns_data_dir = data_dir is None
        self._data_dir = data_dir or tempfile.mkdtemp(prefix="nl2sql_vermeer_")

        # The join projection is table-level and small; keeping it in-process is
        # what makes path reconstruction from a distance vector possible.
        self._join_graph = to_join_graph(schema)

        self._loaded: Dict[str, bool] = {}
        self._allocated = False
        self._isolated_nodes = 0
        self._dangling_warned = False
        # (node, alpha) -> scores ; node -> distances
        self._ppr_cache: Dict[Tuple[str, float], Dict[str, float]] = {}
        self._dist_cache: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="vermeer",
            distributed=True,
            # Vermeer's ppr operator has no edge-weight parameter.
            weighted_ppr=False,
            weighted_paths=True,
            community_algorithms=("louvain", _LOUVAIN, _LEIDEN),
        )

    @property
    def join_graph(self) -> nx.Graph:
        """Table-only projection held locally for path reconstruction."""
        return self._join_graph

    @property
    def schema_graph_name(self) -> str:
        return f"{self._prefix}_schema"

    @property
    def join_graph_name(self) -> str:
        return f"{self._prefix}_join"

    @property
    def data_dir(self) -> str:
        return self._data_dir

    # ------------------------------------------------------------------
    # materialisation
    # ------------------------------------------------------------------

    def _hosts_or_detect(self) -> List[str]:
        if self._hosts:
            return self._hosts
        hosts = self._client.worker_hosts()
        if not hosts:
            raise VermeerError(
                "no workers registered with the Vermeer master; cannot decide "
                "which host should read the load files"
            )
        self._hosts = hosts
        log.debug("vermeer data hosts auto-detected: %s", hosts)
        return hosts

    def _file_map(self, path: str) -> str:
        """Render ``{"<ip>": "<path>"}`` for ``load.*_files``."""
        import json

        return json.dumps({host: path for host in self._hosts_or_detect()})

    @staticmethod
    def _check_id(node_id: str) -> str:
        """Reject ids that would corrupt the load file rather than mangle them."""
        if FIELD_SEP in node_id or "\n" in node_id or "\r" in node_id:
            raise ValueError(
                f"node id contains a tab or newline and cannot be loaded into "
                f"Vermeer: {node_id!r}"
            )
        return node_id

    def _write_schema_files(self) -> Tuple[str, str]:
        """Write the full schema graph (tables + columns + terms).

        Every edge is emitted in both directions, and duplicate undirected
        pairs are collapsed so that the transition probabilities match the
        networkx projection (``nx.Graph`` keeps one edge per pair, Vermeer
        would keep both and double-count it).
        """
        vertex_path = os.path.join(self._data_dir, "schema_vertices.csv")
        edge_path = os.path.join(self._data_dir, "schema_edges.csv")

        degree: Dict[str, int] = {}
        seen: set = set()
        with open(edge_path, "w", encoding="utf-8") as fh:
            for edge in self._schema.edges:
                src, dst = edge.source, edge.target
                if src == dst:
                    continue
                if src not in self._schema.nodes or dst not in self._schema.nodes:
                    continue
                key = (src, dst) if src <= dst else (dst, src)
                if key in seen:
                    continue
                seen.add(key)
                a = self._check_id(src)
                b = self._check_id(dst)
                fh.write(f"{a}{FIELD_SEP}{b}\n")
                fh.write(f"{b}{FIELD_SEP}{a}\n")
                degree[a] = degree.get(a, 0) + 1
                degree[b] = degree.get(b, 0) + 1

        with open(vertex_path, "w", encoding="utf-8") as fh:
            for node_id in self._schema.nodes:
                fh.write(f"{self._check_id(node_id)}\n")

        self._isolated_nodes = sum(
            1 for node_id in self._schema.nodes if degree.get(node_id, 0) == 0
        )
        log.debug(
            "vermeer schema files: %s vertices, %s undirected edges, "
            "%s isolated nodes",
            len(self._schema.nodes), len(seen), self._isolated_nodes,
        )
        return vertex_path, edge_path

    def _write_join_files(self) -> Tuple[str, str]:
        """Write the table-only join projection with cost + affinity."""
        vertex_path = os.path.join(self._data_dir, "join_vertices.csv")
        edge_path = os.path.join(self._data_dir, "join_edges.csv")

        with open(vertex_path, "w", encoding="utf-8") as fh:
            for node_id in self._join_graph.nodes():
                fh.write(f"{self._check_id(node_id)}\n")

        with open(edge_path, "w", encoding="utf-8") as fh:
            for u, v, data in self._join_graph.edges(data=True):
                cost = float(data.get("weight", 1.0))
                if not math.isfinite(cost) or cost <= 0.0:
                    # weighted_sssp skips weight <= 0, and an infinite cost is
                    # not a join at all; both are dropped at projection time.
                    continue
                props = f"{cost!r}{PROP_SEP}{affinity_of(cost)!r}"
                a = self._check_id(u)
                b = self._check_id(v)
                fh.write(f"{a}{FIELD_SEP}{b}{FIELD_SEP}{props}\n")
                fh.write(f"{b}{FIELD_SEP}{a}{FIELD_SEP}{props}\n")

        log.debug("vermeer join files: %s tables, %s edges",
                  self._join_graph.number_of_nodes(),
                  self._join_graph.number_of_edges())
        return vertex_path, edge_path

    def _alloc_once(self) -> None:
        if not self._allocated:
            self._client.alloc_worker_group()
            self._allocated = True

    def _load(self, graph: str, vertex_path: str, edge_path: str,
              with_property: bool) -> None:
        params = {
            "load.type": "local",
            "load.vertex_files": self._file_map(vertex_path),
            "load.edge_files": self._file_map(edge_path),
            "load.delimiter": FIELD_SEP,
            "load.parallel": "1",
            "load.use_outedge": "1",
            "load.use_out_degree": "1",
        }
        if with_property:
            # Value type 0 == Float32 (structure.ValueTypeFloat32 is the first
            # iota member); the second number is the field index.
            params["load.use_property"] = "1"
            params["load.edge_property"] = (
                f'{{"{JOIN_COST_PROP}":"0,0","{AFFINITY_PROP}":"0,1"}}'
            )
        self._alloc_once()
        self._client.create_graph(graph)
        self._client.run_load(graph, params)
        log.info("vermeer graph %s loaded", graph)

    def _ensure_schema_graph(self) -> None:
        name = self.schema_graph_name
        if self._loaded.get(name):
            return
        vertex_path, edge_path = self._write_schema_files()
        self._load(name, vertex_path, edge_path, with_property=False)
        self._loaded[name] = True

    def _ensure_join_graph(self) -> None:
        name = self.join_graph_name
        if self._loaded.get(name):
            return
        vertex_path, edge_path = self._write_join_files()
        self._load(name, vertex_path, edge_path, with_property=True)
        self._loaded[name] = True

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    def _compute(self, graph: str, algorithm: str,
                 extra: Dict[str, str]) -> Dict[str, str]:
        params = {
            "compute.algorithm": algorithm,
            "compute.max_step": str(self._max_step),
            "compute.parallel": str(self._compute_parallel),
            "output.type": "local",
            "output.file_path": os.path.join(
                self._data_dir, "out", f"{graph}_{algorithm}"
            ),
            "output.delimiter": PROP_SEP,
            "output.parallel": "1",
            # Without this the master discards the result set (see docstring).
            "output.need_query": "1",
        }
        params.update(extra)
        task_id = self._client.submit_task("compute", graph, params)
        self._client.wait_task(task_id, COMPUTE_DONE_STATES)
        return self._client.compute_values(task_id)

    # ---- L1: schema linking ----

    def personalized_pagerank(
        self, seeds: Dict[str, float], alpha: float = 0.85
    ) -> Dict[str, float]:
        active = {
            node: float(weight)
            for node, weight in seeds.items()
            if node in self._schema.nodes and float(weight) > 0.0
        }
        if not active:
            return {}
        self._ensure_schema_graph()

        if len(active) > 1 and self._isolated_nodes and not self._dangling_warned:
            self._dangling_warned = True
            log.warning(
                "vermeer ppr: %s isolated node(s) in the schema graph. Vermeer "
                "redistributes dangling mass to the single ppr.source, so the "
                "multi-seed sum is an approximation rather than an identity.",
                self._isolated_nodes,
            )

        total = sum(active.values())
        combined: Dict[str, float] = {}
        for node in sorted(active):
            share = active[node] / total
            for target, score in self._ppr_single(node, alpha).items():
                combined[target] = combined.get(target, 0.0) + share * score

        mass = sum(combined.values())
        if mass <= 0.0:
            return {}
        return {node: score / mass for node, score in combined.items()}

    def _ppr_single(self, source: str, alpha: float) -> Dict[str, float]:
        key = (source, float(alpha))
        cached = self._ppr_cache.get(key)
        if cached is not None:
            return cached
        raw = self._compute(
            self.schema_graph_name,
            "ppr",
            {
                "ppr.source": source,
                "ppr.damping": str(float(alpha)),
                "ppr.diff_threshold": str(self._ppr_diff_threshold),
            },
        )
        scores: Dict[str, float] = {}
        for node, value in raw.items():
            score = _to_float(value)
            if score is not None and score > 0.0:
                scores[node] = score
        self._ppr_cache[key] = scores
        return scores

    # ---- L2: join paths ----

    def _distances(self, source: str) -> Dict[str, float]:
        """Single-source shortest distances over the join graph."""
        cached = self._dist_cache.get(source)
        if cached is not None:
            return cached
        self._ensure_join_graph()
        raw = self._compute(
            self.join_graph_name,
            "weighted_sssp",
            {
                "sssp.source": source,
                "sssp.edge_weight_property": JOIN_COST_PROP,
                "sssp.diff_threshold": str(self._sssp_diff_threshold),
            },
        )
        distances: Dict[str, float] = {}
        for node, value in raw.items():
            dist = _to_float(value)
            if dist is None or dist >= _UNREACHABLE:
                continue
            distances[node] = dist
        self._dist_cache[source] = distances
        return distances

    def shortest_join_path(
        self, source: str, target: str
    ) -> Optional[List[str]]:
        if source == target:
            return [source] if source in self._join_graph else None
        if source not in self._join_graph or target not in self._join_graph:
            return None

        distances = self._distances(source)
        if not math.isfinite(distances.get(target, math.inf)):
            return None

        path = self._rebuild_path(source, target, distances)
        if path is not None:
            return path

        # A distance without a matching walk means the distance vector and the
        # local projection disagree. Say so loudly instead of returning a
        # silently wrong answer; the local Dijkstra keeps the caller working.
        log.warning(
            "vermeer weighted_sssp: could not rebuild %s -> %s from the "
            "distance vector; falling back to the local projection",
            source, target,
        )
        try:
            return nx.shortest_path(
                self._join_graph, source, target, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _rebuild_path(
        self, source: str, target: str, distances: Dict[str, float]
    ) -> Optional[List[str]]:
        """Walk back from ``target`` along edges that explain the distances."""
        path = [target]
        current = target
        limit = self._join_graph.number_of_nodes() + 1
        while current != source:
            if len(path) > limit:
                return None
            here = distances.get(current)
            if here is None:
                return None
            best: Optional[Tuple[float, str]] = None
            for neighbour in self._join_graph.neighbors(current):
                prev = distances.get(neighbour)
                if prev is None:
                    continue
                weight = float(
                    self._join_graph[current][neighbour].get("weight", 1.0)
                )
                if abs(prev + weight - here) > _PATH_TOL * max(1.0, abs(here)):
                    continue
                candidate = (prev, neighbour)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                return None
            current = best[1]
            path.append(current)
        path.reverse()
        return path

    def steiner_join_tree(
        self, terminals: List[str]
    ) -> Tuple[List[str], float]:
        wanted = [t for t in dict.fromkeys(terminals) if t in self._join_graph]
        if not wanted:
            return [], 0.0
        if len(wanted) == 1:
            return wanted, 0.0

        distances = {t: self._distances(t) for t in wanted}
        root = wanted[0]
        for terminal in wanted[1:]:
            if not math.isfinite(distances[root].get(terminal, math.inf)):
                return [], 0.0

        # Metric closure + MST (Kou et al.): a 2-approximation, and the same
        # construction the local engine uses, so the two are comparable.
        closure: List[Tuple[float, str, str]] = []
        for i, a in enumerate(wanted):
            for b in wanted[i + 1:]:
                dist = distances[a].get(b)
                if dist is None:
                    dist = distances[b].get(a)
                if dist is None or not math.isfinite(dist):
                    continue
                closure.append((dist, a, b))
        closure.sort()

        parent = {t: t for t in wanted}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree_edges: List[Tuple[str, str]] = []
        for dist, a, b in closure:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            parent[ra] = rb
            tree_edges.append((a, b))
            if len(tree_edges) == len(wanted) - 1:
                break
        if len(tree_edges) < len(wanted) - 1:
            return [], 0.0

        # Cost is the sum over the deduplicated edges actually walked, not the
        # closure distances: legs that share a segment must not pay twice. This
        # is also what the local engine reports, so the two stay comparable.
        nodes: List[str] = [root]
        seen = {root}
        seen_edges: set = set()
        total_cost = 0.0
        for a, b in tree_edges:
            leg = self._rebuild_path(a, b, distances[a])
            if leg is None:
                leg = self.shortest_join_path(a, b) or [a, b]
            for left, right in zip(leg, leg[1:]):
                key = (left, right) if left <= right else (right, left)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edge = self._join_graph.get_edge_data(left, right) or {}
                total_cost += float(edge.get("weight", 1.0))
            for node in leg:
                if node not in seen:
                    seen.add(node)
                    nodes.append(node)
        return nodes, total_cost

    # ---- L3: subject domains ----

    def communities(
        self, resolution: float = 1.0, algorithm: str = "louvain"
    ) -> Dict[str, int]:
        resolved = _COMMUNITY_ALIASES.get(algorithm)
        if resolved is None:
            raise ValueError(
                f"vermeer engine supports "
                f"{sorted(set(_COMMUNITY_ALIASES))}, got {algorithm!r}"
            )
        if self._join_graph.number_of_nodes() == 0:
            return {}
        self._ensure_join_graph()

        if resolved == _LOUVAIN:
            extra = {
                "louvain.resolution": str(float(resolution)),
                # Affinity, not cost: modularity wants tightly coupled pairs to
                # carry the larger weight.
                "louvain.edge_weight_property": AFFINITY_PROP,
                "louvain.step": str(self._louvain_step),
                "louvain.threshold": str(self._louvain_threshold),
            }
        else:
            # leiden reads no edge-weight property, so it runs unweighted.
            extra = {
                "leiden.resolution": str(float(resolution)),
                "leiden.step": str(self._louvain_step),
                "leiden.threshold": str(self._louvain_threshold),
            }
        raw = self._compute(self.join_graph_name, resolved, extra)
        return _compact_labels(raw)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._keep_graphs:
            for name in list(self._loaded):
                self._client.delete_graph(name)
            self._loaded.clear()
        if self._owns_data_dir:
            shutil.rmtree(self._data_dir, ignore_errors=True)
        self._client.close()


def _to_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _compact_labels(raw: Dict[str, str]) -> Dict[str, int]:
    """Renumber Vermeer community ids to ``0..k-1``.

    Vermeer labels a community by an internal vertex index, so the raw ids are
    sparse and depend on partitioning. Renumbering by the lexicographically
    smallest member makes the output deterministic and directly comparable to
    the local engine, which returns ``0..k-1``.
    """
    groups: Dict[str, List[str]] = {}
    for node, value in raw.items():
        try:
            key = str(int(float(value)))
        except (TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(node)
    ordered = sorted(groups.values(), key=lambda members: min(members))
    labels: Dict[str, int] = {}
    for cid, members in enumerate(ordered):
        for node in members:
            labels[node] = cid
    return labels
