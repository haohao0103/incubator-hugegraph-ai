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

"""Community detection using HugeGraph's own algorithms.

Primary: Vermeer (Go in-memory engine) via PyVermeerClient task API.
  Supported algorithms: louvain, wcc, label_propagation, pagerank, etc.

Fallback: HugeGraph-Computer (Java OLAP) via REST API.
  Supported algorithms: louvain, wcc, clustering_coefficient, etc.

Last resort: Python local — Leiden (preferred, via leidenalg) or Louvain
  (via networkx). Requires fetching all vertices/edges into memory.

The module auto-detects which engine is available and uses the best option.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from hugegraph_llm.utils.log import log

# ── Engine availability detection ────────────────────────────

try:
    from pyvermeer.client.client import PyVermeerClient
    from pyvermeer.structure.task_data import TaskCreateRequest

    HAS_VERMEER = True
except ImportError:
    HAS_VERMEER = False

# Check for leidenalg (Leiden algorithm — preferred over Louvain)
try:
    import leidenalg  # noqa: F401
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


# ── Algorithm constants ──────────────────────────────────────

# Algorithms supported by HugeGraph-Computer / Vermeer
ALGORITHM_LOUVAIN = "louvain"
ALGORITHM_LEIDEN = "leiden"  # Preferred when leidenalg is available locally
ALGORITHM_WCC = "wcc"
ALGORITHM_LABEL_PROPAGATION = "label_propagation"
ALGORITHM_PAGERANK = "pagerank"
ALGORITHM_CLUSTERING = "clustering_coefficient"

# For community detection, we prefer these algorithms in order
COMMUNITY_ALGORITHMS = [
    ALGORITHM_LEIDEN if HAS_LEIDEN else ALGORITHM_LOUVAIN,
    ALGORITHM_LOUVAIN,
    ALGORITHM_WCC,
    ALGORITHM_LABEL_PROPAGATION,
]

# Default Vermeer master endpoint
DEFAULT_VERMEER_IP = "127.0.0.1"
DEFAULT_VERMEER_PORT = 8688

# HugeGraph-Computer OLAP REST endpoint (relative to HugeGraph server)
COMPUTER_ALGORITHM_PATH = "/graphs/{graph_name}/jobs/{algorithm}"


class CommunityDetect:
    """Detect communities in a HugeGraph knowledge graph.

    Uses HugeGraph's own algorithms in priority order:
    1. Vermeer (Go in-memory engine) — sub-second interactive
    2. HugeGraph-Computer (Java OLAP) — batch Pregel/BSP
    3. networkx Louvain — local fallback for small graphs

    Usage:
        # Auto-detect engine
        detector = CommunityDetect(client=hugegraph_client)
        result = detector.run(context)

        # Explicit engine
        detector = CommunityDetect(
            client=hugegraph_client,
            engine="vermeer",
            vermeer_ip="192.168.1.1",
            vermeer_port=8688,
        )
    """

    def __init__(
        self,
        client: Any = None,
        engine: str = "auto",  # "auto", "vermeer", "computer", "networkx"
        algorithm: str = ALGORITHM_LOUVAIN,
        min_community_size: int = 3,
        vermeer_ip: str = DEFAULT_VERMEER_IP,
        vermeer_port: int = DEFAULT_VERMEER_PORT,
        vermeer_token: str = "",
        poll_interval: float = 0.5,
        poll_timeout: float = 60.0,
    ):
        """Initialize the community detector.

        Args:
            client: HugeGraph PyHugeClient instance.
            engine: Which compute engine to use.
            algorithm: Algorithm name (louvain, wcc, label_propagation).
            min_community_size: Communities smaller than this are filtered out.
            vermeer_ip: Vermeer master IP.
            vermeer_port: Vermeer master HTTP port.
            vermeer_token: Vermeer auth token.
            poll_interval: Seconds between task status polls.
            poll_timeout: Maximum seconds to wait for task completion.
        """
        self._client = client
        self._engine = engine
        self._algorithm = algorithm
        self._min_community_size = min_community_size
        self._vermeer_ip = vermeer_ip
        self._vermeer_port = vermeer_port
        self._vermeer_token = vermeer_token
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout

        # Resolve actual engine
        self._resolved_engine = self._resolve_engine()

    def _resolve_engine(self) -> str:
        """Determine which engine to use."""
        if self._engine == "vermeer" and HAS_VERMEER:
            return "vermeer"
        if self._engine == "vermeer" and not HAS_VERMEER:
            log.warning(
                "Vermeer engine requested but pyvermeer not installed. "
                "Install with: pip install vermeer-python-client"
            )
            # Fall through to next best option

        # Auto-detect
        if HAS_VERMEER:
            log.info("Using Vermeer (Go) for community detection")
            return "vermeer"

        # Try HugeGraph-Computer via client
        if self._client is not None:
            log.info("Trying HugeGraph-Computer (Java OLAP) for community detection")
            return "computer"

        # Last resort
        log.warning(
            "No HugeGraph compute engine available. "
            "Falling back to networkx Louvain (local, memory-bound). "
            "For large graphs, start Vermeer or HugeGraph-Computer."
        )
        return "networkx"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run community detection using the resolved engine.

        Reads from context:
            - Graph name (from config) for submitting tasks

        Writes to context:
            communities: List of community dicts with vertices, edges.
            community_count: Total community count.
            engine_used: Which engine was actually used.
        """
        if self._resolved_engine == "networkx":
            return self._run_networkx(context)

        if self._resolved_engine == "vermeer":
            return self._run_vermeer(context)

        if self._resolved_engine == "computer":
            return self._run_computer(context)

        context["communities"] = []
        context["community_count"] = 0
        context["engine_used"] = "none"
        return context

    # ── Vermeer engine ────────────────────────────────────────

    def _run_vermeer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run community detection via Vermeer task API.

        Submits a task to Vermeer master, polls for completion,
        and parses the result into community assignments.
        """
        from hugegraph_llm.config import huge_settings

        graph_name = huge_settings.graph_name
        vermeer_client = PyVermeerClient(
            ip=self._vermeer_ip,
            port=self._vermeer_port,
            token=self._vermeer_token,
        )

        # Submit algorithm task
        log.info(
            "Submitting Vermeer task: algorithm=%s, graph=%s",
            self._algorithm,
            graph_name,
        )
        try:
            response = vermeer_client.tasks.create_task(
                create_task=TaskCreateRequest(
                    task_type=self._algorithm,
                    graph_name=graph_name,
                    params={},
                )
            )
            task_id = response.task.id
            log.info("Vermeer task created: id=%d", task_id)
        except Exception as e:
            log.error("Failed to create Vermeer task: %s", e)
            return self._fallback_networkx(context)

        # Poll for completion
        elapsed = 0.0
        while elapsed < self._poll_timeout:
            time.sleep(self._poll_interval)
            elapsed += self._poll_interval

            try:
                task_response = vermeer_client.tasks.get_task(task_id)
                state = task_response.task.state
                if state in ("SUCCESS", "FAILED", "CANCELLED"):
                    break
            except Exception as e:
                log.warning("Error polling Vermeer task %d: %s", task_id, e)

        if state == "SUCCESS":
            task_data = task_response.task.to_dict()
            communities = self._parse_vermeer_result(task_data)
            context["communities"] = communities
            context["community_count"] = len(communities)
            context["engine_used"] = "vermeer"
            log.info(
                "Vermeer detected %d communities via %s",
                len(communities),
                self._algorithm,
            )
        else:
            log.warning(
                "Vermeer task %d ended with state=%s, falling back to networkx",
                task_id,
                state,
            )
            return self._fallback_networkx(context)

        return context

    @staticmethod
    def _parse_vermeer_result(task_data: Dict) -> List[Dict]:
        """Parse Vermeer task result into community assignments.

        Expected format: task params/results contain vertex → community mapping.
        This implementation handles common Vermeer output formats.
        """
        params = task_data.get("params", {})
        communities = []

        # Vermeer typically returns results in params.result as dict or list
        result = params.get("result", params)

        if isinstance(result, dict):
            # Format: {community_id: [vertex_ids]}
            for cid, vertices in result.items():
                if isinstance(vertices, list) and len(vertices) >= 2:
                    communities.append({
                        "id": str(cid),
                        "level": 0,
                        "vertices": vertices,
                        "size": len(vertices),
                    })
        elif isinstance(result, list):
            # Format: [{vertex_id, community_id, ...}]
            comm_map = {}
            for item in result:
                vid = item.get("vertex_id") or item.get("id")
                cid = item.get("community_id") or item.get("community")
                if vid and cid is not None:
                    comm_map.setdefault(str(cid), []).append(str(vid))
            for cid, vertices in comm_map.items():
                communities.append({
                    "id": f"C{cid}",
                    "level": 0,
                    "vertices": vertices,
                    "size": len(vertices),
                })

        return [c for c in communities if c["size"] >= 2]

    # ── HugeGraph-Computer engine ─────────────────────────────

    def _run_computer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run community detection via HugeGraph-Computer REST API.

        Submits an OLAP job to the HugeGraph server's computer endpoint.
        """
        from hugegraph_llm.config import huge_settings

        graph_name = huge_settings.graph_name
        graph_url = huge_settings.graph_url.rstrip("/")

        # HugeGraph-Computer endpoint
        endpoint = COMPUTER_ALGORITHM_PATH.format(
            graph_name=graph_name, algorithm=self._algorithm
        )
        url = f"{graph_url}{endpoint}"

        log.info(
            "Submitting HugeGraph-Computer job: algorithm=%s, graph=%s",
            self._algorithm,
            graph_name,
        )

        try:
            import requests as req

            resp = req.post(
                url,
                json={"algorithm": self._algorithm},
                timeout=30,
                auth=(huge_settings.graph_user, huge_settings.graph_pwd)
                if huge_settings.graph_user
                else None,
            )
            if resp.status_code == 200:
                data = resp.json()
                communities = self._parse_computer_result(data)
                context["communities"] = communities
                context["community_count"] = len(communities)
                context["engine_used"] = "computer"
                log.info(
                    "HugeGraph-Computer detected %d communities via %s",
                    len(communities),
                    self._algorithm,
                )
            else:
                log.warning(
                    "Computer API returned %d: %s, falling back",
                    resp.status_code,
                    resp.text[:200],
                )
                return self._fallback_networkx(context)
        except Exception as e:
            log.warning("Computer API call failed: %s, falling back", e)
            return self._fallback_networkx(context)

        return context

    @staticmethod
    def _parse_computer_result(data: Dict) -> List[Dict]:
        """Parse HugeGraph-Computer result into community assignments.

        Expected format: {vertices: [{id, community_id}, ...]} or {communities: ...}
        """
        communities = []
        vertices = data.get("vertices", data.get("result", []))

        if isinstance(vertices, list):
            comm_map = {}
            for item in vertices:
                vid = item.get("id") or item.get("vertex_id")
                cid = item.get("community_id") or item.get("community") or item.get("cluster")
                if vid and cid is not None:
                    comm_map.setdefault(str(cid), []).append(str(vid))
            for cid, verts in comm_map.items():
                communities.append({
                    "id": f"C{cid}",
                    "level": 0,
                    "vertices": verts,
                    "size": len(verts),
                })
        elif isinstance(vertices, dict):
            # Map of community_id → vertex list
            for cid, verts in vertices.items():
                if isinstance(verts, list):
                    communities.append({
                        "id": str(cid),
                        "level": 0,
                        "vertices": verts,
                        "size": len(verts),
                    })

        return [c for c in communities if c["size"] >= 2]

    # ── networkx fallback ─────────────────────────────────────

    def _fallback_networkx(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fall back to local networkx Louvain (for small graphs)."""
        log.info("Falling back to networkx Louvain for community detection")
        self._resolved_engine = "networkx"
        return self._run_networkx(context)

    def _run_networkx(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run community detection using local Python libraries.

        Prefers Leiden (via leidenalg) over Louvain when available.
        Falls back to networkx Louvain for small graphs if leidenalg missing.
        """
        # Use Leiden if available and requested
        if HAS_LEIDEN and self._algorithm in (ALGORITHM_LEIDEN, ALGORITHM_LOUVAIN, "auto"):
            return self._run_leiden(context)

        # Fall back to Louvain
        return self._run_louvain(context)

    def _run_leiden(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Leiden community detection using leidenalg library.

        Leiden (Traag et al., 2019) improves on Louvain by guaranteeing
        connected communities and offering better partition quality.
        Requires: pip install leidenalg python-igraph
        """
        import igraph as ig

        from hugegraph_llm.config import huge_settings

        vertices = context.get("vertices")
        edges = context.get("edges")

        if vertices is None or edges is None:
            vertices, edges = self._fetch_graph_from_hugegraph()

        if not vertices:
            log.warning("No vertices found for Leiden community detection")
            context["communities"] = []
            context["community_count"] = 0
            context["engine_used"] = "leiden"
            return context

        # Build igraph graph (required by leidenalg)
        node_ids = []
        node_map = {}  # id -> index

        for v in vertices:
            vid = v.get("id", "")
            if vid:
                node_map[vid] = len(node_ids)
                node_ids.append(vid)

        g = ig.Graph()
        g.add_vertices(len(node_ids))

        edge_list = []
        edge_weights = []
        for e in edges:
            out_v = e.get("outV", "")
            in_v = e.get("inV", "")
            if out_v in node_map and in_v in node_map:
                src_idx = node_map[out_v]
                dst_idx = node_map[in_v]
                edge_list.append((src_idx, dst_idx))
                edge_weights.append(e.get("weight", 1))

        if edge_list:
            g.add_edges(edge_list)
            g.es["weight"] = edge_weights

        log.info(
            "Built igraph for Leiden: %d nodes, %d edges",
            g.vcount(), g.ecount(),
        )

        # Run Leiden algorithm
        partition = None
        try:
            partition = leidenalg.find_partition(
                g, leidenalg.ModularityVertexPartition,
                seed=42,
            )
        except TypeError:
            # Some leidenalg versions don't support all params on ModularityVertexPartition
            try:
                partition = leidenalg.find_partition(
                    g, leidenalg.CPMVertexPartition, seed=42
                )
            except Exception:
                pass

        if partition is None:
            log.warning("Leiden algorithm failed, falling back to Louvain")
            return self._run_louvain(context)

        communities = {}
        for idx, comm_id in enumerate(partition.membership):
            vid = node_ids[idx]
            communities.setdefault(comm_id, []).append(vid)

        result = []
        for cid, vids in sorted(communities.items()):
            if len(vids) >= self._min_community_size:
                result.append({
                    "id": f"LD_C{cid}",
                    "level": 0,
                    "vertices": vids,
                    "size": len(vids),
                })

        # Enrich with vertex/edge details
        vertex_map = {v["id"]: v for v in vertices if "id" in v}
        G_nx = nx.Graph()
        for v in vertices:
            vid = v.get("id", "")
            if vid:
                G_nx.add_node(vid, label=v.get("label", ""), props=v.get("props", {}))
        for e in edges:
            out_v = e.get("outV", "")
            in_v = e.get("inV", "")
            if out_v and in_v:
                G_nx.add_edge(out_v, in_v, weight=e.get("weight", 1), label=e.get("label", ""))

        for comm in result:
            vset = set(comm["vertices"])
            comm["vertex_details"] = [
                {
                    "id": vid,
                    "label": vertex_map[vid].get("label", "unknown") if vid in vertex_map else "unknown",
                    "props": vertex_map[vid].get("props", {}) if vid in vertex_map else {},
                }
                for vid in comm["vertices"][:50]
            ]
            comm["edge_details"] = [
                {"label": G_nx[u][v].get("label", ""), "outV": u, "inV": v}
                for u, v in G_nx.edges() if u in vset and v in vset
            ][:50]
            n = len(comm["vertices"])
            comm["density"] = (
                len(comm["edge_details"]) / (n * (n - 1) / 2) if n > 1 else 0.0
            )
            comm["modularity_class"] = "leiden"

        context["communities"] = result
        context["community_count"] = len(result)
        context["engine_used"] = "leiden"
        log.info(
            "Leiden: %d communities from %d vertices (quality > Louvain)",
            len(result), len(vertices),
        )
        return context

    def _run_louvain(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Louvain community detection using networkx locally.

        Fetches all vertices/edges from HugeGraph, builds a networkx Graph,
        and runs Louvain. Only suitable for graphs < ~10K vertices.
        Fallback when leidenalg is not available.

        Fetches all vertices/edges from HugeGraph, builds a networkx Graph,
        and runs Louvain. Only suitable for graphs < ~10K vertices.
        """
        from hugegraph_llm.config import huge_settings

        # Try to use data from context first
        vertices = context.get("vertices")
        edges = context.get("edges")

        if vertices is None or edges is None:
            vertices, edges = self._fetch_graph_from_hugegraph()

        if not vertices:
            log.warning("No vertices found for community detection")
            context["communities"] = []
            context["community_count"] = 0
            context["engine_used"] = "networkx"
            return context

        # Build networkx graph
        G = nx.Graph()
        for v in vertices:
            vid = v.get("id", "")
            if vid:
                G.add_node(vid, label=v.get("label", ""), props=v.get("props", {}))
        for e in edges:
            out_v = e.get("outV", "")
            in_v = e.get("inV", "")
            if out_v and in_v:
                if G.has_edge(out_v, in_v):
                    G[out_v][in_v]["weight"] += 1
                else:
                    G.add_edge(out_v, in_v, weight=1, label=e.get("label", ""))

        log.info("Built graph for Louvain: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        # Run Louvain
        from networkx.algorithms.community import louvain_communities

        raw = louvain_communities(G, weight="weight", resolution=1.0, seed=42)

        communities = []
        for i, nodes in enumerate(raw):
            if len(nodes) >= self._min_community_size:
                communities.append({
                    "id": f"L0_C{i}",
                    "level": 0,
                    "vertices": list(nodes),
                    "size": len(nodes),
                })

        # Enrich with vertex/edge details
        vertex_map = {v["id"]: v for v in vertices if "id" in v}
        for comm in communities:
            vset = set(comm["vertices"])
            comm["vertex_details"] = [
                {"id": vid, "label": vertex_map[vid].get("label", "unknown") if vid in vertex_map else "unknown"}
                for vid in comm["vertices"][:50]
            ]
            comm["edge_details"] = [
                {"label": G[u][v].get("label", ""), "outV": u, "inV": v}
                for u, v in G.edges() if u in vset and v in vset
            ][:50]
            comm["density"] = (
                len(comm["edge_details"]) / (len(comm["vertices"]) * (len(comm["vertices"]) - 1) / 2)
                if len(comm["vertices"]) > 1 else 0
            )

        context["communities"] = communities
        context["community_count"] = len(communities)
        context["engine_used"] = "networkx"
        log.info("networkx Louvain: %d communities from %d vertices", len(communities), len(vertices))
        return context

    def _fetch_graph_from_hugegraph(self) -> tuple:
        """Fetch vertices and edges from HugeGraph for local processing."""
        from hugegraph_llm.config import huge_settings

        groovy = f"""
        def res = [:];
        res.vertices = g.V().project('id','label','props')
            .by(id()).by(label()).by(valueMap().by(unfold()))
            .limit({huge_settings.max_graph_items * 100}).toList();
        res.edges = g.E().project('id','label','inV','outV','props')
            .by(id()).by(label())
            .by(inV().id()).by(outV().id())
            .by(valueMap().by(unfold()))
            .limit({huge_settings.max_graph_items * 200}).toList();
        return res;
        """
        try:
            resp = self._client.gremlin().exec(groovy)
            data = resp.get("data", [{}])[0] if isinstance(resp, dict) else {}
            return data.get("vertices", []), data.get("edges", [])
        except Exception as e:
            log.error("Failed to fetch graph data: %s", e)
            return [], []
