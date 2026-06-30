# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not in this file except in compliance
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

"""
Community detection integration for GraphRAG.

Implements Leiden/Louvain community detection on graph data stored in
HugeGraph, producing hierarchical community structures for global
query answering.

Inspired by Microsoft GraphRAG's community detection approach, adapted
to work with HugeGraph's Gremlin-based query interface.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx

from hugegraph_llm.utils.log import log


class CommunityDetector:
    """
    Detect communities in graph data using Leiden or Louvain algorithms.

    Works with graph data extracted from HugeGraph, producing a hierarchical
    community structure that enables global-level query answering.

    The community detection pipeline:
    1. Load graph data from HugeGraph (or from context)
    2. Build a NetworkX graph
    3. Run community detection (Leiden or Louvain)
    4. Produce hierarchical community structure
    5. Optionally merge with existing communities for incremental updates
    """

    def __init__(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        max_levels: int = 5,
        min_community_size: int = 3,
    ):
        """
        Args:
            algorithm: Community detection algorithm ('louvain' or 'leiden').
            resolution: Resolution parameter for community detection.
                       Higher values produce more, smaller communities.
            max_levels: Maximum number of hierarchy levels.
            min_community_size: Minimum community size to keep.
        """
        self.algorithm = algorithm
        self.resolution = resolution
        self.max_levels = max_levels
        self.min_community_size = min_community_size

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run community detection on graph data.

        Args:
            context: Dict containing 'graph_result' or raw vertices/edges,
                     and optionally 'graph_client' for loading from HugeGraph.

        Returns:
            Updated context with 'communities' and 'community_hierarchy'.
        """
        # Phase 1: Build NetworkX graph from available data
        graph = self._build_networkx_graph(context)

        if graph.number_of_nodes() == 0:
            log.warning("No graph data available for community detection")
            context["communities"] = []
            context["community_hierarchy"] = {}
            context["community_count"] = 0
            context["community_algorithm"] = self.algorithm
            return context

        # Phase 2: Run community detection
        if self.algorithm == "leiden":
            communities = self._detect_leiden(graph)
        else:
            communities = self._detect_louvain(graph)

        # Phase 3: Build hierarchy
        hierarchy = self._build_hierarchy(graph, communities)

        # Phase 4: Filter small communities
        communities = [c for c in communities if len(c) >= self.min_community_size]

        log.info(
            "Detected %d communities (algorithm=%s, resolution=%.2f, nodes=%d, edges=%d)",
            len(communities),
            self.algorithm,
            self.resolution,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        context["communities"] = communities
        context["community_hierarchy"] = hierarchy
        context["community_count"] = len(communities)
        context["community_algorithm"] = self.algorithm
        return context

    def _build_networkx_graph(self, context: Dict[str, Any]) -> nx.Graph:
        """
        Build a NetworkX graph from context data.

        Tries multiple data sources in order:
        1. Direct vertices/edges in context
        2. graph_result (from previous GraphQueryNode)
        3. graph_client for loading from HugeGraph
        """
        graph = nx.Graph()

        # Try loading from vertices/edges
        vertices = context.get("vertices", [])
        edges = context.get("edges", [])

        if vertices or edges:
            for vertex in vertices:
                vid = vertex.get("id", vertex.get("name", ""))
                if vid:
                    label = vertex.get("label", "unknown")
                    props = vertex.get("properties", {})
                    graph.add_node(vid, label=label, **props)

            for edge in edges:
                source = edge.get("outV", edge.get("start", ""))
                target = edge.get("inV", edge.get("end", ""))
                label = edge.get("label", edge.get("type", "unknown"))
                if source and target:
                    graph.add_edge(source, target, label=label)
            return graph

        # Try loading from graph_result (flat relation format)
        graph_result = context.get("graph_result", [])
        if graph_result:
            for item in graph_result:
                if isinstance(item, str):
                    # Parse flat relation format: "vertexA--[edge]-->vertexB"
                    nodes, edge_label = self._parse_flat_relation(item)
                    if nodes:
                        for node in nodes:
                            if not graph.has_node(node):
                                graph.add_node(node)
                        if len(nodes) == 2:
                            graph.add_edge(nodes[0], nodes[1], label=edge_label)
            return graph

        return graph

    def _detect_louvain(self, graph: nx.Graph) -> List[List[str]]:
        """
        Detect communities using the Louvain algorithm.

        Returns a list of communities, where each community is a list of node IDs.
        """
        try:
            from networkx.algorithms.community import louvain_communities

            communities_set = louvain_communities(graph, resolution=self.resolution, seed=42)
            communities = [list(c) for c in communities_set]
        except ImportError:
            log.warning("Louvain not available, falling back to greedy modularity")
            from networkx.algorithms.community import greedy_modularity_communities

            communities_set = greedy_modularity_communities(graph, resolution=self.resolution)
            communities = [list(c) for c in communities_set]

        return communities

    def _detect_leiden(self, graph: nx.Graph) -> List[List[str]]:
        """
        Detect communities using the Leiden algorithm.

        Falls back to Louvain if python-igraph is not available.
        """
        try:
            import igraph as ig

            # Convert NetworkX graph to igraph
            node_mapping = {n: i for i, n in enumerate(graph.nodes())}
            edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]

            ig_graph = ig.Graph(n=graph.number_of_nodes(), edges=edges, directed=False)

            # Run Leiden
            partition = ig_graph.community_leiden(resolution_parameter=self.resolution, n_iterations=5)

            # Convert back to node IDs
            communities = []
            for community in partition:
                node_ids = [list(graph.nodes())[i] for i in community if i < graph.number_of_nodes()]
                if node_ids:
                    communities.append(node_ids)

        except ImportError:
            log.warning("python-igraph not available, falling back to Louvain for Leiden mode")
            communities = self._detect_louvain(graph)

        return communities

    def _build_hierarchy(self, graph: nx.Graph, communities: List[List[str]]) -> Dict[str, Any]:
        """
        Build a hierarchical community structure.

        Creates multiple levels of abstraction by recursively grouping
        communities based on inter-community edges.
        """
        hierarchy: Dict[str, Any] = {
            "levels": [],
            "total_levels": 0,
        }

        current_level_communities = communities
        level = 0

        while len(current_level_communities) > 1 and level < self.max_levels:
            level_info = {
                "level": level,
                "community_count": len(current_level_communities),
                "communities": [],
            }

            for i, community in enumerate(current_level_communities):
                community_info = {
                    "id": f"L{level}_C{i}",
                    "size": len(community),
                    "members": community,
                    "density": self._compute_community_density(graph, community),
                }
                level_info["communities"].append(community_info)

            hierarchy["levels"].append(level_info)

            # Create next level by treating communities as nodes
            if len(current_level_communities) <= 1:
                break

            meta_graph = nx.Graph()
            for i, comm_a in enumerate(current_level_communities):
                for j, comm_b in enumerate(current_level_communities):
                    if i < j:
                        inter_edges = self._count_inter_community_edges(graph, comm_a, comm_b)
                        if inter_edges > 0:
                            meta_graph.add_edge(f"C{i}", f"C{j}", weight=inter_edges)

            if meta_graph.number_of_nodes() <= 1:
                break

            # Detect communities at the meta level
            meta_communities = self._detect_louvain(meta_graph)
            current_level_communities = [
                [
                    node
                    for meta_node in meta_comm
                    for node in self._resolve_meta_community(meta_node, current_level_communities)
                ]
                for meta_comm in meta_communities
            ]
            level += 1

        hierarchy["total_levels"] = len(hierarchy["levels"])
        return hierarchy

    def _resolve_meta_community(self, meta_node_id: str, current_communities: List[List[str]]) -> List[str]:
        """Resolve a meta-level community node back to original node IDs."""
        try:
            idx = int(meta_node_id.replace("C", ""))
            if idx < len(current_communities):
                return current_communities[idx]
        except (ValueError, IndexError):
            pass
        return []

    @staticmethod
    def _compute_community_density(graph: nx.Graph, community: List[str]) -> float:
        """Compute the edge density within a community."""
        if len(community) < 2:
            return 0.0
        subgraph = graph.subgraph(community)
        possible_edges = len(community) * (len(community) - 1) / 2
        actual_edges = subgraph.number_of_edges()
        return actual_edges / possible_edges if possible_edges > 0 else 0.0

    @staticmethod
    def _count_inter_community_edges(graph: nx.Graph, comm_a: List[str], comm_b: List[str]) -> int:
        """Count edges between two communities."""
        count = 0
        set_b = set(comm_b)
        for node in comm_a:
            for neighbor in graph.neighbors(node):
                if neighbor in set_b:
                    count += 1
        return count

    @staticmethod
    def _parse_flat_relation(relation_str: str) -> Tuple[List[str], str]:
        """
        Parse a flat relation string into nodes and edge label.

        Handles formats like:
        - "vertexA--[edgeLabel]-->vertexB"
        - "vertexA<--[edgeLabel]--vertexB"
        - "vertexA{props}--[edgeLabel]-->vertexB{props}"
        """
        import re

        # Match the edge pattern
        edge_match = re.search(r"--\[([^\]]+)\]-->", relation_str)
        edge_match_rev = re.search(r"<--\[([^\]]+)\]--", relation_str)

        edge_label = ""
        if edge_match:
            edge_label = edge_match.group(1)
        elif edge_match_rev:
            edge_label = edge_match_rev.group(1)

        # Extract node IDs (before { and after } or edge pattern)
        nodes = []
        parts = re.split(r"(?:--\[.*?\]-->|<--\[.*?\]--)", relation_str)
        for part in parts:
            part = part.strip()
            if part:
                # Extract just the ID part before any {props}
                id_match = re.match(r"^([^-<>\[\]{\s]+)", part)
                if id_match:
                    nodes.append(id_match.group(1))

        return nodes, edge_label
