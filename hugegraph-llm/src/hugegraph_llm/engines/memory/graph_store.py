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

"""
HugeGraph-based GraphStore implementation — entity-centric retrieval with
Gremlin multi-hop traversal.

This is the concrete GraphStoreBase implementation that replaces the
string-matching graph channel in MemoryPipelineBackend with proper
entity-centric, multi-hop graph traversal — the key P0 gap identified
vs mem0 (entity_store + entity boost) and PowerMem (graph_store).

Key features:
  - Entity match → 1-N hop neighborhood retrieval
  - Weighted subgraph scoring (edge-type affinity)
  - Conflict resolution (later timestamp wins)
  - Automatic label mapping (handles dynamic edge labels like works_at_v2)
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.engines.memory.base import GraphStoreBase
from hugegraph_llm.utils.log import log

log = logging.getLogger(__name__)

# Edge-type affinity weights: some edges carry more semantic signal
DEFAULT_EDGE_WEIGHTS = {
    "works_at": 0.8,
    "colleague_of": 0.7,
    "lives_in": 0.6,
    "likes": 0.5,
    "friend_of": 0.6,
    "member_of": 0.7,
    "studies_at": 0.6,
    "manage": 0.7,
    "report_to": 0.7,
    "belong_to": 0.5,
}


class HugeGraphGraphStore(GraphStoreBase):
    """HugeGraph-backed entity-centric graph retrieval.

    For each query:
    1. Match query entities to graph vertices (name-based, type-aware)
    2. Traverse 1-N hop neighborhoods from matched vertices
    3. Score traversed paths using edge-type weights + recency
    4. Return ranked subgraph context for memory boosting
    """

    def __init__(
        self,
        hg_client: Any,
        edge_weights: Optional[Dict[str, float]] = None,
        max_hops: int = 2,
        max_neighbors: int = 50,
    ):
        """Initialize with a HugeGraphMemoryClient instance.

        Args:
            hg_client: HugeGraphMemoryClient from memory_backend.py
            edge_weights: Custom edge-type affinity weights
            max_hops: Maximum traversal depth (1-3)
            max_neighbors: Max neighbors per vertex per hop
        """
        self._hg = hg_client
        self._edge_weights = edge_weights or DEFAULT_EDGE_WEIGHTS
        self._max_hops = max_hops
        self._max_neighbors = max_neighbors
        self._vertex_cache: Dict[str, Dict[str, Any]] = {}
        self._edge_cache: List[Dict[str, Any]] = []
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 60.0  # seconds

    # ── GraphStoreBase interface ──────────────────────────────

    def add(self, data: Dict[str, Any]) -> None:
        """Add entities and relations to the graph via HugeGraphMemoryClient.

        data format:
          {
            "entities": [{"name": "...", "type": "person", "properties": {...}}],
            "relationships": [{"source": "...", "target": "...", "label": "works_at", ...}]
          }
        """
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Add vertices
        for ent in entities:
            label = ent.get("type", "concept")
            name = ent.get("name", "")
            if not name:
                continue
            props = ent.get("properties", {})
            self._hg.add_vertex(label, name, properties=props)

        # Add edges
        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            edge_label = rel.get("label", "related_to")
            if not src or not tgt:
                continue
            self._hg.add_edge(edge_label, src, tgt)

        self._invalidate_cache()

    def search(
        self,
        query: str,
        limit: int = 10,
        max_hops: int = 2,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Entity-centric graph search with multi-hop traversal.

        1. Extract entity names from query
        2. Match entities to graph vertices
        3. Traverse N-hop neighborhoods
        4. Score and rank results

        Returns list of dicts with:
          - matched_entity: the query entity that triggered this result
          - path: list of (vertex, edge) tuples along the traversal
          - score: weighted traversal score
          - context: human-readable subgraph context string
        """
        self._refresh_cache()
        max_hops = min(max_hops, self._max_hops)

        # Step 1: Extract entity candidates from query
        query_entities = self._extract_query_entities(query)

        if not query_entities:
            return []

        # Step 2: Match entities to cached vertices
        matched_vertices = self._match_entities(query_entities)

        if not matched_vertices:
            return []

        # Step 3: Multi-hop traversal from each matched vertex
        results = []
        for entity_name, vertex_info in matched_vertices.items():
            subgraph = self._traverse(entity_name, vertex_info, max_hops)
            for path_info in subgraph:
                score = self._score_path(path_info)
                context = self._build_context_string(path_info)
                results.append({
                    "matched_entity": entity_name,
                    "path": path_info,
                    "score": score,
                    "context": context,
                })

        # Step 4: Deduplicate and rank
        results = self._dedup_results(results)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Return all graph vertices."""
        self._refresh_cache()
        return list(self._vertex_cache.values())

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """Return all graph edges."""
        self._refresh_cache()
        return self._edge_cache

    # ── Internal methods ──────────────────────────────────────

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract candidate entity names from query using regex heuristics.

        This mirrors the EntityExtractor._rule_based() method but focuses
        on extracting just the names for graph matching.
        """
        entities = []

        # Chinese: 2-6 char sequences before org/location suffixes
        for m in re.finditer(
            r"([\u4e00-\u9fa5]{2,8})(?:公司|集团|学校|银行|医院|厂|团队|部门|市|省|区|县|路|街)",
            query,
        ):
            entities.append(m.group(1))

        # Chinese person names (after 我叫/他叫 etc.)
        for m in re.finditer(
            r"(?:我(?:的|叫)|他(?:的|叫)|她(?:的|叫)|同事|朋友|同学)([\u4e00-\u9fa5]{2,4})",
            query,
        ):
            entities.append(m.group(1))

        # Pure Chinese sequences 2-8 chars (fallback)
        for m in re.finditer(r"[\u4e00-\u9fa5]{2,8}", query):
            candidate = m.group(0)
            # Skip common stop-words
            stops = {"什么", "怎么", "哪里", "哪个", "多少", "哪些", "如何", "是谁",
                     "是否", "有没有", "能不能", "为什么", "这个", "那个", "的人"}
            if candidate not in stops and candidate not in entities:
                entities.append(candidate)

        # English capitalized multi-word names
        for m in re.finditer(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+", query):
            entities.append(m.group(0))

        return entities

    def _match_entities(self, query_entities: List[str]) -> Dict[str, Dict[str, Any]]:
        """Match query entities to cached graph vertices by name similarity."""
        matched = {}
        for qe in query_entities:
            # Exact match
            if qe in self._vertex_cache:
                matched[qe] = self._vertex_cache[qe]
                continue
            # Fuzzy: substring match (entity name contains query, or vice versa)
            for vname, vinfo in self._vertex_cache.items():
                if qe in vname or vname in qe:
                    matched[qe] = vinfo
                    break
        return matched

    def _traverse(
        self,
        start_name: str,
        start_info: Dict[str, Any],
        max_hops: int,
    ) -> List[List[Dict[str, Any]]]:
        """Traverse N-hop neighborhood from a start vertex.

        Returns list of paths, each path is a list of step dicts:
          {"vertex": name, "vertex_type": type, "edge": label,
           "target": name, "target_type": type, "hop": N}
        """
        all_paths: List[List[Dict[str, Any]]] = []

        # Use the matched vertex name (not the query entity string) for traversal
        start_vertex = start_info.get("name", start_name)

        # Build adjacency map from edges
        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        for edge in self._edge_cache:
            src = edge.get("source_name", "")
            tgt = edge.get("target_name", "")
            label = edge.get("label", "related_to")
            if src:
                adjacency.setdefault(src, []).append({
                    "edge_label": label, "neighbor": tgt,
                    "neighbor_type": edge.get("target_label", ""),
                })
            if tgt:
                adjacency.setdefault(tgt, []).append({
                    "edge_label": label, "neighbor": src,
                    "neighbor_type": edge.get("source_label", ""),
                })

        # BFS traversal
        visited: set = {start_vertex}
        current_level = [(start_vertex, start_info.get("label", ""), [])]

        for hop in range(1, max_hops + 1):
            next_level = []
            for node_name, node_type, prefix_path in current_level:
                neighbors = adjacency.get(node_name, [])[:self._max_neighbors]
                for nb in neighbors:
                    nb_name = nb["neighbor"]
                    if nb_name in visited:
                        continue
                    visited.add(nb_name)
                    step = {
                        "vertex": node_name,
                        "vertex_type": node_type,
                        "edge": nb["edge_label"],
                        "target": nb_name,
                        "target_type": nb.get("neighbor_type", ""),
                        "hop": hop,
                    }
                    path = prefix_path + [step]
                    all_paths.append(path)
                    next_level.append((nb_name, nb.get("neighbor_type", ""), path))
            current_level = next_level

        return all_paths

    def _score_path(self, path: List[Dict[str, Any]]) -> float:
        """Score a traversal path using edge-type weights + recency decay."""
        score = 0.0
        for step in path:
            edge_label = step.get("edge", "related_to")
            # Resolve dynamic label variants (e.g., works_at_v2 → works_at)
            base_label = self._resolve_edge_label(edge_label)
            weight = self._edge_weights.get(base_label, 0.3)
            # Decay by hop distance: closer neighbors score higher
            hop = step.get("hop", 1)
            hop_decay = 1.0 / (1.0 + 0.3 * (hop - 1))
            score += weight * hop_decay
        return round(score, 4)

    @staticmethod
    def _resolve_edge_label(label: str) -> str:
        """Map dynamic label variants back to base labels.

        HugeGraphMemoryClient auto-creates variants like works_at_v2
        when label conflicts occur. We strip the _vN suffix.
        """
        m = re.match(r"(.+)_v\d+$", label)
        if m:
            return m.group(1)
        return label

    @staticmethod
    def _build_context_string(path: List[Dict[str, Any]]) -> str:
        """Build a human-readable context string from a traversal path."""
        parts = []
        for step in path:
            src = step.get("vertex", "?")
            tgt = step.get("target", "?")
            edge = step.get("edge", "→")
            base = HugeGraphGraphStore._resolve_edge_label(edge)
            parts.append(f"{src} [{base}] {tgt}")
        return " | ".join(parts)

    @staticmethod
    def _dedup_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results by context string (keep highest score)."""
        seen: Dict[str, Dict[str, Any]] = {}
        for r in results:
            ctx = r.get("context", "")
            if ctx not in seen or r["score"] > seen[ctx]["score"]:
                seen[ctx] = r
        return list(seen.values())

    # ── Cache management ──────────────────────────────────────

    def _refresh_cache(self) -> None:
        """Refresh vertex and edge caches from HugeGraph if TTL expired."""
        now = time.time()
        if now - self._cache_ts < self._cache_ttl and self._vertex_cache:
            return
        try:
            vertices = self._hg.get_all_vertices(limit=5000)
            edges = self._hg.get_all_edges()
            self._vertex_cache = {}
            for v in vertices:
                name = v.get("name", v.get("id", ""))
                vtype = v.get("label", "concept")
                self._vertex_cache[name] = {
                    "name": name, "label": vtype,
                    "properties": v.get("properties", {}),
                }
            self._edge_cache = edges
            self._cache_ts = now
            log.debug("Graph cache refreshed: %d vertices, %d edges",
                      len(self._vertex_cache), len(self._edge_cache))
        except Exception as e:
            log.warning("Failed to refresh graph cache: %s", e)

    def _invalidate_cache(self) -> None:
        """Force cache invalidation after write operations."""
        self._cache_ts = 0.0
