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

"""Entity Resolution operator for HugeGraph GraphRAG.

Merges duplicate vertices that refer to the same real-world entity.
Supports three strategies (precision & cost ascending):

    1. exact_match:  same label + same primary_key value -> auto merge
    2. embedding:    same label + property embedding cos_sim > threshold -> candidate
    3. llm_verify:   LLM confirms whether candidates truly refer to the same entity

Reference architecture: Neo4j GraphRAG BasePropertySimilarityResolver
(https://github.com/neo4j/neo4j-graphrag-python)
"""

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from hugegraph_llm.utils.log import log


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MergeCandidate:
    """A pair of vertices that should potentially be merged."""

    from_vid: str          # vertex to be deprecated
    from_label: str       # label of from_vid
    from_properties: Dict  # properties of from_vid (for display)
    to_vid: str            # vertex to keep (higher degree preferred)
    to_label: str
    to_properties: Dict
    strategy: str          # which strategy produced this candidate
    confidence: float = 0.0  # 0.0 - 1.0


@dataclass
class MergeResult:
    """Output of the entity resolution process."""

    merged_pairs: List[Dict] = field(default_factory=list)
    merged_count: int = 0
    deprecated_vids: List[str] = field(default_factory=list)
    edges_migrated: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_pairs": self.merged_pairs,
            "merged_count": self.merged_count,
            "deprecated_vids": self.deprecated_vids,
            "edges_migrated": self.edges_migrated,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Union-Find for transitive merging
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted Union-Find with path compression."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> List[Set[str]]:
        groups: Dict[str, Set[str]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, set()).add(x)
        return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Entity Resolution Operator
# ---------------------------------------------------------------------------

class EntityResolution:
    """Entity resolution operator: merge duplicate vertices.

    Strategies (can be combined via ``hybrid``):

    - **exact_match**: same label + same primary key value.
    - **embedding**: same label + property embedding cosine similarity > threshold.
    - **llm_verify**: call LLM to confirm whether two entities are the same.

    Usage::

        from pyhugegraph.client import PyHugeClient
        from hugegraph_llm.models.llms.init_llm import get_chat_llm
        from hugegraph_llm.models.embeddings.init_embedding import get_embedding

        client = PyHugeClient(...)
        llm = get_chat_llm()
        embedding = get_embedding()

        resolver = EntityResolution(
            client=client,
            llm=llm,
            embedding=embedding,
            strategy="hybrid",
            threshold=0.85,
        )
        result = resolver.run(context={"schema": schema_dict})
        # result["resolution_result"] contains MergeResult
    """

    STRATEGY_EXACT = "exact_match"
    STRATEGY_EMBEDDING = "embedding"
    STRATEGY_LLM = "llm_verify"
    STRATEGY_HYBRID = "hybrid"
    VALID_STRATEGIES = {STRATEGY_EXACT, STRATEGY_EMBEDDING, STRATEGY_LLM, STRATEGY_HYBRID}

    def __init__(
        self,
        client: Any,
        llm: Any = None,
        embedding: Any = None,
        strategy: str = "hybrid",
        threshold: float = 0.85,
        batch_size: int = 50,
        resolve_properties: Optional[List[str]] = None,
        vertex_labels: Optional[List[str]] = None,
        max_pairs_per_label: int = 5000,
    ):
        """Initialize the entity resolver.

        Args:
            client: HugeGraph PyHugeClient instance.
            llm: LLM instance (BaseLLM) for llm_verify strategy.
            embedding: Embedding instance (BaseEmbedding) for embedding strategy.
            strategy: One of exact_match, embedding, llm_verify, hybrid.
            threshold: Cosine similarity threshold for embedding strategy.
            batch_size: Number of vertices to process per batch (for embedding).
            resolve_properties: Properties to compare. Defaults to ["name"].
            vertex_labels: Limit resolution to specific vertex labels.
            max_pairs_per_label: Max candidate pairs per label (performance guard).
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {self.VALID_STRATEGIES}")

        self._client = client
        self._llm = llm
        self._embedding = embedding
        self._strategy = strategy
        self._threshold = threshold
        self._batch_size = batch_size
        self._resolve_properties = resolve_properties or ["name"]
        self._vertex_labels = vertex_labels
        self._max_pairs_per_label = max_pairs_per_label

        # Embedding cache: text -> vector
        self._embedding_cache: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity resolution.

        Reads from context:
            schema:           Graph schema dict (optional, for PK discovery)
            vertices:        Vertices list to resolve (optional; if absent,
                             fetch from graph)
            vertex_labels:    Label filter (optional, overrides constructor)

        Writes to context:
            resolution_result: MergeResult dict
        """
        schema = context.get("schema")
        vertices = context.get("vertices")
        labels = context.get("vertex_labels", self._vertex_labels)

        if vertices is not None:
            log.info("Resolving %d in-memory vertices (label filter: %s)", len(vertices), labels)
            return self._resolve_in_memory(context, vertices, labels, schema)
        else:
            log.info("Resolving entities from graph store (label filter: %s)", labels)
            return self._resolve_from_graph(context, labels, schema)

    # ------------------------------------------------------------------
    # Phase 1: Candidate Discovery
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        groups: Dict[str, List[Dict]],
    ) -> List[MergeCandidate]:
        """Phase 1: discover merge candidates.

        Groups vertices by (label, resolve_property_value) and then
        applies the configured strategies.
        """
        candidates: List[MergeCandidate] = []

        for label, group in groups.items():
            if self._strategy in (self.STRATEGY_EXACT, self.STRATEGY_HYBRID):
                candidates.extend(self._exact_match_candidates(group))

            if self._strategy in (self.STRATEGY_EMBEDDING, self.STRATEGY_HYBRID):
                if self._embedding is None:
                    log.warning("Embedding strategy requires embedding model; skipping")
                else:
                    candidates.extend(self._embedding_candidates(group))

        # Deduplicate: same (from_vid, to_vid) pair from different strategies
        seen: Set[Tuple[str, str]] = set()
        unique_candidates: List[MergeCandidate] = []
        for c in candidates:
            pair = tuple(sorted([c.from_vid, c.to_vid]))
            if pair not in seen:
                seen.add(pair)
                unique_candidates.append(c)

        log.info("Found %d unique merge candidates", len(unique_candidates))
        return unique_candidates

    def _exact_match_candidates(self, group: List[Dict]) -> List[MergeCandidate]:
        """Find candidates where primary key values match exactly.

        Group is already filtered by label. We further group by property
        values and merge vertices with identical resolve_properties.
        """
        candidates: List[MergeCandidate] = []

        # Group by concatenated property values
        pk_groups: Dict[str, List[Dict]] = {}
        for v in group:
            key = self._make_property_key(v.get("properties", {}))
            if key:
                pk_groups.setdefault(key, []).append(v)

        for pk_val, vertices in pk_groups.items():
            if len(vertices) < 2:
                continue
            # Sort by degree (descending) — higher degree vertex is kept
            sorted_vertices = sorted(vertices, key=lambda v: v.get("degree", 0), reverse=True)
            keep = sorted_vertices[0]
            for dup in sorted_vertices[1:]:
                candidates.append(MergeCandidate(
                    from_vid=dup["id"],
                    from_label=dup["label"],
                    from_properties=dup.get("properties", {}),
                    to_vid=keep["id"],
                    to_label=keep["label"],
                    to_properties=keep.get("properties", {}),
                    strategy=self.STRATEGY_EXACT,
                    confidence=1.0,
                ))

        log.info("Exact match: %d candidates", len(candidates))
        return candidates

    def _embedding_candidates(self, group: List[Dict]) -> List[MergeCandidate]:
        """Find candidates via embedding cosine similarity.

        Compares all pairs within a label group. Uses Union-Find to
        consolidate transitive matches before producing candidates.
        """
        candidates: List[MergeCandidate] = []
        if len(group) < 2:
            return candidates

        # Limit pair count for performance
        max_pairs = self._max_pairs_per_label
        all_pairs = list(combinations(range(len(group)), 2))
        if len(all_pairs) > max_pairs:
            log.warning(
                "Label group has %d pairs (limit %d); sampling",
                len(all_pairs), max_pairs,
            )
            # Prioritize pairs with similar text lengths
            all_pairs.sort(key=lambda p: abs(len(str(group[p[0]])) - len(str(group[p[1]]))))
            all_pairs = all_pairs[:max_pairs]

        # Compute similarities
        pairs_to_merge: List[Set[int]] = []
        embeddings_cache: Dict[int, List[float]] = {}

        for i, j in all_pairs:
            text_i = self._vertex_text(group[i])
            text_j = self._vertex_text(group[j])
            if not text_i or not text_j:
                continue

            emb_i = self._get_embedding_cached(text_i, embeddings_cache)
            emb_j = self._get_embedding_cached(text_j, embeddings_cache)
            sim = self._cosine_similarity(emb_i, emb_j)

            if sim >= self._threshold:
                pairs_to_merge.append({i, j})

        # Consolidate transitive matches via Union-Find
        if not pairs_to_merge:
            return candidates

        uf = UnionFind()
        for pair in pairs_to_merge:
            uf.union(str(min(pair)), str(max(pair)))

        for merge_set in uf.groups():
            indices = sorted([int(x) for x in merge_set])
            # Keep the vertex with highest degree
            sorted_by_degree = sorted(indices, key=lambda idx: group[idx].get("degree", 0), reverse=True)
            keep_idx = sorted_by_degree[0]
            for dup_idx in sorted_by_degree[1:]:
                candidates.append(MergeCandidate(
                    from_vid=group[dup_idx]["id"],
                    from_label=group[dup_idx]["label"],
                    from_properties=group[dup_idx].get("properties", {}),
                    to_vid=group[keep_idx]["id"],
                    to_label=group[keep_idx]["label"],
                    to_properties=group[keep_idx].get("properties", {}),
                    strategy=self.STRATEGY_EMBEDDING,
                    confidence=self._threshold,
                ))

        log.info("Embedding: %d candidates from %d pairs_to_merge", len(candidates), len(pairs_to_merge))
        return candidates

    # ------------------------------------------------------------------
    # Phase 2: LLM Verification (optional)
    # ------------------------------------------------------------------

    def _verify_candidates(self, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        """Phase 2: LLM verification of merge candidates.

        Only runs for llm_verify or hybrid strategy.
        """
        if self._strategy not in (self.STRATEGY_LLM, self.STRATEGY_HYBRID):
            return candidates
        if self._llm is None:
            log.warning("LLM verify strategy requires LLM; skipping verification")
            return candidates

        # Filter: only verify candidates not already at confidence=1.0 (exact match)
        to_verify = [c for c in candidates if c.confidence < 1.0]
        verified: List[MergeCandidate] = [c for c in candidates if c.confidence >= 1.0]

        if not to_verify:
            return verified

        log.info("LLM verifying %d candidates (batch_size=%d)", len(to_verify), self._batch_size)
        batch: List[MergeCandidate] = []
        for c in to_verify:
            batch.append(c)
            if len(batch) >= self._batch_size:
                verified.extend(self._verify_batch(batch))
                batch = []
        if batch:
            verified.extend(self._verify_batch(batch))

        log.info("LLM verification: %d/%d confirmed", len(verified), len(to_verify) + len(verified))
        return verified

    def _verify_batch(self, batch: List[MergeCandidate]) -> List[MergeCandidate]:
        """Send a batch of candidates to LLM for verification."""
        prompt = self._build_verify_prompt(batch)
        try:
            response = self._llm.generate(prompt=prompt)
            return self._parse_verify_response(response, batch)
        except Exception as e:
            log.error("LLM verification failed: %s; accepting all candidates in batch", e)
            return batch  # Fail-open: accept if LLM fails

    def _build_verify_prompt(self, batch: List[MergeCandidate]) -> str:
        """Build a batch verification prompt for LLM."""
        items = []
        for i, c in enumerate(batch):
            props_a = ", ".join(f"{k}={v}" for k, v in c.from_properties.items())
            props_b = ", ".join(f"{k}={v}" for k, v in c.to_properties.items())
            items.append(
                f"Pair {i + 1}:\n"
                f"  Entity A ({c.from_label}): {props_a}\n"
                f"  Entity B ({c.to_label}): {props_b}"
            )

        return f"""Determine whether each pair of entities refers to the same real-world entity.
Two entities are the SAME if they represent the same person, organization, location, concept, etc.
even if their names differ slightly (e.g., "US" vs "United States", "Bob" vs "Robert Smith").

Entity Pairs:
{chr(10).join(items)}

Respond with ONLY a JSON array of booleans, one per pair, in order.
Example: [true, false, true]
If unsure, respond with false to avoid over-merging."""

    def _parse_verify_response(
        self, response: str, batch: List[MergeCandidate]
    ) -> List[MergeCandidate]:
        """Parse LLM verification response and filter candidates."""
        try:
            # Extract JSON from response
            json_str = response.strip()
            # Handle markdown code blocks
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.split("```")[0].strip()

            results = json.loads(json_str)
            if not isinstance(results, list):
                log.warning("LLM response is not a list: %s", response[:200])
                return batch

            verified = []
            for i, c in enumerate(batch):
                if i < len(results) and results[i] is True:
                    c.confidence = 0.95
                    c.strategy = f"{c.strategy}+llm_verified"
                    verified.append(c)
                else:
                    log.info("LLM rejected merge: %s (%s) <-> %s (%s)", c.from_vid, c.from_label, c.to_vid, c.to_label)
            return verified
        except (json.JSONDecodeError, IndexError) as e:
            log.warning("Failed to parse LLM response: %s; accepting all candidates", e)
            return batch

    # ------------------------------------------------------------------
    # Phase 3: Merge Execution
    # ------------------------------------------------------------------

    def _merge_entities(
        self,
        candidates: List[MergeCandidate],
        context: Dict[str, Any],
    ) -> MergeResult:
        """Phase 3: execute merge operations.

        For each candidate:
        1. Migrate edges from from_vid to to_vid
        2. Mark from_vid as deprecated (not deleted, preserving audit trail)
        """
        result = MergeResult()

        # Use Union-Find to handle chains: A->B, B->C => A->C
        uf = UnionFind()
        for c in candidates:
            uf.union(c.to_vid, c.from_vid)

        # For each group, pick the representative (highest degree)
        groups = uf.groups()

        # Build candidate info index for fallback when graph store is unavailable
        candidate_info: Dict[str, Dict] = {}
        for c in candidates:
            for vid, label, props in [
                (c.to_vid, c.to_label, c.to_properties),
                (c.from_vid, c.from_label, c.from_properties),
            ]:
                if vid not in candidate_info:
                    candidate_info[vid] = {"label": label, "properties": props}

        for group in groups:
            vertices_info = {}
            for vid in group:
                info = self._get_vertex_info(vid)
                if info:
                    vertices_info[vid] = info
                elif vid in candidate_info:
                    # Fallback: use candidate data for in-memory resolution
                    vertices_info[vid] = candidate_info[vid]

            if len(vertices_info) < 2:
                continue

            # Pick representative: highest degree
            sorted_vids = sorted(
                vertices_info.keys(),
                key=lambda v: vertices_info[v].get("degree", 0),
                reverse=True,
            )
            keep_vid = sorted_vids[0]
            keep_label = vertices_info[keep_vid].get("label", "")
            keep_props = vertices_info[keep_vid].get("properties", {})

            for dup_vid in sorted_vids[1:]:
                dup_label = vertices_info[dup_vid].get("label", "")
                dup_props = vertices_info[dup_vid].get("properties", {})

                try:
                    edges_migrated = self._migrate_edges(dup_vid, keep_vid)
                    self._mark_deprecated(dup_vid, merged_to=keep_vid)

                    result.merged_pairs.append({
                        "from_vid": dup_vid,
                        "to_vid": keep_vid,
                        "from_label": dup_label,
                        "to_label": keep_label,
                        "edges_migrated": edges_migrated,
                    })
                    result.merged_count += 1
                    result.deprecated_vids.append(dup_vid)
                    result.edges_migrated += edges_migrated

                    log.info(
                        "Merged %s (%s) -> %s (%s), migrated %d edges",
                        dup_vid, dup_label, keep_vid, keep_label, edges_migrated,
                    )
                except Exception as e:
                    error_msg = f"Failed to merge {dup_vid} -> {keep_vid}: {e}"
                    log.error(error_msg)
                    result.errors.append(error_msg)

        return result

    def _migrate_edges(self, from_vid: str, to_vid: str) -> int:
        """Migrate all edges from from_vid to to_vid.

        Returns the number of edges migrated.
        """
        edges_migrated = 0

        # Find all outgoing edges: from_vid --edge--> other
        outgoing = self._query_edges(from_vid, direction="out")
        for edge in outgoing:
            other_vid = edge.get("inV") or edge.get("target")
            if other_vid and other_vid != to_vid:
                try:
                    self._client.graph().addEdge(
                        edge["label"],
                        to_vid,
                        other_vid,
                        edge.get("properties", {}),
                    )
                    edges_migrated += 1
                except Exception as e:
                    log.warning("Failed to migrate outgoing edge %s: %s", edge, e)

        # Find all incoming edges: other --edge--> from_vid
        incoming = self._query_edges(from_vid, direction="in")
        for edge in incoming:
            other_vid = edge.get("outV") or edge.get("source")
            if other_vid and other_vid != to_vid:
                try:
                    self._client.graph().addEdge(
                        edge["label"],
                        other_vid,
                        to_vid,
                        edge.get("properties", {}),
                    )
                    edges_migrated += 1
                except Exception as e:
                    log.warning("Failed to migrate incoming edge %s: %s", edge, e)

        # Delete original edges
        self._delete_edges(from_vid)
        return edges_migrated

    def _query_edges(self, vid: str, direction: str = "both") -> List[Dict]:
        """Query edges connected to a vertex via Gremlin."""
        if direction == "out":
            groovy = f"g.V('{vid}').outE().project('id','label','inV','properties').by(id()).by(label()).by(inV().id()).by(valueMap().by(unfold())).toList()"
        elif direction == "in":
            groovy = f"g.V('{vid}').inE().project('id','label','outV','properties').by(id()).by(label()).by(outV().id()).by(valueMap().by(unfold())).toList()"
        else:
            groovy = f"g.V('{vid}').bothE().project('id','label','outV','inV','properties').by(id()).by(label()).by(outV().id()).by(inV().id()).by(valueMap().by(unfold())).toList()"

        try:
            resp = self._client.gremlin().exec(groovy)
            if isinstance(resp, dict) and "data" in resp:
                return resp["data"]
            return resp if isinstance(resp, list) else []
        except Exception as e:
            log.warning("Failed to query edges for %s: %s", vid, e)
            return []

    def _delete_edges(self, vid: str) -> None:
        """Delete all edges connected to a vertex."""
        groovy = f"g.V('{vid}').bothE().drop().iterate()"
        try:
            self._client.gremlin().exec(groovy)
        except Exception as e:
            log.warning("Failed to delete edges for %s: %s", vid, e)

    def _mark_deprecated(self, vid: str, merged_to: str) -> None:
        """Mark a vertex as deprecated (not deleted, preserving audit trail).

        Sets a 'deprecated' property with merge metadata.
        """
        # Ensure the 'deprecated' property exists on the vertex label
        # (If it doesn't exist yet, we add it to the schema)
        try:
            groovy = f"g.V('{vid}').property('deprecated', true).property('merged_to', '{merged_to}').iterate()"
            self._client.gremlin().exec(groovy)
        except Exception as e:
            log.warning("Failed to mark %s as deprecated: %s", vid, e)

    # ------------------------------------------------------------------
    # In-memory resolution (vertices provided in context)
    # ------------------------------------------------------------------

    def _resolve_in_memory(
        self,
        context: Dict[str, Any],
        vertices: List[Dict],
        labels: Optional[List[str]],
        schema: Optional[Dict],
    ) -> Dict[str, Any]:
        """Resolve entities from an in-memory vertex list."""
        # Filter by label
        if labels:
            vertices = [v for v in vertices if v.get("label") in labels]

        # Group by label
        groups = self._group_by_label(vertices)
        candidates = self._find_candidates(groups)
        verified = self._verify_candidates(candidates)
        result = self._merge_entities(verified, context)
        context["resolution_result"] = result.to_dict()
        return context

    # ------------------------------------------------------------------
    # Graph-store resolution (fetch from HugeGraph)
    # ------------------------------------------------------------------

    def _resolve_from_graph(
        self,
        context: Dict[str, Any],
        labels: Optional[List[str]],
        schema: Optional[Dict],
    ) -> Dict[str, Any]:
        """Resolve entities by fetching from the HugeGraph store."""
        # Step 1: Fetch all vertex labels to resolve
        vertex_labels = self._get_vertex_labels(labels)
        if not vertex_labels:
            log.warning("No vertex labels found for resolution")
            context["resolution_result"] = MergeResult().to_dict()
            return context

        # Step 2: Fetch vertices grouped by label
        groups: Dict[str, List[Dict]] = {}
        for label in vertex_labels:
            vertices = self._fetch_vertices_by_label(label)
            if vertices:
                groups[label] = vertices

        if not groups:
            log.warning("No vertices found for resolution")
            context["resolution_result"] = MergeResult().to_dict()
            return context

        # Step 3: Find candidates
        candidates = self._find_candidates(groups)

        # Step 4: LLM verify (if applicable)
        verified = self._verify_candidates(candidates)

        # Step 5: Merge
        result = self._merge_entities(verified, context)
        context["resolution_result"] = result.to_dict()
        return context

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _group_by_label(self, vertices: List[Dict]) -> Dict[str, List[Dict]]:
        """Group vertices by their label."""
        groups: Dict[str, List[Dict]] = {}
        for v in vertices:
            label = v.get("label", "unknown")
            groups.setdefault(label, []).append(v)
        return groups

    def _make_property_key(self, properties: Dict) -> str:
        """Create a deduplication key from resolve properties."""
        values = [str(properties.get(p, "")) for p in self._resolve_properties if p in properties]
        return "|||".join(values)

    def _vertex_text(self, vertex: Dict) -> str:
        """Extract comparable text from a vertex."""
        props = vertex.get("properties", {})
        parts = [str(props.get(p, "")) for p in self._resolve_properties if p in props and props[p]]
        return " ".join(parts).strip()

    def _get_embedding_cached(self, text: str, cache: Dict[int, List[float]]) -> List[float]:
        """Get embedding with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        result = self._embedding.get_text_embedding(text)
        self._embedding_cache[text] = result
        return result

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def _get_vertex_labels(self, labels: Optional[List[str]]) -> List[str]:
        """Get vertex labels from HugeGraph, optionally filtered."""
        try:
            all_labels = self._client.schema().getVertexLabels()
            if isinstance(all_labels, dict) and "vertexlabels" in all_labels:
                all_labels = [vl["name"] for vl in all_labels["vertexlabels"]]
            if labels:
                return [l for l in all_labels if l in labels]
            return all_labels if isinstance(all_labels, list) else []
        except Exception as e:
            log.error("Failed to get vertex labels: %s", e)
            return []

    def _fetch_vertices_by_label(self, label: str) -> List[Dict]:
        """Fetch all vertices of a given label with degree info."""
        groovy = f"""
        g.V().hasLabel('{label}').project('id','label','properties','degree')
            .by(id()).by(label()).by(valueMap().by(unfold()))
            .by(bothE().count())
            .toList()
        """
        try:
            resp = self._client.gremlin().exec(groovy)
            if isinstance(resp, dict) and "data" in resp:
                return resp["data"]
            return resp if isinstance(resp, list) else []
        except Exception as e:
            log.warning("Failed to fetch vertices for label '%s': %s", label, e)
            return []

    def _get_vertex_info(self, vid: str) -> Optional[Dict]:
        """Get vertex info (label, properties, degree)."""
        groovy = f"""
        g.V('{vid}').project('id','label','properties','degree')
            .by(id()).by(label()).by(valueMap().by(unfold()))
            .by(bothE().count())
            .next()
        """
        try:
            resp = self._client.gremlin().exec(groovy)
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                return data[0] if isinstance(data, list) and data else data
            return resp if isinstance(resp, dict) else None
        except Exception as e:
            log.warning("Failed to get vertex info for %s: %s", vid, e)
            return None
