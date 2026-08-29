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

    from_vid: str  # vertex to be deprecated
    from_label: str  # label of from_vid
    from_properties: Dict  # properties of from_vid (for display)
    to_vid: str  # vertex to keep (higher degree preferred)
    to_label: str
    to_properties: Dict
    strategy: str  # which strategy produced this candidate
    confidence: float = 0.0  # 0.0 - 1.0


@dataclass
class MergeResult:
    """Output of the entity resolution process."""

    merged_pairs: List[Dict] = field(default_factory=list)
    merged_count: int = 0
    deprecated_vids: List[str] = field(default_factory=list)
    edges_migrated: int = 0
    synonym_edges: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_pairs": self.merged_pairs,
            "merged_count": self.merged_count,
            "deprecated_vids": self.deprecated_vids,
            "edges_migrated": self.edges_migrated,
            "synonym_edges": self.synonym_edges,
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

    Resolution behaviour is controlled by ``merge_mode``: ``"merge"`` (default)
    physically merges duplicate vertices (migrate edges + soft-deprecate);
    ``"synonym_edge"`` keeps both surface-form vertices and links them with a
    synonymy edge (HippoRAG-style), so downstream 2-hop / PPR retrieval can bridge
    near-duplicates without losing provenance.

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
        merge_mode: str = "merge",
        synonym_edge_label: str = "SYNONYM_OF",
        blocking_key: Optional[Any] = None,
        ann_topk: int = 20,
        ann_retriever: Optional[Any] = None,
        llm_fail_open: bool = False,
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
            merge_mode: "merge" (default) physically merges vertices + migrates
                edges; "synonym_edge" keeps both vertices and writes a
                ``synonym_edge_label`` edge between them (HippoRAG-style: preserves
                surface forms + provenance, lets 2-hop / PPR retrieval bridge them).
            synonym_edge_label: Edge label used in "synonym_edge" mode. It must
                already exist in the graph schema before resolution runs.
            blocking_key: Optional callable(vertex)->str partitioning a label group
                into blocks before pairwise / ANN comparison, cutting the O(n^2)
                candidate space. Default: first whitespace token of the resolved
                text, lowercased.
            ann_topk: Neighbours retrieved per vertex when ``ann_retriever`` is set.
            ann_retriever: Optional callable(texts, k) -> List[List[int]] returning,
                for each query text, the indices (into the input list) of its
                approximate nearest neighbours within the same block. When provided,
                replaces the O(n^2) similarity scan with ANN candidate generation
                (e.g. Milvus / OceanBase vector index, faiss, HNSW). If None, falls
                back to blocking + exact pairwise comparison bounded by block size.
            llm_fail_open: If True, keep the legacy fail-open behaviour (LLM error
                or unparseable response -> accept all candidates). Default False
                (fail-closed: reject candidates that were only embedding-confirmed,
                avoiding over-merging when the LLM is unavailable).
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {self.VALID_STRATEGIES}")
        if merge_mode not in ("merge", "synonym_edge"):
            raise ValueError(f"Invalid merge_mode '{merge_mode}'. Must be 'merge' or 'synonym_edge'")

        self._client = client
        self._llm = llm
        self._embedding = embedding
        self._strategy = strategy
        self._threshold = threshold
        self._batch_size = batch_size
        self._resolve_properties = resolve_properties or ["name"]
        self._vertex_labels = vertex_labels
        self._max_pairs_per_label = max_pairs_per_label
        self._merge_mode = merge_mode
        self._synonym_edge_label = synonym_edge_label
        self._blocking_key = blocking_key or self._default_blocking_key
        self._ann_topk = ann_topk
        self._ann_retriever = ann_retriever
        self._llm_fail_open = llm_fail_open

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
                candidates.append(
                    MergeCandidate(
                        from_vid=dup["id"],
                        from_label=dup["label"],
                        from_properties=dup.get("properties", {}),
                        to_vid=keep["id"],
                        to_label=keep["label"],
                        to_properties=keep.get("properties", {}),
                        strategy=self.STRATEGY_EXACT,
                        confidence=1.0,
                    )
                )

        log.info("Exact match: %d candidates", len(candidates))
        return candidates

    def _embedding_candidates(self, group: List[Dict]) -> List[MergeCandidate]:
        """Find candidates via embedding cosine similarity.

        Scalability: instead of an O(n^2) all-pairs comparison within a label
        group, vertices are first partitioned into **blocks** via
        ``_blocking_key`` (cheap, high-recall). Candidate pairs are then generated
        inside each block either by an approximate-nearest-neighbour retriever
        (``ann_retriever``) or by exact pairwise comparison bounded by block size.
        This removes the old 5000-pair sampling truncation that silently dropped
        true duplicates on large labels.
        """
        candidates: List[MergeCandidate] = []
        if len(group) < 2:
            return candidates

        blocks = self._blocking_groups(group)
        log.info("Embedding: %d vertices split into %d blocks", len(group), len(blocks))

        pairs_to_merge: List[Tuple[str, str]] = []
        for block in blocks:
            if len(block) < 2:
                continue
            for i, j in self._candidate_pairs_in_block(block):
                text_i = self._vertex_text(block[i])
                text_j = self._vertex_text(block[j])
                if not text_i or not text_j:
                    continue

                emb_i = self._get_embedding_cached(text_i)
                emb_j = self._get_embedding_cached(text_j)
                if self._cosine_similarity(emb_i, emb_j) >= self._threshold:
                    pairs_to_merge.append((block[i]["id"], block[j]["id"]))

        if not pairs_to_merge:
            return candidates

        # Consolidate transitive matches via Union-Find over vertex ids.
        uf = UnionFind()
        for a, b in pairs_to_merge:
            uf.union(a, b)

        id_to_vertex = {v["id"]: v for v in group}
        for merge_set in uf.groups():
            vids = list(merge_set)
            # Keep the vertex with highest degree
            sorted_by_degree = sorted(vids, key=lambda vid: id_to_vertex[vid].get("degree", 0), reverse=True)
            keep_vid = sorted_by_degree[0]
            for dup_vid in sorted_by_degree[1:]:
                dup = id_to_vertex[dup_vid]
                keep = id_to_vertex[keep_vid]
                candidates.append(
                    MergeCandidate(
                        from_vid=dup_vid,
                        from_label=dup.get("label", ""),
                        from_properties=dup.get("properties", {}),
                        to_vid=keep_vid,
                        to_label=keep.get("label", ""),
                        to_properties=keep.get("properties", {}),
                        strategy=self.STRATEGY_EMBEDDING,
                        confidence=self._threshold,
                    )
                )

        log.info("Embedding: %d candidates from %d pairs_to_merge", len(candidates), len(pairs_to_merge))
        return candidates

    def _blocking_groups(self, group: List[Dict]) -> List[List[Dict]]:
        """Partition a label group into blocks via the blocking key.

        Blocking reduces the O(n^2) comparison space to the sum of block-size
        squares. The default key (first token of the resolved text) is a classic
        high-recall blocking scheme; callers can pass a stronger key (entity type,
        first character, phonetic code) via ``blocking_key``.
        """
        blocks: Dict[str, List[Dict]] = {}
        for v in group:
            key = self._blocking_key(v)
            blocks.setdefault(key, []).append(v)
        return list(blocks.values())

    def _candidate_pairs_in_block(self, block: List[Dict]) -> List[Tuple[int, int]]:
        """Generate candidate index pairs within a block.

        Uses ``ann_retriever`` (if provided) for ANN top-k, otherwise exact
        pairwise comparison. Block size bounds the cost either way; the old
        global 5000-pair sampling cap no longer applies.
        """
        n = len(block)
        if n < 2:
            return []

        if self._ann_retriever is not None:
            texts = [self._vertex_text(v) for v in block]
            try:
                nn_indices = self._ann_retriever(texts, self._ann_topk)
                pairs: List[Tuple[int, int]] = []
                for qi, neighbours in enumerate(nn_indices):
                    if qi >= n:
                        # Defensive: some ANN retrievers return more neighbour
                        # lists than query texts; ignore the overflow.
                        continue
                    for ni in neighbours:
                        if ni != qi and 0 <= ni < n:
                            a, b = (qi, ni) if qi < ni else (ni, qi)
                            pairs.append((a, b))
                return pairs
            except Exception as e:
                log.warning("ANN retriever failed (%s); falling back to pairwise", e)

        # Exact pairwise within block (bounded by block size).
        max_pairs = self._max_pairs_per_label
        all_pairs = list(combinations(range(n), 2))
        if len(all_pairs) > max_pairs:
            log.warning("Block has %d pairs (limit %d); sampling", len(all_pairs), max_pairs)
            all_pairs.sort(key=lambda p: abs(len(str(block[p[0]])) - len(str(block[p[1]]))))
            all_pairs = all_pairs[:max_pairs]
        return all_pairs

    def _default_blocking_key(self, vertex: Dict) -> str:
        """Default blocking key: first whitespace token of the resolved text."""
        parts = [
            str(vertex.get("properties", {}).get(p, ""))
            for p in self._resolve_properties
            if p in vertex.get("properties", {}) and vertex["properties"][p]
        ]
        text = " ".join(parts).strip()
        if not text:
            return "__none__"
        return text.split()[0].lower()

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
        """Send a batch of candidates to LLM for verification.

        Fail-closed by default: on LLM error the batch is rejected (returns []),
        so an unavailable LLM cannot silently over-merge. Set ``llm_fail_open``
        (constructor) to restore the legacy accept-all behaviour.
        """
        prompt = self._build_verify_prompt(batch)
        try:
            response = self._llm.generate(prompt=prompt)
            return self._parse_verify_response(response, batch)
        except Exception as e:
            if self._llm_fail_open:
                log.error("LLM verification failed: %s; fail-open accepting batch", e)
                return batch
            log.error("LLM verification failed: %s; fail-closed rejecting batch", e)
            return []

    def _build_verify_prompt(self, batch: List[MergeCandidate]) -> str:
        """Build a batch verification prompt for LLM."""
        items = []
        for i, c in enumerate(batch):
            props_a = ", ".join(f"{k}={v}" for k, v in c.from_properties.items())
            props_b = ", ".join(f"{k}={v}" for k, v in c.to_properties.items())
            items.append(f"Pair {i + 1}:\n  Entity A ({c.from_label}): {props_a}\n  Entity B ({c.to_label}): {props_b}")

        return f"""Determine whether each pair of entities refers to the same real-world entity.
Two entities are the SAME if they represent the same person, organization, location, concept, etc.
even if their names differ slightly (e.g., "US" vs "United States", "Bob" vs "Robert Smith").

Entity Pairs:
{chr(10).join(items)}

Respond with ONLY a JSON array of booleans, one per pair, in order.
Example: [true, false, true]
If unsure, respond with false to avoid over-merging."""

    def _parse_verify_response(self, response: str, batch: List[MergeCandidate]) -> List[MergeCandidate]:
        """Parse LLM verification response and filter candidates.

        Fail-closed by default: an unparseable or malformed response rejects the
        batch instead of accepting it. Set ``llm_fail_open`` to accept instead.
        """
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
                return batch if self._llm_fail_open else []

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
            log.warning("Failed to parse LLM response: %s", e)
            return batch if self._llm_fail_open else []

    # ------------------------------------------------------------------
    # Phase 3: Merge Execution
    # ------------------------------------------------------------------

    def _merge_entities(
        self,
        candidates: List[MergeCandidate],
        context: Dict[str, Any],
    ) -> MergeResult:
        """Phase 3: execute resolution operations.

        When ``merge_mode == "merge"`` (default): for each candidate, migrate
        edges from from_vid to to_vid and mark from_vid deprecated.
        When ``merge_mode == "synonym_edge"``: keep both vertices and write a
        ``synonym_edge_label`` edge between them (no migration, no deprecation).
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

            # Pick representative: highest degree. Both endpoints are guaranteed to
            # be present via `_get_vertex_info` or the candidate_info fallback built
            # above, so a group always yields >= 2 resolved vertices here.
            sorted_vids = sorted(
                vertices_info.keys(),
                key=lambda v: vertices_info[v].get("degree", 0),
                reverse=True,
            )
            keep_vid = sorted_vids[0]
            keep_label = vertices_info[keep_vid].get("label", "")

            for dup_vid in sorted_vids[1:]:
                dup_label = vertices_info[dup_vid].get("label", "")

                try:
                    if self._merge_mode == "synonym_edge":
                        ok = self._add_synonym_edge(dup_vid, keep_vid, keep_label, dup_label)
                        if not ok:
                            # Non-fatal: keep resolution going, but do not count a
                            # synonym edge that was never written.
                            error_msg = (
                                f"Failed to write synonym edge {dup_vid} -> {keep_vid}"
                            )
                            log.error(error_msg)
                            result.errors.append(error_msg)
                            continue
                        result.merged_pairs.append(
                            {
                                "from_vid": dup_vid,
                                "to_vid": keep_vid,
                                "from_label": dup_label,
                                "to_label": keep_label,
                                "mode": "synonym_edge",
                                "edges_migrated": 0,
                            }
                        )
                        result.merged_count += 1
                        result.synonym_edges += 1
                        log.info(
                            "Synonym edge %s (%s) -[%s]-> %s (%s)",
                            dup_vid, dup_label, self._synonym_edge_label, keep_vid, keep_label,
                        )
                        continue

                    edges_migrated = self._migrate_edges(dup_vid, keep_vid)
                    self._mark_deprecated(dup_vid, merged_to=keep_vid)

                    result.merged_pairs.append(
                        {
                            "from_vid": dup_vid,
                            "to_vid": keep_vid,
                            "from_label": dup_label,
                            "to_label": keep_label,
                            "edges_migrated": edges_migrated,
                        }
                    )
                    result.merged_count += 1
                    result.deprecated_vids.append(dup_vid)
                    result.edges_migrated += edges_migrated

                    log.info(
                        "Merged %s (%s) -> %s (%s), migrated %d edges",
                        dup_vid,
                        dup_label,
                        keep_vid,
                        keep_label,
                        edges_migrated,
                    )
                except Exception as e:
                    error_msg = f"Failed to resolve {dup_vid} -> {keep_vid}: {e}"
                    log.error(error_msg)
                    result.errors.append(error_msg)

        return result

    def _add_synonym_edge(self, from_vid: str, to_vid: str, to_label: str, from_label: str) -> bool:
        """Write a synonymy edge instead of merging (HippoRAG-style).

        Keeps both surface-form vertices and connects them with
        ``self._synonym_edge_label`` so downstream 2-hop / PPR retrieval can bridge
        near-duplicate entities without losing provenance.

        The ``self._synonym_edge_label`` edge label must already exist in the
        graph schema (with source/target labels covering the resolved vertex
        labels, or as a permissive super-edge). A duplicate edge add is tolerated
        (logged, non-fatal) — it does not mutate the graph schema.

        Returns ``True`` if the edge was written (or already tolerated), ``False``
        if the write raised (so the caller can record the failure instead of
        counting a non-existent synonym edge).
        """
        try:
            self._client.graph().addEdge(
                self._synonym_edge_label,
                from_vid,
                to_vid,
                {},
            )
            return True
        except Exception as e:
            log.warning("Failed to add synonym edge %s -> %s: %s", from_vid, to_vid, e)
            return False

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
            groovy = f"g.V({self._g_id(vid)}).outE().project('id','label','inV','properties').by(id()).by(label()).by(inV().id()).by(valueMap().by(unfold())).toList()"
        elif direction == "in":
            groovy = f"g.V({self._g_id(vid)}).inE().project('id','label','outV','properties').by(id()).by(label()).by(outV().id()).by(valueMap().by(unfold())).toList()"
        else:
            groovy = f"g.V({self._g_id(vid)}).bothE().project('id','label','outV','inV','properties').by(id()).by(label()).by(outV().id()).by(inV().id()).by(valueMap().by(unfold())).toList()"

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
        groovy = f"g.V({self._g_id(vid)}).bothE().drop().iterate()"
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
            groovy = f"g.V({self._g_id(vid)}).property('deprecated', true).property('merged_to', '{merged_to}').iterate()"
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
    # Pure in-memory resolution (pre-commit, no graph access)
    # ------------------------------------------------------------------

    def resolve_in_memory_pure(
        self,
        vertices: List[Dict],
        edges: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Resolve duplicate vertices entirely in memory (pre-commit).

        Unlike :meth:`_resolve_in_memory`, this never touches the graph store:
        it builds a ``deprecated_vid -> canonical_vid`` mapping from the merge
        candidates, then rewrites the vertex list (dropping deprecated vertices
        and merging their properties into the keeper) and the edge list
        (repointing ``outV``/``inV`` endpoints and dropping self-loops created
        by a merge).

        This is the "resolve before commit" step for ingestion pipelines: call
        it on freshly extracted vertices/edges, then commit the deduplicated
        result to HugeGraph.

        Args:
            vertices: Extracted vertex dicts (``id``, ``label``, ``properties``).
            edges: Extracted edge dicts (``outV``, ``inV``); optional.

        Returns:
            Dict with ``vertices``, ``edges`` and ``resolution_result``
            (a :class:`MergeResult` dict).
        """
        edges = list(edges or [])
        vertices = list(vertices or [])
        result = MergeResult()

        if not vertices:
            return {
                "vertices": vertices,
                "edges": edges,
                "resolution_result": result.to_dict(),
            }

        # Pre-commit vertices carry no degree; derive it from edge endpoints so
        # "keep the highest-degree vertex" works without a graph query.
        degree: Dict[str, int] = {}
        for edge in edges:
            for key in ("outV", "inV"):
                vid = edge.get(key)
                if vid:
                    degree[vid] = degree.get(vid, 0) + 1
        vertex_by_id: Dict[str, Dict] = {}
        for vertex in vertices:
            vertex.setdefault("degree", 0)
            vertex["degree"] = degree.get(vertex.get("id"), 0)
            vertex_by_id[vertex.get("id")] = vertex

        candidates = self._find_candidates(self._group_by_label(vertices))
        verified = self._verify_candidates(candidates)

        # Union-Find over candidate pairs → canonical mapping per merge group.
        uf = UnionFind()
        for candidate in verified:
            uf.union(candidate.to_vid, candidate.from_vid)

        vid_map: Dict[str, str] = {}
        for group in uf.groups():
            # Keep the highest-degree vertex (tie-break on shorter id for determinism).
            keep_vid = max(
                group,
                key=lambda vid: (vertex_by_id.get(vid, {}).get("degree", 0), -len(str(vid))),
            )
            for dup_vid in group:
                if dup_vid == keep_vid:
                    continue
                vid_map[dup_vid] = keep_vid
                result.merged_pairs.append(
                    {
                        "from_vid": dup_vid,
                        "to_vid": keep_vid,
                        "from_label": vertex_by_id.get(dup_vid, {}).get("label", ""),
                        "to_label": vertex_by_id.get(keep_vid, {}).get("label", ""),
                        "edges_migrated": 0,
                    }
                )
                result.deprecated_vids.append(dup_vid)
        result.merged_count = len(result.deprecated_vids)

        # Merge properties: keeper wins on conflict, deprecated fills gaps.
        for dup_vid, keep_vid in vid_map.items():
            keep = vertex_by_id.get(keep_vid)
            dup = vertex_by_id.get(dup_vid)
            if keep is not None and dup is not None:
                merged = dict(dup.get("properties", {}))
                merged.update(keep.get("properties", {}))
                keep["properties"] = merged

        kept_vertices = [v for v in vertices if v.get("id") not in vid_map]

        # Rewrite edges: repoint endpoints, drop self-loops from merges.
        kept_edges: List[Dict] = []
        for edge in edges:
            new_edge = dict(edge)
            repointed = False
            for key in ("outV", "inV"):
                vid = new_edge.get(key)
                if vid in vid_map:
                    new_edge[key] = vid_map[vid]
                    repointed = True
            if new_edge.get("outV") == new_edge.get("inV"):
                continue
            if repointed:
                result.edges_migrated += 1
            kept_edges.append(new_edge)

        return {
            "vertices": kept_vertices,
            "edges": kept_edges,
            "resolution_result": result.to_dict(),
        }

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

    @staticmethod
    def _g_id(vid: Any) -> str:
        """Return a Gremlin-safe vertex-id literal.

        HugeGraph stores numeric (AUTOMATIC / LONG) ids and requires them
        *unquoted* in Gremlin (``g.V(123)``); string ids must be quoted. The
        previous implementation always quoted the id, which made every
        graph-store query 500 on numeric ids — the common case for the
        default AUTOMATIC id strategy.
        """
        if isinstance(vid, bool):
            pass  # fall through to string quoting below
        elif isinstance(vid, int):
            return str(vid)
        elif isinstance(vid, str) and vid.lstrip("-").isdigit():
            return vid
        # string id -> quote, escaping any embedded single quotes
        return "'%s'" % str(vid).replace("'", "\\'")

    def _get_embedding_cached(self, text: str) -> List[float]:
        """Get embedding with caching (uses ``self._embedding_cache``)."""
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
        """Get vertex labels from HugeGraph, optionally filtered.

        Handles the varying return shapes across ``hugegraph-python-client``
        versions: a ``{"vertexlabels": [...]}`` dict, a plain list of name
        strings, a list of ``{"name": ...}`` dicts, or a list of
        ``VertexLabel`` objects (exposing a ``.name`` attribute).
        """
        try:
            raw = self._client.schema().getVertexLabels()
            names: List[str] = []
            items = (
                raw.get("vertexlabels")
                if isinstance(raw, dict) and "vertexlabels" in raw
                else raw
            )
            for vl in items or []:
                if isinstance(vl, str):
                    names.append(vl)
                elif isinstance(vl, dict):
                    names.append(vl.get("name"))
                else:
                    names.append(getattr(vl, "name", None))
            all_labels = [n for n in names if n]
            if labels:
                return [label for label in all_labels if label in labels]
            return all_labels
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
        g.V({self._g_id(vid)}).project('id','label','properties','degree')
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
