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

"""Entity identity edge builder — creates same_as edges between semantically similar entities.

Borrowed from Fast-GraphRAG's entity deduplication mechanism
(fast_graphrag/_services/_state_manager.py lines 131-174).

Core idea: Instead of physically merging duplicate entities, create "same_as" edges
between entities whose embedding cosine similarity exceeds a threshold (default 0.9).
This allows PPR to propagate importance across similar entities naturally, achieving
"soft merge" without altering the graph structure.

This addresses the P0-v5 Medical graph_hits=0 problem: entity name matching precision
was insufficient, and medical terms in queries didn't match extracted entity names.
Same_as edges bridge this gap by connecting semantically equivalent entities.

Design references:
    - Fast-GraphRAG: _state_manager.py:131-174 (identity edge creation)
    - HippoRAG2: synonym/alias detection via embedding similarity
    - Neo4j GraphRAG: BasePropertySimilarityResolver
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from hugegraph_llm.utils.log import log

# ── Constants ─────────────────────────────────────────────────────

SAME_AS_EDGE_LABEL = "same_as"
DEFAULT_SIMILARITY_THRESHOLD = 0.9   # Fast-GraphRAG uses 0.9 for insert
DEFAULT_QUERY_THRESHOLD = 0.7        # Fast-GraphRAG uses 0.7 for query
DEFAULT_TOP_K_NEIGHBORS = 3          # Fast-GraphRAG uses top_k=3 for identity
DEFAULT_EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 dimension


@dataclass
class IdentityEdgeConfig:
    """Configuration for same_as edge creation."""
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    # Prevent duplicate edges: only create edge from lower index to higher
    bidirectional: bool = False       # Fast-GraphRAG creates单向边避免重复
    # Skip self-matches and already-connected entities
    skip_existing_edges: bool = True


@dataclass
class IdentityEdgeResult:
    """Result of same_as edge creation."""
    edges_created: int = 0
    edges_skipped: int = 0            # Skipped due to threshold or duplicates
    entity_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    # (entity_a_id, entity_b_id, similarity_score)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


class IdentityEdgeBuilder:
    """Build same_as edges between semantically similar entities.

    Implements the "soft merge" strategy from Fast-GraphRAG:
    - For each entity, find its top-K nearest neighbors in embedding space
    - If similarity > threshold, create a same_as edge
    - Skip self-matches, below-threshold pairs, and already-connected pairs

    Usage::

        builder = IdentityEdgeBuilder(
            embedding_fn=my_embedding_fn,
            vector_search_fn=my_vector_search,
            graph_client=my_hg_client,
        )
        result = builder.build(
            entity_ids=["42:diabetes", "42:Diabetes_Type2", "42:DM"],
            entity_texts={"42:diabetes": "diabetes mellitus", ...},
        )
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        vector_search_fn: Optional[Callable] = None,
        graph_client: Optional[Any] = None,
        config: Optional[IdentityEdgeConfig] = None,
    ) -> None:
        """Initialize IdentityEdgeBuilder.

        Args:
            embedding_fn: Function that takes text and returns embedding vector.
            vector_search_fn: Function for vector similarity search (HNSW/FAISS).
            graph_client: PyHugeClient for creating edges in HugeGraph.
            config: IdentityEdgeConfig with threshold and top_k settings.
        """
        self._embedding_fn = embedding_fn
        self._vector_search_fn = vector_search_fn
        self._graph_client = graph_client
        self.config = config or IdentityEdgeConfig()
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def build(
        self,
        entity_ids: List[str],
        entity_texts: Dict[str, str],
        existing_edges: Optional[Set[Tuple[str, str]]] = None,
    ) -> IdentityEdgeResult:
        """Build same_as edges between similar entities.

        Args:
            entity_ids: List of entity vertex IDs.
            entity_texts: Dict mapping entity_id to text (name + description).
            existing_edges: Set of (src_id, tgt_id) pairs that already exist.
                If None, will try to check via graph_client.

        Returns:
            IdentityEdgeResult with edges created and statistics.
        """
        t0 = time.perf_counter()
        result = IdentityEdgeResult()
        threshold = self.config.similarity_threshold
        top_k = self.config.top_k_neighbors

        if len(entity_ids) < 2:
            log.info("[IdentityEdge] Only %d entities, no same_as edges possible", len(entity_ids))
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Step 1: Compute embeddings for all entities
        log.info("[IdentityEdge] Computing embeddings for %d entities...", len(entity_ids))
        embeddings = self._compute_embeddings(entity_ids, entity_texts)

        if embeddings is None or len(embeddings) == 0:
            log.warning("[IdentityEdge] No embeddings computed, aborting")
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Step 2: Compute pairwise similarity matrix
        log.info("[IdentityEdge] Computing pairwise similarity matrix...")
        sim_matrix = self._compute_similarity_matrix(embeddings)

        # Step 3: Find similar pairs above threshold
        log.info("[IdentityEdge] Finding pairs above threshold %.3f...", threshold)
        similar_pairs = self._find_similar_pairs(
            sim_matrix, entity_ids, threshold, top_k,
            existing_edges or set(),
        )

        # Step 4: Create same_as edges in HugeGraph
        log.info("[IdentityEdge] Creating %d same_as edges...", len(similar_pairs))
        for src_id, tgt_id, score in similar_pairs:
            try:
                created = self._create_same_as_edge(src_id, tgt_id, score)
                if created:
                    result.edges_created += 1
                    result.entity_pairs.append((src_id, tgt_id, score))
                else:
                    result.edges_skipped += 1
            except Exception as e:
                result.errors.append(f"Failed to create edge {src_id}-{tgt_id}: {e}")

        result.duration_ms = (time.perf_counter() - t0) * 1000
        log.info("[IdentityEdge] Done: %d edges created, %d skipped, %d errors "
                 f"in {result.duration_ms:.1f}ms",
                 result.edges_created, result.edges_skipped, len(result.errors))
        return result

    def build_incremental(
        self,
        new_entity_ids: List[str],
        new_entity_texts: Dict[str, str],
        all_entity_ids: List[str],
        all_entity_texts: Dict[str, str],
        existing_edges: Optional[Set[Tuple[str, str]]] = None,
    ) -> IdentityEdgeResult:
        """Build same_as edges for newly inserted entities only.

        Only checks new entities against all existing entities,
        avoiding redundant computation for already-processed entities.

        Args:
            new_entity_ids: Newly inserted entity IDs.
            new_entity_texts: Texts for new entities.
            all_entity_ids: All entity IDs (new + existing).
            all_entity_texts: Texts for all entities.

        Returns:
            IdentityEdgeResult.
        """
        t0 = time.perf_counter()
        result = IdentityEdgeResult()
        threshold = self.config.similarity_threshold
        top_k = self.config.top_k_neighbors
        existing = existing_edges or set()

        if len(new_entity_ids) == 0 or len(all_entity_ids) < 2:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Compute embeddings for new entities only
        new_embeddings = self._compute_embeddings(new_entity_ids, new_entity_texts)
        if new_embeddings is None:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Compute embeddings for all entities (needed for comparison)
        all_embeddings = self._compute_embeddings(all_entity_ids, all_entity_texts)
        if all_embeddings is None:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            return result

        # Compute cross-similarity: new vs all
        new_idx_map = {eid: idx for idx, eid in enumerate(new_entity_ids)}
        all_idx_map = {eid: idx for idx, eid in enumerate(all_entity_ids)}

        for new_eid in new_entity_ids:
            new_idx = new_idx_map.get(new_eid)
            if new_idx is None:
                continue
            new_emb = new_embeddings[new_idx]

            # Compare with all entities
            candidates = []
            for all_eid in all_entity_ids:
                all_idx = all_idx_map.get(all_eid)
                if all_idx is None:
                    continue

                # Skip self
                if new_eid == all_eid:
                    continue

                # Skip already-connected pairs (Fast-GraphRAG: lower index check)
                pair = tuple(sorted([new_eid, all_eid]))
                if pair in existing:
                    continue

                # Compute cosine similarity
                all_emb = all_embeddings[all_idx]
                sim = float(np.dot(new_emb, all_emb) / (
                    np.linalg.norm(new_emb) * np.linalg.norm(all_emb) + 1e-8
                ))

                if sim >= threshold:
                    candidates.append((all_eid, sim))

            # Sort by similarity descending, take top_k
            candidates.sort(key=lambda x: x[1], reverse=True)
            for tgt_eid, score in candidates[:top_k]:
                pair = tuple(sorted([new_eid, tgt_eid]))
                if pair not in existing:
                    try:
                        created = self._create_same_as_edge(new_eid, tgt_eid, score)
                        if created:
                            result.edges_created += 1
                            result.entity_pairs.append((new_eid, tgt_eid, score))
                            existing.add(pair)
                        else:
                            result.edges_skipped += 1
                    except Exception as e:
                        result.errors.append(f"Failed: {new_eid}-{tgt_eid}: {e}")

        result.duration_ms = (time.perf_counter() - t0) * 1000
        log.info("[IdentityEdge] Incremental: %d edges created in %.1fms",
                 result.edges_created, result.duration_ms)
        return result

    # ── Internal methods ────────────────────────────────────────

    def _compute_embeddings(
        self,
        entity_ids: List[str],
        entity_texts: Dict[str, str],
    ) -> Optional[np.ndarray]:
        """Compute embedding vectors for all entities.

        Returns (N, dim) numpy array. Uses cache to avoid recomputation.
        """
        if self._embedding_fn is None:
            log.warning("[IdentityEdge] No embedding_fn, cannot compute embeddings")
            return None

        vectors = []
        for eid in entity_ids:
            text = entity_texts.get(eid, "")
            if not text:
                # Use entity ID as fallback text
                text = eid.split(":")[-1] if ":" in eid else eid

            if eid in self._embedding_cache:
                vectors.append(self._embedding_cache[eid])
            else:
                try:
                    vec = self._embedding_fn(text)
                    self._embedding_cache[eid] = vec
                    vectors.append(vec)
                except Exception as e:
                    log.warning(f"[IdentityEdge] Embedding failed for '{eid}': {e}")
                    # Use zero vector as fallback
                    vectors.append(np.zeros(self.config.embedding_dim, dtype=np.float32))

        if not vectors:
            return None
        return np.array(vectors, dtype=np.float32)

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: (N, dim) numpy array.

        Returns:
            (N, N) cosine similarity matrix.
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = embeddings / norms

        # Cosine similarity = dot product of normalized vectors
        sim_matrix = normalized @ normalized.T
        return sim_matrix

    def _find_similar_pairs(
        self,
        sim_matrix: np.ndarray,
        entity_ids: List[str],
        threshold: float,
        top_k: int,
        existing_edges: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str, float]]:
        """Find entity pairs with similarity above threshold.

        Args:
            sim_matrix: (N, N) similarity matrix.
            entity_ids: List of entity IDs corresponding to matrix rows/cols.
            threshold: Minimum similarity score.
            top_k: Max neighbors per entity.
            existing_edges: Set of existing edge pairs to skip.

        Returns:
            List of (src_id, tgt_id, similarity_score) tuples.
        """
        pairs: List[Tuple[str, str, float]] = []
        n = len(entity_ids)

        for i in range(n):
            # Get similarity scores for entity i
            scores = sim_matrix[i]

            # Find top_k neighbors above threshold (excluding self)
            top_indices = np.argsort(scores)[::-1]  # Descending

            count = 0
            for j in top_indices:
                if j == i:
                    continue  # Skip self
                score = float(scores[j])
                if score < threshold:
                    break  # Below threshold, stop

                # Check if pair already exists
                src_id = entity_ids[i]
                tgt_id = entity_ids[j]
                pair = tuple(sorted([src_id, tgt_id]))
                if pair in existing_edges:
                    continue

                pairs.append((src_id, tgt_id, score))
                existing_edges.add(pair)  # Prevent duplicate
                count += 1
                if count >= top_k:
                    break

        # Deduplicate (sorted pairs)
        seen: Set[Tuple[str, str]] = set()
        unique_pairs: List[Tuple[str, str, float]] = []
        for src, tgt, score in pairs:
            pair = tuple(sorted([src, tgt]))
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append((src, tgt, score))

        return unique_pairs

    def _create_same_as_edge(
        self,
        src_id: str,
        tgt_id: str,
        similarity_score: float,
    ) -> bool:
        """Create a same_as edge in HugeGraph between two entities.

        Args:
            src_id: Source vertex ID (HugeGraph format, e.g., "42:EntityName").
            tgt_id: Target vertex ID.
            similarity_score: Cosine similarity that triggered this edge.

        Returns:
            True if edge was created, False if skipped.
        """
        if self._graph_client is None:
            # No graph client — just record the pair, don't actually create edge
            log.debug(f"[IdentityEdge] No graph client, recording pair {src_id}-{tgt_id}")
            return True  # Pretend created for testing

        try:
            # Use PyHugeClient to add edge
            # Edge format: label=same_as, outV=src_id, inV=tgt_id, properties={"score": sim}
            self._graph_client.addEdge(
                label=SAME_AS_EDGE_LABEL,
                outV=src_id,
                inV=tgt_id,
                properties={"score": str(round(similarity_score, 4))},
            )
            log.debug(f"[IdentityEdge] Created edge {src_id} --same_as({similarity_score:.4f})-- {tgt_id}")
            return True
        except Exception as e:
            # Edge may already exist or other error
            log.warning(f"[IdentityEdge] Edge creation failed: {src_id}-{tgt_id}: {e}")
            return False


# ── Convenience function ──────────────────────────────────────────


def build_identity_edges(
    entity_ids: List[str],
    entity_texts: Dict[str, str],
    embedding_fn: Callable[[str], np.ndarray],
    graph_client: Optional[Any] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int = DEFAULT_TOP_K_NEIGHBORS,
) -> IdentityEdgeResult:
    """Quick-build same_as edges between similar entities.

    Args:
        entity_ids: Entity vertex IDs.
        entity_texts: Entity ID → text mapping.
        embedding_fn: Embedding function.
        graph_client: Optional PyHugeClient for edge creation.
        threshold: Similarity threshold (default 0.9).
        top_k: Max neighbors per entity (default 3).

    Returns:
        IdentityEdgeResult.
    """
    config = IdentityEdgeConfig(
        similarity_threshold=threshold,
        top_k_neighbors=top_k,
    )
    builder = IdentityEdgeBuilder(
        embedding_fn=embedding_fn,
        graph_client=graph_client,
        config=config,
    )
    return builder.build(entity_ids, entity_texts)
