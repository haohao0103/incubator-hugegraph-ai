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

"""Cascade propagation: Entity → Relation → Chunk three-layer score spreading.

Borrowed from Fast-GraphRAG's sparse matrix chain propagation
(fast_graphrag/_services/_state_manager.py lines 291-309).

Core algorithm:
    1. PPR propagates importance scores among Entity nodes (already in ppr_retriever.py)
    2. Entity scores * e2r_matrix → Relation scores (sparse matrix dot)
    3. Relation scores * r2c_matrix → Chunk scores (sparse matrix dot)

This replaces the previous RRF three-channel parallel fusion (Vector + BM25 + Graph)
with a serial cascade: Vector seed → PPR propagation → Relation scoring → Chunk scoring.
BM25 is retained as an optional enhancement plugin (default disabled).

Design references:
    - Fast-GraphRAG: _state_manager.py:296-309 (e2r/r2c matrix chain)
    - HippoRAG2: personalized_pagerank seed → entity scoring
    - LightRAG: ll_keywords → entities VDB → _get_node_data (entity→relation expansion)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, vstack

from hugegraph_llm.operators.graph_op.ppr_retriever import PPRRetriever
from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion, RRFResults
from hugegraph_llm.utils.log import log

# ── Ranking policies (borrowed from Fast-GraphRAG) ───────────────


@dataclass
class RankingConfig:
    """Configuration for each layer's ranking policy."""
    entity_threshold: float = 0.005   # PPR score threshold for entity filtering
    entity_max_count: int = 128       # Max entities to keep
    relation_top_k: int = 64          # Top-K relations to keep
    chunk_top_k: int = 8              # Top-K chunks to keep


@dataclass
class CascadeConfig:
    """Configuration for the cascade propagation pipeline."""
    # PPR parameters
    ppr_alpha: float = 0.85           # Damping factor (Fast-GraphRAG uses 0.85)
    ppr_epsilon: float = 1e-6         # Convergence threshold
    ppr_max_depth: int = 3            # Subgraph expansion depth for PPR

    # Vector seed parameters
    vector_top_k: int = 20            # Top-K vector search results as PPR seeds
    vector_threshold: float = 0.5     # Minimum similarity for vector seed
    named_entity_top_k: int = 1       # Top-K for named entity (precise match, threshold=0.7)
    named_entity_threshold: float = 0.7

    # Ranking policies
    ranking: RankingConfig = field(default_factory=RankingConfig)

    # BM25 optional enhancement
    bm25_enabled: bool = False        # BM25 as optional plugin (default OFF)
    bm25_top_k: int = 10
    bm25_weight: float = 0.3          # Weight for BM25 channel in optional RRF fusion

    # RRF fusion (for optional BM25 integration)
    rrf_k: int = 60


@dataclass
class CascadeResult:
    """Result of cascade propagation retrieval."""
    entity_scores: Dict[str, float] = field(default_factory=dict)   # entity_id → PPR score
    relation_scores: Dict[str, float] = field(default_factory=dict) # relation_id → propagated score
    chunk_scores: Dict[str, float] = field(default_factory=dict)    # chunk_id → propagated score
    seed_entities: List[str] = field(default_factory=list)          # Initial vector seed entities
    bm25_results: List[str] = field(default_factory=list)           # BM25 results (if enabled)
    stats: Dict[str, Any] = field(default_factory=dict)             # Pipeline statistics


# ── Sparse matrix utilities ──────────────────────────────────────


def csr_from_indices_list(
    data: List[List[int]],
    shape: Tuple[int, int],
) -> csr_matrix:
    """Build a binary CSR matrix from a list of index lists.

    Borrowed from Fast-GraphRAG _utils.py:95-109.

    Args:
        data: List of lists, where data[i] = [col_indices for row i].
        shape: (num_rows, num_cols) matrix dimensions.

    Returns:
        Binary csr_matrix with 1s at (row, col) positions.
    """
    num_rows = len(data)
    row_indices = np.repeat(np.arange(num_rows), [len(row) for row in data])
    col_indices = np.concatenate(data) if num_rows > 0 else np.array([], dtype=np.int64)
    values = np.broadcast_to(1, len(row_indices))
    return csr_matrix((values, (row_indices, col_indices)), shape=shape)


def apply_threshold_ranking(
    scores: csr_matrix,
    threshold: float,
    max_count: int,
) -> csr_matrix:
    """Apply threshold + top-K filtering on sparse score matrix.

    Borrowed from Fast-GraphRAG RankingPolicy_WithThreshold.

    Args:
        scores: (1, N) sparse score matrix.
        threshold: Minimum score to keep.
        max_count: Maximum number of non-zero entries to keep.

    Returns:
        Filtered csr_matrix with scores below threshold zeroed out.
    """
    scores = scores.copy()
    # Zero out scores below threshold
    scores.data[scores.data < threshold] = 0
    # If too many entries, keep only top max_count
    if scores.nnz > max_count:
        smallest_indices = np.argpartition(scores.data, -max_count)[:len(scores.data) - max_count]
        scores.data[smallest_indices] = 0
    scores.eliminate_zeros()
    return scores


def apply_topk_ranking(
    scores: csr_matrix,
    top_k: int,
) -> csr_matrix:
    """Apply top-K filtering on sparse score matrix.

    Borrowed from Fast-GraphRAG RankingPolicy_TopK.

    Args:
        scores: (1, N) sparse score matrix.
        top_k: Number of top entries to keep.

    Returns:
        Filtered csr_matrix keeping only top_k entries.
    """
    scores = scores.copy()
    if scores.nnz <= top_k:
        return scores
    smallest_indices = np.argpartition(scores.data, -top_k)[:len(scores.data) - top_k]
    scores.data[smallest_indices] = 0
    scores.eliminate_zeros()
    return scores


# ── Cascade Propagation Engine ───────────────────────────────────


class CascadePropagation:
    """Three-layer cascade propagation: Entity → Relation → Chunk.

    This is the core retrieval architecture replacing the previous
    Vector + BM25 + Graph → RRF parallel fusion.

    Pipeline:
        1. Vector search → seed entity scores (1, #entities)
        2. PPR propagation → entity scores (1, #entities)
        3. Entity scores * e2r → relation scores (1, #relations)
        4. Relation scores * r2c → chunk scores (1, #chunks)

    Optional BM25 enhancement:
        When bm25_enabled=True, chunk scores from cascade and BM25
        chunk results are fused via RRF as a post-processing step.

    Usage::

        cascade = CascadePropagation(config=CascadeConfig())
        result = cascade.retrieve(
            query="What is the treatment for diabetes?",
            vector_search_fn=my_vector_search,
            graph_client=my_hg_client,
            e2r_matrix=my_e2r,
            r2c_matrix=my_r2c,
            entity_index_map=my_entity_map,
        )
    """

    def __init__(self, config: Optional[CascadeConfig] = None) -> None:
        self.config = config or CascadeConfig()

    def retrieve(
        self,
        query: str,
        vector_search_fn,         # Callable: (query, top_k, threshold) → [(entity_id, score), ...]
        ppr_retriever: Optional[PPRRetriever] = None,  # PPRRetriever instance
        graph_client=None,        # PyHugeClient for fallback subgraph fetch
        e2r_matrix: Optional[csr_matrix] = None,
        r2c_matrix: Optional[csr_matrix] = None,
        entity_index_map: Optional[Dict[str, int]] = None,  # entity_id → row index in e2r
        relation_index_map: Optional[Dict[str, int]] = None,
        chunk_index_map: Optional[Dict[str, int]] = None,
        bm25_search_fn=None,     # Optional: (query, top_k) → [chunk_id, ...]
    ) -> CascadeResult:
        """Execute cascade propagation retrieval.

        Args:
            query: User question string.
            vector_search_fn: Function that returns (entity_id, similarity_score) pairs.
            ppr_retriever: Optional PPRRetriever for graph-based PPR computation.
            graph_client: Optional PyHugeClient for subgraph fetch.
            e2r_matrix: Entity→Relation sparse mapping matrix.
            r2c_matrix: Relation→Chunk sparse mapping matrix.
            entity_index_map: Maps entity_id to column index in seed vector / e2r rows.
            relation_index_map: Maps relation_id to column index in e2r / r2c rows.
            chunk_index_map: Maps chunk_id to column index in r2c.
            bm25_search_fn: Optional BM25 search function (if bm25_enabled=True).

        Returns:
            CascadeResult with entity/relation/chunk scores and metadata.
        """
        import time
        t0 = time.perf_counter()
        result = CascadeResult()
        ranking = self.config.ranking

        # ── Step 1: Vector seed extraction ──────────────────────
        log.info("[Cascade] Step 1: Vector seed extraction...")
        seed_results = vector_search_fn(
            query,
            top_k=self.config.vector_top_k,
            threshold=self.config.vector_threshold,
        )
        seed_entities = [eid for eid, _ in seed_results]
        result.seed_entities = seed_entities

        if not seed_entities:
            log.warning("[Cascade] No vector seed entities found, returning empty result")
            result.stats["duration_ms"] = (time.perf_counter() - t0) * 1000
            return result

        # Build seed vector (1, #entities) sparse matrix
        num_entities = len(entity_index_map) if entity_index_map else max(
            (entity_index_map.get(eid, -1) for eid in seed_entities), default=0
        ) + 1

        seed_scores = np.zeros(num_entities, dtype=np.float32)
        for eid, score in seed_results:
            idx = entity_index_map.get(eid)
            if idx is not None and idx < num_entities:
                seed_scores[idx] = max(seed_scores[idx], score)  # Take max across seeds
        seed_matrix = csr_matrix(seed_scores.reshape(1, -1))

        # Row-normalize seed scores
        row_sum = seed_matrix.sum()
        if row_sum > 0:
            seed_matrix = seed_matrix / row_sum

        log.info(f"[Cascade] Seed vector built: {len(seed_entities)} entities, "
                 f"seed_matrix shape={seed_matrix.shape}")

        # ── Step 2: PPR propagation on entity layer ─────────────
        log.info("[Cascade] Step 2: PPR propagation...")
        ppr_start = time.perf_counter()

        if ppr_retriever and graph_client:
            # Use real HugeGraph PPR
            entity_scores_matrix = self._ppr_via_hugegraph(
                seed_entities, seed_matrix, ppr_retriever,
                entity_index_map, num_entities,
            )
        else:
            # Fallback: use seed scores directly (no PPR propagation)
            log.warning("[Cascade] No PPR retriever provided, using seed scores directly")
            entity_scores_matrix = seed_matrix

        # Apply entity ranking policy
        entity_scores_matrix = apply_threshold_ranking(
            entity_scores_matrix,
            threshold=ranking.entity_threshold,
            max_count=ranking.entity_max_count,
        )

        ppr_elapsed = time.perf_counter() - ppr_start
        log.info(f"[Cascade] PPR done: {entity_scores_matrix.nnz} entities kept "
                 f"in {ppr_elapsed:.3f}s")

        # Extract entity scores dict for result
        entity_scores_dict = self._sparse_to_score_dict(
            entity_scores_matrix, entity_index_map
        )
        result.entity_scores = entity_scores_dict

        # ── Step 3: Entity → Relation propagation ───────────────
        if e2r_matrix is not None:
            log.info("[Cascade] Step 3: Entity → Relation propagation...")
            relation_scores_matrix = entity_scores_matrix.dot(e2r_matrix)

            # Apply relation ranking policy
            relation_scores_matrix = apply_topk_ranking(
                relation_scores_matrix, top_k=ranking.relation_top_k,
            )

            relation_scores_dict = self._sparse_to_score_dict(
                relation_scores_matrix, relation_index_map
            )
            result.relation_scores = relation_scores_dict
            log.info(f"[Cascade] Relation scores: {relation_scores_matrix.nnz} relations kept")
        else:
            log.warning("[Cascade] No e2r matrix, skipping Entity→Relation propagation")

        # ── Step 4: Relation → Chunk propagation ────────────────
        if r2c_matrix is not None and relation_scores_matrix is not None:
            log.info("[Cascade] Step 4: Relation → Chunk propagation...")
            chunk_scores_matrix = relation_scores_matrix.dot(r2c_matrix)

            # Apply chunk ranking policy
            chunk_scores_matrix = apply_topk_ranking(
                chunk_scores_matrix, top_k=ranking.chunk_top_k,
            )

            chunk_scores_dict = self._sparse_to_score_dict(
                chunk_scores_matrix, chunk_index_map
            )
            result.chunk_scores = chunk_scores_dict
            log.info(f"[Cascade] Chunk scores: {chunk_scores_matrix.nnz} chunks kept")
        else:
            log.warning("[Cascade] No r2c matrix, skipping Relation→Chunk propagation")

        # ── Optional Step 5: BM25 enhancement ───────────────────
        if self.config.bm25_enabled and bm25_search_fn:
            log.info("[Cascade] Step 5: BM25 optional enhancement...")
            bm25_chunks = bm25_search_fn(query, self.config.bm25_top_k)
            result.bm25_results = bm25_chunks

            # RRF fuse cascade chunks with BM25 chunks
            cascade_chunk_ids = list(result.chunk_scores.keys())
            rrf = ReciprocalRankFusion(k=self.config.rrf_k)
            fused = rrf.fuse([
                ("cascade", cascade_chunk_ids),
                ("bm25", bm25_chunks),
            ])
            # Update chunk_scores with fused ranking
            result.chunk_scores = {
                item: fused.scores.get(item, 0.0)
                for item in fused.items
            }
            log.info(f"[Cascade] BM25 enhancement: {len(bm25_chunks)} BM25 results "
                     f"fused → {len(fused.items)} final chunks")

        total_elapsed = time.perf_counter() - t0
        result.stats = {
            "duration_ms": total_elapsed * 1000,
            "seed_count": len(seed_entities),
            "ppr_elapsed_ms": ppr_elapsed * 1000,
            "entity_count": len(entity_scores_dict),
            "relation_count": len(result.relation_scores),
            "chunk_count": len(result.chunk_scores),
            "bm25_enabled": self.config.bm25_enabled,
        }
        log.info(f"[Cascade] Complete: {len(result.chunk_scores)} chunks "
                 f"in {total_elapsed:.3f}s")
        return result

    def _ppr_via_hugegraph(
        self,
        seed_entities: List[str],
        seed_matrix: csr_matrix,
        ppr_retriever: PPRRetriever,
        entity_index_map: Dict[str, int],
        num_entities: int,
    ) -> csr_matrix:
        """Run PPR via HugeGraph REST API and map results to sparse matrix.

        For each seed entity, run PPRRetriever.search() to get PPR scores,
        then aggregate into a (1, #entities) sparse matrix.
        """
        all_ppr_scores = np.zeros(num_entities, dtype=np.float32)

        for seed_id in seed_entities:
            try:
                results = ppr_retriever.search(
                    source_id=seed_id,
                    max_depth=self.config.ppr_max_depth,
                    alpha=self.config.ppr_alpha,
                    epsilon=self.config.ppr_epsilon,
                    top_k=self.config.ranking.entity_max_count,
                )
                for r in results:
                    eid = r.get("node_id", "")
                    score = r.get("ppr_score", 0.0)
                    idx = entity_index_map.get(eid)
                    if idx is not None and idx < num_entities:
                        # Take max score across all seeds
                        all_ppr_scores[idx] = max(all_ppr_scores[idx], score)
            except Exception as e:
                log.warning(f"[Cascade] PPR failed for seed '{seed_id}': {e}")

        # Combine seed scores with PPR scores (weighted blend)
        # seed_matrix has normalized similarity scores, all_ppr_scores has PPR importance
        seed_arr = seed_matrix.toarray().flatten()
        combined = np.maximum(seed_arr, all_ppr_scores)  # Take max as Fast-GraphRAG does

        return csr_matrix(combined.reshape(1, -1))

    @staticmethod
    def _sparse_to_score_dict(
        scores: csr_matrix,
        index_map: Optional[Dict[str, int]],
    ) -> Dict[str, float]:
        """Convert sparse score matrix to id→score dictionary.

        Args:
            scores: (1, N) sparse matrix.
            index_map: Reverse map: index → id. If None, uses integer indices.

        Returns:
            Dict mapping id (or str(index)) to score.
        """
        result: Dict[str, float] = {}
        if scores.shape[1] == 0:
            return result

        coo = scores.tocoo()
        # Build reverse index map: col_index → id
        if index_map:
            reverse_map = {v: k for k, v in index_map.items()}
        else:
            reverse_map = {}

        for col, val in zip(coo.col, coo.data):
            id_str = reverse_map.get(col, str(col))
            if val > 0:
                result[id_str] = float(val)
        return result


# ── Matrix builder utilities ─────────────────────────────────────


class CascadeMatrixBuilder:
    """Build e2r and r2c sparse matrices from HugeGraph knowledge graph.

    Reads the KG structure from HugeGraph and constructs:
    - e2r: (num_entities, num_relations) binary mapping matrix
    - r2c: (num_relations, num_chunks) mapping matrix

    These matrices are used by CascadePropagation for the dot-product
    propagation chain.

    Usage::

        builder = CascadeMatrixBuilder(graph_client=pyhugeclient)
        e2r, r2c, entity_map, relation_map, chunk_map = builder.build(
            graph_name="hugegraph",
            entity_label="Entity",
            relation_edge_labels=["relation", "same_as"],
            chunk_label="Chunk",
        )
    """

    def __init__(self, graph_client=None) -> None:
        """Initialize with a PyHugeClient instance.

        Args:
            graph_client: PyHugeClient connected to HugeGraph server.
        """
        self._client = graph_client

    def build(
        self,
        graph_name: str = "hugegraph",
        entity_label: str = "Entity",
        relation_edge_labels: Optional[List[str]] = None,
        chunk_label: str = "Chunk",
        chunk_text_property: str = "text",
        relation_source_property: str = "source_id",
    ) -> Tuple[csr_matrix, csr_matrix, Dict[str, int], Dict[str, int], Dict[str, int]]:
        """Build cascade propagation matrices from HugeGraph.

        Args:
            graph_name: Graph space name in HugeGraph.
            entity_label: Vertex label for entity nodes.
            relation_edge_labels: Edge labels for relation edges.
            chunk_label: Vertex label for chunk nodes.
            chunk_text_property: Property name for chunk text content.
            relation_source_property: Property name on edges pointing to source chunks.

        Returns:
            Tuple of (e2r, r2c, entity_index_map, relation_index_map, chunk_index_map).
        """
        if not self._client:
            log.warning("[CascadeMatrixBuilder] No graph client, returning empty matrices")
            return (
                csr_matrix((0, 0)),
                csr_matrix((0, 0)),
                {}, {}, {},
            )

        # Fetch all entities
        entities = self._fetch_all_vertices(entity_label, graph_name)
        entity_index_map = {eid: idx for idx, eid in enumerate(entities)}
        num_entities = len(entities)

        # Fetch all edges (relations)
        relation_edge_labels = relation_edge_labels or ["relation"]
        edges = self._fetch_all_edges(relation_edge_labels, graph_name)
        # Create relation IDs from edge tuples (src, tgt, edge_label)
        relation_ids = [
            f"{e.get('source', '')}_{e.get('target', '')}_{e.get('label', '')}"
            for e in edges
        ]
        relation_index_map = {rid: idx for idx, rid in enumerate(relation_ids)}
        num_relations = len(edges)

        # Fetch all chunks
        chunks = self._fetch_all_vertices(chunk_label, graph_name)
        chunk_index_map = {cid: idx for idx, cid in enumerate(chunks)}
        num_chunks = len(chunks)

        # Build e2r matrix: for each entity, which relation edges it participates in
        e2r_data: List[List[int]] = []
        for eid in entities:
            entity_relations = []
            for ridx, edge in enumerate(edges):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                if eid == src or eid == tgt:
                    entity_relations.append(ridx)
            e2r_data.append(entity_relations)

        e2r = csr_from_indices_list(e2r_data, shape=(num_entities, num_relations))

        # Build r2c matrix: for each relation, which chunks it references
        r2c_data: List[List[int]] = []
        for edge in edges:
            # Each relation references chunks via source_id property or
            # by the chunks that contain the source entity
            source_id = edge.get("properties", {}).get(relation_source_property, "")
            chunk_indices = []
            if source_id and source_id in chunk_index_map:
                chunk_indices.append(chunk_index_map[source_id])
            # Also add chunks that reference the source/target entities
            src_entity = edge.get("source", "")
            tgt_entity = edge.get("target", "")
            for cid, chunk_text in self._get_chunk_texts(chunk_label, graph_name).items():
                if src_entity.lower() in chunk_text.lower() or tgt_entity.lower() in chunk_text.lower():
                    if cid in chunk_index_map:
                        chunk_indices.append(chunk_index_map[cid])
            r2c_data.append(list(set(chunk_indices)))

        r2c = csr_from_indices_list(r2c_data, shape=(num_relations, num_chunks))

        log.info(f"[CascadeMatrixBuilder] Built matrices: "
                 f"e2r={e2r.shape} ({e2r.nnz} non-zeros), "
                 f"r2c={r2c.shape} ({r2c.nnz} non-zeros), "
                 f"entities={num_entities}, relations={num_relations}, chunks={num_chunks}")

        return e2r, r2c, entity_index_map, relation_index_map, chunk_index_map

    def build_from_local(
        self,
        entities: List[str],
        relations: List[Dict[str, Any]],
        chunks: List[str],
        chunk_texts: Optional[Dict[str, str]] = None,
    ) -> Tuple[csr_matrix, csr_matrix, Dict[str, int], Dict[str, int], Dict[str, int]]:
        """Build cascade matrices from local data (no HugeGraph connection needed).

        Useful for testing and offline processing.

        Args:
            entities: List of entity IDs.
            relations: List of relation dicts with 'source' and 'target' keys.
            chunks: List of chunk IDs.
            chunk_texts: Optional dict mapping chunk_id to text content.

        Returns:
            Same as build().
        """
        entity_index_map = {eid: idx for idx, eid in enumerate(entities)}
        relation_ids = [
            f"{r.get('source', '')}_{r.get('target', '')}_{r.get('label', 'relation')}"
            for r in relations
        ]
        relation_index_map = {rid: idx for idx, rid in enumerate(relation_ids)}
        chunk_index_map = {cid: idx for idx, cid in enumerate(chunks)}

        # Build e2r
        e2r_data: List[List[int]] = []
        for eid in entities:
            entity_relations = []
            for ridx, rel in enumerate(relations):
                if eid == rel.get("source", "") or eid == rel.get("target", ""):
                    entity_relations.append(ridx)
            e2r_data.append(entity_relations)

        e2r = csr_from_indices_list(e2r_data, shape=(len(entities), len(relations)))

        # Build r2c
        r2c_data: List[List[int]] = []
        chunk_texts = chunk_texts or {}
        for rel in relations:
            source_id = rel.get("properties", {}).get("source_id", "")
            src_entity = rel.get("source", "")
            tgt_entity = rel.get("target", "")
            chunk_indices = []
            if source_id and source_id in chunk_index_map:
                chunk_indices.append(chunk_index_map[source_id])
            for cid, text in chunk_texts.items():
                if src_entity.lower() in text.lower() or tgt_entity.lower() in text.lower():
                    if cid in chunk_index_map:
                        chunk_indices.append(chunk_index_map[cid])
            r2c_data.append(list(set(chunk_indices)))

        r2c = csr_from_indices_list(r2c_data, shape=(len(relations), len(chunks)))

        return e2r, r2c, entity_index_map, relation_index_map, chunk_index_map

    def _fetch_all_vertices(self, label: str, graph_name: str) -> List[str]:
        """Fetch all vertex IDs for a given label from HugeGraph."""
        if not self._client:
            return []
        try:
            # Use PyHugeClient to get vertices
            vertices = self._client.getVertexByCondition(label=label, limit=10000)
            return [v.get("id", "") for v in vertices]
        except Exception as e:
            log.warning(f"[CascadeMatrixBuilder] Failed to fetch vertices for '{label}': {e}")
            return []

    def _fetch_all_edges(self, edge_labels: List[str], graph_name: str) -> List[Dict]:
        """Fetch all edges for given labels from HugeGraph."""
        if not self._client:
            return []
        try:
            all_edges = []
            for elabel in edge_labels:
                edges = self._client.getEdgeByCondition(edge_label=elabel, limit=10000)
                all_edges.extend(edges or [])
            return all_edges
        except Exception as e:
            log.warning(f"[CascadeMatrixBuilder] Failed to fetch edges: {e}")
            return []

    def _get_chunk_texts(
        self, chunk_label: str, graph_name: str
    ) -> Dict[str, str]:
        """Fetch chunk texts from HugeGraph for r2c matrix building."""
        if not self._client:
            return {}
        try:
            chunks = self._client.getVertexByCondition(label=chunk_label, limit=10000)
            return {
                c.get("id", ""): c.get("properties", {}).get("text", "")
                for c in chunks
            }
        except Exception as e:
            log.warning(f"[CascadeMatrixBuilder] Failed to fetch chunk texts: {e}")
            return {}
