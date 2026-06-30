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
G4: DRIFT / Global Search Retrieval — 对标 MS GraphRAG drift_search + LightRAG global mode

实现基于社区的全局推理检索模式，补充HG-AI现有的local(k_neighbor)检索。
设计参考:
  - MS GraphRAG: packages/graphrag/graphrag/query/drift_search.py (DRIFT多跳推理)
  - LightRAG: lightrag/operate.py _get_edge_data() + global mode (关系向量库+Round-Robin合并)
  - MS GraphRAG: packages/graphrag/graphrag/query/local_search/ (Local Search子图+文本+Covariates)

核心思想:
  - **Global Search**: 按社区分组检索 → LLM生成社区级摘要 → 综合回答
  - **DRIFT**: 多跳推理链 (entity → relation → entity → ...) 动态扩展上下文
  - 与现有 local search 的 Round-Robin 合并策略

特性:
  - 社区感知检索 (community-aware retrieval)
  - DRIFT多跳推理链构建
  - Local + Global 结果智能合并
  - 可配置最大跳数、社区数量、结果上限
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures for search modes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedContext:
    """A single piece of retrieved context with provenance."""
    content: str
    source_type: str  # "local_entity", "local_relation", "global_community", "drift_hop", "vector_chunk"
    source_id: str = ""
    score: float = 0.0
    hop_distance: int = 0  # For DRIFT: how many hops from query entities


@dataclass
class SearchResult:
    """Result of a single search operation."""
    query: str
    contexts: List[RetrievedContext] = field(default_factory=list)
    raw_llm_response: str = ""
    mode: str = "unknown"
    duration_ms: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalSearchConfig:
    """Configuration for Global / DRIFT search."""
    max_communities: int = 5          # Max communities to include in global context
    max_hops: int = 3                 # Max hops for DRIFT chain
    max_entities_per_hop: int = 10    # Entities to expand at each hop
    max_relations_per_hop: int = 15   # Relations to follow at each hop
    community_report_top_k: int = 3   # Top-K community reports to include
    round_robin_merge: bool = True    # Merge strategy: True=round-robin, False=global-first
    drift_enabled: bool = True        # Enable DRIFT multi-hop expansion
    global_enabled: bool = True       # Enable community-level retrieval


# ---------------------------------------------------------------------------
# DRIFT Chain Builder
# ---------------------------------------------------------------------------


class DriftChainBuilder:
    """Build multi-hop reasoning chains starting from seed entities.

    DRIFT algorithm (simplified):
      1. Start with entities found in user query (via NER or vector match)
      2. Hop 1: Get all relations of seed entities → collect neighbor entities
      3. Hop 2: Get relations of new neighbors → expand further
      4. Repeat up to max_hops, deduplicating visited nodes
      5. Score each path by relevance weight / distance penalty

    Reference: Microsoft GraphRAG DRIFT Search
    """

    def __init__(self, config: Optional[GlobalSearchConfig] = None) -> None:
        self.config = config or GlobalSearchConfig()

    def build_chains(
        self,
        seed_entities: List[str],
        graph_getter,  # Callable[[str], List[Dict]]: entity_name -> [relations]
        *,
        scorer=None,  # Optional callable(relation_dict) -> float
    ) -> List[Dict[str, Any]]:
        """Build DRIFT reasoning chains from seed entities.

        Parameters
        ----------
        seed_entities : Starting entity names (e.g., extracted from query)
        graph_getter : Function(entity_name) → list of relation dicts.
            Each dict must have ``target`` (or ``tgt_id``, ``source`` etc.)
        scorer : Optional function to score each relation's relevance.

        Returns
        -------
        List of chain dicts, each with:
          {
            "path": [(entity, relation, entity), ...],
            "total_score": float,
            "length": int,
            "entities": set[str],
            "relations": list[dict],
          }
        """
        if not seed_entities:
            return []

        visited: Set[str] = set(seed_entities)
        chains: List[Dict[str, Any]] = []
        current_frontier: List[str] = list(seed_entities)

        for hop in range(1, self.config.max_hops + 1):
            next_frontier: List[str] = []
            frontier_rels: List[Tuple[str, Dict]] = []  # (source_entity, rel_dict)

            for entity in current_frontier:
                try:
                    relations = graph_getter(entity)
                except Exception as e:
                    log.debug("Graph lookup failed for %s: %s", entity, e)
                    continue

                for rel in relations or []:
                    target = (
                        rel.get("target")
                        or rel.get("tgt_id")
                        or rel.get("to", "")
                    )
                    if not target or target == entity:
                        continue

                    score = scorer(rel) if scorer else rel.get("weight", 1.0)
                    frontier_rels.append((entity, {**rel, "_score": score, "_hop": hop}))

                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)

            if not frontier_rels:
                break  # No new relations found; terminate early

            # Record this hop's expansions
            chains.append({
                "hop": hop,
                "relations": [r[1] for r in frontier_rels],
                "new_entities": list(next_frontier),
                "num_expanded": len(frontier_rels),
                "frontier_size": len(next_frontier),
            })

            current_frontier = next_frontier

        return chains

    @staticmethod
    def format_chain_context(
        chains: List[Dict[str, Any]],
        max_relations_per_hop: int = 10,
    ) -> Tuple[str, int]:
        """Format DRIFT chains into a text context block.

        Returns (formatted_text, total_relations_included).
        """
        parts = []
        total = 0
        for step in chains:
            hop_num = step["hop"]
            rels = step["relations"][:max_relations_per_hop]
            total += len(rels)
            lines = [f"\n### DRIFT Hop {hop_num} ({len(rels)} relations)"]
            for r in rels:
                src = r.get("source") or r.get("src_id", "?")
                tgt = r.get("target") or r.get("tgt_id", "?")
                desc = r.get("description") or r.get("relation", "")
                sc = r.get("_score", "?")
                lines.append(f"  - [{sc}] {src} → **{tgt}**: {desc}")
            parts.append("\n".join(lines))

        header = f"<drift_chain>\n## DRIFT Reasoning ({len(chains)} hops, {total} relations)"
        return f"{header}\n{''.join(parts)}\n</drift_chain>", total


# ---------------------------------------------------------------------------
# Global Search (Community-Aware Retrieval)
# ---------------------------------------------------------------------------


class GlobalSearchRetriever:
    """Combines community-level retrieval with local k_neighbor results.

    Implements the "Map-Reduce over communities" pattern from MS GraphRAG:
      - Map phase: Retrieve relevant community reports via embedding similarity
      - Reduce phase: Combine community contexts + generate answer

    Also supports LightRAG-style round-robin merge between local and global results.
    """

    def __init__(
        self,
        config: Optional[GlobalSearchConfig] = None,
        community_reports: Optional[List[Any]] = None,  # List[CommunityReport]
    ) -> None:
        self.config = config or GlobalSearchConfig()
        self._community_reports = community_reports or []
        self._drift_builder = DriftChainBuilder(config)

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        *,
        local_contexts: List[RetrievedContext] | None = None,
        graph_client=None,  # PyHugeClient instance for graph queries
        node_to_community: Optional[Dict[str, int]] = None,
        community_detector=None,  # Optional CommunityDetector for on-demand detection
        embed_model=None,  # sentence-transformers model for report ranking
        llm_generate_fn=None,  # Optional: for final synthesis
    ) -> SearchResult:
        """Execute global search combining multiple strategies.

        Parameters
        ----------
        query : User question string
        query_embedding : Optional pre-computed query embedding
        local_contexts : Existing local retrieval results (k_neighbor, etc.)
        graph_client : HugeGraph PyHugeClient for graph traversal
        node_to_community : Precomputed community assignments
        embed_model : SentenceTransformer for report similarity
        llm_generate_fn : Optional LLM fn for final answer synthesis

        Returns
        -------
        SearchResult with merged contexts from all sources.
        """
        import time
        t0 = time.monotonic()
        all_contexts: List[RetrievedContext] = []

        # --- Phase 1: Community Report Retrieval ---
        if self.config.global_enabled and self._community_reports:
            comm_ctxs = await self._retrieve_community_reports(
                query, query_embedding, embed_model
            )
            all_contexts.extend(comm_ctxs)

        # --- Phase 2: DRIFT Multi-Hop Expansion ---
        if self.config.drift_enabled and graph_client is not None and local_contexts:
            # Extract seed entities from local contexts
            seed_entities = self._extract_seed_entities(local_contexts)
            if seed_entities:
                def _graph_getter(entity):
                    try:
                        edges = graph_client.graph().getEdgesByVertexId(
                            entity, limit=self.config.max_relations_per_hop
                        )
                        return edges if isinstance(edges, list) else []
                    except Exception:
                        return []

                chains = self._drift_builder.build_chains(seed_entities, _graph_getter)
                if chains:
                    drift_text, n_rels = DriftChainBuilder.format_chain_context(chains)
                    all_contexts.append(RetrievedContext(
                        content=drift_text,
                        source_type="drift_chain",
                        score=n_rels,
                        hop_distance=max(c.get("hop", 0) for c in chains),
                    ))

        # --- Phase 3: Merge with Local Results ---
        final_contexts = self._merge_results(
            local_contexts or [], all_contexts
        )

        duration = time.monotonic() - t0
        result = SearchResult(
            query=query,
            contexts=final_contexts,
            mode="global_drift",
            duration_ms=duration * 1000,
            stats={
                "local_count": len(local_contexts) if local_contexts else 0,
                "community_count": sum(
                    1 for c in all_contexts if c.source_type == "global_community"
                ),
                "drift_count": sum(
                    1 for c in all_contexts if c.source_type == "drift_hop"
                ),
                "final_context_count": len(final_contexts),
            },
        )

        log.info(
            "Global+DRIFT search: %d contexts (%.1fms)",
            len(final_contexts), duration * 1000,
        )
        return result

    async def _retrieve_community_reports(
        self,
        query: str,
        query_emb: Optional[List[float]],
        embed_model=None,
    ) -> List[RetrievedContext]:
        """Rank and select top-K community reports by semantic similarity."""
        if not self._community_reports:
            return []

        # Simple keyword matching fallback if no embedding model
        if query_emb is None and embed_model is None:
            query_lower = query.lower()
            scored = []
            for cr in self._community_reports:
                title_score = sum(
                    1 for w in query_lower.split() if w in cr.title.lower()
                )
                summary_score = sum(
                    1 for w in query_lower.split() if w in cr.summary.lower()
                )
                scored.append((cr, title_score * 2 + summary_score))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[: self.config.max_communities]

        elif embed_model is not None and query_emb is None:
            q_emb = embed_model.encode(query, normalize_embeddings=True).tolist()
            selected = []
            for cr in self._community_reports:
                cr_emb = getattr(cr, "embedding", None) or getattr(cr, "full_content_embedding", None)
                if cr_emb is None:
                    # Encode summary as fallback
                    cr_emb = embed_model.encode(cr.summary[:500], normalize_embeddings=True).tolist()

                import numpy as np
                sim = float(np.dot(q_emb, cr_emb)) if len(q_emb) == len(cr_emb) else 0.0
                selected.append((cr, sim))
            selected.sort(key=lambda x: x[1], reverse=True)
            selected = selected[: self.config.max_communities]
        else:
            selected = list(zip(self._community_reports, [0.0] * len(self._community_records)))[:self.config.max_communities]

        ctxs = []
        for cr, score in selected:
            ctxs.append(RetrievedContext(
                content=f"[Community Report: {cr.title}]\n{cr.summary}\n\nFindings:\n" +
                         "\n".join(f"- {f.summary}: {f.explanation}" for f in cr.findings),
                source_type="global_community",
                source_id=str(cr.id),
                score=score,
            ))
        return ctxs

    def _merge_results(
        self,
        local: List[RetrievedContext],
        global_: List[RetrievedContext],
    ) -> List[RetrievedContext]:
        """Merge local and global results using configurable strategy.

        Round-robin (LightRAG style): Interleave local/global items.
        Global-first (MS GraphRAG style): Global first, then local fill-in.
        """
        if self.config.round_robin_merge:
            seen: Set[str] = set()
            merged: List[RetrievedContext] = []
            max_len = max(len(local), len(global_))
            for i in range(max_len):
                if i < len(local):
                    key = local[i].source_id or f"local_{i}"
                    if key not in seen:
                        merged.append(local[i])
                        seen.add(key)
                if i < len(global_):
                    key = global_[i].source_id or f"global_{i}"
                    if key not in seen:
                        merged.append(global_[i])
                        seen.add(key)
            return merged
        else:
            # Global-first
            return list(global_) + list(local)

    @staticmethod
    def _extract_seed_entities(contexts: List[RetrievedContext]) -> List[str]:
        """Extract unique entity names from retrieved contexts."""
        entities: Set[str] = set()
        for ctx in contexts:
            if ctx.source_type in ("local_entity", "local_relation"):
                # Try to parse entity name from content
                for word in ctx.content.split():
                    if word.strip(",.:;\"'[](){}"):
                        entities.add(word.strip())
        return sorted(entities)[:20]  # Cap seeds to prevent explosion
