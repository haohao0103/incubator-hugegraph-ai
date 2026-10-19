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

"""Online dependency-aware retrieval (the L2 layer).

This implements the **Graph-of-Skills** retrieval recipe:

  1. **Hybrid seed retrieval** — a query is scored against every skill twice:
     lexically (BM25 over ``name + description + tags``) and semantically
     (cosine over embeddings). The two rankings are fused with Reciprocal Rank
     Fusion (RRF) into the seed set.
  2. **Reverse Personalized PageRank** — from the seeds, probability mass is
     propagated *backwards along the dependency direction* (from a skill to its
     prerequisites), so the whole prerequisite chain lights up with a graded
     importance score. This is what makes retrieval *dependency-aware* instead of
     flat.
  3. **Budgeted hydration** — seeds are emitted first (highest value), then
     prerequisites in PPR order, greedily packed under a token budget. Skills
     beyond the budget are dropped, which is exactly the token saving the graph
     buys us over a flat skill library.

Everything is pure Python and deterministic; no external graph/vector library is
required at query time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from hugegraph_llm.engines.skill_graph.embedder import Embedder, HashingEmbedder, tokenize
from hugegraph_llm.engines.skill_graph.schema import (
    PREREQUISITE_EDGE_TYPES,
    EdgeType,
    SkillGraph,
    SkillNode,
    cosine,
)


@dataclass
class RetrievedSkill:
    """A single retrieved skill with provenance of *why* it was included."""

    id: str
    name: str
    score: float
    reason: str  # seed | prerequisite | composition | similar
    depth: int = 0
    token_estimate: int = 0
    description: str = ""
    body: str = ""

    def to_dict(self, include_body: bool = False) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "score": round(self.score, 6),
            "reason": self.reason,
            "depth": self.depth,
            "token_estimate": self.token_estimate,
            "description": self.description,
        }
        if include_body:
            data["body"] = self.body
        return data


@dataclass
class RetrievalResult:
    """Ordered, budgeted retrieval output."""

    query: str
    seeds: List[RetrievedSkill] = field(default_factory=list)
    skills: List[RetrievedSkill] = field(default_factory=list)
    total_tokens: int = 0
    dropped_tokens: int = 0

    def to_context(self, include_body: bool = True) -> str:
        """Render the retrieved skills as an LLM-ready markdown context block."""
        lines: List[str] = []
        for skill in self.skills:
            header = f"## {skill.name}  _(reason={skill.reason}, depth={skill.depth})_"
            lines.append(header)
            if skill.description:
                lines.append(skill.description)
            if include_body and skill.body:
                lines.append(skill.body)
            lines.append("")
        return "\n".join(lines).strip()

    def to_dict(self, include_body: bool = False) -> Dict[str, object]:
        return {
            "query": self.query,
            "seeds": [s.to_dict(include_body) for s in self.seeds],
            "skills": [s.to_dict(include_body) for s in self.skills],
            "total_tokens": self.total_tokens,
            "dropped_tokens": self.dropped_tokens,
        }


class _BM25:
    """Minimal BM25 index over skill ``index_text`` (lexical seed channel)."""

    def __init__(self, docs: Dict[str, List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_len = {doc_id: len(toks) for doc_id, toks in docs.items()}
        self.avgdl = (sum(self.doc_len.values()) / len(docs)) if docs else 0.0
        self.tf: Dict[str, Dict[str, int]] = {}
        df: Dict[str, int] = {}
        for doc_id, toks in docs.items():
            counts: Dict[str, int] = {}
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf[doc_id] = counts
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        n = len(docs) or 1
        self.idf = {tok: math.log(1 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()}

    def score(self, query_tokens: Iterable[str]) -> Dict[str, float]:
        scores = {doc_id: 0.0 for doc_id in self.docs}
        for tok in query_tokens:
            if tok not in self.idf:
                continue
            idf = self.idf[tok]
            for doc_id, counts in self.tf.items():
                f = counts.get(tok, 0)
                if not f:
                    continue
                dl = self.doc_len[doc_id] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc_id] += idf * (f * (self.k1 + 1)) / denom
        return scores


def _rrf(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion of several ranked id lists."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class SkillRetriever:
    """Dependency-aware skill retriever over a :class:`SkillGraph`.

    Args:
        graph: A built skill graph (embeddings populated by :class:`GraphBuilder`).
        embedder: Embedder used to embed the *query*. Must match the embedder used
            to build the graph (same vector space / dimension). Defaults to the
            offline :class:`HashingEmbedder`, matching the builder default.
        ppr_edge_types: Edge types traversed by the reverse PPR walk. Defaults to
            the dependency backbone plus compositions.
    """

    def __init__(
        self,
        graph: SkillGraph,
        embedder: Optional[Embedder] = None,
        ppr_edge_types: Optional[Set[EdgeType]] = None,
        ppr_damping: float = 0.85,
        ppr_iterations: int = 60,
    ):
        self.graph = graph
        self.embedder = embedder or HashingEmbedder()
        self.ppr_edge_types = ppr_edge_types or {
            EdgeType.DEPENDS_ON,
            EdgeType.COMPOSES_WITH,
        }
        self.ppr_damping = ppr_damping
        self.ppr_iterations = ppr_iterations

        self._skills: Dict[str, SkillNode] = {n.id: n for n in graph.skills()}
        self._bm25 = _BM25(
            {sid: tokenize(node.index_text) for sid, node in self._skills.items()}
        )

    # ── public API ───────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k_seeds: int = 5,
        token_budget: int = 4000,
        max_skills: Optional[int] = None,
        include_compositions: bool = True,
        include_similar: bool = False,
    ) -> RetrievalResult:
        """Retrieve a budgeted, dependency-complete skill set for ``query``."""
        if not self._skills:
            return RetrievalResult(query=query)

        seeds = self.retrieve_seeds(query, top_k=top_k_seeds)
        seed_ids = [s.id for s in seeds]

        reasons, depths = self._walk_dependencies(
            seed_ids,
            include_compositions=include_compositions,
            include_similar=include_similar,
        )
        ppr = self._reverse_ppr(seed_ids)

        # Candidate ordering: seeds first (by fused score), then graph-reached
        # skills by descending PPR mass.
        seed_rank = {sid: i for i, sid in enumerate(seed_ids)}
        ordered: List[str] = list(seed_ids)
        reached = [sid for sid in ppr if sid not in seed_rank]
        reached.sort(key=lambda sid: (-ppr[sid], sid))

        result = RetrievalResult(query=query)
        result.seeds = [
            self._make_skill(
                sid,
                score=seeds[seed_rank[sid]].score,
                reason="seed",
                depth=0,
            )
            for sid in seed_ids
        ]

        budget_left = max(0, token_budget)
        limit = max_skills or len(self._skills) + 1
        for sid in ordered + reached:
            if len(result.skills) >= limit:
                break
            node = self._skills[sid]
            cost = node.token_estimate()
            if cost > budget_left and result.skills:
                result.dropped_tokens += cost
                continue
            budget_left -= cost
            if sid in seed_rank:
                reason, score, depth = "seed", seeds[seed_rank[sid]].score, 0
            else:
                reason = reasons.get(sid, "prerequisite")
                score = ppr.get(sid, 0.0)
                depth = depths.get(sid, 1)
            result.skills.append(self._make_skill(sid, score=score, reason=reason, depth=depth))

        result.total_tokens = sum(s.token_estimate for s in result.skills)
        return result

    def retrieve_seeds(self, query: str, top_k: int = 5) -> List[RetrievedSkill]:
        """Hybrid (BM25 + semantic) seed retrieval fused with RRF."""
        query_tokens = tokenize(query)

        lexical = self._bm25.score(query_tokens)
        lexical_rank = [
            sid
            for sid in sorted(lexical, key=lambda s: (-lexical[s], s))
            if lexical[sid] > 0
        ]

        rankings: List[List[str]] = [lexical_rank]
        if self._has_embeddings() and query.strip():
            qvec = self.embedder.embed(query)
            semantic = {
                sid: cosine(qvec, node.embedding) for sid, node in self._skills.items()
            }
            semantic_rank = [
                sid
                for sid in sorted(semantic, key=lambda s: (-semantic[s], s))
                if semantic[sid] > 0
            ]
            rankings.append(semantic_rank)

        fused = _rrf(rankings)
        if not fused:  # no lexical hit and no embeddings: fall back to all
            fused = {sid: 0.0 for sid in self._skills}

        ranked = sorted(fused, key=lambda s: (-fused[s], s))[: max(1, top_k)]
        return [
            self._make_skill(sid, score=fused[sid], reason="seed", depth=0) for sid in ranked
        ]

    # ── internals ────────────────────────────────────────────
    def _has_embeddings(self) -> bool:
        return any(n.embedding for n in self._skills.values())

    def _make_skill(self, sid: str, score: float, reason: str, depth: int) -> RetrievedSkill:
        node = self._skills[sid]
        return RetrievedSkill(
            id=sid,
            name=node.name,
            score=score,
            reason=reason,
            depth=depth,
            token_estimate=node.token_estimate(),
            description=node.description,
            body=node.body,
        )

    def _walk_dependencies(
        self,
        seed_ids: List[str],
        include_compositions: bool,
        include_similar: bool,
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """BFS over dependency edges; returns per-node reason and depth."""
        edge_types: Set[EdgeType] = set(PREREQUISITE_EDGE_TYPES)
        if include_compositions:
            edge_types.add(EdgeType.COMPOSES_WITH)
        if include_similar:
            edge_types.add(EdgeType.SIMILAR_TO)

        reasons: Dict[str, str] = {}
        depths: Dict[str, int] = {sid: 0 for sid in seed_ids}
        frontier = list(seed_ids)
        depth = 0
        while frontier:
            depth += 1
            nxt: List[str] = []
            for sid in frontier:
                for edge in self.graph.out_edges(sid, edge_types):
                    if edge.target not in depths:
                        depths[edge.target] = depth
                        reasons[edge.target] = (
                            "prerequisite"
                            if edge.edge_type == EdgeType.DEPENDS_ON
                            else ("composition" if edge.edge_type == EdgeType.COMPOSES_WITH else "similar")
                        )
                        nxt.append(edge.target)
            frontier = nxt
        return reasons, depths

    def _reverse_ppr(self, seed_ids: List[str]) -> Dict[str, float]:
        """Reverse Personalized PageRank from seeds along the dependency backbone.

        Transition follows ``source -> prerequisite`` (i.e. *backwards* relative
        to the prerequisite relation), so mass concentrates on the prerequisite
        closure of the seeds.
        """
        if not seed_ids:
            return {}

        nodes = list(self._skills)
        # out_adj[u] = list of (v, weight)
        out_adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
        for edge in self.graph.edges:
            if edge.edge_type not in self.ppr_edge_types:
                continue
            if edge.source in out_adj and edge.target in out_adj:
                out_adj[edge.source].append((edge.target, max(edge.weight, 1e-6)))

        # personalization vector (normalized over seeds)
        personal = {n: 0.0 for n in nodes}
        for sid in seed_ids:
            if sid in personal:
                personal[sid] += 1.0
        total = sum(personal.values()) or 1.0
        personal = {n: v / total for n, v in personal.items()}

        rank = dict(personal)
        damping = self.ppr_damping
        for _ in range(self.ppr_iterations):
            nxt = {n: (1.0 - damping) * personal[n] for n in nodes}
            dangling_mass = 0.0
            for u in nodes:
                ru = rank[u]
                if ru == 0.0:
                    continue
                edges = out_adj[u]
                if not edges:
                    dangling_mass += ru
                    continue
                wsum = sum(w for _, w in edges) or 1.0
                for v, w in edges:
                    nxt[v] += damping * ru * (w / wsum)
            if dangling_mass:
                for n in nodes:
                    nxt[n] += damping * dangling_mass * personal[n]
            rank = nxt
        return {n: v for n, v in rank.items() if v > 1e-9}