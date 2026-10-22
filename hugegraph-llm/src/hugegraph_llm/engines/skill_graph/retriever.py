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
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hugegraph_llm.engines.skill_graph.embedder import (
    Embedder,
    HashingEmbedder,
    embedder_dim,
    embedder_identity,
    tokenize,
)
from hugegraph_llm.engines.skill_graph.errors import EmbeddingMismatchError, GraphReadError
from hugegraph_llm.engines.skill_graph.schema import (
    PREREQUISITE_EDGE_TYPES,
    EdgeType,
    NodeType,
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
    #: True when part of the result could not be served from the active build
    #: (e.g. a seed missing from HugeGraph, or a candidate without a body).
    #: Degradation is always *reported*, never silent.
    degraded: bool = False
    warnings: List[str] = field(default_factory=list)

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
            "degraded": self.degraded,
            "warnings": list(self.warnings),
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


#: BM25 index per immutable graph snapshot. A service that constructs a
#: retriever per request would otherwise re-tokenize and re-index the whole
#: skill set every time; the snapshot cache keeps the graph object stable, so
#: the index built for it can be reused. Weak keys mean the index dies with the
#: graph (e.g. when the snapshot cache evicts an old build) instead of pinning
#: memory. A graph must not be mutated after indexing - which retriever
#: construction already assumes, since it also caches ``_skills``.
_BM25_CACHE: "weakref.WeakKeyDictionary[SkillGraph, _BM25]" = weakref.WeakKeyDictionary()
_BM25_LOCK = threading.Lock()


def _bm25_for(graph: SkillGraph, skills: Dict[str, SkillNode]) -> _BM25:
    """Return the graph's BM25 index, building it once (thread-safe)."""
    with _BM25_LOCK:
        index = _BM25_CACHE.get(graph)
    if index is not None:
        return index
    index = _BM25({sid: tokenize(node.index_text) for sid, node in skills.items()})
    with _BM25_LOCK:
        _BM25_CACHE[graph] = index
    return index


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
        self._validate_embedder()
        self._bm25 = _bm25_for(graph, self._skills)

    def _validate_embedder(self) -> None:
        """Refuse to query with an embedder the graph was not built with.

        This is checked once at construction (not per query) so misconfiguration
        fails at startup instead of degrading every result. Legacy artifacts
        without a manifest are still checked dimension-wise, which catches the
        common "switched providers" mistake.
        """
        dims = {len(n.embedding) for n in self._skills.values() if n.embedding}
        if len(dims) > 1:
            raise EmbeddingMismatchError(
                f"graph contains mixed embedding dimensions {sorted(dims)}; rebuild it from "
                "a single embedder before querying"
            )
        graph_dim = next(iter(dims)) if dims else 0
        query_dim = embedder_dim(self.embedder)
        identity = embedder_identity(self.embedder)

        manifest = getattr(self.graph, "manifest", None)
        if manifest is not None:
            stored_id = getattr(manifest, "embedder_id", "") or ""
            stored_dim = int(getattr(manifest, "embedder_dim", 0) or 0)
            if stored_dim and query_dim and stored_dim != query_dim:
                raise EmbeddingMismatchError(
                    f"query embedder produces {query_dim}-d vectors but the graph was built "
                    f"with {stored_dim}-d ones; use a matching embedder or rebuild"
                )
            if stored_id and stored_id != identity:
                raise EmbeddingMismatchError(
                    f"query embedder {identity!r} does not match the graph's embedder "
                    f"{stored_id!r}; semantic search would compare different vector spaces. "
                    "Use the same embedding configuration or rebuild the graph."
                )

        if graph_dim and query_dim and graph_dim != query_dim:
            raise EmbeddingMismatchError(
                f"query embedder produces {query_dim}-d vectors but the graph holds "
                f"{graph_dim}-d ones; use a matching embedder or rebuild"
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
        return self.retrieve_from_seeds(
            query,
            seeds,
            token_budget=token_budget,
            max_skills=max_skills,
            include_compositions=include_compositions,
            include_similar=include_similar,
        )

    def retrieve_from_seeds(
        self,
        query: str,
        seeds: List[RetrievedSkill],
        token_budget: int = 4000,
        max_skills: Optional[int] = None,
        include_compositions: bool = True,
        include_similar: bool = False,
    ) -> RetrievalResult:
        """Expand an externally-produced seed ranking into the final result.

        Seeds come from the caller - in the service path OceanBase has already
        fused fulltext + vector recall - so the retriever only runs dependency
        walking, PPR and budget packing over the graph it was given.
        """
        seed_ids = [s.id for s in seeds]
        if not seed_ids:
            return RetrievalResult(query=query)

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
        # Seeds are iterated before graph-reached skills, so the user's own
        # matches always claim the budget first; reached skills fill the rest.
        for sid in ordered + reached:
            if len(result.skills) >= limit:
                break
            cost = self._skills[sid].token_estimate()
            # Always admit the very first skill so a result is never empty.
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


class ServiceSkillRetriever:
    """Query-path retriever for the production (query-down) architecture.

    Flow per query::

        pointer = HugeGraph.active_pointer()          # one tiny read, the snapshot
        seeds   = OceanBase.hybrid_seed_search(...)   # bounded fulltext + vector RRF
        subgraph= HugeGraph.expand_prerequisites(...) # bounded BFS, depth/node caps
        bodies  = OceanBase.fetch_bodies(...)         # bounded PK lookup
        result  = SkillRetriever.retrieve_from_seeds(...)

    Nothing loads the whole graph, and every read is pinned to the ``build_id``
    resolved once at the start of the query, so a concurrent publish/rollback
    can never produce a mixed-version answer.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        hg_url: Optional[str] = None,
        hg_user: str = "admin",
        hg_pwd: str = "admin",
        hg_graphspace: Optional[str] = None,
        hg_graph: Optional[str] = None,
        hg_store: Optional[Any] = None,
        ob_host: Optional[str] = None,
        ob_port: Optional[int] = None,
        ob_user: Optional[str] = None,
        ob_password: Optional[str] = None,
        ob_database: Optional[str] = None,
        seed_candidates: int = 100,
        per_channel: int = 50,
        max_depth: int = 2,
        max_nodes: int = 200,
        max_edges: Optional[int] = None,
    ):
        if hg_store is None:
            from hugegraph_llm.engines.skill_graph.service_loader import _open_hg_store

            hg_store = _open_hg_store(hg_url, hg_user, hg_pwd, hg_graphspace, hg_graph)
        self.hg_store = hg_store
        self.embedder = embedder or HashingEmbedder()
        self._ob_kwargs = {
            "host": ob_host,
            "port": ob_port,
            "user": ob_user,
            "password": ob_password,
            "database": ob_database,
        }
        self.seed_candidates = max(1, int(seed_candidates))
        self.per_channel = max(1, int(per_channel))
        self.max_depth = max(0, int(max_depth))
        self.max_nodes = max(1, int(max_nodes))
        self.max_edges = max_edges

    def retrieve(
        self,
        query: str,
        top_k_seeds: int = 5,
        token_budget: int = 4000,
        max_skills: Optional[int] = None,
        include_compositions: bool = True,
        include_similar: bool = False,
    ) -> RetrievalResult:
        """Answer ``query`` from the active build with bounded reads only."""
        pointer = self.hg_store.active_pointer()
        if pointer is None:
            raise GraphReadError(
                "HugeGraph has no active skill-graph build pointer; publish a build first"
            )
        self._validate_pointer(pointer)

        table = getattr(pointer, "ob_table", None)
        if not table:
            raise GraphReadError(
                f"active build {pointer.build_id} has no OceanBase table recorded; "
                "republish it with an index table"
            )

        from hugegraph_llm.engines.skill_graph.ob_store import OceanBaseSkillStore

        ob = OceanBaseSkillStore(
            table=table,
            dim=int(getattr(pointer, "embedder_dim", 0) or 1536),
            **self._ob_kwargs,
        )
        try:
            qvec = self.embedder.embed(query) if query.strip() else None
            hits = ob.hybrid_seed_search(
                query,
                qvec,
                per_channel=self.per_channel,
                limit=max(self.seed_candidates, top_k_seeds),
            )
            if not hits:
                return RetrievalResult(query=query)

            seed_ids = [h.node_id for h in hits[: max(1, top_k_seeds)]]
            edge_types = [EdgeType.DEPENDS_ON]
            if include_compositions:
                edge_types.append(EdgeType.COMPOSES_WITH)
            if include_similar:
                edge_types.append(EdgeType.SIMILAR_TO)

            subgraph = self.hg_store.expand_prerequisites(
                seed_ids,
                pointer.build_id,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
                edge_types=edge_types,
            )
            if not subgraph.nodes:
                raise GraphReadError(
                    f"OceanBase returned {len(seed_ids)} seed(s) but HugeGraph build "
                    f"{pointer.build_id} contains none of them; the build and its index "
                    "table are out of sync"
                )

            bodies = ob.fetch_bodies(list(subgraph.nodes))
        finally:
            ob.close()

        for node_id, body in bodies.items():
            if node_id in subgraph.nodes:
                subgraph.nodes[node_id].body = body

        seed_nodes: List[RetrievedSkill] = []
        for hit in hits:
            node = subgraph.nodes.get(hit.node_id)
            if node is None:
                continue
            seed_nodes.append(
                RetrievedSkill(
                    id=hit.node_id,
                    name=node.name,
                    score=hit.score,
                    reason="seed",
                    depth=0,
                    token_estimate=node.token_estimate(),
                    description=node.description,
                    body=node.body,
                )
            )
            if len(seed_nodes) >= max(1, top_k_seeds):
                break

        result = SkillRetriever(subgraph, embedder=self.embedder).retrieve_from_seeds(
            query,
            seed_nodes,
            token_budget=token_budget,
            max_skills=max_skills,
            include_compositions=include_compositions,
            include_similar=include_similar,
        )

        # Degradation is reported loudly instead of silently serving a partial
        # answer: ops tooling can alert on either flag.
        warnings: List[str] = []
        missing_seeds = [sid for sid in seed_ids if sid not in subgraph.nodes]
        if missing_seeds:
            warnings.append(
                f"{len(missing_seeds)} seed(s) are indexed in OceanBase but missing from "
                f"HugeGraph build {pointer.build_id}: {', '.join(missing_seeds[:5])}"
            )
        missing_bodies = [
            nid
            for nid, node in subgraph.nodes.items()
            if node.node_type == NodeType.SKILL and nid not in bodies
        ]
        if missing_bodies:
            warnings.append(
                f"{len(missing_bodies)} candidate(s) have no body in OceanBase table "
                f"{table}; served with descriptions only"
            )
        result.warnings = warnings
        result.degraded = bool(warnings)
        return result

    def _validate_pointer(self, pointer: Any) -> None:
        """Fail fast when the active vectors cannot be queried with our embedder."""
        stored_id = str(getattr(pointer, "embedder_id", "") or "")
        stored_dim = int(getattr(pointer, "embedder_dim", 0) or 0)
        identity = embedder_identity(self.embedder)
        dim = embedder_dim(self.embedder)
        if stored_id and identity and stored_id != identity:
            raise EmbeddingMismatchError(
                f"active build {pointer.build_id} was indexed with embedder {stored_id!r} "
                f"but the query embedder is {identity!r}"
            )
        if stored_dim and dim and stored_dim != dim:
            raise EmbeddingMismatchError(
                f"active build {pointer.build_id} was indexed with {stored_dim}-d vectors "
                f"but the query embedder produces {dim}-d vectors"
            )