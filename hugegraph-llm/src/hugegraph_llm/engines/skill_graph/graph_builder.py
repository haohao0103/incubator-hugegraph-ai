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

"""Offline skill-graph construction (the L1 layer).

Given a :class:`SkillGraph` that already carries the *explicit* edges parsed by
:mod:`skill_loader`, the builder:

  1. **Embeds** every skill node (name + description + tags) so the online
     retriever can do semantic seed matching.
  2. **Induces ``similar_to`` edges** from embedding cosine similarity, keeping
     only mutually-close nodes (mutual top-k above a threshold). This mirrors
     SkillNet's *relation layer*, where near-duplicate skills are linked.
  3. **Materialises symmetric edges** (``similar_to`` / ``composes_with`` /
     ``conflicts_with``) in both directions.
  4. **Enforces the dependency DAG** — the ``depends_on`` backbone must be
     acyclic (Graph-of-Skills / SkillDAG). Cycles are broken by dropping the
     weakest edge in each cycle and reported.
  5. **Prunes conflict edges** that contradict a constructive relation.

The result is a self-contained graph that can be persisted to JSON and/or
written to HugeGraph. Everything here is deterministic and offline: the default
embedder is the dependency-free :class:`~...embedder.HashingEmbedder`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.engines.skill_graph.embedder import (
    Embedder,
    HashingEmbedder,
    embedder_dim,
    embedder_identity,
)
from hugegraph_llm.engines.skill_graph.errors import BuildValidationError, EmbeddingMismatchError
from hugegraph_llm.engines.skill_graph.schema import (
    PREREQUISITE_EDGE_TYPES,
    SYMMETRIC_EDGE_TYPES,
    EdgeType,
    NodeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
    cosine,
)

log = logging.getLogger(__name__)


def _known_dim(graph: SkillGraph) -> int:
    """Dimension of any already-embedded node (0 when none), for incremental reports."""
    for node in graph.nodes.values():
        if node.embedding:
            return len(node.embedding)
    return 0


def assert_incremental_compatible(graph: SkillGraph, embedder: object) -> int:
    """Fail fast when an incremental build would mix vector spaces.

    Two failure modes are indistinguishable at query time unless we check the
    *identity* now:

    * a different dimension (obvious), and
    * a **different model with the same dimension** (silent - cosine still
      returns plausible numbers, but they are meaningless).

    Legacy graphs without a manifest can only be checked dimension-wise; the
    returned dim lets the builder keep reporting it.
    """
    dims = {len(n.embedding) for n in graph.nodes.values() if n.embedding}
    if len(dims) > 1:
        raise EmbeddingMismatchError(
            f"the base graph already mixes embedding dimensions {sorted(dims)}; "
            "rebuild it from scratch before using --base"
        )
    graph_dim = next(iter(dims)) if dims else 0

    current_id = embedder_identity(embedder)
    current_dim = embedder_dim(embedder)

    manifest = getattr(graph, "manifest", None)
    if manifest is not None:
        stored_id = getattr(manifest, "embedder_id", "") or ""
        stored_dim = int(getattr(manifest, "embedder_dim", 0) or 0)
        # Dimension first: it is the specific, easy-to-action failure; the
        # identity check below also fires for same-dimension model switches.
        if stored_dim and current_dim and stored_dim != current_dim:
            raise EmbeddingMismatchError(
                f"embedding dimension mismatch: base graph is {stored_dim}-d, current "
                f"embedder produces {current_dim}-d; rebuild from scratch"
            )
        if stored_id and stored_id != current_id:
            raise EmbeddingMismatchError(
                f"embedder mismatch: the base graph was built with {stored_id!r} but the "
                f"current embedder is {current_id!r}. Vectors from different models are not "
                "comparable (even at the same dimension); rebuild from scratch or use the "
                "original embedder."
            )

    if graph_dim and current_dim and graph_dim != current_dim:
        raise EmbeddingMismatchError(
            f"embedding dimension mismatch: the persisted graph holds {graph_dim}-d vectors "
            f"but the current embedder produces {current_dim}-d ones; mixed dimensions break "
            "cosine similarity. Rebuild from scratch (omit the incremental --base) with the "
            "new embedder."
        )
    return graph_dim or current_dim


@dataclass
class BuildReport:
    """Summary of a single build pass (useful for CLI output and tests)."""

    skills: int = 0
    categories: int = 0
    explicit_edges: int = 0
    similarity_edges: int = 0
    llm_edges: int = 0
    llm_induction: Dict[str, Any] = field(default_factory=dict)
    symmetric_edges_added: int = 0
    pruned_conflicts: int = 0
    cycles_broken: int = 0
    dangling_edges: int = 0
    embedding_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skills": self.skills,
            "categories": self.categories,
            "explicit_edges": self.explicit_edges,
            "similarity_edges": self.similarity_edges,
            "llm_edges": self.llm_edges,
            "llm_induction": self.llm_induction,
            "symmetric_edges_added": self.symmetric_edges_added,
            "pruned_conflicts": self.pruned_conflicts,
            "cycles_broken": self.cycles_broken,
            "dangling_edges": self.dangling_edges,
            "embedding_dim": self.embedding_dim,
        }


class GraphBuilder:
    """Deterministic, offline builder for the skill graph.

    Args:
        embedder: Any object implementing :class:`Embedder`. Defaults to the
            dependency-free, offline :class:`HashingEmbedder`. Pass the result of
            :func:`resolve_embedder` to use the project's production embedder.
        similarity_threshold: Minimum cosine similarity for an induced
            ``similar_to`` edge. The default (0.30) suits the offline hashing
            embedder; raise it (e.g. 0.72) when using a real embedding model.
        max_similar_per_node: Maximum induced ``similar_to`` neighbours per node
            (keeps the graph sparse and prevents a hub blow-up).
        enforce_dag: When True (default), break ``depends_on`` cycles.
        edge_inducer: Optional typed-edge inducer (e.g.
            :class:`~...edge_inducer.LLMEdgeInducer`). When provided, it replaces
            the default similarity pass and induces *typed* edges (the
            industry-standard SkillDAG recipe). Anything exposing
            ``induce(graph, skills) -> InductionReport`` works.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.30,
        max_similar_per_node: int = 5,
        enforce_dag: bool = True,
        edge_inducer: Optional[Any] = None,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.similarity_threshold = similarity_threshold
        self.max_similar_per_node = max(1, max_similar_per_node)
        self.enforce_dag = enforce_dag
        self.edge_inducer = edge_inducer

    # ── public API ───────────────────────────────────────────
    def build(
        self,
        graph: SkillGraph,
        induce_similarity: bool = True,
        new_skill_ids: Optional[Set[str]] = None,
    ) -> Tuple[SkillGraph, BuildReport]:
        """Return ``(graph, report)`` after embedding + edge induction.

        The graph is mutated in place and also returned for convenience.

        ``new_skill_ids`` switches to the **incremental** path: only those
        skills are (re-)embedded, their previously induced edges (provenance
        ``similarity``/``llm``) are dropped first so they are re-induced fresh,
        and the typed-edge inducer is told to only propose pairs touching them.
        Explicit front-matter edges of *all* skills are always preserved and the
        cheap full-graph invariants (symmetric closure, conflict pruning, DAG
        enforcement) are re-run so the merged graph stays consistent. Pass
        ``None`` (default) for a full build.
        """
        report = BuildReport()
        skills = graph.skills()
        report.skills = len(skills)
        report.categories = len(graph.categories())

        if new_skill_ids is not None:
            new_ids = {i for i in new_skill_ids if i in graph.nodes}
            # Fail before spending a single embedding call on an incompatible base.
            expected_dim = assert_incremental_compatible(graph, self.embedder)
            self._drop_induced_edges(graph, new_ids)
            targets = [n for n in skills if n.id in new_ids]
            new_dim = self._embed_nodes(targets)
            if new_dim and expected_dim and new_dim != expected_dim:
                raise EmbeddingMismatchError(
                    f"embedding dimension mismatch: the persisted graph holds {expected_dim}-d "
                    f"vectors but the current embedder produced {new_dim}-d ones; rebuild from "
                    "scratch (omit the incremental --base)"
                )
            report.embedding_dim = new_dim or expected_dim
        else:
            new_ids = None
            report.embedding_dim = self._embed_nodes(skills)

        self._assert_no_dangling_edges(graph)
        explicit_before = len(graph.edges)
        report.explicit_edges = len(graph.edges)

        # 2. edge induction — typed (LLM, SkillDAG recipe) or similarity-only
        if self.edge_inducer is not None:
            induction = self.edge_inducer.induce(graph, skills, new_skill_ids=new_ids)
            report.llm_edges = induction.edges_added
            report.llm_induction = induction.to_dict()
        elif induce_similarity:
            report.similarity_edges = self._induce_similarity(graph, skills, new_ids=new_ids)

        # 3. symmetric closure
        report.symmetric_edges_added = self._materialise_symmetric(graph)

        # 4. conflict pruning (after all positive edges exist)
        report.pruned_conflicts = graph.prune_conflicts()

        # 5. DAG enforcement
        if self.enforce_dag:
            report.cycles_broken = self._break_cycles(graph)

        self._attach_manifest(graph, report, incremental=new_skill_ids is not None)

        log.info(
            "built skill graph: %d skills, %d edges (sim=%d, sym=%d, dag_breaks=%d) "
            "from %d input edges",
            report.skills,
            len(graph.edges),
            report.similarity_edges,
            report.symmetric_edges_added,
            report.cycles_broken,
            explicit_before,
        )
        return graph, report

    def build_from_directory(self, skills_dir: str, **kwargs: Any) -> Tuple[SkillGraph, BuildReport]:
        """Convenience: load ``SKILL.md`` files then build the graph."""
        from hugegraph_llm.engines.skill_graph.skill_loader import load_skills

        graph = load_skills(skills_dir)
        return self.build(graph, **kwargs)

    # ── internals ────────────────────────────────────────────
    def _attach_manifest(self, graph: SkillGraph, report: BuildReport, incremental: bool) -> None:
        """Stamp the graph with a verifiable build identity.

        The previous manifest is captured *before* replacement so lineage
        (``previous_build``) survives and rollback knows what to go back to.
        """
        from hugegraph_llm.engines.skill_graph.manifest import build_manifest

        previous = getattr(graph, "manifest", None)
        dim = report.embedding_dim or embedder_dim(self.embedder)
        if not dim:
            # Empty graph / embedder that reports its dimension only per call:
            # there is nothing to verify yet, so do not fabricate a fingerprint.
            return
        graph.manifest = build_manifest(
            graph,
            embedder_id=embedder_identity(self.embedder),
            embedder_dim=dim,
            previous_build=getattr(previous, "build_id", None),
            mode="incremental" if incremental else "full",
        )

    def _embed_nodes(self, skills: List[SkillNode]) -> int:
        dim = 0
        for node in skills:
            vec = self.embedder.embed(node.index_text)
            node.embedding = vec
            dim = len(vec) or dim
        return dim

    def _assert_no_dangling_edges(self, graph: SkillGraph) -> None:
        """Fail on edges whose endpoint does not exist.

        Dropping them silently hides author typos (``requires: data-ingest``
        instead of ``data-ingestion``): the dependency simply disappears and
        retrieval returns an incomplete plan that looks valid.
        """
        missing = [
            edge
            for edge in graph.edges
            if edge.source not in graph.nodes or edge.target not in graph.nodes
        ]
        if not missing:
            return
        preview = "; ".join(
            f"{e.source} -{e.edge_type.value}-> {e.target}"
            + (f" (missing {e.target})" if e.source in graph.nodes else f" (missing {e.source})")
            for e in missing[:5]
        )
        raise BuildValidationError(
            f"{len(missing)} dangling edge(s): {preview}"
            f"{'...' if len(missing) > 5 else ''}. Fix the 'relations:' entries or add the "
            "missing SKILL.md files."
        )

    def _drop_induced_edges(self, graph: SkillGraph, node_ids: Set[str]) -> int:
        """Remove induced edges touching the given nodes (explicit edges kept).

        Used by the incremental path: a new/updated skill must have its
        similarity/LLM edges re-induced from its fresh content, while explicit
        front-matter edges are the author's declared truth and always survive.
        """
        if not node_ids:
            return 0
        kept: List[SkillEdge] = []
        dropped = 0
        for edge in graph.edges:
            if edge.provenance in ("similarity", "llm") and (edge.source in node_ids or edge.target in node_ids):
                dropped += 1
                continue
            kept.append(edge)
        if dropped:
            graph.edges = kept
            graph._edge_keys = {e.key() for e in kept}  # noqa: SLF001 - internal container
        return dropped

    def _induce_similarity(
        self,
        graph: SkillGraph,
        skills: List[SkillNode],
        new_ids: Optional[Set[str]] = None,
    ) -> int:
        """Add ``similar_to`` edges for mutually-close skills.

        Each node proposes its top-k most similar peers above the threshold; an
        edge survives only if the proposal is *mutual* (kNN-graph style), which
        avoids one-directional hub noise.

        In the incremental path (``new_ids`` given) mutual proposals are still
        computed over the whole pool (cheap, no LLM), but only pairs touching a
        new/updated skill are actually added — old-old edges are untouched.
        """
        embedded = [n for n in skills if n.embedding]
        if len(embedded) < 2:
            return 0

        neighbours: Dict[str, List[Tuple[str, float]]] = {}
        for i, a in enumerate(embedded):
            scored: List[Tuple[str, float]] = []
            for j, b in enumerate(embedded):
                if i == j:
                    continue
                sim = cosine(a.embedding, b.embedding)
                if sim >= self.similarity_threshold:
                    scored.append((b.id, sim))
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
            neighbours[a.id] = scored[: self.max_similar_per_node]

        added = 0
        seen: Set[Tuple[str, str]] = set()
        for src, cands in neighbours.items():
            for tgt, sim in cands:
                pair = (src, tgt) if src < tgt else (tgt, src)
                if pair in seen:
                    continue
                if new_ids is not None and src not in new_ids and tgt not in new_ids:
                    continue
                # Mutual check (kNN graph): keep the edge only if both ends
                # nominated each other, which avoids one-directional hub noise.
                if src not in {cid for cid, _ in neighbours.get(tgt, [])}:
                    continue
                seen.add(pair)
                if graph.add_edge(
                    SkillEdge(
                        source=src,
                        target=tgt,
                        edge_type=EdgeType.SIMILAR_TO,
                        weight=round(sim, 4),
                        provenance="similarity",
                    )
                ):
                    added += 1
        return added

    def _materialise_symmetric(self, graph: SkillGraph) -> int:
        added = 0
        existing = {e.key() for e in graph.edges}
        for edge in list(graph.edges):
            if edge.edge_type not in SYMMETRIC_EDGE_TYPES:
                continue
            rev = SkillEdge(
                source=edge.target,
                target=edge.source,
                edge_type=edge.edge_type,
                weight=edge.weight,
                provenance=edge.provenance,
            )
            if rev.key() not in existing:
                if graph.add_edge(rev):
                    existing.add(rev.key())
                    added += 1
        return added

    def _break_cycles(self, graph: SkillGraph) -> int:
        """Remove the weakest ``depends_on`` edge in each cycle until acyclic."""
        removed = 0
        # Bounded iterations: each pass removes >=1 edge, edges are finite.
        for _ in range(len(graph.edges) + 1):
            cycles = graph.find_cycles()
            if not cycles:
                break
            cycle = cycles[0]
            # pick weakest edge inside the cycle
            cycle_set = set(cycle)
            candidates = [
                e
                for e in graph.edges
                if e.edge_type in PREREQUISITE_EDGE_TYPES
                and e.source in cycle_set
                and e.target in cycle_set
            ]
            if not candidates:
                break
            weakest = min(candidates, key=lambda e: (e.weight, e.source, e.target))
            graph.edges = [e for e in graph.edges if e.key() != weakest.key()]
            graph._edge_keys = {e.key() for e in graph.edges}  # noqa: SLF001
            removed += 1
            log.warning(
                "broke dependency cycle: dropped %s -> %s (weight=%.3f)",
                weakest.source,
                weakest.target,
                weakest.weight,
            )
        return removed


def write_to_hugegraph(
    graph: SkillGraph,
    graph_name: str = "skillgraph",
    url: str = "http://127.0.0.1:8080",
    user: str = "admin",
    pwd: str = "admin",
    graphspace: Optional[str] = None,
    clear: bool = False,
) -> Dict[str, int]:
    """Best-effort write of the skill graph into HugeGraph.

    Uses ``PyHugeClient`` with CUSTOMIZE_STRING ids so the stable node ids
    (``skill::...`` / ``category::...``) are used directly as vertex ids. The
    schema is created on demand. Returns vertex/edge counts written.

    ``graphspace`` targets a named HugeGraph graphspace (HugeGraph 1.7+); when
    omitted the client falls back to the ``DEFAULT`` space.

    This is intentionally optional — the whole engine works offline from JSON —
    and raises only on schema/connection errors the caller does not handle.
    """
    from pyhugegraph.client import PyHugeClient

    client = PyHugeClient(url=url, user=user, pwd=pwd, graph=graph_name, graphspace=graphspace)
    schema = client.schema()

    # property keys
    for pk, ptype in (
        ("node_name", "TEXT"),
        ("description", "TEXT"),
        ("category", "TEXT"),
        ("tags", "TEXT"),
        ("node_type", "TEXT"),
        ("source_path", "TEXT"),
        ("token_estimate", "INT"),
        ("weight", "DOUBLE"),
        ("provenance", "TEXT"),
    ):
        getattr(schema.propertyKey(pk), f"as{ptype.capitalize()}")().ifNotExist().create()

    # vertex labels
    skill_props = [
        "node_name",
        "description",
        "category",
        "tags",
        "node_type",
        "source_path",
        "token_estimate",
    ]
    for label in ("skill", "category"):
        schema.vertexLabel(label).properties(*skill_props).useCustomizeStringId().ifNotExist().create()

    # edge labels
    edge_specs = {
        "depends_on": ("skill", "skill"),
        "composes_with": ("skill", "skill"),
        "similar_to": ("skill", "skill"),
        "conflicts_with": ("skill", "skill"),
        "belongs_to": ("skill", "category"),
    }
    for label, (src, tgt) in edge_specs.items():
        try:
            schema.edgeLabel(label).properties("weight", "provenance").sourceLabel(src).targetLabel(tgt).ifNotExist().create()
        except Exception as exc:  # label may already exist with a different shape
            log.warning("edge label %s not (re)created: %s", label, exc)

    g = client.graph()
    if clear:
        for label in ("skill", "category"):
            try:
                # PyHugeClient 1.x exposes no traversal() on GraphManager; drop
                # through the Gremlin endpoint instead.
                client.gremlin().exec(f"g.V().hasLabel('{label}').drop()")
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("failed to clear %s: %s", label, exc)

    vertex_count = 0
    for node in graph.nodes.values():
        label = "skill" if node.node_type == NodeType.SKILL else "category"
        props = {
            "node_name": node.name,
            "description": node.description,
            "category": node.category,
            "tags": ",".join(node.tags),
            "node_type": node.node_type.value,
            "source_path": node.source_path or "",
            "token_estimate": node.token_estimate(),
        }
        try:
            g.addVertex(label, props, node.id)
            vertex_count += 1
        except Exception as exc:  # pragma: no cover - depends on live server
            log.warning("add vertex %s failed: %s", node.id, exc)

    edge_count = 0
    for edge in graph.edges:
        props = {"weight": float(edge.weight), "provenance": edge.provenance}
        try:
            g.addEdge(edge.edge_type.value, edge.source, edge.target, props)
            edge_count += 1
        except Exception as exc:  # pragma: no cover - depends on live server
            log.warning("add edge %s failed: %s", edge.key(), exc)

    return {"vertices": vertex_count, "edges": edge_count}