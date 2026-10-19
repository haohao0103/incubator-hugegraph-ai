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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.engines.skill_graph.embedder import Embedder, HashingEmbedder
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


@dataclass
class BuildReport:
    """Summary of a single build pass (useful for CLI output and tests)."""

    skills: int = 0
    categories: int = 0
    explicit_edges: int = 0
    similarity_edges: int = 0
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
            ``similar_to`` edge.
        max_similar_per_node: Maximum induced ``similar_to`` neighbours per node
            (keeps the graph sparse and prevents a hub blow-up).
        enforce_dag: When True (default), break ``depends_on`` cycles.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.72,
        max_similar_per_node: int = 5,
        enforce_dag: bool = True,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.similarity_threshold = similarity_threshold
        self.max_similar_per_node = max(1, max_similar_per_node)
        self.enforce_dag = enforce_dag

    # ── public API ───────────────────────────────────────────
    def build(self, graph: SkillGraph, induce_similarity: bool = True) -> Tuple[SkillGraph, BuildReport]:
        """Return ``(graph, report)`` after embedding + edge induction.

        The graph is mutated in place and also returned for convenience.
        """
        report = BuildReport()
        skills = graph.skills()
        report.skills = len(skills)
        report.categories = len(graph.categories())

        # 1. embeddings
        report.embedding_dim = self._embed_nodes(skills)

        explicit_before = len(graph.edges)
        report.dangling_edges = self._drop_dangling_edges(graph)
        report.explicit_edges = len(graph.edges)

        # 2. similarity induction
        if induce_similarity:
            report.similarity_edges = self._induce_similarity(graph, skills)

        # 3. symmetric closure
        report.symmetric_edges_added = self._materialise_symmetric(graph)

        # 4. conflict pruning (after all positive edges exist)
        report.pruned_conflicts = graph.prune_conflicts()

        # 5. DAG enforcement
        if self.enforce_dag:
            report.cycles_broken = self._break_cycles(graph)

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
    def _embed_nodes(self, skills: List[SkillNode]) -> int:
        dim = 0
        for node in skills:
            vec = self.embedder.embed(node.index_text)
            node.embedding = vec
            dim = len(vec) or dim
        return dim

    def _drop_dangling_edges(self, graph: SkillGraph) -> int:
        kept: List[SkillEdge] = []
        dropped = 0
        for edge in graph.edges:
            if edge.source in graph.nodes and edge.target in graph.nodes:
                kept.append(edge)
            else:
                dropped += 1
        if dropped:
            graph.edges = kept
            graph._edge_keys = {e.key() for e in kept}  # noqa: SLF001 - internal container
        return dropped

    def _induce_similarity(self, graph: SkillGraph, skills: List[SkillNode]) -> int:
        """Add ``similar_to`` edges for mutually-close skills.

        Each node proposes its top-k most similar peers above the threshold; an
        edge survives only if the proposal is *mutual* (kNN-graph style), which
        avoids one-directional hub noise.
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
    clear: bool = False,
) -> Dict[str, int]:
    """Best-effort write of the skill graph into HugeGraph.

    Uses ``PyHugeClient`` with CUSTOMIZE_STRING ids so the stable node ids
    (``skill::...`` / ``category::...``) are used directly as vertex ids. The
    schema is created on demand. Returns vertex/edge counts written.

    This is intentionally optional — the whole engine works offline from JSON —
    and raises only on schema/connection errors the caller does not handle.
    """
    from pyhugegraph.client import PyHugeClient

    client = PyHugeClient(url=url, user=user, pwd=pwd, graph=graph_name)
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
                g.traversal().V().hasLabel(label).drop().iterate()
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
            g.addVertex(label, node.id, props)
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