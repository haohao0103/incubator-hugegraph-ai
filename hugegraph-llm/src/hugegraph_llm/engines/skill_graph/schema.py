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

"""Skill Graph ontology and in-memory container.

The schema follows the industry-mature designs this project aligns with:

  - **SkillNet** (zjunlp): three layers — category layer, relation layer,
    skill-package layer. Its relation set is ``compose_with / depend_on /
    similar_to / belong_to``.
  - **Graph-of-Skills (GoS)**: typed edges with a dependency DAG backbone;
    retrieval walks the prerequisites *backwards* from semantic seeds.
  - **SkillDAG**: typed edges where ``conflicts_with`` is used *only* for
    pruning, and ``depends_on`` must stay acyclic.

Node types
    - ``skill``:    an atomic or compound capability (one ``SKILL.md``).
    - ``category``: a taxonomy/grouping node (target of ``belongs_to``).

Edge types
    - ``depends_on``:     source requires target as a prerequisite (DAG).
    - ``composes_with``:  source is composed together with target (symmetric).
    - ``similar_to``:     near-duplicate / alternative skill (symmetric).
    - ``belongs_to``:     source belongs to a category node.
    - ``conflicts_with``: mutually exclusive; used ONLY for pruning (symmetric).
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.engines.skill_graph.errors import EmbeddingMismatchError


class NodeType(str, Enum):
    """Vertex labels in the skill graph."""

    SKILL = "skill"
    CATEGORY = "category"


class EdgeType(str, Enum):
    """Typed relations between skill/category nodes."""

    DEPENDS_ON = "depends_on"
    COMPOSES_WITH = "composes_with"
    SIMILAR_TO = "similar_to"
    BELONGS_TO = "belongs_to"
    CONFLICTS_WITH = "conflicts_with"


# Symmetric relations (a) point both ways semantically and (b) may be created
# from a single declared direction.
SYMMETRIC_EDGE_TYPES: Set[EdgeType] = {
    EdgeType.COMPOSES_WITH,
    EdgeType.SIMILAR_TO,
    EdgeType.CONFLICTS_WITH,
}

# Prerequisite relations form the acyclic backbone walked during retrieval.
PREREQUISITE_EDGE_TYPES: Set[EdgeType] = {EdgeType.DEPENDS_ON}


def slugify(text: str) -> str:
    """Lowercase kebab-case slug used to derive stable node ids."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unnamed"


def skill_id(name: str) -> str:
    """Stable id for a skill node."""
    return f"skill::{slugify(name)}"


def category_id(name: str) -> str:
    """Stable id for a category node."""
    return f"category::{slugify(name)}"


def cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Cosine similarity between two vectors (0.0 when either is missing).

    A *dimension mismatch* is never a legitimate 0.0: it means the two vectors
    were produced by different embedders, and silently returning 0.0 would make
    the semantic channel look healthy while comparing incomparable spaces.
    Raise instead so the caller fails loudly.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise EmbeddingMismatchError(
            f"cannot compute cosine between {len(a)}-d and {len(b)}-d vectors: "
            "the graph and the query were embedded by different models/dimensions; "
            "rebuild the graph with the current embedder"
        )
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SkillNode:
    """A vertex in the skill graph."""

    id: str
    name: str
    description: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    node_type: NodeType = NodeType.SKILL
    source_path: Optional[str] = None
    body: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text used for embedding and BM25 indexing (not the full body)."""
        parts = [self.name, self.description, " ".join(self.tags)]
        return " ".join(p for p in parts if p)

    def token_estimate(self) -> int:
        """Rough context cost of this node when hydrated (~4 chars per token)."""
        payload = self.body or self.index_text
        return max(1, math.ceil(len(payload) / 4))

    def to_props(self) -> Dict[str, Any]:
        """Flat property dict for a HugeGraph vertex."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": ",".join(self.tags),
            "node_type": self.node_type.value,
            "source_path": self.source_path or "",
            "token_estimate": self.token_estimate(),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["node_type"] = self.node_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillNode":
        data = dict(data)
        data["node_type"] = NodeType(data.get("node_type", NodeType.SKILL.value))
        return cls(**data)


@dataclass
class SkillEdge:
    """A typed relation between two nodes."""

    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    provenance: str = "explicit"  # explicit | similarity | heuristic

    def key(self) -> Tuple[str, str, str]:
        return (self.source, self.target, self.edge_type.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillEdge":
        data = dict(data)
        data["edge_type"] = EdgeType(data["edge_type"])
        return cls(**data)


class SkillGraph:
    """In-memory skill graph with id-deduplicated nodes and edges.

    The graph is intentionally storage-agnostic: it can be produced by the
    offline builder, persisted to HugeGraph / JSON, and consumed by the
    retriever — all without a live database.
    """

    def __init__(
        self,
        nodes: Optional[List[SkillNode]] = None,
        edges: Optional[List[SkillEdge]] = None,
        manifest: Optional[Any] = None,
    ):
        self.nodes: Dict[str, SkillNode] = {}
        self.edges: List[SkillEdge] = []
        self._edge_keys: Set[Tuple[str, str, str]] = set()
        #: Optional :class:`~...manifest.BuildManifest`. Populated by the builder
        #: and persisted with the graph so readers can verify embedder identity,
        #: content hash and build lineage.
        self.manifest = manifest
        for node in nodes or []:
            self.add_node(node)
        for edge in edges or []:
            self.add_edge(edge)

    # ── mutation ─────────────────────────────────────────────
    def add_node(self, node: SkillNode) -> None:
        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and every edge touching it (keeps the edge index exact)."""
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        kept = [e for e in self.edges if e.source != node_id and e.target != node_id]
        if len(kept) != len(self.edges):
            self.edges = kept
            self._edge_keys = {e.key() for e in kept}
        return True

    def add_edge(self, edge: SkillEdge) -> bool:
        """Add an edge if both endpoints exist and it is not a duplicate."""
        if edge.source == edge.target:
            return False
        if edge.source not in self.nodes or edge.target not in self.nodes:
            return False
        if edge.key() in self._edge_keys:
            return False
        self._edge_keys.add(edge.key())
        self.edges.append(edge)
        return True

    # ── queries ──────────────────────────────────────────────
    def skills(self) -> List[SkillNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.SKILL]

    def categories(self) -> List[SkillNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.CATEGORY]

    def out_edges(
        self, node_id: str, edge_types: Optional[Set[EdgeType]] = None
    ) -> List[SkillEdge]:
        return [
            e
            for e in self.edges
            if e.source == node_id and (edge_types is None or e.edge_type in edge_types)
        ]

    def in_edges(
        self, node_id: str, edge_types: Optional[Set[EdgeType]] = None
    ) -> List[SkillEdge]:
        return [
            e
            for e in self.edges
            if e.target == node_id and (edge_types is None or e.edge_type in edge_types)
        ]

    def prerequisites(self, node_id: str) -> List[str]:
        """Direct prerequisites (``depends_on`` targets) of a node."""
        return [e.target for e in self.out_edges(node_id, PREREQUISITE_EDGE_TYPES)]

    def dependents(self, node_id: str) -> List[str]:
        """Nodes that directly depend on ``node_id``."""
        return [e.source for e in self.in_edges(node_id, PREREQUISITE_EDGE_TYPES)]

    def prerequisite_closure(self, node_id: str) -> Set[str]:
        """Transitive prerequisite closure (excluding the node itself)."""
        seen: Set[str] = set()
        stack = list(self.prerequisites(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in self.nodes:
                continue
            seen.add(cur)
            stack.extend(self.prerequisites(cur))
        return seen

    def find_cycles(self) -> List[List[str]]:
        """Detect cycles in the ``depends_on`` backbone (should be empty)."""
        adj: Dict[str, List[str]] = {
            nid: self.prerequisites(nid) for nid in self.nodes
        }
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in self.nodes}
        cycles: List[List[str]] = []
        path: List[str] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            path.append(node)
            for nxt in adj.get(node, []):
                if color.get(nxt) == GRAY:
                    idx = path.index(nxt)
                    cycles.append(path[idx:] + [nxt])
                elif color.get(nxt) == WHITE:
                    dfs(nxt)
            path.pop()
            color[node] = BLACK

        for nid in list(self.nodes):
            if color[nid] == WHITE:
                dfs(nid)
        return cycles

    def prune_conflicts(self) -> int:
        """Remove ``conflicts_with`` edges that contradict a positive edge.

        Per SkillDAG, ``conflicts_with`` is a *pruning* signal: if a pair is
        simultaneously connected by a constructive relation (``depends_on`` /
        ``composes_with`` / ``similar_to``), the conflict edge is dropped.
        """
        positive: Set[Tuple[str, str]] = set()
        for e in self.edges:
            if e.edge_type in (
                EdgeType.DEPENDS_ON,
                EdgeType.COMPOSES_WITH,
                EdgeType.SIMILAR_TO,
            ):
                positive.add((e.source, e.target))
                positive.add((e.target, e.source))
        kept: List[SkillEdge] = []
        removed = 0
        for e in self.edges:
            if e.edge_type == EdgeType.CONFLICTS_WITH and (e.source, e.target) in positive:
                removed += 1
                continue
            kept.append(e)
        if removed:
            self.edges = kept
            self._edge_keys = {e.key() for e in kept}
        return removed

    def stats(self) -> Dict[str, Any]:
        edge_counts: Dict[str, int] = {}
        for e in self.edges:
            edge_counts[e.edge_type.value] = edge_counts.get(e.edge_type.value, 0) + 1
        return {
            "nodes": len(self.nodes),
            "skills": len(self.skills()),
            "categories": len(self.categories()),
            "edges": len(self.edges),
            "edge_types": edge_counts,
            "cycles": len(self.find_cycles()),
        }

    # ── persistence ──────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.manifest is not None:
            data["manifest"] = self.manifest.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillGraph":
        from hugegraph_llm.engines.skill_graph.manifest import BuildManifest

        nodes = [SkillNode.from_dict(d) for d in data.get("nodes", [])]
        edges = [SkillEdge.from_dict(d) for d in data.get("edges", [])]
        raw_manifest = data.get("manifest")
        manifest = BuildManifest.from_dict(raw_manifest) if raw_manifest else None
        return cls(nodes=nodes, edges=edges, manifest=manifest)

    def save(self, path: str) -> None:
        """Atomically write the artifact (temp file in the same directory).

        A crash mid-write must never leave a truncated JSON that a later
        incremental build would happily load as the base graph.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "SkillGraph":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))