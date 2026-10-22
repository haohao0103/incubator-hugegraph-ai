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

"""LLM-driven, typed edge induction (an L1 substep).

Why this module exists
----------------------
The offline builder can only *induce* ``similar_to`` edges from embedding
cosine; anything else must have been declared by hand in each ``SKILL.md``
front-matter. Real corpora such as ``anthropics/skills`` declare nothing, so a
similarity-only build degenerates into a graph where every edge just means
"these two read alike" — which is **not** how the field builds skill graphs.

2026 state of the art induces **typed** edges from skill *text*:

  - **SkillDAG** — two-view embeddings (``e_self`` = "what it does" +
    ``e_needs`` = the LLM's guess of "what invoking it requires"); candidate
    pairs by cosine; then an LLM classifier assigns the edge *type*
    (``depends_on`` / ``composes_with`` / ``similar_to`` / ``conflicts_with``).
  - **CaSKG** — multi-signal candidates + counterfactual probing + Bayesian
    calibration before an edge survives.
  - **SkillGraph** (tool sequences) — mines precedence edges from real
    execution trajectories.

This module implements the **SkillDAG recipe** — the lightest one that still
yields typed edges — on top of any object exposing ``generate(prompt=...) -> str``
(the project's chat LLM client). It is strictly **opt-in**: :class:`GraphBuilder`
keeps the deterministic offline similarity as the default so tests and offline
builds stay reproducible.

Usage::

    from hugegraph_llm.engines.skill_graph.edge_inducer import LLMEdgeInducer
    from hugegraph_llm.models.llms.init_llm import LLMs

    inducer = LLMEdgeInducer(LLMs().get_chat_llm())
    graph, report = GraphBuilder(edge_inducer=inducer).build(graph)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from hugegraph_llm.engines.skill_graph.embedder import Embedder, HashingEmbedder
from hugegraph_llm.engines.skill_graph.schema import (
    EdgeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
    cosine,
)

log = logging.getLogger(__name__)

# Labels the classifier may return, mapped onto the graph ontology. ``belongs_to``
# is structural (skill -> category) and is not induced from skill-to-skill text.
_LABEL_TO_EDGE: Dict[str, EdgeType] = {
    "depends_on": EdgeType.DEPENDS_ON,
    "composes_with": EdgeType.COMPOSES_WITH,
    "similar_to": EdgeType.SIMILAR_TO,
    "conflicts_with": EdgeType.CONFLICTS_WITH,
}

_NEEDS_PROMPT = (
    "You are building a dependency-aware skill graph.\n"
    "For the skill below, list IN ONE SHORT SENTENCE the concrete capabilities "
    "or prerequisites an agent would need in order to use this skill well "
    "(e.g. what it consumes, what must already exist).\n"
    "Reply with the sentence only — no preamble, no bullet points.\n\n"
    "Skill: {name}\n"
    "Description: {description}\n"
    "Tags: {tags}\n"
)

_CLASSIFY_PROMPT = (
    "You are labelling the relationship between two skills in a skill graph.\n\n"
    "Skill A: {a_name}\n  {a_desc}\n\n"
    "Skill B: {b_name}\n  {b_desc}\n\n"
    "Choose exactly ONE label describing how A relates to B:\n"
    '- "depends_on": A requires B as a prerequisite (A is hard to do without B)\n'
    '- "composes_with": A and B are commonly used together; neither requires the other\n'
    '- "similar_to": A and B are near-duplicates / interchangeable\n'
    '- "conflicts_with": A and B must not be used together\n'
    '- "none": no meaningful relationship\n\n'
    'Reply with JSON only: {{"type": "<label>"}}\n'
)


@dataclass
class InductionReport:
    """Summary of one LLM edge-induction pass."""

    needs_generated: int = 0
    candidate_pairs: int = 0
    classified_pairs: int = 0
    edges_added: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "needs_generated": self.needs_generated,
            "candidate_pairs": self.candidate_pairs,
            "classified_pairs": self.classified_pairs,
            "edges_added": self.edges_added,
            "by_type": dict(self.by_type),
        }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of an LLM reply (fenced or bare)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*?\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None


class LLMEdgeInducer:
    """Induce typed skill-to-skill edges with the SkillDAG two-view recipe.

    Args:
        llm: Any object exposing ``generate(prompt=...) -> str`` (a chat LLM).
        embedder: Embedder for the two views. Defaults to the offline
            :class:`HashingEmbedder` so the pass is runnable without extra deps.
        threshold: Min cosine similarity for a pair to become a *candidate*.
            The candidate step is only a cheap recall filter — the LLM makes the
            final typed decision.
        candidate_top_k: Max candidate partners proposed per skill.
        max_pairs: Hard cap on LLM classification calls (cost guard).
        needs_max_chars: Truncate generated ``needs`` text before embedding.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Optional[Embedder] = None,
        threshold: float = 0.30,
        candidate_top_k: int = 8,
        max_pairs: int = 200,
        needs_max_chars: int = 400,
    ):
        if llm is None or not hasattr(llm, "generate"):
            raise ValueError("LLMEdgeInducer requires an object with a generate(prompt=...) method")
        self.llm = llm
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.candidate_top_k = max(1, candidate_top_k)
        self.max_pairs = max(1, max_pairs)
        self.needs_max_chars = max(1, needs_max_chars)

    # ── public API ───────────────────────────────────────────
    def induce(
        self,
        graph: SkillGraph,
        skills: Optional[Sequence[SkillNode]] = None,
        new_skill_ids: Optional[Set[str]] = None,
    ) -> InductionReport:
        """Generate ``e_needs``, propose candidates, classify, and add typed edges.

        When ``new_skill_ids`` is given this runs the **incremental** path:

          - fresh LLM ``needs`` views are generated only for those skills; the
            rest reuse the ``metadata['needs']`` text persisted by the previous
            pass, so an incremental build costs O(new skills) LLM calls instead
            of O(all skills);
          - candidate pairs are proposed only *from* the new skills (against the
            whole pool), so old-old pairs already classified last time are left
            untouched.

        With ``new_skill_ids=None`` every skill is treated as new (full build).
        """
        report = InductionReport()
        skills = list(skills if skills is not None else graph.skills())
        if len(skills) < 2:
            return report
        new_ids: Set[str] = set(new_skill_ids) if new_skill_ids is not None else {n.id for n in skills}
        if not new_ids:
            return report

        needs_vecs = self._compute_needs(skills, new_ids, report)
        pairs = self._candidate_pairs(skills, needs_vecs, new_ids, report)
        report.candidate_pairs = len(pairs)

        for a, b, score in pairs:
            label = self._classify(a, b)
            report.classified_pairs += 1
            edge_type = _LABEL_TO_EDGE.get(label)
            if edge_type is None:
                continue
            src, tgt = a.id, b.id
            if edge_type == EdgeType.DEPENDS_ON:
                src, tgt = self._orient_dependency(a, b, needs_vecs)
            if graph.add_edge(
                SkillEdge(
                    source=src,
                    target=tgt,
                    edge_type=edge_type,
                    weight=round(score, 4),
                    provenance="llm",
                )
            ):
                report.edges_added += 1
                report.by_type[edge_type.value] = report.by_type.get(edge_type.value, 0) + 1

        log.info(
            "llm edge induction: %d skills, %d needs, %d candidates, %d classified, %d edges %s",
            len(skills),
            report.needs_generated,
            report.candidate_pairs,
            report.classified_pairs,
            report.edges_added,
            report.by_type,
        )
        return report

    # ── internals ────────────────────────────────────────────
    def _generate(self, prompt: str) -> str:
        try:
            return self.llm.generate(prompt=prompt) or ""
        except TypeError:
            # Some clients accept a positional prompt.
            return self.llm.generate(prompt) or ""

    def _compute_needs(
        self,
        skills: Sequence[SkillNode],
        new_ids: Set[str],
        report: InductionReport,
    ) -> Dict[str, List[float]]:
        """View 2: embed "what this skill requires".

        New skills get a fresh LLM sentence; previously-built skills reuse the
        ``metadata['needs']`` text persisted in ``skill_graph.json`` (no LLM
        call), which is what makes incremental builds cheap.
        """
        needs_vecs: Dict[str, List[float]] = {}
        for node in skills:
            if node.id in new_ids:
                prompt = _NEEDS_PROMPT.format(
                    name=node.name,
                    description=node.description or "(none)",
                    tags=", ".join(node.tags) or "(none)",
                )
                text = (self._generate(prompt) or "").strip()[: self.needs_max_chars]
                if not text:
                    # Fall back to the self view so the node still participates.
                    text = node.index_text
                else:
                    report.needs_generated += 1
                node.metadata["needs"] = text
            else:
                text = str(node.metadata.get("needs") or node.index_text)
            needs_vecs[node.id] = self.embedder.embed(text)
        return needs_vecs

    def _candidate_pairs(
        self,
        skills: Sequence[SkillNode],
        needs_vecs: Dict[str, List[float]],
        new_ids: Set[str],
        report: InductionReport,
    ) -> List[Tuple[SkillNode, SkillNode, float]]:
        """Recall candidates by two-view cosine (self-self and needs-to-self).

        Proposals are made only *from* skills in ``new_ids`` (against the whole
        pool), so an incremental pass never re-classifies old-old pairs.
        """
        by_id = {n.id: n for n in skills}

        def pair_score(a: SkillNode, b: SkillNode) -> float:
            return max(
                cosine(a.embedding, b.embedding),
                cosine(needs_vecs.get(a.id), b.embedding),
                cosine(a.embedding, needs_vecs.get(b.id)),
            )

        proposals: Dict[str, List[Tuple[str, float]]] = {}
        for a in skills:
            if a.id not in new_ids:
                continue  # incremental: only new/updated skills propose
            scored: List[Tuple[str, float]] = []
            for b in skills:
                if a.id == b.id:
                    continue
                score = pair_score(a, b)
                if score >= self.threshold:
                    scored.append((b.id, score))
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
            proposals[a.id] = scored[: self.candidate_top_k]

        pairs: List[Tuple[SkillNode, SkillNode, float]] = []
        seen: Set[Tuple[str, str]] = set()
        for a in skills:
            for b_id, score in proposals.get(a.id, []):
                key = (a.id, b_id) if a.id < b_id else (b_id, a.id)
                if key in seen:
                    continue
                seen.add(key)
                b = by_id.get(b_id)
                if b is not None:
                    pairs.append((a, b, score))
                if len(pairs) >= self.max_pairs:
                    log.warning("candidate pairs capped at max_pairs=%d", self.max_pairs)
                    return pairs
        return pairs

    def _classify(self, a: SkillNode, b: SkillNode) -> str:
        prompt = _CLASSIFY_PROMPT.format(
            a_name=a.name,
            a_desc=a.description or "(none)",
            b_name=b.name,
            b_desc=b.description or "(none)",
        )
        data = _extract_json(self._generate(prompt))
        if not isinstance(data, dict):
            return "none"
        label = str(data.get("type", "")).strip().lower()
        return label if label in _LABEL_TO_EDGE or label == "none" else "none"

    @staticmethod
    def _orient_dependency(
        a: SkillNode,
        b: SkillNode,
        needs_vecs: Dict[str, List[float]],
    ) -> Tuple[str, str]:
        """Point ``depends_on`` from the *needer* to the *provider*.

        If A needs B, A's ``e_needs`` should sit closer to B's ``e_self`` than the
        reverse; ties keep the (A -> B) order.
        """
        a_needs_b = cosine(needs_vecs.get(a.id), b.embedding)
        b_needs_a = cosine(needs_vecs.get(b.id), a.embedding)
        if b_needs_a > a_needs_b:
            return b.id, a.id
        return a.id, b.id