# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Retrieval: recall -> expand -> prune -> budget.

This is the component that turns a semantic layer from "a graph you can
query" into "the right slice of schema, in the prompt, within budget".

The pipeline deliberately mirrors the shape neocarta validated, with two
additions it does not have:

* **Connectivity is guaranteed.** Tables are expanded to a connected
  subgraph (Steiner tree when an engine is supplied, BFS otherwise), so the
  model is never handed two tables that cannot be joined -- the single most
  common cause of hallucinated JOIN clauses.
* **The output is budgeted.** Context is degraded to fit a token budget
  instead of being truncated mid-table; see :mod:`semantic_layer.budget`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hugegraph_llm.semantic_layer.bm25 import BM25
from hugegraph_llm.semantic_layer.budget import BudgetResult, fit_context
from hugegraph_llm.semantic_layer.context import (
    ColumnContext,
    TableContext,
    estimate_tokens,
)
from hugegraph_llm.semantic_layer.fusion import rrf_fuse
from hugegraph_llm.semantic_layer.readers import SemanticGraphReader, SemanticProjection

__all__ = [
    "RetrievalConfig",
    "RetrievalResult",
    "SemanticLayerRetriever",
]


@dataclass
class RetrievalConfig:
    """Tunables for one retrieval call."""

    #: Tables recalled directly before expansion.
    top_k: int = 8
    #: Graph expansion hops from each seed.
    hops: int = 2
    #: Score multiplier per hop: a table two joins away is weaker evidence
    #: than one directly recalled.
    hop_decay: float = 0.5
    #: Hard cap on tables considered, after expansion.
    max_tables: int = 12
    #: Prompt budget for schema context.
    max_tokens: int = 4000
    #: Tokens held back for instructions and the question.
    reserve_tokens: int = 500
    #: Per-source RRF weights: [vector, business-term, bm25].
    fusion_weights: Tuple[float, ...] = (1.0, 1.5, 1.0)


@dataclass
class RetrievalResult:
    """What the caller needs to build a prompt, plus how it was produced."""

    question: str
    budget: BudgetResult
    #: Tables recalled directly, in fused-rank order.
    seeds: List[str] = field(default_factory=list)
    #: Tables added by graph expansion.
    expanded: List[str] = field(default_factory=list)
    #: Tables removed because they could not be joined to the rest.
    disconnected: List[str] = field(default_factory=list)
    #: Which recall paths fired; empty means retrieval found nothing.
    sources_used: List[str] = field(default_factory=list)

    @property
    def tables(self) -> List[str]:
        return [c.table_name for c in self.budget.contexts]

    @property
    def ok(self) -> bool:
        return bool(self.budget.contexts) and self.budget.ok

    def render(self) -> str:
        return self.budget.render()

    def to_dicts(self) -> List[dict]:
        return self.budget.to_dicts()

    def summary(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "seeds": self.seeds,
            "expanded": self.expanded,
            "disconnected": self.disconnected,
            "tables": self.tables,
            "dropped_by_budget": self.budget.dropped,
            "levels": self.budget.levels,
            "used_tokens": self.budget.used_tokens,
            "budget": self.budget.budget,
            "sources_used": self.sources_used,
        }


class SemanticLayerRetriever:
    """Retrieves a budgeted, connected slice of the semantic layer."""

    def __init__(
        self,
        reader: SemanticGraphReader,
        *,
        vector_store: Any = None,
        embed: Optional[Callable[[str], List[float]]] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.reader = reader
        self.vector_store = vector_store
        self.embed = embed
        self.config = config or RetrievalConfig()
        self._bm25: Optional[BM25] = None
        self._documents: Dict[str, str] = {}

    # -- index -------------------------------------------------------------

    @property
    def bm25(self) -> BM25:
        """Lazily build the lexical index from the projection."""
        if self._bm25 is None:
            self._bm25 = BM25().fit(self._corpus().items())
        return self._bm25

    def _corpus(self) -> Dict[str, str]:
        """One document per table: identity plus every column it owns.

        Column names and comments are folded into the table document so that
        a query naming a column ("which orders have an amount over 100?")
        recalls the table that owns it.
        """
        if self._documents:
            return self._documents
        proj = self.reader.projection()
        docs: Dict[str, str] = {}
        for name, table in proj.tables.items():
            parts = [name, table.comment]
            for col in proj.columns_of(name):
                parts.append(col.name)
                if col.comment:
                    parts.append(col.comment)
            for term in proj.table_terms.get(name, []):
                parts.append(term)
                term_row = proj.terms.get(term)
                if term_row and term_row.description:
                    parts.append(term_row.description)
            docs[name] = " ".join(p for p in parts if p)
        self._documents = docs
        return docs

    def invalidate(self) -> None:
        """Drop cached indexes; call after the graph is re-ingested."""
        self._bm25 = None
        self._documents = {}

    # -- recall ------------------------------------------------------------

    def _recall_vector(self, question: str) -> List[Tuple[str, float]]:
        if not (self.vector_store and self.embed):
            return []
        try:
            hits = self.vector_store.search(self.embed(question), self.config.top_k * 3)
        except Exception:  # noqa: BLE001 - vector store is best-effort
            return []
        proj = self.reader.projection()
        out: List[Tuple[str, float]] = []
        for node_id, score in hits:
            table = self._node_to_table(node_id, proj)
            if table and not any(t == table for t, _ in out):
                out.append((table, float(score)))
        return out

    @staticmethod
    def _node_to_table(node_id: str, proj: SemanticProjection) -> Optional[str]:
        """Map a vector-store id to a table name.

        The store may hold table, column or term entries (it is populated
        per node type), so each is resolved back to a table.
        """
        raw = str(node_id)
        if raw.startswith("table:"):
            name = raw.split(":", 1)[1]
            return name if name in proj.tables else None
        if raw.startswith("column:"):
            qualified = raw.split(":", 1)[1]
            col = proj.columns.get(qualified)
            return col.table if col else None
        if raw.startswith("term:"):
            term = raw.split(":", 1)[1]
            for column in proj.term_columns.get(term, []):
                return column.split(".", 1)[0]
            for table, terms in proj.table_terms.items():
                if term in terms:
                    return table
        return None

    def _recall_terms(self, question: str) -> List[Tuple[str, float]]:
        """Exact business-term recall.

        This is the path that makes the semantic layer worth having: a
        business term ("月活", "MAU") is matched by name or alias and mapped
        straight to the tables and columns that define it, with no semantic
        similarity guesswork.
        """
        proj = self.reader.projection()
        if not proj.terms:
            return []
        lowered = question.lower()
        out: List[Tuple[str, float]] = []
        scored: Dict[str, float] = {}
        for term, row in proj.terms.items():
            names = [term] + list(row.aliases)
            for candidate in names:
                probe = str(candidate).strip().lower()
                if not probe:
                    continue
                if probe in lowered:
                    # Prefer the longest match: "MAU" inside "mauve" is noise,
                    # "monthly active users" is signal.
                    scored[term] = max(scored.get(term, 0.0), float(len(probe)))
        for term, weight in sorted(scored.items(), key=lambda kv: -kv[1]):
            tables = set()
            for column in proj.term_columns.get(term, []):
                tables.add(column.split(".", 1)[0])
            for table, terms in proj.table_terms.items():
                if term in terms:
                    tables.add(table)
            # Multi-hop: term -> metric -> expression column -> table.
            # This is what resolves an acronym with no direct table binding
            # ("ARR") to the table that actually computes it.
            for metric in proj.term_metrics.get(term, []):
                for column in proj.metric_columns.get(metric, []):
                    tables.add(column.split(".", 1)[0])
            for table in sorted(tables):
                if not any(t == table for t, _ in out):
                    out.append((table, weight))
        return out

    def _recall_bm25(self, question: str) -> List[Tuple[str, float]]:
        if not self._corpus():
            return []
        try:
            return self.bm25.search(question, self.config.top_k * 3)
        except Exception:  # noqa: BLE001
            return []

    # -- expansion ---------------------------------------------------------

    def _expand(
        self,
        proj: SemanticProjection,
        seeds: Sequence[Tuple[str, float]],
        cfg: RetrievalConfig,
    ) -> Dict[str, Tuple[float, List[str]]]:
        """BFS outward from seeds over join-capable edges.

        Returns ``table -> (score, reasons)``. Tag edges are excluded by
        ``tables_adjacent`` -- they annotate, they do not join.

        ``cfg`` is passed explicitly rather than read from ``self.config``:
        a caller overriding hops for one call must not silently get the
        instance default.
        """
        seed_scores = {name: score for name, score in seeds}
        found: Dict[str, Tuple[float, List[str]]] = {
            name: (score, ["recall"]) for name, score in seeds
        }
        frontier = list(seed_scores)
        for hop in range(1, cfg.hops + 1):
            next_frontier: List[str] = []
            for current in frontier:
                base = seed_scores.get(current)
                if base is None:
                    base = found[current][0] if current in found else 0.0
                for neighbour, reasons in proj.tables_adjacent(current).items():
                    score = base * (cfg.hop_decay ** hop)
                    if neighbour in found:
                        prev_score, prev_reasons = found[neighbour]
                        found[neighbour] = (
                            max(prev_score, score),
                            sorted(set(prev_reasons + reasons)),
                        )
                        continue
                    found[neighbour] = (score, sorted(set(reasons)))
                    next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return found

    def _ensure_connected(
        self, proj: SemanticProjection, tables: Sequence[str]
    ) -> Tuple[List[str], List[str]]:
        """Keep only the tables reachable from the strongest seed.

        ``tables`` must be ordered strongest-first; the first entry is the
        root, because the retrieved slice is anchored on the best recall hit
        rather than on an arbitrary member of the candidate set.

        Returns ``(connected, disconnected)``. A table that cannot be joined
        to the rest is worse than absent: it invites the model to invent a
        join, so it is removed and reported rather than passed through.

        When a :class:`GraphEngine` is supplied to :meth:`retrieve` its
        Steiner tree is preferred; this BFS is the dependency-free fallback
        and produces the same guarantee (one connected component).
        """
        if len(tables) <= 1:
            return list(tables), []
        ordered = list(tables)
        root = ordered[0]
        seen = {root}
        queue = [root]
        while queue:
            current = queue.pop(0)
            for neighbour in proj.tables_adjacent(current):
                if neighbour in ordered and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        connected = [t for t in ordered if t in seen]
        disconnected = [t for t in ordered if t not in seen]
        return connected, disconnected

    # -- assembly ----------------------------------------------------------

    def _build_context(
        self,
        proj: SemanticProjection,
        table: str,
        score: float,
        reasons: List[str],
        is_seed: bool,
    ) -> TableContext:
        row = proj.tables[table]
        columns: List[ColumnContext] = []
        for col in sorted(proj.columns_of(table), key=lambda c: c.name):
            key_type = "primary" if col.is_primary_key else (
                "foreign" if col.is_foreign_key else ""
            )
            columns.append(
                ColumnContext(
                    name=col.name,
                    data_type=col.data_type,
                    comment=col.comment,
                    key_type=key_type,
                    references=list(proj.references.get(col.qualified, [])),
                    sample_values=list(col.sample_values),
                    is_time_dimension=col.is_time_dimension,
                    confidence=col.confidence,
                )
            )
        return TableContext(
            table_name=table,
            table_description=row.comment,
            database_name=row.database,
            schema_name=row.schema,
            columns=columns,
            primary_key=[c.name for c in columns if c.key_type == "primary"],
            score=score,
            is_seed=is_seed,
            row_count=row.row_count,
            confidence=row.confidence,
            freshness_ts=row.freshness_ts,
            reasons=reasons,
        )

    # -- entry point -------------------------------------------------------

    def retrieve(
        self,
        question: str,
        *,
        max_tokens: Optional[int] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        """Retrieve a connected, budgeted slice for ``question``."""
        cfg = config or self.config
        proj = self.reader.projection()
        result = RetrievalResult(
            question=question,
            budget=BudgetResult(budget=max_tokens or cfg.max_tokens),
        )
        if proj.is_empty:
            return result

        vector_hits = self._recall_vector(question)
        term_hits = self._recall_terms(question)
        bm25_hits = self._recall_bm25(question)
        result.sources_used = [
            name
            for name, hits in (
                ("vector", vector_hits),
                ("business_term", term_hits),
                ("bm25", bm25_hits),
            )
            if hits
        ]

        fused = rrf_fuse(
            [vector_hits, term_hits, bm25_hits],
            weights=list(cfg.fusion_weights),
        )
        if not fused:
            return result

        seeds = [name for name, _score in fused[: cfg.top_k]]
        result.seeds = seeds

        candidates = self._expand(proj, [(s, dict(fused)[s]) for s in seeds], cfg)
        ordered = sorted(candidates.items(), key=lambda kv: -kv[1][0])
        ordered = ordered[: cfg.max_tables]

        selected = [name for name, _ in ordered]
        connected, disconnected = self._ensure_connected(proj, selected)
        result.disconnected = disconnected
        result.expanded = [t for t in connected if t not in seeds]

        contexts = [
            self._build_context(
                proj,
                name,
                candidates[name][0],
                candidates[name][1],
                is_seed=name in seeds,
            )
            for name in connected
        ]
        result.budget = fit_context(
            contexts,
            max_tokens or cfg.max_tokens,
            reserve_tokens=cfg.reserve_tokens,
        )
        return result


def render_for_prompt(result: RetrievalResult) -> str:
    """Render a retrieval result as prompt-ready text with a token note."""
    body = result.render()
    header = (
        f"Retrieved schema context: {len(result.tables)} tables, "
        f"{estimate_tokens(body)} tokens"
    )
    if result.disconnected:
        header += f"; omitted (not joinable): {', '.join(result.disconnected)}"
    if result.budget.dropped:
        header += f"; omitted (budget): {', '.join(result.budget.dropped)}"
    return f"{header}\n\n{body}"
