"""Multi-recall + fusion schema retrieval (P0: 检索层重建).

KgSchemaLinker is a single *lexical* pass: 2-gram scoring over
name/comment/definition/formula. That misses Chinese semantic equivalents
("客单价" vs "平均每单成交金额") when the wording differs. This module turns
schema retrieval into a production-shaped **multi-recall pipeline**:

    question
      ├─ graph structure : entity link (exact/alias/prefix on names) + 1-hop
      │                   subgraph expansion (table -> fields/metrics)
      ├─ fulltext        : HugeGraph SEARCH index (Text.contains) over
      │                   comment/definition/formula; in-memory fallback
      ├─ lexical         : existing score_vertex ranking (KgSchemaLinker)
      └─ vector          : SLOT - plugged in when an embedding endpoint exists
    → RRF fusion (reciprocal rank) → top-k schema context

``KgMultiSchemaLinker`` keeps the exact ``SchemaContext`` contract so the
NL2SQL pipeline / voter / authority consume it unchanged; the single-pass
``KgSchemaLinker`` stays as one of the recall paths (and as the default for
callers that want the old behaviour).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hugegraph_llm.operators.graph_op.kg_rule_engine import GraphData
from hugegraph_llm.operators.graph_op.kg_schema_linker import (
    KgSchemaLinker,
    SchemaContext,
)
from hugegraph_llm.operators.graph_op.kg_query_understanding import (
    DEFAULT_INTENT_BOOST,
)

logger = logging.getLogger(__name__)

NODE_LABELS = ("Table", "Field", "Metric")


@dataclass
class RetrievedVertex:
    """One vertex recalled by a single retriever (pre-fusion)."""

    label: str  # Table | Field | Metric
    name: str
    score: float = 0.0
    source: str = "retriever"


@dataclass
class MultiRecallConfig:
    """Budgets + fusion knobs for the multi-recall linker."""

    max_tables: int = 5
    max_metrics: int = 5
    max_fields_per_table: int = 12
    max_evidence: int = 12
    min_score: float = 0.3
    fusion_k: int = 60
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "graph": 1.0,
            "fulltext": 0.5,  # recall backstop: HugeGraph SEARCH is
                              # character-level for Chinese, so it over-recalls
                              # vertices sharing a single char; low weight keeps
                              # it from dominating the fusion ranking
            "lexical": 1.0,
            "vector": 1.0,  # inactive until an embedding retriever is attached
        }
    )
    # entity-importance re-ranking (generalized from the 货拉拉 metadata-GraphRAG
    # 表/字段权重公式): fused score *= (1 + importance_weight * importance).
    # 0 disables the re-rank (default, keeps prior behaviour).
    importance_weight: float = 0.0
    retriever_top_k: int = 20
    # fusion strategy: "score" (weighted score sum; the default — each path's
    # score is on the same scale: name exact 3.0/2.0, substring 1.0, fulltext
    # boolean 1.0) or "rrf" (reciprocal-rank; kept for compatibility)
    fusion: str = "score"
    # question-intent type weighting: when >0, the fused score of the label
    # the user asked FOR (from QueryIntent.intent_type) is boosted by
    # (1 + intent_weight), so "在哪个表" surfaces tables above metrics.
    # 0 disables (default, keeps prior behaviour).
    intent_weight: float = 0.0


class SchemaRetriever(ABC):
    """One recall path. Returns ranked vertex candidates (label, name)."""

    source: str = "retriever"

    @abstractmethod
    def retrieve(
        self,
        question: str,
        data: GraphData,
        client: Optional[Any] = None,
        terms: Optional[Sequence[str]] = None,
    ) -> List[RetrievedVertex]:
        """Retrieve candidates.

        ``terms`` optionally overrides the internally extracted terms (the
        query-understanding stage passes expanded terms this way); None falls
        back to the retriever's own extraction.
        """
        ...


class GraphStructureRetriever(SchemaRetriever):
    """Entity link on names (exact / alias-expanded / prefix) + 1-hop expansion.

    Aliases come from the linker's jargon map (synonyms), so "客单价" mapped to
    a canonical metric name contributes here without any embedding.
    """

    source = "graph"
    min_term_len: int = 3

    def __init__(self, linker: Optional[KgSchemaLinker] = None) -> None:
        self._linker = linker or KgSchemaLinker()

    def _alias_terms(self, question: str, terms: Sequence[str]) -> List[str]:
        out = [t for t in terms]
        # aliases are full terms ("客单价"), terms are 2-grams — match the
        # alias against the raw question instead of the token stream
        for alias, canon in self._linker._synonyms.items():
            if alias and alias in question and canon not in out:
                out.append(canon)
        return out

    def retrieve(self, question, data, client=None, terms=None) -> List[RetrievedVertex]:
        base_terms = list(terms) if terms is not None else self._linker.extract_terms(question)
        # generic short tokens ("ID") hit every *_id field name with the same
        # score and add no discriminative signal; drop them like fulltext does
        base_terms = [t for t in base_terms if len(t) >= self.min_term_len]
        terms = self._alias_terms(question, base_terms)
        hits: List[RetrievedVertex] = []
        # 1) name-level entity links (score by match strength)
        for label in NODE_LABELS:
            for v in data.get("vertices", {}).get(label, []):
                name = v.get("name")
                if not name:
                    continue
                low = str(name).lower()
                best = 0.0
                for t in terms:
                    tl = t.lower()
                    if low == tl:
                        best = max(best, 3.0)
                    elif tl in low or low in tl:
                        best = max(best, 2.0)
                if best >= 1.0:
                    hits.append(RetrievedVertex(label, name, best, self.source))
        # 2) 1-hop subgraph: fields of hit tables, table/field sources of metrics
        vertices = data.get("vertices", {})
        edges = data.get("edges", {})
        hit_tables = {h.name for h in hits if h.label == "Table"}
        hit_metrics = {h.name for h in hits if h.label == "Metric"}
        field_by_name = {f.get("name"): f for f in vertices.get("Field", []) if f.get("name")}
        table_names = {t.get("name") for t in vertices.get("Table", []) if t.get("name")}
        for src, dst in edges.get("hasColumn", []):
            if src in hit_tables and dst in field_by_name:
                hits.append(RetrievedVertex("Field", dst, 1.5, self.source))
        for src, dst in edges.get("computedFrom", []):
            if src in hit_metrics and dst in table_names:
                hits.append(RetrievedVertex("Table", dst, 1.5, self.source))
        for src, dst in edges.get("computedFromField", []):
            if src in hit_metrics and dst in field_by_name:
                hits.append(RetrievedVertex("Field", dst, 1.5, self.source))
        return hits


class FulltextRetriever(SchemaRetriever):
    """SEARCH-index recall via Text.contains (HugeGraph) or in-memory fallback.

    With a client it runs real ``g.V().has(label, prop, Text.contains(term))``
    queries (the SEARCH indexes are built); without one it degrades to a
    substring scan over the same fields so unit tests stay offline.

    Term guards (learned on the live server):
    - ``len(term) >= min_term_len``: HugeGraph SEARCH matches substrings
      aggressively, so 2-gram terms like "表在" hit every comment containing
      "表" (the 货拉拉 badcase '不存在也答').
    - ``len(term) <= max_term_len`` (CJK only): HugeGraph's SEARCH tokenizer
      is character-level for Chinese, so a long sentence fragment like
      "完全无关的随机词汇" matches any definition sharing a single character
      ("完"/"的"). Whole-sentence terms must not be sent to Text.contains.
    """

    source = "fulltext"
    min_term_len: int = 3
    # CJK sentence fragments longer than this are NOT sent to Text.contains:
    # HugeGraph's SEARCH tokenizer is character-level for Chinese, so a long
    # fragment like "完全无关的随机词汇" (9 chars) matches any definition
    # sharing a single character. Heuristic bound: noun phrases up to 8 chars
    # ("平均每单成交金额") still match sensibly; longer sentences are noise.
    max_term_len: int = 8
    # character-overlap guard for SEARCH hits: HugeGraph SEARCH is
    # character-level for Chinese, so Text.contains(term) returns every
    # vertex sharing even one char ("订单总额" hits "订单表" via 订/单).
    # A hit is kept only when the term shares >= min_overlap_ratio of its
    # characters with the matched text (4-char term, 2 shared = 0.5 < 0.6).
    min_overlap_ratio: float = 0.6

    _TEXT_FIELDS: Tuple[Tuple[str, str], ...] = (
        ("Table", "comment"),
        ("Field", "comment"),
        ("Metric", "definition"),
        ("Metric", "formula"),
    )

    def __init__(self, linker: Optional[KgSchemaLinker] = None) -> None:
        self._linker = linker or KgSchemaLinker()

    def retrieve(self, question, data, client=None, terms=None) -> List[RetrievedVertex]:
        raw = list(terms) if terms is not None else self._linker.extract_terms(question)
        terms = [
            t for t in raw
            if len(t) >= self.min_term_len and self._term_ok(t)
        ]
        if not terms:
            return []
        if client is not None:
            return self._retrieve_live(client, terms)
        return self._retrieve_memory(data, terms)

    @staticmethod
    def _term_ok(term: str) -> bool:
        """Reject long CJK sentence fragments (character-level SEARCH noise)."""
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in term)
        if not has_cjk:
            return True  # latin tokens are word-indexed, safe at any length
        return len(term) <= FulltextRetriever.max_term_len

    @classmethod
    def _overlap_ratio(cls, term: str, text: str) -> float:
        """Fraction of the term's characters shared with the text (0..1)."""
        if not term or not text:
            return 0.0
        term_chars = set(term)
        text_chars = set(text)
        shared = sum(1 for ch in term_chars if ch in text_chars)
        return shared / len(term_chars)

    def _keep(self, term: str, text: str) -> bool:
        """A SEARCH hit is kept only when the overlap is strong enough."""
        return self._overlap_ratio(term, text) >= self.min_overlap_ratio

    def _retrieve_live(self, client: Any, terms: Sequence[str]) -> List[RetrievedVertex]:
        out: List[RetrievedVertex] = []
        for label, prop in self._TEXT_FIELDS:
            for term in terms:
                safe = term.replace("'", "\\'")
                try:
                    resp = client.gremlin().exec(
                        f"g.V().hasLabel('{label}').has('{prop}', "
                        f"Text.contains('{safe}')).elementMap()"
                    )
                    rows = resp.get("data") if isinstance(resp, dict) else (resp or [])
                    for row in rows or []:
                        name = row.get("name")
                        if name and self._keep(term, str(row.get(prop) or "")):
                            out.append(RetrievedVertex(label, name, 1.0, self.source))
                except Exception as exc:  # noqa: BLE001 - index/term miss
                    logger.debug("fulltext %s.%s %s failed: %s", label, prop, term, exc)
        return out

    def _retrieve_memory(self, data: GraphData, terms: Sequence[str]) -> List[RetrievedVertex]:
        out: List[RetrievedVertex] = []
        for label, prop in self._TEXT_FIELDS:
            for v in data.get("vertices", {}).get(label, []):
                text = str(v.get(prop) or "")
                if any(self._keep(t, text) for t in terms):
                    name = v.get("name")
                    if name:
                        out.append(RetrievedVertex(label, name, 1.0, self.source))
        return out


class LexicalRetriever(SchemaRetriever):
    """The existing single-pass ranking, exposed as one recall path."""

    source = "lexical"

    def __init__(self, linker: Optional[KgSchemaLinker] = None, top_k: int = 20) -> None:
        self._linker = linker or KgSchemaLinker()
        self._top_k = top_k

    def retrieve(self, question, data, client=None, terms=None) -> List[RetrievedVertex]:
        terms = list(terms) if terms is not None else self._linker.extract_terms(question)
        scored: List[Tuple[float, str, str]] = []
        for label in NODE_LABELS:
            for score, v in self._linker._rank(data.get("vertices", {}).get(label, []), label, terms):
                name = v.get("name")
                if name:  # pragma: no branch - _rank already filters nameless
                    scored.append((score, label, name))
        # global score ordering across labels (a Metric scoring 1.2 must rank
        # above a Table scoring 1.0, not after all tables/fields)
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            RetrievedVertex(label, name, score, self.source)
            for score, label, name in scored[: self._top_k]
        ]


def rrf_fuse(
    lists: Sequence[Sequence[RetrievedVertex]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[RetrievedVertex]:
    """Reciprocal-rank fusion across recall paths.

    Each path contributes ``weight / (k + rank)`` to a vertex key; ties merge
    into one vertex (source label keeps the highest-scoring path).
    """
    weights = weights or {}
    acc: Dict[Tuple[str, str], float] = defaultdict(float)
    best: Dict[Tuple[str, str], RetrievedVertex] = {}
    for path in lists:
        w = float(weights.get(path[0].source, 1.0)) if path else 1.0
        for rank, rv in enumerate(path):
            key = (rv.label, rv.name)
            acc[key] += w / (k + rank + 1)
            if key not in best or rv.score > best[key].score:
                best[key] = rv
    ordered = sorted(acc.items(), key=lambda kv: -kv[1])
    return [replace(best[key], score=score) for key, score in ordered]


def score_fuse(
    lists: Sequence[Sequence[RetrievedVertex]],
    weights: Optional[Dict[str, float]] = None,
) -> List[RetrievedVertex]:
    """Weighted score-sum fusion across recall paths.

    Unlike RRF (which flattens rank differences to ~1/(k+rank) so tiny noise
    decides ties), score fusion sums each path's semantic score — all paths
    share the same scale (name exact 3.0 / substring 2.0 / 1.0, fulltext
    boolean 1.0), so a metric scoring 1.2 in lexical + 1.0 in fulltext
    (2.2) beats a table scoring 1.0 + 1.0 (2.0) deterministically.
    """
    weights = weights or {}
    acc: Dict[Tuple[str, str], float] = defaultdict(float)
    best: Dict[Tuple[str, str], RetrievedVertex] = {}
    for path in lists:
        w = float(weights.get(path[0].source, 1.0)) if path else 1.0
        for rv in path:
            key = (rv.label, rv.name)
            acc[key] += w * rv.score
            if key not in best or rv.score > best[key].score:
                best[key] = rv
    ordered = sorted(acc.items(), key=lambda kv: -kv[1])
    return [replace(best[key], score=score) for key, score in ordered]


def compute_entity_importance(data: GraphData) -> Dict[str, Dict[str, float]]:
    """Per-vertex importance in [0, 1] for re-ranking fused results.

    Generalized from the 货拉拉 metadata-GraphRAG weighting: tables are more
    important when referenced by metrics and rich in fields; metrics when
    marked authoritative / high-priority; fields inherit their table's weight.
    """
    edges = data.get("edges", {})
    vertices = data.get("vertices", {})
    metric_refs: Dict[str, int] = defaultdict(int)
    for _src, dst in edges.get("computedFrom", []):
        metric_refs[dst] += 1
    table_fields: Dict[str, int] = defaultdict(int)
    for src, _dst in edges.get("hasColumn", []):
        table_fields[src] += 1

    table_imp: Dict[str, float] = {}
    for t in vertices.get("Table", []):
        name = t.get("name")
        if not name:
            continue
        ref = min(metric_refs.get(name, 0), 3) / 3.0
        fields = min(table_fields.get(name, 0), 10) / 10.0
        auth = 1.0 if str(t.get("authoritative") or "").lower() == "true" else 0.0
        table_imp[name] = 0.6 * ref + 0.3 * fields + 0.1 * auth

    metric_imp: Dict[str, float] = {}
    for m in vertices.get("Metric", []):
        name = m.get("name")
        if not name:
            continue
        auth = 1.0 if str(m.get("authoritative") or "").lower() == "true" else 0.0
        try:
            prio = min(max(int(m.get("priority") or 0), 0), 100) / 100.0
        except (TypeError, ValueError):  # pragma: no cover - dirty data guard
            prio = 0.0
        metric_imp[name] = 0.7 * auth + 0.3 * prio

    field_imp: Dict[str, float] = {}
    for f in vertices.get("Field", []):
        name, owner = f.get("name"), f.get("table")
        if name and owner:
            field_imp[name] = table_imp.get(owner, 0.0)
    return {"Table": table_imp, "Metric": metric_imp, "Field": field_imp}


class KgMultiSchemaLinker:
    """Multi-recall schema linker: fuse paths -> SchemaContext (compatible).

    Usage::

        linker = KgMultiSchemaLinker(client=hg_client)   # fulltext goes live
        ctx = linker.link("客单价是多少")                    # same contract as before
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
        config: Optional[MultiRecallConfig] = None,
        retrievers: Optional[List[SchemaRetriever]] = None,
        query_understanding: Optional[Any] = None,
    ) -> None:
        self._client = client
        self._config = config or MultiRecallConfig()
        base = KgSchemaLinker(client=client, synonyms=synonyms)
        self._base = base
        # default three paths (no embedding yet); vector slot is a no-op list
        self._retrievers: List[SchemaRetriever] = retrievers or [
            GraphStructureRetriever(base),
            FulltextRetriever(base),
            LexicalRetriever(base, top_k=self._config.retriever_top_k),
        ]
        self._vector_retriever: Optional[SchemaRetriever] = None
        # optional query-understanding stage: dual-level keywords + synonym
        # expansion feeding the expanded terms into every recall path
        self._understanding = query_understanding

    def attach_vector_retriever(self, retriever: SchemaRetriever) -> None:
        """Slot for the embedding path; call once an embedding endpoint exists."""
        self._vector_retriever = retriever

    def link(self, question: str, data: Optional[GraphData] = None) -> SchemaContext:
        data = data if data is not None else self._base.load_graph()
        terms: Optional[List[str]] = None
        intent: Dict[str, Any] = {}
        if self._understanding is not None:
            qi = self._understanding.understand(question)
            terms = qi.expanded_terms
            intent = qi.to_dict()
        paths = [r.retrieve(question, data, client=self._client, terms=terms)
                 for r in self._retrievers]
        if self._vector_retriever is not None:
            paths.append(self._vector_retriever.retrieve(question, data, client=self._client,
                                                         terms=terms))
        if self._config.fusion == "rrf":
            fused = rrf_fuse(paths, k=self._config.fusion_k, weights=self._config.weights)
        else:
            fused = score_fuse(paths, weights=self._config.weights)
        # NOTE: RRF scores are 1/(k+rank)-scaled (~0.016), so no absolute
        # min_score filter here; each path filters its own results and the
        # assembly below truncates to the budgets.
        if self._config.importance_weight > 0:
            importance = compute_entity_importance(data)
            for rv in fused:
                rv.score *= 1.0 + self._config.importance_weight * importance.get(
                    rv.label, {}
                ).get(rv.name, 0.0)
            fused.sort(key=lambda rv: -rv.score)
        if self._config.intent_weight > 0 and intent:
            boost = DEFAULT_INTENT_BOOST.get(intent.get("intent_type"), {})
            if boost:
                for rv in fused:
                    if rv.label in boost:
                        rv.score *= 1.0 + self._config.intent_weight * boost[rv.label]
                fused.sort(key=lambda rv: -rv.score)

        # assemble the SchemaContext from the fused candidates, reusing the
        # single-pass linker for relations/evidence construction
        vertices = data.get("vertices", {})
        by_name = {
            label: {v.get("name"): v for v in vertices.get(label, []) if v.get("name")}
            for label in NODE_LABELS
        }
        tables = [
            by_name["Table"][rv.name] for rv in fused
            if rv.label == "Table" and rv.name in by_name["Table"]
        ][: self._config.max_tables]
        metrics = [
            by_name["Metric"][rv.name] for rv in fused
            if rv.label == "Metric" and rv.name in by_name["Metric"]
        ][: self._config.max_metrics]
        hit_fields = [
            by_name["Field"][rv.name] for rv in fused
            if rv.label == "Field" and rv.name in by_name["Field"]
        ]
        # fields that recall hit imply their owner table: a question asking
        # for "订单金额和支付金额" must surface order/payment tables too
        # (the platform needs the table name to build SQL)
        field_owner = {}
        for src, dst in data.get("edges", {}).get("hasColumn", []):
            field_owner[dst] = src
        table_names = {t.get("name") for t in tables}
        for f in hit_fields:
            owner = f.get("table") or field_owner.get(f.get("name"))
            if owner and owner in by_name["Table"] and owner not in table_names:
                if len(tables) >= self._config.max_tables:
                    break
                tables.append(by_name["Table"][owner])
                table_names.add(owner)
        base_ctx = self._base.link(question, data=data)
        fields = list(hit_fields)
        # fields the single-pass linker already resolved for the hit tables
        # (via hasColumn / Field.table) but no recall path returned: append
        # them, bounded by the field budget
        for f in base_ctx.fields:
            if len(fields) >= self._config.max_fields_per_table:
                break
            if f not in fields:
                fields.append(f)
        fields = fields[: self._config.max_fields_per_table]

        # fused relevance ranking; owner tables promoted by field hits are
        # inserted right after their first field so the platform sees the
        # table next to the field that implies it
        ranking = [(rv.label, rv.name) for rv in fused]
        promoted = set()
        for f in hit_fields:
            owner = f.get("table") or field_owner.get(f.get("name"))
            if owner and owner in by_name["Table"] and owner in table_names \
                    and (owner, "Table") not in ranking and owner not in promoted:  # pragma: no branch
                for i, (_label, rname) in enumerate(ranking):
                    if _label == "Field" and field_owner.get(rname) == owner:  # pragma: no branch
                        ranking.insert(i + 1, ("Table", owner))
                        promoted.add(owner)
                        break

        return SchemaContext(
            tables=tables,
            fields=fields,
            metrics=metrics,
            relations=self._base._collect_relations(tables, metrics, data.get("edges", {})),
            evidence=self._base._build_evidence(tables, metrics, fields)[: self._config.max_evidence],
            matched_terms=base_ctx.matched_terms,
            query_intent=intent,
            ranking=ranking,
        )
