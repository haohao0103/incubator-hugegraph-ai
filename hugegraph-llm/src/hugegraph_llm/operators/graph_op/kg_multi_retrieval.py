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
            "fulltext": 1.0,
            "lexical": 1.0,
            "vector": 1.0,  # inactive until an embedding retriever is attached
        }
    )
    # entity-importance re-ranking (generalized from the 货拉拉 metadata-GraphRAG
    # 表/字段权重公式): fused score *= (1 + importance_weight * importance).
    # 0 disables the re-rank (default, keeps prior behaviour).
    importance_weight: float = 0.0
    retriever_top_k: int = 20


class SchemaRetriever(ABC):
    """One recall path. Returns ranked vertex candidates (label, name)."""

    source: str = "retriever"

    @abstractmethod
    def retrieve(
        self,
        question: str,
        data: GraphData,
        client: Optional[Any] = None,
    ) -> List[RetrievedVertex]:
        ...


class GraphStructureRetriever(SchemaRetriever):
    """Entity link on names (exact / alias-expanded / prefix) + 1-hop expansion.

    Aliases come from the linker's jargon map (synonyms), so "客单价" mapped to
    a canonical metric name contributes here without any embedding.
    """

    source = "graph"

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

    def retrieve(self, question, data, client=None) -> List[RetrievedVertex]:
        terms = self._alias_terms(question, self._linker.extract_terms(question))
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

    Only terms with ``len(term) >= min_term_len`` are queried: HugeGraph's
    SEARCH tokenizer matches substrings aggressively, so 2-gram terms like
    "表在" hit every comment containing "表" and cause out-of-scope
    false recalls (the 货拉拉 badcase '不存在也答').
    """

    source = "fulltext"
    min_term_len: int = 3

    _TEXT_FIELDS: Tuple[Tuple[str, str], ...] = (
        ("Table", "comment"),
        ("Field", "comment"),
        ("Metric", "definition"),
        ("Metric", "formula"),
    )

    def __init__(self, linker: Optional[KgSchemaLinker] = None) -> None:
        self._linker = linker or KgSchemaLinker()

    def retrieve(self, question, data, client=None) -> List[RetrievedVertex]:
        terms = [t for t in self._linker.extract_terms(question) if len(t) >= self.min_term_len]
        if not terms:
            return []
        if client is not None:
            return self._retrieve_live(client, terms)
        return self._retrieve_memory(data, terms)

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
                        if name:
                            out.append(RetrievedVertex(label, name, 1.0, self.source))
                except Exception as exc:  # noqa: BLE001 - index/term miss
                    logger.debug("fulltext %s.%s %s failed: %s", label, prop, term, exc)
        return out

    def _retrieve_memory(self, data: GraphData, terms: Sequence[str]) -> List[RetrievedVertex]:
        out: List[RetrievedVertex] = []
        for label, prop in self._TEXT_FIELDS:
            for v in data.get("vertices", {}).get(label, []):
                text = str(v.get(prop) or "").lower()
                if any(t.lower() in text for t in terms):
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

    def retrieve(self, question, data, client=None) -> List[RetrievedVertex]:
        terms = self._linker.extract_terms(question)
        out: List[RetrievedVertex] = []
        for label in NODE_LABELS:
            scored = self._linker._rank(data.get("vertices", {}).get(label, []), label, terms)
            for score, v in scored[: self._top_k]:
                out.append(RetrievedVertex(label, v.get("name"), score, self.source))
        return out


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

    def attach_vector_retriever(self, retriever: SchemaRetriever) -> None:
        """Slot for the embedding path; call once an embedding endpoint exists."""
        self._vector_retriever = retriever

    def link(self, question: str, data: Optional[GraphData] = None) -> SchemaContext:
        data = data if data is not None else self._base.load_graph()
        paths = [r.retrieve(question, data, client=self._client) for r in self._retrievers]
        if self._vector_retriever is not None:
            paths.append(self._vector_retriever.retrieve(question, data, client=self._client))
        fused = rrf_fuse(paths, k=self._config.fusion_k, weights=self._config.weights)
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

        return SchemaContext(
            tables=tables,
            fields=fields,
            metrics=metrics,
            relations=self._base._collect_relations(tables, metrics, data.get("edges", {})),
            evidence=self._base._build_evidence(tables, metrics, fields)[: self._config.max_evidence],
            matched_terms=base_ctx.matched_terms,
        )
