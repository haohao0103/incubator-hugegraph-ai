"""Schema linking for NL2SQL over the HugeGraph metadata KG.

Targeted at the NL2SQL anchor case (Huolala data warehouse). A warehouse has
hundreds/thousands of tables: dumping the whole schema into an LLM prompt
blows the context window and dilutes attention. This module implements the
**schema linking** step (BIRD-leaderboard style "oracle knowledge"):

1. pull the natural-language terms out of the question
2. locate the matching Table / Field / Metric vertices on the graph
3. expand to a bounded neighbourhood (hasColumn / computedFrom /
   computedFromField) to get the join-relevant subgraph
4. assemble a *compact* prompt context: DDL-ish schema + metric definitions
   + field comments (the external-knowledge evidence that pushes BIRD
   accuracy from ~58 to 75-82)

The linking is deterministic (graph matching, no LLM call), so it is cheap,
testable and auditable; the LLM only sees the reduced context.

Typical use::

    linker = KgSchemaLinker(client)                 # or pass graph data
    ctx = linker.link("上个月各城市的完单金额是多少")
    prompt_ctx = ctx.to_prompt_context()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Question words that carry no schema signal (CN + EN).
_STOPWORDS: Set[str] = {
    "的", "了", "是", "在", "有", "和", "与", "或", "多少", "哪些", "什么", "怎么",
    "查询", "统计", "计算", "求", "看", "给", "我", "请", "帮我", "一下", "按",
    "每个", "各个", "分别", "所有", "全部", "最近", "本月", "今日", "所有",
    "the", "a", "an", "of", "in", "on", "for", "by", "what", "how", "many",
    "show", "list", "get", "give", "me", "please", "total", "count",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[一-鿿]+")
_CJK_RE = re.compile(r"[一-鿿]")

# label -> (text fields used for matching, weight multiplier)
_MATCH_FIELDS: Dict[str, Tuple[Tuple[str, ...], float]] = {
    "Table": (("name", "comment"), 1.0),
    "Field": (("name", "comment", "table"), 1.0),
    "Metric": (("name", "definition", "formula"), 1.2),
}


@dataclass
class SchemaLinkConfig:
    """Budget knobs for the reduced context."""

    max_tables: int = 5
    max_metrics: int = 5
    max_fields_per_table: int = 12
    max_evidence: int = 12
    min_score: float = 0.5
    include_comments: bool = True


@dataclass
class SchemaContext:
    """The reduced schema + evidence produced by schema linking."""

    tables: List[Dict[str, Any]] = field(default_factory=list)
    fields: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)

    def to_prompt_context(self, include_evidence: bool = True) -> str:
        """Render the context as prompt text (schema + metrics + evidence)."""
        lines: List[str] = ["== SCHEMA =="]
        if not self.tables and not self.metrics:
            lines.append("(no schema matched)")
        for table in self.tables:
            comment = table.get("comment") or ""
            lines.append(f"table {table.get('name')}" + (f"  -- {comment}" if comment else ""))
        for f in self.fields:
            comment = f.get("comment") or ""
            ftype = f.get("type") or ""
            suffix = "  -- " + " / ".join(p for p in (comment, ftype) if p) if (comment or ftype) else ""
            lines.append(f"  - {f.get('name')}{suffix}")
        if self.metrics:
            lines.append("== METRICS ==")
            for m in self.metrics:
                lines.append(f"metric {m.get('name')}")
                if m.get("definition"):
                    lines.append(f"  definition: {m['definition']}")
                if m.get("formula"):
                    lines.append(f"  formula: {m['formula']}")
        if include_evidence and self.evidence:
            lines.append("== EVIDENCE (external knowledge) ==")
            for i, ev in enumerate(self.evidence, 1):
                lines.append(f"{i}. {ev}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": self.tables,
            "fields": self.fields,
            "metrics": self.metrics,
            "relations": [list(r) for r in self.relations],
            "evidence": self.evidence,
            "matched_terms": self.matched_terms,
        }


class KgSchemaLinker:
    """Locate the schema subgraph relevant to a natural-language question."""

    def __init__(
        self,
        client: Optional[Any] = None,
        config: Optional[SchemaLinkConfig] = None,
        synonyms: Optional[Dict[str, str]] = None,
    ) -> None:
        self._client = client
        self._config = config or SchemaLinkConfig()
        # alias -> canonical (e.g. {"完单": "order"}); aliases are matched as
        # if the question contained the canonical term.
        self._synonyms = dict(synonyms or {})

    # -- term extraction -----------------------------------------------------

    def extract_terms(self, question: str) -> List[str]:
        """Tokenize a question into schema-relevant terms (CN + EN)."""
        if not question:
            return []
        terms: List[str] = []
        for raw in _TOKEN_RE.findall(str(question)):
            token = raw.lower()
            if token in _STOPWORDS or len(token) < 2 and not token.isdigit():
                continue
            # snake_case identifiers also contribute their parts
            if "_" in token:
                terms.extend(p for p in token.split("_") if len(p) >= 2)
            # Chinese has no spaces: a run of >=3 chars also contributes 2-grams
            # so "完单金额" can still match the term/alias "完单".
            if _CJK_RE.search(token) and len(token) >= 3:
                terms.extend(
                    token[i : i + 2]
                    for i in range(len(token) - 1)
                    if token[i : i + 2] not in _STOPWORDS
                )
            terms.append(token)
        # alias expansion: '完单' also searches 'order'
        expanded = list(terms)
        for term in terms:
            canonical = self._synonyms.get(term)
            if canonical and canonical not in expanded:
                expanded.append(canonical.lower())
        return list(dict.fromkeys(expanded))

    # -- scoring -------------------------------------------------------------

    @staticmethod
    def _vertex_text(label: str, vertex: Dict[str, Any]) -> Dict[str, str]:
        fields, _ = _MATCH_FIELDS.get(label, (("name",), 1.0))
        return {f: str(vertex.get(f) or "") for f in fields}

    def score_vertex(self, label: str, vertex: Dict[str, Any], terms: Sequence[str]) -> float:
        """Relevance of one vertex to the question terms (0 when unrelated)."""
        texts = self._vertex_text(label, vertex)
        name = texts.get("name", "").lower()
        best = 0.0
        for term in terms:
            term = term.lower()
            if not term:
                continue
            score = 0.0
            if name and name == term:
                score = 3.0
            elif name and term in name:
                score = 2.0
            else:
                for key, value in texts.items():
                    if key == "name" or not value:
                        continue
                    if term in value.lower():
                        score = max(score, 1.0)
                        break
            if score:
                best = max(best, score)
        _, weight = _MATCH_FIELDS.get(label, (("name",), 1.0))
        return best * weight

    # -- linking -------------------------------------------------------------

    def link(
        self, question: str, data: Optional[Dict[str, Any]] = None
    ) -> SchemaContext:
        """Return the schema subgraph relevant to ``question``."""
        if data is None:
            data = self.load_graph()
        vertices = data.get("vertices", {})
        edges = data.get("edges", {})
        terms = self.extract_terms(question)
        if not terms:
            return SchemaContext()

        scored_tables = self._rank(vertices.get("Table", []), "Table", terms)
        scored_metrics = self._rank(vertices.get("Metric", []), "Metric", terms)
        scored_fields = self._rank(vertices.get("Field", []), "Field", terms)

        tables = [v for _, v in scored_tables[: self._config.max_tables]]
        metrics = [v for _, v in scored_metrics[: self._config.max_metrics]]
        fields = self._expand_fields(tables, metrics, scored_fields, vertices, edges)

        return SchemaContext(
            tables=tables,
            fields=fields,
            metrics=metrics,
            relations=self._collect_relations(tables, metrics, edges),
            evidence=self._build_evidence(tables, metrics, fields),
            matched_terms=terms,
        )

    def _rank(
        self, rows: Sequence[Dict[str, Any]], label: str, terms: Sequence[str]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored = [
            (self.score_vertex(label, v, terms), v)
            for v in rows
            if v.get("name")
        ]
        scored = [(s, v) for s, v in scored if s >= self._config.min_score]
        scored.sort(key=lambda item: (-item[0], str(item[1].get("name"))))
        return scored

    def _expand_fields(
        self,
        tables: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
        scored_fields: List[Tuple[float, Dict[str, Any]]],
        vertices: Dict[str, List[Dict[str, Any]]],
        edges: Dict[str, List[Tuple[str, str]]],
    ) -> List[Dict[str, Any]]:
        """Fields of the matched tables + metric sources + directly matched fields."""
        table_names = {t.get("name") for t in tables}
        all_fields = {f.get("name"): f for f in vertices.get("Field", []) if f.get("name")}

        selected: Dict[str, Dict[str, Any]] = {}
        # 1) fields belonging to a matched table (hasColumn or Field.table)
        for src, dst in edges.get("hasColumn", []):
            if src in table_names and dst in all_fields:
                selected[dst] = all_fields[dst]
        for name, f in all_fields.items():
            if f.get("table") in table_names:
                selected[name] = f
        # 2) fields a matched metric is computed from
        for src, dst in edges.get("computedFromField", []):
            if dst in all_fields and any(m.get("name") == src for m in metrics):
                selected[dst] = all_fields[dst]
        # 3) fields matched directly by the question
        for score, f in scored_fields:
            if len(selected) >= self._config.max_fields_per_table * max(1, len(table_names)):
                break
            selected.setdefault(f.get("name"), f)
        ordered = sorted(selected.values(), key=lambda f: str(f.get("name")))
        return ordered

    def _collect_relations(
        self,
        tables: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
        edges: Dict[str, List[Tuple[str, str]]],
    ) -> List[Tuple[str, str, str]]:
        names = {t.get("name") for t in tables} | {m.get("name") for m in metrics}
        out: List[Tuple[str, str, str]] = []
        for label, pairs in edges.items():
            for src, dst in pairs:
                if src in names or dst in names:
                    out.append((src, label, dst))
        return out

    # -- evidence (P0-2) -----------------------------------------------------

    def _build_evidence(
        self,
        tables: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
        fields: List[Dict[str, Any]],
    ) -> List[str]:
        """External-knowledge evidence: how the warehouse defines things."""
        evidence: List[str] = []
        for m in metrics:
            if m.get("formula"):
                evidence.append(f"指标 {m.get('name')} = {m['formula']}")
            elif m.get("definition"):
                evidence.append(f"指标 {m.get('name')} 口径：{m['definition']}")
        if self._config.include_comments:
            for t in tables:
                if t.get("comment"):
                    evidence.append(f"表 {t.get('name')} 表示{t['comment']}")
            for f in fields:
                if f.get("comment"):
                    evidence.append(
                        f"字段 {f.get('name')} 是{f['comment']}"
                        + (f"（类型 {f['type']}）" if f.get("type") else "")
                    )
        return evidence[: self._config.max_evidence]

    # -- graph loading -------------------------------------------------------

    def load_graph(self) -> Dict[str, Any]:
        """Pull the metadata graph through the live HugeGraph client."""
        if self._client is None:
            return {"vertices": {}, "edges": {}}
        from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine

        return KgRuleEngine(self._client).load_graph()
