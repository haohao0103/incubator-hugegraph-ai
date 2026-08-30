"""Golden-SQL feedback loop backed by the metadata KG (NL2SQL P0-4).

Vanna (23k stars) proves the single highest-leverage NL2SQL improvement is the
*golden-SQL feedback loop*: every SQL a human verifies is stored and replayed
as a few-shot example for the next similar question, so accuracy compounds over
time instead of resetting each session. This module plants that loop on the
same HugeGraph metadata graph:

* A ``Query`` vertex holds ``(question, sql, schema_refs, domain, created_at)``.
* ``references`` edges connect each Query to the ``Table``/``Field`` vertices it
  touches, so the graph stays navigable and governable (consistent with the
  KgRuleEngine "every reference must resolve" principle).
* Retrieval is **deterministic** (no embedding needed - our ``glm-5.3`` has no
  embedding endpoint): it ranks stored golden queries by lexical + schema
  overlap with the new question, using the same term extractor the schema
  linker uses. This is what gets injected back into the LLM prompt as few-shot
  context.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker
from hugegraph_llm.operators.graph_op.kg_sql_validator import parse_sql

logger = logging.getLogger(__name__)

_QUERY_LABEL = "Query"
_REF_EDGE = "references"
# schema_refs are stored denormalized (semicolon-joined) on the vertex for fast
# single-call retrieval; the references edges keep the graph navigable.
_REF_SEP = ";"

_LINKER = KgSchemaLinker()


def _schema_refs_of(sql: str) -> Tuple[List[str], List[str], Set[str]]:
    """Return (tables, qualified_cols, deduplicated ref tokens) for a SQL."""
    parsed = parse_sql(sql)
    tables = [t for t, _ in parsed["tables"]]
    cols = [f"{tbl}.{col}" for tbl, col in parsed["qualified_cols"]]
    refs: Set[str] = set(tables)
    refs.update(cols)
    for col in parsed["bare_cols"]:
        # bare columns cannot be unambiguously attributed without the graph;
        # keep the basename so schema-overlap can still match by suffix.
        refs.add(col)
    return tables, cols, refs


@dataclass
class GoldenRecord:
    """One stored golden (question, sql) pair with its schema footprint."""

    question: str
    sql: str
    schema_refs: Set[str] = field(default_factory=set)
    domain: Optional[str] = None
    vertex_id: Optional[str] = None

    def to_prompt_fewshot(self) -> str:
        return f"-- Q: {self.question}\n-- A: {self.sql}"


def score_golden(terms: Set[str], rec: GoldenRecord, linked_names: Optional[Set[str]] = None) -> int:
    """Lexical + schema-overlap score between a question and a golden record.

    ``terms`` are the term tokens of the new question (see
    ``KgSchemaLinker.extract_terms``). ``linked_names`` are the table/field
    names the question links to in the live graph (via schema-linking); matches
    there dominate because they reflect the actual semantic intent, not just
    surface word overlap. Higher is more relevant.
    """
    q_terms = _LINKER.extract_terms(rec.question)
    term_hits = len(terms & set(q_terms)) * 2
    schema_hits = 0
    for t in terms:
        if t in rec.schema_refs:
            schema_hits += 1
        else:
            # match by basename suffix, e.g. term 'amount' hits 'order.amount'
            if any(r.endswith("." + t) for r in rec.schema_refs):
                schema_hits += 1
    linked_hits = 0
    for n in (linked_names or set()):
        if n in rec.schema_refs:
            linked_hits += 3
        elif any(r.endswith("." + n) for r in rec.schema_refs):
            linked_hits += 3
    return term_hits + schema_hits + linked_hits


class KgGoldenSqlStore:
    """Store and retrieve verified (question, SQL) pairs in HugeGraph."""

    def __init__(self, client: Any, graph_name: Optional[str] = None) -> None:
        self._client = client
        self._graph_name = graph_name

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        question: str,
        sql: str,
        domain: Optional[str] = None,
        verify: bool = True,
    ) -> Optional[str]:
        """Store a verified golden SQL.

        Creates a ``Query`` vertex plus ``references`` edges to the
        Table/Field vertices it actually touches (skipping any that do not
        exist in the graph). Returns the new vertex id, or ``None`` if the
        write failed.
        """
        tables, cols, refs = _schema_refs_of(sql)
        ref_str = _REF_SEP.join(sorted(refs))
        try:
            resp = self._client.gremlin().exec(
                "g.addV('%s').property('question', %s)"
                ".property('sql', %s).property('schema_refs', %s)"
                ".property('domain', %s).property('created_at', %s)"
                % (
                    _QUERY_LABEL,
                    _quote(question),
                    _quote(sql),
                    _quote(ref_str),
                    _quote(domain or ""),
                    _quote(str(int(time.time()))),
                )
            )
        except Exception as exc:  # pragma: no cover - network/permission guard
            logger.error("failed to add golden SQL: %s", exc)
            return None
        vid = self._first_vertex_id(resp)
        if vid is None:
            return None
        if verify:
            self._link_references(vid, tables, cols)
        return vid

    def _link_references(self, vid: str, tables: List[str], cols: List[str]) -> None:
        for t in tables:
            self._add_ref_edge(vid, "Table", t)
        for c in cols:
            self._add_ref_edge(vid, "Field", c)

    def _add_ref_edge(self, vid: str, label: str, name: str) -> None:
        try:
            self._client.gremlin().exec(
                "g.V(%s).as('q').V().has('%s','name',%s)"
                ".addE('%s').from('q')"
                % (_quote(vid), label, _quote(name), _REF_EDGE)
            )
        except Exception as exc:  # pragma: no cover - missing vertex is non-fatal
            logger.debug("skip reference %s->%s.%s: %s", vid, label, name, exc)

    # ------------------------------------------------------------------
    # Read / retrieve
    # ------------------------------------------------------------------

    def get_similar(self, question: str, top_k: int = 3) -> List[GoldenRecord]:
        """Rank stored golden queries by relevance to ``question``."""
        records = self._load_records()
        if not records:
            return []
        terms = set(_LINKER.extract_terms(question))
        # graph-aware: link the question to tables/fields so the retrieval
        # reflects semantic intent, not just surface word overlap.
        linked = self._linked_names(question)
        scored = [(score_golden(terms, r, linked), r) for r in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k] if _ > 0]

    def _linked_names(self, question: str) -> Set[str]:
        """Table/field names the question links to in the live graph."""
        try:
            linker = KgSchemaLinker(self._client)
            data = linker.load_graph()
            ctx = linker.link(question, data)
            names: Set[str] = set()
            for t in ctx.tables:
                name = t.get("name")
                if name:
                    names.add(name)
            for f in ctx.fields:
                name = f.get("name")
                if name:
                    names.add(name)
                    names.add(name.split(".", 1)[-1])
            return names
        except Exception:  # pragma: no cover - graph unavailable -> lexical only
            return set()

    def _load_records(self) -> List[GoldenRecord]:
        try:
            resp = self._client.gremlin().exec(
                f"g.V().hasLabel('{_QUERY_LABEL}').elementMap()"
            )
        except Exception as exc:  # pragma: no cover - network guard
            logger.error("failed to load golden SQLs: %s", exc)
            return []
        rows = resp.get("data") if isinstance(resp, dict) else (resp or [])
        out: List[GoldenRecord] = []
        for row in rows or []:
            props = {k: v for k, v in row.items() if k not in ("id", "label")}
            ref_str = props.get("schema_refs") or ""
            out.append(
                GoldenRecord(
                    question=props.get("question", ""),
                    sql=props.get("sql", ""),
                    schema_refs={
                        r for r in ref_str.split(_REF_SEP) if r
                    },
                    domain=props.get("domain") or None,
                    vertex_id=self._vertex_id(row),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vertex_id(row: Dict[str, Any]) -> Optional[str]:
        raw = row.get("id")
        if raw is None:
            return None
        text = raw.get("id") if isinstance(raw, dict) else str(raw)
        if isinstance(text, str) and ":" in text:
            return text.split(":", 1)[-1]
        return str(text)

    @staticmethod
    def _first_vertex_id(resp: Any) -> Optional[str]:
        if not isinstance(resp, dict):
            return None
        rows = resp.get("data") or []
        if not rows:
            return None
        return KgGoldenSqlStore._vertex_id(rows[0])


def _quote(value: str) -> str:
    """Gremlin single-quoted string literal (escapes internal quotes)."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"
