"""Multi-candidate SQL voting for NL2SQL over the metadata KG (P1-4).

When an NL2SQL generator emits *several* candidate SQLs (beam search,
multi-prompt, self-consistency), the platform must deterministically pick the
best one before execution -- without spending another LLM call (our glm-5.3
endpoint is rate-limited). This module ranks candidates against the *same*
metadata graph the other NL2SQL stages use, purely by:

* **validity** -- each candidate is validated by :class:`KgSqlValidator`
  (table/column existence, 口径/aggregate match, join connectivity);
* **口径 caliber** -- reward candidates that actually hit a metric's canonical
  aggregate (SUM vs COUNT ...);
* **join connectivity** -- prefer fully-connected join graphs;
* **schema overlap** -- prefer candidates that reference the entities the
  question links to (schema-linking signal);
* **golden overlap** -- prefer candidates whose schema footprint matches
  verified golden SQLs for similar questions (few-shot consistency).

The result is a ranked list of :class:`SqlVote` with an auditable score
breakdown, so the chosen SQL is explainable ("why this one").

The voter never calls an LLM; it is deterministic, testable and cheap, and one
instance is reused across one request's candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.graph_op.kg_rule_engine import (
    KgRuleEngine,
    GraphData,
)
from hugegraph_llm.operators.graph_op.kg_sql_validator import (
    KgSqlValidator,
    parse_sql,
)
from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker
from hugegraph_llm.operators.graph_op.kg_golden_sql import (
    _schema_refs_of,
    GoldenRecord,
)

# --- scoring weights (all higher == better, summed into one score) ----------
VALIDITY_BASE = 1000        # a fully-valid candidate starts here
WARNING_PENALTY = 10        # per non-fatal validation warning
CALIBER_WEIGHT = 20         # per metric whose aggregate口径 is matched
JOIN_OK_BONUS = 10          # per joinable table-pair in the FROM/JOIN
JOIN_WARN_PENALTY = 15      # per non-joinable (SQL-J1) table-pair
SCHEMA_OVERLAP_WEIGHT = 5   # per schema entity shared with the question
GOLDEN_OVERLAP_WEIGHT = 4   # per schema entity shared with a golden SQL


@dataclass
class SqlVote:
    """One candidate SQL with its vote score and auditable breakdown."""

    sql: str
    score: float
    valid: bool
    issue_count: int
    breakdown: Dict[str, float] = field(default_factory=dict)
    report: Any = None  # SqlValidationReport (kept for introspection)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "score": self.score,
            "valid": self.valid,
            "issue_count": self.issue_count,
            "breakdown": dict(self.breakdown),
            "validation": self.report.to_dict() if self.report is not None else {},
        }

    def explain(self) -> str:
        """One-line rationale, e.g. ``score=1030 valid=True [validity=1000,caliber=20,...]``."""
        parts = [f"{k}={v}" for k, v in self.breakdown.items()]
        return (
            f"score={self.score} valid={self.valid} issues={self.issue_count} "
            f"[{', '.join(parts)}]"
        )


class KgSqlVoter:
    """Rank multiple candidate SQLs and pick the best one, deterministically."""

    def __init__(
        self,
        question: Optional[str] = None,
        graph_data: Optional[GraphData] = None,
        client: Optional[Any] = None,
        graph_name: Optional[str] = None,
        golden_records: Optional[List[GoldenRecord]] = None,
    ) -> None:
        if graph_data is not None:
            self._graph_data: GraphData = graph_data
        elif client is not None:
            self._graph_data = KgRuleEngine(client, graph_name).load_graph()
        else:
            raise ValueError(
                "KgSqlVoter requires graph_data or client to validate candidates"
            )
        self._graph_name = graph_name
        self._question = question
        self._validator = KgSqlValidator(self._graph_data, graph_name)
        self._golden_records = list(golden_records or [])
        self._linked_names: Set[str] = (
            self._compute_linked_names() if question else set()
        )

    # -- public API ----------------------------------------------------------

    def vote(self, candidates: List[str]) -> List[SqlVote]:
        """Return all candidates ranked best->worst (stable on ties)."""
        if not candidates:
            return []
        scored: List[Tuple[float, int, SqlVote]] = []
        for idx, sql in enumerate(candidates):
            vote = self._score(sql)
            scored.append((vote.score, -idx, vote))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [t[2] for t in scored]

    def best(self, candidates: List[str]) -> Optional[str]:
        """Return the winning SQL string, or None when there are no candidates."""
        ranked = self.vote(candidates)
        return ranked[0].sql if ranked else None

    def best_vote(self, candidates: List[str]) -> Optional[SqlVote]:
        """Return the winning :class:`SqlVote`, or None when there are none."""
        ranked = self.vote(candidates)
        return ranked[0] if ranked else None

    # -- scoring -------------------------------------------------------------

    def _score(self, sql: str) -> SqlVote:
        report = self._validator.validate(sql)
        issues = report.issues

        bd: Dict[str, float] = {}

        # validity gate: an invalid (error-level) candidate starts at zero
        bd["validity"] = float(VALIDITY_BASE) if report.is_valid else 0.0
        # non-fatal warnings still drag the score down
        bd["warnings"] = float(-WARNING_PENALTY * sum(
            1 for i in issues if i.level == "warning"
        ))

        # 口径 caliber: reward metrics whose aggregate matches the canonical formula
        n_checked = len(report.metrics_checked)
        n_mismatch = sum(1 for i in issues if i.rule_id == "SQL-B1")
        bd["caliber"] = float(CALIBER_WEIGHT * (n_checked - n_mismatch))

        # join connectivity: reward fully-connected FROM/JOIN graphs
        tables = list(dict.fromkeys(report.tables_referenced))
        n_pairs = len(tables) * (len(tables) - 1) // 2
        j1 = sum(1 for i in issues if i.rule_id == "SQL-J1")
        bd["join"] = float(JOIN_OK_BONUS * (n_pairs - j1) - JOIN_WARN_PENALTY * j1)

        # schema overlap with the linked question entities
        bd["schema_overlap"] = float(SCHEMA_OVERLAP_WEIGHT * self._schema_overlap(sql, report))

        # golden overlap with the verified SQL pool
        bd["golden_overlap"] = float(GOLDEN_OVERLAP_WEIGHT * self._golden_overlap(sql))

        score = sum(bd.values())
        return SqlVote(
            sql=sql,
            score=score,
            valid=report.is_valid,
            issue_count=len(issues),
            breakdown=bd,
            report=report,
        )

    def _schema_overlap(self, sql: str, report: Any) -> int:
        if not self._linked_names:
            return 0
        refs: Set[str] = set(report.tables_referenced)
        parsed = parse_sql(sql)
        for _tbl, col in parsed["qualified_cols"]:
            refs.add(col)
        for col in parsed["bare_cols"]:
            refs.add(col)
        return len(refs & self._linked_names)

    def _golden_overlap(self, sql: str) -> int:
        if not self._golden_records:
            return 0
        _tables, _cols, refs = _schema_refs_of(sql)
        total = 0
        for rec in self._golden_records:
            total += len(refs & rec.schema_refs)
            for r in refs:
                base = r.split(".", 1)[-1] if "." in r else r
                for s in rec.schema_refs:
                    if s not in refs and s.endswith("." + base):
                        total += 1
        return total

    def _compute_linked_names(self) -> Set[str]:
        """Table/field/metric names the question links to (schema-linking signal)."""
        linked: Set[str] = set()
        try:
            linker = KgSchemaLinker()
            ctx = linker.link(self._question, data=self._graph_data)
        except Exception:  # pragma: no cover - graph/linker failure -> no signal
            return linked
        for t in ctx.tables:
            name = t.get("name")
            if name:
                linked.add(name)
        for f in ctx.fields:
            name = f.get("name")
            if name:
                linked.add(name)
                linked.add(name.split(".", 1)[-1])
        for m in ctx.metrics:
            name = m.get("name")
            if name:
                linked.add(name)
        return linked
