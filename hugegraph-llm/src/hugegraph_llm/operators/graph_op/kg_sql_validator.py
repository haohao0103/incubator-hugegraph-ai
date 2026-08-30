"""Validate a generated NL2SQL SQL string against the metadata KG.

The anchor case is NL2SQL over a large data-warehouse graph: the LLM emits a
SQL string, but a wrong table/column name or a mismatched aggregate (SUM vs
COUNT vs AVG) silently yields wrong numbers. This module validates the SQL
*deterministically* against the same HugeGraph metadata graph that feeds
schema-linking (see ``kg_schema_linker``), and returns actionable issues the
caller can feed back into the LLM for self-correction.

The checks mirror the KgRuleEngine governance families so the two layers stay
consistent:

* **A-family (structural)** -- every table/column the SQL references must exist
  in the graph, exactly as ``KgRuleEngine`` A1/A2 guard the graph itself.
* **B-family (metric semantics)** -- when the SQL aggregates a column that a
  ``Metric`` defines, the aggregate function must match the metric's canonical
  formula (``SUM(order.amount)`` not ``AVG``). This is the NL2SQL analogue of
  ``KgRuleEngine`` B2 (formula dangling reference) and B1 (metric source).
* **J-family (join connectivity)** -- tables joined in the SQL must be
  connectable in the metadata graph (share a field name, or be co-referenced by
  a metric), otherwise the join condition is almost certainly wrong.

The validator never calls an LLM: it is pure, testable and cheap, so it can run
on every generated SQL before execution.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.graph_op.kg_rule_engine import (
    KgRuleEngine,
    GraphData,
)

logger = logging.getLogger(__name__)

# SQL keywords that must not be mistaken for bare column identifiers.
_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "join", "on",
    "and", "or", "as", "limit", "distinct", "not", "in", "is", "null", "like",
    "between", "asc", "desc", "inner", "left", "right", "outer", "full", "union",
    "all", "with", "case", "when", "then", "else", "end", "count", "sum", "avg",
    "max", "min", "case", "exists", "intersect", "except",
}

_AGG_RE = re.compile(
    r"(SUM|COUNT|AVG|MAX|MIN)\s*\(\s*(?:DISTINCT\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)?",
    re.IGNORECASE,
)
_QUALIFIED_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")
_FROM_RE = re.compile(
    r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:AS\s+)?"
    # an alias must not be a reserved word (otherwise the optional group would
    # greedily swallow the next keyword, e.g. "FROM a JOIN b" -> b becomes a's alias)
    r"(?!(?:JOIN|INNER|LEFT|RIGHT|OUTER|FULL|CROSS|NATURAL|ON|WHERE|GROUP|"
    r"ORDER|HAVING|UNION|LIMIT|SET|AND|OR)\b)"
    r"([A-Za-z_][A-Za-z0-9_]*)?",
    re.IGNORECASE,
)
# An aggregate function can start a metric formula, e.g. "SUM(order.amount)".
_METRIC_FUNC_RE = re.compile(
    r"^(SUM|COUNT|AVG|MAX|MIN)\s*\(", re.IGNORECASE
)
# SELECT-list portion for alias extraction (lazy until the first FROM).
_SELECT_LIST_RE = re.compile(r"\bSELECT\s+(.*?)\s+FROM\b", re.IGNORECASE | re.DOTALL)


def _split_select_items(text: str) -> List[str]:
    """Split a SELECT list on top-level commas (ignores commas inside parens)."""
    items: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    return [i.strip() for i in items]


@dataclass
class SqlIssue:
    """One validation problem, carrying a concrete suggested fix."""

    rule_id: str
    level: str  # "error" | "warning"
    target: str
    message: str
    suggested_fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "target": self.target,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class SqlValidationReport:
    """Outcome of validating one SQL string against the KG."""

    issues: List[SqlIssue] = field(default_factory=list)
    tables_referenced: List[str] = field(default_factory=list)
    columns_resolved: List[str] = field(default_factory=list)
    metrics_checked: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> List[SqlIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[SqlIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "tables_referenced": self.tables_referenced,
            "columns_resolved": self.columns_resolved,
            "metrics_checked": self.metrics_checked,
        }

    def to_prompt_feedback(self) -> str:
        """Concise, LLM-injectable block describing what to fix.

        Empty string when the SQL is valid, so the caller can append it to the
        self-correction prompt only when there is something to fix.
        """
        if not self.issues:
            return ""
        lines = ["The generated SQL failed validation against the metadata graph. Fix it:"]
        for i, issue in enumerate(self.issues, 1):
            head = f"{i}. [{issue.level.upper()}] ({issue.rule_id}) {issue.message}"
            if issue.suggested_fix:
                head += f" -> {issue.suggested_fix}"
            lines.append(head)
        return "\n".join(lines)


def load_graph(client: Any, graph_name: Optional[str] = None) -> GraphData:
    """Reuse KgRuleEngine's live-HugeGraph loader (same GraphData shape)."""
    return KgRuleEngine(client, graph_name).load_graph()


def parse_sql(sql: str) -> Dict[str, Any]:
    """Public wrapper around the lightweight SQL extractor (see ``_parse_sql``)."""
    return KgSqlValidator._parse_sql(sql)


class KgSqlValidator:
    """Validate generated SQL against the metadata KG.

    Build once per graph snapshot, then call :meth:`validate` per SQL string.
    """

    def __init__(self, graph_data: GraphData, graph_name: Optional[str] = None) -> None:
        self._graph_name = graph_name
        self._build_indexes(graph_data)

    @classmethod
    def from_client(cls, client: Any, graph_name: Optional[str] = None) -> "KgSqlValidator":
        return cls(load_graph(client, graph_name), graph_name)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_indexes(self, data: GraphData) -> None:
        vertices = data.get("vertices", {})
        edges = data.get("edges", {})

        self.table_names: Set[str] = {
            t.get("name") for t in vertices.get("Table", []) if t.get("name")
        }
        self.field_type: Dict[Tuple[str, str], str] = {}
        # The KG stores Field names in fully-qualified form (``order.amount``),
        # so every index is keyed by the full name; bare SQL columns are matched
        # by their basename suffix.
        self.field_full: Set[str] = set()                 # all full field names
        self.field_owner: Dict[str, Set[str]] = defaultdict(set)  # full -> tables
        self.table_fields: Dict[str, Set[str]] = defaultdict(set)  # table -> full
        self.table_col_bases: Dict[str, Set[str]] = defaultdict(set)  # table -> bases
        self.col_basename: Dict[str, List[str]] = defaultdict(list)  # base -> [full]

        field_by_name: Dict[str, Dict[str, Any]] = {}
        for f in vertices.get("Field", []):
            name = f.get("name")
            if name:
                field_by_name[name] = f
                self.field_full.add(name)
                base = name.split(".", 1)[-1]
                self.col_basename[base].append(name)

        for src, dst in edges.get("hasColumn", []):
            self.field_owner[dst].add(src)
            self.table_fields[src].add(dst)
            self.table_col_bases[src].add(dst.split(".", 1)[-1])
            ftype = field_by_name.get(dst, {}).get("type")
            if ftype:
                self.field_type[(src, dst)] = ftype

        # metrics: name -> {formula, definition, tables it spans}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        metric_tables: Dict[str, Set[str]] = defaultdict(set)
        for src, dst in edges.get("computedFromField", []):
            # a field can be owned by several tables; the metric spans all of them
            for owner in self.field_owner.get(dst, set()):
                metric_tables[src].add(owner)
        for m in vertices.get("Metric", []):
            name = m.get("name")
            if not name:
                continue
            formula = m.get("formula") or ""
            self.metrics[name] = {
                "formula": formula,
                "definition": m.get("definition") or "",
                "func": self._metric_func(formula),
                "tables": metric_tables.get(name, set()),
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, sql: str) -> SqlValidationReport:
        parsed = self._parse_sql(sql)
        report = SqlValidationReport()
        report.tables_referenced = [t for t, _ in parsed["tables"]]

        if not parsed["tables"]:
            report.issues.append(
                SqlIssue(
                    rule_id="SQL-P1",
                    level="warning",
                    target="sql",
                    message="未能从 SQL 解析出任何表名（无 FROM/JOIN）",
                    suggested_fix="确认 SQL 是否引用了正确的表",
                )
            )
            return report

        # A-family: table existence
        alias_map = parsed["alias_map"]
        for raw, _alias in parsed["tables"]:
            if raw not in self.table_names:
                report.issues.append(
                    SqlIssue(
                        rule_id="SQL-A1",
                        level="error",
                        target=f"table:{raw}",
                        message=f"表 {raw!r} 不在元数据图中（可能拼写错误或不存在）",
                        suggested_fix=f"使用图中存在的表：{', '.join(sorted(self.table_names)) or '（空）'}",
                    )
                )

        known_tables = {t for t, _ in parsed["tables"] if t in self.table_names}

        # A-family: qualified column ownership
        for tbl_or_alias, col in parsed["qualified_cols"]:
            table = alias_map.get(tbl_or_alias, tbl_or_alias)
            resolved = self._resolve_col(table, col)
            if resolved is None:
                report.issues.append(
                    SqlIssue(
                        rule_id="SQL-A2",
                        level="error",
                        target=f"col:{tbl_or_alias}.{col}",
                        message=f"列 {tbl_or_alias}.{col} 不属于表 {table!r}（或表不存在）",
                        suggested_fix=self._suggest_column(table, col),
                    )
                )
            else:
                report.columns_resolved.append(f"{resolved}.{col}")

        # A-family: bare column resolution (matched by basename suffix)
        for col in parsed["bare_cols"]:
            fulls = self.col_basename.get(col, [])
            owners: Set[str] = set()
            for full in fulls:
                owners |= self.field_owner.get(full, set())
            if not owners:
                report.issues.append(
                    SqlIssue(
                        rule_id="SQL-A2",
                        level="error",
                        target=f"col:{col}",
                        message=f"列 {col!r} 不在任何表的元数据中",
                        suggested_fix=self._suggest_column(None, col),
                    )
                )
            elif len(owners) > 1:
                report.issues.append(
                    SqlIssue(
                        rule_id="SQL-A2",
                        level="warning",
                        target=f"col:{col}",
                        message=f"列 {col!r} 属于多个表（{', '.join(sorted(owners))}），存在歧义",
                        suggested_fix=(
                            f"用 表名.{col} 显式限定其一"
                            f"（可选：{', '.join(sorted(owners))}）"
                        ),
                    )
                )
                # resolve to every owner for downstream metric checks
                for o in owners:
                    report.columns_resolved.append(f"{o}.{col}")
            else:
                report.columns_resolved.append(f"{next(iter(owners))}.{col}")

        # B-family: metric 口径 (aggregate function must match metric formula)
        for func, tbl_or_alias, col in parsed["aggregates"]:
            table = alias_map.get(tbl_or_alias, tbl_or_alias) if tbl_or_alias else None
            col_ref = f"{table}.{col}" if table else col
            matched_metric = self._metric_for_column(col_ref, table)
            if matched_metric is None:
                continue
            report.metrics_checked.append(matched_metric)
            expected = self.metrics[matched_metric]["func"]
            if expected and func.upper() != expected.upper():
                report.issues.append(
                    SqlIssue(
                        rule_id="SQL-B1",
                        level="error",
                        target=f"metric:{matched_metric}",
                        message=(
                            f"指标 {matched_metric!r} 的口径是 {expected}"
                            f"（公式：{self.metrics[matched_metric]['formula']!r}），"
                            f"但 SQL 使用了 {func.upper()}"
                        ),
                        suggested_fix=(
                            f"将 {func.upper()}({col_ref}) 改为 "
                            f"{expected}({col_ref})"
                        ),
                    )
                )

        # J-family: join connectivity across referenced (known) tables
        self._check_joins(known_tables, report)

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_col(self, table: Optional[str], col: str) -> Optional[str]:
        """Return the owning table if ``col`` is owned by ``table``.

        Field names in the KG are fully-qualified (``order.amount``), so the
        lookup uses the ``table.column`` form.
        """
        if table is None:
            return None
        full = f"{table}.{col}"
        if full in self.table_fields.get(table, set()):
            return table
        return None

    def _metric_for_column(self, col_ref: str, table: Optional[str]) -> Optional[str]:
        """Find a Metric whose formula references ``col_ref``.

        ``col_ref`` may be fully-qualified (``order.amount``) or a bare basename
        (``amount``) when the SQL column was unqualified.
        """
        for name, meta in self.metrics.items():
            refs = self._ref_fields(meta["formula"])
            for ref in refs:
                if ref == col_ref:
                    return name
                if table is None and ref.split(".", 1)[-1] == col_ref:
                    return name
        return None

    @staticmethod
    def _ref_fields(formula: str) -> List[str]:
        if not formula:
            return []
        return [
            m.group(0) for m in _QUALIFIED_RE.finditer(formula)
        ]

    def _suggest_column(self, table: Optional[str], col: str) -> str:
        """Build a fix hint listing candidate columns close to ``col``."""
        if table and f"{table}.{col}" not in self.table_fields.get(table, set()):
            cols = sorted(self.table_fields.get(table, set()))
            if cols:
                return f"表 {table} 拥有的列：{', '.join(cols)}"
        # search any table owning a column with the same basename
        owners = sorted({t for full in self.col_basename.get(col, [])
                         for t in self.field_owner.get(full, set())})
        if owners:
            return f"列 {col} 实际属于表：{', '.join(owners)}"
        return "请检查列名拼写"

    def _check_joins(self, tables: Set[str], report: SqlValidationReport) -> None:
        known = [t for t in tables if t in self.table_names]
        if len(known) < 2:
            return
        for i in range(len(known)):
            for j in range(i + 1, len(known)):
                a, b = known[i], known[j]
                if self._tables_joinable(a, b):
                    continue
                report.issues.append(
                    SqlIssue(
                        rule_id="SQL-J1",
                        level="warning",
                        target=f"join:{a}~{b}",
                        message=f"表 {a} 与 {b} 在元数据中不可连接",
                        suggested_fix=(
                            "两表在元数据中无共享字段、也无共同指标，"
                            "JOIN 条件很可能错误"
                        ),
                    )
                )

    def _tables_joinable(self, a: str, b: str) -> bool:
        if self.table_col_bases[a] & self.table_col_bases[b]:
            return True
        for meta in self.metrics.values():
            if a in meta["tables"] and b in meta["tables"]:
                return True
        return False

    @staticmethod
    def _metric_func(formula: str) -> Optional[str]:
        m = _METRIC_FUNC_RE.match(formula.strip())
        return m.group(1).upper() if m else None

    @staticmethod
    def _select_aliases(sql: str) -> Set[str]:
        """Column aliases declared in the SELECT list (``AS x`` or implicit).

        SQL engines resolve a bare identifier in ORDER BY / GROUP BY / HAVING
        to the SELECT alias when one exists, so the validator must too --
        otherwise natural LLM output like
        ``SELECT city, SUM(order.amount) AS order_amount ... ORDER BY order_amount``
        is wrongly flagged as an unknown column (SQL-A2).

        Explicit ``AS alias`` everywhere; implicit aliases only when the
        trailing identifier follows an expression (contains ``(`` or ``.``),
        so a plain real column like ``SELECT city`` is never misread as one.
        """
        aliases: Set[str] = set()
        for m in re.finditer(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, re.IGNORECASE):
            aliases.add(m.group(1))
        m = _SELECT_LIST_RE.search(sql)
        if m is not None:
            for item in _split_select_items(m.group(1)):
                am = re.match(r"^(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", item)
                if am is None:
                    continue
                lead, cand = am.group(1), am.group(2)
                if cand.lower() in _SQL_KEYWORDS:
                    continue
                if "(" not in lead and "." not in lead:
                    continue
                aliases.add(cand)
        return aliases

    @staticmethod
    def _parse_sql(sql: str) -> Dict[str, Any]:
        """Lightweight SQL extractor (table/alias/qualified/bare/aggregates).

        Not a full parser -- it targets the shape NL2SQL systems emit.
        """
        tables: List[Tuple[str, Optional[str]]] = []
        alias_map: Dict[str, str] = {}
        for m in _FROM_RE.finditer(sql):
            raw = m.group(1)
            alias = m.group(2)
            tables.append((raw, alias))
            if alias:
                alias_map[alias] = raw

        qualified: List[Tuple[str, str]] = [
            (m.group(1), m.group(2)) for m in _QUALIFIED_RE.finditer(sql)
        ]
        # SELECT aliases must not be treated as bare columns (SQL-A2): a bare
        # alias in ORDER BY / GROUP BY / HAVING refers to the SELECT expression.
        select_aliases = KgSqlValidator._select_aliases(sql)
        alias_lower = {a.lower() for a in select_aliases}
        # bare columns: identifiers that are not keywords, not part of a
        # qualified reference (either side of the dot), not function calls,
        # and not SELECT aliases.
        qualified_left = {q[0] for q in qualified}
        bare: List[str] = []
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", sql):
            tok = m.group(0)
            low = tok.lower()
            if low in _SQL_KEYWORDS:
                continue
            if low in alias_lower:
                continue
            if low in qualified_left:
                continue
            # skip the right side of a qualified reference (preceded by '.')
            if m.start() > 0 and sql[m.start() - 1] == ".":
                continue
            # skip if immediately followed by '(' (function call)
            nxt = sql[m.end():m.end() + 1]
            if nxt == "(":
                continue
            bare.append(tok)

        aggregates: List[Tuple[str, Optional[str], Optional[str]]] = []
        for m in _AGG_RE.finditer(sql):
            func = m.group(1).upper()
            arg = m.group(2)
            if arg and "." in arg:
                t, c = arg.split(".", 1)
                aggregates.append((func, t, c))
            elif arg:
                aggregates.append((func, None, arg))
            else:
                aggregates.append((func, None, None))

        return {
            "tables": tables,
            "alias_map": alias_map,
            "qualified_cols": qualified,
            "bare_cols": bare,
            "aggregates": aggregates,
            "aliases": sorted(select_aliases),
        }
