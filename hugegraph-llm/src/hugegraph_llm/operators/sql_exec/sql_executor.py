"""SQL execution layer for the NL2SQL loop ("问 -> SQL -> 执行 -> 答案").

The NL2SQL pipeline produces *text* SQL; running it needs a real engine. This
module defines the execution contract (``SqlExecutor``) plus one concrete
embedded implementation (``DuckDbExecutor``) used by the demo/Gradio loop.

Engine-agnostic on purpose: a warehouse integration (Hive/StarRocks/ClickHouse)
later only needs to implement ``SqlExecutor`` -- the runner and the UI do not
change.

Typical use::

    ex = DuckDbExecutor()              # in-memory, seeded with sample rows
    result = ex.execute("SELECT city, SUM(order.amount) AS amount FROM order GROUP BY city")
    print(result.row_count, result.columns)
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log

# Sample row-level data aligned with the metadata slice (order/payment/user)
# so generated SQL actually returns rows in the demo.
DEFAULT_SAMPLE_DATA: Dict[str, List[Dict[str, Any]]] = {
    "order": [
        {"order_id": 1, "city": "北京", "amount": 120.5, "user_id": 101},
        {"order_id": 2, "city": "上海", "amount": 88.0, "user_id": 102},
        {"order_id": 3, "city": "北京", "amount": 240.0, "user_id": 103},
        {"order_id": 4, "city": "深圳", "amount": 66.5, "user_id": 101},
        {"order_id": 5, "city": "上海", "amount": 320.0, "user_id": 104},
        {"order_id": 6, "city": "北京", "amount": 45.0, "user_id": 102},
        {"order_id": 7, "city": "广州", "amount": 199.9, "user_id": 105},
        {"order_id": 8, "city": "深圳", "amount": 510.0, "user_id": 103},
    ],
    "payment": [
        {"pay_id": 9001, "order_id": 1, "amount": 120.5},
        {"pay_id": 9002, "order_id": 2, "amount": 88.0},
        {"pay_id": 9003, "order_id": 3, "amount": 240.0},
        {"pay_id": 9004, "order_id": 4, "amount": 66.5},
        {"pay_id": 9005, "order_id": 5, "amount": 320.0},
        {"pay_id": 9006, "order_id": 6, "amount": 45.0},
        {"pay_id": 9007, "order_id": 7, "amount": 199.9},
        {"pay_id": 9008, "order_id": 8, "amount": 510.0},
    ],
    "user": [
        {"user_id": 101, "name": "张三"},
        {"user_id": 102, "name": "李四"},
        {"user_id": 103, "name": "王五"},
        {"user_id": 104, "name": "赵六"},
        {"user_id": 105, "name": "钱七"},
    ],
}


@dataclass
class ExecutionResult:
    """Outcome of running one SQL against a data engine."""

    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


def _infer_type(value: Any) -> str:
    """Map a Python sample value to a DuckDB column type."""
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "DOUBLE"
    return "VARCHAR"


# --- read-only SQL policy ---------------------------------------------------
# Only these first keywords may start an executable statement. Everything else
# (DDL / DML / admin) is refused before it reaches the engine.
_SELECT_ONLY_FIRST = {"select", "with", "values", "explain", "describe"}
# Tokens banned anywhere in the statement (after masking string literals), so a
# crafted SELECT cannot smuggle side effects through subqueries or pragmas.
_BLOCKED_TOKENS = (
    "attach", "detach", "load", "install", "copy", "export", "import",
    "pragma", "create", "insert", "update", "delete", "drop", "alter",
    "truncate", "replace", "merge", "call", "set", "reset", "checkpoint",
    "vacuum", "dump", "restore",
)


def _mask_strings(sql: str) -> str:
    """Replace string literals so their *contents* never trip the policy."""
    masked = re.sub(r"'([^']|'')*'", "''", sql)
    return re.sub(r'"([^"]|"")*"', '""', masked)


def _first_token(sql: str) -> Optional[str]:
    """First keyword of the first real statement (comments stripped)."""
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = s.strip().lstrip("(").strip().lower().rstrip(";")
    if not s:
        return None
    m = re.match(r"[a-z_]+", s)
    return m.group(0) if m else s.split()[0]


def _validate_sql(sql: str) -> Optional[str]:
    """Return an error message when ``sql`` must not run, else ``None``.

    Policy: single read-only statement only. Multi-statement injection, DDL,
    DML and engine-admin tokens are all refused.
    """
    stmts = [
        s for s in re.split(r";", sql)
        if s.strip() and not re.match(r"^\s*(--|/\*)", s)
    ]
    if len(stmts) > 1:
        return f"multi-statement SQL is not allowed ({len(stmts)} statements)"
    tok = _first_token(sql)
    if tok is None:
        return "empty SQL"
    if tok not in _SELECT_ONLY_FIRST:
        return f"only read-only statements allowed, got {tok!r}"
    masked = _mask_strings(sql).lower()
    for bad in _BLOCKED_TOKENS:
        if re.search(rf"\b{bad}\b", masked):
            return f"statement blocked by policy: {bad}"
    return None



class SqlExecutor(ABC):
    """Contract every SQL engine adapter must satisfy."""

    @abstractmethod
    def execute(self, sql: str, limit: int = 50) -> ExecutionResult:
        """Run ``sql`` and return up to ``limit`` rows (plus the total count)."""

    def close(self) -> None:  # pragma: no cover - optional lifecycle hook
        """Release any engine resources (default: no-op)."""


class DuckDbExecutor(SqlExecutor):
    """In-memory DuckDB adapter seeded with the demo sample tables.

    Production hardening: every statement passes through ``_validate_sql``
    (read-only policy), runs on a single worker with a wall-clock timeout, and
    is limited to ``limit + 1`` fetched rows. All executions are audit-logged.
    """

    def __init__(
        self,
        sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        timeout_s: float = 10.0,
    ) -> None:
        import duckdb

        self._conn = duckdb.connect(":memory:")
        self._timeout_s = timeout_s
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="duckdb-exec"
        )
        self._table_names: set = set()
        self._col_names: set = set()
        self._register(sample_data if sample_data is not None else DEFAULT_SAMPLE_DATA)

    def _register(self, tables: Dict[str, List[Dict[str, Any]]]) -> None:
        for name, rows in tables.items():
            if not rows:
                continue
            self._table_names.add(name)
            for row in rows:
                self._col_names.update(row.keys())
            col_types = {k: _infer_type(v) for k, v in rows[0].items()}
            cols = ", ".join(f'"{k}" {t}' for k, t in col_types.items())
            self._conn.execute(f'CREATE TABLE "{name}" ({cols})')
            for row in rows:
                self._conn.execute(
                    f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in row)})',
                    list(row.values()),
                )

    def _quote_identifiers(self, sql: str) -> str:
        """Quote known table/column identifiers for DuckDB (keywords like
        ``order`` need quoting; string literals are masked first so words
        inside values are untouched)."""
        literals: List[str] = []

        def _mask(m: "re.Match[str]") -> str:
            literals.append(m.group(0))
            return f"__LIT{len(literals) - 1}__"

        masked = re.sub(r"'[^']*'", _mask, sql)
        for name in sorted(self._table_names | self._col_names, key=len, reverse=True):
            if not name or not name.isidentifier():
                continue
            masked = re.sub(rf"\b{re.escape(name)}\b", f'"{name}"', masked)
        for i, lit in enumerate(literals):
            masked = masked.replace(f"__LIT{i}__", lit)
        return masked

    def execute(self, sql: str, limit: int = 50) -> ExecutionResult:
        start = time.monotonic()
        policy_err = _validate_sql(sql)
        if policy_err:
            log.warning("sql executor: BLOCKED (%s) sql=%.140s", policy_err, sql)
            return ExecutionResult(
                error=f"policy: {policy_err}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        sql = self._quote_identifiers(sql)
        try:
            future = self._executor.submit(self._run, sql, limit)
            result = future.result(timeout=self._timeout_s)
            log.info(
                "sql executor: ok rows=%s ms=%.1f sql=%.140s",
                result.row_count, result.duration_ms, sql,
            )
            return result
        except concurrent.futures.TimeoutError:
            log.warning(
                "sql executor: TIMEOUT after %.1fs sql=%.140s", self._timeout_s, sql
            )
            return ExecutionResult(
                error=f"query timeout after {self._timeout_s:.1f}s",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - surface any engine error
            return ExecutionResult(
                error=str(exc), duration_ms=(time.monotonic() - start) * 1000
            )

    def _run(self, sql: str, limit: int) -> ExecutionResult:
        """Execute on the worker thread; only fetches ``limit + 1`` rows."""
        start = time.monotonic()
        cur = self._conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(limit + 1)
        truncated = len(fetched) > limit
        rows = fetched[:limit]
        total = len(fetched)
        if truncated:
            # best-effort total for the header; falls back to fetched count
            try:
                count_sql = (
                    f"SELECT COUNT(*) AS __n FROM ( {sql.rstrip().rstrip(';')} ) _t"
                )
                total = int(self._conn.execute(count_sql).fetchone()[0])
            except Exception:  # noqa: BLE001 - VALUES/EXPLAIN etc.
                pass
        return ExecutionResult(
            columns=columns,
            rows=[list(r) for r in rows],
            row_count=total,
            truncated=truncated,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - already closed
            pass
