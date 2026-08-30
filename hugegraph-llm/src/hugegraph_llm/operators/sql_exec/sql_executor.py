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

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


class SqlExecutor(ABC):
    """Contract every SQL engine adapter must satisfy."""

    @abstractmethod
    def execute(self, sql: str, limit: int = 50) -> ExecutionResult:
        """Run ``sql`` and return up to ``limit`` rows (plus the total count)."""

    def close(self) -> None:  # pragma: no cover - optional lifecycle hook
        """Release any engine resources (default: no-op)."""


class DuckDbExecutor(SqlExecutor):
    """In-memory DuckDB adapter seeded with the demo sample tables."""

    def __init__(self, sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        import duckdb

        self._conn = duckdb.connect(":memory:")
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
        sql = self._quote_identifiers(sql)
        try:
            cur = self._conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchall()
            total = len(fetched)
            truncated = total > limit
            rows = fetched[:limit]
            return ExecutionResult(
                columns=columns,
                rows=[list(r) for r in rows],
                row_count=total,
                truncated=truncated,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - surface any engine error
            return ExecutionResult(
                error=str(exc), duration_ms=(time.monotonic() - start) * 1000
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - already closed
            pass
