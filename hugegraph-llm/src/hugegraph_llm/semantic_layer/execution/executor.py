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

"""Read-only SQL execution against a SQLite warehouse."""

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["ExecutionResult", "SqliteExecutor"]


@dataclass
class ExecutionResult:
    """What one statement produced. ``error`` set means it did not run."""

    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows[:50],
            "row_count": len(self.rows),
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    def results_equal(self, other: "ExecutionResult") -> bool:
        """Order-insensitive result comparison (rows as multisets).

        GROUP BY ordering is not part of an answer's meaning unless the
        question says "top N" -- comparing multisets judges the SQL, not an
        incidental sort order.
        """
        if not self.ok or not other.ok:
            return False
        if self.columns != other.columns:
            return False
        def normalise(rows: List[List[Any]]) -> List[List[Any]]:
            return sorted(
                (tuple(round(v, 6) if isinstance(v, float) else v for v in row)
                 for row in rows)
            )
        return normalise(self.rows) == normalise(other.rows)


class SqliteExecutor:
    """Executes read-only SQL against a SQLite database.

    Opens in ``mode=ro`` so no generated (or LLM-generated) statement can
    write -- NL2SQL is a read path, and a read-only connection makes that a
    property of the runtime rather than a prompt instruction.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> ExecutionResult:
        import time

        result = ExecutionResult()
        start = time.time()
        try:
            cursor = self._connect().execute(sql, params or ())
            result.columns = [d[0] for d in cursor.description or []]
            result.rows = [[v for v in row] for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            result.error = str(exc)
        result.elapsed_ms = (time.time() - start) * 1000
        return result

    def table_names(self) -> List[str]:
        rows = self._connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
