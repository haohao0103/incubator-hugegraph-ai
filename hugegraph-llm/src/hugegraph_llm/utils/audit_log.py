# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Audit log for memory operations.

Records every add/search/update/delete event to a SQLite database so that
production deployments can trace memory usage, debug retrieval failures, and
meet compliance requirements. The design mirrors the telemetry/audit layers
found in Mem0 and PowerMem.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditEvent:
    """One audit event describing a memory operation."""
    id: str
    timestamp: float
    operation: str
    user_id: str
    memory_id: Optional[str]
    query: Optional[str]
    content: Optional[str]
    latency_ms: float
    success: bool
    error: Optional[str]
    metadata: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Thread-safe SQLite-backed audit logger for memory operations."""

    _instances: Dict[str, "AuditLogger"] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None) -> "AuditLogger":
        db_path = cls._resolve_path(db_path)
        with cls._lock:
            if db_path not in cls._instances:
                instance = super().__new__(cls)
                instance._db_path = db_path
                instance._local = threading.local()
                instance._init_schema()
                cls._instances[db_path] = instance
            return cls._instances[db_path]

    @classmethod
    def _resolve_path(cls, db_path: Optional[str]) -> str:
        if db_path:
            return str(db_path)
        env_path = os.environ.get("MEMORY_AUDIT_LOG_PATH")
        if env_path:
            return env_path
        default_dir = Path.cwd() / "poc_data"
        default_dir.mkdir(parents=True, exist_ok=True)
        return str(default_dir / "memory_audit.db")

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> AuditEvent:
        return AuditEvent(
            id=row["id"],
            timestamp=row["timestamp"],
            operation=row["operation"],
            user_id=row["user_id"],
            memory_id=row["memory_id"],
            query=row["query"],
            content=row["content"],
            latency_ms=row["latency_ms"],
            success=bool(row["success"]),
            error=row["error"],
            metadata=row["metadata"],
        )

    @contextmanager
    def _connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_audit (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    memory_id TEXT,
                    query TEXT,
                    content TEXT,
                    latency_ms REAL DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    error TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_user_time ON memory_audit(user_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_operation ON memory_audit(operation, timestamp)"
            )
            conn.commit()

    def log(
        self,
        operation: str,
        user_id: str = "demo_user",
        memory_id: Optional[str] = None,
        query: Optional[str] = None,
        content: Optional[str] = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Record a single audit event."""
        event = AuditEvent(
            id=str(uuid.uuid4())[:12],
            timestamp=time.time(),
            operation=operation,
            user_id=user_id,
            memory_id=memory_id,
            query=query,
            content=content,
            latency_ms=latency_ms,
            success=success,
            error=error,
            metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_audit
                (id, timestamp, operation, user_id, memory_id, query, content,
                 latency_ms, success, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp,
                    event.operation,
                    event.user_id,
                    event.memory_id,
                    event.query,
                    event.content,
                    event.latency_ms,
                    1 if event.success else 0,
                    event.error,
                    event.metadata,
                ),
            )
            conn.commit()
        return event

    def get_events(
        self,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Retrieve audit events with optional filtering."""
        with self._connection() as conn:
            sql = "SELECT * FROM memory_audit WHERE 1=1"
            params: List[Any] = []
            if user_id:
                sql += " AND user_id=?"
                params.append(user_id)
            if operation:
                sql += " AND operation=?"
                params.append(operation)
            sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_event(row) for row in rows]

    def count(self, user_id: Optional[str] = None, operation: Optional[str] = None) -> int:
        """Count audit events matching the filters."""
        with self._connection() as conn:
            sql = "SELECT COUNT(*) FROM memory_audit WHERE 1=1"
            params: List[Any] = []
            if user_id:
                sql += " AND user_id=?"
                params.append(user_id)
            if operation:
                sql += " AND operation=?"
                params.append(operation)
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0

    def get_stats(self) -> Dict[str, Any]:
        """Return high-level audit statistics."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memory_audit").fetchone()[0]
            success = conn.execute(
                "SELECT COUNT(*) FROM memory_audit WHERE success=1"
            ).fetchone()[0]
            ops = conn.execute(
                "SELECT operation, COUNT(*) FROM memory_audit GROUP BY operation"
            ).fetchall()
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) FROM memory_audit"
            ).fetchone()[0]
            return {
                "total_events": total,
                "successful_events": success,
                "failed_events": total - success,
                "avg_latency_ms": round(avg_latency or 0, 2),
                "operations": {op: count for op, count in ops},
            }

    def clear(self) -> int:
        """Delete all audit events. Returns the number of rows deleted."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM memory_audit")
            conn.commit()
            return cursor.rowcount


def get_audit_logger(db_path: Optional[str] = None) -> AuditLogger:
    """Factory helper for the default audit logger."""
    return AuditLogger(db_path)
