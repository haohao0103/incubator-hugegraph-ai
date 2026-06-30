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
Memory History Tracking — aligned with mem0's SQLite history table.

Tracks the full edit history of every memory: ADD, UPDATE, DELETE events
with old/new content snapshots, timestamps, and actor/role metadata.

This replaces our audit_log which only tracked operations without content diffs.
Now both audit_log (operational metrics) and MemoryHistory (content diffs) exist
as complementary tracking layers.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


@dataclass
class HistoryEvent:
    """A single history event for a memory (mem0-style)."""
    id: str
    memory_id: str
    event: str  # "ADD", "UPDATE", "DELETE"
    old_memory: Optional[str] = None
    new_memory: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    is_deleted: bool = False
    actor_id: Optional[str] = None
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryHistoryTracker:
    """SQLite-backed memory history tracker (mem0-style).

    Features:
      - ADD/UPDATE/DELETE event tracking with old/new content
      - Per-memory version history (history() returns full timeline)
      - Batch insert for efficient bulk operations
      - Thread-safe with per-thread SQLite connections
      - Auto-migration of schema if needed
    """

    _instances: Dict[str, "MemoryHistoryTracker"] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None) -> "MemoryHistoryTracker":
        """Singleton pattern per db_path."""
        path = db_path or cls._default_path()
        with cls._lock:
            if path not in cls._instances:
                instance = super().__new__(cls)
                instance._db_path = path
                instance._local = threading.local()
                instance._init_db()
                cls._instances[path] = instance
            return cls._instances[path]

    @staticmethod
    def _default_path() -> str:
        import os
        base = os.environ.get("MEMORY_DATA_DIR", os.path.join(os.getcwd(), "poc_data"))
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "memory_history.db")

    def _init_db(self) -> None:
        """Create tables and indexes."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_history (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                event TEXT NOT NULL,
                old_memory TEXT,
                new_memory TEXT,
                created_at REAL,
                updated_at REAL,
                is_deleted INTEGER DEFAULT 0,
                actor_id TEXT,
                role TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_memory_id
            ON memory_history(memory_id, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_event
            ON memory_history(event, created_at)
        """)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    # ── Core API ────────────────────────────────────────────

    def add_history(
        self,
        memory_id: str,
        event: str,
        old_memory: Optional[str] = None,
        new_memory: Optional[str] = None,
        *,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        is_deleted: bool = False,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEvent:
        """Record a history event for a memory.

        Args:
            memory_id: The memory's unique ID
            event: "ADD", "UPDATE", or "DELETE"
            old_memory: Previous content (for UPDATE/DELETE)
            new_memory: New content (for ADD/UPDATE)
            created_at: Event timestamp (defaults to now)
            actor_id: Who triggered the event (user_id or agent_id)
            role: Actor role ("user", "assistant", "system")
            metadata: Additional event metadata

        Returns:
            HistoryEvent dataclass
        """
        now = time.time()
        evt = HistoryEvent(
            id=uuid.uuid4().hex[:16],
            memory_id=memory_id,
            event=event,
            old_memory=old_memory,
            new_memory=new_memory,
            created_at=created_at or now,
            updated_at=updated_at or now,
            is_deleted=is_deleted,
            actor_id=actor_id,
            role=role,
            metadata=metadata,
        )
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO memory_history
               (id, memory_id, event, old_memory, new_memory,
                created_at, updated_at, is_deleted, actor_id, role, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evt.id,
                evt.memory_id,
                evt.event,
                evt.old_memory,
                evt.new_memory,
                evt.created_at,
                evt.updated_at,
                int(evt.is_deleted),
                evt.actor_id,
                evt.role,
                json.dumps(evt.metadata) if evt.metadata else None,
            ),
        )
        conn.commit()
        return evt

    def batch_add_history(self, records: List[Dict[str, Any]]) -> None:
        """Batch insert history events (for efficient bulk operations).

        Each record dict should have the same keys as add_history() parameters.
        """
        conn = self._get_conn()
        now = time.time()
        rows = []
        for rec in records:
            rows.append((
                uuid.uuid4().hex[:16],
                rec.get("memory_id", ""),
                rec.get("event", "ADD"),
                rec.get("old_memory"),
                rec.get("new_memory"),
                rec.get("created_at", now),
                rec.get("updated_at", now),
                int(rec.get("is_deleted", False)),
                rec.get("actor_id"),
                rec.get("role"),
                json.dumps(rec.get("metadata")) if rec.get("metadata") else None,
            ))
        conn.executemany(
            """INSERT INTO memory_history
               (id, memory_id, event, old_memory, new_memory,
                created_at, updated_at, is_deleted, actor_id, role, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()

    def get_history(self, memory_id: str) -> List[HistoryEvent]:
        """Return full version history of a memory (mem0-style history())."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM memory_history
               WHERE memory_id = ?
               ORDER BY created_at ASC""",
            (memory_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_recent_history(
        self,
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
    ) -> List[HistoryEvent]:
        """Query recent history events across all memories."""
        conn = self._get_conn()
        if event_type:
            rows = conn.execute(
                """SELECT * FROM memory_history
                   WHERE event = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (event_type, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM memory_history
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_events(
        self,
        memory_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> int:
        """Count history events."""
        conn = self._get_conn()
        if memory_id and event_type:
            return conn.execute(
                "SELECT COUNT(*) FROM memory_history WHERE memory_id=? AND event=?",
                (memory_id, event_type),
            ).fetchone()[0]
        elif memory_id:
            return conn.execute(
                "SELECT COUNT(*) FROM memory_history WHERE memory_id=?",
                (memory_id,),
            ).fetchone()[0]
        elif event_type:
            return conn.execute(
                "SELECT COUNT(*) FROM memory_history WHERE event=?",
                (event_type,),
            ).fetchone()[0]
        else:
            return conn.execute(
                "SELECT COUNT(*) FROM memory_history"
            ).fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Return history statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
        adds = conn.execute(
            "SELECT COUNT(*) FROM memory_history WHERE event='ADD'"
        ).fetchone()[0]
        updates = conn.execute(
            "SELECT COUNT(*) FROM memory_history WHERE event='UPDATE'"
        ).fetchone()[0]
        deletes = conn.execute(
            "SELECT COUNT(*) FROM memory_history WHERE event='DELETE'"
        ).fetchone()[0]
        unique_memories = conn.execute(
            "SELECT COUNT(DISTINCT memory_id) FROM memory_history"
        ).fetchone()[0]
        return {
            "total_events": total,
            "add_events": adds,
            "update_events": updates,
            "delete_events": deletes,
            "unique_memories": unique_memories,
        }

    def clear(self) -> int:
        """Clear all history records."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
        conn.execute("DELETE FROM memory_history")
        conn.commit()
        return count

    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Internal ────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> HistoryEvent:
        """Convert a SQLite row to HistoryEvent."""
        return HistoryEvent(
            id=row["id"],
            memory_id=row["memory_id"],
            event=row["event"],
            old_memory=row["old_memory"],
            new_memory=row["new_memory"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_deleted=bool(row["is_deleted"]),
            actor_id=row["actor_id"],
            role=row["role"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )
