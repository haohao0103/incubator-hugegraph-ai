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

"""Storage interface stubs for future service-ization.

Inspired by LightRAG's storage layer (12 backends: 3 vector + 7 KV + 1 graph
+ 1 DocStatus) and HippoRAG2 PR#184's pluggable vector store pattern.

These are INTERFACE DEFINITIONS only — concrete implementations will be
added in Phase 2 (GraphRAG Engine service-ization). Current Phase 1
operators continue using direct FAISS/BM25/HugeGraph connections.

Design principles:
- ABC + Factory pattern (borrowed from HippoRAG2's BaseEmbeddingStore)
- Lazy import for optional dependencies (borrowed from HippoRAG2)
- Default implementations: JsonFile for KV, SQLite for DocStatus
- HugeGraph remains the sole graph backend (our differentiation)

Usage (future):
    from hugegraph_llm.operators.graph_op.storage_interfaces import (
        BaseKVStorage, KVStorageFactory,
        BaseDocStatusStorage, DocStatusFactory,
    )

    kv = KVStorageFactory.create("json_file", working_dir="./data")
    kv.set("key", "value")
    result = kv.get("key")
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.utils.log import log


# ═══════════════════════════════════════════════════════════════════════
# KV Storage Interface
# ═══════════════════════════════════════════════════════════════════════

class BaseKVStorage(ABC):
    """Abstract base class for key-value storage.

    Inspired by LightRAG's JsonFileStorage (default) and HippoRAG2's
    BaseEmbeddingStore pattern.

    Minimal interface: get, set, delete, keys, clear.
    Async variants will be added in Phase 2.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Retrieve a value by key. Returns None if not found."""
        ...

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store a key-value pair. Overwrites if key exists."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed and was deleted."""
        ...

    @abstractmethod
    def keys(self) -> List[str]:
        """Return all stored keys."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all key-value pairs."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Return the number of stored key-value pairs."""
        ...


class JsonFileKVStorage(BaseKVStorage):
    """JSON file-based KV storage — LightRAG's default pattern.

    Stores all key-value pairs in a single JSON file.
    Thread-safe via file-level atomic write (write temp → rename).
    """

    def __init__(self, working_dir: str = ".", namespace: str = "kv_store"):
        self._path = Path(working_dir) / f"{namespace}.json"
        self._data: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load data from JSON file."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("JsonFileKVStorage: failed to load %s: %s", self._path, e)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Atomic write: write to temp file, then rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)
        except OSError as e:
            log.error("JsonFileKVStorage: failed to save %s: %s", self._path, e)

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()
        self._save()

    def size(self) -> int:
        return len(self._data)


class InMemoryKVStorage(BaseKVStorage):
    """In-memory KV storage for testing and ephemeral use."""

    def __init__(self):
        self._data: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()

    def size(self) -> int:
        return len(self._data)


class KVStorageFactory:
    """Factory for creating KV storage instances.

    Inspired by HippoRAG2's factory pattern with lazy import
    for optional backends (Redis, MongoDB, etc.).
    """

    _BACKENDS = {
        "json_file": JsonFileKVStorage,
        "memory": InMemoryKVStorage,
    }

    @classmethod
    def create(
        cls,
        backend: str = "json_file",
        **kwargs: Any,
    ) -> BaseKVStorage:
        """Create a KV storage instance.

        Args:
            backend: Backend type ("json_file", "memory", "redis", "mongodb").
            **kwargs: Backend-specific configuration.

        Returns:
            BaseKVStorage instance.

        Raises:
            ValueError: If backend is not supported.
        """
        if backend in cls._BACKENDS:
            return cls._BACKENDS[backend](**kwargs)

        # Lazy import for optional backends — fallback to json_file if not yet implemented
        if backend == "redis":
            try:
                # Future: RedisKVStorage
                raise NotImplementedError("Redis KV backend not yet implemented")
            except (ImportError, NotImplementedError) as e:
                log.warning("Redis KV backend not available (%s), falling back to json_file", e)
                return JsonFileKVStorage(**kwargs)

        if backend == "mongodb":
            try:
                # Future: MongoDBKVStorage
                raise NotImplementedError("MongoDB KV backend not yet implemented")
            except (ImportError, NotImplementedError) as e:
                log.warning("MongoDB KV backend not available (%s), falling back to json_file", e)
                return JsonFileKVStorage(**kwargs)

        raise ValueError(f"Unknown KV backend: {backend}. Available: {list(cls._BACKENDS.keys())}")


# ═══════════════════════════════════════════════════════════════════════
# Doc Status Storage Interface
# ═══════════════════════════════════════════════════════════════════════

class DocStatus(Enum):
    """Document processing status lifecycle.

    Inspired by LightRAG's document pipeline status tracking.
    """
    PENDING = "PENDING"
    PARSING = "PARSING"
    ANALYZING = "ANALYZING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


@dataclass
class DocStatusRecord:
    """Record tracking a document's processing lifecycle."""
    doc_id: str
    file_path: str
    status: DocStatus = DocStatus.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0
    error_message: str = ""
    chunks_count: int = 0
    entities_count: int = 0
    relations_count: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at


class BaseDocStatusStorage(ABC):
    """Abstract base for document status tracking.

    Inspired by LightRAG's DocStatus storage (PostgreSQL/SQLite/JsonFile).
    """

    @abstractmethod
    def get(self, doc_id: str) -> Optional[DocStatusRecord]:
        """Get a document's status record."""
        ...

    @abstractmethod
    def upsert(self, record: DocStatusRecord) -> None:
        """Insert or update a document status record."""
        ...

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete a document status record."""
        ...

    @abstractmethod
    def get_by_status(self, status: DocStatus) -> List[DocStatusRecord]:
        """Get all records with a specific status."""
        ...

    @abstractmethod
    def get_pending(self) -> List[DocStatusRecord]:
        """Get all PENDING documents for processing."""
        ...

    @abstractmethod
    def count_by_status(self) -> Dict[str, int]:
        """Count documents per status."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all records."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Total number of tracked documents."""
        ...


class SQLiteDocStatusStorage(BaseDocStatusStorage):
    """SQLite-based doc status storage — LightRAG's default pattern.

    Uses a single SQLite table for lightweight status tracking.
    """

    def __init__(self, working_dir: str = ".", db_name: str = "doc_status"):
        self._path = Path(working_dir) / f"{db_name}.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_status (
                doc_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                error_message TEXT DEFAULT '',
                chunks_count INTEGER DEFAULT 0,
                entities_count INTEGER DEFAULT 0,
                relations_count INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def get(self, doc_id: str) -> Optional[DocStatusRecord]:
        row = self._conn.execute(
            "SELECT * FROM doc_status WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return DocStatusRecord(
            doc_id=row["doc_id"],
            file_path=row["file_path"],
            status=DocStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error_message=row["error_message"],
            chunks_count=row["chunks_count"],
            entities_count=row["entities_count"],
            relations_count=row["relations_count"],
        )

    def upsert(self, record: DocStatusRecord) -> None:
        record.updated_at = time.time()
        self._conn.execute("""
            INSERT OR REPLACE INTO doc_status
            (doc_id, file_path, status, created_at, updated_at,
             error_message, chunks_count, entities_count, relations_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.doc_id, record.file_path, record.status.value,
            record.created_at, record.updated_at, record.error_message,
            record.chunks_count, record.entities_count, record.relations_count,
        ))
        self._conn.commit()

    def delete(self, doc_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM doc_status WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_by_status(self, status: DocStatus) -> List[DocStatusRecord]:
        rows = self._conn.execute(
            "SELECT * FROM doc_status WHERE status = ?", (status.value,)
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_pending(self) -> List[DocStatusRecord]:
        return self.get_by_status(DocStatus.PENDING)

    def count_by_status(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM doc_status GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def clear(self) -> None:
        self._conn.execute("DELETE FROM doc_status")
        self._conn.commit()

    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM doc_status").fetchone()
        return row["cnt"]

    def _row_to_record(self, row: sqlite3.Row) -> DocStatusRecord:
        return DocStatusRecord(
            doc_id=row["doc_id"],
            file_path=row["file_path"],
            status=DocStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error_message=row["error_message"],
            chunks_count=row["chunks_count"],
            entities_count=row["entities_count"],
            relations_count=row["relations_count"],
        )

    def close(self) -> None:
        self._conn.close()


class InMemoryDocStatusStorage(BaseDocStatusStorage):
    """In-memory doc status storage for testing."""

    def __init__(self):
        self._records: Dict[str, DocStatusRecord] = {}

    def get(self, doc_id: str) -> Optional[DocStatusRecord]:
        return self._records.get(doc_id)

    def upsert(self, record: DocStatusRecord) -> None:
        record.updated_at = time.time()
        self._records[record.doc_id] = record

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._records:
            del self._records[doc_id]
            return True
        return False

    def get_by_status(self, status: DocStatus) -> List[DocStatusRecord]:
        return [r for r in self._records.values() if r.status == status]

    def get_pending(self) -> List[DocStatusRecord]:
        return self.get_by_status(DocStatus.PENDING)

    def count_by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._records.values():
            s = r.status.value
            counts[s] = counts.get(s, 0) + 1
        return counts

    def clear(self) -> None:
        self._records.clear()

    def size(self) -> int:
        return len(self._records)


class DocStatusFactory:
    """Factory for creating DocStatus storage instances."""

    _BACKENDS = {
        "sqlite": SQLiteDocStatusStorage,
        "memory": InMemoryDocStatusStorage,
    }

    @classmethod
    def create(
        cls,
        backend: str = "sqlite",
        **kwargs: Any,
    ) -> BaseDocStatusStorage:
        if backend in cls._BACKENDS:
            return cls._BACKENDS[backend](**kwargs)

        if backend == "json_file":
            # Future: JsonFileDocStatusStorage
            log.warning("json_file DocStatus not implemented, falling back to sqlite")
            return SQLiteDocStatusStorage(**kwargs)

        if backend == "postgresql":
            # Future: PostgreSQLDocStatusStorage
            raise NotImplementedError("PostgreSQL DocStatus not yet implemented")

        raise ValueError(f"Unknown DocStatus backend: {backend}. Available: {list(cls._BACKENDS.keys())}")
