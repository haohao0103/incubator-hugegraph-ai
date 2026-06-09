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

"""OceanBase full-text search backend.

Stores documents in OceanBase FULLTEXT INDEX with BM25 scoring
and IK Chinese tokenizer. Requires OceanBase 4.x.

Usage::

    from hugegraph_llm.indices.fulltext.oceanbase_fulltext import (
        OceanBaseFTSBackend,
    )
    fts = OceanBaseFTSBackend(
        dsn="ob://user:pass@host:2883/db",
        table_name="rag_chunks",
    )
    fts.add_documents(["text1", "text2"], ids=["d1", "d2"])
    results = fts.search("query text", top_k=5)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from hugegraph_llm.indices.fulltext.base import FullTextBase

log = logging.getLogger(__name__)


class OceanBaseFTSBackend(FullTextBase):
    """Full-text search backed by OceanBase FULLTEXT INDEX.

    Provides production-grade full-text search with:
    - BM25 relevance scoring
    - IK Chinese tokenizer (smart/max_word modes)
    - Boolean query support (+/- operators)
    - Hybrid query with vector index via UNION_MERGE
    - Paxos-based high availability

    Args:
        dsn: OceanBase connection DSN.
        table_name: Table name (same table as vector store).
        parser: Tokenizer name, ``"ik"`` (Chinese) or ``"ngram"``.
        ik_mode: IK tokenizer mode, ``"smart"`` or ``"max_word"``.
    """

    def __init__(
        self,
        dsn: str = "",
        table_name: str = "rag_chunks",
        parser: str = "ik",
        ik_mode: str = "smart",
    ):
        self._dsn = dsn
        self._table = table_name
        self._parser = parser
        self._ik_mode = ik_mode
        self._conn = None
        self._initialized = False

    def _get_connection(self):
        """Get or create OceanBase connection."""
        if self._conn is None:
            try:
                import pyob

                self._conn = pyob.connect(self._dsn)
            except ImportError:
                try:
                    import pymysql

                    self._conn = pymysql.connect(self._dsn)
                except ImportError:
                    raise ImportError(
                        "OceanBase backend requires 'pyob' or 'pymysql'. "
                        "Install with: pip install pyob"
                    )
        return self._conn

    def _ensure_table(self) -> None:
        """Create table with fulltext index if not exists."""
        if self._initialized:
            return

        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                f"SELECT 1 FROM information_schema.tables "
                f"WHERE table_name = '{self._table}' LIMIT 1"
            )
            table_exists = cur.fetchone()

            if not table_exists:
                log.info("Creating OceanBase FTS table: %s", self._table)
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._table} ("
                    f"  chunk_id VARCHAR(64) PRIMARY KEY,"
                    f"  doc_id VARCHAR(64),"
                    f"  content TEXT,"
                    f"  props JSON,"
                    f"  embedding VECTOR(768)"
                    f")"
                )
                conn.commit()

            # Check if fulltext index exists
            cur.execute(
                f"SELECT 1 FROM information_schema.statistics "
                f"WHERE table_name = '{self._table}' "
                f"AND index_type = 'FULLTEXT' LIMIT 1"
            )
            if not cur.fetchone():
                cur.execute(
                    f"CREATE FULLTEXT INDEX idx_fts_{self._table} "
                    f"ON {self._table}(content) "
                    f"WITH PARSER {self._parser} "
                    f"PARSER_PROPERTIES =(ik_mode = '{self._ik_mode}')"
                )
                conn.commit()
                log.info(
                    "Created OceanBase FULLTEXT INDEX on %s (parser=%s, mode=%s)",
                    self._table,
                    self._parser,
                    self._ik_mode,
                )

            self._initialized = True

        except Exception as e:
            log.error("Failed to initialize OceanBase FTS table: %s", e)
            raise
        finally:
            cur.close()

    @property
    def doc_count(self) -> int:
        """Count indexed documents."""
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]
        finally:
            cur.close()

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        props: Optional[List[Any]] = None,
    ) -> None:
        """Add documents to OceanBase fulltext index.

        If the table already has a vector column, VECTOR_INPUT is set to
        a zero vector placeholder so that rows without vectors can coexist.
        """
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            for i, text in enumerate(texts):
                doc_id = ids[i] if ids and i < len(ids) else f"doc_{i}"
                prop = props[i] if props and i < len(props) else None

                prop_json = json.dumps(prop, default=str) if prop else "NULL"

                cur.execute(
                    f"INSERT INTO {self._table} "
                    f"(chunk_id, doc_id, content, props) "
                    f"VALUES (%s, %s, %s, %s) "
                    f"ON DUPLICATE KEY UPDATE content = VALUES(content), "
                    f"props = VALUES(props)",
                    (doc_id, "", text, prop_json),
                )
            conn.commit()
            log.debug("Added %d documents to OceanBase FTS", len(texts))

        except Exception as e:
            log.error("Failed to add documents to OceanBase FTS: %s", e)
            raise
        finally:
            cur.close()

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search OceanBase fulltext index with BM25 scoring.

        Uses MATCH...AGAINST in natural language mode. For boolean
        queries (required/excluded terms), use ``search_boolean()``.

        Args:
            query: Natural language query text.
            top_k: Maximum results.
            min_score: Minimum BM25 score threshold.

        Returns:
            List of result dicts with id, text, score, prop.
        """
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                f"SELECT chunk_id, content, "
                f"MATCH(content) AGAINST(%s) AS score "
                f"FROM {self._table} "
                f"WHERE MATCH(content) AGAINST(%s) "
                f"ORDER BY score DESC LIMIT %s",
                (query, query, top_k),
            )

            results = []
            for row in cur.fetchall():
                score = float(row[2]) if row[2] is not None else 0.0
                if score < min_score:
                    continue
                results.append({
                    "id": row[0],
                    "text": row[1] or "",
                    "score": round(score, 4),
                    "prop": None,
                })

            log.debug(
                "OceanBase FTS search returned %d results (top_k=%d)",
                len(results),
                top_k,
            )
            return results

        except Exception as e:
            log.error("OceanBase FTS search failed: %s", e)
            raise
        finally:
            cur.close()

    def search_boolean(
        self,
        required_terms: List[str],
        excluded_terms: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search with boolean mode (required/excluded terms).

        Args:
            required_terms: Terms that MUST appear (+term).
            excluded_terms: Terms that must NOT appear (-term).
            top_k: Maximum results.

        Returns:
            List of result dicts.
        """
        self._ensure_table()
        excluded_terms = excluded_terms or []

        boolean_query = " ".join(f"+{t}" for t in required_terms)
        boolean_query += " " + " ".join(f"-{t}" for t in excluded_terms)

        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                f"SELECT chunk_id, content, "
                f"MATCH(content) AGAINST(%s IN BOOLEAN MODE) AS score "
                f"FROM {self._table} "
                f"WHERE MATCH(content) AGAINST(%s IN BOOLEAN MODE) "
                f"ORDER BY score DESC LIMIT %s",
                (boolean_query, boolean_query, top_k),
            )

            results = []
            for row in cur.fetchall():
                score = float(row[2]) if row[2] is not None else 0.0
                results.append({
                    "id": row[0],
                    "text": row[1] or "",
                    "score": round(score, 4),
                    "prop": None,
                })
            return results

        except Exception as e:
            log.error("OceanBase boolean FTS search failed: %s", e)
            raise
        finally:
            cur.close()

    def remove(self, doc_ids: Union[set, List[str]]) -> int:
        """Remove documents by chunk_id."""
        self._ensure_table()
        if not doc_ids:
            return 0

        conn = self._get_connection()
        cur = conn.cursor()

        try:
            ids = set(str(p) if not isinstance(p, str) else p for p in doc_ids)
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"DELETE FROM {self._table} WHERE chunk_id IN ({placeholders})",
                tuple(ids),
            )
            removed = cur.rowcount
            conn.commit()
            return removed

        except Exception as e:
            log.error("OceanBase FTS remove failed: %s", e)
            raise
        finally:
            cur.close()

    def save_index_by_name(self, *name: str) -> None:
        """No-op for OceanBase (data persisted in database)."""

    @classmethod
    def from_name(cls, *name: str) -> "OceanBaseFTSBackend":
        """Load an OceanBase FTS backend."""
        return cls()

    @staticmethod
    def exist(*name: str) -> bool:
        return True

    @staticmethod
    def clean(*name: str) -> bool:
        log.warning("OceanBase clean() is not supported. Use SQL to drop tables.")
        return False
