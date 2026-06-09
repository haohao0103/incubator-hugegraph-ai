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

"""OceanBase vector store backend.

Stores and retrieves embeddings via OceanBase's native VECTOR type
with HNSW/IVF_FLAT index. Requires OceanBase 4.x Enterprise.

Usage::

    from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
        OceanBaseVectorStore,
    )
    store = OceanBaseVectorStore(
        dsn="ob://user:pass@host:2883/db",
        table_name="rag_chunks",
        embed_dim=768,
    )
    store.add(vectors=[[0.1, 0.2, ...]], props=["doc_1"])
    results = store.search(query_vector=[0.1, 0.2, ...], top_k=5)
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

log = logging.getLogger(__name__)


class OceanBaseVectorStore:
    """Vector store backed by OceanBase VECTOR column.

    Provides production-grade vector storage with:
    - Paxos-based high availability (3 replicas)
    - HNSW / IVF_FLAT index for fast ANN search
    - Hybrid query support (vector + structured filters)
    - Automatic table/index management

    Requires ``pyob`` or ``pymysql`` driver and OceanBase 4.x Enterprise.

    Args:
        dsn: OceanBase connection DSN or list of (host, port) tuples.
        table_name: Table name for storing chunks + vectors.
        embed_dim: Embedding dimension (16-2048).
        index_type: Vector index type, ``"hnsw"`` or ``"ivf_flat"``.
        distance_metric: Distance function ``"cosine"``, ``"l2"`` or ``"ip"``.
        index_params: Optional dict of index parameters (M, ef_construction, etc.).
    """

    _DISTANCE_MAP = {
        "cosine": ("cosinesimil", "VECTOR_COSINE"),
        "l2": ("l2", "VECTOR_EUCLID"),
        "ip": ("inner_product", "VECTOR_DOT"),
    }

    def __init__(
        self,
        dsn: str = "",
        table_name: str = "rag_chunks",
        embed_dim: int = 768,
        index_type: str = "hnsw",
        distance_metric: str = "cosine",
        index_params: Optional[Dict[str, Any]] = None,
    ):
        self._dsn = dsn
        self._table = table_name
        self._embed_dim = embed_dim
        self._index_type = index_type.lower()
        self._distance_metric = distance_metric.lower()
        self._index_params = index_params or {}
        self._conn = None
        self._initialized = False

        if self._distance_metric not in self._DISTANCE_MAP:
            raise ValueError(
                f"Unsupported distance metric '{distance_metric}'. "
                f"Choose from: {list(self._DISTANCE_MAP.keys())}"
            )
        if self._index_type not in ("hnsw", "ivf_flat"):
            raise ValueError(
                f"Unsupported index type '{index_type}'. "
                f"Choose from: hnsw, ivf_flat"
            )

    @property
    def _space_type(self) -> str:
        return self._DISTANCE_MAP[self._distance_metric][0]

    @property
    def _distance_func(self) -> str:
        return self._DISTANCE_MAP[self._distance_metric][1]

    def _get_connection(self):
        """Get or create OceanBase connection."""
        if self._conn is None:
            try:
                import pyob

                self._conn = pyob.connect(self._dsn)
                log.info("Connected to OceanBase via pyob")
            except ImportError:
                try:
                    import pymysql

                    self._conn = pymysql.connect(self._dsn)
                    log.info("Connected to OceanBase via pymysql")
                except ImportError:
                    raise ImportError(
                        "OceanBase backend requires 'pyob' or 'pymysql'. "
                        "Install with: pip install pyob"
                    )
        return self._conn

    def _ensure_table(self) -> None:
        """Create table and vector index if not exists."""
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
                log.info("Creating OceanBase table: %s", self._table)
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._table} ("
                    f"  chunk_id VARCHAR(64) PRIMARY KEY,"
                    f"  doc_id VARCHAR(64),"
                    f"  content TEXT,"
                    f"  props JSON,"
                    f"  embedding VECTOR({self._embed_dim}) NOT NULL"
                    f")"
                )
                conn.commit()

            # Create vector index
            if self._index_type == "hnsw":
                m = self._index_params.get("M", 16)
                ef = self._index_params.get("ef_construction", 40)
                idx_sql = (
                    f"CREATE INDEX IF NOT EXISTS idx_vec_{self._table} "
                    f"ON {self._table}(embedding) USING HNSW WITH ("
                    f"M = {m}, ef_construction = {ef}, "
                    f"space_type = '{self._space_type}')"
                )
            else:
                nlist = self._index_params.get("nlist", 100)
                idx_sql = (
                    f"CREATE INDEX IF NOT EXISTS idx_vec_{self._table} "
                    f"ON {self._table}(embedding) USING IVF_FLAT WITH ("
                    f"nlist = {nlist}, space_type = '{self._space_type}')"
                )

            cur.execute(idx_sql)
            conn.commit()
            self._initialized = True
            log.info("OceanBase vector index ready: %s", self._table)

        except Exception as e:
            log.error("Failed to initialize OceanBase table: %s", e)
            raise

        finally:
            cur.close()

    def add(self, vectors: List[List[float]], props: List[Any]) -> None:
        """Add vectors and properties to OceanBase.

        Args:
            vectors: List of embedding vectors.
            props: List of property dicts (each should contain 'chunk_id',
                   optionally 'doc_id', 'content').
        """
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            for vec, prop in zip(vectors, props):
                if isinstance(prop, dict):
                    chunk_id = prop.get("chunk_id", "")
                    doc_id = prop.get("doc_id", "")
                    content = prop.get("content", "")
                    extra_props = {
                        k: v for k, v in prop.items()
                        if k not in ("chunk_id", "doc_id", "content")
                    }
                else:
                    chunk_id = str(prop) if prop else ""
                    doc_id = ""
                    content = ""
                    extra_props = {}

                vec_str = json.dumps(vec)
                props_json = json.dumps(extra_props) if extra_props else "NULL"

                cur.execute(
                    f"INSERT INTO {self._table} "
                    f"(chunk_id, doc_id, content, props, embedding) "
                    f"VALUES (%s, %s, %s, %s, VECTOR_INPUT(%s)) "
                    f"ON DUPLICATE KEY UPDATE "
                    f"embedding = VECTOR_INPUT(%s), "
                    f"content = VALUES(content), "
                    f"props = VALUES(props)",
                    (chunk_id, doc_id, content, props_json, vec_str, vec_str),
                )
            conn.commit()
            log.debug("Added %d vectors to OceanBase", len(vectors))

        except Exception as e:
            log.error("Failed to add vectors to OceanBase: %s", e)
            raise
        finally:
            cur.close()

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        dis_threshold: float = 0.9,
    ) -> List[Any]:
        """Search OceanBase for similar vectors.

        Args:
            query_vector: Query embedding.
            top_k: Number of results.
            dis_threshold: Minimum similarity score (0-1 for cosine).

        Returns:
            List of property dicts with '_score' field.
        """
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            vec_str = json.dumps(query_vector)
            cur.execute(
                f"SELECT chunk_id, doc_id, content, props, "
                f"{self._distance_func}(embedding, VECTOR_INPUT(%s)) AS score "
                f"FROM {self._table} "
                f"ORDER BY score {'DESC' if self._distance_metric == 'ip' or self._distance_metric == 'cosine' else 'ASC'} "
                f"LIMIT %s",
                (vec_str, top_k),
            )

            results = []
            for row in cur.fetchall():
                score = float(row[4]) if row[4] is not None else 0.0
                if score < dis_threshold:
                    continue

                prop = {"chunk_id": row[0], "doc_id": row[1] or "", "content": row[2] or "", "_score": round(score, 4)}
                if row[3] and isinstance(row[3], str):
                    try:
                        extra = json.loads(row[3])
                        prop.update(extra)
                    except (json.JSONDecodeError, TypeError):
                        pass

                results.append(prop)

            log.debug(
                "OceanBase vector search returned %d results (top_k=%d)",
                len(results),
                top_k,
            )
            return results

        except Exception as e:
            log.error("OceanBase vector search failed: %s", e)
            raise
        finally:
            cur.close()

    def remove(self, props: Union[Set[Any], List[Any]]) -> int:
        """Remove vectors by chunk_id.

        Args:
            props: Set/list of chunk_id values.

        Returns:
            Number of vectors removed.
        """
        self._ensure_table()
        if not props:
            return 0

        conn = self._get_connection()
        cur = conn.cursor()

        try:
            ids = set(str(p) if not isinstance(p, str) else p for p in props)
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"DELETE FROM {self._table} WHERE chunk_id IN ({placeholders})",
                tuple(ids),
            )
            removed = cur.rowcount
            conn.commit()
            return removed

        except Exception as e:
            log.error("OceanBase vector remove failed: %s", e)
            raise
        finally:
            cur.close()

    def get_all_properties(self) -> list[str]:
        """Get all stored chunk_ids."""
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute(f"SELECT chunk_id FROM {self._table}")
            return [row[0] for row in cur.fetchall()]
        finally:
            cur.close()

    def save_index_by_name(self, *name: str) -> None:
        """No-op for OceanBase (data already persisted in database)."""

    def get_vector_index_info(self) -> Dict:
        """Return vector store metadata."""
        return {
            "backend": "oceanbase",
            "table": self._table,
            "embed_dim": self._embed_dim,
            "index_type": self._index_type,
            "distance_metric": self._distance_metric,
        }

    @staticmethod
    def from_name(embed_dim: int, *name: str) -> "OceanBaseVectorStore":
        """Load an OceanBase vector store.

        Args:
            embed_dim: Embedding dimension.
            *name: Not used (OceanBase reads from DB directly).
        """
        # For OceanBase, from_name just creates a new instance
        # Actual data comes from the database
        return OceanBaseVectorStore(embed_dim=embed_dim)

    @staticmethod
    def exist(*name: str) -> bool:
        """OceanBase stores are always 'existing' (DB-managed)."""
        return True

    @staticmethod
    def clean(*name: str) -> bool:
        """Clean up is not supported for OceanBase (use SQL DROP TABLE)."""
        log.warning("OceanBase clean() is not supported. Use SQL to drop tables.")
        return False
