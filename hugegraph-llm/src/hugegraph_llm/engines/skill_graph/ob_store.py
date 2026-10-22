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

"""OceanBase persistence for the skill graph (the vector/fulltext substrate).

OceanBase speaks both dialects the engine needs in one engine:

* a **vector** column with an HNSW index (``cosine_distance``), used for the
  semantic seed channel;
* a **fulltext** column with the ``ik`` parser, used for the lexical channel.

Every table this store touches lives under a dedicated ``skill_graph_`` prefix,
so the demo is an isolated namespace inside a shared database and never
collides with other tables (e.g. the nl2sql set).

Rows mirror the project-wide "semantic row" shape:
``node_id`` (PK) + ``embedding`` + ``surface_text`` + metadata.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from hugegraph_llm.engines.skill_graph.schema import SkillNode

log = logging.getLogger(__name__)

DEFAULT_TABLE = "skill_graph_skill_semantic"

# Table names are interpolated into DDL/DML (identifiers cannot be bound as
# parameters), including the base name from ``--ob-table-base`` and the stored
# manifest. Restrict them to a plain identifier so a crafted value cannot break
# out of the backticks.
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _vec_literal(vec: Sequence[float]) -> str:
    """OceanBase accepts vector values as ``'[v1,v2,...]'`` string literals."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


@dataclass
class SemanticHit:
    """A single hit from either the vector or the fulltext channel."""

    node_id: str
    title: str
    surface_text: str
    score: float
    channel: str  # vector | fulltext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "score": round(self.score, 6),
            "channel": self.channel,
        }


class OceanBaseSkillStore:
    """Thin OceanBase adapter for skill vectors + surface text.

    Connection settings default to ``OCEANBASE_*`` environment variables so the
    demo can stay configuration-free::

        OCEANBASE_HOST / OCEANBASE_PORT / OCEANBASE_USER
        OCEANBASE_PASSWORD / OCEANBASE_DATABASE
        OCEANBASE_TABLE (default ``skill_graph_skill_semantic``)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        table: Optional[str] = None,
        dim: int = 1536,
    ):
        self.host = host or os.environ.get("OCEANBASE_HOST", "")
        self.port = int(port or os.environ.get("OCEANBASE_PORT", "3306"))
        self.user = user or os.environ.get("OCEANBASE_USER", "")
        self.password = password if password is not None else os.environ.get("OCEANBASE_PASSWORD", "")
        self.database = database or os.environ.get("OCEANBASE_DATABASE", "")
        self.table = table or os.environ.get("OCEANBASE_TABLE", DEFAULT_TABLE)
        if not _TABLE_RE.match(self.table):
            raise ValueError(
                f"invalid OceanBase table name {self.table!r}; "
                "use letters, digits and underscores only"
            )
        self.dim = dim
        if not self.host or not self.user or not self.database:
            raise ValueError(
                "OceanBaseSkillStore needs host/user/database (pass args or set OCEANBASE_* env vars)"
            )
        self._conn = None

    # ── connection ───────────────────────────────────────────
    @property
    def connection(self):
        if self._conn is None:
            import pymysql

            self._conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                autocommit=True,
                connect_timeout=15,
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── schema ───────────────────────────────────────────────
    def ensure_schema(self) -> None:
        """Create the isolated table (idempotent). Existing data is untouched."""
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{self.table}` (
          `node_id` varchar(512) NOT NULL,
          `embedding` VECTOR({self.dim}) DEFAULT NULL,
          `surface_text` text DEFAULT NULL,
          `content` longtext DEFAULT NULL,
          `entity_type` varchar(128) DEFAULT NULL,
          `entity_id` varchar(512) DEFAULT NULL,
          `title` varchar(512) DEFAULT NULL,
          `content_hash` varchar(64) DEFAULT NULL,
          `is_valid` tinyint(4) NOT NULL DEFAULT 1,
          PRIMARY KEY (`node_id`),
          FULLTEXT KEY `ftx` (`surface_text`) WITH PARSER ik PARSER_PROPERTIES=(ik_mode="smart") BLOCK_SIZE 16384,
          VECTOR KEY `vidx` (`embedding`) WITH (DISTANCE=COSINE, TYPE=HNSW, LIB=VSAG,
            M=16, EF_CONSTRUCTION=200, EF_SEARCH=64) BLOCK_SIZE 16384
        ) ORGANIZATION INDEX DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_general_ci
        """
        with self.connection.cursor() as cur:
            cur.execute(ddl)
        log.info("ensured OceanBase table %s", self.table)

    def count(self) -> int:
        with self.connection.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{self.table}` WHERE is_valid = 1")
            return int(cur.fetchone()[0])

    # ── writes ───────────────────────────────────────────────
    def upsert_nodes(self, nodes: Sequence[SkillNode], batch_size: int = 50) -> int:
        """Insert/replace skill rows (embedding + surface text + body)."""
        rows = []
        for node in nodes:
            if not node.embedding:
                continue
            surface = node.index_text
            rows.append(
                (
                    node.id,
                    _vec_literal(node.embedding),
                    surface,
                    node.body or "",
                    node.node_type.value,
                    node.id,
                    node.name,
                    hashlib.md5(surface.encode("utf-8")).hexdigest(),
                    1,
                )
            )
        sql = (
            f"INSERT INTO `{self.table}` "
            "(node_id, embedding, surface_text, content, entity_type, entity_id, title, content_hash, is_valid) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE embedding=VALUES(embedding), surface_text=VALUES(surface_text), "
            "content=VALUES(content), entity_type=VALUES(entity_type), entity_id=VALUES(entity_id), "
            "title=VALUES(title), content_hash=VALUES(content_hash), is_valid=1"
        )
        written = 0
        with self.connection.cursor() as cur:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                cur.executemany(sql, batch)
                written += len(batch)
        log.info("upserted %d skill rows into %s", written, self.table)
        return written

    # ── reads ────────────────────────────────────────────────
    def vector_search(self, query_embedding: Sequence[float], top_k: int = 5) -> List[SemanticHit]:
        """Nearest skills by cosine distance (HNSW index)."""
        sql = (
            f"SELECT node_id, title, surface_text, cosine_distance(embedding, %s) AS dist "
            f"FROM `{self.table}` WHERE is_valid = 1 "
            "ORDER BY dist ASC LIMIT %s"
        )
        with self.connection.cursor() as cur:
            cur.execute(sql, (_vec_literal(query_embedding), top_k))
            rows = cur.fetchall()
        return [
            SemanticHit(node_id=r[0], title=r[1], surface_text=r[2], score=1.0 - float(r[3]), channel="vector")
            for r in rows
        ]

    def fulltext_search(self, query: str, top_k: int = 5) -> List[SemanticHit]:
        """Lexical skills via the ik fulltext index (BM25-style relevance)."""
        sql = (
            f"SELECT node_id, title, surface_text, MATCH(surface_text) AGAINST(%s) AS score "
            f"FROM `{self.table}` WHERE is_valid = 1 AND MATCH(surface_text) AGAINST(%s) "
            "ORDER BY score DESC LIMIT %s"
        )
        with self.connection.cursor() as cur:
            cur.execute(sql, (query, query, top_k))
            rows = cur.fetchall()
        return [
            SemanticHit(node_id=r[0], title=r[1], surface_text=r[2], score=float(r[3]), channel="fulltext")
            for r in rows
        ]

    def hybrid_seed_search(
        self,
        query: str,
        query_embedding: Optional[Sequence[float]] = None,
        per_channel: int = 50,
        limit: int = 100,
        rrf_k: int = 60,
    ) -> List[SemanticHit]:
        """Return bounded seed candidates from both channels, fused with RRF.

        This is the query-path counterpart of the bulk ``read everything`` load:
        each channel is capped at ``per_channel`` rows *inside* OceanBase and
        only the fused top ``limit`` ids leave the database, so query cost does
        not grow with the number of skills.
        """
        if per_channel <= 0 or limit <= 0:
            return []
        text_hits = self.fulltext_search(query, top_k=per_channel) if query.strip() else []
        vector_hits = (
            self.vector_search(query_embedding, top_k=per_channel) if query_embedding else []
        )
        if not text_hits and not vector_hits:
            return []

        metadata: Dict[str, SemanticHit] = {}
        rankings: List[List[str]] = []
        for hits in (text_hits, vector_hits):
            ranking: List[str] = []
            for hit in hits:
                metadata.setdefault(hit.node_id, hit)
                ranking.append(hit.node_id)
            rankings.append(ranking)

        fused: Dict[str, float] = {}
        for ranking in rankings:
            for rank, node_id in enumerate(ranking):
                fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        ordered = sorted(fused, key=lambda nid: (-fused[nid], nid))[:limit]
        return [
            SemanticHit(
                node_id=nid,
                title=metadata[nid].title,
                surface_text=metadata[nid].surface_text,
                score=fused[nid],
                channel="hybrid",
            )
            for nid in ordered
        ]

    def fetch_bodies(self, node_ids: Sequence[str], batch_size: int = 200) -> Dict[str, str]:
        """Fetch bodies for ``node_ids`` only (bounded hydration, PK lookups)."""
        ids = list(dict.fromkeys(str(nid) for nid in node_ids))
        bodies: Dict[str, str] = {}
        if not ids:
            return bodies
        with self.connection.cursor() as cur:
            for start in range(0, len(ids), batch_size):
                chunk = ids[start : start + batch_size]
                placeholders = ",".join(["%s"] * len(chunk))
                cur.execute(
                    f"SELECT node_id, content FROM `{self.table}` "
                    f"WHERE is_valid = 1 AND node_id IN ({placeholders})",
                    tuple(chunk),
                )
                for node_id, content in cur.fetchall():
                    bodies[str(node_id)] = str(content or "")
        return bodies


def write_to_oceanbase(
    graph: Any,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    table: Optional[str] = None,
    dim: Optional[int] = None,
) -> Dict[str, int]:
    """Persist every embedded node's vector + surface text into OceanBase.

    Creates the isolated ``skill_graph_`` table on demand and upserts one row per
    skill. Nodes without an embedding (e.g. category hubs) are skipped. The vector
    dimension is inferred from the first embedding unless ``dim`` is given.
    """
    nodes = [n for n in graph.nodes.values() if n.embedding]
    if not nodes:
        raise ValueError("graph has no embeddings; build with an embedder before writing to OceanBase")
    resolved_dim = dim or len(nodes[0].embedding)
    store = OceanBaseSkillStore(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        table=table,
        dim=resolved_dim,
    )
    try:
        store.ensure_schema()
        written = store.upsert_nodes(nodes)
    finally:
        store.close()
    return {"rows": written, "dim": resolved_dim}