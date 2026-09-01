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
Vector store abstraction for NL2SQL semantic (P2) schema linking.

Schema linking only needs two operations: load the node embeddings once and
ask for the nearest neighbours of a question. That is deliberately smaller
than the RAG subsystem's :class:`~hugegraph_llm.indices.vector_index.base.
VectorStoreBase` (which also does save/load/clean/info and returns *properties
only*, without similarity scores). P2 seed weighting needs the similarity, so
the NL2SQL layer defines its own minimal contract::

    store.upsert(ids, vectors)                 # one-shot / incremental build
    store.search(query, top_k) -> [(id, sim)]  # best first, sim in [0, 1]

Implementations
---------------
``NumpySchemaVectorStore``
    In-process default. Exact cosine similarity, unit-normalised rows built
    once. This is the reference implementation the stress test is pinned
    against; identical numbers to the pre-abstraction inline matrix.
``MilvusSchemaVectorStore``
    Direct pymilvus 3.x ``MilvusClient`` with ``COSINE`` metric; the returned
    ``distance`` *is* the cosine similarity. Collection is auto-created
    (HNSW by default). No server running -> operations raise, and the linker
    degrades to lexical linking.
``OceanBaseSchemaVectorStore``
    Direct OceanBase 4.x via ``pymysql`` using a native ``VECTOR`` column with
    an HNSW cosine index and ``cosinesimil(...) AS score`` queries. Follows
    the SQL patterns of ``indices/vector_index/oceanbase_vector_store.py``.
``LegacyVectorStoreAdapter``
    Lets the linker reuse any existing RAG ``VectorStoreBase`` (Faiss / Milvus
    / Qdrant / OceanBase from ``hugegraph_llm.indices.vector_index``). The
    base interface hides scores, so the adapter assigns rank-decayed weights
    (1.0, 0.95, 0.90, ...) instead of true cosine similarities.

All failures surface as exceptions; :class:`~hugegraph_llm.nl2sql.linking.
schema_linker.SchemaLinker` catches them and disables P2 (lexical-only).
"""

import json
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from hugegraph_llm.utils.log import log

from .schema_graph.model import NodeType, SchemaGraph


class SchemaVectorStore(ABC):
    """Minimal vector store contract used by schema linking (P2 seeding)."""

    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]]) -> None:
        """Insert or replace ``(id, vector)`` pairs.

        A vector store backend that supports incremental writes should upsert;
        batch-only backends may treat this as a full rebuild.
        """

    @abstractmethod
    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        """Return ``(id, similarity)`` pairs, best first, similarity in [0, 1].

        ``top_k`` is an upper bound; fewer results are fine when nothing
        clears the similarity floor.
        """


# ---------------------------------------------------------------------------
# Reference implementation (in-process, exact cosine)
# ---------------------------------------------------------------------------
class NumpySchemaVectorStore(SchemaVectorStore):
    """In-process cosine store backed by a unit-normalised numpy matrix."""

    def __init__(self):
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - numpy is a runtime dep
            raise ImportError("NumpySchemaVectorStore requires numpy") from exc
        self._np = np
        self._ids: List[str] = []
        self._mat: Optional[object] = None  # (n, dim) float array, rows unit-norm

    @property
    def dimension(self) -> int:
        return int(self._mat.shape[1]) if self._mat is not None else 0

    def upsert(self, ids: List[str], vectors: List[List[float]]) -> None:
        if not ids:
            return
        np = self._np
        mat = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.where(norms == 0, 1.0, norms)
        if self._mat is None:
            self._mat = mat
            self._ids = list(ids)
        else:
            self._mat = np.vstack([self._mat, mat])
            self._ids.extend(ids)

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        if self._mat is None or len(self._ids) == 0 or top_k <= 0:
            return []
        np = self._np
        q = np.asarray(query_vector, dtype=float)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        sims = self._mat @ q
        k = min(top_k, len(self._ids))
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            s = float(sims[i])
            if s > 0:
                out.append((self._ids[int(i)], s))
        return out


# ---------------------------------------------------------------------------
# Milvus (pymilvus 3.x MilvusClient, COSINE metric)
# ---------------------------------------------------------------------------
class MilvusSchemaVectorStore(SchemaVectorStore):
    """Milvus-backed store. ``distance`` from a COSINE search is the
    cosine similarity, so it maps 1:1 onto the linker's seed weights."""

    def __init__(
        self,
        uri: str = "http://127.0.0.1:19530",
        collection_name: str = "nl2sql_schema_nodes",
        dim: int = 512,
        index_params: Optional[dict] = None,
    ):
        self._uri = uri
        self._collection = collection_name
        self._dim = dim
        # defaults follow Milvus HNSW recommendation for cosine
        self._index_params = index_params or {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        self._client = None  # lazily connected

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "MilvusSchemaVectorStore requires pymilvus"
            ) from exc
        client = MilvusClient(uri=self._uri)
        # force a round-trip so a dead server fails fast and clearly
        client.list_collections()
        if not client.has_collection(self._collection):
            client.create_collection(
                collection_name=self._collection,
                dimension=self._dim,
                index_params=self._index_params,
            )
        self._client = client
        return client

    def upsert(self, ids: List[str], vectors: List[List[float]]) -> None:
        if not ids:
            return
        client = self._get_client()
        data = [
            {"id": nid, "vector": [float(v) for v in vec]}
            for nid, vec in zip(ids, vectors)
        ]
        client.insert(self._collection, data)

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []
        client = self._get_client()
        hits = client.search(
            self._collection,
            data=[list(query_vector)],
            limit=top_k,
            output_fields=[],
        )[0]
        out = []
        for hit in hits:
            sim = float(hit["distance"])
            if sim > 0:
                out.append((str(hit["id"]), sim))
        return out


# ---------------------------------------------------------------------------
# OceanBase 4.x (native VECTOR column + HNSW cosine index)
# ---------------------------------------------------------------------------
class OceanBaseSchemaVectorStore(SchemaVectorStore):
    """OceanBase-backed store via ``pymysql``.

    SQL follows ``indices/vector_index/oceanbase_vector_store.py``: a
    ``VECTOR(dim)`` column, HNSW cosine index, ``VECTOR_INPUT()`` binding and
    ``cosinesimil()`` scoring. Validated against those patterns; needs a live
    OceanBase 4.x instance to execute.
    """

    _COSINE_FUNC = "cosinesimil"

    def __init__(
        self,
        dsn: str = "",
        table_name: str = "nl2sql_schema_nodes",
        dim: int = 512,
        index_params: Optional[dict] = None,
    ):
        self._dsn = dsn
        self._table = table_name
        self._dim = dim
        self._index_params = index_params or {"M": 16, "ef_construction": 40}
        self._conn = None
        self._initialized = False

    def _get_connection(self):
        if self._conn is not None:
            return self._conn
        try:
            import pymysql  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OceanBaseSchemaVectorStore requires pymysql "
                "(pip install pymysql)"
            ) from exc
        if not self._dsn:
            raise RuntimeError("OceanBaseSchemaVectorStore requires a dsn")
        self._conn = pymysql.connect(self._dsn)
        return self._conn

    def _ensure_table(self) -> None:
        if self._initialized:
            return
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"  node_id VARCHAR(255) PRIMARY KEY,"
                f"  embedding VECTOR({self._dim}) NOT NULL"
                f")"
            )
            m = self._index_params.get("M", 16)
            ef = self._index_params.get("ef_construction", 40)
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_vec_{self._table} "
                f"ON {self._table}(embedding) USING HNSW WITH ("
                f"M = {m}, ef_construction = {ef}, space_type = 'cosine')"
            )
            conn.commit()
            self._initialized = True
        finally:
            cur.close()

    def upsert(self, ids: List[str], vectors: List[List[float]]) -> None:
        if not ids:
            return
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            for nid, vec in zip(ids, vectors):
                vec_str = json.dumps([float(v) for v in vec])
                cur.execute(
                    f"INSERT INTO {self._table} (node_id, embedding) "
                    f"VALUES (%s, VECTOR_INPUT(%s)) "
                    f"ON DUPLICATE KEY UPDATE embedding = VECTOR_INPUT(%s)",
                    (nid, vec_str, vec_str),
                )
            conn.commit()
        finally:
            cur.close()

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            vec_str = json.dumps(list(query_vector))
            cur.execute(
                f"SELECT node_id, "
                f"{self._COSINE_FUNC}(embedding, VECTOR_INPUT(%s)) AS score "
                f"FROM {self._table} ORDER BY score DESC LIMIT %s",
                (vec_str, top_k),
            )
            out = []
            for row in cur.fetchall():
                sim = float(row[1]) if row[1] is not None else 0.0
                if math.isfinite(sim) and sim > 0:
                    out.append((str(row[0]), sim))
            return out
        finally:
            cur.close()


# ---------------------------------------------------------------------------
# Adapter over the existing RAG VectorStoreBase (Faiss/Milvus/Qdrant/OceanBase)
# ---------------------------------------------------------------------------
class LegacyVectorStoreAdapter(SchemaVectorStore):
    """Plug any existing ``VectorStoreBase`` into schema linking.

    The base ``search()`` returns matched properties only (no scores), so the
    adapter applies rank-decayed weights -- rank i gets ``decay ** i``. Exact
    cosine weighting is available via the dedicated Milvus/OceanBase stores.
    """

    _DECAY = 0.95

    def __init__(self, store, dis_threshold: float = 1e9):
        self._store = store
        self._dis_threshold = dis_threshold  # effectively disables the filter

    def upsert(self, ids: List[str], vectors: List[List[float]]) -> None:
        self._store.add(list(vectors), list(ids))

    def search(
        self, query_vector: List[float], top_k: int
    ) -> List[Tuple[str, float]]:
        props = self._store.search(
            list(query_vector), top_k, dis_threshold=self._dis_threshold
        )
        return [
            (str(p), self._DECAY ** i) for i, p in enumerate(props)
        ]


def as_schema_store(store) -> SchemaVectorStore:
    """Normalise ``None`` / ``SchemaVectorStore`` / ``VectorStoreBase`` into a
    :class:`SchemaVectorStore`."""
    if store is None:
        return NumpySchemaVectorStore()
    if isinstance(store, SchemaVectorStore):
        return store
    # duck-typing: any object exposing the RAG base's add/search works
    return LegacyVectorStoreAdapter(store)


# ---------------------------------------------------------------------------
# Shared embedding helper (write path + read path use the same surfaces)
# ---------------------------------------------------------------------------
def embed_schema_nodes(schema: "SchemaGraph", embedder, store: SchemaVectorStore):
    """Embed every schema node once into ``store``; return ``(count, dim)``.

    This is the single implementation of "which text represents a schema node".
    Both the write path (ingest-time vector refresh) and the read path
    (:meth:`SchemaLinker._ensure_vector_index`) call it, so the vector index is
    always built from the same surfaces — and, with one store instance injected
    at construction, into the *same* vector store. Table / column nodes embed
    ``name + comment + table``; term nodes embed ``name + aliases +
    comment/definition`` so curated business vocabulary participates in
    semantic recall (terms seed the PPR only; they never surface in link()).
    """
    ids: List[str] = []
    texts: List[str] = []
    for node_id, node in schema.nodes.items():
        if node.node_type == NodeType.TERM:
            aliases = " ".join(
                str(a) for a in node.properties.get("aliases", []) if a
            )
            surface = " ".join([
                node.name or "",
                aliases,
                str(node.properties.get("comment", "")),
                str(node.properties.get("definition", "")),
            ])
        else:
            surface = " ".join([
                node.name or "",
                str(node.properties.get("comment", "")),
                str(node.properties.get("table", "")),
            ])
        ids.append(node_id)
        texts.append(surface)
    if not ids:
        return 0, 0
    vecs = [list(embedder(t)) for t in texts]
    store.upsert(ids, vecs)
    return len(ids), len(vecs[0]) if vecs else 0
