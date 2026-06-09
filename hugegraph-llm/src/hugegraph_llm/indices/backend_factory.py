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

"""Storage backend factory.

Creates vector store and full-text search instances based on configuration.
Supports switching between local file storage (dev) and OceanBase (prod).

Configuration via environment variables or config file::

    # Storage backend selection
    VECTOR_BACKEND=faiss|oceanbase|milvus|qdrant
    FULLTEXT_BACKEND=bm25|oceanbase

    # OceanBase connection (required when backend=oceanbase)
    OCEANBASE_DSN=ob://user:pass@host:2883/tenant?database=db

    # OceanBase options
    OCEANBASE_VECTOR_TABLE=rag_chunks
    OCEANBASE_FTS_TABLE=rag_chunks
    OCEANBASE_VECTOR_INDEX_TYPE=hnsw
    OCEANBASE_VECTOR_DIM=768
    OCEANBASE_FTS_PARSER=ik
    OCEANBASE_FTS_IK_MODE=smart
"""

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def create_vector_store(
    embed_dim: int = 768,
    dsn: str = "",
    **kwargs: Any,
):
    """Create a vector store based on configuration.

    Reads ``VECTOR_BACKEND`` env var to select backend:
    - ``faiss`` (default): Local file-based FAISS index. No external deps.
    - ``oceanbase``: OceanBase VECTOR column. Requires OceanBase 4.x.
    - ``milvus``: Milvus server. Requires running Milvus instance.
    - ``qdrant``: Qdrant server. Requires running Qdrant instance.

    Args:
        embed_dim: Embedding vector dimension.
        dsn: Override OceanBase DSN (ignores env var if provided).
        **kwargs: Additional backend-specific parameters.

    Returns:
        A VectorStoreBase subclass instance.
    """
    backend = os.environ.get("VECTOR_BACKEND", "faiss").lower()
    ob_dsn = dsn or os.environ.get("OCEANBASE_DSN", "")

    if backend == "oceanbase":
        from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
            OceanBaseVectorStore,
        )

        if not ob_dsn:
            raise ValueError(
                "OceanBase vector backend requires OCEANBASE_DSN. "
                "Set via env var or pass dsn= parameter."
            )
        return OceanBaseVectorStore(
            dsn=ob_dsn,
            table_name=os.environ.get("OCEANBASE_VECTOR_TABLE", "rag_chunks"),
            embed_dim=int(os.environ.get("OCEANBASE_VECTOR_DIM", str(embed_dim))),
            index_type=os.environ.get("OCEANBASE_VECTOR_INDEX_TYPE", "hnsw"),
            index_params={
                "M": int(os.environ.get("OB_HNSW_M", "16")),
                "ef_construction": int(os.environ.get("OB_HNSW_EF", "40")),
            },
        )

    elif backend == "milvus":
        from hugegraph_llm.indices.vector_index.milvus_vector_store import (
            MilvusVectorStore,
        )

        return MilvusVectorStore(embed_dim=embed_dim, **kwargs)

    elif backend == "qdrant":
        from hugegraph_llm.indices.vector_index.qdrant_vector_store import (
            QdrantVectorStore,
        )

        return QdrantVectorStore(embed_dim=embed_dim, **kwargs)

    elif backend == "faiss":
        from hugegraph_llm.indices.vector_index.faiss_vector_store import (
            FaissVectorIndex,
        )

        return FaissVectorIndex(embed_dim=embed_dim, **kwargs)

    else:
        raise ValueError(
            f"Unknown vector backend '{backend}'. "
            f"Choose from: faiss, oceanbase, milvus, qdrant"
        )


def create_fulltext_store(
    dsn: str = "",
    **kwargs: Any,
):
    """Create a full-text search store based on configuration.

    Reads ``FULLTEXT_BACKEND`` env var to select backend:
    - ``bm25`` (default): Local file-based BM25Okapi. No external deps.
    - ``oceanbase``: OceanBase FULLTEXT INDEX. Requires OceanBase 4.x.

    Args:
        dsn: Override OceanBase DSN (ignores env var if provided).
        **kwargs: Additional backend-specific parameters.

    Returns:
        A FullTextBase subclass instance.
    """
    backend = os.environ.get("FULLTEXT_BACKEND", "bm25").lower()
    ob_dsn = dsn or os.environ.get("OCEANBASE_DSN", "")

    if backend == "oceanbase":
        from hugegraph_llm.indices.fulltext.oceanbase_fulltext import (
            OceanBaseFTSBackend,
        )

        if not ob_dsn:
            raise ValueError(
                "OceanBase FTS backend requires OCEANBASE_DSN. "
                "Set via env var or pass dsn= parameter."
            )
        return OceanBaseFTSBackend(
            dsn=ob_dsn,
            table_name=os.environ.get("OCEANBASE_FTS_TABLE", "rag_chunks"),
            parser=os.environ.get("OCEANBASE_FTS_PARSER", "ik"),
            ik_mode=os.environ.get("OCEANBASE_FTS_IK_MODE", "smart"),
        )

    elif backend == "bm25":
        from hugegraph_llm.indices.fulltext.bm25_fulltext import (
            BM25FullTextBackend,
        )

        return BM25FullTextBackend(**kwargs)

    else:
        raise ValueError(
            f"Unknown full-text backend '{backend}'. "
            f"Choose from: bm25, oceanbase"
        )
