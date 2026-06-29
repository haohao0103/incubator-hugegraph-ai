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
Unified configuration for HugeGraph-AI-Memory (PowerMem-style).

All environment variables are optional and have safe defaults for local
HugeGraph 1.7.0 + sentence-transformers + MiMo (OpenAI-compatible) PoC.

Usage:
    from hugegraph_llm.config.memory_config import memory_settings
    print(memory_settings.hugegraph_url)
"""

import os
from pathlib import Path
from typing import Literal, Optional

from .models import BaseConfig


class MemoryConfig(BaseConfig):
    """Unified AI Memory configuration."""

    # ------------------------------------------------------------------
    # HugeGraph
    # ------------------------------------------------------------------
    hugegraph_url: str = os.environ.get("HUGEGRAPH_URL", "http://127.0.0.1:8080")
    hugegraph_user: str = os.environ.get("HUGEGRAPH_USER", "admin")
    hugegraph_pwd: str = os.environ.get("HUGEGRAPH_PASS", "admin")
    hugegraph_graph: str = os.environ.get("HUGEGRAPH_GRAPH", "hugegraph")

    # ------------------------------------------------------------------
    # LLM (OpenAI-compatible)
    # ------------------------------------------------------------------
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
    llm_model: str = os.environ.get("LLM_MODEL", "mimo-v2.5-pro")
    llm_api_key: Optional[str] = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_CHAT_API_KEY")
    llm_max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
    llm_temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.3"))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    embedding_type: Literal["sentence_transformers", "openai"] = os.environ.get(
        "EMBEDDING_TYPE", "sentence_transformers"
    )
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_api_key: Optional[str] = os.environ.get("EMBEDDING_API_KEY")
    embedding_api_base: Optional[str] = os.environ.get("EMBEDDING_API_BASE")
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "384"))

    # ------------------------------------------------------------------
    # Fulltext (BM25)
    # ------------------------------------------------------------------
    fulltext_backend: Literal["bm25", "oceanbase"] = os.environ.get("FULLTEXT_BACKEND", "bm25")
    bm25_index_name: str = os.environ.get("BM25_INDEX_NAME", "memory_bm25")

    # ------------------------------------------------------------------
    # Vector (FAISS/Milvus/Qdrant)
    # ------------------------------------------------------------------
    vector_backend: Literal["faiss", "milvus", "qdrant", "oceanbase"] = os.environ.get(
        "VECTOR_BACKEND", "faiss"
    )
    faiss_index_path: Optional[str] = os.environ.get("FAISS_INDEX_PATH")

    # ------------------------------------------------------------------
    # Rerank (cross-encoder / API)
    # ------------------------------------------------------------------
    rerank_enabled: bool = os.environ.get("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
    rerank_backend: Literal["sentence_transformers", "jina", "openai", "cohere"] = os.environ.get(
        "RERANK_BACKEND", "sentence_transformers"
    )
    rerank_model: str = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_api_key: Optional[str] = os.environ.get("RERANK_API_KEY")
    rerank_api_base: Optional[str] = os.environ.get("RERANK_API_BASE")
    rerank_top_k: int = int(os.environ.get("RERANK_TOP_K", "10"))
    rerank_batch_size: int = int(os.environ.get("RERANK_BATCH_SIZE", "32"))

    # ------------------------------------------------------------------
    # Sparse vector (SPLADE/BM25 tokenization)
    # ------------------------------------------------------------------
    sparse_enabled: bool = os.environ.get("SPARSE_ENABLED", "false").lower() in ("1", "true", "yes")
    sparse_backend: Literal["splade", "bm25_tokens"] = os.environ.get("SPARSE_BACKEND", "bm25_tokens")
    sparse_weight: float = float(os.environ.get("SPARSE_WEIGHT", "0.3"))

    # ------------------------------------------------------------------
    # Memory behavior
    # ------------------------------------------------------------------
    ebbinghaus_k: float = float(os.environ.get("EBBINGHAUS_K", "0.821"))
    ebbinghaus_reinforce: float = float(os.environ.get("EBBINGHAUS_REINFORCE", "0.3"))
    default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "5"))

    # ------------------------------------------------------------------
    # Server / MCP
    # ------------------------------------------------------------------
    memory_server_host: str = os.environ.get("MEMORY_SERVER_HOST", "127.0.0.1")
    memory_server_port: int = int(os.environ.get("MEMORY_SERVER_PORT", "8765"))
    mcp_server_name: str = os.environ.get("MCP_SERVER_NAME", "hugegraph-memory")
    mcp_server_port: int = int(os.environ.get("MCP_SERVER_PORT", "8848"))

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    memory_db_path: Optional[str] = os.environ.get("MEMORY_DB_PATH")
    memory_data_dir: str = os.environ.get(
        "MEMORY_DATA_DIR", str(Path(__file__).resolve().parents[3] / "poc_data")
    )

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------
    distillation_enabled: bool = os.environ.get("DISTILLATION_ENABLED", "false").lower() in (
        "1", "true", "yes"
    )
    experience_threshold: int = int(os.environ.get("EXPERIENCE_THRESHOLD", "5"))

    # ------------------------------------------------------------------
    # Multimodal
    # ------------------------------------------------------------------
    multimodal_enabled: bool = os.environ.get("MULTIMODAL_ENABLED", "false").lower() in (
        "1", "true", "yes"
    )
    vision_model: Optional[str] = os.environ.get("VISION_MODEL")
    asr_model: Optional[str] = os.environ.get("ASR_MODEL")

    def resolve_db_path(self) -> str:
        if self.memory_db_path:
            return self.memory_db_path
        Path(self.memory_data_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.memory_data_dir) / "memory_backend.db")

    def resolve_faiss_path(self) -> str:
        if self.faiss_index_path:
            return self.faiss_index_path
        Path(self.memory_data_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.memory_data_dir) / "memory_faiss.index")


memory_settings = MemoryConfig()

# Post-initialization fallback: if the dedicated LLM_API_KEY is empty but a
# project-wide OpenAI key is configured, use it so that memory backends can
# initialize without requiring users to duplicate the key.
if not memory_settings.llm_api_key:
    memory_settings.llm_api_key = os.environ.get("OPENAI_CHAT_API_KEY") or os.environ.get("OPENAI_EXTRACT_API_KEY") or os.environ.get("LLM_API_KEY")
    if memory_settings.llm_api_key:
        os.environ["LLM_API_KEY"] = memory_settings.llm_api_key
