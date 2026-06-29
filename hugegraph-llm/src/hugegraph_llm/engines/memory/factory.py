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
Pluggable backend factories (Mem0/PowerMem-style).

These factories allow the engine to swap LLM, embedder, vector store,
graph store, and reranker providers without changing the core pipeline.
"""

from typing import Any, Callable, Dict, Optional, Type

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.engines.memory.base import (
    EmbeddingBase,
    GraphStoreBase,
    LLMBase,
    RerankerBase,
    VectorStoreBase,
)
from hugegraph_llm.utils.log import log


class _Registry:
    """Internal auto-registration helper."""

    def __init__(self):
        self._providers: Dict[str, Type] = {}

    def register(self, name: str, cls: Type) -> Type:
        self._providers[name.lower()] = cls
        return cls

    def create(self, name: str, **kwargs) -> Any:
        key = name.lower()
        if key not in self._providers:
            raise ValueError(
                f"Unknown provider '{name}'. Available: {list(self._providers.keys())}"
            )
        return self._providers[key](**kwargs)


class LLMFactory(_Registry):
    """Factory for LLM backends."""
    pass


class EmbedderFactory(_Registry):
    """Factory for embedding backends."""
    pass


class VectorStoreFactory(_Registry):
    """Factory for vector store backends."""
    pass


class RerankerFactory(_Registry):
    """Factory for reranker backends."""
    pass


class GraphStoreFactory(_Registry):
    """Factory for graph store backends."""
    pass


llm_factory = LLMFactory()
embedder_factory = EmbedderFactory()
vector_store_factory = VectorStoreFactory()
reranker_factory = RerankerFactory()
graph_store_factory = GraphStoreFactory()


def register_llm(name: str) -> Callable[[Type], Type]:
    return lambda cls: llm_factory.register(name, cls)


def register_embedder(name: str) -> Callable[[Type], Type]:
    return lambda cls: embedder_factory.register(name, cls)


def register_vector_store(name: str) -> Callable[[Type], Type]:
    return lambda cls: vector_store_factory.register(name, cls)


def register_reranker(name: str) -> Callable[[Type], Type]:
    return lambda cls: reranker_factory.register(name, cls)


def register_graph_store(name: str) -> Callable[[Type], Type]:
    return lambda cls: graph_store_factory.register(name, cls)


def get_default_vector_store() -> Optional[VectorStoreBase]:
    """Return the vector store configured via memory_settings."""
    backend = memory_settings.vector_backend
    if backend == "faiss":
        # FaissVectorStore is the default; it is initialized lazily by the pipeline.
        return None
    log.warning("Vector backend '%s' factory not yet implemented; using default pipeline.", backend)
    return None


def get_default_reranker() -> Optional[RerankerBase]:
    """Return the reranker configured via memory_settings if enabled."""
    if not memory_settings.rerank_enabled:
        return None
    from hugegraph_llm.indices.rerank_index import get_reranker

    return get_reranker()
