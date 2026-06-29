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

"""HugeGraph-AI-Memory engine abstractions and SDK."""

from hugegraph_llm.engines.memory.base import (
    AccessPermission,
    CollaborationLevel,
    EmbeddingBase,
    GraphStoreBase,
    LLMBase,
    MemoryBase,
    MemoryEntry,
    MemoryScope,
    MemoryType,
    PrivacyLevel,
    RerankerBase,
    RetrievalResult,
    VectorStoreBase,
)
from hugegraph_llm.engines.memory.client import AsyncMemoryClient, MemoryClient
from hugegraph_llm.engines.memory.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LLMFactory,
    RerankerFactory,
    VectorStoreFactory,
    embedder_factory,
    graph_store_factory,
    llm_factory,
    reranker_factory,
    vector_store_factory,
)
from hugegraph_llm.engines.memory.intelligence import (
    EbbinghausDecay,
    EntityExtractor,
    ImportanceEvaluator,
    MemoryOptimizer,
)

__all__ = [
    "MemoryBase",
    "VectorStoreBase",
    "EmbeddingBase",
    "LLMBase",
    "RerankerBase",
    "GraphStoreBase",
    "MemoryEntry",
    "RetrievalResult",
    "MemoryType",
    "MemoryScope",
    "PrivacyLevel",
    "AccessPermission",
    "CollaborationLevel",
    "LLMFactory",
    "EmbedderFactory",
    "VectorStoreFactory",
    "RerankerFactory",
    "GraphStoreFactory",
    "llm_factory",
    "embedder_factory",
    "vector_store_factory",
    "reranker_factory",
    "graph_store_factory",
    "ImportanceEvaluator",
    "EbbinghausDecay",
    "MemoryOptimizer",
    "EntityExtractor",
    "MemoryClient",
    "AsyncMemoryClient",
]
