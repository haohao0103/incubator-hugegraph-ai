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
Base abstractions for the HugeGraph-AI-Memory engine.

Inspired by Mem0 (MemoryBase, VectorStoreBase, EmbeddingBase, LLMBase,
BaseReranker) and PowerMem (StorageAdapter, GraphStoreBase, MemoryType,
MemoryScope). These abstractions allow pluggable LLM/embedding/vector/
graph/rerank backends while keeping the existing MemoryPipelineBackend
as the production-ready reference implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """PowerMem-style memory categorization."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PUBLIC_SHARED = "public_shared"
    PRIVATE_AGENT = "private_agent"
    COLLABORATIVE = "collaborative"
    GROUP_CONSENSUS = "group_consensus"


class MemoryScope(str, Enum):
    """PowerMem-style access scope."""
    PRIVATE = "private"
    AGENT_GROUP = "agent_group"
    USER_GROUP = "user_group"
    PUBLIC = "public"
    RESTRICTED = "restricted"


class PrivacyLevel(str, Enum):
    """PowerMem-style privacy level."""
    STANDARD = "standard"
    SENSITIVE = "sensitive"
    CONFIDENTIAL = "confidential"


class AccessPermission(str, Enum):
    """PowerMem-style access permission."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


class CollaborationLevel(str, Enum):
    """PowerMem-style collaboration level."""
    ISOLATED = "isolated"
    COLLABORATIVE = "collaborative"


@dataclass
class MemoryEntry:
    """Normalized memory entry used by the engine abstractions."""
    id: str
    content: str
    user_id: str = "demo_user"
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.EPISODIC
    scope: MemoryScope = MemoryScope.PRIVATE
    privacy: PrivacyLevel = PrivacyLevel.STANDARD
    importance: float = 0.5
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    last_accessed_at: Optional[float] = None
    access_count: int = 0
    retention: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Normalized retrieval result."""
    memory: MemoryEntry
    score: float
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBase(ABC):
    """High-level memory store interface (Mem0-style)."""

    @abstractmethod
    def add(
        self,
        messages: List[Dict[str, Any]],
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Add memories and return created entries."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories and return ranked entries."""
        raise NotImplementedError

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by id."""
        raise NotImplementedError

    @abstractmethod
    def update(self, memory_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a memory."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory."""
        raise NotImplementedError

    @abstractmethod
    def history(self, memory_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Return the edit history of a memory."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset all memories."""
        raise NotImplementedError


class VectorStoreBase(ABC):
    """Vector store adapter (Mem0-style)."""

    @abstractmethod
    def insert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def colpali(self, query: str) -> List[Dict[str, Any]]:
        """Multi-modal retrieval hook (optional)."""
        raise NotImplementedError

    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Optional full-text search fallback."""
        return []


class EmbeddingBase(ABC):
    """Embedding model adapter (Mem0-style)."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class LLMBase(ABC):
    """LLM adapter (Mem0-style)."""

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


class RerankerBase(ABC):
    """Reranker adapter (Mem0-style)."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GraphStoreBase(ABC):
    """Graph store adapter (PowerMem-style)."""

    @abstractmethod
    def add(self, data: Dict[str, Any]) -> None:
        """Add entities/relations to the graph."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        max_hops: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search graph for relevant context."""
        raise NotImplementedError

    @abstractmethod
    def get_all_entities(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_all_relations(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
