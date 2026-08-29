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
"""KG retriever abstraction (generalized from neo4j-graphrag-python).

Three-layer protocol, mirroring the official Neo4j GraphRAG package's
``Retriever`` (retrievers/base.py):

    search(query, **kwargs)                     # unified public entry
      └─ get_search_results(query, **kwargs)    # subclass implements
      └─ get_result_formatter()                 # item -> RetrieverResultItem
      └─ RetrieverResult(items, metadata)       # unified result + provenance

Every retriever returns a ``RetrieverResult`` through the same ``search``
entry so downstream code (generation, MCP tools, fusion) can consume
retrievers interchangeably. ``metadata["__retriever"]`` records which
retriever produced the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RetrieverResultItem:
    """A single retrieved item: text content + structured metadata + score."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrieverResult:
    """Unified retriever output: ordered items plus retriever-level metadata."""

    items: List[RetrieverResultItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
        }

    @property
    def is_empty(self) -> bool:
        """True when nothing was retrieved (e.g. for response_fallback)."""
        return not self.items


class KGRetriever(ABC):
    """Abstract base class for knowledge-graph retrievers.

    Subclasses implement :meth:`get_search_results` (raw retrieval) and may
    override :meth:`get_result_formatter` / :meth:`_result_items` /
    :meth:`_result_metadata` to adapt their native output shape.
    """

    def search(self, query: str, **kwargs: Any) -> RetrieverResult:
        """Unified public entry: retrieve, format, and wrap in RetrieverResult.

        Args:
            query: The search query text.
            **kwargs: Retriever-specific parameters forwarded to
                :meth:`get_search_results` (e.g. ``rewrite``, ``top_k``).
        """
        raw = self.get_search_results(query, **kwargs)
        items = self._result_items(raw)
        formatter = self.get_result_formatter()
        search_items = [formatter(item) for item in items]
        metadata = dict(self._result_metadata(raw))
        metadata["__retriever"] = self.__class__.__name__
        return RetrieverResult(items=search_items, metadata=metadata)

    @abstractmethod
    def get_search_results(self, query: str, **kwargs: Any) -> Any:
        """Execute the raw retrieval.

        Returns either a list/tuple of items, or an object exposing the
        items (see :meth:`_result_items`).
        """

    def get_result_formatter(self) -> Callable[[Any], RetrieverResultItem]:
        """Return the function mapping one raw item to a RetrieverResultItem."""
        return self._default_formatter

    def _result_items(self, raw: Any) -> List[Any]:
        """Extract the iterable of items from :meth:`get_search_results` output."""
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return list(raw)
        if isinstance(raw, RetrieverResult):
            return raw.items
        if hasattr(raw, "chunks"):  # e.g. KGSearchResult
            return list(raw.chunks)
        return list(raw)

    def _result_metadata(self, raw: Any) -> Dict[str, Any]:
        """Retriever-level metadata from the raw result (e.g. provenance)."""
        if hasattr(raw, "provenance") and isinstance(raw.provenance, dict):
            return dict(raw.provenance)
        if isinstance(raw, RetrieverResult):
            return dict(raw.metadata)
        return {}

    def _default_formatter(self, item: Any) -> RetrieverResultItem:
        """Best-effort item -> RetrieverResultItem conversion."""
        if isinstance(item, RetrieverResultItem):
            return item
        if hasattr(item, "content"):
            return RetrieverResultItem(
                content=str(item.content),
                metadata=dict(getattr(item, "metadata", {}) or {}),
                score=getattr(item, "score", None),
            )
        if hasattr(item, "text"):  # e.g. ScoredChunk-like
            return RetrieverResultItem(
                content=str(item.text),
                metadata={"id": getattr(item, "chunk_id", None)},
                score=getattr(item, "score", None),
            )
        return RetrieverResultItem(content=str(item))
