# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic-similarity channel for memory recall.

Adds the vector signal the temporal channel cannot provide: temporal
gating decides *when* a memory was true, but not whether it is *relevant*
to the question. This channel answers the relevance half, and the two are
merged through the shared RRF implementation.

**Backend is swappable**, because the deployment target is expected to
change: FAISS now (in-process, zero infrastructure), OceanBase in
production (durable, shared, scales out). Both satisfy the same small
contract, so switching is a constructor argument, not a rewrite.

The contract is deliberately narrow -- ``add`` and ``search`` over
``(id, text)`` pairs, with embedding done here so callers never touch
vectors:

    add(items: [(id, text)]) -> None
    search(query: str, top_k: int) -> [(id, score)]

FAISS here wraps the repository's existing ``FaissVectorIndex`` rather than
reaching for faiss directly, so index persistence and property handling
stay consistent with the rest of hugegraph-llm.

**Vector recall is best-effort.** If the backend is unavailable, recall
degrades to the remaining channels instead of failing: memory without
semantic matching is less useful, but memory that throws is useless.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "VectorChannel",
    "FaissChannel",
    "VectorChannelConfig",
    "NoOpChannel",
]


EmbedFn = Callable[[str], List[float]]

#: Distance ceiling that effectively disables filtering. See ``search``:
#: the underlying index drops any hit whose L2 distance is >= this value.
_NO_DISTANCE_LIMIT = float("inf")


class VectorChannelConfig:
    """Shared knobs for a vector channel."""

    def __init__(self, *, top_k: int = 20, min_score: float = 0.0) -> None:
        self.top_k = top_k
        #: Drop hits below this similarity. 0 keeps everything -- useful
        #: when the backend's scale is unknown or the embedder uncalibrated.
        self.min_score = min_score


class VectorChannel(ABC):
    """A source of relevance-ranked memory ids."""

    #: Channel name, used in fusion metadata so ranking is explainable.
    name: str = "vector"

    @abstractmethod
    def add(self, items: Sequence[Tuple[str, str]]) -> None:
        """Index ``(id, text)`` pairs."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Return ``(id, score)`` best first, higher score = more similar."""

    def ranked_ids(self, query: str, top_k: int) -> List[str]:
        """Ids only, for feeding the shared RRF implementation."""
        return [str(doc_id) for doc_id, _score in self.search(query, top_k)]


class NoOpChannel(VectorChannel):
    """Channel that never matches.

    Used when no backend is configured, so the call site does not need a
    branch: recall still works, just without a semantic signal.
    """

    name = "vector_unavailable"

    def add(self, items: Sequence[Tuple[str, str]]) -> None:
        return None

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        return []


class FaissChannel(VectorChannel):
    """In-process FAISS channel (development / single-node).

    :param embed: embedding callable.
    :param config: recall knobs.
    :param index_name: persistence name for the underlying FAISS index.
    """

    name = "vector_faiss"

    def __init__(
        self,
        embed: EmbedFn,
        *,
        config: Optional[VectorChannelConfig] = None,
        index_name: str = "memory",
    ) -> None:
        self.embed = embed
        self.config = config or VectorChannelConfig()
        self.index_name = index_name
        self._index: Any = None
        self._dim: Optional[int] = None
        self._ids: List[str] = []
        #: ``id -> vector``, kept so scores can be recomputed per hit.
        self._vectors: Dict[str, List[float]] = {}

    # -- indexing -----------------------------------------------------------

    def add(self, items: Sequence[Tuple[str, str]]) -> None:
        if not items:
            return
        vectors: List[List[float]] = []
        ids: List[str] = []
        for doc_id, text in items:
            # Embed per item and skip failures. Embedding the whole batch
            # first means one bad item (or one model blip) discards every
            # memory in the batch -- and a silently-empty index is far worse
            # than a partially populated one.
            try:
                vector = self.embed(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("faiss channel: embed failed for %s: %s", doc_id, exc)
                continue
            if not vector:
                continue
            vectors.append(vector)
            ids.append(doc_id)
        if not vectors:
            return
        self._ensure_index(len(vectors[0]))
        try:
            self._index.add(vectors, ids)
            self._ids.extend(ids)
            self._vectors.update(dict(zip(ids, vectors)))
        except Exception as exc:  # noqa: BLE001 - degrade, do not fail recall
            logger.warning("faiss channel: add failed: %s", exc)

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        from hugegraph_llm.indices.vector_index.faiss_vector_store import (
            FaissVectorIndex,
        )

        self._dim = dim
        self._index = FaissVectorIndex(embed_dim=dim)

    # -- search -------------------------------------------------------------

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self._index is None or not self._ids:
            return []
        try:
            vector = self.embed(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("faiss channel: embed failed: %s", exc)
            return []
        try:
            # ``dis_threshold`` is a DISTANCE ceiling (the index keeps
            # ``dist < threshold``), not a similarity floor. Passing 0.0 --
            # the intuitive "no filtering" value for a similarity -- filters
            # out every result. Pass a very large value and rank here.
            hits = self._index.search(vector, top_k, dis_threshold=_NO_DISTANCE_LIMIT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("faiss channel: search failed: %s", exc)
            return []

        scored: Dict[str, float] = {}
        for hit in hits or []:
            doc_id = str(hit)
            if doc_id in scored:
                continue  # the index can return the same id more than once
            score = self._similarity(query, doc_id)
            if score is None:
                continue
            if score < self.config.min_score:
                continue
            scored[doc_id] = score

        # Rank by recomputed similarity. The index's own order is by L2
        # distance, which is not the similarity we score against.
        return sorted(scored.items(), key=lambda kv: -kv[1])

    def _similarity(
        self,
        query: str,
        doc_id: str,
        query_vector: Optional[List[float]] = None,
    ) -> Optional[float]:
        """Cosine similarity for a hit, recomputed from stored vectors.

        The index returns only property payloads -- no scores -- and its
        internal L2 distance is unbounded, so a comparable similarity has to
        be derived here. Vectors are kept alongside ids for exactly this:
        depending on a backend-specific accessor would couple memory to one
        client version.
        """
        try:
            stored = self._vectors.get(doc_id)
            if stored is None:
                return None
            if query_vector is None:
                query_vector = self.embed(query)
            dot = sum(a * b for a, b in zip(query_vector, stored))
            norm_a = sum(a * a for a in query_vector) ** 0.5
            norm_b = sum(b * b for b in stored) ** 0.5
            if not norm_a or not norm_b:
                return 0.0
            return max(0.0, min(1.0, dot / (norm_a * norm_b)))
        except Exception:  # noqa: BLE001 - score is advisory
            return None

    @property
    def size(self) -> int:
        return len(self._ids)
