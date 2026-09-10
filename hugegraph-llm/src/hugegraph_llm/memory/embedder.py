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

"""Embedders for the memory vector channel.

HugeGraph 1.7.0 has no first-class vector index comparable to Neo4j's
``db.index.vector.queryNodes``, so embeddings are held in-process and
vector search is a cosine scan over the cache. That is fine at PoC scale
(hundreds to low-thousands of nodes); production is expected to use
OceanBase for vector storage, where this module is not on the hot path.

**sentence-transformers is optional, and that is deliberate.** Depending on
it pulls in torch, transformers and tokenizers -- roughly a gigabyte on
macOS -- onto every developer machine and CI runner, when:

* production vector search is OceanBase's, not ours,
* validating the recall loop needs plumbing, not semantic quality,
* and the LLM endpoint in use may offer no embedding model at all (the
  configured provider exposes chat/ASR/TTS models only).

So :class:`LocalEmbedder` degrades instead of raising: it uses
sentence-transformers when importable, and otherwise falls back to
:class:`HashingEmbedder`. Callers get the same interface either way, and
:attr:`LocalEmbedder.degraded` says which one is in play so nobody mistakes
fallback output for real semantic similarity.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Iterable, List

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["LocalEmbedder", "HashingEmbedder", "EMBEDDING_AVAILABLE"]

#: True when sentence-transformers (and its heavy deps) are importable.
try:  # pragma: no cover - depends on environment
    import sentence_transformers  # noqa: F401

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class HashingEmbedder:
    """Deterministic fallback embedder. **Not semantic.**

    Projects tokens onto a fixed-dimensional space by hash, with a signed
    contribution so different tokens do not collide into pure magnitude.
    Identical texts give identical vectors and unrelated texts are roughly
    orthogonal, which is enough to exercise ranking and fusion plumbing.

    It cannot judge meaning: "works at Acme" and "employed by Acme" are no
    closer than "works at Acme" and "likes tea". Never use it to evaluate
    retrieval quality -- only to prove the pipeline works.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(str(text).lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


class LocalEmbedder:
    """Embeds text with sentence-transformers, falling back to hashing.

    :param model_name: sentence-transformers model; ignored when degraded.
    :param dim: vector dimension.
    :param allow_fallback: when False, a missing sentence-transformers
        raises instead of degrading. Use this in jobs that would otherwise
        silently produce meaningless similarity scores.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        *,
        allow_fallback: bool = True,
    ) -> None:
        self.dim = dim
        self.model_name = model_name
        self._model = None
        self._fallback: HashingEmbedder | None = None
        # key (usually uuid or text) -> normalized np.array
        self._cache: dict[str, np.ndarray] = {}

        if EMBEDDING_AVAILABLE:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            logger.info("LocalEmbedder loaded: %s (dim=%d)", model_name, dim)
        elif allow_fallback:
            self._fallback = HashingEmbedder(dim=dim)
            logger.warning(
                "sentence-transformers unavailable; LocalEmbedder degraded to "
                "hashing. Vectors are NOT semantic -- install "
                "sentence-transformers for real similarity."
            )
        else:
            raise RuntimeError(
                "sentence-transformers is required but not installed "
                "(pip install sentence-transformers)"
            )

    @property
    def degraded(self) -> bool:
        """True when running on the hashing fallback."""
        return self._model is None

    def embed(self, text: str, key: str | None = None) -> np.ndarray:
        """Embed text; optionally cache under ``key`` (e.g. entity uuid)."""
        k = key or text
        if k in self._cache:
            return self._cache[k]
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            vec = np.asarray(vec, dtype=np.float32)
        else:
            vec = self._fallback.encode(text)
        self._cache[k] = vec
        return vec

    def embed_many(self, texts: Iterable[str]) -> List[np.ndarray]:
        """Embed a batch. The fallback stays per-item on purpose."""
        return [self.embed(t) for t in texts]

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # vectors are L2-normalized

    def _to_arr(self, vec) -> np.ndarray:
        """Convert a list[float] embedding to a normalized np.array."""
        if isinstance(vec, np.ndarray):
            return vec
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def top_k(
        self,
        query_vec: np.ndarray,
        candidates: Iterable[tuple[str, np.ndarray]],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return ``(key, score)`` sorted by cosine similarity desc."""
        scored = [(key, self.cosine(query_vec, self._to_arr(vec)))
                  for key, vec in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
