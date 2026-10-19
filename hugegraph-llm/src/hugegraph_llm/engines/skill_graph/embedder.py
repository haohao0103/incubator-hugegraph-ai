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

"""Text embedders for the skill graph.

The default :class:`HashingEmbedder` is deterministic and dependency-free
(feature hashing over word unigrams/bigrams plus character trigrams). It needs
no model download or network access, which keeps the graph builder and retriever
hermetic and reproducible. Production deployments can inject the project's real
embedder through :func:`resolve_embedder`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface used across the skill graph."""

    def embed(self, text: str) -> List[float]:  # pragma: no cover - protocol
        ...


def tokenize(text: str) -> List[str]:
    """Whitespace/punctuation tokenizer that also emits character trigrams.

    The trigrams give the hashing embedder some robustness for CJK text (which
    has no whitespace) and for partial word matches, without pulling in a
    language-specific tokenizer.
    """
    text = (text or "").lower()
    words = [w for w in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text) if w]
    tokens = list(words)
    joined = "".join(words)
    for i in range(len(joined) - 2):
        tokens.append(joined[i : i + 3])
    return tokens or ["<empty>"]


def _stable_hash(token: str) -> int:
    """Deterministic hash (Python's built-in str hash is salted per process)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class HashingEmbedder:
    """Deterministic feature-hashing embedder with L2-normalized output."""

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = _stable_hash(tok)
            idx = h % self.dim
            sign = 1.0 if ((h >> 31) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class _ProjectEmbedderAdapter:
    """Adapter around the project's ``Embeddings`` backends (OpenAI/Ollama/...)."""

    def __init__(self, backend: object):
        self._backend = backend

    def embed(self, text: str) -> List[float]:
        return list(self._backend.get_text_embedding(text))  # type: ignore[attr-defined]


def resolve_embedder(dim: int = 256) -> Embedder:
    """Return the project embedder if importable, else the hashing fallback.

    The project embedder is only *resolved* (not invoked) here, so no network
    call happens until the first :meth:`embed`. Any failure falls back to
    :class:`HashingEmbedder`.
    """
    try:
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings

        return _ProjectEmbedderAdapter(Embeddings().get_embedding())
    except Exception:  # pragma: no cover - depends on optional runtime config
        return HashingEmbedder(dim=dim)


def mean_vector(vectors: Sequence[Sequence[float]], dim: Optional[int] = None) -> List[float]:
    """Mean of a list of vectors, L2-normalized."""
    if not vectors:
        return []
    dim = dim or len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec[:dim]):
            acc[i] += val
    norm = math.sqrt(sum(v * v for v in acc))
    if norm == 0.0:
        return acc
    return [v / norm for v in acc]