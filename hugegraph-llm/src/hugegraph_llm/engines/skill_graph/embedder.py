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
import logging
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from hugegraph_llm.engines.skill_graph.errors import EmbeddingMismatchError

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface used across the skill graph.

    ``identity`` is the *vector-space fingerprint* persisted in the build
    manifest. It must change whenever the underlying model changes, even if the
    dimension stays the same - two 1536-d models still live in different spaces.
    """

    @property
    def identity(self) -> str:  # pragma: no cover - protocol
        ...

    def embed(self, text: str) -> List[float]:  # pragma: no cover - protocol
        ...


def embedder_identity(embedder: object) -> str:
    """Best-effort stable identity for any embedder-like object."""
    value = getattr(embedder, "identity", None)
    if callable(value):
        try:
            value = value()
        except Exception:  # pragma: no cover - defensive
            value = None
    if value:
        return str(value)
    return f"{type(embedder).__name__.lower()}:unknown"


def embedder_dim(embedder: object) -> int:
    """Declared output dimension (0 when the backend only reports it per call)."""
    value = getattr(embedder, "dim", None)
    try:
        return int(value) if value else 0
    except (TypeError, ValueError):
        return 0


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

    @property
    def identity(self) -> str:
        # Versioned so a future tokenizer change invalidates old vectors.
        return f"hash-v1:{self.dim}"

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


def _parse_headers(raw: Optional[str]) -> Dict[str, str]:
    """Parse ``OPENAI_DEFAULT_HEADERS`` (``k=v,k2=v2`` or ``k:v``) into a dict."""
    headers: Dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        for sep in ("=", ":"):
            if sep in item:
                key, _, value = item.partition(sep)
                headers[key.strip()] = value.strip()
                break
    return headers


class HttpEmbedder:
    """OpenAI-shaped embedder that POSTs to an explicit URL.

    Some gateways expose embeddings on a non-standard path (e.g. ``/embedding``
    rather than ``/v1/embeddings``) and require extra request headers. The
    OpenAI SDK always appends ``/embeddings`` to ``base_url``, so it cannot be
    pointed at those gateways; this thin adapter can. The URL, model, key and
    headers default to the project's ``OPENAI_EMBEDDING_*`` /
    ``OPENAI_DEFAULT_HEADERS`` environment variables.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        dim: Optional[int] = None,
        timeout: float = 60.0,
    ):
        self.url = url or os.environ.get("OPENAI_EMBEDDING_URL", "")
        if not self.url:
            raise ValueError("HttpEmbedder requires a url (or OPENAI_EMBEDDING_URL)")
        self.model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_EMBEDDING_API_KEY", "")
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers.update(_parse_headers(os.environ.get("OPENAI_DEFAULT_HEADERS")))
        if headers:
            self.headers.update(headers)
        self.dim = dim
        self.timeout = timeout

    @property
    def identity(self) -> str:
        # The URL may embed credentials/internal hosts, so fingerprint it rather
        # than persisting it in artifacts.
        url_fp = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:12]
        return f"http:{self.model}#{url_fp}"

    def _post(self, inputs: List[str]) -> List[List[float]]:
        import httpx  # imported lazily so the module stays import-light

        payload: Dict[str, object] = {"input": inputs, "model": self.model}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        items = data["data"] if isinstance(data, dict) else data
        vectors = [item["embedding"] for item in items]
        if self.dim is None and vectors:
            self.dim = len(vectors[0])
        return vectors

    def embed(self, text: str) -> List[float]:
        return self._post([text])[0]

    def embed_many(self, texts: Iterable[str], batch_size: int = 32) -> List[List[float]]:
        texts = list(texts)
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                vecs = self._post(batch)
            except Exception as exc:  # pragma: no cover - gateway may reject batches
                log.warning("batch embedding failed (%s); falling back to per-item", exc)
                vecs = [self._post([t])[0] for t in batch]
            out.extend(vecs)
        return out


class _ProjectEmbedderAdapter:
    """Adapter around the project's ``Embeddings`` backends (OpenAI/Ollama/...)."""

    _MODEL_ATTRS = ("model_name", "model", "model_id", "llm_model_name", "_model_name")
    _DIM_ATTRS = ("dim", "dimension", "dimensions", "embedding_dim", "_dim")

    def __init__(self, backend: object):
        self._backend = backend
        self.dim = self._declared_dim()

    def _declared_dim(self) -> Optional[int]:
        for attr in self._DIM_ATTRS:
            value = getattr(self._backend, attr, None)
            try:
                if value:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _model_name(self) -> Optional[str]:
        for attr in self._MODEL_ATTRS:
            value = getattr(self._backend, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @property
    def identity(self) -> str:
        cls = type(self._backend).__name__
        model = self._model_name() or "unknown"
        # A different backend class with the same model name can still expose a
        # different endpoint/version, so both parts are part of the fingerprint.
        return f"project:{cls}#{model}"

    def embed(self, text: str) -> List[float]:
        vec = list(self._backend.get_text_embedding(text))  # type: ignore[attr-defined]
        if self.dim is None and vec:
            self.dim = len(vec)
        return vec


def resolve_embedder(dim: int = 256, strict: bool = False) -> Embedder:
    """Return the best available production embedder, else the hashing fallback.

    Resolution order:

    1. :class:`HttpEmbedder` when ``OPENAI_EMBEDDING_URL`` is set — this covers
       gateways that expose embeddings on a non-standard path.
    2. The project ``Embeddings`` backend (OpenAI/Ollama/LiteLLM) if importable.
    3. The dependency-free :class:`HashingEmbedder`.

    ``strict=True`` (used for production/service paths) never falls back: a
    misconfigured production embedder must fail the query, because the hashing
    fallback would silently compare 256-d vectors against stored 1536-d ones
    (previously yielding all-zero similarities and an "empty" semantic channel).
    """
    errors: List[str] = []
    if os.environ.get("OPENAI_EMBEDDING_URL"):
        try:
            return HttpEmbedder()
        except Exception as exc:
            errors.append(f"HttpEmbedder: {exc}")
            if strict:
                raise EmbeddingMismatchError(
                    f"OPENAI_EMBEDDING_URL is set but the embedder could not be initialised: {exc}"
                ) from exc
            log.warning("HttpEmbedder init failed; trying project embedder")
    try:
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings

        return _ProjectEmbedderAdapter(Embeddings().get_embedding())
    except Exception as exc:
        errors.append(f"project embedder: {exc}")
        if strict:
            raise EmbeddingMismatchError(
                "no production embedder is available (OPENAI_EMBEDDING_URL unset and the "
                f"project Embeddings backend is unusable): {'; '.join(errors)}. "
                "Set the embedding configuration or pass an explicit embedder."
            ) from exc
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