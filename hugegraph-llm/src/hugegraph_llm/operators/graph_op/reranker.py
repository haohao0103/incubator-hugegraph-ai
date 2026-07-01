"""Reranking interface for GraphRAG retrieval results.

Inspired by LightRAG's ``rerank.py`` but built with HugeGraph-AI's own
architecture.  Provides multiple reranking strategies (identity, cross-encoder,
LLM-based) behind a common async interface.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional


@dataclass
class RerankResult:
    """A single reranked document with its original index, text, and score."""

    index: int
    text: str
    score: float


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[RerankResult]:
        """Rerank *documents* with respect to *query*, returning up to *top_k* results."""
        ...  # pragma: no cover


class IdentityReranker(BaseReranker):
    """Fallback reranker that preserves original order with score=1.0."""

    async def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[RerankResult]:
        results = [
            RerankResult(index=i, text=doc, score=1.0)
            for i, doc in enumerate(documents)
        ]
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Reranker using a sentence-transformers CrossEncoder model.

    Falls back to ``IdentityReranker`` if ``sentence_transformers`` is not
    installed.  The model is loaded lazily on first use.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._identity = IdentityReranker()
        self._available: Optional[bool] = None

    def _try_load_model(self) -> bool:
        """Try to import and instantiate the CrossEncoder.  Returns True on success."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self._model_name)
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    async def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[RerankResult]:
        if self._available is None:
            self._try_load_model()

        if not self._available:
            return await self._identity.rerank(query, documents, top_k)

        # Build pairs and compute scores via CrossEncoder
        pairs = [(query, doc) for doc in documents]
        # CrossEncoder.predict is synchronous; run in thread via asyncio
        import asyncio

        scores = await asyncio.to_thread(self._model.predict, pairs)  # type: ignore[union-attr]

        results = [
            RerankResult(index=i, text=doc, score=float(scores[i]))
            for i, doc in enumerate(documents)
        ]
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class LLMReranker(BaseReranker):
    """Reranker that uses an LLM to score document relevance.

    Accepts either a synchronous ``llm_func`` or an async
    ``llm_func_async`` callable.  The LLM is asked to rate relevance on a
    0–10 scale, and the numeric score is parsed from its output.
    """

    _PROMPT_TEMPLATE = (
        "Rate the relevance of this document to the query on a scale of 0-10.\n"
        "Query: {query}\n"
        "Document: {document}\n"
        "Output only the numeric score."
    )

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        llm_func_async: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> None:
        self._llm_func = llm_func
        self._llm_func_async = llm_func_async

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract the first floating-point number from *raw* LLM output."""
        match = re.search(r"(\d+\.?\d*)", raw)
        if match:
            return float(match.group(1))
        return 0.0

    async def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[RerankResult]:
        import asyncio

        results: List[RerankResult] = []

        if self._llm_func_async is not None:
            tasks = []
            for i, doc in enumerate(documents):
                prompt = self._PROMPT_TEMPLATE.format(query=query, document=doc)
                tasks.append((i, doc, prompt))
            coros = [self._llm_func_async(p) for _, _, p in tasks]
            raw_scores = await asyncio.gather(*coros)
            for (i, doc, _), raw in zip(tasks, raw_scores):
                results.append(RerankResult(index=i, text=doc, score=self._parse_score(raw)))
        elif self._llm_func is not None:
            for i, doc in enumerate(documents):
                prompt = self._PROMPT_TEMPLATE.format(query=query, document=doc)
                raw = await asyncio.to_thread(self._llm_func, prompt)
                results.append(RerankResult(index=i, text=doc, score=self._parse_score(raw)))
        else:
            # No LLM function provided — fallback to identity
            return await IdentityReranker().rerank(query, documents, top_k)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class RerankerFactory:
    """Factory to create rerankers by type name."""

    @staticmethod
    def create(reranker_type: str = "identity", **kwargs) -> BaseReranker:
        """Create a reranker instance.

        Supported *reranker_type* values: ``"identity"``, ``"cross_encoder"``,
        ``"llm"``.  Falls back to ``IdentityReranker`` if the requested type
        is unavailable (e.g. missing dependencies).
        """
        if reranker_type == "identity":
            return IdentityReranker()
        elif reranker_type == "cross_encoder":
            reranker = CrossEncoderReranker(**kwargs)
            # Probe availability; if not available, fall back
            if not reranker._try_load_model():
                return IdentityReranker()
            return reranker
        elif reranker_type == "llm":
            return LLMReranker(**kwargs)
        else:
            return IdentityReranker()
