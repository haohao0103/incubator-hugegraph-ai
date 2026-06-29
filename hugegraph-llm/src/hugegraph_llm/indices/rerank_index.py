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
Rerank module for AI memory retrieval.

Supports:
- Local sentence-transformers cross-encoder (default: ms-marco-MiniLM-L-6-v2)
- API-based rerankers (Jina AI, Cohere, OpenAI)

Usage:
    from hugegraph_llm.indices.rerank_index import get_reranker
    reranker = get_reranker()
    ranked = reranker.rerank("query", [{"id": "m1", "text": "..."}, ...], top_k=5)
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.utils.log import log


def get_reranker() -> "BaseReranker":
    """Factory: return the configured reranker backend."""
    if not memory_settings.rerank_enabled:
        return NoOpReranker()

    backend = memory_settings.rerank_backend
    if backend == "sentence_transformers":
        return SentenceTransformersReranker(
            model_name=memory_settings.rerank_model,
            batch_size=memory_settings.rerank_batch_size,
        )
    if backend == "jina":
        return JinaReranker(
            api_key=memory_settings.rerank_api_key or os.environ.get("JINA_API_KEY"),
            api_base=memory_settings.rerank_api_base or "https://api.jina.ai/v1/rerank",
            model=memory_settings.rerank_model or "jina-reranker-v2-base-multilingual",
        )
    if backend == "cohere":
        return CohereReranker(
            api_key=memory_settings.rerank_api_key or os.environ.get("COHERE_API_KEY"),
            model=memory_settings.rerank_model or "rerank-multilingual-v3.0",
        )
    if backend == "openai":
        return OpenAIReranker(
            api_key=memory_settings.rerank_api_key or os.environ.get("OPENAI_API_KEY"),
            api_base=memory_settings.rerank_api_base or "https://api.openai.com/v1",
            model=memory_settings.rerank_model or "gpt-4o-mini",
        )
    log.warning("Unknown rerank backend %s, fallback to no-op", backend)
    return NoOpReranker()


class BaseReranker(ABC):
    """Unified rerank interface."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query: user query
            candidates: list of dicts with at least 'id' and 'text' keys
            top_k: number of top results to return

        Returns:
            candidates sorted by rerank score, each with added 'rerank_score'
        """

    def _copy_meta(self, cand: Dict[str, Any], score: float) -> Dict[str, Any]:
        out = dict(cand)
        out["rerank_score"] = round(float(score), 6)
        # keep backward compatibility with 'score' field
        out["score"] = out["rerank_score"]
        return out


class NoOpReranker(BaseReranker):
    """Pass-through reranker when rerank is disabled."""

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        return candidates[:top_k]


class SentenceTransformersReranker(BaseReranker):
    """Local cross-encoder reranker (no API key, privacy-safe)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local reranker. "
                "Install: pip install sentence-transformers"
            ) from exc
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c.get("text", c.get("content", ""))) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scored = [(score, cand) for score, cand in zip(scores, candidates)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._copy_meta(cand, score) for score, cand in scored[:top_k]]


class JinaReranker(BaseReranker):
    """Jina AI Reranker API."""

    def __init__(self, api_key: Optional[str], api_base: str, model: str):
        if not api_key:
            raise ValueError("JINA_API_KEY or RERANK_API_KEY is required")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        docs = [c.get("text", c.get("content", "")) for c in candidates]
        resp = requests.post(
            f"{self.api_base}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "query": query, "documents": docs, "top_n": top_k},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        index_map = {i: c for i, c in enumerate(candidates)}
        ranked = []
        for r in results:
            idx = r.get("index")
            if idx is not None and idx in index_map:
                ranked.append(self._copy_meta(index_map[idx], r.get("relevance_score", 0.0)))
        return ranked


class CohereReranker(BaseReranker):
    """Cohere Rerank API."""

    def __init__(self, api_key: Optional[str], model: str):
        if not api_key:
            raise ValueError("COHERE_API_KEY or RERANK_API_KEY is required")
        self.api_key = api_key
        self.model = model

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        docs = [c.get("text", c.get("content", "")) for c in candidates]
        resp = requests.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "query": query, "documents": docs, "top_n": top_k},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        index_map = {i: c for i, c in enumerate(candidates)}
        ranked = []
        for r in results:
            idx = r.get("index")
            if idx is not None and idx in index_map:
                ranked.append(self._copy_meta(index_map[idx], r.get("relevance_score", 0.0)))
        return ranked


class OpenAIReranker(BaseReranker):
    """
    LLM-as-reranker fallback.
    Uses the chat model to score query-document relevance (0-10 scale),
    then normalizes to 0-1. Not as accurate as true cross-encoder but
    works with any OpenAI-compatible endpoint.
    """

    def __init__(self, api_key: Optional[str], api_base: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY or RERANK_API_KEY is required")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        scored = []
        for cand in candidates:
            text = cand.get("text", cand.get("content", ""))
            prompt = (
                f"Rate the relevance of the following memory to the query on a scale of 0-10.\n\n"
                f"Query: {query}\nMemory: {text}\n\n"
                f"Respond with only a number."
            )
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                raw = resp.choices[0].message.content.strip()
                score = max(0.0, min(10.0, float("".join(c for c in raw if c.isdigit() or c == "."))))
                scored.append((score / 10.0, cand))
            except Exception as e:
                log.warning("OpenAI rerank error for candidate %s: %s", cand.get("id"), e)
                scored.append((0.0, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._copy_meta(cand, score) for score, cand in scored[:top_k]]
