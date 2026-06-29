# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not obtain a copy of this License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CachedLLM — LLM Response Cache + Token Budget + Rate Limit Wrapper

Wraps any BaseLLM instance to add:
1. LLM Response Cache (disk-based JSON, hash-keyed by prompt)
2. Token Budget enforcement (per-request + session-wide limits)
3. Rate Limiting (sliding window RPM/TPM)
4. Hit rate tracking and statistics

Usage:
    from hugegraph_llm.models.llms.base import BaseLLM
    from hugegraph_llm.operators.graph_rag_enhancements.cached_llm import CachedLLM

    raw_llm = OpenAILLM(...)  # your existing LLM
    llm = CachedLLM(
        raw_llm,
        cache_dir=".cache/llm_responses",
        max_tokens_per_request=4096,
        max_session_tokens=100000,
        rpm_limit=60,
    )
    # Now all generate() calls go through cache → budget → rate-limit → LLM
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from hugegraph_llm.models.llms.base import BaseLLM

log = logging.getLogger(__name__)

# Import our enhancement modules with fallback if not available
try:
    from hugegraph_llm.operators.graph_rag_enhancements.llm_cache import (
        JsonFileCache,
        InMemoryCache,
        NoopCache,
        create_llm_cache,
    )
    from hugegraph_llm.operators.graph_rag_enhancements.token_budget import (
        TokenCounter,
        TokenBudgetManager,
        BudgetExceededError,
        SlidingWindowRateLimiter,
    )
    _ENHANCEMENTS_AVAILABLE = True
except ImportError:
    _ENHANCEMENTS_AVAILABLE = False
    log.warning(
        "GraphRAG enhancement modules not available. "
        "CachedLLM will operate in pass-through mode. "
        "Install/enable graph_rag_enhancements package for full functionality."
    )


class CachedLLM(BaseLLM):
    """
    Transparent wrapper around BaseLLM that adds caching, budget control,
    and rate limiting.

    All BaseLLM interface methods are delegated to the wrapped LLM instance,
    with caching applied only to stateless generate()/agenerate() calls.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        enable_budget: bool = True,
        enable_rate_limit: bool = True,
        # Cache settings
        cache_backend: str = "json_file",  # json_file | memory | noop
        default_ttl_seconds: int = 86400,  # 24 hours
        max_cache_size_mb: float = 500,
        # Budget settings
        max_tokens_per_request: int = 8192,
        max_session_tokens: int = 500000,  # ~500K tokens per session
        budget_model: str = "cl100k_base",  # tiktoken model
        truncate_mode: str = "tail",  # head | tail | preserve_system
        # Rate limit settings
        rpm_limit: int = 60,  # requests per minute
        tpm_limit: int = 1000000,  # tokens per minute
        window_seconds: int = 60,
    ):
        """
        Args:
            llm: The underlying BaseLLM instance to wrap.
            cache_dir: Directory for disk-based cache files.
            enable_cache: Whether to enable response caching.
            enable_budget: Whether to enable token budget enforcement.
            enable_rate_limit: Whether to enable rate limiting.
            cache_backend: Type of cache backend to use.
            default_ttl_seconds: Cache TTL in seconds.
            max_cache_size_mb: Maximum cache size in MB.
            max_tokens_per_request: Max output tokens per single request.
            max_session_tokens: Max total tokens across the session.
            budget_model: Tiktoken model name for counting.
            truncate_mode: How to truncate over-budget prompts.
            rpm_limit: Max requests per minute.
            tpm_limit: Max tokens per minute.
            window_seconds: Sliding window size in seconds.
        """
        self._inner = llm
        self._enable_cache = enable_cache and _ENHANCEMENTS_AVAILABLE
        self._enable_budget = enable_budget and _ENHANCEMENTS_AVAILABLE
        self._enable_rate_limit = enable_rate_limit and _ENHANCEMENTS_AVAILABLE

        # ── Statistics ──
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "budget_skips": 0,
            "total_calls": 0,
            "total_tokens_saved": 0,
        }

        # ── Initialize Cache ──
        self._cache = None
        if self._enable_cache:
            try:
                if cache_backend == "noop" or not cache_dir:
                    self._cache = NoopCache()
                elif cache_backend == "memory":
                    self._cache = InMemoryCache(default_ttl=default_ttl_seconds)
                else:
                    self._cache = create_llm_cache(
                        backend=cache_backend,
                        cache_dir=cache_dir or ".cache/llm_responses",
                        default_ttl_seconds=default_ttl_seconds,
                        max_size_mb=max_cache_size_mb,
                    )
                log.info("CachedLLM: cache initialized (%s)", type(self._cache).__name__)
            except Exception as e:
                log.warning("CachedLLM: failed to init cache, disabling: %s", e)
                self._cache = NoopCache()
                self._enable_cache = False
        else:
            self._cache = NoopCache()

        # ── Initialize Token Budget ──
        self._budget = None
        if self._enable_budget:
            try:
                counter = TokenCounter(model_name=budget_model)
                self._budget = TokenBudgetManager(
                    token_counter=counter,
                    max_tokens_per_request=max_tokens_per_request,
                    max_session_tokens=max_session_tokens,
                    truncate_mode=truncate_mode,
                )
                log.info(
                    "CachedLLM: budget initialized (max_req=%d, max_session=%d)",
                    max_tokens_per_request,
                    max_session_tokens,
                )
            except Exception as e:
                log.warning("CachedLLM: failed to init budget, disabling: %s", e)
                self._budget = None
                self._enable_budget = False

        # ── Initialize Rate Limiter ──
        self._rate_limiter = None
        if self._enable_rate_limit:
            try:
                self._rate_limiter = SlidingWindowRateLimiter(
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                    window_seconds=window_seconds,
                )
                log.info(
                    "CachedLLM: rate limiter initialized (RPM=%d, TPM=%d)",
                    rpm_limit,
                    tpm_limit,
                )
            except Exception as e:
                log.warning("CachedLLM: failed to init rate limiter, disabling: %s", e)
                self._rate_limiter = None
                self._enable_rate_limit = False

        log.info(
            "CachedLLM: wrapped %s [cache=%s, budget=%s, rate_limit=%s]",
            type(llm).__name__,
            self._enable_cache,
            self._enable_budget,
            self._enable_rate_limit,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Core delegation methods
    # ═══════════════════════════════════════════════════════════════════

    def _make_cache_key(self, prompt: Optional[str] = None,
                        messages: Optional[List[Dict]] = None) -> str:
        """Generate deterministic cache key from prompt/messages."""
        raw = json.dumps({"p": prompt, "m": messages}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def generate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate with cache → budget → rate-limit → LLM pipeline."""
        self._stats["total_calls"] += 1
        cache_key = self._make_cache_key(prompt=prompt, messages=messages)

        # 1) Check cache
        if self._enable_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                self._stats["total_tokens_saved"] += len(cached) // 4  # rough token estimate
                log.debug("CachedLLM: CACHE HIT key=%s", cache_key[:12])
                return cached
            self._stats["cache_misses"] += 1

        # 2) Check token budget
        if self._enable_budget and self._budget:
            text_to_check = prompt or (
                json.dumps(messages) if messages else ""
            )
            try:
                self._budget.check_or_raise(text_to_check)
            except BudgetExceededError as e:
                self._stats["budget_skips"] += 1
                log.warning("CachedLLM: BUDGET SKIP: %s", e)
                return f"[BUDGET_EXCEEDED] {e}"

        # 3) Rate limit wait
        if self._enable_rate_limit and self._rate_limiter:
            self._rate_limiter.wait_if_needed()

        # 4) Call actual LLM
        result = self._inner.generate(messages=messages, prompt=prompt)

        # 5) Store in cache
        if self._enable_cache and result:
            try:
                self._cache.set(cache_key, result)
            except Exception as e:
                log.debug("CachedLLM: cache set error (non-fatal): %s", e)

        return result

    async def agenerate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Async generate with same cache/budget/rate-limit pipeline."""
        self._stats["total_calls"] += 1
        cache_key = self._make_cache_key(prompt=prompt, messages=messages)

        if self._enable_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                self._stats["total_tokens_saved"] += len(cached) // 4
                return cached
            self._stats["cache_misses"] += 1

        if self._enable_budget and self._budget:
            text_to_check = prompt or (json.dumps(messages) if messages else "")
            try:
                self._budget.check_or_raise(text_to_check)
            except BudgetExceededError as e:
                self._stats["budget_skips"] += 1
                return f"[BUDGET_EXCEEDED] {e}"

        if self._enable_rate_limit and self._rate_limiter:
            self._rate_limiter.wait_if_needed()

        result = await self._inner.agenerate(messages=messages, prompt=prompt)

        if self._enable_cache and result:
            try:
                self._cache.set(cache_key, result)
            except Exception:
                pass

        return result

    def generate_streaming(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        on_token_callback: Optional[Any] = None,
    ):
        """Streaming generate — NOT cached (stateful)."""
        # Apply rate limit but skip cache for streaming
        if self._enable_rate_limit and self._rate_limiter:
            self._rate_limiter.wait_if_needed()
        yield from self._inner.generate_streaming(
            messages=messages, prompt=prompt, on_token_callback=on_token_callback
        )

    async def agenerate_streaming(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        on_token_callback: Optional[Any] = None,
    ):
        """Async streaming — NOT cached."""
        if self._enable_rate_limit and self._rate_limiter:
            self._rate_limiter.wait_if_needed()
        async for chunk in self._inner.agenerate_streaming(
            messages=messages, prompt=prompt, on_token_callback=on_token_callback
        ):
            yield chunk

    # ═══════════════════════════════════════════════════════════════════
    # Delegated pass-through methods (no caching)
    # ═══════════════════════════════════════════════════════════════════

    def num_tokens_from_string(self, string: str) -> int:
        return self._inner.num_tokens_from_string(string)

    def max_allowed_token_length(self) -> int:
        return self._inner.max_allowed_token_length()

    def get_llm_type(self) -> str:
        return self._inner.get_llm_type()

    # ═══════════════════════════════════════════════════════════════════
    # Statistics & management
    # ═══════════════════════════════════════════════════════════════════

    @property
    def stats(self) -> Dict[str, int]:
        """Return current usage statistics."""
        s = dict(self._stats)
        if hasattr(self._cache, 'stats'):
            s.update(self._cache.stats.snapshot())
        if self._budget:
            s["session_tokens_used"] = self._budget.session_tokens_used
            s["session_tokens_remaining"] = (
                self._budget.max_session_tokens - self._budget.session_tokens_used
            )
        if self._rate_limiter:
            rl_stats = self._rate_limiter.stats()
            s.update(rl_stats)
        return s

    def stats_summary(self) -> str:
        """Human-readable statistics string."""
        s = self.stats
        total = s.get("total_calls", 0)
        hits = s.get("cache_hits", 0)
        miss = s.get("cache_misses", 0)
        hit_rate = hits / total * 100 if total > 0 else 0
        saved = s.get("total_tokens_saved", 0)
        skipped = s.get("budget_skips", 0)

        parts = [
            f"CachedLLM stats: {total} calls",
            f"  hit={hits} ({hit_rate:.1f}%) miss={miss}",
            f"  ~saved {saved} tokens, {skipped} budget-skips",
        ]
        if self._budget:
            parts.append(
                f"  session tokens: {s.get('session_tokens_used', 0)} / "
                f"{s.get('max_session_tokens', '?')}"
            )
        return "\n".join(parts)

    def reset_stats(self):
        """Reset all counters to zero."""
        self._stats = {
            "cache_hits": 0, "cache_misses": 0, "budget_skips": 0,
            "total_calls": 0, "total_tokens_saved": 0,
        }
        if self._budget:
            self._budget.reset()

    def clear_cache(self):
        """Clear all cached responses."""
        if hasattr(self._cache, 'clear'):
            self._cache.clear()
            log.info("CachedLLM: cache cleared")

    @property
    def inner_llm(self) -> BaseLLM:
        """Access the wrapped LLM instance (for advanced use)."""
        return self._inner
