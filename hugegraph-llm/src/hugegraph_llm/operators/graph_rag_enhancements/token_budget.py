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
G3: Token Budget Manager — 对标 MS GraphRAG token_counter.py + budget.py

提供精确的Token计数、预算管理和OOM防护。
设计参考:
  - MS GraphRAG: packages/graphrag-llm/graphrag_llm/tokenizer/
  - Rate Limiter: packages/graphrag-llm/graphrag_llm/rate_limit/sliding_window_rate_limiter.py

特性:
  - tiktoken 精确计数（支持 cl100k_base / o200k_base 等）
  - 单次/全局双重预算上限
  - 超限自动截断策略 (truncate head/tail/preserve system)
  - 滑动窗口速率限制 (requests_per_period / tokens_per_period)
  - 预算耗尽时返回可读错误而非崩溃
"""

from __future__ import annotations

import time
import threading
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# Token Counter — tiktoken wrapper with fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


@dataclass(frozen=True)
class TokenCountResult:
    """Immutable result of a token counting operation."""
    num_tokens: int
    encoding_name: str = "unknown"


class TokenCounter:
    """Precise token counter using tiktoken with graceful fallback.

    Usage::

        tc = TokenCounter(model="gpt-4o")
        result = tc.count("Hello world")       # -> TokenCountResult(num_tokens=2, ...)
        result = tc.count_messages([{"role": "user", "content": "..."}])
        budget.check_and_adjust(result.num_tokens)  # raises if over budget
    """

    # Model → encoding name mapping (covers most common models)
    MODEL_ENCODING_MAP: Dict[str, str] = {
        # OpenAI
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        # Compatible models (MiMo, etc.)
        "mimo-v2.5-pro": "cl100k_base",
        "mimo-v2-flash": "cl100k_base",
        "default": "cl100k_base",  # fallback
    }

    def __init__(self, model: str = "gpt-4o", *, encoding: Optional[str] = None) -> None:
        self.model = model
        self._encoding_name = encoding or self.MODEL_ENCODING_MAP.get(
            model.lower(), self.MODEL_ENCODING_MAP["default"]
        )
        self._encoding = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(self._encoding_name)
            except Exception:
                warnings.warn(
                    f"tiktoken encoding '{self._encoding_name}' unavailable; "
                    "falling back to rough estimation"
                )

    @property
    def encoding_name(self) -> str:
        return self._encoding_name

    @property
    def available(self) -> bool:
        return self._encoding is not None

    def count(self, text: str) -> TokenCountResult:
        """Count tokens in a plain text string."""
        if self._encoding is not None:
            n = len(self._encoding.encode(text))
            return TokenCountResult(n, self._encoding_name)
        # Rough fallback: ~4 chars per token (conservative for CJK)
        n = len(text) // 3 if text else 0
        return TokenCountResult(n, "rough_estimate")

    def count_messages(self, messages: List[Dict[str, Any]]) -> TokenCountResult:
        """Count tokens in a list of OpenAI-style message dicts.

        Follows the convention used by OpenAI's chat format:
        each message adds 4 tokens (role/name delimiters), plus content tokens.
        """
        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content).num_tokens
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.count(block.get("text", "")).num_tokens
        # Add 2 for assistant reply priming (similar to OpenAI's approach)
        total += 2
        return TokenCountResult(total, self._encoding_name)

    def estimate_input_tokens(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Quick helper: return just the int count."""
        if messages:
            return self.count_messages(messages).num_tokens
        if prompt:
            return self.count(prompt).num_tokens
        return 0


# ---------------------------------------------------------------------------
# Budget Manager — OOM防护
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when a token budget would be exceeded."""

    def __init__(self, requested: int, limit: int, scope: str = "global") -> None:
        self.requested = requested
        self.limit = limit
        self.scope = scope
        super().__init__(
            f"Token budget exceeded: requested={requested}, "
            f"limit={limit} ({scope})"
        )


@dataclass
class BudgetConfig:
    """Budget configuration."""
    max_tokens_per_request: int = 8192      # Single request limit (model context window)
    max_tokens_global: int = 500_000        # Global session limit
    enable_truncation: bool = True          # Auto-truncate instead of raising
    truncation_strategy: str = "head"        # "head" | "tail" | "preserve_system"
    warn_threshold: float = 0.9             # Warn when usage exceeds this ratio


class TokenBudgetManager:
    """Manages token budgets with optional auto-truncation.

    Two-level budget:
    1. **Per-request**: ``max_tokens_per_request`` — prevents single call overflow.
    2. **Global session**: ``max_tokens_global`` — tracks cumulative usage across calls.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        *,
        counter: Optional[TokenCounter] = None,
    ) -> None:
        self.config = config or BudgetConfig()
        self.counter = counter or TokenCounter()
        self._global_used = 0
        self._lock = threading.Lock()

    @property
    def global_used(self) -> int:
        return self._global_used

    @property
    def global_remaining(self) -> int:
        return max(0, self.config.max_tokens_global - self._global_used)

    def check(self, num_tokens: int) -> bool:
        """Return ``True`` if *num_tokens* fits within per-request AND global budget."""
        return (
            num_tokens <= self.config.max_tokens_per_request
            and self._global_used + num_tokens <= self.config.max_tokens_global
        )

    def check_or_raise(self, num_tokens: int, *, scope: str = "request") -> int:
        """Check budget; raise :class:`BudgetExceededError` if over limit.

        Returns the approved token count (may be lower than *num_tokens*
        after adjustment).
        """
        if num_tokens > self.config.max_tokens_per_request:
            if self.config.enable_truncation:
                return self.config.max_tokens_per_request
            raise BudgetExceededError(
                num_tokens, self.config.max_tokens_per_request, scope=scope
            )
        with self._lock:
            if self._global_used + num_tokens > self.config.max_tokens_global:
                available = max(0, self.config.max_tokens_global - self._global_used)
                if available == 0:
                    raise BudgetExceededError(
                        num_tokens, self.config.max_tokens_global, scope="global"
                    )
                if self.config.enable_truncation:
                    return available
                raise BudgetExceededError(
                    num_tokens, self.config.max_tokens_global, scope="global"
                )
            return num_tokens

    def record_usage(self, num_tokens: int) -> None:
        """Record that *num_tokens* have been consumed (after successful LLM call)."""
        with self._lock:
            self._global_used += num_tokens

    def truncate_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        strategy: str = "",
    ) -> List[Dict[str, Any]]:
        """Truncate messages to fit within *max_tokens* budget.

        Strategies:
        - ``"head"``: Keep system + earliest user messages (drop from end).
        - ``"tail"``: Keep system + latest user messages (drop from start).
        - ``"preserve_system"``: Always keep all system/user role messages,
          truncate only assistant/tool content.
        """
        strategy = strategy or self.config.truncation_strategy
        if strategy == "preserve_system":
            return self._truncate_preserve_system(messages, max_tokens)
        if strategy == "tail":
            return self._truncate_tail(messages, max_tokens)
        # default: head
        return self._truncate_head(messages, max_tokens)

    def reset_global(self) -> None:
        """Reset the global usage tracker (new session)."""
        with self._lock:
            self._global_used = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "global_used": self._global_used,
            "global_limit": self.config.max_tokens_global,
            "per_request_limit": self.config.max_tokens_per_request,
            "global_remaining": self.global_remaining,
            "utilization": round(
                self._global_used / self.config.max_tokens_global, 4
            ),
        }

    # -- private truncation strategies -------------------------------------

    def _truncate_head(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Keep system messages + earliest non-system messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        budget_for_other = max_tokens - sum(
            self.counter.count(m.get("content", "")).num_tokens for m in system_msgs
        ) - 10  # small buffer
        if budget_for_other <= 0:
            return system_msgs[-1:] if system_msgs else []

        trimmed: List[Dict[str, Any]] = list(system_msgs)
        current_tokens = 0
        for msg in other_msgs:
            t = self.counter.count(msg.get("content", "")).num_tokens
            if current_tokens + t > budget_for_other:
                break
            trimmed.append(msg)
            current_tokens += t
        return trimmed

    def _truncate_tail(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Keep system messages + latest non-system messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        budget_for_other = max_tokens - sum(
            self.counter.count(m.get("content", "")).num_tokens for m in system_msgs
        ) - 10
        if budget_for_other <= 0:
            return system_msgs[-1:] if system_msgs else []

        trimmed: List[Dict[str, Any]] = list(system_msgs)
        current_tokens = 0
        for msg in reversed(other_msgs):
            t = self.counter.count(msg.get("content", "")).num_tokens
            if current_tokens + t > budget_for_other:
                break
            trimmed.insert(len(system_msgs), msg)
            current_tokens += t
        return trimmed

    def _truncate_preserve_system(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Truncate only non-system/non-user messages first (tool results, assistant)."""
        priority_roles = {"system", "user"}
        high: List[Dict[str, Any]] = []
        low: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") in priority_roles:
                high.append(m)
            else:
                low.append(m)

        high_cost = sum(
            self.counter.count(m.get("content", "")).num_tokens for m in high
        )
        remaining = max_tokens - high_cost - 10
        if remaining <= 0:
            return [high[-1]] if high else []

        trimmed = list(high)
        for msg in low:
            t = self.counter.count(msg.get("content", "")).num_tokens
            if remaining - t < 0:
                break
            trimmed.append(msg)
            remaining -= t
        return trimmed


# ---------------------------------------------------------------------------
# Sliding Window Rate Limiter — 对标 MS GraphRAG SlidingWindowRateLimiter
# ---------------------------------------------------------------------------


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter for LLM API calls.

    Supports two independent limits:
    - **Requests per period** (RPM/RPS style).
    - **Tokens per period** (TPM style).

    Usage as a context manager::

        limiter = SlidingWindowRateLimiter(tokens_per_period=100_000)
        with limiter.acquire(token_count=500):
            response = await llm.agenerate(...)
    """

    def __init__(
        self,
        *,
        period_in_seconds: float = 60.0,
        requests_per_period: Optional[int] = None,
        tokens_per_period: Optional[int] = None,
    ) -> None:
        self._period = period_in_seconds
        self._rpp = requests_per_period
        self._tpp = tokens_per_period
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._token_counts: deque[int] = deque()
        self._last_acquire_time: Optional[float] = None
        self._stagger: float = 0.0
        if self._rpp and self._rpp > 0:
            self._stagger = self._period / self._rpp

    @contextmanager
    def acquire(self, token_count: int = 0) -> Generator[None, None, None]:
        """Block until the request can proceed within rate limits."""
        while True:
            with self._lock:
                now = time.monotonic()

                # Evict expired entries outside the sliding window
                while (
                    self._request_times
                    and self._request_times[0] < now - self._period
                ):
                    self._request_times.popleft()
                    self._token_counts.popleft()

                # Check request limit
                if self._rpp and len(self._request_times) >= self._rpp:
                    pass  # need to wait, continue loop
                elif self._tpp and sum(self._token_counts) >= self._tpp:
                    pass  # need to wait, continue loop
                elif (
                    self._tpp
                    and token_count <= self._tpp
                    and sum(self._token_counts) + token_count > self._tpp
                ):
                    pass  # current request would push us over
                else:
                    # Stagger evenly across the window
                    if (
                        self._stagger > 0
                        and self._last_acquire_time is not None
                    ):
                        elapsed = now - self._last_acquire_time
                        if elapsed < self._stagger:
                            wait = self._stagger - elapsed
                            self._lock.release()
                            time.sleep(wait)
                            self._lock.acquire()
                            now = time.monotonic()  # refresh after sleep

                    self._request_times.append(now)
                    self._token_counts.append(token_count)
                    self._last_acquire_time = now
                    break
            if not self._lock.locked():
                time.sleep(0.01)  # brief backoff before re-checking

        yield

    def reset(self) -> None:
        """Clear all rate-limiting state."""
        with self._lock:
            self._request_times.clear()
            self._token_counts.clear()
            self._last_acquire_time = None


# ---------------------------------------------------------------------------
# Combined LLM Guard — Cache + Budget + Rate Limit integration point
# ---------------------------------------------------------------------------

@dataclass
class LLMCallGuard:
    """Convenience container holding all guard components for an LLM client."""

    cache: Any = field(default_factory=lambda: _noop_cache_instance())
    budget: TokenBudgetManager = field(default_factory=TokenBudgetManager)
    rate_limiter: SlidingWindowRateLimiter = field(
        default_factory=lambda: SlidingWindowRateLimiter(tokens_per_period=100_000)
    )
    counter: TokenCounter = field(default_factory=TokenCounter)


def _noop_cache_instance():
    from hugegraph_llm.operators.graph_rag_enhancements.llm_cache import NoopCache
    return NoopCache()
