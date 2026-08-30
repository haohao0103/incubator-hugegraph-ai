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

"""LLM gateway for the NL2SQL stack: retry, timeout and circuit breaker.

Production notes
----------------
A single chat endpoint (e.g. ``glm-5.3``) can transiently fail with HTTP 000
and self-recover. Without a gateway every blip surfaces as a failed request;
with one the request is retried with backoff and, after repeated failures, a
circuit breaker opens so downstream code can degrade gracefully (lexical-only
linking, cached schema, explicit 503) instead of hanging.

The breaker is *half-open* after the reset window: the next call is allowed
through, and a single success closes it again.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("hugegraph_llm.nl2sql.llm_gateway")


class LLMGatewayError(Exception):
    """Base error raised by the gateway."""


class LLMGatewayOpenError(LLMGatewayError):
    """Circuit is open (or half-open probe failed) -- LLM calls are refused."""


class LLMGatewayTimeoutError(LLMGatewayError):
    """The underlying LLM call exceeded ``timeout_s``."""


class LLMGateway:
    """Retry + timeout + circuit breaker around ``llm.generate(prompt)``.

    ``llm_factory`` is a zero-arg callable returning the configured chat LLM
    (e.g. ``LLMs().get_chat_llm``); it is invoked per attempt so a dead client
    can be re-created instead of reused.
    """

    def __init__(
        self,
        llm_factory: Callable[[], Any],
        max_retries: int = 3,
        retry_base_s: float = 0.5,
        timeout_s: float = 30.0,
        circuit_fail_threshold: int = 5,
        circuit_reset_s: float = 60.0,
    ) -> None:
        self._factory = llm_factory
        self._max_retries = max(1, max_retries)
        self._retry_base_s = retry_base_s
        self._timeout_s = timeout_s
        self._fail_threshold = circuit_fail_threshold
        self._reset_s = circuit_reset_s

        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0  # epoch seconds while open; 0 == closed
        self._last_error: Optional[str] = None
        self._total_attempts = 0
        self._total_failures = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llm-gateway"
        )

    # ------------------------------------------------------------------ state
    @property
    def is_open(self) -> bool:
        if self._open_until == 0.0:
            return False
        if time.monotonic() >= self._open_until:
            # reset window elapsed: probe allowed (half-open) on next call
            return False
        return True

    def state(self) -> Dict[str, Any]:
        with self._lock:
            open_now = self.is_open
            return {
                "circuit": "open" if open_now else "closed",
                "consecutive_failures": self._consecutive_failures,
                "total_attempts": self._total_attempts,
                "total_failures": self._total_failures,
                "last_error": self._last_error,
                "reset_in_s": max(0.0, self._open_until - time.monotonic())
                if self._open_until
                else 0.0,
            }

    # ---------------------------------------------------------------- calling
    def __call__(self, prompt: str) -> str:
        if self.is_open:
            raise LLMGatewayOpenError(
                f"LLM circuit open for {self.state()['reset_in_s']:.0f}s "
                f"(last error: {self._last_error})"
            )
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts <= self._max_retries:
            attempts += 1
            try:
                with self._lock:
                    self._total_attempts += 1
                result = self._invoke(prompt)
                with self._lock:
                    self._consecutive_failures = 0
                    self._open_until = 0.0
                    self._last_error = None
                if attempts > 1:
                    log.warning("llm gateway: recovered after %d attempt(s)", attempts)
                return result
            except Exception as exc:  # noqa: BLE001 - gateway must never die
                last_exc = exc
                with self._lock:
                    self._consecutive_failures += 1
                    self._total_failures += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    if self._consecutive_failures >= self._fail_threshold:
                        self._open_until = time.monotonic() + self._reset_s
                        log.warning(
                            "llm gateway: circuit OPEN after %d consecutive failures",
                            self._consecutive_failures,
                        )
                if attempts <= self._max_retries and not self.is_open:
                    delay = self._retry_base_s * (2 ** (attempts - 1))
                    log.warning(
                        "llm gateway: attempt %d failed (%s), retry in %.1fs",
                        attempts, exc, delay,
                    )
                    time.sleep(delay)
        raise LLMGatewayError(
            f"LLM failed after {attempts} attempts: {last_exc}"
        )

    def _invoke(self, prompt: str) -> str:
        """Run one call with a hard wall-clock timeout."""
        llm = self._factory()
        future = self._executor.submit(llm.generate, prompt=prompt)
        try:
            return future.result(timeout=self._timeout_s)
        except Exception as exc:  # noqa: BLE001 - wrap and re-raise below
            raise LLMGatewayTimeoutError(
                f"LLM call exceeded {self._timeout_s}s: {exc}"
            ) from exc
