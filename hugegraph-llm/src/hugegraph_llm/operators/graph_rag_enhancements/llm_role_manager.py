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

"""LLM Role Manager — per-role LLM routing with priority scheduling.

Inspired by LightRAG's ``_RoleLLMMixin`` (llm_roles.py), adapted for
HugeGraph-AI's operator architecture.

Key differences from LightRAG:
- No mixin pattern — standalone manager class
- 3 roles (extract/keyword/query) instead of 4 (skip vlm for now)
- Per-role asyncio.Semaphore instead of PriorityQueue
- Simplified hot config update (no builder registration)
- Compatible with CachedLLM wrapper chain

Usage:
    from hugegraph_llm.operators.graph_rag_enhancements.llm_role_manager import (
        LLMRoleManager, LLMRole, RoleLLMConfig,
    )

    manager = LLMRoleManager(
        base_llm=my_llm,
        role_configs={
            LLMRole.EXTRACT: RoleLLMConfig(max_async=16, timeout=120),
            LLMRole.KEYWORD: RoleLLMConfig(max_async=8, timeout=60),
            LLMRole.QUERY:   RoleLLMConfig(max_async=5, timeout=90),
        },
    )

    extract_func = manager.get_role_func(LLMRole.EXTRACT)
    result = extract_func("Extract entities from this text...")
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from hugegraph_llm.utils.log import log


class LLMRole(Enum):
    """Canonical LLM roles for GraphRAG operations.

    Maps to LightRAG's ROLES tuple (extract, keyword, query).
    VLM is excluded as P2 (future work).
    """
    EXTRACT = "extract"
    KEYWORD = "keyword"
    QUERY = "query"


# Priority ordering (higher number = lower priority)
ROLE_PRIORITY: Dict[LLMRole, int] = {
    LLMRole.QUERY: 5,     # User-facing, highest priority
    LLMRole.KEYWORD: 8,   # Keyword extraction, medium
    LLMRole.EXTRACT: 10,  # Entity/relation extraction, lowest (batch)
}

ROLE_NAMES = frozenset(r.value for r in LLMRole)


@dataclass
class RoleLLMConfig:
    """Per-role LLM override configuration.

    Any field left as None falls back to the base LLM settings.

    Inspired by LightRAG's RoleLLMConfig, simplified for HugeGraph-AI.
    """
    func: Optional[Callable[..., Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    max_async: Optional[int] = None
    timeout: Optional[int] = None
    priority: Optional[int] = None


@dataclass
class _RoleState:
    """Internal runtime state for one role."""
    func: Optional[Callable[..., Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    max_async: int = 16
    timeout: int = 180
    priority: int = 10
    semaphore: Optional[asyncio.Semaphore] = None
    call_count: int = 0
    error_count: int = 0
    last_call_time: float = 0.0


@dataclass
class RoleCallStats:
    """Observability snapshot for one role."""
    role: str
    max_async: int
    timeout: int
    priority: int
    call_count: int
    error_count: int
    last_call_time: float
    available: bool


class LLMRoleManager:
    """Manage per-role LLM routing with concurrency and priority control.

    Each role gets its own:
    - LLM function (or falls back to base LLM)
    - asyncio.Semaphore for concurrency control
    - Timeout and priority settings
    - Call statistics for observability

    This borrows the core idea from LightRAG's ``_RoleLLMMixin``:
    separate roles for extraction, keyword, and query, each with
    independent concurrency limits. However, we use a simpler
    Semaphore-based approach instead of LightRAG's PriorityQueue
    (which is designed for multi-worker deployment scenarios that
    HugeGraph-AI doesn't currently face).
    """

    def __init__(
        self,
        base_llm: Optional[Callable[..., Any]] = None,
        role_configs: Optional[Dict[LLMRole, RoleLLMConfig]] = None,
        default_max_async: int = 16,
        default_timeout: int = 180,
    ):
        """Initialize the role manager.

        Args:
            base_llm: Default LLM function used when no role-specific func is set.
            role_configs: Per-role override configs.
            default_max_async: Default concurrency limit if not overridden.
            default_timeout: Default timeout (seconds) if not overridden.
        """
        self._base_llm = base_llm
        self._default_max_async = default_max_async
        self._default_timeout = default_timeout

        # Initialize role states
        self._states: Dict[LLMRole, _RoleState] = {}
        for role in LLMRole:
            cfg = (role_configs or {}).get(role, RoleLLMConfig())
            self._states[role] = _RoleState(
                func=cfg.func or base_llm,
                kwargs=cfg.kwargs,
                max_async=cfg.max_async or default_max_async,
                timeout=cfg.timeout or default_timeout,
                priority=cfg.priority or ROLE_PRIORITY[role],
                semaphore=None,  # created lazily in async context
                call_count=0,
                error_count=0,
                last_call_time=0.0,
            )

        log.info(
            "LLMRoleManager initialized: %s roles, base_llm=%s",
            len(self._states),
            type(base_llm).__name__ if base_llm else "None",
        )

    def get_role_func(self, role: LLMRole) -> Optional[Callable[..., Any]]:
        """Get the LLM function for a specific role.

        Returns role-specific func if set, otherwise falls back to base LLM.
        """
        state = self._states[role]
        return state.func or self._base_llm

    async def call_role(
        self,
        role: LLMRole,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM through a specific role with concurrency control.

        Uses asyncio.Semaphore for per-role concurrency limiting,
        similar to LightRAG's priority_limit_async_func_call.

        Args:
            role: The role to use for this call.
            prompt: The prompt to send to the LLM.
            **kwargs: Additional kwargs merged with role kwargs.

        Returns:
            LLM response.
        """
        state = self._states[role]
        llm_func = state.func or self._base_llm

        if llm_func is None:
            raise ValueError(f"No LLM function available for role {role.value}")

        # Lazy semaphore creation
        if state.semaphore is None:
            state.semaphore = asyncio.Semaphore(state.max_async)

        # Merge kwargs: role defaults + call-specific overrides
        merged_kwargs = {}
        if state.kwargs:
            merged_kwargs.update(state.kwargs)
        merged_kwargs.update(kwargs)

        # Track call
        state.call_count += 1
        state.last_call_time = time.time()

        # Call with semaphore and timeout
        async with state.semaphore:
            try:
                # Check if the function is async
                if asyncio.iscoroutinefunction(llm_func):
                    result = await asyncio.wait_for(
                        llm_func(prompt, **merged_kwargs),
                        timeout=state.timeout,
                    )
                else:
                    # Sync function — run in thread pool
                    result = await asyncio.wait_for(
                        asyncio.to_thread(llm_func, prompt, **merged_kwargs),
                        timeout=state.timeout,
                    )
                return result
            except asyncio.TimeoutError:
                state.error_count += 1
                log.warning(
                    "LLMRoleManager: %s call timed out (%ds)",
                    role.value, state.timeout,
                )
                raise
            except Exception as e:
                state.error_count += 1
                log.error("LLMRoleManager: %s call failed: %s", role.value, e)
                raise

    def call_role_sync(
        self,
        role: LLMRole,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """Synchronous wrapper for call_role.

        For use in non-async contexts. Does NOT enforce concurrency limits.
        """
        state = self._states[role]
        llm_func = state.func or self._base_llm

        if llm_func is None:
            raise ValueError(f"No LLM function available for role {role.value}")

        merged_kwargs = {}
        if state.kwargs:
            merged_kwargs.update(state.kwargs)
        merged_kwargs.update(kwargs)

        state.call_count += 1
        state.last_call_time = time.time()

        return llm_func(prompt, **merged_kwargs)

    def update_role_config(
        self,
        role: LLMRole,
        func: Optional[Callable[..., Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        max_async: Optional[int] = None,
        timeout: Optional[int] = None,
        priority: Optional[int] = None,
    ) -> None:
        """Update a role's configuration at runtime.

        If max_async changes, the existing semaphore is invalidated
        and will be recreated on next async call.

        Inspired by LightRAG's update_llm_role_config, simplified
        (no builder registration, no queue cleanup).
        """
        state = self._states[role]

        if func is not None:
            state.func = func
        if kwargs is not None:
            state.kwargs = kwargs
        if max_async is not None:
            state.max_async = max_async
            # Invalidate semaphore — will be recreated lazily
            state.semaphore = None
        if timeout is not None:
            state.timeout = timeout
        if priority is not None:
            state.priority = priority

        log.info(
            "LLMRoleManager: updated role %s (max_async=%d, timeout=%d, priority=%d)",
            role.value, state.max_async, state.timeout, state.priority,
        )

    def get_role_config(self, role: Optional[LLMRole] = None) -> Dict[str, Any]:
        """Return effective role config snapshot (observability).

        If role is None, returns configs for all roles.
        Auth-bearing fields (api_key, etc.) are stripped from kwargs.
        """
        SECRET_MARKERS = ("api_key", "api-key", "apikey", "secret", "token", "password")

        def scrub(d: Dict[str, Any]) -> Dict[str, Any]:
            return {
                k: ("***" if any(m in k.lower() for m in SECRET_MARKERS) else v)
                for k, v in d.items()
            }

        def role_snapshot(r: LLMRole) -> Dict[str, Any]:
            state = self._states[r]
            return {
                "role": r.value,
                "max_async": state.max_async,
                "timeout": state.timeout,
                "priority": state.priority,
                "call_count": state.call_count,
                "error_count": state.error_count,
                "last_call_time": state.last_call_time,
                "has_func": state.func is not None or self._base_llm is not None,
                "kwargs": scrub(state.kwargs or {}),
            }

        if role is not None:
            return role_snapshot(role)
        return {r.value: role_snapshot(r) for r in LLMRole}

    def stats(self) -> Dict[str, Any]:
        """Aggregate stats across all roles."""
        total_calls = sum(s.call_count for s in self._states.values())
        total_errors = sum(s.error_count for s in self._states.values())
        return {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "roles": self.get_role_config(),
        }

    def reset_stats(self) -> None:
        """Reset all call counters."""
        for state in self._states.values():
            state.call_count = 0
            state.error_count = 0
            state.last_call_time = 0.0
