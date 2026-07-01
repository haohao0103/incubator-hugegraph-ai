# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Tests for LLM Role Manager module."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from hugegraph_llm.operators.graph_rag_enhancements.llm_role_manager import (
    LLMRole,
    LLMRoleManager,
    RoleLLMConfig,
    RoleCallStats,
    ROLE_PRIORITY,
    ROLE_NAMES,
)


# ── Test fixtures ──

def _make_sync_llm():
    """Create a mock synchronous LLM function."""
    def mock_llm(prompt, **kwargs):
        return f"Response to: {prompt[:50]}"
    return mock_llm


def _make_async_llm():
    """Create a mock async LLM function."""
    async def mock_llm(prompt, **kwargs):
        await asyncio.sleep(0.01)
        return f"Async response to: {prompt[:50]}"
    return mock_llm


def _make_slow_async_llm(timeout_seconds=10):
    """Create a mock async LLM that takes a configurable time."""
    async def mock_llm(prompt, **kwargs):
        await asyncio.sleep(timeout_seconds)
        return f"Slow response"
    return mock_llm


# ── LLMRole enum tests ──

class TestLLMRole:
    def test_role_values(self):
        assert LLMRole.EXTRACT.value == "extract"
        assert LLMRole.KEYWORD.value == "keyword"
        assert LLMRole.QUERY.value == "query"

    def test_role_names(self):
        assert ROLE_NAMES == frozenset(["extract", "keyword", "query"])

    def test_role_priority(self):
        assert ROLE_PRIORITY[LLMRole.QUERY] < ROLE_PRIORITY[LLMRole.KEYWORD]
        assert ROLE_PRIORITY[LLMRole.KEYWORD] < ROLE_PRIORITY[LLMRole.EXTRACT]


# ── RoleLLMConfig tests ──

class TestRoleLLMConfig:
    def test_default_config(self):
        cfg = RoleLLMConfig()
        assert cfg.func is None
        assert cfg.kwargs is None
        assert cfg.max_async is None
        assert cfg.timeout is None

    def test_custom_config(self):
        func = _make_sync_llm()
        cfg = RoleLLMConfig(func=func, max_async=5, timeout=60)
        assert cfg.func == func
        assert cfg.max_async == 5
        assert cfg.timeout == 60


# ── LLMRoleManager initialization tests ──

class TestLLMRoleManagerInit:
    def test_init_with_base_llm(self):
        llm = _make_sync_llm()
        manager = LLMRoleManager(base_llm=llm)
        assert manager.get_role_func(LLMRole.EXTRACT) == llm
        assert manager.get_role_func(LLMRole.KEYWORD) == llm
        assert manager.get_role_func(LLMRole.QUERY) == llm

    def test_init_with_role_overrides(self):
        base = _make_sync_llm()
        extract = _make_sync_llm()
        manager = LLMRoleManager(
            base_llm=base,
            role_configs={
                LLMRole.EXTRACT: RoleLLMConfig(func=extract, max_async=8),
            },
        )
        assert manager.get_role_func(LLMRole.EXTRACT) == extract
        assert manager.get_role_func(LLMRole.QUERY) == base

    def test_init_no_base_llm(self):
        manager = LLMRoleManager()
        assert manager.get_role_func(LLMRole.EXTRACT) is None

    def test_init_default_values(self):
        manager = LLMRoleManager(default_max_async=32, default_timeout=120)
        stats = manager.get_role_config()
        assert stats["extract"]["max_async"] == 32
        assert stats["extract"]["timeout"] == 120


# ── Sync call tests ──

class TestLLMRoleManagerSyncCall:
    def test_call_role_sync(self):
        llm = _make_sync_llm()
        manager = LLMRoleManager(base_llm=llm)
        result = manager.call_role_sync(LLMRole.QUERY, "What is Python?")
        assert "What is Python?" in result

    def test_call_role_sync_with_kwargs(self):
        llm = MagicMock(return_value="OK")
        manager = LLMRoleManager(base_llm=llm)
        result = manager.call_role_sync(LLMRole.EXTRACT, "text", temperature=0.5)
        llm.assert_called_once_with("text", temperature=0.5)

    def test_call_role_sync_no_llm_raises(self):
        manager = LLMRoleManager()
        with pytest.raises(ValueError, match="No LLM function"):
            manager.call_role_sync(LLMRole.QUERY, "prompt")

    def test_call_role_sync_role_specific_func(self):
        base = MagicMock(return_value="base_response")
        extract = MagicMock(return_value="extract_response")
        manager = LLMRoleManager(
            base_llm=base,
            role_configs={LLMRole.EXTRACT: RoleLLMConfig(func=extract)},
        )
        result = manager.call_role_sync(LLMRole.EXTRACT, "text")
        assert result == "extract_response"
        extract.assert_called_once()

    def test_call_role_sync_increments_count(self):
        llm = _make_sync_llm()
        manager = LLMRoleManager(base_llm=llm)
        manager.call_role_sync(LLMRole.QUERY, "q1")
        manager.call_role_sync(LLMRole.QUERY, "q2")
        stats = manager.get_role_config(LLMRole.QUERY)
        assert stats["call_count"] == 2


# ── Async call tests ──

class TestLLMRoleManagerAsyncCall:
    @pytest.mark.asyncio
    async def test_call_role_async(self):
        llm = _make_async_llm()
        manager = LLMRoleManager(base_llm=llm)
        result = await manager.call_role(LLMRole.QUERY, "async query")
        assert "async query" in result

    @pytest.mark.asyncio
    async def test_call_role_async_with_kwargs(self):
        llm = AsyncMock(return_value="async OK")
        manager = LLMRoleManager(base_llm=llm)
        result = await manager.call_role(LLMRole.KEYWORD, "text", temperature=0.7)
        llm.assert_called_once_with("text", temperature=0.7)

    @pytest.mark.asyncio
    async def test_call_role_async_timeout(self):
        llm = _make_slow_async_llm(timeout_seconds=10)
        manager = LLMRoleManager(
            base_llm=llm,
            role_configs={LLMRole.QUERY: RoleLLMConfig(timeout=1)},
        )
        with pytest.raises(asyncio.TimeoutError):
            await manager.call_role(LLMRole.QUERY, "will timeout")

    @pytest.mark.asyncio
    async def test_call_role_async_semaphore(self):
        """Test concurrency control via semaphore."""
        call_times = []

        async def tracked_llm(prompt, **kwargs):
            call_times.append(time.time())
            await asyncio.sleep(0.1)
            return f"Response {len(call_times)}"

        manager = LLMRoleManager(
            base_llm=tracked_llm,
            role_configs={LLMRole.EXTRACT: RoleLLMConfig(max_async=2)},
        )

        # Launch 4 concurrent calls — only 2 should run at a time
        tasks = [
            manager.call_role(LLMRole.EXTRACT, f"prompt_{i}")
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_call_role_async_no_llm_raises(self):
        manager = LLMRoleManager()
        with pytest.raises(ValueError, match="No LLM function"):
            await manager.call_role(LLMRole.QUERY, "prompt")

    @pytest.mark.asyncio
    async def test_call_role_async_sync_function(self):
        """Sync LLM function should be wrapped in to_thread."""
        llm = _make_sync_llm()
        manager = LLMRoleManager(base_llm=llm)
        result = await manager.call_role(LLMRole.QUERY, "sync via async")
        assert "sync via async" in result


# ── Config update tests ──

class TestLLMRoleManagerUpdateConfig:
    def test_update_max_async(self):
        manager = LLMRoleManager()
        manager.update_role_config(LLMRole.QUERY, max_async=32)
        stats = manager.get_role_config(LLMRole.QUERY)
        assert stats["max_async"] == 32

    def test_update_timeout(self):
        manager = LLMRoleManager()
        manager.update_role_config(LLMRole.EXTRACT, timeout=60)
        stats = manager.get_role_config(LLMRole.EXTRACT)
        assert stats["timeout"] == 60

    def test_update_func(self):
        new_func = _make_sync_llm()
        manager = LLMRoleManager()
        manager.update_role_config(LLMRole.KEYWORD, func=new_func)
        assert manager.get_role_func(LLMRole.KEYWORD) == new_func

    def test_update_kwargs(self):
        manager = LLMRoleManager()
        manager.update_role_config(LLMRole.QUERY, kwargs={"model": "gpt-4"})
        stats = manager.get_role_config(LLMRole.QUERY)
        assert stats["kwargs"]["model"] == "gpt-4"

    def test_update_invalidates_semaphore(self):
        manager = LLMRoleManager(base_llm=_make_sync_llm())
        # Pre-create semaphore by getting the state
        state = manager._states[LLMRole.EXTRACT]
        state.semaphore = asyncio.Semaphore(16)
        # Update max_async — should invalidate
        manager.update_role_config(LLMRole.EXTRACT, max_async=8)
        assert state.semaphore is None


# ── Observability tests ──

class TestLLMRoleManagerObservability:
    def test_get_all_role_configs(self):
        manager = LLMRoleManager(base_llm=_make_sync_llm())
        configs = manager.get_role_config()
        assert "extract" in configs
        assert "keyword" in configs
        assert "query" in configs
        assert configs["extract"]["has_func"] is True

    def test_stats(self):
        manager = LLMRoleManager(base_llm=_make_sync_llm())
        manager.call_role_sync(LLMRole.QUERY, "q1")
        stats = manager.stats()
        assert stats["total_calls"] == 1
        assert "roles" in stats

    def test_reset_stats(self):
        manager = LLMRoleManager(base_llm=_make_sync_llm())
        manager.call_role_sync(LLMRole.QUERY, "q1")
        manager.reset_stats()
        stats = manager.stats()
        assert stats["total_calls"] == 0

    def test_scrub_secret_kwargs(self):
        manager = LLMRoleManager(
            base_llm=_make_sync_llm(),
            role_configs={
                LLMRole.QUERY: RoleLLMConfig(
                    kwargs={"api_key": "sk-secret123", "model": "gpt-4"},
                ),
            },
        )
        config = manager.get_role_config(LLMRole.QUERY)
        assert config["kwargs"]["api_key"] == "***"
        assert config["kwargs"]["model"] == "gpt-4"

    def test_error_count_tracking(self):
        manager = LLMRoleManager(base_llm=_make_sync_llm())

        @pytest.mark.asyncio
        async def test_async_error():
            async def failing_llm(prompt, **kwargs):
                raise RuntimeError("LLM error")

            manager.update_role_config(LLMRole.QUERY, func=failing_llm)
            try:
                await manager.call_role(LLMRole.QUERY, "will fail")
            except RuntimeError:
                pass

            stats = manager.get_role_config(LLMRole.QUERY)
            assert stats["error_count"] == 1

        asyncio.run(test_async_error())
