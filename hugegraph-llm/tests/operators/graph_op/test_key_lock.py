"""Tests for key_lock module — KeyLockManager, EntityKeyLockManager, RelationKeyLockManager."""

import asyncio
import pytest

from hugegraph_llm.operators.graph_op.key_lock import (
    EntityKeyLockManager,
    KeyLockManager,
    RelationKeyLockManager,
)


# ---------------------------------------------------------------------------
# KeyLockManager core tests
# ---------------------------------------------------------------------------


class TestKeyLockManagerBasic:
    """acquire / release / stats / lazy creation / cleanup."""

    @pytest.mark.asyncio
    async def test_acquire_release_basic(self):
        mgr = KeyLockManager()
        await mgr.acquire("alpha")
        assert mgr.stats()["active_locks"] == 1
        await mgr.release("alpha")
        assert mgr.stats()["active_locks"] == 0

    @pytest.mark.asyncio
    async def test_total_acquisitions_counter(self):
        mgr = KeyLockManager()
        await mgr.acquire("k1")
        await mgr.release("k1")
        assert mgr.stats()["total_acquisitions"] == 1
        await mgr.acquire("k2")
        await mgr.release("k2")
        assert mgr.stats()["total_acquisitions"] == 2

    @pytest.mark.asyncio
    async def test_lazy_lock_creation(self):
        mgr = KeyLockManager()
        assert len(mgr._locks) == 0
        await mgr.acquire("new_key")
        assert "new_key" in mgr._locks
        await mgr.release("new_key")
        # Lock should be cleaned up (no waiters, not held)
        assert "new_key" not in mgr._locks

    @pytest.mark.asyncio
    async def test_cleanup_only_when_no_waiters(self):
        mgr = KeyLockManager()
        await mgr.acquire("held_key")
        # Simulate a waiter
        waiter_task = asyncio.create_task(mgr.acquire("held_key"))
        await asyncio.sleep(0.01)  # let waiter start waiting
        assert mgr.stats()["pending_acquisitions"] >= 1

        await mgr.release("held_key")  # first holder releases, waiter gets it
        await asyncio.sleep(0.01)
        # Lock should still exist because waiter holds it
        assert "held_key" in mgr._locks

        await mgr.release("held_key")  # waiter releases
        await asyncio.sleep(0.01)
        assert "held_key" not in mgr._locks
        waiter_task.cancel()

    @pytest.mark.asyncio
    async def test_release_nonexistent_key_no_error(self):
        mgr = KeyLockManager()
        # Releasing a key that was never acquired should not raise
        await mgr.release("ghost")

    @pytest.mark.asyncio
    async def test_multiple_keys_independent(self):
        mgr = KeyLockManager()
        await mgr.acquire("a")
        await mgr.acquire("b")
        assert mgr.stats()["active_locks"] == 2
        await mgr.release("a")
        assert mgr.stats()["active_locks"] == 1
        await mgr.release("b")
        assert mgr.stats()["active_locks"] == 0


class TestKeyLockManagerContextManager:
    """async with_key_lock context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_acquires_and_releases(self):
        mgr = KeyLockManager()
        async with mgr.with_key_lock("ctx_key"):
            assert mgr.stats()["active_locks"] == 1
        assert mgr.stats()["active_locks"] == 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exit(self):
        mgr = KeyLockManager()
        async with mgr.with_key_lock("tmp"):
            pass
        assert "tmp" not in mgr._locks

    @pytest.mark.asyncio
    async def test_context_manager_exception_still_releases(self):
        mgr = KeyLockManager()
        try:
            async with mgr.with_key_lock("err_key"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert mgr.stats()["active_locks"] == 0


class TestKeyLockManagerConcurrent:
    """Concurrent async tasks acquiring the same key."""

    @pytest.mark.asyncio
    async def test_serialized_access_same_key(self):
        mgr = KeyLockManager()
        results = []

        async def worker(ident: int):
            async with mgr.with_key_lock("shared"):
                results.append(ident)
                await asyncio.sleep(0.05)

        tasks = [asyncio.create_task(worker(i)) for i in range(5)]
        await asyncio.gather(*tasks)
        # All workers should have completed (order may vary but all present)
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_pending_acquisitions_count(self):
        mgr = KeyLockManager()
        await mgr.acquire("busy")
        # Start 3 waiters
        waiters = [asyncio.create_task(mgr.acquire("busy")) for _ in range(3)]
        await asyncio.sleep(0.05)
        assert mgr.stats()["pending_acquisitions"] == 3
        # Release lock so first waiter can acquire
        await mgr.release("busy")
        await asyncio.sleep(0.05)
        # Now one waiter holds the lock, 2 are still pending
        # Release to let remaining waiters proceed one by one
        for _ in range(2):
            await mgr.release("busy")
            await asyncio.sleep(0.02)
        # Final release for the last waiter
        await mgr.release("busy")
        # All done — cancel any lingering tasks
        for w in waiters:
            if not w.done():
                w.cancel()


class TestKeyLockManagerStats:
    """Stats reporting."""

    @pytest.mark.asyncio
    async def test_stats_empty_manager(self):
        mgr = KeyLockManager()
        stats = mgr.stats()
        assert stats["active_locks"] == 0
        assert stats["total_acquisitions"] == 0
        assert stats["pending_acquisitions"] == 0


# ---------------------------------------------------------------------------
# EntityKeyLockManager tests
# ---------------------------------------------------------------------------


class TestEntityKeyLockManager:
    """Key normalization: lowercase, strip, truncate to 256."""

    def test_normalize_lowercase(self):
        mgr = EntityKeyLockManager()
        assert mgr.normalize_entity_key("HelloWorld") == "helloworld"

    def test_normalize_strip(self):
        mgr = EntityKeyLockManager()
        assert mgr.normalize_entity_key("  padded  ") == "padded"

    def test_normalize_truncate_256(self):
        mgr = EntityKeyLockManager()
        long_key = "a" * 300
        assert len(mgr.normalize_entity_key(long_key)) == 256

    def test_normalize_combined(self):
        mgr = EntityKeyLockManager()
        long_key = "  " + "B" * 300 + "  "
        result = mgr.normalize_entity_key(long_key)
        assert result == "b" * 256
        assert len(result) == 256

    @pytest.mark.asyncio
    async def test_entity_keys_normalized_in_locks(self):
        mgr = EntityKeyLockManager()
        await mgr.acquire("  MyEntity  ")
        # Internal dict uses normalized key
        assert "myentity" in mgr._locks
        await mgr.release("  MyEntity  ")
        assert "myentity" not in mgr._locks


# ---------------------------------------------------------------------------
# RelationKeyLockManager tests
# ---------------------------------------------------------------------------


class TestRelationKeyLockManager:
    """Key normalization: sorted pair, joined by |."""

    def test_normalize_sorted_pair(self):
        mgr = RelationKeyLockManager()
        key = mgr.normalize_relation_key("B", "A")
        assert key == "a|b"

    def test_normalize_same_direction(self):
        mgr = RelationKeyLockManager()
        key1 = mgr.normalize_relation_key("A", "B")
        key2 = mgr.normalize_relation_key("B", "A")
        assert key1 == key2

    def test_normalize_with_whitespace_and_case(self):
        mgr = RelationKeyLockManager()
        key = mgr.normalize_relation_key("  Alpha  ", "  Beta  ")
        assert key == "alpha|beta"

    def test_normalize_truncate_each_endpoint(self):
        mgr = RelationKeyLockManager()
        long_src = "X" * 300
        long_tgt = "Y" * 300
        key = mgr.normalize_relation_key(long_src, long_tgt)
        parts = key.split("|")
        assert len(parts[0]) == 256
        assert len(parts[1]) == 256

    @pytest.mark.asyncio
    async def test_relation_key_used_in_acquire(self):
        mgr = RelationKeyLockManager()
        norm_key = mgr.normalize_relation_key("NodeB", "NodeA")
        await mgr.acquire(norm_key)
        assert norm_key in mgr._locks
        await mgr.release(norm_key)
        assert norm_key not in mgr._locks
