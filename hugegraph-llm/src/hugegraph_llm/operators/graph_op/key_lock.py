"""Per-key asyncio locks for concurrent entity/relation merging.

Borrowed from LightRAG's ``get_storage_keyed_lock`` pattern — each key (entity
name or sorted relation endpoint pair) gets its own ``asyncio.Lock``, created
lazily on first acquire and cleaned up when no longer held.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional


class KeyLockManager:
    """Manage per-key asyncio locks with lazy creation and automatic cleanup.

    When a lock for a given key is first requested, it is created.  When the
    lock is released and no other waiter is pending, the lock entry is removed
    from the internal dict so that memory does not grow unboundedly.
    """

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._total_acquisitions: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def acquire(self, key: str) -> None:
        """Acquire the lock for *key*.  Creates the lock lazily if needed."""
        normalized = self._normalize_key(key)
        if normalized not in self._locks:
            self._locks[normalized] = asyncio.Lock()
        self._total_acquisitions += 1
        await self._locks[normalized].acquire()

    async def release(self, key: str) -> None:
        """Release the lock for *key* and clean up if no waiters remain."""
        normalized = self._normalize_key(key)
        lock = self._locks.get(normalized)
        if lock is None:
            return
        lock.release()
        # Cleanup: remove the lock entry if nobody is waiting on it
        waiters = getattr(lock, "_waiters", None)
        if not lock.locked() and not waiters:
            self._locks.pop(normalized, None)

    @asynccontextmanager
    async def with_key_lock(self, key: str):
        """Async context manager that acquires and releases the lock for *key*."""
        await self.acquire(key)
        try:
            yield
        finally:
            await self.release(key)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return dict with active_locks, total_acquisitions, pending_acquisitions."""
        active = sum(1 for lock in self._locks.values() if lock.locked())
        pending = sum(
            len(getattr(lock, "_waiters", None) or []) for lock in self._locks.values()
        )
        return {
            "active_locks": active,
            "total_acquisitions": self._total_acquisitions,
            "pending_acquisitions": pending,
        }

    # ------------------------------------------------------------------
    # Key normalization (override in subclasses)
    # ------------------------------------------------------------------

    def _normalize_key(self, key: str) -> str:
        """Default normalization — just strip whitespace."""
        return key.strip()


class EntityKeyLockManager(KeyLockManager):
    """Specialized lock manager for entities.

    Keys are entity names, normalized by lowercasing, stripping whitespace,
    and truncating to 256 characters (mirrors LightRAG's
    ``_truncate_entity_identifier``).
    """

    def normalize_entity_key(self, name: str) -> str:
        """Public normalization: lowercase + strip + truncate to 256 chars."""
        return name.lower().strip()[:256]

    def _normalize_key(self, key: str) -> str:
        return self.normalize_entity_key(key)


class RelationKeyLockManager(KeyLockManager):
    """Specialized lock manager for relations.

    Keys are *sorted* endpoint pairs joined by ``|``, each endpoint normalized
    like an entity key.  Sorting ensures that ``A→B`` and ``B→A`` map to the
    same lock.
    """

    def normalize_relation_key(self, src: str, tgt: str) -> str:
        """Normalize a relation key: sort endpoints, normalize each, join with |."""
        norm_src = src.lower().strip()[:256]
        norm_tgt = tgt.lower().strip()[:256]
        return "|".join(sorted([norm_src, norm_tgt]))

    def _normalize_key(self, key: str) -> str:
        """For relations the key is already a pre-normalized sorted-pair string."""
        return key.strip()
