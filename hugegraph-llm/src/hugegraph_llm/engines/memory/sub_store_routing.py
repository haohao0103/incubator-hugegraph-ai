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
Sub-store routing (PowerMem-style SubStore alignment).

PowerMem routes memories to different "sub stores" based on metadata
(user_id, agent_id, app_name), each with its own embedding dimension
and storage backend. We implement a lighter version:

  - RouteStore manages multiple FAISS/BM25 index shards by routing key.
  - Routing key is derived from metadata (user_id, agent_id, app_name).
  - Each shard has its own FAISS index and BM25 corpus.
  - Search can target a specific shard or union across shards.
  - Supports isolation (private per-user) and sharing (cross-shard search).

This is the engine-level module; memory_backend.py will integrate it.
"""

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.utils.log import log


def compute_routing_key(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_name: Optional[str] = None,
    scope: Optional[str] = None,
) -> str:
    """Compute a routing key from metadata fields.

    The routing key determines which shard a memory belongs to.
    If scope is 'public' or 'user_group', the key is global (all users share).
    If scope is 'private' or 'agent_group', the key is per-user or per-agent.

    Args:
        user_id: User identifier.
        agent_id: Agent identifier.
        app_name: Application name (e.g. 'chatbot', 'code-review').
        scope: Access scope ('private', 'agent_group', 'user_group', 'public').

    Returns:
        A string routing key like "user:alice:app:chatbot" or "global:public".
    """
    scope = scope or "private"

    if scope in ("public", "user_group"):
        # Shared across all users
        parts = ["global", scope]
        if app_name:
            parts.append(f"app:{app_name}")
        return ":".join(parts)

    # Per-user or per-agent
    parts = []
    if user_id:
        parts.append(f"user:{user_id}")
    else:
        parts.append("user:default")

    if agent_id:
        parts.append(f"agent:{agent_id}")

    if app_name:
        parts.append(f"app:{app_name}")

    return ":".join(parts)


class ShardMetadata:
    """Metadata for a single shard."""

    def __init__(
        self,
        routing_key: str,
        user_id: str = "",
        agent_id: str = "",
        app_name: str = "",
        scope: str = "private",
        embedding_dim: int = 384,
        created_at: float = 0.0,
        memory_count: int = 0,
    ):
        self.routing_key = routing_key
        self.user_id = user_id
        self.agent_id = agent_id
        self.app_name = app_name
        self.scope = scope
        self.embedding_dim = embedding_dim
        self.created_at = created_at
        self.memory_count = memory_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_key": self.routing_key,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "app_name": self.app_name,
            "scope": self.scope,
            "embedding_dim": self.embedding_dim,
            "created_at": self.created_at,
            "memory_count": self.memory_count,
        }


class RouteStore:
    """Route memories to different index shards based on metadata.

    This is the core sub-store routing module. Each shard maintains its own
    FAISS index (via FaissMemoryIndex) and BM25 corpus. The routing key
    determines which shard a memory is stored in and searched from.

    Args:
        base_dir: Base directory for shard index files.
        default_scope: Default scope when not specified in metadata.
    """

    def __init__(
        self,
        base_dir: str = "/tmp/hg_memory_shards",
        default_scope: str = "private",
    ):
        self.base_dir = base_dir
        self.default_scope = default_scope
        self._shards: Dict[str, Dict[str, Any]] = {}  # routing_key -> shard dict
        self._lock = threading.RLock()
        os.makedirs(base_dir, exist_ok=True)

    def route(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute routing key from metadata dict.

        Args:
            metadata: Dict with keys 'user_id', 'agent_id', 'app_name', 'scope'.

        Returns:
            Routing key string.
        """
        meta = metadata or {}
        return compute_routing_key(
            user_id=meta.get("user_id"),
            agent_id=meta.get("agent_id"),
            app_name=meta.get("app_name"),
            scope=meta.get("scope", self.default_scope),
        )

    def get_shard(self, routing_key: str) -> Dict[str, Any]:
        """Get or create a shard for the given routing key.

        Each shard is a dict containing:
          - routing_key: the shard's routing key
          - metadata: ShardMetadata
          - faiss_metadata: list of memory metadata entries (for FaissMemoryIndex)
          - bm25_corpus: list of (doc_id, text) tuples
          - index_path: path to shard-specific FAISS index file
        """
        with self._lock:
            if routing_key not in self._shards:
                shard_dir = os.path.join(self.base_dir, routing_key.replace(":", "_"))
                os.makedirs(shard_dir, exist_ok=True)
                self._shards[routing_key] = {
                    "routing_key": routing_key,
                    "metadata": ShardMetadata(routing_key=routing_key),
                    "faiss_metadata": [],
                    "bm25_corpus": [],
                    "index_path": os.path.join(shard_dir, "faiss_index.bin"),
                    "bm25_path": os.path.join(shard_dir, "bm25_index"),
                }
            return self._shards[routing_key]

    def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a memory to the appropriate shard.

        Args:
            memory_id: Unique memory identifier.
            content: Memory content text.
            metadata: Routing metadata (user_id, agent_id, app_name, scope).

        Returns:
            The routing key used.
        """
        routing_key = self.route(metadata)
        shard = self.get_shard(routing_key)

        shard["faiss_metadata"].append({
            "memory_id": memory_id,
            "content": content,
            "routing_key": routing_key,
            "metadata": metadata or {},
        })
        shard["bm25_corpus"].append((memory_id, content))
        shard["metadata"].memory_count += 1

        log.debug("Added memory %s to shard %s", memory_id, routing_key)
        return routing_key

    def search_shard(
        self,
        routing_key: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search within a specific shard.

        This is a metadata-only search (no FAISS/BM25 call).
        The actual FAISS/BM25 search is delegated to memory_backend.py
        which manages the real indices.

        Args:
            routing_key: Shard to search in.
            query: Search query.
            limit: Max results.

        Returns:
            List of matching memory metadata dicts from this shard.
        """
        shard = self.get_shard(routing_key)
        # Simple keyword match on content (real search uses FAISS/BM25)
        results = []
        query_lower = query.lower()
        for meta in shard["faiss_metadata"]:
            if query_lower in meta["content"].lower():
                results.append(meta)
                if len(results) >= limit:
                    break
        return results

    def search_multi_shard(
        self,
        routing_keys: List[str],
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search across multiple shards and merge results.

        Args:
            routing_keys: List of shards to search.
            query: Search query.
            limit: Max total results.

        Returns:
            Merged results from all shards, deduplicated by memory_id.
        """
        all_results = []
        seen_ids: Set[str] = set()
        for key in routing_keys:
            shard_results = self.search_shard(key, query, limit=limit)
            for r in shard_results:
                mid = r.get("memory_id", "")
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    all_results.append(r)
                    if len(all_results) >= limit:
                        return all_results
        return all_results

    def search_accessible(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        app_name: Optional[str] = None,
        query: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search all shards accessible to the given user/agent.

        Accessibility rules (PowerMem-style):
          - private shards: only accessible by the owning user_id
          - agent_group shards: accessible by same agent_id
          - user_group shards: accessible by same user_id (any agent)
          - public shards: accessible by everyone

        Args:
            user_id: User performing the search.
            agent_id: Agent performing the search.
            app_name: App context for the search.
            query: Search query.
            limit: Max results.

        Returns:
            Results from all accessible shards.
        """
        accessible_keys = self._get_accessible_keys(user_id, agent_id, app_name)
        if not accessible_keys:
            return []
        return self.search_multi_shard(accessible_keys, query, limit)

    def _get_accessible_keys(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> List[str]:
        """Determine which routing keys are accessible."""
        accessible = []

        # 1. The user's own private shard
        private_key = compute_routing_key(
            user_id=user_id, agent_id=agent_id, app_name=app_name, scope="private"
        )
        if private_key in self._shards:
            accessible.append(private_key)

        # 2. Agent group shards (same agent_id)
        if agent_id:
            agent_key = compute_routing_key(
                user_id=None, agent_id=agent_id, app_name=app_name, scope="agent_group"
            )
            if agent_key in self._shards:
                accessible.append(agent_key)

        # 3. User group shards (same user_id)
        user_group_key = compute_routing_key(
            user_id=user_id, agent_id=None, app_name=app_name, scope="user_group"
        )
        if user_group_key in self._shards:
            accessible.append(user_group_key)

        # 4. Public shards
        public_key = compute_routing_key(
            user_id=None, agent_id=None, app_name=app_name, scope="public"
        )
        if public_key in self._shards:
            accessible.append(public_key)

        # Also check all shards for any that match accessibility rules
        for key, shard in self._shards.items():
            if key in accessible:
                continue
            shard_meta = shard["metadata"]
            if shard_meta.scope == "public":
                accessible.append(key)
            elif shard_meta.scope == "user_group" and shard_meta.user_id == user_id:
                accessible.append(key)
            elif shard_meta.scope == "agent_group" and shard_meta.agent_id == agent_id:
                accessible.append(key)

        return accessible

    def delete_memory(
        self,
        memory_id: str,
        routing_key: Optional[str] = None,
    ) -> bool:
        """Remove a memory from its shard.

        Args:
            memory_id: Memory to remove.
            routing_key: If known, direct removal. Otherwise search all shards.

        Returns:
            True if found and removed, False otherwise.
        """
        if routing_key:
            shard = self.get_shard(routing_key)
            shard["faiss_metadata"] = [
                m for m in shard["faiss_metadata"] if m.get("memory_id") != memory_id
            ]
            shard["bm25_corpus"] = [
                (did, txt) for did, txt in shard["bm25_corpus"] if did != memory_id
            ]
            shard["metadata"].memory_count -= 1
            return True

        # Search all shards
        for key, shard in self._shards.items():
            before = len(shard["faiss_metadata"])
            shard["faiss_metadata"] = [
                m for m in shard["faiss_metadata"] if m.get("memory_id") != memory_id
            ]
            shard["bm25_corpus"] = [
                (did, txt) for did, txt in shard["bm25_corpus"] if did != memory_id
            ]
            after = len(shard["faiss_metadata"])
            if before > after:
                shard["metadata"].memory_count -= 1
                return True
        return False

    def list_shards(self) -> List[Dict[str, Any]]:
        """Return metadata for all active shards."""
        return [shard["metadata"].to_dict() for shard in self._shards.values()]

    def get_shard_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all shards."""
        total_memories = sum(s["metadata"].memory_count for s in self._shards.values())
        return {
            "total_shards": len(self._shards),
            "total_memories": total_memories,
            "shards": self.list_shards(),
        }

    def clear_shard(self, routing_key: str) -> bool:
        """Clear all memories from a specific shard."""
        with self._lock:
            if routing_key not in self._shards:
                return False
            shard = self._shards[routing_key]
            shard["faiss_metadata"] = []
            shard["bm25_corpus"] = []
            shard["metadata"].memory_count = 0
            return True

    def clear_all(self) -> Dict[str, int]:
        """Clear all shards. Returns count of cleared memories."""
        total = 0
        for key in list(self._shards.keys()):
            shard = self._shards[key]
            total += shard["metadata"].memory_count
            shard["faiss_metadata"] = []
            shard["bm25_corpus"] = []
            shard["metadata"].memory_count = 0
        return {"cleared": total}
