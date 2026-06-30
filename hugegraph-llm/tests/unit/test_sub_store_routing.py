"""Unit tests for Sub-store Routing."""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.sub_store_routing import (
    RouteStore,
    compute_routing_key,
    ShardMetadata,
)


class TestComputeRoutingKey:

    def test_private_user_key(self):
        key = compute_routing_key(user_id="alice", scope="private")
        assert key == "user:alice"

    def test_private_user_with_app(self):
        key = compute_routing_key(user_id="alice", app_name="chatbot", scope="private")
        assert key == "user:alice:app:chatbot"

    def test_agent_group_key(self):
        key = compute_routing_key(user_id="alice", agent_id="bot1", scope="agent_group")
        assert "agent:bot1" in key

    def test_public_key(self):
        key = compute_routing_key(scope="public")
        assert key == "global:public"

    def test_public_with_app(self):
        key = compute_routing_key(scope="public", app_name="review")
        assert key == "global:public:app:review"

    def test_user_group_key(self):
        key = compute_routing_key(user_id="alice", scope="user_group")
        assert "global:user_group" in key

    def test_default_user(self):
        key = compute_routing_key(scope="private")
        assert "user:default" in key


class TestShardMetadata:

    def test_to_dict(self):
        meta = ShardMetadata(
            routing_key="user:alice",
            user_id="alice",
            scope="private",
            memory_count=5,
        )
        d = meta.to_dict()
        assert d["routing_key"] == "user:alice"
        assert d["user_id"] == "alice"
        assert d["memory_count"] == 5


class TestRouteStore:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RouteStore(base_dir=self.tmpdir)

    def test_route_from_metadata(self):
        key = self.store.route({"user_id": "alice", "scope": "private"})
        assert key == "user:alice"

    def test_route_default_scope(self):
        key = self.store.route({"user_id": "bob"})
        assert "bob" in key

    def test_add_memory(self):
        rk = self.store.add_memory("m1", "hello world", {"user_id": "alice"})
        assert rk == "user:alice"
        shard = self.store.get_shard(rk)
        assert len(shard["faiss_metadata"]) == 1

    def test_add_multiple_memories(self):
        self.store.add_memory("m1", "hello", {"user_id": "alice"})
        self.store.add_memory("m2", "world", {"user_id": "alice"})
        shard = self.store.get_shard("user:alice")
        assert len(shard["faiss_metadata"]) == 2

    def test_add_to_different_shards(self):
        self.store.add_memory("m1", "alice memory", {"user_id": "alice"})
        self.store.add_memory("m2", "bob memory", {"user_id": "bob"})
        assert len(self.store.list_shards()) == 2

    def test_search_shard(self):
        self.store.add_memory("m1", "张三在货拉拉工作", {"user_id": "alice"})
        self.store.add_memory("m2", "李四在北京", {"user_id": "alice"})
        results = self.store.search_shard("user:alice", "货拉拉")
        assert len(results) >= 1
        assert any("货拉拉" in r["content"] for r in results)

    def test_search_multi_shard(self):
        self.store.add_memory("m1", "alice info", {"user_id": "alice"})
        self.store.add_memory("m2", "bob info", {"user_id": "bob"})
        results = self.store.search_multi_shard(["user:alice", "user:bob"], "info")
        assert len(results) >= 2

    def test_search_accessible_private(self):
        self.store.add_memory("m1", "alice secret", {"user_id": "alice", "scope": "private"})
        self.store.add_memory("m2", "bob secret", {"user_id": "bob", "scope": "private"})
        # Alice can only see her own private shard
        results = self.store.search_accessible(user_id="alice", query="secret")
        assert any(r["content"] == "alice secret" for r in results)
        # Should NOT see bob's private shard
        assert not any(r["content"] == "bob secret" for r in results)

    def test_search_accessible_public(self):
        self.store.add_memory("m1", "public info", {"scope": "public"})
        results = self.store.search_accessible(user_id="anyone", query="public")
        assert len(results) >= 1

    def test_delete_memory_with_routing_key(self):
        self.store.add_memory("m1", "to delete", {"user_id": "alice"})
        result = self.store.delete_memory("m1", routing_key="user:alice")
        assert result is True
        shard = self.store.get_shard("user:alice")
        assert len(shard["faiss_metadata"]) == 0

    def test_delete_memory_without_routing_key(self):
        self.store.add_memory("m1", "to delete", {"user_id": "alice"})
        result = self.store.delete_memory("m1")
        assert result is True

    def test_delete_nonexistent(self):
        result = self.store.delete_memory("nonexistent")
        assert result is False

    def test_list_shards(self):
        self.store.add_memory("m1", "hello", {"user_id": "alice"})
        self.store.add_memory("m2", "world", {"user_id": "bob"})
        shards = self.store.list_shards()
        assert len(shards) == 2

    def test_get_shard_stats(self):
        self.store.add_memory("m1", "hello", {"user_id": "alice"})
        self.store.add_memory("m2", "world", {"user_id": "bob"})
        stats = self.store.get_shard_stats()
        assert stats["total_shards"] == 2
        assert stats["total_memories"] == 2

    def test_clear_shard(self):
        self.store.add_memory("m1", "hello", {"user_id": "alice"})
        result = self.store.clear_shard("user:alice")
        assert result is True
        shard = self.store.get_shard("user:alice")
        assert len(shard["faiss_metadata"]) == 0

    def test_clear_nonexistent_shard(self):
        result = self.store.clear_shard("nonexistent")
        assert result is False

    def test_clear_all(self):
        self.store.add_memory("m1", "hello", {"user_id": "alice"})
        self.store.add_memory("m2", "world", {"user_id": "bob"})
        result = self.store.clear_all()
        assert result["cleared"] == 2
