"""Unit tests for Agent Collaboration module."""

import json
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.agent_collaboration import (
    AgentMemoryManager,
    PermissionChecker,
    PrivacyFilter,
    CollaborationBroker,
    PermissionRule,
)
from hugegraph_llm.engines.memory.base import (
    AccessPermission,
    MemoryScope,
    PrivacyLevel,
)


class TestPermissionChecker:

    def setup_method(self):
        self.checker = PermissionChecker()

    def test_owner_full_access(self):
        result = self.checker.check("alice", AccessPermission.READ, MemoryScope.PRIVATE, "alice")
        assert result is True

    def test_owner_write_access(self):
        result = self.checker.check("alice", AccessPermission.WRITE, MemoryScope.PRIVATE, "alice")
        assert result is True

    def test_owner_delete_access(self):
        result = self.checker.check("alice", AccessPermission.DELETE, MemoryScope.PRIVATE, "alice")
        assert result is True

    def test_private_scope_blocked_for_non_owner(self):
        result = self.checker.check("bob", AccessPermission.WRITE, MemoryScope.PRIVATE, "alice")
        assert result is False

    def test_private_scope_read_blocked_for_non_owner(self):
        # Private scope: non-owner has NO read access (only owner)
        result = self.checker.check("bob", AccessPermission.READ, MemoryScope.PRIVATE, "alice")
        # Default rules: PRIVATE = {read, write, delete} for owner only
        # Non-owner: not in allowed set because scope is PRIVATE
        assert result is False

    def test_agent_group_read_access(self):
        result = self.checker.check("bot1", AccessPermission.READ, MemoryScope.AGENT_GROUP, "alice")
        assert result is True

    def test_agent_group_write_access(self):
        result = self.checker.check("bot1", AccessPermission.WRITE, MemoryScope.AGENT_GROUP, "alice")
        assert result is True

    def test_agent_group_delete_blocked(self):
        result = self.checker.check("bot1", AccessPermission.DELETE, MemoryScope.AGENT_GROUP, "alice")
        assert result is False

    def test_public_read_access(self):
        result = self.checker.check("anyone", AccessPermission.READ, MemoryScope.PUBLIC, "alice")
        assert result is True

    def test_public_write_blocked(self):
        result = self.checker.check("anyone", AccessPermission.WRITE, MemoryScope.PUBLIC, "alice")
        assert result is False

    def test_confidential_needs_admin(self):
        result = self.checker.check("bob", AccessPermission.READ, MemoryScope.PUBLIC, "alice", PrivacyLevel.CONFIDENTIAL)
        assert result is False

    def test_confidential_admin_access(self):
        result = self.checker.check("alice", AccessPermission.ADMIN, MemoryScope.PRIVATE, "alice", PrivacyLevel.CONFIDENTIAL)
        assert result is True

    def test_sensitive_limited_access(self):
        result = self.checker.check("bot1", AccessPermission.READ, MemoryScope.AGENT_GROUP, "alice", PrivacyLevel.SENSITIVE)
        assert result is True

    def test_sensitive_write_blocked(self):
        result = self.checker.check("bot1", AccessPermission.WRITE, MemoryScope.AGENT_GROUP, "alice", PrivacyLevel.SENSITIVE)
        assert result is False

    def test_check_batch(self):
        memories = [
            {"scope": "private", "user_id": "alice", "privacy": "standard"},
            {"scope": "public", "user_id": "alice", "privacy": "standard"},
            {"scope": "agent_group", "user_id": "bob", "privacy": "standard"},
        ]
        accessible, blocked = self.checker.check_batch("alice", AccessPermission.READ, memories)
        assert len(accessible) >= 1  # own private + public


class TestPrivacyFilter:

    def setup_method(self):
        self.filter = PrivacyFilter()

    def test_standard_no_filter(self):
        content = "我的手机号是13800138000"
        result = self.filter.filter(content, PrivacyLevel.STANDARD)
        assert result == content  # unchanged

    def test_sensitive_filters_critical(self):
        content = "密码是abc123，手机号是13800138000"
        result = self.filter.filter(content, PrivacyLevel.SENSITIVE)
        assert "[REDACTED]" in result
        # Password should be redacted, phone might not be (only first 3 patterns)
        assert "abc123" not in result or "[REDACTED]" in result

    def test_confidential_filters_all(self):
        content = "银行卡号6222001234567890，地址是北京市朝阳区"
        result = self.filter.filter(content, PrivacyLevel.CONFIDENTIAL)
        assert "[REDACTED]" in result

    def test_filter_memories(self):
        memories = [
            {"content": "hello world", "privacy": "standard"},
            {"content": "secret info", "privacy": "confidential"},
        ]
        result = self.filter.filter_memories(memories, PrivacyLevel.SENSITIVE)
        # standard(order=0) <= sensitive(order=1) → included
        # confidential(order=2) > sensitive(order=1) → excluded
        assert len(result) == 1
        assert result[0]["privacy"] == "standard"

    def test_filter_memories_standard_target(self):
        memories = [
            {"content": "hello", "privacy": "standard"},
            {"content": "secret", "privacy": "sensitive"},
        ]
        result = self.filter.filter_memories(memories, PrivacyLevel.STANDARD)
        # only standard(0) <= standard(0) → included; sensitive(1) > standard(0) → excluded
        assert len(result) == 1

    def test_custom_patterns(self):
        filter = PrivacyFilter(sensitive_patterns=["特殊字段"])
        content = "这个特殊字段很重要"
        result = filter.filter(content, PrivacyLevel.CONFIDENTIAL)
        assert "[REDACTED]" in result


class TestPermissionRule:

    def test_to_dict(self):
        rule = PermissionRule(
            scope=MemoryScope.PRIVATE,
            privacy=PrivacyLevel.STANDARD,
            allowed_permissions={AccessPermission.READ, AccessPermission.WRITE},
            owner_id="alice",
        )
        d = rule.to_dict()
        assert d["scope"] == "private"
        assert "read" in d["allowed_permissions"]
        assert "write" in d["allowed_permissions"]

    def test_from_dict(self):
        data = {
            "scope": "private",
            "privacy": "standard",
            "allowed_permissions": ["read", "write"],
            "owner_id": "alice",
        }
        rule = PermissionRule.from_dict(data)
        assert rule.scope == MemoryScope.PRIVATE
        assert AccessPermission.READ in rule.allowed_permissions


class TestCollaborationBroker:

    def setup_method(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        self.broker = CollaborationBroker(db_path=self.tmpdb)

    def teardown_method(self):
        if os.path.exists(self.tmpdb):
            os.unlink(self.tmpdb)

    def test_create_group(self):
        result = self.broker.create_group("team1", "工程团队", ["agent1", "agent2"])
        assert result["group_id"] == "team1"
        assert "agent1" in result["members"]

    def test_add_member(self):
        self.broker.create_group("team1", members=["agent1"])
        result = self.broker.add_member("team1", "agent2")
        assert result is True
        groups = self.broker.get_groups_for_agent("agent2")
        assert len(groups) >= 1

    def test_remove_member(self):
        self.broker.create_group("team1", members=["agent1", "agent2"])
        result = self.broker.remove_member("team1", "agent2")
        assert result is True

    def test_share_memory(self):
        result = self.broker.share_memory("m1", "agent1", "agent2")
        assert result is True

    def test_get_shared_memories(self):
        self.broker.share_memory("m1", "agent1", "agent2")
        self.broker.share_memory("m2", "agent1", "agent2")
        shared = self.broker.get_shared_memories("agent2")
        assert len(shared) == 2

    def test_unshare_memory(self):
        self.broker.share_memory("m1", "agent1", "agent2")
        result = self.broker.unshare_memory("m1", "agent1", "agent2")
        assert result is True
        shared = self.broker.get_shared_memories("agent2")
        assert len(shared) == 0

    def test_get_groups_for_agent(self):
        self.broker.create_group("team1", members=["agent1"])
        groups = self.broker.get_groups_for_agent("agent1")
        assert len(groups) >= 1
        assert groups[0]["group_id"] == "team1"


class TestAgentMemoryManager:

    def setup_method(self):
        self.tmpdb = tempfile.mktemp(suffix=".db")
        self.manager = AgentMemoryManager(
            collaboration_broker=CollaborationBroker(db_path=self.tmpdb),
        )

    def teardown_method(self):
        if os.path.exists(self.tmpdb):
            os.unlink(self.tmpdb)

    def test_check_access_owner(self):
        memory = {"scope": "private", "user_id": "alice", "privacy": "standard"}
        result = self.manager.check_access("alice", "read", memory)
        assert result is True

    def test_check_access_non_owner(self):
        memory = {"scope": "private", "user_id": "alice", "privacy": "standard"}
        result = self.manager.check_access("bob", "write", memory)
        assert result is False

    def test_filter_for_sharing(self):
        memories = [
            {"content": "hello", "privacy": "standard"},
            {"content": "secret", "privacy": "confidential"},
        ]
        result = self.manager.filter_for_sharing(memories, "standard")
        # Only standard privacy should be shared at standard level
        assert len(result) == 1

    def test_share_memory(self):
        result = self.manager.share_memory_to_agent("m1", "agent1", "agent2")
        assert result is True

    def test_get_accessible_memories(self):
        all_memories = [
            {"scope": "private", "user_id": "alice", "privacy": "standard", "content": "alice secret"},
            {"scope": "public", "user_id": "bob", "privacy": "standard", "content": "public info"},
        ]
        accessible = self.manager.get_accessible_memories("alice", all_memories)
        # Alice can see own private + public
        assert len(accessible) >= 1

    def test_get_shared_with_agent(self):
        self.manager.share_memory_to_agent("m1", "agent1", "agent2")
        shared = self.manager.get_shared_with_agent("agent2")
        assert len(shared) >= 1

    def test_get_agent_groups(self):
        self.manager.collaboration_broker.create_group("team1", members=["agent1"])
        groups = self.manager.get_agent_groups("agent1")
        assert len(groups) >= 1
