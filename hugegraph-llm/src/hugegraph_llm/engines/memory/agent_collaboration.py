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
Multi-agent collaboration and permissions (PowerMem AgentMemory aligned).

PowerMem's AgentMemory provides:
  - Memory scope: private / agent_group / user_group / public
  - Privacy levels: standard / sensitive / confidential
  - Access permissions: read / write / delete / share / admin
  - Collaboration: isolated vs collaborative mode
  - Cross-agent memory sharing with privacy filtering

We implement:
  - AgentMemoryManager: manage multi-agent memory access and sharing
  - PermissionChecker: check access permissions for operations
  - CollaborationBroker: facilitate cross-agent memory exchange
  - PrivacyFilter: filter sensitive information before sharing
"""

import json
import os
import sqlite3
import time
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.engines.memory.base import (
    AccessPermission,
    CollaborationLevel,
    MemoryScope,
    PrivacyLevel,
)
from hugegraph_llm.utils.log import log

# Privacy level ordering for comparison (string enums don't have natural ordering)
_PRIVACY_ORDER = {
    PrivacyLevel.STANDARD: 0,
    PrivacyLevel.SENSITIVE: 1,
    PrivacyLevel.CONFIDENTIAL: 2,
}


# Default DB path for agent permissions
_DEFAULT_PERM_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "poc", "agent_permissions.db"
)


class PermissionRule:
    """A single permission rule for a memory scope."""

    def __init__(
        self,
        scope: MemoryScope,
        privacy: PrivacyLevel,
        allowed_permissions: Set[AccessPermission],
        owner_id: str = "",
        target_id: str = "",  # who this rule applies to
    ):
        self.scope = scope
        self.privacy = privacy
        self.allowed_permissions = allowed_permissions
        self.owner_id = owner_id
        self.target_id = target_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "privacy": self.privacy.value,
            "allowed_permissions": [p.value for p in self.allowed_permissions],
            "owner_id": self.owner_id,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionRule":
        return cls(
            scope=MemoryScope(data.get("scope", "private")),
            privacy=PrivacyLevel(data.get("privacy", "standard")),
            allowed_permissions={AccessPermission(p) for p in data.get("allowed_permissions", ["read"])},
            owner_id=data.get("owner_id", ""),
            target_id=data.get("target_id", ""),
        )


# Default permission rules per scope
# These define what NON-OWNER agents can do for each scope.
# Owners always have full access regardless of scope.
_DEFAULT_NON_OWNER_RULES: Dict[MemoryScope, Set[AccessPermission]] = {
    MemoryScope.PRIVATE: set(),  # Non-owners have NO access to private memories
    MemoryScope.AGENT_GROUP: {AccessPermission.READ, AccessPermission.WRITE},
    MemoryScope.USER_GROUP: {AccessPermission.READ},
    MemoryScope.PUBLIC: {AccessPermission.READ},
    MemoryScope.RESTRICTED: set(),
}


class PermissionChecker:
    """Check whether an agent has permission to perform an operation.

    Args:
        custom_rules: Optional override of default permission rules.
    """

    def __init__(
        self,
        custom_rules: Optional[Dict[MemoryScope, Set[AccessPermission]]] = None,
    ):
        self.rules = custom_rules or _DEFAULT_NON_OWNER_RULES

    def check(
        self,
        agent_id: str,
        operation: AccessPermission,
        memory_scope: MemoryScope,
        memory_owner: str,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
    ) -> bool:
        """Check if agent_id has permission for operation on a memory.

        Args:
            agent_id: The agent requesting access.
            operation: The operation being requested (read/write/delete/share/admin).
            memory_scope: The memory's scope.
            memory_owner: The user_id that owns the memory.
            privacy_level: The memory's privacy level.

        Returns:
            True if permission is granted, False otherwise.
        """
        # Owner always has full access
        if agent_id == memory_owner:
            return True

        # Check scope-based rules
        allowed = self.rules.get(memory_scope, set())
        if operation not in allowed:
            return False

        # Privacy-level override: confidential memories need admin permission
        if privacy_level == PrivacyLevel.CONFIDENTIAL and operation != AccessPermission.ADMIN:
            return False

        # Sensitive memories: only read for non-owners
        if privacy_level == PrivacyLevel.SENSITIVE and operation not in {AccessPermission.READ, AccessPermission.ADMIN}:
            return False

        return True

    def check_batch(
        self,
        agent_id: str,
        operation: AccessPermission,
        memories: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Check permissions for a batch of memories.

        Args:
            agent_id: Agent requesting access.
            operation: Operation type.
            memories: List of memory dicts with 'scope', 'owner', 'privacy' keys.

        Returns:
            (accessible, blocked) tuple.
        """
        accessible, blocked = [], []
        for mem in memories:
            scope = MemoryScope(mem.get("scope", "private"))
            owner = mem.get("user_id", mem.get("owner", ""))
            privacy = PrivacyLevel(mem.get("privacy", "standard"))

            if self.check(agent_id, operation, scope, owner, privacy):
                accessible.append(mem)
            else:
                blocked.append(mem)
        return accessible, blocked


class PrivacyFilter:
    """Filter sensitive information from memories before sharing.

    Args:
        sensitive_patterns: Regex patterns for sensitive content.
        redaction_replacement: Text to replace sensitive content with.
    """

    _DEFAULT_SENSITIVE_PATTERNS = [
        r"\d{6,}",  # long numbers (IDs, account numbers)
        r"密码|password|passwd|pwd",  # passwords
        r"身份证|ID card|social security|SSN",  # identity documents
        r"银行卡|bank account|信用卡|credit card",  # financial info
        r"手机号|phone number|电话|tel",  # contact info
        r"地址|address|住址|住处",  # address
    ]

    def __init__(
        self,
        sensitive_patterns: Optional[List[str]] = None,
        redaction_replacement: str = "[REDACTED]",
    ):
        self.sensitive_patterns = sensitive_patterns or self._DEFAULT_SENSITIVE_PATTERNS
        self.redaction_replacement = redaction_replacement

    def filter(
        self,
        content: str,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
    ) -> str:
        """Filter sensitive information from content based on privacy level.

        Args:
            content: Original content string.
            privacy_level: Privacy level determining filter strength.

        Returns:
            Filtered content string.
        """
        if privacy_level == PrivacyLevel.STANDARD:
            # No filtering for standard privacy
            return content

        import re as _re
        filtered = content

        if privacy_level == PrivacyLevel.SENSITIVE:
            # Filter only the most critical patterns
            critical_patterns = self.sensitive_patterns[:3]  # passwords, IDs, financial
            for pat in critical_patterns:
                filtered = _re.sub(pat, self.redaction_replacement, filtered, flags=_re.IGNORECASE)

        elif privacy_level == PrivacyLevel.CONFIDENTIAL:
            # Filter all sensitive patterns
            for pat in self.sensitive_patterns:
                filtered = _re.sub(pat, self.redaction_replacement, filtered, flags=_re.IGNORECASE)

        return filtered

    def filter_memories(
        self,
        memories: List[Dict[str, Any]],
        target_privacy: PrivacyLevel = PrivacyLevel.STANDARD,
    ) -> List[Dict[str, Any]]:
        """Filter a batch of memories for sharing.

        Args:
            memories: List of memory dicts with 'content' and 'privacy' keys.
            target_privacy: Maximum privacy level to share.

        Returns:
            List of filtered memory dicts.
        """
        result = []
        for mem in memories:
            privacy = PrivacyLevel(mem.get("privacy", "standard"))
            # Only share memories at or below the target privacy level
            # Use explicit ordering since string enums don't have numeric ordering
            if _PRIVACY_ORDER.get(privacy, 0) <= _PRIVACY_ORDER.get(target_privacy, 0):
                filtered_content = self.filter(mem.get("content", ""), privacy)
                result.append({**mem, "content": filtered_content})
        return result


class CollaborationBroker:
    """Broker cross-agent memory sharing.

    Manages shared memory pools and collaboration rules between agents.

    Args:
        db_path: Path to SQLite database for collaboration state.
    """

    def __init__(self, db_path: str = _DEFAULT_PERM_DB):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        """Initialize collaboration tables."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            db = sqlite3.connect(self.db_path)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS collaboration_groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    members TEXT DEFAULT '[]',
                    scope TEXT DEFAULT 'agent_group',
                    created_at REAL
                );

                CREATE TABLE IF NOT EXISTS shared_memories (
                    memory_id TEXT,
                    source_agent TEXT,
                    target_agent TEXT,
                    scope TEXT DEFAULT 'agent_group',
                    shared_at REAL,
                    PRIMARY KEY (memory_id, source_agent, target_agent)
                );

                CREATE TABLE IF NOT EXISTS permission_rules (
                    owner_id TEXT,
                    target_id TEXT,
                    scope TEXT,
                    privacy TEXT,
                    permissions TEXT,
                    PRIMARY KEY (owner_id, target_id, scope)
                );
            """)
            db.commit()
            db.close()

    def create_group(
        self,
        group_id: str,
        name: str = "",
        members: Optional[List[str]] = None,
        scope: MemoryScope = MemoryScope.AGENT_GROUP,
    ) -> Dict[str, Any]:
        """Create a collaboration group.

        Args:
            group_id: Unique group identifier.
            name: Human-readable group name.
            members: List of agent_ids in the group.
            scope: Default sharing scope for the group.

        Returns:
            Group info dict.
        """
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.execute("""
                INSERT OR REPLACE INTO collaboration_groups
                (group_id, name, members, scope, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                group_id, name,
                json.dumps(members or []),
                scope.value,
                time.time(),
            ))
            db.commit()
            db.close()

        return {
            "group_id": group_id,
            "name": name,
            "members": members or [],
            "scope": scope.value,
        }

    def add_member(
        self,
        group_id: str,
        agent_id: str,
    ) -> bool:
        """Add an agent to a collaboration group."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            row = db.execute(
                "SELECT members FROM collaboration_groups WHERE group_id=?", (group_id,)
            ).fetchone()
            if not row:
                db.close()
                return False

            members = json.loads(row[0])
            if agent_id not in members:
                members.append(agent_id)
                db.execute(
                    "UPDATE collaboration_groups SET members=? WHERE group_id=?",
                    (json.dumps(members), group_id),
                )
                db.commit()
            db.close()
        return True

    def remove_member(
        self,
        group_id: str,
        agent_id: str,
    ) -> bool:
        """Remove an agent from a collaboration group."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            row = db.execute(
                "SELECT members FROM collaboration_groups WHERE group_id=?", (group_id,)
            ).fetchone()
            if not row:
                db.close()
                return False

            members = json.loads(row[0])
            if agent_id in members:
                members.remove(agent_id)
                db.execute(
                    "UPDATE collaboration_groups SET members=? WHERE group_id=?",
                    (json.dumps(members), group_id),
                )
                db.commit()
            db.close()
        return True

    def share_memory(
        self,
        memory_id: str,
        source_agent: str,
        target_agent: str,
        scope: MemoryScope = MemoryScope.AGENT_GROUP,
    ) -> bool:
        """Share a memory from one agent to another.

        Args:
            memory_id: Memory to share.
            source_agent: Agent sharing the memory.
            target_agent: Agent receiving access.
            scope: Sharing scope.

        Returns:
            True if sharing was recorded.
        """
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.execute("""
                INSERT OR REPLACE INTO shared_memories
                (memory_id, source_agent, target_agent, scope, shared_at)
                VALUES (?, ?, ?, ?, ?)
            """, (memory_id, source_agent, target_agent, scope.value, time.time()))
            db.commit()
            db.close()
        return True

    def get_shared_memories(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all memories shared with the given agent.

        Args:
            agent_id: Agent receiving shared memories.
            limit: Max results.

        Returns:
            List of shared memory records.
        """
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.row_factory = sqlite3.Row
            rows = db.execute("""
                SELECT * FROM shared_memories WHERE target_agent=? ORDER BY shared_at DESC LIMIT ?
            """, (agent_id, limit)).fetchall()
            db.close()

        return [
            {
                "memory_id": r["memory_id"],
                "source_agent": r["source_agent"],
                "scope": r["scope"],
                "shared_at": r["shared_at"],
            }
            for r in rows
        ]

    def get_groups_for_agent(
        self,
        agent_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all collaboration groups an agent belongs to."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM collaboration_groups"
            ).fetchall()
            db.close()

        groups = []
        for r in rows:
            members = json.loads(r["members"])
            if agent_id in members:
                groups.append({
                    "group_id": r["group_id"],
                    "name": r["name"],
                    "members": members,
                    "scope": r["scope"],
                })
        return groups

    def unshare_memory(
        self,
        memory_id: str,
        source_agent: str,
        target_agent: str,
    ) -> bool:
        """Remove a sharing relationship."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.execute("""
                DELETE FROM shared_memories
                WHERE memory_id=? AND source_agent=? AND target_agent=?
            """, (memory_id, source_agent, target_agent))
            db.commit()
            db.close()
        return True


class AgentMemoryManager:
    """Manage multi-agent memory access, sharing, and permissions.

    This is the top-level module that combines PermissionChecker,
    PrivacyFilter, and CollaborationBroker.

    Args:
        permission_checker: PermissionChecker instance.
        privacy_filter: PrivacyFilter instance.
        collaboration_broker: CollaborationBroker instance.
    """

    def __init__(
        self,
        permission_checker: Optional[PermissionChecker] = None,
        privacy_filter: Optional[PrivacyFilter] = None,
        collaboration_broker: Optional[CollaborationBroker] = None,
    ):
        self.permission_checker = permission_checker or PermissionChecker()
        self.privacy_filter = privacy_filter or PrivacyFilter()
        self.collaboration_broker = collaboration_broker or CollaborationBroker()

    def check_access(
        self,
        agent_id: str,
        operation: str,
        memory: Dict[str, Any],
    ) -> bool:
        """Check if an agent can perform an operation on a memory.

        Args:
            agent_id: Agent requesting access.
            operation: "read", "write", "delete", "share", "admin".
            memory: Memory dict with scope, owner, privacy keys.

        Returns:
            True if access is granted.
        """
        perm = AccessPermission(operation)
        scope = MemoryScope(memory.get("scope", "private"))
        owner = memory.get("user_id", memory.get("owner", ""))
        privacy = PrivacyLevel(memory.get("privacy", "standard"))

        return self.permission_checker.check(agent_id, perm, scope, owner, privacy)

    def filter_for_sharing(
        self,
        memories: List[Dict[str, Any]],
        target_privacy: str = "standard",
    ) -> List[Dict[str, Any]]:
        """Filter memories for sharing to a target privacy level.

        Args:
            memories: List of memory dicts.
            target_privacy: Maximum privacy level to share ("standard", "sensitive", "confidential").

        Returns:
            List of filtered memories suitable for sharing.
        """
        privacy = PrivacyLevel(target_privacy)
        return self.privacy_filter.filter_memories(memories, privacy)

    def share_memory_to_agent(
        self,
        memory_id: str,
        source_agent: str,
        target_agent: str,
        scope: str = "agent_group",
    ) -> bool:
        """Share a memory from one agent to another.

        Args:
            memory_id: Memory to share.
            source_agent: Source agent.
            target_agent: Target agent.
            scope: Sharing scope.

        Returns:
            True if sharing was recorded.
        """
        return self.collaboration_broker.share_memory(
            memory_id, source_agent, target_agent, MemoryScope(scope)
        )

    def get_accessible_memories(
        self,
        agent_id: str,
        all_memories: List[Dict[str, Any]],
        operation: str = "read",
    ) -> List[Dict[str, Any]]:
        """Get all memories accessible to an agent for a given operation.

        Args:
            agent_id: Agent requesting access.
            all_memories: All available memories.
            operation: Operation type.

        Returns:
            List of memories the agent can access, filtered for privacy.
        """
        accessible, _ = self.permission_checker.check_batch(
            agent_id, AccessPermission(operation), all_memories
        )
        return self.privacy_filter.filter_memories(accessible, PrivacyLevel.STANDARD)

    def get_shared_with_agent(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get memories explicitly shared with an agent."""
        return self.collaboration_broker.get_shared_memories(agent_id, limit)

    def get_agent_groups(
        self,
        agent_id: str,
    ) -> List[Dict[str, Any]]:
        """Get collaboration groups for an agent."""
        return self.collaboration_broker.get_groups_for_agent(agent_id)
