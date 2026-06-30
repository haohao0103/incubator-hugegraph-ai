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
User Profile independent module (PowerMem UserMemory + UserProfileStore aligned).

PowerMem separates UserMemory from the core Memory class, providing:
  - UserProfileStore: per-user profile (name, preferences, topics, relationships)
  - UserMemory: user-specific memory operations with profile-aware retrieval
  - topics: auto-extracted interest topics from memory history

We implement:
  - UserProfileStore: SQLite-backed per-user profile storage
  - UserProfile: data class for structured user information
  - TopicExtractor: extract interest topics from memory content
  - ProfileInjector: inject user profile into query rewrite / retrieval
"""

import json
import os
import re
import sqlite3
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Set

from hugegraph_llm.utils.log import log


# Default DB path for user profiles
_DEFAULT_PROFILE_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "poc", "user_profile.db"
)


class UserProfile:
    """Structured user profile data class."""

    def __init__(
        self,
        user_id: str,
        name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        relationships: Optional[Dict[str, str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        summary: Optional[str] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.user_id = user_id
        self.name = name or ""
        self.preferences = preferences or {}
        self.topics = topics or []
        self.relationships = relationships or {}
        self.aliases = aliases or {}
        self.summary = summary or ""
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "preferences": self.preferences,
            "topics": self.topics,
            "relationships": self.relationships,
            "aliases": self.aliases,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data.get("user_id", ""),
            name=data.get("name"),
            preferences=data.get("preferences"),
            topics=data.get("topics"),
            relationships=data.get("relationships"),
            aliases=data.get("aliases"),
            summary=data.get("summary"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata"),
        )

    def update_from_memories(self, memories: List[str]) -> "UserProfile":
        """Auto-update profile fields from a list of memory contents.

        Extracts:
          - name: first occurrence of "我叫/我的名字是/My name is"
          - preferences: "喜欢/讨厌/偏好/prefer/like/hate" patterns
          - topics: auto-extracted via TopicExtractor
          - aliases: "也叫/also known as" patterns
        """
        extractor = TopicExtractor()
        all_text = "\n".join(memories)

        # Extract name
        name_patterns = [
            r"我叫([\u4e00-\u9fa5]{2,4})",
            r"我的名字(?:是|叫)([\u4e00-\u9fa5]{2,4})",
            r"My name is ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            r"I'm ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
        ]
        if not self.name:
            for pat in name_patterns:
                m = re.search(pat, all_text)
                if m:
                    self.name = m.group(1).strip()
                    break

        # Extract preferences
        pref_patterns = [
            (r"喜欢([\u4e00-\u9fa5]{2,10})", "likes"),
            (r"讨厌([\u4e00-\u9fa5]{2,10})", "dislikes"),
            (r"偏好([\u4e00-\u9fa5]{2,10})", "prefers"),
            (r"擅长([\u4e00-\u9fa5]{2,10})", "good_at"),
        ]
        for pat, category in pref_patterns:
            for m in re.finditer(pat, all_text):
                item = m.group(1).strip()
                if category not in self.preferences:
                    self.preferences[category] = []
                if item not in self.preferences[category]:
                    self.preferences[category].append(item)

        # Extract topics
        new_topics = extractor.extract(all_text)
        for topic in new_topics:
            if topic not in self.topics:
                self.topics.append(topic)

        # Extract aliases
        alias_patterns = [
            r"([\u4e00-\u9fa5]{2,6})也(?:叫|称为|叫作)([\u4e00-\u9fa5]{2,6})",
            r"([\u4e00-\u9fa5]{2,6})(?:也叫|又称)([\u4e00-\u9fa5]{2,6})",
        ]
        for pat in alias_patterns:
            for m in re.finditer(pat, all_text):
                alias = m.group(1).strip()
                canonical = m.group(2).strip()
                self.aliases[alias] = canonical

        # Update summary if empty
        if not self.summary and memories:
            self.summary = extractor.summarize(all_text)

        self.updated_at = time.time()
        return self

    def get_search_profile(self) -> str:
        """Return a search-optimized profile string for query rewrite."""
        parts = []
        if self.name:
            parts.append(f"用户名:{self.name}")
        if self.topics:
            parts.append(f"兴趣:{','.join(self.topics[:10])}")
        if self.preferences:
            likes = self.preferences.get("likes", [])
            if likes:
                parts.append(f"喜欢:{','.join(likes[:5])}")
        return " ".join(parts) if parts else ""


class UserProfileStore:
    """SQLite-backed per-user profile storage.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str = _DEFAULT_PROFILE_DB):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        """Initialize the user_profiles table."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            db = sqlite3.connect(self.db_path)
            db.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT DEFAULT '',
                    preferences TEXT DEFAULT '{}',
                    topics TEXT DEFAULT '[]',
                    relationships TEXT DEFAULT '{}',
                    aliases TEXT DEFAULT '{}',
                    summary TEXT DEFAULT '',
                    created_at REAL,
                    updated_at REAL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            db.commit()
            db.close()

    def get(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by user_id."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM user_profiles WHERE user_id=?", (user_id,)
            ).fetchone()
            db.close()

            if not row:
                return None
            return UserProfile.from_dict({
                "user_id": row["user_id"],
                "name": row["name"],
                "preferences": json.loads(row["preferences"]),
                "topics": json.loads(row["topics"]),
                "relationships": json.loads(row["relationships"]),
                "aliases": json.loads(row["aliases"]),
                "summary": row["summary"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"]),
            })

    def save(self, profile: UserProfile) -> UserProfile:
        """Save or update a user profile."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, name, preferences, topics, relationships, aliases,
                 summary, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.name,
                json.dumps(profile.preferences),
                json.dumps(profile.topics),
                json.dumps(profile.relationships),
                json.dumps(profile.aliases),
                profile.summary,
                profile.created_at,
                profile.updated_at,
                json.dumps(profile.metadata),
            ))
            db.commit()
            db.close()
        return profile

    def update_from_memories(
        self,
        user_id: str,
        memories: List[str],
    ) -> UserProfile:
        """Update user profile from recent memories.

        Args:
            user_id: User to update.
            memories: List of memory content strings.

        Returns:
            Updated UserProfile.
        """
        profile = self.get(user_id) or UserProfile(user_id=user_id)
        profile.update_from_memories(memories)
        return self.save(profile)

    def delete(self, user_id: str) -> bool:
        """Delete a user profile."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            db.execute("DELETE FROM user_profiles WHERE user_id=?", (user_id,))
            db.commit()
            db.close()
        return True

    def list_users(self) -> List[str]:
        """List all user_ids with profiles."""
        with self._lock:
            db = sqlite3.connect(self.db_path)
            rows = db.execute("SELECT user_id FROM user_profiles").fetchall()
            db.close()
        return [r[0] for r in rows]

    def get_all_profiles(self) -> List[UserProfile]:
        """Get all user profiles."""
        profiles = []
        for uid in self.list_users():
            p = self.get(uid)
            if p:
                profiles.append(p)
        return profiles


class TopicExtractor:
    """Extract interest topics from memory content.

    Uses a combination of:
      - Keyword frequency analysis
      - Named entity extraction (Chinese/English)
      - Domain-specific topic patterns
    """

    # Domain-specific topic patterns
    _TOPIC_PATTERNS = {
        "编程": [r"编程", r"开发", r"代码", r"Python", r"Java", r"算法"],
        "音乐": [r"音乐", r"歌曲", r"乐器", r"吉他", r"钢琴", r"演唱"],
        "运动": [r"运动", r"健身", r"跑步", r"游泳", r"篮球", r"足球"],
        "旅行": [r"旅行", r"旅游", r"出差", r"航班", r"酒店"],
        "阅读": [r"阅读", r"读书", r"书籍", r"小说", r"论文"],
        "投资": [r"投资", r"股票", r"基金", r"理财", r"收益"],
        "烹饪": [r"烹饪", r"做饭", r"食谱", r"厨艺"],
        "电影": [r"电影", r"影片", r"导演", r"演员", r"票房"],
    }

    def extract(self, text: str, max_topics: int = 20) -> List[str]:
        """Extract topics from text.

        Args:
            text: Input text (can be concatenation of multiple memories).
            max_topics: Maximum topics to return.

        Returns:
            List of topic strings.
        """
        topics: Set[str] = set()

        # 1. Domain pattern matching
        for topic, patterns in self._TOPIC_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    topics.add(topic)
                    break

        # 2. Chinese keyword frequency
        words = re.findall(r"[\u4e00-\u9fa5]{2,6}", text)
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        # High-frequency words > 2 occurrences are likely topics
        for w, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count >= 2 and len(w) >= 2:
                topics.add(w)

        # 3. English capitalized terms
        for m in re.finditer(r"[A-Z][a-zA-Z]{2,}", text):
            topics.add(m.group(0))

        return list(topics)[:max_topics]

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a simple extractive summary.

        Takes the first N meaningful sentences as the summary.
        """
        sentences = re.split(r"[。！？\n.!?]+", text)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 10]
        summary = " ".join(meaningful[:5])
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary


class ProfileInjector:
    """Inject user profile into query rewrite and retrieval.

    This bridges UserProfileStore → LLMQueryRewriteEngine / search_memory,
    ensuring that user-specific context (aliases, preferences, topics)
    is used for better retrieval.
    """

    def __init__(self, profile_store: Optional[UserProfileStore] = None):
        self.profile_store = profile_store or UserProfileStore()

    def get_profile_for_rewrite(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get profile data formatted for query rewrite engine.

        Returns:
            Dict with keys: 'user_profile', 'aliases', 'topics'
        """
        profile = self.profile_store.get(user_id)
        if not profile:
            return {"user_profile": "", "aliases": {}, "topics": []}

        return {
            "user_profile": profile.get_search_profile(),
            "aliases": profile.aliases,
            "topics": profile.topics,
        }

    def inject_aliases(
        self,
        user_id: str,
        existing_aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Merge user-specific aliases with existing alias mapping."""
        profile = self.profile_store.get(user_id)
        merged = dict(existing_aliases or {})
        if profile:
            merged.update(profile.aliases)
        return merged
