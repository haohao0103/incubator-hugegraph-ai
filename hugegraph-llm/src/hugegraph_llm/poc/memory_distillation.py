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
Experience + Skill distillation layer for HugeGraph-AI-Memory.

Aligns with PowerMem v1.2.0:
  - Experience: condensed, long-term factual summaries from atomic memories
  - Skill: reusable, generalised knowledge derived from experiences

Storage:
  - SQLite: experiences / skills tables + FAISS/BM25 indexes
  - HugeGraph: Experience/Skill vertices with derived_from / applies_to edges
"""

import json
import os
import sqlite3
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.utils.log import log


# SQLite helpers (same DB as memory_backend)
def _get_db():
    from hugegraph_llm.poc.memory_backend import get_metadata_db

    return get_metadata_db()


def _init_schema():
    db = _get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS experiences (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'demo_user',
            summary TEXT NOT NULL,
            source_memories TEXT NOT NULL,  -- JSON list of memory ids
            entities TEXT,                  -- JSON list of involved entities
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_exp_user ON experiences(user_id);

        CREATE TABLE IF NOT EXISTS skills (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'demo_user',
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            pattern TEXT,                   -- when to apply this skill
            source_experiences TEXT,        -- JSON list of experience ids
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skill_user ON skills(user_id);
    """)
    db.commit()
    db.close()


class ExperienceStore:
    """Condense atomic memories into persistent Experience entries."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        _init_schema()

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        if self.llm is None:
            return ""
        try:
            resp = self.llm.chat.completions.create(
                model=memory_settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.error("Experience LLM call failed: %s", e)
            return ""

    def distill(
        self,
        memories: List[Dict[str, Any]],
        user_id: str = "demo_user",
        focus_entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Distill a list of atomic memories into Experience summaries.

        Args:
            memories: list of {"id": str, "content": str, "created_at": float}
            user_id: scope identifier
            focus_entities: optional entities to focus summarisation on

        Returns:
            list of created experience dicts
        """
        if not memories:
            return []

        focus = ", ".join(focus_entities) if focus_entities else "所有实体"
        mem_text = "\n".join(
            f"[{i+1}] {m.get('content', '')}" for i, m in enumerate(memories)
        )
        prompt = (
            "你是一个记忆蒸馏助手。请将以下原子记忆凝练为几条高层次的\"经验\"(Experience)。\n\n"
            "要求：\n"
            "1. 每个经验用一句话总结，包含明确的主体、行为/属性、时间或背景。\n"
            f"2. 重点关注这些实体：{focus}\n"
            "3. 合并重复或相似的记忆，去除细节噪音。\n"
            "4. 输出严格JSON数组，格式：\n"
            '[{"summary": "...", "entities": ["entity1", "entity2"]}]\n\n'
            f"记忆列表：\n{mem_text}\n\n"
            "只输出JSON，不要解释。"
        )
        raw = self._call_llm(prompt)
        if not raw:
            return []
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            # try to extract JSON array
            import re

            m = re.search(r"\[.*\]", raw, re.DOTALL)
            arr = json.loads(m.group()) if m else []

        created = []
        now = time.time()
        mem_ids = [m.get("id") for m in memories]
        db = _get_db()
        for item in arr:
            exp_id = str(uuid.uuid4())[:8]
            summary = item.get("summary", "").strip()
            if not summary:
                continue
            entities = item.get("entities", [])
            db.execute(
                "INSERT INTO experiences (id, user_id, summary, source_memories, entities, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (
                    exp_id,
                    user_id,
                    summary,
                    json.dumps(mem_ids, ensure_ascii=False),
                    json.dumps(entities, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            created.append(
                {
                    "id": exp_id,
                    "user_id": user_id,
                    "summary": summary,
                    "entities": entities,
                    "source_memories": mem_ids,
                }
            )
        db.commit()
        db.close()
        return created

    def retrieve(
        self,
        query: str,
        user_id: str = "demo_user",
        top_k: int = 5,
        memory_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a query or to a set of memory ids."""
        db = _get_db()
        sql = "SELECT * FROM experiences WHERE user_id=?"
        params = [user_id]
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            sql += f" AND source_memories IN ({placeholders})"
            params.extend(memory_ids)
        sql += " ORDER BY updated_at DESC"
        rows = db.execute(sql, params).fetchall()
        db.close()

        # Simple keyword overlap ranking
        qtok = set(query.lower().split()) if query else set()
        scored = []
        for row in rows:
            summary = row["summary"]
            stok = set(summary.lower().split())
            overlap = len(qtok & stok)
            scored.append((overlap, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r["id"],
                "summary": r["summary"],
                "entities": json.loads(r["entities"] or "[]"),
                "source_memories": json.loads(r["source_memories"] or "[]"),
                "score": sc,
            }
            for sc, r in scored[:top_k]
        ]


class SkillStore:
    """Derive reusable Skills from Experiences."""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        _init_schema()

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        if self.llm is None:
            return ""
        try:
            resp = self.llm.chat.completions.create(
                model=memory_settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.error("Skill LLM call failed: %s", e)
            return ""

    def distill(
        self,
        experiences: List[Dict[str, Any]],
        user_id: str = "demo_user",
    ) -> List[Dict[str, Any]]:
        """Turn experiences into reusable skill rules."""
        if not experiences:
            return []

        exp_text = "\n".join(
            f"[{i+1}] {e.get('summary', '')}" for i, e in enumerate(experiences)
        )
        prompt = (
            "你是一个技能提取助手。请从以下经验中抽象出可复用的\"技能\"(Skill)。\n\n"
            "要求：\n"
            "1. 每个技能包含：name(技能名), description(一句话描述), pattern(触发条件/适用场景)。\n"
            "2. 技能应该是通用、可指导未来行为的规则，而非具体事实。\n"
            "3. 输出严格JSON数组，格式：\n"
            '[{"name": "...", "description": "...", "pattern": "..."}]\n\n'
            f"经验列表：\n{exp_text}\n\n"
            "只输出JSON，不要解释。"
        )
        raw = self._call_llm(prompt)
        if not raw:
            return []
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            import re

            m = re.search(r"\[.*\]", raw, re.DOTALL)
            arr = json.loads(m.group()) if m else []

        created = []
        now = time.time()
        exp_ids = [e.get("id") for e in experiences]
        db = _get_db()
        for item in arr:
            skill_id = str(uuid.uuid4())[:8]
            name = item.get("name", "").strip()
            desc = item.get("description", "").strip()
            pattern = item.get("pattern", "").strip()
            if not name or not desc:
                continue
            db.execute(
                "INSERT INTO skills (id, user_id, name, description, pattern, "
                "source_experiences, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    skill_id,
                    user_id,
                    name,
                    desc,
                    pattern,
                    json.dumps(exp_ids, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            created.append(
                {
                    "id": skill_id,
                    "user_id": user_id,
                    "name": name,
                    "description": desc,
                    "pattern": pattern,
                    "source_experiences": exp_ids,
                }
            )
        db.commit()
        db.close()
        return created

    def retrieve(
        self,
        query: str,
        user_id: str = "demo_user",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve skills relevant to a query."""
        db = _get_db()
        rows = db.execute(
            "SELECT * FROM skills WHERE user_id=? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()
        db.close()

        qtok = set(query.lower().split()) if query else set()
        scored = []
        for row in rows:
            text = f"{row['name']} {row['description']} {row['pattern']}"
            stok = set(text.lower().split())
            overlap = len(qtok & stok)
            scored.append((overlap, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "pattern": r["pattern"],
                "score": sc,
            }
            for sc, r in scored[:top_k]
        ]


class DistillationPipeline:
    """High-level facade: run full Experience + Skill distillation for a user."""

    def __init__(self, llm_client=None):
        self.exp_store = ExperienceStore(llm_client=llm_client)
        self.skill_store = SkillStore(llm_client=llm_client)

    def distill_all(
        self,
        memories: List[Dict[str, Any]],
        user_id: str = "demo_user",
        threshold: int = None,
    ) -> Dict[str, Any]:
        """
        Distill all provided memories into experiences, then into skills.

        Args:
            memories: atomic memory dicts (id/content/created_at)
            user_id: scope
            threshold: min number of memories required to trigger distillation

        Returns:
            {"experiences": [...], "skills": [...], "input_count": int}
        """
        threshold = threshold or memory_settings.experience_threshold
        if len(memories) < threshold:
            return {"experiences": [], "skills": [], "input_count": len(memories), "threshold": threshold}

        experiences = self.exp_store.distill(memories, user_id=user_id)
        skills = self.skill_store.distill(experiences, user_id=user_id) if experiences else []
        return {
            "experiences": experiences,
            "skills": skills,
            "input_count": len(memories),
            "threshold": threshold,
        }
