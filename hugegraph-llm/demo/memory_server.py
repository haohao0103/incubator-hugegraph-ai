"""
HugeGraph Memory Demo Server
=============================
Lightweight Flask server with:
- SQLite graph store (no HugeGraph dependency)
- LLM-powered entity extraction (xiaomimimo API)
- REST API for memory CRUD + search
- Ebbinghaus forgetting curve
- Serves interactive HTML frontend

Usage:
    python demo/memory_server.py [--port 8765]
"""

import json
import math
import sqlite3
import time
import uuid
import os
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# ============================================================================
# Config
# ============================================================================

LLM_BASE_URL = "https://api.xiaomimimo.com/v1"
LLM_MODEL = "mimo-v2.5-pro"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-cs5kqi80f6upqy2e3k3xi39jtizhpgf6dkdd3j9ysoupfw7p")

# Ebbinghaus constants (same as PowerMem)
EBBINGHAUS_K = 0.821  # decay constant
EBBINGHAUS_REINFORCE = 0.3  # access reinforcement

DB_PATH = os.path.join(os.path.dirname(__file__), "memory_demo.db")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


# ============================================================================
# Database Layer (SQLite as graph store)
# ============================================================================

def get_db():
    """Get a thread-local SQLite connection."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db


def init_db():
    """Initialize the database schema."""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'demo_user',
            created_at REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            initial_score REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            properties TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            memory_id TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (source_id) REFERENCES nodes(id),
            FOREIGN KEY (target_id) REFERENCES nodes(id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relationship);
        CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
    """)
    db.commit()
    db.close()


# ============================================================================
# LLM Entity Extraction
# ============================================================================

EXTRACT_PROMPT = """你是一个知识图谱实体和关系抽取器。从用户的输入文本中提取实体和关系。

规则：
1. 实体类型：person(人名)、organization(组织机构)、location(地点)、skill(技能/爱好)、concept(概念/事物)
2. 关系类型：works_at(在...工作)、lives_in(住在...)、likes(喜欢/爱好)、colleague_of(同事)、friend_of(朋友)、part_of(属于)、located_in(位于)
3. 人物识别："我叫XX"表示说话人名字是XX，"我的同事XX"表示同事名字是XX。直接提取具体人名，不要用代词。
4. 推理能力：如果文本说"我的同事也在腾讯"，推断该同事也在腾讯工作
5. 如果文本中同时出现了说话人名字和"我/我的"，用说话人名字替代"我/我的"

请严格按以下JSON格式输出，不要输出其他内容：
{{
    "entities": [
        {{"name": "实体名", "type": "实体类型"}}
    ],
    "relationships": [
        {{"source": "源实体名", "relationship": "关系类型", "target": "目标实体名"}}
    ]
}}

如果无法提取任何信息，返回空数组。"""

EXTRACT_SYSTEM = """你是一个精确的知识图谱信息提取器。只输出JSON，不要解释。"""


def _extract_json_from_response(response) -> dict:
    """Extract JSON from LLM response, handling reasoning models and varied key names."""
    import re
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = (msg.reasoning_content or "").strip()
    if not content:
        return {"entities": [], "relationships": []}

    # Strip markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]  # remove first line (```json or ```)
        content = "\n".join(lines)
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    # Try direct parse first
    try:
        result = json.loads(content)
        return _normalize_keys(result)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON object with entities key (handle nested content from reasoning)
    # Match the outermost { ... } that contains "entities"
    json_patterns = [
        r'\{[^{}]*"entities"\s*:\s*\[[^\]]*\][^{}]*"relationships"\s*:\s*\[[^\]]*\][^{}]*\}',
        r'\{[^{}]*"entities"\s*:\s*\[[^\]]*\][^{}]*\}',
        r'\{[^{}]*"entities"\s*:.*?\}',
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            for m in reversed(matches):
                try:
                    result = json.loads(m)
                    return _normalize_keys(result)
                except (json.JSONDecodeError, TypeError):
                    continue

    # Last resort: try to extract using regex for key-value pairs
    entities = []
    relationships = []

    # Extract entities: "name": "XX", "type": "YY"  or  "name":"XX","type":"YY"
    ent_pattern = r'"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"'
    for m in re.finditer(ent_pattern, content):
        entities.append({"name": m.group(1), "type": m.group(2).lower()})

    # Extract relationships: try both key styles
    rel_patterns = [
        r'"source"\s*:\s*"([^"]+)"\s*,\s*"relationship"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"',
        r'"subject"\s*:\s*"([^"]+)"\s*,\s*"(?:relation|relationship)"\s*:\s*"([^"]+)"\s*,\s*"(?:target|object)"\s*:\s*"([^"]+)"',
    ]
    for pattern in rel_patterns:
        for m in re.finditer(pattern, content):
            relationships.append({"source": m.group(1), "relationship": m.group(2), "target": m.group(3)})

    return {"entities": entities, "relationships": relationships}


def _normalize_keys(result: dict) -> dict:
    """Normalize LLM output key names to standard format."""
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])

    normalized_entities = []
    for e in entities:
        name = e.get("name") or e.get("entity") or e.get("value", "")
        etype = (e.get("type") or e.get("category") or e.get("label", "concept")).lower()
        # Normalize common type variations
        type_map = {
            "person": "person", "people": "person", "人": "person", "人物": "person",
            "organization": "organization", "org": "organization", "公司": "organization", "机构": "organization",
            "location": "location", "地点": "location", "地方": "location", "城市": "location",
            "skill": "skill", "技能": "skill", "爱好": "skill",
            "beverage": "concept", "drink": "concept", "sport": "skill",
        }
        etype = type_map.get(etype, etype)
        # Skip self-reference placeholders like "我"
        if name in ("我", "自己", "本人", "您"):
            continue
        normalized_entities.append({"name": name, "type": etype})

    normalized_rels = []
    for r in relationships:
        source = r.get("source") or r.get("subject") or r.get("from") or r.get("head", "")
        target = r.get("target") or r.get("object") or r.get("to") or r.get("tail", "")
        rel = r.get("relationship") or r.get("relation") or r.get("predicate") or r.get("type", "")
        if source and target and rel:
            normalized_rels.append({"source": source, "relationship": rel, "target": target})

    return {"entities": normalized_entities, "relationships": normalized_rels}


def extract_entities_and_relations(text: str) -> dict:
    """Use LLM to extract entities and relations from text."""
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": f"{EXTRACT_PROMPT}\n\n用户输入：{text}"}
            ],
            temperature=0.1,
            max_completion_tokens=2048,
        )
        result = _extract_json_from_response(response)
        import sys
        print(f"[DEBUG] Extracted {len(result.get('entities',[]))} entities, {len(result.get('relationships',[]))} rels", file=sys.stderr, flush=True)
        return result
    except Exception as e:
        import sys
        print(f"[LLM extraction error] {e}", file=sys.stderr, flush=True)
        return {"entities": [], "relationships": []}


SEARCH_PROMPT = """你是一个记忆检索器。用户提出了一个问题，请从以下记忆列表中找出最相关的记忆。

用户问题：{query}

记忆列表：
{memories}

请返回最相关的记忆ID列表，按相关性排序，每条附上相关性分数(0-1)。
严格按JSON格式输出：
[{{"memory_id": "ID", "score": 0.95, "reason": "简要原因"}}]

如果没有相关记忆，返回空数组。"""


def _get_llm_text(response) -> str:
    """Get text from LLM response, handling reasoning models."""
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content
    return (msg.reasoning_content or "").strip()


ANSWER_PROMPT = """你是一个拥有记忆能力的AI助手。根据用户的记忆信息回答问题。

用户问题：{query}

相关记忆：
{memories}

图谱关系：
{graph_context}

请用简洁自然的中文回答。如果记忆中没有足够信息，诚实说明。"""


def generate_answer(query: str, memories: list, graph_context: str = "") -> str:
    """Use LLM to generate answer based on memories."""
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        memory_text = "\n".join([
            f"- {m['content']}" for m in memories
        ])
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是HugeGraph Memory助手，基于用户的记忆回答问题。回答要简洁。"},
                {"role": "user", "content": ANSWER_PROMPT.format(
                    query=query, memories=memory_text, graph_context=graph_context or "无"
                )}
            ],
            temperature=0.3,
            max_completion_tokens=2048,
        )
        content = _get_llm_text(response)
        if not content:
            return "无法生成回答。"
        return content
    except Exception as e:
        print(f"[LLM answer error] {e}", file=sys.stderr, flush=True)
        return f"生成回答时出错: {e}"


# ============================================================================
# Business Logic
# ============================================================================

class MemoryStore:
    """Memory store with Ebbinghaus forgetting curve."""

    def __init__(self):
        pass

    def _get_user_name(self, db, user_id: str) -> str:
        """Extract the user's name from past memories (e.g., '我叫张三')."""
        import re
        rows = db.execute(
            "SELECT content FROM memories WHERE user_id=? ORDER BY created_at ASC",
            (user_id,)
        ).fetchall()
        for row in rows:
            # Match "我叫XX" or "我是XX" patterns
            m = re.search(r'(?:我叫|我是)\s*([\u4e00-\u9fa5]{2,4})', row["content"])
            if m:
                return m.group(1)
        return ""

    def _dedup_entities(self, entities: list, relationships: list, db) -> tuple:
        """Deduplicate and merge entities (e.g., '腾讯深圳' -> '腾讯' + '深圳')."""
        import re
        # Get existing node names
        existing_names = set(
            row["name"] for row in db.execute("SELECT name FROM nodes")
        )
        merged = {}  # old_name -> new_name
        new_entities = []

        for ent in entities:
            name = ent["name"]
            etype = ent["type"]
            # Check if this name contains an existing name (e.g., "腾讯深圳" contains "腾讯")
            best_match = None
            for ename in existing_names:
                if ename in name and ename != name:
                    # "腾讯深圳" contains "腾讯" -> split into "腾讯" + "深圳"
                    best_match = ename
                    remainder = name.replace(ename, "").strip()
                    if remainder and remainder not in existing_names:
                        # Determine remainder type: if it looks like a city, mark as location
                        if re.match(r'^[\u4e00-\u9fa5]{2,3}$', remainder):
                            new_entities.append({"name": remainder, "type": "location"})
                    break
            if best_match:
                merged[name] = best_match
                # Don't add the merged entity as new; it already exists
            elif name not in existing_names:
                new_entities.append(ent)

        # Update relationship references
        for rel in relationships:
            if rel["source"] in merged:
                rel["source"] = merged[rel["source"]]
            if rel["target"] in merged:
                rel["target"] = merged[rel["target"]]

        return new_entities, relationships

    def _infer_colleague(self, relationships: list, entities: list, db, user_id: str) -> list:
        """Infer colleague_of relationships between persons at the same org."""
        person_names = set(e["name"] for e in entities if e["type"] == "person")

        # Also include all persons from relationships (they may not be in current entities)
        for rel in relationships:
            if rel["relationship"] == "works_at":
                person_names.add(rel["source"])
        # And persons from existing edges
        for row in db.execute(
            """SELECT n1.name as src FROM edges e JOIN nodes n1 ON e.source_id=n1.id
               WHERE e.relationship='works_at' AND n1.type='person'"""
        ).fetchall():
            person_names.add(row["src"])

        if len(person_names) < 2:
            return relationships

        # Check if any two persons work at the same org (existing + new edges)
        all_rels = list(relationships)
        existing_edges = db.execute(
            """SELECT n1.name as src, e.relationship, n2.name as tgt
               FROM edges e JOIN nodes n1 ON e.source_id=n1.id JOIN nodes n2 ON e.target_id=n2.id
               WHERE e.relationship='works_at'"""
        ).fetchall()
        for ee in existing_edges:
            all_rels.append({"source": ee["src"], "relationship": ee["relationship"], "target": ee["tgt"]})

        # Build person -> org mapping
        person_orgs = {}
        for rel in all_rels:
            if rel["relationship"] == "works_at":
                if rel["source"] in person_names or rel["source"] in person_orgs:
                    person_orgs.setdefault(rel["source"], set()).add(rel["target"])
                if rel["target"] in person_names:
                    person_orgs.setdefault(rel["source"], set()).add(rel["target"])

        # For persons sharing an org, add colleague_of
        persons = list(person_names)
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                orgs1 = person_orgs.get(p1, set())
                orgs2 = person_orgs.get(p2, set())
                if orgs1 & orgs2:  # share at least one org
                    # Avoid duplicate
                    exists = any(
                        r["relationship"] == "colleague_of" and
                        set([r["source"], r["target"]]) == set([p1, p2])
                        for r in relationships
                    )
                    if not exists:
                        relationships.append({
                            "source": p1, "relationship": "colleague_of", "target": p2
                        })
        return relationships

    def _extract_missing_rels(self, content: str, entities: list, relationships: list, db) -> list:
        """Fallback: extract relationships from content when LLM missed them."""
        import re, sys
        entity_names = {e["name"] for e in entities}
        # Also include existing node names
        for row in db.execute("SELECT name, type FROM nodes"):
            if row["name"] not in entity_names:
                entity_names.add(row["name"])

        existing_rels = {(r["source"], r["relationship"], r["target"]) for r in relationships}

        # Find known orgs mentioned in content
        orgs_in_content = set()
        for row in db.execute("SELECT name FROM nodes WHERE type='organization'"):
            if row["name"] in content:
                orgs_in_content.add(row["name"])
        for e in entities:
            if e["type"] == "organization" and e["name"] in content:
                orgs_in_content.add(e["name"])

        persons_in_content = [e["name"] for e in entities if e["type"] == "person"]

        for person in persons_in_content:
            for org in orgs_in_content:
                if (person, "works_at", org) not in existing_rels:
                    # Pattern: "person" and "org" both in content with "在" nearby
                    pattern = rf'{person}.*?在.*?{org}|{org}.*{person}|{person}.*{org}'
                    if re.search(pattern, content):
                        relationships.append({"source": person, "relationship": "works_at", "target": org})
                        existing_rels.add((person, "works_at", org))

        # Also handle "likes" for skill/concept entities
        user_name = self._get_user_name(db, "demo_user")
        skills = [e["name"] for e in entities if e["type"] in ("skill", "concept")]
        if user_name and skills and "喜欢" in content:
            for skill in skills:
                if (user_name, "likes", skill) not in existing_rels and skill in content:
                    relationships.append({"source": user_name, "relationship": "likes", "target": skill})

        return relationships

    def add_memory(self, content: str, user_id: str = "demo_user") -> dict:
        """Add a new memory, return pipeline trace."""
        db = get_db()
        now = time.time()
        memory_id = str(uuid.uuid4())[:8]

        # Step 1: Extract entities and relations via LLM
        extraction = extract_entities_and_relations(content)
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        # Step 1.5: Post-processing - self-reference resolution
        # Find the user's name from "我叫XX" patterns in past memories
        user_name = self._get_user_name(db, user_id)
        if user_name:
            # Replace "我/自己/本人" in relationship sources/targets with the known user name
            for rel in relationships:
                if rel["source"] in ("我", "自己", "本人"):
                    rel["source"] = user_name
                if rel["target"] in ("我", "自己", "本人"):
                    rel["target"] = user_name
            # Also check if any relationship references "我" as a concept
            # and ensure the user_name node exists
            if not any(e["name"] == user_name for e in entities):
                entities.append({"name": user_name, "type": "person"})

        # Step 2: Check for conflicts with existing memories
        existing = db.execute(
            "SELECT id, content FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()

        # Simple similarity check
        action = "ADD"
        conflict_reason = ""
        for ex in existing:
            common_words = set(content) & set(ex["content"])
            if len(common_words) > min(len(content), len(ex["content"])) * 0.6:
                action = "SKIP"
                conflict_reason = f"与已有记忆 #{ex['id']} 内容高度相似"
                break

        # Step 2.5: Entity dedup - merge "腾讯深圳" into "腾讯" + "深圳"
        entities, relationships = self._dedup_entities(entities, relationships, db)

        # Step 2.6: Fallback relationship extraction from content keywords
        # (Run before colleague inference so both persons' works_at edges exist)
        relationships = self._extract_missing_rels(content, entities, relationships, db)

        # Step 2.7: Infer colleague_of relationships
        relationships = self._infer_colleague(relationships, entities, db, user_id)

        # Step 3: Store memory
        if action != "SKIP":
            db.execute(
                "INSERT INTO memories (id, content, user_id, created_at, last_accessed_at, access_count) VALUES (?,?,?,?,?,?)",
                (memory_id, content, user_id, now, now, 1)
            )

            # Step 4: Store nodes
            node_id_map = {}  # name -> node_id
            for ent in entities:
                name = ent["name"]
                ent_type = ent["type"]
                # Check if node already exists
                existing_node = db.execute(
                    "SELECT id FROM nodes WHERE name=?", (name,)
                ).fetchone()
                if existing_node:
                    node_id_map[name] = existing_node["id"]
                else:
                    node_id = str(uuid.uuid4())[:8]
                    db.execute(
                        "INSERT INTO nodes (id, name, type, properties) VALUES (?,?,?,?)",
                        (node_id, name, ent_type, json.dumps({}))
                    )
                    node_id_map[name] = node_id

            # Also resolve IDs for existing nodes referenced in relationships
            for rel in relationships:
                for name in (rel["source"], rel["target"]):
                    if name not in node_id_map:
                        existing_node = db.execute(
                            "SELECT id FROM nodes WHERE name=?", (name,)
                        ).fetchone()
                        if existing_node:
                            node_id_map[name] = existing_node["id"]

            # Step 5: Store edges
            stored_rels = []
            for rel in relationships:
                src_name = rel["source"]
                tgt_name = rel["target"]
                rel_type = rel["relationship"]
                src_id = node_id_map.get(src_name)
                tgt_id = node_id_map.get(tgt_name)
                if src_id and tgt_id:
                    # Check duplicate edge
                    dup = db.execute(
                        "SELECT id FROM edges WHERE source_id=? AND target_id=? AND relationship=?",
                        (src_id, tgt_id, rel_type)
                    ).fetchone()
                    if not dup:
                        edge_id = str(uuid.uuid4())[:8]
                        db.execute(
                            "INSERT INTO edges (id, source_id, target_id, relationship, memory_id, created_at) VALUES (?,?,?,?,?,?)",
                            (edge_id, src_id, tgt_id, rel_type, memory_id, now)
                        )
                        stored_rels.append(rel)
        else:
            # Still try to extract and store new entities/relations from skipped memory
            node_id_map = {}
            for ent in entities:
                name = ent["name"]
                ent_type = ent["type"]
                existing_node = db.execute(
                    "SELECT id FROM nodes WHERE name=?", (name,)
                ).fetchone()
                if existing_node:
                    node_id_map[name] = existing_node["id"]
                else:
                    node_id = str(uuid.uuid4())[:8]
                    db.execute(
                        "INSERT INTO nodes (id, name, type, properties) VALUES (?,?,?,?)",
                        (node_id, name, ent_type, json.dumps({}))
                    )
                    node_id_map[name] = node_id
            # Also resolve IDs for existing nodes referenced in relationships
            for rel in relationships:
                for name in (rel["source"], rel["target"]):
                    if name not in node_id_map:
                        existing_node = db.execute(
                            "SELECT id FROM nodes WHERE name=?", (name,)
                        ).fetchone()
                        if existing_node:
                            node_id_map[name] = existing_node["id"]
            stored_rels = []
            for rel in relationships:
                src_name = rel["source"]
                tgt_name = rel["target"]
                rel_type = rel["relationship"]
                src_id = node_id_map.get(src_name)
                tgt_id = node_id_map.get(tgt_name)
                if src_id and tgt_id:
                    dup = db.execute(
                        "SELECT id FROM edges WHERE source_id=? AND target_id=? AND relationship=?",
                        (src_id, tgt_id, rel_type)
                    ).fetchone()
                    if not dup:
                        edge_id = str(uuid.uuid4())[:8]
                        db.execute(
                            "INSERT INTO edges (id, source_id, target_id, relationship, memory_id, created_at) VALUES (?,?,?,?,?,?)",
                            (edge_id, src_id, tgt_id, rel_type, None, now)
                        )
                        stored_rels.append(rel)

        db.commit()
        db.close()

        return {
            "memory_id": memory_id if action != "SKIP" else None,
            "action": action,
            "reason": conflict_reason or "新记忆，无冲突" if action == "ADD" else conflict_reason,
            "entities": entities,
            "relationships": stored_rels,
        }

    def search_memory(self, query: str, user_id: str = "demo_user", top_k: int = 5) -> dict:
        """Search memories and generate answer."""
        db = get_db()
        now = time.time()

        # Get all memories for user
        rows = db.execute(
            "SELECT id, content, created_at, last_accessed_at, access_count, initial_score FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()

        memories = []
        for row in rows:
            elapsed_hours = (now - row["created_at"]) / 3600
            retention = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            retention = min(1.0, retention + row["access_count"] * EBBINGHAUS_REINFORCE)
            retention = max(0.0, min(1.0, retention))
            memories.append({
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "last_accessed_at": row["last_accessed_at"],
                "access_count": row["access_count"],
                "retention": round(retention, 4),
            })

        # Get graph context for richer search
        graph_context = self._build_graph_context(db)

        # Use LLM to rank memories with graph context
        llm_results = self._rank_memories_with_llm(query, memories, graph_context)
        results = []
        for r in llm_results[:top_k]:
            mem_id = r.get("memory_id")
            mem = next((m for m in memories if m["id"] == mem_id), None)
            if mem:
                results.append({
                    "memory": mem,
                    "score": r.get("score", 0.5),
                    "reason": r.get("reason", ""),
                })
                # Reinforce accessed memories
                db.execute(
                    "UPDATE memories SET access_count=access_count+1, last_accessed_at=? WHERE id=?",
                    (now, mem_id)
                )

        # Generate answer with graph context
        relevant_memories = [r["memory"] for r in results]
        answer = generate_answer(query, relevant_memories, graph_context)

        # If no results from memory search, try direct graph-based answer
        if not results and graph_context:
            answer = generate_answer(query, [], graph_context)

        db.commit()
        db.close()

        return {
            "query": query,
            "results": results,
            "answer": answer,
            "graph_context": graph_context,
        }

    def _rank_memories_with_llm(self, query: str, memories: list, graph_context: str) -> list:
        """Use LLM to rank memories by relevance, considering graph context."""
        if not memories:
            return []
        try:
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            memory_text = "\n".join([
                f"[{m['id']}] {m['content']}" for m in memories
            ])
            extra = ""
            if graph_context:
                extra = f"\n\n图谱关系上下文：\n{graph_context}\n请也考虑图谱关系来匹配记忆。"
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个精确的记忆检索器。只输出JSON数组。"},
                    {"role": "user", "content": SEARCH_PROMPT.format(
                        query=query, memories=memory_text
                    ) + extra}
                ],
                temperature=0.1,
                max_completion_tokens=2048,
            )
            content = _get_llm_text(response)
            import re
            arr_match = re.search(r'\[.*\]', content, re.DOTALL)
            if arr_match:
                content = arr_match.group()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0]
            return json.loads(content)
        except Exception as e:
            print(f"[LLM rank error] {e}", file=sys.stderr, flush=True)
            return []

    def _build_graph_context(self, db) -> str:
        """Build graph context string for LLM."""
        edges = db.execute(
            """SELECT n1.name as src, e.relationship, n2.name as tgt
               FROM edges e
               JOIN nodes n1 ON e.source_id = n1.id
               JOIN nodes n2 ON e.target_id = n2.id
               ORDER BY e.created_at DESC LIMIT 20"""
        ).fetchall()
        if not edges:
            return ""
        return "\n".join([
            f"{row['src']} --[{row['relationship']}]--> {row['tgt']}" for row in edges
        ])

    def get_stats(self, user_id: str = "demo_user") -> dict:
        """Get memory store statistics."""
        db = get_db()
        now = time.time()
        mem_count = db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id=?", (user_id,)
        ).fetchone()[0]
        node_count = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        # Node type distribution
        type_dist = {}
        for row in db.execute("SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type"):
            type_dist[row["type"]] = row["cnt"]

        # Ebbinghaus scores for all memories
        ebbinghaus = []
        for row in db.execute(
            "SELECT id, content, created_at, last_accessed_at, access_count, initial_score FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ):
            elapsed_hours = (now - row["created_at"]) / 3600
            retention = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            retention = min(1.0, retention + row["access_count"] * EBBINGHAUS_REINFORCE)
            ebbinghaus.append({
                "id": row["id"],
                "content": row["content"],
                "retention": round(max(0.0, min(1.0, retention)), 4),
                "elapsed_hours": round(elapsed_hours, 2),
                "access_count": row["access_count"],
            })

        db.close()
        return {
            "total_memories": mem_count,
            "total_nodes": node_count,
            "total_edges": edge_count,
            "node_type_distribution": type_dist,
            "ebbinghaus_scores": ebbinghaus,
        }

    def get_graph_data(self) -> dict:
        """Get graph data for visualization with resolved names."""
        db = get_db()
        nodes = []
        for row in db.execute("SELECT id, name, type FROM nodes ORDER BY id"):
            nodes.append({"id": row["id"], "name": row["name"], "type": row["type"]})

        # Build name lookup
        name_map = {n["id"]: n["name"] for n in nodes}

        edges = []
        for row in db.execute("SELECT id, source_id, target_id, relationship FROM edges"):
            edges.append({
                "id": row["id"],
                "source": row["source_id"],
                "source_name": name_map.get(row["source_id"], "?"),
                "target": row["target_id"],
                "target_name": name_map.get(row["target_id"], "?"),
                "relationship": row["relationship"],
            })

        db.close()
        return {"nodes": nodes, "edges": edges}

    def get_memories(self, user_id: str = "demo_user") -> list:
        """Get all memories."""
        db = get_db()
        now = time.time()
        memories = []
        for row in db.execute(
            "SELECT id, content, created_at, last_accessed_at, access_count, initial_score FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ):
            elapsed_hours = (now - row["created_at"]) / 3600
            retention = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            retention = min(1.0, retention + row["access_count"] * EBBINGHAUS_REINFORCE)
            memories.append({
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "retention": round(max(0.0, min(1.0, retention)), 4),
                "access_count": row["access_count"],
            })
        db.close()
        return memories

    def clear_all(self, user_id: str = "demo_user"):
        """Clear all data for a user."""
        db = get_db()
        db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM edges")
        db.execute("DELETE FROM nodes")
        db.commit()
        db.close()


store = MemoryStore()


# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__, template_folder=TEMPLATE_DIR)


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "memory_frontend.html")


# ============================================================================
# LLM Intent Classification (primary) + Regex (fallback)
# ============================================================================

CLASSIFY_PROMPT = """判断以下用户输入是要存储新记忆(ADD)还是查询已有记忆(QUERY)。

示例：
- "我的同事李四也在腾讯" → ADD
- "我的同事有哪些" → QUERY
- "我喜欢喝咖啡" → ADD
- "我喜欢什么" → QUERY
- "我叫张三，在腾讯工作" → ADD
- "我是谁" → QUERY

用户输入：{text}

直接回答 ADD 或 QUERY，不要解释。只输出一个词。"""


def classify_intent_llm(text: str) -> Optional[dict]:
    """Use LLM to classify user intent. Returns {"action": "ADD/QUERY", "reason": "..."} or None."""
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你只输出ADD或QUERY，不要输出任何其他内容。"},
                {"role": "user", "content": CLASSIFY_PROMPT.format(text=text)},
            ],
            temperature=0.0,
            max_completion_tokens=1024,
        )
        content = _get_llm_text(response).strip().upper()
        # Extract ADD or QUERY from the response
        if content.startswith('ADD'):
            return {"action": "ADD", "reason": "LLM classified as ADD"}
        elif content.startswith('QUERY'):
            return {"action": "QUERY", "reason": "LLM classified as QUERY"}
        # Fallback: check if either word appears
        if 'ADD' in content:
            return {"action": "ADD", "reason": "LLM classified as ADD"}
        elif 'QUERY' in content:
            return {"action": "QUERY", "reason": "LLM classified as QUERY"}
        return None
    except Exception as e:
        print(f"[LLM classify error] {e}", file=sys.stderr, flush=True)
        return None


def classify_intent_regex(text: str) -> dict:
    """Regex-based fallback classification."""
    has_qmark = bool(re.search(r'[？?]', text))
    has_stmt_hint = bool(re.search(r'也在|也喜欢|也认识|一起|都[是在]', text))
    starts_my = bool(re.match(r'^(我|我的|咱|咱们)(的?|们?)(同事|朋友|认识|有哪些|谁|什么|了解|知道|记得|之前|上次)', text))
    starts_q = bool(re.match(r'^(谁|什么|哪里|哪个|哪些|有多少|有哪些|你喜欢|我喜欢|我有什么|帮我查)', text))

    is_query = has_qmark or (starts_my and not has_stmt_hint) or starts_q
    reason = "regex: question mark" if has_qmark else \
             "regex: query pattern" if is_query else "regex: default statement"
    return {"action": "QUERY" if is_query else "ADD", "reason": reason}


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify user input as ADD (store memory) or QUERY (search memory).
    Strategy: LLM first, regex fallback."""
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    # Primary: LLM classification
    result = classify_intent_llm(text)
    method = "llm"
    if result and result.get("action") in ("ADD", "QUERY"):
        return jsonify({**result, "method": "llm"})

    # Fallback: regex
    result = classify_intent_regex(text)
    return jsonify({**result, "method": "regex"})


@app.route("/api/memory/add", methods=["POST"])
def api_add_memory():
    data = request.json
    content = data.get("content", "").strip()
    user_id = data.get("user_id", "demo_user")
    if not content:
        return jsonify({"error": "content is required"}), 400
    result = store.add_memory(content, user_id)
    return jsonify(result)


@app.route("/api/memory/search", methods=["POST"])
def api_search_memory():
    data = request.json
    query = data.get("query") or data.get("content", "").strip()
    user_id = data.get("user_id", "demo_user")
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = store.search_memory(query, user_id)
    return jsonify(result)


@app.route("/api/memory/list", methods=["GET"])
def api_list_memories():
    user_id = request.args.get("user_id", "demo_user")
    return jsonify(store.get_memories(user_id))


@app.route("/api/stats", methods=["GET"])
def api_stats():
    user_id = request.args.get("user_id", "demo_user")
    return jsonify(store.get_stats(user_id))


@app.route("/api/graph", methods=["GET"])
def api_graph():
    return jsonify(store.get_graph_data())


@app.route("/api/clear", methods=["POST"])
def api_clear():
    data = request.json or {}
    user_id = data.get("user_id", "demo_user")
    store.clear_all(user_id)
    return jsonify({"status": "cleared"})


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HugeGraph Memory Demo Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--reset", action="store_true", help="Reset database")
    args = parser.parse_args()

    if args.reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[INFO] Database reset.")

    init_db()
    print(f"[INFO] HugeGraph Memory Demo Server")
    print(f"[INFO] Database: {DB_PATH}")
    print(f"[INFO] LLM: {LLM_BASE_URL} ({LLM_MODEL})")
    print(f"[INFO] API Key: {'set' if LLM_API_KEY else 'NOT SET - set LLM_API_KEY env var!'}")
    print(f"[INFO] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
