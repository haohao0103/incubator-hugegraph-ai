"""
HugeGraph Memory Demo Server
=============================
Flask server — 100% HugeGraph Server as storage (REST API, no Gremlin, no SQLite).

Architecture:
  - memory_node vertices   → extracted entities (person/org/location/skill/concept)
  - memory_item vertices   → raw memory text + Ebbinghaus metadata
  - memory_* edges          → relationships between entities
  - memory_mentions edges   → memory_item → memory_node (which memory mentions which entity)

Usage:
    python demo/memory_server.py [--port 8765]
"""

import json
import math
import time
import uuid
import os
import re
import sys
import argparse
import urllib.parse
from typing import Optional

import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# ============================================================================
# Config — LLM (bdi-gpt-stg)
# ============================================================================

LLM_BASE_URL = "http://bdi-gpt-stg.myhll.cn"
LLM_MODEL = "doubao-seed-1-8-251228"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-N2ozBL2daXuMQVOOHc5dLtv5gGrjBnhhz6j4gMjUYccjtjWB")
LLM_HEADERS = {"hller": os.environ.get("LLM_HLLER", "alan578.zhao")}
LLM_TIMEOUT = (5, 45)  # nginx 60s 全局超时，留安全余量

# ============================================================================
# Config — HugeGraph Server (唯一存储)
# ============================================================================

HG_BASE = os.environ.get("HG_BASE", "http://127.0.0.1:8080")
HG_GRAPH = os.environ.get("HG_GRAPH", "memory")
HG_SCHEMA = f"{HG_BASE}/graphs/{HG_GRAPH}/schema"
HG_API = f"{HG_BASE}/graphs/{HG_GRAPH}/graph"

EBBINGHAUS_K = 0.821
EBBINGHAUS_REINFORCE = 0.3

KNOWN_EDGE_LABELS = [
    "works_at", "lives_in", "likes", "colleague_of", "friend_of",
    "part_of", "located_in", "knows", "has_skill", "member_of",
    "related_to", "studies_at", "manages", "owns", "uses",
    "interested_in", "founded", "located_at", "partner_of", "teaches",
]


# ============================================================================
# HugeGraph REST helpers
# ============================================================================

def hg(path, method="GET", params=None, data=None):
    """Unified HugeGraph REST call. Returns (status_code, json_or_None)."""
    url = f"{path}" if path.startswith("http") else f"{HG_API}{path}"
    try:
        if method == "GET":
            r = http_requests.get(url, params=params, timeout=10)
        elif method == "POST":
            r = http_requests.post(url, json=data, timeout=10)
        elif method == "DELETE":
            r = http_requests.delete(url, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        body = r.json() if r.text else None
        return r.status_code, body
    except http_requests.exceptions.HTTPError:
        raise
    except Exception as e:
        print(f"[HG] {method} {url}: {e}", file=sys.stderr, flush=True)
        return 0, None


def hg_schema(method, path, data=None):
    url = f"{HG_SCHEMA}{path}"
    if method == "GET":
        r = http_requests.get(url, timeout=10)
    else:
        r = http_requests.post(url, json=data, timeout=10)
    return r.status_code, (r.json() if r.text else None)


def _qid(vid):
    """Quote vertex ID for HugeGraph DELETE (String IDs need \"id\" format)."""
    return urllib.parse.quote(f'"{vid}"', safe='')


# ============================================================================
# Schema init (idempotent)
# ============================================================================

def init_schema():
    print("[SCHEMA] Checking HugeGraph schema...", file=sys.stderr, flush=True)

    # Property keys
    for name, dtype in [
        ("m_name", "TEXT"), ("m_type", "TEXT"), ("m_user_id", "TEXT"),
        ("m_content", "TEXT"), ("m_created_at", "DOUBLE"),
        ("m_access_count", "INT"), ("m_initial_score", "DOUBLE"),
    ]:
        _, existing = hg_schema("GET", "/propertykeys")
        names = [p["name"] for p in (existing or {}).get("propertykeys", [])]
        if name not in names:
            hg_schema("POST", "/propertykeys", {"name": name, "data_type": dtype, "cardinality": "SINGLE"})

    # Vertex labels
    _, vls = hg_schema("GET", "/vertexlabels")
    vl_names = [v["name"] for v in (vls or {}).get("vertexlabels", [])]

    if "memory_node" not in vl_names:
        hg_schema("POST", "/vertexlabels", {
            "name": "memory_node", "id_strategy": "CUSTOMIZE_STRING",
            "properties": ["m_name", "m_type", "m_user_id"],
            "nullable_keys": ["m_type", "m_user_id"],
        })

    if "memory_item" not in vl_names:
        hg_schema("POST", "/vertexlabels", {
            "name": "memory_item", "id_strategy": "CUSTOMIZE_STRING",
            "properties": ["m_content", "m_user_id", "m_created_at", "m_access_count", "m_initial_score"],
            "nullable_keys": ["m_user_id", "m_content", "m_created_at", "m_access_count", "m_initial_score"],
        })

    # Edge labels
    _, els = hg_schema("GET", "/edgelabels")
    el_names = [e["name"] for e in (els or {}).get("edgelabels", [])]

    for rel in KNOWN_EDGE_LABELS:
        label = f"memory_{rel}"
        if label not in el_names:
            hg_schema("POST", "/edgelabels", {
                "name": label, "source_label": "memory_node", "target_label": "memory_node",
                "frequency": "SINGLE", "nullable_keys": [], "properties": [],
            })

    if "memory_mentions" not in el_names:
        hg_schema("POST", "/edgelabels", {
            "name": "memory_mentions", "source_label": "memory_item", "target_label": "memory_node",
            "frequency": "SINGLE", "nullable_keys": [], "properties": [],
        })

    # Secondary indexes (required for property-based queries)
    _, idxs = hg_schema("GET", "/indexlabels")
    idx_names = [i["name"] for i in (idxs or {}).get("indexlabels", [])]

    for idx_name, base_val, fields in [
        ("memory_node_by_name", "memory_node", ["m_name"]),
        ("memory_node_by_type", "memory_node", ["m_type"]),
        ("memory_item_by_user", "memory_item", ["m_user_id"]),
    ]:
        if idx_name not in idx_names:
            hg_schema("POST", "/indexlabels", {
                "name": idx_name, "base_value": base_val,
                "base_type": "VERTEX_LABEL", "index_type": "SECONDARY",
                "fields": fields,
            })

    print("[SCHEMA] Ready.", file=sys.stderr, flush=True)


# ============================================================================
# Graph CRUD
# ============================================================================

def find_node(name: str) -> Optional[dict]:
    """Find memory_node by m_name."""
    code, body = hg("/vertices", params={"label": "memory_node", "properties": json.dumps({"m_name": name})})
    if code == 200 and body:
        verts = body.get("vertices", [])
        return verts[0] if verts else None
    return None


def get_or_create_node(name: str, node_type: str, user_id="demo_user") -> dict:
    existing = find_node(name)
    if existing:
        return existing
    vid = f"n_{uuid.uuid4().hex[:12]}"
    code, body = hg("/vertices", "POST", data={
        "label": "memory_node", "id": vid,
        "properties": {"m_name": name, "m_type": node_type, "m_user_id": user_id},
    })
    if code in (200, 201) and body:
        return body
    # Race condition — try again
    existing = find_node(name)
    if existing:
        return existing
    raise RuntimeError(f"Failed to create node {name}: {code}")


def create_memory_item(mid: str, content: str, user_id: str, created_at: float) -> dict:
    code, body = hg("/vertices", "POST", data={
        "label": "memory_item", "id": f"mem_{mid}",
        "properties": {
            "m_content": content, "m_user_id": user_id,
            "m_created_at": created_at, "m_access_count": 1, "m_initial_score": 1.0,
        },
    })
    return body or {}


def ensure_edge_label(rel_type: str):
    label = f"memory_{rel_type}"
    _, els = hg_schema("GET", "/edgelabels")
    names = [e["name"] for e in (els or {}).get("edgelabels", [])]
    if label not in names:
        hg_schema("POST", "/edgelabels", {
            "name": label, "source_label": "memory_node", "target_label": "memory_node",
            "frequency": "SINGLE", "nullable_keys": [], "properties": [],
        })


def create_edge(src_id: str, tgt_id: str, rel_type: str) -> Optional[dict]:
    ensure_edge_label(rel_type)
    code, body = hg("/edges", "POST", data={
        "label": f"memory_{rel_type}", "outV": src_id, "inV": tgt_id, "properties": {},
    })
    if code in (200, 201):
        return body
    if code == 400:
        return None  # duplicate
    return None


def get_all_nodes() -> list:
    code, body = hg("/vertices", params={"label": "memory_node", "limit": 500})
    return (body or {}).get("vertices", [])


def get_all_memory_items() -> list:
    code, body = hg("/vertices", params={"label": "memory_item", "limit": 500})
    return (body or {}).get("vertices", [])


def get_all_edges() -> list:
    """Collect all memory_* edges across all known labels."""
    result = []
    # Get all edge labels from schema
    _, els = hg_schema("GET", "/edgelabels")
    for el in (els or {}).get("edgelabels", []):
        name = el["name"]
        if name.startswith("memory_") and name != "memory_mentions":
            code, body = hg("/edges", params={"label": name, "limit": 500})
            if code == 200 and body:
                result.extend(body.get("edges", []))
    return result


def update_vertex_property(vid: str, label: str, props: dict):
    """Update properties on an existing vertex (PUT)."""
    encoded = _qid(vid)
    url = f"{HG_API}/vertices/{encoded}"
    data = {"label": label, "properties": props}
    try:
        r = http_requests.put(url, json=data, timeout=10)
        return r.status_code in (200, 201)
    except Exception as e:
        print(f"[HG] update vertex {vid}: {e}", file=sys.stderr)
        return False


def delete_vertex(vid: str) -> bool:
    code, _ = hg(f"/vertices/{_qid(vid)}", "DELETE")
    return code in (200, 204)


def clear_all_data():
    """Delete all memory_item then memory_node vertices."""
    for label in ["memory_item", "memory_node"]:
        code, body = hg("/vertices", params={"label": label, "limit": 1000})
        if code == 200 and body:
            for v in body.get("vertices", []):
                delete_vertex(v["id"])


# ============================================================================
# LLM helpers (bdi-gpt-stg doubao)
# ============================================================================

EXTRACT_SYSTEM = "你是一个精确的知识图谱信息提取器。只输出JSON，不要解释。"
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

SEARCH_PROMPT = """你是一个记忆检索器。用户提出了一个问题，请从以下记忆列表中找出最相关的记忆。

用户问题：{query}

记忆列表：
{memories}

请返回最相关的记忆ID列表，按相关性排序，每条附上相关性分数(0-1)。
严格按JSON格式输出：
[{{"memory_id": "ID", "score": 0.95, "reason": "简要原因"}}]

如果没有相关记忆，返回空数组。"""

ANSWER_PROMPT = """你是一个拥有记忆能力的AI助手。根据用户的记忆信息回答问题。

用户问题：{query}

相关记忆：
{memories}

图谱关系：
{graph_context}

请用简洁自然的中文回答。如果记忆中没有足够信息，诚实说明。"""


def _call_llm(model: str, system: str, user_msg: str,
              temperature=0.1, max_tokens=2048) -> str:
    # bdi-gpt-stg 统一用 /v1/chat/completions，model 字段指定模型
    client = OpenAI(base_url=f"{LLM_BASE_URL}/v1", api_key=LLM_API_KEY,
                    default_headers=LLM_HEADERS, timeout=LLM_TIMEOUT)
    resp = client.chat.completions.create(
        model=model, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ], temperature=temperature, max_completion_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    return (msg.content or getattr(msg, "reasoning_content", "") or "").strip()


def _parse_json(content: str) -> dict:
    if not content:
        return {"entities": [], "relationships": []}
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
        content = content.rsplit("```", 1)[0]
    content = content.strip()
    try:
        return _normalize(json.loads(content))
    except (json.JSONDecodeError, TypeError):
        pass
    for pat in [
        r'\{[^{}]*"entities"\s*:\s*\[[^\]]*\][^{}]*"relationships"\s*:\s*\[[^\]]*\][^{}]*\}',
        r'\{[^{}]*"entities"\s*:\s*\[[^\]]*\][^{}]*\}',
    ]:
        for m in reversed(re.findall(pat, content, re.DOTALL)):
            try:
                return _normalize(json.loads(m))
            except (json.JSONDecodeError, TypeError):
                continue
    # Regex fallback
    ents = [{"name": m.group(1), "type": m.group(2).lower()}
            for m in re.finditer(r'"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"', content)]
    rels = []
    for pat in [
        r'"source"\s*:\s*"([^"]+)"\s*,\s*"relationship"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"',
        r'"subject"\s*:\s*"([^"]+)"\s*,\s*"(?:relation|relationship)"\s*:\s*"([^"]+)"\s*,\s*"(?:target|object)"\s*:\s*"([^"]+)"',
    ]:
        for m in re.finditer(pat, content):
            rels.append({"source": m.group(1), "relationship": m.group(2), "target": m.group(3)})
    return {"entities": ents, "relationships": rels}


_TMAP = {
    "person": "person", "people": "person", "人": "person", "人物": "person",
    "organization": "organization", "org": "organization", "公司": "organization", "机构": "organization",
    "location": "location", "地点": "location", "地方": "location", "城市": "location",
    "skill": "skill", "技能": "skill", "爱好": "skill",
    "beverage": "concept", "drink": "concept", "sport": "skill",
}


def _normalize(raw: dict) -> dict:
    ents = []
    for e in raw.get("entities", []):
        name = e.get("name") or e.get("entity") or e.get("value", "")
        etype = (e.get("type") or e.get("category") or "concept").lower()
        etype = _TMAP.get(etype, etype)
        if name in ("我", "自己", "本人", "您"):
            continue
        ents.append({"name": name, "type": etype})
    rels = []
    for r in raw.get("relationships", []):
        s = r.get("source") or r.get("subject") or r.get("from") or ""
        t = r.get("target") or r.get("object") or r.get("to") or ""
        rel = r.get("relationship") or r.get("relation") or r.get("predicate") or ""
        if s and t and rel:
            rels.append({"source": s, "relationship": rel, "target": t})
    return {"entities": ents, "relationships": rels}


def extract(text: str) -> dict:
    try:
        content = _call_llm(LLM_MODEL, EXTRACT_SYSTEM, f"{EXTRACT_PROMPT}\n\n用户输入：{text}")
        result = _parse_json(content)
        print(f"[EXTRACT] {len(result['entities'])} entities, {len(result['relationships'])} rels", file=sys.stderr, flush=True)
        return result
    except Exception as e:
        print(f"[LLM extract error] {e}", file=sys.stderr, flush=True)
        return {"entities": [], "relationships": []}


def classify_intent(text: str) -> str:
    """Returns 'ADD' or 'QUERY'."""
    # LLM first
    try:
        content = _call_llm(LLM_MODEL, "你只输出ADD或QUERY，不要输出任何其他内容。",
                            CLASSIFY_PROMPT.format(text=text), max_tokens=50)
        content = content.strip().upper()
        if "ADD" in content:
            return "ADD"
        if "QUERY" in content:
            return "QUERY"
    except Exception as e:
        print(f"[LLM classify error] {e}", file=sys.stderr, flush=True)
    # Regex fallback
    has_q = bool(re.search(r'[？?]', text))
    starts_q = bool(re.match(r'^(谁|什么|哪里|哪个|哪些|有多少|有哪些|帮我查)', text))
    return "QUERY" if (has_q or starts_q) else "ADD"


# ============================================================================
# MemoryStore (100% HugeGraph)
# ============================================================================

class MemoryStore:

    # --- helpers ---

    def _all_memories(self) -> list:
        """Return list of {id, content, created_at, access_count, initial_score}."""
        items = get_all_memory_items()
        result = []
        for v in items:
            p = v.get("properties", {})
            mid = v["id"].replace("mem_", "", 1) if v["id"].startswith("mem_") else v["id"]
            result.append({
                "id": mid,
                "content": p.get("m_content", ""),
                "created_at": p.get("m_created_at", 0),
                "access_count": p.get("m_access_count", 0),
                "initial_score": p.get("m_initial_score", 1.0),
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    def _get_user_name(self) -> str:
        """Extract user name from '我叫XX' patterns in stored memories."""
        for m in self._all_memories():
            hit = re.search(r'(?:我叫|我是)\s*([\u4e00-\u9fa5]{2,4})', m["content"])
            if hit:
                return hit.group(1)
        return ""

    def _ebbinghaus(self, mem: dict, now: float) -> float:
        elapsed_h = (now - mem["created_at"]) / 3600
        r = mem["initial_score"] * math.exp(-EBBINGHAUS_K * max(0, elapsed_h))
        r = min(1.0, r + mem["access_count"] * EBBINGHAUS_REINFORCE)
        return max(0.0, min(1.0, r))

    def _build_graph_context(self) -> str:
        edges = get_all_edges()
        if not edges:
            return ""
        nodes = get_all_nodes()
        name_map = {n["id"]: n.get("properties", {}).get("m_name", "?") for n in nodes}
        lines = []
        for e in edges[:20]:
            src = name_map.get(e.get("outV", ""), "?")
            tgt = name_map.get(e.get("inV", ""), "?")
            rel = e.get("label", "").replace("memory_", "")
            lines.append(f"{src} --[{rel}]--> {tgt}")
        return "\n".join(lines)

    # --- core ops ---

    def add_memory(self, content: str, user_id="demo_user") -> dict:
        now = time.time()
        memory_id = str(uuid.uuid4())[:8]

        # 1. LLM extract
        ext = extract(content)
        entities = ext.get("entities", [])
        relationships = ext.get("relationships", [])

        # 2. Self-reference resolution
        user_name = self._get_user_name()
        if user_name:
            for rel in relationships:
                if rel["source"] in ("我", "自己", "本人"):
                    rel["source"] = user_name
                if rel["target"] in ("我", "自己", "本人"):
                    rel["target"] = user_name
            if not any(e["name"] == user_name for e in entities):
                entities.append({"name": user_name, "type": "person"})

        # 3. Conflict check (against existing memory_items in HugeGraph)
        existing = self._all_memories()
        action = "ADD"
        conflict_reason = ""
        for ex in existing:
            common = set(content) & set(ex["content"])
            if len(common) > min(len(content), len(ex["content"])) * 0.6:
                action = "SKIP"
                conflict_reason = f"与已有记忆 #{ex['id']} 内容高度相似"
                break

        # 4. Entity dedup against existing HugeGraph nodes
        all_nodes = get_all_nodes()
        existing_names = {n.get("properties", {}).get("m_name", "") for n in all_nodes}
        merged = {}
        new_entities = []
        for ent in entities:
            nm = ent["name"]
            best = None
            for en in existing_names:
                if en in nm and en != nm:
                    best = en
                    rem = nm.replace(en, "").strip()
                    if rem and rem not in existing_names and re.match(r'^[\u4e00-\u9fa5]{2,3}$', rem):
                        new_entities.append({"name": rem, "type": "location"})
                    break
            if best:
                merged[nm] = best
            elif nm not in existing_names:
                new_entities.append(ent)
        entities = new_entities
        for rel in relationships:
            if rel["source"] in merged:
                rel["source"] = merged[rel["source"]]
            if rel["target"] in merged:
                rel["target"] = merged[rel["target"]]

        # 5. Fallback rel extraction
        entity_names = {e["name"] for e in entities} | existing_names
        existing_rels = {(r["source"], r["relationship"], r["target"]) for r in relationships}
        orgs_in_text = set()
        for n in all_nodes:
            p = n.get("properties", {})
            if p.get("m_type") == "organization" and p.get("m_name", "") in content:
                orgs_in_text.add(p["m_name"])
        for e in entities:
            if e["type"] == "organization" and e["name"] in content:
                orgs_in_text.add(e["name"])
        for person in [e["name"] for e in entities if e["type"] == "person"]:
            for org in orgs_in_text:
                if (person, "works_at", org) not in existing_rels:
                    if re.search(rf'{person}.*?在.*?{org}|{org}.*{person}|{person}.*{org}', content):
                        relationships.append({"source": person, "relationship": "works_at", "target": org})
                        existing_rels.add((person, "works_at", org))
        skills = [e["name"] for e in entities if e["type"] in ("skill", "concept")]
        if user_name and skills and "喜欢" in content:
            for sk in skills:
                if (user_name, "likes", sk) not in existing_rels and sk in content:
                    relationships.append({"source": user_name, "relationship": "likes", "target": sk})

        # 6. Colleague inference
        person_names = {e["name"] for e in entities if e["type"] == "person"}
        for rel in relationships:
            if rel["relationship"] == "works_at":
                person_names.add(rel["source"])
        all_edges = get_all_edges()
        all_rels = list(relationships)
        for edge in all_edges:
            if edge.get("label") == "memory_works_at":
                src_n = find_node_by_id(edge.get("outV", ""))
                tgt_n = find_node_by_id(edge.get("inV", ""))
                if src_n and tgt_n:
                    sn = src_n.get("properties", {}).get("m_name", "")
                    tn = tgt_n.get("properties", {}).get("m_name", "")
                    if sn and tn:
                        all_rels.append({"source": sn, "relationship": "works_at", "target": tn})
                        if src_n.get("properties", {}).get("m_type") == "person":
                            person_names.add(sn)
        person_orgs = {}
        for rel in all_rels:
            if rel["relationship"] == "works_at" and rel["source"] in person_names:
                person_orgs.setdefault(rel["source"], set()).add(rel["target"])
        persons = list(person_names)
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                if person_orgs.get(p1, set()) & person_orgs.get(p2, set()):
                    dup = any(r["relationship"] == "colleague_of" and
                              set([r["source"], r["target"]]) == {p1, p2} for r in relationships)
                    if not dup:
                        relationships.append({"source": p1, "relationship": "colleague_of", "target": p2})

        # 7. Store memory_item in HugeGraph
        if action != "SKIP":
            try:
                create_memory_item(memory_id, content, user_id, now)
            except Exception as e:
                print(f"[HG] memory_item create failed: {e}", file=sys.stderr)

        # 8. Create/update entity nodes
        node_id_map = {}
        for ent in entities:
            try:
                v = get_or_create_node(ent["name"], ent["type"], user_id)
                node_id_map[ent["name"]] = v["id"]
            except Exception as e:
                print(f"[HG] node {ent['name']}: {e}", file=sys.stderr)

        # Resolve IDs for existing nodes referenced in rels
        for rel in relationships:
            for nm in (rel["source"], rel["target"]):
                if nm not in node_id_map:
                    n = find_node(nm)
                    if n:
                        node_id_map[nm] = n["id"]

        # 9. Create edges
        stored_rels = []
        for rel in relationships:
            sid = node_id_map.get(rel["source"])
            tid = node_id_map.get(rel["target"])
            if sid and tid:
                r = create_edge(sid, tid, rel["relationship"])
                if r:
                    stored_rels.append(rel)

        # 10. memory_mentions edges
        if action != "SKIP":
            for nid in node_id_map.values():
                try:
                    code, _ = hg("/edges", "POST", data={
                        "label": "memory_mentions",
                        "outV": f"mem_{memory_id}", "inV": nid, "properties": {},
                    })
                except Exception:
                    pass

        return {
            "memory_id": memory_id if action != "SKIP" else None,
            "action": action,
            "reason": conflict_reason or "新记忆，无冲突" if action == "ADD" else conflict_reason,
            "entities": entities,
            "relationships": stored_rels,
        }

    def search_memory(self, query: str, user_id="demo_user", top_k=5) -> dict:
        now = time.time()
        memories = self._all_memories()

        # Add retention scores
        for m in memories:
            m["retention"] = round(self._ebbinghaus(m, now), 4)

        graph_ctx = self._build_graph_context()

        # LLM rank
        llm_results = []
        if memories:
            try:
                mem_text = "\n".join([f"[{m['id']}] {m['content']}" for m in memories])
                extra = f"\n\n图谱关系上下文：\n{graph_ctx}\n请也考虑图谱关系来匹配记忆。" if graph_ctx else ""
                content = _call_llm(LLM_MODEL, "你是一个精确的记忆检索器。只输出JSON数组。",
                                    SEARCH_PROMPT.format(query=query, memories=mem_text) + extra)
                arr = re.search(r'\[.*\]', content, re.DOTALL)
                if arr:
                    content = arr.group()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                    content = content.rsplit("```", 1)[0]
                llm_results = json.loads(content)
            except Exception as e:
                print(f"[LLM rank error] {e}", file=sys.stderr, flush=True)

        results = []
        for r in llm_results[:top_k]:
            mid = r.get("memory_id")
            mem = next((m for m in memories if m["id"] == mid), None)
            if mem:
                results.append({"memory": mem, "score": r.get("score", 0.5), "reason": r.get("reason", "")})
                # Reinforce access count in HugeGraph
                new_count = mem["access_count"] + 1
                update_vertex_property(f"mem_{mid}", "memory_item",
                                       {"m_access_count": new_count})

        relevant = [r["memory"] for r in results]
        answer = generate_answer(query, relevant, graph_ctx)
        if not results and graph_ctx:
            answer = generate_answer(query, [], graph_ctx)

        return {"query": query, "results": results, "answer": answer, "graph_context": graph_ctx}

    def get_stats(self, user_id="demo_user") -> dict:
        now = time.time()
        memories = self._all_memories()
        nodes = get_all_nodes()
        edges = get_all_edges()

        type_dist = {}
        for n in nodes:
            t = n.get("properties", {}).get("m_type", "unknown")
            type_dist[t] = type_dist.get(t, 0) + 1

        ebbinghaus = []
        for m in memories:
            elapsed_h = (now - m["created_at"]) / 3600
            retention = self._ebbinghaus(m, now)
            ebbinghaus.append({
                "id": m["id"], "content": m["content"],
                "retention": round(retention, 4),
                "elapsed_hours": round(elapsed_h, 2),
                "access_count": m["access_count"],
            })

        return {
            "total_memories": len(memories),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_type_distribution": type_dist,
            "ebbinghaus_scores": ebbinghaus,
        }

    def get_graph_data(self) -> dict:
        nodes_raw = get_all_nodes()
        nodes = [{"id": n["id"],
                   "name": n.get("properties", {}).get("m_name", "?"),
                   "type": n.get("properties", {}).get("m_type", "unknown")}
                 for n in nodes_raw]
        name_map = {n["id"]: n["name"] for n in nodes}
        edges_raw = get_all_edges()
        edges = [{"id": e.get("id", ""),
                   "source": e.get("outV", ""),
                   "source_name": name_map.get(e.get("outV", ""), "?"),
                   "target": e.get("inV", ""),
                   "target_name": name_map.get(e.get("inV", ""), "?"),
                   "relationship": e.get("label", "").replace("memory_", "")}
                 for e in edges_raw
                 if e.get("outV", "") in name_map or e.get("inV", "") in name_map]
        return {"nodes": nodes, "edges": edges}

    def get_memories(self, user_id="demo_user") -> list:
        now = time.time()
        result = []
        for m in self._all_memories():
            result.append({
                "id": m["id"], "content": m["content"],
                "created_at": m["created_at"],
                "retention": round(self._ebbinghaus(m, now), 4),
                "access_count": m["access_count"],
            })
        return result

    def clear_all(self, user_id="demo_user"):
        clear_all_data()


def find_node_by_id(vid: str) -> Optional[dict]:
    """Fetch a single vertex by ID (needs quoted format for String IDs)."""
    try:
        code, body = hg(f"/vertices/{_qid(vid)}")
        if code == 200:
            return body
    except Exception:
        pass
    return None


def generate_answer(query: str, memories: list, graph_ctx: str = "") -> str:
    try:
        mem_text = "\n".join([f"- {m['content']}" for m in memories])
        content = _call_llm(LLM_MODEL,
                            "你是HugeGraph Memory助手，基于用户的记忆回答问题。回答要简洁。",
                            ANSWER_PROMPT.format(query=query, memories=mem_text,
                                                 graph_context=graph_ctx or "无"),
                            temperature=0.3)
        return content or "无法生成回答。"
    except Exception as e:
        print(f"[LLM answer error] {e}", file=sys.stderr, flush=True)
        return f"生成回答时出错: {e}"


store = MemoryStore()

# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "memory_frontend.html")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    action = classify_intent(text)
    return jsonify({"action": action, "method": "llm+regex"})


@app.route("/api/memory/add", methods=["POST"])
def api_add():
    data = request.json or {}
    content = data.get("content", "").strip()
    user_id = data.get("user_id", "demo_user")
    if not content:
        return jsonify({"error": "content is required"}), 400
    return jsonify(store.add_memory(content, user_id))


@app.route("/api/memory/search", methods=["POST"])
def api_search():
    data = request.json or {}
    query = data.get("query") or data.get("content", "").strip()
    user_id = data.get("user_id", "demo_user")
    if not query:
        return jsonify({"error": "query is required"}), 400
    return jsonify(store.search_memory(query, user_id))


@app.route("/api/memory/list", methods=["GET"])
def api_list():
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
    store.clear_all(data.get("user_id", "demo_user"))
    return jsonify({"status": "cleared"})


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HugeGraph Memory Demo Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    init_schema()

    print(f"[INFO] HugeGraph Memory Demo Server")
    print(f"[INFO] Storage: HugeGraph Server @ {HG_BASE}/graphs/{HG_GRAPH}")
    print(f"[INFO] LLM: {LLM_BASE_URL} ({LLM_MODEL})")
    print(f"[INFO] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
