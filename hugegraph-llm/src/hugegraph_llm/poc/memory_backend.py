"""
HugeGraph Memory Backend — Production-grade AI Memory Server
==========================================================
Architecture (aligned with PowerMem v1.1.2):
  Graph Storage: HugeGraph 1.7.0 (via pyhugegraph-python-client)
  Vector Index:  FAISS (memory content embedding → semantic search)
  LLM Engine:    MiMo v2.5 Pro API (entity extract / rank / generate)

Storage mapping vs PowerMem SQLite:
  memories table → FAISS index + SQLite metadata (Ebbinghaus scores)
  nodes table    → HugeGraph Vertices (person/organization/location/skill/concept)
  edges table    → HugeGraph Edges (works_at/lives_in/likes/colleague_of/friend_of)

Pipeline alignment:
  ADD:  LLMExtract→SelfResolve→ConflictDetect→Dedup→RelComplete→ColleagueInfer→Store(7步)
  QUERY: Classify→Ebbinghaus+VectorSearch→GraphContext→LLMAnswer(4步)

Usage:
    python memory_backend.py --port 8765  # standalone server
    # or import as module:
    from memory_backend import MemoryPipelineBackend, create_app
"""

import json
import math
import os
import re
import sys
import time
import uuid
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ============================================================================
# Config
# ============================================================================

HUGEGRAPH_URL = os.environ.get("HUGEGRAPH_URL", "http://127.0.0.1:8080")
HUGEGRAPH_USER = os.environ.get("HUGEGRAPH_USER", "admin")
HUGEGRAPH_PASS = os.environ.get("HUGEGRAPH_PASS", "admin")
HUGEGRAPH_GRAPH = os.environ.get("HUGEGRAPH_GRAPH", "hugegraph")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "mimo-v2.5-pro")
LLM_API_KEY = os.environ.get(
    "LLM_API_KEY",
    "sk-cs5kqi80f6upqy2e3k3xi39jtizhpgf6dkdd3j9ysoupfw7p",
)

# Ebbinghaus constants (same as PowerMem)
EBBINGHAUS_K = 0.821
EBBINGHAUS_REINFORCE = 0.3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_backend.db")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_faiss.index")

# Vertex / Edge labels for Memory graph schema
VERTEX_LABELS = ["person", "organization", "location", "skill", "concept"]
# Edge labels are now created DYNAMICALLY — no hardcoding needed.
# EDGE_LABELS is only used as LLM prompt guidance, not schema enforcement.
EDGE_LABELS = ["works_at", "lives_in", "likes", "colleague_of", "friend_of", "part_of", "located_in",
               "based_in", "manages", "attends", "founded", "owns", "member_of", "studies_at"]

# ============================================================================
# LLM Prompts (aligned with PowerMem memory_server.py)
# ============================================================================

EXTRACT_PROMPT = """你是一个知识图谱实体和关系抽取器。从用户的输入文本中提取实体和关系。

规则：
1. 实体类型：person(人名)、organization(组织机构)、location(地点)、skill(技能/爱好)、concept(概念/事物)
   如果有明确的不属于上述类型的实体，可以自定义类型（如event、product、project等）
2. 关系类型：自由提取，不需要局限于预定义类型。例如：works_at(在...工作)、lives_in(住在...)、likes(喜欢)、located_in(位于)、manages(管理)、attends(参加)、founded(创立)、owns(拥有)、member_of(成员)等
   如果文本中存在语义关系但无现成类型名，请自造合理的英文关系名（如 "reports_to"、"participates_in"）
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

EXTRACT_SYSTEM = "你是一个精确的知识图谱信息提取器。只输出JSON，不要解释。"

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

回答规则（必须严格遵守）：
1. **只根据记忆回答**，绝不能编造、推测或联想记忆中没有的信息。
2. **不能跨记忆推断**：不能从"A参加了X活动"和"B在Y公司工作"推断出"A也在Y公司工作"。
3. 如果记忆中没有明确回答用户问题的信息，直接回答"记忆中没有这个信息。"然后停止。
4. **只输出最终答案**，不要输出推理过程或分析步骤。回答不超过2句话。"""

CLASSIFY_PROMPT = """判断以下用户输入是要存储新记忆(ADD)还是查询已有记忆(QUERY)。

核心判断标准：
- QUERY: 用户在**提问/询问**（有疑问语气，想知道某信息）→ 输出 QUERY
- ADD: 用户在**陈述/告知**（告诉系统一个事实/信息）→ 输出 ADD

QUERY 示例（疑问句/询问语气）：
- "我的同事有哪些" → QUERY
- "我喜欢什么" → QUERY
- "我是谁" → QUERY
- "谁在货拉拉工作" → QUERY
- "张三的同事有谁" → QUERY
- "李四在哪里上班" → QUERY
- "帮我回忆一下王五的信息" → QUERY
- "货拉拉有多少员工" → QUERY
- "陈铨的同事有谁" → QUERY
- "有哪些人在深圳" → QUERY
- "赵六是什么职位" → QUERY

ADD 示例（陈述句/告知语气）：
- "我的同事李四也在腾讯" → ADD
- "我喜欢喝咖啡" → ADD
- "我叫张三，在腾讯工作" → ADD
- "李四参加了2026技术峰会" → ADD
- "王五创立了字节跳动" → ADD
- "陈铨在深圳货拉拉工作" → ADD
- "赵六是货拉拉的技术总监" → ADD

注意：
- 包含"谁/什么/哪里/哪些/多少/几/吗/呢/有没有"等疑问词 → 很可能是 QUERY
- "XX的同事有谁" / "谁在XX" / "XX是什么" → 一定是 QUERY
- "XX在YY工作" / "XX参加了ZZ" / "XX是YY" → 如果是陈述事实则 ADD

用户输入：{text}

直接回答 ADD 或 QUERY，不要解释。只输出一个词。"""


# ============================================================================
# HugeGraph Client Wrapper
# ============================================================================

class HugeGraphMemoryClient:
    """HugeGraph client with DYNAMIC Memory schema management.

    Key design: All edge labels and vertex labels are created on-demand.
    No hardcoding — if the LLM extracts a new type, the schema adapts.
    """

    def __init__(self, url=HUGEGRAPH_URL, user=HUGEGRAPH_USER, pwd=HUGEGRAPH_PASS,
                 graph=HUGEGRAPH_GRAPH):
        from pyhugegraph.client import PyHugeClient
        self.client = PyHugeClient(url=url, user=user, pwd=pwd, graph=graph)
        self._schema_initialized = False
        # Cache: which (edge_label, src_label, tgt_label) combos have been ensured
        self._edge_cache = {}   # (edge_label, src, tgt) -> actual_label_used
        self._vl_cache = set()  # vertex labels already ensured

    def init_schema(self):
        """Initialize property keys and base vertex labels. Edge labels are created on-demand."""
        if self._schema_initialized:
            return
        s = self.client.schema()

        # Property keys
        s.propertyKey("name").asText().ifNotExist().create()
        s.propertyKey("type").asText().ifNotExist().create()
        s.propertyKey("content").asText().ifNotExist().create()
        s.propertyKey("created_at").asDouble().ifNotExist().create()
        s.propertyKey("memory_id").asText().ifNotExist().create()
        s.propertyKey("access_count").asInt().ifNotExist().create()
        s.propertyKey("initial_score").asDouble().ifNotExist().create()
        s.propertyKey("last_accessed_at").asDouble().ifNotExist().create()

        # Base vertex labels (ensured, not enforced — more can be added dynamically)
        for vl in VERTEX_LABELS:
            self._ensure_vertex_label(vl)

        self._schema_initialized = True
        print("[HugeGraph] Schema initialized (dynamic mode)", file=sys.stderr, flush=True)

    def _ensure_vertex_label(self, label: str) -> bool:
        """Ensure a vertex label exists. Create it if needed. Returns True if created/exists."""
        if label in self._vl_cache:
            return True
        s = self.client.schema()
        try:
            s.vertexLabel(label).properties("name", "type").useCustomizeStringId().ifNotExist().create()
            self._vl_cache.add(label)
            return True
        except Exception as e:
            # Label may already exist with different properties
            self._vl_cache.add(label)
            # Verify it actually exists
            try:
                s.getVertexLabel(label)
                return True
            except Exception:
                print(f"[HugeGraph] Failed to create vertex label '{label}': {e}", file=sys.stderr)
                return False

    def _ensure_edge_label(self, edge_label: str, src_label: str, tgt_label: str) -> str:
        """Ensure an edge label exists for given source/target vertex types.

        Strategy:
        1. Try to create edge_label with (src_label -> tgt_label) — ifNotExist() is safe
        2. If creation fails because label exists with DIFFERENT source/target:
           - Try (edge_label + "_v2") etc. until we find an unused name
           - Cache the mapping so we reuse the same variant consistently
        3. Return the actual label name that was used

        Returns: the actual edge label name that should be used.
        """
        cache_key = (edge_label, src_label, tgt_label)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]

        s = self.client.schema()

        # Step 1: Try direct creation with ifNotExist()
        try:
            s.edgeLabel(edge_label).sourceLabel(src_label).targetLabel(tgt_label).ifNotExist().create()
            self._edge_cache[cache_key] = edge_label
            return edge_label
        except Exception as e1:
            print(f"[Schema] Step1 failed for '{edge_label}' ({src_label}->{tgt_label}): {e1}", file=sys.stderr)

        # Step 2: Check if the label already exists and is compatible
        try:
            existing = s.getEdgeLabel(edge_label)
            srcs = existing.get("source_label", [])
            tgts = existing.get("target_label", [])
            if isinstance(srcs, str):
                srcs = [srcs]
            if isinstance(tgts, str):
                tgts = [tgts]
            # If our src/tgt pair is already covered, reuse this label
            if src_label in srcs and tgt_label in tgts:
                self._edge_cache[cache_key] = edge_label
                return edge_label
            else:
                print(f"[Schema] Step2: '{edge_label}' exists but incompatible "
                      f"(has {srcs}->{tgts}, need {src_label}->{tgt_label})", file=sys.stderr)
        except Exception as e2:
            print(f"[Schema] Step2 check failed: {e2}", file=sys.stderr)

        # Step 3: Label exists with different source/target — create a variant
        print(f"[Schema] Step3: Creating variant for '{edge_label}' ({src_label}->{tgt_label})", file=sys.stderr)
        for i in range(2, 10):
            candidate = f"{edge_label}_v{i}"
            try:
                s.edgeLabel(candidate).sourceLabel(src_label).targetLabel(tgt_label).ifNotExist().create()
                self._edge_cache[cache_key] = candidate
                print(f"[HugeGraph] Edge label '{edge_label}' conflicts, using variant '{candidate}' "
                      f"({src_label}->{tgt_label})", file=sys.stderr)
                return candidate
            except Exception as e3:
                print(f"[Schema] Step3 candidate '{candidate}' failed: {e3}", file=sys.stderr)
                continue

        # Step 4: All variants failed — use the original and hope for the best
        self._edge_cache[cache_key] = edge_label
        return edge_label

    def add_vertex(self, label: str, name: str, properties: dict = None) -> Optional[str]:
        """Add a vertex, return its ID. Upsert by name. Creates label on-demand."""
        # Ensure vertex label exists (dynamic creation)
        self._ensure_vertex_label(label)

        g = self.client.graph()
        custom_id = f"{label}:{name}"
        props = {"name": name, "type": label}
        if properties:
            props.update(properties)

        # Check if exists by name
        existing = self.get_vertex_by_name(name)
        if existing:
            return existing["id"]

        try:
            v = g.addVertex(label, props, id=custom_id)
            return v.id if v else None
        except Exception as e:
            print(f"[HugeGraph] add_vertex error: {e} (label={label}, name={name})", file=sys.stderr)
            return None

    def get_vertex_by_name(self, name: str) -> Optional[dict]:
        """Get a vertex by its 'name' property."""
        g = self.client.graph()
        vertices = g.getVertexByCondition(limit=100)
        if vertices:
            for v in vertices:
                if getattr(v, 'properties', {}).get('name') == name or \
                   getattr(v, 'property', {}).get('name') == name:
                    return {"id": v.id, "label": v.label,
                            "properties": getattr(v, 'properties', {})}
        # Fallback: use gremlin
        result = self.exec_gremlin(f"g.V().has(\"name\",\"{name}\")")
        if result and len(result) > 0:
            return {"id": result[0]["id"], "label": result[0].get("label", "?")}
        return None

    def add_edge(self, edge_label: str, src_name: str, tgt_name: str,
                 properties: dict = None) -> Optional[str]:
        """Add an edge between two vertices. Schema is created dynamically.

        Flow:
        1. Find source/target vertices by name
        2. Get their actual vertex labels
        3. Ensure edge label exists for this (src_label, tgt_label) combo
        4. Check for duplicate edges
        5. Create the edge
        """
        src_v = self.get_vertex_by_name(src_name)
        tgt_v = self.get_vertex_by_name(tgt_name)
        if not src_v or not tgt_v:
            return None

        src_label = src_v.get("label", "")
        tgt_label = tgt_v.get("label", "")

        # Dynamically ensure the edge label exists for this source/target pair
        actual_label = self._ensure_edge_label(edge_label, src_label, tgt_label)

        # Check for duplicate edge
        try:
            existing = self.exec_gremlin(
                f'g.E().hasLabel("{actual_label}")'
                f'.where(outV().has("name","{src_name}"))'
                f'.where(inV().has("name","{tgt_name}"))'
            )
            if existing and len(existing) > 0:
                return existing[0].get("id") if isinstance(existing[0], dict) else str(existing[0])
        except Exception:
            pass

        # Create the edge
        g = self.client.graph()
        try:
            e = g.addEdge(actual_label, src_v["id"], tgt_v["id"], {})
            if actual_label != edge_label:
                print(f"[HugeGraph] Edge '{src_name}' --[{edge_label}→{actual_label}]--> '{tgt_name}'", file=sys.stderr)
            return e.id if e else None
        except Exception as ex:
            print(f"[HugeGraph] add_edge error: {ex} (label={actual_label}, {src_name}->{tgt_name})", file=sys.stderr)
            return None

    def exec_gremlin(self, query: str) -> list:
        """Execute a Gremlin query and return results."""
        try:
            gm = self.client.gremlin()
            result = gm.exec(query)
            if result is None:
                return []
            # Handle various response formats
            if isinstance(result, list):
                return [{"id": r.get("id", ""), "label": r.get("label", ""),
                         "properties": r.get("properties", {})} if isinstance(r, dict) else r
                        for r in result]
            return []
        except Exception as e:
            print(f"[Gremlin error] {e}", file=sys.stderr, flush=True)
            return []

    def get_all_vertices(self) -> list:
        """Get all vertices for visualization using REST API directly."""
        vertices = []
        try:
            g = self.client.graph()
            session = g._sess
            resp = session.request("graph/vertices?limit=500&page")
            if resp and "vertices" in resp:
                for v in resp["vertices"]:
                    props = v.get("properties", {}) or {}
                    vertices.append({
                        "id": v.get("id", ""),
                        "name": props.get("name", v.get("id", "")),
                        "type": props.get("type", v.get("label", "")),
                        "label": v.get("label", ""),
                        "properties": props,
                    })
        except Exception as e:
            print(f"[HugeGraph] get_all_vertices error: {e}", file=sys.stderr)
        return vertices

    def get_all_edges(self) -> list:
        """Get all edges for visualization."""
        edges = []
        try:
            from pyhugegraph.api.graph import GraphManager
            g = self.client.graph()
            # Use REST API directly for edges
            session = g._sess
            resp = session.request("graph/edges?limit=500&page")
            if resp and "edges" in resp:
                for e in resp["edges"]:
                    edges.append({
                        "id": e.get("id", ""),
                        "source": e.get("outV", ""),
                        "target": e.get("inV", ""),
                        "relationship": e.get("label", ""),
                        "properties": e.get("properties", {}),
                    })
        except Exception as ex:
            print(f"[HugeGraph] get_all_edges error: {ex}", file=sys.stderr)
        # Build source/target names from vertices
        vmap = {}
        for v in self.get_all_vertices():
            vmap[v["id"]] = v.get("name", "?")
        for e in edges:
            e["source_name"] = vmap.get(e["source"], "?")
            e["target_name"] = vmap.get(e["target"], "?")
        return edges

    def get_edges_by_vertex(self, vertex_name: str) -> list:
        """Get all edges connected to a vertex by name (REST API)."""
        edges = []
        try:
            g = self.client.graph()
            session = g._sess
            # Find vertex ID by scanning all vertices with matching name
            resp = session.request("graph/vertices?limit=500&page")
            if resp and "vertices" in resp:
                for v in resp["vertices"]:
                    props = v.get("properties", {}) or {}
                    if props.get("name") == vertex_name:
                        vid = v.get("id", "")
                        if vid:
                            e_resp = session.request(
                                f"graph/vertices/{vid}/edges?limit=100&direction=BOTH")
                            if e_resp and "edges" in e_resp:
                                edges.extend([{
                                    "label": e.get("label", ""),
                                    "source": e.get("outV", ""),
                                    "target": e.get("inV", ""),
                                } for e in e_resp["edges"]])
                        break  # Found the vertex, no need to continue
        except Exception as ex:
            print(f"[HugeGraph] get_edges_by_vertex error: {ex}",
                  file=sys.stderr)
        return edges

    def clear_graph(self):
        """Clear all vertices and edges using REST API."""
        try:
            g = self.client.graph()
            session = g._sess
            resp = session.request("graph/vertices?limit=9999&page")
            if resp and "vertices" in resp:
                for v in resp["vertices"][:200]:
                    try:
                        g.removeVertexById(v.get("id"))
                    except Exception:
                        pass
            print("[HugeGraph] Graph cleared", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[HugeGraph] clear error: {e}", file=sys.stderr, flush=True)


# ============================================================================
# FAISS Vector Index for Memory Search
# ============================================================================

class FaissMemoryIndex:
    """FAISS-based vector index for semantic memory search."""

    def __init__(self, dim: int = 1536, index_path: str = FAISS_INDEX_PATH):
        self.dim = dim
        self.index_path = index_path
        # Use Inner Product (cosine similarity friendly) or L2
        self.index = faiss.IndexFlatIP(dim)  # Inner Product for cosine sim
        self.metadata = []  # list of {memory_id, content, created_at}
        self.embedding_client = None

    def _get_embedding_client(self):
        """Lazy-init OpenAI-compatible embedding client."""
        if self.embedding_client is None:
            self.embedding_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        return self.embedding_client

    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding vector for text using MiMo API."""
        try:
            client = self._get_embedding_client()
            # Use a simple approach: call chat completions and extract features
            # For production, use proper embedding endpoint
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:800],
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            # Fallback: deterministic hash-based pseudo-embedding
            print(f"[FAISS] Embedding API error ({e}), using fallback", file=sys.stderr, flush=True)
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Deterministic hash-based pseudo-embedding when API unavailable."""
        # Create a sparse-like but fixed-dim vector from text hash
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, ch in enumerate(text[:self.dim]):
            vec[i % self.dim] += ord(ch) * 0.01
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add_memory(self, memory_id: str, content: str, created_at: float = None):
        """Add a memory to the FAISS index."""
        vec = self.embed_text(content)
        # Reshape for FAISS: (1, dim)
        vec = vec.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.metadata.append({
            "memory_id": memory_id,
            "content": content,
            "created_at": created_at or time.time(),
            "index_pos": len(self.metadata),  # position in FAISS
        })

    def search(self, query: str, top_k: int = 5, ebbinghaus_weights: dict = None) -> list:
        """Search memories by semantic similarity with optional Ebbinghaus weighting.

        Args:
            query: search query text
            top_k: number of results
            ebbinghaus_weights: {memory_id: retention_score} to weight results

        Returns:
            List of {memory_id, content, score, retention} sorted by weighted score
        """
        if self.index.ntotal == 0:
            return []

        qvec = self.embed_text(query).reshape(1, -1).astype(np.float32)

        # Search more than needed to allow re-ranking
        k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(qvec, k)

        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < len(self.metadata):
                meta = self.metadata[int(idx)]
                mid = meta["memory_id"]
                if mid in seen:
                    continue
                seen.add(mid)

                raw_score = float(score)
                # Apply Ebbinghaus weight if available
                retention = 1.0
                if ebbinghaus_weights and mid in ebbinghaus_weights:
                    retention = ebbinghaus_weights[mid]
                weighted_score = raw_score * (0.3 + 0.7 * retention)

                results.append({
                    "memory_id": mid,
                    "content": meta["content"],
                    "raw_score": round(raw_score, 4),
                    "retention": round(retention, 4),
                    "weighted_score": round(weighted_score, 4),
                })

        # Sort by weighted score
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dim,
            "index_type": "IndexFlatIP (Inner Product)",
        }

    def save(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)
        meta_path = self.index_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self):
        """Load FAISS index from disk."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            meta_path = self.index_path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)

    def clear(self):
        """Clear the index."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []


# ============================================================================
# SQLite metadata store (for Ebbinghaus scores + memory content)
# ============================================================================

def get_metadata_db():
    """Get thread-local SQLite connection for metadata."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db


def init_metadata_db():
    """Initialize metadata database schema."""
    db = get_metadata_db()
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
        CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
    """)
    db.commit()
    db.close()


# ============================================================================
# LLM Helpers
# ============================================================================

def _extract_json_from_response(response) -> dict:
    """Extract JSON from LLM response (handles reasoning models)."""
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = (msg.reasoning_content or "").strip()
    if not content:
        return {"entities": [], "relationships": []}

    # Strip markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]
        content = "\n".join(lines)
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    try:
        result = json.loads(content)
        return _normalize_keys(result)
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback patterns
    entities = []
    relationships = []

    ent_pattern = r'"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"'
    for m in re.finditer(ent_pattern, content):
        entities.append({"name": m.group(1), "type": m.group(2).lower()})

    rel_patterns = [
        r'"source"\s*:\s*"([^"]+)"\s*,\s*"relationship"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"',
        r'"subject"\s*:\s*"([^"]+)"\s*,\s*"(?:relation|relationship)"\s*:\s*"([^"]+)"\s*,\s*"(?:target|object)"\s*:\s*"([^"]+)"',
    ]
    for pattern in rel_patterns:
        for m in re.finditer(pattern, content):
            relationships.append({"source": m.group(1), "relationship": m.group(2), "target": m.group(3)})

    return {"entities": entities, "relationships": relationships}


def _normalize_keys(result: dict) -> dict:
    """Normalize LLM output key names to standard format.
    Unknown entity/relationship types are passed through as-is (dynamic schema will create them)."""
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])

    type_map = {
        "person": "person", "people": "person", "人": "person", "人物": "person",
        "organization": "organization", "org": "organization", "公司": "organization",
        "enterprise": "organization", "机构": "organization", "企业": "organization",
        "location": "location", "地点": "location", "地方": "location", "城市": "location",
        "skill": "skill", "技能": "skill", "爱好": "concept",
        "concept": "concept", "概念": "concept",
        "event": "event", "事件": "event",
        "product": "product", "产品": "product",
        "project": "project", "项目": "project",
    }

    normalized_entities = []
    for e in entities:
        name = e.get("name") or e.get("entity") or e.get("value", "")
        etype = (e.get("type") or e.get("category") or "concept").lower().strip()
        # Normalize known types; pass through unknown types as-is (dynamic schema handles it)
        etype = type_map.get(etype, etype)
        if name not in ("我", "自己", "本人"):
            normalized_entities.append({"name": name, "type": etype})

    normalized_rels = []
    for r in relationships:
        source = r.get("source") or r.get("subject") or ""
        target = r.get("target") or r.get("object") or ""
        rel = r.get("relationship") or r.get("relation") or ""
        if source and target and rel:
            normalized_rels.append({"source": source, "relationship": rel, "target": target})

    return {"entities": normalized_entities, "relationships": normalized_rels}


def _get_llm_text(response) -> str:
    msg = response.choices[0].message
    return (msg.content or "").strip() or (msg.reasoning_content or "").strip()


# ============================================================================
# Main Pipeline Backend
# ============================================================================

class MemoryPipelineBackend:
    """
    Production-grade Memory Pipeline — HugeGraph + FAISS + MiMo LLM.
    Aligned with PowerMem v1.1.2 add_memory (7-step) / search_memory (4-step).
    """

    def __init__(self, hg_client: HugeGraphMemoryClient = None,
                 faiss_index: FaissMemoryIndex = None):
        self.hg = hg_client or HugeGraphMemoryClient()
        self.hg.init_schema()
        self.faiss = faiss_index or FaissMemoryIndex()
        # Try loading saved FAISS index
        try:
            self.faiss.load()
        except Exception:
            pass
        self.llm_base_url = LLM_BASE_URL
        self.llm_model = LLM_MODEL
        self.llm_api_key = LLM_API_KEY

    # ---- LLM Operations ----

    def _llm_extract(self, text: str) -> dict:
        """Step 1: Extract entities and relations via MiMo LLM."""
        client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": f"{EXTRACT_PROMPT}\n\n\u7528\u6237\u8f93\u5165\uff1a{text}"},
                ],
                temperature=0.1,
                max_completion_tokens=2048,
            )
            result = _extract_json_from_response(response)
            print(f"[LLM Extract] {len(result.get('entities',[]))} entities, "
                  f"{len(result.get('relationships',[]))} rels", file=sys.stderr, flush=True)
            return result
        except Exception as e:
            print(f"[LLM Extract error] {e}", file=sys.stderr, flush=True)
            return {"entities": [], "relationships": []}

    def _rule_classify_intent(self, text: str) -> Optional[str]:
        """Fast rule-based classification. Returns ADD/QUERY or None if uncertain."""
        # Explicit question patterns — high confidence QUERY signals
        # These take priority over ADD patterns
        query_patterns = [
            r'有谁', r'有哪些', r'有多少', r'有几个', r'有什么',
            r'是谁', r'是谁的', r'是谁？',
            r'是什么', r'是什么职位', r'是什么工作',
            r'在哪里', r'在哪上班', r'在哪个',
            r'哪些人', r'谁在', r'谁在.*工作', r'谁在.*上班',
            r'叫什么', r'叫什么名字',
            r'帮我回忆', r'帮我查', r'回忆一下', r'查一下',
            r'介绍一下', r'告诉我.*信息',
            r'[？?]$',  # ends with question mark
            r'^谁',  # starts with "who" — always a question
        ]
        for pat in query_patterns:
            if re.search(pat, text):
                return "QUERY"

        # Explicit statement patterns — high confidence ADD signals
        add_patterns = [
            r'^(我在|我在.|我叫|我是.{1,4}(，|,|，在|在).{2,10}(工作|上班|任职))',
            r'^(今天|昨天|上周|最近).+(去了|见了|完成了|做了)',
        ]
        for pat in add_patterns:
            if re.search(pat, text):
                return "ADD"

        return None  # uncertain — need LLM

    def _llm_classify_intent(self, text: str) -> Optional[dict]:
        """Classify intent as ADD or QUERY. Rule-based fast path, LLM fallback."""
        # Fast rule-based classification first (no LLM call needed)
        rule_result = self._rule_classify_intent(text)
        if rule_result:
            return {"action": rule_result, "method": "rule", "reason": "Rule-based fast path"}

        # LLM classification for ambiguous cases
        try:
            client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content":
                     "你只输出ADD或QUERY，不输出任何其他内容。"},
                    {"role": "user", "content": CLASSIFY_PROMPT.format(text=text)},
                ],
                temperature=0.0,
                max_completion_tokens=512,
            )
            content = _get_llm_text(response).strip().upper()
            if content.startswith("QUERY") or "QUERY" in content:
                return {"action": "QUERY", "method": "llm", "reason": "LLM classified as QUERY"}
            if content.startswith("ADD") or "ADD" in content:
                return {"action": "ADD", "method": "llm", "reason": "LLM classified as ADD"}
            return None
        except Exception as e:
            print(f"[LLM classify error] {e}", file=sys.stderr, flush=True)
            return None

    def _llm_rank_memories(self, query: str, memories: list, graph_context: str = "") -> list:
        """Rank memories by relevance using LLM (with graph context)."""
        if not memories:
            return []
        try:
            client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
            memory_text = "\n".join([f"[{m['id']}] {m['content']}" for m in memories])
            extra = ""
            if graph_context:
                extra = (f"\n\n\u56fe\u8c31\u5173\u7cfb\u4e0a\u4e0b\u6587\uff1a\n{graph_context}\n"
                         f"\u8bf7\u4e5f\u8003\u8651\u56fe\u8c31\u5173\u7cfb\u6765\u5339\u914d\u8bb0\u5fc6\u3002")

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content":
                     "\u4f60\u662f\u4e00\u4e2a\u7cbe\u786e\u7684\u8bb0\u5fc6\u68c0\u7d22\u5668\u3002\u53ea\u8f93\u51faJSON\u6570\u7ec4\u3002"},
                    {"role": "user", "content":
                     SEARCH_PROMPT.format(query=query, memories=memory_text) + extra},
                ],
                temperature=0.1,
                max_completion_tokens=2048,
            )
            content = _get_llm_text(response)
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

    def _llm_generate_answer(self, query: str, memories: list, graph_context: str = "") -> str:
        """Generate answer using LLM based on memories and graph context."""
        try:
            client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
            memory_text = "\n".join([f"- {m['content']}" for m in memories])
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content":
                     "你是HugeGraph Memory助手，基于用户的记忆回答问题"
                     "。回答要简短，不超过2句话。不要输出推理过程。"},
                    {"role": "user", "content": ANSWER_PROMPT.format(
                        query=query, memories=memory_text,
                        graph_context=graph_context or "无")},
                ],
                temperature=0.3,
                max_completion_tokens=256,
            )
            content = _get_llm_text(response)
            if not content:
                content = "无法生成回答。"
            # Post-process: strip MiMo chain-of-thought reasoning
            content = self._strip_reasoning(content)
            return content
        except Exception as e:
            print(f"[LLM answer error] {e}", file=sys.stderr, flush=True)
            return f"\u751f\u6210\u56de\u7b54\u65f6\u51fa\u9519: {e}"

    # ---- ADD Pipeline (7 steps, aligned with PowerMem) ----

    def add_memory(self, content: str, user_id: str = "demo_user") -> dict:
        """
        Add a new memory through the full 7-step pipeline.
        Aligned with PowerMem MemoryStore.add_memory().
        """
        # Auto-routing: if intent is QUERY, delegate to search_memory
        classify_result = self._llm_classify_intent(content)
        if not classify_result:
            has_q = bool(re.search(r'[？?]', content))
            starts_q = bool(re.match(r'^(谁|什么|哪里|哪个|哪些|多少|几|怎么|如何)', content))
            classify_result = {"action": "QUERY" if (has_q or starts_q) else "ADD",
                             "method": "regex", "reason": "Fallback"}
        if classify_result.get("action") == "QUERY":
            return self.search_memory(content, user_id)

        start_time = time.time()
        db = get_metadata_db()
        now = time.time()
        memory_id = str(uuid.uuid4())[:8]
        trace = []  # pipeline execution trace for frontend display

        # Step 1: LLM Entity Extraction
        step_start = time.time()
        extraction = self._llm_extract(content)
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])
        trace.append({"step": 1, "name": "LLM\u5b9e\u4f53\u62bd\u53d6",
                      "detail": f"{len(entities)} entities, {len(relationships)} relations",
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": {"entities": entities, "relationships": relationships}})

        # Step 1.5: Self-reference Resolution
        step_start = time.time()
        user_name = self._get_user_name(db, user_id)
        self_refs = []
        if user_name:
            for rel in relationships:
                if rel["source"] in ("\u6211", "\u81ea\u5df1", "\u672c\u4eba"):
                    old = rel["source"]; rel["source"] = user_name
                    self_refs.append(f"\"{old}\" \\u2192 \"{user_name}\"")
                if rel["target"] in ("\u6211", "\u81ea\u5df1", "\u672c\u4eba"):
                    old = rel["target"]; rel["target"] = user_name
                    if f"\"{old}\" \\u2192 \"{user_name}\"" not in self_refs:
                        self_refs.append(f"\"{old}\" \\u2192 \"{user_name}\"")
            if not any(e["name"] == user_name for e in entities):
                entities.append({"name": user_name, "type": "person"})
        trace.append({"step": 15, "name": "\u6307\u4ee3\u6d88\u89e3",
                      "detail": self_refs if self_refs else "\u65e0\u9700\u6d88\u89e3",
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": {"user_name": user_name, "resolved": self_refs}})

        # Step 2: Conflict Detection
        step_start = time.time()
        action = "ADD"
        conflict_reason = ""
        conflict_detail = ""
        existing_rows = db.execute(
            "SELECT id, content FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()

        # Entity-level conflict detection
        new_persons = [e["name"] for e in entities if e["type"] == "person"]
        new_orgs = [e["name"] for e in entities if e["type"] == "organization"]
        for ex in existing_rows:
            sp = any(p in ex["content"] for p in new_persons)
            so = any(o in ex["content"] for o in new_orgs)
            if sp and so:
                common = sum(1 for ch in content if ch in ex["content"])
                sim = common / min(len(content), len(ex["content"]))
                if sim > 0.85:
                    action = "SKIP"
                    conflict_reason = f"\u5b9e\u4f53({','.join(new_persons)}@{','.join(new_orgs)})\u9ad8\u5ea6\u76f8\u4f3c{round(sim*100)}%"
                    conflict_detail = f"\u4e0e #{ex['id']} \u53ef\u80fd\u662f\u91cd\u590d\u66f4\u65b0"
                    break
            # Literal duplicate check (>90% same)
            cc = sum(1 for ch in content if ch in ex["content"])
            s2 = cc / min(len(content), len(ex["content"]))
            if s2 > 0.9:
                action = "SKIP"
                conflict_reason = f"\u6587\u672c\u51e0\u4e4e\u76f8\u540c{round(s2*100)}%"
                conflict_detail = f"\u53ef\u80fd\u91cd\u590d\u63d0\u4ea4"
                break

        trace.append({"step": 2, "name": "\u51b2\u7a81\u68c0\u6d4b (\u5b9e\u4f53\u7ea7)",
                      "detail": conflict_reason if action=="SKIP" else
                               "\u2714 \u65e0\u51b2\u7a81(\u65b0\u8bb0\u5fc6\u72ec\u7acb)",
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": {"action": action, "reason": conflict_reason}})

        # Step 3: Entity Dedup (merge "腾讯深圳" → "腾讯" + "深圳")
        step_start = time.time()
        deduped_entities, relationships = self._dedup_entities(entities, relationships)
        if len(deduped_entities) < len(entities):
            ne = len(entities)
            nde = len(deduped_entities)
            trace.append({"step": 3, "name": "\u5b9e\u4f53\u53bb\u91cd",
                          "detail": f"{ne} \u2192 {nde} (\u5408\u5e76{ne-nde}\u4e2a)",
                          "elapsed_ms": round((time.time()-step_start)*1000)})
        else:
            trace.append({"step": 3, "name": "\u5b9e\u4f53\u53bb\u91cd",
                          "detail": "\u65e0\u9700\u53bb\u91cd",
                          "elapsed_ms": round((time.time()-step_start)*1000)})
        entities = deduped_entities

        # Step 4: Relationship Completion (regex fallback when LLM misses)
        step_start = time.time()
        orig_rel_count = len(relationships)
        relationships = self._extract_missing_rels(content, entities, relationships)
        new_rels = len(relationships) - orig_rel_count
        trace.append({"step": 4, "name": "\u5173\u7cfb\u8865\u5168",
                      "detail": f"+{new_rels} \u6761\u8865\u5145\u5173\u7cfb" if new_rels > 0 else "\u65e0\u9700\u8865\u5168",
                      "elapsed_ms": round((time.time()-step_start)*1000)})

        # Step 5: Colleague Inference (cross-memory, via HugeGraph Gremlin)
        step_start = time.time()
        colleague_result = self._infer_colleague(relationships, entities)
        trace.append({"step": 5, "name": "\u540c\u4e8b\u63a8\u7406",
                      "detail": colleague_result["reason"],
                      "triggered": colleague_result["trigger"],
                      "inferred": colleague_result["inferred"],
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": colleague_result})

        # Step 6 & 7: Store to HugeGraph (nodes + edges) + FAISS + SQLite metadata
        # P0 fix: Always store graph entities/edges even on SKIP (align with PowerMem)
        step_start = time.time()
        stored_nodes = []
        stored_edges = []

        # 6a. Always add vertices to HugeGraph (even SKIP — new graph structure)
        node_ids = {}
        for ent in entities:
            vid = self.hg.add_vertex(ent["type"], ent["name"])
            if vid:
                node_ids[ent["name"]] = vid
                stored_nodes.append({"name": ent["name"], "type": ent["type"], "id": vid})

        # 6b. Always add edges to HugeGraph (even SKIP — new graph structure)
        for rel in relationships:
            eid = self.hg.add_edge(rel["relationship"], rel["source"], rel["target"], {})
            if eid:
                stored_edges.append({**rel, "edge_id": eid})

        # 6c-6d. Only store memory metadata + vector when NOT SKIP
        if action != "SKIP":
            db.execute(
                "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count)"
                " VALUES (?,?,?,?,?,?)",
                (memory_id, content, user_id, now, now, 1),
            )
            self.faiss.add_memory(memory_id, content, now)
            try:
                self.faiss.save()
            except Exception:
                pass

        db.commit()
        db.close()

        total_elapsed = round((time.time() - start_time) * 1000)
        trace.append({"step": 67, "name": "\u5b58\u50a8 (HugeGraph+FAISS+SQLite)",
                      "detail": f"{len(stored_nodes)} nodes, {len(stored_edges)} edges, "
                              f"1 memory, action={action}",
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": {
                          "memory_id": memory_id if action != "SKIP" else None,
                          "action": action,
                          "stored_nodes": stored_nodes,
                          "stored_edges": stored_edges,
                      }})

        return {
            "memory_id": memory_id if action != "SKIP" else None,
            "action": action,
            "reason": conflict_reason or conflict_detail or
                    ("\u65b0\u8bb0\u5fc6\uff0c\u65e0\u51b2\u7a81" if action == "ADD" else conflict_reason),
            "trace": trace,
            "entities": entities,
            "relationships": stored_edges,
            "total_elapsed_ms": total_elapsed,
        }

    # ---- QUERY Pipeline (4 steps, aligned with PowerMem) ----

    def search_memory(self, query: str, user_id: str = "demo_user", top_k: int = 5) -> dict:
        """
        Search memories through the 4-step pipeline.
        Aligned with PowerMem MemoryStore.search_memory().
        """
        start_time = time.time()
        db = get_metadata_db()
        now = time.time()
        trace = []

        # Step 1: Intent Classification (LLM primary + regex fallback)
        step_start = time.time()
        classify_result = self._llm_classify_intent(query)
        if not classify_result:
            # Regex fallback (broader patterns)
            has_qmark = bool(re.search(r'[？?]', query))
            starts_q = bool(re.match(r'^(谁|什么|哪里|哪个|哪些|多少|几|怎么|如何)', query))
            starts_my = bool(re.match(
                r'^(我|我的)(的?|们?)(同事|朋友|认识|有哪些|有谁|叫什么|在哪)',
                query))
            has_query_pattern = bool(re.search(
                r'(有谁|有哪些|有多少|是什么|在哪里|是什么职位|叫什么|帮我回忆|帮我查)',
                query))
            is_query = has_qmark or starts_q or starts_my or has_query_pattern
            classify_result = {"action": "QUERY" if is_query else "ADD",
                             "method": "regex", "reason": "Fallback classification"}

            ca = classify_result["action"]
            cm = classify_result.get("method","llm")
            trace.append({"step": 1, "name": "\u610f\u56fe\u5206\u7c7b",
                      "detail": f"{ca} ({cm})",
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": classify_result})

        if classify_result["action"] != "QUERY":
            db.close()
            return {"query": query, "error": "NOT_A_QUERY",
                    "hint": "\u8bf7\u8f93\u5165\u7591\u95ee\u53e5\u6765\u67e5\u8be2\u8bb0\u5fc6",
                    "trace": trace}

        # Step 2: Ebbinghaus Scoring + FAISS Vector Search + LLM Rerank
        step_start = time.time()

        # Get all memories for Ebbinghaus calculation
        rows = db.execute(
            "SELECT id,content,created_at,last_accessed_at,access_count,initial_score "
            "FROM memories WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()

        # Compute Ebbinghaus weights
        eb_weights = {}
        memories_list = []
        for row in rows:
            elapsed_hours = (now - row["created_at"]) / 3600
            ret = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            ret = min(1.0, max(0.0, ret + row["access_count"] * EBBINGHAUS_REINFORCE))
            eb_weights[row["id"]] = round(ret, 4)
            memories_list.append({"id": row["id"], "content": row["content"],
                                   "retention": ret, "access_count": row["access_count"]})

        # FAISS semantic search with Ebbinghaus weighting
        faiss_results = self.faiss.search(query, top_k=top_k * 2, ebbinghaus_weights=eb_weights)

        # Merge with Ebbinghaus-only results for memories without embeddings
        faiss_ids = set(r["memory_id"] for r in faiss_results)
        for m in memories_list:
            if m["id"] not in faiss_ids:
                faiss_results.append({
                    "memory_id": m["id"], "content": m["content"],
                    "raw_score": m["retention"], "retention": m["retention"],
                    "weighted_score": m["retention"] * 0.5,
                })

        # Sort by weighted score
        faiss_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        top_candidates = faiss_results[:top_k * 2]

        # LLM reranking with graph context
        graph_ctx = self._build_graph_context()
        llm_ranks = self._llm_rank_memories(query, top_candidates, graph_ctx)

        # Merge LLM ranks with candidate scores
        results = []
        llm_score_map = {r.get("memory_id"): r.get("score", 0.5) for r in llm_ranks}
        for cand in top_candidates[:top_k]:
            llm_s = llm_score_map.get(cand["memory_id"])
            final_score = llm_s if llm_s else cand["weighted_score"]
            mem_data = next((m for m in memories_list if m["id"] == cand["memory_id"]), None)
            results.append({
                "memory": mem_data or {"id": cand["memory_id"], "content": cand["content"]},
                "score": round(final_score, 4),
                "source": "llm_rerank" if llm_s else "vector_ebbinghaus",
            })
            # Reinforce accessed memories
            if cand["memory_id"]:
                db.execute(
                    "UPDATE memories SET access_count=access_count+1, last_accessed_at=? WHERE id=?",
                    (now, cand["memory_id"]),
                )

        trace.append({"step": 2, "name": "Ebbinghaus+\u5411\u91cf\u68c0\u7d22+LLM\u91cd\u6392",
                      "detail": f"{len(memories_list)} memories, FAISS={self.faiss.index.ntotal}, "
                              f"Top-{len(results)} results",
                      "ebbinghaus_scores": [
                          {"id": m["id"], "content": m["content"][:30],
                           "retention": eb_weights.get(m["id"], 0)}
                          for m in memories_list[:5]],
                      "elapsed_ms": round((time.time()-step_start)*1000)})

        # Step 3: Graph Context Retrieval (from HugeGraph)
        step_start = time.time()
        # Already computed above, but log it
        trace.append({"step": 3, "name": "\u56fe\u8c31\u4e0a\u4e0b\u6587\u68c0\u7d22",
                      "detail": f"{len(graph_ctx.split(chr(10)))} edges retrieved"
                      if graph_ctx else "\u56fe\u8c31\u4e3a\u7a7a",
                      "elapsed_ms": round((time.time()-step_start)*1000)})

        # Step 4: LLM Answer Generation
        step_start = time.time()
        relevant_memories = [r["memory"] for r in results if r.get("memory")]
        all_edges = self.hg.get_all_edges()

        # Extract all potential entity names from query
        query_names = set()
        # Split by query particles first, then take 2-3 char segments
        parts = re.split(r'[的了在是有和也都哪些多少几怎么如何谁什么哪里哪个有没有这个信息记忆同事朋友共事员工上班工作总部公司参加创立技术城市总监告诉我帮回忆]', query)
        for part in parts:
            part = part.strip()
            if len(part) >= 2 and len(part) <= 4 and re.match(r'^[\u4e00-\u9fa5]+$', part):
                query_names.add(part)

        # Check: does ANY query entity exist in the system?
        known_in_system = set()
        for v in self.hg.get_all_vertices():
            known_in_system.add(v.get("name", ""))
        for mem in relevant_memories:
            for m2 in re.finditer(r'[\u4e00-\u9fa5]{2,4}', mem["content"]):
                known_in_system.add(m2.group())

        # If NO query entity exists at all → direct "not found"
        query_known = query_names & known_in_system
        if query_names and not query_known:
            answer = f"记忆中没有这个信息。"
            trace.append({"step": 4, "name": "实体存在性检查",
                          "detail": f"{','.join(query_names)} 均不在系统中",
                          "elapsed_ms": round((time.time()-step_start)*1000)})
            db.commit()
            db.close()
            return {
                "query": query, "action": "QUERY", "results": results,
                "answer": answer, "graph_context": graph_ctx,
                "trace": trace, "total_elapsed_ms": round((time.time() - start_time) * 1000),
            }

        # Graph-based direct reasoning for common query patterns
        is_colleague_query = bool(re.search(r'同事|共事|teammate', query))
        is_org_employee_query = bool(re.search(r'员工|有哪些人|谁在.*工作|有哪些.*人', query))
        is_workplace_query = bool(re.search(r'在哪.*上班|在哪.*工作|哪里工作|在哪里上班', query))
        is_position_query = bool(re.search(r'什么职位|什么岗位|是.*的.*总|是.*的.*监|是.*的.*长', query))
        if (is_colleague_query or is_org_employee_query or is_workplace_query or is_position_query) and query_known:
            if is_colleague_query:
                answer = self._graph_colleague_answer(list(query_known), all_edges)
            elif is_org_employee_query:
                answer = self._graph_org_employee_answer(list(query_known), all_edges)
            elif is_workplace_query:
                answer = self._graph_workplace_answer(list(query_known), all_edges)
            elif is_position_query:
                answer = self._graph_position_answer(list(query_known))
            if answer:
                trace.append({"step": 4, "name": "图谱直接推理",
                              "detail": "从works_at边计算(无需LLM)",
                              "elapsed_ms": round((time.time()-step_start)*1000)})
                db.commit()
                db.close()
                return {
                    "query": query, "action": "QUERY", "results": results,
                    "answer": answer, "graph_context": graph_ctx,
                    "trace": trace, "total_elapsed_ms": round((time.time() - start_time) * 1000),
                }

        # Fallback: LLM answer generation
        if relevant_memories:
            answer = self._llm_generate_answer(query, relevant_memories, graph_ctx)
        elif graph_ctx:
            answer = self._llm_generate_answer(query, [], graph_ctx)
        else:
            answer = "记忆中没有相关信息。"

        trace.append({"step": 4, "name": "LLM\u56de\u7b54\u751f\u6210",
                      "detail": f"{len(answer)} chars",
                      "elapsed_ms": round((time.time()-step_start)*1000)})

        db.commit()
        db.close()

        return {
            "query": query,
            "action": "QUERY",
            "results": results,
            "answer": answer,
            "graph_context": graph_ctx,
            "trace": trace,
            "total_elapsed_ms": round((time.time() - start_time) * 1000),
        }

    # ---- Helper Methods (aligned with PowerMem MemoryStore) ----

    def _graph_colleague_answer(self, query_entities: list, edges: list) -> str:
        """Compute colleague relationships directly from graph works_at edges.
        Returns answer string or None if can't determine.
        """
        # Build org -> persons map from works_at edges
        org_persons = {}  # org_name -> [person_names]
        person_org = {}   # person_name -> org_name
        for e in edges:
            elabel = e.get("label") or e.get("relationship") or ""
            if elabel not in ("works_at", "based_in", "employed_by"):
                continue
            sname = e.get("source_name") or ""
            tname = e.get("target_name") or ""
            # Source is person, target is org
            if tname and sname:
                org_persons.setdefault(tname, []).append(sname)
                person_org[sname] = tname

        for ent in query_entities:
            org = person_org.get(ent)
            if not org:
                continue
            colleagues = [p for p in org_persons.get(org, []) if p != ent]
            if colleagues:
                return f"根据图谱，{ent}的同事有{'、'.join(colleagues)}。"
            else:
                return f"图谱中{ent}在{org}工作，但没有找到其他同事。"

        # If query entities have no works_at, check if they're completely unknown
        known = set()
        for e in edges:
            known.add(e.get("source_name", ""))
            known.add(e.get("target_name", ""))
        unknown = [e for e in query_entities if e not in known]
        if unknown:
            return f"记忆中没有关于{'、'.join(unknown)}的信息。"

        return None  # Can't determine, let LLM handle

    def _graph_org_employee_answer(self, query_entities: list, edges: list) -> str:
        """Find all people working at queried organizations."""
        org_persons = {}
        for e in edges:
            elabel = e.get("label") or e.get("relationship") or ""
            if elabel not in ("works_at", "based_in", "employed_by"):
                continue
            sname = e.get("source_name") or ""
            tname = e.get("target_name") or ""
            org_persons.setdefault(tname, []).append(sname)

        for ent in query_entities:
            if ent in org_persons:
                persons = org_persons[ent]
                return f"根据图谱，{ent}的员工有{'、'.join(persons)}。"

        return None  # Let LLM handle

    def _graph_workplace_answer(self, query_entities: list, edges: list) -> str:
        """Find where a person works from graph edges + memory content."""
        person_org = {}
        for e in edges:
            elabel = e.get("label") or e.get("relationship") or ""
            if elabel in ("works_at", "based_in", "employed_by", "founded"):
                sname = e.get("source_name") or ""
                tname = e.get("target_name") or ""
                person_org[sname] = tname
        for ent in query_entities:
            org = person_org.get(ent)
            if org:
                # Also check memory for position info
                db = get_metadata_db()
                rows = db.execute(
                    "SELECT content FROM memories WHERE content LIKE ?",
                    (f'%{ent}%',)
                ).fetchall()
                db.close()
                for row in rows:
                    m = re.search(r'(\S+是\S+的(\S+))', row["content"])
                    if m:
                        return f"根据记忆，{ent}{m.group(1)}。"
                return f"根据图谱，{ent}在{org}。"
        return None

    def _graph_position_answer(self, query_entities: list) -> str:
        """Find a person's position/role from memory content."""
        for ent in query_entities:
            db = get_metadata_db()
            rows = db.execute(
                "SELECT content FROM memories WHERE content LIKE ?",
                (f'%{ent}%',)
            ).fetchall()
            db.close()
            for row in rows:
                # Pattern: "X是Y的Z" (e.g., "赵六是货拉拉的技术总监")
                m = re.search(re.escape(ent) + r'是(\S+的\S+)', row["content"])
                if m:
                    return f"根据记忆，{ent}是{m.group(1)}。"
                # Pattern: "X担任Y" or "X负责Y"
                m = re.search(re.escape(ent) + r'(担任|负责|作为)(\S+)', row["content"])
                if m:
                    return f"根据记忆，{ent}{m.group(1)}{m.group(2)}。"
            return f"记忆中没有关于{ent}的职位信息。"
        return None

    def _strip_reasoning(self, text: str) -> str:
        """Strip chain-of-thought from MiMo model. Take last meaningful sentence."""
        if not text:
            return text
        text = text.strip()
        if len(text) <= 60:
            return text
        # Take last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            # If last line is too long, take last sentence (split by 。)
            if len(last) > 60:
                sents = last.split("。")
                sents = [s.strip() for s in sents if s.strip()]
                if sents:
                    last = sents[-1] if len(sents[-1]) <= 60 else sents[-2] if len(sents) > 1 else last[:60]
            return last
        return text[:60]

    def _extract_query_entities(self, query: str) -> list:
        """Extract known entity names mentioned in the query.
        Uses existing graph vertices + memory content as the entity dictionary,
        so only names the system already knows about are extracted.
        """
        known_names = set()
        # From HugeGraph vertices
        try:
            hg_verts = self.hg.get_all_vertices()
            for v in hg_verts:
                if v.get("name"):
                    known_names.add(v["name"])
        except Exception:
            pass
        # From SQLite memories
        try:
            db = get_metadata_db()
            rows = db.execute("SELECT content FROM memories").fetchall()
            db.close()
            for row in rows:
                for m in re.finditer(r'[\u4e00-\u9fa5]{2,4}', row["content"]):
                    name = m.group()
                    if name not in ("的", "在", "是", "有", "了", "和", "也", "都",
                                    "公司", "总部", "参加", "创立", "工作", "上班",
                                    "技术", "城市", "员工", "同事", "总监"):
                        known_names.add(name)
        except Exception:
            pass
        # Check which known names appear in the query
        found = [name for name in known_names if name in query]
        return list(set(found))

    def _get_user_name(self, db, user_id: str) -> str:
        rows = db.execute(
            "SELECT content FROM memories WHERE user_id=? ORDER BY created_at ASC", (user_id,)
        ).fetchall()
        for row in rows:
            m = re.search(r'(?:\u6211\u53eb|\u6211\u662f)\s*([\u4e00-\u9fa5]{2,4})', row["content"])
            if m:
                return m.group(1)
        return ""

    def _dedup_entities(self, entities: list, relationships: list) -> tuple:
        """Deduplicate entities (e.g., merge '腾讯深圳' into '腾讯' + '深圳')."""
        hg_verts = self.hg.get_all_vertices()
        existing_names = set(v["name"] for v in hg_verts)
        merged = {}
        new_entities = []
        for ent in entities:
            name = ent["name"]
            best_match = None
            for ename in existing_names:
                if ename in name and ename != name:
                    best_match = ename
                    remainder = name.replace(ename, "").strip()
                    if remainder and remainder not in existing_names:
                        if re.match(r'^[\u4e00-\u9fa5]{2,3}$', remainder):
                            new_entities.append({"name": remainder, "type": "location"})
                    break
            if best_match:
                merged[name] = best_match
            elif name not in existing_names:
                new_entities.append(ent)
        for rel in relationships:
            if rel["source"] in merged:
                rel["source"] = merged[rel["source"]]
            if rel["target"] in merged:
                rel["target"] = merged[rel["target"]]
        return new_entities, relationships

    def _extract_missing_rels(self, content: str, entities: list, relationships: list) -> list:
        """Regex-fallback relationship extraction when LLM misses some."""
        entity_names = {e["name"] for e in entities}
        hg_verts = self.hg.get_all_vertices()
        for v in hg_verts:
            entity_names.add(v["name"])

        existing_rels = {(r["source"], r["relationship"], r["target"]) for r in relationships}
        orgs_in_content = set()
        for v in hg_verts:
            if v.get("type") == "organization" and v["name"] in content:
                orgs_in_content.add(v["name"])
        for e in entities:
            if e["type"] == "organization":
                orgs_in_content.add(e["name"])

        persons_in_content = [e["name"] for e in entities if e["type"] == "person"]
        for person in persons_in_content:
            for org in orgs_in_content:
                if (person, "works_at", org) not in existing_rels:
                    pat = rf"{person}.*?\u5728.*?{org}|{org}.*{person}|{person}.*{org}"
                    if re.search(pat, content):
                        relationships.append({"source": person, "relationship": "works_at", "target": org})
                        existing_rels.add((person, "works_at", org))

        user_name = ""
        try:
            mdb = get_metadata_db()
            user_name = self._get_user_name(mdb, "demo_user")
            mdb.close()
        except Exception:
            pass
        skills = [e["name"] for e in entities if e["type"] in ("skill", "concept")]
        if user_name and skills and "\u559c\u6b22" in content:
            for skill in skills:
                if (user_name, "likes", skill) not in existing_rels and skill in content:
                    relationships.append({"source": user_name, "relationship": "likes", "target": skill})

        return relationships

    def _infer_colleague(self, relationships: list, entities: list) -> dict:
        """Infer colleague relationships via HugeGraph cross-memory graph traversal."""
        person_names = set(e["name"] for e in entities if e["type"] == "person")
        for rel in relationships:
            if rel["relationship"] == "works_at":
                person_names.add(rel["source"])

        # Also fetch persons from HugeGraph
        all_work_rels = list(relationships)
        hg_edges = self.hg.get_all_edges()
        for e in hg_edges:
            if e.get("relationship") == "works_at":
                person_names.add(e.get("source_name", ""))
                all_work_rels.append({
                    "source": e.get("source_name", ""),
                    "relationship": "works_at",
                    "target": e.get("target_name", ""),
                })

        if len(person_names) < 2:
            pn = len(person_names)
            return {"trigger": False, "inferred": [],
                    "reason": "仅检测到" + str(pn) + "个person"}

        groups = {}
        for rel in all_work_rels:
            if rel["relationship"] == "works_at":
                groups.setdefault(rel["target"], []).append(rel["source"])

        inferred = []
        for org, members in groups.items():
            if len(members) >= 2:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        p1, p2 = members[i], members[j]
                        involves_new = (p1 in person_names) or (p2 in person_names)
                        if involves_new:
                            inferred.append({
                                "source": p1, "relationship": "colleague_of",
                                "target": p2, "org": org,
                            })

        if not inferred:
            if all_work_rels:
                pname = ",".join(person_names)
                return {"trigger": False, "inferred": [],
                        "reason": pname + "加入后无新同事(各自在不同组织)"}
            ar = len(all_work_rels)
            return {"trigger": False, "inferred": [],
                    "reason": "图谱中仅" + str(ar) + "条works_at"}

        ni = len(inferred)
        gk = ",".join(groups.keys())
        return {"trigger": True, "inferred": inferred,
                "reason": "发现" + str(ni) + "对新同事(共享 " + gk + ")"}

    def _build_graph_context(self) -> str:
        """Build graph context string from HugeGraph for LLM.
        Prioritize memory-related edges (works_at, colleague_of, etc.)
        over supply chain data.
        """
        edges = self.hg.get_all_edges()
        if not edges:
            return ""
        # Priority edges: memory/person/organization relations
        priority_labels = {"works_at", "colleague_of", "attends", "founded",
                          "headquartered_in", "based_in", "employed_by",
                          "located_in", "manages", "participates_in"}
        priority = [e for e in edges if (e.get("relationship") or e.get("label") or "") in priority_labels]
        other = [e for e in edges if e not in priority]
        ordered = priority + other
        return "\n".join([
            (e.get("source_name","?") + " --[" + (e.get("relationship") or e.get("label") or "?") + "]--> " +
             e.get("target_name","?"))
            for e in ordered[:20]
        ])

    def get_stats(self, user_id: str = "demo_user") -> dict:
        db = get_metadata_db()
        now = time.time()
        mem_count = db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id=?", (user_id,)
        ).fetchone()[0]

        hg_verts = self.hg.get_all_vertices()
        hg_edges = self.hg.get_all_edges()

        type_dist = {}
        for v in hg_verts:
            t = v.get("type", v.get("label", "unknown"))
            type_dist[t] = type_dist.get(t, 0) + 1

        ebbinghaus = []
        for row in db.execute(
            "SELECT id,content,created_at,last_accessed_at,access_count,initial_score "
            "FROM memories WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ):
            elapsed_hours = (now - row["created_at"]) / 3600
            ret = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            ret = min(1.0, max(0.0, ret + row["access_count"] * EBBINGHAUS_REINFORCE))
            ebbinghaus.append({
                "id": row["id"], "content": row["content"],
                "retention": round(ret, 4),
                "elapsed_hours": round(elapsed_hours, 2),
                "access_count": row["access_count"],
            })

        db.close()
        faiss_stats = self.faiss.get_stats()

        return {
            "total_memories": mem_count,
            "total_nodes": len(hg_verts),
            "total_edges": len(hg_edges),
            "node_type_distribution": type_dist,
            "ebbinghaus_scores": ebbinghaus,
            "faiss": faiss_stats,
        }

    def get_graph_data(self) -> dict:
        nodes = self.hg.get_all_vertices()
        edges = self.hg.get_all_edges()
        return {"vertices": nodes, "edges": edges}

    def get_memories(self, user_id: str = "demo_user") -> list:
        db = get_metadata_db()
        now = time.time()
        memories = []
        for row in db.execute(
            "SELECT id,content,created_at,last_accessed_at,access_count,initial_score "
            "FROM memories WHERE user_id=? ORDER BY created_at DESC", (user_id,),
        ):
            elapsed_hours = (now - row["created_at"]) / 3600
            ret = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            ret = min(1.0, max(0.0, ret + row["access_count"] * EBBINGHAUS_REINFORCE))
            memories.append({
                "id": row["id"], "content": row["content"],
                "retention": round(ret, 4), "access_count": row["access_count"],
            })
        db.close()
        return memories

    def clear_all(self, user_id: str = "demo_user"):
        db = get_metadata_db()
        db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        db.commit()
        db.close()
        self.hg.clear_graph()
        self.faiss.clear()
        try:
            self.faiss.save()
        except Exception:
            pass


# ============================================================================
# Flask App Factory
# ============================================================================

def create_app(backend: MemoryPipelineBackend = None) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    store = backend or MemoryPipelineBackend()

    @app.route("/api/memory/add", methods=["POST"])
    def api_add_memory():
        data = request.json or {}
        content = data.get("content", "").strip()
        user_id = data.get("user_id", "demo_user")
        if not content:
            return jsonify({"error": "content is required"}), 400
        result = store.add_memory(content, user_id)
        return jsonify(result)

    @app.route("/api/memory/search", methods=["POST"])
    def api_search_memory():
        data = request.json or {}
        query = data.get("query") or data.get("content", "").strip()
        user_id = data.get("user_id", "demo_user")
        if not query:
            return jsonify({"error": "query is required"}), 400
        result = store.search_memory(query, user_id)
        return jsonify(result)

    @app.route("/api/memory/classify", methods=["POST"])
    def api_classify():
        data = request.json or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400
        result = store._llm_classify_intent(text)
        if not result:
            has_q = bool(re.search(r'[？?]', text))
            starts_q = bool(re.match(r'^(\u8c01|\u4ec0\u4e48|\u54ea\u91cc)', text))
            result = {"action": "QUERY" if (has_q or starts_q) else "ADD",
                     "method": "regex", "reason": "Fallback"}
        return jsonify(result)

    @app.route("/api/memory/list", methods=["GET"])
    def api_list_memories():
        return jsonify(store.get_memories())

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        return jsonify(store.get_stats())

    @app.route("/api/graph", methods=["GET"])
    def api_graph():
        return jsonify(store.get_graph_data())

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        data = request.json or {}
        store.clear_all(data.get("user_id", "demo_user"))
        return jsonify({"status": "cleared"})

    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HugeGraph Memory Backend Server "
                                     "(HugeGraph + FAISS + MiMo LLM)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host bind")
    parser.add_argument("--reset", action="store_true", help="Reset all data")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        meta = FAISS_INDEX_PATH + ".meta.json"
        if os.path.exists(meta):
            os.remove(meta)
        print("[INFO] All data reset.")

    init_metadata_db()

    backend = MemoryPipelineBackend()
    app = create_app(backend)

    print("=" * 60)
    print("[INFO] HugeGraph Memory Backend Server")
    print(f"[INFO] Graph: {HUGEGRAPH_URL} (graph={HUGEGRAPH_GRAPH})")
    print(f"[INFO] Vector: FAISS (dim=1536)")
    print(f"[INFO] LLM: {LLM_BASE_URL} ({LLM_MODEL})")
    print(f"[INFO] DB: {DB_PATH}")
    print(f"[INFO] http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=False)
