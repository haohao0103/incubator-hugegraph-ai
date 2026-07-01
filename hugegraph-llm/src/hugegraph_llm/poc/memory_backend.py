"""
HugeGraph Memory Backend — Engineering-grade AI Memory Server
==========================================================
Architecture (GraphRAG-enhanced, aligned with PowerMem v1.1.2):
  Graph Storage:  HugeGraph 1.7.0 (via pyhugegraph-python-client)
  Vector Index:   FAISS (semantic search) + BM25 (fulltext search) + RRF fusion
  LLM Engine:     DeepSeek Reasoner API (entity extract / rank / generate)
  Provenance:     Memory → Entity → Chunk source tracking

Retrieval Pipeline (3-channel RRF fusion):
  Channel 1: FAISS vector semantic search (with Ebbinghaus decay weighting)
  Channel 2: BM25 fulltext keyword search (jieba tokenization)
  Channel 3: Graph context score (entity/edge relevance to query)
  Fusion:     Reciprocal Rank Fusion (k=60) → unified ranked results

Storage mapping vs PowerMem SQLite:
  memories table → FAISS index + BM25 index + SQLite metadata (Ebbinghaus scores)
  nodes table    → HugeGraph Vertices (person/organization/location/skill/concept)
  edges table    → HugeGraph Edges (works_at/lives_in/likes/colleague_of/friend_of)

Pipeline alignment:
  ADD:  LLMExtract→SelfResolve→ConflictDetect→EntityResolution→RelComplete→ColleagueInfer→Store(7步)
  QUERY: Classify→3ChRetrieve(Ebbinghaus+FAISS+BM25+Graph)→RRFFuse→GraphDirectReason/LLMAnswer(4步)

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
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import faiss
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# GraphRAG components — production-grade retrieval operators
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from hugegraph_llm.indices.fulltext.bm25_fulltext import BM25FullTextBackend
    from hugegraph_llm.indices.rerank_index import get_reranker
    from hugegraph_llm.operators.graph_op.rrf_fusion import (
        ReciprocalRankFusion, fuse_results_with_scores
    )
    HAS_GRAPHRAG_OPS = True
except ImportError:
    HAS_GRAPHRAG_OPS = False
    print("[WARN] GraphRAG operators not available, falling back to FAISS-only search",
          file=sys.stderr, flush=True)

# ============================================================================
# Config (unified via memory_config)
# ============================================================================

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.poc.memory_distillation import DistillationPipeline
from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.audit_log import get_audit_logger
from hugegraph_llm.engines.memory import (
    MemoryScope,
    PrivacyLevel,
    AccessPermission,
    ImportanceEvaluator,
    EbbinghausDecay,
    EntityExtractor,
    QueryRewriteEngine,
    HugeGraphGraphStore,
    AdditiveExtractionPipeline,
    content_hash_md5,
    MemoryHistoryTracker,
    score_and_rank,
    compute_entity_boosts,
    extract_query_entities_simple,
    get_bm25_params,
    ENTITY_BOOST_WEIGHT,
    LLMQueryRewriteEngine,
    RouteStore,
    compute_routing_key,
    UserProfileStore,
    UserProfile,
    TopicExtractor,
    ProfileInjector,
    FaissDeletableIndex,
    MemoryCompressor,
    AgentMemoryManager,
    PermissionChecker,
    PrivacyFilter,
    CollaborationBroker,
)

HUGEGRAPH_URL = memory_settings.hugegraph_url
HUGEGRAPH_USER = memory_settings.hugegraph_user
HUGEGRAPH_PASS = memory_settings.hugegraph_pwd
HUGEGRAPH_GRAPH = memory_settings.hugegraph_graph

LLM_BASE_URL = memory_settings.llm_base_url
LLM_MODEL = memory_settings.llm_model
LLM_API_KEY = memory_settings.llm_api_key

# Ebbinghaus constants (same as PowerMem)
EBBINGHAUS_K = memory_settings.ebbinghaus_k
EBBINGHAUS_REINFORCE = memory_settings.ebbinghaus_reinforce

DB_PATH = memory_settings.resolve_db_path()
FAISS_INDEX_PATH = memory_settings.resolve_faiss_path()

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
1. 实体类型：person(人名)、organization(组织机构)、location(地点)、skill(技能/爱好)、concept(概念/事物)、time(时间/日期)、event(事件)、feeling(情感/情绪)、number(数字/数量)
   如果有明确的不属于上述类型的实体，可以自定义类型（如product、project、hobby、identity、certification等）
   **重要**：务必提取所有时间、日期、年份信息为 time 类型实体（如 "7 May 2023"、"2022"、"yesterday"）
   **重要**：务必提取具体数量为 number 类型实体（如 "300 cities"、"12 RegionServers"）
2. 关系类型：自由提取，不需要局限于预定义类型。例如：works_at(在...工作)、lives_in(住在...)、likes(喜欢)、located_in(位于)、manages(管理)、attends(参加)、founded(创立)、owns(拥有)、member_of(成员)、happened_on(发生在某时间)、studied(学习)、researched(研究)、painted(画了)、keen_on(热衷于)、went_to(去了)、supports(支持)等
   如果文本中存在语义关系但无现成类型名，请自造合理的英文关系名（如 "reports_to"、"participates_in"、"identity_is"、"certified_in"）
3. 人物识别："我叫XX" / "I am XX" / "My name is XX" 表示说话人名字是XX，"我的同事XX" / "My colleague XX" 表示同事名字是XX。直接提取具体人名，不要用代词。
4. 推理能力：如果文本说"我的同事也在腾讯"/"my colleague is also at Tencent"，推断该同事也在腾讯工作
5. 如果文本中同时出现了说话人名字和"我/我的/I/my"，用说话人名字替代代词
6. **多语言支持**：文本可能是中文或英文，请按原文语言提取实体名和关系名。英文实体名保持英文原文。

请严格按以下JSON格式输出，不要输出其他内容：
{
    "entities": [
        {"name": "实体名", "type": "实体类型"}
    ],
    "relationships": [
        {"source": "源实体名", "relationship": "关系类型", "target": "目标实体名"}
    ]
}

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
        self._entity_cache = {}  # (label, name) -> vertex_id

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
            s.vertexLabel(label).properties("name", "type").usePrimaryKeyId().primaryKeys("name").ifNotExist().create()
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

        def _existing_labels():
            """Return (src_labels, tgt_labels) for the given edge label if it exists."""
            try:
                existing = s.getEdgeLabel(edge_label)
                if existing is None:
                    return None
                srcs = existing.sourceLabel if hasattr(existing, "sourceLabel") else existing.get("source_label", [])
                tgts = existing.targetLabel if hasattr(existing, "targetLabel") else existing.get("target_label", [])
                if isinstance(srcs, str):
                    srcs = [srcs]
                if isinstance(tgts, str):
                    tgts = [tgts]
                return srcs, tgts
            except Exception as e2:
                print(f"[Schema] getEdgeLabel check failed for '{edge_label}': {e2}", file=sys.stderr)
                return None

        # Step 1: Check if the label already exists and is compatible.
        existing = _existing_labels()
        if existing:
            srcs, tgts = existing
            if src_label in srcs and tgt_label in tgts:
                self._edge_cache[cache_key] = edge_label
                return edge_label
            print(f"[Schema] Edge label '{edge_label}' exists but incompatible "
                  f"(has {srcs}->{tgts}, need {src_label}->{tgt_label})", file=sys.stderr)
        else:
            print(f"[Schema] Edge label '{edge_label}' not found, creating new", file=sys.stderr)

        # Step 2: Try to create the label directly (or a variant if incompatible).
        # IMPORTANT: ifNotExist() returns success even when a label with the
        # same NAME already exists with incompatible source/target.  We MUST
        # verify after creation that the label actually supports our combo.
        candidates = [edge_label] if existing is None else [f"{edge_label}_v{i}" for i in range(2, 10)]
        for candidate in candidates:
            try:
                s.edgeLabel(candidate).sourceLabel(src_label).targetLabel(tgt_label).ifNotExist().create()

                # --- Verify the created (or pre-existing) label is compatible ---
                try:
                    verify = s.getEdgeLabel(candidate)
                    # Parse source/target from the response (varies by client version)
                    v_srcs = (getattr(verify, "sourceLabel", None) or
                              getattr(verify, "source_label", None))
                    v_tgts = (getattr(verify, "targetLabel", None) or
                              getattr(verify, "target_label", None))
                    if isinstance(v_srcs, str):
                        v_srcs = [v_srcs]
                    if isinstance(v_tgts, str):
                        v_tgts = [v_tgts]
                    if v_srcs and v_tgts and src_label not in v_srcs:
                        # Label exists but with INcompatible source — skip this candidate
                        print(f"[Schema] Candidate '{candidate}' exists as {v_srcs}->{v_tgts}, "
                              f"incompatible with {src_label}->{tgt_label}; trying next",
                              file=sys.stderr)
                        continue
                    if v_srcs and v_tgts and tgt_label not in v_tgts:
                        print(f"[Schema] Candidate '{candidate}' exists as {v_srcs}->{v_tgts}, "
                              f"incompatible with {src_label}->{tgt_label}; trying next",
                              file=sys.stderr)
                        continue
                except Exception:
                    # Verification failed — but ifNotExist().create() said OK,
                    # assume the label is fresh and compatible
                    pass

                self._edge_cache[cache_key] = candidate
                if candidate != edge_label:
                    print(f"[HugeGraph] Edge label '{edge_label}' conflicts, using variant '{candidate}' "
                          f"({src_label}->{tgt_label})", file=sys.stderr)
                return candidate
            except Exception as e3:
                print(f"[Schema] Candidate '{candidate}' failed: {e3}", file=sys.stderr)
                continue

        # Step 3: All attempts failed — fall back to original and let addEdge report the error.
        self._edge_cache[cache_key] = edge_label
        return edge_label

    def add_vertex(self, label: str, name: str, properties: dict = None) -> Optional[str]:
        """Add a vertex, return its ID. Upsert by name. Creates label on-demand.

        Uses PRIMARY_KEY strategy on 'name'; do NOT pass an explicit id.
        """
        if not self._ensure_vertex_label(label):
            return None

        cache_key = (label, name)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        g = self.client.graph()
        props = {"name": name, "type": label}
        if properties:
            props.update(properties)

        # Check if exists by name
        existing = self.get_vertex_by_name(name)
        if existing:
            self._entity_cache[cache_key] = existing["id"]
            return existing["id"]

        try:
            v = g.addVertex(label, props)
            if v:
                self._entity_cache[cache_key] = v.id
                return v.id
        except Exception as e:
            print(f"[HugeGraph] add_vertex error: {e} (label={label}, name={name})", file=sys.stderr)
        return None

    def get_vertex_by_name(self, name: str) -> Optional[dict]:
        """Get a vertex by its 'name' property via REST (no Gremlin fallback)."""
        g = self.client.graph()
        for label in VERTEX_LABELS:
            try:
                vertices = g.getVertexByCondition(label=label, limit=200)
            except Exception:
                vertices = None
            if vertices:
                for v in vertices:
                    props = getattr(v, 'properties', {}) or getattr(v, 'property', {})
                    if isinstance(props, dict) and props.get('name') == name:
                        return {"id": v.id, "label": v.label, "properties": props}
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

        # Skip duplicate-edge check to avoid Gremlin dependency on HugeGraph 1.7.0.
        # The graph will tolerate multiple edges; deduplication can be done at
        # retrieval time if needed.

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

    def get_all_vertices(self, limit: int = 500) -> list:
        """Get all vertices for visualization using REST API directly."""
        vertices = []
        try:
            g = self.client.graph()
            session = g._sess
            resp = session.request(f"graph/vertices?limit={limit}&page")
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
    """FAISS-based vector index for semantic memory search.

    Uses local sentence-transformers model for real embeddings.
    """

    _model = None

    def __init__(self, dim: int = 384, index_path: str = FAISS_INDEX_PATH,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.dim = dim
        self.index_path = index_path
        self.model_name = model_name
        # Use Inner Product (cosine similarity friendly) or L2
        self.index = faiss.IndexFlatIP(dim)  # Inner Product for cosine sim
        self.metadata = []  # list of {memory_id, content, created_at}
        self._load_model()

    def _load_model(self):
        if FaissMemoryIndex._model is None:
            from sentence_transformers import SentenceTransformer
            FaissMemoryIndex._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding vector for text using local sentence-transformers model."""
        emb = FaissMemoryIndex._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)

    def add_memory(self, memory_id: str, content: str, created_at: float = None,
                   embed_text_override: str = None):
        """Add a memory to the FAISS index.

        Args:
            content: text to store in metadata (used for answer verification)
            embed_text_override: if provided, embed this text instead of content.
                Used when content is raw dialogue but we want to embed factual summaries.
        """
        embed_source = embed_text_override or content
        vec = self.embed_text(embed_source)
        # Reshape for FAISS: (1, dim)
        vec = vec.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.metadata.append({
            "memory_id": memory_id,
            "content": content,
            "embed_text": embed_source,
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

    def delete_memory(self, memory_id: str):
        """Rebuild the index without the given memory_id."""
        new_metadata = [m for m in self.metadata if m.get("memory_id") != memory_id]
        if len(new_metadata) == len(self.metadata):
            return
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        for meta in new_metadata:
            vec = self.embed_text(meta["content"]).reshape(1, -1).astype(np.float32)
            self.index.add(vec)
            self.metadata.append({
                "memory_id": meta["memory_id"],
                "content": meta["content"],
                "created_at": meta.get("created_at", time.time()),
                "index_pos": len(self.metadata),
            })


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
    """Initialize metadata database schema (memories + persona)."""
    db = get_metadata_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'demo_user',
            created_at REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            initial_score REAL DEFAULT 1.0,
            agent_id TEXT,
            run_id TEXT,
            scope TEXT DEFAULT 'private',
            privacy TEXT DEFAULT 'standard',
            importance REAL DEFAULT 0.5,
            metadata TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);

        CREATE TABLE IF NOT EXISTS personas (
            user_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL DEFAULT 0
        );
    """)
    # Schema migration: add new columns if upgrading an old DB file
    existing = {row[1] for row in db.execute("PRAGMA table_info(memories)")}
    migrations = [
        ("agent_id", "ALTER TABLE memories ADD COLUMN agent_id TEXT"),
        ("run_id", "ALTER TABLE memories ADD COLUMN run_id TEXT"),
        ("scope", "ALTER TABLE memories ADD COLUMN scope TEXT DEFAULT 'private'"),
        ("privacy", "ALTER TABLE memories ADD COLUMN privacy TEXT DEFAULT 'standard'"),
        ("importance", "ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5"),
        ("metadata", "ALTER TABLE memories ADD COLUMN metadata TEXT"),
    ]
    for col, sql in migrations:
        if col not in existing:
            try:
                db.execute(sql)
            except Exception as e:
                log.warning("Metadata DB migration for %s failed: %s", col, e)
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
    if not response.choices:
        return ""
    msg = response.choices[0].message
    return (msg.content or "").strip() or (msg.reasoning_content or "").strip()


# ============================================================================
# Main Pipeline Backend
# ============================================================================

class MemoryPipelineBackend:
    """
    Engineering-grade Memory Pipeline — HugeGraph + FAISS + BM25 + RRF + MiMo LLM.
    Aligned with PowerMem v1.1.2 add_memory (7-step) / search_memory (4-step).
    Enhanced with GraphRAG operators: BM25 fulltext search, RRF fusion, provenance.
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
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY environment variable is required. Please set it before running memory_backend.")

        # P0: BM25 fulltext index (GraphRAG component)
        self._bm25 = None
        if HAS_GRAPHRAG_OPS:
            try:
                self._bm25 = BM25FullTextBackend()
                # Restore BM25 from persistent storage if available
                bm25_dir = os.path.dirname(os.path.abspath(__file__))
                bm25_pkl_path = os.path.join(bm25_dir, "memory_bm25.pkl")
                if os.path.exists(bm25_pkl_path):
                    self._bm25 = BM25FullTextBackend.from_name(bm25_dir, "memory_bm25")
                print(f"[BM25] Initialized, {self._bm25.doc_count} docs in index",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[BM25] Init error (non-critical): {e}",
                      file=sys.stderr, flush=True)

        # P0: Additive hybrid scoring (mem0-style) replaces old RRF rank fusion
        # 借鉴自: mem0/utils/scoring.py score_and_rank
        self._additive_scoring_available = HAS_GRAPHRAG_OPS

        # P0: Optional cross-encoder / API reranker
        self._reranker = get_reranker() if HAS_GRAPHRAG_OPS else None

        # P1: Provenance tracking (memory_id → [{entity, relation}])
        self._provenance: Dict[str, List[Dict[str, str]]] = {}
        self._provenance_db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "memory_provenance.json")
        self._load_provenance()

        # P1: Experience + Skill distillation pipeline
        self._distillation = DistillationPipeline(llm_client=None)

        # P2: Intelligence / lifecycle components (PowerMem-style)
        self._importance_evaluator = ImportanceEvaluator()
        self._ebbinghaus = EbbinghausDecay()
        self._entity_extractor = EntityExtractor()

        # Ensure metadata SQLite schema exists when used as a library.
        init_metadata_db()

        # P0: Audit logging (PowerMem-style telemetry/audit)
        self._audit = get_audit_logger()

        # P0: Entity-centric graph store for multi-hop retrieval
        # 借鉴自: engines/memory/graph_store.py HugeGraphGraphStore
        self._graph_store = HugeGraphGraphStore(self.hg, max_hops=2, max_neighbors=50)

        # P0: Memory history tracking (mem0-style SQLite history)
        # 借鉴自: mem0/memory/storage.py SQLiteManager + engines/memory/memory_history.py
        self._history_db_path = os.path.join(os.path.dirname(DB_PATH), "memory_history.db")
        self._history = MemoryHistoryTracker(db_path=self._history_db_path)

        # P0: V3 additive extraction pipeline (mem0-style ADD-only extraction)
        # 借鉴自: mem0/memory/main.py _add_to_vector_store + engines/memory/additive_extraction.py
        self._additive_pipeline = AdditiveExtractionPipeline(
            llm_callback=self._additive_llm_callback
        )

        # P1: LLM-enhanced query rewrite engine (PowerMem QueryRewriter aligned)
        # 借鉴自: engines/memory/llm_query_rewrite.py LLMQueryRewriteEngine
        self._query_rewrite = LLMQueryRewriteEngine()

        # P1: Sub-store routing (PowerMem SubStore aligned)
        # 借鉴自: engines/memory/sub_store_routing.py RouteStore
        self._route_store = RouteStore(
            base_dir=os.path.join(memory_settings.memory_data_dir, "route_shards")
        )

        # P1: User profile store (PowerMem UserMemory aligned)
        # 借鉴自: engines/memory/user_profile.py UserProfileStore + TopicExtractor
        self._user_profile = UserProfileStore(
            db_path=os.path.join(memory_settings.memory_data_dir, "user_profile.db")
        )
        self._profile_injector = ProfileInjector(profile_store=self._user_profile)

        # P2: Faiss deletable index (PowerMem efficient delete aligned)
        # 借鉴自: engines/memory/faiss_deletable.py FaissDeletableIndex
        # Auxiliary index for compression workflows; primary index remains self.faiss
        self._faiss_deletable = FaissDeletableIndex(
            dim=memory_settings.embedding_dim,
            model_name=memory_settings.embedding_model or "all-MiniLM-L6-v2",
        )

        # P2: Memory compressor (PowerMem MemoryOptimizer aligned)
        # 借鉴自: engines/memory/memory_compressor.py MemoryCompressor
        self._compressor = MemoryCompressor(llm_callback=self._compressor_llm_callback)

        # P2: Multi-agent collaboration / permissions (PowerMem AgentMemory aligned)
        # 借鉴自: engines/memory/agent_collaboration.py AgentMemoryManager
        self._agent_manager = AgentMemoryManager(
            collaboration_broker=CollaborationBroker(
                db_path=os.path.join(memory_settings.memory_data_dir, "agent_permissions.db")
            )
        )

    def _additive_llm_callback(self, prompt: str) -> str:
        """LLM callback wrapper for AdditiveExtractionPipeline."""
        client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a memory extraction engine."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=2048,
            )
            return _get_llm_text(response)
        except Exception as e:
            print(f"[Additive LLM error] {e}", file=sys.stderr, flush=True)
            return ""

    def _compressor_llm_callback(self, prompt: str) -> str:
        """LLM callback wrapper for MemoryCompressor summaries."""
        client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=256,
            )
            return _get_llm_text(response)
        except Exception as e:
            print(f"[Compressor LLM error] {e}", file=sys.stderr, flush=True)
            return ""
    def _load_provenance(self):
        """Load provenance tracking data from disk."""
        try:
            if os.path.exists(self._provenance_db_path):
                with open(self._provenance_db_path, "r") as f:
                    self._provenance = json.load(f)
        except Exception:
            self._provenance = {}

    def _save_provenance(self):
        """Persist provenance tracking data to disk."""
        try:
            with open(self._provenance_db_path, "w") as f:
                json.dump(self._provenance, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Provenance] Save error: {e}", file=sys.stderr, flush=True)

    def _track_provenance(self, memory_id: str, entities: list, relationships: list):
        """Track which entities/relationships were extracted from which memory."""
        links = []
        for ent in entities:
            links.append({"entity": ent["name"], "type": ent.get("type", ""),
                          "relation": "extracted_from"})
        for rel in relationships:
            links.append({"entity": rel["source"], "type": "source",
                          "relation": rel["relationship"],
                          "target": rel["target"]})
        self._provenance[memory_id] = links
        self._save_provenance()

    def _get_provenance_for_entities(self, entity_names: list) -> List[Dict]:
        """Get source memories that contributed to the given entities."""
        sources = []
        for mem_id, links in self._provenance.items():
            for link in links:
                if link.get("entity") in entity_names or link.get("target") in entity_names:
                    sources.append({"memory_id": mem_id, "link": link})
                    break  # one match per memory is enough
        return sources

    def _get_stored_hashes(self, user_id: str) -> set:
        """Return MD5 hashes of all stored memories for a user (mem0-style dedup)."""
        db = get_metadata_db()
        try:
            rows = db.execute(
                "SELECT content FROM memories WHERE user_id=?", (user_id,)
            ).fetchall()
            return {content_hash_md5(row["content"]) for row in rows}
        finally:
            db.close()

    def _build_factual_summary(self, content: str,
                                entities: list, relationships: list) -> str:
        """Build a factual summary from entities and relationships for BM25/FAISS indexing.

        The raw dialogue text is too long and noisy for effective retrieval.
        This method constructs a concise factual summary that captures the key
        information, making BM25 keyword search and FAISS semantic search much
        more effective.

        Strategy:
        1. If entities/relationships exist → build "EntityA relB EntityC" sentences
        2. Append key sentences from content that contain factual info
        3. If no structured data → extract top sentences by information density
        """
        parts = []

        # 1. Entity-relationship sentences (most reliable facts)
        if relationships:
            for rel in relationships[:20]:
                src = rel.get("source", "")
                tgt = rel.get("target", "")
                label = rel.get("relationship", "")
                if src and tgt and label:
                    parts.append(f"{src} {label} {tgt}")

        # 2. Entity name+type facts
        if entities:
            for ent in entities[:15]:
                name = ent.get("name", "")
                typ = ent.get("type", "")
                if name:
                    parts.append(f"{name} ({typ})")

        # 3. Extract key factual sentences from content
        #    Skip conversational fillers (greetings, acknowledgments, etc.)
        if content:
            # Split multi-turn dialogue into individual sentences
            sentences = []
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove speaker prefix (e.g., "Caroline: ", "Melanie: ")
                colon_pos = line.find(": ")
                if colon_pos > 0 and colon_pos < 15:
                    line = line[colon_pos + 2:]
                # Split into sentences
                for sent in re.split(r'[.!?。！？]', line):
                    sent = sent.strip()
                    if len(sent) >= 10:  # Skip very short filler
                        # Skip obvious conversational filler
                        if re.match(r'^(hey|hi|hello|good to see|nice to|thanks|thank you|bye|goodbye|ok|okay|sure|yeah|yes|no|hmm|wow|cool|great|awesome|lol|haha)', sent.lower()):
                            continue
                        # Skip if just acknowledgments
                        if len(sent) < 20 and not re.search(r'\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|yesterday|today|last|ago', sent.lower()):
                            continue
                        sentences.append(sent)

            # Pick sentences with highest information density
            # Prioritize sentences containing entities, dates, or specific facts
            entity_names = {e.get("name", "") for e in entities} if entities else set()
            scored_sentences = []
            for sent in sentences[:50]:
                score = 0
                # Contains an extracted entity name
                for en in entity_names:
                    if en.lower() in sent.lower():
                        score += 3
                # Contains date/time patterns
                if re.search(r'\d{4}|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|yesterday|last\s+(week|month|year)|\d+\s+(days|weeks|months|years)\s+ago', sent.lower()):
                    score += 2
                # Contains specific quantities
                if re.search(r'\d+\s*(people|cities|kids|books|years|months|hours|times|members|servers)', sent.lower()):
                    score += 2
                # Longer sentences tend to have more info
                score += len(sent) / 50
                scored_sentences.append((score, sent))

            scored_sentences.sort(reverse=True)
            # Add top 10 factual sentences
            for _, sent in scored_sentences[:10]:
                if sent not in parts:  # Avoid duplication with entity sentences
                    parts.append(sent)

        # 4. If still no content, use raw content excerpt (last resort)
        if not parts and content:
            # Take first 200 chars as a compact summary
            parts.append(content[:200].strip())

        summary = "; ".join(parts)
        # Cap at 500 chars to keep embedding quality high
        if len(summary) > 500:
            summary = summary[:500]
        return summary
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

    def extract_entities_relationships(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Public wrapper for the LLM entity/relation extraction step.
        Useful for callers that want to parallelize extraction before storing.
        """
        return self._llm_extract(content)

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
        """Generate answer using LLM based on memories and graph context.
        Falls back to rule-based answer when LLM output is clearly garbage."""
        try:
            client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
            memory_text = "\n".join([f"- {m['content']}" for m in memories])
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content":
                     "你是HugeGraph Memory助手。只根据提供的记忆内容直接回答用户问题。"
                     "回答1-2句话，不要输出编号列表，不要复述规则，不要输出推理过程。"},
                    {"role": "user", "content":
                     f"用户问题：{query}\n\n相关记忆：\n{memory_text}\n\n"
                     f"请直接回答问题，只用记忆中的信息。"},
                ],
                temperature=0.3,
                max_completion_tokens=128,
            )
            content = _get_llm_text(response)
            if not content:
                content = ""
            # Post-process: strip MiMo chain-of-thought reasoning
            content = self._strip_reasoning(content)
            # Detect prompt-regurgitation: if answer starts with numbered rules, fall back
            if self._is_prompt_regurgitation(content):
                content = self._rule_based_answer(query, memories)
            return content if content else self._rule_based_answer(query, memories)
        except Exception as e:
            print(f"[LLM answer error] {e}", file=sys.stderr, flush=True)
            return self._rule_based_answer(query, memories)

    def _is_prompt_regurgitation(self, text: str) -> bool:
        """Detect if the LLM output is regurgitating prompt rules instead of answering."""
        if not text:
            return True
        # Starts with numbered rule pattern like "1." or "2."
        if re.match(r'^\d+\.\s', text):
            return True
        # Contains meta-rule keywords that shouldn be in an answer
        meta_keywords = ['不能跨记忆推断', '不能编造', '只根据记忆回答', '不要输出推理过程',
                         '回答规则', '严格按', '绝不能']
        for kw in meta_keywords:
            if kw in text:
                return True
        return False

    def _rule_based_answer(self, query: str, memories: list) -> str:
        """Simple rule-based answer from search results when LLM fails."""
        if not memories:
            return "记忆中没有这个信息。"
        # Just present the most relevant memory content directly
        best = memories[0]
        content = best.get('content', '')
        # Try to match common query patterns
        if '同事' in query and '同事' in content:
            return f"根据记忆：{content}"
        if '偏好' in query or '喜欢' in query or '爱好' in query:
            relevant = [m for m in memories if any(kw in m.get('content','') for kw in ['偏好','喜欢','爱好','偏好使用'])]
            if relevant:
                return f"根据记忆：{relevant[0]['content']}"
        if '谁' in query or '哪' in query:
            return f"根据记忆找到：{content}"
        # Generic fallback
        return f"最相关的记忆：{content}"

    # ---- ADD Pipeline (7 steps, aligned with PowerMem) ----

    def add_memory(
        self,
        content: str,
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        scope: MemoryScope = MemoryScope.PRIVATE,
        privacy: PrivacyLevel = PrivacyLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None,
        skip_index_save: bool = False,
    ) -> dict:
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

        # P0: V3 additive extraction + MD5 hash dedup (mem0-style ADD-only pipeline)
        # 借鉴自: mem0/memory/main.py _add_to_vector_store + engines/memory/additive_extraction.py
        step_start = time.time()
        existing_rows = db.execute(
            "SELECT id, content FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        stored_hashes = self._get_stored_hashes(user_id)
        existing_memories = [ex["content"] for ex in existing_rows]
        additive_result = self._additive_pipeline.run(
            content, stored_hashes=stored_hashes, existing_memories=existing_memories
        )
        new_facts = additive_result.get("new_facts", [])
        dup_facts = additive_result.get("duplicate_facts", [])
        if not new_facts:
            db.close()
            total_elapsed = round((time.time() - start_time) * 1000)
            result = {
                "memory_id": None,
                "action": "SKIP",
                "reason": "V3 additive dedup: all facts duplicate",
                "trace": trace + [{"step": 1.8, "name": "V3 ADD-only\u53bb\u91cd",
                                    "detail": "all facts duplicate, skip insert",
                                    "elapsed_ms": round((time.time()-step_start)*1000)}],
                "entities": additive_result.get("entities", []),
                "relationships": [],
                "total_elapsed_ms": total_elapsed,
            }
            try:
                self._audit.log(
                    operation="add_memory",
                    user_id=user_id,
                    memory_id=None,
                    content=content[:2000] if content else None,
                    latency_ms=total_elapsed,
                    success=True,
                    metadata={"action": "SKIP", "reason": "V3 additive dedup"},
                )
            except Exception:
                pass
            return result
        if dup_facts:
            content = "\uff1b".join(new_facts)
            trace.append({"step": 1.8, "name": "V3 ADD-only\u53bb\u91cd",
                          "detail": f"{len(new_facts) + len(dup_facts)} facts \u2192 {len(new_facts)} new, {len(dup_facts)} dup",
                          "elapsed_ms": round((time.time()-step_start)*1000)})
        else:
            trace.append({"step": 1.8, "name": "V3 ADD-only\u63d0\u53d6",
                          "detail": f"{len(new_facts)} facts, no duplicates",
                          "elapsed_ms": round((time.time()-step_start)*1000)})

        # P2: importance scoring and Ebbinghaus retention (PowerMem-style)
        importance = self._importance_evaluator.score(content)
        initial_score = max(importance, 0.3)

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
        # existing_rows already fetched during V3 additive extraction

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
            # Store entities in metadata for search answer construction
            enriched_metadata = dict(metadata or {})
            enriched_metadata["entities"] = entities
            enriched_metadata["relationships"] = relationships
            db.execute(
                "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
                "initial_score,agent_id,run_id,scope,privacy,importance,metadata)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory_id, content, user_id, now, now, 1,
                    initial_score,
                    agent_id,
                    run_id,
                    scope.value,
                    privacy.value,
                    importance,
                    json.dumps(enriched_metadata, ensure_ascii=False),
                ),
            )
            # Build factual summary for FAISS embedding and BM25 indexing
            # Raw dialogue is too noisy; embed/search factual summaries instead
            factual_summary = self._build_factual_summary(content, entities, relationships)
            # FAISS: embed factual summary, store raw content in metadata for answer verification
            self.faiss.add_memory(memory_id, content, now, embed_text_override=factual_summary)
            if not skip_index_save:
                try:
                    self.faiss.save()
                except Exception:
                    pass

            # P0: BM25 fulltext indexing — index factual summary, not raw dialogue
            if self._bm25 is not None and not skip_index_save:
                try:
                    self._bm25.add_documents([factual_summary], [memory_id])
                    bm25_dir = os.path.dirname(os.path.abspath(__file__))
                    self._bm25.save_index_by_name(bm25_dir, "memory_bm25")
                except Exception as e:
                    print(f"[BM25] Add error: {e}", file=sys.stderr, flush=True)

            # P1: Sub-store routing (PowerMem SubStore aligned)
            # 借鉴自: engines/memory/sub_store_routing.py RouteStore.add_memory
            try:
                self._route_store.add_memory(
                    memory_id, content,
                    {"user_id": user_id, "agent_id": agent_id, "scope": scope.value}
                )
            except Exception as e:
                print(f"[RouteStore] Add error: {e}", file=sys.stderr, flush=True)

            # P1: User profile update (PowerMem UserMemory aligned)
            # 借鉴自: engines/memory/user_profile.py UserProfileStore.update_from_memories
            try:
                self._user_profile.update_from_memories(user_id, [content])
            except Exception as e:
                print(f"[UserProfile] Update error: {e}", file=sys.stderr, flush=True)

            # P1: Provenance tracking
            self._track_provenance(memory_id, entities, relationships)

            # P0: Memory history tracking (mem0-style ADD event)
            # 借鉴自: mem0/memory/storage.py SQLiteManager.batch_add_history
            self._history.add_history(
                memory_id=memory_id,
                event="ADD",
                new_memory=content,
                actor_id=user_id or agent_id,
                role="user",
                metadata={"entities": entities, "relationships": stored_edges},
            )

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

        result = {
            "memory_id": memory_id if action != "SKIP" else None,
            "action": action,
            "reason": conflict_reason or conflict_detail or
                    ("\u65b0\u8bb0\u5fc6\uff0c\u65e0\u51b2\u7a81" if action == "ADD" else conflict_reason),
            "trace": trace,
            "entities": entities,
            "relationships": stored_edges,
            "total_elapsed_ms": total_elapsed,
        }

        # Audit log
        try:
            self._audit.log(
                operation="add_memory",
                user_id=user_id,
                memory_id=result["memory_id"],
                content=content[:2000] if content else None,
                latency_ms=result["total_elapsed_ms"],
                success=True,
                metadata={"action": action, "entity_count": len(entities), "edge_count": len(stored_edges)},
            )
        except Exception:
            pass

        return result

    def add_memory_bypass_classify(
        self,
        content: str,
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        scope: MemoryScope = MemoryScope.PRIVATE,
        privacy: PrivacyLevel = PrivacyLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[Dict[str, str]]] = None,
        relationships: Optional[List[Dict[str, str]]] = None,
        skip_index_save: bool = False,
    ) -> dict:
        """
        Same as add_memory but bypasses intent classification.
        Useful for benchmark ingestion where every input is known to be a fact
        to remember rather than a question to answer.

        If ``entities`` and ``relationships`` are provided, the LLM extraction step
        is skipped entirely. This is useful for benchmarking where extraction
        can be parallelized outside the critical path.
        """
        start_time = time.time()
        db = get_metadata_db()
        now = time.time()
        memory_id = str(uuid.uuid4())[:8]
        trace = []

        # P0: V3 additive extraction + MD5 hash dedup (mem0-style ADD-only pipeline)
        # 借鉴自: mem0/memory/main.py _add_to_vector_store + engines/memory/additive_extraction.py
        step_start = time.time()
        existing_rows = db.execute(
            "SELECT id, content FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        stored_hashes = self._get_stored_hashes(user_id)
        existing_memories = [ex["content"] for ex in existing_rows]
        # When entities/relationships are pre-extracted (benchmark mode), skip the
        # LLM extraction step and only do lightweight MD5 hash dedup.  This avoids
        # a full LLM call per benchmark batch — the primary bottleneck when using
        # reasoning models.  The pre-extracted entities already capture the semantics.
        if entities is not None and relationships is not None:
            from hugegraph_llm.engines.memory.additive_extraction import content_hash_md5
            content_hash = content_hash_md5(content)
            if content_hash in stored_hashes:
                new_facts = []
                dup_facts = [content]
            else:
                new_facts = [content]
                dup_facts = []
                stored_hashes.add(content_hash)
        else:
            additive_result = self._additive_pipeline.run(
                content, stored_hashes=stored_hashes, existing_memories=existing_memories
            )
            new_facts = additive_result.get("new_facts", [])
            dup_facts = additive_result.get("duplicate_facts", [])

        if not new_facts:
            db.close()
            return {
                "memory_id": None,
                "action": "SKIP",
                "reason": "V3 additive dedup: all facts duplicate",
                "entities": [],
                "relationships": [],
                "total_elapsed_ms": round((time.time() - start_time) * 1000),
            }
        if dup_facts:
            content = "\uff1b".join(new_facts)
            trace.append({"step": 1.8, "name": "V3 ADD-only\u53bb\u91cd",
                          "detail": f"{len(new_facts) + len(dup_facts)} facts \u2192 {len(new_facts)} new, {len(dup_facts)} dup",
                          "elapsed_ms": round((time.time()-step_start)*1000)})
        else:
            trace.append({"step": 1.8, "name": "V3 ADD-only\u63d0\u53d6",
                          "detail": f"{len(new_facts)} facts, no duplicates",
                          "elapsed_ms": round((time.time()-step_start)*1000)})

        # P2: importance scoring (PowerMem-style)
        importance = self._importance_evaluator.score(content)
        initial_score = max(importance, 0.3)

        # Step 1: LLM Entity Extraction (skip when pre-extracted entities/relations are provided)
        step_start = time.time()
        if entities is not None and relationships is not None:
            extraction = {"entities": entities, "relationships": relationships}
        else:
            extraction = self._llm_extract(content)
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])
        trace.append({"step": 1, "name": "LLM实体抽取",
                      "detail": f"{len(entities)} entities, {len(relationships)} relations",
                      "elapsed_ms": round((time.time()-step_start)*1000),
                      "data": {"entities": entities, "relationships": relationships}})

        # Conflict detection (token-level Jaccard similarity, not char-level)
        # 借鉴自: mem0 的 event_dedup + PowerMem 的 content similarity
        # 旧版字符级重叠（s2>0.9）把不同对话片段误判为"几乎相同"
        # → 改为 token/word Jaccard 阈值 0.85，只跳过真正重复的内容
        action = "ADD"
        conflict_reason = ""
        existing_rows = db.execute(
            "SELECT id, content FROM memories WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        content_tokens = set(content.lower().split())
        for ex in existing_rows:
            ex_tokens = set(ex["content"].lower().split())
            if not content_tokens or not ex_tokens:
                continue
            intersection = len(content_tokens & ex_tokens)
            union = len(content_tokens | ex_tokens)
            jaccard = intersection / union if union > 0 else 0
            if jaccard > 0.95:  # Raised from 0.85 — English dialogue chunks share many common filler words
                action = "SKIP"
                conflict_reason = f"token Jaccard={round(jaccard*100)}%"
                break

        # Dedup + missing relations
        entities, relationships = self._dedup_entities(entities, relationships)
        relationships = self._extract_missing_rels(content, entities, relationships)

        # Store to HugeGraph + FAISS + SQLite
        stored_nodes = []
        stored_edges = []
        for ent in entities:
            vid = self.hg.add_vertex(ent["type"], ent["name"])
            if vid:
                stored_nodes.append({"name": ent["name"], "type": ent["type"], "id": vid})
        for rel in relationships:
            eid = self.hg.add_edge(rel["relationship"], rel["source"], rel["target"], {})
            if eid:
                stored_edges.append({**rel, "edge_id": eid})

        if action != "SKIP":
            # Store entities in metadata for search answer construction
            enriched_metadata = dict(metadata or {})
            enriched_metadata["entities"] = entities
            enriched_metadata["relationships"] = relationships
            db.execute(
                "INSERT INTO memories (id,content,user_id,created_at,last_accessed_at,access_count,"
                "initial_score,agent_id,run_id,scope,privacy,importance,metadata)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory_id, content, user_id, now, now, 1,
                    initial_score,
                    agent_id,
                    run_id,
                    scope.value,
                    privacy.value,
                    importance,
                    json.dumps(enriched_metadata, ensure_ascii=False),
                ),
            )
            # Build factual summary for FAISS embedding and BM25 indexing
            factual_summary = self._build_factual_summary(content, entities, relationships)
            self.faiss.add_memory(memory_id, content, now, embed_text_override=factual_summary)
            try:
                self.faiss.save()
            except Exception:
                pass
            if self._bm25 is not None:
                try:
                    self._bm25.add_documents([factual_summary], [memory_id])
                    bm25_dir = os.path.dirname(os.path.abspath(__file__))
                    self._bm25.save_index_by_name(bm25_dir, "memory_bm25")
                except Exception as e:
                    print(f"[BM25] Add error: {e}", file=sys.stderr, flush=True)

            # P1: Sub-store routing
            try:
                self._route_store.add_memory(
                    memory_id, content,
                    {"user_id": user_id, "agent_id": agent_id, "scope": scope.value}
                )
            except Exception as e:
                print(f"[RouteStore] Add error: {e}", file=sys.stderr, flush=True)

            # P1: User profile update
            try:
                self._user_profile.update_from_memories(user_id, [content])
            except Exception as e:
                print(f"[UserProfile] Update error: {e}", file=sys.stderr, flush=True)

            self._track_provenance(memory_id, entities, relationships)

            # P0: Memory history tracking (mem0-style ADD event)
            # 借鉴自: mem0/memory/storage.py SQLiteManager.batch_add_history
            self._history.add_history(
                memory_id=memory_id,
                event="ADD",
                new_memory=content,
                actor_id=user_id or agent_id,
                role="user",
                metadata={"entities": entities, "relationships": stored_edges},
            )

        db.commit()
        db.close()
        result = {
            "memory_id": memory_id if action != "SKIP" else None,
            "action": action,
            "reason": conflict_reason or "新记忆，无冲突",
            "entities": entities,
            "relationships": stored_edges,
            "total_elapsed_ms": round((time.time() - start_time) * 1000),
        }

        # Audit log
        try:
            self._audit.log(
                operation="add_memory_bypass_classify",
                user_id=user_id,
                memory_id=result["memory_id"],
                content=content[:2000] if content else None,
                latency_ms=result["total_elapsed_ms"],
                success=True,
                metadata={"action": action, "entity_count": len(entities), "edge_count": len(stored_edges)},
            )
        except Exception:
            pass

        return result

    def search_memory(
        self,
        query: str,
        user_id: str = "demo_user",
        top_k: int = 5,
        fast_eval: bool = False,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        update_access_count: bool = True,
    ) -> dict:
        """
        Search memories through the 4-step pipeline.
        Aligned with PowerMem MemoryStore.search_memory().

        Args:
            fast_eval: If True, skip all LLM calls (classify / rerank / answer)
                       and return retrieval-only results. Used for fast benchmarks.
        """
        start_time = time.time()
        db = get_metadata_db()
        now = time.time()
        trace = []

        # Step 1: Intent Classification (LLM primary + regex fallback)
        step_start = time.time()
        if fast_eval:
            classify_result = {"action": "QUERY", "method": "fast_eval",
                               "reason": "Benchmark fast-eval: bypass LLM classify"}
        else:
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
        cm = classify_result.get("method", "llm")
        trace.append({"step": 1, "name": "\u610f\u56fe\u5206\u7c7b",
                  "detail": f"{ca} ({cm})",
                  "elapsed_ms": round((time.time()-step_start)*1000),
                  "data": classify_result})

        if classify_result["action"] != "QUERY":
            db.close()
            try:
                self._audit.log(
                    operation="search_memory",
                    user_id=user_id,
                    query=query[:1000] if query else None,
                    latency_ms=round((time.time() - start_time) * 1000),
                    success=True,
                    metadata={"error": "NOT_A_QUERY", "method": classify_result.get("method")},
                )
            except Exception:
                pass
            return {"query": query, "error": "NOT_A_QUERY",
                    "hint": "\u8bf7\u8f93\u5165\u7591\u95ee\u53e5\u6765\u67e5\u8be2\u8bb0\u5fc6",
                    "trace": trace}

        # P1: LLM query rewrite + user profile injection (PowerMem QueryRewriter aligned)
        # 借鉴自: engines/memory/llm_query_rewrite.py LLMQueryRewriteEngine.rewrite
        # In fast_eval mode, skip LLM rewrite to avoid API blocking
        step_start = time.time()
        if fast_eval:
            # Rule-based rewrite only (no LLM call)
            rewrite_result = {"rewritten": query, "entities": [], "intent": "general", "method": "fast_eval_skip"}
            rewritten_query = query
            query_entities_extra = []
            query_intent = "general"
            rewrite_method = "fast_eval_skip"
        else:
            try:
                profile_ctx = self._profile_injector.get_profile_for_rewrite(user_id)
                rewrite_result = self._query_rewrite.rewrite(
                    query, context="", user_profile=profile_ctx.get("user_profile", "")
                )
                rewritten_query = rewrite_result.get("rewritten", query)
                query_entities_extra = [
                    e.get("name", "") if isinstance(e, dict) else str(e)
                    for e in rewrite_result.get("entities", [])
                    if (e.get("name", "") if isinstance(e, dict) else e)
                ]
                query_intent = rewrite_result.get("intent", "general")
                rewrite_method = rewrite_result.get("method", "rule")
            except Exception as e:
                print(f"[QueryRewrite] Error: {e}", file=sys.stderr, flush=True)
                rewritten_query = query
                query_entities_extra = []
                query_intent = "general"
                rewrite_method = "error_fallback"
        trace.append({"step": 1.5, "name": "查询改写与用户画像注入",
                      "detail": f"intent={query_intent}, method={rewrite_method}",
                      "elapsed_ms": round((time.time() - step_start)*1000),
                      "data": {"rewritten": rewritten_query, "intent": query_intent}})

        # Step 2: 3-Channel Retrieval + RRF Fusion (FAISS + BM25 + Graph)
        step_start = time.time()

        # Get all memories for Ebbinghaus calculation (with optional scope/agent/run filters)
        sql = (
            "SELECT id,content,created_at,last_accessed_at,access_count,initial_score,"
            "scope,privacy,importance "
            "FROM memories WHERE user_id=?"
        )
        params: List[Any] = [user_id]
        if agent_id is not None:
            sql += " AND (agent_id=? OR agent_id IS NULL)"
            params.append(agent_id)
        if run_id is not None:
            sql += " AND (run_id=? OR run_id IS NULL)"
            params.append(run_id)
        if filters:
            if filters.get("scope"):
                sql += " AND scope=?"
                params.append(filters["scope"])
            if filters.get("privacy"):
                sql += " AND privacy=?"
                params.append(filters["privacy"])
            if filters.get("memory_type"):
                sql += " AND (metadata LIKE ?)"
                params.append(f'%"memory_type": "{filters["memory_type"]}"%')
        sql += " ORDER BY created_at DESC"
        rows = db.execute(sql, params).fetchall()

        # Compute Ebbinghaus weights
        eb_weights = {}
        memories_map = {}  # id -> full memory data
        for row in rows:
            elapsed_hours = (now - row["created_at"]) / 3600
            ret = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            ret = min(1.0, max(0.0, ret + row["access_count"] * EBBINGHAUS_REINFORCE))
            eb_weights[row["id"]] = round(ret, 4)
            memories_map[row["id"]] = {"id": row["id"], "content": row["content"],
                                       "retention": ret, "access_count": row["access_count"]}

        # --- Channel 1: FAISS Vector Semantic Search (with Ebbinghaus) ---
        faiss_results = self.faiss.search(rewritten_query, top_k=top_k * 3, ebbinghaus_weights=eb_weights)
        # Build ranked list: memory_id ordered by FAISS relevance
        channel_faiss = [r["memory_id"] for r in faiss_results if r.get("memory_id")]

        # Add Ebbinghaus-only memories as low-ranked channel entries
        for mid, mem in memories_map.items():
            if mid not in channel_faiss and mem["retention"] > 0.1:
                channel_faiss.append(mid)

        # --- Channel 2: BM25 Fulltext Keyword Search ---
        channel_bm25 = []
        bm25_scores = {}
        if self._bm25 is not None and self._bm25.doc_count > 0:
            try:
                bm25_raw = self._bm25.search(rewritten_query, top_k=top_k * 3, min_score=0.0)
                for item in bm25_raw:
                    mid = item.get("id")
                    if mid:
                        channel_bm25.append(mid)
                        bm25_scores[mid] = item.get("score", 0.0)
            except Exception as e:
                print(f"[BM25] Search error: {e}", file=sys.stderr, flush=True)

        # --- Channel 3: Graph Entity-Centric Multi-Hop Search ---
        # 借鉴自: engines/memory/graph_store.py HugeGraphGraphStore.search
        graph_results = []
        graph_entity_names: set = set()
        try:
            graph_results = self._graph_store.search(
                rewritten_query, limit=top_k * 3, max_hops=2
            )
            for gr in graph_results:
                matched = gr.get("matched_entity", "")
                if matched:
                    graph_entity_names.add(matched)
                ctx = gr.get("context", "")
                for part in re.split(r"\s+\[[\w_]+\]\s+", ctx):
                    part = part.strip()
                    if 2 <= len(part) <= 8 and re.match(r"^[\u4e00-\u9fa5]+$", part):
                        graph_entity_names.add(part)
        except Exception as e:
            print(f"[Graph Channel] Error: {e}", file=sys.stderr, flush=True)

        # --- Build memory entity map for entity boosting ---
        # 借鉴自: mem0/memory/main.py _compute_entity_boosts
        memory_entities: Dict[str, List[str]] = {}
        for mid, mem in memories_map.items():
            entities_in_mem: set = set()
            # Extract Chinese entity candidates from memory content
            for m in re.finditer(r"[\u4e00-\u9fa5]{2,8}", mem["content"]):
                entities_in_mem.add(m.group(0))
            # Add graph-derived entities that appear in this memory
            for ename in graph_entity_names:
                if ename in mem["content"]:
                    entities_in_mem.add(ename)
            memory_entities[mid] = list(entities_in_mem)

        # Query entities for boosting (search terms + graph-matched entities + rewrite entities)
        query_entities = extract_query_entities_simple(rewritten_query)
        query_entities = list(set(query_entities) | set(query_entities_extra) | graph_entity_names)

        # Compute entity boosts (mem0-style: +0.5 per matching entity)
        entity_boosts = compute_entity_boosts(
            query_entities, memory_entities, boost_weight=ENTITY_BOOST_WEIGHT
        )

        # --- Additive Hybrid Scoring (mem0-style) ---
        # 借鉴自: mem0/utils/scoring.py score_and_rank
        semantic_results = []
        for r in faiss_results:
            mid = r.get("memory_id")
            if mid and mid in memories_map:
                semantic_results.append({
                    "id": mid,
                    "content": memories_map[mid]["content"],
                    "score": r.get("weighted_score", r.get("raw_score", 0.0)),
                    "metadata": {
                        "retention": memories_map[mid].get("retention", 0.0),
                        "access_count": memories_map[mid].get("access_count", 0),
                    },
                })

        # Add Ebbinghaus-only memories as low-score semantic entries
        for mid, mem in memories_map.items():
            if mid not in {sr["id"] for sr in semantic_results} and mem["retention"] > 0.1:
                semantic_results.append({
                    "id": mid,
                    "content": mem["content"],
                    "score": mem["retention"] * 0.3,
                    "metadata": {
                        "retention": mem["retention"],
                        "access_count": mem["access_count"],
                    },
                })

        if HAS_GRAPHRAG_OPS and semantic_results:
            ranked = score_and_rank(
                semantic_results=semantic_results,
                bm25_scores=bm25_scores,
                entity_boosts=entity_boosts,
                threshold=0.0,
                top_k=top_k * 2,
                explain=False,
            )

            # Build source tracking sets
            faiss_id_set = {r["memory_id"] for r in faiss_results if r.get("memory_id")}
            bm25_id_set = set(channel_bm25)
            graph_id_set = set()
            for gr in graph_results:
                matched = gr.get("matched_entity", "")
                for mid, mem in memories_map.items():
                    if matched and matched in mem["content"]:
                        graph_id_set.add(mid)

            results = []
            for r in ranked:
                mid = r["id"]
                if mid not in memories_map:
                    continue
                mem = memories_map[mid]
                channels = []
                if mid in faiss_id_set:
                    channels.append("faiss")
                if mid in bm25_id_set:
                    channels.append("bm25")
                if mid in graph_id_set or entity_boosts.get(mid, 0.0) > 0:
                    channels.append("graph")
                source = "+".join(channels) if channels else "additive"
                results.append({
                    "memory": mem,
                    "score": r["score"],
                    "source": source,
                })
            trace.append({
                "step": 2, "name": "\u52a0\u6027\u6df7\u5408\u68c0\u7d22",
                "detail": f"FAISS={len(faiss_results)}, BM25={len(channel_bm25)}, "
                         f"Graph={len(graph_results)} \u2192 additive Top-{len(results)}",
                "channels": {"faiss": len(faiss_results), "bm25": len(channel_bm25),
                             "graph": len(graph_results)},
                "elapsed_ms": round((time.time()-step_start)*1000)})
        else:
            # Fallback path when GraphRAG operators not available
            results = []
            for r in faiss_results[:top_k]:
                mid = r.get("memory_id")
                if mid and mid in memories_map:
                    results.append({
                        "memory": memories_map[mid],
                        "score": r.get("weighted_score", r.get("retention", 0.5)),
                        "source": "faiss_ebbinghaus",
                    })
            trace.append({"step": 2, "name": "FAISS+Ebbinghaus\u68c0\u7d22",
                          "detail": f"Top-{len(results)} (BM25/Graph\u4e0d\u53ef\u7528)",
                          "elapsed_ms": round((time.time()-step_start)*1000)})

        # P2: Agent collaboration / privacy filtering (PowerMem AgentMemory aligned)
        # 借鉴自: engines/memory/agent_collaboration.py AgentMemoryManager.get_accessible_memories
        if results and agent_id:
            try:
                step_start_privacy = time.time()
                accessible = self._agent_manager.get_accessible_memories(
                    agent_id=agent_id,
                    all_memories=[r["memory"] for r in results],
                    operation="read",
                )
                accessible_ids = {m.get("id") for m in accessible if m.get("id")}
                filtered_results = [r for r in results if r["memory"]["id"] in accessible_ids]
                if len(filtered_results) < len(results):
                    trace.append({"step": 2.1, "name": "\u6743\u9650\u4e0e\u9690\u79c1\u8fc7\u6ee4",
                                  "detail": f"{len(filtered_results)}/{len(results)} accessible",
                                  "elapsed_ms": round((time.time() - step_start_privacy) * 1000)})
                results = filtered_results
            except Exception as e:
                print(f"[AgentCollaboration] Filter error: {e}", file=sys.stderr, flush=True)

        # Cross-encoder rerank (optional) before LLM rerank
        if self._reranker is not None and results and not fast_eval:
            try:
                step_start = time.time()
                candidates = [
                    {
                        "id": r["memory"]["id"],
                        "text": r["memory"]["content"],
                        "memory": r["memory"],
                        "source": r.get("source", ""),
                    }
                    for r in results
                ]
                reranked = self._reranker.rerank(rewritten_query, candidates, top_k=top_k * 2)
                results = [
                    {
                        "memory": r["memory"],
                        "score": r["rerank_score"],
                        "source": (r.get("source", "") + "+rerank").lstrip("+"),
                    }
                    for r in reranked
                ]
                trace.append({
                    "step": 2.5, "name": "Cross-encoder rerank",
                    "detail": f"Top-{len(results)}",
                    "elapsed_ms": round((time.time() - step_start) * 1000),
                })
            except Exception as e:
                print(f"[Rerank] Error: {e}", file=sys.stderr, flush=True)

        # LLM reranking with graph context (only for top candidates)
        graph_ctx = self._build_graph_context()
        top_candidates = results[:top_k]
        if not fast_eval and top_candidates and len(top_candidates) > 1:
            try:
                llm_ranks = self._llm_rank_memories(rewritten_query, top_candidates, graph_ctx)
                llm_score_map = {r.get("memory_id"): r.get("score", 0.5) for r in llm_ranks}
                for r in top_candidates:
                    mid = r["memory"]["id"]
                    llm_s = llm_score_map.get(mid)
                    if llm_s is not None:
                        r["score"] = round(r["score"] * 0.4 + llm_s * 0.6, 4)
                        r["source"] += "+llm_rerank"
            except Exception:
                pass  # LLM rerank failure is non-critical

        # Sort final results and reinforce accessed memories (unless disabled for benchmarks)
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        if update_access_count:
            for r in results:
                mid = r["memory"]["id"]
                if mid:
                    db.execute(
                        "UPDATE memories SET access_count=access_count+1, last_accessed_at=? WHERE id=?",
                        (now, mid),
                    )

        # Step 3: Graph Context Retrieval (from HugeGraph)
        step_start = time.time()
        # Already computed above, but log it
        trace.append({"step": 3, "name": "\u56fe\u8c31\u4e0a\u4e0b\u6587\u68c0\u7d22",
                      "detail": f"{len(graph_ctx.split(chr(10)))} edges retrieved"
                      if graph_ctx else "\u56fe\u8c31\u4e3a\u7a7a",
                      "elapsed_ms": round((time.time()-step_start)*1000)})

        # Fast-eval: bypass Step 4 LLM answer generation and return retrieval-only results
        # 旧版直接返回 results[0]["memory"]["content"]（原始对话文本）
        # → 改为拼接实体+关系+top内容，生成事实摘要而非原始对话
        if fast_eval:
            if results:
                top = results[0]
                mem = top["memory"]
                # Build factual answer from extracted entities/relationships
                # NOT from raw content (which is the original conversation text)
                factual_parts = []
                # 1. Entities as facts
                if mem.get("metadata") and mem["metadata"].get("entities"):
                    ents = mem["metadata"]["entities"]
                    factual_parts.extend(
                        f"{e.get('name','?')}({e.get('type','?')})" for e in ents[:8]
                    )
                # 2. Relationships as facts
                if mem.get("metadata") and mem["metadata"].get("relationships"):
                    rels = mem["metadata"]["relationships"]
                    for r in rels[:6]:
                        src = r.get("source", "?")
                        tgt = r.get("target", "?")
                        label = r.get("label", "?")
                        factual_parts.append(f"{src} {label} {tgt}")
                # 3. If no structured facts, fall back to content excerpt
                if factual_parts:
                    answer = "; ".join(factual_parts)
                else:
                    # Fallback: extract key phrases from content (not full conversation)
                    content = mem.get("content", "")
                    # Take first 100 chars as a factual snippet, not the whole conversation
                    answer = content[:100] if content else "（无事实）"
            else:
                answer = "记忆中没有相关信息。"
            trace.append({"step": 4, "name": "Fast-eval retrieval-only",
                          "detail": f"Top-{len(results)} result returned, no LLM answer",
                          "elapsed_ms": 0})
            db.commit()
            db.close()
            total_ms = round((time.time() - start_time) * 1000)
            try:
                self._audit.log(
                    operation="search_memory",
                    user_id=user_id,
                    query=query[:1000] if query else None,
                    latency_ms=total_ms,
                    success=True,
                    metadata={"mode": "fast_eval", "result_count": len(results)},
                )
            except Exception:
                pass
            return {
                "query": query, "action": "QUERY", "results": results,
                "answer": answer, "graph_context": graph_ctx,
                "trace": trace, "total_elapsed_ms": total_ms,
            }

        # Step 4: LLM Answer Generation
        step_start = time.time()
        relevant_memories = [r["memory"] for r in results if r.get("memory")]
        all_edges = self.hg.get_all_edges()

        # Extract all potential entity names from query
        query_names = set()
        # Split by query particles AND common 2-char action/attribute words
        # to avoid treating "张明擅长" as a single entity (should be 张明 + 擅长)
        parts = re.split(r'[的了在是有和也都哪些多少几怎么如何谁什么哪里哪个有没有这个信息记忆同事朋友共事员工上班工作总部公司参加创立技术城市总监告诉我帮回忆擅长喜欢负责从事属于位于来自使用学习了解知道做过担任开发管理运维设计构建运营推动研究负责关注掌握精通]', query)
        for part in parts:
            part = part.strip()
            if len(part) >= 2 and len(part) <= 4 and re.match(r'^[\u4e00-\u9fa5]+$', part):
                query_names.add(part)

        # Check: does ANY query entity exist in the system?
        # Skip this check if search already found results — the entity existence
        # gate was causing false negatives (e.g., "张明擅长" treated as single entity)
        # when FAISS/BM25/Graph channels already returned relevant memories.
        known_in_system = set()
        for v in self.hg.get_all_vertices():
            known_in_system.add(v.get("name", ""))
        for mem in relevant_memories:
            for m2 in re.finditer(r'[\u4e00-\u9fa5]{2,4}', mem["content"]):
                known_in_system.add(m2.group())

        # If NO query entity exists AND no search results → direct "not found"
        query_known = query_names & known_in_system
        if query_names and not query_known and not results:
            answer = f"记忆中没有这个信息。"
            trace.append({"step": 4, "name": "实体存在性检查",
                          "detail": f"{','.join(query_names)} 均不在系统中",
                          "elapsed_ms": round((time.time()-step_start)*1000)})
            db.commit()
            db.close()
            total_ms = round((time.time() - start_time) * 1000)
            try:
                self._audit.log(
                    operation="search_memory",
                    user_id=user_id,
                    query=query[:1000] if query else None,
                    latency_ms=total_ms,
                    success=True,
                    metadata={"mode": "not_found", "query_entities": list(query_names)},
                )
            except Exception:
                pass
            return {
                "query": query, "action": "QUERY", "results": results,
                "answer": answer, "graph_context": graph_ctx,
                "trace": trace, "total_elapsed_ms": total_ms,
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
                # P1: Provenance for graph direct reasoning answers
                graph_prov = []
                prov_sources = self._get_provenance_for_entities(list(query_known))
                graph_prov = prov_sources[:2]
                trace.append({"step": 4, "name": "图谱直接推理",
                              "detail": "从works_at边计算(无需LLM)",
                              "provenance_count": len(graph_prov),
                              "elapsed_ms": round((time.time()-step_start)*1000)})
                db.commit()
                db.close()
                total_ms = round((time.time() - start_time) * 1000)
                try:
                    self._audit.log(
                        operation="search_memory",
                        user_id=user_id,
                        query=query[:1000] if query else None,
                        latency_ms=total_ms,
                        success=True,
                        metadata={"mode": "graph_direct", "answer_length": len(answer)},
                    )
                except Exception:
                    pass
                return {
                    "query": query, "action": "QUERY", "results": results,
                    "answer": answer, "graph_context": graph_ctx,
                    "provenance": graph_prov,
                    "trace": trace, "total_elapsed_ms": total_ms,
                }

        # Fallback: LLM answer generation
        # P1: Inject user profile context into answer query
        profile_text = self._profile_injector.get_profile_for_rewrite(user_id).get("user_profile", "")
        answer_query = rewritten_query
        if profile_text:
            answer_query = f"[用户背景: {profile_text}] {rewritten_query}"
        if relevant_memories:
            answer = self._llm_generate_answer(answer_query, relevant_memories, graph_ctx)
        elif graph_ctx:
            answer = self._llm_generate_answer(answer_query, [], graph_ctx)
        else:
            answer = "记忆中没有相关信息。"

        # P1: Provenance — attach source memory citations
        provenance_info = []
        if relevant_memories:
            entity_names_in_query = list(query_known) if query_known else []
            if entity_names_in_query:
                prov = self._get_provenance_for_entities(entity_names_in_query)
                provenance_info = prov[:3]  # max 3 provenance entries
            else:
                # Use top result's memory as source
                top_mem_id = results[0]["memory"]["id"] if results else None
                if top_mem_id and top_mem_id in self._provenance:
                    provenance_info = [{"memory_id": top_mem_id,
                                       "link": self._provenance[top_mem_id][0]}]

        trace.append({"step": 4, "name": "LLM\u56de\u7b54\u751f\u6210",
                      "detail": f"{len(answer)} chars",
                      "provenance_count": len(provenance_info),
                      "elapsed_ms": round((time.time()-step_start)*1000)})

        db.commit()
        db.close()
        total_ms = round((time.time() - start_time) * 1000)
        try:
            self._audit.log(
                operation="search_memory",
                user_id=user_id,
                query=query[:1000] if query else None,
                latency_ms=total_ms,
                success=True,
                metadata={"mode": "llm_answer", "answer_length": len(answer), "result_count": len(results)},
            )
        except Exception:
            pass

        return {
            "query": query,
            "action": "QUERY",
            "results": results,
            "answer": answer,
            "graph_context": graph_ctx,
            "provenance": provenance_info,
            "trace": trace,
            "total_elapsed_ms": total_ms,
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
        """Entity resolution: merge duplicates using 3 strategies (inspired by GraphRAG EntityResolution).
        Strategy 1: Substring match (e.g., "腾讯深圳" → "腾讯" + "深圳")
        Strategy 2: Exact type+name match (case/whitespace insensitive)
        Strategy 3: Embedding cosine similarity (>0.85 threshold) — LLM-free fast path
        """
        hg_verts = self.hg.get_all_vertices()
        existing_names = {}  # name -> {type, ...}
        for v in hg_verts:
            existing_names[v.get("name", "")] = {"type": v.get("type", v.get("label", ""))}

        merged = {}  # old_name -> canonical_name
        new_entities = []

        # Strategy 1: Substring containment (original logic, enhanced)
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

        # Strategy 2: Cross-entity dedup within current extraction
        # If two new entities have same type and high name overlap, merge
        seen_names = {}
        final_entities = []
        for ent in new_entities:
            name = ent["name"]
            etype = ent.get("type", "")
            # Check against already-accepted entities
            deduped = False
            for accepted_name, accepted in seen_names.items():
                if accepted.get("type") == etype and accepted_name != name:
                    # Use substring containment as primary signal
                    if accepted_name in name or name in accepted_name:
                        keep = accepted_name if len(accepted_name) <= len(name) else name
                        remove = name if keep == accepted_name else accepted_name
                        if remove == name:
                            merged[name] = accepted_name
                        else:
                            # Update the already-accepted entry
                            merged[accepted_name] = name
                            for rel in relationships:
                                if rel["source"] == accepted_name:
                                    rel["source"] = name
                                if rel["target"] == accepted_name:
                                    rel["target"] = name
                        deduped = True
                        break
            if not deduped:
                seen_names[name] = ent
                final_entities.append(ent)

        # Strategy 3: Embedding similarity (fast, no LLM call)
        # Only when we have >= 2 new entities of the same type
        type_groups = {}
        for ent in final_entities:
            type_groups.setdefault(ent.get("type", ""), []).append(ent)

        for etype, group in type_groups.items():
            if len(group) < 2:
                continue
            try:
                names = [e["name"] for e in group]
                # Use FAISS index's embedding function for fast comparison
                embed_fn = getattr(self.faiss, '_get_embedding_client', None)
                if embed_fn is None:
                    continue
                client = embed_fn()
                embeddings = []
                for n in names:
                    resp = client.embeddings.create(model="text-embedding-ada-002",
                                                     input=n[:50])
                    embeddings.append(np.array(resp.data[0].embedding, dtype=np.float32))
                # Cosine similarity check
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        cos_sim = float(np.dot(embeddings[i], embeddings[j]) /
                                        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8))
                        if cos_sim > 0.85:
                            # Merge: keep the one that's shorter or already in graph
                            keep, remove = (names[i], names[j]) if len(names[i]) <= len(names[j]) else (names[j], names[i])
                            if remove not in merged:
                                merged[remove] = keep
                                # Update relationships
                                for rel in relationships:
                                    if rel["source"] == remove:
                                        rel["source"] = keep
                                    if rel["target"] == remove:
                                        rel["target"] = keep
                                final_entities = [e for e in final_entities if e["name"] != remove]
            except Exception as e:
                print(f"[EntityResolution] Embedding check error (non-critical): {e}",
                      file=sys.stderr, flush=True)

        # Apply all merged mappings to relationships
        for rel in relationships:
            if rel["source"] in merged:
                rel["source"] = merged[rel["source"]]
            if rel["target"] in merged:
                rel["target"] = merged[rel["target"]]

        return final_entities, relationships

    def _apply_merged_to_rels(self, relationships: list, merged: dict) -> list:
        """Apply entity name mappings to all relationships."""
        for rel in relationships:
            if rel["source"] in merged:
                rel["source"] = merged[rel["source"]]
            if rel["target"] in merged:
                rel["target"] = merged[rel["target"]]
        return relationships

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

        # Probe HugeGraph connectivity
        hg_connected = False
        try:
            _ = self.hg.get_all_vertices(limit=1)
            hg_connected = True
        except Exception:
            pass

        graph_name = HUGEGRAPH_GRAPH
        try:
            graph_name = self.hg.client.cfg.graph_name
        except Exception:
            pass

        return {
            "memories": mem_count,
            "entities": len(hg_verts),
            "edges": len(hg_edges),
            "vectors": faiss_stats.get("total_vectors", 0),
            "avg_latency_ms": 0,
            "hugegraph_connected": hg_connected,
            "graph": graph_name,
            "total_memories": mem_count,
            "total_nodes": len(hg_verts),
            "total_edges": len(hg_edges),
            "node_type_distribution": type_dist,
            "ebbinghaus_scores": ebbinghaus,
            "faiss": faiss_stats,
            "bm25": {
                "doc_count": self._bm25.doc_count if self._bm25 else 0,
                "available": self._bm25 is not None,
            },
            "rrf_available": False,
            "additive_scoring_available": self._additive_scoring_available,
            "provenance_count": len(self._provenance),
            "graphrag_ops": HAS_GRAPHRAG_OPS,
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
            "SELECT id,content,created_at,last_accessed_at,access_count,initial_score,"
            "scope,privacy,importance,agent_id,run_id,metadata "
            "FROM memories WHERE user_id=? ORDER BY created_at DESC", (user_id,),
        ):
            elapsed_hours = (now - row["created_at"]) / 3600
            ret = row["initial_score"] * math.exp(-EBBINGHAUS_K * elapsed_hours)
            ret = min(1.0, max(0.0, ret + row["access_count"] * EBBINGHAUS_REINFORCE))
            memories.append({
                "id": row["id"], "content": row["content"],
                "retention": round(ret, 4), "access_count": row["access_count"],
                "scope": row["scope"], "privacy": row["privacy"],
                "importance": row["importance"], "agent_id": row["agent_id"],
                "run_id": row["run_id"],
                "metadata": json.loads(row["metadata"] or "{}"),
            })
        db.close()
        return memories

    def distill_user_memories(
        self, user_id: str = "demo_user", threshold: int = None
    ) -> dict:
        """Run Experience + Skill distillation for all memories of a user."""
        memories = self.get_memories(user_id=user_id)
        # DistillationPipeline expects id/content/created_at
        atomics = [
            {"id": m["id"], "content": m["content"], "created_at": time.time()}
            for m in memories
        ]
        result = self._distillation.distill_all(
            atomics, user_id=user_id, threshold=threshold
        )
        return result

    def get_experiences(self, query: str = "", user_id: str = "demo_user", top_k: int = 5) -> list:
        """Retrieve distilled experiences for a user."""
        return self._distillation.exp_store.retrieve(query, user_id=user_id, top_k=top_k)

    def get_skills(self, query: str = "", user_id: str = "demo_user", top_k: int = 5) -> list:
        """Retrieve distilled skills for a user."""
        return self._distillation.skill_store.retrieve(query, user_id=user_id, top_k=top_k)
        """Legacy alias; prefer forget_user() for scoped deletion."""
        return self.forget_user(user_id=user_id)

    def forget_user(self, user_id: str = "demo_user"):
        """Delete all memories for a user and rebuild FAISS/BM25 without them.

        Graph vertices/edges are intentionally retained as global knowledge,
        matching the PowerMem semantics where graph structure is shared.
        """
        db = get_metadata_db()
        db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        db.commit()

        # Rebuild FAISS from remaining memories
        self.faiss.clear()
        for row in db.execute(
            "SELECT id, content, created_at FROM memories ORDER BY created_at ASC"
        ):
            self.faiss.add_memory(row["id"], row["content"], row["created_at"])
        try:
            self.faiss.save()
        except Exception:
            pass

        # Rebuild BM25 from remaining memories
        if self._bm25 is not None:
            try:
                self._bm25 = BM25FullTextBackend()
                docs, ids = [], []
                for row in db.execute("SELECT id, content FROM memories ORDER BY created_at ASC"):
                    docs.append(row["content"])
                    ids.append(row["id"])
                if docs:
                    self._bm25.add_documents(docs, ids)
                bm25_dir = os.path.dirname(os.path.abspath(__file__))
                self._bm25.save_index_by_name(bm25_dir, "memory_bm25")
            except Exception as e:
                print(f"[BM25] Rebuild error: {e}", file=sys.stderr, flush=True)

        db.close()

        # Remove provenance entries tied to this user (best-effort)
        mem_ids = set()
        db2 = get_metadata_db()
        for row in db2.execute("SELECT id FROM memories WHERE user_id=?", (user_id,)):
            mem_ids.add(row["id"])
        db2.close()
        for mid in list(self._provenance.keys()):
            if mid not in mem_ids:
                self._provenance.pop(mid, None)
        self._save_provenance()

    def get_persona(self, user_id: str = "demo_user") -> dict:
        """Retrieve the L3 persona / user profile for a scope."""
        db = get_metadata_db()
        row = db.execute(
            "SELECT summary, updated_at FROM personas WHERE user_id=?", (user_id,)
        ).fetchone()
        db.close()
        if row:
            return {
                "user_id": user_id,
                "summary": row["summary"],
                "updated_at": row["updated_at"],
            }
        return {"user_id": user_id, "summary": "", "updated_at": 0}

    def update_persona(self, user_id: str = "demo_user", summary: str = ""):
        """Update the L3 persona / user profile for a scope."""
        db = get_metadata_db()
        db.execute(
            "INSERT INTO personas(user_id, summary, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (user_id, summary, time.time()),
        )
        db.commit()
        db.close()

    # ---------------------------------------------------------------------------
    # P2: CRUD + profile helpers aligned with Mem0 / PowerMem SDK surface
    # ---------------------------------------------------------------------------

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by id."""
        db = get_metadata_db()
        row = db.execute(
            "SELECT id,content,created_at,last_accessed_at,access_count,initial_score,"
            "scope,privacy,importance,agent_id,run_id,metadata "
            "FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        db.close()
        if not row:
            return None
        return {
            "id": row["id"], "content": row["content"],
            "created_at": row["created_at"], "last_accessed_at": row["last_accessed_at"],
            "access_count": row["access_count"], "initial_score": row["initial_score"],
            "scope": row["scope"], "privacy": row["privacy"], "importance": row["importance"],
            "agent_id": row["agent_id"], "run_id": row["run_id"],
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    def update_memory(
        self,
        memory_id: str,
        content: str,
        user_id: str = "demo_user",
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a memory's content and metadata. Re-indexes FAISS/BM25."""
        db = get_metadata_db()
        existing = db.execute(
            "SELECT id, content, scope, privacy FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)
        ).fetchone()
        if not existing:
            db.close()
            return {"error": "NOT_FOUND", "memory_id": memory_id}

        # P2: Agent collaboration permission check (PowerMem AgentMemory aligned)
        # 借鉴自: engines/memory/agent_collaboration.py PermissionChecker.check
        if agent_id:
            can_write = self._agent_manager.check_access(
                agent_id=agent_id,
                operation="write",
                memory={
                    "scope": existing["scope"] or MemoryScope.PRIVATE.value,
                    "user_id": user_id,
                    "privacy": existing["privacy"] or PrivacyLevel.STANDARD.value,
                },
            )
            if not can_write:
                db.close()
                return {"error": "FORBIDDEN", "memory_id": memory_id}

        old_content = existing["content"]
        now = time.time()
        merged_meta = json.dumps(metadata or {}, ensure_ascii=False)
        db.execute(
            "UPDATE memories SET content=?, last_accessed_at=?, metadata=? WHERE id=?",
            (content, now, merged_meta, memory_id),
        )
        db.commit()
        db.close()

        # P0: Memory history tracking (mem0-style UPDATE event)
        # 借鉴自: mem0/memory/storage.py SQLiteManager.batch_add_history
        try:
            self._history.add_history(
                memory_id=memory_id,
                event="UPDATE",
                old_memory=old_content,
                new_memory=content,
                actor_id=user_id,
                role="user",
                metadata={"source": "update_memory"},
            )
        except Exception as e:
            print(f"[History] Update event error: {e}", file=sys.stderr, flush=True)

        # P1: User profile update after memory change
        try:
            self._user_profile.update_from_memories(user_id, [content])
        except Exception as e:
            print(f"[UserProfile] Update error: {e}", file=sys.stderr, flush=True)

        # Re-index vector store (best-effort: remove + add)
        try:
            self.faiss.delete_memory(memory_id)
        except Exception:
            pass
        self.faiss.add_memory(memory_id, content, now)
        try:
            self.faiss.save()
        except Exception:
            pass

        try:
            self._audit.log(
                operation="update_memory",
                user_id=user_id,
                memory_id=memory_id,
                content=content[:2000] if content else None,
                latency_ms=0,
                success=True,
            )
        except Exception:
            pass

        return {"status": "ok", "memory_id": memory_id, "action": "updated"}

    def delete_memory(
        self,
        memory_id: str,
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a memory from SQLite, FAISS and BM25. Graph provenance is kept."""
        db = get_metadata_db()
        row = db.execute(
            "SELECT id, content, scope, privacy FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)
        ).fetchone()
        if not row:
            db.close()
            return {"error": "NOT_FOUND", "memory_id": memory_id}

        # P2: Agent collaboration permission check
        if agent_id:
            can_delete = self._agent_manager.check_access(
                agent_id=agent_id,
                operation="delete",
                memory={
                    "scope": row["scope"] or MemoryScope.PRIVATE.value,
                    "user_id": user_id,
                    "privacy": row["privacy"] or PrivacyLevel.STANDARD.value,
                },
            )
            if not can_delete:
                db.close()
                return {"error": "FORBIDDEN", "memory_id": memory_id}

        old_content = row["content"]
        db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        db.commit()
        db.close()

        # P0: Memory history tracking (mem0-style DELETE event)
        # 借鉴自: mem0/memory/storage.py SQLiteManager.batch_add_history
        try:
            self._history.add_history(
                memory_id=memory_id,
                event="DELETE",
                old_memory=old_content,
                actor_id=user_id,
                role="user",
                metadata={"source": "delete_memory"},
            )
        except Exception as e:
            print(f"[History] Delete event error: {e}", file=sys.stderr, flush=True)

        try:
            self.faiss.delete_memory(memory_id)
            self.faiss.save()
        except Exception:
            pass

        # P2: Faiss deletable index auxiliary cleanup (PowerMem efficient delete aligned)
        # 借鉴自: engines/memory/faiss_deletable.py FaissDeletableIndex.remove_by_id
        try:
            self._faiss_deletable.remove_by_id(memory_id)
        except Exception:
            pass

        if self._bm25 is not None:
            try:
                self._bm25.delete_document(memory_id)
                bm25_dir = os.path.dirname(os.path.abspath(__file__))
                self._bm25.save_index_by_name(bm25_dir, "memory_bm25")
            except Exception as e:
                log.warning("BM25 delete error: %s", e)

        try:
            self._audit.log(
                operation="delete_memory",
                user_id=user_id,
                memory_id=memory_id,
                latency_ms=0,
                success=True,
            )
        except Exception:
            pass

        return {"status": "ok", "memory_id": memory_id, "action": "deleted"}

    def compress_memories(
        self,
        user_id: str = "demo_user",
        max_age_hours: float = 0,
    ) -> Dict[str, Any]:
        """Compress memories for a user using MemoryCompressor.

        Clusters similar memories, summarizes them, prunes low-value ones,
        and archives originals. New summaries are added back as compressed
        memories. Aligned with PowerMem MemoryOptimizer.compress().
        """
        db = get_metadata_db()
        rows = db.execute(
            "SELECT id, content, created_at, access_count, initial_score, importance "
            "FROM memories WHERE user_id=?", (user_id,)
        ).fetchall()
        memories = [
            {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
                "access_count": row["access_count"],
                "initial_importance": row["initial_score"],
                "importance": row["importance"],
            }
            for row in rows
        ]
        db.close()

        # P2: Memory compression (PowerMem MemoryOptimizer aligned)
        # 借鉴自: engines/memory/memory_compressor.py MemoryCompressor.compress
        result = self._compressor.compress(memories)

        archived_ids = {
            m.get("id", m.get("memory_id", ""))
            for m in result.get("archived", [])
        }
        pruned_ids = {
            m.get("id", m.get("memory_id", ""))
            for m in result.get("pruned", [])
        }
        removed_ids = archived_ids | pruned_ids

        for mid in removed_ids:
            try:
                self.delete_memory(mid, user_id=user_id)
            except Exception as e:
                print(f"[Compress] Delete {mid} error: {e}", file=sys.stderr, flush=True)

        new_memory_ids = []
        for summary_item in result.get("summaries", []):
            summary_text = summary_item.get("summary", "")
            if not summary_text:
                continue
            try:
                add_res = self.add_memory_bypass_classify(
                    content=summary_text,
                    user_id=user_id,
                    metadata={
                        "memory_type": "compressed_summary",
                        "source_ids": summary_item.get("source_ids", []),
                    },
                    skip_index_save=True,
                )
                new_memory_ids.append(add_res.get("memory_id"))
            except Exception as e:
                print(f"[Compress] Add summary error: {e}", file=sys.stderr, flush=True)

        self.save_index()
        result["removed_memory_ids"] = list(removed_ids)
        result["new_memory_ids"] = new_memory_ids
        return result

    def save_index(self) -> None:
        """Persist FAISS and BM25 indices to disk."""
        try:
            self.faiss.save()
        except Exception:
            pass
        if self._bm25 is not None:
            try:
                bm25_dir = os.path.dirname(os.path.abspath(__file__))
                self._bm25.save_index_by_name(bm25_dir, "memory_bm25")
            except Exception as e:
                log.warning("BM25 save error: %s", e)

    def list_memories(self, user_id: str = "demo_user") -> list:
        """Alias of get_memories for SDK consistency."""
        return self.get_memories(user_id=user_id)

    def get_user_profile(self, user_id: str = "demo_user") -> Dict[str, Any]:
        """Alias of get_persona for SDK consistency."""
        return self.get_persona(user_id=user_id)

    def update_user_profile(self, user_id: str = "demo_user", summary: str = "") -> Dict[str, Any]:
        """Alias of update_persona for SDK consistency."""
        self.update_persona(user_id=user_id, summary=summary)
        return self.get_persona(user_id=user_id)

    def rewrite_query(
        self,
        query: str,
        user_id: str = "demo_user",
        aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Rewrite a query for better retrieval using profile context and aliases."""
        profile = self.get_user_profile(user_id)
        profile_text = profile.get("summary", "") if isinstance(profile, dict) else ""
        engine = QueryRewriteEngine(aliases=aliases, user_profile=profile_text)
        return engine.expand_query(query)

    def add_skill(self, content: str, user_id: str = "demo_user") -> Dict[str, Any]:
        """Store a procedural/skill memory."""
        return self.add_memory(
            content=content,
            user_id=user_id,
            metadata={"memory_type": "procedural"},
        )

    def search_skills(self, query: str, user_id: str = "demo_user", top_k: int = 5) -> list:
        """Search procedural/skilled memories via the skill store."""
        return self.get_skills(query=query, user_id=user_id, top_k=top_k)


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
        result = store.add_memory(
            content=content,
            user_id=user_id,
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            scope=MemoryScope(data.get("scope", "private")),
            privacy=PrivacyLevel(data.get("privacy", "standard")),
            metadata=data.get("metadata"),
        )
        return jsonify(result)

    @app.route("/api/memory/search", methods=["POST"])
    def api_search_memory():
        data = request.json or {}
        query = data.get("query") or data.get("content", "").strip()
        user_id = data.get("user_id", "demo_user")
        if not query:
            return jsonify({"error": "query is required"}), 400
        result = store.search_memory(
            query=query,
            user_id=user_id,
            top_k=int(data.get("top_k", 5)),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            filters=data.get("filters"),
        )
        return jsonify(result)

    @app.route("/api/memory/get", methods=["GET"])
    def api_get_memory():
        memory_id = request.args.get("id")
        if not memory_id:
            return jsonify({"error": "id is required"}), 400
        result = store.get_memory_by_id(memory_id)
        if result is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(result)

    @app.route("/api/memory/update", methods=["POST"])
    def api_update_memory():
        data = request.json or {}
        memory_id = data.get("id") or data.get("memory_id")
        content = data.get("content", "").strip()
        user_id = data.get("user_id", "demo_user")
        if not memory_id or not content:
            return jsonify({"error": "id/memory_id and content are required"}), 400
        return jsonify(store.update_memory(
            memory_id=memory_id,
            content=content,
            user_id=user_id,
            metadata=data.get("metadata"),
        ))

    @app.route("/api/memory/delete", methods=["POST"])
    def api_delete_memory():
        data = request.json or {}
        memory_id = data.get("id") or data.get("memory_id")
        user_id = data.get("user_id", "demo_user")
        if not memory_id:
            return jsonify({"error": "id/memory_id is required"}), 400
        return jsonify(store.delete_memory(memory_id=memory_id, user_id=user_id))

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
        return jsonify(store.get_memories(user_id=request.args.get("user_id", "demo_user")))

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        return jsonify(store.get_stats())

    @app.route("/api/graph", methods=["GET"])
    def api_graph():
        return jsonify(store.get_graph_data())

    @app.route("/api/locomo", methods=["GET"])
    def api_locomo():
        """Serve cached LOCOMO benchmark result if available."""
        import glob
        candidates = [
            "locomo_result_full.json",
            "tests/locomo_result_full.json",
            "../tests/locomo_result_full.json",
            "locomo_result_sample.json",
            "tests/locomo_result_sample.json",
            "../tests/locomo_result_sample.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        return jsonify(json.load(f))
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
        return jsonify({"status": "not_ready", "metrics": {}})

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        data = request.json or {}
        user_id = data.get("user_id", "demo_user")
        # Clear all stores: HugeGraph + FAISS + BM25 + SQLite
        try:
            store.hg.clear_graph()
        except Exception as e:
            print(f"[Clear] HugeGraph error: {e}", file=sys.stderr)
        store.faiss.clear()
        try:
            store.faiss.save()
        except Exception:
            pass
        # BM25: clear by creating a new empty index (BM25FullTextBackend is conditionally imported)
        if store._bm25 is not None:
            try:
                store._bm25 = store._bm25.__class__()  # Create empty instance of same class
            except Exception:
                store._bm25 = None
        try:
            db = get_metadata_db()
            db.execute("DELETE FROM memories")
            db.commit()
            db.close()
        except Exception as e:
            print(f"[Clear] SQLite error: {e}", file=sys.stderr)
        return jsonify({"status": "cleared"})

    @app.route("/api/memory/distill", methods=["POST"])
    def api_distill():
        data = request.json or {}
        user_id = data.get("user_id", "demo_user")
        threshold = data.get("threshold")
        if threshold is not None:
            threshold = int(threshold)
        return jsonify(store.distill_user_memories(user_id=user_id, threshold=threshold))

    @app.route("/api/memory/experiences", methods=["POST"])
    def api_experiences():
        data = request.json or {}
        return jsonify(store.get_experiences(
            query=data.get("query", ""),
            user_id=data.get("user_id", "demo_user"),
            top_k=int(data.get("top_k", 5)),
        ))

    @app.route("/api/memory/skills", methods=["POST"])
    def api_skills():
        data = request.json or {}
        return jsonify(store.get_skills(
            query=data.get("query", ""),
            user_id=data.get("user_id", "demo_user"),
            top_k=int(data.get("top_k", 5)),
        ))

    @app.route("/api/memory/persona", methods=["GET", "POST"])
    def api_persona():
        if request.method == "GET":
            return jsonify(store.get_persona(request.args.get("user_id", "demo_user")))
        data = request.json or {}
        store.update_persona(
            user_id=data.get("user_id", "demo_user"),
            summary=data.get("summary", ""),
        )
        return jsonify(store.get_persona(data.get("user_id", "demo_user")))

    # ------------------------------------------------------------------
    # NEW: P0/P1/P2 capability endpoints (aligned with mem0/PowerMem)
    # ------------------------------------------------------------------

    # --- Memory Version History ---
    _history_tracker = MemoryHistoryTracker()

    @app.route("/api/memory/history", methods=["GET"])
    def api_memory_history():
        """Get version history for a specific memory or recent global events."""
        memory_id = request.args.get("memory_id")
        limit = int(request.args.get("limit", "50"))
        offset = int(request.args.get("offset", "0"))
        event_type = request.args.get("event_type")
        if memory_id:
            events = _history_tracker.get_history(memory_id)
            return jsonify({"memory_id": memory_id, "events": [e.to_dict() if hasattr(e, 'to_dict') else e for e in events]})
        events = _history_tracker.get_recent_history(limit=limit, offset=offset, event_type=event_type)
        return jsonify({"events": [e.to_dict() if hasattr(e, 'to_dict') else e for e in events]})

    @app.route("/api/history/stats", methods=["GET"])
    def api_history_stats():
        """History tracker statistics."""
        return jsonify(_history_tracker.get_stats())

    # --- Entity-centric Graph Traversal ---
    @app.route("/api/graph/entities", methods=["GET"])
    def api_graph_entities():
        """List all entities in the graph (entity-centric traversal)."""
        query = request.args.get("query", "")
        limit = int(request.args.get("limit", "50"))
        try:
            gs = HugeGraphGraphStore(hg_client=store.hg)
            if query:
                # Search entities matching query
                results = gs.search(query=query, limit=limit, max_hops=1)
                return jsonify({"entities": results, "count": len(results), "query": query})
            entities = gs.get_all_entities()
            return jsonify({"entities": entities, "count": len(entities)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/graph/relations", methods=["GET"])
    def api_graph_relations():
        """List all relations (edges) in the graph."""
        entity = request.args.get("entity", "")
        try:
            gs = HugeGraphGraphStore(hg_client=store.hg)
            relations = gs.get_all_relations()
            if entity:
                # Filter relations involving this entity
                filtered = [r for r in relations if entity in str(r)]
                return jsonify({"relations": filtered, "count": len(filtered), "entity": entity})
            return jsonify({"relations": relations, "count": len(relations)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/graph/search", methods=["POST"])
    def api_graph_search():
        """Entity-centric graph search with multi-hop traversal."""
        data = request.json or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400
        try:
            gs = HugeGraphGraphStore(hg_client=store.hg)
            results = gs.search(
                query=query,
                limit=int(data.get("limit", 10)),
                max_hops=int(data.get("max_hops", 2)),
            )
            return jsonify({"query": query, "results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- V3 Additive Extraction ---
    @app.route("/api/memory/extract", methods=["POST"])
    def api_extract_facts():
        """V3-style ADD-only extraction pipeline with MD5 dedup."""
        data = request.json or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400
        try:
            pipeline = AdditiveExtractionPipeline()
            stored_hashes = set(data.get("stored_hashes", []))
            existing_memories = data.get("existing_memories")
            result = pipeline.run(
                text=text,
                stored_hashes=stored_hashes,
                existing_memories=existing_memories,
                use_llm_dedup=bool(data.get("use_llm_dedup", False)),
            )
            # Ensure all values are JSON-serializable (convert sets to lists)
            if isinstance(result.get("hashes"), set):
                result["hashes"] = list(result["hashes"])
            if isinstance(result.get("entities"), set):
                result["entities"] = list(result["entities"])
            for k, v in result.items():
                if isinstance(v, set):
                    result[k] = list(v)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/memory/dedup", methods=["POST"])
    def api_dedup():
        """MD5 batch dedup check."""
        data = request.json or {}
        facts = data.get("facts", [])
        stored_hashes = set(data.get("stored_hashes", []))
        from hugegraph_llm.engines.memory.additive_extraction import batch_dedup
        new, dup, all_hashes = batch_dedup(facts, stored_hashes)
        return jsonify({"new_facts": new, "duplicate_facts": dup, "all_hashes": sorted(list(all_hashes))})

    # --- Hybrid Scoring Breakdown ---
    @app.route("/api/scoring/explain", methods=["POST"])
    def api_scoring_explain():
        """Show hybrid scoring breakdown for a query (sigmoid BM25 + entity boost + semantic)."""
        data = request.json or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400
        try:
            query_entities = extract_query_entities_simple(query)
            bm25_params = get_bm25_params(query)
            return jsonify({
                "query": query,
                "extracted_entities": query_entities,
                "bm25_sigmoid_params": {"midpoint": bm25_params[0], "steepness": bm25_params[1]},
                "explanation": f"BM25 sigmoid: midpoint={bm25_params[0]:.2f}, steepness={bm25_params[1]:.2f}; "
                               f"Entity boost: {len(query_entities)} entities extracted for boost computation; "
                               f"Fusion: additive (semantic + bm25_normalized + entity_boost)",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- LLM Query Rewrite ---
    @app.route("/api/query/rewrite", methods=["POST"])
    def api_query_rewrite():
        """LLM-enhanced query rewrite with entity extraction and intent classification."""
        data = request.json or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400
        try:
            engine = LLMQueryRewriteEngine()
            result = engine.rewrite(
                query=query,
                context=data.get("context"),
                user_profile=data.get("user_profile"),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Sub-Store Routing ---
    _route_store = RouteStore()

    @app.route("/api/routing/shards", methods=["GET"])
    def api_routing_shards():
        """List all active shards and their stats."""
        return jsonify(_route_store.get_shard_stats())

    @app.route("/api/routing/route", methods=["POST"])
    def api_routing_route():
        """Compute routing key for given metadata."""
        data = request.json or {}
        rk = compute_routing_key(
            user_id=data.get("user_id", "demo_user"),
            agent_id=data.get("agent_id"),
            app_name=data.get("app_name"),
            scope=data.get("scope", "private"),
        )
        return jsonify({"routing_key": rk, "metadata": data})

    @app.route("/api/routing/search", methods=["POST"])
    def api_routing_search():
        """Search across accessible shards."""
        data = request.json or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400
        results = _route_store.search_accessible(
            user_id=data.get("user_id", "demo_user"),
            agent_id=data.get("agent_id"),
            app_name=data.get("app_name"),
            query=query,
            limit=int(data.get("limit", 10)),
        )
        return jsonify({"results": results})

    # --- User Profile Auto-Extraction ---
    _profile_store = UserProfileStore()

    @app.route("/api/profile/auto", methods=["POST"])
    def api_profile_auto():
        """Auto-extract user profile from memory content (topics, name, preferences, aliases)."""
        data = request.json or {}
        user_id = data.get("user_id", "demo_user")
        memories = data.get("memories", [])
        if not memories:
            # Use existing memories from the backend
            existing = store.get_memories(user_id=user_id)
            memories = [m.get("content", "") for m in (existing.get("memories", []) if isinstance(existing, dict) else existing)]
        try:
            profile = _profile_store.update_from_memories(user_id=user_id, memories=memories)
            return jsonify(profile.to_dict() if hasattr(profile, 'to_dict') else profile)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/profile/users", methods=["GET"])
    def api_profile_users():
        """List all users with profiles."""
        return jsonify({"users": _profile_store.list_users()})

    @app.route("/api/profile/all", methods=["GET"])
    def api_profile_all():
        """Get all user profiles."""
        profiles = _profile_store.get_all_profiles()
        return jsonify({"profiles": [p.to_dict() if hasattr(p, 'to_dict') else p for p in profiles]})

    @app.route("/api/profile/inject", methods=["POST"])
    def api_profile_inject():
        """Get profile data formatted for query rewrite injection."""
        data = request.json or {}
        user_id = data.get("user_id", "demo_user")
        injector = ProfileInjector(profile_store=_profile_store)
        result = injector.get_profile_for_rewrite(user_id)
        return jsonify(result)

    # --- Memory Compression ---
    _compressor = MemoryCompressor()

    @app.route("/api/memory/compress", methods=["POST"])
    def api_memory_compress():
        """Run memory compression pipeline: cluster → summarize → prune → archive."""
        data = request.json or {}
        user_id = data.get("user_id", "demo_user")
        try:
            # Get user's memories for compression
            mem_data = store.get_memories(user_id=user_id)
            memories = mem_data.get("memories", []) if isinstance(mem_data, dict) else mem_data
            if not memories:
                return jsonify({"status": "no_memories", "stats": {}})
            result = _compressor.compress(memories=memories)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Multi-Agent Collaboration ---
    _agent_mgr = AgentMemoryManager()

    @app.route("/api/agent/check_access", methods=["POST"])
    def api_agent_check_access():
        """Check if an agent has permission to access a memory."""
        data = request.json or {}
        requesting_agent_id = data.get("requesting_agent_id", "")
        owner_agent_id = data.get("owner_agent_id", "")
        operation = data.get("operation", "read")
        scope = data.get("scope", "SHARED")
        privacy = data.get("privacy", "CONFIDENTIAL")
        if not requesting_agent_id:
            return jsonify({"error": "requesting_agent_id is required"}), 400
        # Normalize scope aliases: SHARED -> agent_group
        _scope_map = {"SHARED": "agent_group", "PERSONAL": "private",
                      "PRIVATE": "private", "PUBLIC": "public", "RESTRICTED": "restricted"}
        normalized_scope = _scope_map.get(scope.upper(), scope.lower())
        # Normalize privacy aliases
        _privacy_map = {"PUBLIC": "standard", "PRIVATE": "confidential",
                        "SENSITIVE": "sensitive", "CONFIDENTIAL": "confidential", "STANDARD": "standard"}
        normalized_privacy = _privacy_map.get(privacy.upper(), privacy.lower())
        # Build a memory dict for permission checking
        memory = {"scope": normalized_scope, "owner": owner_agent_id, "privacy": normalized_privacy}
        result = _agent_mgr.check_access(agent_id=requesting_agent_id, operation=operation, memory=memory)
        return jsonify({"allowed": result, "requesting_agent_id": requesting_agent_id,
                        "owner_agent_id": owner_agent_id, "operation": operation,
                        "scope": normalized_scope, "privacy": normalized_privacy})

    @app.route("/api/agent/filter_privacy", methods=["POST"])
    def api_agent_filter_privacy():
        """Filter memories for sharing based on privacy level (with sanitization)."""
        data = request.json or {}
        memories = data.get("memories", [])
        target_privacy = data.get("target_privacy", "standard")
        try:
            # Normalize privacy values: "PUBLIC" -> "standard", "PRIVATE" -> "confidential"
            _privacy_map = {"PUBLIC": "standard", "PRIVATE": "confidential",
                           "SENSITIVE": "sensitive", "CONFIDENTIAL": "confidential", "STANDARD": "standard"}
            normalized_target = _privacy_map.get(target_privacy.upper(), target_privacy.lower())
            normalized_memories = []
            for m in memories:
                norm_m = dict(m)
                if "privacy" in norm_m:
                    norm_m["privacy"] = _privacy_map.get(str(norm_m["privacy"]).upper(),
                                                         str(norm_m["privacy"]).lower())
                normalized_memories.append(norm_m)
            result = _agent_mgr.filter_for_sharing(memories=normalized_memories, target_privacy=normalized_target)
            return jsonify({"filtered_memories": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/agent/share", methods=["POST"])
    def api_agent_share():
        """Share a memory from one agent to another."""
        data = request.json or {}
        memory_id = data.get("memory_id", "")
        source_agent = data.get("source_agent", "")
        target_agent = data.get("target_agent", "")
        scope = data.get("scope", "agent_group")
        if not memory_id or not source_agent or not target_agent:
            return jsonify({"error": "memory_id, source_agent, target_agent are required"}), 400
        result = _agent_mgr.share_memory_to_agent(
            memory_id=memory_id, source_agent=source_agent,
            target_agent=target_agent, scope=scope,
        )
        return jsonify({"success": result})

    @app.route("/api/agent/groups", methods=["GET"])
    def api_agent_groups():
        """List collaboration groups for an agent."""
        agent_id = request.args.get("agent_id", "")
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400
        groups = _agent_mgr.get_agent_groups(agent_id=agent_id)
        return jsonify({"groups": groups})

    @app.route("/api/agent/shared_with", methods=["GET"])
    def api_agent_shared_with():
        """List memories shared with an agent."""
        agent_id = request.args.get("agent_id", "")
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400
        shared = _agent_mgr.get_shared_with_agent(agent_id=agent_id)
        return jsonify({"shared_memories": shared})

    # --- Collaboration Group Management ---
    _collab_broker = CollaborationBroker()

    @app.route("/api/collab/create_group", methods=["POST"])
    def api_collab_create_group():
        """Create a collaboration group."""
        data = request.json or {}
        group_id = data.get("group_id", "")
        name = data.get("name", data.get("group_name", ""))
        members = data.get("members", [])
        scope = data.get("scope", "agent_group")
        if not group_id or not name:
            return jsonify({"error": "group_id and name/group_name are required"}), 400
        try:
            result = _collab_broker.create_group(
                group_id=group_id, name=name, members=members, scope=MemoryScope(scope),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/collab/add_member", methods=["POST"])
    def api_collab_add_member():
        """Add an agent to a collaboration group."""
        data = request.json or {}
        group_id = data.get("group_id", "")
        agent_id = data.get("agent_id", "")
        if not group_id or not agent_id:
            return jsonify({"error": "group_id and agent_id are required"}), 400
        result = _collab_broker.add_member(group_id=group_id, agent_id=agent_id)
        return jsonify({"success": result})

    # --- FAISS Index Management ---
    @app.route("/api/faiss/stats", methods=["GET"])
    def api_faiss_stats():
        """FAISS index statistics (vectors, tombstones, dimension)."""
        try:
            stats = store.get_stats()
            faiss_info = stats.get("faiss", {})
            return jsonify(faiss_info)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/faiss/compact", methods=["POST"])
    def api_faiss_compact():
        """Compact FAISS index: rebuild without tombstones to reclaim space."""
        try:
            # Use store's existing FAISS index if available
            if hasattr(store, 'faiss_index') and store.faiss_index is not None:
                count = store.faiss_index.compact()
            else:
                idx = FaissDeletableIndex()
                count = idx.compact()
            return jsonify({"compact_count": count, "status": "compacted"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Privacy Demo (Sanitization) ---
    @app.route("/api/privacy/filter", methods=["POST"])
    def api_privacy_filter():
        """Filter/sanitize text content based on privacy level (regex-based PII removal)."""
        data = request.json or {}
        content = data.get("content", "")
        privacy_level = data.get("privacy_level", "standard")
        # Normalize: accept uppercase aliases
        _pl_map = {"PUBLIC": "standard", "PRIVATE": "confidential",
                   "SENSITIVE": "sensitive", "CONFIDENTIAL": "confidential", "STANDARD": "standard"}
        normalized = _pl_map.get(privacy_level.upper(), privacy_level.lower())
        try:
            pf = PrivacyFilter()
            filtered = pf.filter(content=content, privacy_level=PrivacyLevel(normalized))
            return jsonify({"original": content, "filtered": filtered, "privacy_level": normalized})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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
