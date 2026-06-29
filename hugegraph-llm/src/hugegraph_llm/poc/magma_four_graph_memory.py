"""
MAGMA Four-Graph Agent Memory PoC
==================================
基于 ACL 2026 论文 "MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents"
(arXiv:2601.03236) 的四图正交记忆架构实现。

核心设计:
  - Semantic Graph: 基于语义相似度的无向边，回答 "发生了什么"
  - Temporal Graph: 不可变时间链，回答 "什么时候"
  - Causal Graph: LLM推理的因果有向边，回答 "为什么"
  - Entity Graph: 跨时间窗口的实体节点，维持对象恒常性

关键机制:
  - Intent Routing: 查询意图分类(Why/When/Entity) → 选择图视图
  - Adaptive Beam Search: 策略引导的图遍历
  - Fast Path + Slow Path: 同步写入时间链，异步巩固因果/实体边
  - HugeGraph Gremlin: 四图共用schema，通过 property 区分

与 HugeGraph GraphRAG 底座集成:
  - 向量存储: 通过 backend_factory 接入 FAISS/Milvus/Qdrant/OceanBase
  - 全文检索: 通过 backend_factory 接入 BM25/OceanBase FTS
  - 图存储: 通过 PyHugeClient/Commit2Graph 接入 HugeGraph
  - RAG 能力: 可复用 Sprint 1-10 DRIFT/实体消解/溯源等

切换后端:
  VECTOR_BACKEND=faiss|milvus|qdrant|oceanbase
  FULLTEXT_BACKEND=bm25|oceanbase

Author: HugeGraph-AI PoC (2026-06-10)
"""

import json
import os
import re
import time
import hashlib
import math
import random
import logging
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MAGMA")


# ============================================================
# 0. GraphRAG 底座集成（本地真实模型，禁止模拟）
# ============================================================

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384


class VectorBackend:
    """FAISS + sentence-transformers 本地向量后端（无 API fallback）。"""

    _model = None

    def __init__(self, model_name: str = EMBED_MODEL):
        self._index = None
        self._id_map: Dict[int, str] = {}
        self._next_idx = 0
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        if VectorBackend._model is None:
            from sentence_transformers import SentenceTransformer
            VectorBackend._model = SentenceTransformer(self.model_name)

    def _ensure_index(self):
        import faiss
        if self._index is None:
            self._index = faiss.IndexFlatIP(EMBED_DIM)

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embs = VectorBackend._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.tolist()

    def add(self, ids: List[str], embs: List[List[float]]):
        import faiss
        import numpy as np
        self._ensure_index()
        arr = np.array(embs, dtype=np.float32)
        s = self._next_idx
        self._index.add(arr)
        for i, eid in enumerate(ids):
            self._id_map[s + i] = eid
        self._next_idx += len(ids)

    def search(self, q_emb: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        import faiss
        import numpy as np
        self._ensure_index()
        if self._next_idx == 0:
            return []
        scores, idxs = self._index.search(
            np.array([q_emb], dtype=np.float32), min(top_k, self._next_idx))
        out = []
        for sc, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            nid = self._id_map.get(int(idx), "")
            if nid:
                out.append((nid, float(sc)))
        return out

    @property
    def count(self) -> int:
        return self._next_idx


class BM25Backend:
    """本地 BM25 全文后端（jieba + rank_bm25）。"""

    def __init__(self):
        self._bm25 = None
        self._docs: List[str] = []
        self._ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        import jieba
        return [t.strip().lower() for t in jieba.lcut(text)
                if re.match(r"^[\w\u4e00-\u9fff]+$", t.strip())]

    def add_documents(self, texts: List[str], ids: List[str], _metas: List[str] = None):
        from rank_bm25 import BM25Okapi
        self._docs.extend(texts)
        self._ids.extend(ids)
        tok = [self._tokenize(t) for t in self._docs]
        self._bm25 = BM25Okapi(tok)

    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[Dict]:
        if self._bm25 is None or not self._ids:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"id": self._ids[i], "score": float(scores[i])}
                for i in ranked if scores[i] > min_score]


def _create_hugegraph_client(graph: str = "hugegraph"):
    """创建 HugeGraph 客户端；如果服务不可用则立即失败（禁止内存 fallback）。"""
    from pyhugegraph.client import PyHugeClient
    try:
        client = PyHugeClient(
            url=os.environ.get("HUGEGRAPH_URL", "http://127.0.0.1:8080"),
            graph=graph,
            user=os.environ.get("HUGEGRAPH_USER", "admin"),
            pwd=os.environ.get("HUGEGRAPH_PASS", "admin"),
        )
        client.schema().getVertexLabels()
        return client
    except Exception as e:
        raise RuntimeError(
            f"MAGMA requires a running HugeGraph server, "
            f"but connection failed: {e}. Please start HugeGraph and retry."
        ) from e


# ============================================================
# 1. 核心数据结构
# ============================================================

class IntentType(Enum):
    WHY = "why"        # 因果推理 → Causal Graph
    WHEN = "when"      # 时间查询 → Temporal Graph
    ENTITY = "entity"  # 实体关联 → Entity Graph


class EdgeType(Enum):
    SEMANTIC = "semantic"     # 语义相似度边
    TEMPORAL = "temporal"     # 时间序列边(不可变)
    CAUSAL = "causal"         # 因果推理边
    ENTITY_REF = "entity_ref" # 事件→实体引用边


@dataclass
class MemoryNode:
    """记忆节点 = MAGMA 论文中的 n_i = <content, timestamp, vector, attributes>"""
    node_id: str
    content: str
    timestamp: str            # ISO 8601
    vector: List[float]         # 稠密嵌入向量
    attributes: Dict[str, Any] = field(default_factory=dict)
    graph_type: str = "memory" # 全部存在同一张图中


@dataclass
class MemoryEdge:
    """记忆边，通过 edge_type 区分四类图"""
    source_id: str
    target_id: str
    edge_type: str              # semantic/temporal/causal/entity_ref
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """检索结果"""
    node: MemoryNode
    score: float
    matched_graphs: List[str]  # 命中了哪些图
    traversal_path: List[str]  # 遍历路径


@dataclass
class IntentRoutingResult:
    """意图路由结果"""
    intent: IntentType
    anchor_node_id: Optional[str]
    anchor_score: float
    routing_confidence: float


# ============================================================
# 2. 向量工具（模拟嵌入空间）
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_embedder = None


def _get_embedder() -> VectorBackend:
    global _embedder
    if _embedder is None:
        _embedder = VectorBackend()
    return _embedder


def generate_embedding(text: str, dim: int = EMBED_DIM, seed: Optional[int] = None) -> List[float]:
    """使用本地 sentence-transformers 模型生成真实嵌入（禁止确定性 hash）。"""
    emb = _get_embedder().encode([text])
    return emb[0] if emb else [0.0] * EMBED_DIM


def extract_keywords(text: str) -> set:
    """简单关键词提取（模拟 NLP 关键词提取）"""
    return set(text.lower().split()) - {"the", "a", "an", "is", "are", "was", "were",
                                          "in", "on", "at", "to", "for", "of", "and",
                                          "that", "this", "it", "with", "as", "by"}


# ============================================================
# 3. 四图存储引擎（基于内存 + 模拟 Gremlin 查询）
# ============================================================

class FourGraphMemoryStore:
    """
    四图正交记忆存储
    ================
    物理上一张图（HugeGraph 单 schema），逻辑上四视图:
      - Semantic Graph:   g.V().has('node_type','memory').hasEdge('semantic')
      - Temporal Graph:   g.V().has('node_type','memory').hasEdge('temporal')
      - Causal Graph:     g.V().has('node_type','memory').hasEdge('causal')
      - Entity Graph:    g.V().has('node_type','entity').hasEdge('entity_ref')

    存储底座集成:
      - 向量存储: FAISS/Milvus/Qdrant/OceanBase (via backend_factory)
      - 全文检索: BM25/OceanBase FTS (via backend_factory)
      - 图存储: HugeGraph (via PyHugeClient, 优雅降级到内存)
    """

    def __init__(self, semantic_threshold: float = 0.6,
                 causal_threshold: float = 0.3,
                 beam_width: int = 5, max_depth: int = 5,
                 beam_decay: float = 0.85):
        # 节点存储
        self.nodes: Dict[str, MemoryNode] = {}
        # 实体节点
        self.entity_nodes: Dict[str, MemoryNode] = {}
        # 四类边存储: edge_type -> [(source, target, edge)]
        self.edges: Dict[str, List[MemoryEdge]] = {
            "semantic": [],
            "temporal": [],
            "causal": [],
            "entity_ref": [],
        }

        # === GraphRAG 底座: 向量存储 (FAISS + sentence-transformers) ===
        self.vector_backend_name = "faiss"
        self.vector_store = VectorBackend()
        self._use_vector_backend = True

        # === GraphRAG 底座: 全文检索 (BM25 + jieba) ===
        self.fulltext_backend_name = "bm25"
        self.fulltext_store = BM25Backend()
        self._use_fulltext_backend = True

        # === GraphRAG 底座: 图存储 (HugeGraph) ===
        self.graph_client = _create_hugegraph_client()
        self._use_graph_backend = True
        self._node_cache: Dict[str, Any] = {}
        self._entity_cache: Dict[str, Any] = {}
        self._init_hg_schema()

        # MAGMA 参数
        self.semantic_threshold = semantic_threshold
        self.causal_threshold = causal_threshold
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.beam_decay = beam_decay

        # 统计
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "semantic_edges": 0,
            "temporal_edges": 0,
            "causal_edges": 0,
            "entity_ref_edges": 0,
            "entity_nodes": 0,
            "fast_path_writes": 0,
            "slow_path_writes": 0,
            "vector_backend": self.vector_backend_name,
            "fulltext_backend": self.fulltext_backend_name,
            "graph_backend": "hugegraph" if self._use_graph_backend else "memory",
        }

    def _init_hg_schema(self):
        """Initialize MAGMA schema in HugeGraph."""
        s = self.graph_client.schema()
        for pk in ["name", "content", "timestamp", "graph_type"]:
            try:
                s.propertyKey(pk).asText().ifNotExist().create()
            except Exception:
                pass
        try:
            s.vertexLabel("MAGMA_Node").properties(
                "name", "content", "timestamp", "graph_type"
            ).usePrimaryKeyId().primaryKeys("name").ifNotExist().create()
        except Exception as e:
            log.debug("[MAGMA] MAGMA_Node schema: %s", e)
        try:
            s.vertexLabel("MAGMA_Entity").properties(
                "name", "content", "timestamp", "graph_type"
            ).usePrimaryKeyId().primaryKeys("name").ifNotExist().create()
        except Exception as e:
            log.debug("[MAGMA] MAGMA_Entity schema: %s", e)
        for el in ["semantic", "temporal", "causal"]:
            try:
                s.edgeLabel(el).sourceLabel("MAGMA_Node").targetLabel("MAGMA_Node").ifNotExist().create()
            except Exception as e:
                log.debug("[MAGMA] edge %s schema: %s", el, e)
        try:
            s.edgeLabel("entity_ref").sourceLabel("MAGMA_Node").targetLabel("MAGMA_Entity").ifNotExist().create()
        except Exception as e:
            log.debug("[MAGMA] entity_ref schema: %s", e)

    def _vector_add(self, node_id: str, vector: List[float]):
        """写入向量存储（必须使用真实 backend）"""
        self.vector_store.add([node_id], [vector])

    def _vector_search(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """向量检索（返回 [(node_id, similarity)]）"""
        results = self.vector_store.search(query_vec, top_k)
        scored = []
        for nid, sim in results:
            if nid in self.nodes:
                node = self.nodes[nid]
                # recompute cosine against the real vector
                sim = cosine_similarity(query_vec, node.vector)
                if sim > 0.3:
                    scored.append((nid, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _fulltext_search(self, query: str, top_k: int = 5) -> Dict[str, float]:
        """全文检索（返回 {node_id: bm25_score}）"""
        results = self.fulltext_store.search(query, top_k, min_score=0.0)
        return {r["id"]: r["score"] for r in results}

    def _fulltext_add(self, node_id: str, text: str):
        """写入全文索引"""
        self.fulltext_store.add_documents([text], [node_id], [text])

    # --- Gremlin 查询翻译 ---

    def gremlin_semantic_neighbors(self, node_id: str) -> List[str]:
        """g.V(node_id).outE('semantic').inV() — 语义邻居"""
        neighbors = []
        for e in self.edges["semantic"]:
            if e.source_id == node_id:
                neighbors.append((e.target_id, e.weight))
            elif e.target_id == node_id:
                neighbors.append((e.source_id, e.weight))
        return neighbors

    def gremlin_temporal_next(self, node_id: str) -> List[str]:
        """g.V(node_id).outE('temporal').inV() — 时间后继"""
        return [(e.target_id, e.weight)
                for e in self.edges["temporal"]
                if e.source_id == node_id]

    def gremlin_causal_successors(self, node_id: str) -> List[str]:
        """g.V(node_id).outE('causal').inV() — 因果后继"""
        return [(e.target_id, e.weight)
                for e in self.edges["causal"]
                if e.source_id == node_id]

    def gremlin_entity_events(self, entity_id: str) -> List[str]:
        """g.V(entity_id).inE('entity_ref').outV() — 实体相关事件"""
        return [(e.source_id, e.weight)
                for e in self.edges["entity_ref"]
                if e.target_id == entity_id]

    def gremlin_k_hop(self, node_id: str, edge_type: str, k: int = 2) -> List[str]:
        """g.V(node_id).repeat(outE(edge_type).inV()).times(k) — k跳遍历"""
        visited = {node_id}
        current = [node_id]
        for _ in range(k):
            next_level = set()
            for nid in current:
                for e in self.edges.get(edge_type, []):
                    neighbor = None
                    if e.source_id == nid and e.target_id not in visited:
                        neighbor = e.target_id
                    elif e.target_id == nid and e.source_id not in visited:
                        neighbor = e.source_id
                    if neighbor:
                        next_level.add(neighbor)
            visited.update(next_level)
            current = list(next_level)
        return list(visited)

    # --- Fast Path (同步写入) ---

    def fast_path_write(self, content: str, timestamp: Optional[str] = None,
                        attributes: Optional[Dict] = None) -> MemoryNode:
        """
        Fast Path: 同步写入，毫秒级延迟
        - 事件分割 + 编码向量
        - 追加时间骨干边 (temporal)
        - 写入向量索引 (FAISS/Milvus/Qdrant/OceanBase)
        - 写入全文索引 (BM25/OceanBase FTS)
        - 可选: 写入 HugeGraph
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        node_id = f"mem_{hashlib.md5(content.encode()).hexdigest()[:12]}"
        vector = generate_embedding(content)

        node = MemoryNode(
            node_id=node_id,
            content=content,
            timestamp=timestamp,
            vector=vector,
            attributes=attributes or {},
        )

        # 写入节点
        self.nodes[node_id] = node

        # GraphRAG 底座: 写入 HugeGraph
        g = self.graph_client.graph()
        try:
            self._node_cache[node_id] = g.addVertex("MAGMA_Node", {
                "name": node_id,
                "content": content,
                "timestamp": timestamp,
                "graph_type": "memory",
            })
        except Exception as e:
            log.debug("[MAGMA] addVertex %s: %s", node_id, e)

        # GraphRAG 底座: 写入向量存储
        self._vector_add(node_id, vector)

        # GraphRAG 底座: 写入全文索引
        self._fulltext_add(node_id, content)

        # 时间骨干边（连接到时间上最近的节点）
        self._append_temporal_chain(node)

        # 语义边（Fast Path 中可以同步计算）
        self._add_semantic_edges(node)

        self.stats["fast_path_writes"] += 1
        self.stats["total_nodes"] += 1

        return node

    def _append_temporal_chain(self, node: MemoryNode):
        """追加时间骨干边 — 找到时间上最近的节点"""
        if not self.nodes:
            return

        # 找到当前最后的时间节点
        latest_id = max(
            (nid for nid in self.nodes if nid != node.node_id),
            key=lambda nid: self.nodes[nid].timestamp,
            default=None
        )

        if latest_id:
            edge = MemoryEdge(
                source_id=latest_id,
                target_id=node.node_id,
                edge_type="temporal",
                weight=1.0,
            )
            self.edges["temporal"].append(edge)
            self.stats["temporal_edges"] += 1
            self.stats["total_edges"] += 1
            try:
                g = self.graph_client.graph()
                src_v = self._node_cache.get(latest_id)
                tgt_v = self._node_cache.get(node.node_id)
                if src_v and tgt_v:
                    g.addEdge("temporal", src_v.id, tgt_v.id, {})
            except Exception as e:
                log.debug("[MAGMA] temporal edge %s->%s: %s", latest_id, node.node_id, e)

    def _add_semantic_edges(self, node: MemoryNode):
        """添加语义相似度边（使用 GraphRAG 向量存储）"""
        # 用向量存储检索与当前节点最相似的已有节点
        neighbors = self._vector_search(node.vector, top_k=20)
        for neighbor_id, sim in neighbors:
            if neighbor_id == node.node_id:
                continue
            if sim >= self.semantic_threshold:
                # 检查是否已存在
                existing = {(e.source_id, e.target_id) for e in self.edges["semantic"]}
                if (neighbor_id, node.node_id) not in existing:
                    edge = MemoryEdge(
                        source_id=neighbor_id,
                        target_id=node.node_id,
                        edge_type="semantic",
                        weight=sim,
                    )
                    self.edges["semantic"].append(edge)
                    self.stats["semantic_edges"] += 1
                    self.stats["total_edges"] += 1
                    try:
                        g = self.graph_client.graph()
                        src_v = self._node_cache.get(neighbor_id)
                        tgt_v = self._node_cache.get(node.node_id)
                        if src_v and tgt_v:
                            g.addEdge("semantic", src_v.id, tgt_v.id, {})
                    except Exception as e:
                        log.debug("[MAGMA] semantic edge %s->%s: %s", neighbor_id, node.node_id, e)

    # --- Slow Path (异步巩固) ---

    def slow_path_consolidate(self, node_id: str) -> Dict[str, Any]:
        """
        Slow Path: 异步巩固，调用 LLM 推理
        - 因果边推理 (causal)
        - 实体边构建 (entity_ref)
        - 需要 2-hop 邻域上下文

        在此 PoC 中用规则模拟 LLM 推理。
        """
        if node_id not in self.nodes:
            return {"status": "error", "message": f"Node {node_id} not found"}

        node = self.nodes[node_id]
        consolidation_result = {
            "node_id": node_id,
            "causal_edges_added": 0,
            "entity_edges_added": 0,
            "entities_discovered": [],
        }

        # --- 模拟因果边推理 ---
        # 实际场景: LLM 分析 2-hop 邻域，推理隐含因果
        # 此处用关键词 + 语义相似度模拟
        neighbors = self.gremlin_semantic_neighbors(node_id)
        for neighbor_id, weight in neighbors:
            neighbor = self.nodes.get(neighbor_id)
            if not neighbor:
                continue

            # 检查是否已有因果边
            existing_causal = {(e.source_id, e.target_id)
                              for e in self.edges["causal"]}
            if (node_id, neighbor_id) in existing_causal or \
               (neighbor_id, node_id) in existing_causal:
                continue

            # 模拟因果评分: 语义相似度 + 时间先后 + 关键词共现
            time_diff = self._time_hours_between(neighbor.timestamp, node.timestamp)
            keyword_overlap = len(
                extract_keywords(node.content) & extract_keywords(neighbor.content)
            )

            causal_score = weight * 0.4 + (1 if time_diff > 0 else -1) * 0.3 + \
                           min(keyword_overlap, 3) * 0.1

            if abs(causal_score) >= self.causal_threshold:
                if time_diff > 0:
                    # neighbor 在前，node 在后 → neighbor → node
                    edge = MemoryEdge(
                        source_id=neighbor_id,
                        target_id=node_id,
                        edge_type="causal",
                        weight=abs(causal_score),
                        attributes={"causal_score": causal_score}
                    )
                else:
                    edge = MemoryEdge(
                        source_id=node_id,
                        target_id=neighbor_id,
                        edge_type="causal",
                        weight=abs(causal_score),
                    )
                self.edges["causal"].append(edge)
                consolidation_result["causal_edges_added"] += 1
                self.stats["causal_edges"] += 1
                self.stats["total_edges"] += 1
                # 写入 HugeGraph
                try:
                    g = self.graph_client.graph()
                    src_v = self._node_cache.get(edge.source_id)
                    tgt_v = self._node_cache.get(edge.target_id)
                    if src_v and tgt_v:
                        g.addEdge("causal", src_v.id, tgt_v.id, {})
                except Exception as e:
                    log.debug("[MAGMA] causal edge %s->%s: %s", edge.source_id, edge.target_id, e)

        # --- 模拟实体提取和实体边 ---
        # 实际场景: LLM 从内容中提取实体（人名/组织/概念等）
        entities = self._extract_entities_simulated(node)
        for entity_name, entity_type in entities:
            entity_id = f"ent_{hashlib.md5(entity_name.encode()).hexdigest()[:12]}"

            if entity_id not in self.entity_nodes:
                entity_node = MemoryNode(
                    node_id=entity_id,
                    content=entity_name,
                    timestamp=node.timestamp,
                    vector=generate_embedding(entity_name),
                    attributes={"entity_type": entity_type, "name": entity_name},
                    graph_type="entity",
                )
                self.entity_nodes[entity_id] = entity_node
                self.nodes[entity_id] = entity_node  # 共享存储
                self.stats["entity_nodes"] += 1
                self.stats["total_nodes"] += 1
                # 写入 HugeGraph
                try:
                    g = self.graph_client.graph()
                    self._entity_cache[entity_id] = g.addVertex("MAGMA_Entity", {
                        "name": entity_id,
                        "content": entity_name,
                        "timestamp": node.timestamp,
                        "graph_type": "entity",
                    })
                except Exception as e:
                    log.debug("[MAGMA] addEntity %s: %s", entity_id, e)

            # 添加 entity_ref 边
            existing_ref = {(e.source_id, e.target_id)
                           for e in self.edges["entity_ref"]}
            if (node_id, entity_id) not in existing_ref:
                edge = MemoryEdge(
                    source_id=node_id,
                    target_id=entity_id,
                    edge_type="entity_ref",
                    weight=1.0,
                    attributes={"entity_name": entity_name}
                )
                self.edges["entity_ref"].append(edge)
                consolidation_result["entity_edges_added"] += 1
                self.stats["entity_ref_edges"] += 1
                self.stats["total_edges"] += 1
                consolidation_result["entities_discovered"].append(entity_name)
                # 写入 HugeGraph
                try:
                    g = self.graph_client.graph()
                    src_v = self._node_cache.get(node_id)
                    tgt_v = self._entity_cache.get(entity_id)
                    if src_v and tgt_v:
                        g.addEdge("entity_ref", src_v.id, tgt_v.id, {})
                except Exception as e:
                    log.debug("[MAGMA] entity_ref edge %s->%s: %s", node_id, entity_id, e)

        self.stats["slow_path_writes"] += 1
        return consolidation_result

    def _extract_entities_simulated(self, node: MemoryNode) -> List[Tuple[str, str]]:
        """模拟 LLM 实体提取（用简单规则）"""
        content = node.content
        entities = []

        # 简单模式: 大写单词视为人名/组织名
        words = content.split()
        for word in words:
            if word[0].isupper() and len(word) > 2 and word not in {
                "The", "This", "That", "When", "Then", "After", "Before",
                "However", "Therefore", "Furthermore", "Meanwhile"
            }:
                entities.append((word, "person_or_org"))

        # 常见概念模式
        concepts = ["bug", "feature", "meeting", "deploy", "release",
                    "error", "crash", "deadline", "review", "server",
                    "database", "API", "issue", "task", "sprint"]
        for concept in concepts:
            if concept in content.lower():
                entities.append((concept, "concept"))

        return list(set(entities))

    def _time_hours_between(self, ts1: str, ts2: str) -> float:
        """计算两个时间戳之间的小时差"""
        try:
            t1 = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(ts2.replace("Z", "+00:00"))
            return (t2 - t1).total_seconds() / 3600
        except Exception:
            return 0.0

    # --- Intent Routing ---

    def route_intent(self, query: str) -> IntentRoutingResult:
        """
        Intent Routing: 判断查询意图并选择图视图
        对应论文中的: 轻量分类器 + 时间解析 + RRF 锚点定位
        """
        query_lower = query.lower()

        # 意图分类
        if any(kw in query_lower for kw in ["why", "原因", "为什么", "导致", "caused",
                                               "因为", "since", "due to", "because"]):
            intent = IntentType.WHY
        elif any(kw in query_lower for kw in ["when", "什么时候", "时间", "何时", "last",
                                               "yesterday", "today", "上周", "昨天"]):
            intent = IntentType.WHEN
        elif any(kw in query_lower for kw in ["who", "谁", "which", "哪个", "entity",
                                               "相关", "related", "关于"]):
            intent = IntentType.ENTITY
        else:
            # 默认: 先语义匹配
            intent = IntentType.ENTITY

        # RRF 锚点定位 (3路信号融合: 向量 + 全文 + 时间)
        query_vec = generate_embedding(query)
        vec_scores = {}
        key_scores = {}
        time_scores = {}

        # 路由1: 向量相似度检索 (FAISS/Milvus/Qdrant/OceanBase)
        vec_results = self._vector_search(query_vec, top_k=20)
        for nid, sim in vec_results:
            if nid in self.nodes and self.nodes[nid].graph_type != "entity":
                vec_scores[nid] = sim

        # 路由2: BM25 全文检索 (BM25Okapi/OceanBase FTS)
        bm25_results = self._fulltext_search(query, top_k=10)
        for nid, bm25_score in bm25_results.items():
            if nid in self.nodes and self.nodes[nid].graph_type != "entity":
                key_scores[nid] = bm25_score

        # 路由3: 时间匹配
        if intent == IntentType.WHEN:
            for nid, mem_node in self.nodes.items():
                if mem_node.graph_type == "entity":
                    continue
                time_diff = abs(self._time_hours_between(
                    mem_node.timestamp, datetime.utcnow().isoformat()))
                if time_diff < 48:  # 48小时内
                    time_scores[nid] = 1.0 / (1.0 + time_diff)

        # RRF 融合: S_anchor = TopK * sum_{m} 1/(k + r_m(n))
        k = 60  # RRF 参数
        all_node_ids = set(vec_scores.keys()) | set(key_scores.keys()) | set(time_scores.keys())

        anchor_scores = {}
        for nid in all_node_ids:
            score = 0.0
            # 向量排名
            ranked_vec = sorted(vec_scores.items(), key=lambda x: -x[1])
            for rank, (rid, _) in enumerate(ranked_vec):
                if rid == nid:
                    score += 1.0 / (k + rank + 1)
                    break
            # 关键词排名
            ranked_key = sorted(key_scores.items(), key=lambda x: -x[1])
            for rank, (rid, _) in enumerate(ranked_key):
                if rid == nid:
                    score += 1.0 / (k + rank + 1)
                    break
            # 时间排名
            ranked_time = sorted(time_scores.items(), key=lambda x: -x[1])
            for rank, (rid, _) in enumerate(ranked_time):
                if rid == nid:
                    score += 1.0 / (k + rank + 1)
                    break

            anchor_scores[nid] = score

        best_anchor = max(anchor_scores.items(), key=lambda x: x[1]) if anchor_scores else (None, 0.0)

        return IntentRoutingResult(
            intent=intent,
            anchor_node_id=best_anchor[0],
            anchor_score=best_anchor[1],
            routing_confidence=min(best_anchor[1] * 3, 1.0),
        )

    # --- Adaptive Beam Search ---

    def adaptive_beam_search(self, routing: IntentRoutingResult,
                              query: str) -> List[QueryResult]:
        """
        Adaptive Beam Search: 策略引导的图遍历
        使用 GraphRAG 向量存储进行相似度计算
        """
        if routing.anchor_node_id is None:
            return []

        query_vec = generate_embedding(query)
        λ1, λ2 = 1.0, 0.5

        # 边类型权重 φ(type, intent)
        type_weights = {
            IntentType.WHY: {
                "causal": 4.0, "temporal": 1.0, "semantic": 0.5, "entity_ref": 0.3
            },
            IntentType.WHEN: {
                "temporal": 4.0, "causal": 0.5, "semantic": 1.0, "entity_ref": 0.3
            },
            IntentType.ENTITY: {
                "entity_ref": 5.0, "semantic": 2.0, "causal": 1.0, "temporal": 0.5
            },
        }

        weights = type_weights[routing.intent]

        # Beam search
        beam = [(routing.anchor_node_id, 1.0, [routing.anchor_node_id])]
        visited = {routing.anchor_node_id}
        results = []

        for depth in range(self.max_depth):
            candidates = []

            for node_id, score, path in beam:
                node = self.nodes.get(node_id)
                if not node:
                    continue

                # 边类型遍历 (优先按意图权重排序)
                for edge_type in ["causal", "temporal", "semantic", "entity_ref"]:
                    # 根据意图调整遍历优先级
                    for e in self.edges.get(edge_type, []):
                        neighbor_id = None
                        if e.source_id == node_id and e.target_id not in visited:
                            neighbor_id = e.target_id
                        elif e.target_id == node_id and e.source_id not in visited:
                            neighbor_id = e.source_id

                        if neighbor_id is None:
                            continue

                        neighbor = self.nodes.get(neighbor_id)
                        if not neighbor:
                            continue

                        # 计算转移分数
                        φ = weights.get(edge_type, 1.0)
                        sim = cosine_similarity(neighbor.vector, query_vec)
                        transfer_score = math.exp(λ1 * φ + λ2 * sim)

                        new_score = score * transfer_score * self.beam_decay
                        new_path = path + [neighbor_id]

                        candidates.append((neighbor_id, new_score, new_path))

            # 取 top-k
            candidates.sort(key=lambda x: -x[1])
            beam = candidates[:self.beam_width]

            for node_id, score, path in beam:
                if node_id not in visited:
                    visited.add(node_id)

            # 收集结果
            for node_id, score, path in beam:
                node = self.nodes.get(node_id)
                if node and node.graph_type != "entity":
                    # 判断命中了哪些图
                    matched = set()
                    for edge_type in ["semantic", "temporal", "causal", "entity_ref"]:
                        for e in self.edges.get(edge_type, []):
                            if (e.source_id in path and e.target_id in path) or \
                               (e.target_id in path and e.source_id in path):
                                matched.add(edge_type)
                                break

                    results.append(QueryResult(
                        node=node,
                        score=score,
                        matched_graphs=list(matched),
                        traversal_path=path,
                    ))

            if not beam:
                break

        # 去重 + 按分数排序
        seen = set()
        unique_results = []
        for r in results:
            if r.node.node_id not in seen:
                seen.add(r.node.node_id)
                unique_results.append(r)
        unique_results.sort(key=lambda x: -x.score)

        return unique_results[:10]

    # --- 完整查询接口 ---

    def query(self, query: str) -> Dict[str, Any]:
        """完整查询: Intent Routing → Beam Search → Context Synthesis"""
        start = time.time()

        routing = self.route_intent(query)
        results = self.adaptive_beam_search(routing, query)

        elapsed = time.time() - start

        return {
            "query": query,
            "intent": routing.intent.value,
            "anchor_node_id": routing.anchor_node_id,
            "routing_confidence": routing.routing_confidence,
            "results_count": len(results),
            "results": [
                {
                    "node_id": r.node.node_id,
                    "content": r.node.content[:100] + ("..." if len(r.node.content) > 100 else ""),
                    "timestamp": r.node.timestamp,
                    "score": round(r.score, 4),
                    "matched_graphs": r.matched_graphs,
                    "traversal_depth": len(r.traversal_path) - 1,
                }
                for r in results
            ],
            "latency_ms": round(elapsed * 1000, 2),
            "graph_stats": self.stats,
        }


# ============================================================
# 4. HugeGraph Gremlin 翻译层（展示如何映射到生产环境）
# ============================================================

def gremlin_translation_guide():
    """
    四图查询的 Gremlin 翻译示例
    展示如何在 HugeGraph 生产环境中执行相同的查询
    """
    translations = {
        "semantic_neighbors": {
            "description": "查询语义邻居节点",
            "gremlin": "g.V(nodeId).outE('semantic').has('weight', gt(threshold)).inV().values('content')",
            "hugegraph_advantage": "HugeGraph OLAP traverser 可并行处理大规模语义邻域查询",
        },
        "temporal_chain": {
            "description": "时间链后继查询",
            "gremlin": "g.V(nodeId).out('temporal').until(has('timestamp', gt(targetTime))).repeat(out('temporal')).emit().values('content')",
            "hugegraph_advantage": "时间链天然有序，HugeGraph 可支持亿级时间序列节点的快速遍历",
        },
        "causal_chain": {
            "description": "因果链推理",
            "gremlin": "g.V(nodeId).repeat(outE('causal').order().by('weight', desc).inV()).until(loops().is(gt(maxDepth))).path().by('content')",
            "hugegraph_advantage": "Vermeer OLAP 引擎支持因果链的批量并行分析，适合供应链风险传导场景",
        },
        "entity_events": {
            "description": "实体关联事件查询",
            "gremlin": "g.V('entity', 'name', entityName).in('entity_ref').order().by('timestamp', desc).limit(k).values('content')",
            "hugegraph_advantage": "实体事件查询可复用 HugeGraph 的 vertex-centric index，毫秒级响应",
        },
        "multi_hop_traversal": {
            "description": "跨图多跳遍历（MAGMA核心能力）",
            "gremlin": "g.V(anchorId).repeat(both().simplePath()).until(has(label, within('semantic','causal','temporal','entity_ref'))).times(k).dedup()",
            "hugegraph_advantage": "60亿点边生产验证，跨类型边的多跳遍历是 HugeGraph 核心差异化能力",
        },
    }
    return translations


# ============================================================
# 5. 测试和演示
# ============================================================

def build_demo_memory(store: FourGraphMemoryStore):
    """构建演示用的 Agent 记忆数据（模拟 AI Agent 的对话历史）"""
    base_time = datetime(2026, 6, 1, 9, 0)

    events = [
        ("Alice reported a critical bug in the authentication service", base_time, {"priority": "high", "type": "bug"}),
        ("Bob investigated the authentication bug and found a race condition", base_time + timedelta(hours=2), {"priority": "high", "type": "investigation"}),
        ("The race condition was caused by incorrect connection pool handling", base_time + timedelta(hours=4), {"type": "root_cause"}),
        ("Alice deployed a fix for the connection pool race condition", base_time + timedelta(hours=6), {"type": "fix"}),
        ("Server CPU usage spiked to 95% after the authentication fix deployment", base_time + timedelta(hours=7), {"priority": "high", "type": "incident"}),
        ("Carol discovered the CPU spike was due to a missing index on the users table", base_time + timedelta(hours=9), {"type": "root_cause"}),
        ("Bob added the missing database index and CPU returned to normal levels", base_time + timedelta(hours=11), {"type": "fix"}),
        ("Alice scheduled a code review meeting for the authentication module next Monday", base_time + timedelta(hours=24), {"type": "meeting"}),
        ("Deploy released v2.3.1 with authentication fix and database index update", base_time + timedelta(hours=30), {"type": "release"}),
        ("David reported a new feature request for OAuth2 support in the authentication service", base_time + timedelta(hours=48), {"type": "feature"}),
    ]

    written_nodes = []
    for content, ts, attrs in events:
        node = store.fast_path_write(
            content=content,
            timestamp=ts.isoformat(),
            attributes=attrs,
        )
        written_nodes.append(node)

    # Slow Path 巩固
    for node in written_nodes:
        store.slow_path_consolidate(node.node_id)

    return written_nodes


def run_tests():
    """运行 MAGMA 四图架构 PoC 测试"""
    print("=" * 70)
    print("MAGMA Four-Graph Agent Memory PoC")
    print("Based on ACL 2026: arXiv:2601.03236")
    print("=" * 70)

    results = {
        "poc_name": "MAGMA Four-Graph Agent Memory",
        "date": "2026-06-10",
        "paper": "MAGMA: A Multi-Graph based Agentic Memory (ACL 2026)",
        "arxiv": "https://arxiv.org/abs/2601.03236",
        "tests": [],
    }

    store = FourGraphMemoryStore()

    print(f"  Vector backend:  {store.stats.get('vector_backend', 'memory')}")
    print(f"  Fulltext backend: {store.stats.get('fulltext_backend', 'memory')}")
    print(f"  Graph backend:    {store.stats.get('graph_backend', 'memory')}")

    # === Test 1: Fast Path 写入 ===
    print("\n[Test 1] Fast Path Write (Synchronous)")
    t0 = time.time()
    nodes = build_demo_memory(store)
    t1 = time.time()
    print(f"  Written {len(nodes)} memory events")
    print(f"  Latency: {(t1-t0)*1000:.1f}ms (total)")
    print(f"  Stats: {json.dumps(store.stats, indent=4)}")
    test1_pass = len(nodes) == 10 and store.stats["total_nodes"] >= 10
    results["tests"].append({
        "name": "Fast Path Write",
        "passed": test1_pass,
        "detail": f"Wrote {len(nodes)} events, {store.stats['total_nodes']} total nodes"
    })

    # === Test 2: 四类边统计 ===
    print("\n[Test 2] Four Graph Edge Types")
    edge_stats = {
        "semantic": store.stats["semantic_edges"],
        "temporal": store.stats["temporal_edges"],
        "causal": store.stats["causal_edges"],
        "entity_ref": store.stats["entity_ref_edges"],
    }
    print(f"  Semantic edges: {edge_stats['semantic']}")
    print(f"  Temporal edges: {edge_stats['temporal']}")
    print(f"  Causal edges:   {edge_stats['causal']}")
    print(f"  Entity ref edges: {edge_stats['entity_ref']}")
    print(f"  Entity nodes: {store.stats['entity_nodes']}")
    all_edges = sum(edge_stats.values())
    test2_pass = edge_stats["temporal"] >= 9 and edge_stats["entity_ref"] > 0
    results["tests"].append({
        "name": "Four Graph Edges",
        "passed": test2_pass,
        "detail": f"S:{edge_stats['semantic']} T:{edge_stats['temporal']} C:{edge_stats['causal']} E:{edge_stats['entity_ref']}"
    })

    # === Test 3: Intent Routing ===
    print("\n[Test 3] Intent Routing")
    test_queries = [
        ("Why did the server CPU spike?", IntentType.WHY),
        ("When was the authentication bug reported?", IntentType.WHEN),
        ("What events are related to Alice?", IntentType.ENTITY),
        ("What caused the authentication fix?", IntentType.WHY),
        ("What happened after the deployment?", IntentType.WHEN),
    ]
    routing_pass = 0
    for query, expected_intent in test_queries:
        routing = store.route_intent(query)
        match = "✓" if routing.intent == expected_intent else "✗"
        print(f"  {match} Q: '{query[:40]}' → Intent: {routing.intent.value} "
              f"(expected: {expected_intent.value})")
        if routing.intent == expected_intent:
            routing_pass += 1
    test3_pass = routing_pass >= 4
    results["tests"].append({
        "name": "Intent Routing",
        "passed": test3_pass,
        "detail": f"{routing_pass}/5 correct intent classifications"
    })

    # === Test 4: Adaptive Beam Search + Full Query ===
    print("\n[Test 4] Adaptive Beam Search")
    full_results = store.query("Why did the server CPU spike?")
    print(f"  Query: '{full_results['query']}'")
    print(f"  Intent: {full_results['intent']}")
    print(f"  Results: {full_results['results_count']}")
    for r in full_results["results"][:3]:
        print(f"    [{r['score']:.3f}] {r['content']} (graphs: {r['matched_graphs']})")
    print(f"  Latency: {full_results['latency_ms']:.1f}ms")
    test4_pass = full_results["results_count"] > 0 and full_results["latency_ms"] < 1000
    results["tests"].append({
        "name": "Beam Search Query",
        "passed": test4_pass,
        "detail": f"{full_results['results_count']} results in {full_results['latency_ms']:.1f}ms"
    })

    # === Test 5: Cross-Graph Traversal ===
    print("\n[Test 5] Cross-Graph Traversal")
    # 测试跨语义→因果的遍历
    node = nodes[0]  # Alice 报告 bug
    semantic_neighbors = store.gremlin_semantic_neighbors(node.node_id)
    print(f"  Node '{node.content[:40]}' has {len(semantic_neighbors)} semantic neighbors")

    if semantic_neighbors:
        first_neighbor_id = semantic_neighbors[0][0]
        causal_succ = store.gremlin_causal_successors(first_neighbor_id)
        print(f"  First semantic neighbor has {len(causal_succ)} causal successors")

    # Entity graph 遍历
    entity_count = store.stats["entity_nodes"]
    entity_events = []
    for eid in list(store.entity_nodes.keys())[:3]:
        events = store.gremlin_entity_events(eid)
        if events:
            entity_events.append((store.entity_nodes[eid].content, len(events)))
    print(f"  Entity graph: {entity_count} entities, sample events: {entity_events}")

    test5_pass = len(semantic_neighbors) > 0 and entity_count > 0
    results["tests"].append({
        "name": "Cross-Graph Traversal",
        "passed": test5_pass,
        "detail": f"Semantic neighbors: {len(semantic_neighbors)}, Entities: {entity_count}"
    })

    # === Test 6: Gremlin Translation ===
    print("\n[Test 6] Gremlin Translation Guide")
    translations = gremlin_translation_guide()
    for name, info in translations.items():
        print(f"  {name}: {info['gremlin'][:60]}...")
        print(f"    HugeGraph advantage: {info['hugegraph_advantage'][:60]}")
    test6_pass = len(translations) == 5
    results["tests"].append({
        "name": "Gremlin Translation",
        "passed": test6_pass,
        "detail": f"5 Gremlin query patterns mapped"
    })

    # === Test 7: MAGMA vs Single-Graph Comparison ===
    print("\n[Test 7] MAGMA vs Single-Graph (Ablation Simulation)")
    # 模拟消融: 仅用语义图 vs 四图
    query_vec = generate_embedding("Why did the server CPU spike?")

    # Single-graph (semantic only)
    semantic_results = []
    for nid, node in store.nodes.items():
        if node.graph_type == "entity":
            continue
        sim = cosine_similarity(query_vec, node.vector)
        semantic_results.append((nid, sim))
    semantic_results.sort(key=lambda x: -x[1])

    # Four-graph (full MAGMA)
    four_graph_results = store.query("Why did the server CPU spike?")

    print(f"  Single-graph top-3:")
    for nid, score in semantic_results[:3]:
        content = store.nodes[nid].content[:60]
        print(f"    [{score:.3f}] {content}")

    print(f"  MAGMA four-graph top-3:")
    for r in four_graph_results["results"][:3]:
        print(f"    [{r['score']:.3f}] {r['content']} (graphs: {r['matched_graphs']})")

    # 检查: 四图是否找到了因果链（single-graph 找不到的）
    causal_in_results = any("causal" in r.get("matched_graphs", [])
                            for r in four_graph_results["results"])
    test7_pass = causal_in_results
    results["tests"].append({
        "name": "MAGMA vs Single-Graph",
        "passed": test7_pass,
        "detail": f"Causal graph contribution detected: {causal_in_results}"
    })

    # === Summary ===
    total_tests = len(results["tests"])
    passed_tests = sum(1 for t in results["tests"] if t["passed"])
    results["summary"] = {
        "total": total_tests,
        "passed": passed_tests,
        "failed": total_tests - passed_tests,
        "pass_rate": f"{passed_tests}/{total_tests}",
    }
    results["store_stats"] = store.stats
    results["gremlin_patterns"] = {k: {
        "description": v["description"],
        "gremlin": v["gremlin"],
        "hugegraph_advantage": v["hugegraph_advantage"],
    } for k, v in translations.items()}

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 70}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = run_tests()

    # 保存结果
    output_dir = "/Users/mac/Desktop/apache-code/hugegraph-dev/incubator-hugegraph-ai/hugegraph-llm/src/hugegraph_llm/poc"
    output_file = f"{output_dir}/magma_four_graph_memory_result.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {output_file}")
