"""
PowerMem Pure PoC — Exact PowerMem v1.2.0 Alignment

Architecture:
  - 4-Channel Search: Dense Vector + Full-Text (BM25) + Sparse Vector + Graph
  - Channels 1-3: Weighted Average Fusion (NOT RRF)
  - Channel 4 (Graph): Independent, returns in separate 'relations' field
  - Ebbinghaus Plugin: lifecycle hooks (on_add/on_search/on_update/on_delete)
  - MemoryGraph interface: NetworkX fallback, HugeGraph optional
  - 5 relationship types: Temporal, Causal, Topical, Referential, Contradictory
  - Add flow (infer=True): extract_facts → decide_actions → execute

Dependencies:
  - faiss-cpu
  - rank_bm25
  - numpy
  - networkx
  - sentence-transformers (optional)
  - requests (optional, for LLM API)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

try:
    import networkx as nx  # type: ignore

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from hugegraph_llm.indices.backend_factory import get_vector_index, get_fulltext_index  # type: ignore

    HAS_GRAPHRAG_OPS = True
except ImportError:
    HAS_GRAPHRAG_OPS = False

try:
    from hugegraph.client import PyHugeClient  # type: ignore

    HAS_HUGEGRAPH = True
except ImportError:
    HAS_HUGEGRAPH = False

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 128
_tokenizer_re = re.compile(r"\w+")


def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Deterministic hash-based embedding (no ML model required)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenizer_re.findall(text.lower())
    for i, token in enumerate(tokens):
        h = hashlib.sha256(token.encode()).digest()
        for j in range(dim):
            byte_idx = j % 32
            byte_val = h[byte_idx]
            val = (byte_val - 128) / 128.0
            vec[j] += val / (i + 1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def embed_text(text: str) -> List[float]:
    """Embed text: sentence-transformers if available, else hash-based."""
    if HAS_SENTENCE_TRANSFORMERS:
        emb = _ST_MODEL.encode(text, normalize_embeddings=True)
        return emb.tolist()
    return _hash_embed(text)


# ---------------------------------------------------------------------------
# LLM helper (optional)
# ---------------------------------------------------------------------------

XIAOMI_API_KEY = os.environ.get("XIAOMI_API_KEY", "")
LLM_BASE_URL = "https://api.xiaomimimo.com/v1"


def _llm_call(prompt: str, system: str = "") -> str:
    """Call MiMo v2.5 Pro if API key available, else return empty."""
    if not XIAOMI_API_KEY:
        return ""
    try:
        import requests  # type: ignore

        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {XIAOMI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mimo-v2.5-pro",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM] API call failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES = ["Temporal", "Causal", "Topical", "Referential", "Contradictory"]


# ---------------------------------------------------------------------------
# MemoryNode
# ---------------------------------------------------------------------------


@dataclass
class MemoryNode:
    """Memory with PowerMem v1.2.0 metadata fields."""

    id: str
    text: str
    embedding: List[float]
    sparse_embedding: Dict[int, float] = field(default_factory=dict)
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0
    user_id: str = "default"
    metadata: dict = field(default_factory=dict)
    # Ebbinghaus fields
    retention_score: float = 1.0
    lifecycle_stage: str = "LONG_TERM"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "retention_score": self.retention_score,
            "lifecycle_stage": self.lifecycle_stage,
        }


# ---------------------------------------------------------------------------
# Ebbinghaus Plugin
# ---------------------------------------------------------------------------


class EbbinghausPlugin:
    """Exact PowerMem v1.2.0 Ebbinghaus implementation.

    R(t) = exp(-λ * t_hours), where λ = 0.821 (= -ln(0.44))
    base_retention_1h = 0.44
    min_retention = 0.2
    Reinforcement: +0.3 per access, capped at 1.0
    """

    DECAY_CONSTANT = 0.821  # -ln(0.44)
    MIN_RETENTION = 0.2
    REINFORCEMENT = 0.3
    STAGES = {"LONG_TERM": 0.8, "SHORT_TERM": 0.6, "WORKING": 0.3}

    def on_add(self, memory_data: dict) -> dict:
        """Inject retention_score=1.0, access_count=0, lifecycle_stage."""
        memory_data["retention_score"] = 1.0
        memory_data["access_count"] = 0
        memory_data["lifecycle_stage"] = "LONG_TERM"
        return memory_data

    def on_search(self, query: str, memories: List[dict]) -> List[dict]:
        """Apply decay + reinforcement, update access_count (not persisted)."""
        now = time.time()
        for mem in memories:
            last_accessed = mem.get("last_accessed_at", now)
            access_count = mem.get("access_count", 0)
            hours = (now - last_accessed) / 3600.0
            base_retention = math.exp(-self.DECAY_CONSTANT * hours)
            retention = max(base_retention, self.MIN_RETENTION)
            reinforcement = min(access_count * self.REINFORCEMENT, 1.0)
            retention = min(retention + reinforcement, 1.0)
            mem["retention_score"] = round(retention, 6)
            mem["lifecycle_stage"] = self._classify_stage(retention)
            # Update access count (transient, not persisted in on_search)
            mem["access_count"] = access_count + 1
        return memories

    def on_update(self, memory_id: str, update_data: dict) -> dict:
        """Recalculate retention on update (persisted)."""
        now = time.time()
        last_accessed = update_data.get("last_accessed_at", now)
        access_count = update_data.get("access_count", 0)
        hours = (now - last_accessed) / 3600.0
        base_retention = math.exp(-self.DECAY_CONSTANT * hours)
        retention = max(base_retention, self.MIN_RETENTION)
        reinforcement = min(access_count * self.REINFORCEMENT, 1.0)
        retention = min(retention + reinforcement, 1.0)
        update_data["retention_score"] = round(retention, 6)
        update_data["lifecycle_stage"] = self._classify_stage(retention)
        return update_data

    def on_delete(self, memory_id: str):
        """No-op for delete."""
        pass

    def _calculate_retention(self, memory: Any, now: float) -> float:
        """R(t) = exp(-λ * t_hours), floored at MIN_RETENTION."""
        hours = (now - memory.last_accessed_at) / 3600.0
        base_retention = math.exp(-self.DECAY_CONSTANT * hours)
        return max(base_retention, self.MIN_RETENTION)

    def _classify_stage(self, retention: float) -> str:
        """Classify retention into lifecycle stage."""
        if retention >= self.STAGES["LONG_TERM"]:
            return "LONG_TERM"
        if retention >= self.STAGES["SHORT_TERM"]:
            return "SHORT_TERM"
        if retention >= self.STAGES["WORKING"]:
            return "WORKING"
        return "EXPIRED"


# ---------------------------------------------------------------------------
# MemoryGraph Interface
# ---------------------------------------------------------------------------


class MemoryGraph(ABC):
    """Graph store interface matching PowerMem v1.2.0."""

    @abstractmethod
    def add_node(self, memory_id: str, content: str, metadata: dict) -> str:
        """Add a node to the graph. Returns node_id."""

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0):
        """Add an edge between nodes."""

    @abstractmethod
    def traverse(self, start_id: str, max_hops: int = 2, filters: Optional[dict] = None) -> List[dict]:
        """Multi-hop traversal from start_id."""

    @abstractmethod
    def get_neighbors(self, memory_id: str, relationship_types: Optional[List[str]] = None) -> List[dict]:
        """Get direct neighbors of a node."""

    @abstractmethod
    def search_graph(self, query: str, filters: Optional[dict] = None) -> List[dict]:
        """Search graph for nodes matching query."""

    @abstractmethod
    def delete_node(self, memory_id: str):
        """Delete a node and all its edges."""


# ---------------------------------------------------------------------------
# NetworkX Graph Store
# ---------------------------------------------------------------------------


class NetworkXGraph(MemoryGraph):
    """NetworkX fallback graph store."""

    def __init__(self):
        if not HAS_NETWORKX:
            raise RuntimeError("networkx is required for NetworkXGraph")
        self._graph = nx.MultiDiGraph()
        self._node_content: Dict[str, str] = {}

    def add_node(self, memory_id: str, content: str, metadata: dict) -> str:
        self._graph.add_node(memory_id, **metadata)
        self._node_content[memory_id] = content
        return memory_id

    def add_edge(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0):
        if source_id not in self._graph:
            self._graph.add_node(source_id)
        if target_id not in self._graph:
            self._graph.add_node(target_id)
        self._graph.add_edge(source_id, target_id, key=relationship, relationship=relationship, weight=weight)

    def traverse(self, start_id: str, max_hops: int = 2, filters: Optional[dict] = None) -> List[dict]:
        if start_id not in self._graph:
            return []
        results = []
        visited = set()
        visited.add(start_id)
        frontier = [start_id]
        for _ in range(max_hops):
            next_frontier = []
            for nid in frontier:
                # Collect unique neighbors from all edges (out + in)
                neighbor_data = {}  # neighbor_id -> best edge data
                for src, dst, key, edge_data in self._graph.out_edges(nid, data=True, keys=True):
                    neighbor_id = dst
                    if neighbor_id in visited:
                        continue
                    if neighbor_id not in neighbor_data:
                        neighbor_data[neighbor_id] = edge_data
                for src, dst, key, edge_data in self._graph.in_edges(nid, data=True, keys=True):
                    neighbor_id = src
                    if neighbor_id in visited:
                        continue
                    if neighbor_id not in neighbor_data:
                        neighbor_data[neighbor_id] = edge_data
                for neighbor_id, edge_data in neighbor_data.items():
                    visited.add(neighbor_id)
                    next_frontier.append(neighbor_id)
                    result = {
                        "node_id": neighbor_id,
                        "content": self._node_content.get(neighbor_id, ""),
                        "relationship": edge_data.get("relationship", "unknown"),
                        "weight": edge_data.get("weight", 1.0),
                    }
                    if filters:
                        if "relationship_types" in filters:
                            if result["relationship"] not in filters["relationship_types"]:
                                continue
                    results.append(result)
            frontier = next_frontier
        return results

    def get_neighbors(self, memory_id: str, relationship_types: Optional[List[str]] = None) -> List[dict]:
        if memory_id not in self._graph:
            return []
        results = []
        seen = set()
        for src, dst, key, edge_data in self._graph.out_edges(memory_id, data=True, keys=True):
            rel = edge_data.get("relationship", "unknown")
            if relationship_types and rel not in relationship_types:
                continue
            if dst in seen:
                continue
            seen.add(dst)
            results.append({
                "node_id": dst,
                "content": self._node_content.get(dst, ""),
                "relationship": rel,
                "weight": edge_data.get("weight", 1.0),
            })
        for src, dst, key, edge_data in self._graph.in_edges(memory_id, data=True, keys=True):
            rel = edge_data.get("relationship", "unknown")
            if relationship_types and rel not in relationship_types:
                continue
            if src in seen:
                continue
            seen.add(src)
            results.append({
                "node_id": src,
                "content": self._node_content.get(src, ""),
                "relationship": rel,
                "weight": edge_data.get("weight", 1.0),
            })
        return results

    def search_graph(self, query: str, filters: Optional[dict] = None) -> List[dict]:
        """Search graph nodes by content matching (BM25-style keyword matching)."""
        query_tokens = set(_tokenizer_re.findall(query.lower()))
        results = []
        for nid, content in self._node_content.items():
            content_tokens = set(_tokenizer_re.findall(content.lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            node_data = self._graph.nodes.get(nid, {})
            result = {
                "node_id": nid,
                "content": content,
                "score": round(score, 4),
                "metadata": dict(node_data),
            }
            if filters:
                if "user_id" in filters and node_data.get("user_id") != filters["user_id"]:
                    continue
            results.append(result)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def delete_node(self, memory_id: str):
        if memory_id in self._graph:
            self._graph.remove_node(memory_id)
        self._node_content.pop(memory_id, None)


# ---------------------------------------------------------------------------
# HugeGraph Store
# ---------------------------------------------------------------------------


class HugeGraphStore(MemoryGraph):
    """HugeGraph implementation via PyHugeClient. Graceful fallback."""

    def __init__(self, host: str = "localhost", port: int = 8080,
                 graph_name: str = "powermem", user: str = "admin", pwd: str = "pwd"):
        if not HAS_HUGEGRAPH:
            raise RuntimeError("PyHugeClient is required for HugeGraphStore")
        self._client = PyHugeClient(host, port, graph_name, user, pwd)
        try:
            self._client.open()
            self._connected = True
        except Exception:
            self._connected = False
        self._node_content: Dict[str, str] = {}

    def _ensure_schema(self):
        """Create vertex/edge labels if not exist."""
        if not self._connected:
            return
        try:
            self._client.schema().vertexLabel("memory").ifNotExists().properties(
                {"name": "content", "type": "TEXT"},
                {"name": "user_id", "type": "TEXT"},
            ).create()
            for rel_type in RELATIONSHIP_TYPES:
                self._client.schema().edgeLabel(rel_type.lower()).sourceLabel("memory").targetLabel(
                    "memory").properties({"name": "weight", "type": "DOUBLE"}).create()
        except Exception:
            pass

    def add_node(self, memory_id: str, content: str, metadata: dict) -> str:
        if not self._connected:
            self._node_content[memory_id] = content
            return memory_id
        self._ensure_schema()
        try:
            self._client.graph().addVertex("memory", content=content, user_id=metadata.get("user_id", ""),
                                             pk=memory_id)
        except Exception:
            pass
        self._node_content[memory_id] = content
        return memory_id

    def add_edge(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0):
        if not self._connected:
            return
        rel_lower = relationship.lower()
        try:
            self._client.graph().addEdge(
                f"{source_id}:{source_id}", f"{target_id}:{target_id}", rel_lower, weight=weight
            )
        except Exception:
            pass

    def traverse(self, start_id: str, max_hops: int = 2, filters: Optional[dict] = None) -> List[dict]:
        if not self._connected:
            return []
        try:
            results = []
            step = 1
            while step <= max_hops:
                resp = self._client.graph().traverse().V(start_id).out(
                    RELATIONSHIP_TYPES if not filters else filters.get("relationship_types", RELATIONSHIP_TYPES)
                ).depth(step).withResult().toList()
                for item in resp:
                    nid = item.get("id", "")
                    results.append({
                        "node_id": nid,
                        "content": self._node_content.get(nid, ""),
                        "relationship": item.get("label", "unknown"),
                        "weight": item.get("weight", 1.0),
                    })
                step += 1
            return results
        except Exception:
            return []

    def get_neighbors(self, memory_id: str, relationship_types: Optional[List[str]] = None) -> List[dict]:
        if not self._connected:
            return []
        try:
            results = []
            rels = relationship_types if relationship_types else RELATIONSHIP_TYPES
            for rel in rels:
                resp = self._client.graph().traverse().V(memory_id).out(rel.lower()).depth(1).withResult().toList()
                for item in resp:
                    nid = item.get("id", "")
                    results.append({
                        "node_id": nid,
                        "content": self._node_content.get(nid, ""),
                        "relationship": rel,
                        "weight": item.get("weight", 1.0),
                    })
            return results
        except Exception:
            return []

    def search_graph(self, query: str, filters: Optional[dict] = None) -> List[dict]:
        if not self._connected:
            return []
        try:
            results = []
            resp = self._client.graph().queryVertices("memory", page="", limit=100)
            for item in resp:
                content = item.get("content", "")
                nid = item.get("id", "")
                query_tokens = set(_tokenizer_re.findall(query.lower()))
                content_tokens = set(_tokenizer_re.findall(content.lower()))
                overlap = len(query_tokens & content_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(len(query_tokens), 1)
                results.append({
                    "node_id": nid,
                    "content": content,
                    "score": round(score, 4),
                    "metadata": item,
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        except Exception:
            return []

    def delete_node(self, memory_id: str):
        if not self._connected:
            self._node_content.pop(memory_id, None)
            return
        try:
            self._client.graph().deleteVertex(memory_id)
        except Exception:
            pass
        self._node_content.pop(memory_id, None)


# ---------------------------------------------------------------------------
# Sparse Embedding (BM25-weight sparse vector)
# ---------------------------------------------------------------------------


def _sparse_embedding(text: str, idf: Optional[Dict[str, float]] = None) -> Dict[int, float]:
    """Generate sparse vector from BM25 weights (SPLADE-like approximation).

    Token hashed to int index, value is term weight.
    If idf is provided, uses idf weights; otherwise uses 1.0 (TF-only).
    """
    tokens = _tokenizer_re.findall(text.lower())
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    sparse = {}
    for token, count in tf.items():
        h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        tf_val = count / total
        idf_val = idf.get(token, 1.0) if idf else 1.0
        sparse[h] = tf_val * idf_val
    return sparse


def _sparse_cosine_similarity(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# PowerMemStore
# ---------------------------------------------------------------------------


class PowerMemStore:
    """PowerMem v1.2.0 aligned memory store.

    4-Channel Search:
      1. Dense Vector (FAISS) — semantic embedding similarity
      2. Full-Text Search (BM25Okapi) — BM25 keyword search
      3. Sparse Vector — BM25-weight sparse embedding (SPLADE-like)
      4. Graph Search (NetworkX or HugeGraph) — multi-hop traversal

    Fusion: Weighted average for channels 1-3 (NOT RRF).
    Graph: Independent, returns in separate 'relations' field.
    Ebbinghaus: Plugin with lifecycle hooks.
    """

    def __init__(self, user_id: str = "default", embedding_dim: int = EMBEDDING_DIM,
                 graph_store: Optional[MemoryGraph] = None,
                 weights: Optional[Dict[str, float]] = None):
        self.embedding_dim = embedding_dim
        self._memories: Dict[str, MemoryNode] = {}
        self._user_index: Dict[str, List[str]] = {}

        # Channel weights (configurable)
        self._weights = weights or {"dense": 0.5, "fts": 0.3, "sparse": 0.2}

        # Channel 1: Dense vector (FAISS)
        self._faiss_index: Optional[object] = None
        self._id_list: List[str] = []
        self._init_vector_store()

        # Channel 2: Full-text (BM25Okapi)
        self._bm25: Optional[object] = None
        self._bm25_docs: List[str] = []
        self._bm25_ids: List[str] = []
        self._bm25_tokenized: List[List[str]] = []

        # Channel 3: Sparse vector store
        self._sparse_vectors: Dict[str, Dict[int, float]] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0

        # Channel 4: Graph store
        if graph_store is not None:
            self._graph = graph_store
        elif HAS_NETWORKX:
            self._graph = NetworkXGraph()
        else:
            self._graph = None

        # Plugin: Ebbinghaus
        self._ebbinghaus = EbbinghausPlugin()

    # ---- Initialization helpers ----

    def _init_vector_store(self):
        if HAS_FAISS:
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self._faiss_index = None

    def _rebuild_bm25(self):
        if HAS_RANK_BM25 and self._bm25_tokenized:
            self._bm25 = BM25Okapi(self._bm25_tokenized)

    def _rebuild_faiss(self):
        if not HAS_FAISS or not self._memories:
            return
        dim = self.embedding_dim
        ids = list(self._memories.keys())
        embs = np.array([self._memories[mid].embedding for mid in ids], dtype=np.float32)
        index = faiss.IndexFlatIP(dim)
        if embs.shape[0] > 0:
            index.add(embs)
        self._faiss_index = index
        self._id_list = ids

    def _rebuild_idf(self):
        """Rebuild IDF from all documents."""
        df: Dict[str, int] = defaultdict(int)
        self._doc_count = len(self._memories)
        for node in self._memories.values():
            tokens = set(_tokenizer_re.findall(node.text.lower()))
            for t in tokens:
                df[t] += 1
        self._idf = {}
        for t, freq in df.items():
            self._idf[t] = math.log((self._doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def _rebuild_indices(self):
        """Rebuild all indices from memory dict."""
        self._rebuild_faiss()
        self._rebuild_bm25_index()
        self._rebuild_idf()
        self._rebuild_sparse_vectors()

    def _rebuild_bm25_index(self):
        self._bm25_docs = []
        self._bm25_ids = []
        self._bm25_tokenized = []
        for mid, node in self._memories.items():
            self._bm25_ids.append(mid)
            self._bm25_docs.append(node.text)
            self._bm25_tokenized.append(_tokenizer_re.findall(node.text.lower()))
        self._rebuild_bm25()

    def _rebuild_sparse_vectors(self):
        """Rebuild sparse vectors using current IDF."""
        self._sparse_vectors = {}
        for mid, node in self._memories.items():
            self._sparse_vectors[mid] = _sparse_embedding(node.text, self._idf)

    # ---- Public API ----

    def add(self, text: str, user_id: str = "default", infer: bool = True) -> dict:
        """Add with infer=True flow:
        1. _extract_facts (LLM or rule-based)
        2. For each fact: search similar memories
        3. _decide_actions (LLM or rule-based): ADD/UPDATE/DELETE/NONE
        4. Execute actions + graph construction
        5. EbbinghausPlugin.on_add()
        """
        facts = self._extract_facts(text) if infer else [text.strip()]

        results = []
        for fact in facts:
            fact = fact.strip()
            if not fact:
                continue
            if infer:
                existing = self._search_existing(fact, user_id, top_k=3)
                action, target_id = self._decide_memory_actions(fact, existing)
            else:
                action, target_id = "ADD", None

            if action == "ADD":
                node = self._create_memory(fact, user_id)
                results.append({"operation": "add", "id": node.id, "text": fact})
            elif action == "UPDATE" and target_id:
                self._update_memory(target_id, fact)
                results.append({"operation": "update", "id": target_id, "text": fact})
            elif action == "DELETE" and target_id:
                self.delete(target_id)
                results.append({"operation": "delete", "id": target_id, "text": fact})
            else:
                results.append({"operation": "none", "text": fact})

        # Build graph edges between related facts (infer only)
        if infer and len(results) > 1:
            self._build_graph_relations(results, user_id)

        return {
            "added": len([r for r in results if r["operation"] == "add"]),
            "updated": len([r for r in results if r["operation"] == "update"]),
            "deleted": len([r for r in results if r["operation"] == "delete"]),
            "unchanged": len([r for r in results if r["operation"] == "none"]),
            "results": results,
        }

    def search(self, query: str, user_id: str = "default", limit: int = 30,
               threshold: Optional[float] = None) -> dict:
        """4-channel search:
        1. Dense vector + FTS + Sparse: weighted average fusion → quality_score
        2. Threshold filtering
        3. Graph search: independent, returns in 'relations' key
        4. EbbinghausPlugin.on_search()
        Returns: {"results": [...], "relations": [...]}
        """
        user_ids = self._user_index.get(user_id, [])
        if not user_ids:
            return {"results": [], "relations": []}

        # Parallel channel 1-3 search
        dense_results = self._dense_search(query, top_k=len(user_ids), user_ids=user_ids)
        fts_results = self._fts_search(query, top_k=len(user_ids), user_ids=user_ids)
        sparse_results = self._sparse_search(query, top_k=len(user_ids), user_ids=user_ids)

        # Weighted average fusion (channels 1-3 only)
        w1 = self._weights["dense"]
        w2 = self._weights["fts"]
        w3 = self._weights["sparse"]
        fused = self._weighted_average_fusion(dense_results, fts_results, sparse_results,
                                             w1=w1, w2=w2, w3=w3)

        # Threshold filtering
        if threshold is not None:
            fused = [(mid, score) for mid, score in fused if score >= threshold]

        # Sort by quality_score descending, limit
        fused.sort(key=lambda x: x[1], reverse=True)
        fused = fused[:limit]

        # Build results
        now = time.time()
        results = []
        for mid, quality_score in fused:
            node = self._memories.get(mid)
            if node:
                # Apply Ebbinghaus: calculate retention and update access (transient)
                retention = self._ebbinghaus._calculate_retention(node, now)
                reinforcement = min(node.access_count * self._ebbinghaus.REINFORCEMENT, 1.0)
                retention = min(retention + reinforcement, 1.0)
                stage = self._ebbinghaus._classify_stage(retention)
                results.append({
                    **node.to_dict(),
                    "quality_score": round(quality_score, 4),
                    "retention_score": round(retention, 6),
                    "lifecycle_stage": stage,
                })

        # Ebbinghaus on_search: update access_count on results (transient)
        if results:
            results = self._ebbinghaus.on_search(query, results)
            # Persist the access_count update back to MemoryNode
            for r in results:
                node = self._memories.get(r["id"])
                if node:
                    node.access_count = r["access_count"]
                    node.last_accessed_at = now

        # Channel 4: Independent graph search
        relations = []
        if self._graph is not None:
            relations = self._graph.search_graph(query, filters={"user_id": user_id})

        return {"results": results, "relations": relations}

    def update(self, memory_id: str, new_text: str) -> dict:
        """Update memory text, re-embed, trigger Ebbinghaus on_update."""
        if memory_id not in self._memories:
            return {"success": False, "error": f"Memory {memory_id} not found"}
        self._update_memory(memory_id, new_text)
        return {"success": True, "updated": memory_id, "text": new_text}

    def delete(self, memory_id: str) -> dict:
        """Delete a memory by ID from all stores."""
        if memory_id not in self._memories:
            return {"success": False, "error": f"Memory {memory_id} not found"}
        node = self._memories.pop(memory_id)
        uid = node.user_id
        if uid in self._user_index and memory_id in self._user_index[uid]:
            self._user_index[uid].remove(memory_id)
        self._sparse_vectors.pop(memory_id, None)
        self._ebbinghaus.on_delete(memory_id)
        # Remove from graph
        if self._graph is not None:
            self._graph.delete_node(memory_id)
        self._rebuild_indices()
        return {"success": True, "deleted": memory_id}

    def get_all(self, user_id: str = "default") -> List[dict]:
        """Get all memories for a user."""
        ids = self._user_index.get(user_id, [])
        return [self._memories[mid].to_dict() for mid in ids if mid in self._memories]

    def reset(self, user_id: str = "default") -> dict:
        """Delete all memories for a user."""
        ids = self._user_index.get(user_id, [])
        count = 0
        for mid in list(ids):
            if mid in self._memories:
                self._ebbinghaus.on_delete(mid)
                if self._graph is not None:
                    self._graph.delete_node(mid)
                del self._memories[mid]
                count += 1
        self._user_index[user_id] = []
        self._rebuild_indices()
        return {"success": True, "deleted_count": count}

    # ---- Internal: Add flow ----

    def _create_memory(self, text: str, user_id: str) -> MemoryNode:
        """Create MemoryNode, embed, store, apply Ebbinghaus on_add."""
        now = time.time()
        emb = embed_text(text)
        sparse_emb = _sparse_embedding(text, self._idf)
        mid = str(uuid.uuid4())

        memory_data = {
            "id": mid,
            "text": text,
            "embedding": emb,
            "sparse_embedding": sparse_emb,
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "user_id": user_id,
            "metadata": {"source": "powermem_pure"},
        }

        # Ebbinghaus on_add: inject retention metadata
        memory_data = self._ebbinghaus.on_add(memory_data)

        node = MemoryNode(
            id=memory_data["id"],
            text=memory_data["text"],
            embedding=memory_data["embedding"],
            sparse_embedding=memory_data["sparse_embedding"],
            created_at=memory_data["created_at"],
            last_accessed_at=memory_data["last_accessed_at"],
            access_count=memory_data["access_count"],
            user_id=memory_data["user_id"],
            metadata=memory_data["metadata"],
            retention_score=memory_data["retention_score"],
            lifecycle_stage=memory_data["lifecycle_stage"],
        )

        self._memories[mid] = node
        if user_id not in self._user_index:
            self._user_index[user_id] = []
        self._user_index[user_id].append(mid)

        # Add to graph
        if self._graph is not None:
            self._graph.add_node(mid, text, {"user_id": user_id})

        self._sparse_vectors[mid] = sparse_emb
        self._rebuild_indices()
        return node

    def _update_memory(self, memory_id: str, new_text: str):
        """Update memory text, re-embed, trigger Ebbinghaus on_update."""
        if memory_id not in self._memories:
            return
        node = self._memories[memory_id]
        node.text = new_text
        node.embedding = embed_text(new_text)
        node.last_accessed_at = time.time()
        node.sparse_embedding = _sparse_embedding(new_text, self._idf)

        # Ebbinghaus on_update: recalculate retention
        update_data = self._ebbinghaus.on_update(
            memory_id,
            {"last_accessed_at": node.last_accessed_at, "access_count": node.access_count},
        )
        node.retention_score = update_data["retention_score"]
        node.lifecycle_stage = update_data["lifecycle_stage"]

        self._sparse_vectors[memory_id] = node.sparse_embedding
        self._rebuild_indices()

    def _search_existing(self, fact: str, user_id: str, top_k: int = 3) -> List[MemoryNode]:
        """Quick search for similar existing memories (for consolidation)."""
        results = self.search(fact, user_id=user_id, limit=top_k)
        nodes = []
        for r in results.get("results", []):
            node = self._memories.get(r["id"])
            if node:
                nodes.append(node)
        return nodes

    def _extract_facts(self, text: str) -> List[str]:
        """LLM-driven fact extraction (or rule-based fallback)."""
        if XIAOMI_API_KEY:
            result = _llm_call(
                prompt=f"Extract atomic facts from this text. Return one fact per line, nothing else:\n\n{text}",
                system="You are a fact extraction assistant. Extract concise, atomic facts.",
            )
            if result:
                facts = [line.strip() for line in result.strip().split("\n") if line.strip()]
                if facts:
                    return facts

        # Rule-based fallback: split by sentences
        sentences = re.split(r'[.!?。！？\n]', text)
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) >= 10:
                facts.append(s)
        return facts if facts else [text.strip()]

    def _decide_memory_actions(self, fact: str, existing: List[MemoryNode]) -> Tuple[str, Optional[str]]:
        """LLM decides ADD/UPDATE/DELETE/NONE. Returns (action, target_id)."""
        if XIAOMI_API_KEY and existing:
            existing_text = "\n".join(f"[{n.id}] {n.text}" for n in existing)
            result = _llm_call(
                prompt=(
                    f"New fact: {fact}\n\n"
                    f"Existing memories:\n{existing_text}\n\n"
                    f"Decide: If the new fact is novel, say ADD.\n"
                    f"If it updates an existing memory, say UPDATE and the ID.\n"
                    f"If it contradicts and should replace, say DELETE and the ID.\n"
                    f"If redundant/already exists, say NONE.\n"
                    f"Respond with exactly one line: OPERATION [ID]\n"
                    f"Example: ADD\nExample: UPDATE abc-123\nExample: DELETE def-456\nExample: NONE"
                ),
                system="You are a memory consolidation assistant. Be precise.",
            )
            if result:
                parts = result.strip().upper().split()
                op = parts[0]
                if op in ("ADD", "UPDATE", "DELETE", "NONE"):
                    target_id = parts[1] if len(parts) > 1 else None
                    return op, target_id

        # Rule-based: check overlap
        for node in existing:
            overlap = len(set(fact.split()) & set(node.text.split()))
            if overlap >= max(len(fact.split()), len(node.text.split())) * 0.7:
                return "UPDATE", node.id
        return "ADD", None

    def _build_graph_relations(self, action_results: List[dict], user_id: str):
        """Build graph edges between related facts from the same add batch."""
        if not self._graph:
            return
        added_ids = [r["id"] for r in action_results if r["operation"] in ("add", "update") and "id" in r]
        if len(added_ids) < 2:
            return
        # Connect sequential facts with Topical relationship
        for i in range(len(added_ids) - 1):
            self._graph.add_edge(added_ids[i], added_ids[i + 1], "Topical", weight=1.0)

    # ---- Channel 1: Dense Vector Search ----

    def _dense_search(self, query: str, top_k: int, user_ids: List[str]) -> List[Tuple[str, float]]:
        """Channel 1: Semantic embedding similarity search via FAISS."""
        if not self._faiss_index or not self._id_list:
            return []
        valid_ids = set(user_ids)
        query_emb = np.array(embed_text(query), dtype=np.float32)
        query_emb_np = query_emb.reshape(1, -1).astype(np.float32)
        actual_k = min(top_k, len(self._id_list))
        if actual_k == 0:
            return []
        scores, indices = self._faiss_index.search(query_emb_np, actual_k)
        result = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            mid = self._id_list[idx]
            if mid in valid_ids:
                result.append((mid, float(score)))
        return result

    # ---- Channel 2: Full-Text Search (BM25) ----

    def _fts_search(self, query: str, top_k: int, user_ids: List[str]) -> List[Tuple[str, float]]:
        """Channel 2: BM25 fulltext search."""
        if not self._bm25 or not self._bm25_ids:
            return []
        valid_ids = set(user_ids)
        tokens = _tokenizer_re.findall(query.lower())
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Normalize BM25 scores to [0, 1]
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = np.ones_like(scores)
        id_score_pairs = [
            (self._bm25_ids[i], float(scores_norm[i]))
            for i in range(len(self._bm25_ids))
            if self._bm25_ids[i] in valid_ids
        ]
        id_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return id_score_pairs[:top_k]

    # ---- Channel 3: Sparse Vector Search ----

    def _sparse_search(self, query: str, top_k: int, user_ids: List[str]) -> List[Tuple[str, float]]:
        """Channel 3: BM25-weight sparse embedding search."""
        valid_ids = set(user_ids)
        query_sparse = _sparse_embedding(query, self._idf)
        if not query_sparse or not self._sparse_vectors:
            return []
        scored = []
        for mid, sparse_vec in self._sparse_vectors.items():
            if mid not in valid_ids:
                continue
            sim = _sparse_cosine_similarity(query_sparse, sparse_vec)
            scored.append((mid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---- Weighted Average Fusion ----

    def _weighted_average_fusion(self, dense_results: List[Tuple[str, float]],
                                  fts_results: List[Tuple[str, float]],
                                  sparse_results: List[Tuple[str, float]],
                                  w1: float = 0.5, w2: float = 0.3, w3: float = 0.2) -> List[Tuple[str, float]]:
        """Weighted average fusion (NOT RRF!).

        Channels 1-3 scores are normalized to [0,1], then:
        final_score = w1 * dense + w2 * fts + w3 * sparse
        """
        # Collect all candidate IDs
        all_ids = set()
        for mid, _ in dense_results:
            all_ids.add(mid)
        for mid, _ in fts_results:
            all_ids.add(mid)
        for mid, _ in sparse_results:
            all_ids.add(mid)

        # Build score maps
        dense_map = dict(dense_results)
        fts_map = dict(fts_results)
        sparse_map = dict(sparse_results)

        # Normalize dense scores to [0,1]
        dense_scores = [s for _, s in dense_results]
        if dense_scores:
            d_min, d_max = min(dense_scores), max(dense_scores)
        else:
            d_min, d_max = 0.0, 1.0
        d_range = d_max - d_min if d_max > d_min else 1.0

        # FTS already normalized in _fts_search
        # Sparse cosine already in [0,1]

        fused = []
        for mid in all_ids:
            d_score = (dense_map.get(mid, 0.0) - d_min) / d_range if mid in dense_map else 0.0
            f_score = fts_map.get(mid, 0.0)
            s_score = sparse_map.get(mid, 0.0)
            final = w1 * d_score + w2 * f_score + w3 * s_score
            fused.append((mid, round(final, 6)))
        return fused

    # ---- Low-level access for benchmarking ----

    def _get_node(self, memory_id: str) -> Optional[MemoryNode]:
        return self._memories.get(memory_id)

    def _set_time(self, node_id: str, created_at: float, last_accessed_at: float, access_count: int = 0):
        """Override timestamps for benchmarking."""
        node = self._memories.get(node_id)
        if node:
            node.created_at = created_at
            node.last_accessed_at = last_accessed_at
            node.access_count = access_count


# ---------------------------------------------------------------------------
# Convenience: standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = PowerMemStore()
    print("=== PowerMemStore v1.2.0 Quick Test ===")
    print(f"FAISS available: {HAS_FAISS}")
    print(f"BM25 available: {HAS_RANK_BM25}")
    print(f"SentenceTransformers: {HAS_SENTENCE_TRANSFORMERS}")
    print(f"NetworkX available: {HAS_NETWORKX}")
    print(f"Graph store: {type(store._graph).__name__}")
    print()

    # Add with infer=True
    r = store.add("Alice works in the Engineering department at Google.", user_id="test_user")
    print(f"Add: {r}")

    r = store.add("Bob prefers dark mode and uses Vim as his editor.", user_id="test_user")
    print(f"Add: {r}")

    # Search
    results = store.search("Who works in Engineering?", user_id="test_user")
    print(f"\nSearch 'Who works in Engineering?':")
    for r in results["results"]:
        print(f"  [{r['quality_score']}] {r['text']}")
    if results["relations"]:
        print(f"  Relations: {results['relations']}")

    # Get all
    all_mem = store.get_all(user_id="test_user")
    print(f"\nAll memories ({len(all_mem)}):")
    for m in all_mem:
        print(f"  [{m['id'][:8]}] {m['text']} (stage={m['lifecycle_stage']})")

    # Reset
    store.reset(user_id="test_user")
    print(f"\nAfter reset: {len(store.get_all(user_id='test_user'))} memories")
