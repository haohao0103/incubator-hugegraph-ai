"""
MAGMA Four-Graph Memory Visualization
=====================================
Interactive visualization of MAGMA (ACL 2026) four-graph agent memory.
Shows Semantic/Temporal/Causal/Entity graphs with intent routing and beam search.

Usage: python3.10 magma_visualizer.py --port 5003
"""

import json
import hashlib
import math
import random
import threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import requests as _requests
import gzip as _gzip


# ============================================================
# HugeGraph REST Client (1.7.0 graphspace API)
# ============================================================

class HugeGraphClient:
    """Lightweight HugeGraph 1.7.0 REST client for the visualizer."""
    MAGMA_EDGE_TYPES = {"semantic", "temporal", "causal", "entity_ref"}
    MAGMA_VERTEX_LABELS = {"memory_event", "entity"}

    def __init__(self, base_url: str = "http://localhost:8080/graphspaces/DEFAULT/graphs/hugegraph"):
        self.base = base_url
        self._s = _requests.Session()
        self._s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
        self._alive = None

    def _decode(self, r):
        raw = r.content
        if raw and raw[:2] == b'\x1f\x8b':
            raw = _gzip.decompress(raw)
        return json.loads(raw) if raw else {}

    def alive(self) -> bool:
        if self._alive is not None:
            return self._alive
        try:
            r = self._s.get(f"{self.base}/schema/vertexlabels", timeout=3)
            self._alive = r.status_code == 200
        except Exception:
            self._alive = False
        return self._alive

    def get_graph_elements(self, include_supply_chain: bool = False) -> Dict[str, Any]:
        """Fetch all MAGMA vertices and edges from HugeGraph, return Cytoscape format."""
        elements = {"nodes": [], "edges": []}
        if not self.alive():
            return elements

        try:
            # Vertices
            vdata = self._decode(self._s.get(f"{self.base}/graph/vertices?page_size=500"))
            all_vertices = vdata.get("vertices", [])

            # Edges
            edata = self._decode(self._s.get(f"{self.base}/graph/edges?page_size=1000"))
            all_edges = edata.get("edges", [])

            # Filter MAGMA vertices
            if not include_supply_chain:
                vertices = [v for v in all_vertices if v["label"] in self.MAGMA_VERTEX_LABELS]
            else:
                vertices = all_vertices

            # Build node map for edge resolution
            vid_map = {v["id"]: v for v in vertices}

            # Add nodes
            for v in vertices:
                props = v.get("properties", {})
                label = props.get("name", v["label"])
                content = props.get("content", label)
                color = "#1A73E8" if v["label"] == "memory_event" else "#FF6D00" if v["label"] == "entity" else "#666"
                shape = "diamond" if v["label"] == "entity" else "ellipse"
                elements["nodes"].append({
                    "data": {
                        "id": str(v["id"]),
                        "label": (content[:28] + "...") if len(content) > 28 else content,
                        "full_content": content,
                        "timestamp": props.get("timestamp", ""),
                        "type": v["label"],
                        "attributes": props,
                        "vertex_label": v["label"],
                    },
                    "classes": v["label"],
                })

            # Add MAGMA edges
            for e in all_edges:
                etype = e["label"]
                if etype not in self.MAGMA_EDGE_TYPES and not include_supply_chain:
                    continue
                src_id = str(e.get("outV", ""))
                tgt_id = str(e.get("inV", ""))
                if src_id not in vid_map and tgt_id not in vid_map:
                    continue
                if src_id not in vid_map or tgt_id not in vid_map:
                    continue

                edge_colors = {"semantic": "#4FC3F7", "temporal": "#81C784", "causal": "#FFB74D",
                               "entity_ref": "#CE93D8"}
                w = e.get("properties", {}).get("weight", 1)
                directed = etype != "semantic"

                elements["edges"].append({
                    "data": {
                        "source": src_id,
                        "target": tgt_id,
                        "type": etype,
                        "weight": w,
                        "color": edge_colors.get(etype, "#555"),
                        "directed": directed,
                    }
                })

        except Exception as ex:
            print(f"[HugeGraphClient] Error: {ex}")

        return elements

    def get_stats(self) -> Dict[str, Any]:
        if not self.alive():
            return {"backend": "in-memory (HugeGraph unreachable)"}
        try:
            vdata = self._decode(self._s.get(f"{self.base}/graph/vertices?page_size=500"))
            edata = self._decode(self._s.get(f"{self.base}/graph/edges?page_size=1000"))
            vlabels = {}
            for v in vdata.get("vertices", []):
                vlabels[v["label"]] = vlabels.get(v["label"], 0) + 1
            elabels = {}
            for e in edata.get("edges", []):
                elabels[e["label"]] = elabels.get(e["label"], 0) + 1
            return {"backend": "HugeGraph 1.7.0 (LIVE)", "vertex_labels": vlabels,
                    "edge_labels": elabels, "total_vertices": len(vdata.get("vertices", [])),
                    "total_edges": len(edata.get("edges", []))}
        except Exception:
            return {"backend": "error"}


# Global HugeGraph client
hg_client = HugeGraphClient()


# ============================================================
# Data Models (shared with PoC)
# ============================================================

class IntentType(Enum):
    WHY = "why"
    WHEN = "when"
    ENTITY = "entity"


@dataclass
class MemoryNode:
    node_id: str
    content: str
    timestamp: str
    vector: List[float]
    attributes: Dict[str, Any] = field(default_factory=dict)
    graph_type: str = "memory"


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Lightweight MAGMA Store (for visualization)
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def generate_embedding(text: str, dim: int = 32) -> List[float]:
    text_lower = text.lower().strip()
    char_features = []
    for i in range(len(text_lower)):
        char_features.append(ord(text_lower[i]))
        if i > 0:
            char_features.append(ord(text_lower[i]) * 31 + ord(text_lower[i-1]))
        if i > 1:
            char_features.append(ord(text_lower[i]) * 961 + ord(text_lower[i-1]) * 31 + ord(text_lower[i-2]))
    keywords = set(text_lower.split()) - {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and"}
    for kw in keywords:
        for c in kw:
            char_features.append(ord(c) * 127)
    rng = random.Random(sum(char_features) % (2**32))
    base = sum(char_features) & 0xFFFFFFFF
    vec = []
    for i in range(dim):
        val = rng.gauss(0, 1)
        fi = (base + i * 7) % len(char_features) if char_features else 0
        val += (char_features[fi] / 128.0) * 0.5
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def extract_entities(text: str) -> List[Tuple[str, str]]:
    entities = []
    words = text.split()
    for word in words:
        if word[0].isupper() and len(word) > 2 and word not in {
            "The", "This", "That", "When", "Then", "After", "Before",
            "However", "Therefore", "Furthermore", "Meanwhile", "Server",
            "CPU", "Carol", "David"
        }:
            entities.append((word, "person"))
    concepts = ["bug", "feature", "deploy", "release", "error", "crash",
                "deadline", "review", "server", "database", "API", "issue",
                "authentication", "race condition", "index", "fix"]
    for concept in concepts:
        if concept in text.lower():
            entities.append((concept, "concept"))
    return list(set(entities))


class MAGMAStore:
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.entity_nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, List[MemoryEdge]] = {
            "semantic": [], "temporal": [], "causal": [], "entity_ref": []
        }
        self.vector_index: List[Tuple[str, List[float]]] = []
        self._built = False

    def build(self):
        if self._built:
            return
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
        written = []
        for content, ts, attrs in events:
            nid = f"mem_{hashlib.md5(content.encode()).hexdigest()[:12]}"
            vec = generate_embedding(content)
            node = MemoryNode(node_id=nid, content=content, timestamp=ts.isoformat(), vector=vec, attributes=attrs)
            self.nodes[nid] = node
            self.vector_index.append((nid, vec))
            written.append(node)
            # temporal chain
            if len(written) > 1:
                self.edges["temporal"].append(MemoryEdge(source_id=written[-2].node_id, target_id=nid, edge_type="temporal"))
        # semantic edges
        for i, (id1, v1) in enumerate(self.vector_index):
            for id2, v2 in self.vector_index[i+1:]:
                sim = cosine_similarity(v1, v2)
                if sim >= 0.55:
                    self.edges["semantic"].append(MemoryEdge(source_id=id1, target_id=id2, edge_type="semantic", weight=round(sim, 3)))
        # causal edges (simulated)
        cause_pairs = [
            (written[0].node_id, written[1].node_id),  # bug → investigation
            (written[1].node_id, written[2].node_id),  # investigation → root cause
            (written[2].node_id, written[3].node_id),  # root cause → fix
            (written[3].node_id, written[4].node_id),  # fix → CPU spike
            (written[4].node_id, written[5].node_id),  # CPU spike → discovery
            (written[5].node_id, written[6].node_id),  # discovery → index fix
        ]
        for src, tgt in cause_pairs:
            if src in self.nodes and tgt in self.nodes:
                self.edges["causal"].append(MemoryEdge(source_id=src, target_id=tgt, edge_type="causal", weight=0.9))
        # entity edges
        for node in written:
            for ename, etype in extract_entities(node.content):
                eid = f"ent_{hashlib.md5(ename.encode()).hexdigest()[:12]}"
                if eid not in self.entity_nodes:
                    self.entity_nodes[eid] = MemoryNode(node_id=eid, content=ename, timestamp=node.timestamp,
                                                          vector=generate_embedding(ename), attributes={"entity_type": etype}, graph_type="entity")
                    self.nodes[eid] = self.entity_nodes[eid]
                self.edges["entity_ref"].append(MemoryEdge(source_id=node.node_id, target_id=eid, edge_type="entity_ref"))
        self._built = True

    def route_intent(self, query: str) -> str:
        q = query.lower()
        if any(kw in q for kw in ["why", "caused", "原因", "导致", "because"]):
            return "why"
        if any(kw in q for kw in ["when", "什么时候", "时间", "yesterday", "after"]):
            return "when"
        return "entity"

    def get_graph_data(self, graph_type: Optional[str] = None) -> Dict[str, Any]:
        """Export graph as Cytoscape.js format"""
        elements = {"nodes": [], "edges": []}
        seen = set()
        edge_types = [graph_type] if graph_type else ["semantic", "temporal", "causal", "entity_ref"]
        colors = {"semantic": "#4FC3F7", "temporal": "#81C784", "causal": "#FFB74D", "entity_ref": "#CE93D8"}
        node_colors = {"memory": "#1A73E8", "entity": "#FF6D00"}
        for etype in edge_types:
            for e in self.edges.get(etype, []):
                src_node = self.nodes.get(e.source_id)
                tgt_node = self.nodes.get(e.target_id)
                if not src_node or not tgt_node:
                    continue
                for n in [src_node, tgt_node]:
                    if n.node_id not in seen:
                        seen.add(n.node_id)
                        color = node_colors.get(n.graph_type, "#666")
                        label = n.content[:30] + ("..." if len(n.content) > 30 else "")
                        node_data = {"data": {"id": n.node_id, "label": label, "full_content": n.content,
                                              "timestamp": n.timestamp, "type": n.graph_type,
                                              "attributes": n.attributes}, "classes": n.graph_type}
                        elements["nodes"].append(node_data)
                edge_data = {"data": {"source": e.source_id, "target": e.target_id,
                                      "type": etype, "weight": e.weight,
                                      "color": colors.get(etype, "#666")}}
                elements["edges"].append(edge_data)
        return elements

    def query(self, query: str) -> Dict[str, Any]:
        intent = self.route_intent(query)
        query_vec = generate_embedding(query)
        results = []
        for nid, node in self.nodes.items():
            if node.graph_type == "entity":
                continue
            sim = cosine_similarity(query_vec, node.vector)
            # Check which graph types connect to this node
            connected_types = set()
            for etype in ["semantic", "temporal", "causal", "entity_ref"]:
                for e in self.edges[etype]:
                    if e.source_id == nid or e.target_id == nid:
                        connected_types.add(etype)
            # Bonus for intent-relevant graph types
            type_weights = {"why": {"causal": 3.0, "temporal": 1.0, "semantic": 1.5, "entity_ref": 0.5},
                          "when": {"temporal": 3.0, "causal": 0.5, "semantic": 1.0, "entity_ref": 0.5},
                          "entity": {"entity_ref": 3.0, "semantic": 2.0, "causal": 1.0, "temporal": 0.5}}
            w = type_weights.get(intent, {})
            bonus = sum(w.get(t, 0.5) for t in connected_types)
            score = sim * (1 + bonus * 0.3)
            if score > 0.1:
                results.append({"node_id": nid, "content": node.content[:80] + ("..." if len(node.content) > 80 else ""),
                               "timestamp": node.timestamp, "score": round(score, 3),
                               "matched_graphs": list(connected_types)})
        results.sort(key=lambda x: -x["score"])
        return {"query": query, "intent": intent, "results": results[:10]}

    def get_stats(self) -> Dict[str, int]:
        return {
            "memory_nodes": sum(1 for n in self.nodes.values() if n.graph_type == "memory"),
            "entity_nodes": len(self.entity_nodes),
            "semantic_edges": len(self.edges["semantic"]),
            "temporal_edges": len(self.edges["temporal"]),
            "causal_edges": len(self.edges["causal"]),
            "entity_ref_edges": len(self.edges["entity_ref"]),
        }


# ============================================================
# Research Findings (today's survey)
# ============================================================

RESEARCH_FINDINGS = [
    {"rank": 1, "title": "MAGMA: Multi-Graph Agentic Memory (ACL 2026)", "source": "https://arxiv.org/abs/2601.03236",
     "desc": "四图正交记忆架构(Semantic/Temporal/Causal/Entity)，Intent Routing + Adaptive Beam Search，与Sprint1-10互补",
     "relevance": "高", "action": "已落地PoC"},
    {"rank": 2, "title": "Graph Database MCP生态 12+竞品分析", "source": "https://chatforest.com/reviews/graph-database-mcp-servers/",
     "desc": "Neo4j/TigerGraph/ArangoDB等12+图数据库已发布MCP Server，HugeGraph完全缺席",
     "relevance": "高", "action": "可立即落地"},
    {"rank": 3, "title": "RAG 2026四大新范式演进指南", "source": "https://radarai.top/articles/2026-年-RAG-技术最新进展与落地实践指南",
     "desc": "Graph-RAG工具化、Agent记忆框架、低成本私有部署三条新机会线，评价体系从Recall转向任务完成率",
     "relevance": "高", "action": "深入研究"},
    {"rank": 4, "title": "ICML 2026: 33篇LLM×Graph论文", "source": "https://cloud.tencent.com/developer/article/2671418",
     "desc": "图基础模型(GLAD/OpenMAG)、多模态属性图、KG问答等前沿方向，11篇知识图谱相关",
     "relevance": "中", "action": "跟踪"},
    {"rank": 5, "title": "AI Agent Memory架构综述", "source": "https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/",
     "desc": "从Context Window到持久知识，涵盖分类学、生产实现、检索策略，Memory是2026核心赛道",
     "relevance": "高", "action": "深入研究"},
    {"rank": 6, "title": "Multi-Agent增量KG融合管线", "source": "https://www.sciencedirect.com/science/article/pii/S2667305326000499",
     "desc": "多Agent协作增量构建KG，支持结构化/半结构化/非结构化三类数据融合，与飞书采集场景匹配",
     "relevance": "中", "action": "跟踪"},
    {"rank": 7, "title": "CodeGraph 3.1K Stars代码图谱", "source": "https://github.com/colbymchenry/codegraph",
     "desc": "Pre-indexed code KG，AST解析+函数调用图，给Agent预建知识图谱避免文件扫描，可对标代码图谱P2场景",
     "relevance": "中", "action": "跟踪"},
    {"rank": 8, "title": "Graphify: 代码库转KG实践", "source": "https://openclawapi.org/en/blog/2026-04-12-graphify-knowledge-graph",
     "desc": "Tree-sitter AST提取(无需LLM)+实体关系构建+English查询，与HugeGraph代码图谱方案技术栈一致",
     "relevance": "中", "action": "跟踪"},
]


# ============================================================
# HTTP Server
# ============================================================

store = MAGMAStore()
store.build()

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MAGMA Four-Graph Memory | HugeGraph-AI PoC</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.0/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.2/dagre.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-dagre/2.5.0/cytoscape-dagre.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; overflow-x: hidden; }
.header { background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); padding: 20px 30px; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 20px; }
.header h1 { font-size: 22px; color: #58a6ff; font-weight: 600; }
.header .badge { background: #238636; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; }
.header .meta { color: #8b949e; font-size: 13px; }
.tabs { display: flex; background: #161b22; border-bottom: 1px solid #30363d; padding: 0 20px; }
.tab { padding: 10px 20px; cursor: pointer; color: #8b949e; border-bottom: 2px solid transparent; transition: all 0.2s; font-size: 13px; }
.tab:hover { color: #c9d1d9; }
.tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.panel { display: none; padding: 20px; }
.panel.active { display: block; }

/* Stats Bar */
.stats-bar { display: flex; gap: 16px; padding: 16px 20px; background: #161b22; border-bottom: 1px solid #30363d; flex-wrap: wrap; }
.stat-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px 18px; min-width: 130px; }
.stat-card .label { font-size: 11px; color: #8b949e; letter-spacing: 1px; }
.stat-card .value { font-size: 24px; font-weight: 700; margin-top: 2px; }
.stat-card.semantic .value { color: #4FC3F7; }
.stat-card.temporal .value { color: #81C784; }
.stat-card.causal .value { color: #FFB74D; }
.stat-card.entity .value { color: #CE93D8; }
.stat-card.total .value { color: #58a6ff; }

/* Graph Layout */
.graph-container { display: flex; height: calc(100vh - 180px); }
.graph-controls { width: 240px; background: #161b22; border-right: 1px solid #30363d; padding: 16px; overflow-y: auto; }
.graph-viewport { flex: 1; position: relative; }
#cy { width: 100%; height: 100%; }
.control-group { margin-bottom: 20px; }
.control-group h3 { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.filter-btn { display: block; width: 100%; padding: 8px 12px; margin-bottom: 6px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; cursor: pointer; text-align: left; font-size: 13px; transition: all 0.2s; }
.filter-btn:hover { background: #30363d; }
.filter-btn.active { border-color: #58a6ff; background: #0d1117; }
.filter-btn .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }

/* Query Panel */
.query-container { padding: 16px; max-width: 900px; margin: 0 auto; }
.query-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.query-input { width: 100%; padding: 12px 16px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; color: #c9d1d9; font-size: 14px; outline: none; font-family: inherit; }
.query-input:focus { border-color: #58a6ff; }
.query-btn { padding: 10px 24px; background: #238636; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; margin-top: 10px; font-family: inherit; }
.query-btn:hover { background: #2ea043; }
.query-examples { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.example-chip { padding: 4px 12px; background: #21262d; border: 1px solid #30363d; border-radius: 16px; font-size: 12px; color: #8b949e; cursor: pointer; }
.example-chip:hover { color: #58a6ff; border-color: #58a6ff; }
.intent-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.intent-why { background: #FFB74D33; color: #FFB74D; }
.intent-when { background: #81C78433; color: #81C784; }
.intent-entity { background: #CE93D833; color: #CE93D8; }
.result-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 8px; transition: all 0.2s; }
.result-card:hover { border-color: #58a6ff; }
.result-card .score { font-size: 20px; font-weight: 700; color: #58a6ff; }
.result-card .content { color: #c9d1d9; font-size: 13px; margin-top: 4px; line-height: 1.5; }
.result-card .graphs { display: flex; gap: 6px; margin-top: 8px; }
.graph-tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; }
.graph-tag.semantic { background: #4FC3F722; color: #4FC3F7; }
.graph-tag.temporal { background: #81C78422; color: #81C784; }
.graph-tag.causal { background: #FFB74D22; color: #FFB74D; }
.graph-tag.entity_ref { background: #CE93D822; color: #CE93D8; }

/* Research Panel */
.research-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.research-table th { text-align: left; padding: 10px 12px; background: #161b22; color: #8b949e; font-weight: 600; border-bottom: 1px solid #30363d; }
.research-table td { padding: 10px 12px; border-bottom: 1px solid #21262d; vertical-align: top; }
.research-table tr:hover { background: #161b2288; }
.relevance-high { color: #FFB74D; font-weight: 600; }
.relevance-mid { color: #4FC3F7; }
.relevance-low { color: #8b949e; }
.action-land { color: #238636; font-weight: 600; }
.action-deep { color: #58a6ff; }
.action-track { color: #8b949e; }

/* Architecture Panel */
.arch-svg { text-align: center; padding: 20px; }
.arch-svg svg { max-width: 100%; height: auto; }

/* Node tooltip */
.node-tooltip { position: absolute; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; max-width: 300px; pointer-events: none; z-index: 100; display: none; }
.node-tooltip .tt-content { font-size: 12px; line-height: 1.5; color: #c9d1d9; }
.node-tooltip .tt-type { font-size: 11px; color: #8b949e; margin-top: 4px; }
</style>
</head>
<body>

<div class="header">
  <h1>MAGMA 四图记忆架构</h1>
  <span class="badge">PoC</span>
  <span class="meta">ACL 2026 | arXiv:2601.03236 | HugeGraph-AI</span>
  <span class="badge" id="backendBadge" style="background:#da3633">__BACKEND__</span>
</div>

<div class="tabs">
  <div class="tab active" data-panel="graph">图视图</div>
  <div class="tab" data-panel="query">查询与意图路由</div>
  <div class="tab" data-panel="arch">架构总览</div>
  <div class="tab" data-panel="research">调研发现 (2026-06-10)</div>
</div>

<div class="stats-bar" id="statsBar"></div>

<!-- Graph Panel -->
<div class="panel active" id="panel-graph">
  <div class="graph-container">
    <div class="graph-controls">
      <div class="control-group">
        <h3>图过滤</h3>
        <button class="filter-btn active" data-filter="all"><span class="dot" style="background:#58a6ff"></span>全部</button>
        <button class="filter-btn" data-filter="semantic"><span class="dot" style="background:#4FC3F7"></span>语义图</button>
        <button class="filter-btn" data-filter="temporal"><span class="dot" style="background:#81C784"></span>时间图</button>
        <button class="filter-btn" data-filter="causal"><span class="dot" style="background:#FFB74D"></span>因果图</button>
        <button class="filter-btn" data-filter="entity_ref"><span class="dot" style="background:#CE93D8"></span>实体引用</button>
      </div>
      <div class="control-group">
        <h3>布局</h3>
        <button class="filter-btn" id="btn-dagre">层级布局 (Dagre)</button>
        <button class="filter-btn" id="btn-cose">力导向布局 (CoSE)</button>
        <button class="filter-btn" id="btn-breadth">广度优先 (BFS)</button>
      </div>
      <div class="control-group">
        <h3>节点详情</h3>
        <div id="selectedInfo" style="font-size:12px;color:#8b949e;">点击节点查看详情</div>
      </div>
    </div>
    <div class="graph-viewport">
      <div id="cy"></div>
    </div>
  </div>
</div>

<!-- Query Panel -->
<div class="panel" id="panel-query">
  <div class="query-container">
    <div class="query-box">
      <h3 style="font-size:14px;color:#58a6ff;margin-bottom:12px;">MAGMA 意图路由 + Beam Search 检索</h3>
      <input class="query-input" id="queryInput" placeholder="输入关于 Agent 记忆的查询..." value="为什么服务器 CPU 飙升了？">
      <button class="query-btn" id="queryBtn">查询</button>
      <div class="query-examples">
        <span class="example-chip" data-q="Why did the server CPU spike?">为什么 CPU 飙升了？</span>
        <span class="example-chip" data-q="When was the authentication bug reported?">Bug 什么时候报告的？</span>
        <span class="example-chip" data-q="What events are related to Alice?">Alice 相关事件</span>
        <span class="example-chip" data-q="What caused the authentication fix deployment?">部署原因是什么？</span>
        <span class="example-chip" data-q="What happened after the race condition was discovered?">发现竞态条件后发生了什么？</span>
      </div>
    </div>
    <div id="queryResults"></div>
  </div>
</div>

<!-- Architecture Panel -->
<div class="panel" id="panel-arch">
  <div class="arch-svg">
    <svg viewBox="0 0 900 620" xmlns="http://www.w3.org/2000/svg" font-family="SF Mono, Menlo, Consolas, monospace">
      <!-- Title -->
      <text x="450" y="30" text-anchor="middle" fill="#58a6ff" font-size="18" font-weight="600">MAGMA 四图记忆架构</text>
      <text x="450" y="50" text-anchor="middle" fill="#8b949e" font-size="12">ACL 2026 | arXiv:2601.03236</text>

      <!-- Query Input -->
      <rect x="320" y="70" width="260" height="40" rx="8" fill="#21262d" stroke="#30363d"/>
      <text x="450" y="95" text-anchor="middle" fill="#c9d1d9" font-size="13">查询输入</text>

      <!-- Intent Router -->
      <rect x="330" y="130" width="240" height="50" rx="8" fill="#0d1117" stroke="#58a6ff"/>
      <text x="450" y="155" text-anchor="middle" fill="#58a6ff" font-size="13" font-weight="600">意图路由器</text>
      <text x="450" y="170" text-anchor="middle" fill="#8b949e" font-size="10">为什么 / 什么时候 / 实体</text>
      <line x1="450" y1="110" x2="450" y2="130" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arrow)"/>

      <!-- RRF Anchor -->
      <rect x="50" y="130" width="180" height="50" rx="8" fill="#0d1117" stroke="#4FC3F7"/>
      <text x="140" y="155" text-anchor="middle" fill="#4FC3F7" font-size="12" font-weight="600">RRF 锚定</text>
      <text x="140" y="170" text-anchor="middle" fill="#8b949e" font-size="10">向量 + 关键词 + 时间</text>

      <!-- Arrow from RRF to Intent -->
      <line x1="230" y1="155" x2="330" y2="155" stroke="#4FC3F7" stroke-width="1" stroke-dasharray="4"/>

      <!-- Four Graphs -->
      <!-- Semantic -->
      <rect x="30" y="220" width="190" height="130" rx="10" fill="#0d1117" stroke="#4FC3F7"/>
      <text x="125" y="245" text-anchor="middle" fill="#4FC3F7" font-size="14" font-weight="600">Semantic Graph</text>
      <text x="125" y="265" text-anchor="middle" fill="#8b949e" font-size="11">"发生了什么？"</text>
      <circle cx="70" cy="300" r="12" fill="#4FC3F733" stroke="#4FC3F7"/>
      <circle cx="130" cy="310" r="12" fill="#4FC3F733" stroke="#4FC3F7"/>
      <circle cx="100" cy="325" r="12" fill="#4FC3F733" stroke="#4FC3F7"/>
      <line x1="82" y1="300" x2="118" y2="310" stroke="#4FC3F7" stroke-width="1"/>
      <line x1="122" y1="318" x2="108" y2="322" stroke="#4FC3F7" stroke-width="1"/>
      <text x="125" y="340" text-anchor="middle" fill="#4FC3F7" font-size="10">无向加权边</text>

      <!-- Temporal -->
      <rect x="240" y="220" width="190" height="130" rx="10" fill="#0d1117" stroke="#81C784"/>
      <text x="335" y="245" text-anchor="middle" fill="#81C784" font-size="14" font-weight="600">时间图</text>
      <text x="335" y="265" text-anchor="middle" fill="#8b949e" font-size="11">"什么时候发生？"</text>
      <rect x="280" y="285" width="24" height="24" rx="3" fill="#81C78433" stroke="#81C784"/>
      <rect x="320" y="285" width="24" height="24" rx="3" fill="#81C78433" stroke="#81C784"/>
      <rect x="360" y="285" width="24" height="24" rx="3" fill="#81C78433" stroke="#81C784"/>
      <line x1="304" y1="297" x2="320" y2="297" stroke="#81C784" stroke-width="1.5" marker-end="url(#arrowG)"/>
      <line x1="344" y1="297" x2="360" y2="297" stroke="#81C784" stroke-width="1.5" marker-end="url(#arrowG)"/>
      <text x="335" y="330" text-anchor="middle" fill="#81C784" font-size="10">不可变时间链</text>

      <!-- Causal -->
      <rect x="450" y="220" width="190" height="130" rx="10" fill="#0d1117" stroke="#FFB74D"/>
      <text x="545" y="245" text-anchor="middle" fill="#FFB74D" font-size="14" font-weight="600">因果图</text>
      <text x="545" y="265" text-anchor="middle" fill="#8b949e" font-size="11">"为什么发生？"</text>
      <polygon points="495,295 505,280 515,295" fill="#FFB74D33" stroke="#FFB74D"/>
      <polygon points="555,310 565,295 575,310" fill="#FFB74D33" stroke="#FFB74D"/>
      <polygon points="525,325 535,310 545,325" fill="#FFB74D33" stroke="#FFB74D"/>
      <line x1="505" y1="295" x2="555" y2="310" stroke="#FFB74D" stroke-width="1.5" marker-end="url(#arrowO)"/>
      <line x1="565" y1="310" x2="535" y2="310" stroke="#FFB74D" stroke-width="1.5"/>
      <text x="545" y="340" text-anchor="middle" fill="#FFB74D" font-size="10">LLM 推理，有向</text>

      <!-- Entity -->
      <rect x="660" y="220" width="190" height="130" rx="10" fill="#0d1117" stroke="#CE93D8"/>
      <text x="755" y="245" text-anchor="middle" fill="#CE93D8" font-size="14" font-weight="600">实体图</text>
      <text x="755" y="265" text-anchor="middle" fill="#8b949e" font-size="11">"涉及哪些实体？"</text>
      <rect x="710" y="285" width="30" height="30" rx="15" fill="#CE93D833" stroke="#CE93D8"/>
      <rect x="770" y="285" width="30" height="30" rx="15" fill="#CE93D833" stroke="#CE93D8"/>
      <line x1="740" y1="300" x2="770" y2="300" stroke="#CE93D8" stroke-width="1" stroke-dasharray="3"/>
      <text x="755" y="335" text-anchor="middle" fill="#CE93D8" font-size="10">跨时间窗口实体链接</text>

      <!-- Lines from Intent Router to graphs -->
      <line x1="400" y1="180" x2="125" y2="220" stroke="#4FC3F7" stroke-width="1" stroke-dasharray="4"/>
      <line x1="430" y1="180" x2="335" y2="220" stroke="#81C784" stroke-width="1.5"/>
      <line x1="470" y1="180" x2="545" y2="220" stroke="#FFB74D" stroke-width="1" stroke-dasharray="4"/>
      <line x1="500" y1="180" x2="755" y2="220" stroke="#CE93D8" stroke-width="1" stroke-dasharray="4"/>

      <!-- Adaptive Beam Search -->
      <rect x="250" y="390" width="400" height="50" rx="8" fill="#0d1117" stroke="#58a6ff"/>
      <text x="450" y="415" text-anchor="middle" fill="#58a6ff" font-size="14" font-weight="600">自适应束搜索</text>
      <text x="450" y="430" text-anchor="middle" fill="#8b949e" font-size="10">S(n_j|n_i,q) = exp(lam1*phi(type) + lam2*sim(n_j,q))</text>
      <line x1="450" y1="350" x2="450" y2="390" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arrow)"/>

      <!-- Context Synthesis -->
      <rect x="300" y="480" width="300" height="45" rx="8" fill="#0d1117" stroke="#238636"/>
      <text x="450" y="505" text-anchor="middle" fill="#238636" font-size="14" font-weight="600">上下文综合</text>
      <text x="450" y="518" text-anchor="middle" fill="#8b949e" font-size="10">跨图证据聚合</text>
      <line x1="450" y1="440" x2="450" y2="480" stroke="#238636" stroke-width="1.5" marker-end="url(#arrowG)"/>

      <!-- Response -->
      <rect x="350" y="560" width="200" height="35" rx="8" fill="#21262d" stroke="#58a6ff"/>
      <text x="450" y="582" text-anchor="middle" fill="#c9d1d9" font-size="13">返回给 Agent</text>
      <line x1="450" y1="525" x2="450" y2="560" stroke="#58a6ff" stroke-width="1" marker-end="url(#arrow)"/>

      <!-- Fast Path / Slow Path labels -->
      <rect x="30" y="400" width="160" height="55" rx="8" fill="#21262d" stroke="#30363d"/>
      <text x="110" y="420" text-anchor="middle" fill="#81C784" font-size="12" font-weight="600">快路径（同步）</text>
      <text x="110" y="435" text-anchor="middle" fill="#8b949e" font-size="10">时间链 + 语义图</text>
      <text x="110" y="448" text-anchor="middle" fill="#8b949e" font-size="10">+ 向量索引</text>

      <rect x="30" y="470" width="160" height="55" rx="8" fill="#21262d" stroke="#30363d"/>
      <text x="110" y="490" text-anchor="middle" fill="#FFB74D" font-size="12" font-weight="600">慢路径（异步）</text>
      <text x="110" y="505" text-anchor="middle" fill="#8b949e" font-size="10">因果图（LLM 推理）</text>
      <text x="110" y="518" text-anchor="middle" fill="#8b949e" font-size="10">+ 实体抽取</text>

      <!-- HugeGraph box -->
      <rect x="660" y="400" width="190" height="130" rx="10" fill="#21262d" stroke="#238636"/>
      <text x="755" y="425" text-anchor="middle" fill="#238636" font-size="13" font-weight="600">HugeGraph</text>
      <text x="755" y="445" text-anchor="middle" fill="#8b949e" font-size="10">一图四视图</text>
      <text x="755" y="465" text-anchor="middle" fill="#8b949e" font-size="10">Gremlin: g.V().hasEdge(type)</text>
      <text x="755" y="485" text-anchor="middle" fill="#8b949e" font-size="10">Vermeer OLAP: 60 亿节点</text>
      <text x="755" y="505" text-anchor="middle" fill="#8b949e" font-size="10">MCP: 10 个工具</text>
      <text x="755" y="520" text-anchor="middle" fill="#8b949e" font-size="10">edge_type 属性过滤</text>

      <!-- Arrows -->
      <defs>
        <marker id="arrow" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="10" markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#58a6ff"/></marker>
        <marker id="arrowG" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="10" markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#81C784"/></marker>
        <marker id="arrowO" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="10" markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#FFB74D"/></marker>
      </defs>
    </svg>
  </div>
</div>

<!-- Research Panel -->
<div class="panel" id="panel-research">
  <div style="max-width:1000px;margin:0 auto;">
    <table class="research-table">
      <thead>
        <tr><th>#</th><th>标题</th><th>描述</th><th>关联度</th><th>建议</th></tr>
      </thead>
      <tbody id="researchBody"></tbody>
    </table>
  </div>
</div>

<div class="node-tooltip" id="tooltip">
  <div class="tt-content" id="ttContent"></div>
  <div class="tt-type" id="ttType"></div>
</div>

<script>
const STATS = __STATS__;
const GRAPH_DATA = __GRAPH_DATA__;
const RESEARCH = __RESEARCH__;

// Normalize stats (works for both in-memory and HugeGraph formats)
const STATS_N = {
  semantic_edges: STATS.semantic_edges || (STATS.edge_labels && STATS.edge_labels.semantic) || 0,
  temporal_edges: STATS.temporal_edges || (STATS.edge_labels && STATS.edge_labels.temporal) || 0,
  causal_edges: STATS.causal_edges || (STATS.edge_labels && STATS.edge_labels.causal) || 0,
  entity_ref_edges: STATS.entity_ref_edges || (STATS.edge_labels && STATS.edge_labels.entity_ref) || 0,
  memory_nodes: STATS.memory_nodes || (STATS.vertex_labels && (STATS.vertex_labels.memory_event || 0)) || 0,
  entity_nodes: STATS.entity_nodes || (STATS.vertex_labels && (STATS.vertex_labels.entity || 0)) || 0,
  backend: STATS.backend || STATS.get("backend", "unknown"),
};

// Init stats
const statsHTML = [
  `<div class="stat-card semantic"><div class="label">语义边</div><div class="value">${STATS_N.semantic_edges}</div></div>`,
  `<div class="stat-card temporal"><div class="label">时间边</div><div class="value">${STATS_N.temporal_edges}</div></div>`,
  `<div class="stat-card causal"><div class="label">因果边</div><div class="value">${STATS_N.causal_edges}</div></div>`,
  `<div class="stat-card entity"><div class="label">实体引用</div><div class="value">${STATS_N.entity_ref_edges}</div></div>`,
  `<div class="stat-card total"><div class="label">记忆节点</div><div class="value">${STATS_N.memory_nodes}</div></div>`,
  `<div class="stat-card"><div class="label">实体节点</div><div class="value" style="color:#FF6D00">${STATS_N.entity_nodes}</div></div>`,
  `<div class="stat-card total"><div class="label">后端</div><div class="value" style="color:#da3633;font-size:12px">${STATS_N.backend}</div></div>`,
].join('');
document.getElementById('statsBar').innerHTML = statsHTML;

// Research table
const rBody = document.getElementById('researchBody');
RESEARCH.forEach(r => {
  const relClass = r.relevance === '高' ? 'relevance-high' : r.relevance === '中' ? 'relevance-mid' : 'relevance-low';
  const actClass = r.action === '已落地PoC' || r.action === '可立即落地' ? 'action-land' : r.action === '深入研究' ? 'action-deep' : 'action-track';
  rBody.innerHTML += `<tr>
    <td>${r.rank}</td>
    <td><a href="${r.source}" target="_blank" style="color:#58a6ff;text-decoration:none">${r.title}</a></td>
    <td style="font-size:12px;color:#8b949e;max-width:350px">${r.desc}</td>
    <td class="${relClass}">${r.relevance}</td>
    <td class="${actClass}">${r.action}</td>
  </tr>`;
});

// Tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.panel).classList.add('active');
    if (tab.dataset.panel === 'graph') setTimeout(() => cy.resize(), 100);
  });
});

// Cytoscape.js
const edgeColors = { semantic: '#4FC3F7', temporal: '#81C784', causal: '#FFB74D', entity_ref: '#CE93D8' };

let currentFilter = 'all';
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: GRAPH_DATA,
  style: [
    { selector: 'node.memory_event', style: { 'background-color': '#1A73E8', 'label': 'data(label)', 'font-size': '10px', 'color': '#c9d1d9', 'text-valign': 'center', 'text-halign': 'center', 'width': 40, 'height': 40, 'text-wrap': 'wrap', 'text-max-width': '80px', 'text-overflow-wrap': 'anywhere' }},
    { selector: 'node.entity', style: { 'background-color': '#FF6D00', 'label': 'data(label)', 'font-size': '11px', 'color': '#fff', 'text-valign': 'center', 'text-halign': 'center', 'width': 35, 'height': 35, 'shape': 'diamond', 'text-wrap': 'wrap', 'text-max-width': '80px' }},
    { selector: 'edge', style: { 'width': 2, 'line-color': '#555', 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#555', 'arrow-scale': 0.8, 'opacity': 0.7 }},
    { selector: 'node:active', style: { 'overlay-opacity': 0.1 }},
    { selector: '.highlighted', style: { 'opacity': 1, 'z-index': 999 }},
    { selector: '.dimmed', style: { 'opacity': 0.15 }},
  ],
  layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.2, padding: 30 },
});

function applyEdgeStyles() {
  cy.edges().forEach(e => {
    const type = e.data('type');
    const color = edgeColors[type] || '#555';
    e.style({ 'line-color': color, 'target-arrow-color': color });
    if (type === 'semantic') e.style({ 'target-arrow-shape': 'none' });
  });
}
applyEdgeStyles();

function filterGraph(type) {
  if (type === 'all') {
    cy.elements().removeClass('dimmed highlighted').style({ 'opacity': 1 });
    applyEdgeStyles();
  } else {
    cy.edges().forEach(e => {
      if (e.data('type') === type) { e.removeClass('dimmed'); e.style({ 'opacity': 0.8 }); }
      else { e.addClass('dimmed'); e.style({ 'opacity': 0.08 }); }
    });
    const connected = cy.edges().filter(e => e.data('type') === type).connectedNodes();
    cy.nodes().difference(connected).addClass('dimmed').style({ 'opacity': 0.08 });
    connected.removeClass('dimmed').style({ 'opacity': 1 });
  }
}

document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn[data-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    filterGraph(currentFilter);
  });
});

// Layout buttons
document.getElementById('btn-dagre').addEventListener('click', () => {
  try { cy.layout({ name: 'dagre', rankDir: 'TB', spacingFactor: 0.8, padding: 30, animate: true, animationDuration: 500 }).run(); }
  catch(e) { console.error('dagre layout error:', e); }
});
document.getElementById('btn-cose').addEventListener('click', () => {
  try { cy.layout({ name: 'cose-bilkent', nodeRepulsion: 8000, idealEdgeLength: 80, padding: 30, animate: true, animationDuration: 500 }).run(); }
  catch(e) { console.error('cose layout error:', e); }
});
document.getElementById('btn-breadth').addEventListener('click', () => {
  try { cy.layout({ name: 'breadthfirst', directed: true, spacingFactor: 1.2, padding: 30, animate: true, animationDuration: 500 }).run(); }
  catch(e) { console.error('breadthfirst layout error:', e); }
});

// Node click
cy.on('tap', 'node', e => {
  const node = e.target;
  document.getElementById('selectedInfo').innerHTML =
    `<strong style="color:#58a6ff">${node.data('label')}</strong><br><span style="color:#8b949e">${node.data('full_content') || ''}</span><br><span style="color:#484f58">${node.data('timestamp') || ''}</span>`;
});

// Tooltip
cy.on('mouseover', 'node', e => {
  const node = e.target;
  const tt = document.getElementById('tooltip');
  document.getElementById('ttContent').textContent = node.data('full_content') || node.data('label');
  document.getElementById('ttType').textContent = node.data('type') + ' | ' + (node.data('timestamp') || '');
  tt.style.display = 'block';
});
cy.on('mouseout', 'node', () => { document.getElementById('tooltip').style.display = 'none'; });

// Query
function doQuery(q) {
  document.getElementById('queryInput').value = q;
  fetch('/api/query?q=' + encodeURIComponent(q)).then(r => r.json()).then(data => {
    const intentClass = data.intent === 'why' ? 'intent-why' : data.intent === 'when' ? 'intent-when' : 'intent-entity';
    const intentLabel = data.intent === 'why' ? '原因' : data.intent === 'when' ? '时间' : '实体';
    // Map intent to primary edge type filter
    const intentEdgeMap = { why: 'causal', when: 'temporal', entity: 'entity_ref' };
    const primaryEdge = intentEdgeMap[data.intent] || null;
    let html = `<div style="margin-bottom:16px"><span class="intent-badge ${intentClass}">意图: ${intentLabel}</span> <span style="color:#8b949e;font-size:12px;margin-left:8px">${data.results.length} 条结果</span></div>`;
    data.results.forEach(r => {
      const graphTags = r.matched_graphs.map(g =>
        `<span class="graph-tag ${g}">${g}</span>`
      ).join('');
      html += `<div class="result-card">
        <div class="score">${r.score}</div>
        <div class="content">${r.content}</div>
        <div class="graphs">${graphTags}</div>
      </div>`;
    });
    document.getElementById('queryResults').innerHTML = html;

    // Switch to primary graph filter based on intent
    if (primaryEdge) {
      cy.edges().forEach(e => {
        if (e.data('type') === primaryEdge) { e.removeClass('dimmed'); e.style({ 'opacity': 0.8, 'line-width': 3 }); }
        else { e.addClass('dimmed'); e.style({ 'opacity': 0.05, 'line-width': 1 }); }
      });
    }

    // Highlight matched nodes, dim others
    const resultIds = new Set(data.results.map(r => r.node_id));
    cy.nodes().forEach(n => {
      if (resultIds.has(n.id())) { n.removeClass('dimmed'); n.style({ 'opacity': 1, 'z-index-compare': 1 }); }
      else { n.addClass('dimmed'); n.style({ 'opacity': 0.1, 'z-index-compare': 0 }); }
    });

    // Update filter buttons to show active intent
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.filter === primaryEdge);
    });

    // Switch to graph tab to see highlights
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelector('.tab[data-panel="graph"]').classList.add('active');
    document.getElementById('panel-graph').classList.add('active');
    setTimeout(() => cy.resize(), 100);
  });
}

document.getElementById('queryBtn').addEventListener('click', () => doQuery(document.getElementById('queryInput').value));
document.getElementById('queryInput').addEventListener('keydown', e => { if (e.key === 'Enter') doQuery(e.target.value); });
document.querySelectorAll('.example-chip').forEach(c => c.addEventListener('click', () => doQuery(c.dataset.q)));

// Auto-query on load
setTimeout(() => doQuery('Why did the server CPU spike?'), 500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            # Use HugeGraph if alive, fallback to in-memory store
            if hg_client.alive():
                stats = hg_client.get_stats()
                graph_data = hg_client.get_graph_elements(include_supply_chain=False)
                backend = "HugeGraph LIVE"
            else:
                stats = store.get_stats()
                graph_data = store.get_graph_data()
                backend = "In-Memory"
            html = HTML_TEMPLATE.replace('__STATS__', json.dumps(stats)) \
                                  .replace('__GRAPH_DATA__', json.dumps(graph_data)) \
                                  .replace('__RESEARCH__', json.dumps(RESEARCH_FINDINGS)) \
                                  .replace('__BACKEND__', backend)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        elif self.path.startswith('/api/query'):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get('q', [''])[0]
            result = store.query(query)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        elif self.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            stats = hg_client.get_stats() if hg_client.alive() else store.get_stats()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
        elif self.path == '/api/graph':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if hg_client.alive():
                self.wfile.write(json.dumps(hg_client.get_graph_elements()).encode('utf-8'))
            else:
                self.wfile.write(json.dumps(store.get_graph_data()).encode('utf-8'))
        elif self.path == '/api/hg-stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(hg_client.get_stats()).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5003)
    args = parser.parse_args()

    print(f"\n  MAGMA Four-Graph Memory Visualization")
    print(f"  http://localhost:{args.port}")
    print(f"  4 graphs | Intent Routing | Beam Search | Research Dashboard")
    print()
    server = HTTPServer(('0.0.0.0', args.port), Handler)
    server.serve_forever()
