"""
MAGMA Agent Integration + Four-Graph Pipeline Visualizer
============================================================
Demonstrates:
  1. Agent <-> Memory API contract (what agent sends, what memory returns)
  2. Four-graph construction pipeline (Fast Path + Slow Path)
  3. Live interaction: send events as agent, query memory, see graph evolution

Usage: python3.10 magma_agent_pipeline.py --port 5004
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
import argparse

# ============================================================
# HugeGraph REST Client
# ============================================================
class HugeGraphClient:
    MAGMA_EDGE_TYPES = {"semantic", "temporal", "causal", "entity_ref"}
    MAGMA_VERTEX_LABELS = {"memory_event", "entity"}

    def __init__(self, base_url="http://localhost:8080/graphspaces/DEFAULT/graphs/hugegraph"):
        self.base = base_url
        self._s = _requests.Session()
        self._s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
        self._alive = None

    def _decode(self, r):
        raw = r.content
        if raw and raw[:2] == b'\x1f\x8b': raw = _gzip.decompress(raw)
        return json.loads(raw) if raw else {}

    def alive(self) -> bool:
        if self._alive is not None: return self._alive
        try:
            r = self._s.get(f"{self.base}/schema/vertexlabels", timeout=3)
            self._alive = r.status_code == 200
        except Exception: self._alive = False
        return self._alive

    def get_graph_elements(self, include_supply_chain=False):
        elements = {"nodes": [], "edges": []}
        if not self.alive(): return elements
        try:
            vdata = self._decode(self._s.get(f"{self.base}/graph/vertices?page_size=500"))
            edata = self._decode(self._s.get(f"{self.base}/graph/edges?page_size=1000"))
            vertices = [v for v in vdata.get("vertices", []) if v["label"] in self.MAGMA_VERTEX_LABELS] if not include_supply_chain else vdata.get("vertices", [])
            vid_map = {v["id"]: v for v in vertices}
            for v in vertices:
                props = v.get("properties", {})
                content = props.get("content", props.get("name", v["label"]))
                elements["nodes"].append({"data": {
                    "id": str(v["id"]), "label": (content[:28] + "...") if len(content) > 28 else content,
                    "full_content": content, "timestamp": props.get("timestamp", ""),
                    "type": v["label"], "vertex_label": v["label"], "attributes": props,
                }, "classes": v["label"]})
            for e in edata.get("edges", []):
                etype = e["label"]
                if etype not in self.MAGMA_EDGE_TYPES and not include_supply_chain: continue
                src, tgt = str(e.get("outV", "")), str(e.get("inV", ""))
                if src not in vid_map or tgt not in vid_map: continue
                elements["edges"].append({"data": {
                    "source": src, "target": tgt, "type": etype,
                    "weight": e.get("properties", {}).get("weight", 1),
                    "color": {"semantic": "#4FC3F7", "temporal": "#81C784", "causal": "#FFB74D", "entity_ref": "#CE93D8"}.get(etype, "#555"),
                    "directed": etype != "semantic",
                }})
        except Exception: pass
        return elements

    def get_stats(self):
        if not self.alive(): return {"backend": "离线"}
        try:
            vdata = self._decode(self._s.get(f"{self.base}/graph/vertices?page_size=500"))
            edata = self._decode(self._s.get(f"{self.base}/graph/edges?page_size=1000"))
            vl, el = {}, {}
            for v in vdata.get("vertices", []): vl[v["label"]] = vl.get(v["label"], 0) + 1
            for e in edata.get("edges", []): el[e["label"]] = el.get(e["label"], 0) + 1
            return {"backend": "HugeGraph 1.7.0 (LIVE)", "vertex_labels": vl, "edge_labels": el,
                    "total_vertices": len(vdata.get("vertices", [])), "total_edges": len(edata.get("edges", []))}
        except Exception: return {"backend": "error"}

hg_client = HugeGraphClient()

# ============================================================
# MAGMA Core: Agent Interface + Four-Graph Construction
# ============================================================

class IntentType(Enum):
    WHY = "why"
    WHEN = "when"
    ENTITY = "entity"
    GENERAL = "general"

@dataclass
class MemoryNode:
    node_id: str
    content: str
    timestamp: str
    vector: List[float] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)
    graph_type: str = "memory_event"

@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    edge_type: str  # semantic / temporal / causal / entity_ref
    weight: float = 1.0
    description: str = ""

def generate_embedding(text: str, dim=32):
    """Deterministic embedding simulation based on character n-gram hash."""
    text_lower = text.lower().strip()
    features = []
    for i, c in enumerate(text_lower):
        features.append(ord(c))
        if i > 0: features.append(ord(c) * 31 + ord(text_lower[i-1]))
        if i > 1: features.append(ord(c) * 961 + ord(text_lower[i-1]) * 31 + ord(text_lower[i-2]))
    rng = random.Random(sum(features) % (2**32))
    base = sum(features) & 0xFFFFFFFF
    vec = []
    for i in range(dim):
        val = rng.gauss(0, 1)
        fi = (base + i * 7) % len(features) if features else 0
        val += (features[fi] / 128.0) * 0.5
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0

def extract_keywords(text):
    words = set()
    for w in text.lower().split():
        if len(w) > 3: words.add(w)
    return list(words)

def extract_entities(text):
    """Simple entity extraction: capitalized words and quoted strings."""
    entities = []
    import re
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
    for e in caps:
        if e.lower() not in ("the", "this", "that", "then", "when", "after", "before", "while", "with", "from"):
            entities.append(e)
    return entities


class MAGMAAgentMemory:
    """
    MAGMA Memory System - Agent Integration Interface
    =================================================
    This class demonstrates the complete data contract between
    an AI Agent and the MAGMA memory system.
    """

    def __init__(self):
        # Storage
        self.nodes: Dict[str, MemoryNode] = {}
        self.entity_nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, List[MemoryEdge]] = {"semantic": [], "temporal": [], "causal": [], "entity_ref": []}
        self.event_timeline: List[str] = []  # ordered node IDs

        # Pipeline tracking
        self.pipeline_log: List[Dict] = []  # records of each write/query step
        self.step_counter = 0

        # Hyperparameters (from MAGMA paper Table 6)
        self.RRF_K = 60
        self.MAX_DEPTH = 5
        self.BEAM_WIDTH = 5
        self.LAMBDA_1 = 1.0  # structural alignment weight
        self.LAMBDA_2 = 0.5  # semantic affinity weight
        self.semantic_threshold = 0.5

    # ---- AGENT -> MEMORY: Write Interface ----

    def agent_observe(self, interaction: str, metadata: Dict = None) -> Dict:
        """
        Agent sends an observation/event to memory.
        This triggers the dual-stream write pipeline.

        Agent Input Format:
        {
            "interaction": "Alice reported a login authentication bug",
            "metadata": {"source": "slack", "priority": "high"},
            "timestamp": "2026-06-10T10:30:00"
        }

        Returns: Fast Path result (synchronous, immediate)
        {
            "status": "stored",
            "node_id": "mem_abc123",
            "fast_path_edges": [{"type": "temporal", ...}],
            "slow_path_queued": true,
            "latency_ms": 12
        }
        """
        ts = metadata.get("timestamp", datetime.now().isoformat()[:19]) if metadata else datetime.now().isoformat()[:19]
        self.step_counter += 1

        # === FAST PATH (Synchronous) ===
        fast_log = {"step": self.step_counter, "phase": "FAST_PATH", "timestamp": ts}

        # 1. Event Segmentation
        node_id = f"mem_{hashlib.md5(interaction.encode()).hexdigest()[:12]}"
        node = MemoryNode(
            node_id=node_id,
            content=interaction,
            timestamp=ts,
            vector=generate_embedding(interaction),
            attributes=metadata or {},
            graph_type="memory_event"
        )
        self.nodes[node_id] = node
        self.event_timeline.append(node_id)
        fast_log["action"] = "segment_event"
        fast_log["node_id"] = node_id
        fast_log["content_preview"] = interaction[:80]

        # 2. Vector Indexing
        fast_log["vector_dim"] = len(node.vector)
        fast_log["vector_indexed"] = True

        # 3. Temporal Chain (n_{t-1} -> n_t)
        temporal_edges = []
        if len(self.event_timeline) >= 2:
            prev_id = self.event_timeline[-2]
            edge = MemoryEdge(source_id=prev_id, target_id=node_id, edge_type="temporal",
                            weight=1, description="temporal_chain")
            self.edges["temporal"].append(edge)
            temporal_edges.append({"from": prev_id[:12], "to": node_id[:12]})
        fast_log["temporal_edges"] = temporal_edges

        fast_path_result = {
            "status": "stored",
            "node_id": node_id,
            "fast_path_edges": temporal_edges,
            "slow_path_queued": True,
            "latency_ms": random.randint(8, 25),
            "phase": "fast_path"
        }

        # === SLOW PATH (Asynchronous - simulated immediately for demo) ===
        slow_log = self._run_slow_path(node_id, node)
        slow_log["step"] = self.step_counter

        self.pipeline_log.append({"fast": fast_log, "slow": slow_log})

        return fast_path_result

    def _run_slow_path(self, node_id: str, node: MemoryNode) -> Dict:
        """
        Slow Path: Structural Consolidation
        Uses LLM to infer causal and entity edges.
        In production, this runs asynchronously as a background worker.
        """
        slow_log = {"phase": "SLOW_PATH", "node_id": node_id}

        # Get local neighborhood (2-hop)
        neighborhood = self._get_neighborhood(node_id, hops=2)
        slow_log["neighborhood_size"] = len(neighborhood)

        # 1. Semantic edges (embedding similarity)
        semantic_edges = []
        for nid, other_node in self.nodes.items():
            if nid == node_id or other_node.graph_type == "entity": continue
            sim = cosine_sim(node.vector, other_node.vector)
            if sim > self.semantic_threshold:
                edge = MemoryEdge(source_id=node_id, target_id=nid, edge_type="semantic",
                                weight=round(sim * 10), description=f"sim={sim:.2f}")
                self.edges["semantic"].append(edge)
                semantic_edges.append({"to": nid[:12], "similarity": round(sim, 3)})
        slow_log["semantic_edges_found"] = len(semantic_edges)
        slow_log["semantic_details"] = semantic_edges[:5]

        # 2. Causal edges (LLM-inferred - simulated)
        causal_edges = self._infer_causal_edges(node_id, node, neighborhood)
        slow_log["causal_edges_found"] = len(causal_edges)
        slow_log["causal_details"] = causal_edges

        # 3. Entity extraction and linking
        entity_edges = self._extract_and_link_entities(node_id, node)
        slow_log["entity_edges_found"] = len(entity_edges)
        slow_log["entity_details"] = entity_edges

        return slow_log

    def _infer_causal_edges(self, node_id, node, neighborhood):
        """Simulate LLM causal reasoning: find events that could cause/be caused by this event."""
        causal_edges = []
        content_lower = node.content.lower()

        # Simple heuristic causal patterns
        cause_keywords = {"fix", "deploy", "resolve", "restart", "patch", "update", "changed"}
        effect_keywords = {"crash", "spike", "error", "fail", "slow", "timeout", "bug", "issue"}

        for nid, other in self.nodes.items():
            if nid == node_id or other.graph_type == "entity": continue
            other_lower = other.content.lower()
            # Check if other could be cause of current
            if any(w in other_lower for w in cause_keywords) and any(w in content_lower for w in effect_keywords):
                if other.timestamp < node.timestamp:
                    edge = MemoryEdge(source_id=nid, target_id=node_id, edge_type="causal",
                                    weight=1, description="potential_cause")
                    self.edges["causal"].append(edge)
                    causal_edges.append({"from": nid[:12], "to": node_id[:12], "reason": "cause_effect"})
            # Check if current could be cause of other
            elif any(w in content_lower for w in cause_keywords) and any(w in other_lower for w in effect_keywords):
                if node.timestamp < other.timestamp:
                    edge = MemoryEdge(source_id=node_id, target_id=nid, edge_type="causal",
                                    weight=1, description="potential_cause")
                    self.edges["causal"].append(edge)
                    causal_edges.append({"from": node_id[:12], "to": nid[:12], "reason": "cause_effect"})

        return causal_edges

    def _extract_and_link_entities(self, node_id, node):
        """Extract entities from event and link to entity nodes."""
        entity_edges = []
        entities = extract_entities(node.content)

        for ename in entities:
            eid = f"ent_{hashlib.md5(ename.encode()).hexdigest()[:12]}"
            if eid not in self.entity_nodes:
                self.entity_nodes[eid] = MemoryNode(
                    node_id=eid, content=ename, timestamp=node.timestamp,
                    vector=generate_embedding(ename), attributes={"entity_type": "extracted"},
                    graph_type="entity"
                )
                self.nodes[eid] = self.entity_nodes[eid]
            edge = MemoryEdge(source_id=node_id, target_id=eid, edge_type="entity_ref",
                            weight=1, description=f"contains_{ename}")
            self.edges["entity_ref"].append(edge)
            entity_edges.append({"entity": ename, "entity_id": eid[:12]})

        return entity_edges

    def _get_neighborhood(self, node_id, hops=2):
        """Get nodes within N hops of the given node."""
        visited = {node_id}
        frontier = [node_id]
        for _ in range(hops):
            next_frontier = []
            for nid in frontier:
                for etype in self.edges.values():
                    for e in etype:
                        if e.source_id == nid and e.target_id not in visited:
                            visited.add(e.target_id)
                            next_frontier.append(e.target_id)
                        elif e.target_id == nid and e.source_id not in visited:
                            visited.add(e.source_id)
                            next_frontier.append(e.source_id)
            frontier = next_frontier
        return {nid: self.nodes[nid] for nid in visited if nid in self.nodes}

    # ---- AGENT -> MEMORY: Query Interface ----

    def agent_query(self, query: str) -> Dict:
        """
        Agent queries memory for relevant context.

        Agent Input:
        {
            "query": "Why did the CPU spike?",
            "max_tokens": 4000,
            "intent": "auto"  // or "why"/"when"/"entity"
        }

        Memory Returns:
        {
            "intent": "why",
            "context": "<t:2026-06-01T19:00> Server CPU usage spiked to 95% <ref:mem_abc123>\n<t:2026-06-01T18:00> Database query caused lock contention <ref:mem_def456>\n...2 intermediate events...",
            "nodes_used": 5,
            "token_estimate": 280,
            "traversal_path": [...],
            "beam_search_log": [...]
        }
        """
        self.step_counter += 1
        ts = datetime.now().isoformat()[:19]

        # === Phase 1: Intent Classification ===
        intent = self._classify_intent(query)

        # === Phase 2: Multi-Signal Anchor (RRF) ===
        q_vec = generate_embedding(query)
        q_keywords = extract_keywords(query)
        anchors = self._rrf_anchors(q_vec, q_keywords)

        # === Phase 3: Adaptive Beam Search ===
        beam_log = self._beam_search(anchors, q_vec, intent)

        # === Phase 4: Context Synthesis ===
        sorted_nodes = self._topological_sort(beam_log["visited"], intent)
        context = self._synthesize_context(sorted_nodes, intent)

        result = {
            "query": query,
            "intent": intent,
            "intent_label": {"why": "原因", "when": "时间", "entity": "实体", "general": "通用"}[intent],
            "context": context,
            "nodes_used": len(sorted_nodes),
            "token_estimate": len(context) // 4,  # rough estimate
            "anchors": [nid[:12] for nid in anchors],
            "traversal_path": beam_log["traversal_path"],
            "beam_search_depth": beam_log["depth"],
            "total_visited": beam_log["total_visited"],
        }

        self.pipeline_log.append({
            "query": result, "step": self.step_counter, "timestamp": ts
        })
        return result

    def _classify_intent(self, query):
        q = query.lower()
        if any(kw in q for kw in ["why", "cause", "原因", "导致", "因为", "caused", "what led"]):
            return "why"
        if any(kw in q for kw in ["when", "什么时候", "时间", "after", "before", "顺序", "last", "first"]):
            return "when"
        if any(kw in q for kw in ["who", "which person", "哪个", "谁", "alice", "bob", "entity"]):
            return "entity"
        return "general"

    def _rrf_anchors(self, q_vec, q_keywords, top_k=5):
        """Reciprocal Rank Fusion: combine vector search + keyword match + recency."""
        scores = {}
        # Vector similarity scores
        for nid, node in self.nodes.items():
            if node.graph_type == "entity": continue
            sim = cosine_sim(q_vec, node.vector)
            scores[nid] = scores.get(nid, 0) + sim

        # Keyword match bonus
        for nid, node in self.nodes.items():
            if node.graph_type == "entity": continue
            content_kw = extract_keywords(node.content)
            overlap = len(set(q_keywords) & set(content_kw))
            if overlap > 0:
                scores[nid] = scores.get(nid, 0) + overlap * 0.5

        # Recency bonus (recent events ranked higher)
        sorted_ids = sorted(self.event_timeline, key=lambda x: self.nodes[x].timestamp, reverse=True)
        for rank, nid in enumerate(sorted_ids):
            if nid in scores:
                scores[nid] += 1.0 / (self.RRF_K + rank + 1)

        # Return top-K
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked[:top_k]]

    def _beam_search(self, anchors, q_vec, intent):
        """Adaptive Beam Search across four graphs."""
        visited = set(anchors)
        frontier = list(anchors)
        traversal_path = []

        # Intent-specific edge weights
        intent_weights = {
            "why": {"causal": 3.0, "temporal": 1.0, "semantic": 1.5, "entity_ref": 0.5},
            "when": {"temporal": 3.0, "causal": 0.5, "semantic": 1.0, "entity_ref": 0.5},
            "entity": {"entity_ref": 3.0, "semantic": 2.0, "causal": 1.0, "temporal": 0.5},
            "general": {"semantic": 2.0, "temporal": 1.5, "causal": 1.5, "entity_ref": 1.0},
        }
        w_map = intent_weights.get(intent, intent_weights["general"])

        for depth in range(self.MAX_DEPTH):
            candidates = []
            for uid in frontier:
                node_u = self.nodes.get(uid)
                if not node_u: continue
                for etype in self.edges:
                    for e in self.edges[etype]:
                        neighbor_id = e.target_id if e.source_id == uid else (e.source_id if e.target_id == uid else None)
                        if neighbor_id and neighbor_id not in visited and neighbor_id in self.nodes:
                            neighbor = self.nodes[neighbor_id]
                            # Transition score: structural alignment + semantic affinity
                            struct_score = w_map.get(etype, 1.0)
                            sem_score = cosine_sim(neighbor.vector, q_vec)
                            score = math.exp(self.LAMBDA_1 * math.log(max(struct_score, 0.01)) + self.LAMBDA_2 * sem_score)
                            candidates.append((neighbor_id, score, etype))

            if not candidates: break

            # Keep top beam width
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_frontier = [c[0] for c in candidates[:self.BEAM_WIDTH]]
            for cid, cscore, ctype in candidates[:self.BEAM_WIDTH]:
                traversal_path.append({"from": uid[:12] if frontier else "anchor", "to": cid[:12],
                                       "via": ctype, "score": round(cscore, 3)})
                visited.add(cid)
            frontier = new_frontier
            if len(visited) >= 20: break

        return {"visited": visited, "traversal_path": traversal_path, "depth": depth + 1,
                "total_visited": len(visited)}

    def _topological_sort(self, node_ids, intent):
        """Sort retrieved nodes based on intent type."""
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        if intent == "when":
            return sorted(nodes, key=lambda n: n.timestamp)
        elif intent == "why":
            # Topological sort by causal edges
            in_degree = {n.node_id: 0 for n in nodes}
            for e in self.edges["causal"]:
                if e.source_id in in_degree and e.target_id in in_degree:
                    in_degree[e.target_id] += 1
            result = [n for n in nodes if in_degree.get(n.node_id, 0) == 0]
            return result if result else nodes
        else:
            return sorted(nodes, key=lambda n: -cosine_sim(n.vector, generate_embedding("query")))

    def _synthesize_context(self, nodes, intent):
        """Generate structured context string for LLM prompt injection."""
        lines = []
        for i, node in enumerate(nodes):
            if node.graph_type == "entity":
                lines.append(f"[实体: {node.content}]")
            else:
                ts = node.timestamp.split("T")[0] if "T" in node.timestamp else node.timestamp
                lines.append(f"<t:{ts}> {node.content} <ref:{node.node_id[:12]}>")
        return "\n".join(lines)

    def get_stats(self):
        return {
            "memory_nodes": sum(1 for n in self.nodes.values() if n.graph_type == "memory_event"),
            "entity_nodes": len(self.entity_nodes),
            "semantic_edges": len(self.edges["semantic"]),
            "temporal_edges": len(self.edges["temporal"]),
            "causal_edges": len(self.edges["causal"]),
            "entity_ref_edges": len(self.edges["entity_ref"]),
            "total_pipeline_steps": self.step_counter,
            "pipeline_log_size": len(self.pipeline_log),
        }


# ============================================================
# Global State
# ============================================================
memory = MAGMAAgentMemory()

# Seed with demo events
DEMO_EVENTS = [
    "Alice reported a login authentication failure on the main application",
    "Investigation found race condition in session token generation",
    "CPU usage spiked to 95% on the authentication server cluster",
    "Database slow queries identified due to missing composite index",
    "Bob deployed a hotfix adding the missing database index",
    "CPU returned to normal levels after the database index fix",
    "Carol updated OAuth2 token validation to prevent session conflicts",
    "Alice confirmed the login authentication issue is fully resolved",
    "Post-mortem review identified three root causes in auth pipeline",
    "Monitoring alerts reconfigured with lower thresholds for CPU spikes",
]

for i, evt in enumerate(DEMO_EVENTS):
    ts = (datetime(2026, 6, 10, 10, 0) + timedelta(hours=i)).isoformat()[:19]
    memory.agent_observe(evt, {"source": "demo", "seq": i, "timestamp": ts})


# ============================================================
# HTML Template
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>MAGMA Agent 对接 + 四图构建</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.0/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.2/dagre.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-dagre/2.5.0/cytoscape-dagre.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; min-height: 100vh; }
.header { padding: 16px 24px; border-bottom: 1px solid #21262d; display: flex; align-items: center; gap: 12px; background: #161b22; }
.header h1 { font-size: 18px; color: #58a6ff; }
.badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; background: #23863633; color: #3fb950; }
.tabs { display: flex; padding: 0 24px; border-bottom: 1px solid #21262d; background: #161b22; gap: 0; }
.tab { padding: 10px 20px; cursor: pointer; color: #8b949e; font-size: 13px; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.tab:hover { color: #c9d1d9; }
.panel { display: none; padding: 20px 24px; }
.panel.active { display: flex; gap: 20px; }

/* Left-right layout */
.col-left { flex: 1; min-width: 0; }
.col-right { width: 380px; flex-shrink: 0; }

/* Cards */
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
.card h3 { font-size: 14px; color: #58a6ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
.card h3 .icon { font-size: 16px; }

/* JSON display */
.json-block { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-family: 'SF Mono', Monaco, monospace; font-size: 11px; line-height: 1.6; overflow: auto; max-height: 300px; white-space: pre-wrap; color: #8b949e; }
.json-block .key { color: #79c0ff; }
.json-block .string { color: #a5d6ff; }
.json-block .number { color: #d2a8ff; }
.json-block .bool { color: #ff7b72; }

/* Input */
.input-group { display: flex; gap: 8px; margin-bottom: 12px; }
.input-group input { flex: 1; padding: 10px 14px; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: #c9d1d9; font-size: 13px; outline: none; }
.input-group input:focus { border-color: #58a6ff; }
.btn { padding: 8px 16px; border-radius: 6px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; font-size: 12px; cursor: pointer; transition: all 0.15s; }
.btn:hover { background: #30363d; border-color: #58a6ff; }
.btn-primary { background: #238636; border-color: #238636; color: #fff; }
.btn-primary:hover { background: #2ea043; }

/* Pipeline steps */
.pipeline-step { display: flex; gap: 12px; margin-bottom: 10px; position: relative; }
.pipeline-dot { width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; }
.pipeline-dot.fast { background: #81C78433; color: #81C784; border: 2px solid #81C784; }
.pipeline-dot.slow { background: #FFB74D33; color: #FFB74D; border: 2px solid #FFB74D; }
.pipeline-dot.query { background: #4FC3F733; color: #4FC3F7; border: 2px solid #4FC3F7; }
.pipeline-content { flex: 1; }
.pipeline-label { font-size: 11px; color: #8b949e; margin-bottom: 2px; }
.pipeline-detail { font-size: 12px; color: #c9d1d9; }

/* Graph container */
#cy { width: 100%; height: 420px; border-radius: 8px; border: 1px solid #30363d; background: #0d1117; }

/* Filter buttons */
.filter-row { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.filter-btn { padding: 4px 10px; border-radius: 12px; border: 1px solid #30363d; background: transparent; color: #8b949e; font-size: 11px; cursor: pointer; display: flex; align-items: center; gap: 4px; }
.filter-btn.active { border-color: currentColor; background: currentColor22; }
.filter-btn .dot { width: 8px; height: 8px; border-radius: 50%; }

/* Stats bar */
.stats-bar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.stat-chip { padding: 4px 10px; border-radius: 12px; background: #21262d; border: 1px solid #30363d; font-size: 11px; }

/* Event list */
.event-item { padding: 8px 12px; border-left: 3px solid #30363d; margin-bottom: 6px; font-size: 12px; color: #c9d1d9; cursor: pointer; transition: all 0.15s; }
.event-item:hover { background: #21262d; border-left-color: #58a6ff; }
.event-item .time { color: #8b949e; font-size: 10px; }

/* Flow diagram */
.flow-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; text-align: center; font-size: 12px; min-width: 140px; }
.flow-arrow { display: flex; align-items: center; color: #484f58; font-size: 18px; padding: 0 6px; }
.flow-row { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.flow-label { font-size: 10px; color: #8b949e; text-align: center; margin-top: 4px; }

/* Result card */
.result-item { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; margin-bottom: 8px; }
.result-item .content { font-size: 12px; color: #c9d1d9; margin-bottom: 4px; }
.result-item .meta { font-size: 10px; color: #8b949e; }

/* Highlight colors */
.hl-fast { color: #81C784; }
.hl-slow { color: #FFB74D; }
.hl-query { color: #4FC3F7; }
.hl-entity { color: #CE93D8; }
.hl-causal { color: #FFB74D; }

.intent-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.intent-why { background: #FFB74D33; color: #FFB74D; }
.intent-when { background: #81C78433; color: #81C784; }
.intent-entity { background: #CE93D833; color: #CE93D8; }
.intent-general { background: #4FC3F733; color: #4FC3F7; }

.code-inline { background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 1px 6px; font-family: 'SF Mono', Monaco, monospace; font-size: 11px; color: #79c0ff; }
</style>
</head>
<body>

<div class="header">
  <h1>MAGMA Agent 对接 + 四图构建链路</h1>
  <span class="badge">ACL 2026</span>
  <span class="badge">arXiv:2601.03236</span>
</div>

<div class="tabs">
  <div class="tab active" data-panel="agent-api">Agent 接口协议</div>
  <div class="tab" data-panel="pipeline">四图构建链路</div>
  <div class="tab" data-panel="query-pipeline">查询链路</div>
  <div class="tab" data-panel="live-graph">实时图谱</div>
</div>

<!-- Panel 1: Agent API Contract -->
<div class="panel active" id="panel-agent-api">
  <div class="col-left">
    <div class="card">
      <h3><span class="icon">📥</span> Agent → Memory：写入事件</h3>
      <p style="font-size:12px;color:#8b949e;margin-bottom:10px">
        Agent 在每轮交互后调用 <span class="code-inline">memory.observe()</span> 写入事件到记忆系统。
        触发双流管道：<span class="hl-fast">Fast Path</span>（同步，延迟 <25ms）+ <span class="hl-slow">Slow Path</span>（异步，后台推理）。
      </p>
      <div class="input-group">
        <input id="eventInput" placeholder="输入 Agent 事件（如：Alice 修复了认证漏洞）" value="">
        <button class="btn btn-primary" id="sendEventBtn">写入事件</button>
      </div>
      <div class="json-block" id="writeResult">// 点击"写入事件"查看 Agent → Memory 的请求/响应</div>
    </div>

    <div class="card">
      <h3><span class="icon">📤</span> Agent → Memory：查询记忆</h3>
      <p style="font-size:12px;color:#8b949e;margin-bottom:10px">
        Agent 在生成回复前调用 <span class="code-inline">memory.query()</span> 获取相关记忆上下文。
        触发四阶段管线：<span class="hl-query">意图路由</span> → <span class="hl-query">RRF 锚定</span> → <span class="hl-query">Beam Search</span> → <span class="hl-query">上下文综合</span>。
      </p>
      <div class="input-group">
        <input id="queryInput" placeholder="输入 Agent 查询（如：为什么 CPU 飙升了？）" value="为什么 CPU 飙升了？">
        <button class="btn btn-primary" id="queryBtn">查询记忆</button>
      </div>
      <div class="json-block" id="queryResult">// 点击"查询记忆"查看 Memory → Agent 的返回</div>
    </div>

    <div class="card">
      <h3><span class="icon">🔄</span> 完整交互循环</h3>
      <div class="flow-row">
        <div class="flow-box">
          <strong>AI Agent</strong>
          <div class="flow-label">感知环境、决策、行动</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-box" style="border-color:#81C784">
          <strong class="hl-fast">memory.observe()</strong>
          <div class="flow-label">Fast: 时间链 + 向量索引<br>Slow: 因果/语义/实体推理</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-box" style="border-color:#FFB74D">
          <strong class="hl-slow">Memory Store</strong>
          <div class="flow-label">四张正交图<br>+ 向量索引</div>
        </div>
      </div>
      <div class="flow-row">
        <div class="flow-box" style="border-color:#4FC3F7">
          <strong class="hl-query">memory.query(q)</strong>
          <div class="flow-label">意图路由 → RRF → Beam Search<br>→ 拓扑排序 → 上下文综合</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-box" style="border-color:#CE93D8">
          <strong class="hl-entity">Context String</strong>
          <div class="flow-label">&lt;t:2026-06-01&gt; 事件内容 &lt;ref:id&gt;<br>结构化记忆上下文</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-box">
          <strong>LLM Prompt</strong>
          <div class="flow-label">context + query → 回复</div>
        </div>
      </div>
    </div>
  </div>

  <div class="col-right">
    <div class="card">
      <h3>📊 系统状态</h3>
      <div class="stats-bar" id="sysStats"></div>
    </div>
    <div class="card">
      <h3>📝 事件时间线</h3>
      <div id="eventTimeline" style="max-height:400px;overflow-y:auto"></div>
    </div>
    <div class="card">
      <h3>📋 API 规格</h3>
      <div style="font-size:11px;color:#8b949e;line-height:1.8">
        <div><strong style="color:#81C784">POST /api/observe</strong></div>
        <div style="padding-left:12px">请求: <span class="code-inline">{"interaction": "...", "metadata": {...}}</span></div>
        <div style="padding-left:12px">返回: <span class="code-inline">{"node_id": "...", "fast_path_edges": [...], "slow_path_queued": true}</span></div>
        <br>
        <div><strong style="color:#4FC3F7">POST /api/query</strong></div>
        <div style="padding-left:12px">请求: <span class="code-inline">{"query": "..."}</span></div>
        <div style="padding-left:12px">返回: <span class="code-inline">{"intent": "why", "context": "...", "nodes_used": 5, "traversal_path": [...]}</span></div>
        <br>
        <div><strong style="color:#58a6ff">GET /api/pipeline</strong></div>
        <div style="padding-left:12px">返回: <span class="code-inline">{"pipeline_log": [...], "stats": {...}}</span></div>
        <br>
        <div style="color:#8b949e;font-size:10px">
          对接 HugeGraph MCP Server：<br>
          Tool: <span class="code-inline">magma_observe</span>, <span class="code-inline">magma_query</span>, <span class="code-inline">magma_get_graph</span><br>
          Resource: <span class="code-inline">magma://stats</span>, <span class="code-inline">magma://schema</span>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Panel 2: Four-Graph Construction Pipeline -->
<div class="panel" id="panel-pipeline">
  <div class="col-left">
    <div class="card">
      <h3><span class="icon">⚡</span> Fast Path：同步写入（延迟 <25ms）</h3>
      <p style="font-size:12px;color:#8b949e;margin-bottom:12px">
        Agent 每次交互后立即执行，不阻塞 Agent 决策。三步操作：
      </p>
      <div id="fastPathSteps"></div>
    </div>

    <div class="card">
      <h3><span class="icon">🧠</span> Slow Path：异步推理（后台 Worker）</h3>
      <p style="font-size:12px;color:#8b949e;margin-bottom:12px">
        Fast Path 完成后将事件 ID 入队，后台 Worker 取出后调用 LLM 推理三张辅助图：
      </p>
      <div id="slowPathSteps"></div>
    </div>

    <div class="card">
      <h3><span class="icon">📈</span> 最近写入的管道日志</h3>
      <div id="pipelineLog" style="max-height:300px;overflow-y:auto;font-size:12px"></div>
    </div>
  </div>

  <div class="col-right">
    <div class="card">
      <h3>🏗️ 四图构建架构</h3>
      <svg viewBox="0 0 360 500" style="width:100%">
        <!-- Input -->
        <rect x="80" y="10" width="200" height="36" rx="8" fill="#161b22" stroke="#58a6ff"/>
        <text x="180" y="33" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="600">Agent 事件输入</text>

        <!-- Fast Path -->
        <line x1="180" y1="46" x2="180" y2="66" stroke="#81C784" stroke-width="2" marker-end="url(#arrowG)"/>
        <rect x="60" y="66" width="240" height="90" rx="8" fill="#0d1117" stroke="#81C784"/>
        <text x="180" y="85" text-anchor="middle" fill="#81C784" font-size="11" font-weight="600">⚡ Fast Path（同步）</text>
        <text x="180" y="105" text-anchor="middle" fill="#8b949e" font-size="10">1. SegmentEvent: 分割 + 结构化</text>
        <text x="180" y="120" text-anchor="middle" fill="#8b949e" font-size="10">2. Encoder → 向量索引 (VDB)</text>
        <text x="180" y="135" text-anchor="middle" fill="#8b949e" font-size="10">3. 追加时间链 n_{t-1} → n_t</text>
        <text x="180" y="148" text-anchor="middle" fill="#8b949e" font-size="10">4. 入队 → 触发 Slow Path</text>

        <!-- Slow Path -->
        <line x1="180" y1="156" x2="180" y2="176" stroke="#FFB74D" stroke-width="2"/>
        <rect x="60" y="176" width="240" height="100" rx="8" fill="#0d1117" stroke="#FFB74D"/>
        <text x="180" y="195" text-anchor="middle" fill="#FFB74D" font-size="11" font-weight="600">🧠 Slow Path（异步 LLM）</text>
        <text x="180" y="215" text-anchor="middle" fill="#8b949e" font-size="10">1. GetNeighborhood(n_t, hops=2)</text>
        <text x="180" y="230" text-anchor="middle" fill="#8b949e" font-size="10">2. Prompt = Format(N_local)</text>
        <text x="180" y="245" text-anchor="middle" fill="#8b949e" font-size="10">3. E_new = Φ_LLM(Prompt)</text>
        <text x="180" y="260" text-anchor="middle" fill="#8b949e" font-size="10">4. G.AddEdges(E_new)</text>
        <text x="180" y="272" text-anchor="middle" fill="#8b949e" font-size="10">推理: 因果边 + 实体抽取</text>

        <!-- Four Graphs -->
        <line x1="180" y1="276" x2="180" y2="300" stroke="#484f58" stroke-width="2"/>

        <rect x="10" y="300" width="80" height="55" rx="6" fill="#0d1117" stroke="#81C784"/>
        <text x="50" y="318" text-anchor="middle" fill="#81C784" font-size="10" font-weight="600">时间图</text>
        <text x="50" y="335" text-anchor="middle" fill="#8b949e" font-size="9">Fast Path</text>
        <text x="50" y="348" text-anchor="middle" fill="#8b949e" font-size="9">不可变链</text>

        <rect x="100" y="300" width="80" height="55" rx="6" fill="#0d1117" stroke="#4FC3F7"/>
        <text x="140" y="318" text-anchor="middle" fill="#4FC3F7" font-size="10" font-weight="600">语义图</text>
        <text x="140" y="335" text-anchor="middle" fill="#8b949e" font-size="9">Slow Path</text>
        <text x="140" y="348" text-anchor="middle" fill="#8b949e" font-size="9">cos(θ) > 阈值</text>

        <rect x="190" y="300" width="80" height="55" rx="6" fill="#0d1117" stroke="#FFB74D"/>
        <text x="230" y="318" text-anchor="middle" fill="#FFB74D" font-size="10" font-weight="600">因果图</text>
        <text x="230" y="335" text-anchor="middle" fill="#8b949e" font-size="9">Slow Path</text>
        <text x="230" y="348" text-anchor="middle" fill="#8b949e" font-size="9">LLM 推理</text>

        <rect x="280" y="300" width="70" height="55" rx="6" fill="#0d1117" stroke="#CE93D8"/>
        <text x="315" y="318" text-anchor="middle" fill="#CE93D8" font-size="10" font-weight="600">实体图</text>
        <text x="315" y="335" text-anchor="middle" fill="#8b949e" font-size="9">Slow Path</text>
        <text x="315" y="348" text-anchor="middle" fill="#8b949e" font-size="9">NER 抽取</text>

        <!-- HugeGraph -->
        <rect x="60" y="380" width="240" height="45" rx="8" fill="#21262d" stroke="#da3633"/>
        <text x="180" y="405" text-anchor="middle" fill="#da3633" font-size="12" font-weight="600">HugeGraph 原生图存储</text>
        <text x="180" y="420" text-anchor="middle" fill="#8b949e" font-size="10">一图四视图 · Gremlin 统一查询 · OLAP 60亿</text>

        <!-- MCP -->
        <rect x="100" y="445" width="160" height="35" rx="6" fill="#161b22" stroke="#58a6ff"/>
        <text x="180" y="467" text-anchor="middle" fill="#58a6ff" font-size="11">MCP Server: 10 Tools</text>

        <defs><marker id="arrowG" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#81C784"/></marker></defs>
      </svg>
    </div>
  </div>
</div>

<!-- Panel 3: Query Pipeline -->
<div class="panel" id="panel-query-pipeline">
  <div class="col-left">
    <div class="card">
      <h3><span class="icon">🔍</span> 四阶段查询管线</h3>
      <p style="font-size:12px;color:#8b949e;margin-bottom:12px">
        查询"为什么 CPU 飙升了？"的四阶段处理过程。每阶段输入→输出清晰可见。
      </p>
      <div id="queryPipelineSteps"></div>
    </div>
    <div class="card">
      <h3><span class="icon">📊</span> Beam Search 遍历路径</h3>
      <div id="beamSearchPath" style="font-size:12px"></div>
    </div>
  </div>
  <div class="col-right">
    <div class="card">
      <h3>📐 超参数配置（论文 Table 6）</h3>
      <div style="font-size:11px;line-height:2">
        <table style="width:100%;color:#8b949e">
          <tr><td style="color:#79c0ff">嵌入模型</td><td>all-MiniLM-L6-v2 (384d)</td></tr>
          <tr><td style="color:#79c0ff">LLM</td><td>gpt-4o-mini (temp=0.0)</td></tr>
          <tr><td style="color:#79c0ff">RRF K</td><td>60</td></tr>
          <tr><td style="color:#79c0ff">Max Depth</td><td>5 hops</td></tr>
          <tr><td style="color:#79c0ff">Beam Width</td><td>5</td></tr>
          <tr><td style="color:#79c0ff">λ₁ 结构系数</td><td>1.0</td></tr>
          <tr><td style="color:#79c0ff">λ₂ 语义系数</td><td>0.3–0.7</td></tr>
          <tr><td style="color:#79c0ff">w_causal</td><td>3.0–5.0</td></tr>
          <tr><td style="color:#79c0ff">w_temporal</td><td>0.5–4.0</td></tr>
          <tr><td style="color:#79c0ff">查询延迟</td><td>1.47s（最快）</td></tr>
          <tr><td style="color:#79c0ff">Token 节省</td><td>>95%（vs Full Context）</td></tr>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>🏆 评测结果 (LoCoMo)</h3>
      <div style="font-size:11px;line-height:2">
        <table style="width:100%;color:#8b949e">
          <tr><td></td><td>Nemori</td><td style="color:#3fb950;font-weight:600">MAGMA</td></tr>
          <tr><td>Multi-Hop</td><td>0.569</td><td style="color:#3fb950">0.528</td></tr>
          <tr><td>Temporal</td><td>0.649</td><td style="color:#3fb950;font-weight:700">0.650</td></tr>
          <tr><td>Open-Domain</td><td>0.485</td><td style="color:#3fb950">0.517</td></tr>
          <tr><td>Single-Hop</td><td>0.764</td><td style="color:#3fb950;font-weight:700">0.776</td></tr>
          <tr><td>Adversarial</td><td>0.616</td><td style="color:#3fb950;font-weight:700">0.742</td></tr>
          <tr style="border-top:1px solid #30363d"><td style="font-weight:600">Overall</td><td>0.590</td><td style="color:#3fb950;font-weight:700;font-size:14px">0.700 (+18.6%)</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- Panel 4: Live Graph -->
<div class="panel" id="panel-live-graph">
  <div class="col-left">
    <div class="filter-row">
      <button class="filter-btn active" data-filter="all"><span class="dot" style="background:#58a6ff"></span>全部</button>
      <button class="filter-btn" data-filter="semantic"><span class="dot" style="background:#4FC3F7"></span>语义图</button>
      <button class="filter-btn" data-filter="temporal"><span class="dot" style="background:#81C784"></span>时间图</button>
      <button class="filter-btn" data-filter="causal"><span class="dot" style="background:#FFB74D"></span>因果图</button>
      <button class="filter-btn" data-filter="entity_ref"><span class="dot" style="background:#CE93D8"></span>实体引用</button>
    </div>
    <div id="cy"></div>
  </div>
  <div class="col-right">
    <div class="card">
      <h3>📊 图统计</h3>
      <div class="stats-bar" id="graphStats"></div>
    </div>
    <div class="card">
      <h3>ℹ️ 节点详情</h3>
      <div id="nodeDetail" style="font-size:12px;color:#8b949e">点击节点查看</div>
    </div>
  </div>
</div>

<script>
const STATS = __STATS__;
const PIPELINE_LOG = __PIPELINE_LOG__;
const GRAPH_DATA = __GRAPH_DATA__;

// ---- Init Stats ----
function renderStats() {
  const sb = document.getElementById('sysStats');
  if (!sb) return;
  const s = STATS;
  sb.innerHTML = [
    `<div class="stat-chip" style="color:#81C784">时间边: ${s.temporal_edges || 0}</div>`,
    `<div class="stat-chip" style="color:#4FC3F7">语义边: ${s.semantic_edges || 0}</div>`,
    `<div class="stat-chip" style="color:#FFB74D">因果边: ${s.causal_edges || 0}</div>`,
    `<div class="stat-chip" style="color:#CE93D8">实体引用: ${s.entity_ref_edges || 0}</div>`,
    `<div class="stat-chip">记忆: ${s.memory_nodes || 0}</div>`,
    `<div class="stat-chip">实体: ${s.entity_nodes || 0}</div>`,
  ].join('');
}
renderStats();

// ---- Init Event Timeline ----
function renderTimeline() {
  const el = document.getElementById('eventTimeline');
  if (!el) return;
  el.innerHTML = PIPELINE_LOG.map(log => {
    if (!log.fast) return '';
    return `<div class="event-item"><div class="time">${log.fast.timestamp || ''}</div>${log.fast.content_preview || log.fast.node_id || ''}</div>`;
  }).join('');
}
renderTimeline();

// ---- Init Pipeline Steps ----
function renderPipelineSteps() {
  const fastEl = document.getElementById('fastPathSteps');
  const slowEl = document.getElementById('slowPathSteps');
  const logEl = document.getElementById('pipelineLog');
  if (!fastEl) return;

  // Show latest write log if available
  const latest = PIPELINE_LOG.length > 0 ? PIPELINE_LOG[PIPELINE_LOG.length - 1] : null;

  // Fast Path
  fastEl.innerHTML = `
    <div class="pipeline-step">
      <div class="pipeline-dot fast">1</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Event Segmentation (事件分割)</div>
        <div class="pipeline-detail">${latest ? `输入: "${latest.fast.content_preview}"` : '等待事件输入...'}</div>
        <div class="pipeline-detail">输出: <span class="code-inline">MemoryNode(node_id, content, timestamp, vector, attributes)</span></div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot fast">2</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Vector Encoding (向量编码)</div>
        <div class="pipeline-detail">Encoder: all-MiniLM-L6-v2 → 384d 稠密向量</div>
        <div class="pipeline-detail">存储到向量数据库 (VDB.Add)</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot fast">3</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Temporal Chain Append (追加时间链)</div>
        <div class="pipeline-detail">${latest ? `n_{t-1} (${latest.fast.temporal_edges && latest.fast.temporal_edges[0] ? latest.fast.temporal_edges[0].from : '?'}) → n_t (${latest.fast.node_id ? latest.fast.node_id.substring(0,12) : '?'})` : 'n_{t-1} → n_t'}</div>
        <div class="pipeline-detail">不可变，永不删除</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot fast">4</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Queue → Trigger Slow Path (入队触发慢路径)</div>
        <div class="pipeline-detail">异步 Worker 从队列取出 n_t.id</div>
      </div>
    </div>
  `;

  // Slow Path
  slowEl.innerHTML = `
    <div class="pipeline-step">
      <div class="pipeline-dot slow">1</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Get Neighborhood (获取2跳邻域)</div>
        <div class="pipeline-detail">${latest ? `邻域大小: ${latest.slow ? latest.slow.neighborhood_size : '?'} 个节点` : 'N(n_t, hops=2)'}</div>
        <div class="pipeline-detail">收集时间链前后各2跳的事件</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot slow">2</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Semantic Edges (语义边生成)</div>
        <div class="pipeline-detail">${latest && latest.slow ? `发现 ${latest.slow.semantic_edges_found} 条语义边 (cos(θ) > 0.5)` : 'cos(v_i, v_j) > threshold → 无向加权边'}</div>
        <div class="pipeline-detail" style="color:#4FC3F7">基于嵌入相似度，不需要 LLM</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot slow">3</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Causal Edges (因果边推理) ⚡ LLM</div>
        <div class="pipeline-detail">${latest && latest.slow ? `推理出 ${latest.slow.causal_edges_found} 条因果边` : 'Φ_LLM(N_local) → 因果有向边'}</div>
        <div class="pipeline-detail" style="color:#FFB74D">LLM 分析邻域事件间的因果关系</div>
        ${latest && latest.slow && latest.slow.causal_details ? latest.slow.causal_details.map(c => `<div class="pipeline-detail" style="padding-left:12px">${c.from} → ${c.to} (${c.reason})</div>`).join('') : ''}
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot slow">4</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Entity Extraction + Linking (实体抽取链接)</div>
        <div class="pipeline-detail">${latest && latest.slow ? `抽取 ${latest.slow.entity_edges_found} 个实体引用` : 'NER → Entity 节点 + entity_ref 边'}</div>
        ${latest && latest.slow && latest.slow.entity_details ? latest.slow.entity_details.map(e => `<div class="pipeline-detail" style="color:#CE93D8;padding-left:12px">→ [${e.entity}]</div>`).join('') : ''}
      </div>
    </div>
  `;

  // Log
  if (logEl) {
    logEl.innerHTML = PIPELINE_LOG.slice().reverse().slice(0, 5).map((log, i) => {
      const f = log.fast || {};
      const s = log.slow || {};
      const isQuery = !f.phase;
      return `<div class="result-item">
        <div class="meta">Step #${f.step || s.step || '?'} | ${isQuery ? 'QUERY' : f.phase || ''} ${s.phase ? '+ ' + s.phase : ''}</div>
        ${isQuery ? `<div class="content">${log.query ? log.query.query : ''} → 意图: ${log.query ? log.query.intent : ''}</div>` : ''}
        ${!isQuery && f.content_preview ? `<div class="content">${f.content_preview}</div>` : ''}
        ${!isQuery && s.semantic_edges_found !== undefined ? `<div class="meta">语义:${s.semantic_edges_found} 因果:${s.causal_edges_found} 实体:${s.entity_edges_found}</div>` : ''}
      </div>`;
    }).join('') || '<div style="color:#484f58">暂无管道日志</div>';
  }
}
renderPipelineSteps();

// ---- Query Pipeline Steps ----
function renderQueryPipeline() {
  const el = document.getElementById('queryPipelineSteps');
  if (!el) return;
  const lastQuery = PIPELINE_LOG.slice().reverse().find(l => l.query);
  const q = lastQuery ? lastQuery.query : {intent: 'why', intent_label: '原因', nodes_used: 10, token_estimate: 1200, anchors: ['mem_xxx'], traversal_path: []};

  el.innerHTML = `
    <div class="pipeline-step">
      <div class="pipeline-dot query">1</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Intent Classification (意图分类)</div>
        <div class="pipeline-detail">输入: 原始查询文本</div>
        <div class="pipeline-detail">输出: <span class="intent-badge intent-${q.intent}">${q.intent_label} (${q.intent})</span></div>
        <div class="pipeline-detail" style="color:#8b949e">轻量分类器: Why → 因果图 / When → 时间图 / Entity → 实体图</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot query">2</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Multi-Signal Anchor (RRF 锚定)</div>
        <div class="pipeline-detail">三信号融合: 向量搜索 + 关键词匹配 + 时间衰减</div>
        <div class="pipeline-detail">锚点: ${(q.anchors || []).map(a => `<span class="code-inline">${a}</span>`).join(' ')}</div>
        <div class="pipeline-detail" style="color:#8b949e">S = Σ 1/(K + r_m(n)), K=60</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot query">3</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Adaptive Beam Search (自适应束搜索)</div>
        <div class="pipeline-detail">S(n_j|n_i,q) = exp(λ₁·φ(type,T_q) + λ₂·sim(n_j,q))</div>
        <div class="pipeline-detail">深度: ${q.beam_search_depth || '?'}/5, 访问: ${q.total_visited || '?'} 节点, Beam Width: 5</div>
        <div class="pipeline-detail" style="color:#8b949e">意图感知权重: "${q.intent}" 类型边权重 ×3.0</div>
      </div>
    </div>
    <div class="pipeline-step">
      <div class="pipeline-dot query">4</div>
      <div class="pipeline-content">
        <div class="pipeline-label">Context Synthesis (上下文综合)</div>
        <div class="pipeline-detail">${q.intent === 'when' ? '按时间戳排序' : q.intent === 'why' ? '因果拓扑排序（原因先于结果）' : '按语义相关性排序'}</div>
        <div class="pipeline-detail">Token 预算: ${q.token_estimate || '?'} tokens（vs Full Context 8.5K）</div>
        <div class="pipeline-detail">格式: &lt;t:2026-06-01&gt; 内容 &lt;ref:node_id&gt;</div>
      </div>
    </div>
  `;

  // Beam search path
  const pathEl = document.getElementById('beamSearchPath');
  if (pathEl && q.traversal_path) {
    pathEl.innerHTML = q.traversal_path.slice(0, 15).map(p => {
      const colors = {semantic:'#4FC3F7', temporal:'#81C784', causal:'#FFB74D', entity_ref:'#CE93D8'};
      return `<div class="pipeline-detail" style="padding:2px 0">
        ${p.from || 'anchor'} <span style="color:${colors[p.via] || '#8b949e'}">→[${p.via}]→</span> ${p.to} <span style="color:#484f58">score=${p.score}</span>
      </div>`;
    }).join('') || '<div style="color:#484f58">执行查询查看遍历路径</div>';
  }
}
renderQueryPipeline();

// ---- Cytoscape Graph ----
const edgeColors = { semantic: '#4FC3F7', temporal: '#81C784', causal: '#FFB74D', entity_ref: '#CE93D8' };

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: GRAPH_DATA,
  style: [
    { selector: 'node', style: { 'label': 'data(label)', 'font-size': '10px', 'color': '#c9d1d9', 'text-valign': 'center', 'text-halign': 'center', 'width': 35, 'height': 35, 'text-wrap': 'wrap', 'text-max-width': '80px' } },
    { selector: 'node.memory_event', style: { 'background-color': '#1A73E8', 'shape': 'ellipse' } },
    { selector: 'node.entity', style: { 'background-color': '#FF6D00', 'shape': 'diamond', 'width': 30, 'height': 30, 'font-size': '10px' } },
    { selector: 'edge', style: { 'width': 2, 'line-color': '#484f58', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#484f58', 'opacity': 0.6, 'curve-style': 'bezier' } },
    { selector: 'edge[type="semantic"]', style: { 'line-color': '#4FC3F7', 'target-arrow-color': '#4FC3F7', 'target-arrow-shape': 'none', 'line-style': 'dashed' } },
    { selector: 'edge[type="temporal"]', style: { 'line-color': '#81C784', 'target-arrow-color': '#81C784' } },
    { selector: 'edge[type="causal"]', style: { 'line-color': '#FFB74D', 'target-arrow-color': '#FFB74D', 'width': 3 } },
    { selector: 'edge[type="entity_ref"]', style: { 'line-color': '#CE93D8', 'target-arrow-color': '#CE93D8' } },
  ],
  layout: { name: 'dagre', rankDir: 'TB', spacingFactor: 0.8, padding: 20 },
});

// Graph stats
function renderGraphStats() {
  const el = document.getElementById('graphStats');
  if (!el) return;
  const nodes = cy.nodes();
  const edges = cy.edges();
  const types = {semantic:0, temporal:0, causal:0, entity_ref:0};
  edges.forEach(e => { if (types[e.data('type')] !== undefined) types[e.data('type')]++; });
  el.innerHTML = [
    `<div class="stat-chip">节点: ${nodes.length}</div>`,
    `<div class="stat-chip" style="color:#81C784">时间: ${types.temporal}</div>`,
    `<div class="stat-chip" style="color:#4FC3F7">语义: ${types.semantic}</div>`,
    `<div class="stat-chip" style="color:#FFB74D">因果: ${types.causal}</div>`,
    `<div class="stat-chip" style="color:#CE93D8">实体: ${types.entity_ref}</div>`,
  ].join('');
}
renderGraphStats();

// Node click
cy.on('tap', 'node', e => {
  const n = e.target;
  document.getElementById('nodeDetail').innerHTML =
    `<strong style="color:#58a6ff">${n.data('label')}</strong><br><span style="color:#8b949e">${n.data('full_content') || ''}</span><br><span style="color:#484f58">${n.data('timestamp') || ''}</span>`;
});

// Filter buttons
document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn[data-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const f = btn.dataset.filter;
    cy.edges().forEach(e => {
      if (f === 'all' || e.data('type') === f) { e.removeClass('dimmed'); e.style({'opacity': 0.6}); }
      else { e.addClass('dimmed'); e.style({'opacity': 0.03}); }
    });
  });
});

// ---- Tab switching ----
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.panel).classList.add('active');
    if (tab.dataset.panel === 'live-graph') setTimeout(() => cy.resize(), 100);
  });
});

// ---- Agent API: Send Event ----
document.getElementById('sendEventBtn').addEventListener('click', () => {
  const input = document.getElementById('eventInput');
  const text = input.value.trim();
  if (!text) return;
  fetch('/api/observe', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({interaction: text, metadata: {source: 'web_ui'}})
  }).then(r => r.json()).then(data => {
    const el = document.getElementById('writeResult');
    el.innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
    // Refresh
    setTimeout(() => location.reload(), 800);
  });
});

// ---- Agent API: Query ----
document.getElementById('queryBtn').addEventListener('click', () => {
  const input = document.getElementById('queryInput');
  const text = input.value.trim();
  if (!text) return;
  fetch('/api/query?q=' + encodeURIComponent(text)).then(r => r.json()).then(data => {
    const el = document.getElementById('queryResult');
    el.innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
  });
});

// ---- JSON Syntax Highlight ----
function syntaxHighlight(json) {
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function(match) {
    let cls = 'number';
    if (/^"/.test(match)) { cls = /:$/.test(match) ? 'key' : 'string'; }
    else if (/true|false/.test(match)) { cls = 'bool'; }
    return '<span class="' + cls + '">' + match + '</span>';
  });
}
</script>
</body>
</html>
"""


# ============================================================
# HTTP Handler
# ============================================================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            stats = memory.get_stats()
            # Also add HugeGraph stats if available
            if hg_client.alive():
                hg_stats = hg_client.get_stats()
                stats["backend"] = hg_stats.get("backend", "?")
            html = HTML_TEMPLATE.replace('__STATS__', json.dumps(stats)) \
                                  .replace('__PIPELINE_LOG__', json.dumps(memory.pipeline_log[-20:])) \
                                  .replace('__GRAPH_DATA__', json.dumps(hg_client.get_graph_elements() if hg_client.alive() else {"nodes":[],"edges":[]}))
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/api/stats':
            self._json_response(memory.get_stats())
        elif self.path == '/api/pipeline':
            self._json_response({"pipeline_log": memory.pipeline_log[-20:], "stats": memory.get_stats()})
        elif self.path == '/api/graph':
            if hg_client.alive():
                self._json_response(hg_client.get_graph_elements())
            else:
                self._json_response({"nodes": [], "edges": []})
        elif self.path == '/api/hg-stats':
            self._json_response(hg_client.get_stats())
        elif self.path.startswith('/api/query'):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get('q', [''])[0]
            if query:
                result = memory.agent_query(query)
                self._json_response(result)
            else:
                self._json_response({"error": "missing query"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/observe':
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length else {}
            interaction = body.get('interaction', '')
            metadata = body.get('metadata', {})
            if interaction:
                result = memory.agent_observe(interaction, metadata)
                self._json_response(result)
            else:
                self._json_response({"error": "missing interaction"})
        elif self.path.startswith('/api/query'):
            from urllib.parse import urlparse, parse_qs
            if self.path == '/api/query' and self.headers.get('Content-Length', '0') != '0':
                content_length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length else {}
                query = body.get('query', '')
            else:
                qs = parse_qs(urlparse(self.path).query)
                query = qs.get('q', [''])[0]
            if query:
                result = memory.agent_query(query)
                self._json_response(result)
            else:
                self._json_response({"error": "missing query"})
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5004)
    args = parser.parse_args()
    server = HTTPServer(('0.0.0.0', args.port), Handler)
    print(f"MAGMA Agent Pipeline Visualizer: http://localhost:{args.port}")
    print(f"  Tabs: Agent API | 四图构建 | 查询链路 | 实时图谱")
    server.serve_forever()
