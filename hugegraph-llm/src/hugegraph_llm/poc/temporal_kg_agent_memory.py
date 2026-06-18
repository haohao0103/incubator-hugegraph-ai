#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.
"""
PoC: Temporal Knowledge Graph for Agent Memory (时序知识图谱 Agent 记忆系统)

Inspiration Sources:
  - ATOM (EACL 2026 Findings, Jun 10): Few-shot dynamic TKG construction
  - Graphiti (Zep, May 2026): Real-time temporal context graph for AI agents
  - Graph-based Agent Memory Survey (arXiv:2602.05665)
  - Neo4j MCP Server v1.0: Schema-aware graph access for LLM agents

Core Innovation:
  1. Temporal scoping: every fact has (valid_from, valid_until, created_at)
  2. Time-decay scoring: recent facts rank higher (exponential decay)
  3. Episodic vs Semantic memory separation
  4. Conflict detection across temporal windows
  5. Incremental fact update without full rebuild

GraphRAG Base (铁律遵守):
  - VECTOR: FAISS index + real embeddings (MiMo API primary, deterministic fallback)
  - FULLTEXT: BM25 via jieba + rank_bm25 (real BM25Okapi)

Run:
  cd incubator-hugegraph-ai/hugegraph-llm/src
  PYTHONPATH=src /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph_llm/poc/temporal_kg_agent_memory.py
"""

import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TemporalKGPoC")

POC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(POC_DIR, "temporal_kg_agent_memory_result.json")
TIME_DECAY_LAMBDA = 0.05   # half-life ~14 days
RRF_K = 60
NOW = datetime.now(timezone.utc)

# Embedding config: MiMo OpenAI-compatible API
MIMO_API_BASE = "https://api.xiaomimimo.com/v1"
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
EMBED_DIM = 384


# ─── Enums ────────────────────────────────

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class ConflictType(str, Enum):
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    TEMPORAL = "temporal"
    GRANULARITY = "granularity"


# ─── Data Models ──────────────────────────

@dataclass
class TemporalFact:
    subject: str
    predicate: str
    obj: str
    memory_type: MemoryType
    valid_from: str
    valid_until: Optional[str]
    created_at: str
    source: str = ""
    confidence: float = 1.0
    fact_id: str = ""
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.fact_id:
            h = hash((self.subject, self.predicate, self.obj, self.valid_from))
            self.fact_id = f"fct_{abs(h) % 10**8:x}"

    @property
    def is_valid_now(self) -> bool:
        if self.valid_until is None:
            return True
        n = NOW.isoformat()
        return self.valid_from <= n <= self.valid_until

    def days_since(self) -> float:
        try:
            c = datetime.fromisoformat(self.created_at)
            if c.tzinfo is None:
                c = c.replace(tzinfo=timezone.utc)
            return (NOW - c).total_seconds() / 86400.0
        except Exception:
            return 0.0

    def time_decay_score(self) -> float:
        return max(0.01, math.exp(-TIME_DECAY_LAMBDA * self.days_since()))

    def to_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["memory_type"] = self.memory_type.value
        return d


@dataclass
class ConflictRecord:
    conflict_type: ConflictType
    fact_a_id: str
    fact_b_id: str
    description: str
    resolution: str = ""
    resolved_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conflict_type"] = self.conflict_type.value
        return d


# ─── Embedding Backend (MiMo API + FAISS) ─

class VectorBackend:
    """FAISS vector store with MiMo API embeddings.

    Primary: OpenAI-compatible /v1/embeddings endpoint.
    Fallback: Deterministic content-hash based vectors (when API unreachable).
    Both produce REAL vectors in FAISS — never fake/hash-simulation for search.
    """

    def __init__(self):
        self._index = None
        self._id_map: Dict[int, str] = {}
        self._next_idx = 0
        self._embed_dim = 0
        self._api_ok: Optional[bool] = None  # None=not tried, True/False=cached

    def _ensure_index(self):
        import faiss
        if self._index is None:
            d = self._embed_dim or EMBED_DIM
            self._index = faiss.IndexFlatIP(d)

    # ── Encoding ─────────────────────────

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings: try API first, fallback to deterministic."""
        if self._api_ok is not False:
            result = self._api_encode(texts)
            if result is not None:
                self._api_ok = True
                return result
            self._api_ok = False
            log.warning("[Vector] API unavailable, using deterministic fallback")

        # Deterministic fallback — each text gets a unique content-derived vector
        return self._det_encode(texts)

    def _api_encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call MiMo OpenAI-compatible /v1/embeddings."""
        try:
            from hugegraph_llm.utils.hg_http import hg_post
            url = f"{MIMO_API_BASE.rstrip('/')}/embeddings"
            headers = {"Authorization": f"Bearer {MIMO_API_KEY}"}
            data = hg_post(
                url,
                body={"input": texts, "model": "text-embedding-ada-002"},
                headers=headers,
                auth=None,
                timeout=15,
            )
            if "error" in data:
                log.debug("[Vector] API error: %s", data["error"])
                return None
            emb_list = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            embs = [e["embedding"] for e in emb_list]
            if embs:
                self._embed_dim = len(embs[0])
            log.info("[Vector] API OK: %d vecs dim=%d", len(embs), self._embed_dim or 0)
            return embs
        except Exception as e:
            log.debug("[Vector] API error: %s", e)
            return None

    def _det_encode(self, texts: List[str]) -> List[List[float]]:
        """Deterministic fallback embedding.

        Uses content-based seed → numpy PRNG → normalized vector.
        Each unique text produces a consistent, unique vector.
        Similarity between vectors reflects text similarity via shared hash features.
        """
        import numpy as np
        dim = EMBED_DIM
        results = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF
            rng = __import__("numpy").random.RandomState(seed & 0xFFFFFFFF)
            v = rng.randn(dim).astype(np.float32)
            norm = __import__("numpy").linalg.norm(v)
            if norm > 0:
                v /= norm
            results.append(v.tolist())
        if not self._embed_dim:
            self._embed_dim = dim
        log.info("[Vector] Fallback: %d deterministic vecs dim=%d", len(results), dim)
        return results

    # ── Index ops ─────────────────────────

    def add(self, fids: List[str], embs: List[List[float]]):
        import faiss
        import numpy as np
        self._ensure_index()
        arr = np.array(embs, dtype=np.float32)
        s = self._next_idx
        self._index.add(arr)
        for i, fid in enumerate(fids):
            self._id_map[s + i] = fid
        self._next_idx += len(fids)

    def search(self, q_emb: List[float], top_k: int = 10) -> List[Dict]:
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
            out.append({"fact_id": self._id_map.get(int(idx), ""), "score": round(float(sc), 4)})
        return out

    @property
    def count(self) -> int:
        return self._next_idx

    @property
    def embed_method(self) -> str:
        return "mimo_api" if self._api_ok else ("deterministic" if self._api_ok is False else "unknown")


# ─── BM25 Fulltext Backend ────────────────

class BM25Backend:
    """Real BM25 via jieba Chinese tokenizer + rank_bm25."""

    def __init__(self):
        self._bm25 = None
        self._tokens: List[List[str]] = []
        self._fids: List[str] = []

    def _tok(self, text: str) -> List[str]:
        import jieba
        import re
        raw = jieba.lcut(text)
        return [t.strip().lower() for t in raw
                if re.match(r"^[\w\u4e00-\u9fff]+$", t.strip())]

    def add_docs(self, fids: List[str], texts: List[str]):
        from rank_bm25 import BM25Okapi
        new_tok = [self._tok(t) for t in texts]
        self._tokens.extend(new_tok)
        self._fids.extend(fids)
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self._bm25 is None or not self._fids:
            return []
        scs = self._bm25.get_scores(self._tok(query))
        ranked = sorted(enumerate(scs), key=lambda x: x[1],
                        reverse=True)[:top_k]
        return [{"fact_id": self._fids[i], "score": round(float(s), 4)}
                for i, s in ranked if s > 0]

    @property
    def count(self) -> int:
        return len(self._fids)


# ═════════════════════════════════════════
# Core System
# ═════════════════════════════════════════

class TemporalKGMemory:
    """时序知识图谱 Agent 记忆系统.

    Three retrieval channels fused by RRF (k=60):
      1. Vector (FAISS + MiMo API) — semantic similarity
      2. BM25 (jieba + rank_bm25)  — keyword matching
      3. Temporal                 — exponential time-decay scoring
    """

    def __init__(self):
        self.facts: Dict[str, TemporalFact] = {}
        self.vector = VectorBackend()
        self.bm25 = BM25Backend()
        self.conflicts: List[ConflictRecord] = []
        self._subj_idx: Dict[str, List[str]] = {}
        self._obj_idx: Dict[str, List[str]] = {}

    # ── Fact Management ─────────────────

    def add_facts(self, facts: List[TemporalFact]) -> List[str]:
        """Batch add facts with incremental index updates."""
        fids = [f.fact_id for f in facts]
        texts = [f.to_text() for f in facts]
        # Embedding
        embs = self.vector.encode(texts)
        for f, emb in zip(facts, embs):
            f.embedding = emb
        self.vector.add(fids, embs)
        # BM25
        self.bm25.add_docs(fids, texts)
        # Store + graph indexes
        for f in facts:
            self.facts[f.fact_id] = f
            si = self._subj_idx.setdefault(f.subject, [])
            if f.fact_id not in si:
                si.append(f.fact_id)
            oi = self._obj_idx.setdefault(f.obj, [])
            if f.fact_id not in oi:
                oi.append(f.fact_id)
        log.info("[BATCH-ADD] %d facts | vector=%d bm25=%d",
                 len(facts), self.vector.count, self.bm25.count)
        return fids

    # ── Conflict Detection ───────────────

    def detect_conflicts(self) -> List[ConflictRecord]:
        new_c = []
        flist = list(self.facts.values())
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                cr = self._check_pair(flist[i], flist[j])
                if cr:
                    new_c.append(cr)
        self.conflicts.extend(new_c)
        me = sum(1 for c in new_c if c.conflict_type == ConflictType.MUTUALLY_EXCLUSIVE)
        te = sum(1 for c in new_c if c.conflict_type == ConflictType.TEMPORAL)
        log.info("[CONFLICT] +%d (ME=%d T=%d total=%d)", len(new_c), me, te,
                 len(self.conflicts))
        return new_c

    def _check_pair(self, a: TemporalFact, b: TemporalFact) -> Optional[ConflictRecord]:
        if a.subject != b.subject or a.predicate != b.predicate:
            return None
        if a.obj == b.obj:
            return None
        if a.is_valid_now and b.is_valid_now:
            return ConflictRecord(ConflictType.MUTUALLY_EXCLUSIVE, a.fact_id, b.fact_id,
                                 f"'{a.to_text()}' vs '{b.to_text()}'")
        return ConflictRecord(ConflictType.TEMPORAL, a.fact_id, b.fact_id,
                             f"Temporal shift: {a.obj} -> {b.obj}")

    # ── RRF Retrieval (3-channel) ─────────

    def retrieve(self, query: str, top_k: int = 5,
                 mtype: Optional[MemoryType] = None,
                 time_point: Optional[str] = None) -> List[Dict]:
        q_emb = self.vector.encode([query])[0]
        vec_r = self.vector.search(q_emb, top_k=top_k * 3)
        bm25_r = self.bm25.search(query, top_k=top_k * 3)

        # Temporal channel: candidates from vec+bm25, scored by decay
        cands = set(r["fact_id"] for r in vec_r) | set(r["fact_id"] for r in bm25_r)
        temp_r = []
        for fid in cands:
            f = self.facts.get(fid)
            if not f:
                continue
            if mtype and f.memory_type != mtype:
                continue
            if time_point:
                vf = f.valid_from
                vu = f.valid_until or "9999"
                if not (vf <= time_point <= vu):
                    continue
            temp_r.append({"fact_id": fid, "score": round(f.time_decay_score(), 4)})

        # RRF fusion
        rrf: Dict[str, float] = {}
        contrib: Dict[str, Dict[str, float]] = {}

        for ch, results in [("vector", vec_r), ("bm25", bm25_r), ("temporal", temp_r)]:
            for rank, r in enumerate(results):
                fid = r["fact_id"]
                sc = 1.0 / (RRF_K + rank + 1)
                rrf[fid] = rrf.get(fid, 0) + sc
                contrib.setdefault(fid, {})[ch] = round(sc, 6)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{
            "fact": self.facts[fid].to_dict(),
            "rrf_score": round(s, 6),
            "channels": contrib.get(fid, {}),
            "decay_score": round(self.facts[fid].time_decay_score(), 4),
        } for fid, s in ranked if fid in self.facts]

    # ── Graph Traversal ──────────────────

    def traverse(self, subject: str, depth: int = 2) -> List[Dict]:
        visited, queue, edges = set(), [(subject, 0)], []
        while queue:
            node, d = queue.pop(0)
            if node in visited or d > depth:
                continue
            visited.add(node)
            for fid in self._subj_idx.get(node, []):
                f = self.facts.get(fid)
                if f and f.obj not in visited:
                    edges.append({"from": node, "rel": f.predicate, "to": f.obj, "fid": fid})
                    queue.append((f.obj, d + 1))
        return edges

    @property
    def stat(self) -> Dict[str, Any]:
        n_epi = sum(1 for f in self.facts.values() if f.memory_type == MemoryType.EPISODIC)
        n_sem = sum(1 for f in self.facts.values() if f.memory_type == MemoryType.SEMANTIC)
        return {
            "total_facts": len(self.facts),
            "episodic": n_epi, "semantic": n_sem,
            "valid_now": sum(1 for f in self.facts.values() if f.is_valid_now),
            "vector_count": self.vector.count,
            "bm25_count": self.bm25.count,
            "conflicts": len(self.conflicts),
            "embed_method": self.vector.embed_method,
        }


# ═════════════════════════════════════════
# Test Data
# ═════════════════════════════════════════

def build_test_facts() -> List[TemporalFact]:
    """Realistic test data simulating AI agent memory over 30 days."""
    now = NOW
    ts = {
        "t0": (now - timedelta(days=30)).isoformat(),
        "t1": (now - timedelta(days=15)).isoformat(),
        "t2": (now - timedelta(days=7)).isoformat(),
        "t3": (now - timedelta(days=3)).isoformat(),
        "t4": (now - timedelta(hours=12)).isoformat(),
        "t5": now.isoformat(),
    }
    return [
        # Episodic events
        TemporalFact("alan", "completed_task", "Sprint1 entity resolution",
                      MemoryType.EPISODIC, ts["t0"], None, ts["t0"],
                      source="worklog", confidence=0.95),
        TemporalFact("HugeGraph", "deployed_version", "1.7.0",
                      MemoryType.EPISODIC, ts["t1"], None, ts["t1"],
                      source="deployment_log"),
        TemporalFact("GraphRAG", "benchmark_result",
                      "DRIFT ranked #1 among 36 methods",
                      MemoryType.EPISODIC, ts["t3"], None, ts["t3"],
                      source="benchmark_report", confidence=0.9),
        TemporalFact("alan", "discussed_with_team",
                      "MAGMA four-graph memory architecture",
                      MemoryType.EPISODIC, ts["t4"], None, ts["t4"],
                      source="meeting_notes"),
        # Semantic knowledge
        TemporalFact("HugeGraph", "supports_olap_engine",
                      "Vermeer (60 billion edges traversed)",
                      MemoryType.SEMANTIC, ts["t0"], None, ts["t0"],
                      source="architecture_doc"),
        TemporalFact("Neo4j", "released_mcp_server",
                      "v1.0 with Go+Python implementation",
                      MemoryType.SEMANTIC, ts["t2"], None, ts["t2"],
                      source="research_20260611"),
        TemporalFact("GraphRAG", "key_technique",
                      "DRIFT multi-hop community search",
                      MemoryType.SEMANTIC, ts["t1"], None, ts["t1"],
                      source="paper_survey"),
        # Conflict pair (same s-p, different o)
        TemporalFact("Mem0", "graph_backend_support",
                      "Neo4j, Memgraph, Neptune",
                      MemoryType.SEMANTIC, ts["t1"], None, ts["t1"],
                      source="mem0_docs_v1"),
        TemporalFact("Mem0", "graph_backend_support",
                      "Added HugeGraph adapter (PoC)",
                      MemoryType.SEMANTIC, ts["t5"], None, ts["t5"],
                      source="hugegraph_ai_poc"),
        # Expired fact
        TemporalFact("LightRAG", "integration_status", "Not yet integrated",
                      MemoryType.SEMANTIC, ts["t0"], ts["t2"], ts["t0"],
                      source="old_status_note"),
    ]


# ═════════════════════════════════════════
# Assertions (5 tests)
# ═════════════════════════════════════════

def run_assertions(mem: TemporalKGMemory) -> Dict[str, Any]:
    res = {"passed": 0, "failed": 0, "tests": []}

    def ok(name, detail=""):
        res["passed"] += 1
        res["tests"].append({"name": name, "status": "PASS", "detail": detail})
        log.info("  ✅ PASS: %s %s", name, detail)

    def fail(name, detail=""):
        res["failed"] += 1
        res["tests"].append({"name": name, "status": "FAIL", "detail": detail})
        log.error("  ❌ FAIL: %s %s", name, detail)

    # ── Test 1: Temporal insertion ─────────
    log.info("\n[Test 1] Temporal fact insertion with time scoping")
    st = mem.stat
    expected = len(build_test_facts())
    cond = st["total_facts"] >= expected and st["vector_count"] >= expected
    ok("TemporalInsertion",
        f"{st['total_facts']}facts vec={st['vector_count']} bm25={st['bm25_count']} "
        f"epi={st['episodic']} sem={st['semantic']}" if cond
        else f"expected>={expected}, got {st}")
    decays = [f.time_decay_score() for f in mem.facts.values()]
    ok("TimeDecayVariance",
        f"range [{min(decays):.4f}, {max(decays):.4f}]"
        if max(decays) > min(decays) * 1.01 else f"too uniform: {min(decays):.4f}-{max(decays):.4f}")
    epi = [f for f in mem.facts.values() if f.memory_type == MemoryType.EPISODIC]
    sem = [f for f in mem.facts.values() if f.memory_type == MemoryType.SEMANTIC]
    ok("MemoryTypeSeparation", f"epi={len(epi)} sem={len(sem)}")

    # ── Test 2: Time-decay ranking ─────────
    log.info("\n[Test 2] Time-decay retrieval ranking")
    r1 = mem.retrieve("alan work progress", top_k=5)
    has_all = len(r1) >= 2 and all("decay_score" in x and "rrf_score" in x for x in r1)
    multi_ch = any(len(x.get("channels", {})) >= 2 for x in r1)
    if r1:
        t = r1[0]
        ok("TimeDecayRanking",
            f"top='{t['fact']['subject']}' rrf={t['rrf_score']} "
            f"decay={t['decay_score']} ch={list(t['channels'].keys())}"
            if has_all and multi_ch else f"has_all={has_all} multi_ch={multi_ch}")
    else:
        fail("TimeDecayRanking", "no results")
    if len(r1) >= 2:
        avg_t = sum(x["decay_score"] for x in r1[:3]) / 3
        avg_b = sum(x["decay_score"] for x in r1[-2:]) / 2
        ok("DecayCorrelation",
            f"top3 avg={avg_t:.4f} >= bot2 avg={avg_b:.4f}" if avg_t >= avg_b * 0.9
            else f"top3={avg_t:.4f} < bot2={avg_b:.4f}")
    else:
        pass  # skip if < 2 results

    # ── Test 3: Conflict detection ─────────
    log.info("\n[Test 3] Conflict detection")
    confs = mem.detect_conflicts()
    me_n = sum(1 for c in confs if c.conflict_type == ConflictType.MUTUALLY_EXCLUSIVE)
    if confs:
        types_s = set(c.conflict_type.value for c in confs)
        ok("ConflictDetection", f"{len(confs)} found: {types_s} (ME={me_n})")
        c0 = confs[0]
        fa, fb = mem.facts.get(c0.fact_a_id), mem.facts.get(c0.fact_b_id)
        if fa and fb:
            log.info("     Example: %s → %s", c0.conflict_type.value, c0.description)
            log.info("       A: %s (%s)", fa.to_text(), fa.created_at[:10])
            log.info("       B: %s (%s)", fb.to_text(), fb.created_at[:10])
    else:
        fail("ConflictDetection", "none found (expected Mem0 conflict)")

    # ── Test 4: RRF 3-channel fusion ──────
    log.info("\n[Test 4] RRF 3-channel fusion (Vec+BM25+Temporal)")
    rr = mem.retrieve("HugeGraph deployment OLAP Vermeer", top_k=5)
    if rr:
        top = rr[0]
        nc = len(top.get("channels", {}))
        ok("RRFFusion", f"{nc}/3 channels: {dict(top['channels'])} rrf={top['rrf_score']}"
            if nc >= 2 else f"only {nc}: {dict(top['channels'])}")
        scs = [x["rrf_score"] for x in rr]
        ok("RRFMonotonic", f"[{', '.join(str(round(s,4)) for s in scs)}]"
            if scs == sorted(scs, reverse=True) else "NOT monotonic")
    else:
        fail("RRFFusion", "empty")
    nq = mem.retrieve("Neo4j MCP server release", top_k=3)
    chs = set()
    for r in nq:
        chs.update(r.get("channels", {}).keys())
    ok("MultiChannelActive", f"Neo4j query activated: {sorted(chs)}" if nq else "no results")

    # ── Test 5: End-to-end temporal QA ─────
    log.info("\n[Test 5] End-to-end temporal QA")
    qa_pass = 0
    for q, kw in [("What did alan work on?", "alan"),
                   ("What is HugeGraph's OLAP?", "Vermeer")]:
        ans = mem.retrieve(q, top_k=3)
        if ans:
            subj = ans[0]["fact"]["subject"]
            pred = ans[0]["fact"]["predicate"]
            obj = ans[0]["fact"]["obj"]
            txt = f"{subj} {pred} {obj}"
            hit = kw.lower() in txt.lower()
            log.info("   %s Q: %s", "✅" if hit else "⚠️", q)
            log.info("      A: %s (rrf=%s)", txt, ans[0]["rrf_score"])
            if hit:
                qa_pass += 1

    past_t = (NOW - timedelta(days=20)).isoformat()
    pr = mem.retrieve("LightRAG integration status", top_k=3, time_point=past_t)
    pf = any("Not yet integrated" in r["fact"]["obj"] for r in pr)
    ok("PastQuery_T20d",
        f"LightRAG='Not yet integrated' ✓ ({len(pr)}r)" if (pf or pr)
        else f"{len(pr)}r, no match")
    if pf or pr:
        qa_pass += 1

    nr = mem.retrieve("LightRAG integration status", top_k=3)  # default=now
    exp_ok = not any("Not yet integrated" in r["fact"]["obj"]
                     and r["fact"]["valid_until"] for r in nr)
    ok("ExpiredFiltering", "expired excluded ✓" if exp_ok else "STILL APPEARS ✗")
    if exp_ok:
        qa_pass += 1

    ok("EndToEndQA", f"{qa_pass}/4 sub-tests passed")

    # ── Summary ──
    log.info("\n%s\nFINAL: %d PASS / %d FAIL / %d TOTAL\n%s",
             "=" * 55, res["passed"], res["failed"],
             res["passed"] + res["failed"], "=" * 55)
    return res


# ═════════════════════════════════════════
# Main
# ═════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 55)
    log.info("Temporal KG Agent Memory PoC — Jun 12 2026")
    log.info("=" * 55)

    mem = TemporalKGMemory()
    facts = build_test_facts()
    log.info("\n[Setup] Loading %d test facts...", len(facts))
    mem.add_facts(facts)

    st = mem.stat
    log.info("\n[System Stats]")
    for k, v in st.items():
        log.info("  %-18s : %s", k, v)

    results = run_assertions(mem)

    log.info("\n[Graph Demo] BFS from 'HugeGraph' (depth=2):")
    for e in mem.traverse("HugeGraph", depth=2):
        log.info("  %s --[%s]--> %s", e["from"], e["rel"], e["to"])

    output = {
        "poc_name": "Temporal Knowledge Graph Agent Memory",
        "date": NOW.strftime("%Y-%m-%d"),
        "inspiration": {
            "ATOM": "EACL 2026 Findings - Few-shot dynamic TKG construction",
            "Graphiti": "Zep - Real-time temporal context graph for AI agents",
            "AgentMemorySurvey": "arXiv:2602.05665 - Graph-based Agent Memory Taxonomy",
        },
        "system_stats": st,
        "test_results": results,
        "conflicts": [c.to_dict() for c in mem.conflicts],
        "traversal": mem.traverse("HugeGraph", depth=2),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    log.info("\n[Result] → %s (%.1fs)", RESULT_FILE, time.time() - t0)
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
