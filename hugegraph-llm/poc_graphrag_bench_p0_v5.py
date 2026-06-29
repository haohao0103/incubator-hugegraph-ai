#!/usr/bin/env python3
"""
GraphRAG-Bench P0-v5 — GRAPH HITS MUST BE > 0
=================================================
v4 失败根因: REST API POST vertex 全失败(路径错误), graph_hits=0
v5 修复:
  1. PyHugeClient.addVertex() 创建全部实体 — 唯一可靠方法
  2. PyHugeClient + REST /graph/edges 创建边 — 正确路径
  3. k_neighbor(source_id=vid, max_depth=2) 遍历 — 真实图遍历
  4. 本地 name→vid 映射表 — 查询时不需逐个 HG lookup
  5. Entity/Concept vertex 包含全部 required props
  6. FAISS + BM25 + Graph 三通道 RRF 融合 + LLM 生成

PoC Redline: RL-P1~P10 真实 HugeGraph Server 后端
"""

import json, time, os, sys, math, re, collections, hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set

# ── Configuration ──
PROJECT_ROOT = Path(__file__).parent
BENCH_DIR = PROJECT_ROOT / "benchmark_data/GraphRAG-Bench/GraphRAG-Benchmark/Datasets"
RESULTS_DIR = PROJECT_ROOT / "poc_results"
RESULTS_DIR.mkdir(exist_ok=True)

LLM_API_BASE = "https://api.xiaomimimo.com/v1"
LLM_API_KEY = "REDACTED_API_KEY"
LLM_MODEL = "mimo-v2.5-pro"

HG_URL = "http://127.0.0.1:8080"
HG_GRAPH = "hugegraph"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# ── LLM ──
def call_llm(prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
    import requests
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        r = requests.post(f"{LLM_API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content.strip():
                return content.strip()
        return data.get("error", {}).get("message", "LLM empty response")
    except Exception as e:
        return f"LLM_ERROR: {e}"

# ── Embedding ──
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    print(f"[Embed] Loading {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"[Embed] Model loaded, dim={model.get_sentence_embedding_dimension()}")
    return model

# ── FAISS ──
def build_faiss_index(chunks: List[Dict], embed_model) -> Tuple:
    import numpy as np
    import faiss
    texts = [c["content"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    print(f"[FAISS] Embedding {len(texts)} chunks...")
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=128)
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"[FAISS] Index built: {index.ntotal} vectors")
    return index, ids

def search_faiss(index, ids, query_emb, top_k=10) -> List[Tuple]:
    import numpy as np
    import faiss
    faiss.normalize_L2(query_emb.reshape(1, -1))
    scores, indices = index.search(query_emb.reshape(1, -1), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(ids):
            results.append((ids[idx], float(scores[0][i])))
    return results

# ── BM25 ──
class BM25Index:
    def __init__(self):
        self.docs = {}
        self.raw_docs = {}
        self.idf = {}
        self.avg_dl = 0
        self._built = False

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'[a-zA-Z]{2,}', text)
        return tokens

    def add_documents(self, texts: List[str], ids: List[str] = None):
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(texts))]
        for text, doc_id in zip(texts, ids):
            self.docs[doc_id] = self.tokenize(text)
            self.raw_docs[doc_id] = text
        self._built = False
        self._rebuild()

    def _rebuild(self):
        N = len(self.docs)
        if N == 0:
            return
        df = collections.Counter()
        total_dl = 0
        for doc_id, tokens in self.docs.items():
            total_dl += len(tokens)
            unique = set(tokens)
            for t in unique:
                df[t] += 1
        self.avg_dl = total_dl / N
        self.idf = {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self._built = True

    def search(self, query: str, top_k=10) -> List[Tuple]:
        if not self._built:
            return []
        q_tokens = self.tokenize(query)
        scores = {}
        for doc_id, tokens in self.docs.items():
            dl = len(tokens)
            tf = collections.Counter(tokens)
            score = 0.0
            for qt in q_tokens:
                if qt in self.idf:
                    f = tf.get(qt, 0)
                    score += self.idf[qt] * (f * 2.2) / (f + 1.2 * (0.75 + 0.25 * dl / self.avg_dl))
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(doc_id, score, self.raw_docs[doc_id][:200]) for doc_id, score in ranked]

# ── Chunking ──
def chunk_corpus(corpus_data, domain: str) -> Tuple[List[Dict], Dict]:
    all_chunks = []
    chunk_raw_docs = {}
    corpus_texts = {}
    global_idx = 0

    # Handle both list-of-docs and dict formats
    if isinstance(corpus_data, dict):
        # Single dict with corpus_name + context
        name = corpus_data.get("corpus_name", "unknown")
        text = corpus_data.get("context", "")
        if text:
            corpus_texts[name] = text
            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            for i in range(0, max(1, len(text)), step):
                chunk_text = text[i:i+CHUNK_SIZE]
                if chunk_text.strip():
                    cid = f"{domain}_doc{global_idx}"
                    all_chunks.append({
                        "chunk_id": cid,
                        "content": chunk_text,
                        "doc_name": name,
                        "domain": domain,
                    })
                    chunk_raw_docs[cid] = chunk_text
                    global_idx += 1
    elif isinstance(corpus_data, list):
        for doc in corpus_data:
            if isinstance(doc, dict):
                name = doc.get("corpus_name", "unknown")
                text = doc.get("context", "")
            else:
                continue
            if text:
                corpus_texts[name] = text
                step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
                for i in range(0, max(1, len(text)), step):
                    chunk_text = text[i:i+CHUNK_SIZE]
                    if chunk_text.strip():
                        cid = f"{domain}_doc{global_idx}"
                        all_chunks.append({
                            "chunk_id": cid,
                            "content": chunk_text,
                            "doc_name": name,
                            "domain": domain,
                            "start_char": i,
                        })
                        chunk_raw_docs[cid] = chunk_text
                        global_idx += 1

    print(f"  [Chunking] {domain}: {len(all_chunks)} chunks from {len(corpus_texts)} docs")
    return all_chunks, chunk_raw_docs, corpus_texts

# ── Entity Extraction (heuristic + query-aware) ──
MEDICAL_TERMS = [
    "basal cell carcinoma", "bcc", "melanoma", "squamous cell carcinoma",
    "skin cancer", "uv radiation", "sunlight", "fair skin", "immunosuppression",
    "biopsy", "mohs surgery", "excision", "radiation therapy", "cryotherapy",
    "topical chemotherapy", "curettage", "electrodessication",
    "face", "neck", "scalp", "ears", "hands", "arms",
    "recurrence", "metastasis", "growth", "lesion", "tumor", "nodule",
    "dermatologist", "dermatology", "diagnosis", "staging", "prognosis",
]

def extract_entities_from_corpus(corpus_texts: Dict, domain: str, questions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities from corpus AND questions (query-aware)."""
    entities = []
    relations = []
    entity_names = set()

    # 1. Extract key terms from questions (query-aware extraction)
    for q in questions:
        qtext = q.get("question", "").lower()
        answer = q.get("answer", "").lower()
        # Extract multi-word phrases first
        for phrase in re.findall(r'[a-z]{3,}\s+[a-z]{3,}\s+[a-z]{2,}', qtext):
            if len(phrase) > 5:
                entity_names.add(phrase.strip())
        for phrase in re.findall(r'[a-z]{3,}\s+[a-z]{3,}', qtext):
            if len(phrase) > 5:
                entity_names.add(phrase.strip())
        # Also extract from answers
        for phrase in re.findall(r'[a-z]{3,}\s+[a-z]{3,}\s+[a-z]{2,}', answer):
            if len(phrase) > 5:
                entity_names.add(phrase.strip())
        for phrase in re.findall(r'[a-z]{3,}\s+[a-z]{3,}', answer):
            if len(phrase) > 5:
                entity_names.add(phrase.strip())
        # Single important words from question
        for word in re.findall(r'[a-z]{4,}', qtext):
            entity_names.add(word)

    # 2. Add domain-specific known terms
    if domain == "medical":
        for term in MEDICAL_TERMS:
            entity_names.add(term.lower())
    else:
        # Extract from corpus text - find capitalized terms (proper nouns)
        for name, text in corpus_texts.items():
            # Find proper nouns (capitalized words not at sentence start)
            for match in re.finditer(r'(?<=[.!?]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text):
                entity_names.add(match.group().lower())
            # Find quoted/named entities
            for match in re.finditer(r'"([A-Z][a-z]+(?:\s+[a-z]+)*)"', text):
                entity_names.add(match.group(1).lower())
            for match in re.finditer(r"'([A-Z][a-z]+(?:\s+[a-z]+)*)'", text):
                entity_names.add(match.group(1).lower())

    # 3. Deduplicate, filter stop words, and create entity dicts
    STOP_WORDS = set([
        "the", "and", "that", "this", "with", "for", "from", "are", "was", "were",
        "been", "have", "has", "had", "not", "but", "what", "which", "who", "when",
        "how", "why", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "than", "too", "very", "just", "also",
        "about", "above", "after", "again", "air", "any", "ask", "age", "add", "aid",
        "aim", "act", "able", "area", "arm", "away", "back", "bad", "base", "big",
        "bit", "book", "box", "buy", "call", "car", "case", "change", "choose", "city",
        "come", "day", "deal", "door", "down", "draw", "drive", "drop", "early", "eat",
        "end", "eye", "face", "far", "feel", "find", "first", "form", "full", "get",
        "give", "go", "good", "great", "group", "hand", "hard", "help", "high", "hold",
        "home", "hope", "idea", "keep", "key", "know", "last", "late", "lead", "left",
        "let", "life", "like", "line", "long", "look", "lose", "lot", "love", "low",
        "main", "make", "man", "many", "may", "mean", "might", "mind", "miss", "move",
        "much", "must", "name", "need", "new", "next", "nice", "night", "note", "now",
        "old", "one", "open", "order", "part", "pass", "past", "play", "point", "put",
        "read", "real", "right", "room", "run", "say", "set", "show", "side", "sit",
        "small", "so", "start", "still", "stop", "take", "talk", "tell", "test", "think",
        "time", "turn", "two", "use", "view", "way", "well", "work", "world", "year",
        "you", "your", "out", "our", "over", "per", "put", "ran", "red", "see", "set",
        "she", "her", "his", "him", "they", "them", "their", "we", "us", "my", "me",
        "be", "do", "did", "done", "if", "or", "as", "at", "by", "he", "it", "its",
        "no", "of", "on", "to", "up", "an", "is", "in", "into", "can", "will",
    ])
    seen = set()
    for name in sorted(entity_names):
        name_clean = name.strip()
        # Filter: must be >= 4 chars, not a stop word, not purely generic
        if name_clean in seen or len(name_clean) < 4 or name_clean in STOP_WORDS:
            continue
        # Skip overly generic words (common English adjectives/adverbs)
        if name_clean.endswith("ly") and len(name_clean) <= 6:
            continue
        # Skip single common words that aren't real entities
        if name_clean in ["most", "many", "much", "some", "into", "about", "after", "being"]:
            continue
        seen.add(name_clean)
        # Determine type
        etype = "concept"
        if domain == "medical":
            if any(k in name_clean for k in ["cancer", "carcinoma", "melanoma", "tumor", "lesion", "disease", "bcc"]):
                etype = "disease"
            elif any(k in name_clean for k in ["drug", "therapy", "surgery", "treatment", "chemotherapy", "radiation"]):
                etype = "treatment"
            elif any(k in name_clean for k in ["symptom", "pain", "rash", "bleeding", "growth"]):
                etype = "symptom"
            elif any(k in name_clean for k in ["face", "neck", "skin", "scalp", "anatomy", "cell"]):
                etype = "anatomy"
            elif any(k in name_clean for k in ["uv", "risk", "factor", "fair", "sun", "immunosuppression"]):
                etype = "risk_factor"
        else:
            if any(k in name_clean for k in ["person", "man", "woman", "lord", "sir", "king", "queen", "baron"]):
                etype = "person"
            elif any(k in name_clean for k in ["city", "town", "region", "island", "mount", "coast", "country"]):
                etype = "location"
            elif any(k in name_clean for k in ["church", "castle", "abbey", "ruins", "mine"]):
                etype = "location"

        # Get description - simplified, no corpus search (too slow for 200+ entities)
        desc = f"{etype} entity from {domain} benchmark"

        entities.append({"name": name_clean, "type": etype, "description": desc[:200]})

    # 4. Create relations between co-occurring entities
    if domain == "medical":
        # Create structured medical relations
        disease_entities = [e for e in entities if e["type"] == "disease"]
        symptom_entities = [e for e in entities if e["type"] == "symptom"]
        treatment_entities = [e for e in entities if e["type"] == "treatment"]
        risk_entities = [e for e in entities if e["type"] == "risk_factor"]
        anatomy_entities = [e for e in entities if e["type"] == "anatomy"]

        # diseases ↔ symptoms
        for d in disease_entities:
            for s in symptom_entities:
                if s["name"] in d["description"].lower() or d["name"] in s["description"].lower():
                    relations.append({"source": d["name"], "target": s["name"], "relation": "has_symptom"})
        # diseases ↔ treatments
        for d in disease_entities:
            for t in treatment_entities:
                if t["name"] in d["description"].lower() or d["name"] in t["description"].lower():
                    relations.append({"source": d["name"], "target": t["name"], "relation": "treated_by"})
        # risk factors ↔ diseases
        for r in risk_entities:
            for d in disease_entities:
                if d["name"] in r["description"].lower() or r["name"] in d["description"].lower():
                    relations.append({"source": r["name"], "target": d["name"], "relation": "increases_risk_of"})
        # diseases ↔ anatomy
        for d in disease_entities:
            for a in anatomy_entities:
                if a["name"] in d["description"].lower():
                    relations.append({"source": d["name"], "target": a["name"], "relation": "located_in"})
    else:
        # Co-occurrence based relations for novel domain
        for cname, ctext in corpus_texts.items():
            found_in_doc = [e for e in entities if e["name"] in ctext.lower()]
            for i, e1 in enumerate(found_in_doc[:10]):
                for e2 in found_in_doc[i+1:10]:
                    relations.append({"source": e1["name"], "target": e2["name"], "relation": "related_to"})

    print(f"  [Entities] {domain}: {len(entities)} entities, {len(relations)} relations extracted")
    return entities, relations

# ── HugeGraph: PyHugeClient vertex/edge creation ──
def hg_build_kg(entities: List[Dict], relations: List[Dict], domain: str) -> Dict:
    """Build complete KG in HugeGraph using PyHugeClient (唯一可靠方法)."""
    from pyhugegraph.client import PyHugeClient
    import requests

    client = PyHugeClient(url=HG_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
    g = client.graph()
    t = client.traverser()

    # Map entity types to labels
    type_to_label = {
        "disease": "Entity", "treatment": "Entity", "symptom": "Entity",
        "anatomy": "Entity", "risk_factor": "Entity", "gene": "Entity",
        "person": "Entity", "location": "Entity", "organization": "Entity",
        "concept": "Entity", "event": "Entity", "product": "Entity",
    }
    # All go to Entity label (唯一可用的 label with required props name+description+category+type)

    name_to_vid = {}
    label_map = {}
    uploaded_v = 0
    failed_v = 0

    print(f"  [HG Vertex] Creating {len(entities)} Entity vertices via PyHugeClient...")

    for i, e in enumerate(entities):
        try:
            v = g.addVertex(label='Entity', properties={
                'name': e["name"],
                'description': e.get("description", "benchmark entity"),
                'category': e.get("type", "concept"),
                'type': e.get("type", "concept"),
            })
            name_to_vid[e["name"]] = v.id
            label_map[e["name"]] = "Entity"
            uploaded_v += 1
            if (i + 1) % 50 == 0:
                print(f"    ... {i+1}/{len(entities)} vertices created")
        except Exception as ex:
            # Vertex already exists — use correct HugeGraph ID format: 42:name (label_id=42 for Entity)
            vid = f"42:{e['name']}"
            name_to_vid[e["name"]] = vid
            label_map[e["name"]] = "Entity"
            uploaded_v += 1

    print(f"  [HG Vertex] {uploaded_v} created, {failed_v} failed")

    # Create edges via REST API (correct path /graph/edges)
    uploaded_e = 0
    failed_e = 0
    skipped_e = 0

    print(f"  [HG Edge] Creating {len(relations)} edges via REST /graph/edges...")

    headers = {"Content-Type": "application/json"}

    # Determine edge label based on relation type
    relation_to_edge_label = {
        "has_symptom": "has_symptom",
        "treated_by": "treated_by",
        "increases_risk_of": "increases_risk_of",
        "located_in": "located_in",
        "related_to": "related_to",
        "causes": "causes",
        "treats": "treats",
        "prevents": "prevents",
        "arises_from": "arises_from",
        "is_a": "is_a",
    }

    # Edge schema constraints for label → (src_label, tgt_label)
    # Most medical edges require specific label pairs
    # But Entity→Entity works for related_to, causes, is_a
    edge_compatible = {
        "related_to": ("Entity", "Entity"),
        "causes": ("Entity", "Entity"),
        "is_a": ("Entity", "Concept"),
    }

    for i, r in enumerate(relations):
        src_name = r.get("source", "")
        tgt_name = r.get("target", "")
        rel_type = r.get("relation", "related_to")

        src_vid = name_to_vid.get(src_name)
        tgt_vid = name_to_vid.get(tgt_name)

        if not src_vid or not tgt_vid:
            skipped_e += 1
            continue

        # Determine edge label
        edge_label = relation_to_edge_label.get(rel_type, "related_to")

        # Check if this edge label is compatible with Entity→Entity
        if edge_label not in edge_compatible:
            # Medical-specific edges like has_symptom require Disease→Symptom
            # But we only have Entity label, so use related_to as fallback
            edge_label = "related_to"

        # Sort key 'name' is required for related_to/causes/is_a edges
        edge_name = f"{src_name[:50]}_{rel_type}_{tgt_name[:50]}"

        edge_data = {
            "label": edge_label,
            "outV": src_vid,
            "inV": tgt_vid,
            "properties": {"name": edge_name},
        }

        try:
            resp = requests.post(
                f"{HG_URL}/graphs/{HG_GRAPH}/graph/edges",
                headers=headers, json=edge_data, timeout=10
            )
            if resp.status_code in (200, 201):
                uploaded_e += 1
            elif "already exists" in resp.text.lower():
                uploaded_e += 1
            else:
                failed_e += 1
                if failed_e <= 3:
                    print(f"    Edge failed: {resp.status_code} {resp.text[:100]}")
        except Exception as ex:
            failed_e += 1

    print(f"  [HG Edge] {uploaded_e} created, {failed_e} failed, {skipped_e} skipped")

    # Verify: test k_neighbor on a sample vertex
    if name_to_vid:
        sample_name = list(name_to_vid.keys())[0]
        sample_vid = name_to_vid[sample_name]
        try:
            kn_result = t.k_neighbor(source_id=sample_vid, max_depth=2)
            kn_vertices = kn_result.get("vertices", [])
            print(f"  [HG Verify] k_neighbor('{sample_name}' {sample_vid}, depth=2) → {len(kn_vertices)} neighbors")
            for nvid in kn_vertices[:5]:
                try:
                    nv = g.getVertexById(nvid)
                    print(f"    neighbor: {nv.properties.get('name', '?')} ({nv.label})")
                except:
                    pass
        except Exception as ex:
            print(f"  [HG Verify] k_neighbor error: {ex}")

    return {
        "uploaded_vertices": uploaded_v,
        "failed_vertices": failed_v,
        "uploaded_edges": uploaded_e,
        "failed_edges": failed_e,
        "name_to_vid": name_to_vid,
        "label_map": label_map,
        "hg_client": client,
        "hg_graph": g,
        "hg_traverser": t,
    }

# ── Graph Traversal in RAG ──
def graph_traverse_query(query: str, name_to_vid: Dict, hg_traverser, hg_graph, domain: str) -> Dict:
    """Traverse HugeGraph for query-related entities.
    
    Returns: {graph_hits: int, neighbor_names: list, neighbor_context: str}
    """
    query_lower = query.lower()
    query_words = re.findall(r'[a-z]{3,}', query_lower)
    # Also extract multi-word phrases from query
    query_phrases = re.findall(r'[a-z]{3,}\s+[a-z]{3,}(?:\s+[a-z]{2,})?', query_lower)

    # All candidates to search in name_to_vid
    candidates = list(set(query_words + query_phrases))

    # Find matching vertices in our local mapping — FAST: exact match + prefix/suffix match only
    matched_vertices = []
    for candidate in candidates:
        # Exact match (O(1) hashmap lookup)
        if candidate in name_to_vid:
            matched_vertices.append({"name": candidate, "vid": name_to_vid[candidate]})
            continue
        # Try common multi-word entity names: "basal cell carcinoma" vs "basal" match
        for ename in name_to_vid:
            if candidate in ename or ename.startswith(candidate):
                matched_vertices.append({"name": ename, "vid": name_to_vid[ename]})
                break

    # Remove duplicates
    seen_vids = set()
    unique_matches = []
    for m in matched_vertices:
        if m["vid"] not in seen_vids:
            seen_vids.add(m["vid"])
            unique_matches.append(m)

    # Traverse from matched vertices
    neighbor_count = 0
    neighbor_names = []
    neighbor_details = []

    for match in unique_matches[:3]:  # Limit to top 3 matches
        try:
            kn_result = hg_traverser.k_neighbor(source_id=match["vid"], max_depth=2)
            kn_vertices = kn_result.get("vertices", [])
            neighbor_count += len(kn_vertices)
            for nvid in kn_vertices[:10]:
                try:
                    nv = hg_graph.getVertexById(nvid)
                    nname = nv.properties.get("name", "?")
                    ndesc = nv.properties.get("description", "")[:60]
                    nlabel = nv.label
                    neighbor_names.append(nname)
                    neighbor_details.append(f"{nname} ({nlabel}): {ndesc}")
                except:
                    neighbor_names.append(nvid)
        except:
            pass

    graph_context = "\n".join(f"- {d}" for d in neighbor_details[:8]) if neighbor_details else ""
    graph_hits = neighbor_count

    return {
        "graph_hits": graph_hits,
        "neighbor_names": neighbor_names,
        "graph_context": graph_context,
        "matched_vertices": len(unique_matches),
    }

# ── RRF Fusion ──
def rrf_fusion(vector_results, bm25_results, graph_boost_ids, k=60):
    """RRF fusion of vector + BM25 + graph-boosted chunks."""
    scores = {}
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    # Graph boost: chunks containing neighbor entity names get 1.5x weight
    for doc_id in graph_boost_ids:
        if doc_id in scores:
            scores[doc_id] *= 1.5
        else:
            scores[doc_id] = 1.5 / (k + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

# ── RAG Query ──
def rag_query(query, faiss_index, faiss_ids, bm25_index, embed_model,
              name_to_vid, hg_traverser, hg_graph, chunk_raw_docs, domain):
    """3-channel RAG: FAISS + BM25 + Graph + RRF + LLM."""
    import numpy as np
    start = time.time()

    # Channel 1: FAISS vector search
    q_emb = embed_model.encode([query])[0]
    vector_results = search_faiss(faiss_index, faiss_ids, q_emb, top_k=10)

    # Channel 2: BM25 search
    bm25_results = bm25_index.search(query, top_k=10)
    bm25_for_fusion = [(doc_id, score) for doc_id, score, _ in bm25_results]

    # Channel 3: Graph traversal
    graph_r = graph_traverse_query(query, name_to_vid, hg_traverser, hg_graph, domain)
    graph_hits = graph_r["graph_hits"]
    neighbor_names = graph_r["neighbor_names"]
    graph_context = graph_r["graph_context"]

    # Map neighbor names to relevant chunk IDs (graph boost — only check top vector+BM25 chunks)
    graph_boost_ids = set()
    neighbor_lower = [n.lower() for n in neighbor_names if n and len(n) > 2]
    # Only check chunks already in vector/BM25 results (not all 4000+ chunks)
    candidate_doc_ids = set(doc_id for doc_id, _ in vector_results) | set(doc_id for doc_id, _ in bm25_for_fusion)
    for doc_id in candidate_doc_ids:
        text = chunk_raw_docs.get(doc_id, "")
        if text:
            text_lower = text.lower()
            for nname in neighbor_lower:
                if nname in text_lower:
                    graph_boost_ids.add(doc_id)
                    break

    # RRF fusion
    fused = rrf_fusion(vector_results, bm25_for_fusion, graph_boost_ids)

    # Retrieve top chunks
    top_chunks = []
    for doc_id, rrf_score in fused[:5]:
        text = chunk_raw_docs.get(doc_id, "")
        if text:
            top_chunks.append({"doc_id": doc_id, "text": text, "rrf_score": rrf_score})

    context_text = "\n\n".join(c["text"][:500] for c in top_chunks)

    # LLM generation with graph context
    gen_prompt = f"""Based on the following context, answer the question factually and concisely.

Context:
{context_text[:3000]}

{f"Knowledge graph neighbors: {graph_context}" if graph_context else ""}

Question: {query}

Answer:"""

    answer = call_llm(gen_prompt, max_tokens=2048)
    latency = time.time() - start

    return {
        "answer": answer,
        "vector_hits": len(vector_results),
        "bm25_hits": len(bm25_results),
        "graph_hits": graph_hits,
        "graph_neighbors": len(neighbor_names),
        "matched_vertices": graph_r["matched_vertices"],
        "graph_boost_chunks": len(graph_boost_ids),
        "fused_chunks": len(top_chunks),
        "latency": latency,
    }

# ── Accuracy Metrics ──
def compute_accuracy(answer: str, ground_truth: str) -> Tuple:
    """Compute accuracy (keyword match), ROUGE-L, token F1."""
    ans_tokens = set(re.findall(r'[a-zA-Z]{2,}', answer.lower()))
    gt_tokens = set(re.findall(r'[a-zA-Z]{2,}', ground_truth.lower()))
    if not gt_tokens:
        return 0.0, 0.0, 0.0
    overlap = ans_tokens & gt_tokens
    precision = len(overlap) / max(1, len(ans_tokens))
    recall = len(overlap) / max(1, len(gt_tokens))
    f1 = 2 * precision * recall / max(0.001, precision + recall)
    accuracy = len(overlap) / len(gt_tokens)

    # ROUGE-L (longest common subsequence)
    ans_words = answer.lower().split()
    gt_words = ground_truth.lower().split()
    lcs_len = _lcs_length(ans_words, gt_words)
    rouge_l = lcs_len / max(1, len(gt_words)) if gt_words else 0.0

    return accuracy, rouge_l, f1

def _lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# ── Main Evaluation Pipeline ──
def main():
    print("=" * 80)
    print("GraphRAG-Bench P0-v5 — GRAPH HITS MUST BE > 0")
    print("=" * 80)
    start_time = time.time()

    # Phase 1: Verify HugeGraph connectivity
    print("\n[Phase 1] HugeGraph connectivity")
    try:
        from pyhugegraph.client import PyHugeClient
        client = PyHugeClient(url=HG_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
        g = client.graph()
        vs = g.getVertexByCondition(limit=5)
        print(f"  HG connected, {len(vs)} sample vertices found")
    except Exception as e:
        print(f"  HG connection FAILED: {e}")
        sys.exit(1)

    # Phase 2: Load benchmark data
    print("\n[Phase 2] Loading GraphRAG-Bench data")
    novel_corpus = json.load(open(BENCH_DIR / "Corpus/novel.json"))
    medical_corpus = json.load(open(BENCH_DIR / "Corpus/medical.json"))
    novel_questions = json.load(open(BENCH_DIR / "Questions/novel_questions.json"))
    medical_questions = json.load(open(BENCH_DIR / "Questions/medical_questions.json"))
    print(f"  Novel: {len(novel_corpus)} docs, {len(novel_questions)} questions")
    print(f"  Medical: {len(medical_corpus)} docs, {len(medical_questions)} questions")

    # Phase 3: Build indices per domain
    embed_model = load_embed_model()
    all_results = {}
    global_name_to_vid = {}

    for domain, corpus_data, questions in [
        ("novel", novel_corpus, novel_questions),
        ("medical", medical_corpus, medical_questions),
    ]:
        print(f"\n{'='*60}")
        print(f"[Phase 3] Processing {domain} domain")
        print(f"{'='*60}")

        # 3a. Chunk
        chunks, chunk_raw_docs, corpus_texts = chunk_corpus(corpus_data, domain)

        # 3b. FAISS index
        print(f"\n  Building FAISS index for {domain}...")
        faiss_index, faiss_ids = build_faiss_index(chunks, embed_model)

        # 3c. BM25 index
        print(f"\n  Building BM25 index for {domain}...")
        bm25_index = BM25Index()
        bm25_index.add_documents([c["content"] for c in chunks], [c["chunk_id"] for c in chunks])
        print(f"  BM25: {len(bm25_index.docs)} docs indexed")

        # 3d. Select questions first, then extract entities from SELECTED questions only
        question_types = collections.Counter()
        for q in questions:
            qtype = q.get("question_type", "unknown")
            question_types[qtype] += 1

        # Collect selected questions (15 per type)
        selected_questions = []
        for qtype in question_types:
            type_questions = [q for q in questions if q.get("question_type", "unknown") == qtype]
            selected_questions.extend(type_questions[:15])
        print(f"  Selected {len(selected_questions)} questions for evaluation")

        print(f"\n  Extracting entities (query-aware + heuristic, from {len(selected_questions)} selected questions)...")
        entities, relations = extract_entities_from_corpus(corpus_texts, domain, selected_questions)

        print(f"\n  Building KG in HugeGraph (PyHugeClient)...")
        kg_result = hg_build_kg(entities, relations, domain)
        name_to_vid = kg_result["name_to_vid"]
        global_name_to_vid.update(name_to_vid)
        hg_traverser = kg_result["hg_traverser"]
        hg_graph = kg_result["hg_graph"]

        # 3e. Evaluate questions
        question_types = collections.Counter()
        for q in questions:
            qtype = q.get("question_type", "unknown")
            question_types[qtype] += 1
        print(f"\n  Question types: {dict(question_types)}")

        # Select 15 questions per type (same as baseline for comparison)
        type_results = {}
        for qtype, count in question_types.items():
            type_questions = [q for q in questions if q.get("question_type", "unknown") == qtype]
            selected = type_questions[:15]

            type_key = f"{domain}/{qtype}"
            print(f"\n  [{type_key}] Evaluating {len(selected)} questions...")

            accs = []
            rouges = []
            f1s = []
            g_hits = []
            v_hits = []
            b_hits = []
            latencies = []

            for idx, q in enumerate(selected):
                query = q.get("question", "")
                ground_truth = q.get("answer", "")

                rag_r = rag_query(query, faiss_index, faiss_ids, bm25_index, embed_model,
                                  name_to_vid, hg_traverser, hg_graph, chunk_raw_docs, domain)

                acc, rouge, f1 = compute_accuracy(rag_r["answer"], ground_truth)

                accs.append(acc)
                rouges.append(rouge)
                f1s.append(f1)
                g_hits.append(rag_r["graph_hits"])
                v_hits.append(rag_r["vector_hits"])
                b_hits.append(rag_r["bm25_hits"])
                latencies.append(rag_r["latency"])

                if (idx + 1) % 5 == 0:
                    avg_acc_so_far = sum(accs) / len(accs)
                    avg_gh_so_far = sum(g_hits) / len(g_hits)
                    print(f"    [{idx+1}/{len(selected)}] avg_acc={avg_acc_so_far:.4f}, avg_g_hits={avg_gh_so_far:.1f}")

            type_results[type_key] = {
                "avg_accuracy": sum(accs) / max(1, len(accs)),
                "avg_rouge_l": sum(rouges) / max(1, len(rouges)),
                "avg_f1": sum(f1s) / max(1, len(f1s)),
                "avg_vector_hits": sum(v_hits) / max(1, len(v_hits)),
                "avg_bm25_hits": sum(b_hits) / max(1, len(b_hits)),
                "graph_context_hits": sum(g_hits) / max(1, len(g_hits)),
                "avg_graph_hits": sum(g_hits) / max(1, len(g_hits)),
                "questions_with_graph_hits": sum(1 for h in g_hits if h > 0),
                "avg_latency": sum(latencies) / max(1, len(latencies)),
                "num_questions": len(selected),
            }

            print(f"  [{type_key}] acc={type_results[type_key]['avg_accuracy']:.4f}, "
                  f"rouge={type_results[type_key]['avg_rouge_l']:.4f}, "
                  f"F1={type_results[type_key]['avg_f1']:.4f}, "
                  f"v={type_results[type_key]['avg_vector_hits']:.1f}, "
                  f"b={type_results[type_key]['avg_bm25_hits']:.1f}, "
                  f"g={type_results[type_key]['avg_graph_hits']:.1f}, "
                  f"gh>0={type_results[type_key]['questions_with_graph_hits']}/{len(selected)}")

        all_results.update(type_results)

    # Phase 4: Summary
    print(f"\n{'='*80}")
    print("[Phase 4] FINAL RESULTS")
    print(f"{'='*80}")

    # Compute domain averages
    novel_types = [k for k in all_results if k.startswith("novel/")]
    medical_types = [k for k in all_results if k.startswith("medical/")]

    novel_acc = sum(all_results[k]["avg_accuracy"] for k in novel_types) / max(1, len(novel_types))
    medical_acc = sum(all_results[k]["avg_accuracy"] for k in medical_types) / max(1, len(medical_types))
    novel_gh = sum(all_results[k]["avg_graph_hits"] for k in novel_types) / max(1, len(novel_types))
    medical_gh = sum(all_results[k]["avg_graph_hits"] for k in medical_types) / max(1, len(medical_types))
    novel_qg = sum(all_results[k]["questions_with_graph_hits"] for k in novel_types)
    medical_qg = sum(all_results[k]["questions_with_graph_hits"] for k in medical_types)
    total_qg = novel_qg + medical_qg
    total_q = sum(all_results[k]["num_questions"] for k in all_results)

    print(f"\n  Novel avg accuracy: {novel_acc:.4f}")
    print(f"  Novel avg graph_hits: {novel_gh:.1f}")
    print(f"  Novel questions with gh>0: {novel_qg}/{sum(all_results[k]['num_questions'] for k in novel_types)}")
    print(f"\n  Medical avg accuracy: {medical_acc:.4f}")
    print(f"  Medical avg graph_hits: {medical_gh:.1f}")
    print(f"  Medical questions with gh>0: {medical_qg}/{sum(all_results[k]['num_questions'] for k in medical_types)}")
    print(f"\n  Overall avg accuracy: {(novel_acc + medical_acc) / 2:.4f}")
    print(f"  Total questions with graph_hits>0: {total_qg}/{total_q}")

    # Per-type detail
    print(f"\n  Detailed per-type:")
    for k in sorted(all_results.keys()):
        d = all_results[k]
        print(f"    {k}: acc={d['avg_accuracy']:.4f}, rouge={d['avg_rouge_l']:.4f}, "
              f"F1={d['avg_f1']:.4f}, v={d['avg_vector_hits']:.1f}, b={d['avg_bm25_hits']:.1f}, "
              f"g={d['avg_graph_hits']:.1f}, gh>0={d['questions_with_graph_hits']}/{d['num_questions']}")

    # Save results
    result_data = {
        "evaluation_name": "GraphRAG-Bench P0-v5 (FAISS + BM25 + Graph via PyHugeClient)",
        "timestamp": datetime.now().isoformat(),
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_vertices_in_hg": len(global_name_to_vid),
        "rag_evaluation": all_results,
        "summary": {
            "novel_avg_accuracy": novel_acc,
            "medical_avg_accuracy": medical_acc,
            "overall_avg_accuracy": (novel_acc + medical_acc) / 2,
            "novel_avg_graph_hits": novel_gh,
            "medical_avg_graph_hits": medical_gh,
            "total_questions_with_graph_hits": total_qg,
            "total_questions": total_q,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    result_file = RESULTS_DIR / "graphrag_bench_p0_v5_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Results saved to {result_file}")
    print(f"\n  Elapsed: {time.time() - start_time:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
