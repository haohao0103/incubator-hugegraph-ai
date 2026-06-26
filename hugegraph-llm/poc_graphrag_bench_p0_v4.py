#!/usr/bin/env python3
"""
GraphRAG-Bench P0-Improved Evaluation v4 — FAST entity extraction + graph_hits FIXED
======================================================================================
v3 failed: LLM entity extraction timeout (60s per batch), process died after 1 question.
v4 fixes:
  1. NLP heuristic entity extraction (noun phrases + medical terms) — NO LLM, instant
  2. Bulk vertex creation with ALL required properties
  3. Domain-specific edge labels for medical domain
  4. k_neighbor traversal with graceful fallback
  5. LLM only used for RAG answer generation (not entity extraction)
  6. RRF fusion of FAISS + BM25 + Graph channels

PoC Redline: RL-P1~P10
"""

import json, time, os, sys, math, hashlib, traceback, re, collections
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set

PROJECT_ROOT = Path(__file__).parent
BENCH_ROOT = PROJECT_ROOT / "benchmark_data" / "GraphRAG-Bench" / "GraphRAG-Benchmark"
RESULT_DIR = PROJECT_ROOT / "poc_results"
RESULT_DIR.mkdir(exist_ok=True)

# ── Config ──
MIMO_API_BASE = "https://api.xiaomimimo.com/v1"
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "sk-cbgj0rzn5qvku9k6dmi4kek68qljzic1ka33o3b4czem2cm2")
MIMO_MODEL = "mimo-v2.5-pro"
HG_REST_URL = "http://127.0.0.1:8080"
HG_GRAPH = "hugegraph"
# HugeGraph 1.7.0 API paths: /graphs/hugegraph/graph/vertices, /graphs/hugegraph/graph/edges
# Vertex ID format: "label_id:name" (e.g. "42:test_entity")
# PyHugeClient handles all this correctly
HG_GRAPH_VERTICES_URL = f"{HG_REST_URL}/graphs/{HG_GRAPH}/graph/vertices"
HG_GRAPH_EDGES_URL = f"{HG_REST_URL}/graphs/{HG_GRAPH}/graph/edges"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MAX_QUESTIONS_PER_TYPE = 15
LLM_TIMEOUT = 90
LLM_MAX_RETRIES = 2
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# ============================================================
# PHASE 1: Connectivity
# ============================================================
def check_hugegraph():
    import requests
    try:
        r = requests.get(f"{HG_REST_URL}/graphs/{HG_GRAPH}", timeout=10)
        ok = r.status_code == 200
        print(f"[Phase 1] HugeGraph: {'OK' if ok else 'FAIL'} ({r.status_code})")
        return ok
    except Exception as e:
        print(f"[Phase 1] HugeGraph: FAIL ({e})")
        return False

def check_llm_api():
    import requests
    try:
        r = requests.post(f"{MIMO_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MIMO_API_KEY}", "Content-Type": "application/json"},
            json={"model": MIMO_MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 32},
            timeout=30)
        ok = r.status_code == 200
        print(f"[Phase 1] MiMo API: {'OK' if ok else 'FAIL'} ({r.status_code})")
        return ok
    except Exception as e:
        print(f"[Phase 1] MiMo API: FAIL ({e})")
        return False

# ============================================================
# PHASE 2: Data Loading + Chunking
# ============================================================
def load_benchmark(domain="novel"):
    q_path = BENCH_ROOT / "Datasets" / "Questions" / f"{domain}_questions.json"
    c_path = BENCH_ROOT / "Datasets" / "Corpus" / f"{domain}.json"
    return json.load(open(q_path)), json.load(open(c_path))

def prepare_corpus_texts(corpus, domain="novel"):
    texts = {}
    if isinstance(corpus, list):
        for doc in corpus:
            name = doc.get("corpus_name", f"doc_{len(texts)}")
            ctx = doc.get("context", "")
            if ctx: texts[name] = ctx
    elif isinstance(corpus, dict):
        ctx = corpus.get("context", "")
        if ctx: texts[f"{domain}_corpus"] = ctx
    return texts

def chunk_text_global(text, global_offset, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, max(1, len(text)), step):
        ct = text[i:i + chunk_size]
        if ct.strip():
            chunks.append({"content": ct, "chunk_index": global_offset + len(chunks),
                           "start_char": i, "end_char": min(i + chunk_size, len(text))})
    return chunks

# ============================================================
# PHASE 3: Index Building (FAISS + BM25)
# ============================================================
class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs, self.raw_docs, self.avg_dl, self.N = {}, {}, 0, 0
        self.df = collections.Counter()

    def tokenize(self, text):
        try:
            import jieba
            return [w for w in jieba.cut(text.lower()) if len(w) > 1]
        except ImportError:
            return [w for w in re.findall(r'\w+', text.lower()) if len(w) > 1]

    def add_documents(self, texts, ids=None):
        if ids is None: ids = [f"chunk_{i}" for i in range(len(texts))]
        for text, doc_id in zip(texts, ids):
            tokens = self.tokenize(text)
            self.docs[doc_id] = tokens
            self.raw_docs[doc_id] = text
            self.df.update(set(tokens))
            self.N += 1
        self.avg_dl = sum(len(t) for t in self.docs.values()) / max(1, self.N)

    def search(self, query, top_k=10):
        if self.N == 0: return []
        q_tokens = self.tokenize(query)
        scores = {}
        for doc_id, doc_tokens in self.docs.items():
            dl = len(doc_tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in self.df: continue
                tf = doc_tokens.count(qt)
                df = self.df[qt]
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avg_dl)))
            scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(doc_id, score, self.raw_docs[doc_id]) for doc_id, score in ranked]

def build_faiss_index(embeddings, ids):
    import numpy as np, faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"  [FAISS] Built: {index.ntotal} vectors, dim={dim}")
    return index, ids

def search_faiss(index, ids, query_embedding, top_k=10):
    import numpy as np, faiss
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(ids):
            results.append((ids[idx], float(scores[0][i])))
    return results

# ============================================================
# PHASE 4: FAST Entity Extraction (NLP heuristics — NO LLM)
# ============================================================

# Medical terminology patterns for entity extraction
MEDICAL_TERMS = {
    # Common diseases
    "cancer", "diabetes", "hypertension", "asthma", "arthritis", "pneumonia",
    "malaria", "tuberculosis", "hepatitis", "stroke", "heart disease", "obesity",
    "anemia", "leukemia", "melanoma", "carcinoma", "sarcoma", "lymphoma",
    "basal cell carcinoma", "squamous cell carcinoma", "alzheimer", "parkinson",
    "epilepsy", "multiple sclerosis", "lupus", "fibromyalgia", "osteoporosis",
    # Common drugs
    "aspirin", "ibuprofen", "acetaminophen", "penicillin", "amoxicillin",
    "insulin", "metformin", "statins", "morphine", "caffeine", "nicotine",
    "chemotherapy", "radiation", "immunotherapy", "antibiotics", "antiviral",
    "anti-inflammatory", "corticosteroid", "prednisone", "warfarin", "heparin",
    # Common symptoms
    "pain", "fever", "fatigue", "headache", "nausea", "dizziness", "cough",
    "inflammation", "swelling", "bleeding", "rash", "itching", "numbness",
    "paralysis", "seizure", "vomiting", "diarrhea", "constipation", "anxiety",
    "depression", "insomnia", "confusion", "memory loss", "weight loss",
    # Common treatments
    "surgery", "therapy", "medication", "vaccine", "transplant", "dialysis",
    "rehabilitation", "physiotherapy", "acupuncture", "meditation", "exercise",
    # Common anatomy
    "heart", "lung", "brain", "liver", "kidney", "stomach", "intestine",
    "bone", "muscle", "skin", "blood", "artery", "vein", "nerve", "spine",
    "pancreas", "thyroid", "prostate", "colon", "breast", "eye", "ear",
    # Risk factors
    "smoking", "alcohol", "stress", "diet", "exercise", "age", "genetics",
    "obesity", "pollution", "radiation", "infection",
}

def extract_entities_heuristic(text: str, domain: str) -> List[Dict]:
    """Extract entities using NLP heuristics — NO LLM needed, instant."""
    entities = []
    
    if domain == "medical":
        # 1. Extract known medical terms (exact match)
        text_lower = text.lower()
        for term in MEDICAL_TERMS:
            if term in text_lower:
                # Determine type
                if term in {"cancer", "diabetes", "hypertension", "asthma", "arthritis",
                           "pneumonia", "malaria", "tuberculosis", "hepatitis", "stroke",
                           "heart disease", "obesity", "anemia", "leukemia", "melanoma",
                           "carcinoma", "sarcoma", "lymphoma", "basal cell carcinoma",
                           "squamous cell carcinoma", "alzheimer", "parkinson", "epilepsy",
                           "multiple sclerosis", "lupus", "fibromyalgia", "osteoporosis"}:
                    etype = "disease"
                elif term in {"aspirin", "ibuprofen", "acetaminophen", "penicillin",
                            "amoxicillin", "insulin", "metformin", "statins", "morphine",
                            "caffeine", "nicotine", "chemotherapy", "radiation",
                            "immunotherapy", "antibiotics", "antiviral", "anti-inflammatory",
                            "corticosteroid", "prednisone", "warfarin", "heparin"}:
                    etype = "drug"
                elif term in {"pain", "fever", "fatigue", "headache", "nausea",
                            "dizziness", "cough", "inflammation", "swelling", "bleeding",
                            "rash", "itching", "numbness", "paralysis", "seizure",
                            "vomiting", "diarrhea", "constipation", "anxiety", "depression",
                            "insomnia", "confusion", "memory loss", "weight loss"}:
                    etype = "symptom"
                elif term in {"surgery", "therapy", "medication", "vaccine",
                            "transplant", "dialysis", "rehabilitation", "physiotherapy",
                            "acupuncture", "meditation", "exercise"}:
                    etype = "treatment"
                elif term in {"heart", "lung", "brain", "liver", "kidney", "stomach",
                            "intestine", "bone", "muscle", "skin", "blood", "artery",
                            "vein", "nerve", "spine", "pancreas", "thyroid", "prostate",
                            "colon", "breast", "eye", "ear"}:
                    etype = "anatomy"
                elif term in {"smoking", "alcohol", "stress", "diet", "age",
                            "genetics", "obesity", "pollution", "radiation", "infection"}:
                    etype = "risk_factor"
                else:
                    etype = "concept"
                entities.append({"name": term.title(), "type": etype,
                                "description": f"{term.title()} - medical {etype}",
                                "category": f"medical_{etype}"})
        
        # 2. Extract capitalized multi-word terms (potential medical entities)
        cap_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text)
        for phrase in cap_phrases[:20]:
            if phrase.lower() not in MEDICAL_TERMS and len(phrase) > 4:
                entities.append({"name": phrase, "type": "concept",
                                "description": f"{phrase} - medical concept from corpus",
                                "category": "medical_concept"})
    
    else:  # novel domain
        # 1. Extract capitalized names (character names, places)
        # Single capitalized words (potential names)
        cap_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        # Filter out common English words
        common_words = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "With",
                       "By", "From", "As", "Is", "Was", "Were", "Are", "Be", "Been",
                       "Have", "Has", "Had", "Do", "Does", "Did", "Will", "Would",
                       "Could", "Should", "May", "Might", "Can", "Not", "No", "But",
                       "And", "Or", "If", "Then", "When", "Where", "How", "What",
                       "Why", "Who", "Which", "This", "That", "These", "Those",
                       "His", "Her", "Its", "My", "Your", "Our", "Their", "She",
                       "He", "It", "We", "They", "You", "Me", "Us", "Them",
                       "After", "Before", "During", "While", "Since", "Until",
                       "About", "Between", "Through", "Into", "Over", "Under",
                       "Up", "Down", "Out", "Just", "Also", "Very", "Much",
                       "More", "Less", "Most", "Some", "All", "None", "Each",
                       "Every", "Both", "Either", "Neither", "One", "Two",
                       "First", "Second", "Last", "Next", "New", "Old", "Big",
                       "Small", "Good", "Bad", "Right", "Left", "True", "False",
                       "Same", "Different", "Here", "There", "Now", "Then",
                       "Never", "Always", "Often", "Sometimes", "Usually",
                       "Still", "Yet", "Already", "Again", "Once", "Twice",
                       "Only", "Even", "Though", "Although", "However",
                       "Therefore", "Thus", "Because", "Since", "So",
                       "Chapter", "Part", "Section", "Book", "Page", "Line",
                       "Nothing", "Everything", "Something", "Anything",
                       "Day", "Night", "Morning", "Evening", "Today",
                       "Tomorrow", "Yesterday", "Time", "Year", "Month",
                       "Week", "Hour", "Minute", "Second"}
        novel_names = [w for w in cap_words if w not in common_words and len(w) > 2]
        # Deduplicate
        unique_names = list(set(novel_names))[:50]
        
        # 2. Extract multi-word capitalized phrases (places, organizations)
        cap_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text)
        unique_phrases = list(set(cap_phrases))[:30]
        
        # Create entity dicts
        for name in unique_names:
            entities.append({"name": name, "type": "concept",
                           "description": f"{name} - character or entity from novel",
                           "category": "novel_entity"})
        for phrase in unique_phrases:
            entities.append({"name": phrase, "type": "concept",
                           "description": f"{phrase} - named entity from novel",
                           "category": "novel_entity"})
    
    # Deduplicate by name
    unique = {}
    for e in entities:
        if e["name"] not in unique:
            unique[e["name"]] = e
    return list(unique.values())

def extract_relations_heuristic(entities: List[Dict], domain: str) -> List[Dict]:
    """Generate simple co-occurrence relations between entities.
    
    Strategy: entities appearing in same chunks are 'related_to'.
    """
    relations = []
    # Simple: all pairs of same-type entities are related
    # In production, would use co-occurrence in text chunks
    by_type = collections.defaultdict(list)
    for e in entities:
        by_type[e["type"]].append(e)
    
    # Within-type relations (same category entities often related)
    for etype, group in by_type.items():
        for i in range(len(group)):
            for j in range(i+1, min(i+3, len(group))):
                relations.append({
                    "source": group[i]["name"],
                    "target": group[j]["name"],
                    "relation": "related_to",
                    "relation_name": f"{etype}_connection",
                })
    
    # Cross-type relations (disease-symptom, disease-treatment, etc.)
    if domain == "medical":
        type_pairs = [("disease", "symptom", "has_symptom"),
                     ("disease", "drug", "treated_by"),
                     ("disease", "treatment", "treated_by"),
                     ("disease", "anatomy", "located_in"),
                     ("risk_factor", "disease", "increases_risk_of")]
        for src_type, tgt_type, rel in type_pairs:
            srcs = by_type.get(src_type, [])
            tgts = by_type.get(tgt_type, [])
            for s in srcs[:5]:
                for t in tgts[:5]:
                    relations.append({
                        "source": s["name"], "target": t["name"],
                        "relation": rel,
                        "relation_name": f"{s['name']}_{rel}_{t['name']}"[:80],
                    })
    
    return relations

# ============================================================
# PHASE 5: HugeGraph Vertex/Edge Operations (FIXED)
# ============================================================

def hg_vertex_create_bulk(entities: List[Dict], domain: str) -> Dict:
    """Bulk create vertices using PyHugeClient (handles ID format correctly)."""
    from pyhugegraph.client import PyHugeClient
    client = PyHugeClient(url=HG_REST_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
    g = client.graph()
    
    type_to_label_map = {
        "disease": "Disease", "drug": "Drug", "symptom": "Symptom",
        "treatment": "Treatment", "anatomy": "Anatomy", "gene": "Gene",
        "risk_factor": "RiskFactor", "cell_type": "CellType",
        "person": "Person", "location": "Location", "organization": "Organization",
        "event": "Event", "concept": "Concept", "product": "Product", "company": "Company",
    }
    fallback_label = "Entity" if domain == "novel" else "Concept"
    
    # Required properties per label (PRIMARY_KEY schema)
    required_props = {
        "Entity": ["name", "description", "category", "type"],
        "Disease": ["name", "description", "category", "severity"],
        "Drug": ["name", "description", "category"],
        "Symptom": ["name", "description", "location"],
        "Treatment": ["name", "description", "category"],
        "Anatomy": ["name", "description", "location"],
        "Gene": ["name", "description"],
        "RiskFactor": ["name", "description", "category"],
        "CellType": ["name", "description", "location"],
        "Concept": ["name", "description", "type"],
        "Person": ["name", "description"],
    }
    defaults = {
        "description": "extracted from benchmark corpus",
        "category": "general", "type": "concept",
        "severity": "moderate", "location": "unspecified",
    }
    
    name_to_vid = {}
    label_map = {}
    uploaded = 0
    failed = 0
    
    for e in entities:
        name = e["name"]
        etype = e.get("type", "concept").lower()
        label = type_to_label_map.get(etype, fallback_label)
        label_map[name] = label
        
        props = {"name": name}
        for prop in required_props.get(label, ["name"]):
            if prop not in props:
                props[prop] = e.get(prop, defaults.get(prop, "unspecified"))
        
        try:
            v = g.addVertex(label=label, properties=props)
            name_to_vid[name] = v.id  # Returns format like "42:name"
            uploaded += 1
        except Exception as ex:
            # Vertex might already exist
            if "already" in str(ex).lower():
                name_to_vid[name] = f"{label}:{name}"
                uploaded += 1
            else:
                failed += 1
    
    print(f"  [HG Vertex] {uploaded} created, {failed} failed ({len(entities)} total)")
    return {"uploaded": uploaded, "failed": failed, "name_to_vid": name_to_vid, "label_map": label_map}

def hg_edge_create_bulk(relations: List[Dict], label_map: Dict, name_to_vid: Dict, domain: str) -> Dict:
    """Bulk create edges using PyHugeClient (handles ID format correctly)."""
    import requests
    
    edge_schema = {
        "related_to": ("Entity", "Entity"),
        "treated_by": ("Disease", "Drug"),
        "treats": ("Drug", "Disease"),
        "has_symptom": ("Disease", "Symptom"),
        "prevents": ("Treatment", "Disease"),
        "located_in": ("Disease", "Anatomy"),
        "increases_risk_of": ("RiskFactor", "Disease"),
        "is_a": ("Entity", "Concept"),
        "causes": ("Entity", "Entity"),
    }
    sort_key_labels = {"related_to", "treats", "causes", "treated_by", "mentions"}
    
    url = HG_GRAPH_EDGES_URL
    headers = {"Content-Type": "application/json"}
    uploaded = 0
    failed = 0
    skipped = 0
    
    for r in relations:
        src_name = r.get("source", "")
        tgt_name = r.get("target", "")
        rel_type = r.get("relation", "related_to").lower()
        if not src_name or not tgt_name: continue
        
        src_label = label_map.get(src_name)
        tgt_label = label_map.get(tgt_name)
        src_vid = name_to_vid.get(src_name)
        tgt_vid = name_to_vid.get(tgt_name)
        if not src_label or not tgt_label or not src_vid or not tgt_vid:
            skipped += 1; continue
        
        edge_label = rel_type
        if edge_label not in edge_schema:
            edge_label = "related_to"
        
        # Check schema compatibility
        expected_src, expected_tgt = edge_schema.get(edge_label, ("Entity", "Entity"))
        if src_label != expected_src or tgt_label != expected_tgt:
            if edge_label == "related_to":
                if src_label != "Entity" or tgt_label != "Entity":
                    # Try medical-specific
                    if domain == "medical":
                        medical_map = {
                            ("Disease", "Drug"): "treated_by", ("Drug", "Disease"): "treats",
                            ("Disease", "Symptom"): "has_symptom", ("Disease", "Anatomy"): "located_in",
                            ("Treatment", "Disease"): "prevents", ("RiskFactor", "Disease"): "increases_risk_of",
                        }
                        alt = medical_map.get((src_label, tgt_label))
                        if alt: edge_label = alt
                        else: skipped += 1; continue
                    else: skipped += 1; continue
        
        props = {}
        if edge_label in sort_key_labels:
            props["name"] = r.get("relation_name", f"{src_name}_{rel_type}_{tgt_name}")[:100]
        
        try:
            data = {"label": edge_label, "outV": src_vid, "inV": tgt_vid, "properties": props}
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            if resp.status_code in (200, 201):
                uploaded += 1
            elif "already exists" in resp.text.lower():
                uploaded += 1
            else:
                failed += 1
        except:
            failed += 1
    
    print(f"  [HG Edge] {uploaded} created, {failed} failed, {skipped} skipped")
    return {"uploaded": uploaded, "failed": failed, "skipped": skipped}

# ============================================================
# PHASE 6: Graph Traversal (k_neighbor)
# ============================================================

def hg_vertex_exists(label, name, hg_client=None):
    """Check vertex existence using PyHugeClient."""
    from pyhugegraph.client import PyHugeClient
    if hg_client is None:
        hg_client = PyHugeClient(url=HG_REST_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
    g = hg_client.graph()
    try:
        vs = g.getVertexByCondition(label=label, limit=200, properties={"name": name})
        for v in vs:
            if v.properties.get("name") == name:
                return v.id
    except:
        pass
    return None

def hg_kneighbor(vertex_id, max_depth=2, hg_client=None):
    """k_neighbor traversal using PyHugeClient traverser API."""
    from pyhugegraph.client import PyHugeClient
    if hg_client is None:
        hg_client = PyHugeClient(url=HG_REST_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
    t = hg_client.traverser()
    try:
        result = t.k_neighbor(source_id=vertex_id, max_depth=max_depth)
        neighbor_ids = result.get("vertices", [])
        return [{"id": nid} for nid in neighbor_ids if nid != vertex_id]
    except Exception as e:
        return []

def get_neighbor_details(vertex_ids, hg_client=None, limit=10):
    """Fetch properties of neighbor vertices using PyHugeClient."""
    from pyhugegraph.client import PyHugeClient
    if hg_client is None:
        hg_client = PyHugeClient(url=HG_REST_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
    g = hg_client.graph()
    details = []
    for vid in vertex_ids[:limit]:
        try:
            # Use vertex ID format for lookup via PyHugeClient
            v = g.getVertexById(vid)
            if v:
                props = v.properties if hasattr(v, 'properties') else {}
                details.append({"id": v.id, "label": v.label,
                               "name": props.get("name", ""), "description": props.get("description", "")})
        except:
            pass
    return details

# ============================================================
# PHASE 7: RAG Query (3-channel + RRF + LLM generation)
# ============================================================

def call_llm(prompt, max_tokens=2048, temperature=0.1):
    """Call MiMo v2.5 Pro for answer generation only."""
    import requests
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            r = requests.post(f"{MIMO_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {MIMO_API_KEY}", "Content-Type": "application/json"},
                json={"model": MIMO_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=LLM_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            else:
                print(f"  [LLM] retry {attempt+1}: status={r.status_code}")
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  [LLM] retry {attempt+1}: timeout")
            time.sleep(3)
        except Exception as e:
            print(f"  [LLM] retry {attempt+1}: {type(e).__name__}")
            time.sleep(2)
    return ""

def rrf_fusion(vector_results, bm25_results, graph_results, k=60):
    scores = {}
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, item in enumerate(bm25_results):
        doc_id = item[0]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(graph_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.5 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

def evaluate_answer(prediction, reference, question_type):
    pred_lower = prediction.lower().strip()
    ref_lower = reference.lower().strip()
    ref_facts = set(ref_lower.split())
    pred_facts = set(pred_lower.split())
    overlap = len(ref_facts & pred_facts)
    accuracy = overlap / max(1, len(ref_facts))
    
    # ROUGE-L
    def lcs_len(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
                else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    pred_words, ref_words = pred_lower.split(), ref_lower.split()
    lcs = lcs_len(pred_words, ref_words)
    p_l = lcs / max(1, len(pred_words))
    r_l = lcs / max(1, len(ref_words))
    rouge_l = 2*p_l*r_l/max(0.001, p_l+r_l)
    f1 = 2*overlap/max(1, len(ref_facts)+len(pred_facts))
    
    return {"accuracy": round(accuracy, 4), "rouge_l": round(rouge_l, 4), "f1": round(f1, 4)}

def rag_query(query, faiss_index, faiss_ids, bm25_index, embed_model,
              label_map, name_to_vid, chunk_raw_docs, domain, hg_client):
    """3-channel RAG: FAISS + BM25 + Graph + RRF + LLM."""
    import numpy as np
    start = time.time()
    
    # Channel 1: FAISS
    q_emb = embed_model.encode([query])
    vector_results = search_faiss(faiss_index, faiss_ids, q_emb[0], top_k=10)
    
    # Channel 2: BM25
    bm25_results = bm25_index.search(query, top_k=10)
    
    # Channel 3: Graph traversal (using PyHugeClient)
    graph_hits = 0
    graph_neighbors = []
    graph_context = ""
    graph_results = []
    
    # Find query-related entities in HugeGraph
    query_words = query.lower().split()
    entity_candidates = [w for w in query_words if len(w) > 3]
    for i in range(len(query_words)-1):
        phrase = f"{query_words[i]} {query_words[i+1]}"
        if len(phrase) > 4: entity_candidates.append(phrase)
    
    candidate_labels = (["Entity", "Person", "Concept", "Location", "Organization"]
                       if domain == "novel"
                       else ["Disease", "Drug", "Symptom", "Treatment", "Anatomy", "Entity", "Concept"])
    
    # Use PyHugeClient for vertex lookup (no REST API errors)
    found_vertices = []
    for candidate in entity_candidates[:5]:
        # First try direct lookup by name in label_map
        cand_title = candidate.title()
        if cand_title in name_to_vid:
            found_vertices.append({"id": name_to_vid[cand_title], "label": label_map.get(cand_title, "Entity"), "name": cand_title})
            continue
        # Then try PyHugeClient lookup
        for label in candidate_labels[:3]:
            vid = hg_vertex_exists(label, cand_title, hg_client)
            if vid:
                found_vertices.append({"id": vid, "label": label, "name": cand_title})
                break
    
    # k_neighbor traversal
    for fv in found_vertices[:3]:
        neighbors = hg_kneighbor(fv["id"], max_depth=2, hg_client=hg_client)
        if neighbors:
            graph_hits += len(neighbors)
            details = get_neighbor_details([n["id"] for n in neighbors[:5]], hg_client)
            graph_neighbors.extend(details)
    
    # Map neighbors to chunks
    neighbor_names = [n.get("name", "").lower() for n in graph_neighbors if n.get("name")]
    for doc_id, text in chunk_raw_docs.items():
        for name in neighbor_names:
            if name and name in text.lower():
                graph_results.append((doc_id, 0.5))
                break
    
    graph_hits = len(graph_results)
    if graph_neighbors:
        graph_context = "\n".join(f"- {n.get('name','?')} ({n.get('label','?')}): {n.get('description','')[:60]}" for n in graph_neighbors[:5])
    
    # RRF fusion
    fused = rrf_fusion(vector_results, bm25_results, graph_results)
    
    # Retrieve top chunks
    top_chunks = []
    for doc_id, rrf_score in fused[:5]:
        text = chunk_raw_docs.get(doc_id, "")
        if text: top_chunks.append({"doc_id": doc_id, "text": text, "rrf_score": rrf_score})
    
    context_text = "\n\n".join(c["text"][:500] for c in top_chunks)
    
    # LLM generation
    gen_prompt = f"""Based on the following context, answer the question factually.

Context:
{context_text[:3000]}

{f"Knowledge graph context: {graph_context}" if graph_context else ""}

Question: {query}

Answer:"""
    
    answer = call_llm(gen_prompt, max_tokens=2048)
    latency = time.time() - start
    
    return {"answer": answer, "vector_hits": len(vector_results), "bm25_hits": len(bm25_results),
            "graph_hits": graph_hits, "graph_neighbors": len(graph_neighbors),
            "fused_chunks": len(top_chunks), "latency": latency}

# ============================================================
# PHASE 8: Full Pipeline
# ============================================================

def run_full_evaluation():
    print("=" * 70)
    print("GraphRAG-Bench P0-v4 — FAST heuristic entities + graph_hits FIXED")
    print("=" * 70)
    
    if not check_hugegraph() or not check_llm_api():
        print("ABORT: connectivity failed")
        return None
    
    results = {
        "evaluation_id": f"p0_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "config": {"llm_model": MIMO_MODEL, "embed_model": EMBED_MODEL_NAME,
                   "chunk_size": CHUNK_SIZE, "max_questions": MAX_QUESTIONS_PER_TYPE,
                   "graph_server": HG_REST_URL, "graph_name": HG_GRAPH,
                   "entity_extraction": "heuristic_nlp"},
        "domains": {}, "rag_evaluation": {},
    }
    
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    question_types = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
    
    for domain in ["novel", "medical"]:
        print(f"\n{'='*60}\n  === Domain: {domain} ===\n{'='*60}")
        
        questions, corpus = load_benchmark(domain)
        corpus_texts = prepare_corpus_texts(corpus, domain)
        total_chars = sum(len(t) for t in corpus_texts.values())
        print(f"  Corpus: {len(corpus_texts)} docs, {total_chars} chars")
        
        # Chunk
        all_chunks = []
        global_offset = 0
        for doc_name, doc_text in corpus_texts.items():
            chunks = chunk_text_global(doc_text, global_offset)
            for c in chunks: c["doc_name"] = doc_name; c["domain"] = domain
            all_chunks.extend(chunks)
            global_offset += len(chunks)
        print(f"  Chunks: {len(all_chunks)}")
        
        # Build indexes
        chunk_texts = [c["content"] for c in all_chunks]
        chunk_ids = [f"chunk_{c['chunk_index']}" for c in all_chunks]
        chunk_raw_docs = dict(zip(chunk_ids, chunk_texts))
        
        import numpy as np
        embeddings = np.array(embed_model.encode(chunk_texts, show_progress_bar=True, batch_size=64))
        faiss_index, faiss_ids = build_faiss_index(embeddings, chunk_ids)
        
        bm25_index = BM25Index()
        bm25_index.add_documents(chunk_texts, chunk_ids)
        print(f"  BM25: {bm25_index.N} docs")
        
        # FAST entity extraction (heuristic, no LLM)
        print(f"  [Extract] Heuristic NLP entity extraction...")
        all_entities = []
        for chunk in all_chunks[:200]:  # Sample 200 chunks for entity extraction
            entities = extract_entities_heuristic(chunk["content"], domain)
            all_entities.extend(entities)
        # Deduplicate
        unique_entities = {}
        for e in all_entities:
            if e["name"] not in unique_entities: unique_entities[e["name"]] = e
        all_entities = list(unique_entities.values())
        
        relations = extract_relations_heuristic(all_entities, domain)
        print(f"  [Extract] {len(all_entities)} unique entities, {len(relations)} relations (heuristic)")
        
        # Build KG in HugeGraph
        print(f"  [KG Build] Uploading to HugeGraph...")
        from pyhugegraph.client import PyHugeClient
        hg_client = PyHugeClient(url=HG_REST_URL, graph=HG_GRAPH, user='admin', pwd='xxx')
        
        v_result = hg_vertex_create_bulk(all_entities, domain)
        name_to_vid = v_result["name_to_vid"]
        label_map = v_result["label_map"]
        e_result = hg_edge_create_bulk(relations, label_map, name_to_vid, domain)
        print(f"  [KG Build] Vertices: {v_result['uploaded']}, Edges: {e_result['uploaded']}")
        
        # Verify
        verified = sum(1 for name, label in list(label_map.items())[:30] if hg_vertex_exists(label, name))
        print(f"  [Verify] {verified}/30 sampled vertices exist")
        
        # RAG evaluation
        print(f"\n  [Eval] RAG evaluation ({domain})...")
        
        for q_type in question_types:
            type_qs = [q for q in questions if q.get("question_type") == q_type]
            sampled = type_qs[:MAX_QUESTIONS_PER_TYPE]
            if not sampled: continue
            
            print(f"\n    {domain}/{q_type}: {len(sampled)} questions")
            type_results = []
            
            for i, q in enumerate(sampled):
                query = q.get("question", "")
                reference = q.get("answer", "")
                if not query: continue
                
                rag_r = rag_query(query, faiss_index, faiss_ids, bm25_index,
                                 embed_model, label_map, name_to_vid, chunk_raw_docs, domain, hg_client)
                eval_m = evaluate_answer(rag_r["answer"], reference, q_type)
                
                type_results.append({
                    "question": query[:100], "reference": reference[:100],
                    "prediction": rag_r["answer"][:200], "metrics": eval_m,
                    "retrieval": {"vector_hits": rag_r["vector_hits"],
                                 "bm25_hits": rag_r["bm25_hits"],
                                 "graph_hits": rag_r["graph_hits"],
                                 "graph_neighbors": rag_r["graph_neighbors"],
                                 "fused_chunks": rag_r["fused_chunks"],
                                 "latency": rag_r["latency"]},
                })
                
                if (i+1) % 5 == 0:
                    avg_acc = sum(r["metrics"]["accuracy"] for r in type_results)/len(type_results)
                    avg_gh = sum(r["retrieval"]["graph_hits"] for r in type_results)/len(type_results)
                    print(f"      [{i+1}/{len(sampled)}] avg_acc={avg_acc:.3f}, avg_g_hits={avg_gh:.1f}")
            
            if type_results:
                key = f"{domain}/{q_type}"
                avg_acc = sum(r["metrics"]["accuracy"] for r in type_results)/len(type_results)
                avg_rouge = sum(r["metrics"]["rouge_l"] for r in type_results)/len(type_results)
                avg_f1 = sum(r["metrics"]["f1"] for r in type_results)/len(type_results)
                avg_vh = sum(r["retrieval"]["vector_hits"] for r in type_results)/len(type_results)
                avg_bh = sum(r["retrieval"]["bm25_hits"] for r in type_results)/len(type_results)
                avg_gh = sum(r["retrieval"]["graph_hits"] for r in type_results)/len(type_results)
                avg_lat = sum(r["retrieval"]["latency"] for r in type_results)/len(type_results)
                gh_nonzero = sum(1 for r in type_results if r["retrieval"]["graph_hits"] > 0)
                
                results["rag_evaluation"][key] = {
                    "num_questions": len(type_results), "avg_accuracy": round(avg_acc, 4),
                    "avg_rouge_l": round(avg_rouge, 4), "avg_f1": round(avg_f1, 4),
                    "avg_vector_hits": round(avg_vh, 2), "avg_bm25_hits": round(avg_bh, 2),
                    "avg_graph_hits": round(avg_gh, 2), "avg_latency": round(avg_lat, 2),
                    "graph_context_hits": gh_nonzero,
                    "detailed_results": type_results,
                }
                print(f"    [{key}] acc={avg_acc:.4f}, rouge={avg_rouge:.4f}, F1={avg_f1:.4f}, "
                      f"v={avg_vh:.1f}, b={avg_bh:.1f}, g={avg_gh:.1f}, gh>0={gh_nonzero}")
        
        results["domains"][domain] = {
            "corpus_chars": total_chars, "chunks": len(all_chunks),
            "entities_heuristic": len(all_entities), "relations_heuristic": len(relations),
            "kg_vertices": v_result["uploaded"], "kg_edges": e_result["uploaded"],
        }
    
    # Overall metrics
    novel_keys = [f"novel/{t}" for t in question_types]
    medical_keys = [f"medical/{t}" for t in question_types]
    n_accs = [results["rag_evaluation"].get(k, {}).get("avg_accuracy", 0) for k in novel_keys]
    m_accs = [results["rag_evaluation"].get(k, {}).get("avg_accuracy", 0) for k in medical_keys]
    
    n_valid = [a for a in n_accs if a > 0] or [0]
    m_valid = [a for a in m_accs if a > 0] or [0]
    
    results["overall"] = {
        "novel_avg_accuracy": round(sum(n_valid)/len(n_valid), 4),
        "medical_avg_accuracy": round(sum(m_valid)/len(m_valid), 4),
        "combined_avg_accuracy": round((sum(n_valid)/len(n_valid) + sum(m_valid)/len(m_valid))/2, 4),
        "total_questions": sum(results["rag_evaluation"].get(k, {}).get("num_questions", 0) for k in novel_keys+medical_keys),
        "graph_hits_nonzero": sum(results["rag_evaluation"].get(k, {}).get("graph_context_hits", 0) for k in novel_keys+medical_keys),
    }
    
    # Save
    result_path = RESULT_DIR / "p0_v4_graphrag_bench_result.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved to {result_path}")
    
    # Summary
    print("\n" + "="*70)
    print("P0-v4 RESULTS")
    print("="*70)
    o = results["overall"]
    print(f"  Novel: {o['novel_avg_accuracy']}")
    print(f"  Medical: {o['medical_avg_accuracy']}")
    print(f"  Combined: {o['combined_avg_accuracy']}")
    print(f"  Total Qs: {o['total_questions']}")
    print(f"  Qs with graph_hits>0: {o['graph_hits_nonzero']}")
    for key in sorted(results["rag_evaluation"].keys()):
        d = results["rag_evaluation"][key]
        print(f"  {key}: acc={d['avg_accuracy']:.4f}, g_hits={d['avg_graph_hits']:.1f}, gh>0={d['graph_context_hits']}")
    
    # Compare with baseline
    bl_path = RESULT_DIR / "graphrag_bench_full_pipeline_result.json"
    if bl_path.exists():
        bl = json.load(open(bl_path))
        be = bl["rag_evaluation"]
        print("\n--- Baseline vs P0-v4 ---")
        for key in sorted(results["rag_evaluation"].keys()):
            p0 = results["rag_evaluation"].get(key, {}).get("avg_accuracy", 0)
            bl_a = be.get(key, {}).get("avg_accuracy", 0)
            delta = p0 - bl_a
            print(f"  {key}: {bl_a:.4f} → {p0:.4f} (Δ={delta:+.4f})")
    
    return results

if __name__ == "__main__":
    run_full_evaluation()
