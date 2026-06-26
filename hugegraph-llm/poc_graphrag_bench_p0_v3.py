#!/usr/bin/env python3
"""
GraphRAG-Bench P0-Improved Evaluation v3 — graph_hits FIXED
=============================================================
Root causes of graph_hits=0:
  1. Entity vertex creation missing required properties (description, category, type)
  2. Edge creation missing sort_key (name) for related_to label
  3. Vertex lookup flooding 10730 ERRORs → process died after 1 question
  4. No benchmark KG data in HugeGraph at all (only 40 old PoC vertices)

Fixes:
  1. Entity/Disease/Drug/Symptom vertex with ALL required properties
  2. Edge creation matching schema (related_to: Entity→Entity, sort_keys=['name'])
  3. Domain-specific edge labels for medical (treated_by, has_symptom, prevents, etc.)
  4. k_neighbor traversal with graceful fallback (skip if vertex not found)
  5. 3-channel retrieval: FAISS + BM25 + Graph traversal (RRF fusion)
  6. LLM timeout + retry logic

PoC Redline compliance: RL-P1~P10
"""

import json, time, os, sys, math, hashlib, traceback, re, collections
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set

# ── Project paths ──
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
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MAX_QUESTIONS_PER_TYPE = 15  # 4 types × 15 = 60 per domain, 120 total
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 2

# ── Embedding model ──
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# ============================================================
# PHASE 1: Connectivity Checks
# ============================================================

def check_hugegraph():
    """Verify HugeGraph Server is reachable."""
    import requests
    try:
        r = requests.get(f"{HG_REST_URL}/graphs/{HG_GRAPH}", timeout=10)
        ok = r.status_code == 200
        print(f"[Phase 1] HugeGraph Server: {'OK' if ok else 'FAIL'} (status={r.status_code})")
        return ok
    except Exception as e:
        print(f"[Phase 1] HugeGraph Server: FAIL ({e})")
        return False

def check_llm_api():
    """Verify MiMo v2.5 Pro API is reachable."""
    import requests
    try:
        r = requests.post(
            f"{MIMO_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MIMO_API_KEY}", "Content-Type": "application/json"},
            json={"model": MIMO_MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 32},
            timeout=30,
        )
        ok = r.status_code == 200
        print(f"[Phase 1] MiMo API: {'OK' if ok else 'FAIL'} (status={r.status_code})")
        return ok
    except Exception as e:
        print(f"[Phase 1] MiMo API: FAIL ({e})")
        return False

# ============================================================
# PHASE 2: Data Loading + Chunking
# ============================================================

def load_benchmark(domain="novel"):
    """Load GraphRAG-Bench questions and corpus."""
    q_path = BENCH_ROOT / "Datasets" / "Questions" / f"{domain}_questions.json"
    c_path = BENCH_ROOT / "Datasets" / "Corpus" / f"{domain}.json"
    questions = json.load(open(q_path))
    corpus = json.load(open(c_path))
    return questions, corpus

def prepare_corpus_texts(corpus, domain="novel"):
    """Extract and flatten corpus into doc_name -> text mapping."""
    texts = {}
    if isinstance(corpus, list):
        for doc in corpus:
            name = doc.get("corpus_name", f"doc_{len(texts)}")
            ctx = doc.get("context", "")
            if ctx:
                texts[name] = ctx
    elif isinstance(corpus, dict):
        ctx = corpus.get("context", "")
        if ctx:
            texts[f"{domain}_corpus"] = ctx
    return texts

def chunk_text_global(text: str, global_offset: int, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks with globally unique indices."""
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, max(1, len(text)), step):
        ct = text[i:i + chunk_size]
        if ct.strip():
            chunks.append({
                "content": ct,
                "chunk_index": global_offset + len(chunks),
                "start_char": i,
                "end_char": min(i + chunk_size, len(text)),
            })
    return chunks

# ============================================================
# PHASE 3: Index Building (FAISS + BM25)
# ============================================================

class BM25Index:
    """Simple BM25 fulltext index using jieba + TF-IDF scoring."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}
        self.raw_docs = {}
        self.avg_dl = 0
        self.N = 0
        self.df = collections.Counter()
        self._dirty = True

    def tokenize(self, text):
        try:
            import jieba
            return [w for w in jieba.cut(text.lower()) if len(w) > 1]
        except ImportError:
            return [w for w in re.findall(r'\w+', text.lower()) if len(w) > 1]

    def add_documents(self, texts, ids=None):
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(texts))]
        for text, doc_id in zip(texts, ids):
            tokens = self.tokenize(text)
            self.docs[doc_id] = tokens
            self.raw_docs[doc_id] = text
            self.df.update(set(tokens))
            self.N += 1
        total_dl = sum(len(t) for t in self.docs.values())
        self.avg_dl = total_dl / max(1, self.N)
        self._dirty = False

    def search(self, query, top_k=10):
        if self._dirty or self.N == 0:
            return []
        q_tokens = self.tokenize(query)
        scores = {}
        for doc_id, doc_tokens in self.docs.items():
            dl = len(doc_tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in self.df:
                    continue
                tf = doc_tokens.count(qt)
                df = self.df[qt]
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avg_dl))
                score += idf * numerator / denominator
            scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(doc_id, score, self.raw_docs[doc_id]) for doc_id, score in ranked]

def build_faiss_index(embeddings, ids):
    """Build FAISS index from pre-computed embeddings."""
    import numpy as np
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity (after normalization)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"  [FAISS] Built index: {index.ntotal} vectors, dim={dim}")
    return index, ids

def embed_texts(texts, model_name=EMBED_MODEL_NAME):
    """Embed texts using sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    print(f"  [Embed] Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  [Embed] Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings

def search_faiss(index, ids, query_embedding, top_k=10):
    """Search FAISS index for similar vectors."""
    import numpy as np
    import faiss
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(ids):
            results.append((ids[idx], float(scores[0][i])))
    return results

# ============================================================
# PHASE 4: Knowledge Graph Construction (FIXED)
# ============================================================

def call_llm(prompt, max_tokens=2048, temperature=0.1):
    """Call MiMo v2.5 Pro with retry logic."""
    import requests
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{MIMO_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {MIMO_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MIMO_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=LLM_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                print(f"  [LLM] Attempt {attempt+1}: status={r.status_code}, resp={r.text[:100]}")
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  [LLM] Attempt {attempt+1}: timeout ({LLM_TIMEOUT}s)")
            time.sleep(3)
        except Exception as e:
            print(f"  [LLM] Attempt {attempt+1}: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)
    return ""

ENTITY_EXTRACT_PROMPT = """Extract entities and their relationships from the following text chunk.

Return a JSON array with this exact format:
[
  {"name": "entity_name", "type": "person|location|organization|disease|drug|symptom|treatment|gene|concept|event|anatomy|risk_factor|product|company", "description": "brief description of this entity", "category": "domain category"},
  ...
]

Then a second JSON array for relationships:
[
  {"source": "entity1_name", "target": "entity2_name", "relation": "relation_type", "relation_name": "short name for this relation"},
  ...
]

Rules:
- Entity names must be specific (e.g. "Basal cell carcinoma", not "cancer")
- Each entity MUST have name, type, description, category
- description: 1-2 sentence factual description
- category: domain grouping (e.g. "medical_condition", "pharmaceutical", "character", "location")
- relation types: treats, causes, prevents, has_symptom, located_in, is_a, related_to, part_of, works_at, belongs_to, mentions
- Return ONLY valid JSON, no other text

Text chunk:
"""

def extract_entities_from_chunks(chunks, batch_size=3):
    """Extract entities from text chunks using LLM."""
    all_entities = []
    all_relations = []
    total_chunks = len(chunks)

    for batch_start in range(0, total_chunks, batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        combined_text = "\n---\n".join(c["content"][:600] for c in batch)
        prompt = ENTITY_EXTRACT_PROMPT + combined_text[:3000]

        response = call_llm(prompt, max_tokens=4096)
        if not response:
            continue

        # Parse JSON from response
        try:
            # Find JSON arrays in response
            json_blocks = re.findall(r'\[[\s\S]*?\]', response)
            if len(json_blocks) >= 1:
                entities = json.loads(json_blocks[0])
                for e in entities:
                    if isinstance(e, dict) and e.get("name"):
                        e["name"] = str(e["name"]).strip()
                        e["type"] = str(e.get("type", "concept")).strip().lower()
                        e["description"] = str(e.get("description", f"Entity: {e['name']}")).strip()
                        e["category"] = str(e.get("category", "general")).strip()
                        all_entities.append(e)
            if len(json_blocks) >= 2:
                relations = json.loads(json_blocks[1])
                for r in relations:
                    if isinstance(r, dict) and r.get("source") and r.get("target"):
                        all_relations.append(r)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  [Extract] Batch {batch_start}: JSON parse error ({type(e).__name__})")
            continue

        if (batch_start + batch_size) % 15 == 0 or batch_start + batch_size >= total_chunks:
            print(f"  [Extract] {min(batch_start + batch_size, total_chunks)}/{total_chunks} chunks processed, "
                  f"{len(all_entities)} entities, {len(all_relations)} relations")

    return all_entities, all_relations

# ── HugeGraph Vertex/Edge Operations (FIXED) ──

def hg_vertex_create(label: str, properties: Dict) -> Optional[str]:
    """Create a vertex in HugeGraph with ALL required properties.
    
    Returns vertex ID if successful, None otherwise.
    Uses PRIMARY_KEY strategy: ID = label:primary_key_value
    """
    import requests
    # Ensure ALL required properties are present
    # Different labels have different required property sets
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
        "Location": ["name", "country", "description", "region"],
        "Organization": ["name", "country", "description", "type"],
        "Event": ["name", "description", "date", "location"],
        "Document": ["name", "description", "category"],
        "Person": ["name", "description"],
        "Company": ["name", "description", "revenue", "market_share", "stock_code"],
        "Product": ["name", "description", "param_count", "users", "duration"],
    }
    
    req_props = required_props.get(label, ["name"])
    for prop in req_props:
        if prop not in properties:
            # Fill with sensible defaults
            defaults = {
                "description": f"{properties.get('name', 'unknown')} - extracted from benchmark corpus",
                "category": properties.get("type", "general"),
                "type": properties.get("category", "concept"),
                "severity": "moderate",
                "location": "unspecified",
                "country": "unspecified",
                "region": "unspecified",
                "date": "unspecified",
                "revenue": 0.0,
                "market_share": 0.0,
                "stock_code": "N/A",
                "param_count": "0",
                "users": "unspecified",
                "duration": "unspecified",
            }
            properties[prop] = defaults.get(prop, "unspecified")
    
    # Ensure property types match schema (TEXT for most, DOUBLE/INT for specific)
    type_conversions = {
        "revenue": float,
        "market_share": float,
        "growth_rate": float,
        "amount": float,
        "param_count": int,
        "category_count": int,
        "importance": float,
        "sprint_num": int,
        "access_count": int,
        "initial_score": float,
        "decay_score": float,
    }
    for prop, conv in type_conversions.items():
        if prop in properties:
            try:
                properties[prop] = conv(properties[prop])
            except (ValueError, TypeError):
                properties[prop] = conv(0) if conv != str else "unspecified"
    
    url = f"{HG_REST_URL}/graphs/{HG_GRAPH}/vertices"
    headers = {"Content-Type": "application/json"}
    data = {"label": label, "properties": properties}
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        if r.status_code in (200, 201):
            vid = r.json().get("id", f"{label}:{properties.get('name', '')}")
            return vid
        else:
            # Vertex might already exist (duplicate primary key)
            if "already exists" in r.text.lower():
                return f"{label}:{properties.get('name', '')}"
            return None
    except Exception as e:
        print(f"  [HG Vertex] Create {label}/{properties.get('name','?')} error: {str(e)[:60]}")
        return None

def hg_edge_create(edge_label: str, src_label: str, src_name: str, tgt_label: str, tgt_name: str, 
                   properties: Dict = None):
    """Create an edge in HugeGraph matching schema constraints.
    
    For related_to: Entity→Entity with sort_keys=['name']
    For domain-specific edges: uses correct label mappings
    """
    import requests
    
    sort_key_labels = {"related_to", "treats", "causes", "mentions", "treated_by",
                       "invests_in", "reports_revenue", "has_market_share_in", "supplies_to",
                       "subject_of", "object_of", "precedes", "supersedes", "conflicts_with"}
    
    if properties is None:
        properties = {}
    
    # Sort key required: add 'name' property for sort_key labels
    if edge_label in sort_key_labels and "name" not in properties:
        properties["name"] = f"{src_name}_{edge_label}_{tgt_name}"
    
    url = f"{HG_REST_URL}/graphs/{HG_GRAPH}/edges"
    headers = {"Content-Type": "application/json"}
    data = {
        "label": edge_label,
        "outV": f"{src_label}:{src_name}",
        "inV": f"{tgt_label}:{tgt_name}",
        "properties": properties,
    }
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        if r.status_code in (200, 201):
            return True
        else:
            if "already exists" in r.text.lower():
                return True
            return False
    except Exception as e:
        print(f"  [HG Edge] Create {edge_label} error: {str(e)[:60]}")
        return False

def build_kg_in_hugegraph(entities: List[Dict], relations: List[Dict], domain: str) -> Dict:
    """Build Knowledge Graph in HugeGraph from extracted entities/relations.
    
    Uses domain-appropriate labels:
    - medical: Disease, Drug, Symptom, Treatment, Anatomy, Gene, RiskFactor
    - novel: Entity, Person, Location, Organization, Event, Concept
    
    Returns: {uploaded_vertices, uploaded_edges, name_to_vid, label_map}
    """
    # Map entity types to HugeGraph vertex labels
    type_to_label_map = {
        # Medical domain types → medical labels
        "disease": "Disease", "drug": "Drug", "symptom": "Symptom",
        "treatment": "Treatment", "anatomy": "Anatomy", "gene": "Gene",
        "risk_factor": "RiskFactor", "cell_type": "CellType",
        # General types → Entity label (catch-all)
        "person": "Person", "location": "Location", "organization": "Organization",
        "event": "Event", "concept": "Concept", "product": "Product",
        "company": "Company",
    }
    
    # For novel domain, use Entity as fallback (most flexible)
    fallback_label = "Entity" if domain == "novel" else "Concept"
    
    name_to_vid = {}
    label_map = {}  # entity_name -> label used
    uploaded_v = 0
    failed_v = 0
    
    # Deduplicate entities by name
    unique_entities = {}
    for e in entities:
        name = e["name"]
        if name not in unique_entities:
            unique_entities[name] = e
    
    print(f"  [KG Build] {len(unique_entities)} unique entities to upload for {domain} domain")
    
    for name, e in unique_entities.items():
        entity_type = e.get("type", "concept").lower()
        label = type_to_label_map.get(entity_type, fallback_label)
        label_map[name] = label
        
        properties = {
            "name": name,
            "description": e.get("description", f"{name} - from {domain} corpus"),
            "category": e.get("category", entity_type),
            "type": e.get("type", "concept"),
        }
        
        # Add domain-specific properties for medical labels
        if label == "Disease":
            properties["severity"] = e.get("severity", "moderate")
        elif label in ("Drug", "Treatment"):
            properties["category"] = e.get("category", "medical_treatment")
        elif label in ("Symptom", "Anatomy"):
            properties["location"] = e.get("location", "unspecified")
        elif label == "Gene":
            pass  # Gene only needs name + description
        elif label == "RiskFactor":
            properties["category"] = e.get("category", "risk_factor")
        
        vid = hg_vertex_create(label, properties)
        if vid:
            name_to_vid[name] = vid
            uploaded_v += 1
        else:
            failed_v += 1
    
    print(f"  [KG Build] Vertices: {uploaded_v} uploaded, {failed_v} failed")
    
    # ── Upload edges ──
    uploaded_e = 0
    failed_e = 0
    
    # Map relation types to HugeGraph edge labels
    relation_to_edge_map = {
        # Medical domain edges (source→target)
        "treats": ("Drug", "Treatment", "Disease"),       # Drug/Treatment → Disease
        "treated_by": ("Disease", "Drug"),                 # Disease → Drug  
        "has_symptom": ("Disease", "Symptom"),             # Disease → Symptom
        "prevents": ("Treatment", "Disease"),              # Treatment → Disease
        "causes": ("Entity", "Entity"),                    # Entity → Entity
        "located_in": ("Disease", "Anatomy"),              # Disease → Anatomy
        "is_a": ("Entity", "Concept"),                     # Entity → Concept
        "part_of": ("Entity", "Entity"),                   # Entity → Entity
        "arises_from": ("Disease", "CellType"),            # Disease → CellType
        "affects": ("Symptom", "Anatomy"),                 # Symptom → Anatomy
        "increases_risk_of": ("RiskFactor", "Disease"),    # RiskFactor → Disease
        # General edges
        "related_to": ("Entity", "Entity"),                # Entity → Entity (catch-all)
        "mentions": ("Document", "Entity"),                # Document → Entity
        "works_at": ("Person", "Organization"),            # Person → Organization
        "belongs_to": ("Entity", "Organization"),           # Entity → Organization
        "located_at": ("Entity", "Location"),              # Entity → Location
        "participated_in": ("Person", "Event"),            # Person → Event
    }
    
    unique_relations = {}
    for r in relations:
        key = (r.get("source", ""), r.get("target", ""), r.get("relation", ""))
        if key not in unique_relations:
            unique_relations[key] = r
    
    for key, r in unique_relations.items():
        src_name = r.get("source", "")
        tgt_name = r.get("target", "")
        relation_type = r.get("relation", "related_to").lower()
        
        if not src_name or not tgt_name:
            continue
        if src_name not in name_to_vid or tgt_name not in name_to_vid:
            continue
        
        src_label = label_map.get(src_name, fallback_label)
        tgt_label = label_map.get(tgt_name, fallback_label)
        
        # Determine edge label
        edge_info = relation_to_edge_map.get(relation_type)
        if edge_info:
            edge_label = relation_type
            # Verify src/tgt labels match edge schema
            expected_src = edge_info[0]
            expected_tgt = edge_info[-1]
        else:
            # Fallback: related_to (Entity → Entity)
            edge_label = "related_to"
        
        # For related_to, both must be Entity label
        if edge_label == "related_to" and (src_label != "Entity" or tgt_label != "Entity"):
            # Can't use related_to with non-Entity vertices
            # Try domain-specific edge or skip
            if domain == "medical":
                # Use domain-specific edges based on src/tgt labels
                medical_edges = {
                    ("Disease", "Drug"): "treated_by",
                    ("Drug", "Disease"): "treats",
                    ("Treatment", "Disease"): "prevents",
                    ("Disease", "Symptom"): "has_symptom",
                    ("Disease", "Anatomy"): "located_in",
                    ("RiskFactor", "Disease"): "increases_risk_of",
                }
                edge_label = medical_edges.get((src_label, tgt_label), "")
                if not edge_label:
                    # These labels don't have a matching edge schema - skip
                    continue
            else:
                # Novel domain: force Entity label for related_to edges
                continue
        
        relation_name = r.get("relation_name", f"{src_name}_{relation_type}_{tgt_name}")
        
        ok = hg_edge_create(edge_label, src_label, src_name, tgt_label, tgt_name,
                           {"name": relation_name[:100]})
        if ok:
            uploaded_e += 1
        else:
            failed_e += 1
    
    print(f"  [KG Build] Edges: {uploaded_e} uploaded, {failed_e} failed")
    
    return {
        "uploaded_vertices": uploaded_v,
        "uploaded_edges": uploaded_e,
        "name_to_vid": name_to_vid,
        "label_map": label_map,
    }

# ============================================================
# PHASE 5: Graph Traversal (FIXED - k_neighbor)
# ============================================================

def hg_vertex_exists(label: str, name: str) -> Optional[str]:
    """Check if a vertex exists in HugeGraph by label:name (PRIMARY_KEY).
    
    Returns vertex ID if found, None otherwise.
    No error flooding - single request per check.
    """
    import requests
    try:
        vid = f"{label}:{name}"
        url = f"{HG_REST_URL}/graphs/{HG_GRAPH}/vertices/{requests.utils.quote(vid)}"
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("id", vid)
    except:
        pass
    return None

def hg_kneighbor(vertex_id: str, max_depth: int = 2, limit: int = 10) -> List[Dict]:
    """HugeGraph k_neighbor traversal from a given vertex.
    
    Returns list of neighbor vertex info.
    """
    import requests
    try:
        # Use REST traverser API for kneighbor
        url = f"{HG_REST_URL}/graphs/{HG_GRAPH}/traversers/kneighbor"
        data = {
            "source": vertex_id,
            "direction": "BOTH",
            "max_depth": max_depth,
            "limit": limit,
        }
        r = requests.post(url, headers={"Content-Type": "application/json", "Accept": "application/json"},
                         json=data, timeout=10)
        if r.status_code == 200:
            result = r.json()
            # kneighbor returns {vertex_id: {neighbor_id: distance}}
            neighbors = []
            kneighbor_data = result.get("kneighbor", {})
            if isinstance(kneighbor_data, dict):
                for nid, info in kneighbor_data.items():
                    neighbors.append({"id": nid, "distance": info if isinstance(info, int) else 1})
            elif isinstance(kneighbor_data, list):
                for nid in kneighbor_data:
                    neighbors.append({"id": nid, "distance": 1})
            return neighbors
    except Exception as e:
        print(f"  [k_neighbor] Error: {str(e)[:50]}")
    return []

def get_neighbor_details(vertex_ids: List[str], limit=10) -> List[Dict]:
    """Fetch properties of neighbor vertices."""
    import requests
    details = []
    for vid in vertex_ids[:limit]:
        try:
            url = f"{HG_REST_URL}/graphs/{HG_GRAPH}/vertices/{requests.utils.quote(vid)}"
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=5)
            if r.status_code == 200:
                data = r.json()
                details.append({
                    "id": data.get("id", vid),
                    "label": data.get("label", ""),
                    "name": data.get("properties", {}).get("name", ""),
                    "description": data.get("properties", {}).get("description", ""),
                })
        except:
            pass
    return details

# ============================================================
# PHASE 6: 3-Channel RAG Retrieval + Evaluation
# ============================================================

def rrf_fusion(vector_results, bm25_results, graph_results, k=60):
    """Reciprocal Rank Fusion of 3 retrieval channels."""
    scores = {}
    
    # Vector channel (weight 1.0)
    for rank, (doc_id, score) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    
    # BM25 channel (weight 1.0)
    for rank, (doc_id, score, text) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    
    # Graph channel (weight 1.5 — higher because KG provides structural info)
    for rank, (doc_id, score) in enumerate(graph_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.5 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: -x[1])

def map_graph_neighbor_to_chunks(neighbors: List[Dict], chunk_raw_docs: Dict, name_to_vid: Dict):
    """Map graph neighbor entities back to relevant chunks.
    
    Strategy: Find chunks that mention the neighbor entity names.
    """
    graph_results = []
    neighbor_names = [n.get("name", "") for n in neighbors if n.get("name")]
    
    for doc_id, text in chunk_raw_docs.items():
        for name in neighbor_names:
            if name and name.lower() in text.lower():
                graph_results.append((doc_id, 0.5))
                break
    
    return graph_results

def evaluate_answer(prediction: str, reference: str, question_type: str) -> Dict:
    """Evaluate a RAG answer against reference using multiple metrics."""
    pred_lower = prediction.lower().strip()
    ref_lower = reference.lower().strip()
    
    # 1. Accuracy: key facts from reference present in prediction
    ref_facts = set(ref_lower.split())
    pred_facts = set(pred_lower.split())
    if len(ref_facts) > 0:
        overlap = len(ref_facts & pred_facts)
        accuracy = overlap / len(ref_facts)
    else:
        accuracy = 0.0
    
    # 2. ROUGE-L (Longest Common Subsequence)
    def lcs_len(s1, s2):
        if len(s1) == 0 or len(s2) == 0:
            return 0
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    pred_words = pred_lower.split()
    ref_words = ref_lower.split()
    lcs = lcs_len(pred_words, ref_words)
    precision_l = lcs / max(1, len(pred_words))
    recall_l = lcs / max(1, len(ref_words))
    rouge_l = (2 * precision_l * recall_l) / max(0.001, precision_l + recall_l) if (precision_l + recall_l) > 0 else 0
    
    # 3. F1 score (token-level)
    if len(ref_facts) + len(pred_facts) > 0:
        f1 = 2 * overlap / (len(ref_facts) + len(pred_facts))
    else:
        f1 = 0.0
    
    # 4. Contains correct answer check
    contains_answer = any(kw in pred_lower for kw in ref_lower.split()[:5] if len(kw) > 3)
    
    return {
        "accuracy": round(accuracy, 4),
        "rouge_l": round(rouge_l, 4),
        "f1": round(f1, 4),
        "contains_answer": contains_answer,
        "lcs_length": lcs,
    }

def rag_query(query: str, faiss_index, faiss_ids, bm25_index: BM25Index,
              embed_model, name_to_vid: Dict, label_map: Dict, chunk_raw_docs: Dict,
              domain: str) -> Dict:
    """Full 3-channel RAG query: FAISS + BM25 + Graph traversal + RRF fusion + LLM generation."""
    
    start_time = time.time()
    vector_hits = 0
    bm25_hits = 0
    graph_hits = 0
    graph_context = ""
    graph_neighbors = []
    
    # ── Channel 1: FAISS vector search ──
    import numpy as np
    q_embedding = embed_model.encode([query])
    vector_results = search_faiss(faiss_index, faiss_ids, q_embedding[0], top_k=10)
    vector_hits = len(vector_results)
    
    # ── Channel 2: BM25 search ──
    bm25_results = bm25_index.search(query, top_k=10)
    bm25_hits = len(bm25_results)
    
    # ── Channel 3: Graph traversal ──
    # Try to find query-related entities in HugeGraph
    query_words = query.lower().split()
    # Extract potential entity names (capitalized words, multi-word phrases)
    entity_candidates = []
    # Simple heuristic: capitalized words and known medical/novel terms
    for word in query_words:
        if len(word) > 2:
            entity_candidates.append(word)
    # Also try 2-word phrases
    for i in range(len(query_words) - 1):
        phrase = f"{query_words[i]} {query_words[i+1]}"
        if len(phrase) > 4:
            entity_candidates.append(phrase)
    
    # Look up entities in HugeGraph (limited attempts, no error flooding)
    found_vertices = []
    candidate_labels = ["Entity", "Disease", "Drug", "Symptom", "Treatment", "Concept",
                       "Person", "Location", "Organization", "Event"] if domain == "novel" \
        else ["Disease", "Drug", "Symptom", "Treatment", "Anatomy", "Gene", "RiskFactor", "Entity", "Concept"]
    
    for candidate in entity_candidates[:5]:  # Max 5 candidates to avoid flooding
        for label in candidate_labels[:3]:  # Max 3 labels per candidate
            vid = hg_vertex_exists(label, candidate)
            if vid:
                found_vertices.append({"id": vid, "label": label, "name": candidate})
                break  # Found it, no need to try other labels
    
    # k_neighbor traversal from found vertices
    for fv in found_vertices[:3]:  # Max 3 starting points
        neighbors = hg_kneighbor(fv["id"], max_depth=2, limit=10)
        if neighbors:
            graph_hits += len(neighbors)
            neighbor_details = get_neighbor_details([n["id"] for n in neighbors[:5]])
            graph_neighbors.extend(neighbor_details)
    
    # Map graph neighbors to chunks
    graph_results = map_graph_neighbor_to_chunks(graph_neighbors, chunk_raw_docs, name_to_vid)
    graph_hits = len(graph_results) if graph_results else 0
    if graph_neighbors:
        graph_context = "\n".join(
            f"- {n.get('name', '?')} ({n.get('label', '?')}): {n.get('description', '')[:80]}"
            for n in graph_neighbors[:5]
        )
    
    # ── RRF Fusion ──
    # BM25 results format: (doc_id, score, text)
    bm25_for_fusion = [(doc_id, score) for doc_id, score, _ in bm25_results]
    fused = rrf_fusion(vector_results, bm25_for_fusion, graph_results)
    
    # ── Retrieve top fused chunks ──
    top_chunks = []
    for doc_id, rrf_score in fused[:5]:
        text = chunk_raw_docs.get(doc_id, "")
        if text:
            top_chunks.append({"doc_id": doc_id, "text": text, "rrf_score": rrf_score})
    
    context_text = "\n\n".join(c["text"][:500] for c in top_chunks)
    
    # ── LLM Generation ──
    gen_prompt = f"""Based on the following context information, answer the question.
If the context contains relevant information, use it. If not, provide the best answer you can.

Context:
{context_text[:3000]}

{f"Related entities from knowledge graph: {graph_context}" if graph_context else ""}

Question: {query}

Provide a detailed, factual answer based on the context above."""
    
    answer = call_llm(gen_prompt, max_tokens=2048)
    latency = time.time() - start_time
    
    return {
        "answer": answer,
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "graph_hits": graph_hits,
        "graph_neighbors": len(graph_neighbors),
        "fused_chunks": len(top_chunks),
        "latency": latency,
        "context_used": context_text[:200],
    }

# ============================================================
# PHASE 7: Full Pipeline Execution
# ============================================================

def run_full_evaluation():
    """Run complete P0-improved GraphRAG-Bench evaluation."""
    
    print("=" * 70)
    print("GraphRAG-Bench P0-Improved v3 — graph_hits FIXED")
    print("=" * 70)
    
    # ── Phase 1: Connectivity ──
    print("\n[Phase 1] Connectivity checks...")
    if not check_hugegraph():
        print("ABORT: HugeGraph not reachable")
        return None
    if not check_llm_api():
        print("ABORT: LLM API not reachable")
        return None
    
    results = {
        "evaluation_id": f"p0_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": MIMO_MODEL,
            "llm_api_base": MIMO_API_BASE,
            "embed_model": EMBED_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "max_questions_per_type": MAX_QUESTIONS_PER_TYPE,
            "graph_server": HG_REST_URL,
            "graph_name": HG_GRAPH,
        },
        "domains": {},
        "rag_evaluation": {},
    }
    
    # Load sentence-transformers model once
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    for domain in ["novel", "medical"]:
        print(f"\n{'=' * 60}")
        print(f"  === Domain: {domain} ===")
        print(f"{'=' * 60}")
        
        # ── Phase 2: Load data ──
        questions, corpus = load_benchmark(domain)
        corpus_texts = prepare_corpus_texts(corpus, domain)
        total_chars = sum(len(t) for t in corpus_texts.values())
        print(f"  Corpus: {len(corpus_texts)} docs, {total_chars} chars")
        
        # ── Phase 3: Chunk + Build indexes ──
        all_chunks = []
        global_offset = 0
        for doc_name, doc_text in corpus_texts.items():
            chunks = chunk_text_global(doc_text, global_offset)
            for c in chunks:
                c["doc_name"] = doc_name
                c["domain"] = domain
            all_chunks.extend(chunks)
            global_offset += len(chunks)
        
        print(f"  Chunks: {len(all_chunks)} total")
        
        # FAISS index
        chunk_texts = [c["content"] for c in all_chunks]
        chunk_ids = [f"chunk_{c['chunk_index']}" for c in all_chunks]
        chunk_raw_docs = {chunk_ids[i]: chunk_texts[i] for i in range(len(all_chunks))}
        
        embeddings = embed_model.encode(chunk_texts, show_progress_bar=True, batch_size=64)
        import numpy as np
        embeddings = np.array(embeddings)
        faiss_index, faiss_ids = build_faiss_index(embeddings, chunk_ids)
        
        # BM25 index
        bm25_index = BM25Index()
        bm25_index.add_documents(chunk_texts, chunk_ids)
        print(f"  BM25: {bm25_index.N} docs indexed, avg_dl={bm25_index.avg_dl:.1f}")
        
        # ── Phase 4: KG Construction ──
        # Sample chunks for entity extraction (first 50 + random 50)
        extract_chunks = all_chunks[:50] + all_chunks[len(all_chunks)//2:len(all_chunks)//2+50]
        print(f"  [Extract] Extracting entities from {len(extract_chunks)} chunks via LLM...")
        
        entities, relations = extract_entities_from_chunks(extract_chunks, batch_size=3)
        print(f"  [Extract] Got {len(entities)} entities, {len(relations)} relations")
        
        # Build KG in HugeGraph
        kg_result = build_kg_in_hugegraph(entities, relations, domain)
        name_to_vid = kg_result["name_to_vid"]
        label_map = kg_result["label_map"]
        print(f"  [KG Build] {kg_result['uploaded_vertices']} vertices, {kg_result['uploaded_edges']} edges")
        
        # ── Phase 5: Verify graph data ──
        # Count vertices we just created
        verified_count = 0
        for name, label in list(label_map.items())[:20]:
            vid = hg_vertex_exists(label, name)
            if vid:
                verified_count += 1
        print(f"  [Verify] {verified_count}/20 sampled vertices exist in HugeGraph")
        
        # ── Phase 6: RAG Evaluation ──
        print(f"\n  [Phase 4] P0-Improved RAG evaluation ({domain})...")
        
        question_types = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
        
        for q_type in question_types:
            type_questions = [q for q in questions if q.get("question_type") == q_type]
            sampled = type_questions[:MAX_QUESTIONS_PER_TYPE]
            
            if not sampled:
                print(f"    {domain}/{q_type}: 0 questions (skip)")
                continue
            
            print(f"\n    {domain}/{q_type}: {len(sampled)} questions")
            
            type_results = []
            for i, q in enumerate(sampled):
                query = q.get("question", "")
                reference = q.get("answer", "")
                
                if not query:
                    continue
                
                # Run 3-channel RAG
                rag_result = rag_query(
                    query, faiss_index, faiss_ids, bm25_index, embed_model,
                    name_to_vid, label_map, chunk_raw_docs, domain
                )
                
                # Evaluate
                eval_metrics = evaluate_answer(rag_result["answer"], reference, q_type)
                
                type_results.append({
                    "question_id": q.get("question_id", f"q_{i}"),
                    "question": query[:100],
                    "reference_answer": reference[:100],
                    "prediction": rag_result["answer"][:200],
                    "metrics": eval_metrics,
                    "retrieval": {
                        "vector_hits": rag_result["vector_hits"],
                        "bm25_hits": rag_result["bm25_hits"],
                        "graph_hits": rag_result["graph_hits"],
                        "graph_neighbors": rag_result["graph_neighbors"],
                        "fused_chunks": rag_result["fused_chunks"],
                        "latency": rag_result["latency"],
                    },
                })
                
                if (i + 1) % 5 == 0:
                    avg_acc = sum(r["metrics"]["accuracy"] for r in type_results) / len(type_results)
                    avg_gh = sum(r["retrieval"]["graph_hits"] for r in type_results) / len(type_results)
                    print(f"      [{i+1}/{len(sampled)}] avg_acc={avg_acc:.3f}, avg_graph_hits={avg_gh:.1f}")
            
            # Compute type-level averages
            if type_results:
                avg_acc = sum(r["metrics"]["accuracy"] for r in type_results) / len(type_results)
                avg_rouge = sum(r["metrics"]["rouge_l"] for r in type_results) / len(type_results)
                avg_f1 = sum(r["metrics"]["f1"] for r in type_results) / len(type_results)
                avg_vh = sum(r["retrieval"]["vector_hits"] for r in type_results) / len(type_results)
                avg_bh = sum(r["retrieval"]["bm25_hits"] for r in type_results) / len(type_results)
                avg_gh = sum(r["retrieval"]["graph_hits"] for r in type_results) / len(type_results)
                avg_lat = sum(r["retrieval"]["latency"] for r in type_results) / len(type_results)
                
                key = f"{domain}/{q_type}"
                results["rag_evaluation"][key] = {
                    "num_questions": len(type_results),
                    "avg_accuracy": round(avg_acc, 4),
                    "avg_rouge_l": round(avg_rouge, 4),
                    "avg_f1": round(avg_f1, 4),
                    "avg_vector_hits": round(avg_vh, 2),
                    "avg_bm25_hits": round(avg_bh, 2),
                    "avg_graph_hits": round(avg_gh, 2),
                    "avg_latency": round(avg_lat, 2),
                    "graph_context_hits": int(sum(1 for r in type_results if r["retrieval"]["graph_hits"] > 0)),
                    "detailed_results": type_results,
                }
                
                print(f"    [{key}] acc={avg_acc:.4f}, rouge-L={avg_rouge:.4f}, F1={avg_f1:.4f}, "
                      f"v_hits={avg_vh:.1f}, b_hits={avg_bh:.1f}, g_hits={avg_gh:.1f}, lat={avg_lat:.1f}s")
        
        # Store domain-level info
        results["domains"][domain] = {
            "corpus_docs": len(corpus_texts),
            "corpus_chars": total_chars,
            "total_chunks": len(all_chunks),
            "entities_extracted": len(entities),
            "relations_extracted": len(relations),
            "kg_vertices_uploaded": kg_result["uploaded_vertices"],
            "kg_edges_uploaded": kg_result["uploaded_edges"],
        }
    
    # ── Phase 8: Compute overall metrics ──
    novel_keys = [f"novel/{t}" for t in question_types]
    medical_keys = [f"medical/{t}" for t in question_types]
    
    novel_accs = [results["rag_evaluation"].get(k, {}).get("avg_accuracy", 0) for k in novel_keys]
    medical_accs = [results["rag_evaluation"].get(k, {}).get("avg_accuracy", 0) for k in medical_keys]
    
    results["overall"] = {
        "novel_avg_accuracy": round(sum(novel_accs) / max(1, len([a for a in novel_accs if a > 0])), 4) if novel_accs else 0,
        "medical_avg_accuracy": round(sum(medical_accs) / max(1, len([a for a in medical_accs if a > 0])), 4) if medical_accs else 0,
        "total_questions": sum(results["rag_evaluation"].get(k, {}).get("num_questions", 0) for k in novel_keys + medical_keys),
        "graph_hits_nonzero": sum(results["rag_evaluation"].get(k, {}).get("graph_context_hits", 0) for k in novel_keys + medical_keys),
    }
    
    results["overall"]["combined_avg_accuracy"] = round(
        (results["overall"]["novel_avg_accuracy"] + results["overall"]["medical_avg_accuracy"]) / 2, 4
    )
    
    # ── Phase 9: Save results ──
    result_path = RESULT_DIR / "p0_v3_graphrag_bench_result.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Results saved to {result_path}")
    
    # ── Print summary ──
    print("\n" + "=" * 70)
    print("P0-IMPROVED v3 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Novel avg accuracy:  {results['overall']['novel_avg_accuracy']}")
    print(f"  Medical avg accuracy: {results['overall']['medical_avg_accuracy']}")
    print(f"  Combined avg accuracy: {results['overall']['combined_avg_accuracy']}")
    print(f"  Total questions: {results['overall']['total_questions']}")
    print(f"  Questions with graph_hits>0: {results['overall']['graph_hits_nonzero']}")
    print()
    
    # Per-type breakdown
    for key in sorted(results["rag_evaluation"].keys()):
        d = results["rag_evaluation"][key]
        print(f"  {key}: acc={d['avg_accuracy']:.4f}, rouge={d['avg_rouge_l']:.4f}, "
              f"F1={d['avg_f1']:.4f}, v_hits={d['avg_vector_hits']:.1f}, "
              f"b_hits={d['avg_bm25_hits']:.1f}, g_hits={d['avg_graph_hits']:.1f}")
    
    # ── Baseline comparison ──
    baseline_path = RESULT_DIR / "graphrag_bench_full_pipeline_result.json"
    if baseline_path.exists():
        baseline = json.load(open(baseline_path))
        be = baseline["rag_evaluation"]
        print("\n--- Baseline vs P0-v3 Comparison ---")
        for key in sorted(results["rag_evaluation"].keys()):
            p0_data = results["rag_evaluation"].get(key, {})
            bl_data = be.get(key, {})
            p0_acc = p0_data.get("avg_accuracy", 0)
            bl_acc = bl_data.get("avg_accuracy", 0)
            delta = p0_acc - bl_acc
            print(f"  {key}: baseline={bl_acc:.4f} → P0-v3={p0_acc:.4f} (Δ={delta:+.4f})")
    
    return results

if __name__ == "__main__":
    run_full_evaluation()
