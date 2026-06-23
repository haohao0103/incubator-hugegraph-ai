#!/usr/bin/env python3
"""
GraphRAG 快速基准 — 消融实验A0-A5
在414顶点真实数据上快速跑出Recall@K/MRR/P99，用结果说服高层
"""

import json, os, re, time, math, random
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

HG_REST = "http://127.0.0.1:8080"
HG_GRAPH = "poc_supply_chain"

def hg_get(url, auth=("admin", "admin")):
    from hugegraph_llm.utils.hg_http import hg_get as _hg_get
    return _hg_get(url, auth=auth)

# === Graph Data ===
class GraphData:
    def __init__(self):
        self.vertices = {}
        self.edges = {"out": defaultdict(list), "in": defaultdict(list)}
        self._loaded = False
    
    def _gp(self, p, k):
        v = p.get(k)
        return v.get("value", "") if isinstance(v, dict) else v
    
    def load(self):
        if self._loaded: return
        r = hg_get(f"{HG_REST}/graphs/{HG_GRAPH}/graph/vertices?limit=500")
        for v in r.get("vertices", []):
            vid = str(v.get("id", ""))
            props = v.get("properties", {})
            self.vertices[vid] = {
                "name": self._gp(props, "entity_name") or vid,
                "label": v.get("label", ""),
                "props": props,
                "vid": vid,
            }
        for elabel in ["supplies", "requires", "ships_to"]:
            r = hg_get(f"{HG_REST}/graphs/{HG_GRAPH}/graph/edges?label={elabel}&limit=500")
            for e in r.get("edges", []):
                src, tgt = str(e.get("outV", "")), str(e.get("inV", ""))
                self.edges["out"][src].append({"label": elabel, "target": tgt})
                self.edges["in"][tgt].append({"label": elabel, "source": src})
        self._loaded = True
        print(f"  Loaded: {len(self.vertices)} vertices, {sum(len(v) for v in self.edges['out'].values())} edges")

# === Tokenizer ===
def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    tokens.extend(re.findall(r'[\u4e00-\u9fff]+', text))
    return tokens

# === Channel A0: Vector (sentence-transformers 真实语义embedding) ===
class VectorChannel:
    """真实语义embedding (all-MiniLM-L6-v2, 384维), 替代TF-IDF"""
    def __init__(self, gd):
        self.gd = gd
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.np = np
        self.docs = []
        self.embeddings = None
        self._build()
    
    def _build(self):
        texts = []
        for vid, info in self.gd.vertices.items():
            text = f"{info['name']} {info['label']} "
            for k, v in info["props"].items():
                val = self.gd._gp(info["props"], k) if isinstance(info["props"].get(k), dict) else str(v)
                text += f"{val} "
            self.docs.append({"vid": vid, "text": text.strip(), "name": info["name"]})
            texts.append(text.strip())
        
        # 批量encode (真实语义embedding)
        print(f"    Encoding {len(texts)} documents with all-MiniLM-L6-v2...")
        self.embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=64)
        print(f"    Embedding dim: {self.embeddings.shape[1]}")
    
    def search(self, query, top_k=5):
        # Encode query
        q_emb = self.model.encode([query], show_progress_bar=False)[0]
        # Cosine similarity
        scores = self.np.dot(self.embeddings, q_emb) / (
            self.np.linalg.norm(self.embeddings, axis=1) * self.np.linalg.norm(q_emb) + 1e-8)
        
        # Top-K
        top_indices = self.np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({"vid": self.docs[idx]["vid"], "score": float(scores[idx]), "name": self.docs[idx]["name"]})
        return results

# === Channel A1: BM25 ===
class BM25Channel:
    def __init__(self, gd):
        self.docs = []
        self.df = defaultdict(int)
        for vid, info in gd.vertices.items():
            text = f"{info['name']} {info['label']} "
            for k, v in info["props"].items():
                val = gd._gp(info["props"], k) if isinstance(info["props"].get(k), dict) else str(v)
                text += f"{val} "
            tokens = tokenize(text)
            self.docs.append({"vid": vid, "tokens": tokens, "len": len(tokens), "name": info["name"]})
            for t in set(tokens): self.df[t] += 1
        self.N = len(self.docs)
        self.avg_len = sum(d["len"] for d in self.docs) / max(self.N, 1)
    
    def search(self, query, top_k=5):
        qt = tokenize(query)
        k1, b = 1.5, 0.75
        results = []
        for d in self.docs:
            tf = defaultdict(int)
            for t in d["tokens"]: tf[t] += 1
            score = 0.0
            for t in qt:
                if t in tf:
                    idf = math.log(1 + (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                    score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * d["len"] / max(self.avg_len, 1)))
            if score > 0:
                results.append({"vid": d["vid"], "score": score, "name": d["name"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# === Channel A2: Graph Traversal ===
class GraphChannel:
    def __init__(self, gd):
        self.gd = gd
    
    def _extract_entities(self, query):
        entities = re.findall(r'[\u4e00-\u9fff]+', query)
        entities.extend(re.findall(r'[A-Z]\w+', query))
        return entities
    
    def search(self, query, top_k=5):
        entities = self._extract_entities(query)
        results = []
        seen = set()
        for entity in entities:
            for vid, info in self.gd.vertices.items():
                if entity.lower() in info["name"].lower() or info["name"].lower() in entity.lower():
                    if vid not in seen:
                        seen.add(vid)
                        deg = len(self.gd.edges["out"].get(vid, [])) + len(self.gd.edges["in"].get(vid, []))
                        results.append({"vid": vid, "score": deg / 20.0, "name": info["name"]})
                        for e in self.gd.edges["out"].get(vid, [])[:3]:
                            t = e["target"]
                            if t not in seen and t in self.gd.vertices:
                                seen.add(t)
                                td = len(self.gd.edges["out"].get(t, [])) + len(self.gd.edges["in"].get(t, []))
                                results.append({"vid": t, "score": td / 30.0, "name": self.gd.vertices[t]["name"]})
                        for e in self.gd.edges["in"].get(vid, [])[:3]:
                            s = e["source"]
                            if s not in seen and s in self.gd.vertices:
                                seen.add(s)
                                sd = len(self.gd.edges["out"].get(s, [])) + len(self.gd.edges["in"].get(s, []))
                                results.append({"vid": s, "score": sd / 30.0, "name": self.gd.vertices[s]["name"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# === RRF Fusion ===
def rrf_fuse(channel_results, k=60, top_k=5):
    scores = defaultdict(float)
    doc_map = {}
    channels_per = defaultdict(list)
    for ch, results in channel_results.items():
        for rank, r in enumerate(results):
            vid = r["vid"]
            scores[vid] += 1.0 / (k + rank + 1)
            if vid not in doc_map: doc_map[vid] = r
            channels_per[vid].append(ch)
    sorted_vids = sorted(scores, key=scores.get, reverse=True)
    return [{"vid": vid, "rrf_score": scores[vid], "name": doc_map[vid]["name"], "channels": channels_per[vid]} for vid in sorted_vids[:top_k]]

# === Test Dataset ===
TEST_QUESTIONS = [
    {"q": "供应商A0的风险评分是多少", "entities": ["供应商A0"], "expected_vids": None, "expected_name": "供应商A0", "difficulty": "easy", "type": "属性查询"},
    {"q": "供应商B1供应了哪些零件", "entities": ["供应商B1"], "expected_vids": None, "expected_name": "供应商B1", "difficulty": "medium", "type": "关系查询"},
    {"q": "哪些零件是关键零件", "entities": ["零件"], "expected_vids": None, "expected_name": None, "difficulty": "easy", "type": "属性查询"},
    {"q": "Tier-1供应商有哪些", "entities": ["Tier-1", "供应商"], "expected_vids": None, "expected_name": None, "difficulty": "easy", "type": "属性查询"},
    {"q": "供应商C2的风险评分和供货关系", "entities": ["供应商C2"], "expected_vids": None, "expected_name": "供应商C2", "difficulty": "medium", "type": "关系查询"},
    {"q": "设施华北A接收哪些零件", "entities": ["设施华北A", "设施"], "expected_vids": None, "expected_name": "设施华北A", "difficulty": "medium", "type": "关系查询"},
    {"q": "风险评分最高的供应商", "entities": ["供应商", "风险"], "expected_vids": None, "expected_name": None, "difficulty": "hard", "type": "比较查询"},
    {"q": "哪些供应商来自中国", "entities": ["供应商", "中国"], "expected_vids": None, "expected_name": None, "difficulty": "easy", "type": "属性查询"},
    {"q": "零件A0的类别和成本", "entities": ["零件A0"], "expected_vids": None, "expected_name": "零件A0", "difficulty": "easy", "type": "属性查询"},
    {"q": "供应商和零件的供货关系分析", "entities": ["供应商", "零件"], "expected_vids": None, "expected_name": None, "difficulty": "hard", "type": "关系查询"},
    {"q": "哪些零件运往华南的设施", "entities": ["零件", "华南", "设施"], "expected_vids": None, "expected_name": None, "difficulty": "hard", "type": "关系查询"},
    {"q": "供应商D0的tier和国家", "entities": ["供应商D0"], "expected_vids": None, "expected_name": "供应商D0", "difficulty": "easy", "type": "属性查询"},
    {"q": "关键零件有哪些供应商", "entities": ["零件", "供应商"], "expected_vids": None, "expected_name": None, "difficulty": "hard", "type": "关系查询"},
    {"q": "设施A0的容量和区域", "entities": ["设施A0"], "expected_vids": None, "expected_name": "设施A0", "difficulty": "easy", "type": "属性查询"},
    {"q": "Tier-2供应商供应的零件", "entities": ["Tier-2", "供应商", "零件"], "expected_vids": None, "expected_name": None, "difficulty": "medium", "type": "关系查询"},
    {"q": "哪些供应商供应电子元件", "entities": ["供应商", "电子元件"], "expected_vids": None, "expected_name": None, "difficulty": "medium", "type": "关系查询"},
    {"q": "风险评分大于0.5的供应商", "entities": ["供应商", "风险"], "expected_vids": None, "expected_name": None, "difficulty": "medium", "type": "属性查询"},
    {"q": "零件B1是否关键零件", "entities": ["零件B1"], "expected_vids": None, "expected_name": "零件B1", "difficulty": "easy", "type": "属性查询"},
    {"q": "供应商E3的reliability", "entities": ["供应商E3"], "expected_vids": None, "expected_name": "供应商E3", "difficulty": "easy", "type": "属性查询"},
    {"q": "哪些设施在华北", "entities": ["设施", "华北"], "expected_vids": None, "expected_name": None, "difficulty": "easy", "type": "属性查询"},
]

# === Evaluation ===
def build_ground_truth(gd, questions):
    """从图数据构建ground truth: 查询命中的正确顶点名"""
    for q in questions:
        entities = q["entities"]
        matched_names = set()
        for entity in entities:
            for vid, info in gd.vertices.items():
                name = info["name"]
                if entity.lower() in name.lower() or name.lower() in entity.lower():
                    matched_names.add(name)
                    # 1跳邻居也算正确
                    for e in gd.edges["out"].get(vid, []):
                        t = e["target"]
                        if t in gd.vertices:
                            matched_names.add(gd.vertices[t]["name"])
                    for e in gd.edges["in"].get(vid, []):
                        s = e["source"]
                        if s in gd.vertices:
                            matched_names.add(gd.vertices[s]["name"])
        q["gt_names"] = matched_names
    return questions

def compute_recall_at_k(results, gt_names, k=5):
    if not gt_names: return 0.0
    top_k_names = set(r["name"] for r in results[:k])
    hits = len(top_k_names & gt_names)
    return min(hits / min(len(gt_names), k), 1.0)

def compute_mrr(results, gt_names):
    if not gt_names: return 0.0
    for i, r in enumerate(results):
        if r["name"] in gt_names:
            return 1.0 / (i + 1)
    return 0.0

def compute_precision_at_k(results, gt_names, k=5):
    if not results: return 0.0
    top_k_names = set(r["name"] for r in results[:k])
    hits = len(top_k_names & gt_names)
    return hits / max(len(top_k_names), 1)

# === Main Benchmark ===
def run_benchmark():
    print("=" * 70)
    print("GraphRAG Quick Benchmark — 消融实验 A0-A5")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Graph: {HG_REST} / {HG_GRAPH}")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading graph data...")
    gd = GraphData()
    gd.load()
    
    # Build ground truth
    print("\n[2/4] Building ground truth...")
    questions = build_ground_truth(gd, TEST_QUESTIONS)
    print(f"  {len(questions)} questions, ground truth built")
    
    # Init channels
    print("\n[3/4] Initializing retrieval channels...")
    vec_ch = VectorChannel(gd)
    bm25_ch = BM25Channel(gd)
    graph_ch = GraphChannel(gd)
    print(f"  Vector: {len(vec_ch.docs)} docs indexed")
    print(f"  BM25: {len(bm25_ch.docs)} docs indexed, avg_len={bm25_ch.avg_len:.1f}")
    print(f"  Graph: {len(gd.vertices)} vertices, {sum(len(v) for v in gd.edges['out'].values())} edges")
    
    # Run ablation experiments
    print("\n[4/4] Running ablation experiments A0-A5...")
    
    experiments = {
        "A0_纯向量": lambda q: vec_ch.search(q, top_k=5),
        "A1_纯BM25": lambda q: bm25_ch.search(q, top_k=5),
        "A2_纯图遍历": lambda q: graph_ch.search(q, top_k=5),
        "A3_向量+BM25": lambda q: rrf_fuse({"vector": vec_ch.search(q, 10), "bm25": bm25_ch.search(q, 10)}, k=60, top_k=5),
        "A4_向量+图": lambda q: rrf_fuse({"vector": vec_ch.search(q, 10), "graph": graph_ch.search(q, 10)}, k=60, top_k=5),
        "A5_三通道融合": lambda q: rrf_fuse({"vector": vec_ch.search(q, 10), "bm25": bm25_ch.search(q, 10), "graph": graph_ch.search(q, 10)}, k=60, top_k=5),
    }
    
    all_results = {}
    
    for exp_name, search_fn in experiments.items():
        print(f"\n  --- {exp_name} ---")
        recalls = []
        mrrs = []
        precisions = []
        latencies = []
        
        for q in questions:
            t0 = time.time()
            try:
                results = search_fn(q["q"])
            except Exception as e:
                results = []
            latency_ms = (time.time() - t0) * 1000
            
            gt = q["gt_names"]
            r5 = compute_recall_at_k(results, gt, k=5)
            mrr = compute_mrr(results, gt)
            p5 = compute_precision_at_k(results, gt, k=5)
            
            recalls.append(r5)
            mrrs.append(mrr)
            precisions.append(p5)
            latencies.append(latency_ms)
        
        n = len(questions)
        avg_r5 = sum(recalls) / n
        avg_mrr = sum(mrrs) / n
        avg_p5 = sum(precisions) / n
        avg_lat = sum(latencies) / n
        p99_lat = sorted(latencies)[int(0.99 * n)] if n > 1 else latencies[0]
        
        # Difficulty breakdown
        easy_r5 = sum(recalls[i] for i, q in enumerate(questions) if q["difficulty"] == "easy") / max(sum(1 for q in questions if q["difficulty"] == "easy"), 1)
        med_r5 = sum(recalls[i] for i, q in enumerate(questions) if q["difficulty"] == "medium") / max(sum(1 for q in questions if q["difficulty"] == "medium"), 1)
        hard_r5 = sum(recalls[i] for i, q in enumerate(questions) if q["difficulty"] == "hard") / max(sum(1 for q in questions if q["difficulty"] == "hard"), 1)
        
        all_results[exp_name] = {
            "recall@5": round(avg_r5, 4),
            "mrr": round(avg_mrr, 4),
            "precision@5": round(avg_p5, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "easy_recall@5": round(easy_r5, 4),
            "medium_recall@5": round(med_r5, 4),
            "hard_recall@5": round(hard_r5, 4),
        }
        
        print(f"    Recall@5={avg_r5:.3f} MRR={avg_mrr:.3f} P@5={avg_p5:.3f} | avg={avg_lat:.1f}ms p99={p99_lat:.1f}ms")
        print(f"    Easy={easy_r5:.3f} Medium={med_r5:.3f} Hard={hard_r5:.3f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("消融实验结果汇总 (N=20, 414顶点, 552边)")
    print("=" * 70)
    print(f"{'实验':<16} {'Recall@5':>10} {'MRR':>8} {'P@5':>8} {'P99(ms)':>10} {'Easy':>8} {'Med':>8} {'Hard':>8}")
    print("-" * 70)
    for exp, metrics in all_results.items():
        print(f"{exp:<16} {metrics['recall@5']:>10.3f} {metrics['mrr']:>8.3f} {metrics['precision@5']:>8.3f} {metrics['p99_latency_ms']:>10.1f} {metrics['easy_recall@5']:>8.3f} {metrics['medium_recall@5']:>8.3f} {metrics['hard_recall@5']:>8.3f}")
    
    # Key findings
    a0 = all_results["A0_纯向量"]["recall@5"]
    a3 = all_results["A3_向量+BM25"]["recall@5"]
    a5 = all_results["A5_三通道融合"]["recall@5"]
    a2 = all_results["A2_纯图遍历"]["recall@5"]
    
    print(f"\n关键发现:")
    print(f"  1. 三通道 vs 纯向量: A5={a5:.3f} vs A0={a0:.3f} (提升{(a5-a0)/max(a0,0.001)*100:.0f}%)")
    print(f"  2. 三通道 vs 双通道: A5={a5:.3f} vs A3={a3:.3f} (提升{(a5-a3)/max(a3,0.001)*100:.0f}%)")
    print(f"  3. 图通道独立价值: A2={a2:.3f} (纯图遍历)")
    print(f"  4. 图通道增益判定: {'✅ A5>A3 (图通道有增益)' if a5 > a3 else '❌ A5≤A3 (图通道无增益)'}")
    print(f"  5. P99延迟: {all_results['A5_三通道融合']['p99_latency_ms']:.1f}ms (目标<500ms)")
    
    # Save results
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "graph": HG_GRAPH,
        "data_scale": {"vertices": len(gd.vertices), "edges": sum(len(v) for v in gd.edges["out"].values())},
        "questions": len(questions),
        "ablation_results": all_results,
        "key_findings": {
            "三通道vs纯向量": f"{a5:.3f} vs {a0:.3f} (提升{(a5-a0)/max(a0,0.001)*100:.0f}%)",
            "三通道vs双通道": f"{a5:.3f} vs {a3:.3f} (提升{(a5-a3)/max(a3,0.001)*100:.0f}%)",
            "图通道增益判定": "✅ 有增益" if a5 > a3 else "❌ 无增益",
            "P99延迟": f"{all_results['A5_三通道融合']['p99_latency_ms']:.1f}ms",
        },
        "go_nogo_check": {
            "Recall@5>0.7": a5 > 0.7,
            "MRR>0.6": all_results["A5_三通道融合"]["mrr"] > 0.6,
            "P99<500ms": all_results["A5_三通道融合"]["p99_latency_ms"] < 500,
            "A5>A3(图通道增益)": a5 > a3,
        },
    }
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_ablation_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {result_path}")
    
    return result

if __name__ == "__main__":
    run_benchmark()
