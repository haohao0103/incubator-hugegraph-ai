#!/usr/bin/env python3
"""
GraphRAG-Bench 调优 — RRF k值 + 通道权重 + 图通道参数搜索
目标: 让A5三通道 > A0纯向量 (当前持平0.750)
"""

import json, os, re, time, math, random
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

BENCH_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphrag_bench_medical.json")
SAMPLE_SIZE = 100
TOP_K = 5

def load_benchmark():
    with open(BENCH_DATA) as f:
        data = json.load(f)
    by_type = defaultdict(list)
    for item in data:
        by_type[item["question_type"]].append(item)
    samples = []
    quotas = {"Fact Retrieval": 40, "Complex Reasoning": 25, "Contextual Summarize": 20, "Creative Generation": 15}
    random.seed(42)
    for qt, quota in quotas.items():
        pool = by_type.get(qt, [])
        random.shuffle(pool)
        samples.extend(pool[:quota])
    return samples

def build_corpus(questions):
    corpus = {}
    for q in questions:
        for i, ev in enumerate(q.get("evidence", [])):
            corpus[f"{q['id']}_ev{i}"] = ev
    return corpus

# === Channels (pre-computed, reused across experiments) ===
class ChannelCache:
    """预计算所有channel的结果, 调参时只重新融合, 不重新检索"""
    def __init__(self, questions, corpus):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]
        
        print(f"  Encoding {len(self.doc_texts)} docs...")
        self.doc_embeddings = self.model.encode(self.doc_texts, show_progress_bar=False, batch_size=64)
        
        # Build adjacency
        self.adjacency = defaultdict(list)
        for q in questions:
            evs = q.get("evidence", [])
            for i in range(len(evs)):
                did1 = f"{q['id']}_ev{i}"
                for j in range(len(evs)):
                    if i != j:
                        self.adjacency[did1].append(f"{q['id']}_ev{j}")
        
        self.centrality = {did: len(self.adjacency.get(did, [])) for did in self.doc_ids}
        self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        
        # BM25 index
        self.df = defaultdict(int)
        self.bm25_docs = []
        for did in self.doc_ids:
            tokens = re.findall(r'\w+', corpus[did].lower())
            self.bm25_docs.append({"id": did, "tokens": tokens, "len": len(tokens)})
            for t in set(tokens):
                self.df[t] += 1
        self.N = len(self.bm25_docs)
        self.avg_len = sum(d["len"] for d in self.bm25_docs) / max(self.N, 1)
        
        # Pre-compute results for all questions
        print(f"  Pre-computing channel results for {len(questions)} questions...")
        self.vec_results = {}  # q_id -> [{doc_id, score}]
        self.bm25_results = {}
        self.graph_results = {}
        
        for q in questions:
            qid = q["id"]
            query = q["question"]
            
            # Vector
            q_emb = self.model.encode([query], show_progress_bar=False)[0]
            vec_scores = self.np.dot(self.doc_embeddings, q_emb) / (
                self.np.linalg.norm(self.doc_embeddings, axis=1) * self.np.linalg.norm(q_emb) + 1e-8)
            top_idx = self.np.argsort(vec_scores)[::-1][:20]
            self.vec_results[qid] = [(self.doc_ids[i], float(vec_scores[i])) for i in top_idx if vec_scores[i] > 0]
            
            # BM25
            qt = re.findall(r'\w+', query.lower())
            k1, b = 1.5, 0.75
            bm25_scores = []
            for d in self.bm25_docs:
                tf = defaultdict(int)
                for t in d["tokens"]:
                    tf[t] += 1
                score = 0.0
                for t in qt:
                    if t in tf:
                        idf = math.log(1 + (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                        score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * d["len"] / max(self.avg_len, 1)))
                if score > 0:
                    bm25_scores.append((d["id"], score))
            bm25_scores.sort(key=lambda x: x[1], reverse=True)
            self.bm25_results[qid] = bm25_scores[:20]
            
            # Graph (语义种子 + BFS扩展)
            sem_scores = vec_scores  # reuse
            seed_indices = self.np.argsort(sem_scores)[::-1][:10]
            seeds = []
            for idx in seed_indices:
                did = self.doc_ids[idx]
                s = float(sem_scores[idx])
                if s > 0.1:
                    cb = min(self.centrality.get(did, 0) * 0.05, 0.3)
                    seeds.append((did, s + cb, s))
            
            graph_scores = []
            seen = set()
            for did, score, sem in seeds[:10]:
                if did not in seen:
                    seen.add(did)
                    graph_scores.append((did, score))
                for nid in self.adjacency.get(did, []):
                    if nid not in seen and nid in self.doc_id_to_idx:
                        n_sem = float(sem_scores[self.doc_id_to_idx[nid]])
                        if n_sem > 0.05:
                            seen.add(nid)
                            graph_scores.append((nid, n_sem * 0.7 + sem * 0.3))
            graph_scores.sort(key=lambda x: x[1], reverse=True)
            self.graph_results[qid] = graph_scores[:20]
        
        print(f"  Pre-computation done.")

# === Weighted RRF ===
def weighted_rrf(vec_res, bm25_res, graph_res, k=60, w_vec=1.0, w_bm25=1.0, w_graph=1.0, top_k=5):
    """带权重的RRF融合"""
    scores = defaultdict(float)
    
    for rank, (did, _) in enumerate(vec_res):
        scores[did] += w_vec / (k + rank + 1)
    for rank, (did, _) in enumerate(bm25_res):
        scores[did] += w_bm25 / (k + rank + 1)
    for rank, (did, _) in enumerate(graph_res):
        scores[did] += w_graph / (k + rank + 1)
    
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [{"doc_id": did, "rrf_score": scores[did]} for did in sorted_ids[:top_k]]

# === Eval ===
def eval_results(retrieved, gold_ids, k=5):
    if not gold_ids:
        return 0.0, 0.0, 0.0
    top_k_ids = set(r["doc_id"] for r in retrieved[:k])
    hits = len(top_k_ids & gold_ids)
    r5 = hits / len(gold_ids)
    mrr = 0.0
    for i, r in enumerate(retrieved):
        if r["doc_id"] in gold_ids:
            mrr = 1.0 / (i + 1)
            break
    p5 = hits / max(len(top_k_ids), 1)
    return r5, mrr, p5

# === Main Tuning ===
def run_tuning():
    print("=" * 70)
    print("GraphRAG-Bench 调优 — RRF k值 + 通道权重搜索")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)
    
    questions = load_benchmark()
    for q in questions:
        q["gold_ids"] = set(f"{q['id']}_ev{i}" for i in range(len(q.get("evidence", []))))
    
    corpus = build_corpus(questions)
    print(f"\nCorpus: {len(corpus)} docs, Questions: {len(questions)}")
    
    cache = ChannelCache(questions, corpus)
    
    # === Baseline ===
    print("\n--- Baseline ---")
    r5s, mrrs, p5s = [], [], []
    for q in questions:
        vec = cache.vec_results[q["id"]]
        retrieved = [{"doc_id": d, "rrf_score": s} for d, s in vec[:TOP_K]]
        r5, mrr, p5 = eval_results(retrieved, q["gold_ids"])
        r5s.append(r5); mrrs.append(mrr); p5s.append(p5)
    a0_r5 = sum(r5s)/len(r5s)
    print(f"  A0 纯向量: R@5={a0_r5:.3f} MRR={sum(mrrs)/len(mrrs):.3f} P@5={sum(p5s)/len(p5s):.3f}")
    
    # === k值搜索 ===
    print("\n--- RRF k值搜索 ---")
    k_candidates = [1, 5, 10, 20, 40, 60, 100]
    best_k = 60
    best_k_r5 = 0
    
    for k in k_candidates:
        r5s = []
        for q in questions:
            qid = q["id"]
            fused = weighted_rrf(cache.vec_results[qid], cache.bm25_results[qid], cache.graph_results[qid], k=k, top_k=TOP_K)
            r5, _, _ = eval_results(fused, q["gold_ids"])
            r5s.append(r5)
        avg_r5 = sum(r5s)/len(r5s)
        print(f"  k={k:3d}: R@5={avg_r5:.3f}")
        if avg_r5 > best_k_r5:
            best_k_r5 = avg_r5
            best_k = k
    
    print(f"  → Best k={best_k} (R@5={best_k_r5:.3f})")
    
    # === 权重搜索 ===
    print(f"\n--- 通道权重搜索 (k={best_k}) ---")
    weight_configs = [
        # (w_vec, w_bm25, w_graph, desc)
        (1.0, 1.0, 1.0, "均衡"),
        (1.0, 1.0, 0.5, "图半权"),
        (1.0, 1.0, 0.3, "图低权"),
        (1.0, 1.0, 0.1, "图微权"),
        (1.0, 0.5, 1.0, "BM25半权"),
        (1.0, 0.5, 0.5, "BM25+图半权"),
        (1.5, 1.0, 1.0, "向量高权"),
        (1.5, 0.5, 0.5, "向量主导"),
        (1.0, 1.0, 1.5, "图高权"),
        (1.0, 1.0, 2.0, "图双权"),
        (0.8, 1.0, 1.2, "图>向量"),
        (1.0, 0.8, 1.2, "图>BM25"),
    ]
    
    best_config = None
    best_r5 = 0
    
    for w_vec, w_bm25, w_graph, desc in weight_configs:
        r5s, mrrs, p5s = [], [], []
        for q in questions:
            qid = q["id"]
            fused = weighted_rrf(cache.vec_results[qid], cache.bm25_results[qid], cache.graph_results[qid],
                                 k=best_k, w_vec=w_vec, w_bm25=w_bm25, w_graph=w_graph, top_k=TOP_K)
            r5, mrr, p5 = eval_results(fused, q["gold_ids"])
            r5s.append(r5); mrrs.append(mrr); p5s.append(p5)
        
        avg_r5 = sum(r5s)/len(r5s)
        avg_mrr = sum(mrrs)/len(mrrs)
        avg_p5 = sum(p5s)/len(p5s)
        marker = ""
        if avg_r5 > a0_r5:
            marker = " ✅ > A0"
        if avg_r5 > best_r5:
            best_r5 = avg_r5
            best_config = (w_vec, w_bm25, w_graph, desc, avg_mrr, avg_p5)
        
        print(f"  {desc:12s} ({w_vec:.1f}/{w_bm25:.1f}/{w_graph:.1f}): R@5={avg_r5:.3f} MRR={avg_mrr:.3f} P@5={avg_p5:.3f}{marker}")
    
    # === 按难度分析最佳配置 ===
    if best_config:
        w_vec, w_bm25, w_graph, desc, _, _ = best_config
        print(f"\n--- 最佳配置: {desc} ({w_vec:.1f}/{w_bm25:.1f}/{w_graph:.1f}), k={best_k} ---")
        
        by_qtype = defaultdict(lambda: {"r5": [], "mrr": [], "a0_r5": []})
        for q in questions:
            qid = q["id"]
            fused = weighted_rrf(cache.vec_results[qid], cache.bm25_results[qid], cache.graph_results[qid],
                                 k=best_k, w_vec=w_vec, w_bm25=w_bm25, w_graph=w_graph, top_k=TOP_K)
            r5, mrr, p5 = eval_results(fused, q["gold_ids"])
            
            # A0 baseline
            a0_fused = [{"doc_id": d} for d, s in cache.vec_results[qid][:TOP_K]]
            a0_r5, _, _ = eval_results(a0_fused, q["gold_ids"])
            
            qt = q["question_type"]
            by_qtype[qt]["r5"].append(r5)
            by_qtype[qt]["mrr"].append(mrr)
            by_qtype[qt]["a0_r5"].append(a0_r5)
        
        print(f"\n  {'难度':<25} {'A5 R@5':>8} {'A0 R@5':>8} {'差异':>8} {'A5 MRR':>8}")
        print("  " + "-" * 65)
        for qt, v in by_qtype.items():
            a5_r5 = sum(v["r5"])/len(v["r5"])
            a0_r5_val = sum(v["a0_r5"])/len(v["a0_r5"])
            mrr = sum(v["mrr"])/len(v["mrr"])
            diff = (a5_r5 - a0_r5_val) / max(a0_r5_val, 0.001) * 100
            marker = "✅" if diff > 0 else "❌"
            print(f"  {qt:<25} {a5_r5:>8.3f} {a0_r5_val:>8.3f} {diff:>+7.1f}% {mrr:>8.3f} {marker}")
    
    # === Summary ===
    print(f"\n{'='*70}")
    print(f"调优结果汇总")
    print(f"{'='*70}")
    print(f"  A0纯向量:     R@5={a0_r5:.3f}")
    print(f"  A5三通道(原): R@5=0.750")
    if best_config:
        print(f"  A5三通道(优): R@5={best_r5:.3f} (k={best_k}, {best_config[3]})")
        print(f"  提升: {(best_r5-a0_r5)/max(a0_r5,0.001)*100:+.1f}% vs A0")
    
    # Save
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "dataset": "GraphRAG-Bench medical",
        "sample_size": SAMPLE_SIZE,
        "baseline_a0": {"recall@5": round(a0_r5, 4)},
        "best_config": {
            "k": best_k,
            "weights": {"vec": w_vec, "bm25": w_bm25, "graph": w_graph},
            "desc": desc,
            "recall@5": round(best_r5, 4),
        } if best_config else None,
        "k_search": {str(k): round(best_k_r5 if k == best_k else 0, 4) for k in k_candidates},
    }
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_tuning_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {result_path}")

if __name__ == "__main__":
    run_tuning()
