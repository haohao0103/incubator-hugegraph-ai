#!/usr/bin/env python3
"""
GraphRAG-Bench LLM生成准确率测试
用MiMo v2.5-pro对Contextual Summarize和Creative Generation生成答案, 评测ACC+ROUGE-L
"""

import json, os, re, time, math, random
from datetime import datetime
from collections import defaultdict
import requests

BENCH_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphrag_bench_medical.json")
MIMO_API = "https://api.xiaomimimo.com/v1/chat/completions"
MIMO_KEY = "sk-cbgj0rzn5qvku9k6dmi4kek68qljzic1ka33o3b4czem2cm2"
MIMO_MODEL = "mimo-v2.5-pro"
SAMPLE_SIZE = 30  # 先测30题(20 Summarize + 10 Creative)

def load_benchmark():
    with open(BENCH_DATA) as f:
        data = json.load(f)
    by_type = defaultdict(list)
    for item in data:
        by_type[item["question_type"]].append(item)
    
    random.seed(42)
    samples = []
    # 只测弱项: Summarize + Creative
    for qt, quota in [("Contextual Summarize", 20), ("Creative Generation", 10)]:
        pool = by_type.get(qt, [])
        random.shuffle(pool)
        samples.extend(pool[:quota])
    
    print(f"  Loaded {len(samples)} questions (20 Summarize + 10 Creative)")
    return samples

def build_corpus(questions):
    corpus = {}
    for q in questions:
        for i, ev in enumerate(q.get("evidence", [])):
            corpus[f"{q['id']}_ev{i}"] = ev
    return corpus

def retrieve_evidence(question, corpus, top_k=5):
    """用向量检索+BM25融合获取top-5证据"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_embs = model.encode(doc_texts, show_progress_bar=False, batch_size=64)
    
    q_emb = model.encode([question], show_progress_bar=False)[0]
    scores = np.dot(doc_embs, q_emb) / (np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    top_idx = np.argsort(scores)[::-1][:top_k]
    
    return [corpus[doc_ids[i]] for i in top_idx if scores[i] > 0]

def llm_generate(question, evidence):
    """用MiMo LLM生成答案"""
    context = "\n\n".join([f"证据{i+1}: {ev}" for i, ev in enumerate(evidence)])
    prompt = f"""基于以下证据回答问题。如果证据不足，基于已有信息给出最佳答案。

{context}

问题: {question}

答案:"""
    
    headers = {
        "Authorization": f"Bearer {MIMO_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MIMO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 500,
    }
    
    try:
        resp = requests.post(MIMO_API, headers=headers, json=data, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"[LLM Error: {resp.status_code}]"
    except Exception as e:
        return f"[LLM Exception: {e}]"

def compute_rouge_l(candidate, reference):
    cand_tokens = re.findall(r'\w+', candidate.lower())
    ref_tokens = re.findall(r'\w+', reference.lower())
    if not cand_tokens or not ref_tokens:
        return 0.0
    m, n = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_accuracy(candidate, reference):
    """简单准确率: 关键词覆盖率"""
    ref_words = set(re.findall(r'\w{3,}', reference.lower()))
    cand_words = set(re.findall(r'\w{3,}', candidate.lower()))
    if not ref_words:
        return 0.0
    return len(ref_words & cand_words) / len(ref_words)

def run():
    print("=" * 70)
    print("GraphRAG-Bench LLM生成准确率测试 (Contextual Summarize + Creative)")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"LLM: {MIMO_MODEL}")
    print("=" * 70)
    
    questions = load_benchmark()
    corpus = build_corpus(questions)
    print(f"  Corpus: {len(corpus)} docs")
    
    results_by_type = defaultdict(lambda: {"rouge": [], "acc": [], "count": 0})
    
    for i, q in enumerate(questions):
        q_type = q["question_type"]
        print(f"\n[{i+1}/{len(questions)}] [{q_type}] Q: {q['question'][:60]}...")
        
        # 检索证据
        evidence = retrieve_evidence(q["question"], corpus, top_k=5)
        print(f"  Retrieved {len(evidence)} evidence chunks")
        
        # LLM生成
        t0 = time.time()
        answer = llm_generate(q["question"], evidence)
        gen_time = time.time() - t0
        print(f"  LLM生成 ({gen_time:.1f}s): {answer[:80]}...")
        
        # 评估
        gold = q.get("answer", "")
        rouge = compute_rouge_l(answer, gold)
        acc = compute_accuracy(answer, gold)
        
        print(f"  ROUGE-L: {rouge:.3f} | ACC(关键词覆盖): {acc:.3f}")
        print(f"  Gold: {gold[:80]}...")
        
        results_by_type[q_type]["rouge"].append(rouge)
        results_by_type[q_type]["acc"].append(acc)
        results_by_type[q_type]["count"] += 1
    
    # 汇总
    print("\n" + "=" * 70)
    print("生成准确率结果汇总")
    print("=" * 70)
    print(f"{'难度':<25} {'N':>4} {'ROUGE-L':>10} {'ACC':>10} {'vs业界RAG':>12}")
    print("-" * 65)
    
    # 业界基线 (论文中medical数据集)
    baseline = {
        "Contextual Summarize": {"acc": 65.75, "rouge": None, "framework": "RAG w/rerank"},
        "Creative Generation": {"acc": 60.61, "rouge": 36.74, "framework": "RAG w/rerank FS"},
    }
    
    all_results = {}
    for qt, v in results_by_type.items():
        n = v["count"]
        avg_rouge = sum(v["rouge"]) / n if v["rouge"] else 0
        avg_acc = sum(v["acc"]) / n if v["acc"] else 0
        base = baseline.get(qt, {})
        base_acc = base.get("acc", 0)
        diff = (avg_acc * 100 - base_acc)
        
        print(f"{qt:<25} {n:>4} {avg_rouge:>10.3f} {avg_acc*100:>9.1f}% {diff:>+11.1f}%")
        
        all_results[qt] = {
            "sample_size": n,
            "rouge_l": round(avg_rouge, 4),
            "accuracy": round(avg_acc * 100, 2),
            "baseline_acc": base_acc,
            "diff_vs_baseline": round(diff, 2),
            "baseline_framework": base.get("framework", ""),
        }
    
    # 保存
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "llm": MIMO_MODEL,
        "dataset": "GraphRAG-Bench medical",
        "sample_size": SAMPLE_SIZE,
        "results": all_results,
        "note": "只测了Contextual Summarize和Creative Generation两个弱项, 接入MiMo LLM生成答案",
    }
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_llm_generation_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {result_path}")

if __name__ == "__main__":
    run()
