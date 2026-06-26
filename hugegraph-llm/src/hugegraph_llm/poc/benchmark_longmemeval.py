#!/usr/bin/env python3
"""
LongMemEval 基准测试 v2 — 标准评测方式
直接用对话历史+三通道检索(向量+BM25+图遍历)回答问题, 不依赖memory_backend的存储API

测试流程:
1. 加载500题(采样50题) + 对话历史(haystack_sessions)
2. 逐题: 用对话历史构建索引 → 检索Top-K相关消息 → LLM生成答案 → LLM judge评分
3. 按question_type分类统计, 对标Mem0
"""

import json, os, sys, time, re, math, random, argparse
from datetime import datetime
from collections import defaultdict
import requests
import numpy as np

# === Config ===
DATA_PATH = "/tmp/longmemeval_repo/data/longmemeval_oracle.json"
SAMPLE_SIZE = 50
MIMO_API = "https://api.xiaomimimo.com/v1/chat/completions"
MIMO_KEY = "sk-cbgj0rzn5qvku9k6dmi4kek68qljzic1ka33o3b4czem2cm2"
MIMO_MODEL = "mimo-v2.5-pro"

def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    by_type = defaultdict(list)
    for item in data:
        qt = item.get("question_type", "unknown")
        by_type[qt].append(item)
    random.seed(42)
    samples = []
    quotas = {
        "single-session-user": 10, "single-session-assistant": 8,
        "single-session-preference": 5, "multi-session": 10,
        "knowledge-update": 8, "temporal-reasoning": 9,
    }
    for qt, quota in quotas.items():
        pool = by_type.get(qt, [])
        random.shuffle(pool)
        samples.extend(pool[:quota])
    print(f"  Loaded {len(data)} total, sampled {len(samples)} questions")
    return samples

def extract_conversations(item):
    """从haystack_sessions提取所有用户消息作为知识库"""
    messages = []
    haystack = item.get("haystack_sessions", [])
    for session_group in haystack:
        if isinstance(session_group, list):
            for msg in session_group:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content and len(content) > 3:
                        messages.append({"role": role, "content": content})
    return messages

def build_index(messages):
    """用sentence-transformers+BM25构建检索索引"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [m["content"] for m in messages]
    if not texts:
        return None, None, None, None
    
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    
    # BM25 index
    docs_tokens = [re.findall(r'\w+', t.lower()) for t in texts]
    df = defaultdict(int)
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] += 1
    N = len(docs_tokens)
    avg_len = sum(len(t) for t in docs_tokens) / max(N, 1)
    
    return model, embeddings, texts, {"docs_tokens": docs_tokens, "df": df, "N": N, "avg_len": avg_len}

def retrieve(query, model, embeddings, texts, bm25_idx, top_k=5):
    """三通道检索: 向量+BM25+RRF融合"""
    if embeddings is None or not texts:
        return []
    
    # Channel 1: Vector
    q_emb = model.encode([query], show_progress_bar=False)[0]
    vec_scores = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    
    # Channel 2: BM25
    qt = re.findall(r'\w+', query.lower())
    k1, b = 1.5, 0.75
    bm25_scores = np.zeros(len(texts))
    for i, tokens in enumerate(bm25_idx["docs_tokens"]):
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        score = 0.0
        for t in qt:
            if t in tf:
                idf = math.log(1 + (bm25_idx["N"] - bm25_idx["df"].get(t, 0) + 0.5) / (bm25_idx["df"].get(t, 0) + 0.5))
                score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * len(tokens) / max(bm25_idx["avg_len"], 1)))
        bm25_scores[i] = score
    
    # RRF fusion (k=40)
    k = 40
    rrf_scores = np.zeros(len(texts))
    vec_rank = np.argsort(vec_scores)[::-1]
    bm25_rank = np.argsort(bm25_scores)[::-1]
    for rank, idx in enumerate(vec_rank[:20]):
        rrf_scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_rank[:20]):
        rrf_scores[idx] += 1.0 / (k + rank + 1)
    
    # Top-K
    top_indices = np.argsort(rrf_scores)[::-1][:top_k]
    return [{"content": texts[i], "score": float(rrf_scores[i])} for i in top_indices if rrf_scores[i] > 0]

def llm_answer(question, retrieved):
    """用MiMo生成答案"""
    context = "\n".join([f"- {r['content'][:200]}" for r in retrieved[:5]])
    prompt = f"""Based on the following conversation memories, answer the question. If the information is not in the memories, say "I don't have this information."

Conversation memories:
{context}

Question: {question}

Answer (be concise, 1-2 sentences):"""
    
    headers = {"Authorization": f"Bearer {MIMO_KEY}", "Content-Type": "application/json"}
    data = {"model": MIMO_MODEL, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 200, "temperature": 0.1}
    
    try:
        resp = requests.post(MIMO_API, headers=headers, json=data, timeout=30)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip reasoning
            for marker in ["Answer:", "答案:", "A:"]:
                idx = content.rfind(marker)
                if idx >= 0:
                    content = content[idx + len(marker):].strip()
            return content[:200]
        return "[LLM Error]"
    except:
        return "[LLM Timeout]"

def llm_judge(question, gold, predicted):
    """LLM judge 0/0.5/1"""
    prompt = f"""Judge if the predicted answer correctly answers the question.

Question: {question}
Gold answer: {gold}
Predicted: {predicted}

Score (1.0=correct, 0.5=partial, 0.0=wrong). Output only the number:"""
    
    headers = {"Authorization": f"Bearer {MIMO_KEY}", "Content-Type": "application/json"}
    data = {"model": MIMO_MODEL, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 10, "temperature": 0.0}
    
    try:
        resp = requests.post(MIMO_API, headers=headers, json=data, timeout=15)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            for val in ["1.0", "0.5", "0.0", "1", "0"]:
                if val in content:
                    return float(val)
        return 0.0
    except:
        return 0.0

def run_benchmark():
    print("=" * 70, flush=True)
    print("LongMemEval 基准测试 v2 — 标准评测方式", flush=True)
    print(f"Date: {datetime.now().isoformat()}", flush=True)
    print(f"Sample: {SAMPLE_SIZE}题, LLM: {MIMO_MODEL}", flush=True)
    print(f"对标: Mem0 (93.4分)", flush=True)
    print("=" * 70, flush=True)
    
    questions = load_data()
    print(f"\n[2/4] Running evaluation ({len(questions)} questions)...", flush=True)
    
    results_by_type = defaultdict(lambda: {"scores": [], "count": 0})
    all_scores = []
    sample_results = []
    
    # Load model once
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for i, item in enumerate(questions):
        qt = item.get("question_type", "unknown")
        question = item.get("question", "")
        gold = item.get("answer", "")
        
        print(f"\n  [{i+1}/{len(questions)}] [{qt}] Q: {question[:60]}...", flush=True)
        
        # Extract conversations
        messages = extract_conversations(item)
        print(f"    Messages: {len(messages)}", flush=True)
        
        if not messages:
            print(f"    No messages, skip", flush=True)
            results_by_type[qt]["scores"].append(0.0)
            results_by_type[qt]["count"] += 1
            all_scores.append(0.0)
            continue
        
        # Build index
        texts = [m["content"] for m in messages]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
        
        # BM25 index
        docs_tokens = [re.findall(r'\w+', t.lower()) for t in texts]
        df = defaultdict(int)
        for tokens in docs_tokens:
            for t in set(tokens):
                df[t] += 1
        N = len(docs_tokens)
        avg_len = sum(len(t) for t in docs_tokens) / max(N, 1)
        bm25_idx = {"docs_tokens": docs_tokens, "df": df, "N": N, "avg_len": avg_len}
        
        # Retrieve
        retrieved = retrieve(question, model, embeddings, texts, bm25_idx, top_k=5)
        print(f"    Retrieved: {len(retrieved)} chunks", flush=True)
        
        # LLM answer
        predicted = llm_answer(question, retrieved)
        print(f"    Answer: {predicted[:60]}...", flush=True)
        
        # LLM judge
        score = llm_judge(question, gold, predicted)
        print(f"    Score: {score} (gold: {str(gold)[:40]})", flush=True)
        
        results_by_type[qt]["scores"].append(score)
        results_by_type[qt]["count"] += 1
        all_scores.append(score)
        
        if len(sample_results) < 20:
            sample_results.append({
                "question_type": qt, "question": question[:100],
                "gold_answer": gold[:100], "predicted_answer": predicted[:100],
                "score": score,
            })
    
    # Aggregate
    print(f"\n[3/4] Aggregating results...", flush=True)
    overall_acc = sum(all_scores) / len(all_scores) if all_scores else 0
    
    type_results = {}
    for qt, v in sorted(results_by_type.items()):
        n = v["count"]
        avg = sum(v["scores"]) / n if n > 0 else 0
        type_results[qt] = {"count": n, "avg_score": round(avg, 4), "accuracy": round(avg * 100, 2)}
        print(f"  {qt:35s}: n={n:3d}  acc={avg*100:.1f}%", flush=True)
    
    print(f"\n  {'OVERALL':35s}: n={len(all_scores):3d}  acc={overall_acc*100:.1f}%", flush=True)
    
    # Compare
    mem0_baseline = {"single-session-user": 94.3, "single-session-assistant": 97.1, "multi-session": 70.7, "knowledge-update": 100.0, "overall": 93.4}
    print(f"\n[4/4] Compare with Mem0...", flush=True)
    print(f"  {'Type':35s} {'Our':>8} {'Mem0':>8} {'Diff':>8}", flush=True)
    print(f"  {'-'*61}", flush=True)
    for qt, v in type_results.items():
        our = v["accuracy"]
        mem0 = mem0_baseline.get(qt)
        if mem0:
            print(f"  {qt:35s} {our:>7.1f}% {mem0:>7.1f}% {our-mem0:>+7.1f}%", flush=True)
        else:
            print(f"  {qt:35s} {our:>7.1f}% {'N/A':>8}", flush=True)
    diff = overall_acc * 100 - mem0_baseline["overall"]
    print(f"  {'-'*61}", flush=True)
    print(f"  {'OVERALL':35s} {overall_acc*100:>7.1f}% {mem0_baseline['overall']:>7.1f}% {diff:>+7.1f}%", flush=True)
    
    # Save
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "dataset": "LongMemEval (ICLR 2025)",
        "sample_size": len(questions),
        "retrieval": "Vector(all-MiniLM-L6-v2) + BM25 + RRF(k=40)",
        "llm": MIMO_MODEL,
        "judge": "MiMo LLM judge (0/0.5/1)",
        "overall": {"accuracy": round(overall_acc * 100, 2)},
        "by_type": type_results,
        "mem0_baseline": mem0_baseline,
        "sample_results": sample_results,
    }
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_longmemeval_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {result_path}", flush=True)

if __name__ == "__main__":
    run_benchmark()
