#!/usr/bin/env python3
"""
LongMemEval 基准测试 — AI Memory标准评测
对标Mem0 (LongMemEval_S: 93.4分)

测试流程:
1. 加载500题 + 对话历史
2. 逐题: 将对话历史存入memory_backend → 提问 → LLM judge评分
3. 按question_type分类统计
4. 对标Mem0基线
"""

import json, os, sys, time, re, random, argparse
from datetime import datetime
from collections import defaultdict
import requests

# === Config ===
DATA_PATH = "/tmp/longmemeval_repo/data/longmemeval_oracle.json"
SAMPLE_SIZE = 50  # 先跑50题快速验证
MEMORY_API = "http://127.0.0.1:8765"
MIMO_API = "https://api.xiaomimimo.com/v1/chat/completions"
MIMO_KEY = "sk-cbgj0rzn5qvku9k6dmi4kek68qljzic1ka33o3b4czem2cm2"
MIMO_MODEL = "mimo-v2.5-pro"

def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    # 按question_type分组采样
    by_type = defaultdict(list)
    for item in data:
        qt = item.get("question_type", "unknown")
        by_type[qt].append(item)
    
    # 按比例采样
    random.seed(42)
    samples = []
    quotas = {
        "single-session-user": 10,
        "single-session-assistant": 8,
        "single-session-preference": 5,
        "multi-session": 10,
        "knowledge-update": 8,
        "temporal-reasoning": 9,
    }
    
    for qt, quota in quotas.items():
        pool = by_type.get(qt, [])
        random.shuffle(pool)
        samples.extend(pool[:quota])
    
    print(f"  Loaded {len(data)} total, sampled {len(samples)} questions")
    qtypes = defaultdict(int)
    for s in samples:
        qtypes[s["question_type"]] += 1
    print(f"  Distribution: {dict(qtypes)}")
    return samples

def reset_memory():
    """清空memory_backend"""
    try:
        requests.post(f"{MEMORY_API}/api/clear", json={"user_id": "eval_user"}, timeout=5)
    except:
        pass

def store_conversations(item):
    """将对话历史存入memory_backend"""
    haystack = item.get("haystack_sessions", [])
    user_id = f"eval_{item.get('question_id', 'unknown')}"
    
    stored = 0
    for session_group in haystack:
        if isinstance(session_group, list):
            for msg in session_group:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content and len(content) > 5:
                        try:
                            resp = requests.post(
                                f"{MEMORY_API}/api/memory/add",
                                json={"content": content[:500], "user_id": user_id},
                                timeout=30
                            )
                            if resp.status_code == 200:
                                stored += 1
                        except:
                            pass
    
    return stored

def query_memory(question, user_id):
    """查询memory_backend"""
    try:
        resp = requests.post(
            f"{MEMORY_API}/api/memory/search",
            json={"query": question, "user_id": user_id},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("answer", ""), data
        return f"[Error: {resp.status_code}]", {}
    except Exception as e:
        return f"[Exception: {e}]", {}

def llm_judge(question, gold_answer, predicted_answer):
    """用MiMo做LLM judge评分 (0/0.5/1)"""
    prompt = f"""你是记忆系统评测的裁判。判断预测答案是否正确回答了问题。

问题: {question}
标准答案: {gold_answer}
预测答案: {predicted_answer}

评分规则:
- 1.0: 完全正确（语义匹配标准答案）
- 0.5: 部分正确（包含部分正确信息，但不完整或有轻微错误）
- 0.0: 完全错误或无关

只输出一个数字(0.0, 0.5, 或 1.0)，不要其他文字。"""

    headers = {
        "Authorization": f"Bearer {MIMO_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MIMO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 10,
    }
    
    try:
        resp = requests.post(MIMO_API, headers=headers, json=data, timeout=15)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # 提取数字
            for val in ["1.0", "0.5", "0.0", "1", "0"]:
                if val in content:
                    return float(val)
            return 0.0
        return 0.0
    except:
        return 0.0

def run_benchmark():
    print("=" * 70)
    print("LongMemEval 基准测试 — AI Memory标准评测")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Dataset: LongMemEval (ICLR 2025, 500题, 6类能力)")
    print(f"Sample: {SAMPLE_SIZE}题")
    print(f"Memory: memory_backend (HugeGraph + FAISS + BM25)")
    print(f"LLM: MiMo v2.5-pro")
    print(f"对标: Mem0 (93.4分)")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/4] Loading LongMemEval dataset...")
    questions = load_data()
    
    # 2. Run evaluation
    print(f"\n[2/4] Running evaluation ({len(questions)} questions)...")
    
    results_by_type = defaultdict(lambda: {"scores": [], "count": 0})
    all_scores = []
    sample_results = []
    
    for i, item in enumerate(questions):
        qt = item.get("question_type", "unknown")
        qid = item.get("question_id", f"q{i}")
        question = item.get("question", "")
        gold = item.get("answer", "")
        user_id = f"eval_{qid}"
        
        print(f"\n  [{i+1}/{len(questions)}] [{qt}] Q: {question[:60]}...")
        
        # 2a. Store conversations
        stored = store_conversations(item)
        print(f"    Stored {stored} messages")
        
        # 2b. Query
        predicted, raw = query_memory(question, user_id)
        print(f"    Answer: {str(predicted)[:60]}...")
        
        # 2c. LLM judge
        score = llm_judge(question, gold, predicted)
        print(f"    Score: {score} (gold: {gold[:40]})")
        
        results_by_type[qt]["scores"].append(score)
        results_by_type[qt]["count"] += 1
        all_scores.append(score)
        
        if len(sample_results) < 20:
            sample_results.append({
                "question_id": qid,
                "question_type": qt,
                "question": question[:100],
                "gold_answer": gold[:100],
                "predicted_answer": str(predicted)[:100],
                "score": score,
            })
        
        # Reset for next question
        reset_memory()
    
    # 3. Aggregate results
    print(f"\n[3/4] Aggregating results...")
    
    overall_acc = sum(all_scores) / len(all_scores) if all_scores else 0
    
    type_results = {}
    for qt, v in sorted(results_by_type.items()):
        n = v["count"]
        avg = sum(v["scores"]) / n if n > 0 else 0
        acc_1 = sum(1 for s in v["scores"] if s >= 1.0) / n if n > 0 else 0
        type_results[qt] = {
            "count": n,
            "avg_score": round(avg, 4),
            "accuracy": round(acc_1 * 100, 2),
        }
        print(f"  {qt:30s}: n={n:3d}  avg={avg:.3f}  acc={acc_1*100:.1f}%")
    
    print(f"\n  {'OVERALL':30s}: n={len(all_scores):3d}  avg={overall_acc:.3f}  acc={overall_acc*100:.1f}%")
    
    # 4. Compare with Mem0
    print(f"\n[4/4] Compare with Mem0...")
    
    mem0_baseline = {
        "single-session-user": 94.3,
        "single-session-assistant": 97.1,
        "single-session-preference": None,  # Mem0未单独报告
        "multi-session": 70.7,
        "knowledge-update": 100.0,
        "temporal-reasoning": None,
        "overall": 93.4,
    }
    
    print(f"\n  {'Type':30s} {'Our':>8} {'Mem0':>8} {'Diff':>8}")
    print(f"  {'-'*56}")
    for qt, v in type_results.items():
        our = v["accuracy"]
        mem0 = mem0_baseline.get(qt)
        if mem0:
            diff = our - mem0
            print(f"  {qt:30s} {our:>7.1f}% {mem0:>7.1f}% {diff:>+7.1f}%")
        else:
            print(f"  {qt:30s} {our:>7.1f}% {'N/A':>8}")
    
    diff_overall = overall_acc * 100 - mem0_baseline["overall"]
    print(f"  {'-'*56}")
    print(f"  {'OVERALL':30s} {overall_acc*100:>7.1f}% {mem0_baseline['overall']:>7.1f}% {diff_overall:>+7.1f}%")
    
    # Save results
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "dataset": "LongMemEval (ICLR 2025)",
        "sample_size": len(questions),
        "memory_backend": "HugeGraph + FAISS + BM25 + GraphRAG",
        "llm": MIMO_MODEL,
        "judge": "MiMo LLM judge (0/0.5/1)",
        "overall": {
            "avg_score": round(overall_acc, 4),
            "accuracy": round(overall_acc * 100, 2),
        },
        "by_type": type_results,
        "mem0_baseline": mem0_baseline,
        "sample_results": sample_results,
    }
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "benchmark_longmemeval_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {result_path}")

if __name__ == "__main__":
    run_benchmark()
