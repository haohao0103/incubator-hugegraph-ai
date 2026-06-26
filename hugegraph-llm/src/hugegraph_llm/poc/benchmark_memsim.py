#!/usr/bin/env python3
"""
MemSim 中文记忆基准测试
数据集: https://github.com/nuster1128/MemSim (中文日常生活记忆模拟)
6种QA类型: simple/conditional/comparative/aggregative/post_processing/noisy
每题4选1 (MCQ), 用exact match评分

测试流程:
1. 加载memdaily.json (6 splits, 2954题)
2. 逐题: 用对话历史构建索引 → 检索Top-K → LLM回答MCQ → exact match评分
3. 按split分类统计, 对标AMB排行榜
"""

import json, os, sys, time, re, math, random
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np
import requests
from dotenv import load_dotenv

# === Load environment from hugegraph-llm/.env ===
PROJECT_ROOT = Path(__file__).parent.parent.parent  # hugegraph-llm/
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# === Config ===
DATA_PATH = os.environ.get("MEMSIM_DATA_PATH", str(PROJECT_ROOT / "benchmark_data" / "MemSim" / "memdaily.json"))
LLM_API_BASE = os.environ.get("OPENAI_CHAT_API_BASE", "https://api.xiaomimimo.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("OPENAI_CHAT_API_KEY") or os.environ.get("MIMO_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("Please set OPENAI_CHAT_API_KEY (or MIMO_API_KEY) in hugegraph-llm/.env")
LLM_MODEL = os.environ.get("OPENAI_CHAT_LANGUAGE_MODEL", "mimo-v2.5-pro")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SEC", "60"))
SPLITS = ["simple", "conditional", "comparative", "aggregative", "post_processing", "noisy"]
SAMPLE_PER_SPLIT = int(os.environ.get("MEMSIM_SAMPLE_PER_SPLIT", "20"))  # 每个split采样N题

def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data

def format_message(msg):
    """格式化消息为可读文本"""
    if isinstance(msg, str):
        return msg
    parts = [msg.get("message", "")]
    if msg.get("time"):
        parts.append(f"(时间: {msg['time']})")
    if msg.get("place"):
        parts.append(f"(地点: {msg['place']})")
    return " ".join(p for p in parts if p)

def extract_trajectory(traj):
    """提取对话历史和QA"""
    messages = [format_message(m) for m in traj.get("message_list", [])]
    qa = traj.get("QA", {})
    return messages, qa

def build_and_retrieve(query, messages, model, top_k=5):
    """向量+BM25 RRF检索"""
    if not messages:
        return []
    
    texts = messages
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    
    # Vector
    q_emb = model.encode([query], show_progress_bar=False)[0]
    vec_scores = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    
    # BM25
    docs_tokens = [re.findall(r'\w+', t.lower()) for t in texts]
    # Also do character-level for Chinese
    char_tokens = [list(t) for t in texts]
    
    df = defaultdict(int)
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] += 1
    for tokens in char_tokens:
        for t in set(tokens):
            df[t] += 1
    
    N = len(texts)
    avg_len = sum(len(t) + len(c) for t, c in zip(docs_tokens, char_tokens)) / max(N, 1)
    
    qt_words = re.findall(r'\w+', query.lower())
    qt_chars = list(query)
    
    k1, b = 1.5, 0.75
    bm25_scores = np.zeros(len(texts))
    for i, (tokens, ctokens) in enumerate(zip(docs_tokens, char_tokens)):
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t in ctokens:
            tf[t] += 1
        score = 0.0
        for t in set(qt_words + qt_chars):
            if t in tf:
                idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                doc_len = len(tokens) + len(ctokens)
                score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * doc_len / max(avg_len, 1)))
        bm25_scores[i] = score
    
    # RRF fusion
    k = 40
    rrf_scores = np.zeros(len(texts))
    vec_rank = np.argsort(vec_scores)[::-1]
    bm25_rank = np.argsort(bm25_scores)[::-1]
    for rank, idx in enumerate(vec_rank[:20]):
        rrf_scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_rank[:20]):
        rrf_scores[idx] += 1.0 / (k + rank + 1)
    
    top_indices = np.argsort(rrf_scores)[::-1][:top_k]
    return [texts[i] for i in top_indices if rrf_scores[i] > 0]

def llm_answer_mcq(question, choices, retrieved):
    """用MiMo回答MCQ题"""
    context = "\n".join([f"- {r[:200]}" for r in retrieved[:5]])
    choices_text = "\n".join([f"{k}. {v}" for k, v in sorted(choices.items())])
    
    prompt = f"""根据以下对话记忆，回答选择题。

对话记忆:
{context}

问题: {question}

选项:
{choices_text}

请只输出正确选项的字母(A/B/C/D)，不要输出其他内容。"""
    
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    data = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 2048, "temperature": 0.1}
    
    try:
        resp = requests.post(f"{LLM_API_BASE}/chat/completions", headers=headers, json=data, timeout=LLM_TIMEOUT)
        if resp.status_code == 200:
            result = resp.json()
            msg = result["choices"][0]["message"]
            content = msg.get("content", "").strip()
            reasoning = msg.get("reasoning_content", "").strip()
            # MiMo reasoning model: answer may be in reasoning_content if content is empty
            combined = content + " " + reasoning
            # 提取A/B/C/D — 优先找明确答案标记
            # Pattern 1: "答案是D" / "选D" / "D."
            for pattern in [r"答案[是为：:]\s*([ABCD])", r"选\s*([ABCD])", r"正确选项[是为：:]\s*([ABCD])", r"^([ABCD])[.。]"]:
                m = re.search(pattern, combined)
                if m:
                    return m.group(1)
            # Pattern 2: last occurrence of A/B/C/D as standalone
            matches = re.findall(r'\b([ABCD])\b', combined)
            if matches:
                return matches[-1]
            return "?"
        return "?"
    except:
        return "?"

def run_benchmark():
    print("=" * 70, flush=True)
    print("MemSim 中文记忆基准测试", flush=True)
    print(f"Date: {datetime.now().isoformat()}", flush=True)
    print(f"Dataset: MemSim/MemDaily (中文日常生活记忆模拟)", flush=True)
    print(f"Sample: {SAMPLE_PER_SPLIT}×6 splits = {SAMPLE_PER_SPLIT*6}题", flush=True)
    print(f"LLM: {LLM_MODEL}", flush=True)
    print(f"评分: exact match (A/B/C/D)", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load data
    print("\n[1/3] Loading MemSim dataset...", flush=True)
    data = load_data()
    for split in SPLITS:
        scenarios = data[split]
        total_q = sum(1 for s in scenarios.values() for t in s if t.get("QA"))
        print(f"  {split}: {total_q} questions", flush=True)
    
    # 2. Load model
    print("\n[2/3] Loading embedding model...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded: all-MiniLM-L6-v2", flush=True)
    
    # 3. Run evaluation
    print(f"\n[3/3] Running evaluation ({SAMPLE_PER_SPLIT*6} questions)...", flush=True)
    
    all_results = {}
    random.seed(42)
    
    for split in SPLITS:
        scenarios = data[split]
        trajectories = []
        for scenario, trajs in scenarios.items():
            for traj in trajs:
                traj["_scenario"] = scenario
                trajectories.append(traj)
        
        random.shuffle(trajectories)
        samples = trajectories[:SAMPLE_PER_SPLIT]
        
        correct = 0
        total = 0
        sample_details = []
        
        for i, traj in enumerate(samples):
            messages, qa = extract_trajectory(traj)
            if not qa or not messages:
                continue
            
            question = qa.get("question", "")
            choices = qa.get("choices", {})
            gold = qa.get("ground_truth", "")
            qid = qa.get("qid", f"q{i}")
            
            # Retrieve
            retrieved = build_and_retrieve(question, messages, model, top_k=5)
            
            # LLM answer
            predicted = llm_answer_mcq(question, choices, retrieved)
            
            is_correct = (predicted == gold)
            if is_correct:
                correct += 1
            total += 1
            
            if len(sample_details) < 3:
                sample_details.append({
                    "question": question[:80],
                    "gold": gold,
                    "predicted": predicted,
                    "correct": is_correct,
                })
            
            if (i + 1) % 5 == 0:
                print(f"  [{split}] {i+1}/{len(samples)}: acc={correct}/{total} ({correct/max(total,1)*100:.0f}%)", flush=True)
        
        acc = correct / max(total, 1) * 100
        all_results[split] = {
            "count": total,
            "correct": correct,
            "accuracy": round(acc, 2),
        }
        print(f"  [{split}] DONE: {correct}/{total} = {acc:.1f}%", flush=True)
        for d in sample_details:
            print(f"    Q: {d['question']}... gold={d['gold']} pred={d['predicted']} {'✅' if d['correct'] else '❌'}", flush=True)
    
    # Summary
    total_correct = sum(r["correct"] for r in all_results.values())
    total_count = sum(r["count"] for r in all_results.values())
    overall_acc = total_correct / max(total_count, 1) * 100
    
    print(f"\n{'='*70}", flush=True)
    print(f"MemSim 中文记忆基准结果", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Split':<20} {'N':>5} {'Correct':>8} {'Accuracy':>10}", flush=True)
    print(f"{'-'*45}", flush=True)
    for split in SPLITS:
        r = all_results[split]
        print(f"{split:<20} {r['count']:>5} {r['correct']:>8} {r['accuracy']:>9.1f}%", flush=True)
    print(f"{'-'*45}", flush=True)
    print(f"{'OVERALL':<20} {total_count:>5} {total_correct:>8} {overall_acc:>9.1f}%", flush=True)
    
    # Save
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "dataset": "MemSim/MemDaily (中文)",
        "source": "https://github.com/nuster1128/MemSim",
        "sample_size": total_count,
        "sample_per_split": SAMPLE_PER_SPLIT,
        "retrieval": "Vector(all-MiniLM-L6-v2) + BM25(char+word) + RRF(k=40)",
        "llm": LLM_MODEL,
        "scoring": "exact match (MCQ A/B/C/D)",
        "overall": {"accuracy": round(overall_acc, 2)},
        "by_split": all_results,
        "amb_leaderboard_note": "AMB排行榜上memsim尚无任何框架跑出结果, 我们是第一个",
    }
    
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_memsim_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存: {result_path}", flush=True)

if __name__ == "__main__":
    run_benchmark()
