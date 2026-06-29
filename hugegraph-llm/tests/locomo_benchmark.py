# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
LOCOMO benchmark runner for HugeGraph-AI-Memory.

LOCOMO (Long Context Multi-session Open-domain Conversation) evaluates how well
an agent remembers facts across many sessions. This runner feeds LOCOMO
dialogue sessions into the HugeGraph memory pipeline and evaluates QA accuracy.

Usage:
    python locomo_benchmark.py --data-dir ./locomo_data --max-sessions 50
    python locomo_benchmark.py --data-dir ./locomo_data --sample 20

Output:
    locomo_result.json with R@1, R@5, MRR, latency, token usage estimates.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure this workspace's source tree takes precedence over any venv-installed package.
_WORKSPACE_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _WORKSPACE_SRC not in sys.path:
    sys.path.insert(0, _WORKSPACE_SRC)

from hugegraph_llm.poc.memory_backend import MemoryPipelineBackend, HugeGraphMemoryClient
from hugegraph_llm.utils.log import log


DATA_URL = "https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json"


def download_locomo(data_dir: str) -> str:
    """Download LOCOMO data if not present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    data_file = data_path / "locomo10.json"
    if data_file.exists():
        return str(data_file)

    import urllib.request

    log.info("Downloading LOCOMO from %s ...", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, data_file)
    log.info("Saved to %s", data_file)
    return str(data_file)


def load_locomo_sessions(data_file: str, max_sessions: int = None):
    """
    Load LOCOMO sessions from the official locomo10.json format.

    Each conversation contains chronological sessions (session_1..session_n)
    with QA annotations. We flatten all turns as memory text.
    """
    data_path = Path(data_file)
    if not data_path.exists():
        log.warning("%s not found; using dummy data for smoke test", data_file)
        return _dummy_sessions()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = []
    for conv in data:
        conv_id = conv.get("sample_id", f"conv_{len(sessions)}")
        conv_data = conv.get("conversation", {})

        # Extract sessions in chronological order
        turns = []
        session_idx = 1
        while f"session_{session_idx}" in conv_data:
            session_turns = conv_data.get(f"session_{session_idx}", [])
            session_dt = conv_data.get(f"session_{session_idx}_date_time", "")
            for turn in session_turns:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "").strip()
                if text:
                    turns.append({
                        "speaker": speaker,
                        "text": text,
                        "session": session_idx,
                        "datetime": session_dt,
                    })
            session_idx += 1

        qa_pairs = []
        for qa in conv.get("qa", []):
            ans = qa.get("answer", "")
            # Normalize answer to a list of strings (LOCoMo may contain int IDs or strings)
            if isinstance(ans, list):
                answers = [str(a) for a in ans if a is not None]
            elif ans is not None and ans != "":
                answers = [str(ans)]
            else:
                answers = []
            qa_pairs.append({
                "question": qa.get("question", ""),
                "answers": answers,
                "evidence": qa.get("evidence", []),
                "type": f"category_{qa.get('category', 0)}",
            })

        sessions.append({
            "id": conv_id,
            "turns": turns,
            "qa": qa_pairs,
            "user_id": f"locomo_{conv_id}",
        })
        if max_sessions and len(sessions) >= max_sessions:
            break
    return sessions


def _dummy_sessions():
    """Minimal smoke-test data when real LOCOMO is unavailable."""
    return [
        {
            "id": "dummy_1",
            "user_id": "locomo_dummy_1",
            "turns": [
                {"speaker": "user", "text": "My name is Alice and I work at HugeGraph."},
                {"speaker": "agent", "text": "Nice to meet you, Alice."},
                {"speaker": "user", "text": "I prefer email for work updates."},
            ],
            "qa": [
                {"question": "What is Alice's name?", "answers": ["Alice"], "type": "fact"},
                {"question": "Where does Alice work?", "answers": ["HugeGraph"], "type": "fact"},
                {"question": "How does Alice prefer work updates?", "answers": ["email"], "type": "fact"},
            ],
        }
    ]


def session_to_memory_texts(session: Dict[str, Any], batch_size: int = 10) -> List[str]:
    """Convert a LOCOMO session into atomic memory statements.

    For long LOCOMO sessions we batch consecutive turns into a single memory
    entry to reduce the number of LLM entity-extraction calls while still
    preserving enough context for retrieval.
    """
    turns = [t for t in session["turns"] if t.get("text", "").strip()]
    batches = []
    for i in range(0, len(turns), batch_size):
        chunk = turns[i:i + batch_size]
        texts = [f"{t.get('speaker', 'user')}: {t.get('text', '').strip()}" for t in chunk]
        batches.append("\n".join(texts))
    return batches


def evaluate_answer(prediction: str, answers: List[str]) -> bool:
    """Case-insensitive exact/contains match."""
    pred = prediction.lower().strip(". ").strip("。")
    for ans in answers:
        if ans.lower() in pred or pred in ans.lower():
            return True
    return False


def run_locomo(
    data_dir: str,
    max_sessions: int = None,
    sample_qa: int = None,
    batch_size: int = 10,
    graph_name: str = "hugegraph",
    backend: MemoryPipelineBackend = None,
    fast_eval: bool = False,
) -> Dict[str, Any]:
    """Run the full benchmark pipeline.

    Args:
        fast_eval: If True, use retrieval-only search (no LLM classify/rerank/answer)
                   for fast metrics. Accuracy is then approximated by rank==1.
    """
    data_file = download_locomo(data_dir)
    sessions = load_locomo_sessions(data_file, max_sessions=max_sessions)
    log.info("Loaded %d LOCOMO sessions", len(sessions))

    if backend is None:
        hg_client = HugeGraphMemoryClient(graph=graph_name)
        store = MemoryPipelineBackend(hg_client=hg_client)
    else:
        store = backend

    total_q = 0
    correct = 0
    rr_sum = 0.0
    hit_at_5 = 0
    latencies = []
    token_estimate = 0

    results_per_session = []

    for session in sessions:
        user_id = session["user_id"]
        # Feed batched turns as memories (bypass intent classification: every
        # batch is a fact to remember, not a question).
        for text in session_to_memory_texts(session, batch_size=batch_size):
            try:
                store.add_memory_bypass_classify(text, user_id=user_id)
            except Exception as e:
                log.warning("add_memory failed for %s: %s", user_id, e)

        qa_pairs = session["qa"]
        if sample_qa and len(qa_pairs) > sample_qa:
            qa_pairs = qa_pairs[:sample_qa]

        session_correct = 0
        for qa in qa_pairs:
            question = qa["question"]
            answers = qa.get("answers", [])
            if not question or not answers:
                continue

            t0 = time.perf_counter()
            try:
                resp = store.search_memory(question, user_id=user_id, top_k=5, fast_eval=fast_eval)
            except Exception as e:
                log.warning("search_memory failed for %s: %s", user_id, e)
                continue
            latency = time.perf_counter() - t0
            latencies.append(latency)

            prediction = resp.get("answer", "")
            # Token estimate: input prompt + output (very rough); fast-eval uses no LLM calls
            if fast_eval:
                token_estimate += len(question) // 4 + len(prediction) // 4
            else:
                token_estimate += len(question) // 4 + len(prediction) // 4 + 200

            rank = None
            for i, r in enumerate(resp.get("results", [])):
                content = r.get("memory", {}).get("content", "")
                if any(a.lower() in content.lower() for a in answers):
                    rank = i + 1
                    break

            # In fast-eval mode, accuracy is approximated by whether the gold answer
            # appears in the top-1 retrieved memory (retrieval exact match).
            if fast_eval:
                is_correct = rank == 1
            else:
                is_correct = evaluate_answer(prediction, answers)
            if is_correct:
                correct += 1
                session_correct += 1
                hit_at_5 += 1
                rr_sum += 1.0
            elif rank is not None and rank <= 5:
                hit_at_5 += 1
                rr_sum += 1.0 / rank

            total_q += 1
            results_per_session.append({
                "session_id": session["id"],
                "question": question,
                "answers": answers,
                "prediction": prediction,
                "correct": is_correct,
                "rank": rank,
                "latency_ms": round(latency * 1000, 2),
            })

    metrics = {
        "sessions": len(sessions),
        "total_questions": total_q,
        "correct": correct,
        "accuracy": round(correct / total_q, 4) if total_q else 0,
        "hit_at_5": round(hit_at_5 / total_q, 4) if total_q else 0,
        "mrr": round(rr_sum / total_q, 4) if total_q else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2) if latencies else 0,
        "token_estimate": token_estimate,
    }

    output = {
        "metrics": metrics,
        "details": results_per_session,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description="LOCOMO benchmark for HugeGraph-AI-Memory")
    parser.add_argument("--data-dir", default="./locomo_data", help="Directory to cache LOCOMO data")
    parser.add_argument("--max-sessions", type=int, default=None, help="Max conversation sessions to evaluate")
    parser.add_argument("--sample", type=int, default=None, help="Max QA pairs per session")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch N turns into one memory entry")
    parser.add_argument("--graph-name", type=str, default=os.environ.get("HUGEGRAPH_GRAPH", "poc_memgraphrag"),
                        help="HugeGraph graph name to use")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Retrieval-only fast evaluation (no LLM classify/rerank/answer)")
    parser.add_argument("--output", default="locomo_result.json", help="Result JSON path")
    args = parser.parse_args()

    output_path = args.output
    if args.fast_eval and output_path == "locomo_result.json":
        output_path = "locomo_result_fast_eval.json"

    result = run_locomo(
        data_dir=args.data_dir,
        max_sessions=args.max_sessions,
        sample_qa=args.sample,
        batch_size=args.batch_size,
        graph_name=args.graph_name,
        fast_eval=args.fast_eval,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    metrics = result["metrics"]
    print("\n=== LOCOMO Benchmark Results ===")
    print(f"Sessions: {metrics['sessions']}")
    print(f"Questions: {metrics['total_questions']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Hit@5: {metrics['hit_at_5']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency: {metrics['avg_latency_ms']} ms")
    print(f"P95 latency: {metrics['p95_latency_ms']} ms")
    print(f"Token estimate: {metrics['token_estimate']}")
    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
