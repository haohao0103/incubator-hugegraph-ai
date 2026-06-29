"""Evaluate existing LOCOMO data in the graph with retrieval-only (no LLM)."""
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

spec = importlib.util.spec_from_file_location("locomo", str(Path(__file__).resolve().parent / "locomo_benchmark.py"))
locomo = importlib.util.module_from_spec(spec)
sys.modules["locomo"] = locomo
spec.loader.exec_module(locomo)


def main():
    data_file = "tests/locomo_data/locomo10.json"
    graph_name = "poc_memgraphrag"
    sessions = locomo.load_locomo_sessions(data_file)

    from hugegraph_llm.poc.memory_backend import MemoryPipelineBackend, HugeGraphMemoryClient

    hg_client = HugeGraphMemoryClient(graph=graph_name)
    store = MemoryPipelineBackend(hg_client=hg_client)

    total_q = 0
    correct = 0
    hit_at_5 = 0
    rr_sum = 0.0
    latencies = []
    details = []

    for session in sessions:
        user_id = session["user_id"]
        for qa in session["qa"]:
            question = qa["question"]
            answers = qa.get("answers", [])
            if not question or not answers:
                continue

            t0 = time.perf_counter()
            try:
                resp = store.search_memory(question, user_id=user_id, top_k=5, fast_eval=True, update_access_count=False)
            except Exception as e:
                print(f"search failed: {e}")
                continue
            latency = time.perf_counter() - t0
            latencies.append(latency)

            prediction = resp.get("answer", "")
            rank = None
            for i, r in enumerate(resp.get("results", [])):
                content = r.get("memory", {}).get("content", "")
                if any(a.lower() in content.lower() for a in answers):
                    rank = i + 1
                    break

            is_correct = locomo.evaluate_answer(prediction, answers)
            if is_correct:
                correct += 1
                hit_at_5 += 1
                rr_sum += 1.0
            elif rank is not None and rank <= 5:
                hit_at_5 += 1
                rr_sum += 1.0 / rank

            total_q += 1
            details.append({
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
    }

    result = {"metrics": metrics, "details": details}
    with open("tests/locomo_result_full_fast_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== LOCOMO Full Fast-Eval Results ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
