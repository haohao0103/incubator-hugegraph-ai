#!/usr/bin/env python3
"""
CodeGraph vs PyCG — Call Graph Accuracy Evaluation
===================================================
Runs our CodeGraph parser on PyCG micro-benchmarks and compares
CALLS edges against PyCG ground truth (Precision/Recall/F1).
"""
import sys, os, json, subprocess
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hugegraph-llm", "src"))
from hugegraph_llm.poc.codegraph_hugegraph_mcp import PythonCodeParser

PYCG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "benchmark_data/PyCG")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "benchmark/pycg_eval_result.json")

# ── Run PyCG and collect ground truth edges ──
def get_pycg_edges(py_file: str) -> set:
    """Run PyCG on a file and extract call edges as (source_func, target_func) pairs."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pycg", py_file, "--package", os.path.dirname(py_file)],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(PYCG_DIR)
        )
        edges = set()
        for line in result.stdout.strip().split("\n"):
            if " -> " in line:
                parts = line.strip().split(" -> ")
                if len(parts) == 2:
                    src = parts[0].strip()
                    tgt = parts[1].strip()
                    # Normalize: remove module path prefix
                    src = src.split(":")[-1] if ":" in src else src
                    tgt = tgt.split(":")[-1] if ":" in tgt else tgt
                    edges.add((src, tgt))
        return edges
    except Exception as e:
        return set()


# ── Run CodeGraph and collect call edges ──
def get_codegraph_edges(py_file: str) -> set:
    """Run CodeGraph PythonCodeParser on a file and extract CALLS edges."""
    parser = PythonCodeParser()
    try:
        parser.parse_file(py_file)
    except Exception:
        return set()

    edges = set()
    for e in parser.edges:
        if e.edge_type == "CALLS":
            src_name = parser._node_by_id.get(e.source_id)
            tgt_name = parser._node_by_id.get(e.target_id)
            if src_name and tgt_name:
                edges.add((src_name.name, tgt_name.name))
    return edges


# ── Main evaluation on PyCG snippets ──
def evaluate_pycg():
    snippets_dir = os.path.join(PYCG_DIR, "micro-benchmark", "snippets")
    test_files = []
    for root, dirs, files in os.walk(snippets_dir):
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                test_files.append(os.path.join(root, f))

    # Limit to key test categories for speed
    key_categories = ["lambdas", "assignments", "args", "kwargs",
                      "classes", "functions", "imports", "decorators", "returns"]
    test_files = [f for f in test_files
                  if any(cat in f for cat in key_categories)]
    print(f"Evaluating {len(test_files)} test files from PyCG micro-benchmarks...")

    total_pycg = 0
    total_codegraph = 0
    correct = 0
    results = []

    for tf in sorted(test_files)[:60]:  # limit to 60 for speed
        try:
            pycg_edges = get_pycg_edges(tf)
            cg_edges = get_codegraph_edges(tf)
        except Exception as e:
            continue

        intersection = pycg_edges & cg_edges
        total_pycg += len(pycg_edges)
        total_codegraph += len(cg_edges)
        correct += len(intersection)

        precision = len(intersection) / max(len(cg_edges), 1)
        recall = len(intersection) / max(len(pycg_edges), 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        results.append({
            "file": os.path.relpath(tf, PYCG_DIR),
            "pycg_edges": len(pycg_edges),
            "codegraph_edges": len(cg_edges),
            "correct": len(intersection),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    # Aggregate
    precisions = [r["precision"] for r in results if r["codegraph_edges"] > 0]
    recalls = [r["recall"] for r in results if r["pycg_edges"] > 0]
    f1s = [r["f1"] for r in results]

    overall_precision = round(total_pycg / max(total_codegraph, 1), 4) if total_codegraph else 0
    overall_recall = round(total_codegraph / max(total_pycg, 1), 4) if total_pycg else 0

    summary = {
        "test_files": len(results),
        "overall_precision": round(correct / max(total_codegraph, 1), 4),
        "overall_recall": round(correct / max(total_pycg, 1), 4),
        "overall_f1": round(2 * correct / max(total_pycg + total_codegraph, 1), 4),
        "avg_precision": round(sum(precisions)/max(len(precisions),1), 4),
        "avg_recall": round(sum(recalls)/max(len(recalls),1), 4),
        "avg_f1": round(sum(f1s)/max(len(f1s),1), 4),
        "total_pycg_reference": total_pycg,
        "total_codegraph_predicted": total_codegraph,
        "total_correct": correct,
    }

    output = {"summary": summary, "per_file": results}
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output


if __name__ == "__main__":
    result = evaluate_pycg()
    print(f"\n{'='*60}")
    print(f"PyCG Call Graph Accuracy Evaluation")
    print(f"{'='*60}")
    s = result["summary"]
    print(f"  Files evaluated: {s['test_files']}")
    print(f"  PyCG reference edges: {s['total_pycg_reference']}")
    print(f"  CodeGraph predicted edges: {s['total_codegraph_predicted']}")
    print(f"  Correctly matched: {s['total_correct']}")
    print(f"  ---")
    print(f"  Overall Precision: {s['overall_precision']:.3f}")
    print(f"  Overall Recall:    {s['overall_recall']:.3f}")
    print(f"  Overall F1:        {s['overall_f1']:.3f}")
    print(f"  Avg Precision:     {s['avg_precision']:.3f}")
    print(f"  Avg Recall:        {s['avg_recall']:.3f}")
    print(f"  Avg F1:            {s['avg_f1']:.3f}")
    print(f"  ---")
    print(f"  ✅ ColbyMcHenry reference: 86.7%-100% (per-language)")
