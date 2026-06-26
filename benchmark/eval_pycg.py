#!/usr/bin/env python3
"""
CodeGraph vs PyCG — Call Graph Accuracy Evaluation
===================================================
Uses the PyCG micro-benchmark (industry-standard Python call-graph ground truth)
to measure Precision / Recall / F1 of CodeGraph's CALLS edge extraction.

Run:
    python benchmark/eval_pycg.py
Output:
    benchmark/pycg_eval_result.json
"""
import sys
import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hugegraph-llm", "src"))
from hugegraph_llm.poc.codegraph_hugegraph_mcp import PythonCodeParser, find_python_files

PYCG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "benchmark_data", "PyCG")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "pycg_eval_result.json")

# PyCG's package name collides with its internal lowercase import on case-insensitive FS.
# We register a sys.modules alias so ``from pycg import ...`` resolves to ``PyCG``.
import PyCG as _PyCG_package  # noqa: E402
sys.modules["pycg"] = _PyCG_package
from PyCG.pycg import CallGraphGenerator  # noqa: E402
from PyCG import utils as _pycg_utils  # noqa: E402


def _filename_to_modname(filename: str) -> str:
    """main.py -> main, __init__.py -> package name (caller decides)."""
    base = os.path.basename(filename)
    if base.endswith(".py"):
        base = base[:-3]
    return base


def run_pycg(entry_file: str, package: str) -> Tuple[Dict, Dict]:
    """Run PyCG Python API and return (call_graph dict, module_info dict)."""
    gen = CallGraphGenerator(
        entry_points=[entry_file],
        package=package,
        max_iter=1,
        operation=_pycg_utils.constants.CALL_GRAPH_OP,
    )
    gen.analyze()
    cg = gen.output()           # {caller_ns: set(callee_ns)}
    mods = gen.output_internal_mods()
    return cg, mods


def run_codegraph(package: str, max_files: int = 20) -> PythonCodeParser:
    """Run our PythonCodeParser on all .py files under a package."""
    parser = PythonCodeParser()
    files = find_python_files(package, max_files=max_files)
    for fp in files:
        try:
            parser.parse_file(fp)
        except Exception as exc:
            print(f"    [CodeGraph] skip {fp}: {exc}")
    return parser


def _pycg_ns_to_loc(ns: str, mods: Dict, package: str) -> Tuple[str, str]:
    """
    Convert a PyCG namespace like 'main' or 'main.func' or 'main.Class.method'
    into (relative_filename, func_name). For module-level namespaces func_name is '__module__'.
    Returns (None, None) if we cannot map it to an internal file.
    """
    if ns in mods:
        filename = mods[ns]["filename"]
        return (filename, "__module__")

    if "." not in ns:
        return (None, None)

    # Split into module namespace and member name.
    mod_ns, member = ns.rsplit(".", 1)
    while mod_ns and mod_ns not in mods:
        if "." not in mod_ns:
            break
        mod_ns, _ = mod_ns.rsplit(".", 1)

    if mod_ns in mods:
        filename = mods[mod_ns]["filename"]
        return (filename, member)

    return (None, None)


def _collect_codegraph_call_edges(parser: PythonCodeParser, package: str) -> Set[Tuple[str, str, str]]:
    """
    Normalize CodeGraph calls edges to (relative_filename, source_name, target_name).
    source_name is '__module__' for module-level callers.
    """
    node_by_id = {n.id: n for n in parser.nodes}
    edges: Set[Tuple[str, str, str]] = set()

    for e in parser.edges:
        if e.edge_type != "calls":
            continue
        src_node = node_by_id.get(e.source_id)
        if src_node is None:
            continue
        src_name = "__module__" if src_node.node_type == "module" else src_node.name
        rel_file = os.path.relpath(os.path.normpath(src_node.file_path), os.path.normpath(package))
        edges.add((rel_file, src_name, e.target_id))
    return edges


def _collect_pycg_call_edges(cg: Dict, mods: Dict, package: str) -> Set[Tuple[str, str, str]]:
    """
    Normalize PyCG call graph to (relative_filename, source_name, target_name).
    Target function name is taken from the last component of the callee namespace.
    """
    edges: Set[Tuple[str, str, str]] = set()
    for caller_ns, callees in cg.items():
        src_file, src_name = _pycg_ns_to_loc(caller_ns, mods, package)
        if src_file is None:
            continue
        for callee_ns in callees:
            tgt_file, tgt_name = _pycg_ns_to_loc(callee_ns, mods, package)
            if tgt_file is None:
                continue
            # We compare at function-name level because CodeGraph stores target as bare name.
            edges.add((src_file, src_name, tgt_name))
    return edges


def evaluate_pycg(max_snippets: int = 80):
    snippets_dir = os.path.join(PYCG_DIR, "micro-benchmark", "snippets")
    # Collect leaf directories that contain a main.py
    snippet_dirs = []
    for root, dirs, files in os.walk(snippets_dir):
        if "main.py" in files:
            snippet_dirs.append(root)

    print(f"Found {len(snippet_dirs)} PyCG micro-benchmark snippets.")

    total_ref = 0
    total_pred = 0
    total_correct = 0
    per_snippet = []

    for idx, sdir in enumerate(sorted(snippet_dirs)[:max_snippets]):
        entry = os.path.join(sdir, "main.py")
        try:
            cg, mods = run_pycg(entry, sdir)
            parser = run_codegraph(sdir)
        except Exception as exc:
            print(f"  [{idx+1}] SKIP {os.path.relpath(sdir, PYCG_DIR)}: {exc}")
            continue

        ref_edges = _collect_pycg_call_edges(cg, mods, sdir)
        pred_edges = _collect_codegraph_call_edges(parser, sdir)

        correct = len(ref_edges & pred_edges)
        total_ref += len(ref_edges)
        total_pred += len(pred_edges)
        total_correct += correct

        p = correct / max(len(pred_edges), 1)
        r = correct / max(len(ref_edges), 1)
        f1 = 2 * p * r / max(p + r, 0.0001)

        per_snippet.append({
            "directory": os.path.relpath(sdir, PYCG_DIR),
            "ref_edges": len(ref_edges),
            "pred_edges": len(pred_edges),
            "correct": correct,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })

        if (idx + 1) % 20 == 0:
            print(f"  ... evaluated {idx+1}/{min(len(snippet_dirs), max_snippets)} snippets")

    overall_p = total_correct / max(total_pred, 1)
    overall_r = total_correct / max(total_ref, 1)
    overall_f1 = 2 * overall_p * overall_r / max(overall_p + overall_r, 0.0001)

    valid_p = [s["precision"] for s in per_snippet if s["pred_edges"] > 0]
    valid_r = [s["recall"] for s in per_snippet if s["ref_edges"] > 0]
    valid_f1 = [s["f1"] for s in per_snippet]

    summary = {
        "snippets_evaluated": len(per_snippet),
        "overall_precision": round(overall_p, 4),
        "overall_recall": round(overall_r, 4),
        "overall_f1": round(overall_f1, 4),
        "avg_precision": round(sum(valid_p) / max(len(valid_p), 1), 4),
        "avg_recall": round(sum(valid_r) / max(len(valid_r), 1), 4),
        "avg_f1": round(sum(valid_f1) / max(len(valid_f1), 1), 4),
        "total_reference_edges": total_ref,
        "total_predicted_edges": total_pred,
        "total_correct_edges": total_correct,
        "dataset": "PyCG micro-benchmark snippets",
        "note": "Function-name-level comparison; target file is not matched because CodeGraph stores callee as bare name.",
    }

    output = {"summary": summary, "per_snippet": per_snippet}
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output


if __name__ == "__main__":
    result = evaluate_pycg()
    print(f"\n{'='*60}")
    print("CodeGraph vs PyCG Call-Graph Accuracy")
    print(f"{'='*60}")
    s = result["summary"]
    print(f"  Snippets evaluated: {s['snippets_evaluated']}")
    print(f"  Reference edges (PyCG): {s['total_reference_edges']}")
    print(f"  Predicted edges (CodeGraph): {s['total_predicted_edges']}")
    print(f"  Correctly matched: {s['total_correct_edges']}")
    print(f"  ---")
    print(f"  Overall Precision: {s['overall_precision']:.3f}")
    print(f"  Overall Recall:    {s['overall_recall']:.3f}")
    print(f"  Overall F1:        {s['overall_f1']:.3f}")
    print(f"  Avg Precision:     {s['avg_precision']:.3f}")
    print(f"  Avg Recall:        {s['avg_recall']:.3f}")
    print(f"  Avg F1:            {s['avg_f1']:.3f}")
    print(f"  ---")
    print(f"  Output: {OUTPUT}")
