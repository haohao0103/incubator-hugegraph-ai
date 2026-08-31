"""Cross-encoder rerank evaluation for NL2SQL schema linking.

Compares the PPR retrieval baseline against two-stage retrieval
(recall a candidate_k-wide pool -> cross-encoder rescore -> top_k) and
reports R@1 / R@3 / R@5 / MRR on the 23-question warehouse stress set.

This script lives next to its helpers (p2_corpus / p2_embedder / p2_stress)
under hugegraph-llm/nl2sql_tools/, so the path setup only has to make that
directory and the package `src/` importable.

Usage:
    PYTHONPATH=hugegraph-llm/src P2_MODEL=BAAI/bge-small-zh-v1.5 \\
      hugegraph-llm/.venv/bin/python hugegraph-llm/nl2sql_tools/p2_rerank_eval.py \\
      [--candidates 10 20 30] [--alphas 0.3 0.5] [--model BAAI/bge-reranker-base]

Log: _out/rerank/logs/rerank_eval.log
"""

import argparse
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))            # .../nl2sql_tools
sys.path.insert(0, _THIS)
_HGLLM = os.path.dirname(_THIS)                              # .../hugegraph-llm
sys.path.insert(0, os.path.join(_HGLLM, "src"))

from p2_corpus import build_warehouse_schema  # noqa: E402
from p2_embedder import make_embedder  # noqa: E402
from p2_stress import aggregate, evaluate  # noqa: E402

from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from hugegraph_llm.nl2sql.rerank import CrossEncoderReranker  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(_HGLLM))             # .../incubator-hugegraph-ai
_LOG = os.path.join(_REPO, "_out", "rerank", "logs", "rerank_eval.log")


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(_LOG), exist_ok=True)
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _row(label: str, a: dict) -> str:
    return (
        f"{label:28} | R@1 {a['recall@1']:.3f} | R@3 {a['recall@3']:.3f} | "
        f"R@5 {a['recall@5']:.3f} | MRR {a['mrr']:.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, nargs="*", default=[10, 20, 30])
    ap.add_argument("--alphas", type=float, nargs="*", default=[])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    _log("=" * 78)
    _log("cross-encoder rerank evaluation (two-stage schema linking)")

    schema = build_warehouse_schema()
    embedder = make_embedder()
    _log(f"corpus: {len(schema.tables())} tables, {len(schema.columns())} columns")

    # ---- baseline: single-stage PPR + importance rerank ----
    base = SchemaLinker(schema, embedder=embedder, top_k_vector=3, vector_weight=0.5)
    b = aggregate(evaluate(base, "P0+P2"))
    _log(_row("baseline (no rerank)", b))

    reranker = CrossEncoderReranker(model_name=args.model)
    if not reranker.available:
        _log("RERANKER UNAVAILABLE — model not loaded, aborting variants")
        return 1
    _log(f"reranker model: {reranker.model_name}")

    best = ("baseline", b)
    results = [("baseline (no rerank)", b)]

    # ---- two-stage: pure cross-encoder ordering at several pool widths ----
    for ck in args.candidates:
        rk = CrossEncoderReranker(model_name=args.model, candidate_k=ck)
        lk = SchemaLinker(
            schema, embedder=embedder, top_k_vector=3, vector_weight=0.5,
            reranker=rk,
        )
        a = aggregate(evaluate(lk, "P0+P2"))
        label = f"rerank ce-only k={ck}"
        _log(_row(label, a))
        results.append((label, a))
        if a["mrr"] > best[1]["mrr"]:
            best = (label, a)

    # ---- two-stage with blended PPR + CE score ----
    for alpha in args.alphas:
        for ck in args.candidates[:1]:
            rk = CrossEncoderReranker(
                model_name=args.model, candidate_k=ck, alpha=alpha
            )
            lk = SchemaLinker(
                schema, embedder=embedder, top_k_vector=3, vector_weight=0.5,
                reranker=rk,
            )
            a = aggregate(evaluate(lk, "P0+P2"))
            label = f"rerank blend a={alpha} k={ck}"
            _log(_row(label, a))
            results.append((label, a))
            if a["mrr"] > best[1]["mrr"]:
                best = (label, a)

    _log("-" * 78)
    _log(f"BEST MRR: {best[0]} -> {best[1]['mrr']:.3f} "
         f"(baseline {b['mrr']:.3f}, delta {best[1]['mrr'] - b['mrr']:+.3f})")
    _log(f"BEST R@5: {best[1]['recall@5']:.3f} (baseline {b['recall@5']:.3f}, "
         f"delta {best[1]['recall@5'] - b['recall@5']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
