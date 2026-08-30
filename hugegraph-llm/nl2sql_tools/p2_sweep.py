"""Tuning sweep for P2 semantic linking: vector_weight x top_k_vector + a
P2-fallback strategy (vector seeds only when lexical misses).

Reuses evaluate()/aggregate() from p2_stress. Outputs a compact metrics table
to _out/p2_stress/logs/p2_sweep.log and prints it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from p2_corpus import build_warehouse_schema, WAREHOUSE_QUESTIONS  # noqa: E402
from p2_embedder import make_embedder  # noqa: E402
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from p2_stress import evaluate, aggregate, K_VALUES, TOP_K  # noqa: E402

LOG_PATH = "_out/p2_stress/logs/p2_sweep.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def fallback_rows(p0_rows, vector_rows):
    """Lexical-first; use vector-only results when P0 produced no hit."""
    v_by_q = {r["q"]: r for r in vector_rows}
    out = []
    for r in p0_rows:
        v = v_by_q[r["q"]]
        if r["any_hit"]:
            items = r["top"]
        else:
            items = v["top"]
        rank = None
        for i, nid in enumerate(items, 1):
            if nid in set(r["gold_ids"]):
                rank = i
                break
        out.append({**r, "top": items, "rank": rank,
                    "any_hit": rank is not None})
    return out


def main():
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
    schema = build_warehouse_schema()
    log(f"schema tables={len(schema.tables())} cols={len(schema.columns())} "
        f"terms={len(schema.terms())} questions={len(WAREHOUSE_QUESTIONS)}")

    p0 = SchemaLinker(schema)
    p0_rows = evaluate(p0, "P0")
    embedder = make_embedder()
    log(f"embedder dim={len(embedder('x'))}")

    header = (f"{'config':42} | {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>6} "
              f"{'cov':>5} | {'semR@5':>7} {'p0missR@5':>9}")
    log(header)
    log("-" * len(header))

    rows = {}

    def report(name, rows_):
        agg = aggregate(rows_)
        sem = aggregate(rows_, "semantic")
        p0miss = [r for r in p0_rows if not r["any_hit"]]
        miss_qs = {r["q"] for r in p0miss}
        on_miss = [r for r in rows_ if r["q"] in miss_qs]
        miss_agg = aggregate(on_miss) if on_miss else {}
        line = (f"{name:42} | {agg['recall@1']:5.2f} {agg['recall@3']:5.2f} "
                f"{agg['recall@5']:5.2f} {agg['mrr']:6.3f} {agg['coverage']:5.2f}"
                f" | {sem['recall@5']:7.2f} "
                f"{miss_agg.get('recall@5', 0):9.2f}")
        log(line)
        rows[name] = line

    report("P0 (baseline)", p0_rows)

    for w in (0.5, 0.7, 0.9, 1.0):
        for k in (3, 5):
            lk = SchemaLinker(schema, embedder=embedder,
                              vector_weight=w, top_k_vector=k)
            r = evaluate(lk, "P0+P2")
            report(f"P0+P2 w={w} k={k}", r)
        if w == 0.9:
            lk = SchemaLinker(schema, embedder=embedder,
                              vector_weight=w, top_k_vector=8)
            report(f"P0+P2 w={w} k=8", evaluate(lk, "P0+P2"))

    # P2-fallback strategies
    for w, k in ((0.9, 5), (0.7, 5), (1.0, 8)):
        lk = SchemaLinker(schema, embedder=embedder,
                          vector_weight=w, top_k_vector=k)
        v_rows = evaluate(lk, "P2")
        report(f"P2-fallback w={w} k={k}", fallback_rows(p0_rows, v_rows))

    log("DONE")


if __name__ == "__main__":
    main()
