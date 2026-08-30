"""P2 semantic recall stress test for NL2SQL schema linking.

Compares three linking modes over the labeled warehouse question set:
  - P0   : lexical seeds only            (SchemaLinker(embedder=None))
  - P2   : vector seeds only             (semantic, lexical disabled via internals)
  - P0+P2: combined (production mode)    (SchemaLinker(embedder=real))

Metrics per mode: recall@k (any-gold), gold-fraction@k (stricter, all gold),
MRR, and coverage. Reported overall and by question category (lexical /
semantic / join), plus a P0-missed lift breakdown (the questions P0 fails on,
where P2 must earn its keep).

Run:
  PYTHONPATH=incubator-hugegraph-ai/hugegraph-llm/src \
    /path/to/hg-llm/python scripts/p2_stress.py
Env: P2_MODEL (default shibing624/text2vec-base-chinese)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from p2_corpus import (  # noqa: E402
    build_warehouse_schema, WAREHOUSE_QUESTIONS, gold_node_ids, gold_table_ids,
)
from p2_embedder import make_embedder, model_dimension  # noqa: E402

from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402

LOG_PATH = "_out/p2_stress/logs/p2_stress.log"
REPORT_MD = "_out/p2_stress/p2_stress_report.md"
REPORT_JSON = "_out/p2_stress/p2_stress_report.json"
K_VALUES = [1, 3, 5, 10]
TOP_K = 10


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _link_lexical(linker, q):
    return linker.link(q, top_k=TOP_K)


def _link_vector_only(linker, q):
    linker._ensure_vector_index()
    seeds = linker._vector_seeds(q)
    scores = linker._ppr(seeds)
    return linker._to_items(scores, True, True)[:TOP_K]


def _returned_ids(items):
    return [it.node_id for it in items]


def evaluate(linker, mode, embedder=None):
    rows = []
    for q in WAREHOUSE_QUESTIONS:
        if mode == "P0":
            items = _link_lexical(linker, q["q"])
        elif mode == "P2":
            items = _link_vector_only(linker, q["q"])
        else:  # P0+P2
            items = linker.link(q["q"], top_k=TOP_K)
        ids = _returned_ids(items)
        gids = set(gold_node_ids(q))
        tgids = set(gold_table_ids(q))
        # position of first gold column in ranked list
        rank = None
        for i, nid in enumerate(ids, 1):
            if nid in gids:
                rank = i
                break
        retrieved_gold = gids & set(ids)
        # lenient: gold table also counts
        lenient_hit = bool(retrieved_gold) or bool(tgids & set(ids))
        rows.append({
            "q": q["q"], "category": q["category"],
            "gold": q["gold"], "gold_ids": sorted(gids),
            "rank": rank, "n_gold": len(gids),
            "n_retrieved_gold": len(retrieved_gold),
            "gold_frac": len(retrieved_gold) / len(gids) if gids else 0.0,
            "any_hit": bool(retrieved_gold),
            "lenient_hit": lenient_hit,
            "top": ids[:5],
        })
    return rows


def aggregate(rows, key=None):
    sub = rows if key is None else [r for r in rows if r["category"] == key]
    n = len(sub)
    if n == 0:
        return {}
    agg = {"n": n}
    for k in K_VALUES:
        any_hit = sum(1 for r in sub
                      if any(g in r["top"][:k] for g in r["gold_ids"]))
        # recompute gold_frac@k over first k items
        gf = 0.0
        for r in sub:
            first_k = set(r["top"][:k])
            gf += len(set(r["gold_ids"]) & first_k) / r["n_gold"] if r["n_gold"] else 0
        agg[f"recall@{k}"] = any_hit / n
        agg[f"goldfrac@{k}"] = gf / n
    mrr_num = sum(1.0 / r["rank"] for r in sub if r["rank"])
    agg["mrr"] = mrr_num / n
    agg["coverage"] = sum(1 for r in sub if r["any_hit"]) / n
    return agg


def fmt_agg(a):
    if not a:
        return "-"
    parts = [f"n={a['n']}", f"cov={a['coverage']:.2f}", f"MRR={a['mrr']:.3f}"]
    for k in K_VALUES:
        parts.append(f"R@{k}={a[f'recall@{k}']:.2f}/gf={a[f'goldfrac@{k}']:.2f}")
    return "  ".join(parts)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== P2 semantic recall stress test ===")

    t0 = time.time()
    schema = build_warehouse_schema()
    log(f"schema built: tables={len(schema.tables())} "
        f"columns={len(schema.columns())} terms={len(schema.terms())} "
        f"edges={len(schema.edges)}")
    log(f"questions={len(WAREHOUSE_QUESTIONS)}")

    # P0 linker (lexical only)
    p0 = SchemaLinker(schema)
    p0_rows = evaluate(p0, "P0")
    log("P0 (lexical) evaluated")

    # load embedder
    emb_t0 = time.time()
    embedder = make_embedder()
    dim = model_dimension()
    log(f"embedder loaded in {time.time()-emb_t0:.1f}s dim={dim}")

    # P0+P2 linker (combined, production mode)
    comb = SchemaLinker(schema, embedder=embedder)
    comb_rows = evaluate(comb, "P0+P2")
    log("P0+P2 (combined) evaluated")

    # P2-only (vector seeds only, lexical disabled via internals)
    p2 = SchemaLinker(schema, embedder=embedder)
    p2_rows = evaluate(p2, "P2")
    log("P2 (vector-only) evaluated")

    log(f"total time {time.time()-t0:.1f}s")

    # ---- reporting ----
    cats = ["lexical", "semantic", "join"]
    report = {
        "schema": {"tables": len(schema.tables()),
                   "columns": len(schema.columns()),
                   "terms": len(schema.terms()),
                   "edges": len(schema.edges)},
        "embedder_dim": dim,
        "modes": {},
    }
    summary_lines = []
    for name, rows in [("P0_lexical", p0_rows), ("P2_vector", p2_rows),
                       ("P0+P2_combined", comb_rows)]:
        overall = aggregate(rows)
        by_cat = {c: aggregate(rows, c) for c in cats}
        report["modes"][name] = {"overall": overall, "by_category": by_cat}
        summary_lines.append(f"## {name}")
        summary_lines.append(f"- overall: {fmt_agg(overall)}")
        for c in cats:
            summary_lines.append(f"- {c}: {fmt_agg(by_cat[c])}")
        summary_lines.append("")

    # P0-missed lift: questions where P0 any_hit == False
    p0_miss = [r for r in p0_rows if not r["any_hit"]]
    lift = {
        "n_p0_miss": len(p0_miss),
        "P2_recall@5_on_p0miss": aggregate(p2_rows, None) and None,  # placeholder
    }
    # compute recall@5 on p0-miss subset for P2 and combined
    p0miss_qs = {r["q"] for r in p0_miss}
    p2_on_miss = [r for r in p2_rows if r["q"] in p0miss_qs]
    comb_on_miss = [r for r in comb_rows if r["q"] in p0miss_qs]
    lift["P2_recall@5"] = aggregate(p2_on_miss)["recall@5"] if p2_on_miss else 0
    lift["P0+P2_recall@5"] = aggregate(comb_on_miss)["recall@5"] if comb_on_miss else 0
    lift["P2_goldfrac@5"] = aggregate(p2_on_miss)["goldfrac@5"] if p2_on_miss else 0
    lift["P0+P2_goldfrac@5"] = aggregate(comb_on_miss)["goldfrac@5"] if comb_on_miss else 0
    report["p0_missed_lift"] = lift

    summary_lines.append("## P0-missed lift (questions P0 lexical could NOT hit)")
    summary_lines.append(f"- P0-missed questions: {len(p0_miss)} / {len(p0_rows)}")
    if p0_miss:
        summary_lines.append(f"  - P2 recall@5 on these: {lift['P2_recall@5']:.2f} "
                             f"(goldfrac {lift['P2_goldfrac@5']:.2f})")
        summary_lines.append(f"  - P0+P2 recall@5 on these: {lift['P0+P2_recall@5']:.2f} "
                             f"(goldfrac {lift['P0+P2_goldfrac@5']:.2f})")
        summary_lines.append("  - missed questions: " + ", ".join(r["q"] for r in p0_miss))

    # per-question detail table (markdown)
    detail = ["## Per-question detail", "",
              "| question | cat | gold | P0 rank | P2 rank | P0+P2 rank | P0+P2 top5 |",
              "|---|---|---|---|---|---|---|"]
    p0_by_q = {r["q"]: r for r in p0_rows}
    p2_by_q = {r["q"]: r for r in p2_rows}
    comb_by_q = {r["q"]: r for r in comb_rows}
    for q in WAREHOUSE_QUESTIONS:
        p0r = p0_by_q[q["q"]]; p2r = p2_by_q[q["q"]]; cmr = comb_by_q[q["q"]]
        detail.append(
            f"| {q['q']} | {q['category']} | {', '.join(q['gold'])} | "
            f"{p0r['rank'] or '-'} | {p2r['rank'] or '-'} | {cmr['rank'] or '-'} | "
            f"{', '.join(x.replace('column:dw.','').replace('table:dw.','T:') for x in cmr['top'])} |"
        )

    md = ("# P2 Semantic Recall Stress Test — NL2SQL Schema Linking\n\n"
          f"- schema: {len(schema.tables())} tables / "
          f"{len(schema.columns())} columns / {len(schema.terms())} terms / "
          f"{len(schema.edges)} edges\n"
          f"- embedder dim: {dim}\n"
          f"- questions: {len(WAREHOUSE_QUESTIONS)} "
          f"(lexical={sum(1 for q in WAREHOUSE_QUESTIONS if q['category']=='lexical')}, "
          f"semantic={sum(1 for q in WAREHOUSE_QUESTIONS if q['category']=='semantic')}, "
          f"join={sum(1 for q in WAREHOUSE_QUESTIONS if q['category']=='join')})\n\n"
          + "\n".join(summary_lines) + "\n\n" + "\n".join(detail) + "\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("report written: " + REPORT_MD)
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
