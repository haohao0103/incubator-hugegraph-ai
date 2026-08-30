"""Scale benchmark: LocalEngine vs VermeerEngine PPR on synthetic schemas.

Synthetic warehouse schemas of N tables x 10 columns with FK chains / lineage /
terms, then measures:
  * BM25 index build time;
  * per-question PPR latency (p50/p99) on LocalEngine;
  * VermeerEngine PPR latency (cluster) + top-k agreement vs LocalEngine.

Vermeer cluster must be running (scripts/vermeer_cluster.py start).

Usage:
  python scripts/benchmark_scale.py [--max-tables 300] [--out _out/benchmark/scale_report.md]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from hugegraph_llm.nl2sql.engine.local import LocalEngine  # noqa: E402
from hugegraph_llm.nl2sql.engine.vermeer import VermeerEngine  # noqa: E402
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from hugegraph_llm.nl2sql.schema_graph.model import (  # noqa: E402
    Column, Edge, EdgeType, SchemaGraph, Table, Term,
)

LOG_PATH = "_out/benchmark/logs/benchmark.log"
VERMEER_URL = "http://127.0.0.1:6688"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def gen_large_schema(n_tables: int, cols_per_table: int = 10) -> SchemaGraph:
    g = SchemaGraph()
    for i in range(n_tables):
        tname = f"t_{i:03d}"
        g.add_node(Table(name=tname, database="dw",
                         comment=f"合成表 {tname}", row_count=1000 + i * 100,
                         is_fact=(i % 4 == 0)).to_node())
    # phase 2: all columns first (FK edges may reference later tables)
    for i in range(n_tables):
        tname = f"t_{i:03d}"
        for j in range(cols_per_table):
            col = "id" if j == 0 else f"c_{j:02d}"
            g.add_node(Column(name=col, table=f"dw.{tname}",
                              data_type="decimal" if j % 2 else "bigint",
                              comment=f"{tname}.{col} 的字段注释",
                              is_primary_key=(j == 0)).to_node())
    # phase 3: edges
    for i in range(n_tables):
        tname = f"t_{i:03d}"
        for j in range(cols_per_table):
            col = "id" if j == 0 else f"c_{j:02d}"
            g.add_edge(Edge(source=f"column:dw.{tname}.{col}",
                            target=f"table:dw.{tname}",
                            edge_type=EdgeType.BELONGS_TO, weight=1.0))
        # chain FK: t_i.id -> t_{i+1}.c_01 (last table links back to 0)
        nxt = (i + 1) % n_tables
        g.add_edge(Edge(source=f"column:dw.{tname}.id",
                        target=f"column:dw.t_{nxt:03d}.c_01",
                        edge_type=EdgeType.FOREIGN_KEY, weight=1.0))
    # lineage every 10 tables + terms every 20 tables
    for i in range(0, n_tables, 10):
        if i + 5 < n_tables:
            g.add_edge(Edge(source=f"table:dw.t_{i:03d}",
                            target=f"table:dw.t_{i + 5:03d}",
                            edge_type=EdgeType.LINEAGE, weight=1.0))
    for i in range(0, n_tables, 20):
        tname = f"t_{i:03d}"
        g.add_node(Term(name=f"指标_{i:03d}", comment=f"指标_{i:03d} 口径说明").to_node())
        g.add_edge(Edge(source=f"term:指标_{i:03d}",
                        target=f"column:dw.{tname}.c_02",
                        edge_type=EdgeType.TERM_MAPS, weight=1.0))
    return g


def _p50_p99(latencies):
    s = sorted(latencies)
    n = len(s)
    return s[n // 2], s[int(n * 0.99) - 1] if n > 1 else s[-1]


def bench_local(schema, n_queries=20):
    t0 = time.time()
    linker = SchemaLinker(schema)
    linker.prebuild()
    build_s = time.time() - t0
    seeds = {"table:dw.t_010": 1.0, "term:指标_000": 0.7}
    lat = []
    for _ in range(n_queries):
        t1 = time.time()
        linker._ppr(seeds)
        lat.append((time.time() - t1) * 1000)
    p50, p99 = _p50_p99(lat)
    return {"build_s": round(build_s, 2), "local_p50_ms": round(p50, 2),
            "local_p99_ms": round(p99, 2), "n": len(schema.nodes),
            "tables": len(schema.tables())}


def bench_vermeer(schema, n_queries=20):
    from hugegraph_llm.nl2sql.engine.vermeer_client import VermeerClient

    client = VermeerClient(base_url=VERMEER_URL)
    t0 = time.time()
    engine = VermeerEngine(schema, client=client)
    load_s = time.time() - t0
    seeds = {"table:dw.t_010": 1.0, "term:指标_000": 0.7}
    lat = []
    for _ in range(n_queries):
        t1 = time.time()
        engine.personalized_pagerank(seeds)
        lat.append((time.time() - t1) * 1000)
    p50, p99 = _p50_p99(lat)
    return {"load_s": round(load_s, 2), "vm_p50_ms": round(p50, 2),
            "vm_p99_ms": round(p99, 2)}


def agreement(local_scores, vermeer_scores, top_k=5):
    lk = sorted(local_scores.items(), key=lambda kv: -kv[1])[:top_k]
    vk = sorted(vermeer_scores.items(), key=lambda kv: -kv[1])[:top_k]
    lset = {k for k, _ in lk}
    vset = {k for k, _ in vk}
    return len(lset & vset) / top_k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tables", type=int, default=300)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--out", default="_out/benchmark/scale_report.md")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log(f"=== scale benchmark (max {args.max_tables} tables) ===")
    rows = []
    for n in (100, args.max_tables):
        log(f"--- {n} tables ---")
        schema = gen_large_schema(n)
        lr = bench_local(schema, args.queries)
        log(f"local: {lr}")
        row = {"tables": n, **lr}
        try:
            vr = bench_vermeer(schema, args.queries)
            # agreement on a fresh local run (deterministic seeds)
            local_scores = LocalEngine(schema).personalized_pagerank(
                {"table:dw.t_010": 1.0})
            ver_scores = VermeerEngine(schema).personalized_pagerank(
                {"table:dw.t_010": 1.0})
            ag = agreement(local_scores, ver_scores)
            row.update({**vr, "top5_agreement": round(ag, 2)})
            log(f"vermeer: {vr} agreement@5={ag:.2f}")
        except Exception as exc:  # noqa: BLE001
            log(f"vermeer skipped for {n} tables: {exc}")
            row["vermeer"] = f"skipped: {exc}"
        rows.append(row)
        engine = VermeerEngine  # noqa: F841 - keep import used for typing

    md = ["# NL2SQL 规模基准（合成 schema）", "",
          f"- vermeer: {VERMEER_URL} | queries: {args.queries}", "",
          "| tables | nodes | BM25 build(s) | local p50(ms) | local p99(ms) |"
          " vm load(s) | vm p50(ms) | vm p99(ms) | top5 agree |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['tables']} | {r['n']} | {r['build_s']} | {r['local_p50_ms']} | "
            f"{r['local_p99_ms']} | {r.get('load_s', '-')} | "
            f"{r.get('vm_p50_ms', '-')} | {r.get('vm_p99_ms', '-')} | "
            f"{r.get('top5_agreement', '-')} |")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    log("report: " + args.out)
    log("DONE")


if __name__ == "__main__":
    main()
