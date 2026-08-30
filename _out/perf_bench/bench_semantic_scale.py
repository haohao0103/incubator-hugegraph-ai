"""Scale benchmark for the NL2SQL semantic layer (SchemaLinker PPR).

Synthesizes growing SchemaGraphs (tables x columns + FK mesh + term bindings)
and measures the hot path of the semantic layer on the in-process LocalEngine:

    SchemaLinker.link(question)  -> PPR over the schema graph

Numbers answer the capacity question: at what catalog size does the in-process
engine stop being acceptable, i.e. when should we switch to the Vermeer engine?

Run::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/perf_bench/bench_semantic_scale.py 2>&1 \\
        | tee _out/perf_bench/logs/bench_semantic_scale.log
"""

from __future__ import annotations

import time

from hugegraph_llm.nl2sql.engine.local import LocalEngine
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker
from hugegraph_llm.nl2sql.schema_graph.builder import SchemaGraphBuilder
from hugegraph_llm.nl2sql.schema_graph.model import Column, Table, Term


def build_schema(n_tables: int, cols_per_table: int = 8) -> SchemaGraphBuilder:
    b = SchemaGraphBuilder()
    for t in range(n_tables):
        tbl = f"tbl_{t:04d}"
        b.add_table(Table(name=tbl, comment=f"{tbl} comment"))
        for c in range(cols_per_table):
            b.add_column(Column(
                name=f"col_{c}", table=tbl, data_type="double",
                comment=f"{tbl}.col_{c}",
            ))
        # every table carries a shared *_id column so the FK mesh connects it
        b.add_column(Column(name="user_id", table=tbl, data_type="bigint",
                            comment="user id"))
    # ring FK mesh: tbl_i.user_id <-> tbl_{i+1}.user_id
    for t in range(n_tables - 1):
        b.add_foreign_key(f"tbl_{t:04d}.user_id", f"tbl_{t + 1:04d}.user_id")
    b.add_term(Term(name="支付总额", comment="支付总额（口径：pay_amount 汇总）"))
    b.bind_term("支付总额", "tbl_0000.col_1")
    return b


def main() -> None:
    print("semantic-scale | tables | columns | link(ms) | peak_ppr_nodes")
    for n in (500, 1000, 2000, 5000):
        b = build_schema(n)
        schema = b.build()
        engine = LocalEngine(schema)
        linker = SchemaLinker(schema, engine=engine)
        # warm up (jieba load, graph materialisation)
        linker.link("支付总额是多少", top_k=10)
        best = None
        for _ in range(5):
            t0 = time.monotonic()
            items = linker.link("支付总额是多少", top_k=10)
            dt = (time.monotonic() - t0) * 1000
            best = dt if best is None else min(best, dt)
        cols = len(schema.columns())
        print(f"{n:6d} | {n:6d} | {cols:7d} | {best:8.2f} | {len(items)}")


if __name__ == "__main__":
    main()
