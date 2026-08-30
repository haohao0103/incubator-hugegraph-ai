"""Scale benchmark: KG linking / SQL validation / multi-candidate voting.

Synthesizes in-memory ``GraphData`` of growing size (tables x columns) and
measures the three hot deterministic steps of the NL2SQL pipeline:

* ``KgSqlValidator`` construction (index build over all vertices);
* ``KgSchemaLinker.link`` (question -> relevant subgraph);
* ``KgSqlVoter.vote`` (3 candidates, includes its own validator build).

No live graph is touched -- everything runs on synthetic dicts, so the
numbers reflect algorithmic cost at metadata-KG scale (which a real
warehouse catalog with tens of thousands of tables would hit).

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/perf_bench/bench_kg_scale.py 2>&1 \\
        | tee _out/perf_bench/logs/bench_kg_scale.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_sql_validator import KgSqlValidator  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_sql_voter import KgSqlVoter  # noqa: E402

SCALES = (500, 1000, 2000, 5000)
COLS_PER_TABLE = 6
REPS = 3


def synthetic_graph(n_tables: int) -> Dict[str, Any]:
    """n_tables tables x COLS_PER_TABLE fields + sparse metrics/edges."""
    tables: List[Dict[str, Any]] = []
    fields: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []
    has_column: List[tuple] = []
    computed_from: List[tuple] = []
    computed_field: List[tuple] = []

    for i in range(n_tables):
        # every 100th table carries an 'order'-ish name so the question
        # "order amount" actually links to something at every scale
        name = f"order_{i:05d}" if i % 100 == 0 else f"t_{i:05d}"
        tables.append({"name": name, "comment": f"表 {i}"})
        for c in range(COLS_PER_TABLE):
            fname = f"{name}.col{c}"
            fields.append({"name": fname, "comment": f"列 {c}"})
            has_column.append((name, fname))
        if i % 100 == 0:
            metrics.append({
                "name": f"m_{name}",
                "formula": f"SUM({name}.col0)",
                "definition": f"{name} 合计",
            })
            computed_from.append((f"m_{name}", name))
            computed_field.append((f"m_{name}", f"{name}.col0"))
    return {
        "vertices": {"Table": tables, "Field": fields, "Metric": metrics},
        "edges": {
            "hasColumn": has_column,
            "computedFrom": computed_from,
            "computedFromField": computed_field,
            "dependsOn": [],
        },
    }


def median_ms(fn, reps: int = REPS) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    print("scale | vertices | link(ms) | build(ms) | validate(ms) | vote-shared(ms) | vote-standalone(ms)")
    print("------|----------|----------|----------|--------------|----------------|-------------------")
    baseline = None
    for scale in SCALES:
        data = synthetic_graph(scale)
        n_vertices = sum(len(v) for v in data["vertices"].values())

        link_ms = median_ms(lambda: KgSchemaLinker().link("order amount", data=data))
        build_ms = median_ms(lambda: KgSqlValidator(data))
        validator = KgSqlValidator(data)
        sql = "SELECT SUM(order_00000.col0) FROM order_00000"
        validate_ms = median_ms(lambda: validator.validate(sql))

        def _vote(shared: bool):
            def run():
                voter = KgSqlVoter(
                    question="order amount", graph_data=data,
                    golden_records=[],
                    validator=validator if shared else None,
                )
                voter.vote([
                    "SELECT SUM(order_00000.col0) FROM order_00000",
                    "SELECT order_00000.col1 FROM order_00000",
                    "SELECT SUM(order_00000.col9) FROM order_00000",
                ])
            return run
        vote_shared_ms = median_ms(_vote(True))
        vote_standalone_ms = median_ms(_vote(False))

        growth = ""
        if baseline is not None:
            growth = f"  (vote-shared {vote_shared_ms / baseline:+.1f}x vs 500)"
        baseline = vote_shared_ms if baseline is None else baseline
        print(f"{scale:>5} | {n_vertices:>8} | {link_ms:>8.1f} | {build_ms:>8.1f} "
              f"| {validate_ms:>10.2f} | {vote_shared_ms:>14.1f} | {vote_standalone_ms:>17.1f}{growth}")

    print("\nnotes:")
    print("  - link uses heap top-k truncation (max_tables/max_metrics/max_fields budgets)")
    print("  - build = KgSqlValidator index build (one-off per graph snapshot)")
    print("  - vote-shared reuses that validator (pipeline pattern);")
    print("    vote-standalone is the old per-voter rebuild")
    return 0


if __name__ == "__main__":
    sys.exit(main())
