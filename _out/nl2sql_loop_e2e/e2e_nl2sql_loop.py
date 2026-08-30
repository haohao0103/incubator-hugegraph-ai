"""Real-HugeGraph e2e for the full loop: 问 -> SQL -> 执行 -> 答案.

Combines the real KG-NL2SQL pipeline (live kg_rag metadata graph, real
generation/voting/validation) with the DuckDB executor (sample row-level
order/payment/user tables) and the runner, proving the end-to-end loop on a
live graph:

    question -> linking -> generation -> validation -> voting
              -> execute winning SQL (DuckDB) -> answer with data

Deterministic path (injected candidates) always runs and asserts the executed
rows; the live-LLM path is OPT-IN via KG_E2E_LIVE_LLM=1.

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/nl2sql_loop_e2e/e2e_nl2sql_loop.py 2>&1 \\
        | tee _out/nl2sql_loop_e2e/logs/e2e_nl2sql_loop.log
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline import (  # noqa: E402
    KgNL2SQLPipeline,
)
from hugegraph_llm.operators.sql_exec.nl2sql_runner import (  # noqa: E402
    KgNL2SQLRunner,
)
from hugegraph_llm.operators.sql_exec.sql_executor import DuckDbExecutor  # noqa: E402
from hugegraph_llm.utils.hugegraph_utils import get_hg_client  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")
LIVE_LLM = os.environ.get("KG_E2E_LIVE_LLM") == "1"


def main() -> int:
    try:
        client = get_hg_client()
        client.gremlin().exec("g.V().limit(1).count()")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: graph unreachable: {exc}")
        return 0

    pipe = KgNL2SQLPipeline(question="各城市订单总额", client=client)
    runner = KgNL2SQLRunner(pipe, DuckDbExecutor())

    # ---- deterministic: injected candidates, assert executed rows ---------
    out = runner.run(candidates=[
        "SELECT SUM(payment.amount) FROM payment",
        "SELECT city, SUM(order.amount) FROM order GROUP BY city ORDER BY SUM(order.amount) DESC",
        "SELECT SUM(order.amnt) FROM order",  # invalid
    ])

    assert out.sql.startswith("SELECT city, SUM(order.amount) FROM order"), out.sql
    assert out.execution.ok, out.execution.error
    assert out.execution.row_count == 4, out.execution.row_count  # 北京/上海/深圳/广州
    assert "查询返回 4 行" in out.answer, out.answer
    assert "北京" in out.answer or "上海" in out.answer

    print("PASS: 问 -> SQL -> 执行 -> 答案 (deterministic, live kg_rag)")
    print(f"  question : 各城市订单总额")
    print(f"  sql      : {out.sql}")
    print(f"  valid    : {out.valid}")
    print(f"  columns  : {out.execution.columns}")
    print(f"  rows     : {out.execution.rows}")
    print(f"  row_count: {out.execution.row_count} ({out.execution.duration_ms:.1f} ms)")
    print(f"  answer   : {out.answer}")
    stages = [getattr(s, "stage", None) for s in out.stages]
    print(f"  stages   : {' -> '.join(str(x) for x in stages)}")

    if LIVE_LLM:
        print("\n--- live-LLM loop ---")
        try:
            live = runner.run()
            print(f"  sql   : {live.sql}")
            print(f"  answer: {live.answer}")
        except Exception as exc:  # pragma: no cover
            print(f"  live loop degraded: {exc}")
    else:
        print("\n(tip: set KG_E2E_LIVE_LLM=1 to also run the real glm-5.3 loop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
