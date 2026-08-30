"""One-shot demo seeder: comment-rich slice + a few golden SQLs.

Prepares the live ``kg-rag`` graph for the Gradio "0. Unified Import & Query"
tab's ``mode="nl2sql"`` demo:

1. seeds the comment-rich order/payment/user slice (via ``seed_slice``);
2. adds a handful of verified golden SQLs (domain ``demo_golden``) so the
   golden-feedback loop already has records to retrieve on the first query.

Run from the repo root::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/nl2sql_demo/seed_demo_slice.py 2>&1 \\
        | tee _out/nl2sql_demo/logs/seed_demo.log
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
_SEED_DIR = os.path.abspath(os.path.dirname(__file__))
if _SEED_DIR not in sys.path:
    sys.path.insert(0, _SEED_DIR)

logging.disable(logging.CRITICAL)

from seed_slice import (  # noqa: E402
    drop_slice,
    make_client,
    reachable,
    seed_slice,
)
from hugegraph_llm.operators.graph_op.kg_golden_sql import (  # noqa: E402
    KgGoldenSqlStore,
)

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")
DEMO_DOMAIN = "demo_golden"

# (question, golden_sql) -- same style the demo examples use
DEMO_GOLDEN = [
    # NOTE: SQL-A2 flags SELECT/ORDER BY aliases as unknown columns, so keep
    # the group-by golden alias-free.
    ("各城市订单总额",
     "SELECT city, SUM(order.amount) FROM order "
     "GROUP BY city ORDER BY SUM(order.amount) DESC"),
    ("订单总额",
     "SELECT SUM(order.amount) FROM order"),
    ("订单金额与支付金额对比",
     "SELECT SUM(order.amount) FROM order"),
    ("支付总额",
     "SELECT SUM(payment.amount) FROM payment"),
]


def main() -> int:
    client = make_client(GRAPH)
    if not reachable(client):
        print(f"SKIP: live HugeGraph gremlin endpoint unreachable ({GRAPH})")
        return 0

    drop_slice(client)
    seed_slice(client)

    store = KgGoldenSqlStore(client, GRAPH)
    n = 0
    for question, sql in DEMO_GOLDEN:
        vid = store.add(question, sql, domain=DEMO_DOMAIN)
        if vid is not None:
            n += 1
    print(f"PASS: demo slice seeded; {n}/{len(DEMO_GOLDEN)} golden records added "
          f"(graph={GRAPH}, domain={DEMO_DOMAIN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
