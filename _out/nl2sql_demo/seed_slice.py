"""Shared idempotent seed for the NL2SQL demo / golden-feedback eval slices.

Seeds a small but *comment-rich* order/payment/user metadata slice into the live
``kg-rag`` graph (same label/name conventions as the other P1 e2e scripts):

- tables with Chinese ``comment`` (so Chinese questions link via the linker);
- fields with Chinese ``comment``;
- two metrics ``order_total`` / ``payment_total`` with canonical formulas and
  Chinese ``definition`` (caliber check needs these);
- ``hasColumn`` / ``computedFrom`` / ``computedFromField`` edges.

The seed is idempotent: it drops the slice vertices (plus any ``Query`` golden
vertices in the demo/eval domains) before re-creating them fresh, so every run
starts from a clean, deterministic state.

Run from the repo root::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/nl2sql_demo/seed_slice.py [--graph kg_rag]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.config import huge_settings  # noqa: E402
from pyhugegraph.client import PyHugeClient  # noqa: E402

# (label, name, {extra properties})
SEED_VERTICES = [
    ("Table", "order", {"comment": "订单表"}),
    ("Table", "payment", {"comment": "支付表"}),
    ("Table", "user", {"comment": "用户表"}),
    ("Field", "order.order_id", {"comment": "订单号"}),
    ("Field", "order.amount", {"comment": "订单金额"}),
    ("Field", "order.city", {"comment": "城市"}),
    ("Field", "payment.pay_id", {"comment": "支付流水号"}),
    ("Field", "payment.order_id", {"comment": "订单号"}),
    ("Field", "payment.amount", {"comment": "支付金额"}),
    ("Field", "user.user_id", {"comment": "用户ID"}),
    ("Metric", "order_total", {"definition": "订单总额", "formula": "SUM(order.amount)"}),
    ("Metric", "payment_total", {"definition": "支付总额", "formula": "SUM(payment.amount)"}),
]
# (edge_label, src_label, src, dst_label, dst)
SEED_EDGES = [
    ("hasColumn", "Table", "order", "Field", "order.order_id"),
    ("hasColumn", "Table", "order", "Field", "order.amount"),
    ("hasColumn", "Table", "order", "Field", "order.city"),
    ("hasColumn", "Table", "payment", "Field", "payment.pay_id"),
    ("hasColumn", "Table", "payment", "Field", "payment.order_id"),
    ("hasColumn", "Table", "payment", "Field", "payment.amount"),
    ("hasColumn", "Table", "user", "Field", "user.user_id"),
    ("computedFrom", "Metric", "order_total", "Table", "order"),
    ("computedFrom", "Metric", "payment_total", "Table", "payment"),
    ("computedFromField", "Metric", "order_total", "Field", "order.amount"),
    ("computedFromField", "Metric", "payment_total", "Field", "payment.amount"),
]

GOLDEN_DOMAINS = ("demo_golden", "eval_golden")


def make_client(graph: str):
    return PyHugeClient(
        url=huge_settings.graph_url,
        graph=graph,
        user=huge_settings.graph_user,
        pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )


def reachable(client) -> bool:
    try:
        client.gremlin().exec("g.V().limit(1).count()")
        return True
    except Exception:
        return False


def drop_slice(client) -> None:
    """Drop the seed vertices + golden Query vertices of the known domains."""
    for label, name, _ in SEED_VERTICES:
        try:
            client.gremlin().exec(f"g.V().has('{label}','name','{name}').drop()")
        except Exception:
            pass
    for domain in GOLDEN_DOMAINS:
        try:
            client.gremlin().exec(
                f"g.V().hasLabel('Query').has('domain','{domain}').drop()"
            )
        except Exception:
            pass


def seed_slice(client) -> None:
    """Create the seed vertices + edges fresh (assumes drop_slice already ran)."""
    for label, name, props in SEED_VERTICES:
        props_str = "".join(f".property('{k}','{v}')" for k, v in props.items())
        client.gremlin().exec(
            f"g.addV('{label}').property('name','{name}'){props_str}"
        )
    for label, s_label, src, d_label, dst in SEED_EDGES:
        client.gremlin().exec(
            f"g.V().has('{s_label}','name','{src}').as('s')"
            f".V().has('{d_label}','name','{dst}')"
            f".addE('{label}').from('s')"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default=os.environ.get("KG_E2E_GRAPH", "kg_rag"))
    args = parser.parse_args()

    client = make_client(args.graph)
    if not reachable(client):
        print(f"SKIP: live HugeGraph gremlin endpoint unreachable ({args.graph})")
        return 0
    drop_slice(client)
    seed_slice(client)
    print(f"PASS: seeded {len(SEED_VERTICES)} vertices / {len(SEED_EDGES)} edges "
          f"into graph '{args.graph}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
