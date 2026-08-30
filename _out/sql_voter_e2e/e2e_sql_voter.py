"""Real-HugeGraph e2e for KgSqlVoter (P1-4).

Runs the deterministic multi-candidate SQL voter against a *live* HugeGraph
metadata graph (the same `kg_rag` slice the other NL2SQL P1 modules use),
proving on real data that:

1. a metric-aware, schema-valid candidate beats an unrelated-but-valid one;
2. an invalid candidate (wrong column) is ranked last;
3. golden-SQL overlap nudges a candidate that matches verified SQL upward.

No LLM is involved -- this is pure graph traversal + scoring, so it is safe to
run on every candidate set.

Run::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/sql_voter_e2e/e2e_sql_voter.py

Exits non-zero on assertion failure; prints SKIP (exit 0) when the live graph
is unreachable.
"""

from __future__ import annotations

import logging
import os
import sys

# make the hugegraph-ai package importable when run from repo root
_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# silence third-party INFO noise (config modules re-set their own levels)
logging.disable(logging.CRITICAL)

from hugegraph_llm.config import huge_settings  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_sql_voter import KgSqlVoter  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_golden_sql import GoldenRecord  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine  # noqa: E402
from pyhugegraph.client import PyHugeClient  # noqa: E402


GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")

# idempotent seed: a tiny order/payment slice with one metric
# (label, name, {extra properties})
_SEED_VERTICES = [
    ("Table", "order", {}),
    ("Table", "payment", {}),
    ("Field", "order.amount", {}),
    ("Field", "payment.amount", {}),
    ("Metric", "order_total", {"formula": "SUM(order.amount)"}),
]
# (edge_label, src_label, src, dst_label, dst)
_SEED_EDGES = [
    ("hasColumn", "Table", "order", "Field", "order.amount"),
    ("hasColumn", "Table", "payment", "Field", "payment.amount"),
    ("computedFrom", "Metric", "order_total", "Table", "order"),
    ("computedFromField", "Metric", "order_total", "Field", "order.amount"),
]


def _client():
    return PyHugeClient(
        url=huge_settings.graph_url,
        graph=GRAPH,
        user=huge_settings.graph_user,
        pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )


def _reachable(client) -> bool:
    try:
        client.gremlin().exec("g.V().limit(1).count()")
        return True
    except Exception:
        return False


def _drop_vertices(client):
    for label, name, _ in _SEED_VERTICES:
        try:
            client.gremlin().exec(
                f"g.V().has('{label}','name','{name}').drop()"
            )
        except Exception:
            pass


def _seed(client):
    for label, name, props in _SEED_VERTICES:
        props_str = "".join(f".property('{k}','{v}')" for k, v in props.items())
        client.gremlin().exec(
            f"g.addV('{label}').property('name','{name}'){props_str}"
        )
    for label, s_label, src, d_label, dst in _SEED_EDGES:
        client.gremlin().exec(
            f"g.V().has('{s_label}','name','{src}').as('s')"
            f".V().has('{d_label}','name','{dst}')"
            f".addE('{label}').from('s')"
        )


def main() -> int:
    try:
        client = _client()
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: cannot build client: {exc}")
        return 0

    if not _reachable(client):
        print("SKIP: live HugeGraph gremlin endpoint unreachable")
        return 0

    # idempotent: drop the slice, then re-create it fresh
    _drop_vertices(client)
    _seed(client)

    try:
        data = KgRuleEngine(client, GRAPH).load_graph()
        voter = KgSqlVoter(
            question="订单金额 order amount",
            graph_data=data,
            golden_records=[
                GoldenRecord(
                    question="订单总额",
                    sql="SELECT SUM(order.amount) FROM order",
                    schema_refs={"order", "order.amount"},
                ),
            ],
        )

        candidates = [
            "SELECT payment.amount FROM payment",    # valid, unrelated
            "SELECT SUM(order.amount) FROM order",    # metric-aware + golden match
            "SELECT SUM(order.amnt) FROM order",      # invalid (wrong column)
        ]
        ranked = voter.vote(candidates)

        assert ranked[0].sql == "SELECT SUM(order.amount) FROM order", ranked[0].sql
        assert ranked[0].valid is True
        assert ranked[-1].valid is False
        assert ranked[0].breakdown["caliber"] > 0
        assert ranked[0].breakdown["golden_overlap"] > 0

        print("PASS: KgSqlVoter picks the metric-aware, golden-matched SQL")
        print("  ranking (live HugeGraph %s):" % GRAPH)
        for v in ranked:
            print(f"    {v.score:7.1f}  valid={v.valid}  {v.sql}")
        return 0
    finally:
        # leave the slice in place (harmless, reusable by other P1 e2e)
        pass


if __name__ == "__main__":
    sys.exit(main())
