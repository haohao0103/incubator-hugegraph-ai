"""Real-HugeGraph e2e for KgNL2SQLPipeline (P0 + P1 orchestration).

Runs the full KG-aware NL2SQL pipeline against a *live* HugeGraph metadata graph
(the same `kg_rag` slice the other NL2SQL P1 modules use), proving on real data
that the staged contract is produced end-to-end:

    linking -> sql_generation -> sql_validation -> sql_voting
            -> [authority] -> lineage -> [golden_feedback]

The deterministic path (candidates injected, no LLM) always runs and asserts
the winner. The live-LLM path (``pipe.run()`` with glm-5.3 generation) is
OPT-IN via ``KG_E2E_LIVE_LLM=1`` -- it is guarded and never fails the run, so a
flaky/rate-limited endpoint degrades gracefully to an empty answer.

Run::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/nl2sql_pipeline_e2e/e2e_nl2sql_pipeline.py

    # also exercise the live LLM generation (may be slow / rate-limited):
    KG_E2E_LIVE_LLM=1 \\
        /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/nl2sql_pipeline_e2e/e2e_nl2sql_pipeline.py

Exits non-zero on deterministic assertion failure; prints SKIP (exit 0) when
the live graph is unreachable.
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
from hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline import (  # noqa: E402
    KgNL2SQLPipeline,
)
from pyhugegraph.client import PyHugeClient  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")
LIVE_LLM = os.environ.get("KG_E2E_LIVE_LLM") == "1"

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


def _assert_stages(stages, expect_authority: bool = False) -> None:
    names = [s.stage for s in stages]
    required = [
        "linking",
        "sql_generation",
        "sql_validation",
        "sql_voting",
        "lineage",
    ]
    for r in required:
        assert r in names, f"missing stage {r!r} in {names}"
    if expect_authority:
        assert "authority" in names, f"expected authority stage in {names}"


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

    # ---- deterministic path: injected candidates, no LLM ------------------
    pipe = KgNL2SQLPipeline(question="各城市订单总额", client=client)
    candidates = [
        "SELECT payment.amount FROM payment",          # valid, unrelated
        "SELECT SUM(order.amount) FROM order",          # metric-aware, valid
        "SELECT SUM(order.amnt) FROM order",            # invalid (wrong column)
    ]
    resp = pipe.run(candidates=candidates)

    # contract checks
    assert resp.route == "nl2sql", resp.route
    assert resp.answer == "SELECT SUM(order.amount) FROM order", resp.answer
    _assert_stages(resp.stages)

    # the chosen SQL must sit at the top of the voting ranking
    voting = next(s for s in resp.stages if s.stage == "sql_voting")
    assert voting.output["chosen"] == "SELECT SUM(order.amount) FROM order"
    assert voting.output["ranked"][0]["sql"] == "SELECT SUM(order.amount) FROM order"

    # lineage must explain a real (seeded) entity
    lineage = next(s for s in resp.stages if s.stage == "lineage")
    assert "order_total" in lineage.output["explain"] or "order" in lineage.output["explain"]

    print("PASS: KgNL2SQLPipeline end-to-end (live HugeGraph %s)" % GRAPH)
    print("  route   :", resp.route)
    print("  answer  :", resp.answer)
    print("  stages  :", " -> ".join(s.stage for s in resp.stages))
    print("  ranking :")
    for v in resp.raw["votes"]:
        print(f"    {v['score']:7.1f}  valid={v['valid']}  {v['sql']}")

    # ---- optional live-LLM path (guarded, never fails the run) -----------
    if LIVE_LLM:
        print("\n--- live-LLM generation (glm-5.3) ---")
        try:
            live = pipe.run()  # no candidates -> real LLM generation
            print("  live answer:", live.answer or "(empty -- endpoint degraded gracefully)")
            if live.stages:
                gen = next(
                    (s for s in live.stages if s.stage == "sql_generation"), None
                )
                if gen:
                    print("  generation source:", gen.output.get("source"))
        except Exception as exc:  # pragma: no cover - defensive
            print("  live LLM path skipped/unavailable:", exc)
    else:
        print("\n(tip: set KG_E2E_LIVE_LLM=1 to also exercise glm-5.3 generation)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
