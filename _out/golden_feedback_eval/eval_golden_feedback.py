"""Offline evaluation: how much does golden-SQL feedback improve voting?

Measures the uplift that "golden 回灌" gives the NL2SQL voting stage on a
*real* HugeGraph metadata slice:

1. seed a comment-rich order/payment/user slice into the live graph (idempotent);
2. write verified golden SQLs into the graph via ``KgGoldenSqlStore.add``
   (real ``Query`` vertices + ``references`` edges);
3. for each question, rank the same candidate set twice with ``KgSqlVoter``:
   * baseline  -- ``golden_records=[]`` (no feedback);
   * feedback  -- ``golden_records=store.get_similar(question)`` (real retrieval);
4. aggregate top-1 hit-rate / average rank / score uplift of the golden SQL.

Invariants asserted (never regress): feedback rank <= baseline rank for every
golden, and top-1 rate with feedback >= baseline. The headline number is the
top-1 flip, e.g. a case where a tied distractor wins baseline and the golden
wins only once golden overlap is fed back.

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/golden_feedback_eval/eval_golden_feedback.py 2>&1 \\
        | tee _out/golden_feedback_eval/logs/eval_golden_feedback.log

Exits non-zero on assertion failure; prints SKIP (exit 0) when the live graph
is unreachable.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SEED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nl2sql_demo"))
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
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_sql_voter import KgSqlVoter  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")
GOLDEN_DOMAIN = "eval_golden"

# Each case: the golden SQL is the ground truth; candidates are listed in a
# deliberate order (distractors first) so ties resolve against the golden
# without feedback -- the feedback must then flip it back.
CASES: List[Dict[str, Any]] = [
    {
        # tie-flip showcase: both candidates are metric-caliber-correct and the
        # question links both tables symmetrically -> baseline tie, first wins;
        # golden overlap (+8) breaks the tie toward the golden.
        "question": "订单金额与支付金额对比",
        "golden": "SELECT SUM(order.amount) FROM order",
        "candidates": [
            "SELECT SUM(payment.amount) FROM payment",
            "SELECT SUM(order.amount) FROM order",
        ],
    },
    {
        "question": "订单总额",
        "golden": "SELECT SUM(order.amount) FROM order",
        "candidates": [
            "SELECT SUM(order.amount) FROM order",
            "SELECT order.amount FROM order",
            "SELECT SUM(order.amnt) FROM order",  # invalid: wrong column
        ],
    },
    {
        "question": "订单总额是多少",
        "golden": "SELECT SUM(order.amount) FROM order",
        "candidates": [
            "SELECT AVG(order.amount) FROM order",  # invalid: caliber mismatch
            "SELECT SUM(order.amount) FROM order",
        ],
    },
    {
        "question": "支付总额",
        "golden": "SELECT SUM(payment.amount) FROM payment",
        "candidates": [
            "SELECT SUM(order.amount) FROM order",
            "SELECT SUM(payment.amount) FROM payment",
        ],
    },
    {
        "question": "各城市订单总额",
        # natural LLM-style SQL with a SELECT alias (SQL-A2 now resolves it)
        "golden": (
            "SELECT city, SUM(order.amount) AS order_amount FROM order "
            "GROUP BY city ORDER BY order_amount DESC"
        ),
        "candidates": [
            "SELECT SUM(payment.amount) FROM payment",
            (
                "SELECT city, SUM(order.amount) AS order_amount FROM order "
                "GROUP BY city ORDER BY order_amount DESC"
            ),
        ],
    },
    {
        "question": "订单与支付关联金额",
        "golden": (
            "SELECT payment.order_id, payment.amount FROM payment "
            "JOIN order ON payment.order_id = order.order_id"
        ),
        "candidates": [
            # non-joinable (payment.order_id vs user.user_id) -> SQL-J1 penalty
            "SELECT payment.order_id, payment.amount FROM payment "
            "JOIN user ON payment.order_id = user.user_id",
            (
                "SELECT payment.order_id, payment.amount FROM payment "
                "JOIN order ON payment.order_id = order.order_id"
            ),
        ],
    },
]


def _rank_of(sql: str, votes) -> Optional[int]:
    for i, v in enumerate(votes, 1):
        if v.sql == sql:
            return i
    return None


def _rank_or(sql: str, votes, default: int) -> int:
    r = _rank_of(sql, votes)
    return r if r is not None else default


def main() -> int:
    try:
        client = make_client(GRAPH)
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: cannot build client: {exc}")
        return 0
    if not reachable(client):
        print("SKIP: live HugeGraph gremlin endpoint unreachable")
        return 0

    # clean, fresh, comment-rich slice
    drop_slice(client)
    seed_slice(client)

    # write the verified golden SQLs into the real graph
    store = KgGoldenSqlStore(client, GRAPH)
    added = []
    for case in CASES:
        vid = store.add(case["question"], case["golden"], domain=GOLDEN_DOMAIN)
        added.append(vid)
        assert vid is not None, f"golden store.add failed for {case['question']}"
    print(f"seeded {len(added)} golden records into graph '{GRAPH}' (domain={GOLDEN_DOMAIN})")

    data = KgRuleEngine(client, GRAPH).load_graph()

    rows = []
    n_flip = 0
    n_boost = 0
    for case in CASES:
        q, golden = case["question"], case["golden"]
        retrieved = store.get_similar(q, top_k=3)
        retrieved_sqls = {r.sql for r in retrieved}
        golden_retrieved = golden in retrieved_sqls

        base_voter = KgSqlVoter(question=q, graph_data=data, golden_records=[])
        fb_voter = KgSqlVoter(question=q, graph_data=data, golden_records=retrieved)

        base_votes = base_voter.vote(case["candidates"])
        fb_votes = fb_voter.vote(case["candidates"])

        rank_wo = _rank_or(golden, base_votes, len(case["candidates"]) + 1)
        rank_w = _rank_or(golden, fb_votes, len(case["candidates"]) + 1)
        score_wo = next(v.score for v in base_votes if v.sql == golden)
        score_w = next(v.score for v in fb_votes if v.sql == golden)
        delta = score_w - score_wo
        golden_valid = next(v.valid for v in base_votes if v.sql == golden)
        overlap_wo = next(
            v.breakdown.get("golden_overlap", 0.0)
            for v in base_votes
            if v.sql == golden
        )
        overlap_w = next(
            v.breakdown.get("golden_overlap", 0.0) for v in fb_votes if v.sql == golden
        )

        # the golden SQL must be schema-valid, else voting can never pick it
        if not golden_valid:
            v0 = next(v for v in base_votes if v.sql == golden)
            issues = [
                f"{i.rule_id}:{i.level}:{i.message}"
                for i in (v0.report.issues if v0.report is not None else [])
            ]
            print(f"!! golden invalid for {q!r}\n    {golden}\n    issues={issues}")
        assert golden_valid, f"golden SQL invalid for {q!r}: {golden}"
        # feedback must never make the golden rank worse
        assert rank_w <= rank_wo, f"rank regression for {q!r}: {rank_wo} -> {rank_w}"
        if rank_w < rank_wo:
            n_flip += 1
        if overlap_w > overlap_wo:
            n_boost += 1

        rows.append(
            {
                "question": q,
                "golden": golden,
                "golden_retrieved": golden_retrieved,
                "rank_wo": rank_wo,
                "score_wo": score_wo,
                "rank_w": rank_w,
                "score_w": score_w,
                "delta": delta,
                "overlap_wo": overlap_wo,
                "overlap_w": overlap_w,
                "top1_wo": rank_wo == 1,
                "top1_w": rank_w == 1,
            }
        )

    # ---- summary ---------------------------------------------------------
    n = len(rows)
    top1_wo = sum(1 for r in rows if r["top1_wo"])
    top1_w = sum(1 for r in rows if r["top1_w"])
    avg_rank_wo = sum(r["rank_wo"] for r in rows) / n
    avg_rank_w = sum(r["rank_w"] for r in rows) / n
    avg_delta = sum(r["delta"] for r in rows) / n

    print("\n=== per-question (real kg_rag) ===")
    print(
        f"{'question':<18} {'top1 wo':>8} {'top1 w':>8} {'rank':>6} {'score wo':>9} "
        f"{'score w':>9} {'Δscore':>7} {'golden?':>7}"
    )
    for r in rows:
        print(
            f"{r['question']:<18} {str(r['top1_wo']):>8} {str(r['top1_w']):>8} "
            f"{str(r['rank_wo']) + '/' + str(r['rank_w']):>6} {r['score_wo']:>9.1f} "
            f"{r['score_w']:>9.1f} {r['delta']:>+7.1f} {str(r['golden_retrieved']):>7}"
        )

    print("\n=== aggregate ===")
    print(f"  top-1 without feedback : {top1_wo}/{n} ({top1_wo / n:.0%})")
    print(f"  top-1 with feedback    : {top1_w}/{n} ({top1_w / n:.0%})")
    print(f"  top-1 uplift           : {top1_w - top1_wo}/{n} "
          f"({(top1_w - top1_wo) / n:+.0%} pp)")
    print(f"  avg golden rank        : {avg_rank_wo:.2f} -> {avg_rank_w:.2f}")
    print(f"  avg golden score delta : {avg_delta:+.1f} (golden_overlap weight=4/ref)")
    print(f"  cases where feedback boosted golden_overlap : {n_boost}/{n}")
    print(f"  cases where rank improved (flip)           : {n_flip}/{n}")
    if n_flip:
        print("  flips:")
        for r in rows:
            if r["rank_w"] < r["rank_wo"]:
                print(f"    - {r['question']}: rank {r['rank_wo']} -> {r['rank_w']}")

    assert top1_w >= top1_wo, "feedback must not lower top-1 accuracy"
    print("\nPASS: golden feedback never hurts; top-1 = "
          f"{top1_wo}/{n} -> {top1_w}/{n} ({top1_w - top1_wo:+d} flip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
