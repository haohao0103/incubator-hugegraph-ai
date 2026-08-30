"""Tests for KgSqlVoter (deterministic multi-candidate SQL voting, P1-4)."""

import unittest
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.operators.graph_op.kg_sql_voter import (
    KgSqlVoter,
    SqlVote,
)
from hugegraph_llm.operators.graph_op.kg_golden_sql import GoldenRecord


def _sample_graph() -> Dict[str, Any]:
    """order / payment (joinable) / user (isolated) tables + one metric."""
    return {
        "vertices": {
            "Table": [
                {"name": "order"},
                {"name": "payment"},
                {"name": "user"},
            ],
            "Field": [
                {"name": "order.amount"},
                {"name": "order.city"},
                {"name": "order.order_id"},
                {"name": "payment.amount"},
                {"name": "payment.order_id"},
                {"name": "user.user_id"},
            ],
            "Metric": [
                {"name": "order_total", "formula": "SUM(order.amount)",
                 "definition": "订单总额"},
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.amount"),
                ("order", "order.city"),
                ("order", "order.order_id"),
                ("payment", "payment.amount"),
                ("payment", "payment.order_id"),
                ("user", "user.user_id"),
            ],
            "computedFrom": [("order_total", "order")],
            "computedFromField": [("order_total", "order.amount")],
            "dependsOn": [],
        },
    }


_QUESTION = "订单金额 order amount"


def _voter(
    question: Optional[str] = _QUESTION,
    golden: Optional[List[GoldenRecord]] = None,
) -> KgSqlVoter:
    return KgSqlVoter(
        question=question,
        graph_data=_sample_graph(),
        golden_records=golden,
    )


class TestConstructor(unittest.TestCase):
    def test_requires_graph(self):
        with self.assertRaises(ValueError):
            KgSqlVoter()
        with self.assertRaises(ValueError):
            KgSqlVoter(question=_QUESTION)

    def test_loads_via_client(self):
        class _FakeClient:
            def gremlin(self):
                return self

            def exec(self, q):
                if "hasLabel('Table')" in q:
                    return {"data": [{"id": "1:" + t, "label": "Table", "name": t}
                                     for t in ("order", "payment", "user")]}
                if "hasLabel('Field')" in q:
                    fs = ("order.amount", "order.city", "order.order_id",
                          "payment.amount", "payment.order_id", "user.user_id")
                    return {"data": [{"id": "1:" + f, "label": "Field", "name": f}
                                     for f in fs]}
                if "hasLabel('Metric')" in q:
                    return {"data": [{"id": "1:order_total", "label": "Metric",
                                     "name": "order_total",
                                     "formula": "SUM(order.amount)",
                                     "definition": "订单总额"}]}
                if "hasLabel('hasColumn')" in q:
                    es = [("order", "order.amount"), ("order", "order.city"),
                          ("order", "order.order_id"), ("payment", "payment.amount"),
                          ("payment", "payment.order_id"), ("user", "user.user_id")]
                    return {"data": [{"id": "e", "label": "hasColumn",
                                      "OUT": {"id": "1:" + s}, "IN": {"id": "1:" + d}}
                                     for s, d in es]}
                if "hasLabel('computedFrom')" in q:
                    return {"data": [{"id": "e", "label": "computedFrom",
                                      "OUT": {"id": "1:order_total"},
                                      "IN": {"id": "1:order"}}]}
                if "hasLabel('computedFromField')" in q:
                    return {"data": [{"id": "e", "label": "computedFromField",
                                      "OUT": {"id": "1:order_total"},
                                      "IN": {"id": "1:order.amount"}}]}
                return {"data": []}

        voter = KgSqlVoter(question=_QUESTION, client=_FakeClient())
        # the question links to 'order', so a matching candidate should win
        best = voter.best([
            "SELECT SUM(order.amount) FROM order",
            "SELECT payment.amount FROM payment",
        ])
        self.assertEqual(best, "SELECT SUM(order.amount) FROM order")


class TestVoteRanking(unittest.TestCase):
    def setUp(self):
        self.voter = _voter()

    def test_empty_candidates(self):
        self.assertEqual(self.voter.vote([]), [])
        self.assertIsNone(self.voter.best([]))
        self.assertIsNone(self.voter.best_vote([]))

    def test_picks_metric_aware_over_unrelated(self):
        candidates = [
            "SELECT payment.amount FROM payment",   # valid, no caliber, low overlap
            "SELECT SUM(order.amount) FROM order",   # valid + 口径 + high overlap
            "SELECT SUM(order.amnt) FROM order",     # invalid (wrong column)
        ]
        ranked = self.voter.vote(candidates)
        self.assertEqual(ranked[0].sql, "SELECT SUM(order.amount) FROM order")
        self.assertTrue(ranked[0].valid)
        self.assertFalse(ranked[-1].valid)
        # caliber component is earned by the metric-aware candidate
        self.assertGreater(ranked[0].breakdown["caliber"], 0)
        # golden component is zero (no golden pool supplied)
        self.assertEqual(ranked[0].breakdown["golden_overlap"], 0)

    def test_single_candidate_returned(self):
        ranked = self.voter.vote(["SELECT order.city FROM order"])
        self.assertEqual(len(ranked), 1)
        self.assertTrue(ranked[0].valid)

    def test_tie_is_stable(self):
        a = "SELECT order.city FROM order"
        ranked = self.voter.vote([a, a])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].sql, a)
        self.assertEqual(ranked[1].sql, a)


class TestVoteBudget(unittest.TestCase):
    def test_vote_truncates_to_max_candidates(self):
        v = KgSqlVoter(
            question="订单金额 order amount", graph_data=_sample_graph(),
            max_candidates=5,
        )
        ranked = v.vote([f"SELECT {i} FROM order" for i in range(12)])
        self.assertEqual(len(ranked), 5)

    def test_vote_max_candidates_floor_one(self):
        v = KgSqlVoter(
            question="x", graph_data=_sample_graph(), max_candidates=0
        )
        self.assertEqual(v._max_candidates, 1)

    def test_default_max_candidates_ten(self):
        v = KgSqlVoter(question="x", graph_data=_sample_graph())
        self.assertEqual(v._max_candidates, 10)

    def test_shared_validator_reused(self):
        from hugegraph_llm.operators.graph_op.kg_sql_validator import KgSqlValidator

        validator = KgSqlValidator(_sample_graph())
        v = KgSqlVoter(
            question="订单金额", graph_data=_sample_graph(), validator=validator
        )
        self.assertIs(v._validator, validator)  # no second index build
        ranked = v.vote(["SELECT SUM(order.amount) FROM order"])
        self.assertEqual(len(ranked), 1)
        self.assertTrue(ranked[0].valid)


class TestScoringComponents(unittest.TestCase):
    def setUp(self):
        self.voter = _voter()

    def test_join_penalty_for_unconnected_tables(self):
        bad_join = "SELECT * FROM order JOIN user ON order.order_id = user.user_id"
        good_join = "SELECT * FROM order JOIN payment ON order.order_id = payment.order_id"
        rb = self.voter.best_vote([bad_join]).breakdown
        rg = self.voter.best_vote([good_join]).breakdown
        self.assertLess(rb["join"], 0)        # non-joinable -> penalty
        self.assertGreater(rg["join"], 0)     # joinable -> bonus

    def test_warning_penalty(self):
        # ambiguous bare column 'amount' belongs to two tables -> SQL-A2 warning
        vote = self.voter.best_vote(["SELECT amount FROM order"])
        self.assertLess(vote.breakdown["warnings"], 0)
        self.assertTrue(vote.valid)  # warning, not error

    def test_zero_warning_path(self):
        vote = self.voter.best_vote(["SELECT order.city FROM order"])
        self.assertEqual(vote.breakdown["warnings"], 0)
        self.assertEqual(vote.issue_count, 0)

    def test_caliber_mismatch_invalid(self):
        # AVG instead of SUM -> SQL-B1 error -> invalid, no caliber credit
        vote = self.voter.best_vote(["SELECT AVG(order.amount) FROM order"])
        self.assertFalse(vote.valid)
        self.assertEqual(vote.breakdown["caliber"], 0)


class TestSchemaOverlap(unittest.TestCase):
    def test_overlap_with_linked_question(self):
        v = _voter(question=_QUESTION)
        vote = v.best_vote(["SELECT SUM(order.amount) FROM order"])
        self.assertGreater(vote.breakdown["schema_overlap"], 0)

    def test_no_question_no_overlap(self):
        v = _voter(question=None)  # linked_names empty
        vote = v.best_vote(["SELECT SUM(order.amount) FROM order"])
        self.assertEqual(vote.breakdown["schema_overlap"], 0)


class TestGoldenOverlap(unittest.TestCase):
    def test_exact_and_suffix_match(self):
        golden = [
            GoldenRecord(
                question="订单总额",
                sql="SELECT SUM(order.amount) FROM order",
                schema_refs={"order", "order.amount"},
            ),
        ]
        v = _voter(question=_QUESTION, golden=golden)
        # exact-match candidate
        vote = v.best_vote(["SELECT SUM(order.amount) FROM order"])
        self.assertGreater(vote.breakdown["golden_overlap"], 0)
        # bare 'amount' should still overlap 'order.amount' by suffix
        vote2 = v.best_vote(["SELECT amount FROM order"])
        self.assertGreater(vote2.breakdown["golden_overlap"], 0)

    def test_no_golden_no_overlap(self):
        v = _voter(question=_QUESTION)
        vote = v.best_vote(["SELECT SUM(order.amount) FROM order"])
        self.assertEqual(vote.breakdown["golden_overlap"], 0)

    def test_empty_refs_no_crash(self):
        golden = [GoldenRecord(question="q", sql="x", schema_refs=set())]
        v = _voter(question=_QUESTION, golden=golden)
        vote = v.best_vote(["SELECT 1"])  # no schema refs at all
        self.assertEqual(vote.breakdown["golden_overlap"], 0)


class TestBestSelectors(unittest.TestCase):
    def setUp(self):
        self.voter = _voter()

    def test_best_returns_sql(self):
        self.assertEqual(
            self.voter.best(["SELECT payment.amount FROM payment",
                             "SELECT SUM(order.amount) FROM order"]),
            "SELECT SUM(order.amount) FROM order",
        )

    def test_best_vote_returns_vote(self):
        bv = self.voter.best_vote(["SELECT SUM(order.amount) FROM order"])
        self.assertIsInstance(bv, SqlVote)
        self.assertTrue(bv.valid)


class TestSqlVoteModel(unittest.TestCase):
    def test_to_dict_and_explain(self):
        v = _voter().best_vote(["SELECT SUM(order.amount) FROM order"])
        d = v.to_dict()
        self.assertIn("score", d)
        self.assertIn("validation", d)
        self.assertIn("breakdown", d)
        text = v.explain()
        self.assertIn("score=", text)
        self.assertIn("valid=", text)
        self.assertIn("validity=", text)


class TestLinkedNamesDefensive(unittest.TestCase):
    """Cover the `if name:` guards when linking returns a nameless vertex."""

    def test_nameless_vertices_skipped(self):
        from unittest.mock import patch

        class _Ctx:
            tables = [{"name": "order"}, {}]
            fields = [{"name": "order.amount"}, {}]
            metrics = [{"name": "order_total"}, {}]

        class _FakeLinker:
            def __init__(self, *a, **k):
                pass

            def link(self, q, data=None):
                return _Ctx()

        with patch(
            "hugegraph_llm.operators.graph_op.kg_sql_voter.KgSchemaLinker",
            _FakeLinker,
        ):
            v = KgSqlVoter(question="q", graph_data=_sample_graph())
            # nameless vertices are skipped without crashing
            self.assertIn("order", v._linked_names)
            self.assertTrue(all(isinstance(n, str) for n in v._linked_names))


if __name__ == "__main__":
    unittest.main()
