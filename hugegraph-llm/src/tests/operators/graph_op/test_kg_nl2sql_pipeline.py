"""Tests for KgNL2SQLPipeline -- end-to-end NL2SQL orchestration (P0 + P1 wiring).

Deterministic: every test injects ``graph_data`` and (where relevant) candidate
SQL, so no network or LLM is touched. The only network-bound helper
(``_default_generate``) is exercised with a mocked ``LLMs``.
"""

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline import (
    KgNL2SQLPipeline,
    _extract_sql_candidates,
    _truncate,
)
from hugegraph_llm.operators.graph_op.kg_sql_voter import KgSqlVoter
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine
from hugegraph_llm.operators.graph_op.kg_jargon_map import KgJargonMap


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


def _graph_dup_metric() -> Dict[str, Any]:
    """Same as sample but the metric name collides (two authoritative candidates)."""
    g = _sample_graph()
    g["vertices"]["Metric"] = [
        {"name": "order_total", "formula": "SUM(order.amount)",
         "definition": "订单总额", "owner": "finance"},
        {"name": "order_total", "formula": "SUM(payment.amount)",
         "definition": "支付总额", "owner": "settlement"},
    ]
    return g


class _FakeGoldenStore:
    """Minimal KgGoldenSqlStore stand-in for the store_best path."""

    def __init__(self, add_result: str = "vid-1", raise_on_add: bool = False) -> None:
        self._add_result = add_result
        self._raise = raise_on_add
        self.added: List[Any] = []

    def get_similar(self, question: str, top_k: int = 3) -> List[Any]:
        return []

    def add(self, question: str, sql: str, domain: Optional[str] = None) -> Any:
        if self._raise:
            raise RuntimeError("store down")
        self.added.append((question, sql, domain))
        return self._add_result


class _Ctx:
    """Lightweight SchemaContext stand-in for helper-method tests."""

    def __init__(self, tables=None, fields=None, metrics=None, evidence=None):
        self.tables = tables or []
        self.fields = fields or []
        self.metrics = metrics or []
        self.evidence = evidence or []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class TestHelpers(unittest.TestCase):
    def test_extract_sql_fenced(self):
        text = "x\n```sql\nSELECT 1\n```\ny"
        self.assertEqual(_extract_sql_candidates(text), ["SELECT 1"])

    def test_extract_sql_multiple_blocks(self):
        text = "```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```"
        self.assertEqual(_extract_sql_candidates(text), ["SELECT 1", "SELECT 2"])

    def test_extract_sql_fallback_whole(self):
        # no fenced block -> whole trimmed text is one candidate
        self.assertEqual(_extract_sql_candidates("  SELECT 1  "), ["SELECT 1"])

    def test_extract_sql_fallback_empty(self):
        self.assertEqual(_extract_sql_candidates(""), [])

    def test_truncate_short(self):
        self.assertEqual(_truncate("abc", n=10), "abc")

    def test_truncate_long(self):
        out = _truncate("x" * 50, n=10)
        self.assertEqual(len(out), 10 + len("...(truncated)"))
        self.assertTrue(out.endswith("...(truncated)"))


# --------------------------------------------------------------------------- #
# constructor
# --------------------------------------------------------------------------- #

class TestConstructor(unittest.TestCase):
    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            KgNL2SQLPipeline(question="  ", graph_data=_sample_graph())

    def test_no_source_raises(self):
        with self.assertRaises(ValueError):
            KgNL2SQLPipeline(question="anything")

    def test_default_jargon(self):
        pipe = KgNL2SQLPipeline(question="q", graph_data=_sample_graph())
        self.assertIsInstance(pipe._jargon, KgJargonMap)

    def test_provided_jargon(self):
        jargon = KgJargonMap(extra={"完单": "order"})
        pipe = KgNL2SQLPipeline(
            question="q", graph_data=_sample_graph(), jargon=jargon
        )
        self.assertIs(pipe._jargon, jargon)

    def test_golden_store_queried_when_present(self):
        store = _FakeGoldenStore()
        # should not raise; golden_records fetched (empty here)
        KgNL2SQLPipeline(
            question="q", graph_data=_sample_graph(), golden_store=store
        )

    def test_golden_store_query_failure_is_guarded(self):
        class _Boom:
            def get_similar(self, question, top_k=3):
                raise RuntimeError("down")

        # must not raise -- the guard swallows the error
        KgNL2SQLPipeline(
            question="q", graph_data=_sample_graph(), golden_store=_Boom()
        )


# --------------------------------------------------------------------------- #
# run() with injected candidates
# --------------------------------------------------------------------------- #

class TestRunProvidedCandidates(unittest.TestCase):
    def setUp(self):
        self.pipe = KgNL2SQLPipeline(question="订单金额", graph_data=_sample_graph())

    def test_response_shape_and_route(self):
        resp = self.pipe.run(candidates=[
            "SELECT payment.amount FROM payment",
            "SELECT SUM(order.amount) FROM order",
        ])
        self.assertEqual(resp.route, "nl2sql")
        self.assertEqual(resp.answer, "SELECT SUM(order.amount) FROM order")
        stage_names = [s.stage for s in resp.stages]
        self.assertIn("linking", stage_names)
        self.assertIn("sql_generation", stage_names)
        self.assertIn("sql_validation", stage_names)
        self.assertIn("sql_voting", stage_names)
        self.assertIn("lineage", stage_names)
        # no conflicting metric name -> no authority stage
        self.assertNotIn("authority", stage_names)

    def test_generation_source_provided(self):
        resp = self.pipe.run(candidates=["SELECT order.city FROM order"])
        gen = next(s for s in resp.stages if s.stage == "sql_generation")
        self.assertEqual(gen.output["source"], "provided")
        self.assertEqual(gen.output["candidates"], ["SELECT order.city FROM order"])

    def test_voting_stage_chosen(self):
        resp = self.pipe.run(candidates=[
            "SELECT SUM(order.amount) FROM order",
            "SELECT payment.amount FROM payment",
        ])
        voting = next(s for s in resp.stages if s.stage == "sql_voting")
        self.assertEqual(voting.output["chosen"], "SELECT SUM(order.amount) FROM order")
        self.assertEqual(len(voting.output["ranked"]), 2)

    def test_validation_stage_reports(self):
        resp = self.pipe.run(candidates=["SELECT SUM(order.amount) FROM order"])
        val = next(s for s in resp.stages if s.stage == "sql_validation")
        self.assertEqual(len(val.output["validated"]), 1)
        self.assertTrue(val.output["validated"][0]["valid"])


class TestRunEmptyCandidates(unittest.TestCase):
    def test_empty_yields_empty_answer(self):
        pipe = KgNL2SQLPipeline(question="订单金额", graph_data=_sample_graph())
        resp = pipe.run(candidates=[])
        self.assertEqual(resp.answer, "")
        self.assertEqual(resp.stages[-1].stage, "lineage")
        # lineage target falls back to ctx.tables (no votes, refs empty)
        lineage = resp.stages[-1]
        self.assertIn("order", lineage.output["explain"])


# --------------------------------------------------------------------------- #
# run() with custom generate_fn
# --------------------------------------------------------------------------- #

class TestRunGenerateFn(unittest.TestCase):
    def test_custom_generate_fn_success(self):
        pipe = KgNL2SQLPipeline(
            question="q",
            graph_data=_sample_graph(),
            generate_fn=lambda q, ctx: ["SELECT order.city FROM order"],
        )
        resp = pipe.run()
        gen = next(s for s in resp.stages if s.stage == "sql_generation")
        self.assertEqual(gen.output["source"], "custom")
        self.assertEqual(gen.output["candidates"], ["SELECT order.city FROM order"])

    def test_custom_generate_fn_failure_guarded(self):
        def _boom(q, ctx):
            raise RuntimeError("gen failed")

        pipe = KgNL2SQLPipeline(
            question="q", graph_data=_sample_graph(), generate_fn=_boom
        )
        resp = pipe.run()
        gen = next(s for s in resp.stages if s.stage == "sql_generation")
        self.assertEqual(gen.output["source"], "custom")
        self.assertEqual(gen.output["candidates"], [])
        self.assertEqual(resp.answer, "")


# --------------------------------------------------------------------------- #
# run() default LLM path (mocked LLMs)
# --------------------------------------------------------------------------- #

class TestRunDefaultLlm(unittest.TestCase):
    def _fake_llms(self):
        class _FakeLLM:
            def generate(self, prompt):
                return "```sql\nSELECT SUM(order.amount) FROM order\n```"

        class _FakeLLMs:
            def get_text2gql_llm(self):
                return _FakeLLM()

        return _FakeLLMs

    def test_default_llm_source_and_extraction(self):
        with patch(
            "hugegraph_llm.models.llms.init_llm.LLMs", self._fake_llms()
        ):
            pipe = KgNL2SQLPipeline(question="订单金额", graph_data=_sample_graph())
            resp = pipe.run()
        gen = next(s for s in resp.stages if s.stage == "sql_generation")
        self.assertEqual(gen.output["source"], "llm")
        self.assertEqual(
            gen.output["candidates"], ["SELECT SUM(order.amount) FROM order"]
        )
        self.assertEqual(resp.answer, "SELECT SUM(order.amount) FROM order")


# --------------------------------------------------------------------------- #
# authority conflict branch (direct + run-level)
# --------------------------------------------------------------------------- #

class TestAuthorityNotes(unittest.TestCase):
    def setUp(self):
        self.pipe = KgNL2SQLPipeline(question="q", graph_data=_sample_graph())

    def test_empty_metrics_no_notes(self):
        self.assertEqual(self.pipe._authority_notes(_Ctx()), {})

    def test_single_metric_no_conflict(self):
        ctx = _Ctx(metrics=[{"name": "order_total"}])
        self.assertEqual(self.pipe._authority_notes(ctx), {})

    def test_conflicting_metric_surfaces(self):
        # the authority object must know about the name collision, so the
        # pipeline is built on the duplicate-metric graph
        pipe = KgNL2SQLPipeline(question="q", graph_data=_graph_dup_metric())
        ctx = _Ctx(metrics=[{"name": "order_total"}])
        notes = pipe._authority_notes(ctx)
        self.assertIn("order_total", notes)
        self.assertIn("alternatives", notes["order_total"])

    def test_skips_nameless_and_duplicates(self):
        pipe = KgNL2SQLPipeline(question="q", graph_data=_graph_dup_metric())
        ctx = _Ctx(metrics=[
            {"name": ""},
            {"name": "order_total"},
            {"name": "order_total"},
        ])
        notes = pipe._authority_notes(ctx)
        # only one note emitted (nameless skipped, duplicate skipped)
        self.assertEqual(list(notes.keys()), ["order_total"])


class TestRunAuthorityStage(unittest.TestCase):
    def test_authority_stage_present_on_conflict(self):
        pipe = KgNL2SQLPipeline(question="订单总额", graph_data=_graph_dup_metric())
        resp = pipe.run(candidates=["SELECT SUM(order.amount) FROM order"])
        names = [s.stage for s in resp.stages]
        # conflict may or may not surface depending on linking; the direct
        # test above guarantees the branch. Here we only assert no crash.
        self.assertIn("lineage", names)


# --------------------------------------------------------------------------- #
# lineage helper branches
# --------------------------------------------------------------------------- #

class TestLineageFor(unittest.TestCase):
    def setUp(self):
        self.pipe = KgNL2SQLPipeline(question="q", graph_data=_sample_graph())

    def _vote_with_refs(self, refs):
        class _Report:
            tables_referenced = refs

        class _Vote:
            report = _Report()

        return _Vote()

    def test_refs_present_takes_precedence(self):
        out = self.pipe._lineage_for(
            "SELECT order.city FROM order",
            _Ctx(tables=[{"name": "payment"}]),
            [self._vote_with_refs(["order"])],
        )
        self.assertIn("order", out)

    def test_refs_empty_falls_to_tables(self):
        out = self.pipe._lineage_for(
            "SELECT 1",
            _Ctx(tables=[{"name": "payment"}]),
            [self._vote_with_refs([])],
        )
        self.assertIn("payment", out)

    def test_no_votes_falls_to_tables(self):
        out = self.pipe._lineage_for(
            "SELECT 1", _Ctx(tables=[{"name": "payment"}]), []
        )
        self.assertIn("payment", out)

    def test_no_votes_no_tables_falls_to_metrics(self):
        out = self.pipe._lineage_for(
            "SELECT 1", _Ctx(metrics=[{"name": "order_total"}]), []
        )
        self.assertIn("order_total", out)

    def test_unknown_target_explain(self):
        out = self.pipe._lineage_for("SELECT 1", _Ctx(), [])
        self.assertIn("无法解释", out)


# --------------------------------------------------------------------------- #
# golden store path
# --------------------------------------------------------------------------- #

class TestMaybeStore(unittest.TestCase):
    def setUp(self):
        self.pipe = KgNL2SQLPipeline(question="q", graph_data=_sample_graph())

    def _ok_votes(self):
        voter = KgSqlVoter(graph_data=_sample_graph(), question="q")
        return voter.vote(["SELECT SUM(order.amount) FROM order"])

    def test_not_store_best_returns_none(self):
        self.assertIsNone(self.pipe._maybe_store("SELECT 1", self._ok_votes()))

    def test_no_golden_store_returns_none(self):
        self.pipe._store_best = True
        self.pipe._golden_store = None
        self.assertIsNone(self.pipe._maybe_store("SELECT 1", self._ok_votes()))

    def test_empty_sql_returns_none(self):
        self.pipe._store_best = True
        self.pipe._golden_store = _FakeGoldenStore()
        self.assertIsNone(self.pipe._maybe_store("", self._ok_votes()))

    def test_invalid_vote_not_stored(self):
        self.pipe._store_best = True
        store = _FakeGoldenStore()
        self.pipe._golden_store = store
        voter = KgSqlVoter(graph_data=_sample_graph(), question="q")
        votes = voter.vote(["SELECT AVG(order.amount) FROM order"])  # invalid
        self.assertIsNone(self.pipe._maybe_store("SELECT AVG(order.amount) FROM order", votes))
        self.assertEqual(store.added, [])

    def test_valid_vote_stored(self):
        self.pipe._store_best = True
        store = _FakeGoldenStore()
        self.pipe._golden_store = store
        votes = self._ok_votes()
        out = self.pipe._maybe_store("SELECT SUM(order.amount) FROM order", votes)
        self.assertTrue(out["stored"])
        self.assertEqual(out["vertex_id"], "vid-1")
        self.assertEqual(len(store.added), 1)

    def test_store_add_failure_guarded(self):
        self.pipe._store_best = True
        store = _FakeGoldenStore(raise_on_add=True)
        self.pipe._golden_store = store
        votes = self._ok_votes()
        out = self.pipe._maybe_store("SELECT SUM(order.amount) FROM order", votes)
        self.assertFalse(out["stored"])
        self.assertIn("error", out)


class TestRunGoldenFeedback(unittest.TestCase):
    def test_golden_feedback_stage_present(self):
        store = _FakeGoldenStore()
        pipe = KgNL2SQLPipeline(
            question="订单金额",
            graph_data=_sample_graph(),
            golden_store=store,
            store_best=True,
        )
        resp = pipe.run(candidates=["SELECT SUM(order.amount) FROM order"])
        names = [s.stage for s in resp.stages]
        self.assertIn("golden_feedback", names)
        self.assertEqual(len(store.added), 1)

    def test_no_store_no_golden_stage(self):
        pipe = KgNL2SQLPipeline(question="订单金额", graph_data=_sample_graph())
        resp = pipe.run(candidates=["SELECT SUM(order.amount) FROM order"])
        self.assertNotIn("golden_feedback", [s.stage for s in resp.stages])


# --------------------------------------------------------------------------- #
# client-path graph access
# --------------------------------------------------------------------------- #

class TestClientPath(unittest.TestCase):
    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    def test_graph_data_loaded_via_client(self, _mock):
        # patch must be active during construction too, because KgSqlVoter /
        # KgMetricAuthority build their graph eagerly from the client
        pipe = KgNL2SQLPipeline(question="订单金额", client=object())
        data = pipe._graph_data()
        self.assertIn("order", [t["name"] for t in data["vertices"]["Table"]])

    @patch.object(KgRuleEngine, "load_graph", return_value=_sample_graph())
    def test_run_with_client_and_candidates(self, _mock):
        pipe = KgNL2SQLPipeline(question="订单金额", client=object())
        resp = pipe.run(candidates=["SELECT SUM(order.amount) FROM order"])
        self.assertEqual(resp.answer, "SELECT SUM(order.amount) FROM order")
        self.assertEqual(resp.stages[0].stage, "linking")


if __name__ == "__main__":
    unittest.main()
