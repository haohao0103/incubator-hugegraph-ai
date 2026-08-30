import unittest
from unittest.mock import MagicMock

from hugegraph_llm.operators.graph_op.kg_schema_linker import (
    SchemaContext,
    SchemaLinkConfig,
    KgSchemaLinker,
)


def _graph():
    """Small warehouse: 2 tables, 4 fields, 2 metrics."""
    return {
        "vertices": {
            "Table": [
                {"name": "order", "comment": "订单表", "domain": "trade"},
                {"name": "driver", "comment": "司机表", "domain": "ops"},
            ],
            "Field": [
                {"name": "order.order_id", "comment": "订单号", "type": "LONG", "table": "order"},
                {"name": "order.amount", "comment": "订单金额", "type": "DOUBLE", "table": "order"},
                {"name": "order.city", "comment": "城市", "type": "STRING", "table": "order"},
                {"name": "driver.driver_id", "comment": "司机ID", "type": "LONG", "table": "driver"},
            ],
            "Metric": [
                {"name": "gmv", "definition": "成交额", "formula": "SUM(order.amount)"},
                {"name": "order_cnt", "definition": "完成订单数", "formula": ""},
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.order_id"),
                ("order", "order.amount"),
                ("order", "order.city"),
                ("driver", "driver.driver_id"),
            ],
            "computedFrom": [("gmv", "order"), ("order_cnt", "order")],
            "computedFromField": [("gmv", "order.amount")],
            "dependsOn": [],
        },
    }


class TestExtractTerms(unittest.TestCase):
    def test_empty_and_none(self):
        linker = KgSchemaLinker()
        self.assertEqual(linker.extract_terms(""), [])
        self.assertEqual(linker.extract_terms(None), [])

    def test_english_and_snake_case(self):
        linker = KgSchemaLinker()
        terms = linker.extract_terms("query order_amount by city")
        self.assertIn("order_amount", terms)
        self.assertIn("order", terms)
        self.assertIn("amount", terms)
        self.assertIn("city", terms)

    def test_stopwords_filtered(self):
        linker = KgSchemaLinker()
        terms = linker.extract_terms("查询的订单是多少")
        self.assertNotIn("的", terms)
        self.assertNotIn("查询", terms)

    def test_chinese_segment(self):
        linker = KgSchemaLinker()
        terms = linker.extract_terms("各城市订单金额")
        self.assertIn("订单", terms)
        self.assertIn("金额", terms)
        self.assertIn("城市", terms)

    def test_synonym_expansion(self):
        linker = KgSchemaLinker(synonyms={"完单": "order"})
        terms = linker.extract_terms("完单金额")
        self.assertIn("完单", terms)
        self.assertIn("order", terms)  # alias expanded to canonical

    def test_synonym_not_duplicated(self):
        linker = KgSchemaLinker(synonyms={"order": "order"})
        terms = linker.extract_terms("order amount")
        self.assertEqual(terms.count("order"), 1)


class TestScoreVertex(unittest.TestCase):
    def setUp(self):
        self.linker = KgSchemaLinker()

    def test_exact_name_match_highest(self):
        v = {"name": "order", "comment": "订单表"}
        self.assertAlmostEqual(self.linker.score_vertex("Table", v, ["order"]), 3.0)

    def test_partial_name_match(self):
        v = {"name": "order_detail", "comment": "订单明细"}
        self.assertAlmostEqual(self.linker.score_vertex("Table", v, ["order"]), 2.0)

    def test_comment_match(self):
        v = {"name": "t1", "comment": "订单表"}
        self.assertAlmostEqual(self.linker.score_vertex("Table", v, ["订单"]), 1.0)

    def test_no_match_zero(self):
        v = {"name": "driver", "comment": "司机表"}
        self.assertEqual(self.linker.score_vertex("Table", v, ["订单"]), 0.0)

    def test_metric_definition_match_weighted(self):
        v = {"name": "gmv", "definition": "成交额", "formula": ""}
        self.assertAlmostEqual(self.linker.score_vertex("Metric", v, ["成交额"]), 1.2)

    def test_metric_formula_match(self):
        v = {"name": "gmv", "definition": "", "formula": "SUM(order.amount)"}
        self.assertGreater(self.linker.score_vertex("Metric", v, ["amount"]), 0.0)

    def test_vertex_without_name_still_matches_comment(self):
        # link() skips nameless vertices; scoring itself still reads comments
        self.assertAlmostEqual(
            self.linker.score_vertex("Table", {"comment": "x"}, ["x"]), 1.0
        )

    def test_empty_term_skipped(self):
        v = {"name": "order"}
        self.assertEqual(self.linker.score_vertex("Table", v, [""]), 0.0)

    def test_unknown_label_defaults(self):
        v = {"name": "custom"}
        self.assertAlmostEqual(self.linker.score_vertex("Ghost", v, ["custom"]), 3.0)


class TestLink(unittest.TestCase):
    def setUp(self):
        self.linker = KgSchemaLinker()
        self.data = _graph()

    def test_link_table_by_name(self):
        ctx = self.linker.link("order 表有哪些字段", self.data)
        self.assertEqual([t["name"] for t in ctx.tables], ["order"])
        self.assertIn("order.amount", [f["name"] for f in ctx.fields])

    def test_link_expands_metric_sources(self):
        ctx = self.linker.link("gmv 是多少", self.data)
        self.assertEqual([m["name"] for m in ctx.metrics], ["gmv"])
        # metric is computed from order.amount -> field included
        self.assertIn("order.amount", [f["name"] for f in ctx.fields])

    def test_link_by_metric_definition_chinese(self):
        ctx = self.linker.link("成交额", self.data)
        self.assertEqual([m["name"] for m in ctx.metrics], ["gmv"])

    def test_no_terms_returns_empty(self):
        ctx = self.linker.link("的 是 查询", self.data)
        self.assertEqual(ctx.tables, [])
        self.assertEqual(ctx.fields, [])
        self.assertEqual(ctx.metrics, [])

    def test_no_match_returns_empty_context(self):
        ctx = self.linker.link("xyzzy 不存在", self.data)
        self.assertEqual(ctx.tables, [])
        self.assertEqual(ctx.metrics, [])

    def test_no_match_prompt_context(self):
        ctx = self.linker.link("xyzzy 不存在", self.data)
        self.assertIn("(no schema matched)", ctx.to_prompt_context())

    def test_relations_collected(self):
        ctx = self.linker.link("order 表有哪些字段", self.data)
        self.assertIn(("order", "hasColumn", "order.amount"), ctx.relations)
        self.assertIn(("gmv", "computedFrom", "order"), ctx.relations)

    def test_max_tables_limit(self):
        linker = KgSchemaLinker(config=SchemaLinkConfig(max_tables=1))
        ctx = linker.link("order driver 司机", self.data)
        self.assertLessEqual(len(ctx.tables), 1)

    def test_max_metrics_limit(self):
        linker = KgSchemaLinker(config=SchemaLinkConfig(max_metrics=1))
        ctx = linker.link("gmv order_cnt", self.data)
        self.assertEqual(len(ctx.metrics), 1)

    def test_field_direct_match_included(self):
        ctx = self.linker.link("城市", self.data)
        self.assertIn("order.city", [f["name"] for f in ctx.fields])

    def test_field_budget_respected(self):
        linker = KgSchemaLinker(config=SchemaLinkConfig(max_fields_per_table=1))
        ctx = linker.link("order 司机 driver 城市 订单号", self.data)
        self.assertGreater(len(ctx.fields), 0)

    def test_evidence_built_from_metric_formula(self):
        ctx = self.linker.link("gmv 是多少", self.data)
        self.assertTrue(any("SUM(order.amount)" in e for e in ctx.evidence))

    def test_evidence_uses_definition_when_no_formula(self):
        ctx = self.linker.link("完成订单数", self.data)
        self.assertTrue(any("完成订单数" in e for e in ctx.evidence))

    def test_evidence_includes_table_and_field_comments(self):
        ctx = self.linker.link("订单金额", self.data)
        joined = " ".join(ctx.evidence)
        self.assertIn("订单金额", joined)

    def test_evidence_max_limit(self):
        linker = KgSchemaLinker(config=SchemaLinkConfig(max_evidence=1))
        ctx = linker.link("order gmv 订单金额", self.data)
        self.assertLessEqual(len(ctx.evidence), 1)

    def test_include_comments_false(self):
        linker = KgSchemaLinker(config=SchemaLinkConfig(include_comments=False))
        ctx = linker.link("成交额 gmv", self.data)
        # only metric evidence remains
        self.assertTrue(all(e.startswith("指标") for e in ctx.evidence))

    def test_min_score_filters_weak_matches(self):
        linker = KgSchemaLinker(config=SchemaLinkConfig(min_score=3.0))
        ctx = linker.link("订单表", self.data)  # comment-only match scores 1.0
        self.assertEqual(ctx.tables, [])

    def test_rank_top_k_is_top_scoring(self):
        rows = [
            {"name": "order"}, {"name": "order_dw"}, {"name": "order_pay"},
            {"name": "order_cnt"}, {"name": "order_log"}, {"name": "order_tmp"},
            {"name": "t1"}, {"name": "t2"},
        ]
        full = self.linker._rank(rows, "Table", ["order"])
        top = self.linker._rank(rows, "Table", ["order"], top_k=3)
        self.assertEqual(len(top), 3)
        top_names = {v["name"] for _, v in top}
        top_scores = {s for s, _ in top}
        others = {s for s, v in full if v["name"] not in top_names}
        # heap top-k invariant: every winner scores >= every non-winner
        if others:
            self.assertLessEqual(max(others), min(top_scores))
        # exact name match (3.0) must always be a winner
        self.assertIn("order", top_names)

    def test_rank_top_k_none_returns_full_list(self):
        rows = [{"name": "order"}, {"name": "order_dw"}]
        full = self.linker._rank(rows, "Table", ["order"])
        self.assertEqual(len(full), 2)  # backward-compatible full ranking

    def test_link_respects_max_tables_with_many_matches(self):
        data = {
            "vertices": {
                "Table": [{"name": f"order_{i}"} for i in range(20)] + [{"name": "order"}],
                "Field": [],
                "Metric": [],
            },
            "edges": {
                "hasColumn": [], "computedFrom": [], "computedFromField": [], "dependsOn": [],
            },
        }
        linker = KgSchemaLinker(config=SchemaLinkConfig(max_tables=3))
        ctx = linker.link("order", data)
        self.assertEqual(len(ctx.tables), 3)
        self.assertIn("order", {t["name"] for t in ctx.tables})


class TestEvidenceEdgeCases(unittest.TestCase):
    """Metrics/tables/fields with missing props produce no empty evidence."""

    def setUp(self):
        self.linker = KgSchemaLinker()

    def test_metric_without_definition(self):
        ctx = SchemaContext()
        ev = self.linker._build_evidence(
            [], [{"name": "m1", "definition": "", "formula": "SUM(x)"}], []
        )
        self.assertEqual(ev, ["指标 m1 = SUM(x)"])

    def test_metric_without_formula_or_definition_skipped(self):
        ev = self.linker._build_evidence([], [{"name": "m1"}], [])
        self.assertEqual(ev, [])

    def test_table_without_comment(self):
        ev = self.linker._build_evidence([{"name": "t1"}], [], [])
        self.assertEqual(ev, [])

    def test_field_comment_without_type(self):
        ev = self.linker._build_evidence([], [], [{"name": "f1", "comment": "金额"}])
        self.assertEqual(ev, ["字段 f1 是金额"])

    def test_field_without_comment_skipped(self):
        ev = self.linker._build_evidence([], [], [{"name": "f1", "type": "INT"}])
        self.assertEqual(ev, [])


class TestSchemaContext(unittest.TestCase):
    def test_prompt_context_structure(self):
        ctx = SchemaContext(
            tables=[{"name": "order", "comment": "订单表"}],
            fields=[{"name": "order.amount", "comment": "订单金额", "type": "DOUBLE"}],
            metrics=[{"name": "gmv", "definition": "成交额", "formula": "SUM(order.amount)"}],
            relations=[("order", "hasColumn", "order.amount")],
            evidence=["指标 gmv = SUM(order.amount)"],
            matched_terms=["gmv"],
        )
        text = ctx.to_prompt_context()
        self.assertIn("== SCHEMA ==", text)
        self.assertIn("table order  -- 订单表", text)
        self.assertIn("- order.amount  -- 订单金额 / DOUBLE", text)
        self.assertIn("== METRICS ==", text)
        self.assertIn("formula: SUM(order.amount)", text)
        self.assertIn("== EVIDENCE (external knowledge) ==", text)
        self.assertIn("1. 指标 gmv = SUM(order.amount)", text)

    def test_prompt_context_metric_formula_only(self):
        ctx = SchemaContext(metrics=[{"name": "m1", "definition": "", "formula": "SUM(x)"}])
        text = ctx.to_prompt_context()
        self.assertIn("formula: SUM(x)", text)
        self.assertNotIn("definition:", text)

    def test_prompt_context_metric_definition_only(self):
        ctx = SchemaContext(metrics=[{"name": "m1", "definition": "总额", "formula": ""}])
        text = ctx.to_prompt_context()
        self.assertIn("definition: 总额", text)
        self.assertNotIn("formula:", text)

    def test_prompt_context_without_evidence(self):
        ctx = SchemaContext(evidence=["x"])
        self.assertNotIn("EVIDENCE", ctx.to_prompt_context(include_evidence=False))

    def test_prompt_context_field_without_comment(self):
        ctx = SchemaContext(fields=[{"name": "f1"}])
        self.assertIn("- f1", ctx.to_prompt_context())

    def test_to_dict(self):
        ctx = SchemaContext(
            tables=[{"name": "t"}], relations=[("a", "r", "b")], matched_terms=["q"]
        )
        d = ctx.to_dict()
        self.assertEqual(d["tables"], [{"name": "t"}])
        self.assertEqual(d["relations"], [["a", "r", "b"]])
        self.assertEqual(d["matched_terms"], ["q"])


class TestLoadGraph(unittest.TestCase):
    def test_no_client_returns_empty(self):
        linker = KgSchemaLinker()
        self.assertEqual(linker.load_graph(), {"vertices": {}, "edges": {}})

    def test_delegates_to_rule_engine(self):
        client = MagicMock()
        linker = KgSchemaLinker(client)
        with unittest.mock.patch(
            "hugegraph_llm.operators.graph_op.kg_rule_engine.KgRuleEngine"
        ) as mock_engine:
            mock_engine.return_value.load_graph.return_value = {"vertices": {"Table": []}}
            data = linker.load_graph()
        self.assertEqual(data, {"vertices": {"Table": []}})
        mock_engine.assert_called_once_with(client)

    def test_link_loads_when_data_missing(self):
        linker = KgSchemaLinker()
        with unittest.mock.patch.object(
            KgSchemaLinker, "load_graph", return_value=_graph()
        ):
            ctx = linker.link("order 表")
        self.assertEqual([t["name"] for t in ctx.tables], ["order"])


if __name__ == "__main__":
    unittest.main()
