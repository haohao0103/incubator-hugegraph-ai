"""Tests for KgSqlValidator (NL2SQL SQL validation against the metadata KG)."""

import unittest

from hugegraph_llm.operators.graph_op.kg_sql_validator import (
    KgSqlValidator,
    SqlValidationReport,
    SqlIssue,
    load_graph,
)
from hugegraph_llm.operators.graph_op.kg_rule_engine import GraphData


def _sample_graph() -> GraphData:
    return {
        "vertices": {
            "Table": [
                {"name": "order"},
                {"name": "payment"},
                {"name": "user"},
            ],
            "Field": [
                {"name": "order.amount", "type": "DOUBLE"},
                {"name": "order.order_id", "type": "BIGINT"},
                {"name": "order.city", "type": "STRING"},
                {"name": "payment.amount", "type": "DOUBLE"},
                {"name": "payment.order_id", "type": "BIGINT"},
                {"name": "user.user_id", "type": "BIGINT"},
            ],
            "Metric": [
                {"name": "order_total", "formula": "SUM(order.amount)", "definition": "订单总金额"},
                {"name": "pay_total", "formula": "SUM(payment.amount)", "definition": "支付总金额"},
                {"name": "cnt", "formula": "COUNT(order.order_id)", "definition": "订单数"},
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.amount"),
                ("order", "order.order_id"),
                ("order", "order.city"),
                ("payment", "payment.amount"),
                ("payment", "payment.order_id"),
                ("user", "user.user_id"),
            ],
            "computedFromField": [
                ("order_total", "order.amount"),
                ("pay_total", "payment.amount"),
                ("cnt", "order.order_id"),
            ],
            "computedFrom": [],
            "dependsOn": [],
        },
    }


class TestParseSql(unittest.TestCase):
    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_from_and_join_with_alias(self):
        p = self.v._parse_sql(
            "SELECT a.x FROM order a JOIN payment b ON a.order_id=b.order_id"
        )
        self.assertEqual(p["tables"], [("order", "a"), ("payment", "b")])
        self.assertEqual(p["alias_map"], {"a": "order", "b": "payment"})

    def test_qualified_cols(self):
        p = self.v._parse_sql("SELECT order.amount FROM order")
        self.assertIn(("order", "amount"), p["qualified_cols"])

    def test_bare_cols_excludes_keywords_and_functions(self):
        p = self.v._parse_sql(
            "SELECT order_id FROM order WHERE amount > 10 GROUP BY order_id"
        )
        # 'order_id' appears twice in bare (SELECT + GROUP BY); keywords excluded
        self.assertIn("order_id", p["bare_cols"])
        self.assertNotIn("SELECT", p["bare_cols"])
        self.assertNotIn("FROM", p["bare_cols"])
        self.assertNotIn("WHERE", p["bare_cols"])
        self.assertNotIn("GROUP", p["bare_cols"])
        self.assertNotIn("BY", p["bare_cols"])

    def test_aggregates_qualified_and_bare(self):
        p = self.v._parse_sql(
            "SELECT SUM(order.amount), COUNT(*) FROM order"
        )
        funcs = {(f, t, c) for f, t, c in p["aggregates"]}
        self.assertIn(("SUM", "order", "amount"), funcs)
        self.assertIn(("COUNT", None, None), funcs)

    def test_aggregates_distinct(self):
        p = self.v._parse_sql("SELECT COUNT(DISTINCT order.order_id) FROM order")
        self.assertIn(("COUNT", "order", "order_id"), {(f, t, c) for f, t, c in p["aggregates"]})

    def test_select_aliases_extracted_and_filtered_from_bare(self):
        p = self.v._parse_sql(
            "SELECT city, SUM(order.amount) AS order_amount FROM order "
            "GROUP BY city ORDER BY order_amount DESC"
        )
        self.assertIn("order_amount", p["aliases"])
        # the alias must not be treated as an unknown bare column
        self.assertNotIn("order_amount", p["bare_cols"])
        self.assertIn("city", p["bare_cols"])

    def test_implicit_alias_extracted(self):
        p = self.v._parse_sql(
            "SELECT SUM(order.amount) total FROM order ORDER BY total"
        )
        self.assertIn("total", p["aliases"])
        self.assertNotIn("total", p["bare_cols"])

    def test_plain_column_not_misread_as_implicit_alias(self):
        # "SELECT city" must not register "city" as an alias (no expression)
        p = self.v._parse_sql("SELECT city FROM order")
        self.assertNotIn("city", p["aliases"])
        self.assertIn("city", p["bare_cols"])

    def test_keyword_not_registered_as_implicit_alias(self):
        # trailing identifier that is a SQL keyword is not an alias
        p = self.v._parse_sql("SELECT SUM(order.amount) order FROM order")
        self.assertNotIn("order", p["aliases"])

    def test_distinct_bare_unknown_not_aliased(self):
        # "DISTINCT nope": the leading part has no expression, so "nope" must
        # stay a bare column (and be flagged later as unknown), not an alias
        p = self.v._parse_sql("SELECT DISTINCT nope FROM order")
        self.assertNotIn("nope", p["aliases"])
        self.assertIn("nope", p["bare_cols"])

    def test_public_parse_sql_wrapper(self):
        from hugegraph_llm.operators.graph_op.kg_sql_validator import parse_sql

        p = parse_sql("SELECT order.amount FROM order")
        self.assertEqual(p["tables"], [("order", None)])
        self.assertEqual(p["aliases"], [])


class TestTableExistence(unittest.TestCase):
    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_valid_table_no_issue(self):
        r = self.v.validate("SELECT 1 FROM order")
        self.assertTrue(r.is_valid)
        self.assertFalse(any(i.rule_id == "SQL-A1" for i in r.issues))

    def test_unknown_table_error(self):
        r = self.v.validate("SELECT * FROM nonexistent")
        a1 = [i for i in r.issues if i.rule_id == "SQL-A1"]
        self.assertEqual(len(a1), 1)
        self.assertEqual(a1[0].level, "error")
        self.assertIn("nonexistent", a1[0].message)
        self.assertIn("order", a1[0].suggested_fix)

    def test_no_table_parsed_warning(self):
        r = self.v.validate("SELECT 1")
        self.assertFalse(r.is_valid is False)  # no error, only warning
        self.assertTrue(any(i.rule_id == "SQL-P1" for i in r.issues))


class TestColumnOwnership(unittest.TestCase):
    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_qualified_col_ok(self):
        r = self.v.validate("SELECT order.amount FROM order")
        self.assertTrue(r.is_valid)
        self.assertIn("order.amount", r.columns_resolved)

    def test_qualified_col_not_owned_error(self):
        r = self.v.validate("SELECT order.foo FROM order")
        a2 = [i for i in r.issues if i.rule_id == "SQL-A2"]
        self.assertEqual(len(a2), 1)
        self.assertEqual(a2[0].level, "error")
        self.assertIn("order", a2[0].suggested_fix)

    def test_unknown_bare_col_error(self):
        r = self.v.validate("SELECT nope FROM order")
        a2 = [i for i in r.issues if i.rule_id == "SQL-A2" and "nope" in i.target]
        self.assertEqual(len(a2), 1)
        self.assertEqual(a2[0].level, "error")

    def test_ambiguous_bare_col_warning(self):
        # order_id is owned by both order and payment
        r = self.v.validate(
            "SELECT order_id FROM order JOIN payment ON order.order_id=payment.order_id"
        )
        a2 = [i for i in r.issues if i.rule_id == "SQL-A2" and "order_id" in i.target]
        self.assertEqual(len(a2), 1)
        self.assertEqual(a2[0].level, "warning")
        self.assertIn("order", a2[0].suggested_fix)
        self.assertIn("payment", a2[0].suggested_fix)


class TestSelectAliases(unittest.TestCase):
    """SQL-A2 must resolve SELECT aliases in ORDER BY / GROUP BY / HAVING."""

    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_order_by_explicit_alias_valid(self):
        r = self.v.validate(
            "SELECT city, SUM(order.amount) AS order_amount FROM order "
            "GROUP BY city ORDER BY order_amount DESC"
        )
        self.assertTrue(r.is_valid, [i.message for i in r.issues])

    def test_order_by_implicit_alias_valid(self):
        r = self.v.validate(
            "SELECT SUM(order.amount) total FROM order ORDER BY total"
        )
        self.assertTrue(r.is_valid, [i.message for i in r.issues])

    def test_alias_with_metric_caliber_still_checked(self):
        # alias is resolved, but the aggregate caliber check still applies
        r = self.v.validate(
            "SELECT city, AVG(order.amount) AS avg_amount FROM order "
            "GROUP BY city ORDER BY avg_amount DESC"
        )
        b1 = [i for i in r.issues if i.rule_id == "SQL-B1"]
        self.assertTrue(b1, "AVG must still mismatch the SUM metric")
        self.assertFalse(r.is_valid)

    def test_undefined_alias_still_error(self):
        r = self.v.validate("SELECT order.amount FROM order ORDER BY nope")
        a2 = [i for i in r.issues if i.rule_id == "SQL-A2" and "nope" in i.target]
        self.assertTrue(a2)
        self.assertFalse(r.is_valid)


class TestMetricCaliber(unittest.TestCase):
    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_metric_match_no_issue(self):
        r = self.v.validate("SELECT SUM(order.amount) FROM order")
        self.assertTrue(r.is_valid)
        self.assertIn("order_total", r.metrics_checked)
        self.assertFalse(any(i.rule_id == "SQL-B1" for i in r.issues))

    def test_metric_func_mismatch_error(self):
        r = self.v.validate("SELECT AVG(order.amount) FROM order")
        b1 = [i for i in r.issues if i.rule_id == "SQL-B1"]
        self.assertEqual(len(b1), 1)
        self.assertEqual(b1[0].level, "error")
        self.assertIn("SUM(order.amount)", b1[0].suggested_fix)

    def test_count_metric_match(self):
        r = self.v.validate("SELECT COUNT(order.order_id) FROM order")
        self.assertTrue(r.is_valid)
        self.assertIn("cnt", r.metrics_checked)

    def test_count_metric_mismatch(self):
        r = self.v.validate("SELECT SUM(order.order_id) FROM order")
        b1 = [i for i in r.issues if i.rule_id == "SQL-B1"]
        self.assertEqual(len(b1), 1)
        self.assertIn("COUNT", b1[0].suggested_fix)


class TestJoinConnectivity(unittest.TestCase):
    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_joinable_via_shared_field(self):
        r = self.v.validate(
            "SELECT * FROM order JOIN payment ON order.order_id=payment.order_id"
        )
        self.assertFalse(any(i.rule_id == "SQL-J1" for i in r.issues))

    def test_joinable_via_metric(self):
        # order and payment are co-referenced by no metric directly, but they
        # share order_id, so still joinable; test a metric-spanned pair by
        # adding a metric spanning both is unnecessary -- shared field covers it.
        r = self.v.validate(
            "SELECT SUM(order.amount) FROM order JOIN payment ON order.order_id=payment.order_id"
        )
        self.assertFalse(any(i.rule_id == "SQL-J1" for i in r.issues))

    def test_not_joinable_warning_with_hint(self):
        # order and user share no field and no metric spans both
        r = self.v.validate(
            "SELECT * FROM order JOIN user ON order.order_id=user.user_id"
        )
        j1 = [i for i in r.issues if i.rule_id == "SQL-J1"]
        self.assertEqual(len(j1), 1)
        self.assertEqual(j1[0].level, "warning")
        self.assertIn("无共享字段", j1[0].suggested_fix)

    def test_single_table_no_join_check(self):
        r = self.v.validate("SELECT 1 FROM order")
        self.assertFalse(any(i.rule_id == "SQL-J1" for i in r.issues))

    def test_join_with_unknown_table_skipped_in_connectivity(self):
        # unknown table is reported by A1, not by J1
        r = self.v.validate("SELECT * FROM order JOIN ghost ON order.order_id=ghost.x")
        self.assertTrue(any(i.rule_id == "SQL-A1" for i in r.issues))
        self.assertFalse(any(i.rule_id == "SQL-J1" for i in r.issues))


class TestReportHelpers(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        r = KgSqlValidator(_sample_graph()).validate("SELECT AVG(order.amount) FROM order")
        d = r.to_dict()
        self.assertFalse(d["is_valid"])
        self.assertEqual(len(d["issues"]), 1)

    def test_to_prompt_feedback_empty_when_valid(self):
        r = KgSqlValidator(_sample_graph()).validate("SELECT SUM(order.amount) FROM order")
        self.assertEqual(r.to_prompt_feedback(), "")

    def test_to_prompt_feedback_lists_issues(self):
        r = KgSqlValidator(_sample_graph()).validate("SELECT AVG(order.amount) FROM order")
        fb = r.to_prompt_feedback()
        self.assertIn("AVG", fb)
        self.assertIn("SUM(order.amount)", fb)

    def test_issue_to_dict(self):
        i = SqlIssue("SQL-A1", "error", "t:x", "msg", "fix")
        self.assertEqual(i.to_dict()["suggested_fix"], "fix")


class TestLoadGraphReuse(unittest.TestCase):
    def test_load_graph_uses_rule_engine_signature(self):
        # load_graph must accept (client, graph_name) and return GraphData shape
        class _FakeClient:
            def gremlin(self):
                return self

            def exec(self, q):
                # minimal elementMap responses for the loader
                if "V()" in q:
                    return {"data": [{"id": "1:order", "label": "Table", "name": "order"}]}
                return {"data": []}

        g = load_graph(_FakeClient())
        self.assertIn("vertices", g)
        self.assertIn("edges", g)
        self.assertEqual(g["vertices"]["Table"][0]["name"], "order")

    def test_from_client_builds_validator(self):
        class _FakeClient:
            def gremlin(self):
                return self

            def exec(self, q):
                if "V()" in q:
                    return {"data": [
                        {"id": "1:order", "label": "Table", "name": "order"},
                        {"id": "2:order.amount", "label": "Field", "name": "order.amount"},
                    ]}
                if "E()" in q:
                    return {"data": [{"OUT": {"id": "1:order"}, "IN": {"id": "2:order.amount"}}]}
                return {"data": []}

        v = KgSqlValidator.from_client(_FakeClient())
        self.assertIn("order", v.table_names)


class TestValidatorEdgeCases(unittest.TestCase):
    """Branch coverage for odd-but-real KG shapes and internals."""

    def setUp(self):
        self.g = _sample_graph()
        # a field without a declared type, a field without a name, and a metric
        # with no name
        self.g["vertices"]["Field"].append({"name": "order.status"})  # no type
        self.g["edges"]["hasColumn"].append(("order", "order.status"))
        self.g["vertices"]["Field"].append({"type": "INT"})  # no name
        self.g["vertices"]["Metric"].append(
            {"name": "", "formula": "SUM(order.status)", "definition": "x"}
        )
        # a metric spanning two tables that share NO column basename
        self.g["vertices"]["Metric"].append(
            {"name": "span_m", "formula": "SUM(order.amount) + SUM(user.user_id)",
             "definition": "span"}
        )
        self.g["edges"]["computedFromField"].append(("span_m", "order.amount"))
        self.g["edges"]["computedFromField"].append(("span_m", "user.user_id"))
        self.v = KgSqlValidator(self.g)

    def test_field_without_type_still_resolves(self):
        r = self.v.validate("SELECT order.status FROM order")
        self.assertTrue(r.is_valid)

    def test_metric_without_name_skipped(self):
        self.assertNotIn("", self.v.metrics)

    def test_resolve_col_direct_branches(self):
        self.assertEqual(self.v._resolve_col("order", "amount"), "order")
        self.assertIsNone(self.v._resolve_col(None, "amount"))
        self.assertIsNone(self.v._resolve_col("order", "nope"))

    def test_bare_single_owner_resolved(self):
        r = self.v.validate("SELECT status FROM order")
        self.assertIn("order.status", r.columns_resolved)

    def test_aggregate_no_metric_match_continues(self):
        r = self.v.validate("SELECT MAX(order.status) FROM order")
        self.assertTrue(r.is_valid)
        self.assertEqual(r.metrics_checked, [])

    def test_metric_for_column_bare_basename(self):
        self.assertEqual(self.v._metric_for_column("amount", None), "order_total")
        self.assertIsNone(self.v._metric_for_column("status", None))

    def test_ref_fields_empty(self):
        self.assertEqual(self.v._ref_fields(""), [])

    def test_suggest_column_owners(self):
        # bare unknown col whose basename has known owners
        self.assertIn("order", self.v._suggest_column(None, "amount"))

    def test_report_errors_warnings_props(self):
        r = self.v.validate("SELECT AVG(order.amount) FROM order")
        self.assertEqual(len(r.errors), 1)
        self.assertEqual(len(r.warnings), 0)
        self.assertEqual(len(r.warnings), 0)  # property reads empty list

    def test_prompt_feedback_without_fix(self):
        rep = SqlValidationReport(issues=[SqlIssue("X", "error", "t", "msg")])
        fb = rep.to_prompt_feedback()
        self.assertIn("msg", fb)
        self.assertNotIn("->", fb)

    def test_joinable_via_metric_only(self):
        # order + user share no basename but span_m references both
        r = self.v.validate(
            "SELECT * FROM order JOIN user ON order.order_id=user.user_id"
        )
        self.assertFalse(any(i.rule_id == "SQL-J1" for i in r.issues))


class TestParseAggregatesBare(unittest.TestCase):
    def setUp(self):
        self.v = KgSqlValidator(_sample_graph())

    def test_aggregate_bare_col(self):
        p = self.v._parse_sql("SELECT SUM(amount) FROM order")
        self.assertIn(
            ("SUM", None, "amount"),
            {(f, t, c) for f, t, c in p["aggregates"]},
        )

    def test_function_like_token_skipped_in_bare(self):
        # COALESCE is not a keyword and is followed by '(' -> must be skipped
        p = self.v._parse_sql("SELECT COALESCE(amount, 0) FROM order")
        self.assertNotIn("COALESCE", p["bare_cols"])


if __name__ == "__main__":
    unittest.main()
