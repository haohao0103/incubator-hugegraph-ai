import unittest
from unittest.mock import MagicMock

from hugegraph_llm.operators.graph_op.kg_rule_engine import (
    EDGE_ENDPOINTS,
    KgRuleEngine,
    KgRuleReport,
    KgViolation,
    RULE_IDS,
    derive_c1_dependency_closure,
    derive_c2_bottom_tables,
    run_rules,
)


def _healthy_graph():
    """A fully-valid metadata graph: 2 tables, 2 fields, 2 metrics, all rules pass."""
    return {
        "vertices": {
            "Table": [
                {"name": "order", "domain": "trade"},
                {"name": "payment", "domain": "trade"},
            ],
            "Field": [
                {"name": "order.amount", "type": "DOUBLE", "domain": "trade"},
                {"name": "order.status", "type": "STRING", "domain": "trade"},
                {"name": "payment.id", "type": "LONG", "domain": "trade"},
            ],
            "Metric": [
                {"name": "gmv", "definition": "sum of order.amount", "domain": "trade"},
                {"name": "order_cnt", "definition": "count of orders", "domain": "trade"},
                {"name": "pay_cnt", "definition": "count of payments", "domain": "trade"},
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.amount"),
                ("order", "order.status"),
                ("payment", "payment.id"),
            ],
            "computedFrom": [("gmv", "order"), ("pay_cnt", "payment"), ("order_cnt", "order")],
            "computedFromField": [("gmv", "order.amount")],
            "dependsOn": [("order_cnt", "gmv")],
        },
    }


class TestKgViolation(unittest.TestCase):
    def test_to_dict(self):
        v = KgViolation(rule_id="A1", level="error", target="Field: x", message="m", details={"k": "v"})
        d = v.to_dict()
        self.assertEqual(d["rule_id"], "A1")
        self.assertEqual(d["details"], {"k": "v"})


class TestKgRuleReport(unittest.TestCase):
    def test_counts_and_by_rule(self):
        r = KgRuleReport(
            violations=[
                KgViolation(rule_id="A1", level="error", target="t1", message="m"),
                KgViolation(rule_id="A1", level="error", target="t2", message="m"),
                KgViolation(rule_id="D2", level="warning", target="t3", message="m"),
            ],
            derived={"C1": [{"metric": "gmv"}]},
        )
        self.assertEqual(r.error_count, 2)
        self.assertEqual(r.warning_count, 1)
        self.assertEqual(r.by_rule(), {"A1": 2, "D2": 1})
        d = r.to_dict()
        self.assertEqual(d["stats"]["total"], 3)
        self.assertEqual(d["derived"]["C1"][0]["metric"], "gmv")

    def test_empty_report(self):
        r = KgRuleReport()
        self.assertEqual(r.error_count, 0)
        self.assertEqual(r.warning_count, 0)
        self.assertEqual(r.by_rule(), {})


class TestRulesHealthy(unittest.TestCase):
    """The healthy graph must produce zero violations."""

    def test_healthy_graph_no_violations(self):
        report = run_rules(_healthy_graph())
        self.assertEqual(report.violations, [])
        self.assertIn("C1", report.derived)
        self.assertIn("C2", report.derived)


class TestRulesViolations(unittest.TestCase):
    def _graph(self, **overrides):
        g = _healthy_graph()
        if "vertices" in overrides:
            g["vertices"] = overrides["vertices"]
        if "edges" in overrides:
            g["edges"] = overrides["edges"]
        return g

    def test_a1_orphan_field(self):
        g = self._graph()
        # add a field with no owning hasColumn edge
        g["vertices"]["Field"].append({"name": "order.ghost", "type": "INT", "domain": "trade"})
        report = run_rules(g)
        a1 = [v for v in report.violations if v.rule_id == "A1"]
        self.assertEqual(len(a1), 1)
        self.assertEqual(a1[0].target, "Field: order.ghost")
        self.assertEqual(a1[0].level, "error")

    def test_a2_dangling_edge_source_and_target(self):
        g = self._graph()
        g["edges"]["hasColumn"].append(("order", "no_such_field"))
        g["edges"]["computedFrom"].append(("no_such_metric", "order"))
        report = run_rules(g)
        a2 = [v for v in report.violations if v.rule_id == "A2"]
        self.assertEqual(len(a2), 2)  # target missing + source missing
        targets = [v.target for v in a2]
        self.assertTrue(any("no_such_field" in t for t in targets))
        self.assertTrue(any("no_such_metric" in t for t in targets))

    def test_a2_unknown_edge_label_skipped(self):
        g = self._graph()
        g["edges"]["mystery"] = [("a", "b")]  # not in EDGE_ENDPOINTS -> skipped
        report = run_rules(g)
        self.assertEqual(report.violations, [])

    def test_a3_vertex_without_name_skipped(self):
        g = self._graph()
        g["vertices"]["Table"].append({"domain": "trade"})  # no name
        report = run_rules(g)
        self.assertEqual([v for v in report.violations if v.rule_id == "A3"], [])

    def test_a3_duplicate_pk(self):
        g = self._graph()
        g["vertices"]["Metric"].append({"name": "gmv", "definition": "dup", "domain": "trade"})
        report = run_rules(g)
        a3 = [v for v in report.violations if v.rule_id == "A3"]
        self.assertEqual(len(a3), 1)
        self.assertEqual(a3[0].details["count"], 2)

    def test_a4_invalid_field_type(self):
        g = self._graph()
        g["vertices"]["Field"].append({"name": "x.bad", "type": "NOTATYPE", "domain": "trade"})
        report = run_rules(g)
        a4 = [v for v in report.violations if v.rule_id == "A4"]
        self.assertEqual(len(a4), 1)
        self.assertIn("NOTATYPE", a4[0].message)

    def test_a4_valid_type_case_insensitive(self):
        g = self._graph()
        g["vertices"]["Field"].append({"name": "x.ok", "type": "double", "domain": "trade"})
        report = run_rules(g)
        self.assertEqual([v for v in report.violations if v.rule_id == "A4"], [])

    def test_a5_empty_table(self):
        g = self._graph()
        g["vertices"]["Table"].append({"name": "empty_tbl", "domain": "trade"})
        report = run_rules(g)
        a5 = [v for v in report.violations if v.rule_id == "A5"]
        self.assertEqual(len(a5), 1)
        self.assertEqual(a5[0].level, "warning")

    def test_b1_metric_no_source(self):
        g = self._graph()
        g["vertices"]["Metric"].append({"name": "orphan_metric", "definition": "x", "domain": "trade"})
        report = run_rules(g)
        b1 = [v for v in report.violations if v.rule_id == "B1"]
        self.assertEqual(len(b1), 1)
        self.assertEqual(b1[0].target, "Metric: orphan_metric")

    def test_b2_formula_dangling_ref(self):
        g = self._graph()
        g["vertices"]["Metric"][0]["formula"] = "SUM(order.amount) + SUM(nonexistent.bad)"
        report = run_rules(g)
        b2 = [v for v in report.violations if v.rule_id == "B2"]
        self.assertEqual(len(b2), 1)
        self.assertIn("nonexistent.bad", b2[0].message)

    def test_b3_parallel_cycle_edge_reported_once(self):
        g = self._graph()
        # two parallel back-edges close the same cycle -> dedup by pair
        g["edges"]["dependsOn"].append(("gmv", "order_cnt"))
        g["edges"]["dependsOn"].append(("gmv", "order_cnt"))
        report = run_rules(g)
        b3 = [v for v in report.violations if v.rule_id == "B3"]
        self.assertEqual(len(b3), 1)

    def test_b3_black_node_skipped(self):
        g = self._graph()
        # acyclic diamond a->b->d, a->c->d: when c reaches d, d is BLACK
        g["edges"]["dependsOn"] = [
            ("gmv", "order_cnt"),
            ("order_cnt", "pay_cnt"),
            ("gmv", "pay_cnt"),
        ]
        report = run_rules(g)
        b3 = [v for v in report.violations if v.rule_id == "B3"]
        self.assertEqual(b3, [])

    def test_b3_dependency_cycle_and_self_loop(self):
        g = self._graph()
        g["edges"]["dependsOn"].append(("gmv", "order_cnt"))  # closes the cycle
        g["edges"]["dependsOn"].append(("gmv", "gmv"))  # self loop
        report = run_rules(g)
        b3 = [v for v in report.violations if v.rule_id == "B3"]
        self.assertEqual(len(b3), 2)

    def test_d1_metric_without_name_skipped(self):
        g = self._graph()
        g["vertices"]["Metric"].append({"definition": "no name"})
        report = run_rules(g)
        self.assertEqual([v for v in report.violations if v.rule_id == "D1"], [])

    def test_d1_metric_conflict(self):
        g = self._graph()
        g["vertices"]["Metric"].append({"name": "gmv", "definition": "different!", "domain": "ops"})
        report = run_rules(g)
        d1 = [v for v in report.violations if v.rule_id == "D1"]
        self.assertEqual(len(d1), 1)
        self.assertEqual(d1[0].details["count"], 2)

    def test_d2_missing_domain(self):
        g = self._graph()
        g["vertices"]["Table"].append({"name": "no_domain_tbl"})
        report = run_rules(g)
        d2 = [v for v in report.violations if v.rule_id == "D2"]
        self.assertEqual(len(d2), 1)
        self.assertEqual(d2[0].level, "warning")


class TestDerivations(unittest.TestCase):
    def test_c1_dependency_closure(self):
        g = _healthy_graph()
        closure = derive_c1_dependency_closure(g["vertices"], g["edges"])
        order_cnt = [c for c in closure if c["metric"] == "order_cnt"]
        self.assertEqual(order_cnt[0]["upstream"], ["gmv"])
        self.assertEqual(order_cnt[0]["depth"], 1)

    def test_c2_unknown_field_no_table(self):
        g = _healthy_graph()
        g["edges"]["computedFromField"].append(("gmv", "ghost.field"))
        derived = derive_c2_bottom_tables(g["vertices"], g["edges"])
        gmv = [d for d in derived if d["metric"] == "gmv"]
        self.assertEqual(gmv[0]["tables"], ["order"])  # ghost.field ignored

    def test_c2_metric_all_fields_unowned_skipped(self):
        g = _healthy_graph()
        g["edges"]["computedFromField"] = [("gmv", "ghost.a"), ("gmv", "ghost.b")]
        derived = derive_c2_bottom_tables(g["vertices"], g["edges"])
        self.assertEqual(derived, [])  # no field maps to a known table

    def test_c2_bottom_tables(self):
        g = _healthy_graph()
        derived = derive_c2_bottom_tables(g["vertices"], g["edges"])
        gmv = [d for d in derived if d["metric"] == "gmv"]
        self.assertEqual(gmv[0]["tables"], ["order"])


class TestRuleIds(unittest.TestCase):
    def test_rule_ids_expected_set(self):
        self.assertEqual(
            set(RULE_IDS),
            {"A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C2", "D1", "D2"},
        )

    def test_edge_endpoints_cover_all_labels(self):
        self.assertEqual(
            set(EDGE_ENDPOINTS.keys()),
            {"hasColumn", "computedFrom", "computedFromField", "dependsOn"},
        )


class TestKgRuleEngine(unittest.TestCase):
    def _mock_client(self, vertex_data, edge_data):
        client = MagicMock()
        gremlin = MagicMock()

        def exec(query):
            if ".hasLabel('" in query and "E()" not in query:
                label = query.split(".hasLabel('")[1].split("'")[0]
                return {"data": vertex_data.get(label, [])}
            if "E()" in query:
                label = query.split(".hasLabel('")[1].split("'")[0]
                return {"data": edge_data.get(label, [])}
            return {"data": []}

        gremlin.exec.side_effect = exec
        client.gremlin.return_value = gremlin
        return client

    def test_load_graph_and_run(self):
        client = self._mock_client(
            {
                "Table": [{"id": "1:order", "label": "Table", "name": "order", "domain": "trade"}],
                "Field": [{"id": "2:order.amount", "label": "Field", "name": "order.amount", "type": "DOUBLE"}],
                "Metric": [{"id": "3:gmv", "label": "Metric", "name": "gmv"}],
            },
            {
                "hasColumn": [
                    {"OUT": {"id": "1:order", "label": "Table"}, "IN": {"id": "2:order.amount", "label": "Field"}}
                ],
                "computedFrom": [],
                "computedFromField": [],
                "dependsOn": [],
            },
        )
        engine = KgRuleEngine(client, graph_name="kg_rag")
        data = engine.load_graph()
        self.assertIn("Table", data["vertices"])
        self.assertEqual(data["vertices"]["Table"][0]["name"], "order")
        report = engine.run(data)
        self.assertIsInstance(report, KgRuleReport)
        # endpoint name normalization: "1:order" -> "order" aligns with vertex name
        self.assertEqual(data["edges"]["hasColumn"], [("order", "order.amount")])
        self.assertEqual(report.by_rule().get("B1"), 1)  # gmv has no source edge

    def test_endpoint_name_dict_without_id_skipped(self):
        client = self._mock_client(
            {},
            {"hasColumn": [{"OUT": {"label": "Table"}, "IN": "raw"}]},
        )
        engine = KgRuleEngine(client)
        data = engine.load_graph()
        self.assertEqual(data["edges"]["hasColumn"], [])  # OUT has no id -> skipped

    def test_endpoint_name_non_numeric_prefix_kept(self):
        client = self._mock_client(
            {},
            {"hasColumn": [{"OUT": "abc:def", "IN": {"id": "2:x", "label": "Field"}}]},
        )
        engine = KgRuleEngine(client)
        data = engine.load_graph()
        self.assertEqual(data["edges"]["hasColumn"], [("abc:def", "x")])

    def test_endpoint_name_plain_string_fallback(self):
        client = self._mock_client(
            {},
            {"hasColumn": [{"OUT": "raw1", "IN": "raw2"}]},
        )
        engine = KgRuleEngine(client)
        data = engine.load_graph()
        self.assertEqual(data["edges"]["hasColumn"], [("raw1", "raw2")])

    def test_fetch_edges_skips_row_without_endpoints(self):
        client = self._mock_client(
            {},
            {"hasColumn": [{"no_out": 1}]},
        )
        engine = KgRuleEngine(client)
        data = engine.load_graph()
        self.assertEqual(data["edges"]["hasColumn"], [])

    def test_run_loads_when_data_missing(self):
        client = self._mock_client({}, {})
        engine = KgRuleEngine(client)
        report = engine.run()
        self.assertEqual(report.violations, [])

    def test_run_with_injected_data(self):
        engine = KgRuleEngine(MagicMock())
        report = engine.run(_healthy_graph())
        self.assertEqual(report.violations, [])


if __name__ == "__main__":
    unittest.main()
