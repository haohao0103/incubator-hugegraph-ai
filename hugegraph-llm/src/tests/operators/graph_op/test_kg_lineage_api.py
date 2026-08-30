"""Tests for KgLineageApi (deterministic lineage traversal over the KG)."""

import unittest
from typing import Any, Dict, List, Tuple

from hugegraph_llm.operators.graph_op.kg_lineage_api import (
    KgLineageApi,
    LineageResult,
    LineageNode,
)


def _sample_graph() -> Dict[str, Any]:
    """order / payment tables, two metrics, a metric dependency, one cycle-free."""
    return {
        "vertices": {
            "Table": [
                {"name": "order"},
                {"name": "payment"},
            ],
            "Field": [
                {"name": "order.amount"},
                {"name": "order.order_id"},
                {"name": "payment.amount"},
                {"name": "payment.order_id"},
            ],
            "Metric": [
                {"name": "order_total", "formula": "SUM(order.amount)", "definition": "订单总额"},
                {"name": "payment_total", "formula": "SUM(payment.amount)"},
                {"name": "gmv", "formula": "order_total + payment_total"},
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.amount"),
                ("order", "order.order_id"),
                ("payment", "payment.amount"),
                ("payment", "payment.order_id"),
            ],
            "computedFromField": [
                ("order_total", "order.amount"),
                ("payment_total", "payment.amount"),
                ("gmv", "payment.amount"),
            ],
            "computedFrom": [
                ("order_total", "order"),
                ("gmv", "payment"),
            ],
            "dependsOn": [
                ("gmv", "order_total"),
            ],
        },
    }


class TestUpstream(unittest.TestCase):
    def setUp(self):
        self.api = KgLineageApi(data=_sample_graph())

    def test_metric_upstream_direct(self):
        up = self.api.upstream("order_total")
        names = set(up.names)
        self.assertEqual(names, {"order.amount", "order"})
        self.assertTrue(any(n.name == "order.amount" and n.via == "computedFromField" for n in up.nodes))
        self.assertTrue(any(n.name == "order" and n.via == "computedFrom" for n in up.nodes))

    def test_metric_upstream_transitive(self):
        # gmv depends on order_total, which is built from order.amount/order
        up = self.api.upstream("gmv")
        names = set(up.names)
        self.assertIn("payment.amount", names)   # computedFromField
        self.assertIn("payment", names)           # computedFrom
        self.assertIn("order_total", names)       # dependsOn
        self.assertIn("order.amount", names)      # transitive via order_total
        self.assertIn("order", names)             # transitive via order_total

    def test_field_upstream_is_owning_table(self):
        up = self.api.upstream("order.amount")
        self.assertEqual([(n.kind, n.name, n.via) for n in up.nodes],
                         [("Table", "order", "hasColumn")])

    def test_table_upstream_empty(self):
        up = self.api.upstream("order")
        self.assertEqual(up.nodes, [])

    def test_unknown_target_empty(self):
        self.assertEqual(self.api.upstream("ghost").nodes, [])
        self.assertEqual(self.api.downstream("ghost").nodes, [])

    def test_max_depth_cutoff(self):
        # chain a->b->c is not in sample; build a dependsOn chain
        g = _sample_graph()
        g["vertices"]["Metric"].append({"name": "x1"})
        g["vertices"]["Metric"].append({"name": "x2"})
        g["edges"]["dependsOn"] += [("x1", "x2")]
        api = KgLineageApi(data=g)
        up = api.upstream("x1", max_depth=0)
        self.assertEqual(up.nodes, [])  # depth 0 -> nothing beyond self


class TestDownstream(unittest.TestCase):
    def setUp(self):
        self.api = KgLineageApi(data=_sample_graph())

    def test_metric_downstream(self):
        down = self.api.downstream("order_total")
        self.assertEqual([n.name for n in down.nodes], ["gmv"])

    def test_table_downstream_includes_metric_and_columns(self):
        down = self.api.downstream("order")
        names = set(down.names)
        self.assertIn("order_total", names)   # computedFrom (rev), then gmv transitively
        self.assertIn("order.amount", names)  # hasColumn (fwd)
        self.assertIn("order.order_id", names)  # hasColumn (fwd)
        self.assertIn("gmv", names)           # transitive via order_total
        self.assertNotIn("payment", names)

    def test_field_downstream_metric(self):
        down = self.api.downstream("order.amount")
        names = {n.name for n in down.nodes}
        # order.amount -> order_total (computedFromField), then transitively
        # gmv (which dependsOn order_total)
        self.assertEqual(names, {"order_total", "gmv"})


class TestCycleSafety(unittest.TestCase):
    def test_self_loop(self):
        g = _sample_graph()
        g["edges"]["dependsOn"].append(("gmv", "gmv"))
        api = KgLineageApi(data=g)
        up = api.upstream("gmv")
        # gmv -> gmv would revisit itself; must not loop infinitely
        self.assertLessEqual(len(up.nodes), 10)

    def test_mutual_cycle(self):
        g = _sample_graph()
        g["vertices"]["Metric"].append({"name": "cyc_a"})
        g["vertices"]["Metric"].append({"name": "cyc_b"})
        g["edges"]["dependsOn"] += [("cyc_a", "cyc_b"), ("cyc_b", "cyc_a")]
        api = KgLineageApi(data=g)
        up = api.upstream("cyc_a")
        self.assertEqual(set(up.names), {"cyc_b"})  # only one hop, no revisit


class TestTablesOfMetric(unittest.TestCase):
    def setUp(self):
        self.api = KgLineageApi(data=_sample_graph())

    def test_direct_tables(self):
        self.assertEqual(self.api.tables_of_metric("order_total"), {"order"})
        self.assertEqual(self.api.tables_of_metric("payment_total"), {"payment"})

    def test_tables_via_field_and_computedfrom(self):
        self.assertEqual(self.api.tables_of_metric("gmv"), {"payment"})

    def test_unknown_metric(self):
        self.assertEqual(self.api.tables_of_metric("ghost"), set())


class TestExplain(unittest.TestCase):
    def setUp(self):
        self.api = KgLineageApi(data=_sample_graph())

    def test_explain_metric(self):
        text = self.api.explain("order_total")
        self.assertIn("order_total", text)
        self.assertIn("上游来源", text)
        self.assertIn("order.amount", text)
        self.assertIn("order", text)
        self.assertIn("下游消费", text)
        self.assertIn("gmv", text)  # gmv depends on order_total

    def test_explain_unknown(self):
        self.assertIn("不存在", self.api.explain("ghost"))


class TestEmptyAndNoClient(unittest.TestCase):
    def test_empty_graph(self):
        api = KgLineageApi(data={"vertices": {}, "edges": {}})
        self.assertEqual(api.upstream("order").nodes, [])
        self.assertEqual(api.downstream("order").nodes, [])
        self.assertEqual(api.tables_of_metric("order"), set())

    def test_no_client_no_data(self):
        api = KgLineageApi()
        self.assertEqual(api.upstream("x").nodes, [])
        self.assertEqual(api.tables_of_metric("x"), set())
        self.assertIn("不存在", api.explain("x"))


class TestDataStructures(unittest.TestCase):
    def test_lineage_result_to_dict(self):
        r = LineageResult(target="t", direction="upstream",
                          nodes=[LineageNode(kind="Table", name="order", via="hasColumn")])
        d = r.to_dict()
        self.assertEqual(d["target"], "t")
        self.assertEqual(d["nodes"][0]["kind"], "Table")
        self.assertEqual(r.names, ["order"])


class TestKeepHelpers(unittest.TestCase):
    """Defensive branches of the direction predicates (unknown edge label)."""

    def test_keep_up_unknown_label(self):
        self.assertFalse(KgLineageApi._keep_up("relatedTo", True))
        self.assertFalse(KgLineageApi._keep_up("relatedTo", False))

    def test_keep_down_unknown_label(self):
        self.assertFalse(KgLineageApi._keep_down("relatedTo", True))
        self.assertFalse(KgLineageApi._keep_down("relatedTo", False))


class TestEmptyTarget(unittest.TestCase):
    def setUp(self):
        self.api = KgLineageApi(data=_sample_graph())

    def test_empty_upstream(self):
        self.assertEqual(self.api.upstream("").nodes, [])

    def test_empty_downstream(self):
        self.assertEqual(self.api.downstream("").nodes, [])


class TestExplainIsolated(unittest.TestCase):
    def test_explain_isolated_table(self):
        g = _sample_graph()
        g["vertices"]["Table"].append({"name": "orphan"})
        api = KgLineageApi(data=g)
        text = api.explain("orphan")
        self.assertIn("orphan", text)
        self.assertIn("上游来源：无", text)
        self.assertIn("下游消费：无", text)


class TestClientLoading(unittest.TestCase):
    """Cover the KgRuleEngine-backed load path (no data provided)."""

    def test_loads_via_client(self):
        class _FakeClient:
            def gremlin(self):
                return self

            def exec(self, q):
                if "hasLabel('Table')" in q:
                    return {"data": [{"id": "1:order", "label": "Table", "name": "order"}]}
                if "hasLabel('Field')" in q:
                    return {"data": [{"id": "1:order.amount", "label": "Field", "name": "order.amount"}]}
                if "hasLabel('hasColumn')" in q:
                    return {"data": [{"id": "e1", "label": "hasColumn",
                                      "OUT": {"id": "1:order"}, "IN": {"id": "1:order.amount"}}]}
                return {"data": []}

        api = KgLineageApi(client=_FakeClient())
        # data was pulled lazily from the client
        self.assertEqual(api.upstream("order").nodes, [])
        self.assertEqual(
            [(n.kind, n.name) for n in api.downstream("order").nodes],
            [("Field", "order.amount")],
        )


if __name__ == "__main__":
    unittest.main()
