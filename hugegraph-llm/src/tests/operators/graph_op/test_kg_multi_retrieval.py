"""Tests for multi-recall + fusion schema retrieval (kg_multi_retrieval)."""

import unittest

from hugegraph_llm.operators.graph_op.kg_multi_retrieval import (
    FulltextRetriever,
    GraphStructureRetriever,
    KgMultiSchemaLinker,
    LexicalRetriever,
    MultiRecallConfig,
    RetrievedVertex,
    SchemaRetriever,
    compute_entity_importance,
    rrf_fuse,
)
from hugegraph_llm.operators.graph_op.kg_rule_engine import GraphData
from hugegraph_llm.operators.graph_op.kg_schema_linker import (
    KgSchemaLinker,
    SchemaContext,
)


def _sample_graph() -> GraphData:
    return {
        "vertices": {
            "Table": [
                {"name": "order", "comment": "订单表"},
                {"name": "payment", "comment": "支付表"},
            ],
            "Field": [
                {"name": "order.amount", "comment": "订单金额"},
                {"name": "order.order_id", "comment": "订单号"},
                {"name": "payment.amount", "comment": "支付金额"},
            ],
            "Metric": [
                {
                    "name": "order_total",
                    "definition": "订单总额",
                    "formula": "SUM(order.amount)",
                },
                {
                    "name": "avg_order_value",
                    # deliberately NO '客单价' substring -> lexical misses it
                    "definition": "平均每单成交金额 = 订单总额 / 订单数",
                    "formula": "SUM(order.amount) / COUNT(DISTINCT order.order_id)",
                },
            ],
        },
        "edges": {
            "hasColumn": [
                ("order", "order.amount"),
                ("order", "order.order_id"),
                ("payment", "payment.amount"),
            ],
            "computedFrom": [("order_total", "order"), ("avg_order_value", "order")],
            "computedFromField": [
                ("order_total", "order.amount"),
                ("avg_order_value", "order.amount"),
            ],
            "dependsOn": [],
        },
    }


class TestGraphStructureRetriever(unittest.TestCase):
    def setUp(self):
        self.data = _sample_graph()
        self.r = GraphStructureRetriever()

    def test_exact_name_hit(self):
        hits = self.r.retrieve("order", self.data)
        self.assertTrue(any(h.label == "Table" and h.name == "order" for h in hits))

    def test_alias_canonical_expansion(self):
        linker = KgSchemaLinker(synonyms={"客单价": "avg_order_value"})
        r = GraphStructureRetriever(linker)
        hits = r.retrieve("客单价是多少", self.data)
        self.assertTrue(
            any(h.label == "Metric" and h.name == "avg_order_value" for h in hits),
            [h.name for h in hits],
        )

    def test_one_hop_subgraph_expansion(self):
        hits = self.r.retrieve("order", self.data)
        fields = {h.name for h in hits if h.label == "Field"}
        self.assertIn("order.amount", fields)
        self.assertIn("order.order_id", fields)

    def test_alias_not_in_question_noop(self):
        linker = KgSchemaLinker(synonyms={"客单价": "avg_order_value"})
        r = GraphStructureRetriever(linker)
        hits = r.retrieve("订单", self.data)  # alias not present -> no expansion
        self.assertFalse(any(h.name == "avg_order_value" for h in hits))

    def test_name_substring_match_direction(self):
        # term contained in a longer name (low in tl branch)
        hits = self.r.retrieve("order.amount", self.data)
        self.assertTrue(any(h.label == "Field" and h.name == "order.amount" for h in hits))

    def test_no_match_returns_empty(self):
        data = dict(_sample_graph())
        data["vertices"] = {
            k: list(v) + [{"comment": "无名顶点"}] for k, v in data["vertices"].items()
        }
        hits = self.r.retrieve("完全不相关的词汇xyz", data)
        self.assertEqual(hits, [])


class TestFulltextRetriever(unittest.TestCase):
    def setUp(self):
        self.data = _sample_graph()

    def test_memory_contains_hit(self):
        r = FulltextRetriever()
        hits = r.retrieve("平均每单成交金额", self.data)
        self.assertTrue(
            any(h.label == "Metric" and h.name == "avg_order_value" for h in hits)
        )

    def test_short_terms_skipped(self):
        # '表在' is a 2-gram; SEARCH matches substrings aggressively, so
        # sub-3-char terms are skipped entirely (no false recall)
        r = FulltextRetriever()
        hits = r.retrieve("表在", self.data)
        self.assertEqual(hits, [])

    def test_overlap_guard_filters_single_char_hits(self):
        # '订单总额' shares only 订/单 with comment '订单表' (2/4 chars):
        # below the overlap threshold -> dropped, exactly the 货拉拉
        # '不存在也答' noise source
        data = {
            "vertices": {
                "Table": [{"name": "order", "comment": "订单表"}],
                "Field": [],
                "Metric": [{"name": "order_total", "definition": "订单总额：求和"}],
            },
            "edges": {
                "hasColumn": [], "computedFrom": [], "computedFromField": [],
                "dependsOn": [],
            },
        }
        r = FulltextRetriever()
        hits = r.retrieve("订单总额", data)
        names = {h.name for h in hits}
        self.assertIn("order_total", names)   # strong overlap kept
        self.assertNotIn("order", names)      # weak overlap (订/单) dropped

    def test_overlap_ratio_math(self):
        self.assertGreater(FulltextRetriever._overlap_ratio("订单总额", "订单总额：求和"), 0.8)
        self.assertLess(FulltextRetriever._overlap_ratio("订单总额", "订单表"), 0.6)
        self.assertEqual(FulltextRetriever._overlap_ratio("订单总额", ""), 0.0)

    def test_intent_weight_boosts_asked_label(self):
        from hugegraph_llm.operators.graph_op.kg_query_understanding import (
            QueryUnderstanding,
            QueryUnderstandingConfig,
        )

        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(intent_weight=1.0),
            query_understanding=QueryUnderstanding(
                config=QueryUnderstandingConfig(short_query_threshold=5),
            ),
        )
        # "在哪个表" -> table intent -> Table order boosted above metrics
        ctx = multi.link("订单金额在哪个表", self.data)
        self.assertEqual(ctx.ranking[0][0], "Table")

    def test_intent_weight_general_question_noop(self):
        # a general-intent question (no boost map) must not crash or reorder
        from hugegraph_llm.operators.graph_op.kg_query_understanding import (
            QueryUnderstanding,
            QueryUnderstandingConfig,
        )

        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(intent_weight=1.0),
            query_understanding=QueryUnderstanding(
                config=QueryUnderstandingConfig(short_query_threshold=5),
            ),
        )
        ctx = multi.link("order 数据", self.data)
        self.assertIn("order", [t.get("name") for t in ctx.tables])

    def test_memory_miss_when_absent(self):
        r = FulltextRetriever()
        hits = r.retrieve("客单价是多少", self.data)  # '客单' not in definitions
        self.assertFalse(any(h.name == "avg_order_value" for h in hits))

    def test_memory_skips_nameless_vertex(self):
        data = dict(_sample_graph())
        data["vertices"]["Table"].append({"comment": "订单表补充说明"})
        r = FulltextRetriever()
        hits = r.retrieve("订单表", data)
        # the nameless vertex's comment matches but yields no candidate
        self.assertTrue(all(h.name for h in hits))

    def test_live_via_client(self):
        class _FakeClient:
            def gremlin(self):
                return self

            def exec(self, q):
                if "Text.contains" in q:
                    # the row carries the matched text so the character-overlap
                    # guard can validate the hit
                    return {"data": [
                        {"name": "avg_order_value", "definition": "平均每单成交金额"},
                        {"name": "order_total", "definition": "订单总额"},
                    ]}
                return {"data": []}

        r = FulltextRetriever()
        # "平均每单成交金额" overlaps avg_order_value.definition 8/8; the
        # order_total row shares only 单/成/金/额 (4/8 < 0.6) -> dropped
        hits = r.retrieve("平均每单成交金额", self.data, client=_FakeClient())
        names = {h.name for h in hits}
        self.assertIn("avg_order_value", names)
        self.assertNotIn("order_total", names)
        self.assertTrue(all(h.source == "fulltext" for h in hits))

    def test_live_client_error_skipped(self):
        class _Broken:
            def gremlin(self):
                return self

            def exec(self, q):
                raise RuntimeError("index down")

        r = FulltextRetriever()
        hits = r.retrieve("订单表在哪里", self.data, client=_Broken())
        self.assertEqual(hits, [])


class TestLexicalRetriever(unittest.TestCase):
    def setUp(self):
        self.data = _sample_graph()

    def test_lexical_hit_on_definition_terms(self):
        r = LexicalRetriever()
        hits = r.retrieve("平均每单成交金额", self.data)
        self.assertTrue(any(h.name == "avg_order_value" for h in hits))

    def test_lexical_miss_on_wording_gap(self):
        r = LexicalRetriever()
        hits = r.retrieve("客单价是多少", self.data)
        self.assertFalse(any(h.name == "avg_order_value" for h in hits))


class TestRrfFuse(unittest.TestCase):
    def test_common_hit_ranks_above_single_path(self):
        a = [
            RetrievedVertex("Metric", "m1", 1.0, "lexical"),
            RetrievedVertex("Metric", "m2", 1.0, "lexical"),
        ]
        b = [
            RetrievedVertex("Metric", "m2", 1.0, "graph"),
            RetrievedVertex("Metric", "m3", 1.0, "graph"),
        ]
        fused = rrf_fuse([a, b], k=60)
        names = [rv.name for rv in fused]
        self.assertEqual(names[0], "m2")  # recalled by both paths
        # double-recall ranks above single-recall within the fused list
        self.assertGreater(fused[0].score, fused[1].score)

    def test_weights_shift_ranking(self):
        a = [RetrievedVertex("Metric", "m1", 1.0, "graph")]
        b = [RetrievedVertex("Metric", "m1", 1.0, "fulltext"),
             RetrievedVertex("Metric", "m2", 1.0, "fulltext")]
        fused = rrf_fuse([a, b], k=10, weights={"graph": 100.0, "fulltext": 1.0})
        self.assertEqual(fused[0].name, "m1")


class _FakeVectorRetriever(SchemaRetriever):
    source = "vector"

    def retrieve(self, question, data, client=None, terms=None):
        if "客单价" in question:
            return [RetrievedVertex("Metric", "avg_order_value", 0.9, self.source)]
        return []


class TestEntityImportance(unittest.TestCase):
    def test_table_importance_by_metric_refs_and_fields(self):
        imp = compute_entity_importance(_sample_graph())
        # order is referenced by 2 metrics and owns 2 fields; payment by 1
        self.assertGreater(imp["Table"]["order"], imp["Table"]["payment"])

    def test_metric_importance_authoritative_priority(self):
        data = _sample_graph()
        data["vertices"]["Metric"][0]["authoritative"] = "true"
        data["vertices"]["Metric"][0]["priority"] = "80"
        imp = compute_entity_importance(data)
        self.assertGreater(imp["Metric"]["order_total"], 0.5)

    def test_field_inherits_table_importance(self):
        data = _sample_graph()
        data["vertices"]["Field"][0]["table"] = "order"
        imp = compute_entity_importance(data)
        self.assertEqual(imp["Field"]["order.amount"], imp["Table"]["order"])

    def test_importance_skips_nameless(self):
        data = _sample_graph()
        data["vertices"]["Table"].append({"comment": "无名表"})
        data["vertices"]["Metric"].append({"definition": "无名指标"})
        imp = compute_entity_importance(data)  # must not crash
        self.assertNotIn(None, imp["Table"])
        self.assertNotIn(None, imp["Metric"])

    def test_importance_rerank_boosts_referenced_table(self):
        a = [RetrievedVertex("Table", "order", 0.02, "lexical")]
        b = [RetrievedVertex("Table", "payment", 0.02, "lexical")]
        fused = rrf_fuse([a, b], k=60)
        imp = {"Table": {"order": 1.0, "payment": 0.0}}
        for rv in fused:
            rv.score *= 1.0 + 1.0 * imp.get(rv.label, {}).get(rv.name, 0.0)
        fused.sort(key=lambda rv: -rv.score)
        self.assertEqual(fused[0].name, "order")


class TestKgMultiSchemaLinker(unittest.TestCase):
    def setUp(self):
        self.data = _sample_graph()

    def test_returns_schema_context(self):
        linker = KgMultiSchemaLinker(config=MultiRecallConfig())
        ctx = linker.link("订单总额是多少", self.data)
        self.assertIsInstance(ctx, SchemaContext)
        self.assertIn("order_total", [m.get("name") for m in ctx.metrics])

    def test_alias_recall_rescues_semantic_equivalent(self):
        # single-pass lexical misses '客单价' (wording gap in definition) ...
        single = KgSchemaLinker().link("客单价是多少", self.data)
        self.assertNotIn("avg_order_value", [m.get("name") for m in single.metrics])
        # ... multi-recall with the alias path rescues it
        multi = KgMultiSchemaLinker(
            synonyms={"客单价": "avg_order_value"},
            config=MultiRecallConfig(),
        )
        ctx = multi.link("客单价是多少", self.data)
        self.assertIn("avg_order_value", [m.get("name") for m in ctx.metrics])

    def test_direct_wording_hits_without_alias(self):
        multi = KgMultiSchemaLinker(config=MultiRecallConfig())
        ctx = multi.link("平均每单成交金额", self.data)
        self.assertIn("avg_order_value", [m.get("name") for m in ctx.metrics])

    def test_vector_slot_attached(self):
        multi = KgMultiSchemaLinker(
            synonyms={"客单价": "avg_order_value"}, config=MultiRecallConfig()
        )
        multi.attach_vector_retriever(_FakeVectorRetriever())
        ctx = multi.link("客单价是多少", self.data)
        self.assertIn("avg_order_value", [m.get("name") for m in ctx.metrics])

    def test_vector_slot_empty_result_ok(self):
        multi = KgMultiSchemaLinker(config=MultiRecallConfig())
        multi.attach_vector_retriever(_FakeVectorRetriever())  # returns [] here
        ctx = multi.link("订单", self.data)
        self.assertIn("order_total", [m.get("name") for m in ctx.metrics])

    def test_terms_override_injected_into_retrievers(self):
        # the query-understanding stage passes expanded terms as ``terms``;
        # every retriever must honour the override instead of re-extracting
        g = GraphStructureRetriever()
        ft = FulltextRetriever()
        lx = LexicalRetriever()
        # each path matches on a different field, but the terms override
        # must drive the recall (not the unrelated question text)
        cases = (
            (g, ["order_total"]),       # name-level entity link
            (ft, ["订单总额"]),          # SEARCH on Metric.definition
            (lx, ["order_total"]),      # lexical ranking over name/comment
        )
        for retriever, terms in cases:
            hits = retriever.retrieve("完全无关的句子", self.data, terms=terms)
            names = {h.name for h in hits}
            self.assertTrue(names, f"{retriever.source}: terms override produced no hits")
        # without the override, the same question extracts nothing relevant
        self.assertEqual(g.retrieve("完全无关的句子", self.data), [])

    def test_query_understanding_integrated(self):
        from hugegraph_llm.operators.graph_op.kg_query_understanding import (
            QueryUnderstanding,
            QueryUnderstandingConfig,
        )
        from hugegraph_llm.operators.graph_op.kg_term_graph import KgTermGraph

        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(),
            query_understanding=QueryUnderstanding(
                term_graph=KgTermGraph.from_jargon_map({"大单": "order_total"}),
                config=QueryUnderstandingConfig(short_query_threshold=5),
            ),
        )
        ctx = multi.link("大单是多少", data=self.data)
        # synonym expansion via the understanding stage -> canonical term
        # injected into every recall path
        self.assertIn("order_total", [m.get("name") for m in ctx.metrics])
        # the intent trace is attached to the context
        self.assertIn("expanded_terms", ctx.query_intent)
        self.assertIn("order_total", ctx.query_intent["expanded_terms"])
        self.assertEqual(ctx.query_intent["extraction_method"], "heuristic")

    def test_field_owner_table_budget_break(self):
        # 6 tables each with a hit field -> owner-table promotion stops at the
        # max_tables budget (the break branch)
        from hugegraph_llm.operators.graph_op.kg_query_understanding import (
            QueryUnderstanding,
            QueryUnderstandingConfig,
        )

        tables = [{"name": f"t{i}", "comment": f"表{i}"} for i in range(6)]
        fields = [{"name": f"t{i}.f", "comment": f"字段{i}"} for i in range(6)]
        data = {
            "vertices": {"Table": tables, "Field": fields, "Metric": []},
            "edges": {
                "hasColumn": [(f"t{i}", f"t{i}.f") for i in range(6)],
                "computedFrom": [], "computedFromField": [], "dependsOn": [],
            },
        }
        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(max_tables=5),
            query_understanding=QueryUnderstanding(
                config=QueryUnderstandingConfig(short_query_threshold=100),
            ),
        )
        ctx = multi.link("t0.f t1.f t2.f t3.f t4.f t5.f", data=data)
        # owner tables promoted for the recalled fields, capped at the budget
        self.assertLessEqual(len(ctx.tables), 5)

    def test_intent_weight_without_understanding_noop(self):
        # intent_weight > 0 but no query-understanding stage -> intent dict is
        # empty, the boost branch is skipped without crashing
        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(intent_weight=1.0),
            retrievers=[LexicalRetriever()],
        )
        ctx = multi.link("订单总额", self.data)
        self.assertIn("order_total", [m.get("name") for m in ctx.metrics])

    def test_ranking_promote_owner_via_table_attr_only(self):
        # the field carries `table` but there is no hasColumn edge for it, so
        # field_owner misses it and the ranking-insertion loop runs to the end
        # (no matching field found) without crashing
        data = {
            "vertices": {
                "Table": [{"name": "order", "comment": "订单表"}],
                "Field": [{"name": "order.amount", "comment": "订单金额", "table": "order"}],
                "Metric": [],
            },
            "edges": {
                "hasColumn": [], "computedFrom": [], "computedFromField": [],
                "dependsOn": [],
            },
        }
        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(),
            retrievers=[FulltextRetriever(), LexicalRetriever()],
        )
        ctx = multi.link("订单金额", data)
        self.assertIn("order", [t.get("name") for t in ctx.tables])

    def test_context_empty_property(self):
        multi = KgMultiSchemaLinker(config=MultiRecallConfig())
        ctx = multi.link("完全不相关的词汇xyz", self.data)
        self.assertTrue(ctx.empty)

    def test_lexical_skips_vertices_without_name(self):
        # a vertex without a name must be skipped by the lexical path (and by
        # the fused assembly) without crashing
        data = {
            "vertices": {
                "Table": [{"name": "order", "comment": "订单表"},
                          {"comment": "无名字表"}],
                "Field": [], "Metric": [],
            },
            "edges": {
                "hasColumn": [], "computedFrom": [], "computedFromField": [],
                "dependsOn": [],
            },
        }
        multi = KgMultiSchemaLinker(
            retrievers=[LexicalRetriever()], config=MultiRecallConfig()
        )
        ctx = multi.link("订单", data)
        self.assertIn("order", [t.get("name") for t in ctx.tables])

    def test_rrf_fusion_mode_kept(self):
        # MultiRecallConfig.fusion="rrf" falls back to reciprocal-rank fusion
        multi = KgMultiSchemaLinker(
            config=MultiRecallConfig(fusion="rrf"), retrievers=[LexicalRetriever()]
        )
        ctx = multi.link("订单总额", self.data)
        self.assertIn("order_total", [m.get("name") for m in ctx.metrics])

    def test_context_not_empty_when_linked(self):
        multi = KgMultiSchemaLinker(config=MultiRecallConfig())
        ctx = multi.link("订单总额", self.data)
        self.assertFalse(ctx.empty)

    def test_importance_weight_reranks_tables(self):
        data = _sample_graph()
        # heavily reference 'order' via extra metrics so importance dominates
        for i in range(3):
            data["vertices"]["Metric"].append(
                {"name": f"m_extra_{i}", "definition": "口径", "formula": "COUNT(1)",
                 "source_tables": ["order"]}
            )
            data["edges"]["computedFrom"].append((f"m_extra_{i}", "order"))
        multi = KgMultiSchemaLinker(config=MultiRecallConfig(importance_weight=1.0))
        ctx = multi.link("order payment", data)
        names = [t.get("name") for t in ctx.tables]
        self.assertIn("order", names)
        # 'order' outranks 'payment' after importance re-ranking
        self.assertEqual(names[0], "order")

    def test_no_evidence_empty_context(self):
        multi = KgMultiSchemaLinker(config=MultiRecallConfig())
        ctx = multi.link("风控引擎实时决策表在哪里", self.data)
        self.assertEqual(ctx.tables, [])
        self.assertEqual(ctx.metrics, [])
        self.assertEqual(ctx.fields, [])

    def test_evidence_and_relations_present(self):
        multi = KgMultiSchemaLinker(config=MultiRecallConfig())
        ctx = multi.link("订单总额", self.data)
        self.assertTrue(any("SUM(order.amount)" in e for e in ctx.evidence))
        self.assertTrue(any("computedFrom" in r for r in ctx.relations))

    def test_base_fields_appended_when_no_graph_path(self):
        # no graph retriever attached (only fulltext+lexical): the field is
        # collected by the base linker via the hasColumn edge, so the append
        # branch must add it to the context
        data = {
            "vertices": {
                "Table": [{"name": "order", "comment": "订单表"}],
                "Field": [{"name": "order.amount"}],
                "Metric": [],
            },
            "edges": {
                "hasColumn": [("order", "order.amount")],
                "computedFrom": [], "computedFromField": [], "dependsOn": [],
            },
        }
        multi = KgMultiSchemaLinker(
            retrievers=[FulltextRetriever(), LexicalRetriever()],
            config=MultiRecallConfig(),
        )
        ctx = multi.link("订单表", data)
        self.assertIn("order.amount", [f.get("name") for f in ctx.fields])

    def test_field_budget_break(self):
        # hit fields already exceed the budget -> base-fields loop breaks early
        data = {
            "vertices": {
                "Table": [{"name": "order", "comment": "订单表"}],
                "Field": [{"name": f"order.f{i}"} for i in range(5)],
                "Metric": [],
            },
            "edges": {
                "hasColumn": [(("order"), f"order.f{i}") for i in range(5)],
                "computedFrom": [], "computedFromField": [], "dependsOn": [],
            },
        }
        multi = KgMultiSchemaLinker(config=MultiRecallConfig(max_fields_per_table=3))
        ctx = multi.link("order", data)
        self.assertEqual(len(ctx.fields), 3)


if __name__ == "__main__":
    unittest.main()
