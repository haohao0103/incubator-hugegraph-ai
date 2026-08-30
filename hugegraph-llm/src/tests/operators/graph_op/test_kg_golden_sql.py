"""Tests for KgGoldenSqlStore (NL2SQL golden-SQL feedback loop, P0-4)."""

import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.operators.graph_op.kg_golden_sql import (
    KgGoldenSqlStore,
    GoldenRecord,
    score_golden,
    _schema_refs_of,
    _quote,
)


class _FakeClient:
    """In-memory stand-in for the HugeGraph client used by the store."""

    def __init__(self, tables=None, fields=None, queries=None):
        self.tables = set(tables or [])
        self.fields = set(fields or [])
        self.queries = list(queries or [])
        self.calls = []
        self.edges = []  # only successfully created reference edges
        self.fail_next = False

    def gremlin(self):
        return self

    def exec(self, q):
        self.calls.append(q)
        if self.fail_next:
            raise RuntimeError("boom")
        if "addV('Query')" in q:
            vid = f"1:q{len(self.queries) + 1}"
            self.queries.append({"id": vid})
            return {"data": [{"id": vid, "label": "Query"}]}
        if "hasLabel('Query')" in q and "elementMap" in q:
            return {"data": [
                {
                    "id": f"1:q{i + 1}",
                    "label": "Query",
                    "question": qq.get("question", ""),
                    "sql": qq.get("sql", ""),
                    "schema_refs": qq.get("schema_refs", ""),
                }
                for i, qq in enumerate(self.queries)
            ]}
        # references edge: only succeed when the target vertex exists
        if "addE('references')" in q:
            if "has('Table','name'" in q:
                name = q.split("has('Table','name',")[1].split("'")[1]
                if name not in self.tables:
                    raise RuntimeError(f"no Table {name}")
            if "has('Field','name'" in q:
                name = q.split("has('Field','name',")[1].split("'")[1]
                if name not in self.fields:
                    raise RuntimeError(f"no Field {name}")
            self.edges.append(q)
            return {"data": [{"id": "e1", "label": "references"}]}
        return {"data": []}


class TestSchemaRefs(unittest.TestCase):
    def test_extract_tables_cols(self):
        tables, cols, refs = _schema_refs_of(
            "SELECT order.amount FROM order JOIN payment ON order.order_id=payment.order_id"
        )
        self.assertEqual(tables, ["order", "payment"])
        self.assertIn("order.amount", cols)
        self.assertIn("payment.order_id", cols)
        self.assertIn("order", refs)
        self.assertIn("order.amount", refs)

    def test_bare_column_recorded_by_basename(self):
        tables, cols, refs = _schema_refs_of("SELECT order_id FROM order")
        self.assertEqual(tables, ["order"])
        self.assertEqual(cols, [])
        self.assertIn("order_id", refs)  # bare column kept as basename


class TestScoreGoldenExact(unittest.TestCase):
    def test_exact_schema_match(self):
        rec = GoldenRecord(question="x", sql="", schema_refs={"order", "order.amount"})
        # term 'order' is exactly in schema_refs -> exact-match branch (line 81)
        self.assertGreater(score_golden({"order"}, rec), 0)


class TestScoreGolden(unittest.TestCase):
    def _rec(self, question, refs):
        return GoldenRecord(question=question, sql="", schema_refs=set(refs))

    def test_relevant_ranks_higher(self):
        a = self._rec("订单金额是多少", {"order", "order.amount"})
        b = self._rec("用户数量统计", {"user", "user.user_id"})
        new_terms = set(["订单", "金额"])
        self.assertGreater(score_golden(new_terms, a), score_golden(new_terms, b))

    def test_no_overlap_returns_zero(self):
        a = self._rec("订单金额是多少", {"order", "order.amount"})
        self.assertEqual(score_golden(set(["天气"]), a), 0)

    def test_schema_basename_match(self):
        a = self._rec("x", {"order.amount"})
        # term 'amount' matches ref 'order.amount' by basename suffix
        self.assertGreater(score_golden(set(["amount"]), a), 0)

    def test_linked_names_dominate(self):
        # linked table 'payment' should rank the payment golden above an
        # otherwise lexically-tied order golden.
        order = self._rec("订单金额是多少", {"order", "order.amount"})
        pay = self._rec("支付金额合计", {"payment", "payment.amount"})
        terms = set(["金额"])
        s_order = score_golden(terms, order, linked_names={"payment"})
        s_pay = score_golden(terms, pay, linked_names={"payment"})
        self.assertGreater(s_pay, s_order)

    def test_linked_name_suffix_only_match(self):
        # 'amount' is not a direct ref but matches 'payment.amount' by suffix
        rec = self._rec("x", {"payment.amount"})
        self.assertGreater(score_golden(set(), rec, linked_names={"amount"}), 0)


class TestAdd(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient(
            tables={"order", "payment"},
            fields={"order.amount", "payment.amount"},
        )
        self.store = KgGoldenSqlStore(self.client)

    def test_add_creates_vertex_and_edges(self):
        vid = self.store.add(
            "订单金额是多少", "SELECT SUM(order.amount) FROM order"
        )
        self.assertIsNotNone(vid)
        # vertex created
        self.assertTrue(any("addV('Query')" in c for c in self.client.calls))
        # reference edge to existing Table 'order' and Field 'order.amount'
        self.assertTrue(
            any("has('Table','name','order')" in c for c in self.client.edges)
        )
        self.assertTrue(
            any("has('Field','name','order.amount')" in c for c in self.client.edges)
        )

    def test_add_skips_nonexistent_reference(self):
        # 'ghost' table/field do not exist -> edge attempt must not crash add(),
        # and no edge for ghost is created (the failure is swallowed).
        vid = self.store.add("x", "SELECT ghost.col FROM ghost")
        self.assertIsNotNone(vid)
        self.assertFalse(
            any("has('Table','name','ghost')" in c for c in self.client.edges)
        )

    def test_add_returns_none_on_write_failure(self):
        self.client.fail_next = True
        self.assertIsNone(
            self.store.add("q", "SELECT 1 FROM order")
        )

    def test_add_returns_none_when_vertex_not_created(self):
        # addV returns empty data -> first vertex id is None -> add returns None
        class _EmptyAdd(_FakeClient):
            def exec(self, q):
                if "addV('Query')" in q:
                    return {"data": []}
                return super().exec(q)

        store = KgGoldenSqlStore(_EmptyAdd(tables={"order"}, fields={"order.amount"}))
        self.assertIsNone(store.add("q", "SELECT 1 FROM order"))

    def test_add_no_verify_skips_edges(self):
        vid = self.store.add("q", "SELECT order.amount FROM order", verify=False)
        self.assertIsNotNone(vid)
        # no reference edges attempted when verify=False
        self.assertEqual(self.client.edges, [])


class TestVertexIdHelpers(unittest.TestCase):
    def test_vertex_id_none(self):
        self.assertIsNone(KgGoldenSqlStore._vertex_id({"id": None}))

    def test_vertex_id_plain(self):
        self.assertEqual(KgGoldenSqlStore._vertex_id({"id": "plain"}), "plain")

    def test_vertex_id_nested_strips_prefix(self):
        self.assertEqual(
            KgGoldenSqlStore._vertex_id({"id": {"id": "1:q5"}}), "q5"
        )

    def test_first_vertex_id_non_dict(self):
        self.assertIsNone(KgGoldenSqlStore._first_vertex_id("nope"))

    def test_first_vertex_id_empty_data(self):
        self.assertIsNone(KgGoldenSqlStore._first_vertex_id({"data": []}))


class TestGetSimilar(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient(
            tables={"order", "payment", "user"},
            fields={"order.amount", "user.user_id"},
            queries=[
                {"question": "订单金额是多少",
                 "sql": "SELECT SUM(order.amount) FROM order",
                 "schema_refs": "order;order.amount"},
                {"question": "用户数量统计",
                 "sql": "SELECT COUNT(user.user_id) FROM user",
                 "schema_refs": "user;user.user_id"},
            ],
        )
        self.store = KgGoldenSqlStore(self.client)

    def test_relevant_golden_first(self):
        res = self.store.get_similar("订单的总金额", top_k=3)
        self.assertTrue(res)
        self.assertIn("SUM(order.amount)", res[0].sql)

    def test_unrelated_returns_empty(self):
        res = self.store.get_similar("今天天气怎么样")
        self.assertEqual(res, [])

    def test_empty_graph_returns_empty(self):
        store = KgGoldenSqlStore(_FakeClient())
        self.assertEqual(store.get_similar("订单金额"), [])

    def test_fewshot_format(self):
        rec = GoldenRecord(question="q", sql="SELECT 1")
        self.assertIn("-- Q: q", rec.to_prompt_fewshot())
        self.assertIn("-- A: SELECT 1", rec.to_prompt_fewshot())


class TestQuote(unittest.TestCase):
    def test_quote_escapes(self):
        self.assertEqual(_quote("it's"), "'it\\'s'")
        self.assertEqual(_quote("a\\b"), "'a\\\\b'")


class _StubLinker:
    def __init__(self, client):
        self.client = client

    def load_graph(self):
        return {}

    def link(self, question, data):
        class _CtxMissingName:
            tables = [{"name": None}]
            fields = [{"name": None}]

        class _Ctx:
            tables = [{"name": "payment"}]
            fields = [{"name": "payment.amount"}]

        return _CtxMissingName() if question == "missing" else _Ctx()


class TestLinkedNames(unittest.TestCase):
    def test_linked_names_populated(self):
        import hugegraph_llm.operators.graph_op.kg_golden_sql as mod

        orig = mod.KgSchemaLinker
        mod.KgSchemaLinker = _StubLinker
        try:
            store = KgGoldenSqlStore(_FakeClient())
            names = store._linked_names("支付金额")
            self.assertEqual(names, {"payment", "payment.amount", "amount"})
            # missing-name vertices must be skipped without error
            self.assertEqual(store._linked_names("missing"), set())
        finally:
            mod.KgSchemaLinker = orig

    def test_linked_names_falls_back_on_error(self):
        class _Broken(_FakeClient):
            def exec(self, q):
                if "hasLabel('Table')" in q or "elementMap" in q:
                    raise RuntimeError("graph down")
                return super().exec(q)

        store = KgGoldenSqlStore(_Broken())
        self.assertEqual(store._linked_names("x"), set())


class TestEnsureSchema(unittest.TestCase):
    """KgGoldenSqlStore.ensure_schema guarantees the Query label + indexes."""

    def test_success_then_cached(self):
        store = KgGoldenSqlStore(_FakeClient(), "kg_rag")
        fake_mgr = MagicMock()
        with patch(
            "hugegraph_llm.operators.hugegraph_op.schema_manager.SchemaManager",
            return_value=fake_mgr,
        ):
            self.assertTrue(store.ensure_schema())
            self.assertTrue(store.ensure_schema())
        # second call served by the cached flag, not a second schema round-trip
        self.assertEqual(fake_mgr.ensure_schema.call_count, 1)

    def test_add_passes_query_indexed_schema(self):
        store = KgGoldenSqlStore(_FakeClient(tables=["order"], fields=["order.amount"]), "kg_rag")
        fake_mgr = MagicMock()
        with patch(
            "hugegraph_llm.operators.hugegraph_op.schema_manager.SchemaManager",
            return_value=fake_mgr,
        ):
            vid = store.add("订单总额", "SELECT SUM(order.amount) FROM order")
        self.assertIsNotNone(vid)
        called_schema = fake_mgr.ensure_schema.call_args[0][0]
        index_names = {i["name"] for i in called_schema["indexes"]}
        self.assertIn("QueryByDomain", index_names)
        self.assertIn("QueryByQuestion", index_names)
        # the Query vertex label is declared AUTOMATIC-id (property filters
        # on it are rejected unless secondary indexes exist)
        vl = next(v for v in called_schema["vertexlabels"] if v["name"] == "Query")
        self.assertEqual(vl["id_strategy"], "AUTOMATIC")

    def test_failure_guarded_and_add_still_proceeds(self):
        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("schema api down")

        store = KgGoldenSqlStore(
            _FakeClient(tables=["order"], fields=["order.amount"]), "kg_rag"
        )
        with patch(
            "hugegraph_llm.operators.hugegraph_op.schema_manager.SchemaManager", _Boom
        ):
            self.assertFalse(store.ensure_schema())
            # the write still works when schema self-healing is unavailable
            vid = store.add("订单总额", "SELECT SUM(order.amount) FROM order")
        self.assertIsNotNone(vid)


if __name__ == "__main__":
    unittest.main()
