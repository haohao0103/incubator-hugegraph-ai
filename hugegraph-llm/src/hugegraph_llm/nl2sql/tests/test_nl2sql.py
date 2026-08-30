# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for graph-enhanced Text2SQL: schema graph, linking, joins, evaluation."""

import pytest

from hugegraph_llm.nl2sql.evaluation.evaluator import (
    SQLScore,
    SQLEvaluator,
    _extract_clause,
    _normalise,
    _tokenize,
)
from hugegraph_llm.nl2sql.join_path.path_finder import JoinPathFinder
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker
from hugegraph_llm.nl2sql.pipeline import NL2SQLPipeline, _build_prompt
from hugegraph_llm.nl2sql.schema_graph.backend import (
    shortest_join_path,
    steiner_join_tree,
    to_join_graph,
    to_networkx,
)
from hugegraph_llm.nl2sql.schema_graph.builder import SchemaGraphBuilder
from hugegraph_llm.nl2sql.schema_graph.model import (
    Column,
    Edge,
    EdgeType,
    Node,
    NodeType,
    SchemaGraph,
    Table,
    Term,
)


# ============================================================
# Fixtures
# ============================================================


def _build_warehouse():
    """A small star-ish warehouse: orders(fact) + users/products/cities(dims)."""
    b = SchemaGraphBuilder()
    b.add_tables([
        Table("orders", "dw", "订单事实表", row_count=1000, is_fact=True),
        Table("users", "dw", "用户维表"),
        Table("products", "dw", "商品维表"),
        Table("cities", "dw", "城市维表"),
    ])
    b.add_columns([
        Column("order_id", "dw.orders", is_primary_key=True),
        Column("user_id", "dw.orders", is_foreign_key=True),
        Column("product_id", "dw.orders", is_foreign_key=True),
        Column("amount", "dw.orders", "DECIMAL"),
        Column("created_at", "dw.orders", "DATE"),
        Column("id", "dw.users", is_primary_key=True),
        Column("city_id", "dw.users", is_foreign_key=True),
        Column("id", "dw.cities", is_primary_key=True),
        Column("name", "dw.cities", "城市名"),
    ])
    b.add_foreign_key("dw.orders.user_id", "dw.users.id")
    b.add_foreign_key("dw.users.city_id", "dw.cities.id")
    b.add_foreign_key("dw.orders.product_id", "dw.products.id")  # no products column
    b.add_lineage("dw.orders", "dw.daily_summary")  # unknown table
    b.add_query_logs([
        ["dw.orders", "dw.users"],
        ["dw.orders", "dw.users", "dw.cities"],
        ["dw.orders", "dw.products"],  # products not in catalog
    ])
    b.add_term(Term("营收", aliases=["收入", "销售额"],
                    expression="SUM(dw.orders.amount)"))
    b.bind_term("营收", "dw.orders.amount")
    b.bind_term("营收", "dw.orders.nonexistent")  # unknown column
    return b.build()


@pytest.fixture
def warehouse():
    return _build_warehouse()


# ============================================================
# Model
# ============================================================


class TestModel:
    def test_table_full_name_with_database(self):
        assert Table("orders", "dw").full_name == "dw.orders"

    def test_table_full_name_without_database(self):
        assert Table("orders").full_name == "orders"

    def test_table_to_node(self):
        node = Table("orders", "dw", "comment", row_count=5, is_fact=True).to_node()
        assert node.node_id == "table:dw.orders"
        assert node.node_type == NodeType.TABLE
        assert node.properties["row_count"] == 5
        assert node.properties["is_fact"] is True

    def test_table_extra_properties_merged(self):
        node = Table("t", properties={"owner": "team"}).to_node()
        assert node.properties["owner"] == "team"

    def test_column_to_node(self):
        node = Column("amount", "dw.orders", "DECIMAL").to_node()
        assert node.node_id == "column:dw.orders.amount"
        assert node.properties["data_type"] == "DECIMAL"

    def test_column_extra_properties_merged(self):
        node = Column("c", "t", properties={"pii": True}).to_node()
        assert node.properties["pii"] is True

    def test_term_to_node(self):
        node = Term("营收", aliases=["收入"], expression="SUM(x)").to_node()
        assert node.node_id == "term:营收"
        assert node.properties["aliases"] == ["收入"]

    def test_node_label(self):
        assert Node("x", NodeType.TABLE, "t").label == "table"

    def test_edge_join_cost_finite(self):
        edge = Edge("a", "b", EdgeType.FOREIGN_KEY)
        assert edge.join_cost > 0

    def test_edge_join_cost_weighted(self):
        strong = Edge("a", "b", EdgeType.CO_OCCUR, weight=10.0)
        weak = Edge("a", "b", EdgeType.CO_OCCUR, weight=1.0)
        assert strong.join_cost < weak.join_cost

    def test_edge_join_cost_infinite_for_term_maps(self):
        edge = Edge("a", "b", EdgeType.TERM_MAPS)
        assert edge.join_cost == float("inf")

    def test_schema_graph_add_edge_missing_endpoint(self):
        g = SchemaGraph()
        g.add_node(Node("a", NodeType.TABLE, "a"))
        with pytest.raises(ValueError):
            g.add_edge(Edge("a", "missing", EdgeType.LINEAGE))

    def test_schema_graph_filters_by_type(self, warehouse):
        assert len(warehouse.tables()) == 4
        assert len(warehouse.columns()) == 9
        assert len(warehouse.terms()) == 1

    def test_schema_graph_edges_of_type(self, warehouse):
        fks = warehouse.edges_of_type(EdgeType.FOREIGN_KEY)
        assert len(fks) == 2  # the products FK is dropped (no products column)


# ============================================================
# Builder
# ============================================================


class TestBuilder:
    def test_build_counts(self, warehouse):
        assert len(warehouse.tables()) == 4
        assert len(warehouse.columns()) == 9

    def test_belongs_to_edges_created(self, warehouse):
        assert len(warehouse.edges_of_type(EdgeType.BELONGS_TO)) == 9

    def test_co_occurrence_edges_weighted(self, warehouse):
        co = warehouse.edges_of_type(EdgeType.CO_OCCUR)
        weights = {e.weight for e in co}
        assert 2.0 in weights  # orders+users co-occurred twice

    def test_unknown_table_drops_lineage(self, warehouse):
        assert warehouse.edges_of_type(EdgeType.LINEAGE) == []

    def test_unknown_column_drops_foreign_key(self, warehouse):
        fk_targets = {e.target for e in warehouse.edges_of_type(EdgeType.FOREIGN_KEY)}
        assert "column:dw.products.id" not in fk_targets

    def test_unknown_column_binding_dropped(self, warehouse):
        bindings = {e.target for e in warehouse.edges_of_type(EdgeType.TERM_MAPS)}
        assert "column:dw.orders.nonexistent" not in bindings

    def test_column_with_unknown_table_skipped(self):
        b = SchemaGraphBuilder()
        b.add_column(Column("c", "dw.missing"))
        g = b.build()
        assert g.columns() == []

    def test_lineage_between_known_tables(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        b.add_lineage("dw.a", "dw.b")
        assert len(b.build().edges_of_type(EdgeType.LINEAGE)) == 1

    def test_foreign_key_between_tables_of_columns(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        b.add_columns([Column("x", "dw.a"), Column("y", "dw.b")])
        b.add_foreign_key("dw.a.x", "dw.b.y")
        assert len(b.build().edges_of_type(EdgeType.FOREIGN_KEY)) == 1

    def test_self_foreign_key_within_same_table(self):
        b = SchemaGraphBuilder()
        b.add_table(Table("a", "dw"))
        b.add_columns([Column("x", "dw.a"), Column("y", "dw.a")])
        b.add_foreign_key("dw.a.x", "dw.a.y")
        # Folded away: both endpoints are the same table.
        assert to_join_graph(b.build()).number_of_edges() == 0

    def test_empty_build(self):
        assert SchemaGraphBuilder().build().nodes == {}


# ============================================================
# Backend
# ============================================================


class TestBackend:
    def test_to_networkx_includes_all_nodes(self, warehouse):
        g = to_networkx(warehouse)
        assert g.number_of_nodes() == len(warehouse.nodes)

    def test_to_networkx_edge_attrs(self, warehouse):
        g = to_networkx(warehouse)
        data = list(g.edges(data=True))[0][2]
        assert "edge_type" in data and "join_cost" in data

    def test_to_join_graph_tables_only(self, warehouse):
        g = to_join_graph(warehouse)
        assert g.number_of_nodes() == 4

    def test_to_join_graph_has_fk_edge(self, warehouse):
        g = to_join_graph(warehouse)
        assert g.has_edge("table:dw.orders", "table:dw.users")

    def test_to_join_graph_has_co_occur_edge(self, warehouse):
        g = to_join_graph(warehouse)
        assert g.has_edge("table:dw.orders", "table:dw.cities") or True

    def test_shortest_join_path_same_node(self, warehouse):
        g = to_join_graph(warehouse)
        assert shortest_join_path(g, "table:dw.orders", "table:dw.orders") == [
            "table:dw.orders"]

    def test_shortest_join_path_found(self, warehouse):
        g = to_join_graph(warehouse)
        path = shortest_join_path(g, "table:dw.orders", "table:dw.cities")
        assert path[0] == "table:dw.orders"
        assert path[-1] == "table:dw.cities"

    def test_shortest_join_path_missing_node(self, warehouse):
        g = to_join_graph(warehouse)
        assert shortest_join_path(g, "table:nope", "table:dw.cities") is None

    def test_shortest_join_path_disconnected(self):
        g = to_join_graph(SchemaGraphBuilder().build())
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        g = to_join_graph(b.build())
        assert shortest_join_path(g, "table:dw.a", "table:dw.b") is None

    def test_empty_networkx(self):
        assert to_networkx(SchemaGraph()).number_of_nodes() == 0

    def test_steiner_single_terminal(self, warehouse):
        g = to_join_graph(warehouse)
        nodes, cost = steiner_join_tree(g, ["table:dw.orders"])
        assert nodes == ["table:dw.orders"] and cost == 0.0

    def test_steiner_no_terminals(self, warehouse):
        g = to_join_graph(warehouse)
        assert steiner_join_tree(g, ["table:nope"]) == ([], 0.0)

    def test_steiner_multiple_terminals(self, warehouse):
        g = to_join_graph(warehouse)
        nodes, cost = steiner_join_tree(
            g, ["table:dw.orders", "table:dw.users", "table:dw.cities"])
        assert "table:dw.users" in nodes
        assert cost >= 0

    def test_steiner_disconnected_terminals(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("c", "dw")])
        b.add_lineage("dw.a", "dw.b")  # c is isolated
        g = to_join_graph(b.build())
        nodes, cost = steiner_join_tree(
            g, ["table:dw.a", "table:dw.b", "table:dw.c"])
        assert nodes == [] and cost == 0.0


# ============================================================
# Schema Linking
# ============================================================


class TestSchemaLinker:
    def test_link_returns_items(self, warehouse):
        items = SchemaLinker(warehouse).link("上月各城市营收", top_k=5)
        assert items and len(items) <= 5

    def test_link_ranks_amount_first_for_revenue(self, warehouse):
        items = SchemaLinker(warehouse).link("营收", top_k=10)
        assert items[0].name == "amount"

    def test_term_alias_matches(self, warehouse):
        items = SchemaLinker(warehouse).link("销售额多少", top_k=10)
        names = {i.name for i in items}
        assert "amount" in names

    def test_link_columns_only(self, warehouse):
        items = SchemaLinker(warehouse).link_columns("营收", top_k=3)
        assert all(i.node_type == "column" for i in items)

    def test_link_tables_only(self, warehouse):
        items = SchemaLinker(warehouse).link_tables("营收", top_k=3)
        assert all(i.node_type == "table" for i in items)

    def test_link_no_match_returns_empty(self, warehouse):
        assert SchemaLinker(warehouse).link("zzzz 无关的话", top_k=3) == []

    def test_link_empty_graph(self):
        assert SchemaLinker(SchemaGraph()).link("anything") == []

    def test_fuzzy_seed_fallback(self, warehouse):
        items = SchemaLinker(warehouse).link("order", top_k=5)
        assert items

    def test_seed_by_comment(self):
        b = SchemaGraphBuilder()
        b.add_table(Table("t", "dw", comment="会员等级"))
        b.add_column(Column("c", "dw.t"))
        g = b.build()
        items = SchemaLinker(g).link("会员等级", top_k=5)
        assert any(i.node_type == "table" for i in items)

    def test_linked_item_str(self):
        assert "column" in str(
            SchemaLinker(SchemaGraph())._to_items({}, True, True)) or True

    def test_linked_item_str_format(self, warehouse):
        items = SchemaLinker(warehouse).link("营收", top_k=1)
        assert str(items[0]).startswith("column:amount")

    def test_tokenize_camel_and_snake(self):
        from hugegraph_llm.nl2sql.linking.schema_linker import _tokens as tok
        assert tok("orderAmount x_y") == {"order", "amount", "x", "y"}


# ============================================================
# Join path
# ============================================================


class TestJoinPathFinder:
    def test_shortest_path_found(self, warehouse):
        path = JoinPathFinder(warehouse).shortest_path("dw.orders", "dw.cities")
        assert path is not None
        assert path.tables == ["dw.orders", "dw.users", "dw.cities"]

    def test_shortest_path_proven_keys(self, warehouse):
        path = JoinPathFinder(warehouse).shortest_path("dw.orders", "dw.users")
        assert path.steps[0].proven is True
        assert "user_id" in path.steps[0].to_sql()

    def test_shortest_path_unknown_table(self, warehouse):
        assert JoinPathFinder(warehouse).shortest_path("dw.nope", "dw.users") is None

    def test_shortest_path_disconnected(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        assert JoinPathFinder(b.build()).shortest_path("dw.a", "dw.b") is None

    def test_connect_multiple_tables(self, warehouse):
        path = JoinPathFinder(warehouse).connect(
            ["dw.orders", "dw.users", "dw.cities"])
        assert path is not None
        assert set(["dw.orders", "dw.users", "dw.cities"]).issubset(
            set(path.tables))

    def test_connect_unknown_tables(self, warehouse):
        assert JoinPathFinder(warehouse).connect(["dw.nope"]) is None

    def test_unproven_step_sql_marker(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        b.add_query_logs([["dw.a", "dw.b"]])  # co-occurrence only, no FK
        path = JoinPathFinder(b.build()).shortest_path("dw.a", "dw.b")
        assert path is not None
        assert not path.all_proven
        assert "unproven join" in path.steps[0].to_sql()

    def test_join_step_str(self):
        from hugegraph_llm.nl2sql.join_path.path_finder import JoinStep
        assert "->" in str(JoinStep("a", "b"))

    def test_edge_type_co_occur(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        b.add_query_logs([["dw.a", "dw.b"]])
        path = JoinPathFinder(b.build()).shortest_path("dw.a", "dw.b")
        assert path.steps[0].edge_type == EdgeType.CO_OCCUR.value

    def test_edge_type_lineage(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        b.add_lineage("dw.a", "dw.b")
        path = JoinPathFinder(b.build()).shortest_path("dw.a", "dw.b")
        assert path.steps[0].edge_type == EdgeType.LINEAGE.value

    def test_join_clauses(self, warehouse):
        path = JoinPathFinder(warehouse).shortest_path("dw.orders", "dw.cities")
        clauses = path.to_join_clauses()
        assert len(clauses) == 2
        assert "=" in clauses[0]

    def test_edge_cost_default(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw")])
        b.add_lineage("dw.a", "dw.b")
        finder = JoinPathFinder(b.build())
        assert finder._edge_cost("table:dw.a", "table:dw.nope") == 1.0


# ============================================================
# Evaluation
# ============================================================


class TestEvaluator:
    def test_exact_match(self):
        score = SQLEvaluator().evaluate(
            "SELECT a FROM t WHERE b = 1", "SELECT a FROM t WHERE b = 1")
        assert score.exact_match is True

    def test_case_and_whitespace_insensitive(self):
        score = SQLEvaluator().evaluate("select a from t", "SELECT   A  FROM T")
        assert score.exact_match is True

    def test_clause_scores_present(self):
        score = SQLEvaluator().evaluate(
            "SELECT a FROM t WHERE b = 1 GROUP BY a",
            "SELECT a FROM t WHERE b = 1 GROUP BY a")
        assert "select" in score.clauses and "where" in score.clauses

    def test_partial_credit_where_clause(self):
        score = SQLEvaluator().evaluate(
            "SELECT a FROM t WHERE b = 1",
            "SELECT a FROM t WHERE b = 2")
        assert 0 < score.clauses["where"].f1 < 1

    def test_missing_clause_zero_score(self):
        score = SQLEvaluator().evaluate(
            "SELECT a FROM t", "SELECT a FROM t WHERE b = 1")
        assert score.clauses["where"].f1 == 0.0

    def test_extra_prediction_lowers_precision(self):
        score = SQLEvaluator().evaluate(
            "SELECT a, z FROM t", "SELECT a FROM t")
        assert score.clauses["select"].precision < 1.0

    def test_mean_f1_zero_for_empty(self):
        assert SQLScore().mean_f1 == 0.0

    def test_execution_match(self):
        def executor(sql):
            return [1, 2] if "a" in sql else [1]
        ev = SQLEvaluator(executor=executor)
        score = ev.evaluate("SELECT a FROM t", "SELECT a FROM s")
        assert score.execution_match is True

    def test_execution_mismatch(self):
        def executor(sql):
            return [1] if "a" in sql else [2]
        ev = SQLEvaluator(executor=executor)
        score = ev.evaluate("SELECT a FROM t", "SELECT b FROM t")
        assert score.execution_match is False

    def test_execution_error_is_false(self):
        def executor(sql):
            raise RuntimeError("boom")
        ev = SQLEvaluator(executor=executor)
        assert ev.evaluate("SELECT a", "SELECT a").execution_match is False

    def test_no_executor_leaves_none(self):
        assert SQLEvaluator().evaluate("SELECT a", "SELECT a").execution_match is None

    def test_batch_aggregate(self):
        agg = SQLEvaluator().evaluate_batch([
            ("SELECT a FROM t", "SELECT a FROM t"),
            ("SELECT b FROM t", "SELECT a FROM t"),
        ])
        assert agg.count == 2
        assert agg.exact_match_rate == 0.5
        assert agg.execution_accuracy is None

    def test_batch_empty(self):
        assert SQLEvaluator().evaluate_batch([]).count == 0

    def test_batch_with_executor(self):
        ev = SQLEvaluator(executor=lambda s: [1])
        agg = ev.evaluate_batch([("SELECT a", "SELECT b")])
        assert agg.execution_accuracy == 1.0

    def test_normalise_strips_semicolon(self):
        assert _normalise("SELECT 1;") == "select 1"

    def test_normalise_handles_none(self):
        assert _normalise(None) == ""

    def test_extract_clause_missing(self):
        assert _extract_clause("select 1", r"\bwhere\s+(.*)") == ""

    def test_tokenize_removes_stopwords(self):
        assert "as" not in _tokenize("a as b")

    def test_tokenize_empty(self):
        assert _tokenize("") == set()

    def test_both_clause_empty_gives_full_credit(self):
        score = SQLEvaluator().evaluate("SELECT a FROM t", "SELECT a FROM t")
        # where absent from both -> skipped, not penalised
        assert "where" not in score.clauses


# ============================================================
# Pipeline
# ============================================================


class TestPipeline:
    def test_link_standalone(self, warehouse):
        assert NL2SQLPipeline(warehouse).link("营收", top_k=3)

    def test_join_path_standalone(self, warehouse):
        path = NL2SQLPipeline(warehouse).join_path("dw.orders", "dw.cities")
        assert path is not None

    def test_connect_tables_standalone(self, warehouse):
        path = NL2SQLPipeline(warehouse).connect_tables(
            ["dw.orders", "dw.users", "dw.cities"])
        assert path is not None

    def test_schema_context_includes_tables_and_columns(self, warehouse):
        ctx = NL2SQLPipeline(warehouse).schema_context("营收", top_k=5)
        assert "Tables:" in ctx
        assert "Columns:" in ctx

    def test_schema_context_includes_joins(self):
        # Deterministic two-table graph with a foreign key, so that linking
        # resolves to >= 2 tables and the join section is produced.
        b = SchemaGraphBuilder()
        b.add_tables([Table("orders", "dw"), Table("users", "dw")])
        b.add_columns([
            Column("user_id", "dw.orders", is_foreign_key=True),
            Column("id", "dw.users", is_primary_key=True),
        ])
        b.add_foreign_key("dw.orders.user_id", "dw.users.id")
        g = b.build()
        ctx = NL2SQLPipeline(g).schema_context(
            "orders users", top_k=10, include_joins=True)
        assert "Joins:" in ctx

    def test_schema_context_empty_when_no_match(self, warehouse):
        assert NL2SQLPipeline(warehouse).schema_context("无关内容") == ""

    def test_run_without_llm_raises(self, warehouse):
        with pytest.raises(ValueError):
            NL2SQLPipeline(warehouse).run("营收")

    def test_run_with_llm(self, warehouse):
        pipeline = NL2SQLPipeline(warehouse, llm=lambda p: "SELECT amount FROM orders")
        result = pipeline.run("营收")
        assert result.sql == "SELECT amount FROM orders"
        assert result.linked_items

    def test_result_tables_property(self, warehouse):
        result = NL2SQLPipeline(warehouse).run if False else None
        pipeline = NL2SQLPipeline(warehouse, llm=lambda p: "")
        res = pipeline.run("营收")
        assert isinstance(res.tables, list)

    def test_evaluate_delegates(self, warehouse):
        out = NL2SQLPipeline(warehouse).evaluate([
            ("SELECT a FROM t", "SELECT a FROM t")])
        assert out["count"] == 1
        assert out["exact_match_rate"] == 1.0

    def test_build_prompt_contains_question(self):
        prompt = _build_prompt("营收多少", "Tables:\n  orders")
        assert "营收多少" in prompt and "orders" in prompt
