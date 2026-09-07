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

"""End-to-end retrieval tests over an in-memory projection."""

import pytest

from hugegraph_llm.semantic_layer.context import FULL, KEYS_ONLY
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    InMemorySemanticReader,
    SemanticProjection,
    TableRow,
    TermRow,
)
from hugegraph_llm.semantic_layer.retrieval import (
    RetrievalConfig,
    SemanticLayerRetriever,
    render_for_prompt,
)


def _column(table, name, **kwargs):
    return ColumnRow(name=name, table=table, **kwargs)


def _projection():
    """A small warehouse: orders -> customers, orders -> products, island."""
    proj = SemanticProjection()
    for name, comment, rows in (
        ("orders", "All customer orders", 1000),
        ("customers", "Customer master data", 200),
        ("products", "Product catalogue", 50),
        ("island", "Unrelated lookup table", 5),
    ):
        proj.tables[name] = TableRow(
            name=name,
            database="wh",
            schema="public",
            comment=comment,
            row_count=rows,
        )
    for col in (
        _column("orders", "order_id", is_primary_key=True),
        _column("orders", "customer_id", is_foreign_key=True),
        _column("orders", "product_id", is_foreign_key=True),
        _column("orders", "amount", comment="Order total in USD"),
        _column("customers", "customer_id", is_primary_key=True),
        _column("customers", "name", comment="Customer full name"),
        _column("customers", "city", comment="City of residence"),
        _column("products", "product_id", is_primary_key=True),
        _column("products", "price", comment="List price"),
        _column("island", "code", is_primary_key=True),
    ):
        proj.columns[col.qualified] = col

    proj.references["orders.customer_id"] = ["customers.customer_id"]
    proj.references["orders.product_id"] = ["products.product_id"]
    proj.reference_proven[("orders.customer_id", "customers.customer_id")] = True
    proj.reference_proven[("orders.product_id", "products.product_id")] = True

    proj.terms["营收"] = TermRow(name="营收", description="订单金额总和", aliases=["收入"])
    proj.term_columns["营收"] = ["orders.amount"]
    proj.table_terms["orders"] = ["营收"]
    return proj


@pytest.fixture
def retriever():
    return SemanticLayerRetriever(InMemorySemanticReader(_projection()))


# -- recall ----------------------------------------------------------------


def test_bm25_recall_finds_table_by_column_comment(retriever):
    result = retriever.retrieve("order total in USD")
    assert "orders" in result.tables
    assert "bm25" in result.sources_used


def test_business_term_recall_maps_term_to_table(retriever):
    result = retriever.retrieve("营收是多少")
    assert "orders" in result.seeds
    assert "business_term" in result.sources_used


def test_business_term_recall_uses_alias(retriever):
    result = retriever.retrieve("收入")
    assert "orders" in result.seeds


def test_term_recall_follows_metric_to_table():
    """term -> metric -> expression column -> table.

    This is the hop that resolves an acronym with no direct table binding.
    """
    proj = _projection()
    proj.terms["ARR"] = TermRow(name="ARR", description="Annual recurring revenue")
    proj.term_metrics["ARR"] = ["mrr"]
    proj.metric_columns["mrr"] = ["orders.amount"]
    r = SemanticLayerRetriever(InMemorySemanticReader(proj))
    result = r.retrieve("ARR")
    assert "orders" in result.seeds
    assert "business_term" in result.sources_used


def test_unbound_term_yields_nothing():
    """A term with no bindings must not invent a table."""
    proj = _projection()
    proj.terms["orphan"] = TermRow(name="orphan")
    r = SemanticLayerRetriever(InMemorySemanticReader(proj))
    result = r.retrieve("orphan")
    assert result.tables == []


def test_retrieve_empty_projection_is_safe():
    empty = SemanticLayerRetriever(InMemorySemanticReader(SemanticProjection()))
    result = empty.retrieve("anything")
    assert result.tables == []
    assert not result.ok


def test_no_match_returns_no_tables(retriever):
    result = retriever.retrieve("zzzzz nothing matches this")
    assert result.tables == []


# -- expansion -------------------------------------------------------------


def test_expansion_reaches_joinable_tables(retriever):
    result = retriever.retrieve("customer city")
    assert "customers" in result.tables
    # orders is reachable via the declared foreign key.
    assert "orders" in result.tables


def test_expansion_decays_score_by_hop(retriever):
    proj = _projection()
    r = SemanticLayerRetriever(InMemorySemanticReader(proj))
    scored = r._expand(proj, [("customers", 1.0)], RetrievalConfig())
    assert scored["customers"][0] == 1.0
    assert scored["orders"][0] < 1.0  # one hop away
    assert scored["products"][0] < scored["orders"][0]  # two hops


def test_expansion_reports_reasons(retriever):
    proj = _projection()
    scored = SemanticLayerRetriever(
        InMemorySemanticReader(proj)
    )._expand(proj, [("customers", 1.0)], RetrievalConfig())
    assert "REFERENCES" in scored["orders"][1]


def test_hops_zero_disables_expansion(retriever):
    cfg = RetrievalConfig(hops=0)
    result = retriever.retrieve("customer city", config=cfg)
    assert result.expanded == []


# -- connectivity ----------------------------------------------------------


def test_unjoinable_table_is_removed_and_reported(retriever):
    """`island` has no edges; handing it over invites a hallucinated join."""
    proj = _projection()
    r = SemanticLayerRetriever(InMemorySemanticReader(proj))
    # Ordered strongest-first: the root is orders, so island is dropped.
    connected, disconnected = r._ensure_connected(proj, ["orders", "island"])
    assert connected == ["orders"]
    assert disconnected == ["island"]


def test_symmetric_adjacency_from_referenced_table(retriever):
    """A FK stored orders->customers must be traversable from customers too."""
    proj = _projection()
    assert "orders" in proj.tables_adjacent("customers")
    assert "customers" in proj.tables_adjacent("orders")


def test_island_alone_is_still_returned(retriever):
    """A single disconnected table cannot be dropped -- it may be the answer."""
    result = retriever.retrieve("code from island")
    assert result.tables == ["island"]


def test_connected_component_keeps_joinable_group(retriever):
    result = retriever.retrieve("customer name and product price")
    tables = set(result.tables)
    assert tables <= {"orders", "customers", "products"}
    # Whatever came back must be joinable: no island.
    assert "island" not in tables


def test_single_table_is_always_connected(retriever):
    proj = _projection()
    connected, disconnected = SemanticLayerRetriever(
        InMemorySemanticReader(proj)
    )._ensure_connected(proj, ["island"])
    assert connected == ["island"] and disconnected == []


# -- assembly --------------------------------------------------------------


def test_context_carries_references_and_keys(retriever):
    result = retriever.retrieve("order total")
    orders = next(c for c in result.budget.contexts if c.table_name == "orders")
    customer_col = next(c for c in orders.columns if c.name == "customer_id")
    assert customer_col.references == ["customers.customer_id"]
    assert customer_col.key_type == "foreign"
    assert orders.primary_key == ["order_id"]


def test_context_carries_database_and_schema(retriever):
    result = retriever.retrieve("order total")
    orders = next(c for c in result.budget.contexts if c.table_name == "orders")
    assert orders.database_name == "wh"
    assert orders.schema_name == "public"


def test_seed_flag_is_set(retriever):
    result = retriever.retrieve("customer city")
    seeds = {c.table_name for c in result.budget.contexts if c.is_seed}
    assert "customers" in seeds


# -- budget integration ----------------------------------------------------


def test_retrieve_respects_token_budget(retriever):
    result = retriever.retrieve("customer name and product price and order total",
                                max_tokens=400)
    assert result.budget.used_tokens <= result.budget.budget


def test_tiny_budget_still_returns_something(retriever):
    result = retriever.retrieve("customer name", max_tokens=200)
    # Even a minimal budget must not exceed itself.
    assert result.budget.used_tokens <= 200


def test_summary_is_serialisable(retriever):
    summary = retriever.retrieve("customer city").summary()
    assert set(summary) >= {
        "question", "seeds", "expanded", "tables", "used_tokens", "sources_used",
    }


def test_render_for_prompt_mentions_tables(retriever):
    text = render_for_prompt(retriever.retrieve("customer city"))
    assert "Retrieved schema context" in text
    assert "customers" in text


# -- vector path -----------------------------------------------------------


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, vector, top_k):
        return self._hits


def test_vector_hits_are_mapped_to_tables():
    proj = _projection()
    store = _FakeStore([
        ("column:orders.amount", 0.9),
        ("term:营收", 0.8),
        ("table:customers", 0.7),
        ("column:unknown.x", 0.6),
    ])
    r = SemanticLayerRetriever(
        InMemorySemanticReader(proj),
        vector_store=store,
        embed=lambda text: [1.0, 0.0],
    )
    mapped = r._recall_vector("anything")
    tables = [t for t, _ in mapped]
    assert "orders" in tables and "customers" in tables
    assert all(t in proj.tables for t in tables)
    assert "vector" in r.retrieve("anything").sources_used


def test_vector_store_failure_does_not_break_retrieval():
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("milvus down")

    r = SemanticLayerRetriever(
        InMemorySemanticReader(_projection()),
        vector_store=Broken(),
        embed=lambda t: [1.0],
    )
    result = r.retrieve("customer name")
    assert "customers" in result.tables
    assert "vector" not in result.sources_used


def test_invalidate_rebuilds_bm25(retriever):
    first = retriever.bm25
    retriever.invalidate()
    assert retriever.bm25 is not first
