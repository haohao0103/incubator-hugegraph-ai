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

"""Tests for the MCP tool layer."""

import pytest

from hugegraph_llm.semantic_layer.mcp_tools import SemanticLayerTools
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    InMemorySemanticReader,
    SemanticProjection,
    TableRow,
    TermRow,
)


def _column(table, name, **kw):
    return ColumnRow(name=name, table=table, **kw)


def _proj():
    proj = SemanticProjection()
    proj.tables["orders"] = TableRow(
        name="orders", database="wh", schema="public",
        comment="All customer orders", row_count=1000,
    )
    proj.tables["customers"] = TableRow(
        name="customers", database="wh", schema="public",
        comment="Customer master", row_count=200,
    )
    for col in (
        _column("orders", "order_id", is_primary_key=True),
        _column("orders", "customer_id", is_foreign_key=True),
        _column("orders", "amount", comment="Order total"),
        _column("customers", "customer_id", is_primary_key=True),
        _column("customers", "city", comment="City"),
    ):
        proj.columns[col.qualified] = col
    proj.references["orders.customer_id"] = ["customers.customer_id"]
    proj.reference_proven[("orders.customer_id", "customers.customer_id")] = True
    proj.terms["营收"] = TermRow(name="营收", description="订单金额", aliases=["收入"])
    proj.term_columns["营收"] = ["orders.amount"]
    proj.term_metrics["营收"] = ["gmv"]
    proj.metric_columns["gmv"] = ["orders.amount"]
    return proj


@pytest.fixture
def tools():
    return SemanticLayerTools(InMemorySemanticReader(_proj()))


class _Store:
    def search(self, vector, top_k):
        return [("column:orders.amount", 0.9)]


# -- capability probing ----------------------------------------------------


def test_capabilities_without_vector(tools):
    caps = tools.capabilities()
    assert caps.has_terms is True
    assert caps.has_vector is False
    assert caps.has_bm25 is True
    assert caps.table_count == 2


def test_capabilities_with_vector():
    t = SemanticLayerTools(
        InMemorySemanticReader(_proj()),
        vector_store=_Store(),
        embed=lambda s: [1.0],
    )
    assert t.capabilities().has_vector is True


def test_context_tool_name_reflects_capability(tools):
    """The tool list itself must tell the agent which recall is available."""
    assert "get_context_by_table_full_text_search" in tools.register()


def test_context_tool_name_with_vector_and_terms():
    t = SemanticLayerTools(
        InMemorySemanticReader(_proj()),
        vector_store=_Store(),
        embed=lambda s: [1.0],
    )
    names = set(t.register())
    assert "get_context_by_term_hybrid_search" in names
    assert "get_context_by_table_full_text_search" not in names


def test_only_one_context_tool_registered(tools):
    context_tools = [
        name for name in tools.register() if name.startswith("get_context")
    ]
    assert len(context_tools) == 1


def test_always_on_tools_present(tools):
    names = set(tools.register())
    assert {"list_schemas", "list_tables_by_schema", "get_join_path",
            "get_table_columns", "get_full_metadata_schema"} <= names


def test_term_tool_registered_only_with_terms():
    empty = SemanticLayerTools(InMemorySemanticReader(SemanticProjection()))
    assert "search_business_terms" not in empty.register()
    with_terms = SemanticLayerTools(InMemorySemanticReader(_proj()))
    assert "search_business_terms" in with_terms.register()


def test_empty_graph_still_registers_browse_tools():
    empty = SemanticLayerTools(InMemorySemanticReader(SemanticProjection()))
    assert "list_schemas" in empty.register()


# -- tool descriptions -----------------------------------------------------


def test_tool_descriptions_have_mcp_shape(tools):
    for desc in tools.tool_descriptions():
        assert {"name", "description", "inputSchema"} <= set(desc)
        assert desc["inputSchema"]["type"] == "object"


def test_descriptions_are_non_empty(tools):
    assert all(d["description"].strip() for d in tools.tool_descriptions())


# -- handlers --------------------------------------------------------------


def test_context_handler_returns_tables(tools):
    name = next(n for n in tools.register() if n.startswith("get_context"))
    out = tools.call(name, {"question": "customer city"})
    assert "orders" in out["tables"] or "customers" in out["tables"]
    assert out["contexts"]


def test_context_handler_no_match_gives_hint(tools):
    name = next(n for n in tools.register() if n.startswith("get_context"))
    out = tools.call(name, {"question": "zzzz nothing"})
    assert out["tables"] == []
    assert "hint" in out


def test_list_schemas_groups_tables(tools):
    out = tools.call("list_schemas", {})
    assert out["schemas"]["wh.public"]["tables"] == 2


def test_list_tables_filters_by_name(tools):
    out = tools.call("list_tables_by_schema", {"name_contains": "cust"})
    assert [t["name"] for t in out["tables"]] == ["customers"]


def test_list_tables_respects_limit(tools):
    out = tools.call("list_tables_by_schema", {"limit": 1})
    assert out["returned"] == 1


def test_get_join_path_returns_sql(tools):
    out = tools.call("get_join_path", {"left": "orders", "right": "customers"})
    assert out["found"] is True
    assert out["steps"][0]["sql"] == "orders.customer_id = customers.customer_id"
    assert out["all_proven"] is True


def test_get_join_path_missing_is_honest(tools):
    out = tools.call("get_join_path", {"left": "orders", "right": "nope"})
    assert out["found"] is False
    assert "No column-level join path" in out["render"]


def test_get_table_columns(tools):
    out = tools.call("get_table_columns", {"table": "orders"})
    names = [c["name"] for c in out["columns"]]
    assert names == ["amount", "customer_id", "order_id"]
    pk = [c for c in out["columns"] if c["key_type"] == "primary"]
    assert [c["name"] for c in pk] == ["order_id"]


def test_get_table_columns_unknown_table(tools):
    out = tools.call("get_table_columns", {"table": "nope"})
    assert out["columns"] == []
    assert "error" in out


def test_search_business_terms(tools):
    out = tools.call("search_business_terms", {"query": "营收"})
    assert out["returned"] == 1
    assert out["terms"][0]["tables"] == ["orders"]


def test_search_business_terms_matches_alias(tools):
    out = tools.call("search_business_terms", {"query": "收入"})
    assert out["returned"] == 1


def test_search_business_terms_follows_metric(tools):
    """A term bound only through a metric must still resolve to a table."""
    proj = _proj()
    del proj.term_columns["营收"]  # only the metric link remains
    t = SemanticLayerTools(InMemorySemanticReader(proj))
    out = t.call("search_business_terms", {"query": "营收"})
    assert out["terms"][0]["tables"] == ["orders"]


def test_full_schema_is_capped_and_reports_truncation(tools):
    out = tools.call("get_full_metadata_schema", {"max_tokens": 5})
    assert out["truncated"] is True or out["returned"] == 0
    assert out["total_tables"] == 2
    assert out["used_tokens"] <= 5


def test_full_schema_no_truncation_when_room(tools):
    out = tools.call("get_full_metadata_schema", {"max_tokens": 100000})
    assert out["truncated"] is False
    assert out["returned"] == 2


# -- dispatch --------------------------------------------------------------


def test_call_unknown_tool_raises(tools):
    with pytest.raises(KeyError):
        tools.call("not_a_tool", {})


def test_register_is_cached(tools):
    assert tools.register() is tools.register()


def test_invalidate_forces_reregistration(tools):
    first = tools.register()
    tools.invalidate()
    assert tools.register() is not first
