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

"""Tests for feedback recording (M6)."""

import pytest

from hugegraph_llm.semantic_layer.feedback import FeedbackRecorder, FeedbackResult
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    InMemorySemanticReader,
    SemanticProjection,
    TableRow,
)
from tests.semantic_layer.fakes import FakeHugeGraph


def _projection():
    proj = SemanticProjection()
    for name in ("orders", "customers", "products"):
        proj.tables[name] = TableRow(name=name, comment="", row_count=10)
    # Ossie-style namespaced ids: feedback must use these, not table:<name>.
    proj.table_vids = {
        "orders": "acme:table:orders",
        "customers": "acme:table:customers",
        "products": "acme:table:products",
    }
    for col in (
        ColumnRow(name="order_id", table="orders", is_primary_key=True),
        ColumnRow(name="customer_id", table="orders", is_foreign_key=True),
        ColumnRow(name="customer_id", table="customers", is_primary_key=True),
    ):
        proj.columns[col.qualified] = col
    proj.references["orders.customer_id"] = ["customers.customer_id"]
    return proj


def _recorder(projection=None):
    proj = projection or _projection()
    reader = InMemorySemanticReader(proj)
    # The graph must actually contain the Table vertices the edges point at
    # -- the fake rejects dangling endpoints exactly like the real server,
    # which is the behaviour the namespaced-id test depends on.
    graph = FakeHugeGraph(
        vertices={
            vid: ("Table", {"name": name})
            for name, vid in proj.table_vids.items()
        }
    )
    return FeedbackRecorder(graph, reader), graph, reader


SQL = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id"


def test_record_writes_query_vertex_and_edges():
    rec, graph, _ = _recorder()
    result = rec.record("orders by customer", SQL)

    assert result.ok is True
    assert result.tables == ["orders", "customers"]
    assert result.co_occurrence_edges == 1
    assert graph.count("Query") == 1
    assert graph.edge_count("USES_TABLE") == 2
    assert graph.edge_count("CO_OCCUR") == 1


def test_record_resolves_namespaced_vertex_ids():
    """Ossie namespaces ids; edges must point at the real vertices."""
    rec, graph, _ = _recorder()
    rec.record("orders by customer", SQL)
    endpoints = {(e[1], e[2]) for e in graph.edges if e[0] == "CO_OCCUR"}
    assert endpoints == {("acme:table:orders", "acme:table:customers")}


def test_record_extracts_tables_from_sql_when_not_given():
    rec, graph, _ = _recorder()
    result = rec.record("q", SQL)
    assert result.tables == ["orders", "customers"]


def test_record_filters_unknown_tables():
    rec, graph, _ = _recorder()
    result = rec.record(
        "q", SQL, tables=["orders", "ghost_table"]
    )
    assert result.tables == ["orders"]
    assert result.unknown_tables == ["ghost_table"]


def test_record_no_known_tables_records_nothing():
    rec, graph, _ = _recorder()
    result = rec.record("q", "SELECT * FROM nothing_real")
    assert result.ok is False
    assert graph.count("Query") == 0


def test_single_table_records_no_co_occurrence():
    rec, graph, _ = _recorder()
    result = rec.record("q", "SELECT * FROM orders")
    assert result.tables == ["orders"]
    assert result.co_occurrence_edges == 0


def test_repeat_submission_is_deduplicated():
    """One question, one vote: edges are not double-counted."""
    rec, graph, _ = _recorder()
    first = rec.record("orders by customer", SQL)
    second = rec.record("orders by customer", SQL)

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.exec_count == 2
    assert graph.count("Query") == 1
    assert graph.edge_count("CO_OCCUR") == 1  # not 2


def test_repeat_is_ok_true():
    """A deduplicated repeat recorded fine the first time -- it is not a
    failure, and a caller reporting success by ok must not be misled."""
    rec, _graph, _ = _recorder()
    rec.record("orders by customer", SQL)
    second = rec.record("orders by customer", SQL)
    assert second.deduplicated is True
    assert second.ok is True


def test_repeat_bumps_exec_count_via_append():
    rec, graph, _ = _recorder()
    result = rec.record("orders by customer", SQL)
    vid = result.query_id
    before = graph.vertices[vid]["properties"]["exec_count"]
    rec.record("orders by customer", SQL)
    after = graph.vertices[vid]["properties"]["exec_count"]
    assert after == before + 1


def test_different_questions_are_not_deduplicated():
    rec, graph, _ = _recorder()
    rec.record("orders by customer", SQL)
    rec.record("revenue by customer", SQL)
    assert graph.count("Query") == 2
    # Same table pair twice -> two co-occurrence events.
    assert graph.edge_count("CO_OCCUR") == 2


def test_invalidate_clears_reader_cache():
    """Feedback must become visible to later retrieval in the same process."""
    proj = _projection()
    reader = InMemorySemanticReader(proj)
    graph = FakeHugeGraph()
    rec = FeedbackRecorder(graph, reader)

    cached_before = reader.projection()
    reader.invalidate()
    assert reader.projection() is cached_before  # in-memory reader ignores it

    class _CachedReader(InMemorySemanticReader):
        """Reader that caches, to verify invalidate() is forwarded."""

        def __init__(self, p):
            super().__init__(p)
            self.invalidated = 0

        def invalidate(self):
            self.invalidated += 1

    cached_reader = _CachedReader(proj)
    rec2 = FeedbackRecorder(graph, cached_reader)
    rec2.invalidate()
    assert cached_reader.invalidated == 1


def test_feedback_result_to_dict():
    rec, _graph, _ = _recorder()
    result = rec.record("q", SQL)
    payload = result.to_dict()
    assert payload["ok"] is True
    assert payload["exec_count"] == 1
    assert "query_id" in payload


def test_query_vertex_carries_schema_refs():
    rec, graph, _ = _recorder()
    result = rec.record("orders by customer", SQL)
    props = graph.vertices[result.query_id]["properties"]
    assert props["schema_refs"] == ["orders", "customers"]
    assert props["exec_count"] == 1
