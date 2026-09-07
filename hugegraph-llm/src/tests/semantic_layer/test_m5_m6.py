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

"""M5 + M6 integration: trust import round-trip and the feedback loop."""

import pytest

from hugegraph_llm.semantic_layer.connectors.ossie import TRUST_VENDOR, OssieConnector
from hugegraph_llm.semantic_layer.feedback import FeedbackRecorder
from hugegraph_llm.semantic_layer.mcp_tools import SemanticLayerTools
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    InMemorySemanticReader,
    SemanticProjection,
    TableRow,
)
from tests.semantic_layer.fakes import FakeHugeGraph
from tests.semantic_layer.test_ossie import _Connector, _spec


# -- M5: trust custom_extensions on import ---------------------------------


def test_import_reads_trust_extension():
    spec = _spec()
    spec["semantic_model"][0]["datasets"][0]["custom_extensions"] = [
        {"vendor_name": TRUST_VENDOR,
         "data": {"confidence": 0.8, "freshness_ts": 1700000000}}
    ]
    conn = _Connector(spec)
    conn.ingest()
    table = {vid: n for vid, n in conn._vertices}["acme:table:orders"]
    assert table["properties"]["confidence"] == 0.8
    assert table["properties"]["freshness_ts"] == 1700000000


def test_import_source_system_override():
    spec = _spec()
    spec["semantic_model"][0]["datasets"][0]["custom_extensions"] = [
        {"vendor_name": TRUST_VENDOR, "data": {"source_system": "catalog-x"}}
    ]
    conn = _Connector(spec)
    conn.ingest()
    table = {vid: n for vid, n in conn._vertices}["acme:table:orders"]
    assert table["properties"]["source_system"] == "catalog-x"


def test_import_ignores_foreign_vendor_extensions():
    spec = _spec()
    spec["semantic_model"][0]["datasets"][0]["custom_extensions"] = [
        {"vendor_name": "someone-else", "data": {"confidence": 0.1}}
    ]
    conn = _Connector(spec)
    conn.ingest()
    table = {vid: n for vid, n in conn._vertices}["acme:table:orders"]
    assert "confidence" not in table["properties"]


def test_import_metric_trust_extension():
    spec = _spec()
    spec["semantic_model"][0]["metrics"][0]["custom_extensions"] = [
        {"vendor_name": TRUST_VENDOR, "data": {"confidence": 0.95}}
    ]
    conn = _Connector(spec)
    conn.ingest()
    metric = {vid: n for vid, n in conn._vertices}["acme:metric:total_revenue"]
    assert metric["properties"]["confidence"] == 0.95


def test_trust_round_trip_via_extensions():
    """Export writes trust into custom_extensions; import must read it back."""
    spec = _spec()
    spec["semantic_model"][0]["datasets"][0]["custom_extensions"] = [
        {"vendor_name": TRUST_VENDOR, "data": {"confidence": 0.7}}
    ]
    conn = _Connector(spec)
    conn.ingest()
    exported = conn.export("acme")["semantic_model"][0]["datasets"][0]
    assert exported["custom_extensions"][0]["data"]["confidence"] == 0.7

    # Feed the export back through a fresh connector.
    spec2 = {"version": "0.1.1", "semantic_model": [{
        "name": "acme",
        "datasets": [exported],
        "relationships": [],
        "metrics": [],
    }]}
    conn2 = _Connector(spec2)
    conn2.ingest()
    table = {vid: n for vid, n in conn2._vertices}["acme:table:orders"]
    assert table["properties"]["confidence"] == 0.7


# -- M6: feedback loop through MCP tools -----------------------------------


def _feedback_projection():
    proj = SemanticProjection()
    proj.tables["orders"] = TableRow(name="orders", comment="", row_count=10)
    proj.tables["customers"] = TableRow(name="customers", comment="", row_count=5)
    proj.table_vids = {
        "orders": "acme:table:orders",
        "customers": "acme:table:customers",
    }
    proj.columns["orders.order_id"] = ColumnRow(
        name="order_id", table="orders", is_primary_key=True
    )
    proj.columns["orders.customer_id"] = ColumnRow(
        name="customer_id", table="orders", is_foreign_key=True
    )
    proj.columns["customers.customer_id"] = ColumnRow(
        name="customer_id", table="customers", is_primary_key=True
    )
    proj.references["orders.customer_id"] = ["customers.customer_id"]
    return proj


def _tools_with_feedback():
    proj = _feedback_projection()
    graph = FakeHugeGraph(
        vertices={vid: ("Table", {"name": name})
                  for name, vid in proj.table_vids.items()}
    )
    reader = InMemorySemanticReader(proj)
    feedback = FeedbackRecorder(graph, reader)
    tools = SemanticLayerTools(reader, feedback=feedback)
    return tools, graph, reader, feedback


def test_feedback_tool_registered_only_with_recorder():
    tools, _, _, _ = _tools_with_feedback()
    assert "record_query_feedback" in tools.register()

    bare = SemanticLayerTools(InMemorySemanticReader(_feedback_projection()))
    assert "record_query_feedback" not in bare.register()


def test_feedback_tool_records_and_invalidates():
    tools, graph, _, _ = _tools_with_feedback()
    out = tools.call("record_query_feedback", {
        "question": "orders per customer",
        "sql": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
    })
    assert out["ok"] is True
    assert graph.count("Query") == 1
    assert graph.edge_count("CO_OCCUR") == 1


def test_full_loop_feedback_improves_expansion():
    """The point of M6: a recorded co-occurrence becomes a traversal edge.

    Uses a reloadable reader to model what GremlinSemanticReader does on
    ``invalidate()`` -- re-read the graph, now including the CO_OCCUR edge
    the recorder wrote.
    """
    proj_before = _feedback_projection()
    graph = FakeHugeGraph(
        vertices={vid: ("Table", {"name": name})
                  for name, vid in proj_before.table_vids.items()}
    )
    reader = InMemorySemanticReader(proj_before)
    feedback = FeedbackRecorder(graph, reader)

    before = set(reader.projection().tables_adjacent("customers"))
    feedback.record(
        "orders per customer",
        "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
    )

    # Simulate the post-invalidate re-read: the server now has the edge.
    proj_after = _feedback_projection()
    proj_after.co_occur["customers"] = ["orders"]
    proj_after.co_occur["orders"] = ["customers"]

    class _ReloadableReader(InMemorySemanticReader):
        def __init__(self, after):
            super().__init__(after)

        def invalidate(self):
            self._projection = self._after

    reloadable = _ReloadableReader(proj_after)
    reloadable._after = proj_after
    reloadable.invalidate()
    after = reloadable.projection().tables_adjacent("customers")

    assert before == {"orders"}  # only the FK, pre-existing
    assert "orders" in after
    assert "CO_OCCUR" in after["orders"]  # now also reachable via usage
