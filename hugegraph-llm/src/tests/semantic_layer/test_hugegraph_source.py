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

"""Tests for the HugeGraph -> semantic layer connector (no server required).

The source graph is faked with a stub client, which keeps these tests fast
and lets them encode the two real-world quirks observed on ``kg_rag``:
labels that lie about their contents, and relationships that are simply
absent.
"""

import pytest

from hugegraph_llm.semantic_layer.connectors.hugegraph_source import (
    HugeGraphSourceConnector,
    _split_source_id,
)
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from tests.semantic_layer.fakes import FakeHugeGraph


def _client(vertices, edges):
    """Build a fake source graph keyed by ``vid -> (label, properties)``."""
    return FakeHugeGraph(
        vertices=vertices,
        edges=[dict(e) for e in edges],
    )


def _edges_of(conn, label):
    return [e for e in conn._edges if e[0] == label]


def _target():
    """A separate fake graph acting as the semantic layer target."""
    return FakeHugeGraph()


def _make_connector(vertices, edges, **kwargs):
    return HugeGraphSourceConnector(
        source_client=_client(vertices, edges),
        target_client=_target(),
        **kwargs,
    )


# -- id handling -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("3:orders.id", "orders.id"), ("orders.id", "orders.id"), ("1:营收", "营收")],
)
def test_split_source_id(raw, expected):
    assert _split_source_id(raw) == expected


# -- shape-driven classification -------------------------------------------


def test_metric_without_definition_is_skipped():
    """`kg_rag` stores bare field references under the Metric label."""
    vertices = {
        "1:dim_user.register_at": (
            "Metric", {"id": "dim_user.register_at", "name": "dim_user.register_at"}
        ),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    assert conn.stats["skipped_metric_like"] == 1
    assert conn.stats["terms"] == 0


def test_metric_with_definition_becomes_business_term():
    vertices = {
        "1:营收": ("Metric", {"name": "营收", "definition": "营收", "formula": ""}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    assert conn.stats["terms"] == 1
    term_id = [vid for vid, _n in conn._vertices if vid == "term:营收"]
    assert term_id


def test_query_named_metric_prefix_becomes_metric():
    vertices = {"1:metric_GMV": ("Query", {"id": "metric_GMV", "name": "metric:GMV"})}
    conn = _make_connector(vertices, [])
    conn.ingest()
    assert conn.stats["metrics"] == 1
    # The `metric:` prefix is a source naming convention, not part of the name.
    assert [n["properties"]["name"] for vid, n in conn._vertices
            if vid == "metric:GMV"] == ["GMV"]


# -- structural mapping ----------------------------------------------------


def test_table_and_column_mapping():
    vertices = {
        "1:orders": ("Table", {"name": "orders", "comment": "订单表", "row_count": 0}),
        "2:orders.id": ("Field", {"name": "orders.id", "comment": "主键", "type": "bigint"}),
    }
    edges = [{"label": "hasColumn", "outV": "1:orders", "inV": "2:orders.id"}]
    conn = _make_connector(vertices, edges)
    conn.ingest()

    assert conn.stats["tables"] == 1
    assert conn.stats["columns"] == 1
    assert conn.stats["has_column"] == 1

    by_id = dict(conn._vertices)
    column = by_id["column:orders.id"]
    assert column["properties"]["name"] == "id"
    assert column["properties"]["table"] == "orders"
    assert column["properties"]["comment"] == "主键"


def test_computed_from_field_becomes_term_maps():
    vertices = {
        "1:营收": ("Metric", {"name": "营收", "definition": "营收", "formula": ""}),
        "2:ads.gmv": ("Field", {"name": "ads.gmv", "comment": "", "type": "decimal"}),
    }
    edges = [{"label": "computedFromField", "outV": "1:营收", "inV": "2:ads.gmv"}]
    conn = _make_connector(vertices, edges)
    conn.ingest()
    assert conn.stats["term_maps"] == 1
    assert any(
        label == EdgeLabel.TERM_MAPS.value
        and out == "term:营收"
        and in_v == "column:ads.gmv"
        for label, out, in_v, _p in conn._edges
    )


def test_missing_relationships_are_reported_not_faked():
    """`kg_rag` has no lineage edges; the summary must say zero."""
    vertices = {
        "1:a": ("Table", {"name": "a", "comment": "", "row_count": 0}),
        "2:b": ("Table", {"name": "b", "comment": "", "row_count": 0}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    assert conn.stats["lineage"] == 0
    assert not any(e[0] == EdgeLabel.LINEAGE.value for e in conn._edges)


# -- foreign key inference -------------------------------------------------


def test_foreign_keys_inferred_and_marked_unproven():
    vertices = {
        "1:orders": ("Table", {"name": "orders", "comment": "", "row_count": 0}),
        "2:users": ("Table", {"name": "users", "comment": "", "row_count": 0}),
        "3:orders.user_id": ("Field", {"name": "orders.user_id", "type": "bigint"}),
        "4:users.user_id": ("Field", {"name": "users.user_id", "type": "bigint"}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    fks = [e for e in conn._edges if e[0] == EdgeLabel.REFERENCES.value]
    assert len(fks) == 1
    assert fks[0][3]["proven"] is False  # inferred, never presented as declared


def test_inference_can_be_disabled():
    vertices = {
        "1:orders": ("Table", {"name": "orders", "comment": "", "row_count": 0}),
        "2:users": ("Table", {"name": "users", "comment": "", "row_count": 0}),
        "3:orders.user_id": ("Field", {"name": "orders.user_id", "type": "bigint"}),
        "4:users.user_id": ("Field", {"name": "users.user_id", "type": "bigint"}),
    }
    conn = _make_connector(vertices, [], infer_foreign_keys=False)
    conn.ingest()
    assert conn.stats["foreign_keys"] == 0


def test_same_table_columns_are_not_foreign_keys():
    vertices = {
        "1:orders": ("Table", {"name": "orders", "comment": "", "row_count": 0}),
        "2:orders.id": ("Field", {"name": "orders.id", "type": "bigint"}),
        "3:orders.user_id": ("Field", {"name": "orders.user_id", "type": "bigint"}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    # "id" and "user_id" differ by name, so no shared-name pair exists here.
    assert conn.stats["foreign_keys"] == 0


def test_fk_inference_matches_shared_names_only():
    vertices = {
        "1:orders": ("Table", {"name": "orders", "comment": "", "row_count": 0}),
        "2:users": ("Table", {"name": "users", "comment": "", "row_count": 0}),
        "3:orders.id": ("Field", {"name": "orders.id", "type": "bigint"}),
        "4:users.id": ("Field", {"name": "users.id", "type": "bigint"}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    assert conn.stats["foreign_keys"] == 1


# -- property hygiene ------------------------------------------------------


def test_undeclared_properties_are_dropped():
    vertices = {
        "1:a": ("Table", {"name": "a", "not_a_declared_prop": "x", "row_count": "5"}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    props = dict(conn._vertices)[0][1]["properties"] if False else [
        n["properties"] for _v, n in conn._vertices
    ][0]
    assert "not_a_declared_prop" not in props
    assert props["row_count"] == 5  # coerced to LONG


def test_aliases_string_is_split():
    vertices = {
        "1:营收": ("Metric", {"name": "营收", "definition": "营收", "aliases": "收入;销售额"}),
    }
    conn = _make_connector(vertices, [])
    conn.ingest()
    term = [n for _v, n in conn._vertices][0]
    assert term["properties"]["aliases"] == ["收入", "销售额"]


def test_dangling_edges_are_dropped():
    """An edge pointing at an unmapped vertex must never reach the loader.

    Such an edge would make HugeGraph fail the whole batch with an opaque
    "Invalid vertex id", so it is filtered during transform.
    """
    vertices = {
        "1:营收": ("Metric", {"name": "营收", "definition": "营收", "formula": ""}),
    }
    edges = [{"label": "computedFromField", "outV": "1:营收", "inV": "9:missing"}]
    conn = _make_connector(vertices, edges)
    conn.ingest()
    assert conn.stats["term_maps"] == 0
    assert conn._edges == []
