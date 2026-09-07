# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for deterministic seeding (SemanticModel -> graph data)."""

import pytest

from hugegraph_llm.text2sql.examples import build_order_domain_model
from hugegraph_llm.text2sql.seed import seed_graph

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model():
    return build_order_domain_model()


@pytest.fixture(scope="module")
def graph_data(model):
    return seed_graph(model)


def _by_label(vertices, label):
    return [v for v in vertices if v["label"] == label]


def test_seed_vertex_counts(graph_data):
    vertices = graph_data["vertices"]
    assert len(_by_label(vertices, "table")) == 4
    assert len(_by_label(vertices, "domain")) == 1
    assert len(_by_label(vertices, "metric")) == 2
    assert len(_by_label(vertices, "term")) == 5
    # order has 8 columns, order_detail 4, user 3, driver 3 = 18 columns.
    assert len(_by_label(vertices, "column")) == 18
    # 4 status values + 1 structured filter (gmv's status IN (2,3)).
    assert len(_by_label(vertices, "value")) == 4
    assert len(_by_label(vertices, "filter")) == 1
    assert len(_by_label(vertices, "query_pattern")) == 2


def test_seed_column_ids_are_composite(graph_data):
    columns = _by_label(graph_data["vertices"], "column")
    ids = {c["id"] for c in columns}
    full_names = {c["properties"]["full_name"] for c in columns}
    assert "column:order!status" in ids
    assert "column:order_detail!amount" in ids
    assert "column:driver!name" in ids
    assert "order.status" in full_names
    assert "order_detail.amount" in full_names


def test_seed_vertex_ids_are_label_scoped(graph_data):
    ids_by_label = {}
    for v in graph_data["vertices"]:
        ids_by_label.setdefault(v["label"], set()).add(v["id"])
    # domain "order" and table "order" no longer collide.
    assert "domain:order" in ids_by_label["domain"]
    assert "table:order" in ids_by_label["table"]
    assert "metric:gmv" in ids_by_label["metric"]
    assert "term:GMV" in ids_by_label["term"]


def test_seed_join_edge_carries_on_condition(graph_data):
    joins = [e for e in graph_data["edges"] if e["label"] == "joins"]
    order_to_user = next(e for e in joins if e["outV"] == "table:order" and e["inV"] == "table:user")
    assert order_to_user["properties"]["on_condition"] == "order.user_id = user.id"
    assert order_to_user["properties"]["fanout_risk"] == "none"


def test_seed_term_edges(graph_data):
    edges = graph_data["edges"]
    maps_to = [e for e in edges if e["label"] == "maps_to" and e["outV"] == "term:GMV"]
    assert maps_to == [{"label": "maps_to", "outV": "term:GMV", "inV": "column:order_detail!amount", "properties": {}}]
    maps_metric = [e for e in edges if e["label"] == "maps_to_metric" and e["outV"] == "term:GMV"]
    assert [e["inV"] for e in maps_metric] == ["metric:gmv"]


def test_seed_term_ambiguous_maps_to_multiple_columns(graph_data):
    edges = graph_data["edges"]
    maps_to = [e for e in edges if e["label"] == "maps_to" and e["outV"] == "term:金额"]
    assert [e["inV"] for e in maps_to] == ["column:order!amount", "column:order_detail!amount"]


def test_seed_metric_aggregates_edge(graph_data):
    aggregates = [e for e in graph_data["edges"] if e["label"] == "aggregates"]
    gmv_agg = next(e for e in aggregates if e["outV"] == "metric:gmv")
    assert gmv_agg["inV"] == "column:order_detail!amount"
    assert gmv_agg["properties"]["agg_func"] == "SUM"


def test_seed_metric_dedup_and_structured_filter(graph_data):
    metrics = _by_label(graph_data["vertices"], "metric")
    gmv = next(v for v in metrics if v["id"] == "metric:gmv")
    assert "filters" not in gmv["properties"]
    assert gmv["properties"]["dedup"] is True

    has_filter = [e for e in graph_data["edges"] if e["label"] == "has_filter" and e["outV"] == "metric:gmv"]
    assert len(has_filter) == 1
    filter_vertex = next(v for v in graph_data["vertices"] if v["id"] == has_filter[0]["inV"])
    assert filter_vertex["properties"]["column"] == "order.status"
    assert filter_vertex["properties"]["operator"] == "IN"
    assert filter_vertex["properties"]["values"] == ["2", "3"]


def test_seed_column_role(graph_data):
    columns = _by_label(graph_data["vertices"], "column")
    status = next(c for c in columns if c["properties"]["full_name"] == "order.status")
    assert status["properties"]["role"] == "value"
    assert status["properties"]["name"] == "status"
    assert status["properties"]["table"] == "order"
    amount = next(c for c in columns if c["properties"]["full_name"] == "order_detail.amount")
    assert amount["properties"]["role"] == "measure"
