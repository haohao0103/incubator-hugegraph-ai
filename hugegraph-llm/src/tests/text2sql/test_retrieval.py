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

"""Tests for the deterministic retrieval operators."""

import pytest

from hugegraph_llm.text2sql.examples import build_order_domain_model
from hugegraph_llm.text2sql.retrieval import (
    SemanticGraph,
    TermIndex,
    build_sql_prompt,
    find_join_path,
    render_sql_filter,
    resolve_metric,
    schema_link,
)
from hugegraph_llm.text2sql.seed import seed_graph

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model():
    return build_order_domain_model()


@pytest.fixture(scope="module")
def graph(model):
    return SemanticGraph(seed_graph(model))


def test_term_resolution_exact_alias(graph, model):
    index = TermIndex(model.terms)
    assert index.resolve("成交额") == "GMV"
    assert index.resolve("支付金额") == "GMV"
    assert index.resolve("gmv") == "GMV"


def test_term_resolution_substring_fallback(model):
    index = TermIndex(model.terms)
    assert index.resolve("上个月成交额是多少") == "GMV"


def test_term_resolution_unknown(model):
    index = TermIndex(model.terms)
    assert index.resolve("完全无关的词") is None


def test_schema_link(graph):
    assert schema_link(graph, "GMV") == {"columns": ["order_detail.amount"], "metrics": ["gmv"]}
    assert schema_link(graph, "司机") == {"columns": ["driver.name"], "metrics": []}


def test_find_join_path_direct(graph):
    path = find_join_path(graph, "order", "user")
    assert len(path) == 1
    assert path[0].from_table == "order"
    assert path[0].to_table == "user"
    assert path[0].on_condition == "order.user_id = user.id"


def test_find_join_path_multi_hop(graph):
    path = find_join_path(graph, "order_detail", "driver")
    assert [s.from_table for s in path] == ["order_detail", "order"]
    assert [s.to_table for s in path] == ["order", "driver"]
    assert path[-1].on_condition == "order.driver_id = driver.id"


def test_find_join_path_same_table(graph):
    assert find_join_path(graph, "order", "order") == []


def test_find_join_path_unknown_table(graph):
    assert find_join_path(graph, "order", "nonexistent") == []


def test_resolve_metric(graph):
    metric = resolve_metric(graph, "gmv")
    assert metric is not None
    assert metric.agg_func == "SUM"
    assert metric.measure == "order_detail.amount"
    assert len(metric.filters) == 1
    assert metric.filters[0].column == "order.status"
    assert metric.filters[0].operator == "IN"
    assert metric.filters[0].values == ["2", "3"]
    assert metric.time_column == "order.pay_time"
    assert metric.time_granularity == "month"
    assert metric.dedup is True


def test_render_sql_filter_reconstructs_where(graph):
    metric = resolve_metric(graph, "gmv")
    assert render_sql_filter(metric.filters) == "order.status IN ('2', '3')"


def test_resolve_metric_missing(graph):
    assert resolve_metric(graph, "nonexistent_metric") is None


def test_build_sql_prompt_includes_all_sources(graph):
    path = find_join_path(graph, "order_detail", "order")
    prompt = build_sql_prompt(
        graph,
        question="上个月 GMV 是多少",
        tables=["order", "order_detail"],
        metrics=["gmv"],
        join_paths=path,
        few_shot_sql=["上个月 GMV 是多少 -> SELECT SUM(od.amount) ..."],
    )

    assert "CREATE TABLE order" in prompt
    assert "CREATE TABLE order_detail" in prompt
    assert "gmv = SUM(order_detail.amount)" in prompt
    assert "order.status: 2 = 已支付" in prompt
    assert "ON order_detail.order_id = order.id" in prompt
    assert "上个月 GMV 是多少" in prompt
    assert prompt.rstrip().endswith("SQL:")


def test_build_sql_prompt_minimal(graph):
    prompt = build_sql_prompt(graph, question="订单量", tables=["order"])
    assert "CREATE TABLE order" in prompt
    assert "SQL:" in prompt
    # No metric section when none requested.
    assert "Metric definitions" not in prompt
