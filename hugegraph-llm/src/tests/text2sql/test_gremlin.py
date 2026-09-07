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

"""Tests for the Gremlin mappings of the semantic-layer graph."""

import pytest

from hugegraph_llm.text2sql.gremlin import (
    column_values,
    few_shot_by_metric,
    join_path,
    metric_filters,
    metric_value_map,
    term_to_columns,
    term_to_metrics,
)

pytestmark = pytest.mark.unit


def test_term_to_columns():
    q = term_to_columns("GMV")
    assert "hasLabel('term')" in q
    assert "has('name', 'GMV')" in q
    assert "out('maps_to')" in q


def test_term_to_metrics():
    q = term_to_metrics("订单量")
    assert "out('maps_to_metric')" in q
    assert "has('name', '订单量')" in q


def test_metric_value_map_reads_koujing():
    q = metric_value_map("gmv")
    assert "hasLabel('metric')" in q
    assert "'dedup'" in q and "'time_granularity'" in q and "'measure'" in q
    assert "'filters'" not in q  # filters are structured via has_filter edges now


def test_metric_filters_structured():
    q = metric_filters("gmv")
    assert "out('has_filter')" in q
    assert "'column'" in q and "'operator'" in q and "'values'" in q


def test_column_values_uses_full_name():
    q = column_values("order.status")
    assert "has('full_name', 'order.status')" in q
    assert "out('has_value')" in q


def test_join_path_is_undirected():
    q = join_path("order_detail", "order")
    assert "repeat(both('joins').simplePath())" in q
    assert "has('name', 'order')" in q


def test_column_values_and_few_shot():
    q = column_values("order.status")
    assert "out('has_value')" in q
    q = few_shot_by_metric("gmv")
    assert "out('uses_metric')" in q


def test_quote_escapes_single_quote():
    q = term_to_columns("O'Brien")
    assert "O\\'Brien" in q
