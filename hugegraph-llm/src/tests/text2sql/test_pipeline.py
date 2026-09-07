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

"""Tests for the end-to-end Text2SQLPipeline orchestration."""

from unittest.mock import MagicMock

import pytest

from hugegraph_llm.text2sql.examples import build_order_domain_model
from hugegraph_llm.text2sql.pipeline import Text2SQLPipeline

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model():
    return build_order_domain_model()


def test_plan_resolves_terms_and_links(model):
    pipeline = Text2SQLPipeline(model)
    result = pipeline.plan("上个月成交额是多少")

    assert result.resolved_terms == ["GMV"]
    assert result.linked_metrics == ["gmv"]
    assert "order_detail.amount" in result.linked_columns
    # The metric's measure + time column both derive tables.
    assert set(result.tables) == {"order_detail", "order"}
    assert result.metric_definitions[0].name == "gmv"


def test_plan_finds_join_path(model):
    pipeline = Text2SQLPipeline(model)
    result = pipeline.plan("上个月成交额是多少")
    assert len(result.join_paths) >= 1
    assert result.join_paths[0].on_condition == "order_detail.order_id = order.id"


def test_plan_few_shot_included(model):
    pipeline = Text2SQLPipeline(model)
    result = pipeline.plan("上个月 GMV 是多少")
    assert "SELECT SUM(od.amount) AS gmv" in result.prompt


def test_generate_without_llm_returns_prompt(model):
    pipeline = Text2SQLPipeline(model)
    out = pipeline.generate("上个月 GMV 是多少")
    assert isinstance(out, str)
    assert out.rstrip().endswith("SQL:")


def test_generate_with_llm_returns_sql(model):
    llm = MagicMock()
    llm.generate.return_value = "SELECT SUM(od.amount) AS gmv"
    pipeline = Text2SQLPipeline(model, llm=llm)

    sql = pipeline.generate("上个月 GMV 是多少")

    assert sql == "SELECT SUM(od.amount) AS gmv"
    llm.generate.assert_called_once()
    assert "SQL:" in llm.generate.call_args.kwargs["prompt"]
    assert "order.status IN ('2', '3')" in llm.generate.call_args.kwargs["prompt"]
