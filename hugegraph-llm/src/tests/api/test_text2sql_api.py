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

"""Tests for the Text2SQL HTTP endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, FastAPI, status
from fastapi.testclient import TestClient

from hugegraph_llm.api.models.text2sql_requests import Text2SQLRequest
from hugegraph_llm.api.text2sql_api import Text2SQLApi, text2sql_http_api
from hugegraph_llm.text2sql.examples import build_order_domain_model
from hugegraph_llm.text2sql.pipeline import Text2SQLPipeline

pytestmark = pytest.mark.unit


def _client(pipeline):
    router = APIRouter()
    text2sql_http_api(router, pipeline)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _pipeline_with_llm(sql="SELECT SUM(od.amount) AS gmv"):
    llm = MagicMock()
    llm.generate.return_value = sql
    return Text2SQLPipeline(build_order_domain_model(), llm=llm)


def test_text2sql_returns_envelope():
    pipeline = _pipeline_with_llm()
    response = _client(pipeline).post("/text2sql", json={"question": "上个月成交额是多少"})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["sql"] == "SELECT SUM(od.amount) AS gmv"
    assert "GMV" in body["resolved_terms"]
    assert set(body["tables"]) == {"order_detail", "order"}
    assert body["metrics"] == ["gmv"]


def test_text2sql_sql_none_when_no_llm():
    pipeline = Text2SQLPipeline(build_order_domain_model())
    response = _client(pipeline).post("/text2sql", json={"question": "订单量"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["sql"] is None


def test_text2sql_error_returns_500():
    pipeline = MagicMock()
    pipeline.answer.side_effect = RuntimeError("boom")

    response = _client(pipeline).post("/text2sql", json={"question": "x"})
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_service_answer_maps_result():
    llm = MagicMock()
    llm.generate.return_value = "SELECT 1"
    pipeline = Text2SQLPipeline(build_order_domain_model(), llm=llm)

    resp = Text2SQLApi.answer(Text2SQLRequest(question="订单量"), pipeline)

    assert resp.status == "succeeded"
    assert resp.sql == "SELECT 1"
    assert resp.metrics == ["order_count"]
