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

import json
from unittest.mock import patch

import pytest
from fastapi import APIRouter, FastAPI, status
from fastapi.testclient import TestClient

from hugegraph_llm.api.feishu_ingest_api import FeishuIngestApi, feishu_ingest_http_api
from hugegraph_llm.api.models.feishu_ingest_requests import FeishuIngestRequest
from hugegraph_llm.api.models.feishu_ingest_responses import FeishuIngestResponse
from hugegraph_llm.flows.feishu_ingest import FeishuIngestResult

pytestmark = pytest.mark.unit

INLINE_SCHEMA = {"vertexlabels": [], "edgelabels": []}


def _feishu_client():
    router = APIRouter()
    feishu_ingest_http_api(router)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _ingest_result(**overrides):
    fields = {
        "space_id": "space_1",
        "mode": "full",
        "doc_count": 2,
        "text_count": 2,
        "docs": [{"doc_id": "feishu:t1", "title": "Doc"}],
        "graph_result": json.dumps({"vertices": [{"id": "1"}], "edges": []}),
        "vector_result": json.dumps({"chunk_count": 5}),
    }
    fields.update(overrides)
    return FeishuIngestResult(**fields)


@patch("hugegraph_llm.api.feishu_ingest_api.FeishuIngestService.ingest_wiki")
def test_feishu_ingest_returns_envelope(mock_ingest):
    mock_ingest.return_value = _ingest_result()

    response = _feishu_client().post(
        "/feishu/ingest",
        json={"space_id": "space_1", "app_id": "a", "app_secret": "s", "schema": INLINE_SCHEMA},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] == "succeeded"
    result = body["result"]
    assert result["space_id"] == "space_1"
    assert result["doc_count"] == 2
    assert result["text_count"] == 2
    assert result["graph_result"] == {"vertices": [{"id": "1"}], "edges": []}
    assert result["vector_result"] == {"chunk_count": 5}


@patch("hugegraph_llm.api.feishu_ingest_api.FeishuIngestService.ingest_wiki")
def test_feishu_ingest_forwards_request_fields(mock_ingest):
    mock_ingest.return_value = _ingest_result()

    _feishu_client().post(
        "/feishu/ingest",
        json={
            "space_id": "space_1",
            "app_id": "app_1",
            "app_secret": "secret_1",
            "mode": "incremental",
            "since_timestamp": 123456,
            "schema": INLINE_SCHEMA,
            "language": "en",
            "split_type": "paragraph",
        },
    )

    _, kwargs = mock_ingest.call_args
    assert kwargs["space_id"] == "space_1"
    assert kwargs["app_id"] == "app_1"
    assert kwargs["app_secret"] == "secret_1"
    assert kwargs["mode"] == "incremental"
    assert kwargs["since_timestamp"] == 123456
    assert kwargs["language"] == "en"
    assert kwargs["split_type"] == "paragraph"
    assert kwargs["graph_schema"] == json.dumps(INLINE_SCHEMA, ensure_ascii=False)


@patch("hugegraph_llm.api.feishu_ingest_api.FeishuIngestService.ingest_wiki")
def test_feishu_ingest_build_index_only_needs_no_schema(mock_ingest):
    mock_ingest.return_value = _ingest_result(graph_result=None)

    response = _feishu_client().post(
        "/feishu/ingest",
        json={"space_id": "space_1", "build_graph": False, "build_index": True},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["result"]["graph_result"] is None


def test_feishu_ingest_incremental_requires_timestamp():
    response = _feishu_client().post(
        "/feishu/ingest",
        json={"space_id": "space_1", "mode": "incremental", "schema": INLINE_SCHEMA},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_feishu_ingest_build_graph_requires_schema():
    response = _feishu_client().post("/feishu/ingest", json={"space_id": "space_1", "build_graph": True})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_feishu_ingest_rejects_invalid_schema():
    response = _feishu_client().post(
        "/feishu/ingest",
        json={"space_id": "space_1", "schema": {"vertexlabels": []}},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_feishu_ingest_rejects_both_flags_false():
    response = _feishu_client().post(
        "/feishu/ingest",
        json={"space_id": "space_1", "build_graph": False, "build_index": False},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@patch("hugegraph_llm.api.feishu_ingest_api.FeishuIngestService.ingest_wiki")
def test_feishu_ingest_ingest_error_returns_500(mock_ingest):
    mock_ingest.side_effect = RuntimeError("lark api failed")

    response = _feishu_client().post(
        "/feishu/ingest",
        json={"space_id": "space_1", "schema": INLINE_SCHEMA},
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@patch("hugegraph_llm.api.feishu_ingest_api.FeishuIngestService.ingest_wiki")
def test_service_ingest_sync_builds_envelope(mock_ingest):
    mock_ingest.return_value = _ingest_result()

    resp = FeishuIngestApi.ingest_sync(
        FeishuIngestRequest(space_id="space_1", app_id="a", app_secret="s", schema=INLINE_SCHEMA)
    )

    assert isinstance(resp, FeishuIngestResponse)
    assert resp.status == "succeeded"
    assert resp.result.doc_count == 2
    assert resp.result.graph_result == {"vertices": [{"id": "1"}], "edges": []}
