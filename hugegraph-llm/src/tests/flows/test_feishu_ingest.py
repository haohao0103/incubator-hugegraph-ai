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

"""Tests for the Feishu ingestion bridge (``FeishuIngestService``)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.document.document_loader import Document
from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.feishu_ingest import (
    MODE_FULL,
    MODE_INCREMENTAL,
    FeishuIngestService,
)

pytestmark = pytest.mark.unit

SPACE_ID = "space_123"


def _doc(content, doc_token, title="Doc", updated_time=100):
    return Document(
        content=content,
        metadata={
            "source_type": "feishu_wiki",
            "source": "feishu",
            "title": title,
            "doc_token": doc_token,
            "node_token": f"node_{doc_token}",
            "updated_time": updated_time,
            "url": f"https://open.feishu.cn/wiki/{doc_token}",
        },
        doc_id=f"feishu:{doc_token}",
    )


def _connector_with(docs):
    connector = MagicMock()
    connector.load_wiki_space.return_value = docs
    connector.load_wiki_space_incremental.return_value = docs
    return connector


def _scheduler_mock():
    scheduler = MagicMock()
    scheduler.schedule_flow.return_value = '{"ok": true}'
    return scheduler


def test_docs_to_texts_extracts_content_and_skips_empty():
    docs = [
        _doc("# Heading\n\nbody", "t1"),
        _doc("   ", "t2"),
        _doc("", "t3"),
        _doc("plain text", "t4"),
    ]
    assert FeishuIngestService.docs_to_texts(docs) == ["# Heading\n\nbody", "plain text"]


def test_docs_summary_collects_provenance_metadata():
    docs = [_doc("body", "t1", title="Hello", updated_time=12345)]
    summary = FeishuIngestService._docs_summary(docs)
    assert summary == [
        {
            "doc_id": "feishu:t1",
            "title": "Hello",
            "doc_token": "t1",
            "node_token": "node_t1",
            "url": "https://open.feishu.cn/wiki/t1",
            "updated_time": 12345,
        }
    ]


def test_ingest_wiki_full_mode_runs_graph_and_index():
    connector = _connector_with([_doc("content", "t1"), _doc("content 2", "t2")])
    scheduler = _scheduler_mock()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        result = FeishuIngestService.ingest_wiki(
            SPACE_ID,
            connector=connector,
            graph_schema='{"vertexlabels": [], "edgelabels": []}',
        )

    assert result.mode == MODE_FULL
    assert result.doc_count == 2
    assert result.text_count == 2
    assert result.graph_result == '{"ok": true}'
    assert result.vector_result == '{"ok": true}'
    connector.load_wiki_space.assert_called_once_with(SPACE_ID)
    # Two flows scheduled: graph extract then vector index.
    assert scheduler.schedule_flow.call_count == 2
    flow_names = [call.args[0] for call in scheduler.schedule_flow.call_args_list]
    assert flow_names == [FlowName.GRAPH_EXTRACT, FlowName.BUILD_VECTOR_INDEX]
    # Graph extract receives the plain-text list.
    graph_call = scheduler.schedule_flow.call_args_list[0]
    assert graph_call.args[2] == ["content", "content 2"]
    vector_call = scheduler.schedule_flow.call_args_list[1]
    assert vector_call.args[1] == ["content", "content 2"]


def test_ingest_wiki_incremental_mode():
    connector = _connector_with([_doc("content", "t1")])
    scheduler = _scheduler_mock()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        result = FeishuIngestService.ingest_wiki(
            SPACE_ID,
            connector=connector,
            mode=MODE_INCREMENTAL,
            since_timestamp=500,
            graph_schema='{"vertexlabels": [], "edgelabels": []}',
        )

    assert result.mode == MODE_INCREMENTAL
    connector.load_wiki_space_incremental.assert_called_once_with(SPACE_ID, 500)


def test_ingest_wiki_incremental_requires_timestamp():
    with pytest.raises(ValueError, match="since_timestamp"):
        FeishuIngestService.ingest_wiki(SPACE_ID, connector=_connector_with([]), mode=MODE_INCREMENTAL)


def test_ingest_wiki_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        FeishuIngestService.ingest_wiki(SPACE_ID, connector=_connector_with([]), mode="bad")


def test_ingest_wiki_build_graph_requires_schema():
    with pytest.raises(ValueError, match="graph_schema"):
        FeishuIngestService.ingest_wiki(SPACE_ID, connector=_connector_with([_doc("x", "t1")]), build_graph=True)


def test_ingest_wiki_empty_texts_skips_flows():
    connector = _connector_with([])
    scheduler = _scheduler_mock()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        result = FeishuIngestService.ingest_wiki(SPACE_ID, connector=connector)

    assert result.doc_count == 0
    assert result.text_count == 0
    scheduler.schedule_flow.assert_not_called()


# ---------------------------------------------------------------------------
# ingest_wiki_full (full KG pipeline) tests
# ---------------------------------------------------------------------------

INLINE_SCHEMA = '{"vertexlabels": [], "edgelabels": []}'
EXTRACTED_GRAPH = {
    "vertices": [
        {"id": "p:A", "label": "Person", "properties": {"name": "Alice"}},
        {"id": "p:B", "label": "Person", "properties": {"name": "Alice"}},
    ],
    "edges": [{"label": "knows", "outV": "p:E", "inV": "p:B", "properties": {}}],
}
RESOLVED = {
    "vertices": [{"id": "p:A", "label": "Person", "properties": {"name": "Alice"}}],
    "edges": [{"label": "knows", "outV": "p:E", "inV": "p:A", "properties": {}}],
    "resolution_result": {"merged_count": 1, "deprecated_vids": ["p:B"], "edges_migrated": 1},
}


def _full_scheduler():
    scheduler = MagicMock()

    def se(name, *args, **kwargs):
        if name == FlowName.BUILD_SCHEMA:
            return INLINE_SCHEMA
        if name == FlowName.GRAPH_EXTRACT:
            return json.dumps(EXTRACTED_GRAPH)
        if name == FlowName.IMPORT_GRAPH_DATA:
            return '{"imported": true}'
        if name == FlowName.BUILD_VECTOR_INDEX:
            return '{"indexed": true}'
        return "{}"

    scheduler.schedule_flow.side_effect = se
    return scheduler


def _resolver_mock():
    resolver = MagicMock()
    resolver.resolve_in_memory_pure.return_value = RESOLVED
    return resolver


def test_ingest_wiki_full_runs_extract_resolve_commit_index():
    connector = _connector_with([_doc("content", "t1")])
    scheduler = _full_scheduler()
    resolver = _resolver_mock()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        result = FeishuIngestService.ingest_wiki_full(
            SPACE_ID,
            connector=connector,
            graph_schema=INLINE_SCHEMA,
            resolve=True,
            resolver=resolver,
            commit=True,
            build_index=True,
        )

    flow_names = [c.args[0] for c in scheduler.schedule_flow.call_args_list]
    assert flow_names == [FlowName.GRAPH_EXTRACT, FlowName.IMPORT_GRAPH_DATA, FlowName.BUILD_VECTOR_INDEX]
    resolver.resolve_in_memory_pure.assert_called_once()
    assert result.vertex_count == 1
    assert result.edge_count == 1
    assert result.resolution_result == RESOLVED["resolution_result"]
    assert result.commit_result == '{"imported": true}'
    assert result.vector_result == '{"indexed": true}'
    # Commit receives the deduplicated graph, not the raw extracted one.
    commit_call = scheduler.schedule_flow.call_args_list[1]
    assert json.loads(commit_call.args[1]) == {
        "vertices": RESOLVED["vertices"],
        "edges": RESOLVED["edges"],
    }


def test_ingest_wiki_full_auto_schema_calls_build_schema():
    connector = _connector_with([_doc("content", "t1")])
    scheduler = _full_scheduler()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        result = FeishuIngestService.ingest_wiki_full(
            SPACE_ID,
            connector=connector,
            auto_schema=True,
            commit=False,
            build_index=False,
        )

    flow_names = [c.args[0] for c in scheduler.schedule_flow.call_args_list]
    assert flow_names == [FlowName.BUILD_SCHEMA, FlowName.GRAPH_EXTRACT]
    assert result.vertex_count == 2
    assert result.edge_count == 1
    assert result.commit_result is None


def test_ingest_wiki_full_requires_schema_or_auto():
    connector = _connector_with([_doc("content", "t1")])
    with pytest.raises(ValueError, match="graph_schema"):
        FeishuIngestService.ingest_wiki_full(SPACE_ID, connector=connector)


def test_ingest_wiki_full_resolve_requires_resolver():
    connector = _connector_with([_doc("content", "t1")])
    scheduler = _full_scheduler()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        with pytest.raises(ValueError, match="resolver"):
            FeishuIngestService.ingest_wiki_full(
                SPACE_ID,
                connector=connector,
                graph_schema=INLINE_SCHEMA,
                resolve=True,
                commit=False,
                build_index=False,
            )


def test_ingest_wiki_full_empty_texts_skips_flows():
    connector = _connector_with([])
    scheduler = _full_scheduler()
    with patch("hugegraph_llm.flows.feishu_ingest.SchedulerSingleton.get_instance", return_value=scheduler):
        result = FeishuIngestService.ingest_wiki_full(SPACE_ID, connector=connector, graph_schema=INLINE_SCHEMA)

    assert result.text_count == 0
    scheduler.schedule_flow.assert_not_called()
