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

"""Connector contract tests (milestone M1). Pure logic -- no server needed."""

import pytest

from hugegraph_llm.semantic_layer.connectors import (
    SourceConnector,
    StateError,
    make_id,
)


# -- make_id ---------------------------------------------------------------


def test_make_id_is_deterministic():
    assert make_id("table", "dw.orders") == "table:dw.orders"
    assert make_id("table", "dw.orders") == make_id("table", "dw.orders")


def test_make_id_skips_empty_parts():
    assert make_id("table", "", "dw.orders") == "table:dw.orders"
    assert make_id("term", None, "月活") == "term:月活"


def test_make_id_rejects_all_empty():
    with pytest.raises(ValueError):
        make_id("", None)


# -- stage ordering --------------------------------------------------------


class _RecordingConnector(SourceConnector):
    """Minimal connector used to exercise the inherited orchestration."""

    name = "recording"

    def __init__(self, rows=None):
        super().__init__()
        self._rows = rows or [{"name": "orders"}]
        self.loaded_records = []

    def extract(self):
        return list(self._rows)

    def transform(self, raw):
        return [{"id": make_id("table", r["name"]), "name": r["name"]} for r in raw]

    def load(self, records):
        self.loaded_records = records
        return len(records)


def test_ingest_runs_all_stages():
    connector = _RecordingConnector([{"name": "orders"}, {"name": "customers"}])
    summary = connector.ingest()
    assert summary == {
        "connector": "recording",
        "extracted": 2,
        "transformed": 2,
        "loaded": 2,
    }
    assert [r["id"] for r in connector.loaded_records] == [
        "table:orders",
        "table:customers",
    ]


def test_require_extracted_before_extract_raises():
    with pytest.raises(StateError):
        _RecordingConnector().require_extracted()


def test_require_transformed_before_transform_raises():
    connector = _RecordingConnector()
    connector.extract()
    with pytest.raises(StateError):
        connector.require_transformed()


def test_ingest_is_idempotent_for_same_source():
    """Re-ingesting the same source must not create duplicate ids."""
    connector = _RecordingConnector([{"name": "orders"}])
    connector.ingest()
    first = [r["id"] for r in connector.loaded_records]
    connector.ingest()
    second = [r["id"] for r in connector.loaded_records]
    assert first == second == ["table:orders"]
