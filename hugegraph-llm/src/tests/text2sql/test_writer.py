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

"""Tests for the HugeGraph write path (write_semantic_graph)."""

import json
from unittest.mock import MagicMock

import pytest

from hugegraph_llm.flows import FlowName
from hugegraph_llm.text2sql.examples import build_order_domain_model
from hugegraph_llm.text2sql.writer import build_commit_payload, write_semantic_graph

pytestmark = pytest.mark.unit


def test_build_commit_payload_shape():
    data_str, schema_str = build_commit_payload(build_order_domain_model())

    data = json.loads(data_str)
    schema = json.loads(schema_str)

    assert set(data.keys()) == {"vertices", "edges"}
    assert set(schema.keys()) == {"propertykeys", "vertexlabels", "edgelabels"}
    assert len(schema["vertexlabels"]) == 8
    assert len(schema["edgelabels"]) == 10


def test_build_commit_payload_vertices_have_label_and_properties():
    data_str, _ = build_commit_payload(build_order_domain_model())
    vertices = json.loads(data_str)["vertices"]
    for vertex in vertices:
        assert "id" in vertex and "label" in vertex and "properties" in vertex


def test_write_semantic_graph_schedules_import_flow():
    scheduler = MagicMock()
    scheduler.schedule_flow.return_value = '{"ok": true}'

    result = write_semantic_graph(build_order_domain_model(), scheduler=scheduler)

    assert result == '{"ok": true}'
    scheduler.schedule_flow.assert_called_once()
    args = scheduler.schedule_flow.call_args.args
    assert args[0] == FlowName.IMPORT_GRAPH_DATA
    # data_str is {vertices, edges}, schema_str is the constraint schema.
    assert json.loads(args[1]).keys() == {"vertices", "edges"}
    assert json.loads(args[2]).keys() == {"propertykeys", "vertexlabels", "edgelabels"}
