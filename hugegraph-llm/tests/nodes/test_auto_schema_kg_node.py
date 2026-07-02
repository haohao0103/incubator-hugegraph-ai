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

"""Tests for AutoSchemaKGNode integration in incremental indexing flows."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pycgraph import CStatus

from hugegraph_llm.nodes.llm_node.auto_schema_kg_node import AutoSchemaKGNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


def _schema_response() -> str:
    return json.dumps({
        "propertykeys": [{"name": "name", "data_type": "text", "cardinality": "single"}],
        "vertexlabels": [{
            "name": "Person",
            "properties": ["name"],
            "primary_keys": ["name"],
            "nullable_keys": [],
        }],
        "edgelabels": [],
    })


def _make_node(llm_response: str):
    node = AutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM(llm_response))
    return node


def test_reuses_existing_schema_dict():
    from hugegraph_llm.operators.llm_op.auto_schema_kg import AutoSchemaKGOperator
    node = AutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM("{}"))

    existing = {"vertexlabels": [{"name": "Book"}], "edgelabels": [], "propertykeys": []}
    result = node.operator_schedule({"schema": existing})
    assert result["schema"] == existing


def test_parses_wk_input_schema_json():
    from hugegraph_llm.operators.llm_op.auto_schema_kg import AutoSchemaKGOperator
    node = AutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.wk_input.schema = _schema_response()
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM("{}"))

    result = node.operator_schedule({})
    assert result["schema"]["vertexlabels"][0]["name"] == "Person"


def test_invalid_wk_input_schema_returns_error():
    from hugegraph_llm.operators.llm_op.auto_schema_kg import AutoSchemaKGOperator
    node = AutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.wk_input.schema = "{not json"
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM("{}"))

    result = node.operator_schedule({})
    assert isinstance(result, CStatus) and result.isErr()


def test_infers_schema_from_texts():
    from hugegraph_llm.operators.llm_op.auto_schema_kg import AutoSchemaKGOperator
    node = AutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.wk_input.texts = ["Alice knows Bob."]
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM(_schema_response()))

    result = node.operator_schedule({})
    assert isinstance(result, dict)
    assert result["schema"]["vertexlabels"][0]["name"] == "Person"
    assert "schema_draft" in result


def test_empty_texts_returns_error():
    from hugegraph_llm.operators.llm_op.auto_schema_kg import AutoSchemaKGOperator
    node = AutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.wk_input.texts = ["", "   "]
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM(_schema_response()))

    result = node.operator_schedule({})
    assert isinstance(result, CStatus) and result.isErr()
