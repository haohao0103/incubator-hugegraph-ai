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

"""Tests for multimodal AutoSchemaKG conversion and node."""

import json

import pytest
from pycgraph import CStatus

from hugegraph_llm.nodes.llm_node.multimodal_auto_schema_kg_node import MultimodalAutoSchemaKGNode
from hugegraph_llm.operators.llm_op.auto_schema_kg import (
    AutoSchemaKGOperator,
    MultimodalAutoSchemaKGOperator,
    multimodal_result_to_document,
)
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


def test_multimodal_result_to_document_text_blocks():
    pdf_result = {
        "pages": [
            {
                "page_num": 0,
                "text_blocks": [
                    {"text": "Introduction", "is_heading": True},
                    {"text": "This is body text."},
                    {"text": "", "is_heading": False},  # empty block should be skipped
                ],
            }
        ]
    }
    vlm = [
        {
            "image_id": "img_1",
            "caption": "A chart",
            "detailed_description": "Revenue chart",
            "chart_type": "bar",
            "key_insights": ["Q1 grew"],
            "related_keywords": ["revenue"],
            "object_labels": ["bar", "chart"],
        }
    ]
    doc = multimodal_result_to_document(pdf_extraction_result=pdf_result, vlm_descriptions=vlm)
    assert "Introduction" in doc
    assert "Page 1" in doc
    assert "A chart" in doc
    assert "Revenue chart" in doc
    assert "bar" in doc and "chart" in doc


def test_multimodal_result_to_document_standalone_text_blocks():
    blocks = [
        {"text": "Heading", "is_heading": True},
        {"content": "Body via content key.", "is_heading": False},
        {"text": "", "is_heading": False},
    ]
    doc = multimodal_result_to_document(text_blocks=blocks)
    assert "Heading" in doc
    assert "Body via content key" in doc


def test_multimodal_operator_no_input_raises():
    llm = FakeLLM(_schema_response())
    op = MultimodalAutoSchemaKGOperator(llm=llm, allow_commit=False)
    with pytest.raises(ValueError, match="No document"):
        op.run()


def test_multimodal_operator_with_vlm_descriptions():
    llm = FakeLLM(_schema_response())
    op = MultimodalAutoSchemaKGOperator(llm=llm, allow_commit=False)
    vlm = [{"image_id": "img_1", "caption": "A person", "detailed_description": "Alice is smiling"}]
    result = op.run(vlm_descriptions=vlm)
    assert result.draft.vertex_labels[0].name == "Person"
    assert "A person" in result.draft.source_document


def test_multimodal_operator_appends_to_document():
    llm = FakeLLM(_schema_response())
    op = MultimodalAutoSchemaKGOperator(llm=llm, allow_commit=False)
    vlm = [{"image_id": "img_1", "caption": "Chart", "detailed_description": "Revenue"}]
    result = op.run(document="Alice knows Bob.", vlm_descriptions=vlm)
    assert "Alice knows Bob." in result.draft.source_document
    assert "Revenue" in result.draft.source_document


def test_multimodal_node_reuses_existing_schema():
    node = MultimodalAutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.context = WkFlowState()
    node.operator = AutoSchemaKGOperator(llm=FakeLLM("{}"), allow_commit=False)

    existing = {"vertexlabels": [{"name": "Book"}], "edgelabels": [], "propertykeys": []}
    result = node.operator_schedule({"schema": existing})
    assert result["schema"] == existing


def test_multimodal_node_infers_from_vlm():
    node = MultimodalAutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.context = WkFlowState()
    node.operator = MultimodalAutoSchemaKGOperator(llm=FakeLLM(_schema_response()), allow_commit=False)

    data = {
        "vlm_descriptions": [
            {"image_id": "img_1", "caption": "A person", "detailed_description": "Alice"}
        ]
    }
    result = node.operator_schedule(data)
    assert isinstance(result, dict)
    assert result["schema"]["vertexlabels"][0]["name"] == "Person"
    assert "schema_draft" in result


def test_multimodal_node_no_input_returns_error():
    node = MultimodalAutoSchemaKGNode()
    node.wk_input = WkFlowInput()
    node.context = WkFlowState()
    node.operator = MultimodalAutoSchemaKGOperator(llm=FakeLLM(_schema_response()), allow_commit=False)

    result = node.operator_schedule({})
    assert isinstance(result, CStatus) and result.isErr()
