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

"""Tests for AutoSchemaKG operator."""

import json
from typing import Any, Dict

import pytest

from hugegraph_llm.operators.llm_op.auto_schema_kg import (
    AutoSchemaKGOperator,
    AutoSchemaKGResult,
    BatchAutoSchemaKGOperator,
    EdgeLabelDef,
    PropertyKeyDef,
    SchemaConflictDetector,
    SchemaDiffCalculator,
    SchemaDraft,
    SchemaMerger,
    SchemaReviewResult,
    VertexLabelDef,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class FakeLLM:
    """Simple LLM that returns canned responses."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list = []
        self.fail = False

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("LLM failed")
        if self.response is None:
            return ""
        return self.response


class FakeCommitClient:
    """Records calls to init_schema_if_need."""

    def __init__(self, raise_error: bool = False):
        self.schemas: list = []
        self.raise_error = raise_error

    def init_schema_if_need(self, schema: Dict[str, Any]) -> None:
        if self.raise_error:
            raise RuntimeError("commit failed")
        self.schemas.append(schema)


def make_valid_schema_response() -> str:
    return json.dumps({
        "propertykeys": [
            {"name": "name", "data_type": "text", "cardinality": "single"},
            {"name": "age", "data_type": "int", "cardinality": "single"},
        ],
        "vertexlabels": [
            {
                "name": "Person",
                "properties": ["name", "age"],
                "primary_keys": ["name"],
                "nullable_keys": ["age"],
            }
        ],
        "edgelabels": [
            {
                "name": "KNOWS",
                "source_label": "Person",
                "target_label": "Person",
                "properties": [],
                "nullable_keys": [],
            }
        ],
    })


# ---------------------------------------------------------------------------
# Core parsing and normalization
# ---------------------------------------------------------------------------


def test_basic_schema_generation_and_commit():
    llm = FakeLLM(make_valid_schema_response())
    commit = FakeCommitClient()
    op = AutoSchemaKGOperator(llm=llm, schema_commit_client=commit)
    result = op.run("Alice is a 30-year-old engineer who knows Bob.")

    assert result.draft.vertex_labels[0].name == "Person"
    assert result.draft.property_keys[0].name == "name"
    assert result.review.approved is True
    assert result.committed is True
    assert len(commit.schemas) == 1

    schema_dict = commit.schemas[0]
    assert schema_dict["propertykeys"][0]["name"] == "name"
    assert schema_dict["vertexlabels"][0]["name"] == "Person"
    assert schema_dict["edgelabels"][0]["name"] == "KNOWS"


def test_markdown_code_block_extraction():
    wrapped = "```json\n" + make_valid_schema_response() + "\n```"
    llm = FakeLLM(wrapped)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Some document.")
    assert result.draft.vertex_labels[0].name == "Person"


def test_extract_json_without_code_block():
    inner = json.dumps({"propertykeys": [], "vertexlabels": [], "edgelabels": []})
    text = f"Some text before {inner} and after"
    extracted = AutoSchemaKGOperator._extract_json(text)  # pylint: disable=protected-access
    assert extracted == inner


def test_invalid_json_raises():
    llm = FakeLLM("not json")
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        op.run("Doc")


def test_empty_llm_response_raises():
    llm = FakeLLM("")
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    with pytest.raises(RuntimeError, match="empty"):
        op.run("Doc")


def test_llm_failure_raises():
    llm = FakeLLM("")
    llm.fail = True
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    with pytest.raises(RuntimeError, match="LLM schema generation failed"):
        op.run("Doc")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_unknown_data_type_defaults_to_text():
    raw = json.dumps({
        "propertykeys": [
            {"name": "score", "data_type": "float64", "cardinality": "single"},
        ],
        "vertexlabels": [
            {"name": "Item", "properties": ["score"], "primary_keys": ["score"]}
        ],
        "edgelabels": [],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    assert result.draft.property_keys[0].data_type == "text"


def test_unknown_cardinality_defaults_to_single():
    raw = json.dumps({
        "propertykeys": [
            {"name": "tags", "data_type": "text", "cardinality": "array"},
        ],
        "vertexlabels": [
            {"name": "Post", "properties": ["tags"], "primary_keys": ["tags"]}
        ],
        "edgelabels": [],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    assert result.draft.property_keys[0].cardinality == "single"


def test_missing_primary_key_property_gets_created():
    raw = json.dumps({
        "propertykeys": [],
        "vertexlabels": [
            {"name": "Company", "properties": ["name"], "primary_keys": ["name"]}
        ],
        "edgelabels": [],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    pk_names = {p.name for p in result.draft.property_keys}
    assert "name" in pk_names


def test_vertex_label_without_primary_key_gets_default():
    raw = json.dumps({
        "propertykeys": [
            {"name": "title", "data_type": "text", "cardinality": "single"},
        ],
        "vertexlabels": [
            {"name": "Book", "properties": ["title"], "primary_keys": []}
        ],
        "edgelabels": [],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    assert result.draft.vertex_labels[0].primary_keys == ["title"]


def test_edge_property_gets_created():
    raw = json.dumps({
        "propertykeys": [
            {"name": "name", "data_type": "text", "cardinality": "single"},
        ],
        "vertexlabels": [
            {"name": "User", "properties": ["name"], "primary_keys": ["name"]}
        ],
        "edgelabels": [
            {"name": "FOLLOWS", "source_label": "User", "target_label": "User", "properties": ["since"]}
        ],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    pk_names = {p.name for p in result.draft.property_keys}
    assert "since" in pk_names


# ---------------------------------------------------------------------------
# Review callback
# ---------------------------------------------------------------------------


def test_reject_callback_prevents_commit():
    llm = FakeLLM(make_valid_schema_response())
    commit = FakeCommitClient()

    def reject(_draft):
        return SchemaReviewResult(approved=False, reason="Too risky")

    op = AutoSchemaKGOperator(llm=llm, schema_commit_client=commit, review_callback=reject)
    result = op.run("Doc")
    assert result.review.approved is False
    assert result.committed is False
    assert len(commit.schemas) == 0


def test_modify_callback_changes_committed_schema():
    llm = FakeLLM(make_valid_schema_response())
    commit = FakeCommitClient()

    modified = SchemaDraft(
        property_keys=[PropertyKeyDef(name="id", data_type="text", cardinality="single")],
        vertex_labels=[VertexLabelDef(name="Product", properties=["id"], primary_keys=["id"])],
        edge_labels=[],
        source_document="Doc",
    )

    def modify(draft):
        return SchemaReviewResult(approved=True, modified_schema=modified)

    op = AutoSchemaKGOperator(llm=llm, schema_commit_client=commit, review_callback=modify)
    result = op.run("Doc")
    assert result.committed is True
    assert len(commit.schemas) == 1
    assert commit.schemas[0]["vertexlabels"][0]["name"] == "Product"


def test_auto_approve_without_callback():
    llm = FakeLLM(make_valid_schema_response())
    commit = FakeCommitClient()
    op = AutoSchemaKGOperator(llm=llm, schema_commit_client=commit)
    result = op.run("Doc")
    assert result.review.approved is True
    assert result.committed is True


def test_allow_commit_false_prevents_commit():
    llm = FakeLLM(make_valid_schema_response())
    commit = FakeCommitClient()
    op = AutoSchemaKGOperator(llm=llm, schema_commit_client=commit, allow_commit=False)
    result = op.run("Doc")
    assert result.committed is False
    assert "allow_commit" in result.commit_error


def test_commit_failure_is_recorded():
    llm = FakeLLM(make_valid_schema_response())
    commit = FakeCommitClient(raise_error=True)
    op = AutoSchemaKGOperator(llm=llm, schema_commit_client=commit)
    result = op.run("Doc")
    assert result.committed is False
    assert "commit failed" in result.commit_error


# ---------------------------------------------------------------------------
# Human-readable rendering and validation warnings
# ---------------------------------------------------------------------------


def test_human_readable_contains_labels_and_warnings():
    draft = SchemaDraft(
        property_keys=[PropertyKeyDef(name="name")],
        vertex_labels=[VertexLabelDef(name="Person", properties=["name"], primary_keys=["name"])],
        edge_labels=[EdgeLabelDef(name="WORKS_AT", source_label="Person", target_label="Company")],
        source_document="Alice works at Acme.",
    )
    text = draft.to_human_readable()
    assert "Person" in text
    assert "WORKS_AT" in text
    assert "Company" in text  # warning: target not in vertex labels


def test_empty_draft_human_readable():
    draft = SchemaDraft()
    text = draft.to_human_readable()
    assert "None inferred" in text


def test_to_dict_roundtrip():
    llm = FakeLLM(make_valid_schema_response())
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    d = result.to_dict()
    assert d["committed"] is False
    assert d["draft"]["vertexlabels"][0]["name"] == "Person"
    assert "human_readable" in d


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_non_string_document_raises():
    op = AutoSchemaKGOperator(llm=FakeLLM("{}"), allow_commit=False)
    with pytest.raises(ValueError, match="must be a string"):
        op.run(123)


def test_empty_document_raises():
    op = AutoSchemaKGOperator(llm=FakeLLM("{}"), allow_commit=False)
    with pytest.raises(ValueError, match="must not be empty"):
        op.run("   ")


# ---------------------------------------------------------------------------
# Integration with real HugeGraph schema commit client shape
# ---------------------------------------------------------------------------


def test_normalize_skips_non_dict_and_duplicate_labels():
    raw = json.dumps({
        "propertykeys": [
            {"name": "name", "data_type": "text", "cardinality": "single"},
            "not-a-dict",
            {"name": "name", "data_type": "text", "cardinality": "single"},  # duplicate
            {"name": "", "data_type": "text", "cardinality": "single"},  # empty
        ],
        "vertexlabels": [
            {"name": "Person", "properties": ["name"], "primary_keys": ["name"]},
            123,
            {"name": "Person"},  # duplicate
        ],
        "edgelabels": [
            {"name": "KNOWS", "source_label": "Person", "target_label": "Person"},
            None,
            {"name": "KNOWS"},  # duplicate
        ],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    assert len(result.draft.property_keys) == 1
    assert len(result.draft.vertex_labels) == 1
    assert len(result.draft.edge_labels) == 1


def test_properties_as_comma_string():
    raw = json.dumps({
        "propertykeys": [
            {"name": "name", "data_type": "text", "cardinality": "single"},
        ],
        "vertexlabels": [
            {"name": "Person", "properties": "name, age", "primary_keys": "name"}
        ],
        "edgelabels": [],
    })
    llm = FakeLLM(raw)
    op = AutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = op.run("Doc")
    assert result.draft.vertex_labels[0].properties == ["name", "age"]


def test_as_string_list_unknown_value_returns_empty():
    assert AutoSchemaKGOperator._as_string_list(123) == []  # pylint: disable=protected-access
    assert AutoSchemaKGOperator._as_string_list(None) == []  # pylint: disable=protected-access


def test_no_commit_client_returns_uncommitted():
    llm = FakeLLM(make_valid_schema_response())
    op = AutoSchemaKGOperator(llm=llm, allow_commit=True)  # no commit client
    result = op.run("Doc")
    assert result.review.approved is True
    assert result.committed is False
    assert "No schema_commit_client" in result.commit_error


def test_validation_warnings_for_undefined_and_bad_references():
    draft = SchemaDraft(
        property_keys=[PropertyKeyDef(name="name")],
        vertex_labels=[
            VertexLabelDef(name="Person", properties=["name", "age"], primary_keys=["name", "id"])
        ],
        edge_labels=[
            EdgeLabelDef(name="WORKS_AT", source_label="Company", target_label="Office", properties=["since"])
        ],
    )
    warnings = draft._validation_warnings()  # pylint: disable=protected-access
    assert any("undefined property 'age'" in w for w in warnings)
    assert any("primary key 'id' not in properties" in w for w in warnings)
    assert any("source 'Company' not in vertex labels" in w for w in warnings)
    assert any("target 'Office' not in vertex labels" in w for w in warnings)
    assert any("undefined property 'since'" in w for w in warnings)


# ---------------------------------------------------------------------------
# Schema merge / diff / conflict detection
# ---------------------------------------------------------------------------


def _person_schema() -> SchemaDraft:
    return SchemaDraft(
        property_keys=[
            PropertyKeyDef(name="name", data_type="text", cardinality="single"),
            PropertyKeyDef(name="age", data_type="int", cardinality="single"),
        ],
        vertex_labels=[VertexLabelDef(name="Person", properties=["name", "age"], primary_keys=["name"])],
        edge_labels=[EdgeLabelDef(name="KNOWS", source_label="Person", target_label="Person")],
    )


def _company_schema() -> SchemaDraft:
    return SchemaDraft(
        property_keys=[
            PropertyKeyDef(name="name", data_type="text", cardinality="single"),
            PropertyKeyDef(name="age", data_type="int", cardinality="single"),
            PropertyKeyDef(name="founded_year", data_type="int", cardinality="single"),
            PropertyKeyDef(name="since", data_type="text", cardinality="single"),
        ],
        vertex_labels=[
            VertexLabelDef(name="Person", properties=["name", "age"], primary_keys=["name"]),
            VertexLabelDef(name="Company", properties=["name", "founded_year"], primary_keys=["name"]),
        ],
        edge_labels=[
            EdgeLabelDef(name="WORKS_AT", source_label="Person", target_label="Company", properties=["since"])
        ],
    )


def test_merge_unions_labels_and_properties():
    merger = SchemaMerger()
    merged, conflicts = merger.merge([_person_schema(), _company_schema()])

    pk_names = {pk.name for pk in merged.property_keys}
    assert pk_names == {"name", "age", "founded_year", "since"}
    assert len(merged.vertex_labels) == 2
    assert len(merged.edge_labels) == 2
    assert not conflicts


def test_merge_detects_incomplete_schema_conflicts():
    """If a per-document schema references labels not defined in that document, conflicts are reported."""
    incomplete = SchemaDraft(
        property_keys=[PropertyKeyDef(name="name")],
        vertex_labels=[VertexLabelDef(name="Company", properties=["name"], primary_keys=["name"])],
        edge_labels=[EdgeLabelDef(name="WORKS_AT", source_label="Person", target_label="Company", properties=["since"])],
    )
    merger = SchemaMerger()
    merged, conflicts = merger.merge([_person_schema(), incomplete])
    assert "Person" in {v.name for v in merged.vertex_labels}
    assert "since" in {p.name for p in merged.property_keys}
    assert any(c.conflict_type == "edge_source_missing" for c in conflicts)
    assert any(c.conflict_type == "undefined_edge_property" for c in conflicts)

def test_merge_detects_property_type_conflict_across_drafts():
    draft_a = SchemaDraft(
        property_keys=[PropertyKeyDef(name="score", data_type="int", cardinality="single")],
        vertex_labels=[VertexLabelDef(name="Item", properties=["score"], primary_keys=["score"])],
    )
    draft_b = SchemaDraft(
        property_keys=[PropertyKeyDef(name="score", data_type="double", cardinality="single")],
        vertex_labels=[VertexLabelDef(name="Item", properties=["score"], primary_keys=["score"])],
    )
    merger = SchemaMerger()
    merged, conflicts = merger.merge([draft_a, draft_b])
    assert merged.property_keys[0].data_type == "int"  # first wins
    assert any(c.conflict_type == "cross_draft_property_type_mismatch" for c in conflicts)


def test_merge_detects_edge_endpoint_conflict():
    draft_a = SchemaDraft(
        vertex_labels=[
            VertexLabelDef(name="Person", properties=["name"], primary_keys=["name"]),
            VertexLabelDef(name="Company", properties=["name"], primary_keys=["name"]),
        ],
        edge_labels=[EdgeLabelDef(name="MANAGES", source_label="Person", target_label="Company")],
    )
    draft_b = SchemaDraft(
        vertex_labels=[
            VertexLabelDef(name="Person", properties=["name"], primary_keys=["name"]),
            VertexLabelDef(name="Company", properties=["name"], primary_keys=["name"]),
        ],
        edge_labels=[EdgeLabelDef(name="MANAGES", source_label="Person", target_label="Person")],
    )
    merger = SchemaMerger()
    _, conflicts = merger.merge([draft_a, draft_b])
    assert any(c.conflict_type == "cross_draft_edge_endpoint_mismatch" for c in conflicts)


def test_intra_draft_conflict_detects_missing_primary_key():
    draft = SchemaDraft(
        property_keys=[],
        vertex_labels=[VertexLabelDef(name="Person", properties=["name"], primary_keys=["id"])],
    )
    detector = SchemaConflictDetector()
    conflicts = detector.detect_intra_draft(draft)
    assert any(c.conflict_type == "undefined_primary_key" for c in conflicts)


def test_intra_draft_conflict_detects_edge_source_missing():
    draft = SchemaDraft(
        property_keys=[PropertyKeyDef(name="name")],
        vertex_labels=[VertexLabelDef(name="Person", properties=["name"], primary_keys=["name"])],
        edge_labels=[EdgeLabelDef(name="WORKS_AT", source_label="Company", target_label="Person")],
    )
    detector = SchemaConflictDetector()
    conflicts = detector.detect_intra_draft(draft)
    assert any(c.conflict_type == "edge_source_missing" for c in conflicts)


def test_diff_calculator_reports_added_and_removed():
    base = _person_schema()
    target = SchemaDraft(
        property_keys=[PropertyKeyDef(name="name"), PropertyKeyDef(name="founded_year", data_type="int")],
        vertex_labels=[VertexLabelDef(name="Company", properties=["name", "founded_year"], primary_keys=["name"])],
    )
    diff = SchemaDiffCalculator().diff(base, target)
    assert len(diff.added_vertex_labels) == 1
    assert diff.added_vertex_labels[0].name == "Company"
    assert len(diff.removed_vertex_labels) == 1
    assert diff.removed_vertex_labels[0].name == "Person"


# ---------------------------------------------------------------------------
# Batch AutoSchemaKG
# ---------------------------------------------------------------------------


def test_batch_infers_and_merges_multiple_documents():
    def llm_response(doc: str) -> str:
        if "Person" in doc or "Alice" in doc:
            return json.dumps({
                "propertykeys": [{"name": "name", "data_type": "text", "cardinality": "single"}],
                "vertexlabels": [{"name": "Person", "properties": ["name"], "primary_keys": ["name"]}],
                "edgelabels": [],
            })
        return json.dumps({
            "propertykeys": [{"name": "name", "data_type": "text", "cardinality": "single"}],
            "vertexlabels": [{"name": "Company", "properties": ["name"], "primary_keys": ["name"]}],
            "edgelabels": [],
        })

    class SwitchingLLM:
        def __init__(self):
            self.prompts: list = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return llm_response(prompt)

    llm = SwitchingLLM()
    batch = BatchAutoSchemaKGOperator(llm=llm, allow_commit=False)
    result = batch.run(["Alice is a person.", "Acme is a company."])

    assert len(result.per_document_results) == 2
    vl_names = {vl.name for vl in result.merged_draft.vertex_labels}
    assert vl_names == {"Person", "Company"}
    assert result.diff_from_first is not None


def test_batch_empty_documents_raises():
    batch = BatchAutoSchemaKGOperator(llm=FakeLLM("{}"), allow_commit=False)
    with pytest.raises(ValueError, match="At least one non-empty document"):
        batch.run(["", "   "])


def test_batch_commit_with_review_callback():
    commit = FakeCommitClient()

    def approve(_draft):
        return SchemaReviewResult(approved=True)

    batch = BatchAutoSchemaKGOperator(
        llm=FakeLLM(make_valid_schema_response()),
        schema_commit_client=commit,
        review_callback=approve,
        allow_commit=True,
    )
    result = batch.run(["Doc one.", "Doc two."])
    assert len(commit.schemas) == 1
    assert result.merged_draft.vertex_labels[0].name == "Person"


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------


def test_gradio_handler_generate_batch_schema_draft():
    from unittest.mock import patch

    fake_llm = FakeLLM(make_valid_schema_response())
    with patch("hugegraph_llm.demo.rag_demo.auto_schema_kg_handlers.LLMs") as mock_llms:
        mock_llms.return_value.get_extract_llm.return_value = fake_llm
        from hugegraph_llm.demo.rag_demo.auto_schema_kg_handlers import generate_batch_schema_draft

        md, schema_json, status = generate_batch_schema_draft(
            "Alice is 30.\n\nBob is 25.",
            instructions="focus on people",
        )
        assert "Person" in md
        assert '"Person"' in schema_json
        assert "Merged schema from 2 document(s)" in status


def test_gradio_handler_empty_input():
    from hugegraph_llm.demo.rag_demo.auto_schema_kg_handlers import generate_batch_schema_draft

    md, schema_json, status = generate_batch_schema_draft("   ")
    assert "Error" in status
    assert not schema_json
