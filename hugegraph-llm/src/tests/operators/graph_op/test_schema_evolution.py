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

"""Tests for the schema evolution component."""

from hugegraph_llm.operators.graph_op.schema_evolution import (
    SchemaEvolutionDraft,
    SchemaEvolutionStore,
    apply_schema_draft,
    collect_schema_candidates,
)

BASE_SCHEMA = {
    "vertexlabels": [{"name": "Person", "properties": ["name"], "primary_keys": ["name"]}],
    "edgelabels": [{"name": "works_at", "source_label": "Person", "target_label": "Org", "properties": []}],
    "propertykeys": [{"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"}],
}

DISCARDED = [
    {"type": "vertex", "label": "Skill", "properties": {"name": "Python", "level": "expert"}},
    {"type": "vertex", "label": "Skill", "properties": {"name": "Java"}},
    {"type": "edge", "label": "has_skill", "outVLabel": "Person", "inVLabel": "Skill", "properties": {"years": "5"}},
    # Existing label — should be ignored, not collected as a candidate.
    {"type": "vertex", "label": "Person", "properties": {"name": "Alice"}},
]


def test_collect_schema_candidates_groups_vertex_and_edge():
    draft = collect_schema_candidates(BASE_SCHEMA, DISCARDED)

    assert draft.total_discarded == 4
    assert len(draft.new_vertex_labels) == 1
    skill = draft.new_vertex_labels[0]
    assert skill["name"] == "Skill"
    assert skill["sample_count"] == 2
    assert skill["example_properties"]["name"] == ["Python", "Java"]
    assert skill["example_properties"]["level"] == ["expert"]

    assert len(draft.new_edge_labels) == 1
    has_skill = draft.new_edge_labels[0]
    assert has_skill["name"] == "has_skill"
    assert has_skill["source_label"] == "Person"
    assert has_skill["target_label"] == "Skill"
    assert has_skill["sample_count"] == 1
    assert has_skill["example_properties"]["years"] == ["5"]


def test_collect_schema_candidates_skips_existing_labels():
    draft = collect_schema_candidates(
        BASE_SCHEMA,
        [{"type": "vertex", "label": "Person", "properties": {"name": "Bob"}}],
    )
    assert draft.new_vertex_labels == []
    assert draft.new_edge_labels == []


def test_collect_schema_candidates_caps_sample_values():
    items = [{"type": "vertex", "label": "Skill", "properties": {"name": f"v{i}"}} for i in range(10)]
    draft = collect_schema_candidates(BASE_SCHEMA, items)
    samples = draft.new_vertex_labels[0]["example_properties"]["name"]
    assert len(samples) == 3  # capped at _MAX_SAMPLE_VALUES


def test_apply_schema_draft_merges_and_declares_props():
    draft = collect_schema_candidates(BASE_SCHEMA, DISCARDED)
    new_schema = apply_schema_draft(BASE_SCHEMA, draft)

    vertex_names = {vl["name"] for vl in new_schema["vertexlabels"]}
    assert "Skill" in vertex_names
    skill = next(vl for vl in new_schema["vertexlabels"] if vl["name"] == "Skill")
    assert skill["primary_keys"] == ["name"]
    assert set(skill["properties"]) == {"name", "level"}

    edge_names = {el["name"] for el in new_schema["edgelabels"]}
    assert "has_skill" in edge_names
    has_skill = next(el for el in new_schema["edgelabels"] if el["name"] == "has_skill")
    assert has_skill["source_label"] == "Person"
    assert has_skill["target_label"] == "Skill"
    assert has_skill["properties"] == ["years"]

    prop_names = {pk["name"] for pk in new_schema["propertykeys"]}
    assert {"name", "level", "years"} <= prop_names


def test_apply_schema_draft_does_not_mutate_input():
    draft = collect_schema_candidates(BASE_SCHEMA, DISCARDED)
    apply_schema_draft(BASE_SCHEMA, draft)
    assert BASE_SCHEMA["vertexlabels"] == [{"name": "Person", "properties": ["name"], "primary_keys": ["name"]}]


def test_store_records_and_lists_versions(tmp_path):
    path = str(tmp_path / "schema_evolution.json")
    store = SchemaEvolutionStore(path)

    draft1 = SchemaEvolutionDraft(new_vertex_labels=[{"name": "Skill", "sample_count": 1}])
    draft2 = SchemaEvolutionDraft(new_edge_labels=[{"name": "has_skill", "sample_count": 1}])

    r1 = store.record(draft1)
    r2 = store.record(draft2)

    assert r1["version"] == 1
    assert r2["version"] == 2

    records = store.list()
    assert len(records) == 2
    assert store.latest()["version"] == 2
    assert store.latest()["draft"]["new_edge_labels"][0]["name"] == "has_skill"


def test_store_empty_returns_none(tmp_path):
    store = SchemaEvolutionStore(str(tmp_path / "none.json"))
    assert store.list() == []
    assert store.latest() is None
