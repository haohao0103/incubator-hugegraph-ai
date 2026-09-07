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

"""Schema definition tests. Pure logic -- no HugeGraph server required.

These guard the constraints that are expensive to discover against a live
server (immutable property types, one source/target pair per edge label,
unsupported index types).
"""

import pytest

from hugegraph_llm.semantic_layer import schema_def
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.semantic_layer.schema_def import (
    build_schema_dict,
    coerce_value,
    unknown_property_keys,
    validate,
)


def test_definition_is_self_consistent():
    assert validate() is None


def test_no_undeclared_property_keys():
    assert unknown_property_keys()["undeclared"] == []


def test_every_vertex_label_has_a_name_property():
    """`name` drives the automatic {label}ByName index and `has('name')` filters."""
    for label, props in schema_def.VERTEX_LABELS.items():
        assert "name" in props, f"{label} missing 'name'"


def test_all_vertices_use_customize_string_ids():
    """Logical ids (`table:dw.orders`) must be stable across re-ingest."""
    schema = build_schema_dict()
    for vertex in schema["vertexlabels"]:
        assert vertex["id_strategy"] == "CUSTOMIZE_STRING"


def test_edge_labels_are_unique():
    """HugeGraph allows one source->target pair per edge label."""
    names = [name.value for name, _s, _t, _p in schema_def.EDGE_LABELS]
    assert len(names) == len(set(names))


def test_all_edge_labels_are_declared_in_enum():
    declared = {e.value for e in EdgeLabel}
    for name, _s, _t, _p in schema_def.EDGE_LABELS:
        assert name.value in declared


def test_all_vertex_labels_are_declared_in_enum():
    declared = {v.value for v in VertexLabel}
    assert set(schema_def.VERTEX_LABELS) == set(VertexLabel)


def test_no_search_index():
    """SEARCH (full-text) is declared but unsupported on HugeGraph 1.7."""
    for idx in schema_def.INDEX_LABELS:
        assert idx["index_type"] != "SEARCH"


def test_only_supported_property_data_types():
    supported = {"TEXT", "LONG", "DOUBLE", "BOOLEAN"}
    for name, (dtype, _card) in schema_def.PROPERTY_KEYS.items():
        assert dtype in supported, f"{name}: {dtype}"


def test_build_schema_dict_shape():
    schema = build_schema_dict()
    assert {"propertykeys", "vertexlabels", "edgelabels", "indexes"} <= set(schema)
    assert len(schema["vertexlabels"]) == len(schema_def.VERTEX_LABELS)
    assert len(schema["edgelabels"]) == len(schema_def.EDGE_LABELS)

    for edge in schema["edgelabels"]:
        assert {"name", "source_label", "target_label", "properties"} <= set(edge)


def test_every_index_targets_a_declared_label():
    vertex_names = {v.value for v in VertexLabel}
    edge_names = {e.value for e in EdgeLabel}
    for idx in schema_def.INDEX_LABELS:
        base, on = idx["base_label"], idx["on"]
        if on == "vertex":
            assert base in vertex_names
        else:
            assert base in edge_names


def test_indexed_properties_are_declared_on_their_label():
    """An index on a property the label does not carry fails on the server."""
    vertex_props = {
        label.value: set(props)
        for label, props in schema_def.VERTEX_LABELS.items()
    }
    edge_props = {
        name.value: set(props) for name, _s, _t, props in schema_def.EDGE_LABELS
    }
    for idx in schema_def.INDEX_LABELS:
        pool = vertex_props if idx["on"] == "vertex" else edge_props
        assert idx["field"] in pool[idx["base_label"]]


# -- coerce_value ----------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("row_count", "42", 42),
        ("confidence", "0.95", 0.95),
        ("is_primary_key", "true", True),
        ("is_primary_key", "false", False),
        ("is_primary_key", 1, True),
        ("name", 7, "7"),
    ],
)
def test_coerce_scalar(key, raw, expected):
    assert coerce_value(key, raw) == expected


def test_coerce_list_wraps_scalars():
    assert coerce_value("sample_values", "a") == ["a"]
    assert coerce_value("sample_values", ["a", 1]) == ["a", "1"]


def test_coerce_unknown_key_returns_none():
    assert coerce_value("not_a_property", "x") is None


def test_coerce_bad_number_returns_none():
    assert coerce_value("row_count", "not-a-number") is None


def test_coerce_none_is_preserved_for_scalars():
    assert coerce_value("comment", None) is None
    assert coerce_value("sample_values", None) == []
