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

"""Tests for the constraint schema (Text2SQLSchema)."""

import pytest

from hugegraph_llm.text2sql.schema import Text2SQLSchema

pytestmark = pytest.mark.unit


def test_schema_is_valid():
    assert Text2SQLSchema.validate() == []


def test_to_hugegraph_dict_has_required_keys():
    schema = Text2SQLSchema.to_hugegraph_dict()
    assert set(schema.keys()) == {"propertykeys", "vertexlabels", "edgelabels"}
    assert len(schema["vertexlabels"]) == 8
    assert len(schema["edgelabels"]) == 10


def test_primary_keys_are_non_empty():
    for vertex in Text2SQLSchema.VERTEX_LABELS:
        assert vertex["primary_keys"], f"{vertex['name']} must define primary_keys"


def test_edge_endpoints_are_valid_labels():
    names = set(Text2SQLSchema.vertex_label_names())
    for edge in Text2SQLSchema.EDGE_LABELS:
        assert edge["source_label"] in names
        assert edge["target_label"] in names
