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

"""Constraint schema for the Text2SQL semantic-layer graph.

Mirrors the OpenSPG/KAG philosophy of *schema-constrained* modelling: the
vertex/edge labels are fixed up front, so the graph can only ever express the
``table / column / metric / dimension / term / join / value / query_pattern``
semantic layer — never free-form document entities.  This is the key difference
from an open-extraction document graph.
"""

from typing import Any, Dict, List

# Property keys shared across labels.
PROPERTY_KEYS: List[Dict[str, Any]] = [
    {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "aliases", "data_type": "TEXT", "cardinality": "LIST"},
    {"name": "description", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "data_type", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "role", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "table", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "full_name", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "column", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "on_condition", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "join_type", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "fanout_risk", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "agg_func", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "time_granularity", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "time_column", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "partition_column", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "timezone", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "dedup", "data_type": "BOOLEAN", "cardinality": "SINGLE"},
    {"name": "sql", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "measure", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "dimensions", "data_type": "TEXT", "cardinality": "LIST"},
    {"name": "meaning", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "code", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "operator", "data_type": "TEXT", "cardinality": "SINGLE"},
    {"name": "values", "data_type": "TEXT", "cardinality": "LIST"},
]

VERTEX_LABELS: List[Dict[str, Any]] = [
    {
        "name": "domain",
        "properties": ["name", "description"],
        "primary_keys": ["name"],
        "nullable_keys": [],
    },
    {
        "name": "table",
        "properties": [
            "name",
            "aliases",
            "description",
            "partition_column",
            "time_column",
            "timezone",
        ],
        "primary_keys": ["name"],
        "nullable_keys": ["description", "partition_column", "time_column", "timezone"],
    },
    {
        "name": "column",
        "properties": ["table", "name", "full_name", "data_type", "role", "description"],
        "primary_keys": ["table", "name"],
        "nullable_keys": ["full_name", "role", "description"],
    },
    {
        "name": "term",
        "properties": ["name", "aliases"],
        "primary_keys": ["name"],
        "nullable_keys": [],
    },
    {
        "name": "metric",
        "properties": [
            "name",
            "agg_func",
            "time_granularity",
            "dedup",
            "description",
            "measure",
            "dimensions",
            "time_column",
        ],
        "primary_keys": ["name"],
        "nullable_keys": ["description", "time_granularity", "dimensions", "time_column"],
    },
    {
        "name": "value",
        "properties": ["column", "code", "meaning"],
        "primary_keys": ["column", "code"],
        "nullable_keys": [],
    },
    {
        "name": "filter",
        "properties": ["name", "column", "operator", "values"],
        "primary_keys": ["name"],
        "nullable_keys": [],
    },
    {
        "name": "query_pattern",
        "properties": ["name", "sql", "description"],
        "primary_keys": ["name"],
        "nullable_keys": ["description"],
    },
]

EDGE_LABELS: List[Dict[str, Any]] = [
    {"name": "belongs_to", "source_label": "table", "target_label": "domain", "properties": []},
    {"name": "has_column", "source_label": "table", "target_label": "column", "properties": []},
    {
        "name": "joins",
        "source_label": "table",
        "target_label": "table",
        "properties": ["on_condition", "join_type", "fanout_risk"],
    },
    {"name": "maps_to", "source_label": "term", "target_label": "column", "properties": []},
    {"name": "maps_to_metric", "source_label": "term", "target_label": "metric", "properties": []},
    {
        "name": "aggregates",
        "source_label": "metric",
        "target_label": "column",
        "properties": ["agg_func"],
    },
    {"name": "has_value", "source_label": "column", "target_label": "value", "properties": []},
    {"name": "has_filter", "source_label": "metric", "target_label": "filter", "properties": []},
    {"name": "uses_table", "source_label": "query_pattern", "target_label": "table", "properties": []},
    {"name": "uses_metric", "source_label": "query_pattern", "target_label": "metric", "properties": []},
]


class Text2SQLSchema:
    """The fixed constraint schema for the Text2SQL semantic-layer graph."""

    VERTEX_LABELS = VERTEX_LABELS
    EDGE_LABELS = EDGE_LABELS
    PROPERTY_KEYS = PROPERTY_KEYS

    @classmethod
    def to_hugegraph_dict(cls) -> Dict[str, Any]:
        """Return the schema in the shape ``Commit2Graph`` expects."""
        return {
            "propertykeys": [dict(pk) for pk in cls.PROPERTY_KEYS],
            "vertexlabels": [dict(vl) for vl in cls.VERTEX_LABELS],
            "edgelabels": [dict(el) for el in cls.EDGE_LABELS],
        }

    @classmethod
    def vertex_label_names(cls) -> List[str]:
        return [vl["name"] for vl in cls.VERTEX_LABELS]

    @classmethod
    def edge_label_names(cls) -> List[str]:
        return [el["name"] for el in cls.EDGE_LABELS]

    @classmethod
    def validate(cls) -> List[str]:
        """Return a list of schema integrity problems (empty when valid)."""
        problems: List[str] = []
        vertex_names = cls.vertex_label_names()
        for edge in cls.EDGE_LABELS:
            for key in ("source_label", "target_label"):
                label = edge[key]
                if label not in vertex_names:
                    problems.append(f"edge '{edge['name']}' {key} '{label}' is not a vertex label")
        pk_names = {pk["name"] for pk in cls.PROPERTY_KEYS}
        for vertex in cls.VERTEX_LABELS:
            for prop in vertex["properties"] + vertex.get("primary_keys", []):
                if prop not in pk_names:
                    problems.append(f"vertex '{vertex['name']}' references unknown property '{prop}'")
            for pk in vertex.get("primary_keys", []):
                if pk not in vertex["properties"]:
                    problems.append(f"vertex '{vertex['name']}' primary key '{pk}' not in properties")
        return problems
