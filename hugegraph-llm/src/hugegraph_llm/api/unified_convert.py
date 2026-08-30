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

"""Conversion helpers for the unified ingest API.

The spec (docs/UNIFIED_INGEST_QUERY_API_SPEC.md) describes
``ImportGraphDataFlow(mode="catalog" | "metric")``.  In this codebase
``ImportGraphDataFlow`` is *generic* (``build_flow(data, schema, **kwargs)`` and
has no ``mode`` flag), so the routing is done here: we transform a
catalog/metric payload into the ``{vertices, edges}`` graph JSON that
``ImportGraphDataFlow`` + ``Commit2Graph`` expect, plus a graph schema.

Vertex labels: ``Table`` / ``Field`` / ``Metric``.
Edge labels:    ``hasColumn`` (Table->Field), ``computedFrom`` (Metric->Table),
                ``computedFromField`` (Metric->Field), ``dependsOn`` (Metric->Metric).

Primary keys: each label uses ``name`` as its (single) primary key.  Edge
endpoints therefore use the HugeGraph PK-mapping id ``<Label>:<name>``, e.g.
``Table:order`` or ``Field:order.amount`` (Field.name is table-qualified so it
stays unique across tables).
"""

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# KG graph schema (vertex / edge labels + property keys).  Passed to
# ImportGraphDataFlow as the schema JSON string; SchemaNode -> CheckSchema
# validates it and injects it into data_json for Commit2Graph.
# ---------------------------------------------------------------------------
KG_SCHEMA: Dict[str, Any] = {
    "propertykeys": [
        {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "comment", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "type", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "table", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "domain", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "source", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "definition", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "formula", "data_type": "TEXT", "cardinality": "SINGLE"},
    ],
    "vertexlabels": [
        {
            "name": "Table",
            "properties": ["name", "comment", "domain", "source"],
            "primary_keys": ["name"],
            "nullable_keys": ["comment", "domain", "source"],
        },
        {
            "name": "Field",
            "properties": ["name", "comment", "type", "table", "domain", "source"],
            "primary_keys": ["name"],
            "nullable_keys": ["comment", "type", "table", "domain", "source"],
        },
        {
            "name": "Metric",
            "properties": ["name", "definition", "formula", "domain", "source"],
            "primary_keys": ["name"],
            "nullable_keys": ["definition", "formula", "domain", "source"],
        },
    ],
    "edgelabels": [
        {"name": "hasColumn", "source_label": "Table", "target_label": "Field", "properties": []},
        {"name": "computedFrom", "source_label": "Metric", "target_label": "Table", "properties": []},
        {"name": "computedFromField", "source_label": "Metric", "target_label": "Field", "properties": []},
        {"name": "dependsOn", "source_label": "Metric", "target_label": "Metric", "properties": []},
    ],
    # Every non-PK property gets the index type that matches how it will be
    # queried: short enum-ish values -> SECONDARY (exact filter), long text ->
    # SEARCH (textContains), numbers -> RANGE. `name` needs none: it is the
    # PRIMARY KEY (auto-indexed). HugeGraph supports 5 index types:
    # SECONDARY / RANGE / SEARCH / SHARD / UNIQUE.
    "indexes": [
        {"name": "TableByComment", "base_label": "Table", "field": "comment", "index_type": "SEARCH"},
        {"name": "TableByDomain", "base_label": "Table", "field": "domain", "index_type": "SECONDARY"},
        {"name": "TableBySource", "base_label": "Table", "field": "source", "index_type": "SECONDARY"},
        {"name": "FieldByComment", "base_label": "Field", "field": "comment", "index_type": "SEARCH"},
        {"name": "FieldByType", "base_label": "Field", "field": "type", "index_type": "SECONDARY"},
        {"name": "FieldByTable", "base_label": "Field", "field": "table", "index_type": "SECONDARY"},
        {"name": "FieldByDomain", "base_label": "Field", "field": "domain", "index_type": "SECONDARY"},
        {"name": "FieldBySource", "base_label": "Field", "field": "source", "index_type": "SECONDARY"},
        {"name": "MetricByDefinition", "base_label": "Metric", "field": "definition", "index_type": "SEARCH"},
        {"name": "MetricByFormula", "base_label": "Metric", "field": "formula", "index_type": "SEARCH"},
        {"name": "MetricByDomain", "base_label": "Metric", "field": "domain", "index_type": "SECONDARY"},
        {"name": "MetricBySource", "base_label": "Metric", "field": "source", "index_type": "SECONDARY"},
        {"name": "MetricByAuthoritative", "base_label": "Metric", "field": "authoritative", "index_type": "SECONDARY"},
        {"name": "MetricByPriority", "base_label": "Metric", "field": "priority", "index_type": "SECONDARY"},
    ],
}

# Serialized form consumed by ImportGraphDataFlow (SchemaNode checks
# ``schema.startswith("{")`` to decide JSON vs graph-name mode).
KG_SCHEMA_JSON = json.dumps(KG_SCHEMA, ensure_ascii=False)


def detect_source_type(payload: Dict[str, Any]) -> str:
    """Sniff ``payload`` to pick catalog vs metric for ``source_type="auto"``."""
    if "metrics" in payload:
        return "metric_json"
    if "tables" in payload:
        return "catalog_csv"
    raise ValueError(
        "Cannot auto-detect source_type: payload must contain 'tables' (catalog) "
        "or 'metrics' (metric_json). Pass an explicit source_type instead."
    )


def _vertex(label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"label": label, "properties": properties}


def convert_catalog_to_graph(
    payload: Dict[str, Any],
    domain: str = "default",
    source: str = "catalog_csv",
) -> Dict[str, Any]:
    """Convert a catalog payload to {vertices, edges} for ImportGraphDataFlow.

    Expected payload shape::
        {
          "tables": [
            {"name": "order", "comment": "订单表", "fields": [
              {"name": "order_id", "comment": "订单号", "type": "bigint"},
              ...
            ]},
            ...
          ]
        }
    """
    vertices: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for table in payload.get("tables", []):
        tname = table["name"]
        vertices.append(
            _vertex(
                "Table",
                {
                    "name": tname,
                    "comment": table.get("comment", ""),
                    "domain": domain,
                    "source": source,
                },
            )
        )
        for field in table.get("fields", []):
            fname = field["name"]
            fq = f"{tname}.{fname}"  # table-qualified, unique PK
            vertices.append(
                _vertex(
                    "Field",
                    {
                        "name": fq,
                        "comment": field.get("comment", ""),
                        "type": field.get("type", ""),
                        "table": tname,
                        "domain": domain,
                        "source": source,
                    },
                )
            )
            edges.append(
                {
                    "label": "hasColumn",
                    "outV": f"Table:{tname}",
                    "inV": f"Field:{fq}",
                    "properties": {},
                }
            )

    return {"vertices": vertices, "edges": edges}


def convert_metric_to_graph(
    payload: Dict[str, Any],
    domain: str = "default",
    source: str = "metric_json",
) -> Dict[str, Any]:
    """Convert a metric payload to {vertices, edges} for ImportGraphDataFlow.

    Expected payload shape::
        {
          "metrics": [
            {"name": "refund_rate", "definition": "...", "formula": "...",
             "source_tables": ["order", "payment"],
             "source_fields": ["order.amount"],   # optional, "table.field"
             "depends_on": ["gmv"]},              # optional
            ...
          ]
        }
    """
    vertices: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for metric in payload.get("metrics", []):
        mname = metric["name"]
        vertices.append(
            _vertex(
                "Metric",
                {
                    "name": mname,
                    "definition": metric.get("definition", ""),
                    "formula": metric.get("formula", ""),
                    "domain": domain,
                    "source": source,
                },
            )
        )
        for t in metric.get("source_tables", []):
            edges.append(
                {
                    "label": "computedFrom",
                    "outV": f"Metric:{mname}",
                    "inV": f"Table:{t}",
                    "properties": {},
                }
            )
        for f in metric.get("source_fields", []):  # "table.field"
            edges.append(
                {
                    "label": "computedFromField",
                    "outV": f"Metric:{mname}",
                    "inV": f"Field:{f}",
                    "properties": {},
                }
            )
        for d in metric.get("depends_on", []):
            edges.append(
                {
                    "label": "dependsOn",
                    "outV": f"Metric:{mname}",
                    "inV": f"Metric:{d}",
                    "properties": {},
                }
            )

    return {"vertices": vertices, "edges": edges}


def build_index_texts(
    vertices: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> List[str]:
    """Assemble free-text corpus from catalog/metric vertices for vector index."""
    texts: List[str] = []
    for v in vertices:
        props = v.get("properties", {})
        label = v.get("label")
        if label == "Table":
            texts.append(f"表 {props.get('name', '')}（{props.get('comment', '')}）")
        elif label == "Field":
            texts.append(
                f"字段 {props.get('name', '')}：{props.get('comment', '')}"
                f"（类型 {props.get('type', '')}，所属表 {props.get('table', '')}）"
            )
        elif label == "Metric":
            texts.append(
                f"指标 {props.get('name', '')}：{props.get('definition', '')}"
                f"，计算公式 {props.get('formula', '')}"
            )
    return [t for t in texts if t.strip()]
