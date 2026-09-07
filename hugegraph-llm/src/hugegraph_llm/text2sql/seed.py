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

"""Deterministic seeding: turn a :class:`SemanticModel` into graph data.

The result is a ``{"vertices": [...], "edges": [...]}`` structure that can be
traversed directly (see :mod:`hugegraph_llm.text2sql.retrieval`) or committed to
HugeGraph via ``ImportGraphDataFlow`` with :class:`Text2SQLSchema`.

Vertex ids are label-scoped (``"<label>:<primary_key>"``), mirroring HugeGraph's
label-scoped id space under the ``PRIMARY_KEY`` id strategy:

- ``domain``        -> ``"domain:order"``
- ``table``         -> ``"table:order"``
- ``column``        -> ``"column:order!status"`` (composite ``[table, name]``)
- ``term``          -> ``"term:GMV"``
- ``metric``        -> ``"metric:gmv"``
- ``value``         -> ``"value:order.status!2"`` (composite ``[column, code]``)
- ``filter``        -> ``"filter:<hash>"`` (stable, shareable across metrics)
- ``query_pattern`` -> ``"query_pattern:<hash>"`` (idempotent across re-seeds)
"""

import hashlib
from typing import Any, Dict, List

from hugegraph_llm.text2sql.model import SemanticModel


def _vertex(vid: str, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": vid, "label": label, "properties": properties}


def _edge(label: str, out_v: str, in_v: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"label": label, "outV": out_v, "inV": in_v, "properties": properties}


def _vid(label: str, primary_key: str) -> str:
    """Label-scoped vertex id: ``"column:order!status"``."""
    return f"{label}:{primary_key}"


def _column_full_name(table: str, column: str) -> str:
    return f"{table}.{column}"


def _column_pk_from_full_name(full_name: str) -> str:
    """Composite column primary key: ``"table.column"`` -> ``"table!column"``."""
    return full_name.replace(".", "!", 1)


def _filter_pk(column: str, operator: str, values: List[str]) -> str:
    raw = f"{column}|{operator}|{','.join(values)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def seed_graph(model: SemanticModel) -> Dict[str, Any]:
    """Convert a semantic model into ``{"vertices", "edges"}`` graph data."""
    vertices: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Domain vertex.
    if model.domain:
        vertices.append(_vertex(_vid("domain", model.domain), "domain", {"name": model.domain, "description": ""}))

    # Table + column vertices, belongs_to / has_column edges.
    for table in model.tables:
        table_id = _vid("table", table.name)
        table_props = {
            "name": table.name,
            "aliases": list(table.aliases),
            "description": table.description,
            "partition_column": table.partition_column,
            "time_column": table.time_column,
            "timezone": table.timezone,
        }
        vertices.append(_vertex(table_id, "table", table_props))
        if table.domain:
            edges.append(_edge("belongs_to", table_id, _vid("domain", table.domain), {}))

        for column in table.columns:
            full_name = _column_full_name(table.name, column.name)
            col_id = _vid("column", _column_pk_from_full_name(full_name))
            vertices.append(
                _vertex(
                    col_id,
                    "column",
                    {
                        "table": table.name,
                        "name": column.name,
                        "full_name": full_name,
                        "data_type": column.data_type,
                        "role": column.role,
                        "description": column.description,
                    },
                )
            )
            edges.append(_edge("has_column", table_id, col_id, {}))

    # Join edges.
    for join in model.joins:
        edges.append(
            _edge(
                "joins",
                _vid("table", join.from_table),
                _vid("table", join.to_table),
                {
                    "on_condition": join.on_condition,
                    "join_type": join.join_type,
                    "fanout_risk": join.fanout_risk,
                },
            )
        )

    # Term vertices and their maps_to / maps_to_metric edges.
    for term in model.terms:
        term_id = _vid("term", term.name)
        vertices.append(_vertex(term_id, "term", {"name": term.name, "aliases": list(term.aliases)}))
        for col_ref in term.column_refs:
            edges.append(_edge("maps_to", term_id, _vid("column", _column_pk_from_full_name(col_ref)), {}))
        for metric_ref in term.metric_refs:
            edges.append(_edge("maps_to_metric", term_id, _vid("metric", metric_ref), {}))

    # Metric vertices, aggregates edges, and structured filter vertices.
    for metric in model.metrics:
        metric_id = _vid("metric", metric.name)
        metric_props = {
            "name": metric.name,
            "agg_func": metric.agg_func,
            "time_granularity": metric.time_granularity,
            "dedup": metric.dedup,
            "description": metric.description,
            "measure": metric.measure,
            "dimensions": list(metric.dimensions),
            "time_column": metric.time_column,
        }
        vertices.append(_vertex(metric_id, "metric", metric_props))
        if metric.measure:
            edges.append(
                _edge(
                    "aggregates",
                    metric_id,
                    _vid("column", _column_pk_from_full_name(metric.measure)),
                    {"agg_func": metric.agg_func},
                )
            )
        for filter_ in metric.filters:
            filter_id = _vid("filter", _filter_pk(filter_.column, filter_.operator, filter_.values))
            vertices.append(
                _vertex(
                    filter_id,
                    "filter",
                    {
                        "name": filter_id,
                        "column": filter_.column,
                        "operator": filter_.operator,
                        "values": list(filter_.values),
                    },
                )
            )
            edges.append(_edge("has_filter", metric_id, filter_id, {}))

    # Value vertices and has_value edges.
    for value in model.values:
        value_id = _vid("value", f"{value.column}!{value.code}")
        vertices.append(
            _vertex(
                value_id,
                "value",
                {"column": value.column, "code": value.code, "meaning": value.meaning},
            )
        )
        edges.append(_edge("has_value", _vid("column", _column_pk_from_full_name(value.column)), value_id, {}))

    # Query-pattern vertices and uses_* edges.
    for qp in model.query_patterns:
        digest = hashlib.sha1(f"{qp.question}\0{qp.sql}".encode("utf-8")).hexdigest()[:12]
        qp_id = _vid("query_pattern", digest)
        vertices.append(
            _vertex(
                qp_id,
                "query_pattern",
                {"name": qp_id, "sql": qp.sql, "description": qp.question},
            )
        )
        for table in qp.tables:
            edges.append(_edge("uses_table", qp_id, _vid("table", table), {}))
        for metric in qp.metrics:
            edges.append(_edge("uses_metric", qp_id, _vid("metric", metric), {}))

    return {"vertices": vertices, "edges": edges}
