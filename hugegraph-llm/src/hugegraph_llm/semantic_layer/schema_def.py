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

"""Declarative graph schema for the semantic layer.

The schema is expressed as data and handed to
:class:`~hugegraph_llm.operators.hugegraph_op.schema_manager.SchemaManager`,
which creates every object idempotently (probe-then-create), so re-running
bootstrap on an existing graph is a no-op.

Two HugeGraph constraints shape this module and must be respected when
extending it:

1. **Property keys are global and their data type is immutable.** A key first
   created as ``TEXT`` can never store a number, so the type table below is
   the single source of truth and callers coerce values through
   :func:`coerce_value` before writing.
2. **Every property used in a ``has()`` filter needs an index**, otherwise
   HugeGraph raises ``NoIndexException``. ``SchemaManager.ensure_schema``
   auto-creates ``{label}ByName`` for vertex labels carrying ``name``; every
   other filterable property is listed in :data:`INDEX_LABELS` explicitly.

Deliberate differences from neocarta's Neo4j model:

* Sample values are inlined as a ``sample_values`` list property instead of
  dedicated ``Value`` nodes. At warehouse scale (billions of edges) a node per
  sample value is pure overhead, and the retrieved context is identical.
* ``Glossary`` / ``Category`` are collapsed into ``BusinessTerm.category``
  to save a hop on the hot retrieval path.
"""

from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.operators.hugegraph_op.schema_manager import (
    ID_STRATEGY_CUSTOMIZE_STRING,
)
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel

__all__ = [
    "PROPERTY_KEYS",
    "VERTEX_LABELS",
    "EDGE_LABELS",
    "INDEX_LABELS",
    "build_schema_dict",
    "coerce_value",
    "SINGLE",
    "LIST",
]

SINGLE = "SINGLE"
LIST = "LIST"

# --------------------------------------------------------------------------
# Property keys. Global scope: a name may appear only once, with one type.
# (data_type, cardinality)
# --------------------------------------------------------------------------
PROPERTY_KEYS: Dict[str, Tuple[str, str]] = {
    # identity / descriptive
    "name": ("TEXT", SINGLE),
    "description": ("TEXT", SINGLE),
    "comment": ("TEXT", SINGLE),
    "database": ("TEXT", SINGLE),
    "schema": ("TEXT", SINGLE),
    "table": ("TEXT", SINGLE),
    "data_type": ("TEXT", SINGLE),
    "platform": ("TEXT", SINGLE),
    "service": ("TEXT", SINGLE),
    "category": ("TEXT", SINGLE),
    "owner": ("TEXT", SINGLE),
    "source_system": ("TEXT", SINGLE),
    "expression": ("TEXT", SINGLE),
    "dialect": ("TEXT", SINGLE),
    "grain": ("TEXT", SINGLE),
    "unit": ("TEXT", SINGLE),
    "cardinality": ("TEXT", SINGLE),
    "content": ("TEXT", SINGLE),
    "job_id": ("TEXT", SINGLE),
    # trust layer -- NOT part of the Ossie spec, which standardises
    # definitions but explicitly not whether a definition can be believed.
    # Carried on every governance-relevant label and exported as a vendor
    # custom extension; see connectors/ossie.py::_trust_extension.
    "lineage_ref": ("TEXT", SINGLE),
    # numeric
    "row_count": ("LONG", SINGLE),
    "freshness_ts": ("LONG", SINGLE),
    "exec_count": ("LONG", SINGLE),
    "last_seen_ts": ("LONG", SINGLE),
    "use_count": ("LONG", SINGLE),
    "confidence": ("DOUBLE", SINGLE),
    "weight": ("DOUBLE", SINGLE),
    "resolution": ("DOUBLE", SINGLE),
    # flags
    "is_primary_key": ("BOOLEAN", SINGLE),
    "is_foreign_key": ("BOOLEAN", SINGLE),
    "nullable": ("BOOLEAN", SINGLE),
    "is_time_dimension": ("BOOLEAN", SINGLE),
    "is_fact": ("BOOLEAN", SINGLE),
    "proven": ("BOOLEAN", SINGLE),
    # multi-valued (LIST cardinality)
    "aliases": ("TEXT", LIST),
    "synonyms": ("TEXT", LIST),
    "sample_values": ("TEXT", LIST),
    "schema_refs": ("TEXT", LIST),
    "from_columns": ("TEXT", LIST),
    "to_columns": ("TEXT", LIST),
}

# --------------------------------------------------------------------------
# Vertex labels -> property names. All use CUSTOMIZE_STRING ids so that the
# vertex id is the business-logical id (`table:dw.orders`), which keeps
# ingest idempotent and lets edges reference endpoints without a lookup.
# --------------------------------------------------------------------------
VERTEX_LABELS: Dict[VertexLabel, List[str]] = {
    VertexLabel.DATABASE: ["name", "platform", "service", "description"],
    VertexLabel.SCHEMA: ["name", "description"],
    VertexLabel.TABLE: [
        "name",
        "database",
        "schema",
        "comment",
        "row_count",
        "is_fact",
        "freshness_ts",
        "confidence",
        "owner",
        "source_system",
        "lineage_ref",
    ],
    VertexLabel.COLUMN: [
        "name",
        "table",
        "data_type",
        "comment",
        "is_primary_key",
        "is_foreign_key",
        "nullable",
        "is_time_dimension",
        "sample_values",
        "confidence",
        "source_system",
        "freshness_ts",
        "lineage_ref",
    ],
    VertexLabel.BUSINESS_TERM: [
        "name",
        "description",
        "aliases",
        "synonyms",
        "category",
        "source_system",
        "confidence",
        "freshness_ts",
        "lineage_ref",
    ],
    VertexLabel.METRIC: [
        "name",
        "description",
        "expression",
        "dialect",
        "grain",
        "unit",
        "owner",
        "confidence",
        "source_system",
        "freshness_ts",
        "lineage_ref",
    ],
    VertexLabel.JOIN: [
        "name",
        "from_columns",
        "to_columns",
        "cardinality",
        "proven",
        "source_system",
        "confidence",
        "freshness_ts",
        "lineage_ref",
    ],
    VertexLabel.QUERY: [
        "name",
        "content",
        "exec_count",
        "last_seen_ts",
        "schema_refs",
    ],
    VertexLabel.DOMAIN: ["name", "description", "resolution"],
}

# --------------------------------------------------------------------------
# Edge labels: (name, source_label, target_label, properties)
# --------------------------------------------------------------------------
EDGE_LABELS: List[Tuple[EdgeLabel, VertexLabel, VertexLabel, List[str]]] = [
    (EdgeLabel.HAS_SCHEMA, VertexLabel.DATABASE, VertexLabel.SCHEMA, []),
    (EdgeLabel.HAS_TABLE, VertexLabel.SCHEMA, VertexLabel.TABLE, []),
    (EdgeLabel.HAS_COLUMN, VertexLabel.TABLE, VertexLabel.COLUMN, []),
    (EdgeLabel.REFERENCES, VertexLabel.COLUMN, VertexLabel.COLUMN, ["proven"]),
    (EdgeLabel.LINEAGE, VertexLabel.TABLE, VertexLabel.TABLE, ["job_id"]),
    (EdgeLabel.CO_OCCUR, VertexLabel.TABLE, VertexLabel.TABLE, ["weight"]),
    (
        EdgeLabel.TABLE_TAGGED_WITH,
        VertexLabel.TABLE,
        VertexLabel.BUSINESS_TERM,
        [],
    ),
    (
        EdgeLabel.COLUMN_TAGGED_WITH,
        VertexLabel.COLUMN,
        VertexLabel.BUSINESS_TERM,
        [],
    ),
    (
        EdgeLabel.METRIC_TAGGED_WITH,
        VertexLabel.METRIC,
        VertexLabel.BUSINESS_TERM,
        [],
    ),
    (EdgeLabel.TERM_MAPS, VertexLabel.BUSINESS_TERM, VertexLabel.COLUMN, []),
    (EdgeLabel.HAS_EXPRESSION, VertexLabel.METRIC, VertexLabel.COLUMN, ["dialect"]),
    (
        EdgeLabel.SYNONYM,
        VertexLabel.BUSINESS_TERM,
        VertexLabel.BUSINESS_TERM,
        [],
    ),
    (EdgeLabel.USES_TABLE, VertexLabel.QUERY, VertexLabel.TABLE, ["use_count"]),
    (
        EdgeLabel.USES_COLUMN,
        VertexLabel.QUERY,
        VertexLabel.COLUMN,
        ["use_count"],
    ),
]

# --------------------------------------------------------------------------
# Indexes beyond the automatic `{label}ByName`. Only SECONDARY and RANGE are
# usable on HugeGraph 1.7: SEARCH (full-text) is declared but not supported,
# and there is no vector index at all -- semantic search lives in an external
# store (see nl2sql.vector_store).
# --------------------------------------------------------------------------
INDEX_LABELS: List[Dict[str, str]] = [
    {
        "name": "tableByRowCount",
        "base_label": VertexLabel.TABLE.value,
        "field": "row_count",
        "index_type": "RANGE",
        "on": "vertex",
    },
    {
        "name": "tableByFreshness",
        "base_label": VertexLabel.TABLE.value,
        "field": "freshness_ts",
        "index_type": "RANGE",
        "on": "vertex",
    },
    {
        "name": "cooccurByWeight",
        "base_label": EdgeLabel.CO_OCCUR.value,
        "field": "weight",
        "index_type": "RANGE",
        "on": "edge",
    },
]

_VERTEX_PROP_KEYS = {p for props in VERTEX_LABELS.values() for p in props}
_EDGE_PROP_KEYS = {p for _, _, _, props in EDGE_LABELS for p in props}


def build_schema_dict() -> Dict[str, Any]:
    """Build the schema payload accepted by ``SchemaManager.ensure_schema``."""
    propertykeys = [
        {"name": name, "data_type": dtype, "cardinality": card}
        for name, (dtype, card) in sorted(PROPERTY_KEYS.items())
    ]
    vertexlabels = [
        {
            "name": label.value,
            "properties": list(props),
            "id_strategy": ID_STRATEGY_CUSTOMIZE_STRING,
            # Everything optional: a connector may only know part of the
            # metadata for a given object on any single run.
            "nullable_keys": list(props),
        }
        for label, props in VERTEX_LABELS.items()
    ]
    edgelabels = [
        {
            "name": name.value,
            "source_label": source.value,
            "target_label": target.value,
            "properties": list(props),
        }
        for name, source, target, props in EDGE_LABELS
    ]
    return {
        "propertykeys": propertykeys,
        "vertexlabels": vertexlabels,
        "edgelabels": edgelabels,
        "indexes": [dict(idx) for idx in INDEX_LABELS],
    }


def coerce_value(key: str, value: Any) -> Any:
    """Coerce ``value`` to the declared type of property ``key``.

    Property key types are immutable once created, so a value arriving with
    the wrong Python type must be converted rather than triggering a key
    rebuild (or a 400 from the server). LIST properties accept a single
    scalar and wrap it, so callers need not special-case them.

    Returns ``None`` for unknown keys -- callers should drop those instead
    of writing an undeclared property, which HugeGraph rejects.
    """
    declared = PROPERTY_KEYS.get(key)
    if declared is None:
        return None
    data_type, cardinality = declared
    if cardinality == LIST:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [_scalar_to_text(v) for v in value]
        return [_scalar_to_text(value)]
    return _coerce_scalar(data_type, value)


def _scalar_to_text(value: Any) -> str:
    return "" if value is None else str(value)


def _coerce_scalar(data_type: str, value: Any) -> Any:
    if value is None:
        return None
    try:
        if data_type == "TEXT":
            return value if isinstance(value, str) else str(value)
        if data_type == "LONG":
            return int(value)
        if data_type == "DOUBLE":
            return float(value)
        if data_type == "BOOLEAN":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y", "t")
            return bool(value)
    except (TypeError, ValueError):
        return None
    return value


def unknown_property_keys() -> Dict[str, List[str]]:
    """Self-check: property names referenced but never declared (and vice versa).

    Wired into tests so a typo in a label definition fails fast instead of
    surfacing as a 400 on a live server.
    """
    referenced = _VERTEX_PROP_KEYS | _EDGE_PROP_KEYS
    declared = set(PROPERTY_KEYS)
    return {
        "undeclared": sorted(referenced - declared),
        "unused": sorted(declared - referenced),
    }


def validate() -> Optional[str]:
    """Return an error message when the schema definition is inconsistent."""
    problems = []
    undeclared = unknown_property_keys()["undeclared"]
    if undeclared:
        problems.append(f"undeclared property keys: {undeclared}")
    for name, (dtype, card) in PROPERTY_KEYS.items():
        if dtype not in ("TEXT", "LONG", "DOUBLE", "BOOLEAN"):
            problems.append(f"{name}: unsupported data type {dtype}")
        if card not in (SINGLE, LIST):
            problems.append(f"{name}: unsupported cardinality {card}")
    seen_edges = {}
    for name, source, target, _props in EDGE_LABELS:
        if name in seen_edges:
            problems.append(
                f"{name}: duplicate edge label (HugeGraph allows one "
                f"source->target pair per label); already {seen_edges[name]}"
            )
        seen_edges[name] = f"{source}->{target}"
    for idx in INDEX_LABELS:
        if idx.get("index_type") == "SEARCH":
            problems.append(
                f"{idx.get('name')}: SEARCH (full-text) index is declared "
                "but not supported by HugeGraph 1.7"
            )
    return "; ".join(problems) if problems else None
