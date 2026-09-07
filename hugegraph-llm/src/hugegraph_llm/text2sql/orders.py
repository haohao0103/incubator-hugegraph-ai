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

"""The order-domain sample, expressed in semantic_layer terms.

This replaces the former in-code ``SemanticModel`` stack (model.py,
examples.py, schema.py, retrieval.py, seed.py, writer.py, gremlin.py).
Everything overlapping the graph-native semantic layer is gone; what
remains is **data plus two renderings of it**:

* :func:`orders_projection` builds a :class:`SemanticProjection` for tests
  and offline use -- no server needed.
* :class:`OrderDomainConnector` seeds a live HugeGraph through the M1
  connector contract, using the M0 schema (``Table`` / ``Column`` /
  ``BusinessTerm`` / ``Metric`` / ``Query`` labels, ``CUSTOMIZE_STRING``
  ids in the ``table:`` / ``column:`` / ``term:`` / ``metric:`` /
  ``query:`` conventions). The former PoC wrote its own PRIMARY_KEY-schema
  graph; that is gone -- there is one schema stack now.

Domain concepts that the semantic layer stores as *text* rather than
structure: a metric's 口径 filters and granularity are serialised into its
``expression`` string, and a column's value dictionary (code -> meaning)
is folded into its ``comment``. Both render verbatim into prompts, which
is all the generator needs; a structured filter model would be a
schema_def extension, deliberately not invented here.
"""

import hashlib
from typing import Any, Dict, List

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.semantic_layer.connectors.base import SourceConnector
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.semantic_layer.evaluation.dataset import extract_tables
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    MetricRow,
    QueryPatternRow,
    SemanticProjection,
    TableRow,
    TermRow,
)
from hugegraph_llm.semantic_layer.schema_def import coerce_value
from hugegraph_llm.utils.log import log

__all__ = [
    "orders_projection",
    "OrderDomainConnector",
    "ensure_order_domain",
    "ORDER_TABLES",
    "ORDER_TERMS",
    "ORDER_METRICS",
    "ORDER_QUERIES",
]

# -- the domain, as plain data ----------------------------------------------

#: ``fk`` names the referenced ``table.column`` -- seeded as a proven
#: ``REFERENCES`` edge, the only join evidence generators should trust.
ORDER_TABLES: List[Dict[str, Any]] = [
    {
        "name": "order",
        "comment": "订单主表",
        "columns": [
            {"name": "id", "type": "BIGINT", "comment": "订单ID", "pk": True},
            {"name": "user_id", "type": "BIGINT", "comment": "下单用户ID",
             "fk": "user.id"},
            {"name": "driver_id", "type": "BIGINT", "comment": "接单司机ID",
             "fk": "driver.id"},
            {"name": "status", "type": "TINYINT",
             "comment": "订单状态: 1=待支付, 2=已支付, 3=已完成, 4=已取消",
             "values": True},
            {"name": "amount", "type": "DECIMAL", "comment": "订单金额"},
            {"name": "pay_time", "type": "DATETIME", "comment": "支付时间",
             "time": True},
            {"name": "created_at", "type": "DATETIME", "comment": "下单时间",
             "time": True},
        ],
    },
    {
        "name": "order_detail",
        "comment": "订单明细表",
        "columns": [
            {"name": "order_id", "type": "BIGINT", "comment": "订单ID",
             "fk": "order.id"},
            {"name": "product_id", "type": "BIGINT", "comment": "商品ID"},
            {"name": "amount", "type": "DECIMAL", "comment": "明细金额"},
            {"name": "quantity", "type": "INT", "comment": "数量"},
        ],
    },
    {
        "name": "user",
        "comment": "用户表",
        "columns": [
            {"name": "id", "type": "BIGINT", "comment": "用户ID", "pk": True},
            {"name": "name", "type": "STRING", "comment": "用户姓名"},
            {"name": "level", "type": "STRING", "comment": "用户等级"},
        ],
    },
    {
        "name": "driver",
        "comment": "司机表",
        "columns": [
            {"name": "id", "type": "BIGINT", "comment": "司机ID", "pk": True},
            {"name": "name", "type": "STRING", "comment": "司机姓名"},
            {"name": "region", "type": "STRING", "comment": "所属区域"},
        ],
    },
]

ORDER_TERMS: List[Dict[str, Any]] = [
    {"name": "GMV", "aliases": ["成交额", "支付金额", "gmv"],
     "columns": ["order_detail.amount"], "metrics": ["gmv"]},
    # Ambiguous on purpose: 金额 maps to two columns.
    {"name": "金额", "aliases": [], "columns": ["order.amount", "order_detail.amount"]},
    {"name": "订单量", "aliases": ["订单数", "单量"], "metrics": ["order_count"]},
    {"name": "司机", "aliases": ["骑手", "driver"], "columns": ["driver.name"]},
    {"name": "用户", "aliases": ["顾客", "客户"], "columns": ["user.name"]},
]

#: 口径 filters/granularity are serialised into ``expression`` -- the prompt
#: renders them verbatim and the generator must not recompute them.
ORDER_METRICS: List[Dict[str, Any]] = [
    {
        "name": "gmv",
        "description": "成交总额（已支付+已完成订单的明细金额求和，按月）",
        "expression": "SUM(order_detail.amount) WHERE order.status IN (2, 3)",
        "dialect": "StarRocks",
        "columns": ["order_detail.amount"],
    },
    {
        "name": "order_count",
        "description": "订单量（按日）",
        "expression": "COUNT(order.id)",
        "dialect": "StarRocks",
        "columns": ["order.id"],
    },
]

ORDER_QUERIES: List[Dict[str, Any]] = [
    {
        "question": "上个月 GMV 是多少",
        "sql": "SELECT SUM(od.amount) AS gmv FROM order_detail od "
               "JOIN `order` o ON od.order_id = o.id "
               "WHERE o.status IN (2, 3) AND o.pay_time >= '2024-01-01' "
               "AND o.pay_time < '2024-02-01'",
    },
    {
        "question": "每个司机的接单量",
        "sql": "SELECT d.name AS driver, COUNT(o.id) AS order_count FROM driver d "
               "LEFT JOIN `order` o ON d.id = o.driver_id GROUP BY d.name",
    },
]


def _query_vid(question: str, sql: str) -> str:
    digest = hashlib.sha1(f"{question}\n{sql}".encode("utf-8")).hexdigest()[:16]
    return f"query:{digest}"


# -- rendering 1: projection (tests, offline) --------------------------------


def orders_projection() -> SemanticProjection:
    """Build the order domain as a :class:`SemanticProjection`."""
    proj = SemanticProjection()
    for table in ORDER_TABLES:
        proj.tables[table["name"]] = TableRow(
            name=table["name"], comment=table["comment"],
            source_system="text2sql:order_domain",
        )
        for col in table["columns"]:
            row = ColumnRow(
                name=col["name"], table=table["name"],
                data_type=col.get("type", ""),
                comment=col.get("comment", ""),
                is_primary_key=bool(col.get("pk")),
                is_time_dimension=bool(col.get("time")),
            )
            proj.columns[row.qualified] = row
            if col.get("fk"):
                proj.references[row.qualified] = [col["fk"]]
                proj.reference_proven[(row.qualified, col["fk"])] = True

    for term in ORDER_TERMS:
        proj.terms[term["name"]] = TermRow(
            name=term["name"], aliases=list(term.get("aliases", []))
        )
        for column in term.get("columns", []):
            proj.term_columns.setdefault(term["name"], []).append(column)
        for metric in term.get("metrics", []):
            proj.term_metrics.setdefault(term["name"], []).append(metric)

    for metric in ORDER_METRICS:
        proj.metrics[metric["name"]] = MetricRow(
            name=metric["name"],
            description=metric["description"],
            expression=metric["expression"],
            dialect=metric.get("dialect", ""),
        )
        for column in metric["columns"]:
            proj.metric_columns.setdefault(metric["name"], []).append(column)

    for pattern in ORDER_QUERIES:
        proj.query_patterns.append(QueryPatternRow(
            question=pattern["question"],
            sql=pattern["sql"],
            tables=[t for t in extract_tables(pattern["sql"])
                    if t in proj.tables],
        ))
    return proj


# -- rendering 2: graph seed (M1 connector contract, M0 schema) ---------------


class OrderDomainConnector(SourceConnector):
    """Seeds the order domain into a live HugeGraph, semantic_layer style.

    Idempotent: vertices are written under fixed ``CUSTOMIZE_STRING`` ids,
    so re-running updates rather than duplicates. Edges are appended; for
    this static domain the caller should seed once per graph (see
    :func:`ensure_order_domain` for the guard).
    """

    name = "order_domain"

    def __init__(self, client: PyHugeClient) -> None:
        super().__init__()
        self.client = client

    def extract(self) -> List[Dict[str, Any]]:
        return [dict(t) for t in ORDER_TABLES]

    def transform(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vertices: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for table in ORDER_TABLES:
            vertices.append({
                "label": VertexLabel.TABLE.value,
                "id": f"table:{table['name']}",
                "properties": _clean({
                    "name": table["name"],
                    "comment": table["comment"],
                    "source_system": "text2sql:order_domain",
                }),
            })
            for col in table["columns"]:
                qualified = f"{table['name']}.{col['name']}"
                vertices.append({
                    "label": VertexLabel.COLUMN.value,
                    "id": f"column:{qualified}",
                    "properties": _clean({
                        "name": col["name"],
                        "table": table["name"],
                        "data_type": col.get("type", ""),
                        "comment": col.get("comment", ""),
                        "is_primary_key": bool(col.get("pk")),
                        "is_time_dimension": bool(col.get("time")),
                    }),
                })
                edges.append({
                    "label": EdgeLabel.HAS_COLUMN.value,
                    "out": f"table:{table['name']}",
                    "in": f"column:{qualified}",
                })
                if col.get("fk"):
                    edges.append({
                        "label": EdgeLabel.REFERENCES.value,
                        "out": f"column:{qualified}",
                        "in": f"column:{col['fk']}",
                        "properties": {"proven": True},
                    })

        for term in ORDER_TERMS:
            term_id = f"term:{term['name']}"
            vertices.append({
                "label": VertexLabel.BUSINESS_TERM.value,
                "id": term_id,
                "properties": _clean({
                    "name": term["name"],
                    "aliases": list(term.get("aliases", [])),
                    "source_system": "text2sql:order_domain",
                }),
            })
            for column in term.get("columns", []):
                edges.append({
                    "label": EdgeLabel.TERM_MAPS.value,
                    "out": term_id,
                    "in": f"column:{column}",
                })

        for metric in ORDER_METRICS:
            metric_id = f"metric:{metric['name']}"
            vertices.append({
                "label": VertexLabel.METRIC.value,
                "id": metric_id,
                "properties": _clean({
                    "name": metric["name"],
                    "description": metric["description"],
                    "expression": metric["expression"],
                    "dialect": metric.get("dialect", ""),
                    "source_system": "text2sql:order_domain",
                }),
            })
            for column in metric["columns"]:
                edges.append({
                    "label": EdgeLabel.HAS_EXPRESSION.value,
                    "out": metric_id,
                    "in": f"column:{column}",
                    "properties": {"dialect": metric.get("dialect", "")},
                })
            for term in ORDER_TERMS:
                if metric["name"] in term.get("metrics", []):
                    edges.append({
                        "label": EdgeLabel.METRIC_TAGGED_WITH.value,
                        "out": metric_id,
                        "in": f"term:{term['name']}",
                    })

        for pattern in ORDER_QUERIES:
            vid = _query_vid(pattern["question"], pattern["sql"])
            vertices.append({
                "label": VertexLabel.QUERY.value,
                "id": vid,
                "properties": _clean({
                    "name": pattern["question"],
                    "content": pattern["sql"],
                    "exec_count": 1,
                    "source_system": "text2sql:order_domain",
                }),
            })
            for table in extract_tables(pattern["sql"]):
                if any(t["name"] == table for t in ORDER_TABLES):
                    edges.append({
                        "label": EdgeLabel.USES_TABLE.value,
                        "out": vid,
                        "in": f"table:{table}",
                    })

        return [{"vertices": vertices, "edges": edges}]

    def load(self, records: List[Dict[str, Any]]) -> int:
        payload = records[0] if records else {"vertices": [], "edges": []}
        written = 0
        graph = self.client.graph()
        for node in payload["vertices"]:
            try:
                graph.addVertex(node["label"], node["properties"], id=node["id"])
                written += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("order_domain seed: vertex %s failed: %s",
                            node["id"], exc)
        for edge in payload["edges"]:
            try:
                graph.addEdge(edge["label"], edge["out"], edge["in"],
                              edge.get("properties") or {})
                written += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("order_domain seed: edge %s %s->%s failed: %s",
                            edge["label"], edge["out"], edge["in"], exc)
        return written


def ensure_order_domain(client: PyHugeClient) -> bool:
    """Seed the order domain once. Returns True when seeding happened.

    The guard is the domain's root table: if ``table:order`` exists the
    graph was already seeded (vertices are id-upserts, edges are appends,
    so a second pass would duplicate edges).
    """
    try:
        client.graph().getVertexById("table:order")
        return False  # already seeded
    except Exception:  # noqa: BLE001 - absence surfaces as an error
        pass
    OrderDomainConnector(client).ingest()
    return True


def _clean(props: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in props.items():
        coerced = coerce_value(key, value)
        if coerced is None:
            continue
        cleaned[key] = coerced
    return cleaned
