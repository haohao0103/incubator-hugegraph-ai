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

"""Gradio tab: Unified Import & Query.

One import box (source dropdown + payload JSON) and one query box, wired to the
unified ingest/query API. This block calls the *same* core functions exposed by
the HTTP routes (``unified_ingest`` / ``unified_query``) to keep a single code
path.
"""

import json

import gradio as gr

from hugegraph_llm.api.models.unified_requests import (
    UnifiedIngestRequest,
    UnifiedQueryRequest,
)
from hugegraph_llm.api.unified_ingest_api import unified_ingest
from hugegraph_llm.api.unified_query_api import unified_query
from hugegraph_llm.utils.log import log

SOURCE_TYPES = ["feishu", "catalog_csv", "jdbc", "metric_json", "auto"]
QUERY_MODES = ["auto", "precise", "semantic", "hybrid"]

EXAMPLE_CATALOG = json.dumps(
    {
        "tables": [
            {
                "name": "order",
                "comment": "订单表",
                "fields": [
                    {"name": "order_id", "comment": "订单号", "type": "bigint"},
                    {"name": "amount", "comment": "订单金额", "type": "double"},
                ],
            },
            {
                "name": "payment",
                "comment": "支付表",
                "fields": [
                    {"name": "pay_id", "comment": "支付流水号", "type": "bigint"},
                    {"name": "amount", "comment": "支付金额", "type": "double"},
                ],
            },
        ]
    },
    ensure_ascii=False,
    indent=2,
)

EXAMPLE_METRIC = json.dumps(
    {
        "metrics": [
            {
                "name": "refund_rate",
                "definition": "退款订单数/总订单数",
                "formula": "count(refund)/count(total)",
                "source_tables": ["order", "payment"],
                "source_fields": ["order.amount"],
                "depends_on": [],
            }
        ]
    },
    ensure_ascii=False,
    indent=2,
)

EXAMPLE_FEISHU = json.dumps(
    {
        "space_id": "<wiki_space_id>",
        "app_id": "<app_id>",
        "app_secret": "<app_secret>",
        "mode": "full",
        "graph_schema": "kg-rag",
        "build_graph": True,
        "build_index": True,
    },
    ensure_ascii=False,
    indent=2,
)


def _ingest_handler(source_type, payload_text, domain, er_flag, vi_flag):
    try:
        payload = json.loads(payload_text) if payload_text and payload_text.strip() else {}
    except json.JSONDecodeError as exc:
        return f"❌ payload 不是合法 JSON: {exc}"
    req = UnifiedIngestRequest(
        source_type=source_type,
        payload=payload,
        domain=domain or None,
        options={
            "entity_resolution": bool(er_flag),
            "build_vector_index": bool(vi_flag),
        },
    )
    try:
        res = unified_ingest(req)
        return json.dumps(res.model_dump(), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        log.error("unified ingest handler error: %s", exc)
        return f"❌ 导入失败: {exc}"


def _query_handler(question, mode, domain, top_k):
    req = UnifiedQueryRequest(
        question=question,
        mode=mode,
        domain=domain or None,
        top_k=int(top_k) if top_k else 5,
    )
    try:
        res = unified_query(req)
        return json.dumps(res.model_dump(), ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        log.error("unified query handler error: %s", exc)
        return f"❌ 查询失败: {exc}"


def create_unified_io_block():
    with gr.Tab(label="0. Unified Import & Query 🔌"):
        gr.Markdown(
            "## 统一导入 / 查询\n"
            "库表字段 comment · 指标 · 飞书文档 → 同一张 KG 图 `kg-rag`。\n"
            "导入只写元数据（表/字段/指标定义），**不导行级数据**（行级属于关系图）。"
        )

        with gr.Row():
            # ---- left: import ----
            with gr.Column():
                gr.Markdown("### 导入 Ingest")
                ingest_source = gr.Dropdown(
                    choices=SOURCE_TYPES, value="catalog_csv", label="来源类型 source_type"
                )
                ingest_payload = gr.Code(
                    label="payload (JSON)", language="json", value=EXAMPLE_CATALOG
                )
                with gr.Row():
                    ex_catalog = gr.Button("示例·库表")
                    ex_metric = gr.Button("示例·指标")
                    ex_feishu = gr.Button("示例·飞书")
                ingest_domain = gr.Textbox(
                    label="domain (业务域隔离, 可选)", placeholder="risk_control"
                )
                with gr.Row():
                    er_flag = gr.Checkbox(label="实体消歧 EntityResolution", value=True)
                    vi_flag = gr.Checkbox(label="建向量索引 VectorIndex", value=True)
                ingest_btn = gr.Button("导入 Import", variant="primary")
                ingest_out = gr.Code(label="导入结果", language="json")

            # ---- right: query ----
            with gr.Column():
                gr.Markdown("### 查询 Query")
                query_q = gr.Textbox(
                    label="question",
                    placeholder="去年大促退款率怎么算？相关文档有哪些？",
                )
                query_mode = gr.Dropdown(
                    choices=QUERY_MODES, value="auto", label="mode (auto=智能路由)"
                )
                with gr.Row():
                    query_topk = gr.Number(label="top_k", value=5, precision=0)
                    query_domain = gr.Textbox(label="domain (可选过滤)", placeholder="risk_control")
                query_btn = gr.Button("查询 Query", variant="primary")
                query_out = gr.Code(label="查询结果", language="json")

        ingest_btn.click(
            fn=_ingest_handler,
            inputs=[ingest_source, ingest_payload, ingest_domain, er_flag, vi_flag],
            outputs=ingest_out,
        )
        query_btn.click(
            fn=_query_handler,
            inputs=[query_q, query_mode, query_domain, query_topk],
            outputs=query_out,
        )

        # example loaders also set the source dropdown
        ex_catalog.click(
            lambda: ("catalog_csv", EXAMPLE_CATALOG),
            outputs=[ingest_source, ingest_payload],
        )
        ex_metric.click(
            lambda: ("metric_json", EXAMPLE_METRIC),
            outputs=[ingest_source, ingest_payload],
        )
        ex_feishu.click(
            lambda: ("feishu", EXAMPLE_FEISHU),
            outputs=[ingest_source, ingest_payload],
        )
