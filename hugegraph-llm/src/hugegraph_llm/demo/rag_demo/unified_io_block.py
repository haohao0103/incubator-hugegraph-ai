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

``mode="nl2sql"`` additionally builds the KG-aware NL2SQL pipeline directly
(question -> linking -> generation -> validation -> voting -> lineage) with an
optional golden-SQL feedback loop: when "回灌 golden" is on, the winning SQL is
stored into the graph's ``Query`` vertices and retrieved on the next run, so the
voting visibly improves (the ``golden_feedback`` stage appears and the chosen
candidate is boosted by golden overlap).
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
QUERY_MODES = ["auto", "precise", "semantic", "hybrid", "nl2sql"]

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

# deterministic candidate demo for mode="nl2sql" (no LLM needed); the first
# candidate uses a natural SELECT alias, which the validator now resolves.
EXAMPLE_NL2SQL_CANDIDATES = "\n".join(
    [
        "SELECT city, SUM(order.amount) AS order_amount FROM order GROUP BY city ORDER BY order_amount DESC",
        "SELECT SUM(payment.amount) FROM payment",
        "SELECT SUM(order.amnt) FROM order",
    ]
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


_EXECUTOR = None


def _get_executor():
    """Lazy singleton DuckDB executor (in-memory, seeded once, persists)."""
    global _EXECUTOR
    if _EXECUTOR is None:
        from hugegraph_llm.operators.sql_exec.sql_executor import DuckDbExecutor

        _EXECUTOR = DuckDbExecutor()
    return _EXECUTOR


def _nl2sql_run(question, domain, candidates_text, store_golden):
    """Run the KG-aware NL2SQL pipeline with an optional golden feedback loop,
    then execute the winning SQL ("问 -> SQL -> 执行 -> 答案")."""
    from hugegraph_llm.config import huge_settings
    from hugegraph_llm.operators.graph_op.kg_golden_sql import KgGoldenSqlStore
    from hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline import KgNL2SQLPipeline
    from hugegraph_llm.operators.sql_exec.nl2sql_runner import KgNL2SQLRunner
    from hugegraph_llm.utils.hugegraph_utils import get_hg_client

    client = get_hg_client()
    golden_store = None
    if store_golden:
        golden_store = KgGoldenSqlStore(client, huge_settings.graph_name)
    pipe = KgNL2SQLPipeline(
        question=question,
        client=client,
        domain=domain or None,
        golden_store=golden_store,
        store_best=bool(golden_store),
    )
    if candidates_text and candidates_text.strip():
        # deterministic path: vote on the pasted candidates, no LLM
        cands = [c.strip() for c in candidates_text.splitlines() if c.strip()]
        return KgNL2SQLRunner(pipe, _get_executor()).run(candidates=cands)
    return KgNL2SQLRunner(pipe, _get_executor()).run()  # live glm-5.3


def _render_query_markdown(resp_dict) -> str:
    """Human-readable rendering of a UnifiedQueryResponse (nl2sql friendly)."""
    route = resp_dict.get("route", "")
    answer = resp_dict.get("answer") or ""
    if route != "nl2sql":
        return f"**route**: `{route}`\n\n**answer**:\n```\n{answer or '(empty)'}\n```"

    stages = resp_dict.get("stages") or []
    chain = " → ".join(s.get("stage", "") for s in stages)
    lines = [f"**route**: `nl2sql`", f"**stages**: `{chain}`"]
    lines.append(
        "\n**chosen SQL**:\n```sql\n"
        + (answer or "（未生成可用 SQL——可粘贴候选走确定性投票，或重试 LLM）")
        + "\n```"
    )
    votes = (resp_dict.get("raw") or {}).get("votes") or []
    if votes:
        rows = ["| # | score | valid | SQL |", "|---|------:|:-----:|-----|"]
        for i, v in enumerate(votes, 1):
            rows.append(
                f"| {i} | {float(v.get('score', 0)):.1f} | "
                f"{'✅' if v.get('valid') else '❌'} | `{v.get('sql')}` |"
            )
        lines.append("\n**投票排名 voting**\n" + "\n".join(rows))
    # execution block (runner output: 问->SQL->执行->答案)
    execution = resp_dict.get("execution") or {}
    if execution:
        if execution.get("error"):
            lines.append(f"\n**执行结果 execution**\n```\n❌ {execution['error']}\n```")
        else:
            head = " | ".join(str(c) for c in execution.get("columns", []))
            rows_html = []
            for r in execution.get("rows", [])[:5]:
                rows_html.append("| " + " | ".join(str(c) for c in r) + " |")
            table = "\n".join([f"| {head} |", "|" + "---|" * len(execution.get("columns", []))] + rows_html)
            truncated = "（截断，仅显示前 %d 行）" % len(execution.get("rows", [])) if execution.get("truncated") else ""
            lines.append(
                f"\n**执行结果 execution**：{execution.get('row_count', 0)} 行 {truncated}"
                f"\n{table or '（无数据）'}"
            )
        answer = resp_dict.get("answer")
        if answer:
            lines.append(f"\n**答案 answer**：\n{answer}")
    for s in stages:
        out = s.get("output") or {}
        if s.get("stage") == "lineage":
            lines.append(f"\n**血缘 lineage**\n```\n{out.get('explain', '')}\n```")
        elif s.get("stage") == "authority":
            lines.append(
                "\n**指标权威度 authority**\n```\n"
                + json.dumps(out, ensure_ascii=False, indent=2)
                + "\n```"
            )
        elif s.get("stage") == "golden_feedback":
            lines.append(
                "\n**golden 回灌**\n```\n"
                + json.dumps(out, ensure_ascii=False, indent=2)
                + "\n```"
            )
    return "\n".join(lines)


def _query_handler(question, mode, domain, top_k, candidates_text, store_golden):
    if not question or not str(question).strip():
        return "❌ question 不能为空", "❌ question 不能为空"
    try:
        if mode == "nl2sql":
            resp = _nl2sql_run(question, domain, candidates_text, bool(store_golden))
        else:
            req = UnifiedQueryRequest(
                question=question,
                mode=mode,
                domain=domain or None,
                top_k=int(top_k) if top_k else 5,
            )
            resp = unified_query(req)
        resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else resp.model_dump()
        return (
            _render_query_markdown(resp_dict),
            json.dumps(resp_dict, ensure_ascii=False, indent=2),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("unified query handler error: %s", exc)
        msg = f"❌ 查询失败: {exc}"
        return msg, msg


def create_unified_io_block():
    with gr.Tab(label="0. Unified Import & Query 🔌"):
        gr.Markdown(
            "## 统一导入 / 查询\n"
            "库表字段 comment · 指标 · 飞书文档 → 同一张 KG 图 `kg-rag`。\n"
            "导入只写元数据（表/字段/指标定义），**不导行级数据**（行级属于关系图）。\n"
            "`mode=nl2sql`：KG 元数据生成 SQL——linking → 生成 → 校验 → 投票 → 血缘，可开 golden 回灌。"
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
                    choices=QUERY_MODES,
                    value="auto",
                    label="mode (auto=智能路由 / nl2sql=KG 元数据生成 SQL)",
                )
                with gr.Row():
                    query_topk = gr.Number(label="top_k", value=5, precision=0)
                    query_domain = gr.Textbox(label="domain (可选过滤)", placeholder="risk_control")
                query_candidates = gr.Textbox(
                    label="候选 SQL（nl2sql 专用，可选：每行一条；填了走确定性投票、不调 LLM）",
                    placeholder="SELECT city, SUM(order.amount) AS order_amount FROM order GROUP BY city ...",
                )
                store_golden = gr.Checkbox(
                    label="回灌 golden（nl2sql：把最优 SQL 存入图 Query 顶点，下次查询用于提升投票）",
                    value=False,
                )
                with gr.Row():
                    ex_nl2sql_1 = gr.Button("示例·各城市订单总额")
                    ex_nl2sql_2 = gr.Button("示例·订单支付对比")
                    ex_nl2sql_3 = gr.Button("示例·候选投票")
                query_btn = gr.Button("查询 Query", variant="primary")
                query_out_md = gr.Markdown("_点击查询后显示摘要_")
                query_out = gr.Code(label="查询结果 (JSON)", language="json")

        ingest_btn.click(
            fn=_ingest_handler,
            inputs=[ingest_source, ingest_payload, ingest_domain, er_flag, vi_flag],
            outputs=ingest_out,
        )
        query_btn.click(
            fn=_query_handler,
            inputs=[
                query_q,
                query_mode,
                query_domain,
                query_topk,
                query_candidates,
                store_golden,
            ],
            outputs=[query_out_md, query_out],
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

        # nl2sql example loaders (question + mode + optional candidates)
        ex_nl2sql_1.click(
            lambda: ("各城市订单总额", "nl2sql", ""),
            outputs=[query_q, query_mode, query_candidates],
        )
        ex_nl2sql_2.click(
            lambda: ("订单金额与支付金额对比", "nl2sql", ""),
            outputs=[query_q, query_mode, query_candidates],
        )
        ex_nl2sql_3.click(
            lambda: ("各城市订单总额", "nl2sql", EXAMPLE_NL2SQL_CANDIDATES),
            outputs=[query_q, query_mode, query_candidates],
        )
