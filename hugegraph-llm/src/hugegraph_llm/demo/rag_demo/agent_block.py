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

"""Gradio UI block for Agent-based reasoning and Global Search."""

import json

import gradio as gr

from hugegraph_llm.demo.rag_demo.agent_handlers import (
    agent_answer,
    community_build,
    global_search,
    graph_rag_search,
)
from hugegraph_llm.demo.rag_demo.capability_closure_handlers import query_classifier_demo
from hugegraph_llm.utils.log import log


def create_agent_block():
    """Create the Agent & Global Search Gradio UI tab."""
    gr.Markdown("## 🤖 Agent 多步推理 (Multi-Step Reasoning)")
    gr.Markdown("LLM 驱动的 Agent 自动选择合适的工具来探索知识图谱，回答复杂问题。")

    with gr.Row():
        with gr.Column(scale=3):
            agent_query = gr.Textbox(
                label="🔍 查询 / Query",
                placeholder="例如: 比较实体 A 和 B 的关系，分析 X 如何通过网络影响 Y...",
                lines=2,
            )
        with gr.Column(scale=1):
            agent_max_steps = gr.Slider(
                label="最大步数 / Max Steps",
                minimum=1,
                maximum=20,
                value=10,
                step=1,
            )
            agent_btn = gr.Button("🚀 运行 Agent", variant="primary")

    with gr.Accordion("Agent 推理过程 / Reasoning Trace", open=False):
        agent_trace = gr.JSON(label="推理步骤 / Steps")

    agent_answer_box = gr.Textbox(
        label="📝 Agent 答案",
        lines=8,
        interactive=False,
    )

    agent_btn.click(
        fn=run_agent,
        inputs=[agent_query, agent_max_steps],
        outputs=[agent_answer_box, agent_trace],
    )

    # ── Global Search section ─────────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 🌐 全局搜索 / Global Search")
    gr.Markdown("基于社区摘要的宏观问答，回答跨文档的主题性问题。需先构建社区索引。")

    with gr.Row():
        with gr.Column(scale=3):
            global_query = gr.Textbox(
                label="🔍 全局查询 / Global Query",
                placeholder="例如: 所有文档的主要主题是什么？知识图谱的总体结构是怎样的？",
                lines=2,
            )
        with gr.Column(scale=1):
            global_btn = gr.Button("🌍 全局搜索", variant="secondary")

    global_answer_box = gr.Textbox(
        label="📝 全局答案 / Global Answer",
        lines=8,
        interactive=False,
    )

    global_btn.click(
        fn=run_global_search,
        inputs=[global_query],
        outputs=[global_answer_box],
    )

    # ── Community build section ───────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 🏗️ 社区检测 / Community Detection")
    gr.Markdown("构建社区索引，启用全局搜索。离线操作，通常只需执行一次。")

    with gr.Row():
        community_algo = gr.Dropdown(
            choices=["louvain", "wcc"],
            value="louvain",
            label="算法 / Algorithm",
        )
        community_levels = gr.Slider(
            label="层级 / Levels",
            minimum=1,
            maximum=3,
            value=2,
            step=1,
        )
        community_btn = gr.Button("🔧 构建社区索引", variant="secondary")

    community_status = gr.Textbox(
        label="状态 / Status",
        lines=3,
        interactive=False,
    )

    community_btn.click(
        fn=run_community_build,
        inputs=[community_algo, community_levels],
        outputs=[community_status],
    )

    # ── Graph RAG Search section ─────────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 🔎 图搜索操作 / Graph RAG Search")
    gr.Markdown("直接调用图搜索工具：图遍历、语义ID查找、Text2Gremlin、Schema查询。")

    with gr.Row():
        graph_search_mode = gr.Dropdown(
            choices=["graph_traverse", "semantic_id_lookup", "text2gremlin", "schema_lookup"],
            value="graph_traverse",
            label="操作模式 / Mode",
        )
        graph_search_btn = gr.Button("🔍 执行图搜索", variant="secondary")

    with gr.Row():
        graph_search_query = gr.Textbox(
            label="查询文本 / Query",
            placeholder="输入查询文本（text2gremlin 模式必需）",
            lines=1,
        )
        graph_search_vids = gr.Textbox(
            label="顶点ID列表 / Vertex IDs",
            placeholder="逗号分隔的顶点ID（graph_traverse 模式必需）",
            lines=1,
        )

    with gr.Row():
        graph_search_depth = gr.Slider(
            label="遍历深度 / Max Depth",
            minimum=0,
            maximum=10,
            value=2,
            step=1,
        )
        graph_search_items = gr.Slider(
            label="最大结果数 / Max Items",
            minimum=1,
            maximum=100,
            value=10,
            step=1,
        )

    graph_search_keywords = gr.Textbox(
        label="关键词列表 / Keywords",
        placeholder="逗号分隔的关键词（semantic_id_lookup 模式必需）",
        lines=1,
    )

    graph_search_result = gr.JSON(label="搜索结果 / Search Result")

    graph_search_btn.click(
        fn=run_graph_rag_search,
        inputs=[
            graph_search_mode,
            graph_search_query,
            graph_search_vids,
            graph_search_depth,
            graph_search_items,
            graph_search_keywords,
        ],
        outputs=[graph_search_result],
    )

    # ── Query Classifier (from Capability Closure) ────────────────
    gr.Markdown("---")
    gr.Markdown("## 🎯 查询分类 / Query Classifier")
    gr.Markdown("将查询分类为简单或复杂，决定路由到快速图搜索还是 Agent 多步推理。")

    with gr.Row():
        with gr.Column(scale=2):
            qc_query = gr.Textbox(
                label="查询 / Query",
                placeholder="输入查询以分类为简单或复杂...",
                lines=2,
            )
            qc_use_llm = gr.Checkbox(value=False, label="使用 LLM 进行精细分类")
            qc_btn = gr.Button("分类查询 / Classify", variant="secondary")
        with gr.Column(scale=2):
            qc_out = gr.Code(label="分类结果 / Classification Result", language="json")

    qc_btn.click(
        fn=query_classifier_demo,
        inputs=[qc_query, qc_use_llm],
        outputs=[qc_out],
    )


def run_agent(query: str, max_steps: int) -> tuple:
    """Run the ReAct agent and return answer + trace."""
    if not query or not query.strip():
        return "请输入查询。", []

    try:
        result = agent_answer(query=query, max_steps=max_steps)
        answer = result.get("answer", "")
        trace = result.get("trace", [])
        log.info("Agent UI: completed in %d steps", len(trace))
        return answer, trace
    except Exception as e:
        log.error("Agent UI error: %s", e)
        return f"Agent 运行失败: {str(e)}", []


def run_global_search(query: str) -> str:
    """Run Global Search and return answer."""
    if not query or not query.strip():
        return "请输入查询。"

    try:
        result = global_search(query=query)
        answer = result.get("answer", "")
        used = result.get("communities_used", 0)
        return f"{answer}\n\n*(使用了 {used} 个社区)*"
    except Exception as e:
        log.error("Global search UI error: %s", e)
        return f"全局搜索失败: {str(e)}"


def run_community_build(algorithm: str, max_levels: int) -> str:
    """Trigger community detection and indexing."""
    try:
        result = community_build(algorithm=algorithm, max_levels=max_levels)
        count = result.get("community_count", 0)
        reports = result.get("report_count", 0)
        built = result.get("index_built", False)
        return (
            f"✅ 社区检测完成\n"
            f"- 检测到 {count} 个社区\n"
            f"- 生成了 {reports} 个社区报告\n"
            f"- 索引构建: {'成功' if built else '失败'}"
        )
    except Exception as e:
        log.error("Community build UI error: %s", e)
        return f"❌ 社区检测失败: {str(e)}"


def run_graph_rag_search(
    mode: str,
    query: str,
    vertex_ids_str: str,
    max_depth: int,
    max_items: int,
    keywords_str: str,
) -> dict:
    """Run a direct graph RAG search operation and return JSON result."""
    # Parse vertex IDs and keywords from comma-separated strings
    vertex_ids = None
    if vertex_ids_str and vertex_ids_str.strip():
        vertex_ids = [v.strip() for v in vertex_ids_str.split(",") if v.strip()]

    keywords = None
    if keywords_str and keywords_str.strip():
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

    try:
        result = graph_rag_search(
            mode=mode,
            query=query or None,
            vertex_ids=vertex_ids,
            max_depth=max_depth,
            max_items=max_items,
            keywords=keywords,
        )
        log.info("Graph RAG search UI: mode=%s, success=%s", mode, result.get("success"))
        return result
    except Exception as e:
        log.error("Graph RAG search UI error: %s", e)
        return {"success": False, "error": str(e), "mode": mode}
