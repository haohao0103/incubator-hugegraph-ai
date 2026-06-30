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

"""Gradio UI block for GraphRAG Enhancement features.

Provides interactive demos for capabilities NOT yet shown in the
existing 7 tabs:

- PPR Retriever (Personalized PageRank)
- Cascade Propagation (Entity→Relation→Chunk 3-layer)
- Identity Edge Builder (same_as edges for entity resolution)
- Dual Keyword Extract (hl_keywords + ll_keywords)
- Community Summary Generator
- HyDE Query Enhancement
- Gleaning (Follow-up Extraction)
- Provenance Answer (Source Tracing)
- BM25 Keyword Search (Optional Plugin)

These showcase the P0 gap closure modules aligned with competitors.
"""

import json

import gradio as gr

from hugegraph_llm.demo.rag_demo.graphrag_enhancement_handlers import (
    bm25_demo,
    cascade_propagation_demo,
    community_summary_demo,
    dual_keyword_demo,
    gleaning_demo,
    hyde_demo,
    identity_edge_demo,
    ppr_retriever_demo,
    provenance_demo,
)
from hugegraph_llm.utils.log import log


def create_graphrag_enhancement_block():
    """Create the GraphRAG Enhancement showcase Gradio UI tab."""

    # ── Section 1: PPR Retriever ──────────────────────────────
    gr.Markdown(
        "## 1. PPR Retriever (Personalized PageRank) 🔍"
    )
    gr.Markdown(
        "Personalized PageRank从种子实体出发，在知识图谱上扩散检索相关信息。"
        "对标Fast-GraphRAG/HippoRAG2的核心图检索算法。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            ppr_query = gr.Textbox(
                label="查询 / Query",
                placeholder="Enter a query to seed PPR traversal...",
                lines=2,
            )
            ppr_result = gr.JSON(label="PPR检索结果")

        with gr.Column(scale=1):
            ppr_alpha = gr.Slider(
                label="Alpha (Teleport概率)", minimum=0.01, maximum=0.99,
                value=0.15, step=0.01,
            )
            ppr_depth = gr.Slider(
                label="Max Depth", minimum=1, maximum=5, value=2, step=1,
            )
            ppr_top_k = gr.Slider(
                label="Top-K Results", minimum=1, maximum=50, value=10, step=1,
            )
            ppr_btn = gr.Button("Run PPR Search", variant="primary")

    with gr.Accordion("PPR详细结果", open=False):
        ppr_scores = gr.JSON(label="PPR Scores (Top entities)")
        ppr_context = gr.Textbox(label="Retrieved Context", lines=5, interactive=False)

    ppr_btn.click(
        fn=_run_ppr,
        inputs=[ppr_query, ppr_alpha, ppr_depth, ppr_top_k],
        outputs=[ppr_result, ppr_scores, ppr_context],
    )

    # ── Section 2: Cascade Propagation ────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 2. Cascade Propagation (三层传播) 🌊"
    )
    gr.Markdown(
        "Entity→Relation→Chunk三层稀疏矩阵传播，PPR分数从实体层经关系层"
        "传播到原文chunk层。对标Fast-GraphRAG的级联传播架构。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            cascade_query = gr.Textbox(
                label="查询 / Query",
                placeholder="Enter a query for cascade propagation...",
                lines=2,
            )
            cascade_trace = gr.JSON(label="传播步骤 / Propagation Trace")

        with gr.Column(scale=1):
            cascade_alpha = gr.Slider(
                label="PPR Alpha", minimum=0.01, maximum=0.99,
                value=0.15, step=0.01,
            )
            cascade_threshold = gr.Slider(
                label="Entity Threshold", minimum=0.001, maximum=0.1,
                value=0.01, step=0.001,
            )
            cascade_top_k = gr.Slider(
                label="Chunk Top-K", minimum=1, maximum=30, value=10, step=1,
            )
            cascade_btn = gr.Button("Run Cascade", variant="secondary")

    with gr.Row():
        cascade_entity_scores = gr.JSON(label="Entity Layer Scores")
        cascade_relation_scores = gr.JSON(label="Relation Layer Scores")
        cascade_chunk_scores = gr.JSON(label="Chunk Layer Scores")

    cascade_btn.click(
        fn=_run_cascade,
        inputs=[cascade_query, cascade_alpha, cascade_threshold, cascade_top_k],
        outputs=[cascade_trace, cascade_entity_scores, cascade_relation_scores, cascade_chunk_scores],
    )

    # ── Section 3: Identity Edge Builder ──────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 3. Identity Edge Builder (实体消解) 🔗"
    )
    gr.Markdown(
        "Embedding相似度>阈值的实体间创建same_as边，解决同名/别名/缩写问题。"
        "对标Fast-GraphRAG的identity edges。这是Medical graph_hits=0的根因修复。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            identity_input = gr.Textbox(
                label="实体名称(逗号分隔) / Entity Names (comma-separated)",
                placeholder="Apple Inc, Apple, AAPL, Microsoft Corp, MSFT, MicroSoft",
                lines=2,
            )
            identity_pairs = gr.JSON(label="相似实体对 / Entity Pairs")

        with gr.Column(scale=1):
            identity_threshold = gr.Slider(
                label="相似度阈值 / Similarity Threshold",
                minimum=0.5, maximum=1.0, value=0.9, step=0.05,
            )
            identity_top_k = gr.Slider(
                label="每实体最大边数 / Top-K Neighbors",
                minimum=1, maximum=20, value=5, step=1,
            )
            identity_btn = gr.Button("Build Identity Edges", variant="secondary")

    with gr.Accordion("合并建议 / Merge Suggestions", open=False):
        identity_merge = gr.JSON(label="合并组 / Merge Groups")

    identity_btn.click(
        fn=_run_identity,
        inputs=[identity_input, identity_threshold, identity_top_k],
        outputs=[identity_pairs, identity_merge],
    )

    # ── Section 4: Dual Keyword Extract ──────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 4. Dual Keyword Extract (双层关键词) 📝"
    )
    gr.Markdown(
        "从查询中提取两层关键词：hl_keywords(高层抽象概念，用于社区匹配) "
        "和 ll_keywords(低层具体实体，用于精确检索)。对标LightRAG的双层检索范式。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            dk_query = gr.Textbox(
                label="查询 / Query",
                placeholder="What is the treatment for diabetes?",
                lines=2,
            )
            dk_hl = gr.JSON(label="hl_keywords (高层概念)")
            dk_ll = gr.JSON(label="ll_keywords (低层实体)")

        with gr.Column(scale=1):
            dk_method = gr.Dropdown(
                choices=["heuristic", "llm"],
                value="heuristic",
                label="提取方式 / Method",
            )
            dk_lang = gr.Dropdown(
                choices=["en", "zh"],
                value="en",
                label="语言 / Language",
            )
            dk_btn = gr.Button("Extract Keywords", variant="secondary")

    dk_btn.click(
        fn=_run_dual_keyword,
        inputs=[dk_query, dk_method, dk_lang],
        outputs=[dk_hl, dk_ll],
    )

    # ── Section 5: Community Summary Generator ────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 5. Community Summary Generator (社区摘要) 📊"
    )
    gr.Markdown(
        "社区检测(Louvain/WCC) + LLM/Heuristic社区摘要生成。"
        "对标MS-GraphRAG的社区摘要+Global Search。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            cs_algo = gr.Dropdown(
                choices=["louvain", "wcc"],
                value="louvain",
                label="社区检测算法 / Algorithm",
            )
            cs_levels = gr.Slider(
                label="层级 / Levels", minimum=1, maximum=3, value=2, step=1,
            )
            cs_method = gr.Dropdown(
                choices=["heuristic", "llm"],
                value="heuristic",
                label="摘要方式 / Summary Method",
            )

        with gr.Column(scale=1):
            cs_btn = gr.Button("Generate Summaries", variant="secondary")

    cs_reports = gr.JSON(label="社区摘要报告 / Community Reports")
    cs_stats = gr.Markdown(label="")

    cs_btn.click(
        fn=_run_community_summary,
        inputs=[cs_algo, cs_levels, cs_method],
        outputs=[cs_reports, cs_stats],
    )

    # ── Section 6: HyDE Query Enhancement ─────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 6. HyDE (Hypothetical Document Embedding) 💡"
    )
    gr.Markdown(
        "先让LLM生成假设性答案，再用假设答案的embedding代替原始query embedding"
        "进行检索，提升语义匹配精度。"
    )

    with gr.Row():
        hyde_query = gr.Textbox(
            label="原始查询 / Original Query",
            placeholder="Enter a query to enhance...",
            lines=2,
        )
        hyde_btn = gr.Button("Generate HyDE", variant="secondary")

    with gr.Row():
        hyde_hypo = gr.Textbox(label="假设性答案 / Hypothetical Answer", lines=4, interactive=False)
        hyde_enhanced = gr.Textbox(label="增强查询 / Enhanced Query", lines=2, interactive=False)

    hyde_btn.click(
        fn=_run_hyde,
        inputs=[hyde_query],
        outputs=[hyde_hypo, hyde_enhanced],
    )

    # ── Section 7: Gleaning ────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 7. Gleaning (追问提取) 🔎"
    )
    gr.Markdown(
        "从初始回答中自动提取追问，逐步深化检索直到信息充分。"
        "对标MS-GraphRAG的DRIFT迭代策略。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            glean_query = gr.Textbox(
                label="查询 / Query",
                placeholder="Enter a query for gleaning extraction...",
                lines=2,
            )
            glean_follow_ups = gr.JSON(label="追问列表 / Follow-up Questions")

        with gr.Column(scale=1):
            glean_rounds = gr.Slider(
                label="最大追问轮数 / Max Rounds",
                minimum=1, maximum=5, value=3, step=1,
            )
            glean_btn = gr.Button("Run Gleaning", variant="secondary")

    glean_answers = gr.JSON(label="渐进答案 / Progressive Answers")

    glean_btn.click(
        fn=_run_gleaning,
        inputs=[glean_query, glean_rounds],
        outputs=[glean_follow_ups, glean_answers],
    )

    # ── Section 8: Provenance Answer ────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 8. Provenance Answer (溯源回答) 📜"
    )
    gr.Markdown(
        "回答问题时附带信息溯源路径，标注每个结论来自哪个原文chunk/实体/关系。"
        "确保答案可信可验证。"
    )

    with gr.Row():
        prov_query = gr.Textbox(
            label="查询 / Query",
            placeholder="Enter a query for provenance-traced answer...",
            lines=2,
        )
        prov_btn = gr.Button("Run Provenance Answer", variant="secondary")

    prov_answer = gr.Textbox(label="溯源答案 / Provenance Answer", lines=6, interactive=False)
    prov_sources = gr.JSON(label="溯源路径 / Source Provenance")

    prov_btn.click(
        fn=_run_provenance,
        inputs=[prov_query],
        outputs=[prov_answer, prov_sources],
    )

    # ── Section 9: BM25 Keyword Search ─────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 9. BM25 Keyword Search (可选插件) 📚"
    )
    gr.Markdown(
        "BM25精确关键词检索，捕捉专有名词、缩写、代码标识符等向量搜索"
        "难以匹配的内容。降级为可选插件(4竞品全无BM25独立通道)。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            bm25_query = gr.Textbox(
                label="查询 / Query",
                placeholder="Enter keywords for exact matching...",
                lines=2,
            )
            bm25_results = gr.JSON(label="BM25检索结果")

        with gr.Column(scale=1):
            bm25_top_k = gr.Slider(
                label="Top-K", minimum=1, maximum=30, value=10, step=1,
            )
            bm25_btn = gr.Button("Run BM25 Search", variant="secondary")

    bm25_btn.click(
        fn=_run_bm25,
        inputs=[bm25_query, bm25_top_k],
        outputs=[bm25_results],
    )


# ── UI wrapper functions ──────────────────────────────────────

def _run_ppr(query, alpha, depth, top_k):
    result = ppr_retriever_demo(query, alpha, depth, top_k)
    summary = {
        "total_entities_reached": result.get("total_entities_reached", 0),
        "alpha": result.get("alpha", alpha),
        "max_depth": result.get("max_depth", depth),
        "seed_entities": result.get("seed_entities", [])[:5],
        "error": result.get("error"),
    }
    scores = result.get("ppr_scores", {})
    context = result.get("context", "")
    if result.get("error"):
        context = f"**Error:** {result['error']}"
    return summary, scores, context


def _run_cascade(query, alpha, threshold, top_k):
    result = cascade_propagation_demo(query, alpha, threshold, top_k)
    trace = result.get("propagation_trace", [])
    if result.get("error"):
        trace = [{"error": result["error"]}]
    return (
        trace,
        result.get("entity_scores", {}),
        result.get("relation_scores", {}),
        result.get("chunk_scores", {}),
    )


def _run_identity(entities_text, threshold, top_k):
    result = identity_edge_demo(entities_text, threshold, top_k)
    pairs = result.get("entity_pairs", [])
    if result.get("error"):
        pairs = [{"error": result["error"]}]
    merge = result.get("merge_suggestions", [])
    return pairs, merge


def _run_dual_keyword(query, method, language):
    result = dual_keyword_demo(query, method, language)
    hl = {"keywords": result.get("hl_keywords", []), "count": result.get("hl_count", 0)}
    ll = {"keywords": result.get("ll_keywords", []), "count": result.get("ll_count", 0)}
    if result.get("error"):
        hl["error"] = result["error"]
        ll["error"] = result["error"]
    return hl, ll


def _run_community_summary(algo, levels, method):
    result = community_summary_demo(algo, levels, method)
    reports = result.get("community_reports", [])
    if result.get("error"):
        reports = [{"error": result["error"]}]
    stats = (
        f"**Communities:** {result.get('total_communities', 0)} | "
        f"**Reports:** {result.get('total_reports', 0)} | "
        f"**Algorithm:** {result.get('algorithm', algo)} | "
        f"**Method:** {result.get('summary_method', method)}"
    )
    if result.get("error"):
        stats = f"**Error:** {result['error']}"
    return reports, stats


def _run_hyde(query):
    result = hyde_demo(query)
    hypo = result.get("hypothetical_answer", "")
    enhanced = result.get("enhanced_query", "")
    if result.get("error"):
        hypo = f"**Error:** {result['error']}"
    return hypo, enhanced


def _run_gleaning(query, max_rounds):
    result = gleaning_demo(query, max_rounds)
    follow_ups = result.get("follow_up_questions", [])
    answers = result.get("progressive_answers", [])
    if result.get("error"):
        follow_ups = [{"error": result["error"]}]
    return follow_ups, answers


def _run_provenance(query):
    result = provenance_demo(query)
    answer = result.get("answer", "")
    sources = result.get("provenance", [])
    if result.get("error"):
        answer = f"**Error:** {result['error']}"
    return answer, sources


def _run_bm25(query, top_k):
    result = bm25_demo(query, top_k)
    results = result.get("results", [])
    if result.get("error"):
        results = [{"error": result["error"]}]
    return results
