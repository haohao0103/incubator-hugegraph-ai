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

"""Gradio UI block for GraphRAG Core capabilities.

Merges the former \"Advanced GraphRAG\" (Tab 8) and \"GraphRAG Enhancement\"
(Tab 9) tabs into a single unified tab organized by **retrieval pipeline**:

  Section A — Graph Retrieval Engines (Phase 2 core)
    PPR Retriever / Cascade Propagation / Identity Edge Builder
    + Property Graph Extract (from old Tab 1)

  Section B — Retrieval Enhancement
    Dual Keyword / HyDE / RRF Multi-Channel Fusion

  Section C — Reasoning & Refinement (Phase 1→2 transition)
    DRIFT Multi-Hop Search / Gleaning / Token Budget Control

  Section D — Trustworthy Output
    Provenance Answer / Entity Resolution / Schema Validation

  Section E — Keyword Fallback
    BM25 Exact Match

  Section F — Chunk Graph Enhancement (from old Tab 1)
    Chunk Similarity Edges (SIMILAR edges via KNN)

This layout tells the full GraphRAG retrieval story: pick engine → enhance query →
reason deeply → output trustworthy results → fall back to keywords.
"""

import json

import gradio as gr

from hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers import (
    drift_search_answer,
    entity_resolve,
    rrf_demo,
    schema_validate,
    token_budget_demo,
)
from hugegraph_llm.demo.rag_demo.capability_closure_handlers import (
    chunk_sim_edges_build,
    property_graph_extract,
)
from hugegraph_llm.demo.rag_demo.graphrag_enhancement_handlers import (
    bm25_demo,
    cascade_propagation_demo,
    dual_keyword_demo,
    gleaning_demo,
    hyde_demo,
    identity_edge_demo,
    ppr_retriever_demo,
    provenance_demo,
)
from hugegraph_llm.utils.log import log


def create_graphrag_core_block():
    """Create the unified GraphRAG Core Gradio UI tab."""

    gr.Markdown("# GraphRAG Core")
    gr.Markdown(
        "**Complete GraphRAG capability set** — organized by retrieval pipeline stage.\n\n"
        "| Phase | Section | What it does | Competitor对标 |\n"
        "|-------|---------|-------------|----------------|\n"
        "| Phase 2 | **A. Retrieval Engine** | PPR / Cascade / Identity Edge | Fast-GraphRAG, HippoRAG2 |\n"
        "| Phase 1+2 | **B. Enhancement** | Dual Keyword / HyDE / RRF | LightRAG |\n"
        "| Phase 1+2 | **C. Reasoning** | DRIFT / Gleaning / Token Budget | MS-GraphRAG |\n"
        "| Production | **D. Trustworthy Output** | Provenance / ER / Validate | Enterprise requirement |\n"
        "| Fallback | **E. Keyword** | BM25 exact match | Our unique plugin |\n"
        "| Enhancement | **F. Chunk Graph** | SIMILAR edges via KNN | Internal innovation |"
    )

    # ══════════════════════════════════════════════════════════
    # A. Graph Retrieval Engines
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("A. Graph Retrieval Engines / 图检索引擎", open=True):
        gr.Markdown(
            "**Phase 2 core:** PPR (Personalized PageRank), Cascade Propagation "
            "(Entity→Relation→Chunk), Identity Edge Builder (same_as dedup). "
            "These are lightweight real-time alternatives to pre-built community indexes."
        )

        # --- A1: PPR Retriever ---
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

        ppr_btn.click(fn=_run_ppr, inputs=[ppr_query, ppr_alpha, ppr_depth, ppr_top_k],
                     outputs=[ppr_result, ppr_scores, ppr_context])

        # --- A2: Cascade Propagation ---
        gr.Markdown("---")
        gr.Markdown("### Cascade Propagation (三层传播) 🌊")

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

        cascade_btn.click(fn=_run_cascade,
                          inputs=[cascade_query, cascade_alpha, cascade_threshold, cascade_top_k],
                          outputs=[cascade_trace, cascade_entity_scores,
                                   cascade_relation_scores, cascade_chunk_scores])

        # --- A3: Identity Edge Builder ---
        gr.Markdown("---")
        gr.Markdown("### Identity Edge Builder (实体消解) 🔗")

        with gr.Row():
            with gr.Column(scale=3):
                identity_input = gr.Textbox(
                    label="实体名称(逗号分隔)",
                    placeholder="Apple Inc, Apple, AAPL, Microsoft Corp, MSFT",
                    lines=2,
                )
                identity_pairs = gr.JSON(label="相似实体对 / Entity Pairs")

            with gr.Column(scale=1):
                identity_threshold = gr.Slider(
                    label="Similarity Threshold", minimum=0.5, maximum=1.0,
                    value=0.9, step=0.05,
                )
                identity_top_k = gr.Slider(
                    label="Top-K Neighbors", minimum=1, maximum=20, value=5, step=1,
                )
                identity_btn = gr.Button("Build Identity Edges", variant="secondary")

        with gr.Accordion("合并建议 / Merge Suggestions", open=False):
            identity_merge = gr.JSON(label="合并组 / Merge Groups")

        identity_btn.click(fn=_run_identity,
                           inputs=[identity_input, identity_threshold, identity_top_k],
                           outputs=[identity_pairs, identity_merge])

    # ══════════════════════════════════════════════════════════
    # B. Retrieval Enhancement
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("B. Retrieval Enhancement / 检索增强", open=False):
        gr.Markdown(
            "Boost recall/precision before fusion: Dual Keywords (hl/ll layers), "
            "HyDE (hypothetical document embedding), RRF (multi-channel rank fusion)."
        )

        # --- B1: Dual Keyword ---
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
                    choices=["heuristic", "llm"], value="heuristic",
                    label="提取方式 / Method",
                )
                dk_lang = gr.Dropdown(
                    choices=["en", "zh"], value="en", label="语言 / Language",
                )
                dk_btn = gr.Button("Extract Keywords", variant="secondary")

        dk_btn.click(fn=_run_dual_keyword, inputs=[dk_query, dk_method, dk_lang],
                     outputs=[dk_hl, dk_ll])

        # --- B2: HyDE ---
        gr.Markdown("---")
        gr.Markdown("### HyDE (Hypothetical Document Embedding) 💡")

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

        hyde_btn.click(fn=_run_hyde, inputs=[hyde_query], outputs=[hyde_hypo, hyde_enhanced])

        # --- B3: RRF Fusion ---
        gr.Markdown("---")
        gr.Markdown("### RRF Multi-Channel Fusion")

        with gr.Row():
            rrf_query = gr.Textbox(
                label="Query", placeholder="Enter a search query...", lines=1,
            )
            rrf_top_k = gr.Slider(label="Top-K Results", minimum=1, maximum=20, value=5, step=1)
            rrf_btn = gr.Button("Run RRF Fusion", variant="secondary")

        with gr.Row():
            rrf_channels = gr.JSON(label="Per-Channel Results")
            rrf_fused = gr.JSON(label="RRF Fused Results")

        rrf_btn.click(fn=_run_rrf, inputs=[rrf_query, rrf_top_k],
                      outputs=[rrf_channels, rrf_fused])

    # ══════════════════════════════════════════════════════════
    # C. Reasoning & Refinement
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("C. Reasoning & Refinement / 推理与精炼", open=False):
        gr.Markdown(
            "Multi-hop reasoning and context budgeting: DRIFT (5-step iterative), "
            "Gleaning (follow-up extraction), Token Budget (3-level allocation)."
        )

        # --- C1: DRIFT ---
        gr.Markdown("### DRIFT Multi-Hop Reasoning Search")

        with gr.Row():
            with gr.Column(scale=3):
                drift_query = gr.Textbox(
                    label="Query",
                    placeholder="Ask a complex question requiring multi-hop reasoning...",
                    lines=2,
                )
                drift_answer_box = gr.Markdown(label="Final Answer")

            with gr.Column(scale=1):
                drift_top_k = gr.Slider(
                    label="Communities Top-K", minimum=1, maximum=20, value=5, step=1,
                )
                drift_lang = gr.Dropdown(
                    choices=["cn", "en"], value="cn", label="Language",
                )
                drift_btn = gr.Button("Run DRIFT Search", variant="primary")

        with gr.Accordion("Pipeline Trace (5 Steps)", open=True):
            drift_pipeline = gr.JSON(label="Step-by-step pipeline result")
            drift_metadata = gr.JSON(label="Metadata")
            drift_findings = gr.JSON(label="Top Findings")

        drift_btn.click(fn=_run_drift, inputs=[drift_query, drift_top_k, drift_lang],
                        outputs=[drift_answer_box, drift_pipeline, drift_metadata, drift_findings])

        # --- C2: Gleaning ---
        gr.Markdown("---")
        gr.Markdown("### Gleaning (追问提取) 🔎")

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
                    label="最大追问轮数 / Max Rounds", minimum=1, maximum=5, value=3, step=1,
                )
                glean_btn = gr.Button("Run Gleaning", variant="secondary")

        glean_answers = gr.JSON(label="渐进答案 / Progressive Answers")

        glean_btn.click(fn=_run_gleaning, inputs=[glean_query, glean_rounds],
                        outputs=[glean_follow_ups, glean_answers])

        # --- C3: Token Budget ---
        gr.Markdown("---")
        gr.Markdown("### Token Budget Control")

        with gr.Row():
            tb_query = gr.Textbox(
                label="Query", placeholder="Enter a query to simulate...", lines=1,
            )
            tb_max_tokens = gr.Slider(
                label="Max Tokens", minimum=500, maximum=8000, value=2000, step=100,
            )
            tb_btn = gr.Button("Simulate Budget", variant="secondary")

        with gr.Row():
            tb_summary = gr.JSON(label="Budget Summary")
            tb_context = gr.Textbox(label="Generated Context (truncated)", lines=5, interactive=False)

        tb_btn.click(fn=_run_token_budget, inputs=[tb_query, tb_max_tokens],
                     outputs=[tb_summary, tb_context])

    # ══════════════════════════════════════════════════════════
    # D. Trustworthy Output
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("D. Trustworthy Output / 可信输出", open=False):
        gr.Markdown(
            "Ensure answers are verifiable: Provenance (source tracing), "
            "Entity Resolution (dedup), Schema Validation (constraint check)."
        )

        # --- D1: Provenance ---
        gr.Markdown("### Provenance Answer (溯源回答) 📜")

        with gr.Row():
            prov_query = gr.Textbox(
                label="查询 / Query",
                placeholder="Enter a query for provenance-traced answer...",
                lines=2,
            )
            prov_btn = gr.Button("Run Provenance Answer", variant="secondary")

        prov_answer = gr.Textbox(label="溯源答案 / Provenance Answer", lines=6, interactive=False)
        prov_sources = gr.JSON(label="溯源路径 / Source Provenance")

        prov_btn.click(fn=_run_provenance, inputs=[prov_query],
                       outputs=[prov_answer, prov_sources])

        # --- D2: Entity Resolution ---
        gr.Markdown("---")
        gr.Markdown("### Entity Resolution (实体去重)")

        with gr.Row():
            with gr.Column(scale=2):
                entity_input = gr.Textbox(
                    label="Entity Names (one per line)",
                    placeholder="Apple Inc.\nApple (fruit)\nAAPL\nMicrosoft Corporation\nMSFT",
                    lines=6,
                )
            with gr.Column(scale=1):
                entity_strategy = gr.Dropdown(
                    choices=["exact_match", "embedding", "llm_verify", "hybrid"],
                    value="hybrid", label="Resolution Strategy",
                )
                entity_btn = gr.Button("Resolve Entities", variant="secondary")

        entity_result = gr.JSON(label="Resolution Groups")
        entity_summary = gr.Markdown(label="Summary")

        entity_btn.click(fn=_run_entity_resolve, inputs=[entity_input, entity_strategy],
                         outputs=[entity_result, entity_summary])

        # --- D3: Schema Validation ---
        gr.Markdown("---")
        gr.Markdown("### Schema Constraint Validation")

        schema_example = json.dumps({
            "entities": [
                {"name": "Person", "properties": {"name": "string", "age": "int"}, "cardinality": "SINGLE"},
                {"name": "Company", "properties": {"name": "string", "founded": "date"}, "cardinality": "SINGLE"},
            ],
            "relations": [
                {"source": "Person", "target": "Company", "label": "works_at", "cardinality": "MULTI"},
            ],
        }, indent=2)

        with gr.Row():
            with gr.Column(scale=2):
                schema_input = gr.Code(value=schema_example, label="Schema JSON",
                                       language="json", lines=10)
            with gr.Column(scale=1):
                schema_btn = gr.Button("Validate Schema", variant="secondary")

        schema_result = gr.JSON(label="Validation Result")
        schema_error_box = gr.Markdown(label="")

        schema_btn.click(fn=_run_schema_validate, inputs=[schema_input],
                         outputs=[schema_result, schema_error_box])

    # ══════════════════════════════════════════════════════════
    # E. Keyword Fallback (BM25)
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("E. BM25 Keyword Fallback / 关键词降级插件", open=False):
        gr.Markdown(
            "BM25 exact keyword match catches proper nouns, acronyms, code identifiers "
            "that vector search misses. Unique among all 4 competitors."
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
                bm25_top_k = gr.Slider(label="Top-K", minimum=1, maximum=30, value=10, step=1)
                bm25_btn = gr.Button("Run BM25 Search", variant="secondary")

        bm25_btn.click(fn=_run_bm25, inputs=[bm25_query, bm25_top_k], outputs=[bm25_results])

    # ══════════════════════════════════════════════════════════
    # F. Chunk Graph Enhancement
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("F. Chunk Graph Enhancement / 块图增强", open=False):
        gr.Markdown(
            "Build SIMILAR edges between Chunks using KNN on embeddings. "
            "Enhances RAG recall through chunk-to-chunk connectivity."
        )

        with gr.Row():
            with gr.Column(scale=1):
                cs_label = gr.Textbox(value="Chunk", label="Chunk Vertex Label")
                cs_top_k = gr.Slider(value=3, minimum=1, maximum=10, step=1, label="KNN Top-K")
                cs_min_score = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.05,
                                          label="Min Similarity")
                cs_btn = gr.Button("Build SIMILAR Edges", variant="secondary")
            with gr.Column(scale=2):
                cs_out = gr.Code(label="Chunk Similarity Result", language="json")

        cs_btn.click(fn=chunk_sim_edges_build, inputs=[cs_label, cs_top_k, cs_min_score],
                     outputs=[cs_out])

        # Property Graph Extract (standalone operator)
        gr.Markdown("---")
        gr.Markdown("### Property Graph Extract (独立抽取算子)")

        with gr.Row():
            with gr.Column(scale=2):
                pg_text = gr.Textbox(
                    label="Input Text (Property Graph)",
                    placeholder="Paste text to extract a property graph...",
                    lines=4,
                )
                pg_schema = gr.Code(
                    label="Schema JSON (optional)", language="json",
                    value='{"vertexlabels": [], "edgelabels": []}',
                )
                pg_btn = gr.Button("Extract Property Graph", variant="secondary")
            with gr.Column(scale=2):
                pg_out = gr.Code(label="Property Graph Result", language="json")

        pg_btn.click(fn=property_graph_extract, inputs=[pg_text, pg_schema], outputs=[pg_out])


# ── UI wrapper functions ──────────────────────────────────────

# --- Section A wrappers ---

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


# --- Section B wrappers ---

def _run_dual_keyword(query, method, language):
    result = dual_keyword_demo(query, method, language)
    hl = {"keywords": result.get("hl_keywords", []), "count": result.get("hl_count", 0)}
    ll = {"keywords": result.get("ll_keywords", []), "count": result.get("ll_count", 0)}
    if result.get("error"):
        hl["error"] = result["error"]
        ll["error"] = result["error"]
    return hl, ll


def _run_hyde(query):
    result = hyde_demo(query)
    hypo = result.get("hypothetical_answer", "")
    enhanced = result.get("enhanced_query", "")
    if result.get("error"):
        hypo = f"**Error:** {result['error']}"
    return hypo, enhanced


def _run_rrf(query: str, top_k: int):
    result = rrf_demo(query, top_k)
    channels = {
        "vector": result.get("vector_results", []),
        "graph": result.get("graph_results", []),
        "keyword": result.get("keyword_results", []),
    }
    fused = {
        "fused_results": result.get("fused_results", []),
        "fused_scores": result.get("fused_scores", {}),
    }
    if result.get("error"):
        fused["error"] = result["error"]
    return channels, fused


# --- Section C wrappers ---

def _run_drift(query: str, top_k: int, lang: str):
    result = drift_search_answer(query, top_k, lang)
    answer = result.get("answer", "")
    if result.get("error"):
        answer = f"**Error:** {result['error']}"
    return (
        answer,
        result.get("pipeline", []),
        result.get("metadata", {}),
        result.get("findings", []),
    )


def _run_gleaning(query, max_rounds):
    result = gleaning_demo(query, max_rounds)
    follow_ups = result.get("follow_up_questions", [])
    answers = result.get("progressive_answers", [])
    if result.get("error"):
        follow_ups = [{"error": result["error"]}]
    return follow_ups, answers


def _run_token_budget(query: str, max_tokens: int):
    result = token_budget_demo(query, max_tokens)
    summary = result.get("summary", {})
    if result.get("error"):
        summary = {"error": result["error"]}
    return summary, result.get("context", "")


# --- Section D wrappers ---

def _run_provenance(query):
    result = provenance_demo(query)
    answer = result.get("answer", "")
    sources = result.get("provenance", [])
    if result.get("error"):
        answer = f"**Error:** {result['error']}"
    return answer, sources


def _run_entity_resolve(entities_text: str, strategy: str):
    result = entity_resolve(entities_text, strategy)
    summary = (
        f"**Strategy:** {result.get('strategy', 'N/A')} | "
        f"Total: {result.get('total_entities', 0)} | "
        f"Resolved: {result.get('resolved_count', 0)} | "
        f"Unresolved: {result.get('unresolved_count', 0)}"
    )
    if result.get("error"):
        summary = f"**Error:** {result['error']}"
    return result.get("groups", []), summary


def _run_schema_validate(schema_json: str):
    result = schema_validate(schema_json)
    display = {
        "valid": result["valid"],
        "entity_count": result["entity_count"],
        "relation_count": result["relation_count"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "suggestions": result["suggestions"],
    }
    status = (
        f"**Valid:** {'Yes' if result['valid'] else 'No'} | "
        f"Entities: {result['entity_count']} | Relations: {result['relation_count']}"
    )
    return display, status


# --- Section E wrapper ---

def _run_bm25(query, top_k):
    result = bm25_demo(query, top_k)
    results = result.get("results", [])
    if result.get("error"):
        results = [{"error": result["error"]}]
    return results
