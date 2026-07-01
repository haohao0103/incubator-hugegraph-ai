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

"""Gradio UI block for Advanced GraphRAG features showcase.

Provides interactive demos for:
- DRIFT multi-hop search (5-step pipeline visualization)
- Schema validation (graph constraint checking)
- Entity resolution (deduplication visualization)
- Community reports viewer
- RRF multi-channel fusion demo
- Token Budget allocation demo

Designed for stakeholder demos and product showcases.
"""

import json

import gradio as gr

from hugegraph_llm.demo.rag_demo.advanced_graphrag_handlers import (
    drift_search_answer,
    entity_resolve,
    get_community_reports,
    incremental_index_status,
    rrf_demo,
    schema_validate,
    token_budget_demo,
)
from hugegraph_llm.demo.rag_demo.capability_closure_handlers import (
    incremental_index_flow,
    synonym_add,
    synonym_expand,
    synonym_list,
)
from hugegraph_llm.utils.log import log


def create_advanced_graphrag_block():
    """Create the Advanced GraphRAG showcase Gradio UI tab."""

    # ── Section 1: DRIFT Search ──────────────────────────────
    gr.Markdown(
        "## 1. DRIFT Multi-Hop Reasoning Search"
    )
    gr.Markdown(
        "DRIFT (DRiving Iterative Feedback Search) performs multi-hop "
        "reasoning through a 5-step pipeline: HyDE → Community Match → "
        "Primer → Parallel Local Search → Reduce."
    )

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
                label="Communities Top-K", minimum=1, maximum=20, value=5, step=1
            )
            drift_lang = gr.Dropdown(
                choices=["cn", "en"], value="cn", label="Language"
            )
            drift_btn = gr.Button("Run DRIFT Search", variant="primary")

    with gr.Accordion("Pipeline Trace (5 Steps)", open=True):
        drift_pipeline = gr.JSON(label="Step-by-step pipeline result")
        drift_metadata = gr.JSON(label="Metadata")
        drift_findings = gr.JSON(label="Top Findings")

    drift_btn.click(
        fn=_run_drift,
        inputs=[drift_query, drift_top_k, drift_lang],
        outputs=[drift_answer_box, drift_pipeline, drift_metadata, drift_findings],
    )

    # ── Section 2: Schema Validation ─────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 2. Schema Constraint Validation"
    )
    gr.Markdown(
        "Validate graph schema definitions against type, cardinality, "
        "and relationship constraints."
    )

    schema_example = json.dumps(
        {
            "entities": [
                {"name": "Person", "properties": {"name": "string", "age": "int"}, "cardinality": "SINGLE"},
                {"name": "Company", "properties": {"name": "string", "founded": "date"}, "cardinality": "SINGLE"},
            ],
            "relations": [
                {"source": "Person", "target": "Company", "label": "works_at", "cardinality": "MULTI"},
            ],
        },
        indent=2,
    )

    with gr.Row():
        with gr.Column(scale=2):
            schema_input = gr.Code(
                value=schema_example,
                label="Schema JSON",
                language="json",
                lines=10,
            )
        with gr.Column(scale=1):
            schema_btn = gr.Button("Validate Schema", variant="secondary")

    schema_result = gr.JSON(label="Validation Result")
    schema_error_box = gr.Markdown(label="")

    schema_btn.click(
        fn=_run_schema_validate,
        inputs=[schema_input],
        outputs=[schema_result, schema_error_box],
    )

    # ── Section 3: Entity Resolution ────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 3. Entity Resolution (Deduplication)"
    )
    gr.Markdown(
        "Identify and merge duplicate entities using exact match, "
        "embedding similarity, LLM verification, or hybrid strategy."
    )

    with gr.Row():
        with gr.Column(scale=2):
            entity_input = gr.Textbox(
                label="Entity Names (one per line)",
                placeholder="Apple Inc.\nApple (fruit)\nAAPL\nMicrosoft Corporation\nMSFT\nMicroSoft",
                lines=6,
            )
        with gr.Column(scale=1):
            entity_strategy = gr.Dropdown(
                choices=["exact_match", "embedding", "llm_verify", "hybrid"],
                value="hybrid",
                label="Resolution Strategy",
            )
            entity_btn = gr.Button("Resolve Entities", variant="secondary")

    entity_result = gr.JSON(label="Resolution Groups")
    entity_summary = gr.Markdown(label="Summary")

    entity_btn.click(
        fn=_run_entity_resolve,
        inputs=[entity_input, entity_strategy],
        outputs=[entity_result, entity_summary],
    )

    # ── Section 4: Community Reports ──────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 4. Community Detection & Reports"
    )
    gr.Markdown(
        "View generated community reports with structured summaries, "
        "key entities, and importance scores."
    )

    with gr.Row():
        community_limit = gr.Slider(label="Max Reports", minimum=1, maximum=50, value=10, step=1)
        community_btn = gr.Button("Load Community Reports", variant="secondary")

    community_reports = gr.JSON(label="Community Reports")
    community_status = gr.Markdown(label="")

    community_btn.click(
        fn=_run_community_reports,
        inputs=[community_limit],
        outputs=[community_reports, community_status],
    )

    # ── Section 5: RRF Fusion Demo ────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 5. RRF Multi-Channel Fusion"
    )
    gr.Markdown(
        "Reciprocal Rank Fusion merges results from vector, graph, "
        "and keyword retrieval channels into a single ranked list."
    )

    with gr.Row():
        rrf_query = gr.Textbox(
            label="Query", placeholder="Enter a search query...", lines=1
        )
        rrf_top_k = gr.Slider(label="Top-K Results", minimum=1, maximum=20, value=5, step=1)
        rrf_btn = gr.Button("Run RRF Fusion", variant="secondary")

    with gr.Row():
        rrf_channels = gr.JSON(label="Per-Channel Results")
        rrf_fused = gr.JSON(label="RRF Fused Results")

    rrf_btn.click(
        fn=_run_rrf,
        inputs=[rrf_query, rrf_top_k],
        outputs=[rrf_channels, rrf_fused],
    )

    # ── Section 6: Token Budget Demo ──────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 6. Token Budget Control"
    )
    gr.Markdown(
        "Three-level token allocation (entity / relation / community) "
        "ensures the final LLM context stays within model limits."
    )

    with gr.Row():
        tb_query = gr.Textbox(
            label="Query", placeholder="Enter a query to simulate...", lines=1
        )
        tb_max_tokens = gr.Slider(
            label="Max Tokens", minimum=500, maximum=8000, value=2000, step=100
        )
        tb_btn = gr.Button("Simulate Budget", variant="secondary")

    with gr.Row():
        tb_summary = gr.JSON(label="Budget Summary")
        tb_context = gr.Textbox(label="Generated Context (truncated)", lines=5, interactive=False)

    tb_btn.click(
        fn=_run_token_budget,
        inputs=[tb_query, tb_max_tokens],
        outputs=[tb_summary, tb_context],
    )

    # ── Section 7: Incremental Index Flow ────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 7. Incremental Index Flow")
    gr.Markdown("Add new documents to an existing graph index without rebuilding from scratch.")

    with gr.Row():
        with gr.Column(scale=2):
            inc_text = gr.Textbox(
                label="New Document Texts",
                placeholder="Paste new document content. Use '---' to separate multiple documents.",
                lines=6,
            )
            inc_graph = gr.Textbox(value="", label="Graph Name (optional)")
            inc_btn = gr.Button("Run Incremental Index", variant="secondary")
        with gr.Column(scale=2):
            inc_out = gr.Code(label="Incremental Index Result", language="json")

    inc_btn.click(
        fn=incremental_index_flow,
        inputs=[inc_text, inc_graph],
        outputs=[inc_out],
    )

    # ── Section 8: Synonym Manager ────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 8. Synonym Manager")
    gr.Markdown("Manage synonym groups for query expansion and entity normalization.")

    with gr.Row():
        with gr.Column(scale=1):
            syn_canonical = gr.Textbox(label="Canonical Term", placeholder="physical car model")
            syn_aliases = gr.Textbox(label="Aliases (comma-separated)", placeholder="actual car model, vehicle type")
            syn_category = gr.Textbox(label="Category", value="general")
            syn_add_btn = gr.Button("Add Synonym Group", variant="secondary")
        with gr.Column(scale=1):
            syn_query = gr.Textbox(label="Query to Expand", placeholder="where is the actual car model field")
            syn_expand_btn = gr.Button("Expand Query", variant="secondary")
        with gr.Column(scale=1):
            syn_list_btn = gr.Button("List Synonyms", variant="secondary")

    with gr.Row():
        syn_add_out = gr.Code(label="Add Result", language="json")
        syn_expand_out = gr.Code(label="Expanded Query", language="json")
        syn_list_out = gr.Code(label="Synonym List", language="json")

    syn_add_btn.click(
        fn=synonym_add,
        inputs=[syn_canonical, syn_aliases, syn_category],
        outputs=[syn_add_out],
    )
    syn_expand_btn.click(
        fn=synonym_expand,
        inputs=[syn_query],
        outputs=[syn_expand_out],
    )
    syn_list_btn.click(
        fn=synonym_list,
        inputs=[],
        outputs=[syn_list_out],
    )


# ── UI wrapper functions ──────────────────────────────────────

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


def _run_community_reports(limit: int):
    result = get_community_reports(limit)
    status = f"**Reports loaded:** {result['total_reports']}"
    if result.get("error"):
        status = f"**Error:** {result['error']}"
    return result.get("reports", []), status


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


def _run_token_budget(query: str, max_tokens: int):
    result = token_budget_demo(query, max_tokens)
    summary = result.get("summary", {})
    if result.get("error"):
        summary = {"error": result["error"]}
    return summary, result.get("context", "")
