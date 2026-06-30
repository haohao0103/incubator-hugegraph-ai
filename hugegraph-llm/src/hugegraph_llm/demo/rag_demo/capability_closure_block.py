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

"""Gradio UI block for Capability Closure.

Adds interactive demos for all capabilities that were missing from the
existing Gradio tabs:
- Multimodal RAG
- Property Graph Extraction
- Incremental Index Flow
- Gremlin Self-Correction
- Query Classifier
- Synonym Manager
- Chunk Similarity Edges
"""

import gradio as gr

from hugegraph_llm.demo.rag_demo.capability_closure_handlers import (
    chunk_sim_edges_build,
    gremlin_self_correct,
    incremental_index_flow,
    multimodal_build_kg,
    multimodal_describe_images,
    multimodal_extract_pdf,
    multimodal_search,
    property_graph_extract,
    query_classifier_demo,
    synonym_add,
    synonym_expand,
    synonym_list,
)
from hugegraph_llm.utils.log import log


def create_capability_closure_block():
    """Create the Capability Closure Gradio UI tab."""
    gr.Markdown("# Capability Closure / 能力补齐")
    gr.Markdown(
        "This tab exposes capabilities that exist in the codebase but were not "
        "available in the previous Gradio tabs. Each section provides a standalone "
        "interactive demo."
    )

    # ── Section 1: Multimodal RAG ─────────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 1. Multimodal RAG (PDF → Image + Text → VLM → KG → Search)")

    with gr.Row():
        with gr.Column(scale=1):
            mm_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
            mm_max_pages = gr.Number(value=5, label="Max Pages to Analyze", minimum=1, maximum=50)
            mm_extract_btn = gr.Button("1. Extract PDF", variant="secondary")
            mm_describe_btn = gr.Button("2. Describe Images (VLM)", variant="secondary")
            mm_graph_name = gr.Textbox(value="multimodal_poc", label="Target Graph Name")
            mm_build_btn = gr.Button("3. Build Multimodal KG", variant="secondary")
        with gr.Column(scale=2):
            mm_extract_out = gr.Code(label="Extraction Summary", language="json")

    with gr.Row():
        with gr.Column(scale=1):
            mm_query = gr.Textbox(label="Search Query", placeholder="Ask about charts, figures, or text...")
            mm_mode = gr.Dropdown(choices=["auto", "text_only", "image_aware"], value="auto", label="Search Mode")
            mm_top_k = gr.Slider(value=5, minimum=1, maximum=20, step=1, label="Top-K")
            mm_search_btn = gr.Button("4. Multimodal Search", variant="primary")
        with gr.Column(scale=2):
            mm_describe_out = gr.Code(label="VLM Descriptions", language="json")
            mm_search_out = gr.Code(label="Search Results", language="json")

    mm_extract_btn.click(
        fn=multimodal_extract_pdf,
        inputs=[mm_pdf, mm_max_pages],
        outputs=[mm_extract_out],
    )
    mm_describe_btn.click(
        fn=multimodal_describe_images,
        inputs=[gr.Number(value=3, visible=False), gr.Text(value="xiaomimo", visible=False)],
        outputs=[mm_describe_out],
    )
    mm_build_btn.click(
        fn=multimodal_build_kg,
        inputs=[mm_graph_name],
        outputs=[mm_describe_out],
    )
    mm_search_btn.click(
        fn=multimodal_search,
        inputs=[mm_query, mm_graph_name, mm_top_k, mm_mode],
        outputs=[mm_search_out],
    )

    # ── Section 2: Property Graph Extraction ──────────────────
    gr.Markdown("---")
    gr.Markdown("## 2. Property Graph Extraction")

    with gr.Row():
        with gr.Column(scale=2):
            pg_text = gr.Textbox(
                label="Input Text",
                placeholder="Paste text to extract a property graph...",
                lines=6,
            )
            pg_schema = gr.Code(
                label="Schema JSON (optional, auto-fetch if empty)",
                language="json",
                value='{"vertexlabels": [], "edgelabels": []}',
            )
            pg_btn = gr.Button("Extract Property Graph", variant="secondary")
        with gr.Column(scale=2):
            pg_out = gr.Code(label="Extracted Vertices/Edges", language="json")

    pg_btn.click(
        fn=property_graph_extract,
        inputs=[pg_text, pg_schema],
        outputs=[pg_out],
    )

    # ── Section 3: Incremental Index Flow ─────────────────────
    gr.Markdown("---")
    gr.Markdown("## 3. Incremental Index Flow")

    with gr.Row():
        with gr.Column(scale=2):
            inc_text = gr.Textbox(
                label="New Document Texts",
                placeholder="Paste new document content here. Use '---' on its own line to separate multiple documents.",
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

    # ── Section 4: Gremlin Self-Correction ────────────────────
    gr.Markdown("---")
    gr.Markdown("## 4. Gremlin Self-Correction (Text2Gremlin + Validator + Retry)")

    with gr.Row():
        with gr.Column(scale=2):
            gc_query = gr.Textbox(
                label="Natural Language Query",
                placeholder="e.g., Find all suppliers of part A",
                lines=2,
            )
            gc_retries = gr.Slider(value=3, minimum=1, maximum=5, step=1, label="Max Retries")
            gc_lang = gr.Dropdown(choices=["cn", "en"], value="cn", label="Language")
            gc_btn = gr.Button("Generate & Validate Gremlin", variant="secondary")
        with gr.Column(scale=2):
            gc_out = gr.Code(label="Gremlin Retry Result", language="json")

    gc_btn.click(
        fn=gremlin_self_correct,
        inputs=[gc_query, gc_retries, gc_lang],
        outputs=[gc_out],
    )

    # ── Section 5: Query Classifier ───────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 5. Query Classifier (Agent Routing)")

    with gr.Row():
        with gr.Column(scale=2):
            qc_query = gr.Textbox(
                label="User Query",
                placeholder="Enter a query to classify as simple or complex...",
                lines=2,
            )
            qc_use_llm = gr.Checkbox(value=False, label="Use LLM for nuanced classification")
            qc_btn = gr.Button("Classify Query", variant="secondary")
        with gr.Column(scale=2):
            qc_out = gr.Code(label="Classification Result", language="json")

    qc_btn.click(
        fn=query_classifier_demo,
        inputs=[qc_query, qc_use_llm],
        outputs=[qc_out],
    )

    # ── Section 6: Synonym Manager ────────────────────────────
    gr.Markdown("---")
    gr.Markdown("## 6. Synonym Manager")

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

    # ── Section 7: Chunk Similarity Edges ─────────────────────
    gr.Markdown("---")
    gr.Markdown("## 7. Chunk Similarity Edges")

    with gr.Row():
        with gr.Column(scale=1):
            cs_label = gr.Textbox(value="Chunk", label="Chunk Vertex Label")
            cs_top_k = gr.Slider(value=3, minimum=1, maximum=10, step=1, label="KNN Top-K")
            cs_min_score = gr.Slider(value=0.5, minimum=0.0, maximum=1.0, step=0.05, label="Min Similarity")
            cs_btn = gr.Button("Build SIMILAR Edges", variant="secondary")
        with gr.Column(scale=2):
            cs_out = gr.Code(label="Build Result", language="json")

    cs_btn.click(
        fn=chunk_sim_edges_build,
        inputs=[cs_label, cs_top_k, cs_min_score],
        outputs=[cs_out],
    )
