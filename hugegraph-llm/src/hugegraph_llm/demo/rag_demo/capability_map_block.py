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

"""Gradio UI block for the Capability Map.

Shows the full capability matrix and highlights gaps in the Gradio UI.
Also provides quick demos for a few missing utilities that exist in the
current branch but are not exposed elsewhere.
"""

import gradio as gr

from hugegraph_llm.demo.rag_demo.capability_map_handlers import (
    get_capability_matrix,
    ui_fetch_graph_summary,
    ui_get_graph_schema,
    ui_incremental_tool,
    ui_validate_gremlin,
)
from hugegraph_llm.utils.log import log


_MISSING_COUNT = 13  # approximate number of Missing rows for header text


def create_capability_map_block():
    """Create the Capability Map Gradio UI tab."""
    gr.Markdown("# Capability Map / 能力地图")
    gr.Markdown(
        "This tab shows the full capability matrix of HugeGraph-AI and highlights "
        "which capabilities are already exposed in the Gradio UI and which are still missing. "
        "Use this to understand coverage gaps at a glance."
    )

    # ── Coverage summary ───────────────────────────────────────
    gr.Markdown("## 1. Coverage Overview")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                "### ✅ Exposed in UI\n"
                "- Build RAG Index (Tab 1)\n"
                "- RAG & User Functions (Tab 2)\n"
                "- Text2Gremlin (Tab 3)\n"
                "- Agent & Global Search (Tab 4)\n"
                "- Graph Tools (Tab 5)\n"
                "- Admin Tools (Tab 6)\n"
                "- Advanced GraphRAG (Tab 7)\n"
                "- GraphRAG Enhancement (Tab 8)"
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "### ⚠️ Missing / Partial\n"
                "- Multimodal RAG\n"
                "- Property Graph Extraction\n"
                "- Incremental Index Update\n"
                "- Gremlin Self-Correction Validator\n"
                "- Agent Memory (other branch)\n"
                "- Code Graph + MCP (other branch)\n"
                "- Skills Graph / Code-Review (other branch)\n"
                "- Supply Chain Agent (other branch)"
            )

    # ── Capability matrix DataFrame ───────────────────────────
    gr.Markdown("## 2. Full Capability Matrix")
    gr.Markdown(
        "The table below lists every tracked capability, its current UI exposure, "
        "and priority for closure. Sort by 'Status' to see missing items first."
    )

    matrix_df = gr.DataFrame(
        value=get_capability_matrix(),
        label="Capability Matrix",
        wrap=True,
    )

    refresh_btn = gr.Button("Refresh Matrix", size="sm")
    refresh_btn.click(fn=lambda: get_capability_matrix(), outputs=[matrix_df])

    # ── Quick demos for missing utilities ─────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 3. Quick Demos for Missing Utilities\n"
        "These utilities exist in the current branch but are not exposed in other tabs."
    )

    with gr.Accordion("Fetch Graph Summary", open=False):
        gr.Markdown(
            "Get a lightweight summary of the current HugeGraph instance: "
            "vertex count, edge count, and sample IDs."
        )
        with gr.Row():
            with gr.Column(scale=1):
                v_limit = gr.Number(value=100, label="Max Vertex IDs", minimum=1, maximum=10000)
                e_limit = gr.Number(value=50, label="Max Edge IDs", minimum=1, maximum=1000)
                fetch_btn = gr.Button("Fetch Summary", variant="secondary")
            with gr.Column(scale=2):
                fetch_out = gr.Code(label="Graph Summary", language="json")
        fetch_btn.click(
            fn=ui_fetch_graph_summary,
            inputs=[v_limit, e_limit],
            outputs=[fetch_out],
        )

    with gr.Accordion("Get Graph Schema", open=False):
        gr.Markdown("Retrieve the schema (vertex labels, edge labels, properties) from HugeGraph.")
        get_schema_btn = gr.Button("Get Schema", variant="secondary")
        schema_out = gr.Code(label="Graph Schema", language="json")
        get_schema_btn.click(fn=ui_get_graph_schema, outputs=[schema_out])

    with gr.Accordion("Validate Gremlin Query", open=False):
        gr.Markdown(
            "Use LLM-driven validation to check a Gremlin query for syntax and schema issues. "
            "Schema is fetched automatically from HugeGraph if left empty."
        )
        with gr.Row():
            with gr.Column(scale=3):
                gremlin_input = gr.Textbox(
                    label="Gremlin Query",
                    value="g.V().has('name', 'Alice').out().limit(10)",
                    lines=3,
                )
            with gr.Column(scale=1):
                val_lang = gr.Dropdown(choices=["en", "cn"], value="cn", label="Language")
                val_btn = gr.Button("Validate", variant="secondary")
        val_out = gr.Code(label="Validation Result", language="json")
        val_btn.click(
            fn=ui_validate_gremlin,
            inputs=[gremlin_input, val_lang],
            outputs=[val_out],
        )

    with gr.Accordion("Incremental Index Tools", open=False):
        gr.Markdown(
            "Utilities for incremental community indexing: find affected communities "
            "from new vertices and persist community assignments."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inc_action = gr.Dropdown(
                    choices=["find_affected", "persist_communities"],
                    value="find_affected",
                    label="Action",
                )
                inc_hop = gr.Number(value=1, label="Hop (find_affected)", minimum=1, maximum=5)
                inc_btn = gr.Button("Run", variant="secondary")
            with gr.Column(scale=2):
                inc_vertex_ids = gr.Textbox(
                    label="Vertex IDs (comma-separated, for find_affected)",
                    placeholder="42:alice, 42:bob",
                    lines=1,
                )
                inc_communities = gr.Code(
                    label="Communities JSON (for persist_communities)",
                    language="json",
                    value='[\n  {"id": "1", "vertices": ["42:alice", "42:bob"]}\n]',
                )
                inc_out = gr.Code(label="Result", language="json")
        inc_btn.click(
            fn=ui_incremental_tool,
            inputs=[inc_action, inc_vertex_ids, inc_communities, inc_hop],
            outputs=[inc_out],
        )

    # ── Roadmap note ─────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "## 4. Roadmap Notes\n"
        "The biggest remaining gaps are in separate branches that have not been merged "
        "into `feature/graphrag-baseline` yet:\n"
        "- `feature/agent-memory-collection` → Agent Memory (MAGMA / MemGraphRAG)\n"
        "- `poc/0614-codegraph-hugegraph-mcp` → Code Graph + MCP\n"
        "- `poc/0618-skills-graph-code-review-wiki` → Skills Graph / Code-Review-Graph\n"
        "- `poc/0615-supply-chain-agent-router` → Supply Chain Agent Router\n\n"
        "To fully showcase these in the Gradio UI, the branches need to be merged or "
        "cherry-picked into the current baseline, and dedicated tabs should be added."
    )
