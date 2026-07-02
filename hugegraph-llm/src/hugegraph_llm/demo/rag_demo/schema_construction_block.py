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

"""Gradio UI block for Schema Studio — unified schema construction tab.

Merges AutoSchemaKG (Section A) and EDC Pipeline (Section B) into one tab.
Replaces the old separate Tab 11 (EDC) and Tab 12 (AutoSchemaKG).

Section A — AutoSchemaKG:
  Single or batch document -> LLM schema draft -> review & commit to HugeGraph.
  Supports both single-doc inference and multi-document batch merge with conflict detection.

Section B — EDC Pipeline:
  Extract -> Define -> Canonicalize schema evolution for production scale.
  Content is identical to the old edc_schema_block.py — no refactoring needed.

Cross-section bridge:
  A button in Section A exports the generated schema JSON into the format
  expected by Section B's Guided mode (allowed_vertex_labels / allowed_edge_labels),
  so users can go from "LLM-guessed initial schema" straight into
  "constrained extraction with that schema".
"""

import json

import gradio as gr

from hugegraph_llm.demo.rag_demo.auto_schema_kg_handlers import (
    EXAMPLE_BATCH_DOCUMENTS,
    EXAMPLE_DOCUMENTS,
    approve_and_commit,
    generate_batch_schema_draft,
    generate_schema_draft,
    get_suggested_questions,
    load_example_batch_document,
    load_example_document,
    reset_schema_draft,
)
from hugegraph_llm.demo.rag_demo.edc_schema_block import (
    DEMO_TEXTS,
    create_edc_schema_block,
)


def create_schema_construction_block():
    """Create the Schema Studio Gradio UI tab (Sections A + B)."""

    gr.Markdown("# Schema Studio")
    gr.Markdown(
        "**Build and evolve HugeGraph schemas in one place.**\n\n"
        "| Section | What it does | When to use |"
        "\n|---------|-------------|-------------|"
        "\n"
        "| **A. AutoSchemaKG** | LLM infers schema from documents; review & commit | "
        "Starting fresh, no existing schema |"
        "\n"
        "| **B. EDC Pipeline** | Schema evolution: Extract → Define → Canonicalize | "
        "Production scale, handling type explosion |"
    )

    # ══════════════════════════════════════════════════════════
    # Section A — AutoSchemaKG
    # ══════════════════════════════════════════════════════════

    with gr.Accordion("A. AutoSchemaKG - LLM Schema Inference", open=True):
        gr.Markdown(
            "Paste a document or pick a preset example. The LLM will infer a HugeGraph schema draft "
            "(property keys, vertex labels, edge labels). Review the draft, edit the JSON if needed, "
            "then click **Approve & Commit**."
        )

        with gr.Row():
            mode_toggle = gr.Radio(
                choices=[("Single Document", "single"), ("Batch Multi-document", "batch")],
                value="batch",
                label="Mode",
                info="Single = one doc, one draft. Batch = split by blank lines, merge + detect conflicts.",
            )
            instructions_input = gr.Textbox(
                label="Instructions (optional)",
                placeholder="e.g., focus on supply-chain entities, include timestamp properties",
                lines=3,
            )

        with gr.Row():
            with gr.Column(scale=2):
                single_example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_DOCUMENTS.keys()),
                    value=None,
                    label="Load Single-doc Example",
                    visible=len(EXAMPLE_DOCUMENTS) > 0,
                )
                batch_example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_BATCH_DOCUMENTS.keys()),
                    value=None,
                    label="Load Multi-doc Example",
                    visible=len(EXAMPLE_BATCH_DOCUMENTS) > 0,
                )
                doc_input = gr.Textbox(
                    label="Document(s)",
                    placeholder=(
                        "Paste your document here...\n\n"
                        "(In batch mode, separate documents with blank lines.)"
                    ),
                    lines=10,
                    show_copy_button=True,
                )
                example_note = gr.Markdown(label="Example Note")

            with gr.Column(scale=1):
                generate_btn = gr.Button("Generate Schema Draft", variant="primary")
                approve_btn = gr.Button("Approve & Commit to HugeGraph", variant="secondary")
                reset_btn = gr.Button("Reset", variant="stop", size="sm")
                status_output = gr.Textbox(label="Status", interactive=False, lines=2)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Schema Preview (Markdown)")
                preview_output = gr.Markdown(label="Schema Preview")
            with gr.Column(scale=1):
                gr.Markdown("### Schema JSON (editable before commit)")
                schema_json_output = gr.Code(label="Schema JSON", language="json", lines=16)

        # ── Bridge: export schema to EDC Guided mode ──
        with gr.Row():
            export_to_guided_btn = gr.Button(
                "Export Schema to EDC Guided Mode",
                variant="secondary",
                size="sm",
            )
            guided_config_out = gr.Code(
                label="Guided Mode Config (paste into Section B)",
                language="json",
                lines=4,
                interactive=False,
            )
        gr.Markdown(
            "<small>"
            "This extracts vertex/edge labels from the generated schema and formats them "
            "as EDC **Guided** mode constraints (allowed_vertex_labels / allowed_edge_labels). "
            "Copy this config and paste it into **Section B > Guided** to run constrained extraction."
            "</small>"
        )

        # ── Event bindings: Section A ──

        def _on_generate(doc, instr, mode):
            if mode == "batch":
                return generate_batch_schema_draft(doc, instr)
            return generate_schema_draft(doc, instr)

        generate_btn.click(
            fn=_on_generate,
            inputs=[doc_input, instructions_input, mode_toggle],
            outputs=[preview_output, schema_json_output, status_output],
        )

        approve_btn.click(
            fn=approve_and_commit,
            inputs=[schema_json_output],
            outputs=[preview_output, status_output],
        )

        reset_btn.click(
            fn=reset_schema_draft,
            inputs=[],
            outputs=[preview_output, schema_json_output, status_output],
        )

        if len(EXAMPLE_DOCUMENTS) > 0:
            single_example_dropdown.change(
                fn=load_example_document,
                inputs=[single_example_dropdown],
                outputs=[doc_input, example_note],
            )
        if len(EXAMPLE_BATCH_DOCUMENTS) > 0:
            batch_example_dropdown.change(
                fn=load_example_batch_document,
                inputs=[batch_example_dropdown],
                outputs=[doc_input, example_note],
            )

        def _export_to_guided(schema_json_str):
            """Convert a generated schema JSON into EDC Guided-mode config."""
            try:
                schema = json.loads(schema_json_str) if isinstance(schema_json_str, str) else {}
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"error": "No valid schema JSON yet. Generate a schema first."}, indent=2)
            vls = [v.get("name", "") for v in schema.get("vertexlabels", []) if v.get("name")]
            els = [e.get("name", "") for e in schema.get("edgelabels", []) if e.get("name")]
            if not vls and not els:
                return json.dumps({"error": "Schema has no vertex or edge labels."}, indent=2)
            guided = {
                "mode": "GUIDED",
                "allowed_vertex_labels": vls,
                "allowed_edge_labels": els,
                "guided_allow_dynamic": False,
            }
            return json.dumps(guided, ensure_ascii=False, indent=2)

        export_to_guided_btn.click(
            fn=_export_to_guided,
            inputs=[schema_json_output],
            outputs=[guided_config_out],
        )

        with gr.Accordion("Suggested Review Questions", open=False):
            gr.Markdown(get_suggested_questions())

    # ══════════════════════════════════════════════════════════
    # Section B — EDC Pipeline (old Tab 11 content, unchanged)
    # ══════════════════════════════════════════════════════════

    gr.Markdown("---")

    with gr.Accordion("B. EDC Pipeline - Schema Evolution", open=True):
        gr.Markdown(
            "**EDC = Extract → Define → Canonicalize** — handles type explosion at production scale.\n\n"
            "**Workflow**: Start with **AutoSchemaKG** (Section A) to create an initial schema, "
            "then use **EDC** here to evolve it as more documents are ingested. "
            "Or jump straight into EDC if you already have a schema foundation."
        )
        edc_demo_outputs, edc_load_demo = create_edc_schema_block()

    # ══════════════════════════════════════════════════════════
    # Demo data loader (combines both sections)
    # ══════════════════════════════════════════════════════════

    def _load_all_demo_data():
        """Pre-populate both sections with demo data on page load."""
        # Section A: load first batch example into doc box
        batch_doc = ""
        batch_note = ""
        if EXAMPLE_BATCH_DOCUMENTS:
            first_key = list(EXAMPLE_BATCH_DOCUMENTS.keys())[0]
            batch_doc, batch_note = load_example_batch_document(first_key)

        # Section B: EDC demo data
        edc_data = edc_load_demo() if callable(edc_load_demo) else ()

        return (batch_doc, batch_note) + (tuple(edc_data) if edc_data else ())

    all_outputs = [doc_input, example_note]
    if edc_demo_outputs:
        all_outputs.extend(edc_demo_outputs)

    return all_outputs, _load_all_demo_data
