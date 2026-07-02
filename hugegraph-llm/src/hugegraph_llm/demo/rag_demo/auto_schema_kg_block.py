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

"""Gradio UI block for AutoSchemaKG.

Single-document → LLM schema draft → human review → commit to HugeGraph.
"""

import gradio as gr

from hugegraph_llm.demo.rag_demo.auto_schema_kg_handlers import (
    EXAMPLE_DOCUMENTS,
    approve_and_commit,
    generate_schema_draft,
    get_suggested_questions,
    load_example_document,
    reset_schema_draft,
)


def create_auto_schema_kg_block():
    """Create the AutoSchemaKG Gradio UI tab."""
    gr.Markdown("# AutoSchemaKG 🧬")
    gr.Markdown(
        "Paste a document below, or pick a preset example. The LLM will infer a HugeGraph schema draft "
        "(property keys, vertex labels, edge labels). Review the draft, edit the JSON if needed, "
        "then click **Approve & Commit** to write the schema to HugeGraph."
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_DOCUMENTS.keys()),
                    value=None,
                    label="Load Preset Example",
                    info="Select an example to populate the document box.",
                )
            doc_input = gr.Textbox(
                label="Document",
                placeholder="Paste your document here...",
                lines=12,
                show_copy_button=True,
            )
            example_note = gr.Markdown(label="Example Note")
        with gr.Column(scale=1):
            instructions_input = gr.Textbox(
                label="Instructions (optional)",
                placeholder="e.g., focus on supply-chain entities, include timestamp properties",
                lines=4,
            )
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
            schema_json_output = gr.Code(label="Schema JSON", language="json", lines=18)

    generate_btn.click(
        fn=generate_schema_draft,
        inputs=[doc_input, instructions_input],
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

    example_dropdown.change(
        fn=load_example_document,
        inputs=[example_dropdown],
        outputs=[doc_input, example_note],
    )

    with gr.Accordion("Suggested Review Questions", open=False):
        gr.Markdown(get_suggested_questions())
