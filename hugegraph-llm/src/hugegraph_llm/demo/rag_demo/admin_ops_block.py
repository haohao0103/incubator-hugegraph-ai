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

"""Gradio UI block for Admin & Operations.

Merges the former \"Admin Tools\" (Tab 7) and \"Graph Tools\" (Tab 6) tabs
into a single operations tab with two sub-sections:

  Section A — Graph Tools (Gremlin query, backup, test data init)
  Section B — Admin Log Viewer (password-protected LLM server log)
"""

import asyncio
from collections import deque
from contextlib import asynccontextmanager

import gradio as gr
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from hugegraph_llm.api.admin_api import _is_configured_admin_token
from hugegraph_llm.config import admin_settings
from hugegraph_llm.demo.rag_demo.vector_graph_block import timely_update_vid_embedding
from hugegraph_llm.utils.hugegraph_utils import backup_data, init_hg_test_data, run_gremlin_query
from hugegraph_llm.utils.log import log


async def log_stream(log_path: str, lines: int = 125):
    """
    Stream the content of a log file like `tail -f`.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            buffer = deque(file, maxlen=lines)
            for line in buffer:
                yield line
            while True:
                line = file.readline()
                if line:
                    yield line
                else:
                    await asyncio.sleep(0.1)
    except FileNotFoundError as exc:
        raise Exception(f"Log file not found: {log_path}") from exc
    except Exception as e:
        raise Exception(f"An error occurred while reading the log: {str(e)}") from e


def create_admin_ops_block():
    """Create the unified Admin & Ops Gradio UI tab."""

    gr.Markdown("# Admin & Operations / 管理与运维")
    gr.Markdown(
        "**Section A:** Graph database tools (query, backup, init).  "
        "**Section B:** Password-protected log viewer."
    )

    # ══════════════════════════════════════════════════════════
    # A. Graph Tools
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("A. Graph Tools / 图数据库工具", open=True):
        gr.Markdown("### Gremlin Query Console")

        with gr.Row():
            gremlin_inp = gr.Textbox(
                value="g.V().limit(10)",
                label="Gremlin query",
                show_copy_button=True,
                lines=8,
            )
            gremlin_out = gr.Code(
                label="Output", language="json", elem_classes="code-container-show"
            )
        gremlin_btn = gr.Button("Run Gremlin Query", variant="primary")
        gremlin_btn.click(fn=run_gremlin_query, inputs=[gremlin_inp], outputs=gremlin_out)

        gr.Markdown("---")
        gr.Markdown("### Data Backup")

        with gr.Row():
            backup_out = gr.Textbox(
                label="Backup Result",
                show_copy_button=True,
                info="Auto backup at 1:00 AM everyday via scheduler.",
            )
        backup_btn = gr.Button("Backup Graph Data Now", variant="secondary")
        backup_btn.click(fn=backup_data, inputs=[], outputs=backup_out)

        with gr.Accordion("Init HugeGraph Test Data (BETA 🚧)", open=False):
            with gr.Row():
                init_out = gr.Textbox(label="Init Graph Demo Result", show_copy_button=True)
            init_btn = gr.Button("(BETA) Init HugeGraph Test Data", variant="stop")
            init_btn.click(fn=init_hg_test_data, inputs=[], outputs=init_out)

    # ══════════════════════════════════════════════════════════
    # B. Admin Log Viewer
    # ══════════════════════════════════════════════════════════
    with gr.Accordion("B. Admin Log Viewer / 日志查看（需密码）", open=False):
        password_input = gr.Textbox(
            label="Enter Password / 输入密码",
            type="password",
            placeholder="Enter admin password to access logs...",
        )

        error_message = gr.Textbox(label="", visible=False, interactive=False,
                                    elem_classes="error-message")
        submit_button = gr.Button("Submit / 提交")

        with gr.Row(visible=False) as hidden_row:
            with gr.Column():
                gr.Markdown("### LLM Server Log")
                llm_server_log_output = gr.Code(
                    label="LLM Server Log (llm-server.log)",
                    lines=20,
                    value="",
                    elem_classes="code-container-edit",
                    every=60,
                )
                with gr.Row():
                    clear_llm_server_button = gr.Button(
                        "Clear LLM Server Log", visible=False
                    )
                    refresh_llm_server_button = gr.Button(
                        "Refresh LLM Server Log", visible=False, variant="primary"
                    )

        submit_button.click(
            fn=check_password,
            inputs=[password_input],
            outputs=[
                llm_server_log_output,
                hidden_row,
                clear_llm_server_button,
                refresh_llm_server_button,
                error_message,
            ],
        )
        clear_llm_server_button.click(fn=clear_llm_server_log, inputs=[],
                                      outputs=[llm_server_log_output])
        refresh_llm_server_button.click(fn=read_llm_server_log, inputs=[],
                                        outputs=[llm_server_log_output])


# ── Admin log functions (unchanged from original admin_block.py) ──

def read_llm_server_log(lines=250):
    log_path = "logs/llm-server.log"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except FileNotFoundError:
        log.critical("Log file not found: %s", log_path)
        return "LLM Server log file not found."


def clear_llm_server_log():
    log_path = "logs/llm-server.log"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.truncate(0)
        return "LLM Server log cleared."
    except Exception as e:  # pylint: disable=W0718
        log.error("An error occurred while clearing the log: %s", str(e))
        return "Failed to clear LLM Server log."


def check_password(password, request: gr.Request | None = None):
    client_ip = request.client.host if request else "Unknown IP"
    admin_token = admin_settings.admin_token

    if not _is_configured_admin_token(admin_token):
        log.error("Rejected admin log access with insecure token from IP: %s", client_ip)
        return (
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="Admin token is not configured securely.", visible=True),
        )

    if password == admin_token:
        llm_log = read_llm_server_log()
        log.info("Logs accessed successfully from IP: %s", client_ip)
        return (
            llm_log,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    log.error("Incorrect password attempt from IP: %s", client_ip)
    return (
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="Incorrect password. Access denied.", visible=True),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=W0621
    log.info("Starting background scheduler...")
    scheduler = AsyncIOScheduler()
    scheduler.add_job(backup_data, trigger=CronTrigger(hour=1, minute=0),
                      id="daily_backup", replace_existing=True)
    scheduler.start()

    log.info("Starting vid embedding update task...")
    embedding_task = asyncio.create_task(timely_update_vid_embedding())
    yield

    log.info("Stopping vid embedding update task...")
    embedding_task.cancel()
    try:
        await embedding_task
    except asyncio.CancelledError:
        log.info("Vid embedding update task cancelled.")

    log.info("Shutting down background scheduler...")
    scheduler.shutdown()
