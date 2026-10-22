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

"""Chat LLM adapter for the skill graph's edge induction.

The project's :class:`~hugegraph_llm.models.llms.openai.OpenAIClient` posts a
plain chat-completions body and waits for the full response. The deployment
gateway used here fronts *reasoning* models (``qwen3.6-plus``) that, by default,
stream a long chain-of-thought and therefore never return a non-streaming body
in time.

:class:`GatewayChatLLM` talks to the same OpenAI-compatible endpoint but:

  * posts to ``{OPENAI_CHAT_API_BASE}/chat/completions`` (base URL may or may not
    include ``/v1``);
  * sends the extra ``hller`` header several gateways require (reusing the
    project's ``OPENAI_DEFAULT_HEADERS``);
  * asks the model to skip its thinking trace (``enable_thinking=false``), which
    makes the call return promptly with a short, parseable answer.

It exposes only what :class:`~...edge_inducer.LLMEdgeInducer` needs:
``generate(prompt=...) -> str``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from hugegraph_llm.engines.skill_graph.embedder import _parse_headers

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class GatewayChatLLM:
    """Minimal OpenAI-compatible chat client tuned for gateway-hosted reasoning models."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        disable_thinking: Optional[bool] = None,
        timeout: float = 120.0,
    ):
        base = base_url or os.environ.get("OPENAI_CHAT_API_BASE", "")
        if not base:
            raise ValueError("GatewayChatLLM requires a base_url (or OPENAI_CHAT_API_BASE)")
        self.url = base.rstrip("/") + "/chat/completions"
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_CHAT_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_CHAT_LANGUAGE_MODEL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = (
            _env_flag("OPENAI_CHAT_DISABLE_THINKING", True)
            if disable_thinking is None
            else disable_thinking
        )
        self.timeout = timeout
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers.update(_parse_headers(os.environ.get("OPENAI_DEFAULT_HEADERS")))
        if headers:
            self.headers.update(headers)

    def generate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Return the assistant text for ``messages`` (or a single ``prompt``)."""
        import httpx

        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            payload["enable_thinking"] = False

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices in LLM response: {str(data)[:200]}")
        message = choices[0].get("message", {}) or {}
        content = message.get("content") or message.get("reasoning_content") or ""
        return content.strip()


def resolve_chat_llm():
    """Return the best available chat LLM for edge induction, else ``None``.

    Resolution order:

    1. :class:`GatewayChatLLM` when ``OPENAI_CHAT_API_BASE`` is configured — this
       is the path that actually returns for gateway-hosted reasoning models.
    2. The project chat LLM (:func:`LLMs.get_chat_llm`) if importable.
    """
    if os.environ.get("OPENAI_CHAT_API_BASE"):
        try:
            return GatewayChatLLM()
        except Exception:  # pragma: no cover - bad config falls through
            log.warning("GatewayChatLLM init failed; trying project chat LLM")
    try:
        from hugegraph_llm.models.llms.init_llm import LLMs

        return LLMs().get_chat_llm()
    except Exception:  # pragma: no cover - depends on optional runtime config
        return None