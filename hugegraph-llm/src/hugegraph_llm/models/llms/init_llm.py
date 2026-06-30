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

import logging
from typing import Optional, Union

from hugegraph_llm.config import LLMConfig, llm_settings
from hugegraph_llm.models.llms.custom_endpoint import CustomEndpointLLM
from hugegraph_llm.models.llms.litellm import LiteLLMClient
from hugegraph_llm.models.llms.ollama import OllamaClient
from hugegraph_llm.models.llms.openai import OpenAIClient

log = logging.getLogger(__name__)


def _parse_headers(headers_str: str) -> dict:
    """Parse 'key1=val1,key2=val2' into a dict."""
    if not headers_str:
        return {}
    result = {}
    for pair in headers_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    return result

# Supported LLM type identifiers
_SUPPORTED_TYPES = {"openai", "ollama/local", "litellm"}

# All recognized role names (maps to attribute prefixes on LLMConfig / llm_settings)
_ROLE_NAMES = frozenset(
    ["chat", "extract", "text2gql", "agent", "general"]
)


def _create_llm(
    llm_type: str,
    *,
    api_key: str = "",
    api_base: str = "",
    model_name: str = "",
    max_tokens: int = 4096,
    host: str = "localhost",
    port: int = 11434,
    default_headers: Optional[dict] = None,
    direct_url: str = "",
) -> Union[OpenAIClient, OllamaClient, LiteLLMClient, CustomEndpointLLM]:
    """Unified LLM factory that instantiates the correct client from parameters.

    When *direct_url* is set and *llm_type* is ``openai``, returns a
    :class:`CustomEndpointLLM` that posts directly to that URL.
    """
    if llm_type == "openai":
        if direct_url:
            return CustomEndpointLLM(
                api_key=api_key,
                api_base=direct_url,
                model_name=model_name,
                max_tokens=max_tokens,
                default_headers=default_headers,
            )
        return OpenAIClient(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            max_tokens=max_tokens,
            default_headers=default_headers,
        )
    if llm_type == "ollama/local":
        return OllamaClient(
            model=model_name,
            host=host,
            port=port,
        )
    if llm_type == "litellm":
        return LiteLLMClient(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            max_tokens=max_tokens,
        )
    raise ValueError(
        f"Unsupported LLM type '{llm_type}'. Supported: {_SUPPORTED_TYPES}"
    )


def _role_config(llm_configs: LLMConfig, role: str):
    """Extract the parameter dict for a given role from *llm_configs*."""
    r = role
    return {
        "llm_type": getattr(llm_configs, f"{r}_llm_type", llm_configs.chat_llm_type),
        "api_key": getattr(llm_configs, f"openai_{r}_api_key", ""),
        "api_base": getattr(llm_configs, f"openai_{r}_api_base", ""),
        "model_name": getattr(llm_configs, f"openai_{r}_language_model", ""),
        "max_tokens": getattr(llm_configs, f"openai_{r}_tokens", 4096),
        "host": getattr(llm_configs, f"ollama_{r}_host", "localhost"),
        "port": getattr(llm_configs, f"ollama_{r}_port", 11434),
        "litellm_api_key": getattr(llm_configs, f"litellm_{r}_api_key", ""),
        "litellm_api_base": getattr(llm_configs, f"litellm_{r}_api_base", ""),
        "litellm_model_name": getattr(llm_configs, f"litellm_{r}_language_model", ""),
        "litellm_max_tokens": getattr(llm_configs, f"litellm_{r}_tokens", 4096),
        "default_headers": _parse_headers(getattr(llm_configs, "openai_default_headers", "") or ""),
        "direct_url": getattr(llm_configs, f"openai_{r}_direct_url", "") or "",
    }


def _build_from_role_params(params: dict) -> Union[OpenAIClient, OllamaClient, LiteLLMClient]:
    """Build an LLM instance from role-extracted params, dispatching by type."""
    t = params["llm_type"]
    if t == "litellm":
        return _create_llm(
            t,
            api_key=params["litellm_api_key"],
            api_base=params["litellm_api_base"],
            model_name=params["litellm_model_name"],
            max_tokens=params["litellm_max_tokens"],
        )
    return _create_llm(
        t,
        api_key=params["api_key"],
        api_base=params["api_base"],
        model_name=params["model_name"],
        max_tokens=params["max_tokens"],
        host=params["host"],
        port=params["port"],
        default_headers=params.get("default_headers"),
        direct_url=params.get("direct_url", ""),
    )


# ---------------------------------------------------------------------------
# Module-level convenience factories (accept explicit LLMConfig)
# ---------------------------------------------------------------------------

def get_chat_llm(llm_configs: LLMConfig):
    return _build_from_role_params(_role_config(llm_configs, "chat"))


def get_extract_llm(llm_configs: LLMConfig):
    return _build_from_role_params(_role_config(llm_configs, "extract"))


def get_text2gql_llm(llm_configs: LLMConfig):
    return _build_from_role_params(_role_config(llm_configs, "text2gql"))


def get_agent_llm(llm_configs: LLMConfig):
    return _build_from_role_params(_role_config(llm_configs, "agent"))


def get_general_llm(llm_configs: LLMConfig):
    """Get a general-purpose LLM instance.

    Falls back to the ``chat`` role configuration if no dedicated
    ``general`` role is defined in *llm_configs*.
    """
    try:
        params = _role_config(llm_configs, "general")
        if params["llm_type"] == llm_configs.chat_llm_type:
            raise AttributeError
    except AttributeError:
        params = _role_config(llm_configs, "chat")
    return _build_from_role_params(params)


# ---------------------------------------------------------------------------
# Global-settings factory class
# ---------------------------------------------------------------------------

class LLMs:
    """Lazy LLM factory that reads from the global ``llm_settings`` singleton.

    Typical usage::

        llm = LLMs().get_chat_llm()
    """

    _INSTANCES: dict = {}

    def __init__(self):
        # Pre-discover supported roles from llm_settings for fast lookup
        self._roles = {}
        for role in _ROLE_NAMES:
            type_attr = f"{role}_llm_type"
            if hasattr(llm_settings, type_attr):
                self._roles[role] = getattr(llm_settings, type_attr)

    def _get_type(self, role: str) -> str:
        return self._roles.get(role, llm_settings.chat_llm_type)

    def _build(self, role: str):
        """Build an LLM for *role* using global settings."""
        t = self._get_type(role)
        if t == "litellm":
            return _create_llm(
                t,
                api_key=getattr(llm_settings, f"litellm_{role}_api_key", ""),
                api_base=getattr(llm_settings, f"litellm_{role}_api_base", ""),
                model_name=getattr(llm_settings, f"litellm_{role}_language_model", ""),
                max_tokens=getattr(llm_settings, f"litellm_{role}_tokens", 4096),
            )
        return _create_llm(
            t,
            api_key=getattr(llm_settings, f"openai_{role}_api_key", ""),
            api_base=getattr(llm_settings, f"openai_{role}_api_base", ""),
            model_name=getattr(llm_settings, f"openai_{role}_language_model", ""),
            max_tokens=getattr(llm_settings, f"openai_{role}_tokens", 4096),
            host=getattr(llm_settings, f"ollama_{role}_host", "localhost"),
            port=getattr(llm_settings, f"ollama_{role}_port", 11434),
            default_headers=_parse_headers(getattr(llm_settings, "openai_default_headers", "") or ""),
            direct_url=getattr(llm_settings, f"openai_{role}_direct_url", "") or "",
        )

    # -- public role accessors ------------------------------------------------

    def get_chat_llm(self):
        return self._build("chat")

    def get_extract_llm(self):
        return self._build("extract")

    def get_text2gql_llm(self):
        return self._build("text2gql")

    def get_agent_llm(self):
        """Get the LLM instance for agent reasoning.

        Agent tasks require strong reasoning for multi-step planning
        and tool selection. Uses a dedicated LLM configuration.
        """
        return self._build("agent")

    def get_general_llm(self):
        """Get a general-purpose LLM (fallback to chat LLM).

        Used by operators that do not belong to a specific role
        (e.g. DRIFT search synthesis).
        """
        try:
            t = self._get_type("general")
            if t != llm_settings.chat_llm_type:
                return self._build("general")
        except (AttributeError, KeyError):
            pass
        return self._build("chat")


if __name__ == "__main__":
    client = LLMs().get_chat_llm()
    print(client.generate(prompt="What is the capital of China?"))
    print(
        client.generate(
            messages=[{"role": "user", "content": "What is the capital of China?"}]
        )
    )
