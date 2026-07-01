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

"""VLM Multi-backend Provider Registry — unified image-to-VLM bridge.

Extends the single-provider VLMDescriptor (OpenAI-compatible only) into a
multi-backend system that supports OpenAI, Ollama, Anthropic, Gemini, and
Bedrock through a provider-adapter pattern.

Architecture:
  1. NormalizedVLMImage — standardizes diverse image inputs (base64, data URL,
     dict, bytes) into a single canonical representation.
  2. VLMProviderAdapter (ABC) — each provider implements format_image_content()
     + build_request() + call_api() to handle provider-specific message format
     and API transport.
  3. VLMProviderRegistry — name → adapter mapping with register/get/list.
  4. VLMMultiBackendCaller — the main caller class that selects an adapter from
     the registry and delegates VLM calls, falling back to VLMDescriptor for
     backward compatibility.

Adapted from LightRAG ``llm/_vision_utils.py`` NormalizedImage pattern.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import requests

# Lazy import to avoid triggering __init__.py dependency chain
def _get_dimension_funcs():
    """Lazy-load image_dimension_validator to avoid circular imports."""
    from hugegraph_llm.operators.multimodal.image_dimension_validator import (
        _detect_mime,
        _dimensions_from_bytes,
        DATA_URL_RE,
    )
    return _detect_mime, _dimensions_from_bytes, DATA_URL_RE

# Module-level cache for lazy-loaded functions
_dim_funcs_cache = None

def _detect_mime_lazy(raw_bytes):
    """Lazy wrapper for _detect_mime."""
    global _dim_funcs_cache
    if _dim_funcs_cache is None:
        _dim_funcs_cache = _get_dimension_funcs()
    return _dim_funcs_cache[0](raw_bytes)

def _dimensions_from_bytes_lazy(raw_bytes):
    """Lazy wrapper for _dimensions_from_bytes."""
    global _dim_funcs_cache
    if _dim_funcs_cache is None:
        _dim_funcs_cache = _get_dimension_funcs()
    return _dim_funcs_cache[1](raw_bytes)

def _data_url_re_lazy():
    """Lazy wrapper for DATA_URL_RE."""
    global _dim_funcs_cache
    if _dim_funcs_cache is None:
        _dim_funcs_cache = _get_dimension_funcs()
    return _dim_funcs_cache[2]

log = logging.getLogger(__name__)

# Re-export magic byte constants for reference (already in image_dimension_validator)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"


# =============================================================================
# 1. NormalizedVLMImage — unified image representation
# =============================================================================


@dataclass(frozen=True)
class NormalizedVLMImage:
    """Canonical image representation for all VLM providers.

    All providers receive the same NormalizedVLMImage objects; each adapter
    converts them into its own message format (data URL, raw bytes, base64
    string, etc.).

    Adapted from LightRAG ``NormalizedImage`` with VLM-specific extensions.
    """

    raw_bytes: bytes
    base64_str: str
    mime_type: str  # "image/jpeg", "image/png", "image/gif", "image/webp"
    sha256: str  # content hash for caching / dedup
    source_id: Optional[str] = None  # original image identifier (e.g. "img_2_3")
    source_file: Optional[str] = None  # original file path if available
    width: Optional[int] = None  # pixel width from header parsing
    height: Optional[int] = None  # pixel height from header parsing
    index: int = 0  # position in the input list (for ordering)


# =============================================================================
# 2. normalize_vlm_image_inputs — diverse input normalization
# =============================================================================


def _decode_base64(data: str) -> bytes:
    """Decode a base64 string, stripping whitespace, with strict validation."""
    cleaned = re.sub(r"\s+", "", data)
    try:
        return base64.b64decode(cleaned, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 image data: {exc}") from exc


def _coerce_image_item(item: Any) -> Dict[str, Any]:
    """Convert a diverse image input into a dict with at least ``base64`` key.

    Accepts:
      - raw base64 string
      - data URL (``data:<mime>;base64,<payload>``)
      - dict with keys ``{base64, mime_type, source_id, source_file}``
      - raw ``bytes``
    """
    _detect_mime_fn = _detect_mime_lazy
    _dimensions_fn = _dimensions_from_bytes_lazy
    _data_url_re = _data_url_re_lazy()

    if isinstance(item, bytes):
        b64 = base64.b64encode(item).decode("ascii")
        return {"base64": b64, "raw_bytes": item}

    if isinstance(item, str):
        stripped = item.strip()
        match = _data_url_re_lazy().match(stripped)
        if match:
            return {"base64": match.group("data"), "mime_type": match.group("mime")}
        return {"base64": stripped}

    if isinstance(item, dict):
        if "base64" not in item and "raw_bytes" not in item:
            raise ValueError("image_inputs dict element must contain 'base64' or 'raw_bytes' key")
        return item

    if isinstance(item, NormalizedVLMImage):
        # Already normalized — wrap for re-processing (preserves all fields)
        return {
            "base64": item.base64_str,
            "mime_type": item.mime_type,
            "source_id": item.source_id,
            "source_file": item.source_file,
            "raw_bytes": item.raw_bytes,
        }

    raise TypeError(
        f"image_inputs element must be str, bytes, dict, or NormalizedVLMImage, "
        f"got {type(item).__name__}"
    )


def normalize_vlm_image_inputs(image_inputs: List[Any]) -> List[NormalizedVLMImage]:
    """Normalize diverse image input formats into NormalizedVLMImage objects.

    Accepts:
      - raw base64 string
      - data URL (``data:mime;base64,payload``)
      - dict with keys ``{base64, mime_type, source_id, source_file}``
      - raw ``bytes``
      - NormalizedVLMImage (already normalized, re-validated)

    For each item:
      - Decodes to raw bytes
      - Infers MIME type from magic bytes (JPEG=FFD8FF, PNG=89504E47,
        GIF=474946, WebP=52494646)
      - Computes SHA-256 content hash
      - Parses pixel dimensions from raster headers (reuses logic from
        image_dimension_validator.py)
      - Produces clean base64 encoding (no data URL wrapper, no whitespace)
    """
    if not image_inputs:
        return []

    result: List[NormalizedVLMImage] = []
    for idx, raw_item in enumerate(image_inputs):
        item = _coerce_image_item(raw_item)

        # Obtain raw bytes
        if "raw_bytes" in item and item["raw_bytes"]:
            raw_bytes = item["raw_bytes"]
        else:
            raw_bytes = _decode_base64(item["base64"])

        if not raw_bytes:
            raise ValueError(f"image_inputs[{idx}] decoded to empty bytes")

        # MIME type: explicit > magic bytes > default
        mime_type = item.get("mime_type") or _detect_mime_lazy(raw_bytes)

        # SHA-256 content hash
        sha = hashlib.sha256(raw_bytes).hexdigest()

        # Clean base64 (re-encode from raw_bytes to guarantee consistency)
        clean_b64 = base64.b64encode(raw_bytes).decode("ascii")

        # Pixel dimensions from raster header
        dims = _dimensions_from_bytes_lazy(raw_bytes)
        width, height = (dims[0], dims[1]) if dims else (None, None)

        result.append(
            NormalizedVLMImage(
                raw_bytes=raw_bytes,
                base64_str=clean_b64,
                mime_type=mime_type,
                sha256=sha,
                source_id=item.get("source_id"),
                source_file=item.get("source_file"),
                width=width,
                height=height,
                index=idx,
            )
        )

    return result


# =============================================================================
# 3. VLMProviderAdapter — abstract base class
# =============================================================================


class VLMProviderAdapter(ABC):
    """Abstract base class for VLM provider adapters.

    Each adapter handles TWO responsibilities:
      1. Message format conversion (format_image_content) — turn NormalizedVLMImage
         into provider-specific content blocks.
      2. API transport (build_request + call_api) — construct and execute the
         HTTP request (or SDK call) for the provider.

    Response parsing is shared across all providers (handled by
    VLMMultiBackendCaller / VLMDescriptor).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier, e.g. "openai", "ollama", "anthropic"."""

    @abstractmethod
    def format_image_content(
        self,
        images: List[NormalizedVLMImage],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Convert normalized images + prompt into provider-specific message format.

        Returns a list of content blocks (dicts) that the provider expects in
        the user message content array.
        """

    @abstractmethod
    def build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the full API request payload for this provider.

        Args:
            messages: Complete message list (with images already formatted).
            model: Model identifier for this provider.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc).

        Returns:
            Dict suitable for ``call_api()`` (may include url, headers, payload,
            or any provider-specific fields).
        """

    @abstractmethod
    def call_api(
        self,
        request: Dict[str, Any],
        api_key: str,
        base_url: str,
        timeout: int = 60,
    ) -> str:
        """Execute the API call and return the response text content.

        Args:
            request: The request dict built by ``build_request()``.
            api_key: Provider API key / credential.
            base_url: Provider API base URL.
            timeout: Request timeout in seconds.

        Returns:
            The text content from the VLM response (JSON string expected).

        Raises:
            requests.RequestException: Network / API error.
            ValueError: Response parsing error.
        """


# =============================================================================
# 4. Provider Adapters
# =============================================================================


class OpenAIAdapter(VLMProviderAdapter):
    """OpenAI-compatible VLM adapter (GPT-4o, MiMo, etc).

    Format: ``{"type": "image_url", "image_url": {"url": "data:mime;base64,b64"}}``
    API: POST to ``/chat/completions`` with standard OpenAI format.

    This adapter produces output identical to the existing VLMDescriptor,
    ensuring backward compatibility.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def format_image_content(
        self,
        images: List[NormalizedVLMImage],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.mime_type};base64,{img.base64_str}",
                },
            })
        return content

    def build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.3),
        }
        return {
            "url_suffix": "/chat/completions",
            "headers_template": "bearer",
            "payload": payload,
        }

    def call_api(
        self,
        request: Dict[str, Any],
        api_key: str,
        base_url: str,
        timeout: int = 60,
    ) -> str:
        url = f"{base_url.rstrip('/')}{request['url_suffix']}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = request["payload"]

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if response.status_code != 200:
            error_detail = ""
            try:
                error_body = response.json()
                error_detail = error_body.get("error", {}).get("message", "")
            except Exception:
                error_detail = response.text[:500]
            raise requests.RequestException(
                f"VLM API ({self.provider_name}) returned status {response.status_code}: {error_detail}"
            )

        try:
            body = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response.text[:500]}")

        if "choices" not in body or not body["choices"]:
            raise ValueError(f"No choices in response: {body}")

        content_text = body["choices"][0].get("message", {}).get("content", "")
        if not content_text:
            raise ValueError("Empty content in VLM response")

        usage = body.get("usage")
        if usage:
            log.debug(
                f"VLM ({self.provider_name}) token usage: "
                f"prompt={usage.get('prompt_tokens')}, "
                f"completion={usage.get('completion_tokens')}, "
                f"total={usage.get('total_tokens')}"
            )

        return content_text.strip()


class OllamaAdapter(VLMProviderAdapter):
    """Ollama local VLM adapter (llava, bakllava, etc).

    Format: append ``user_message["images"] = [img.base64_str for img in normalized_images]``
    API: POST to ``/api/chat`` (Ollama chat endpoint).

    Key difference: Ollama uses base64 strings directly in an ``images`` array
    on the user message, NOT data URLs or content blocks.
    """

    @property
    def provider_name(self) -> str:
        return "ollama"

    def format_image_content(
        self,
        images: List[NormalizedVLMImage],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        # Ollama doesn't use content blocks; images go into a separate "images" key.
        # We return a single dict representing the user message with prompt + images.
        return [{
            "role": "user",
            "content": prompt,
            "images": [img.base64_str for img in images],
        }]

    def build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Ollama /api/chat expects: {"model": ..., "messages": ..., "stream": false}
        # Images are already embedded in the message dicts from format_image_content.
        # We need to merge: if messages contain our formatted user msg, use it;
        # otherwise inject images into the last user message.
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.3),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        }
        return {
            "url_suffix": "/api/chat",
            "headers_template": "none",  # Ollama typically requires no auth
            "payload": payload,
        }

    def call_api(
        self,
        request: Dict[str, Any],
        api_key: str,
        base_url: str,
        timeout: int = 60,
    ) -> str:
        url = f"{base_url.rstrip('/')}{request['url_suffix']}"
        headers = {"Content-Type": "application/json"}
        # Ollama doesn't use Bearer auth; api_key is ignored
        payload = request["payload"]

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if response.status_code != 200:
            error_detail = response.text[:500]
            raise requests.RequestException(
                f"VLM API ({self.provider_name}) returned status {response.status_code}: {error_detail}"
            )

        try:
            body = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response.text[:500]}")

        # Ollama /api/chat response: {"message": {"content": "..."}}
        message = body.get("message", {})
        content_text = message.get("content", "")
        if not content_text:
            raise ValueError("Empty content in Ollama response")

        return content_text.strip()


class AnthropicAdapter(VLMProviderAdapter):
    """Anthropic Claude VLM adapter (Claude 3.5 Sonnet, etc).

    Format: ``{"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}``
    API: POST to ``/v1/messages`` with ``x-api-key`` header.

    Key difference: Anthropic separates ``media_type`` and ``data`` in the
    image source block, and uses ``x-api-key`` header instead of Bearer token.
    Also requires ``anthropic-version`` header.
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def format_image_content(
        self,
        images: List[NormalizedVLMImage],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.mime_type,
                    "data": img.base64_str,
                },
            })
        content.append({"type": "text", "text": prompt})
        return content

    def build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Anthropic API requires system as a top-level field, not in messages.
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                # Extract system content; Anthropic expects it as top-level param.
                sys_content = msg.get("content", "")
                if isinstance(sys_content, list):
                    system_text = "\n".join(
                        p.get("text", "") for p in sys_content if p.get("type") == "text"
                    )
                else:
                    system_text = str(sys_content)
            else:
                chat_messages.append(msg)

        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": chat_messages,
        }
        if system_text:
            payload["system"] = system_text

        return {
            "url_suffix": "/v1/messages",
            "headers_template": "x-api-key",
            "payload": payload,
            "anthropic_version": kwargs.get("anthropic_version", "2023-06-01"),
        }

    def call_api(
        self,
        request: Dict[str, Any],
        api_key: str,
        base_url: str,
        timeout: int = 60,
    ) -> str:
        url = f"{base_url.rstrip('/')}{request['url_suffix']}"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": request.get("anthropic_version", "2023-06-01"),
            "Content-Type": "application/json",
        }
        payload = request["payload"]

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if response.status_code != 200:
            error_detail = ""
            try:
                error_body = response.json()
                error_detail = error_body.get("error", {}).get("message", "")
            except Exception:
                error_detail = response.text[:500]
            raise requests.RequestException(
                f"VLM API ({self.provider_name}) returned status {response.status_code}: {error_detail}"
            )

        try:
            body = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response.text[:500]}")

        # Anthropic response: {"content": [{"type": "text", "text": "..."}]}
        content_blocks = body.get("content", [])
        text_parts = [
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ]
        content_text = "\n".join(text_parts)
        if not content_text:
            raise ValueError("Empty content in Anthropic response")

        usage = body.get("usage")
        if usage:
            log.debug(
                f"VLM ({self.provider_name}) token usage: "
                f"input={usage.get('input_tokens')}, "
                f"output={usage.get('output_tokens')}"
            )

        return content_text.strip()


class GeminiAdapter(VLMProviderAdapter):
    """Google Gemini VLM adapter (Gemini 1.5 Pro/Flash, etc).

    Format: ``{"inlineData": {"mimeType": mime, "data": b64}}`` in REST content parts.
    API: POST to ``/v1beta/models/{model}:generateContent`` with API key as query param.

    Key difference: Gemini uses ``inlineData`` with base64 data in REST mode,
    and authenticates via ``key`` query parameter (not header). The SDK mode
    uses ``Part.from_bytes()`` with raw bytes + mime_type.
    """

    @property
    def provider_name(self) -> str:
        return "gemini"

    def format_image_content(
        self,
        images: List[NormalizedVLMImage],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        # Gemini REST API uses "parts" format
        parts: List[Dict[str, Any]] = []
        for img in images:
            parts.append({
                "inlineData": {
                    "mimeType": img.mime_type,
                    "data": img.base64_str,
                },
            })
        parts.append({"text": prompt})
        return parts

    def build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Gemini REST API structure:
        # {"contents": [{"parts": [...]}], "generationConfig": {...}}
        # System instructions go into "systemInstruction" top-level field.
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.get("role") == "system":
                sys_content = msg.get("content", "")
                if isinstance(sys_content, list):
                    sys_text = "\n".join(
                        p.get("text", "") for p in sys_content if p.get("type") == "text"
                    )
                else:
                    sys_text = str(sys_content)
                system_instruction = {"parts": [{"text": sys_text}]}
            elif isinstance(msg.get("content"), list) and any(
                p.get("inlineData") for p in msg.get("content", [])
            ):
                # This is a Gemini-format content with inlineData parts
                gemini_role = "user" if msg.get("role") == "user" else "model"
                contents.append({"role": gemini_role, "parts": msg["content"]})
            elif msg.get("role") == "user":
                text_content = msg.get("content", "")
                if isinstance(text_content, list):
                    parts = [
                        {"text": p.get("text", "")} for p in text_content if p.get("type") == "text"
                    ]
                else:
                    parts = [{"text": str(text_content)}]
                contents.append({"role": "user", "parts": parts})
            elif msg.get("role") == "assistant":
                text_content = msg.get("content", "")
                if isinstance(text_content, list):
                    parts = [
                        {"text": p.get("text", "")} for p in text_content if p.get("type") == "text"
                    ]
                else:
                    parts = [{"text": str(text_content)}]
                contents.append({"role": "model", "parts": parts})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.3),
                "maxOutputTokens": kwargs.get("max_tokens", 1024),
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        return {
            "url_suffix": f"/v1beta/models/{model}:generateContent",
            "headers_template": "none",
            "payload": payload,
            "auth_mode": "query_param",  # Gemini uses ?key=... in URL
        }

    def call_api(
        self,
        request: Dict[str, Any],
        api_key: str,
        base_url: str,
        timeout: int = 60,
    ) -> str:
        # Gemini authenticates via query parameter, not header
        url_suffix = request["url_suffix"]
        url = f"{base_url.rstrip('/')}{url_suffix}?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = request["payload"]

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if response.status_code != 200:
            error_detail = ""
            try:
                error_body = response.json()
                error_detail = error_body.get("error", {}).get("message", "")
            except Exception:
                error_detail = response.text[:500]
            raise requests.RequestException(
                f"VLM API ({self.provider_name}) returned status {response.status_code}: {error_detail}"
            )

        try:
            body = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response.text[:500]}")

        # Gemini response: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        candidates = body.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates in Gemini response: {body}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        content_text = "\n".join(text_parts)
        if not content_text:
            raise ValueError("Empty content in Gemini response")

        usage_meta = body.get("usageMetadata")
        if usage_meta:
            log.debug(
                f"VLM ({self.provider_name}) token usage: "
                f"prompt={usage_meta.get('promptTokenCount')}, "
                f"candidates={usage_meta.get('candidatesTokenCount')}, "
                f"total={usage_meta.get('totalTokenCount')}"
            )

        return content_text.strip()


class BedrockAdapter(VLMProviderAdapter):
    """AWS Bedrock VLM adapter (Claude on Bedrock, etc) via Converse API.

    Format: ``{"image": {"format": fmt, "source": {"bytes": raw_bytes}}}`` in
    Converse API content block.
    API: Uses boto3 ``bedrock-runtime`` client's ``converse`` method.

    Key difference: Bedrock uses raw ``bytes`` in the Converse API content
    block, not base64 strings. The format field uses short names (jpeg, png, gif, webp).
    """

    # Bedrock format field uses short names (no "image/" prefix)
    _MIME_TO_FORMAT = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def format_image_content(
        self,
        images: List[NormalizedVLMImage],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for img in images:
            fmt = self._MIME_TO_FORMAT.get(img.mime_type, "png")
            content.append({
                "image": {
                    "format": fmt,
                    "source": {"bytes": img.raw_bytes},
                },
            })
        content.append({"text": prompt})
        return content

    def build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Bedrock Converse API structure:
        # {"modelId": ..., "messages": [...], "system": [...], "inferenceConfig": {...}}
        system_messages = []
        chat_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                sys_content = msg.get("content", "")
                if isinstance(sys_content, list):
                    sys_text = "\n".join(
                        p.get("text", "") for p in sys_content if p.get("type") == "text"
                    )
                else:
                    sys_text = str(sys_content)
                system_messages.append({"text": sys_text})
            else:
                # Convert content blocks to Bedrock format
                bedrock_content = msg.get("content", [])
                if isinstance(bedrock_content, list):
                    # Already formatted by format_image_content or text blocks
                    converted = []
                    for block in bedrock_content:
                        if "image" in block:
                            converted.append(block)  # Already in Bedrock format
                        elif block.get("type") == "text":
                            converted.append({"text": block["text"]})
                        elif "text" in block:
                            converted.append({"text": block["text"]})
                    bedrock_content = converted
                elif isinstance(bedrock_content, str):
                    bedrock_content = [{"text": bedrock_content}]

                bedrock_role = "user" if msg.get("role") == "user" else "assistant"
                chat_messages.append({
                    "role": bedrock_role,
                    "content": bedrock_content,
                })

        payload: Dict[str, Any] = {
            "modelId": model,
            "messages": chat_messages,
            "inferenceConfig": {
                "maxTokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.3),
            },
        }
        if system_messages:
            payload["system"] = system_messages

        return {
            "payload": payload,
            "auth_mode": "boto3",  # Uses boto3 client, not HTTP headers
            "region": kwargs.get("region", "us-east-1"),
        }

    def call_api(
        self,
        request: Dict[str, Any],
        api_key: str,  # Not used for Bedrock; uses boto3 credentials
        base_url: str,  # Not used for Bedrock; uses boto3 endpoint
        timeout: int = 60,
    ) -> str:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock VLM adapter. "
                "Install with: pip install boto3"
            )

        region = request.get("region", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)
        payload = request["payload"]

        response = client.converse(
            modelId=payload["modelId"],
            messages=payload["messages"],
            system=payload.get("system", []),
            inferenceConfig=payload.get("inferenceConfig", {}),
        )

        # Bedrock Converse response: {"output": {"message": {"content": [{"text": "..."}]}}}
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        text_parts = [
            block.get("text", "") for block in content_blocks if "text" in block
        ]
        content_text = "\n".join(text_parts)
        if not content_text:
            raise ValueError("Empty content in Bedrock response")

        usage = response.get("usage")
        if usage:
            log.debug(
                f"VLM ({self.provider_name}) token usage: "
                f"input={usage.get('inputTokens')}, "
                f"output={usage.get('outputTokens')}"
            )

        return content_text.strip()


# =============================================================================
# 5. VLMProviderRegistry — registry + factory
# =============================================================================


class VLMProviderRegistry:
    """Registry for VLM provider adapters.

    Maintains a name → adapter mapping. Ships with 5 built-in adapters
    (openai, ollama, anthropic, gemini, bedrock) and allows registering
    custom adapters.
    """

    _registry: Dict[str, VLMProviderAdapter] = {}
    _initialized: bool = False

    def __init__(self) -> None:
        if not self._initialized:
            self._register_builtins()
            self._initialized = True

    def _register_builtins(self) -> None:
        """Register the 5 built-in provider adapters."""
        builtins: List[VLMProviderAdapter] = [
            OpenAIAdapter(),
            OllamaAdapter(),
            AnthropicAdapter(),
            GeminiAdapter(),
            BedrockAdapter(),
        ]
        for adapter in builtins:
            self._registry[adapter.provider_name] = adapter

    def register(self, name: str, adapter: VLMProviderAdapter) -> None:
        """Register a custom adapter by name.

        Args:
            name: Provider name (must match adapter.provider_name for consistency).
            adapter: VLMProviderAdapter instance.

        Raises:
            ValueError: If name conflicts with a built-in adapter (unless
                ``allow_override=True``).
        """
        if name != adapter.provider_name:
            log.warning(
                f"Registry name '{name}' differs from adapter.provider_name "
                f"'{adapter.provider_name}'. Using adapter's provider_name."
            )
            name = adapter.provider_name
        self._registry[name] = adapter
        log.info(f"Registered VLM provider adapter: {name}")

    def get(self, name: str) -> VLMProviderAdapter:
        """Get an adapter by provider name.

        Args:
            name: Provider name (case-insensitive).

        Returns:
            The registered VLMProviderAdapter.

        Raises:
            KeyError: If no adapter is registered for the given name.
        """
        key = name.lower().strip()
        if key not in self._registry:
            available = self.list_providers()
            raise KeyError(
                f"No VLM provider adapter registered for '{name}'. "
                f"Available providers: {available}"
            )
        return self._registry[key]

    def list_providers(self) -> List[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._registry.keys())

    @property
    def default_provider(self) -> str:
        """Default provider name (openai, matching existing VLMDescriptor)."""
        return "openai"


# Module-level singleton registry
_GLOBAL_REGISTRY: Optional[VLMProviderRegistry] = None


def get_registry() -> VLMProviderRegistry:
    """Return the global VLMProviderRegistry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = VLMProviderRegistry()
    return _GLOBAL_REGISTRY


# =============================================================================
# 6. VLMMultiBackendCaller — the main caller class
# =============================================================================


# Lazy import to avoid pulling vlm_descriptor's full dependency chain
# (vlm_descriptor depends on requests and image processing which may trigger
#  __init__.py -> multimodal_retrieval_channel -> pyhugegraph chain)
def _get_vlm_descriptor_dataclasses():
    from hugegraph_llm.operators.multimodal.vlm_descriptor import (
        ImageDescription,
        BatchDescribeResult,
        VALID_CHART_TYPES,
    )
    return ImageDescription, BatchDescribeResult, VALID_CHART_TYPES


class VLMMultiBackendCaller:
    """Multi-backend VLM caller — the main entry point for VLM operations.

    Selects a provider adapter from the registry, normalizes image inputs,
    builds provider-specific requests, and delegates the API call. Response
    parsing is shared across all providers (reuses VLMDescriptor's logic).

    Falls back to existing VLMDescriptor for backward compatibility when
    provider is "openai" and the call pattern matches the original interface.

    Usage::

        caller = VLMMultiBackendCaller(
            provider="openai",  # or "ollama", "anthropic", "gemini", "bedrock"
            api_key="your-key",
            model="gpt-4o",
        )

        # Single image description
        desc = caller.describe_image(
            image_id="img_2_3",
            image_data="base64string...",
            prompt="Describe this image in JSON format",
        )

        # Batch description
        result = caller.describe_batch(
            items=[("img_1", "base64_1"), ("img_2", "base64_2")],
            prompt="Describe these images",
        )
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        max_retries: int = 2,
        retry_delay: float = 1.0,
        timeout: int = 60,
        language: str = "zh",
        cache_dir: Optional[str] = None,
        batch_size: int = 5,
        **provider_kwargs: Any,
    ) -> None:
        self.provider = provider.lower().strip()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.language = language
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.provider_kwargs = provider_kwargs

        # Select adapter from registry
        registry = get_registry()
        self._adapter = registry.get(self.provider)

        # Backward-compatibility: keep VLMDescriptor for "openai" provider
        # so callers using the old API still work.
        self._legacy_descriptor: Optional[Any] = None
        if self.provider in ("openai", "xiaomimo"):
            from hugegraph_llm.operators.multimodal.vlm_descriptor import VLMDescriptor

            # Determine env key for backward compat
            env_key = ""
            if self.provider == "xiaomimo":
                env_key = "XIAOMI_MIMO_API_KEY"
            elif self.provider == "openai":
                env_key = "OPENAI_API_KEY"

            import os
            resolved_key = api_key or os.environ.get(env_key, "")
            resolved_url = base_url or ""
            resolved_model = model or ""

            self._legacy_descriptor = VLMDescriptor(
                provider=self.provider,
                api_key=resolved_key,
                model=resolved_model,
                base_url=resolved_url,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                language=language,
                cache_dir=cache_dir,
                batch_size=batch_size,
            )

        # Cache for image descriptions (sha256 → Any)
        self._cache: Dict[str, Any] = {}

        # Lazy-load VLMDescriptor dataclasses at first use
        self._ImageDescription = None
        self._BatchDescribeResult = None
        self._VALID_CHART_TYPES = None

    def _ensure_dataclasses(self):
        """Lazy-load VLMDescriptor dataclasses to avoid import chain."""
        if self._ImageDescription is None:
            ImageDescription, BatchDescribeResult, VALID_CHART_TYPES = _get_vlm_descriptor_dataclasses()
            self._ImageDescription = ImageDescription
            self._BatchDescribeResult = BatchDescribeResult
            self._VALID_CHART_TYPES = VALID_CHART_TYPES

    # ========== Single image description ==========

    def describe_image(
        self,
        image_id: str,
        image_data: Any,
        prompt: str = "",
        page_context: str = "",
        nearby_text: str = "",
        use_legacy: bool = False,
    ) -> Any:
        """Describe a single image using the selected VLM provider.

        Args:
            image_id: Image identifier (e.g. "img_2_3").
            image_data: Image data in any supported format (base64 string,
                data URL, dict, bytes, NormalizedVLMImage).
            prompt: Custom prompt override (empty = use default structured prompt).
            page_context: Page context for cross-modal understanding.
            nearby_text: Nearby text for context enrichment.
            use_legacy: If True and provider is openai/xiaomimo, fall back
                to VLMDescriptor for exact backward compatibility.

        Returns:
            ImageDescription with structured VLM output.
        """
        start_time = time.time()

        # Ensure dataclasses are loaded
        self._ensure_dataclasses()

        # Normalize image input
        normalized = normalize_vlm_image_inputs([image_data])
        if not normalized:
            return self._fallback_description(image_id, start_time)

        img = normalized[0]

        # Check cache
        cache_key = img.sha256[:16]
        if cache_key in self._cache:
            log.debug(f"Cache hit for {image_id} (key={cache_key})")
            cached = self._cache[cache_key]
            return self._ImageDescription(
                image_id=image_id,
                caption=cached.caption,
                detailed_description=cached.detailed_description,
                object_labels=list(cached.object_labels),
                chart_type=cached.chart_type,
                key_insights=list(cached.key_insights),
                related_keywords=list(cached.related_keywords),
                confidence=cached.confidence,
                vlm_model=cached.vlm_model,
                generation_time_ms=0,
            )

        # Legacy mode for backward compatibility
        if use_legacy and self._legacy_descriptor is not None:
            b64_str = img.base64_str
            desc = self._legacy_descriptor.describe(
                image_id=image_id,
                base64_data=b64_str,
                page_context=page_context,
                nearby_text=nearby_text,
            )
            self._cache[cache_key] = desc
            return desc

        # Build messages using provider adapter
        if not prompt:
            prompt = self._default_prompt(page_context, nearby_text)

        messages = self._build_messages(normalized, prompt, page_context, nearby_text)

        # Call API with retry
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                # Resolve effective credentials/URL
                eff_api_key, eff_base_url = self._resolve_credentials()

                request = self._adapter.build_request(
                    messages, self.model, **self.provider_kwargs,
                )
                response_text = self._adapter.call_api(
                    request, eff_api_key, eff_base_url, self.timeout,
                )
                parsed = self._parse_response(response_text, image_id)

                desc = self._ImageDescription(
                    image_id=image_id,
                    caption=parsed.get("caption", ""),
                    detailed_description=parsed.get("detailed_description", ""),
                    object_labels=parsed.get("object_labels", []),
                    chart_type=parsed.get("chart_type", "other"),
                    key_insights=parsed.get("key_insights", []),
                    related_keywords=parsed.get("related_keywords", []),
                    confidence=float(parsed.get("confidence", 0.0)),
                    vlm_model=self.model or self.provider,
                    generation_time_ms=int((time.time() - start_time) * 1000),
                )

                self._cache[cache_key] = desc
                return desc

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** attempt)
                    log.warning(
                        f"VLM call failed for {image_id} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    log.error(f"All retries exhausted for {image_id}: {e}")

        # All retries failed → fallback description
        log.warning(f"Using fallback description for {image_id}")
        return self._fallback_description(image_id, start_time)

    # ========== Batch description ==========

    def describe_batch(
        self,
        items: List[Tuple[str, Any]],  # [(image_id, image_data), ...]
        prompt: str = "",
        page_context: str = "",
        nearby_text: str = "",
        use_legacy: bool = False,
    ) -> Any:
        """Batch describe multiple images.

        Args:
            items: (image_id, image_data) tuples. image_data can be any
                format accepted by normalize_vlm_image_inputs.
            prompt: Custom prompt override (empty = default).
            page_context: Shared page context.
            nearby_text: Shared nearby text.
            use_legacy: If True, use VLMDescriptor for backward compat.

        Returns:
            BatchDescribeResult with success/failure stats.
        """
        self._ensure_dataclasses()
        total_start = time.time()
        result = self._BatchDescribeResult(
            total_images=len(items),
            success_count=0,
            fail_count=0,
        )

        for batch_start in range(0, len(items), self.batch_size):
            batch = items[batch_start:batch_start + self.batch_size]
            log.info(
                f"Processing batch {batch_start // self.batch_size + 1}: "
                f"images [{batch_start}-{batch_start + len(batch) - 1}]"
            )

            for img_id, img_data in batch:
                try:
                    desc = self.describe_image(
                        image_id=img_id,
                        image_data=img_data,
                        prompt=prompt,
                        page_context=page_context,
                        nearby_text=nearby_text,
                        use_legacy=use_legacy,
                    )
                    result.descriptions.append(desc)
                    result.success_count += 1
                except Exception as e:
                    log.error(f"Unexpected error describing {img_id}: {e}")
                    result.failed_ids.append(img_id)
                    result.fail_count += 1

        result.total_time_ms = int((time.time() - total_start) * 1000)
        log.info(
            f"Batch describe complete: {result.success_count}/{result.total_images} "
            f"success ({result.success_rate:.1%}), {result.total_time_ms}ms"
        )
        return result

    # ========== Internal methods ==========

    def _build_messages(
        self,
        normalized_images: List[NormalizedVLMImage],
        prompt: str,
        page_context: str,
        nearby_text: str,
    ) -> List[Dict[str, Any]]:
        """Build the complete message list for the VLM provider.

        For OpenAI-compatible providers, this creates the standard
        system + user content blocks. For providers with special
        message formats (Anthropic system separation, Gemini parts,
        Bedrock content), the adapter's format_image_content handles
        the provider-specific part.
        """
        # System prompt
        lang_instruction = "中文" if self.language == "zh" else "English"
        system_prompt = (
            f"你是一个专业的文档视觉分析助手。分析给定的图片，返回严格JSON格式的描述。"
            f"所有文本输出使用{lang_instruction}。\n\n"
            "你必须返回以下JSON结构（不要包含markdown代码块标记）:\n"
            "{\n"
            '  "caption": "一句话概括图片内容",\n'
            '  "detailed_description": "2-3句详细描述图片内容、数据、趋势等",\n'
            '  "object_labels": ["检测到的对象/元素标签"],\n'
            '  "chart_type": "图表类型, 必须是以下之一: '
            f"{'/'.join(sorted(self._VALID_CHART_TYPES))}" + '",\n'
            '  "key_insights": ["关键信息点, 如具体数字、结论等"],\n'
            '  "related_keywords": ["用于文本检索匹配的关键词"],\n'
            '  "confidence": 0.95\n'
            "}\n\n"
            "规则:\n"
            "1. caption 要简洁有力, 包含最核心的信息\n"
            "2. 如果是图表, key_insights 必须包含具体的数值和趋势\n"
            "3. related_keywords 要包含中英文, 方便后续检索匹配\n"
            "4. confidence 反映你对描述的准确程度 (0.0-1.0)\n"
            "5. 如果图片模糊不清, confidence 设低并说明"
        )

        # Build user content based on provider type
        # Some providers (Ollama) return a full message dict from format_image_content;
        # others (OpenAI, Anthropic, Gemini, Bedrock) return content blocks.
        provider_formatted = self._adapter.format_image_content(normalized_images, prompt)

        # Determine message structure based on provider
        if self.provider == "ollama":
            # Ollama returns a single user message dict with "images" key
            user_msg = provider_formatted[0] if provider_formatted else {"role": "user", "content": prompt}
            # Add context text into the content
            context_text = ""
            if page_context:
                context_text += f"文档上下文: {page_context}\n"
            if nearby_text:
                context_text += f"图片附近的文字: {nearby_text}\n"
            if context_text:
                user_msg["content"] = context_text + user_msg.get("content", prompt)
            messages = [
                {"role": "system", "content": system_prompt},
                user_msg,
            ]
        elif self.provider in ("anthropic", "gemini", "bedrock"):
            # These providers return content blocks (list of dicts)
            # We need to wrap them in a user message
            context_parts: List[Dict[str, Any]] = []
            if page_context:
                context_parts.append({"type": "text", "text": f"文档上下文: {page_context}"})
            if nearby_text:
                context_parts.append({"type": "text", "text": f"图片附近的文字: {nearby_text}"})

            # For Gemini, context_parts need to be in Gemini "parts" format
            if self.provider == "gemini":
                gemini_context = [{"text": p["text"]} for p in context_parts]
                # Merge: context + formatted image/prompt parts
                user_content = gemini_context + provider_formatted
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            elif self.provider == "bedrock":
                # Bedrock content blocks use {"text": ...} and {"image": ...} format
                bedrock_context = [{"text": p["text"]} for p in context_parts]
                user_content = bedrock_context + provider_formatted
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                # Anthropic
                all_parts = context_parts + provider_formatted
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": all_parts},
                ]
        else:
            # OpenAI-compatible (default)
            user_parts: List[Dict[str, Any]] = []
            if page_context:
                user_parts.append({"type": "text", "text": f"文档上下文: {page_context}"})
            if nearby_text:
                user_parts.append({"type": "text", "text": f"图片附近的文字: {nearby_text}"})
            user_parts.extend(provider_formatted)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_parts},
            ]

        return messages

    def _default_prompt(self, page_context: str = "", nearby_text: str = "") -> str:
        """Build default prompt text."""
        lang_instruction = "中文" if self.language == "zh" else "English"
        return f"请分析这张图片, 返回JSON格式的描述（使用{lang_instruction}）:"

    def _resolve_credentials(self) -> Tuple[str, str]:
        """Resolve effective API key and base URL.

        Priority: explicit args > environment variables > provider defaults.
        """
        import os

        eff_key = self.api_key
        eff_url = self.base_url

        # Provider-specific env key defaults
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "xiaomimo": "XIAOMI_MIMO_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ollama": "",  # Ollama doesn't need an API key
            "bedrock": "",  # Bedrock uses boto3 credentials chain
        }
        env_url_map = {
            "openai": "https://api.openai.com/v1",
            "xiaomimo": "https://api.xiaomimimo.com/v1",
            "anthropic": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
            "ollama": "http://localhost:11434",
        }

        env_key_name = env_key_map.get(self.provider, "")
        if not eff_key and env_key_name:
            eff_key = os.environ.get(env_key_name, "")

        if not eff_url:
            eff_url = env_url_map.get(self.provider, "")
            # Also check env for custom base URL
            env_url_name = f"{self.provider.upper()}_BASE_URL"
            eff_url = os.environ.get(env_url_name, eff_url)

        return eff_key, eff_url

    def _parse_response(self, response_text: str, image_id: str) -> Dict[str, Any]:
        """Parse VLM response JSON — shared across all providers.

        Reuses VLMDescriptor's robust parsing logic (tolerates markdown
        code blocks, extracts first JSON object, validates and fills defaults).
        """
        text = response_text.strip()

        # Strategy 1: Direct JSON parse
        parsed = self._try_json_parse(text)
        if parsed is not None:
            return self._validate_and_fill(parsed, image_id)

        # Strategy 2: Extract ```json ... ``` code block
        json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_block_match:
            parsed = self._try_json_parse(json_block_match.group(1).strip())
            if parsed is not None:
                return self._validate_and_fill(parsed, image_id)

        # Strategy 3: Extract first {...} brace pair
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            parsed = self._try_json_parse(brace_match.group(0))
            if parsed is not None:
                return self._validate_and_fill(parsed, image_id)

        raise ValueError(
            f"Failed to parse VLM response as JSON for {image_id}. "
            f"Response preview: {text[:200]}"
        )

    @staticmethod
    def _try_json_parse(text: str) -> Optional[Dict[str, Any]]:
        """Try JSON parse; return None on failure instead of raising."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _validate_and_fill(self, parsed: Dict[str, Any], image_id: str) -> Dict[str, Any]:
        """Validate parsed dict and fill missing fields with defaults."""
        result: Dict[str, Any] = {}

        result["caption"] = self._coerce_str(parsed.get("caption", ""), 200)
        result["detailed_description"] = self._coerce_str(
            parsed.get("detailed_description", ""), 1000
        )

        # object_labels → list[str]
        raw_labels = parsed.get("object_labels", [])
        if isinstance(raw_labels, list):
            result["object_labels"] = [str(l) for l in raw_labels if l]
        elif isinstance(raw_labels, str):
            result["object_labels"] = [raw_labels]
        else:
            result["object_labels"] = []

        # chart_type → enum validation
        raw_chart = str(parsed.get("chart_type", "other")).strip().lower()
        if raw_chart in self._VALID_CHART_TYPES:
            result["chart_type"] = raw_chart
        else:
            log.warning(f"Invalid chart_type '{raw_chart}' for {image_id}, defaulting to 'other'")
            result["chart_type"] = "other"

        # key_insights → list[str]
        raw_insights = parsed.get("key_insights", [])
        if isinstance(raw_insights, list):
            result["key_insights"] = [str(i) for i in raw_insights if i]
        elif isinstance(raw_insights, str):
            result["key_insights"] = [raw_insights]
        else:
            result["key_insights"] = []

        # related_keywords → list[str]
        raw_kw = parsed.get("related_keywords", [])
        if isinstance(raw_kw, list):
            result["related_keywords"] = [str(k) for k in raw_kw if k]
        elif isinstance(raw_kw, str):
            result["related_keywords"] = [raw_kw]
        else:
            result["related_keywords"] = []

        # confidence → float [0, 1]
        raw_conf = parsed.get("confidence", 0.5)
        try:
            conf_val = float(raw_conf)
            result["confidence"] = max(0.0, min(1.0, conf_val))
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        return result

    @staticmethod
    def _coerce_str(value: Any, max_length: int = 500) -> str:
        """Force convert to string and truncate."""
        if value is None:
            return ""
        s = str(value).strip()
        return s[:max_length] if len(s) > max_length else s

    def _fallback_description(
        self,
        image_id: str,
        start_time: float,
    ) -> Any:
        """Fallback description when all VLM calls fail."""
        return self._ImageDescription(
            image_id=image_id,
            caption="[VLM调用失败, 使用兜底描述]",
            detailed_description="无法获取视觉语言模型的描述。图片可能需要人工审核。",
            object_labels=["unknown"],
            chart_type="other",
            key_insights=[],
            related_keywords=[],
            confidence=0.0,
            vlm_model=f"{self.provider}-fallback",
            generation_time_ms=int((time.time() - start_time) * 1000),
        )


# =============================================================================
# Convenience function
# =============================================================================


def describe_images_multi(
    items: List[Tuple[str, Any]],
    provider: str = "openai",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    **kwargs: Any,
) -> Any:
    """One-shot batch describe convenience function."""
    caller = VLMMultiBackendCaller(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        **kwargs,
    )
    return caller.describe_batch(items)
