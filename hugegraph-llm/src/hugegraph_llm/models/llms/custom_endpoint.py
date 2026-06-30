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

"""Custom LLM client for non-standard API endpoints (e.g., model name in URL path)."""

from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.utils.log import log


class CustomEndpointLLM(BaseLLM):
    """LLM client that sends requests directly to a custom URL without appending /chat/completions.

    Useful for endpoints where the model name is part of the URL path (e.g., doubao).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "default",
        max_tokens: int = 8092,
        temperature: float = 0.01,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_url = api_base
        self.api_key = api_key or ""
        self.model = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if default_headers:
            self.headers.update(default_headers)

    def _build_payload(self, messages: List[Dict[str, Any]], stream: bool = False) -> Dict[str, Any]:
        return {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

    def _parse_response(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Empty choices in LLM response: {str(data)[:200]}")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"Empty content in LLM response: {str(data)[:200]}")
        usage = data.get("usage")
        if usage:
            log.info("Token usage: %s", usage)
        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    def generate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a response using httpx directly."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        payload = self._build_payload(messages)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, headers=self.headers, json=payload)
                if response.status_code == 401:
                    log.critical("The provided API key is invalid")
                    return "Error: The provided API key is invalid"
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                log.critical("Fatal: %s", e)
                return f"Error: {e}"
            raise
        except Exception as e:
            log.error("Retrying LLM call %s", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def agenerate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a response asynchronously using httpx."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        payload = self._build_payload(messages)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, headers=self.headers, json=payload)
                if response.status_code == 401:
                    log.critical("The provided API key is invalid")
                    return "Error: The provided API key is invalid"
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                log.critical("Fatal: %s", e)
                return f"Error: {e}"
            raise
        except Exception as e:
            log.error("Retrying LLM call %s", e)
            raise

    def generate_streaming(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        on_token_callback: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """Generate streaming response using SSE."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        payload = self._build_payload(messages, stream=True)
        accumulated = ""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", self.api_url, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated += content
                                if on_token_callback:
                                    on_token_callback(content)
                                yield content
                        except Exception:
                            continue
        except Exception as e:
            log.error("Error in streaming: %s", e)
            raise

    async def agenerate_streaming(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        on_token_callback: Optional[Callable] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response asynchronously using SSE."""
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        payload = self._build_payload(messages, stream=True)
        accumulated = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.api_url, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated += content
                                if on_token_callback:
                                    on_token_callback(content)
                                yield content
                        except Exception:
                            continue
        except Exception as e:
            log.error("Error in streaming: %s", e)
            raise

    def num_tokens_from_string(self, string: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(string) // 3

    def max_allowed_token_length(self) -> int:
        return self.max_tokens

    def get_llm_type(self) -> str:
        return "custom_endpoint"
