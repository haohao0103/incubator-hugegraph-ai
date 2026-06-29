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

"""
Python client SDK for HugeGraph-AI-Memory (Mem0-style).

The sync client calls the local Flask memory server via REST. A direct
backend wrapper is also provided for in-process usage.
"""

from typing import Any, Dict, List, Optional

import requests

from hugegraph_llm.config.memory_config import memory_settings


class MemoryClient:
    """Synchronous HTTP client for the memory server."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or f"http://{memory_settings.memory_server_host}:"
                         f"{memory_settings.memory_server_port}").rstrip("/")
        self.api_key = api_key or memory_settings.llm_api_key

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        resp = requests.post(
            f"{self.base_url}{path}", json=payload, headers=self._headers(), timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = requests.get(
            f"{self.base_url}{path}", params=params, headers=self._headers(), timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    def add(
        self,
        content: str,
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"content": content, "user_id": user_id}
        if agent_id:
            payload["agent_id"] = agent_id
        if run_id:
            payload["run_id"] = run_id
        if metadata:
            payload["metadata"] = metadata
        return self._post("/api/memory/add", payload)

    def search(
        self,
        query: str,
        user_id: str = "demo_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": limit,
        }
        if agent_id:
            payload["agent_id"] = agent_id
        if run_id:
            payload["run_id"] = run_id
        if filters:
            payload["filters"] = filters
        return self._post("/api/memory/search", payload)

    def get(self, memory_id: str, user_id: str = "demo_user") -> Optional[Dict[str, Any]]:
        return self._get("/api/memory/get", {"id": memory_id, "user_id": user_id})

    def list(self, user_id: str = "demo_user") -> List[Dict[str, Any]]:
        return self._get("/api/memory/list", {"user_id": user_id})

    def update(
        self,
        memory_id: str,
        content: str,
        user_id: str = "demo_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"id": memory_id, "content": content, "user_id": user_id}
        if metadata:
            payload["metadata"] = metadata
        return self._post("/api/memory/update", payload)

    def delete(self, memory_id: str, user_id: str = "demo_user") -> Dict[str, Any]:
        return self._post("/api/memory/delete", {"id": memory_id, "user_id": user_id})

    def get_persona(self, user_id: str = "demo_user") -> Dict[str, Any]:
        return self._get("/api/memory/persona", {"user_id": user_id})

    def update_persona(self, user_id: str = "demo_user", summary: str = "") -> Dict[str, Any]:
        return self._post("/api/memory/persona", {"user_id": user_id, "summary": summary})

    def add_skill(
        self,
        content: str,
        user_id: str = "demo_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Add skill as a memory with type=procedural; the server may treat it specially later.
        return self.add(
            content=content,
            user_id=user_id,
            metadata={**(metadata or {}), "memory_type": "procedural"},
        )

    def search_skills(
        self,
        query: str,
        user_id: str = "demo_user",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self._post("/api/memory/skills", {"query": query, "user_id": user_id, "top_k": top_k})

    def get_experiences(
        self,
        query: str = "",
        user_id: str = "demo_user",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self._post("/api/memory/experiences", {"query": query, "user_id": user_id, "top_k": top_k})

    def stats(self) -> Dict[str, Any]:
        return self._get("/api/stats")

    def reset(self, user_id: str = "demo_user") -> Dict[str, Any]:
        return self._post("/api/clear", {"user_id": user_id})


class AsyncMemoryClient:
    """Asynchronous client stub (aligned with Mem0 AsyncMemoryClient)."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self._sync = MemoryClient(base_url=base_url, api_key=api_key)

    async def add(self, *args, **kwargs) -> Dict[str, Any]:
        return self._sync.add(*args, **kwargs)

    async def search(self, *args, **kwargs) -> Dict[str, Any]:
        return self._sync.search(*args, **kwargs)

    async def get(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        return self._sync.get(*args, **kwargs)

    async def list(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return self._sync.list(*args, **kwargs)

    async def update(self, *args, **kwargs) -> Dict[str, Any]:
        return self._sync.update(*args, **kwargs)

    async def delete(self, *args, **kwargs) -> Dict[str, Any]:
        return self._sync.delete(*args, **kwargs)
