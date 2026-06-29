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

"""Tests for hugegraph_llm.engines.memory.client."""

import pytest
from unittest import mock

from hugegraph_llm.engines.memory.client import AsyncMemoryClient, MemoryClient


@pytest.fixture
def mock_requests():
    with mock.patch("hugegraph_llm.engines.memory.client.requests") as m:
        yield m


class TestMemoryClient:
    def test_headers_with_api_key(self):
        client = MemoryClient(base_url="http://localhost:9999", api_key="abc")
        assert client._headers()["Authorization"] == "Bearer abc"

    def test_headers_without_api_key(self, mock_requests):
        with mock.patch("hugegraph_llm.engines.memory.client.memory_settings") as ms:
            ms.llm_api_key = ""
            client = MemoryClient(base_url="http://localhost:9999", api_key=None)
            assert "Authorization" not in client._headers()

    def test_add(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"memory_id": "1"}
        result = client.add("hello", user_id="u1")
        assert result["memory_id"] == "1"
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args.kwargs["json"]["content"] == "hello"
        assert call_args.kwargs["json"]["user_id"] == "u1"

    def test_add_with_optional_fields(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"ok": True}
        client.add("hello", user_id="u1", agent_id="a1", run_id="r1", metadata={"k": "v"})
        payload = mock_requests.post.call_args.kwargs["json"]
        assert payload["agent_id"] == "a1"
        assert payload["run_id"] == "r1"
        assert payload["metadata"] == {"k": "v"}

    def test_search(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"answer": "yes"}
        result = client.search("where", user_id="u1", limit=3, filters={"scope": "private"})
        assert result["answer"] == "yes"
        payload = mock_requests.post.call_args.kwargs["json"]
        assert payload["top_k"] == 3
        assert payload["filters"] == {"scope": "private"}

    def test_get(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.get.return_value.json.return_value = {"id": "1"}
        result = client.get("1", user_id="u1")
        assert result["id"] == "1"

    def test_list(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.get.return_value.json.return_value = [{"id": "1"}]
        result = client.list(user_id="u1")
        assert len(result) == 1

    def test_update(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"status": "ok"}
        result = client.update("1", "new", user_id="u1", metadata={"k": "v"})
        assert result["status"] == "ok"

    def test_delete(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"status": "ok"}
        result = client.delete("1", user_id="u1")
        assert result["status"] == "ok"

    def test_persona(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.get.return_value.json.return_value = {"summary": "s"}
        assert client.get_persona("u1")["summary"] == "s"

        mock_requests.post.return_value.json.return_value = {"summary": "updated"}
        assert client.update_persona("u1", "updated")["summary"] == "updated"

    def test_add_skill(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"memory_id": "s1"}
        result = client.add_skill("skill", user_id="u1")
        assert result["memory_id"] == "s1"
        payload = mock_requests.post.call_args.kwargs["json"]
        assert payload["metadata"]["memory_type"] == "procedural"

    def test_search_skills(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"skills": []}
        result = client.search_skills("q", user_id="u1", top_k=2)
        assert result["skills"] == []

    def test_get_experiences(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"experiences": []}
        result = client.get_experiences(user_id="u1")
        assert result["experiences"] == []

    def test_stats(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.get.return_value.json.return_value = {"memories": 1}
        assert client.stats()["memories"] == 1

    def test_reset(self, mock_requests):
        client = MemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"status": "ok"}
        assert client.reset("u1")["status"] == "ok"


class TestAsyncMemoryClient:
    @pytest.mark.asyncio
    async def test_async_add(self, mock_requests):
        client = AsyncMemoryClient(base_url="http://localhost:9999")
        mock_requests.post.return_value.json.return_value = {"memory_id": "1"}
        result = await client.add("hello", user_id="u1")
        assert result["memory_id"] == "1"
