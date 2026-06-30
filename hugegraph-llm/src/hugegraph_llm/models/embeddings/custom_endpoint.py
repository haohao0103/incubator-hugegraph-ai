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


from typing import Dict, List, Optional

import httpx

from hugegraph_llm.models.embeddings.base import BaseEmbedding


class CustomEndpointEmbedding(BaseEmbedding):
    """Embedding client for non-standard endpoints that don't follow OpenAI's /embeddings path."""

    def __init__(
        self,
        embedding_dimension: int = 1536,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.model = model_name
        self.embedding_dimension = embedding_dimension
        self.api_url = api_url
        self.api_key = api_key or ""
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if default_headers:
            self.headers.update(default_headers)
        self.client = httpx.AsyncClient(timeout=60.0)

    def get_embedding_dim(self) -> int:
        return self.embedding_dimension

    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.api_url,
                headers=self.headers,
                json={"input": text, "model": self.model},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    def get_texts_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts with automatic batch splitting."""
        all_embeddings = []
        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = client.post(
                    self.api_url,
                    headers=self.headers,
                    json={"input": batch, "model": self.model},
                )
                response.raise_for_status()
                data = response.json()
                all_embeddings.extend([item["embedding"] for item in data["data"]])
        return all_embeddings

    async def async_get_texts_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts asynchronously."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json={"input": batch, "model": self.model},
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend([item["embedding"] for item in data["data"]])
        return all_embeddings

    async def async_get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text asynchronously."""
        response = await self.client.post(
            self.api_url,
            headers=self.headers,
            json={"input": text, "model": self.model},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
