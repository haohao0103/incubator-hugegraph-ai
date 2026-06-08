# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership. The ASF licenses this
# file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""LangChain-compatible VectorStore backed by HugeGraph."""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class HugeGraphVectorStore:
    """LangChain VectorStore interface backed by HugeGraph FAISS/Qdrant/Milvus.

    Usage (with LangChain)::

        from langchain.vectorstores import VectorStore

        store = HugeGraphVectorStore(
            embedding=my_embedding,
            vector_index=my_faiss_index,
            graph_name="mygraph",
        )
        store.add_texts(["Hello world", "Second document"])
        results = store.similarity_search("Hello", k=2)
    """

    def __init__(
        self,
        embedding: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        graph_name: str = "hugegraph",
        top_k: int = 5,
        dis_threshold: float = 2.0,
    ):
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_name = graph_name
        self._top_k = top_k
        self._dis_threshold = dis_threshold

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector index.

        :param texts: List of text strings to embed and index.
        :param metadatas: Optional list of metadata dicts.
        :return: List of IDs (placeholder strings).
        """
        if not self._embedding or not self._vector_index:
            log.warning("VectorStore: no embedding or vector index configured")
            return [f"doc_{i}" for i in range(len(texts))]

        try:
            embeddings = self._embedding.get_texts_embeddings(texts)
            for i, emb in enumerate(embeddings):
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                self._vector_index.add_with_ids([emb], [f"doc_{i}"])
                if hasattr(self._vector_index, "add_properties"):
                    self._vector_index.add_properties(f"doc_{i}", meta)
            return [f"doc_{i}" for i in range(len(texts))]
        except Exception as e:
            log.error("VectorStore add_texts failed: %s", e)
            return []

    def similarity_search(
        self, query: str, k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        :param query: Query string.
        :param k: Number of results (default: top_k).
        :return: List of dicts with "content", "metadata", "score".
        """
        if not self._embedding or not self._vector_index:
            return []

        top_k = k or self._top_k
        try:
            query_vec = self._embedding.get_texts_embeddings([query])[0]
            results = self._vector_index.search(query_vec, top_k)
            if not isinstance(results, list):
                return []
            return [
                {
                    "content": str(r),
                    "metadata": {},
                    "score": 0.0,
                }
                for r in results
            ]
        except Exception as e:
            log.error("VectorStore similarity_search failed: %s", e)
            return []

    @classmethod
    def from_texts(
        cls, texts: List[str], embedding: Any, metadatas: Optional[List[Dict]] = None, **kwargs
    ) -> "HugeGraphVectorStore":
        """Create a VectorStore from texts."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store
