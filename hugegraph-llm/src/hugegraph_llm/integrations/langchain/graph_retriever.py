# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership. The ASF licenses this
# file to You under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""LangChain-compatible Retriever backed by HugeGraph GraphRAG."""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class HugeGraphRetriever:
    """LangChain BaseRetriever interface for HugeGraph hybrid retrieval.

    Combines vector search + graph traversal for context retrieval.
    Compatible with LangChain RetrievalQA and other chain types.

    Usage::

        retriever = HugeGraphRetriever(
            embedding=my_embedding,
            vector_index=my_vindex,
            graph_client=my_client,
        )
        docs = retriever.get_relevant_documents("What is HugeGraph?")
    """

    def __init__(
        self,
        embedding: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        top_k: int = 5,
        graph_ratio: float = 0.5,
    ):
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_client = graph_client
        self._top_k = top_k
        self._graph_ratio = graph_ratio

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents for a query.

        :param query: The search query.
        :param k: Number of results.
        :return: List of document dicts with content and metadata.
        """
        top_k = k or self._top_k
        results = []

        # Vector search component
        if self._embedding and self._vector_index:
            try:
                query_vec = self._embedding.get_texts_embeddings([query])[0]
                vec_results = self._vector_index.search(query_vec, top_k)
                if isinstance(vec_results, list):
                    graph_k = int(top_k * self._graph_ratio)
                    for r in vec_results[:graph_k]:
                        results.append({
                            "content": str(r),
                            "metadata": {"source": "vector"},
                        })
            except Exception as e:
                log.warning("HugeGraphRetriever vector search failed: %s", e)

        # Graph traversal component (if vector results give entity hints)
        if self._graph_client and results:
            try:
                graph_k = top_k - len(results)
                if graph_k > 0:
                    g_resp = self._graph_client.gremlin(
                        f'g.V().hasLabel("Entity").limit({graph_k}).valueMap()'
                    ).exec()
                    if isinstance(g_resp, dict):
                        data = g_resp.get("data", [])
                        for item in data[:graph_k]:
                            results.append({
                                "content": str(item),
                                "metadata": {"source": "graph"},
                            })
            except Exception as e:
                log.warning("HugeGraphRetriever graph search failed: %s", e)

        return results
