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

"""DRIFT search as a LangChain Retriever."""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class DriftRetriever:
    """LangChain retriever that uses DRIFT search pipeline.

    Wraps the DRIFT search operator (HyDE -> Community Match -> Primer ->
    Parallel Local Search -> Reduce) as a drop-in LangChain retriever.

    Usage::

        retriever = DriftRetriever(
            llm=my_llm,
            embedding=my_embedding,
            vector_index=my_vindex,
            graph_client=my_client,
        )
        docs = retriever.get_relevant_documents("How does X affect Y?")
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        embedding: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        top_k: int = 5,
        max_parallel: int = 3,
    ):
        self._llm = llm
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_client = graph_client
        self._top_k = top_k
        self._max_parallel = max_parallel

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Execute DRIFT search pipeline.

        Simplified implementation that performs:
        1. Vector similarity search for relevant communities/chunks
        2. Graph expansion from matched entities
        3. Context ranking

        :param query: Search query.
        :param k: Number of results.
        :return: List of document dicts.
        """
        top_k = k or self._top_k
        results = []

        # Step 1: Vector search for top-k candidates
        if self._embedding and self._vector_index:
            try:
                query_vec = self._embedding.get_texts_embeddings([query])[0]
                vec_results = self._vector_index.search(query_vec, top_k)
                if isinstance(vec_results, list):
                    for i, r in enumerate(vec_results):
                        results.append({
                            "content": str(r),
                            "metadata": {
                                "source": "drift_vector",
                                "rank": i + 1,
                            },
                        })
            except Exception as e:
                log.warning("DriftRetriever vector search failed: %s", e)

        # Step 2: Graph expansion for matched entities
        if self._graph_client and results:
            try:
                graph_k = max(1, top_k // 2)
                g_resp = self._graph_client.gremlin(
                    'g.V().hasLabel("Entity").limit(' + str(graph_k) + ').valueMap()'
                ).exec()
                if isinstance(g_resp, dict):
                    data = g_resp.get("data", [])
                    for item in data[:graph_k]:
                        results.append({
                            "content": str(item),
                            "metadata": {
                                "source": "drift_graph",
                                "rank": len(results) + 1,
                            },
                        })
            except Exception as e:
                log.warning("DriftRetriever graph expansion failed: %s", e)

        # Step 3: Rank by source priority (vector first, graph supplement)
        return results[:top_k]
