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

"""LangChain-compatible Retriever backed by HugeGraph GraphRAG.

Implements a dual-path retrieval strategy:
1. **Vector search** – semantic similarity over document embeddings.
2. **Graph expansion** – 1-hop neighbor traversal from entities discovered
   by the vector search, enriching context with structural relationships.

The graph expansion path uses Gremlin ``both()`` traversal with configurable
hop depth to discover related entities, edges, and community context that
pure vector search would miss.
"""

from typing import Any, Dict, List, Optional, Set

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

    # Gremlin query parameter limits
    _MAX_HOP = 2
    _MAX_NEIGHBORS_PER_SEED = 20
    _MAX_GRAPH_RESULTS = 30

    def __init__(
        self,
        embedding: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        top_k: int = 5,
        graph_ratio: float = 0.5,
        hop_depth: int = 1,
    ):
        """Initialize the hybrid retriever.

        Args:
            embedding: Embedding model client.
            vector_index: Vector index for semantic search.
            graph_client: HugeGraph Python client (PyHugeGraph).
            top_k: Total number of results to return.
            graph_ratio: Fraction of *top_k* slots reserved for graph results.
            hop_depth: Graph neighbor expansion depth (1 or 2).
        """
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_client = graph_client
        self._top_k = top_k
        self._graph_ratio = min(max(graph_ratio, 0.0), 1.0)
        self._hop_depth = max(1, min(hop_depth, self._MAX_HOP))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents for a query.

        1. Vector search to find semantically matching entities/chunks.
        2. From vector-hit entity names, launch Gremlin 1-hop (or 2-hop)
           expansion to discover structurally related context.

        :param query: The search query.
        :param k: Number of results (overrides default *top_k*).
        :return: List of document dicts with *content* and *metadata*.
        """
        top_k = k or self._top_k
        results: List[Dict] = []

        # ── Path 1: Vector Search ────────────────────────────────────
        seed_entities: Set[str] = set()
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
                    # Extract entity names from vector hits as seeds
                    seed_entities = self._extract_entity_names(vec_results[:graph_k])
            except Exception as e:
                log.warning("HugeGraphRetriever vector search failed: %s", e)

        # ── Path 2: Graph Expansion ───────────────────────────────
        if self._graph_client:
            graph_k = top_k - len(results)
            if graph_k > 0:
                graph_results = self._graph_expand(seed_entities, graph_k)
                results.extend(graph_results)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_entity_names(self, vec_results: List) -> Set[str]:
        """Extract candidate entity names from vector search results.

        Supports results that are dicts with 'name' key, or string results
        where the first word/token is treated as an entity name.
        """
        names: Set[str] = set()
        for r in vec_results:
            if isinstance(r, dict):
                name = r.get("name") or r.get("entity", "")
                if name:
                    names.add(str(name))
            elif isinstance(r, str):
                # Heuristic: first non-empty segment might be entity name
                parts = r.split("|", 1)[0].strip()
                if parts:
                    names.add(parts)
        return names

    def _graph_expand(self, seed_entities: Set[str], limit: int) -> List[Dict]:
        """Expand from seed entities via graph traversal.

        When seed entities are available, runs a Gremlin ``both()`` traversal
        to discover 1-hop (or 2-hop) neighbors.  Falls back to a general
        high-degree entity query when no seeds exist.
        """
        if seed_entities:
            return self._expand_from_seeds(seed_entities, limit)
        return self._fallback_graph_search(limit)

    def _expand_from_seeds(self, seeds: Set[str], limit: int) -> List[Dict]:
        """Run Gremlin neighbor expansion from known entity seeds.

        Builds a parameterized Gremlin query that:
        1. Finds vertices matching any seed entity name.
        2. Traverses both() edges up to *hop_depth* hops.
        3. Deduplicates and projects the path elements.
        """
        try:
            # Build safe seed list (escape single quotes for Gremlin)
            safe_seeds = [s.replace("'", "\\'") for s in seeds]
            seed_list = ",".join(f"'{s}'" for s in safe_seeds[:10])

            hop = self._hop_depth
            neighbors_limit = self._MAX_NEIGHBORS_PER_SEED

            gremlin = (
                f"g.V().has('Entity', 'name', within({seed_list}))"
                f".both().repeat(__.both()).times({hop - 1})" if hop > 1
                else f"g.V().has('Entity', 'name', within({seed_list})).both()"
            )
            gremlin += (
                f".dedup().limit({neighbors_limit}).valueMap()"
            )

            g_resp = self._graph_client.gremlin(gremlin).exec()
            if not isinstance(g_resp, dict):
                return []

            data = g_resp.get("data", [])
            return [
                {
                    "content": self._format_graph_item(item),
                    "metadata": {"source": "graph", "expansion": "neighbor"},
                }
                for item in data[:min(limit, self._MAX_GRAPH_RESULTS)]
            ]
        except Exception as e:
            log.warning("Graph neighbor expansion failed: %s", e)
            return self._fallback_graph_search(limit)

    def _fallback_graph_search(self, limit: int) -> List[Dict]:
        """Fallback: retrieve high-connectivity entities when no seeds found.

        Uses ``inE().count()`` ordering to return the most connected entities
        as a reasonable approximation of importance.
        """
        try:
            safe_limit = min(limit, self._MAX_GRAPH_RESULTS)
            gremlin = (
                f'g.V().hasLabel("Entity")'
                f".local(__.inE().count()).order().by(__.inE().count(), desc)"
                f".limit({safe_limit}).valueMap()"
            )
            g_resp = self._graph_client.gremlin(gremlin).exec()
            if not isinstance(g_resp, dict):
                return []

            data = g_resp.get("data", [])
            return [
                {
                    "content": self._format_graph_item(item),
                    "metadata": {"source": "graph", "expansion": "high-degree"},
                }
                for item in data[:safe_limit]
            ]
        except Exception as e:
            log.warning("Fallback graph search failed: %s", e)
            return []

    @staticmethod
    def _format_graph_item(item: Any) -> str:
        """Format a graph item (dict from valueMap) into a readable string."""
        if isinstance(item, dict):
            parts = []
            for k, v in item.items():
                val = v if isinstance(v, str) else str(v)
                parts.append(f"{k}: {val}")
            return "; ".join(parts)
        return str(item)
