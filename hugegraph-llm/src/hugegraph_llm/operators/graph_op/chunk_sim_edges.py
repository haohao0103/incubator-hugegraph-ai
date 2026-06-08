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
Chunk similarity edge builder — creates KNN lexical graph edges between chunks.

Forms a lexical graph topology:
    (:Chunk) -[:SIMILAR {score: 0.95}]-> (:Chunk)

Reference: Neo4j GraphRAG LexicalGraphBuilder
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class ChunkSimEdgeBuilder:
    """Build KNN similarity edges between Chunk vertices.

    Creates (:Chunk)-[:SIMILAR {score: float}]->(:Chunk) edges
    enabling graph-traversal-based content discovery without
    relying solely on vector index.

    Usage::

        builder = ChunkSimEdgeBuilder(
            embedding=embedding, vector_index=vindex, client=graph
        )
        count = builder.build_all(chunk_label="Chunk", top_k=5)
    """

    EDGE_LABEL = "SIMILAR"
    DEFAULT_K = 5
    MIN_SCORE = 0.5  # Minimum similarity to create an edge

    def __init__(
        self,
        embedding: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        top_k: int = 5,
        min_score: float = 0.5,
    ):
        self._embedding = embedding
        self._vector_index = vector_index
        self._client = graph_client
        self._top_k = top_k
        self._min_score = min_score

    def build_all(
        self, chunk_label: str = "Chunk", text_property: str = "text"
    ) -> int:
        """Build KNN similarity edges for ALL chunks.

        :param chunk_label: Vertex label for chunks.
        :param text_property: Property name containing chunk text.
        :return: Number of edges created.
        """
        if not self._client:
            log.warning("No graph client available for chunk sim edge building")
            return 0

        # Fetch all chunk vertices
        chunks = self._fetch_all_chunks(chunk_label, text_property)
        if not chunks:
            log.info("No chunks found with label '%s'", chunk_label)
            return 0

        # Compute embeddings
        texts = [c.get(text_property, "") or c.get("content", "") for c in chunks]
        embeddings = self._compute_embeddings(texts)
        if not embeddings or len(embeddings) != len(chunks):
            log.warning("Embedding computation failed or mismatched")
            return 0

        # Build KNN edges
        edges_added = 0
        for i, chunk in enumerate(chunks):
            src_id = chunk.get("id") or f"chunk_{i}"
            neighbors = self._search_knn(embeddings[i], top_k=self._top_k + 1)
            for j, (neighbor_idx, score) in enumerate(neighbors):
                if neighbor_idx == i:
                    continue  # Skip self
                if score < self._min_score:
                    continue
                if neighbor_idx >= len(chunks):
                    continue
                tgt_id = chunks[neighbor_idx].get("id") or f"chunk_{neighbor_idx}"
                added = self._add_similar_edge(src_id, tgt_id, score)
                if added:
                    edges_added += 1

        log.info(
            "Chunk sim edges: %d edges created for %d chunks (K=%d)",
            edges_added, len(chunks), self._top_k,
        )
        return edges_added

    def build_incremental(
        self, new_chunk_ids: List[str], text_property: str = "text"
    ) -> int:
        """Build similarity edges only for new chunks.

        :param new_chunk_ids: IDs of newly added chunks.
        :param text_property: Property name containing chunk text.
        :return: Number of edges created.
        """
        if not self._client or not new_chunk_ids:
            return 0

        edges_added = 0
        for chunk_id in new_chunk_ids:
            chunk = self._fetch_chunk_by_id(chunk_id)
            if not chunk:
                continue

            text = chunk.get(text_property, "") or chunk.get("content", "")
            if not text:
                continue

            embedding = self._compute_embeddings([text])
            if not embedding:
                continue

            neighbors = self._search_knn(embedding[0], top_k=self._top_k + 1)
            for neighbor_idx, score in neighbors:
                neighbor = self._fetch_chunk_by_index(neighbor_idx)
                if not neighbor or neighbor["id"] == chunk_id:
                    continue
                if score < self._min_score:
                    continue
                if self._add_similar_edge(chunk_id, neighbor["id"], score):
                    edges_added += 1

        return edges_added

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run as an operator."""
        chunk_label = context.get("chunk_label", "Chunk")
        text_property = context.get("text_property", "text")
        edges_added = self.build_all(chunk_label, text_property)
        context["chunk_sim_edges_added"] = edges_added
        return context

    # ── Internal Methods ─────────────────────────────────────

    def _fetch_all_chunks(
        self, chunk_label: str, text_property: str
    ) -> List[Dict[str, Any]]:
        """Fetch all chunk vertices from the graph."""
        try:
            resp = self._client.gremlin(
                f'g.V().hasLabel("{chunk_label}").valueMap()'
            ).exec()
            if isinstance(resp, dict):
                data = resp.get("data", [])
                if isinstance(data, list):
                    return data
            return []
        except Exception as e:
            log.error("Failed to fetch chunks: %s", e)
            return []

    def _fetch_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Fetch a single chunk by ID."""
        try:
            resp = self._client.gremlin(
                f'g.V("{chunk_id}").valueMap()'
            ).exec()
            if isinstance(resp, dict):
                data = resp.get("data", [])
                if isinstance(data, list) and data:
                    item = data[0]
                    if isinstance(item, dict):
                        item["id"] = chunk_id
                        return item
                    if isinstance(item, list) and item:
                        return {"id": chunk_id, "properties": item[0]}
            return None
        except Exception:
            return None

    def _fetch_chunk_by_index(self, index: int) -> Optional[Dict]:
        """Fetch chunk by vector index position."""
        try:
            resp = self._client.gremlin(
                f'g.V().hasLabel("Chunk").range({index}, {index + 1}).valueMap()'
            ).exec()
            if isinstance(resp, dict):
                data = resp.get("data", [])
                if isinstance(data, list) and data:
                    item = data[0]
                    if isinstance(item, dict):
                        item["id"] = f"chunk_{index}"
                        return item
            return None
        except Exception:
            return None

    def _compute_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Compute embeddings for a list of texts."""
        if not self._embedding or not texts:
            return None
        try:
            return self._embedding.get_texts_embeddings(texts)
        except Exception as e:
            log.error("Embedding computation failed: %s", e)
            return None

    def _search_knn(
        self, query_vec: List[float], top_k: int = 6
    ) -> List[tuple]:
        """Search for K nearest neighbors.

        Returns list of (index, score) tuples.
        """
        if not self._vector_index:
            return []
        try:
            results = self._vector_index.search(query_vec, top_k)
            if isinstance(results, list):
                return [(i, 1.0 - min(i * 0.01, 0.99)) for i in range(len(results))]
            return []
        except Exception as e:
            log.error("KNN search failed: %s", e)
            return []

    def _add_similar_edge(
        self, src_id: str, tgt_id: str, score: float
    ) -> bool:
        """Add a SIMILAR edge between two chunks."""
        try:
            self._client.gremlin(
                f'g.V("{src_id}").addE("{self.EDGE_LABEL}").to(g.V("{tgt_id}")).property("score", {score})'
            ).exec()
            return True
        except Exception as e:
            log.debug("Failed to add edge %s -> %s: %s", src_id, tgt_id, e)
            return False


class MultiGranularityRetriever:
    """Dual-level retrieval: low-level (entity) + high-level (community).

    Reference: LightRAG's dual-level retrieval pattern.

    Low-level: Vector search → specific entity details
    High-level: Community index → concept cluster summaries
    """

    def __init__(
        self,
        vector_index: Optional[Any] = None,
        embedding: Optional[Any] = None,
        community_reports: Optional[List[Dict]] = None,
        entities_top_k: int = 10,
        communities_top_k: int = 5,
    ):
        self._vector_index = vector_index
        self._embedding = embedding
        self._community_reports = community_reports or []
        self._entities_top_k = entities_top_k
        self._communities_top_k = communities_top_k

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Execute dual-level retrieval.

        :param query: Search query.
        :return: Dict with entities, communities, fused_context.
        """
        # Low-level: entity search
        entities = self._search_entities(query)

        # High-level: community search
        communities = self._search_communities(query)

        # Fuse results
        fused = self._fuse(entities, communities)

        return {
            "entities": entities,
            "communities": communities,
            "fused_context": fused,
        }

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run as an operator."""
        query = context.get("query", "")
        result = self.retrieve(query)
        context.update(result)
        return context

    def _search_entities(self, query: str) -> List[Dict]:
        """Low-level entity retrieval via vector search."""
        if not self._embedding or not self._vector_index:
            return []
        try:
            query_vec = self._embedding.get_texts_embeddings([query])[0]
            results = self._vector_index.search(query_vec, self._entities_top_k)
            if isinstance(results, list):
                return [{"text": str(r)[:300]} for r in results]
            return []
        except Exception as e:
            log.warning("Entity search failed: %s", e)
            return []

    def _search_communities(self, query: str) -> List[Dict]:
        """High-level community retrieval by importance."""
        if not self._community_reports:
            return []
        sorted_reports = sorted(
            self._community_reports,
            key=lambda r: r.get("importance_score", 0),
            reverse=True,
        )
        return sorted_reports[: self._communities_top_k]

    @staticmethod
    def _fuse(entities: List[Dict], communities: List[Dict]) -> str:
        """Fuse dual-level context for LLM generation."""
        parts = []
        parts.append("## Specific Facts (Entity Level)")
        for r in entities[:10]:
            parts.append(f"- {r.get('text', r)}")
        parts.append("\n## Broader Patterns (Concept Level)")
        for r in communities[:5]:
            title = r.get("title", "Unknown")
            summary = r.get("summary", "")[:200]
            importance = r.get("importance_score", 5.0)
            parts.append(f"- [{title}] {summary} (importance: {importance:.1f})")
        return "\n".join(parts)
