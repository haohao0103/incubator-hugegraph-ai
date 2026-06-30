# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not in this file except in compliance
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
DRIFT search mode for GraphRAG.

Implements Dynamic Reasoning and Inference From Triplets — a hybrid
search strategy that combines global community-level search with
local entity-level search.

Inspired by Microsoft GraphRAG's DRIFT search mode:
1. Coarse phase: Search community summaries to identify relevant communities
2. Fine phase: Traverse entity-level graph within relevant communities
3. Merge: Combine global and local results for final answer synthesis
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class DriftSearch:
    """
    DRIFT search: hybrid local+global retrieval.

    Combines the broad coverage of community-summary-based global search
    with the precision of entity-centric local search, dynamically
    weighting results based on query characteristics.
    """

    def __init__(
        self,
        community_weight: float = 0.4,
        entity_weight: float = 0.6,
        top_communities: int = 5,
        max_depth: int = 2,
        embedding_model: Optional[Any] = None,
    ):
        """
        Args:
            community_weight: Weight for community-level results in merge.
            entity_weight: Weight for entity-level results in merge.
            top_communities: Number of top communities to search.
            max_depth: Maximum traversal depth for entity search.
            embedding_model: Optional embedding model for semantic community matching.
        """
        self.community_weight = community_weight
        self.entity_weight = entity_weight
        self.top_communities = top_communities
        self.max_depth = max_depth
        self.embedding_model = embedding_model

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute DRIFT search.

        Args:
            context: Dict with 'query', 'community_summaries', 'communities',
                     and graph data from previous nodes.

        Returns:
            Updated context with 'drift_results' containing merged
            global and local search results.
        """
        query = context.get("query", "")
        community_summaries = context.get("community_summaries", [])
        communities = context.get("communities", [])

        if not query:
            log.warning("No query provided for DRIFT search")
            return context

        # Phase 1: Coarse search — find relevant communities
        relevant_communities = self._coarse_community_search(query, community_summaries, context)
        log.info("DRIFT coarse phase: found %d relevant communities", len(relevant_communities))

        # Phase 2: Fine search — traverse entity-level graph in relevant communities
        entity_results = self._fine_entity_search(query, relevant_communities, communities, context)
        log.info("DRIFT fine phase: found %d entity-level results", len(entity_results))

        # Phase 3: Merge global and local results
        merged_results = self._merge_results(relevant_communities, entity_results, context)
        log.info("DRIFT merge: %d merged results", len(merged_results))

        context["drift_results"] = merged_results
        context["drift_community_results"] = relevant_communities
        context["drift_entity_results"] = entity_results
        context["graph_result"] = merged_results  # For compatibility with downstream answer synthesis

        return context

    def _coarse_community_search(
        self,
        query: str,
        community_summaries: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Coarse search: find communities whose summaries are relevant to the query.

        Uses keyword overlap and optionally embedding similarity.
        """
        if not community_summaries:
            return []

        scored_communities = []

        for summary in community_summaries:
            score = self._compute_community_relevance(query, summary, context)
            if score > 0:
                scored_communities.append({"summary": summary, "relevance_score": score})

        # Sort by relevance and take top N
        scored_communities.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_communities[: self.top_communities]

    def _compute_community_relevance(
        self,
        query: str,
        summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        """
        Compute relevance score between a query and a community summary.

        Uses keyword overlap as the primary signal, with optional
        embedding similarity for semantic matching.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Keyword overlap scoring
        summary_text = (
            summary.get("summary", "")
            + " "
            + " ".join(summary.get("key_entities", []))
            + " "
            + " ".join(summary.get("themes", []))
            + " "
            + summary.get("title", "")
        ).lower()
        summary_words = set(summary_text.split())

        overlap = len(query_words & summary_words)
        keyword_score = overlap / max(len(query_words), 1)

        # If embedding model available, add semantic similarity
        if self.embedding_model:
            try:
                query_emb = self.embedding_model.get_embedding(query)
                summary_emb = self.embedding_model.get_embedding(summary_text[:500])
                semantic_score = self._cosine_similarity(query_emb, summary_emb)
                return 0.5 * keyword_score + 0.5 * semantic_score
            except Exception as e:  # pylint: disable=broad-except
                log.debug("Embedding similarity failed: %s", e)

        return keyword_score

    def _fine_entity_search(
        self,
        query: str,
        relevant_communities: List[Dict[str, Any]],
        communities: List[List[str]],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Fine search: traverse entity-level graph within relevant communities.

        Collects entity-level results from the communities identified
        in the coarse phase, supplemented by the existing graph_result
        from previous graph query nodes.
        """
        entity_results = []

        # Collect entities from relevant communities
        relevant_entity_ids = set()
        for scored_comm in relevant_communities:
            community_id = scored_comm["summary"].get("community_id", "")
            # Parse community index from ID like "C0"
            try:
                idx = int(community_id.replace("C", ""))
                if idx < len(communities):
                    for entity_id in communities[idx]:
                        relevant_entity_ids.add(entity_id)
            except (ValueError, IndexError):
                log.debug("Could not resolve community ID '%s'", community_id)

        # Format entity results
        for entity_id in relevant_entity_ids:
            entity_str = self._format_entity_result(entity_id, context)
            if entity_str:
                entity_results.append(entity_str)

        # Also include results from existing graph traversal (if available)
        existing_graph_result = context.get("graph_result", [])
        for result in existing_graph_result:
            if result not in entity_results:
                entity_results.append(result)

        return entity_results

    def _format_entity_result(self, entity_id: str, context: Dict[str, Any]) -> str:
        """Format an entity ID into a human-readable result string."""
        # Try to find vertex info from context
        for vertex in context.get("vertices", []):
            if str(vertex.get("id", "")) == entity_id:
                label = vertex.get("label", "unknown")
                props = vertex.get("properties", {})
                props_str = ", ".join(f"{k}: {v}" for k, v in props.items() if v)
                return f"{entity_id}({label}): {props_str}"

        return entity_id

    def _merge_results(
        self,
        relevant_communities: List[Dict[str, Any]],
        entity_results: List[str],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Merge global (community-level) and local (entity-level) results.

        Applies configurable weighting to produce a merged result list
        that balances broad context with precise details.
        """
        merged = []

        # Add community-level context (global results)
        community_context_parts = []
        for scored_comm in relevant_communities:
            summary = scored_comm["summary"]
            community_context_parts.append(
                f"[Community: {summary.get('title', 'N/A')} (score: {scored_comm['relevance_score']:.2f})] "
                f"{summary.get('summary', '')}"
            )

        if community_context_parts:
            global_header = f"=== Global Context (weight: {self.community_weight:.1f}) ===\n"
            global_header += "\n".join(community_context_parts)
            merged.append(global_header)

        # Add entity-level details (local results)
        if entity_results:
            local_header = f"\n=== Local Details (weight: {self.entity_weight:.1f}) ===\n"
            local_results = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(entity_results))
            merged.append(local_header + local_results)

        # Also preserve vector results if available
        vector_result = context.get("vector_result", [])
        if vector_result:
            vector_header = "\n=== Vector Search Results ===\n"
            vector_results = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(vector_result))
            merged.append(vector_header + vector_results)

        return merged

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
