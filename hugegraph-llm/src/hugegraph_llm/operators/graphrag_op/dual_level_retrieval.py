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
LightRAG-style dual-level retrieval.

Core insight from LightRAG: different questions need different retrieval
granularity levels:

1. Low-level (entity-centric): For specific fact questions like
   "What is X?" or "Who is Y?" — retrieves 1-2 hop subgraph around
   the target entity.

2. High-level (relationship-centric): For abstract/broad questions like
   "How are X and Y related?" or "What are the key themes?" — retrieves
   relationship paths and aggregates across multiple entities.

This replaces the community-detection-based global/local split from
Microsoft GraphRAG. Instead of requiring expensive community detection
and hierarchical summaries (which prevent incremental updates), we use
a simpler but equally effective approach:

- Low-level: Direct entity lookup + neighbor traversal
- High-level: Relationship path search + multi-entity aggregation

No community detection needed, no global rebuild on updates.

Reference: LightRAG (https://github.com/HKUDS/LightRAG)
         Huolala production implementation (货拉拉元初团队)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from hugegraph_llm.utils.log import log


class RetrievalLevel(str, Enum):
    """Retrieval granularity level."""

    LOW = "low"  # Entity-centric: specific facts
    HIGH = "high"  # Relationship-centric: abstract/broad questions
    HYBRID = "hybrid"  # Both levels merged


class DualLevelRetriever:
    """
    LightRAG-style dual-level retrieval.

    Provides two retrieval granularities without depending on
    community detection:

    - Low-level: Entity-centric retrieval for specific fact questions.
      Looks up entities by keyword/ID and traverses their 1-2 hop
      neighborhoods.

    - High-level: Relationship-centric retrieval for abstract questions.
      Finds relationship paths between entities and aggregates
      multi-entity context.

    Benefits over community-based approaches:
    - Supports incremental updates (no global restructuring)
    - Lower latency (no Map-Reduce over communities)
    - Simpler architecture (fewer moving parts)
    - Proven in production (Huolala: 56%→78% accuracy)
    """

    def __init__(
        self,
        graph_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        low_level_max_depth: int = 2,
        low_level_max_neighbors: int = 20,
        high_level_max_paths: int = 10,
        high_level_max_hops: int = 3,
    ):
        """
        Args:
            graph_client: HugeGraph client for Gremlin queries.
            embedding_model: Optional embedding model for semantic matching.
            low_level_max_depth: Max traversal depth for low-level retrieval.
            low_level_max_neighbors: Max neighbors per entity in low-level.
            high_level_max_paths: Max relationship paths in high-level.
            high_level_max_hops: Max hops for path search in high-level.
        """
        self.graph_client = graph_client
        self.embedding_model = embedding_model
        self.low_level_max_depth = low_level_max_depth
        self.low_level_max_neighbors = low_level_max_neighbors
        self.high_level_max_paths = high_level_max_paths
        self.high_level_max_hops = high_level_max_hops

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute dual-level retrieval.

        Args:
            context: Dict with 'query', 'keywords', and optionally
                     'entity_name_to_id', 'graph_result'.

        Returns:
            Updated context with 'low_level_results', 'high_level_results',
            and 'dual_level_results' (merged).
        """
        query = context.get("query", "")
        keywords = context.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]

        if not query:
            log.warning("No query provided for dual-level retrieval")
            return context

        # Determine retrieval level from query characteristics
        level = self._determine_retrieval_level(query, keywords)
        context["retrieval_level"] = level.value
        log.info("Dual-level retrieval: query='%s', level=%s", query[:50], level.value)

        # Execute retrieval based on level
        low_results = []
        high_results = []

        if level in (RetrievalLevel.LOW, RetrievalLevel.HYBRID):
            low_results = self._low_level_retrieval(query, keywords, context)

        if level in (RetrievalLevel.HIGH, RetrievalLevel.HYBRID):
            high_results = self._high_level_retrieval(query, keywords, context)

        # Merge results
        merged = self._merge_results(low_results, high_results, level)

        context["low_level_results"] = low_results
        context["high_level_results"] = high_results
        context["dual_level_results"] = merged
        # For compatibility with downstream merge_rerank / answer_synthesize
        context["graph_result"] = merged

        log.info(
            "Dual-level results: low=%d, high=%d, merged=%d",
            len(low_results),
            len(high_results),
            len(merged),
        )
        return context

    def _determine_retrieval_level(self, query: str, keywords: List[str]) -> RetrievalLevel:
        """
        Determine the appropriate retrieval level for a query.

        Heuristics:
        - Specific entity questions (who/what/where) → LOW
        - Relationship questions (how/why/compare) → HIGH
        - Multiple entities mentioned → HYBRID
        - Ambiguous → HYBRID (safe default)
        """
        query_lower = query.lower()

        # Low-level indicators: specific entity questions
        low_indicators = [
            "who is",
            "what is",
            "where is",
            "when did",
            "what does",
            "是谁",
            "是什么",
            "什么是",
            "在哪里",
            "什么时候",
            "做什么",
            "define",
            "definition",
            "定义",
            "含义",
        ]

        # High-level indicators: abstract/relationship questions
        high_indicators = [
            "how are",
            "related to",
            "relationship",
            "connection",
            "compare",
            "difference",
            "trend",
            "overview",
            "summarize",
            "如何关联",
            "关系",
            "联系",
            "比较",
            "区别",
            "趋势",
            "概述",
        ]

        low_score = sum(1 for ind in low_indicators if ind in query_lower)
        high_score = sum(1 for ind in high_indicators if ind in query_lower)

        # If multiple keywords/entities mentioned, likely hybrid
        if len(keywords) >= 3:
            return RetrievalLevel.HYBRID

        if low_score > high_score:
            return RetrievalLevel.LOW
        if high_score > low_score:
            return RetrievalLevel.HIGH

        # Default: hybrid for best coverage
        return RetrievalLevel.HYBRID

    def _low_level_retrieval(
        self,
        query: str,
        keywords: List[str],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Low-level retrieval: entity-centric search.

        For each keyword, find the matching entity and traverse its
        1-2 hop neighborhood to gather specific facts.

        This is the primary retrieval mode for most user questions
        (Huolala found 80%+ queries are entity-centric).
        """
        results = []
        entity_name_to_id = context.get("entity_name_to_id", {})

        for keyword in keywords:
            # Step 1: Find entity by keyword
            vertex_id = self._resolve_keyword_to_vertex(keyword, entity_name_to_id)
            if not vertex_id:
                # Fallback: try semantic search if embedding available
                vertex_id = self._semantic_entity_search(keyword, context)

            if not vertex_id:
                log.debug("No entity found for keyword '%s'", keyword)
                continue

            # Step 2: Traverse entity neighborhood
            subgraph = self._traverse_entity_neighborhood(vertex_id)
            results.extend(subgraph)

            # Step 3: Get entity properties
            entity_info = self._get_entity_info(vertex_id)
            if entity_info:
                results.append(entity_info)

        # Also include existing graph results from previous graph query
        existing_graph_result = context.get("graph_result", [])
        for item in existing_graph_result:
            if isinstance(item, str) and item not in results:
                results.append(item)

        return results

    def _high_level_retrieval(
        self,
        query: str,
        keywords: List[str],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        High-level retrieval: relationship-centric search.

        Finds relationship paths between entities and aggregates
        multi-entity context. This handles abstract/broad questions
        without needing community summaries.

        Key difference from community-based global search:
        - No community detection required
        - No hierarchical summaries required
        - Works directly on entity-relationship graph
        - Incremental-update-friendly
        """
        results = []
        entity_name_to_id = context.get("entity_name_to_id", {})

        # Resolve keywords to entity IDs
        entity_ids = []
        for keyword in keywords:
            vid = self._resolve_keyword_to_vertex(keyword, entity_name_to_id)
            if vid:
                entity_ids.append(vid)

        if len(entity_ids) >= 2:
            # Multi-entity: find paths between pairs
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    paths = self._find_relationship_paths(entity_ids[i], entity_ids[j])
                    results.extend(paths)

        if len(entity_ids) >= 1:
            # Aggregate context from all found entities
            for vid in entity_ids:
                agg_context = self._aggregate_entity_context(vid)
                if agg_context:
                    results.append(agg_context)

        # If no entities resolved, try broader keyword-based relationship search
        if not entity_ids and keywords:
            for keyword in keywords:
                rel_context = self._keyword_relationship_search(keyword, context)
                results.extend(rel_context)

        return results

    def _resolve_keyword_to_vertex(self, keyword: str, entity_name_to_id: Dict[str, str]) -> Optional[str]:
        """Resolve a keyword to a vertex ID using name mapping."""
        # Direct name match
        if keyword in entity_name_to_id:
            return entity_name_to_id[keyword]

        # Case-insensitive match
        if keyword.lower() in entity_name_to_id:
            return entity_name_to_id[keyword.lower()]

        # Partial match
        for name, vid in entity_name_to_id.items():
            if keyword.lower() in name.lower() or name.lower() in keyword.lower():
                return vid

        # Try graph client lookup
        if self.graph_client:
            return self._graph_lookup_vertex(keyword)

        return None

    def _graph_lookup_vertex(self, keyword: str) -> Optional[str]:
        """Look up a vertex in the graph by name property."""
        if not self.graph_client:
            return None
        try:
            result = self.graph_client.gremlin().exec(gremlin=f"g.V().has('name', '{keyword}').limit(1).id()")
            data = result.get("data", [])
            if data:
                return str(data[0])
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Graph lookup failed for '%s': %s", keyword, e)
        return None

    def _semantic_entity_search(self, keyword: str, context: Dict[str, Any]) -> Optional[str]:
        """Use embedding similarity to find matching entity."""
        if not self.embedding_model:
            return None

        try:
            self.embedding_model.get_embedding(keyword)
            # This would need a vector index over entity names
            # For now, return None — can be enhanced later with vector index
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Semantic entity search failed for '%s': %s", keyword, e)

        return None

    def _traverse_entity_neighborhood(self, vertex_id: str) -> List[str]:
        """
        Traverse the 1-2 hop neighborhood of an entity.

        Returns formatted strings describing the entity's local context.
        """
        results = []

        if not self.graph_client:
            return results

        try:
            # 1-hop neighbors
            gremlin = f"g.V('{vertex_id}').both().limit({self.low_level_max_neighbors}).elementMap()"
            result = self.graph_client.gremlin().exec(gremlin=gremlin)
            for item in result.get("data", []):
                results.append(self._format_vertex_result(item))

            # Direct edges
            gremlin_edges = f"g.V('{vertex_id}').bothE().limit({self.low_level_max_neighbors}).elementMap()"
            result_edges = self.graph_client.gremlin().exec(gremlin=gremlin_edges)
            for item in result_edges.get("data", []):
                results.append(self._format_edge_result(item))

        except Exception as e:  # pylint: disable=broad-except
            log.debug("Neighborhood traversal failed for '%s': %s", vertex_id, e)

        return results

    def _get_entity_info(self, vertex_id: str) -> Optional[str]:
        """Get formatted entity info."""
        if not self.graph_client:
            return None

        try:
            result = self.graph_client.gremlin().exec(gremlin=f"g.V('{vertex_id}').elementMap()")
            data = result.get("data", [])
            if data:
                return self._format_vertex_result(data[0])
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Entity info lookup failed for '%s': %s", vertex_id, e)

        return None

    def _find_relationship_paths(self, source_id: str, target_id: str) -> List[str]:
        """Find relationship paths between two entities."""
        results = []

        if not self.graph_client:
            return results

        try:
            gremlin = (
                f"g.V('{source_id}')"
                f".repeat both().simplePath()"
                f".until(or(loops().is({self.high_level_max_hops}),"
                f"           is('{target_id}')))"
                f".limit({self.high_level_max_paths})"
                f".path().by('name')"
            )
            result = self.graph_client.gremlin().exec(gremlin=gremlin)
            for path in result.get("data", []):
                if isinstance(path, list):
                    results.append(" → ".join(str(p) for p in path))
                else:
                    results.append(str(path))
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Path search failed between '%s' and '%s': %s", source_id, target_id, e)

        return results

    def _aggregate_entity_context(self, vertex_id: str) -> Optional[str]:
        """Aggregate relationship context around an entity for high-level retrieval."""
        if not self.graph_client:
            return None

        try:
            # Get all relationships for this entity
            gremlin = (
                f"g.V('{vertex_id}).bothE().otherV()"
                f".limit({self.low_level_max_neighbors})"
                f".group().by('label').by('name')"
            )
            result = self.graph_client.gremlin().exec(gremlin=gremlin)
            data = result.get("data", [])
            if data:
                return f"[Entity {vertex_id} relationships]: {data}"
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Context aggregation failed for '%s': %s", vertex_id, e)

        return None

    def _keyword_relationship_search(self, keyword: str, context: Dict[str, Any]) -> List[str]:
        """Search for relationships involving a keyword when no entity is resolved."""
        results = []

        # Try to find from existing graph results
        for item in context.get("graph_result", []):
            if isinstance(item, str) and keyword.lower() in item.lower():
                results.append(item)

        # Try vector results
        for item in context.get("vector_result", []):
            if isinstance(item, str) and keyword.lower() in item.lower():
                results.append(item)

        return results

    def _merge_results(
        self,
        low_results: List[str],
        high_results: List[str],
        level: RetrievalLevel,
    ) -> List[str]:
        """
        Merge low-level and high-level results.

        Deduplicates and applies level-appropriate weighting.
        """
        merged = []
        seen: Set[str] = set()

        # Determine weights based on level
        if level == RetrievalLevel.LOW:
            low_weight, high_weight = 0.8, 0.2
        elif level == RetrievalLevel.HIGH:
            low_weight, high_weight = 0.2, 0.8
        else:
            low_weight, high_weight = 0.5, 0.5

        # Add weighted results (low-level first for entity-centric queries)
        if low_weight >= high_weight:
            ordered = [(low_results, "Entity"), (high_results, "Relationship")]
        else:
            ordered = [(high_results, "Relationship"), (low_results, "Entity")]

        for result_list, label in ordered:
            for result in result_list:
                # Simple dedup by content
                result_key = result.strip().lower()[:100]
                if result_key not in seen:
                    seen.add(result_key)
                    merged.append(result)

        return merged

    @staticmethod
    def _format_vertex_result(element_map: Dict[str, Any]) -> str:
        """Format a vertex elementMap result as readable text."""
        label = element_map.get("label", "unknown")
        name = element_map.get("name", "")
        props = {k: v for k, v in element_map.items() if k not in ("id", "label", "~type", "~id", "name")}
        props_str = ", ".join(f"{k}: {v}" for k, v in props.items() if v)
        parts = [f"[{label}]", name]
        if props_str:
            parts.append(f"({props_str})")
        return " ".join(parts)

    @staticmethod
    def _format_edge_result(element_map: Dict[str, Any]) -> str:
        """Format an edge elementMap result as readable text."""
        label = element_map.get("label", "unknown")
        props = {k: v for k, v in element_map.items() if k not in ("id", "label", "~type", "~id")}
        props_str = ", ".join(f"{k}: {v}" for k, v in props.items() if v)
        if props_str:
            return f"--[{label}: {props_str}]-->"
        return f"--[{label}]-->"
