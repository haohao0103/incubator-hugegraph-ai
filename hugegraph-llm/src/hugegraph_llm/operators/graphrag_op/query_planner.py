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
Simplified query planner for LightRAG-style GraphRAG.

Instead of the complex 7-intent, 4-strategy planner that depended on
community detection (Microsoft GraphRAG approach), this simplified version
maps queries to LightRAG's dual-level retrieval:

- LOW level (entity-centric): For specific fact questions
- HIGH level (relationship-centric): For abstract/broad questions
- HYBRID: Both levels combined

No community detection dependency. No global restructuring needed.
This is the pragmatic "Plan A" approach for quick production landing.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class QueryLevel(str, Enum):
    """Query retrieval level (maps to LightRAG dual-level retrieval)."""

    LOW = "low"  # Entity-centric: specific facts
    HIGH = "high"  # Relationship-centric: abstract questions
    HYBRID = "hybrid"  # Both levels combined


class QueryIntent(str, Enum):
    """Simplified query intent classification."""

    SPECIFIC_ENTITY = "specific_entity"  # "Who is X?" / "What is Y?"
    RELATIONSHIP = "relationship"  # "How are X and Y related?"
    ABSTRACT = "abstract"  # "What are the key themes?"
    UNKNOWN = "unknown"


# Intent classification patterns
INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
    QueryIntent.SPECIFIC_ENTITY: [
        "who is",
        "what is",
        "what does",
        "where is",
        "when did",
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
        "哪个",
    ],
    QueryIntent.RELATIONSHIP: [
        "how are",
        "related to",
        "connection between",
        "relationship",
        "compare",
        "difference",
        "vs",
        "versus",
        "如何关联",
        "关系",
        "联系",
        "之间的",
        "比较",
        "区别",
        "对比",
    ],
    QueryIntent.ABSTRACT: [
        "why",
        "cause",
        "reason",
        "trend",
        "overview",
        "summarize",
        "explain",
        "describe",
        "analysis",
        "为什么",
        "原因",
        "趋势",
        "概述",
        "总结",
        "说明",
        "分析",
    ],
}

# Intent → Level mapping
INTENT_LEVEL_MAP: Dict[QueryIntent, QueryLevel] = {
    QueryIntent.SPECIFIC_ENTITY: QueryLevel.LOW,
    QueryIntent.RELATIONSHIP: QueryLevel.HYBRID,
    QueryIntent.ABSTRACT: QueryLevel.HIGH,
    QueryIntent.UNKNOWN: QueryLevel.HYBRID,
}


class QueryPlanner:
    """
    Simplified query planner for LightRAG-style GraphRAG.

    Maps user queries to the appropriate retrieval level (LOW/HIGH/HYBRID)
    without depending on community detection.

    Design principles:
    - Simple > complex (2 retrieval levels vs 4 strategies)
    - No community dependency (enables incremental updates)
    - Fast classification (pattern-based, optional LLM refinement)
    - Production-ready (proven by Huolala team)
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        default_level: QueryLevel = QueryLevel.HYBRID,
        enable_llm_classification: bool = False,
    ):
        """
        Args:
            llm: Optional LLM for intent classification when patterns are ambiguous.
            default_level: Default retrieval level when intent is unclear.
            enable_llm_classification: Whether to use LLM for ambiguous queries.
        """
        self.llm = llm
        self.default_level = default_level
        self.enable_llm_classification = enable_llm_classification

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the query execution level.

        Args:
            context: Dict with 'query'.

        Returns:
            Updated context with 'query_plan', 'query_intent', 'retrieval_level'.
        """
        query = context.get("query", "")
        if not query:
            log.warning("No query provided for planning")
            context["query_plan"] = self._default_plan()
            context["query_intent"] = QueryIntent.UNKNOWN.value
            context["retrieval_level"] = self.default_level.value
            return context

        # Step 1: Classify intent
        intent = self._classify_intent(query)

        # Step 2: Map intent to retrieval level
        level = INTENT_LEVEL_MAP.get(intent, self.default_level)

        # Step 3: Generate execution plan
        plan = self._generate_plan(query, intent, level)

        context["query_plan"] = plan
        context["query_intent"] = intent.value
        context["retrieval_level"] = level.value
        log.info("Query plan: intent=%s, level=%s", intent.value, level.value)
        return context

    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Classify query intent using pattern matching.

        Optionally uses LLM for ambiguous cases when enabled.
        """
        query_lower = query.lower()

        # Pattern-based classification
        best_intent = QueryIntent.UNKNOWN
        best_score = 0

        for intent, patterns in INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent

        # If pattern matching is ambiguous, try LLM
        if best_score <= 1 and self.llm and self.enable_llm_classification:
            llm_intent = self._classify_intent_by_llm(query)
            if llm_intent != QueryIntent.UNKNOWN:
                return llm_intent

        return best_intent if best_intent != QueryIntent.UNKNOWN else QueryIntent.SPECIFIC_ENTITY

    def _classify_intent_by_llm(self, query: str) -> QueryIntent:
        """Use LLM to classify query intent when patterns are ambiguous."""
        if not self.llm:
            return QueryIntent.UNKNOWN

        prompt = f"""Classify the following question into exactly one category:
- specific_entity: Questions about a specific entity (who, what, where, when, define)
- relationship: Questions about relationships between entities (how related, compare, difference)
- abstract: Questions about causes, reasons, trends, overviews, summaries (why, trend, summarize)

Question: {query}

Category:"""

        try:
            response = self.llm.generate(prompt=prompt).strip().lower()
            for intent in QueryIntent:
                if intent.value in response:
                    return intent
        except Exception as e:  # pylint: disable=broad-except
            log.warning("LLM intent classification failed: %s", e)

        return QueryIntent.UNKNOWN

    def _generate_plan(self, query: str, intent: QueryIntent, level: QueryLevel) -> Dict[str, Any]:
        """Generate execution plan for the chosen retrieval level."""
        plan = {
            "query": query,
            "intent": intent.value,
            "retrieval_level": level.value,
            "steps": [],
            "parameters": {},
        }

        if level == QueryLevel.LOW:
            plan["steps"] = [
                "extract_keywords",
                "semantic_id_query",
                "low_level_entity_retrieval",
                "merge_rerank",
                "answer_synthesize",
            ]
            plan["parameters"] = {
                "max_depth": 2,
                "max_neighbors": 20,
            }

        elif level == QueryLevel.HIGH:
            plan["steps"] = [
                "extract_keywords",
                "high_level_relationship_retrieval",
                "merge_rerank",
                "answer_synthesize",
            ]
            plan["parameters"] = {
                "max_paths": 10,
                "max_hops": 3,
            }

        elif level == QueryLevel.HYBRID:
            plan["steps"] = [
                "extract_keywords",
                "semantic_id_query",
                "dual_level_retrieval",
                "merge_rerank",
                "answer_synthesize",
            ]
            plan["parameters"] = {
                "low_max_depth": 2,
                "low_max_neighbors": 20,
                "high_max_paths": 10,
                "high_max_hops": 3,
            }

        return plan

    def _default_plan(self) -> Dict[str, Any]:
        """Return a default query plan."""
        return {
            "query": "",
            "intent": QueryIntent.UNKNOWN.value,
            "retrieval_level": self.default_level.value,
            "steps": [
                "extract_keywords",
                "dual_level_retrieval",
                "merge_rerank",
                "answer_synthesize",
            ],
            "parameters": {
                "low_max_depth": 2,
                "high_max_hops": 3,
            },
        }
