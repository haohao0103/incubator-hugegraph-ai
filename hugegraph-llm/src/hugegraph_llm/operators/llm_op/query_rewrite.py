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

"""Query Rewrite: decompose complex questions into sub-queries for multi-hop KG retrieval.

Inspired by RAGFlow v0.26.0 KGSearchRetrieval (QueryRewrite prompt builder + response parser)
and Microsoft GraphRAG's multi-step reasoning patterns. The goal is to break a complex user
query into a sequence of simpler, graph-answerable sub-queries that can be executed
against a knowledge graph via entity/relation traversal.

Design references:
    - RAGFlow v0.26.0: QueryRewrite prompt builder + response parser for KGSearchRetrieval
    - MS-GraphRAG: structured local/global search decomposition patterns
    - LightRAG: dual keyword extraction (hl/ll) feeding local/global search modes
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from hugegraph_llm.utils.log import log


logger = logging.getLogger(__name__)


QUERY_REWRITE_PROMPT = """---Role---
You are a query decomposition expert for a Knowledge-Graph-powered Retrieval-Augmented Generation (RAG) system.

---Goal---
Given a complex user question, decompose it into a small set of simpler sub-questions that can be answered by traversing a knowledge graph (entities and relationships). Each sub-question should target a specific fact or relationship needed to answer the original question.

---Instructions---
1. **Analyze the question**: Identify the entities, relationships, and reasoning steps required.
2. **Decompose**: Break the complex question into 1 to 4 ordered sub-questions. Each sub-question should be simple, self-contained, and focused on a single piece of information.
3. **Output format**: Return ONLY a valid JSON object. The first character must be `{{` and the last character must be `}}`.
4. **Required JSON structure**:
{{
  "needs_rewrite": true,
  "sub_queries": [
    "<sub_query_1>",
    "<sub_query_2>",
    ...
  ],
  "reasoning": "<brief explanation of why this decomposition was chosen>"
}}
5. If the query is already simple and directly answerable from a single graph lookup, set `needs_rewrite` to false and return an empty `sub_queries` list.
6. Sub-queries must be in the same language as the user question.

---Example 1---
User Question: "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
Output:
{{
  "needs_rewrite": true,
  "sub_queries": [
    "Who is the CEO of Apple?",
    "Which company designed the A17 chip?",
    "What is the relationship between the CEO of Apple and the designer of the A17 chip?"
  ],
  "reasoning": "The question requires identifying two entities (Apple CEO, A17 chip designer) and then their relationship."
}}

---Example 2---
User Question: "What is the capital of France?"
Output:
{{
  "needs_rewrite": false,
  "sub_queries": [],
  "reasoning": "Simple factual query; can be answered by a single entity lookup."
}}

---Real Data---
User Question: {query}

---Output---
"""


@dataclass
class QueryRewriteResult:
    """Result of query rewrite."""

    original_query: str = ""
    needs_rewrite: bool = False
    sub_queries: List[str] = field(default_factory=list)
    reasoning: str = ""
    raw_llm_output: str = ""
    extraction_method: str = "llm"  # "llm" or "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "needs_rewrite": self.needs_rewrite,
            "sub_queries": self.sub_queries,
            "reasoning": self.reasoning,
            "extraction_method": self.extraction_method,
        }

    @property
    def executable_queries(self) -> List[str]:
        """Return all queries that should be executed against the graph.

        If rewrite was not needed, the original query is the only executable query.
        """
        if not self.needs_rewrite or not self.sub_queries:
            return [self.original_query] if self.original_query else []
        return self.sub_queries


@dataclass
class QueryRewriteConfig:
    """Configuration for query rewriting."""

    max_sub_queries: int = 4
    fallback_to_heuristic: bool = True
    llm_max_retries: int = 2
    # Heuristic thresholds
    simple_query_threshold: int = 50  # chars
    conjunction_markers: Sequence[str] = field(
        default_factory=lambda: (
            " and ", " or ", " between ", " compare ", " difference ",
            " relationship ", " related to ", " connected to ", " impact of ",
            " 和 ", " 与 ", " 关系 ", " 比较 ", " 区别 ",
        )
    )


class QueryRewrite:
    """Rewrite a complex query into sub-queries for multi-hop KG retrieval.

    Two extraction modes:
    1. **LLM mode**: Uses an LLM to decompose the query (RAGFlow-style).
    2. **Heuristic mode**: Uses simple rules when LLM is unavailable.

    Usage::

        rewriter = QueryRewrite(llm=my_llm)
        result = rewriter.run({"query": "What is the relationship between X and Y?"})
        # result["sub_queries"] = ["Who is X?", "Who is Y?", "What is the relationship between X and Y?"]
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[QueryRewriteConfig] = None,
        rewrite_template: Optional[str] = None,
    ) -> None:
        """Initialize QueryRewrite.

        Args:
            llm: LLM instance (BaseLLM) for query decomposition.
            config: QueryRewriteConfig with rewrite parameters.
            rewrite_template: Custom prompt template (overrides default).
        """
        self._llm = llm
        self.config = config or QueryRewriteConfig()
        self._rewrite_template = rewrite_template or QUERY_REWRITE_PROMPT

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Operator protocol: rewrite query in context.

        Reads from context:
            query: User question string.

        Writes to context:
            query_rewrite: QueryRewriteResult dataclass.
        """
        query = context.get("query", "")
        if not query:
            context["query_rewrite"] = QueryRewriteResult()
            return context

        result = self.extract(query)
        context["query_rewrite"] = result
        return context

    def extract(self, query: str) -> QueryRewriteResult:
        """Rewrite a single query.

        Args:
            query: User question string.

        Returns:
            QueryRewriteResult with original query, sub-queries, and metadata.
        """
        if not query or not query.strip():
            return QueryRewriteResult()

        # Heuristic short-circuit: very simple queries need no rewrite
        if self._is_simple_query(query):
            return QueryRewriteResult(
                original_query=query,
                needs_rewrite=False,
                sub_queries=[],
                reasoning="Heuristic: query is simple and directly answerable.",
                extraction_method="heuristic",
            )

        if self._llm is None:
            if self.config.fallback_to_heuristic:
                return self._heuristic_rewrite(query)
            return QueryRewriteResult(
                original_query=query,
                needs_rewrite=False,
                reasoning="No LLM available and heuristic fallback disabled.",
                extraction_method="heuristic",
            )

        return self._llm_rewrite(query)

    def _is_simple_query(self, query: str) -> bool:
        """Heuristic: queries shorter than threshold and without conjunction markers are simple."""
        if len(query) < self.config.simple_query_threshold:
            return not any(marker in query.lower() for marker in self.config.conjunction_markers)
        return False

    def _llm_rewrite(self, query: str) -> QueryRewriteResult:
        """Use LLM to decompose query."""
        prompt = self._rewrite_template.format(query=query)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.llm_max_retries + 1):
            try:
                raw_output = self._call_llm(prompt)
                parsed = self._parse_json(raw_output)
                return self._build_result(query, parsed, raw_output, "llm")
            except Exception as e:  # pylint: disable=broad-except
                last_error = e
                logger.warning("Query rewrite LLM attempt %d failed: %s", attempt, e)

        logger.error("Query rewrite failed after %d attempts: %s", self.config.llm_max_retries, last_error)
        if self.config.fallback_to_heuristic:
            return self._heuristic_rewrite(query)
        return QueryRewriteResult(
            original_query=query,
            needs_rewrite=False,
            reasoning=f"LLM rewrite failed: {last_error}",
            raw_llm_output="",
            extraction_method="heuristic",
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self._llm is None:
            raise ValueError("LLM is not configured")

        # Support both sync and async LLM interfaces
        if hasattr(self._llm, "generate"):
            return self._llm.generate(prompt)
        if hasattr(self._llm, "chat"):
            return self._llm.chat(prompt)
        if hasattr(self._llm, "completion"):
            return self._llm.completion(prompt)
        raise ValueError(f"Unsupported LLM interface: {type(self._llm)}")

    def _parse_json(self, raw_output: str) -> Dict[str, Any]:
        """Extract and parse JSON object from LLM output."""
        # Strip markdown code fences if present
        text = raw_output.strip()
        if text.startswith("```"):
            text = text.strip().removeprefix("```json").removeprefix("```")
            text = text.removesuffix("```").strip()

        # Find outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No valid JSON object found in LLM output: {raw_output[:200]}")

        json_str = text[start : end + 1]
        return json.loads(json_str)

    def _build_result(
        self,
        query: str,
        parsed: Dict[str, Any],
        raw_output: str,
        method: str,
    ) -> QueryRewriteResult:
        """Build QueryRewriteResult from parsed JSON."""
        needs_rewrite = bool(parsed.get("needs_rewrite", False))
        sub_queries = parsed.get("sub_queries", [])
        if not isinstance(sub_queries, list):
            raise ValueError(f"sub_queries must be a list, got {type(sub_queries)}")

        # Filter and deduplicate sub-queries
        cleaned: List[str] = []
        seen = set()
        for sq in sub_queries:
            if not isinstance(sq, str):
                continue
            sq = sq.strip()
            if sq and sq.lower() not in seen and len(cleaned) < self.config.max_sub_queries:
                cleaned.append(sq)
                seen.add(sq.lower())

        # If LLM says rewrite but produced no sub-queries, treat as no rewrite
        if needs_rewrite and not cleaned:
            needs_rewrite = False

        return QueryRewriteResult(
            original_query=query,
            needs_rewrite=needs_rewrite,
            sub_queries=cleaned,
            reasoning=parsed.get("reasoning", ""),
            raw_llm_output=raw_output,
            extraction_method=method,
        )

    def _heuristic_rewrite(self, query: str) -> QueryRewriteResult:
        """Heuristic fallback when LLM is unavailable or fails.

        Splits query by conjunction markers and returns each clause as a sub-query.
        If no split occurs, returns no rewrite.
        """
        query_lower = query.lower()
        for marker in self.config.conjunction_markers:
            if marker in query_lower:
                parts = [p.strip() for p in re.split(re.escape(marker), query_lower, flags=re.IGNORECASE)]
                parts = [p for p in parts if p]
                if len(parts) > 1:
                    # Convert each part into a question-like sub-query if needed
                    sub_queries = []
                    for part in parts:
                        # Remove trailing punctuation
                        part = re.sub(r"[?.,;:!]+$", "", part).strip()
                        if part:
                            sub_queries.append(part.capitalize())
                    return QueryRewriteResult(
                        original_query=query,
                        needs_rewrite=True,
                        sub_queries=sub_queries[: self.config.max_sub_queries],
                        reasoning=f"Heuristic split by marker '{marker}'.",
                        extraction_method="heuristic",
                    )

        return QueryRewriteResult(
            original_query=query,
            needs_rewrite=False,
            sub_queries=[],
            reasoning="Heuristic: no conjunction markers found; query treated as simple.",
            extraction_method="heuristic",
        )
