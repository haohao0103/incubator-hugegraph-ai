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
LLM-powered Query Rewrite (PowerMem QueryRewriter + mem0 entity boost aligned).

Extends the existing rule-based QueryRewriteEngine with:
  - LLM query understanding: reformulate ambiguous queries, expand acronyms,
    resolve coreferences using conversation context.
  - Entity-aware boost: extract query entities and inject them as
    retrieval-side boosts (aligned with mem0 entity_boost scoring).
  - Conversation context injection: carry prior turns for multi-turn
    pronoun resolution.
  - Fallback: when no LLM key is available, the rule-based engine still works.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.engines.memory.query_rewrite import QueryRewriteEngine
from hugegraph_llm.engines.memory.hybrid_scoring import extract_query_entities_simple, compute_entity_boosts
from hugegraph_llm.utils.log import log


# LLM prompt for query understanding
LLM_QUERY_REWRITE_PROMPT = """You are a query understanding assistant for a memory retrieval system.
Given the original query, conversation context, and user profile, produce a JSON object with:

1. "rewritten": a clear, unambiguous reformulation of the query suitable for semantic search
2. "entities": a list of named entities mentioned or implied (each with "name" and "type")
3. "intent": the retrieval intent — one of "fact_lookup", "relationship_query", "temporal_query", "preference_query", "general"

Rules:
- Resolve pronouns (他/她/它/they) to the actual entity name if context provides it
- Expand abbreviations (e.g. "HQ" -> "总部", "PM" -> "项目经理")
- Split compound queries into sub-queries if needed
- Preserve original meaning — do NOT fabricate information not in the query/context
- If the query is already clear, return it unchanged as "rewritten"

Original query: {query}
Conversation context: {context}
User profile: {profile}

Output JSON only, no explanation."""


class LLMQueryRewriteEngine:
    """LLM-enhanced query rewrite with rule-based fallback.

    Args:
        llm_callback: Function that takes a prompt string and returns LLM response text.
                       If None, falls back to pure rule-based QueryRewriteEngine.
        aliases: Alias -> canonical entity mapping (same as QueryRewriteEngine).
        user_profile: Static user profile for pronoun resolution.
        use_llm: Whether to call LLM. Defaults to True if llm_callback is provided.
        model: LLM model name for direct OpenAI API calls (alternative to llm_callback).
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        user_profile: Optional[str] = None,
        use_llm: bool = True,
        model: Optional[str] = None,
    ):
        self.llm_callback = llm_callback
        self.aliases = aliases or {}
        self.user_profile = user_profile or ""
        self.use_llm = use_llm and (llm_callback is not None or memory_settings.llm_api_key)
        self.model = model or memory_settings.llm_model

        # Rule-based fallback engine
        self._rule_engine = QueryRewriteEngine(
            aliases=self.aliases,
            user_profile=self.user_profile,
        )

        # OpenAI client for direct calls (if no custom callback)
        self._openai_client = None
        if self.use_llm and llm_callback is None and memory_settings.llm_api_key:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=memory_settings.llm_api_key,
                base_url=memory_settings.llm_base_url,
            )

    def rewrite(
        self,
        query: str,
        context: Optional[str] = None,
        user_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rewrite a query using LLM (if available) or rule-based fallback.

        Args:
            query: Original user query.
            context: Recent conversation turns for pronoun resolution.
            user_profile: Override user profile for this query.

        Returns:
            Dict with keys:
              - original: original query
              - rewritten: best rewritten query
              - entities: list of {name, type} extracted entities
              - intent: classified retrieval intent
              - boosts: entity boost scores for retrieval
              - variants: list of retrieval variant queries
              - method: "llm" or "rule"
        """
        profile = user_profile or self.user_profile
        if self.use_llm:
            try:
                result = self._llm_rewrite(query, context=context, profile=profile)
                result["method"] = "llm"
                return result
            except Exception as e:
                log.warning("LLM query rewrite failed: %s; falling back to rule-based", e)

        # Rule-based fallback
        result = self._rule_rewrite(query, profile=profile)
        result["method"] = "rule"
        return result

    def _llm_rewrite(
        self,
        query: str,
        context: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call LLM for query understanding."""
        prompt = LLM_QUERY_REWRITE_PROMPT.format(
            query=query,
            context=context or "none",
            profile=profile or "unknown user",
        )

        if self.llm_callback:
            raw = self.llm_callback(prompt)
        elif self._openai_client:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            if not response.choices:
                raw = ""
            else:
                raw = (response.choices[0].message.content or "").strip()
        else:
            raise RuntimeError("No LLM callback or OpenAI client available")

        # Parse LLM response
        parsed = self._parse_llm_response(raw)

        # Merge rule-based variants for diversity
        rule_result = self._rule_engine.expand_query(query)
        variants = parsed.get("variants", [])
        for v in rule_result.get("variants", []):
            if v.lower() not in {x.lower() for x in variants}:
                variants.append(v)

        # Compute entity boosts (requires memory_entities for full scoring,
        # but for query-only boost we use entity names as keys with default weight)
        entities = parsed.get("entities", [])
        entity_names = [e.get("name", "") if isinstance(e, dict) else str(e) for e in entities if (e.get("name", "") if isinstance(e, dict) else e)]
        boosts = compute_entity_boosts(entity_names, {})

        return {
            "original": query,
            "rewritten": parsed.get("rewritten", query),
            "entities": entities,
            "intent": parsed.get("intent", "general"),
            "boosts": boosts,
            "variants": variants or [parsed.get("rewritten", query)],
        }

    def _rule_rewrite(
        self,
        query: str,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rule-based rewrite fallback."""
        engine = self._rule_engine
        if profile and profile != self.user_profile:
            engine = QueryRewriteEngine(aliases=self.aliases, user_profile=profile)

        expansion = engine.expand_query(query)

        # Extract entities using rule-based extractor
        entities_raw = extract_query_entities_simple(query)
        # extract_query_entities_simple returns List[str], convert to list of dicts
        entities = [{"name": e, "type": "unknown"} if isinstance(e, str) else e for e in entities_raw]
        entity_names = [e.get("name", "") if isinstance(e, dict) else str(e) for e in entities if (e.get("name", "") if isinstance(e, dict) else e)]
        boosts = compute_entity_boosts(entity_names, {})

        # Classify intent heuristically
        intent = self._classify_intent_heuristic(query)

        return {
            "original": query,
            "rewritten": expansion.get("rewritten", query),
            "entities": entities,
            "intent": intent,
            "boosts": boosts,
            "variants": expansion.get("variants", [query]),
        }

    @staticmethod
    def _classify_intent_heuristic(query: str) -> str:
        """Simple heuristic intent classification."""
        q = query.lower()
        if any(kw in q for kw in ["什么时候", "何时", "哪天", "when", "日期", "时间", "年", "月"]):
            return "temporal_query"
        if any(kw in q for kw in ["谁", "谁的", "同事", "朋友", "关系", "who", "relation", "认识"]):
            return "relationship_query"
        if any(kw in q for kw in ["喜欢", "讨厌", "擅长", "偏好", "prefer", "like", "hate", "兴趣"]):
            return "preference_query"
        if any(kw in q for kw in ["是什么", "什么是", "定义", "what is", "who is", "哪个"]):
            return "fact_lookup"
        return "general"

    @staticmethod
    def _parse_llm_response(raw: str) -> Dict[str, Any]:
        """Parse LLM JSON response, with fallback for malformed output."""
        # Try direct JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1).strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        # Try to find first { ... } block
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        # Last resort: extract rewritten query from plain text
        lines = raw.strip().split("\n")
        rewritten = lines[0] if lines else ""
        return {
            "rewritten": rewritten.strip() or "",
            "entities": [],
            "intent": "general",
            "variants": [rewritten.strip()] if rewritten.strip() else [],
        }


def llm_rewrite_query(
    query: str,
    context: Optional[str] = None,
    user_profile: Optional[str] = None,
    aliases: Optional[Dict[str, str]] = None,
    llm_callback: Optional[Callable[[str], str]] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Convenience factory for one-shot LLM query rewrite."""
    engine = LLMQueryRewriteEngine(
        llm_callback=llm_callback,
        aliases=aliases,
        user_profile=user_profile,
        use_llm=use_llm,
    )
    return engine.rewrite(query, context=context, user_profile=user_profile)
