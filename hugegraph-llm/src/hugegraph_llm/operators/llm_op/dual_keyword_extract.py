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

"""Dual-level keyword extraction: high-level (hl) and low-level (ll) keywords.

Borrowed from LightRAG's keywords_extraction prompt
(lightrag/lightrag/prompt.py lines 484-515, operate.py lines 4001-4241).

Core idea: Decompose a user query into two keyword types:
- **hl_keywords** (high-level): Overarching concepts, themes, subject areas,
  question types. Used for searching relation/edge vector DB (global search).
- **ll_keywords** (low-level): Specific entities, proper nouns, technical
  terminology, product names, concrete items. Used for searching entity
  vector DB (local search).

This replaces our previous single-layer KeywordExtract with a more nuanced
decomposition that enables dual-path retrieval (local + global).

Design references:
    - LightRAG: prompt.py:484-515 (keywords_extraction prompt template)
    - LightRAG: operate.py:4001-4241 (extract_keywords_only + _parse_keywords_payload)
    - LightRAG: operate.py:4315-4515 (_perform_kg_search with hl/ll dispatch)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hugegraph_llm.utils.log import log


# ── Constants ─────────────────────────────────────────────────────

# LightRAG-style prompt template for dual keyword extraction
DUAL_KEYWORD_EXTRACT_PROMPT = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system powered by a knowledge graph.

---Goal---
Given a user query, extract two distinct types of keywords:
1. **high_level_keywords**: Overarching concepts or themes, capturing the user's core intent, the subject area, or the type of question being asked. These represent abstract, thematic categories.
2. **low_level_keywords**: Specific entities or details, identifying concrete items such as proper nouns, technical jargon, product names, specific people, places, or things mentioned in the query.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else.
2. **Exact JSON Shape**: The JSON object must contain exactly these two keys:
   - `"high_level_keywords"`: an array of strings representing overarching themes
   - `"low_level_keywords"`: an array of strings representing specific entities
3. **JSON Boundary**: The first character of your response must be `{{` and the last character must be `}}`.
4. **Source of Truth**: All keywords must be explicitly derived only from the `User Query`. Do not invent keywords not present in or implied by the query.
5. **Concise & Meaningful**: Prioritize multi-word phrases when they represent a single concept (e.g., "machine learning" over just "machine" and "learning" separately).
6. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical, return empty arrays for both types.
7. **No Duplicates**: Do not repeat the same keyword within a list.
8. **Language**: All extracted keywords MUST be in {language}.
9. **Balance**: Aim for roughly equal numbers of high-level and low-level keywords when possible.

---Output Format Template---
{{
  "high_level_keywords": ["<high_level_keyword_1>", "<high_level_keyword_2>"],
  "low_level_keywords": ["<low_level_keyword_1>", "<low_level_keyword_2>"]
}}

---Real Data---
User Query: {query}

---Output---
"""

# Heuristic fallback prompt (no LLM needed) — used when LLM is unavailable
# Extracts keywords using simple NLP rules
HEURISTIC_STOP_WORDS = {
    "en": set([
        "what", "who", "how", "why", "when", "where", "which", "is", "are",
        "was", "were", "the", "a", "an", "of", "in", "to", "for", "with",
        "on", "at", "by", "from", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no", "not",
        "only", "own", "same", "so", "than", "too", "very", "can", "will",
        "just", "should", "now", "also", "and", "but", "or", "if", "that",
        "this", "these", "those", "it", "its", "they", "them", "their",
        "we", "you", "he", "she", "me", "my", "your", "his", "her",
        "does", "did", "do", "has", "have", "had", "been", "being",
        "would", "could", "may", "might", "shall", "must", "need",
        "describe", "explain", "tell", "list", "name", "give", "find",
        "show", "provide", "compare", "discuss", "analyze", "evaluate",
    ]),
    "zh": set([
        "什么", "谁", "怎么", "如何", "为什么", "什么时候", "哪里", "哪个",
        "是", "有", "在", "到", "对", "为", "与", "从", "关于", "因为",
        "所以", "如果", "那么", "的", "了", "吗", "呢", "啊", "吧",
        "请", "能", "会", "要", "可以", "应该", "需要", "描述", "解释",
        "告诉", "列出", "给出", "找到", "显示", "提供", "比较", "讨论",
    ]),
}


@dataclass
class DualKeywords:
    """Container for dual-level keywords."""
    hl_keywords: List[str] = field(default_factory=list)  # High-level: themes, concepts
    ll_keywords: List[str] = field(default_factory=list)  # Low-level: entities, specifics
    raw_llm_output: str = ""                               # Raw LLM response (for debugging)
    extraction_method: str = "llm"                         # "llm" or "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hl_keywords": self.hl_keywords,
            "ll_keywords": self.ll_keywords,
            "extraction_method": self.extraction_method,
        }

    @property
    def has_keywords(self) -> bool:
        return len(self.hl_keywords) > 0 or len(self.ll_keywords) > 0

    @property
    def hl_str(self) -> str:
        """Join hl_keywords as single string for VDB search."""
        return " ".join(self.hl_keywords)

    @property
    def ll_str(self) -> str:
        """Join ll_keywords as single string for VDB search."""
        return " ".join(self.ll_keywords)


@dataclass
class DualKeywordConfig:
    """Configuration for dual keyword extraction."""
    max_keywords_per_level: int = 5       # Max keywords per level
    min_keyword_length: int = 2           # Minimum keyword length
    language: str = "en"                  # Language for prompt
    llm_max_retries: int = 2              # Max retries for LLM call
    fallback_to_heuristic: bool = True    # Fall back to heuristic if LLM fails
    # Short query (<50 chars) fallback: use entire query as ll_keywords
    short_query_threshold: int = 50


class DualKeywordExtract:
    """Extract high-level and low-level keywords from a query.

    Two extraction modes:
    1. **LLM mode**: Uses an LLM to parse the query into hl/ll keywords
       (LightRAG-style, with JSON response format).
    2. **Heuristic mode**: Uses simple NLP rules when LLM is unavailable.

    Usage::

        extractor = DualKeywordExtract(llm=my_llm, config=DualKeywordConfig())
        keywords = extractor.extract("What is the treatment for type 2 diabetes?")
        # keywords.hl_keywords = ["treatment", "disease management"]
        # keywords.ll_keywords = ["type 2 diabetes", "diabetes treatment"]

        # Heuristic fallback (no LLM needed):
        extractor = DualKeywordExtract(llm=None)
        keywords = extractor.extract("What is the treatment for diabetes?")
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[DualKeywordConfig] = None,
        extract_template: Optional[str] = None,
    ) -> None:
        """Initialize DualKeywordExtract.

        Args:
            llm: LLM instance (BaseLLM) for keyword extraction.
            config: DualKeywordConfig with extraction parameters.
            extract_template: Custom prompt template (overrides default).
        """
        self._llm = llm
        self.config = config or DualKeywordConfig()
        self._extract_template = extract_template or DUAL_KEYWORD_EXTRACT_PROMPT

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Operator protocol: extract dual keywords from context query.

        Reads from context:
            query: User question string.

        Writes to context:
            hl_keywords: List of high-level keywords.
            ll_keywords: List of low-level keywords.
            dual_keywords: DualKeywords object.
        """
        query = context.get("query", "")
        if not query:
            log.warning("[DualKeyword] No query in context")
            context["hl_keywords"] = []
            context["ll_keywords"] = []
            context["dual_keywords"] = DualKeywords()
            return context

        keywords = self.extract(query)
        context["hl_keywords"] = keywords.hl_keywords
        context["ll_keywords"] = keywords.ll_keywords
        context["dual_keywords"] = keywords
        return context

    def extract(self, query: str) -> DualKeywords:
        """Extract dual-level keywords from a query.

        Args:
            query: User question string.

        Returns:
            DualKeywords with hl_keywords and ll_keywords.
        """
        if not query:
            return DualKeywords(extraction_method="empty")

        # Short query fallback (<50 chars): entire query as ll_keywords
        if len(query) < self.config.short_query_threshold and not self._llm:
            log.info(f"[DualKeyword] Short query fallback: '{query}' → ll_keywords")
            return DualKeywords(
                hl_keywords=[],
                ll_keywords=[query],
                extraction_method="short_query_fallback",
            )

        # Try LLM extraction first
        if self._llm:
            for attempt in range(self.config.llm_max_retries):
                try:
                    keywords = self._extract_via_llm(query)
                    if keywords.has_keywords:
                        return keywords
                    log.warning(f"[DualKeyword] LLM returned empty keywords (attempt {attempt+1})")
                except Exception as e:
                    log.warning(f"[DualKeyword] LLM extraction failed (attempt {attempt+1}): {e}")

            # LLM failed after retries
            if self.config.fallback_to_heuristic:
                log.info("[DualKeyword] Falling back to heuristic extraction")
                return self._extract_via_heuristic(query)
            return DualKeywords(extraction_method="llm_failed")

        # No LLM available
        if self.config.fallback_to_heuristic:
            return self._extract_via_heuristic(query)

        return DualKeywords(extraction_method="no_method")

    def _extract_via_llm(self, query: str) -> DualKeywords:
        """Extract keywords using LLM with LightRAG-style prompt."""
        prompt = self._extract_template.format(
            query=query,
            language=self.config.language,
        )

        # Call LLM with JSON response format
        response = self._llm.generate(prompt)

        # Parse JSON response
        hl_keywords, ll_keywords = self._parse_llm_response(response)

        # Normalize and filter
        hl_keywords = self._normalize_keywords(hl_keywords)
        ll_keywords = self._normalize_keywords(ll_keywords)

        return DualKeywords(
            hl_keywords=hl_keywords[:self.config.max_keywords_per_level],
            ll_keywords=ll_keywords[:self.config.max_keywords_per_level],
            raw_llm_output=response,
            extraction_method="llm",
        )

    def _extract_via_heuristic(self, query: str) -> DualKeywords:
        """Extract keywords using simple NLP rules (no LLM needed).

        Heuristic approach:
        1. Split query into tokens
        2. Filter stop words and short tokens
        3. Classify remaining tokens:
           - Proper nouns/capitalized → ll_keywords
           - Common nouns/verbs → hl_keywords
           - Multi-word phrases → check if specific or abstract
        """
        stop_words = HEURISTIC_STOP_WORDS.get(self.config.language, HEURISTIC_STOP_WORDS["en"])

        # Tokenize
        if self.config.language == "zh":
            # Simple Chinese tokenization: character-level for short words
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', query)
        else:
            # English: word-level
            tokens = re.findall(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)*', query.lower())

        # Extract multi-word phrases (2-3 word combinations)
        phrases = self._extract_phrases(query, stop_words)

        # Single tokens: filter stop words and short words
        filtered_tokens = [
            t for t in tokens
            if t.lower() not in stop_words and len(t) >= self.config.min_keyword_length
        ]

        # Classify: proper nouns → ll, common words → hl
        hl_keywords: List[str] = []
        ll_keywords: List[str] = []

        # Check original query for capitalized words (likely proper nouns/entities)
        original_tokens = query.split()
        capitalized = set()
        for tok in original_tokens:
            # A token starting with uppercase (not first word of sentence) is likely a proper noun
            if tok[0].isupper() and tok.lower() not in stop_words and len(tok) >= 2:
                capitalized.add(tok.lower())

        for phrase in phrases:
            # If phrase contains a capitalized word → ll (specific entity)
            phrase_lower = phrase.lower()
            has_capitalized = any(w in capitalized for w in phrase_lower.split())
            if has_capitalized:
                ll_keywords.append(phrase)
            else:
                # Abstract concept → hl
                hl_keywords.append(phrase)

        # Remaining single filtered tokens
        remaining = [t for t in filtered_tokens if t not in set(
            w for p in phrases for w in p.lower().split()
        )]
        for t in remaining:
            if t in capitalized:
                ll_keywords.append(t)
            else:
                hl_keywords.append(t)

        # Deduplicate
        hl_keywords = list(set(hl_keywords))
        ll_keywords = list(set(ll_keywords))

        return DualKeywords(
            hl_keywords=hl_keywords[:self.config.max_keywords_per_level],
            ll_keywords=ll_keywords[:self.config.max_keywords_per_level],
            extraction_method="heuristic",
        )

    def _extract_phrases(self, query: str, stop_words: set) -> List[str]:
        """Extract meaningful multi-word phrases from query."""
        phrases: List[str] = []

        # Simple n-gram extraction (2-3 word phrases)
        words = re.findall(r'[a-zA-Z]+', query)
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                phrase_words = words[i:i + n]
                # Skip phrases where all words are stop words
                if all(w.lower() in stop_words for w in phrase_words):
                    continue
                # Skip phrases starting with stop words
                if phrase_words[0].lower() in stop_words:
                    continue
                phrase = " ".join(phrase_words)
                if len(phrase) >= 4:  # Minimum phrase length
                    phrases.append(phrase.lower())

        return phrases

    @staticmethod
    def _parse_llm_response(response: str) -> Tuple[List[str], List[str]]:
        """Parse LLM JSON response into hl/ll keywords.

        Handles multiple response formats:
        - Clean JSON object
        - Markdown-fenced JSON (```json ... ```)
        - Pydantic model_dump output
        - Malformed JSON (with json_repair fallback)

        Returns:
            (hl_keywords, ll_keywords) tuple.
        """
        # Strip markdown fences
        text = response.strip()
        if text.startswith("```"):
            # Remove first and last fence lines
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try standard JSON parse
        try:
            payload = json.loads(text)
            hl = DualKeywordExtract._normalize_keyword_list(payload.get("high_level_keywords"))
            ll = DualKeywordExtract._normalize_keyword_list(payload.get("low_level_keywords"))
            return hl, ll
        except json.JSONDecodeError:
            pass

        # Try json_repair if available
        try:
            import json_repair
            payload = json_repair.loads(text)
            hl = DualKeywordExtract._normalize_keyword_list(payload.get("high_level_keywords"))
            ll = DualKeywordExtract._normalize_keyword_list(payload.get("low_level_keywords"))
            return hl, ll
        except ImportError:
            pass
        except Exception:
            pass

        # Last resort: regex extraction
        hl_match = re.findall(
            r'"high_level_keywords"\s*:\s*\[([^\]]*)\]', text
        )
        ll_match = re.findall(
            r'"low_level_keywords"\s*:\s*\[([^\]]*)\]', text
        )

        hl = []
        ll = []
        if hl_match:
            hl = [k.strip().strip('"').strip("'") for k in hl_match[0].split(",") if k.strip()]
        if ll_match:
            ll = [k.strip().strip('"').strip("'") for k in ll_match[0].split(",") if k.strip()]

        return hl, ll

    @staticmethod
    def _normalize_keyword_list(raw: Any) -> List[str]:
        """Normalize a keyword list from LLM output.

        Handles None, string, list, and other types.
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            # Split by common delimiters
            keywords = re.split(r'[,;\n]', raw)
            return [k.strip() for k in keywords if k.strip()]
        if isinstance(raw, list):
            result = []
            for item in raw:
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        result.append(item)
                elif isinstance(item, dict):
                    # Some LLMs return {keyword: "..."} format
                    kw = item.get("keyword", item.get("name", str(item)))
                    if kw:
                        result.append(str(kw).strip())
            return result
        return []

    @staticmethod
    def _normalize_keywords(keywords: List[str]) -> List[str]:
        """Filter and normalize extracted keywords."""
        result = []
        seen = set()
        for kw in keywords:
            kw = kw.strip().lower()
            if len(kw) < 2 or kw in seen:
                continue
            seen.add(kw)
            result.append(kw)
        return result


# ── Convenience function ──────────────────────────────────────────


def extract_dual_keywords(
    query: str,
    llm: Optional[Any] = None,
    language: str = "en",
    max_keywords: int = 5,
) -> DualKeywords:
    """Quick-extract dual-level keywords from a query.

    Args:
        query: User question string.
        llm: Optional LLM instance.
        language: Language code ("en" or "zh").
        max_keywords: Max keywords per level.

    Returns:
        DualKeywords.
    """
    config = DualKeywordConfig(
        max_keywords_per_level=max_keywords,
        language=language,
    )
    extractor = DualKeywordExtract(llm=llm, config=config)
    return extractor.extract(query)
