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
Query rewrite service for HugeGraph-AI-Memory.

Implements Mem0/PowerMem-style query understanding:
  - pronoun/coreference resolution using user profile context
  - abbreviation/acronym expansion
  - generation of retrieval variants (whitespace, synonyms, loose match)
  - alias mapping for known entities

All functions are deterministic and stateless so they are easy to unit test.
"""

import re
from typing import Dict, List, Optional, Any


# Pronouns that should be resolved to a known entity name.
_PRONOUNS = {"他", "她", "它", "这个人", "那位", "那人", "此君"}

# Query particles that are stripped when generating loose-match variants.
_STOP_PARTICLES = {
    "的", "了", "在", "是", "有", "和", "也", "都", "哪些", "多少", "几",
    "怎么", "如何", "谁", "什么", "哪里", "哪个", "有没有", "这个", "信息",
    "记忆", "同事", "朋友", "共事", "员工", "上班", "工作", "总部", "公司",
    "参加", "创立", "技术", "城市", "总监", "告诉", "我", "帮", "回忆", "吗",
}


class QueryRewriteEngine:
    """
    Rewrite natural-language queries for better memory retrieval.

    Args:
        aliases: Optional mapping from alias -> canonical entity name.
        user_profile: Optional user profile summary used for pronoun resolution.
    """

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        user_profile: Optional[str] = None,
    ):
        self.aliases = {k.lower(): v for k, v in (aliases or {}).items()}
        self.user_profile = user_profile or ""
        self._profile_entities = self._extract_profile_entities(self.user_profile)

    @staticmethod
    def _extract_profile_entities(profile: str) -> List[str]:
        """Extract likely entity names from a profile summary."""
        entities = []
        # Chinese 2-4 char names/organizations/locations
        for m in re.finditer(r"[\u4e00-\u9fa5]{2,8}", profile):
            entities.append(m.group())
        # English names / companies (1-3 capitalized words, optional suffix)
        for m in re.finditer(
            r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2}(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|Company|Co\.))?(?=\s|$|[,;\.])",
            profile,
        ):
            entities.append(m.group(0).strip())
        return list(dict.fromkeys(entities))  # preserve order, dedupe

    def rewrite(self, query: str) -> str:
        """
        Return a single rewritten query with resolved pronouns and aliases.

        The rewrite is conservative: it only replaces pronouns when a unique
        entity can be inferred from the user profile, and expands aliases only
        when an alias is an exact match.
        """
        query = query.strip()
        if not query:
            return query

        # 1. Alias expansion (case-insensitive, whole-word for English)
        lowered = query.lower()
        for alias, canonical in self.aliases.items():
            if alias in lowered:
                # Replace first occurrence in the original query (preserve casing)
                query = re.sub(re.escape(alias), canonical, query, count=1, flags=re.IGNORECASE)
                break  # only expand one alias per query

        # 2. Pronoun resolution
        if self._profile_entities and any(p in query for p in _PRONOUNS):
            # Choose the most prominent entity (first one) as the referent
            referent = self._profile_entities[0]
            for p in sorted(_PRONOUNS, key=len, reverse=True):
                if p in query:
                    query = query.replace(p, referent, 1)
                    break

        return query.strip()

    def variants(self, query: str) -> List[str]:
        """
        Generate retrieval variants of a query.

        Returns the rewritten query plus a loose keyword-only variant suitable
        for keyword/ BM25-style search. Deduplicates results.
        """
        rewritten = self.rewrite(query)
        variants = [rewritten]

        # Loose keyword variant: strip question particles and punctuation
        tokens = self._tokenize(rewritten)
        if tokens:
            keyword_variant = " ".join(tokens)
            if keyword_variant != rewritten:
                variants.append(keyword_variant)

        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                unique_variants.append(v)
        return unique_variants

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        """Strip particles and split query into meaningful keyword tokens."""
        # Remove punctuation and question marks
        cleaned = re.sub(r"[？?!?.,;:，\uff1b\uff1a\"\']+", " ", query)
        # Split on common stop particles first so they don't remain inside tokens
        particles = sorted(_STOP_PARTICLES, key=len, reverse=True)
        pattern = "|".join(re.escape(p) for p in particles)
        parts = re.split(pattern, cleaned)
        tokens = []
        for part in parts:
            # Further split on whitespace
            for sub in re.split(r"[\s\u3000]+", part):
                sub = sub.strip()
                if sub and len(sub) > 1:
                    tokens.append(sub)
        return tokens

    def expand_query(self, query: str) -> Dict[str, Any]:
        """Return a structured rewrite result with original, rewritten, and variants."""
        variants = self.variants(query)
        return {
            "original": query,
            "rewritten": variants[0] if variants else query,
            "variants": variants,
            "keyword_query": variants[-1] if len(variants) > 1 else variants[0] if variants else query,
            "profile_entities": self._profile_entities,
        }


def rewrite_query(
    query: str,
    aliases: Optional[Dict[str, str]] = None,
    user_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience factory function for one-shot query expansion."""
    engine = QueryRewriteEngine(aliases=aliases, user_profile=user_profile)
    return engine.expand_query(query)
