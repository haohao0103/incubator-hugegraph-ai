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

"""Token Budget Controller for GraphRAG retrieval context.

Manages the allocation of context window tokens across different content
types (entities, relationships, communities, raw chunks) to ensure
the final LLM prompt stays within model limits.

Inspired by LightRAG's multi-level token truncation strategy but adapted
for HugeGraph-AI's community + entity + relationship model.

Usage::

    budget = TokenBudget(
        max_total_tokens=4096,
        max_entity_tokens=1500,
        max_relation_tokens=1000,
        max_community_tokens=800,
    )
    budget.add("entity", "Entity: Apache HugeGraph - A graph database...", 15)
    budget.add("entity", "Entity: Gremlin - A graph traversal language...", 12)
    budget.add("relation", "(HugeGraph)-[implements]->(TinkerPop)", 8)
    context = budget.build_context()
    # context is a formatted string respecting all token budgets
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Rough token estimation (4 chars ≈ 1 token for English; ~2 chars for CJK)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate.

    Uses a simple heuristic: ~4 chars per token for Latin text,
    ~1.5 chars per token for CJK-heavy text.  Falls back to
    ``len(text) // 3`` for mixed content.
    """
    if not text:
        return 0
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ratio = cjk_count / len(text) if len(text) > 0 else 0
    # CJK: ~1.5 chars/token; Latin: ~4 chars/token; blended
    chars_per_token = 4.0 - 2.5 * ratio  # ranges from 1.5 (all CJK) to 4.0 (all Latin)
    return int(len(text) / chars_per_token) + 1


@dataclass
class TokenBudgetConfig:
    """Configuration for token budget allocation.

    Attributes:
        max_total_tokens: Hard cap on the combined context.
        max_entity_tokens: Budget for entity descriptions.
        max_relation_tokens: Budget for relationship descriptions.
        max_community_tokens: Budget for community summaries.
        max_chunk_tokens: Budget for raw document chunks.
        reserve_for_prompt: Tokens reserved for the system prompt +
            question (deducted from max_total before allocation).
    """

    max_total_tokens: int = 4096
    max_entity_tokens: int = 1500
    max_relation_tokens: int = 1000
    max_community_tokens: int = 800
    max_chunk_tokens: int = 1000
    reserve_for_prompt: int = 300

    def effective_total(self) -> int:
        """Total tokens available for context after prompt reservation."""
        return max(0, self.max_total_tokens - self.reserve_for_prompt)


class TokenBudget:
    """Accumulates content entries and enforces per-category token limits.

    Entries are added by category and truncated when the category budget
    is exceeded.  The final ``build_context()`` call assembles all
    accepted entries into a single context string.
    """

    def __init__(self, config: Optional[TokenBudgetConfig] = None):
        self._config = config or TokenBudgetConfig()
        # category -> list of (text, token_count)
        self._entries: Dict[str, List[tuple]] = {
            "entity": [],
            "relation": [],
            "community": [],
            "chunk": [],
        }
        # Track total used tokens across all categories
        self._total_used: int = 0

    @property
    def config(self) -> TokenBudgetConfig:
        return self._config

    @property
    def total_used(self) -> int:
        return self._total_used

    @property
    def remaining(self) -> int:
        return max(0, self._config.effective_total() - self._total_used)

    def _category_limit(self, category: str) -> int:
        """Get the token limit for a given category."""
        mapping = {
            "entity": self._config.max_entity_tokens,
            "relation": self._config.max_relation_tokens,
            "community": self._config.max_community_tokens,
            "chunk": self._config.max_chunk_tokens,
        }
        return mapping.get(category, self._config.effective_total())

    def _category_used(self, category: str) -> int:
        """Get total tokens used in a category."""
        return sum(tc for _, tc in self._entries.get(category, []))

    def add(self, category: str, text: str, estimated_tokens: Optional[int] = None) -> bool:
        """Add a content entry to the budget.

        Args:
            category: One of ``entity``, ``relation``, ``community``, ``chunk``.
            text: The text content to add.
            estimated_tokens: Pre-computed token count. If None, auto-estimated.

        Returns:
            True if the entry was accepted, False if it would exceed budget.
        """
        tc = estimated_tokens if estimated_tokens is not None else _estimate_tokens(text)

        cat_limit = self._category_limit(category)
        cat_used = self._category_used(category)
        total_limit = self._config.effective_total()

        # Check category budget
        if cat_used + tc > cat_limit:
            return False

        # Check total budget
        if self._total_used + tc > total_limit:
            return False

        self._entries.setdefault(category, []).append((text, tc))
        self._total_used += tc
        return True

    def add_truncated(self, category: str, text: str, estimated_tokens: Optional[int] = None) -> str:
        """Add content, truncating to fit if necessary.

        Returns:
            The actually added text (possibly truncated).
        """
        tc = estimated_tokens if estimated_tokens is not None else _estimate_tokens(text)

        cat_limit = self._category_limit(category)
        cat_used = self._category_used(category)
        remaining_cat = cat_limit - cat_used
        remaining_total = self._config.effective_total() - self._total_used

        available = min(remaining_cat, remaining_total)
        if available <= 0:
            return ""

        if tc <= available:
            self.add(category, text, tc)
            return text

        # Truncate proportionally
        ratio = available / tc
        # Rough truncation by character ratio
        trunc_len = max(1, int(len(text) * ratio))
        truncated = text[:trunc_len]
        actual_tc = _estimate_tokens(truncated)
        self.add(category, truncated, actual_tc)
        return truncated

    def build_context(self, separators: Optional[Dict[str, str]] = None) -> str:
        """Assemble all accepted entries into a single context string.

        Args:
            separators: Optional per-category separator strings.

        Returns:
            Formatted context string respecting all token budgets.
        """
        sep = separators or {
            "entity": "\n",
            "relation": "\n",
            "community": "\n\n",
            "chunk": "\n\n",
        }
        parts = []

        # Emit in priority order: community > entity > relation > chunk
        for category in ("community", "entity", "relation", "chunk"):
            entries = self._entries.get(category, [])
            if not entries:
                continue
            cat_texts = [text for text, _ in entries]
            s = sep.get(category, "\n")
            cat_block = s.join(cat_texts)
            parts.append(cat_block)

        return "\n\n".join(parts)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of budget utilization."""
        return {
            "total_used": self._total_used,
            "total_limit": self._config.effective_total(),
            "utilization_pct": (
                round(self._total_used / max(1, self._config.effective_total()) * 100, 1)
            ),
            "by_category": {
                cat: {
                    "used": self._category_used(cat),
                    "limit": self._category_limit(cat),
                    "entries": len(self._entries.get(cat, [])),
                }
                for cat in self._entries
            },
        }

    def reset(self):
        """Clear all entries and reset usage counters."""
        self._entries = {k: [] for k in self._entries}
        self._total_used = 0
