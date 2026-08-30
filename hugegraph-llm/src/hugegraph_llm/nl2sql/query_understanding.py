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

"""Query understanding for NL2SQL: rule-based intent classification +
jargon-driven keyword expansion (borrowed from the parallel NL2SQL demo
branch's ``kg_query_understanding``; LLM keyword extraction replaced by the
local :class:`JargonMap`, so the whole stage runs offline and deterministically).

What the user is asking FOR drives type-weighted re-ranking: "在哪个表" must
surface tables above columns, "是多少/口径" must surface the metric's column.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# stop-word-ish tails that should never pollute retrieval terms.
_NOISE_TAILS = {
    "多少", "怎么", "如何", "哪个", "哪些", "是什么", "在哪里", "在哪个",
    "可以用吗", "可以吗", "能用吗", "怎么取", "怎么看", "去哪里",
    "是哪个", "怎么样", "有没有", "能否", "能不能", "是",
}

# question-intent patterns: what entity type the user is asking FOR.
# Table first because 哪个表/哪张表 are the most specific.
INTENT_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "table": (
        "哪个表", "哪张表", "在哪个表", "在什么表", "是什么表", "哪几张表",
        "哪个库", "哪个报表", "哪个宽表", "在哪个库", "哪个表的", "哪张表的",
        "找表", "有没有表", "在哪张表", "哪个表能",
    ),
    "field": (
        "哪个字段", "什么字段", "哪个列", "什么列", "字段是什么", "哪一列",
        "哪个属性", "哪个栏位", "对应什么字段", "对应哪个字段", "字段名",
        "字段是哪个", "哪个参数",
    ),
    "metric": (
        "是多少", "怎么算", "如何算", "怎么取", "如何计算", "怎么计算",
        "口径", "指标", "计算方式", "公式", "怎么算的", "如何统计", "统计口径",
        "总额", "数量", "单量", "订单数", "单数", "金额",
    ),
}

# default intent -> node type -> boost applied during re-ranking
DEFAULT_INTENT_BOOST: Dict[str, Dict[str, float]] = {
    "table": {"table": 1.5},
    "field": {"column": 1.4},
    "metric": {"column": 1.2},
    "general": {},
}


@dataclass
class QueryIntent:
    """Structured understanding of one user question."""

    question: str
    intent_type: str = "general"
    keywords: List[str] = field(default_factory=list)
    expanded_terms: List[str] = field(default_factory=list)
    synonym_hits: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def has_terms(self) -> bool:
        return len(self.expanded_terms) > 0

    @property
    def retrieval_texts(self) -> List[str]:
        """Extra texts to seed the linker with (canonical terms first)."""
        texts = list(self.expanded_terms)
        for _alias, canon in self.synonym_hits:
            if canon not in texts:
                texts.append(canon)
        return texts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent_type": self.intent_type,
            "keywords": self.keywords,
            "expanded_terms": self.expanded_terms,
            "synonym_hits": [list(h) for h in self.synonym_hits],
        }


def classify_intent(question: str) -> str:
    """Rule-based question-intent classification (table/field/metric/general)."""
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, question):
                return intent
    return "general"


def intent_boost(intent: str) -> Dict[str, float]:
    """Node-type -> score multiplier for a question intent."""
    return DEFAULT_INTENT_BOOST.get(intent or "general", {})


class QueryUnderstanding:
    """Offline, deterministic query understanding for schema linking."""

    def __init__(self, jargon_map=None):
        from .synonym_dict import JargonMap

        self._jargon = jargon_map or JargonMap()

    def understand(self, question: str) -> QueryIntent:
        question = (question or "").strip()
        if not question:
            return QueryIntent(question="")

        # 1) keywords: strip interrogative tails from the raw question chunks
        raw = re.split(r"[，。？?！!；;、\s]+", question)
        keywords: List[str] = []
        for chunk in raw:
            w = self._strip_tail(chunk.strip())
            if w and w not in _NOISE_TAILS and w not in keywords:
                keywords.append(w)

        # 2) jargon expansion (slang -> canonical identifiers)
        hits: List[Tuple[str, str]] = []
        terms: List[str] = []
        seen: set = set()
        for kw in list(keywords):
            canon = self._jargon.lookup(kw)
            if canon and canon.lower() not in seen:
                seen.add(canon.lower())
                terms.append(canon)
                hits.append((kw, canon))
        for alias, canon in [(a, c) for a, c in self._jargon.match(question)]:
            if canon.lower() not in seen:
                seen.add(canon.lower())
                terms.append(canon)
                hits.append((alias, canon))

        return QueryIntent(
            question=question,
            intent_type=classify_intent(question),
            keywords=keywords,
            expanded_terms=terms,
            synonym_hits=hits,
        )

    @staticmethod
    def _strip_tail(word: str) -> str:
        """Strip a trailing interrogative tail, longest first."""
        for tail in sorted(_NOISE_TAILS, key=len, reverse=True):
            if word.endswith(tail) and len(word) > len(tail):
                return word[: -len(tail)]
        return word
