"""Query understanding: dual-level keywords + synonym expansion.

Borrowed from the 货拉拉 metadata-GraphRAG retrieval design: a user query is
first decomposed into **high-level** (themes/subject areas -> global context)
and **low-level** (entities/specifics -> local context) keywords, then the
low-level terms are expanded through the synonym layer so that "客单价" also
contributes its canonical "arpu" for exact schema linking.

Pipeline::

    question
      ├─ DualKeywordExtract (LLM first, heuristic fallback)  -> hl/ll keywords
      ├─ KgTermGraph synonym expansion (ll terms + raw-question aliases)
      └─ QueryIntent { hl_keywords, ll_keywords, expanded_terms,
                       synonym_hits, local_context, global_context }

The extractor is a thin composition over the existing
``dual_keyword_extract`` (LightRAG-style) and the new ``KgTermGraph`` term
layer. The LLM is injected (glm-5.3 works as a chat role); when it is absent
or flaky the heuristic/short-query fallback keeps the path deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hugegraph_llm.operators.graph_op.kg_term_graph import KgTermGraph
from hugegraph_llm.operators.llm_op.dual_keyword_extract import (
    DualKeywordConfig,
    DualKeywordExtract,
)

logger = logging.getLogger(__name__)

# stop-word-ish tails that should never pollute retrieval terms. Exact
# matches are dropped; when a keyword *ends with* a tail, the tail is
# stripped ("实时订单表可以用吗" -> "实时订单表") so the remaining entity
# term can be matched, while pure interrogative chunks ("可以用吗") vanish.
_NOISE_TAILS = {
    "多少", "怎么", "如何", "哪个", "哪些", "是什么", "在哪里", "在哪个",
    "可以用吗", "可以吗", "能用吗", "怎么取", "怎么看", "去哪里",
    "是哪个", "怎么样", "有没有", "能否", "能不能", "是",
}


@dataclass
class QueryIntent:
    """Structured understanding of one user question."""

    question: str
    hl_keywords: List[str] = field(default_factory=list)   # themes (global)
    ll_keywords: List[str] = field(default_factory=list)   # entities (local)
    expanded_terms: List[str] = field(default_factory=list)  # retrieval terms
    synonym_hits: List[Tuple[str, str]] = field(default_factory=list)  # (alias, canonical)
    extraction_method: str = "heuristic"

    @property
    def local_context(self) -> str:
        """Local query context: concrete entities + their canonical names."""
        parts = list(self.ll_keywords)
        for _alias, canon in self.synonym_hits:
            if canon not in parts:
                parts.append(canon)
        return " ".join(parts)

    @property
    def global_context(self) -> str:
        """Global query context: themes / subject areas."""
        return " ".join(self.hl_keywords)

    @property
    def has_terms(self) -> bool:
        return len(self.expanded_terms) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "hl_keywords": self.hl_keywords,
            "ll_keywords": self.ll_keywords,
            "expanded_terms": self.expanded_terms,
            "synonym_hits": [list(h) for h in self.synonym_hits],
            "extraction_method": self.extraction_method,
            "local_context": self.local_context,
            "global_context": self.global_context,
        }


@dataclass
class QueryUnderstandingConfig:
    """Knobs for the query-understanding stage."""

    language: str = "zh"
    max_keywords_per_level: int = 5
    min_keyword_length: int = 2
    short_query_threshold: int = 50
    expand_with_synonyms: bool = True
    include_hl_in_terms: bool = True  # theme words also feed entity retrieval


class QueryUnderstanding:
    """Dual-level keyword extraction + synonym expansion for retrieval."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        term_graph: Optional[KgTermGraph] = None,
        config: Optional[QueryUnderstandingConfig] = None,
    ) -> None:
        self._config = config or QueryUnderstandingConfig()
        self._terms = term_graph or KgTermGraph.default()
        extract_config = DualKeywordConfig(
            max_keywords_per_level=self._config.max_keywords_per_level,
            min_keyword_length=self._config.min_keyword_length,
            language=self._config.language,
            short_query_threshold=self._config.short_query_threshold,
        )
        self._extractor = DualKeywordExtract(llm=llm, config=extract_config)

    @property
    def term_graph(self) -> KgTermGraph:
        return self._terms

    @staticmethod
    def _strip_tail(word: str) -> str:
        """Strip a trailing interrogative tail, longest first."""
        for tail in sorted(_NOISE_TAILS, key=len, reverse=True):
            if word.endswith(tail) and len(word) > len(tail):
                return word[: -len(tail)]
        return word

    def understand(self, question: str) -> QueryIntent:
        """Decompose + expand a question into a :class:`QueryIntent`."""
        question = (question or "").strip()
        if not question:
            return QueryIntent(question="")

        dk = self._extractor.extract(question)

        hl = [self._strip_tail(k) for k in dk.hl_keywords]
        ll = [self._strip_tail(k) for k in dk.ll_keywords]
        hl = [k for k in hl if k and k not in _NOISE_TAILS]
        ll = [k for k in ll if k and k not in _NOISE_TAILS]

        terms: List[str] = []
        seen: set = set()

        def _push(word: str) -> None:
            word = word.strip()
            if word and word.lower() not in seen:
                seen.add(word.lower())
                terms.append(word)

        for k in ll:          # entities first (highest precision)
            _push(k)
        if self._config.include_hl_in_terms:
            for k in hl:
                _push(k)
        # CJK blocks longer than 4 chars also contribute their 2-grams so the
        # lexical path keeps substring-level scoring ("每个城市的订单总额"
        # must still hit definition "订单总额" even though the full block is
        # not a verbatim substring of it). The heuristic extractor classifies
        # Chinese chunks as hl, so both levels feed the bigrams.
        for k in (ll + hl):
            if len(k) > 4 and any("\u4e00" <= ch <= "\u9fff" for ch in k):
                for i in range(len(k) - 1):
                    _push(k[i : i + 2])

        hits: List[Tuple[str, str]] = []
        if self._config.expand_with_synonyms:
            # 1) exact alias lookup on each extracted term
            for term in list(terms):
                canon = self._terms.lookup(term)
                if canon and canon.lower() not in seen:
                    seen.add(canon.lower())
                    terms.append(canon)
                    hits.append((term, canon))
            # 2) alias substring scan against the raw question (2-gram terms
            #    can't be looked up by exact match; aliases are full phrases)
            for alias, canon in self._terms.match(question):
                if canon.lower() not in seen:
                    seen.add(canon.lower())
                    terms.append(canon)
                    hits.append((alias, canon))

        return QueryIntent(
            question=question,
            hl_keywords=hl,
            ll_keywords=ll,
            expanded_terms=terms,
            synonym_hits=hits,
            extraction_method=dk.extraction_method,
        )
