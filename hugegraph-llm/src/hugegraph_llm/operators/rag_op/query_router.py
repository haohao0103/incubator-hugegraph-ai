"""
Query Router: Global vs Local Search Classification
=====================================================

Determines whether a user query should be routed to:
- GLOBAL search: community-level summaries (macro/overview questions)
- LOCAL search: entity/subgraph-level retrieval (micro/specific questions)

Strategy:
1. Rule-based classification (fast, no LLM cost)
2. LLM-based classification (fallback, higher accuracy)
3. Hybrid: rules first, LLM for ambiguous cases

Usage:
    router = QueryRouter(llm=optional_llm)
    route = router.classify("What are Tencent's main businesses?")
    # -> {"type": "global", "confidence": 0.95, "method": "rule"}
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RouteResult:
    """Result of query classification."""
    query_type: str  # "global" or "local"
    confidence: float  # 0.0 ~ 1.0
    method: str  # "rule", "llm", "default"
    reason: str  # human-readable explanation


class QueryRouter:
    """Routes queries to global or local search paths.

    Global queries ask for overview, summary, themes, comparison across
    broad topics. They benefit from community-level summaries.

    Local queries ask about specific entities, relationships, facts.
    They benefit from direct subgraph retrieval.
    """

    # ── Rule-based keywords ──────────────────────────────────────

    GLOBAL_KEYWORDS = {
        "overview", "summary", "summarize", "总结", "概述", "概览",
        "main", "major", "primary", "core", "主要", "核心",
        "theme", "themes", "topic", "topics", "主题",
        "trend", "trends", "development", "发展", "趋势",
        "compare", "comparison", "对比", "比较",
        "industry", "market", "领域", "行业", "市场",
        "impact", "effect", "influence", "影响",
        "what are", "what is", "有哪些", "是什么",
        "how does", "how do", "如何", "怎样",
        "list", "enumerate", "列举",
    }

    LOCAL_KEYWORDS = {
        "who", "whom", "谁",
        "where", "哪里", "在哪",
        "when", "什么时候", "何时",
        "which", "哪个",
        "specific", "detail", "具体", "详细",
        "relationship", "relation", "关系",
        "connection", "connect", "联系", "关联",
        "work", "works", "worked", "工作",
        "located", "location", "位于", "地点",
        "born", "birth", "出生",
        "found", "founded", "成立", "创建",
        "collaborate", "cooperate", "合作",
        "know", "认识", "了解",
    }

    # Query patterns that strongly indicate global search
    GLOBAL_PATTERNS = [
        re.compile(r"^(what\s+(are|is)\s+(the\s+)?)?main\b", re.I),
        re.compile(r"^(what\s+(are|is)\s+(the\s+)?)?(core|primary|key)\b", re.I),
        re.compile(r"(overview|summary|总结|概述)$", re.I),
        re.compile(r"(compare|comparison|对比|比较).+(and|与|和)", re.I),
        re.compile(r"(industry|market|field|领域).+(trend|development|趋势|发展)", re.I),
        re.compile(r"^(介绍|简述|概述|总结|分析).+", re.I),
        re.compile(r".+((有|存在).+(哪些|什么).+(问题|风险|优势|特点))", re.I),
    ]

    # Patterns that strongly indicate local search
    LOCAL_PATTERNS = [
        re.compile(r"^(who|whom|谁)\b", re.I),
        re.compile(r"^(where|哪里|在哪|位于)\b", re.I),
        re.compile(r"^(when|什么时候|何时)\b", re.I),
        re.compile(r"(relationship|relation|关系|联系).+(between|among|之间)", re.I),
        re.compile(r"^(do|does|did|is|are|was|were|can|could).+\?", re.I),
        re.compile(r"(张三|李四|具体|详细).+(谁|什么|哪里|何时)", re.I),
    ]

    def __init__(self, llm=None, threshold: float = 0.7):
        """Initialize router.

        Args:
            llm: Optional LLM client for ambiguous cases.
            threshold: Confidence threshold for rule-based decisions.
                       Below this, falls back to LLM if available.
        """
        self._llm = llm
        self._threshold = threshold

    def classify(self, query: str) -> RouteResult:
        """Classify query as global or local.

        Strategy:
        1. Strong pattern match → immediate decision
        2. Keyword scoring → decision if confidence >= threshold
        3. LLM fallback → if available and ambiguous
        4. Default → global (safer for broad coverage)
        """
        query_lower = query.lower()

        # Step 1: Strong pattern match
        for pattern in self.GLOBAL_PATTERNS:
            if pattern.search(query_lower):
                return RouteResult(
                    query_type="global",
                    confidence=0.95,
                    method="rule",
                    reason=f"Matched global pattern: {pattern.pattern}",
                )

        for pattern in self.LOCAL_PATTERNS:
            if pattern.search(query_lower):
                return RouteResult(
                    query_type="local",
                    confidence=0.95,
                    method="rule",
                    reason=f"Matched local pattern: {pattern.pattern}",
                )

        # Step 2: Keyword scoring
        global_score = 0
        local_score = 0

        # Tokenize: English words + individual CJK characters
        tokens = re.findall(r"[a-z]+|[\u4e00-\u9fff]", query_lower)
        for t in tokens:
            if t in self.GLOBAL_KEYWORDS:
                global_score += 1
            if t in self.LOCAL_KEYWORDS:
                local_score += 1

        # Check multi-word phrases (English phrases and Chinese bigrams)
        for phrase in self.GLOBAL_KEYWORDS:
            if len(phrase) > 1 and phrase in query_lower:
                global_score += 2
        for phrase in self.LOCAL_KEYWORDS:
            if len(phrase) > 1 and phrase in query_lower:
                local_score += 2

        total = global_score + local_score
        if total > 0:
            if global_score > local_score:
                confidence = 0.5 + 0.4 * (global_score / total)
                if confidence >= self._threshold:
                    return RouteResult(
                        query_type="global",
                        confidence=round(confidence, 3),
                        method="rule",
                        reason=f"Keyword score: global={global_score}, local={local_score}",
                    )
            elif local_score > global_score:
                confidence = 0.5 + 0.4 * (local_score / total)
                if confidence >= self._threshold:
                    return RouteResult(
                        query_type="local",
                        confidence=round(confidence, 3),
                        method="rule",
                        reason=f"Keyword score: global={global_score}, local={local_score}",
                    )

        # Step 3: LLM fallback
        if self._llm is not None:
            return self._classify_with_llm(query)

        # Step 4: Default to global (broad coverage, safer)
        return RouteResult(
            query_type="global",
            confidence=0.5,
            method="default",
            reason="No strong signals, defaulting to global",
        )

    def _classify_with_llm(self, query: str) -> RouteResult:
        """Use LLM to classify ambiguous queries."""
        prompt = f"""Classify the user query as GLOBAL or LOCAL.

Definitions:
- GLOBAL: Overview, summary, themes, trends, comparisons, "what are the main..."
- LOCAL: Specific facts about entities, "who", "where", "when", relationships

Query: "{query}"

Respond with ONLY one word: GLOBAL or LOCAL."""

        try:
            # Try OpenAI-compatible API
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.xiaomimimo.com/v1",
                api_key="sk-cs5kqi80f6upqy2e3k3xi39jtizhpgf6dkdd3j9ysoupfw7p",
            )
            response = client.chat.completions.create(
                model="mimo-v2.5-pro",
                messages=[
                    {"role": "system", "content": "You classify queries. Output only GLOBAL or LOCAL."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=32,
            )
            content = (response.choices[0].message.content or "").strip().upper()
            if "GLOBAL" in content:
                return RouteResult(
                    query_type="global",
                    confidence=0.85,
                    method="llm",
                    reason="LLM classified as GLOBAL",
                )
            elif "LOCAL" in content:
                return RouteResult(
                    query_type="local",
                    confidence=0.85,
                    method="llm",
                    reason="LLM classified as LOCAL",
                )
        except Exception as e:
            pass

        return RouteResult(
            query_type="global",
            confidence=0.5,
            method="default",
            reason="LLM failed, defaulting to global",
        )

    def batch_classify(self, queries: list) -> list:
        """Classify multiple queries."""
        return [self.classify(q) for q in queries]


# ── Standalone test ──────────────────────────────────────────

if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        # Global
        "腾讯公司的主要业务是什么？",
        "What are the main themes in this dataset?",
        "Compare Alibaba and Tencent",
        "总结一下这个领域的趋势",
        # Local
        "张三在哪家公司工作？",
        "Who founded Alibaba?",
        "李四和王五是什么关系？",
        "Where is Tencent headquartered?",
        # Ambiguous
        "告诉我关于腾讯的信息",
        "What happened in 2020?",
    ]

    print("Query Router Test Results")
    print("=" * 80)
    for q in test_queries:
        r = router.classify(q)
        print(f"[{r.query_type:6}] ({r.method:5}, conf={r.confidence:.2f}) {q}")
        print(f"       reason: {r.reason}")
