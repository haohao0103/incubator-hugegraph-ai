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
DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search operator.

Implements a 5-step deep retrieval strategy that combines the breadth of
Global Search with the depth of Local Search:

1. HyDE: Generate a hypothetical answer passage for the query
2. Community Match: Find top-K relevant communities via vector search
3. Primer: Generate initial analysis + follow-up sub-questions
4. Parallel Local Search: Iteratively search for specific facts (max 2 depth)
5. Reduce: Synthesize all findings into a comprehensive answer

Reference: Microsoft GraphRAG DRIFT search, Neo4j GraphRAG DRIFT
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log

# ── Step 1: HyDE ──────────────────────────────────────────────

HYDE_PROMPT = (
    "Write a concise passage that answers the following question. "
    "Focus on factual content with key entities and relationships. "
    "Do not worry about perfect accuracy — generate a plausible answer.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)

HYDE_PROMPT_CN = (
    "请写一段简短的文字回答以下问题。"
    "重点包含关键实体和关系。"
    "不必担心信息完全准确——生成一段合理的回答即可。\n\n"
    "问题：{query}\n\n"
    "段落："
)

# ── Step 3: Primer ────────────────────────────────────────────

PRIMER_PROMPT = (
    "You are analyzing a knowledge graph to answer a complex question.\n\n"
    "## User Question\n{query}\n\n"
    "## Community Context\n{community_context}\n\n"
    "## Task\n"
    "Based on the community summaries above, provide:\n"
    '1. An initial analysis (2-3 sentences summarizing key insights)\n'
    "2. 2-4 specific follow-up sub-questions that would help deepen the answer\n\n"
    "Output in JSON format:\n"
    '{{"initial_answer": "...", "follow_up_queries": ["sub-question 1", "sub-question 2", ...]}}'
)

PRIMER_PROMPT_CN = (
    "你正在分析一个知识图谱来回答一个复杂问题。\n\n"
    "## 用户问题\n{query}\n\n"
    "## 社区上下文\n{community_context}\n\n"
    "## 任务\n"
    "基于以上社区摘要，请提供：\n"
    "1. 初步分析（2-3句话总结关键洞察）\n"
    "2. 2-4个具体的后续子问题，帮助深入回答\n\n"
    "以JSON格式输出：\n"
    '{{"initial_answer": "...", "follow_up_queries": ["子问题1", "子问题2", ...]}}'
)

# ── Step 4: Local Search Finding ───────────────────────────────

FINDING_SYNTHESIZE_PROMPT = (
    "Given the following search results for the sub-question, "
    "synthesize a concise finding.\n\n"
    "Sub-question: {sub_query}\n\n"
    "Search Results:\n{search_results}\n\n"
    "Synthesize a brief finding (2-3 sentences) that captures key facts:"
)

FINDING_SYNTHESIZE_PROMPT_CN = (
    "根据以下子问题的搜索结果，综合出一个简洁的发现。\n\n"
    "子问题：{sub_query}\n\n"
    "搜索结果：\n{search_results}\n\n"
    "综合一个简短的发现（2-3句话），捕捉关键事实："
)

# ── Step 4 (depth > 1): Generate new follow-ups from findings ──

DEEP_FOLLOW_UP_PROMPT = (
    "Based on the original question and the findings so far, "
    "generate 2-3 new sub-questions to further deepen the analysis.\n\n"
    "Original Question: {query}\n\n"
    "Findings So Far:\n{findings_text}\n\n"
    "Output a JSON array of sub-question strings:\n"
    '["sub-question 1", "sub-question 2", ...]'
)

DEEP_FOLLOW_UP_PROMPT_CN = (
    "基于原始问题和目前的发现，生成2-3个新的子问题以进一步深入分析。\n\n"
    "原始问题：{query}\n\n"
    "目前的发现：\n{findings_text}\n\n"
    "以JSON数组输出子问题字符串：\n"
    '["子问题1", "子问题2", ...]'
)

# ── Step 5: Reduce ───────────────────────────────────────────

REDUCE_PROMPT = (
    "You are synthesizing a comprehensive answer from multiple investigation steps.\n\n"
    "## User Question\n{query}\n\n"
    "## Initial Analysis\n{initial_answer}\n\n"
    "## Deep Findings\n{findings_text}\n\n"
    "## Task\n"
    "Synthesize everything into a comprehensive, well-structured answer:\n"
    "- Build upon the initial analysis with specific facts from findings\n"
    "- Maintain logical flow and coherence\n"
    "- Cite specific details where relevant\n"
    "- Write in a professional, analytical tone\n\n"
    "Answer:"
)

REDUCE_PROMPT_CN = (
    "你正在从多个调查步骤中综合出一个全面的答案。\n\n"
    "## 用户问题\n{query}\n\n"
    "## 初步分析\n{initial_answer}\n\n"
    "## 深度发现\n{findings_text}\n\n"
    "## 任务\n"
    "将所有内容综合成一个全面、结构清晰的答案：\n"
    "- 在初步分析基础上，用发现中的具体事实补充\n"
    "- 保持逻辑流畅和连贯\n"
    "- 在相关处引用具体细节\n"
    "- 以专业分析性口吻撰写\n\n"
    "答案："
)


class DriftSearch:
    """DRIFT: Dynamic Reasoning and Inference with Flexible Traversal.

    A 5-step search strategy combining Global Search breadth with
    Local Search depth. Suitable for complex analytical questions
    that require both overview understanding and specific facts.

    Usage::

        searcher = DriftSearch(llm=my_llm, embedding=my_embedding)
        context = searcher.run({
            "query": "What are the key risk factors in the supply chain?",
            "community_reports": [...],
        })
        # context["drift_answer"] = comprehensive answer
    """

    MAX_LOCAL_DEPTH = 2
    COMMUNITIES_TOP_K = 5
    LOCAL_SEARCH_TOP_K = 10
    MAX_FINDINGS_PER_DEPTH = 10

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embedding: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        max_local_depth: int = 2,
        communities_top_k: int = 5,
        local_search_top_k: int = 10,
        language: str = "en",
    ):
        """Initialize DRIFT search.

        :param llm: LLM instance for generation tasks.
        :param embedding: Embedding model for vector search.
        :param vector_index: Vector index for community/chunk retrieval.
        :param graph_client: HugeGraph client for graph traversal.
        :param max_local_depth: Max iteration depth for parallel local search (1-2).
        :param communities_top_k: Number of top communities to match.
        :param local_search_top_k: Top-K results for vector search in local search.
        :param language: "en" or "cn" for prompt selection.
        """
        self._llm = llm
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_client = graph_client
        self._max_local_depth = max(1, min(max_local_depth, 3))
        self._communities_top_k = communities_top_k
        self._local_search_top_k = local_search_top_k
        self._language = language

    def _get_llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = LLMs().get_general_llm()
        return self._llm

    def _get_prompt(self, en_prompt: str, cn_prompt: str) -> str:
        return cn_prompt if self._language == "cn" else en_prompt

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full DRIFT search pipeline.

        Reads from context:
            query: The user question.
            community_reports: List of community report dicts.
                Each should have: title, summary, key_entities, importance_score.

        Writes to context:
            drift_answer: Final synthesized answer.
            drift_findings: List of all findings from local searches.
            drift_communities_used: Number of communities used.
            drift_depth_reached: Actual depth reached.
            drift_primer: Primer output (initial_answer, follow_up_queries).
            call_count: Total LLM calls made.
        """
        query = context.get("query", "")
        community_reports = context.get("community_reports", [])

        if not query:
            context["drift_answer"] = ""
            return context

        call_count = context.get("call_count", 0)

        # Step 1: HyDE — generate hypothetical answer
        hyde_passage = self._step1_hyde(query)
        call_count += 1

        # Step 2: Community matching
        matched_communities = self._step2_match_communities(
            hyde_passage or query, community_reports
        )

        # Step 3: Primer — initial analysis + follow-up queries
        primer = self._step3_primer(query, matched_communities)
        call_count += 1

        # Step 4: Iterative parallel local search
        all_findings = []
        follow_ups = primer.get("follow_up_queries", [])
        depth_reached = 0

        for depth in range(self._max_local_depth):
            depth_reached = depth + 1
            if not follow_ups:
                break

            batch_findings = self._step4_parallel_local_search(
                follow_ups, all_findings
            )
            all_findings.extend(batch_findings)
            call_count += len(batch_findings)

            # Generate new follow-ups for next depth
            if depth < self._max_local_depth - 1 and batch_findings:
                follow_ups = self._generate_deep_follow_ups(
                    query, all_findings
                )
                call_count += 1
            else:
                follow_ups = []

        # Step 5: Reduce — synthesize final answer
        final_answer = self._step5_reduce(
            query, primer.get("initial_answer", ""), all_findings
        )
        call_count += 1

        # Write results to context
        context["drift_answer"] = final_answer
        context["drift_findings"] = all_findings
        context["drift_communities_used"] = len(matched_communities)
        context["drift_depth_reached"] = depth_reached
        context["drift_primer"] = primer
        context["call_count"] = call_count

        log.info(
            "DRIFT search: %d findings, %d communities, depth=%d, %d LLM calls",
            len(all_findings),
            len(matched_communities),
            depth_reached,
            call_count,
        )
        return context

    # ── Step 1: HyDE ──────────────────────────────────────────

    def _step1_hyde(self, query: str) -> str:
        """Generate a hypothetical answer passage for the query."""
        prompt = self._get_prompt(HYDE_PROMPT, HYDE_PROMPT_CN)
        try:
            llm = self._get_llm()
            passage = llm.generate(prompt=prompt.format(query=query))
            return passage.strip() if passage else ""
        except Exception as e:
            log.warning("DRIFT Step 1 (HyDE) failed: %s", e)
            return ""

    # ── Step 2: Community Matching ────────────────────────────

    def _step2_match_communities(
        self,
        search_text: str,
        community_reports: List[Dict],
    ) -> List[Dict]:
        """Match query to top-K relevant communities.

        Uses vector similarity if embedding + vector_index are available,
        otherwise falls back to keyword matching.

        Optimized: batch-embeds all community texts in a single API call
        instead of per-community calls (reduces N+1 API round-trips to 2).
        """
        if not community_reports:
            return []

        # If vector index is available, use embedding similarity
        if self._embedding and self._vector_index:
            try:
                query_vec = self._embedding.get_texts_embeddings([search_text])[0]
                # Batch: embed all community texts in one call
                report_texts = [
                    self._community_to_text(r) for r in community_reports
                ]
                report_vecs = self._embedding.get_texts_embeddings(report_texts)
                scored = [
                    (self._cosine_similarity(query_vec, rv), r)
                    for rv, r in zip(report_vecs, community_reports)
                ]

                scored.sort(key=lambda x: x[0], reverse=True)
                return [r for _, r in scored[: self._communities_top_k]]
            except Exception as e:
                log.warning("DRIFT vector community match failed: %s", e)

        # Fallback: sort by importance score
        sorted_reports = sorted(
            community_reports,
            key=lambda r: r.get("importance_score", 0),
            reverse=True,
        )
        return sorted_reports[: self._communities_top_k]

    # ── Step 3: Primer ────────────────────────────────────────

    def _step3_primer(
        self, query: str, communities: List[Dict]
    ) -> Dict[str, Any]:
        """Generate initial analysis and follow-up sub-questions."""
        community_context = "\n\n".join(
            f"### {r.get('title', 'Community')} (importance: {r.get('importance_score', 5.0):.1f})\n"
            f"{r.get('summary', '')}\n"
            f"Key Entities: {', '.join(r.get('key_entities', []))}"
            for r in communities
        )

        prompt = self._get_prompt(PRIMER_PROMPT, PRIMER_PROMPT_CN)
        prompt_text = prompt.format(query=query, community_context=community_context)

        try:
            llm = self._get_llm()
            response = llm.generate(prompt=prompt_text)
            return self._parse_primer_json(response)
        except Exception as e:
            log.warning("DRIFT Step 3 (Primer) failed: %s", e)
            return {
                "initial_answer": "",
                "follow_up_queries": [query],
            }

    # ── Step 4: Parallel Local Search ─────────────────────────

    def _step4_parallel_local_search(
        self,
        queries: List[str],
        existing_findings: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Execute local searches in parallel.

        Local search = vector retrieval + optional graph traversal,
        synthesized into a concise finding.
        """
        existing = existing_findings or []

        async def _search_one(q: str) -> Optional[Dict]:
            try:
                search_results = self._local_vector_search(q)
                finding_text = self._local_synthesize(q, search_results)
                return {
                    "sub_query": q,
                    "finding": finding_text,
                    "sources": search_results[:3],  # Keep top 3 sources
                    "depth_context": f"based on {len(existing)} prior findings",
                }
            except Exception as e:
                log.warning("Local search failed for '%s': %s", q, e)
                return None

        async def _run_all():
            tasks = [_search_one(q) for q in queries]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.new_event_loop()
            results = loop.run_until_complete(_run_all())
            loop.close()
        except RuntimeError:
            # Fallback: synchronous execution
            results = []
            for q in queries:
                try:
                    results.append(asyncio.run(_search_one(q)))
                except Exception:
                    results.append(None)

        return [r for r in results if r is not None]

    def _local_vector_search(self, query: str) -> List[Dict]:
        """Perform vector search for a single sub-query."""
        if not self._embedding or not self._vector_index:
            return []

        try:
            query_vec = self._embedding.get_texts_embeddings([query])[0]
            results = self._vector_index.search(
                query_vec, self._local_search_top_k
            )
            return results if isinstance(results, list) else []
        except Exception as e:
            log.warning("Local vector search failed: %s", e)
            return []

    def _local_synthesize(self, sub_query: str, search_results: List[Dict]) -> str:
        """Synthesize search results into a concise finding."""
        if not search_results:
            return f"No relevant information found for: {sub_query}"

        results_text = "\n".join(
            f"- {str(r)[:300]}" for r in search_results[:5]
        )

        prompt = self._get_prompt(FINDING_SYNTHESIZE_PROMPT, FINDING_SYNTHESIZE_PROMPT_CN)
        prompt_text = prompt.format(sub_query=sub_query, search_results=results_text)

        try:
            llm = self._get_llm()
            return llm.generate(prompt=prompt_text).strip()
        except Exception:
            # Fallback: concatenate top results
            return "; ".join(str(r)[:200] for r in search_results[:2])

    def _generate_deep_follow_ups(
        self, query: str, findings: List[Dict]
    ) -> List[str]:
        """Generate new follow-up questions based on accumulated findings."""
        findings_text = "\n".join(
            f"- {f['finding'][:200]}" for f in findings[-self.MAX_FINDINGS_PER_DEPTH:]
        )

        prompt = self._get_prompt(DEEP_FOLLOW_UP_PROMPT, DEEP_FOLLOW_UP_PROMPT_CN)
        prompt_text = prompt.format(query=query, findings_text=findings_text)

        try:
            llm = self._get_llm()
            response = llm.generate(prompt=prompt_text)
            parsed = self._parse_json_array(response)
            return parsed[:3]  # Max 3 new follow-ups
        except Exception as e:
            log.warning("Deep follow-up generation failed: %s", e)
            return []

    # ── Step 5: Reduce ────────────────────────────────────────

    def _step5_reduce(
        self,
        query: str,
        initial_answer: str,
        findings: List[Dict],
    ) -> str:
        """Synthesize all findings into a comprehensive final answer."""
        findings_text = "\n\n".join(
            f"### Finding: {f.get('sub_query', 'N/A')}\n{f.get('finding', '')}"
            for f in findings[:20]
        )

        if not findings_text.strip():
            return initial_answer or "No sufficient information to answer the question."

        prompt = self._get_prompt(REDUCE_PROMPT, REDUCE_PROMPT_CN)
        prompt_text = prompt.format(
            query=query,
            initial_answer=initial_answer or "No initial analysis available.",
            findings_text=findings_text,
        )

        try:
            llm = self._get_llm()
            answer = llm.generate(prompt=prompt_text)
            return answer.strip()
        except Exception as e:
            log.error("DRIFT Step 5 (Reduce) failed: %s", e)
            # Fallback: concatenate findings
            parts = []
            if initial_answer:
                parts.append(initial_answer)
            parts.extend(f["finding"] for f in findings[:5])
            return "\n\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _community_to_text(report: Dict) -> str:
        """Convert a community report dict to searchable text."""
        return (
            f"{report.get('title', '')} "
            f"{report.get('summary', '')} "
            f"{' '.join(report.get('key_entities', []))} "
            f"{' '.join(report.get('relationship_patterns', []))}"
        )

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Uses numpy when available for vectorized computation (10-50x faster
        on large embeddings), falls back to pure Python otherwise.
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        try:
            import numpy as np

            a = np.asarray(vec_a, dtype=np.float32)
            b = np.asarray(vec_b, dtype=np.float32)
            dot = np.dot(a, b)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(dot / (na * nb))
        except ImportError:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = sum(a * a for a in vec_a) ** 0.5
            norm_b = sum(b * b for b in vec_b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

    @staticmethod
    def _parse_primer_json(response: str) -> Dict[str, Any]:
        """Parse primer LLM response as JSON."""
        # Try to extract JSON from response
        text = response.strip()

        # Direct JSON parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback
        return {
            "initial_answer": text[:500],
            "follow_up_queries": [],
        }

    @staticmethod
    def _parse_json_array(response: str) -> List[str]:
        """Parse a JSON array of strings from LLM response."""
        text = response.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, TypeError):
            pass

        import re

        json_match = re.search(r"\[[^\[\]]*\]", text)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if isinstance(result, list):
                    return [str(item) for item in result]
            except (json.JSONDecodeError, TypeError):
                pass

        return []
