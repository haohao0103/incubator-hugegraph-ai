# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not in this file except in compliance
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
GraphRAG evaluation framework.

Provides multi-dimensional evaluation for GraphRAG systems:
- Comprehensiveness: How completely the answer covers the topic
- Diversity: How varied and multi-perspective the answer is
- Empowerment: How actionable and informative the answer is
- Robustness: How consistent answers are across similar queries
- Retrieval Quality: Precision, recall of retrieved context

Inspired by Microsoft GraphRAG's evaluation methodology and
the GraphRAG survey paper (arXiv 2501.00309).
"""

import json
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


class EvaluationDimension(str, object):
    """Evaluation dimensions for GraphRAG."""

    COMPREHENSIVENESS = "comprehensiveness"
    DIVERSITY = "diversity"
    EMPOWERMENT = "empowerment"
    ROBUSTNESS = "robustness"
    RETRIEVAL_PRECISION = "retrieval_precision"
    RETRIEVAL_RECALL = "retrieval_recall"
    FAITHFULNESS = "faithfulness"


class GraphRAGEvaluator:
    """
    Multi-dimensional evaluator for GraphRAG systems.

    Evaluates RAG answer quality across multiple dimensions,
    supporting both LLM-based and metric-based evaluation.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        dimensions: Optional[List[str]] = None,
    ):
        """
        Args:
            llm: LLM instance for LLM-as-judge evaluation.
            dimensions: List of dimensions to evaluate.
                       Defaults to all available dimensions.
        """
        self.llm = llm
        self.dimensions = dimensions or [
            EvaluationDimension.COMPREHENSIVENESS,
            EvaluationDimension.DIVERSITY,
            EvaluationDimension.EMPOWERMENT,
            EvaluationDimension.RETRIEVAL_PRECISION,
            EvaluationDimension.RETRIEVAL_RECALL,
        ]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate GraphRAG answer quality.

        Args:
            context: Dict containing 'query', 'answer', 'graph_result',
                     'vector_result', and optionally 'ground_truth'.

        Returns:
            Updated context with 'evaluation_results'.
        """
        query = context.get("query", "")
        answer = context.get("answer", "")
        graph_result = context.get("graph_result", [])
        vector_result = context.get("vector_result", [])
        ground_truth = context.get("ground_truth")

        if not query or not answer:
            log.warning("Missing query or answer for evaluation")
            context["evaluation_results"] = {"error": "Missing query or answer"}
            return context

        results = {}

        for dimension in self.dimensions:
            if dimension in (
                EvaluationDimension.COMPREHENSIVENESS,
                EvaluationDimension.DIVERSITY,
                EvaluationDimension.EMPOWERMENT,
            ):
                score = self._evaluate_with_llm(query, answer, dimension)
            elif dimension == EvaluationDimension.RETRIEVAL_PRECISION:
                score = self._compute_retrieval_precision(graph_result, vector_result, ground_truth)
            elif dimension == EvaluationDimension.RETRIEVAL_RECALL:
                score = self._compute_retrieval_recall(graph_result, vector_result, ground_truth)
            elif dimension == EvaluationDimension.ROBUSTNESS:
                score = self._compute_robustness(query, answer, context)
            elif dimension == EvaluationDimension.FAITHFULNESS:
                score = self._evaluate_faithfulness(query, answer, graph_result + vector_result)
            else:
                score = 0.0
            results[dimension] = score

        # Compute overall score (weighted average)
        weights = {
            EvaluationDimension.COMPREHENSIVENESS: 0.25,
            EvaluationDimension.DIVERSITY: 0.15,
            EvaluationDimension.EMPOWERMENT: 0.15,
            EvaluationDimension.RETRIEVAL_PRECISION: 0.2,
            EvaluationDimension.RETRIEVAL_RECALL: 0.15,
            EvaluationDimension.ROBUSTNESS: 0.05,
            EvaluationDimension.FAITHFULNESS: 0.05,
        }
        overall = sum(results.get(d, 0) * weights.get(d, 0) for d in self.dimensions)
        total_weight = sum(weights.get(d, 0) for d in self.dimensions)
        results["overall"] = overall / total_weight if total_weight > 0 else 0.0

        context["evaluation_results"] = results
        log.info("Evaluation results: %s", json.dumps(results, indent=2))
        return context

    def _evaluate_with_llm(self, query: str, answer: str, dimension: str) -> float:
        """Use LLM as judge for qualitative evaluation dimensions."""
        if not self.llm:
            return self._heuristic_score(query, answer, dimension)

        dimension_prompts = {
            EvaluationDimension.COMPREHENSIVENESS: "how completely and thoroughly the answer covers all aspects of the question",
            EvaluationDimension.DIVERSITY: "how varied and multi-perspective the answer is, considering different viewpoints",
            EvaluationDimension.EMPOWERMENT: "how actionable and informative the answer is for the reader",
        }

        aspect = dimension_prompts.get(dimension, "the overall quality")
        prompt = f"""Rate the following answer on a scale of 1-5 based on {aspect}.

Question: {query}
Answer: {answer}

Rating (1-5):"""

        try:
            response = self.llm.generate(prompt=prompt).strip()
            # Extract numeric rating
            import re

            match = re.search(r"[1-5]", response)
            if match:
                return int(match.group()) / 5.0
        except Exception as e:  # pylint: disable=broad-except
            log.warning("LLM evaluation failed for %s: %s", dimension, e)

        return self._heuristic_score(query, answer, dimension)

    def _heuristic_score(self, query: str, answer: str, dimension: str) -> float:
        """Compute heuristic-based score when LLM is unavailable."""
        if dimension == EvaluationDimension.COMPREHENSIVENESS:
            # Heuristic: longer, more detailed answers score higher
            answer_len = len(answer)
            if answer_len < 50:
                return 0.2
            if answer_len < 150:
                return 0.5
            if answer_len < 300:
                return 0.7
            return 0.9

        if dimension == EvaluationDimension.DIVERSITY:
            # Heuristic: unique word ratio
            words = answer.lower().split()
            if not words:
                return 0.0
            unique_ratio = len(set(words)) / len(words)
            return min(unique_ratio * 1.5, 1.0)

        if dimension == EvaluationDimension.EMPOWERMENT:
            # Heuristic: presence of specific details
            detail_indicators = [
                "because",
                "therefore",
                "specifically",
                "for example",
                "such as",
                "因为",
                "因此",
                "具体",
                "例如",
            ]
            count = sum(1 for indicator in detail_indicators if indicator in answer.lower())
            return min(count * 0.25, 1.0)

        return 0.5

    def _compute_retrieval_precision(
        self,
        graph_result: List[Any],
        vector_result: List[Any],
        ground_truth: Optional[str],
    ) -> float:
        """
        Compute retrieval precision.

        If ground truth is available, computes actual precision.
        Otherwise, uses a heuristic based on result relevance.
        """
        total_results = len(graph_result) + len(vector_result)
        if total_results == 0:
            return 0.0

        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            relevant = 0
            for result in graph_result + vector_result:
                result_str = str(result).lower()
                result_words = set(result_str.split())
                if gt_words & result_words:
                    relevant += 1
            return relevant / total_results

        # Heuristic: non-empty results suggest reasonable precision
        return 0.7 if total_results > 0 else 0.0

    def _compute_retrieval_recall(
        self,
        graph_result: List[Any],
        vector_result: List[Any],
        ground_truth: Optional[str],
    ) -> float:
        """Compute retrieval recall against ground truth."""
        if not ground_truth:
            return 0.5  # Unknown without ground truth

        gt_words = set(ground_truth.lower().split())
        if not gt_words:
            return 0.5

        covered = set()
        for result in graph_result + vector_result:
            result_str = str(result).lower()
            for word in gt_words:
                if word in result_str:
                    covered.add(word)

        return len(covered) / len(gt_words)

    def _compute_robustness(self, query: str, answer: str, context: Dict[str, Any]) -> float:
        """
        Compute robustness score.

        Uses answer consistency as a proxy: similar questions should
        yield similar answers. For single-query evaluation, we use
        answer stability heuristics.
        """
        # Heuristic: well-structured answers are more robust
        structure_indicators = ["1.", "2.", "first", "second", "however", "on the other hand"]
        has_structure = any(ind in answer.lower() for ind in structure_indicators)

        # Check if answer is not too short or too long
        answer_len = len(answer)
        length_score = 1.0 if 50 <= answer_len <= 2000 else 0.5

        return 0.5 * (1.0 if has_structure else 0.5) + 0.5 * length_score

    def _evaluate_faithfulness(self, query: str, answer: str, context_items: List[Any]) -> float:
        """
        Evaluate faithfulness — whether the answer is supported by the context.

        Uses word overlap between answer and context as a basic heuristic,
        with optional LLM-based evaluation.
        """
        if not context_items:
            return 0.3  # No context to be faithful to

        context_text = " ".join(str(item) for item in context_items).lower()
        answer_words = set(answer.lower().split())

        # Check what fraction of answer words appear in context
        supported = sum(1 for w in answer_words if w in context_text)
        faithfulness = supported / max(len(answer_words), 1)

        return min(faithfulness, 1.0)


class BenchmarkRunner:
    """
    Run GraphRAG benchmarks with standardized evaluation.

    Provides a framework for running evaluation benchmarks against
    a set of test queries with expected answers, producing aggregate
    metrics across all evaluation dimensions.
    """

    def __init__(
        self,
        evaluator: Optional[GraphRAGEvaluator] = None,
        rag_func: Optional[Any] = None,
    ):
        """
        Args:
            evaluator: GraphRAGEvaluator instance.
            rag_func: Function that takes a query and returns RAG context.
        """
        self.evaluator = evaluator or GraphRAGEvaluator()
        self.rag_func = rag_func
        self._benchmark_history: List[Dict[str, Any]] = []

    def run_benchmark(
        self,
        test_cases: List[Dict[str, str]],
        rag_func: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmark evaluation against a set of test cases.

        Args:
            test_cases: List of dicts with 'query' and optionally 'ground_truth'.
            rag_func: RAG function to evaluate.

        Returns:
            Aggregate benchmark results.
        """
        func = rag_func or self.rag_func
        if not func:
            raise ValueError("No RAG function provided for benchmark")

        all_results = []

        for i, test_case in enumerate(test_cases):
            query = test_case["query"]
            ground_truth = test_case.get("ground_truth")

            try:
                context = func(query)
                if isinstance(context, dict):
                    context["ground_truth"] = ground_truth
                    result = self.evaluator.run(context)
                    all_results.append(result.get("evaluation_results", {}))
                else:
                    log.warning("RAG function returned non-dict for query %d", i)
            except Exception as e:  # pylint: disable=broad-except
                log.error("Benchmark test case %d failed: %s", i, e)
                all_results.append({"error": str(e)})

        # Aggregate results
        aggregate = self._aggregate_results(all_results)

        benchmark_summary = {
            "total_cases": len(test_cases),
            "successful_cases": len([r for r in all_results if "error" not in r]),
            "aggregate_scores": aggregate,
            "per_case_results": all_results,
        }

        self._benchmark_history.append(benchmark_summary)
        return benchmark_summary

    def _aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate evaluation results across all test cases."""
        dimension_sums: Dict[str, float] = {}
        dimension_counts: Dict[str, int] = {}

        for result in all_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key != "overall":
                    dimension_sums[key] = dimension_sums.get(key, 0.0) + value
                    dimension_counts[key] = dimension_counts.get(key, 0) + 1

        return {k: v / dimension_counts[k] for k, v in dimension_sums.items() if dimension_counts[k] > 0}

    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Return all benchmark run history."""
        return list(self._benchmark_history)
