"""HotpotQA Benchmark for HugeGraph GraphRAG.

Evaluates multi-hop question answering performance on the HotpotQA dataset.
Metrics: F1, Exact Match (EM), Recall@5, Precision@5.

Reference: https://hotpotqa.github.io/
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from tests.benchmarks.benchmark_framework import (
    BenchmarkFramework,
    compute_f1,
    compute_exact_match,
    compute_recall_at_k,
    compute_precision_at_k,
    AggregateResult,
)


class HotpotQAEvaluator:
    """Evaluator for HotpotQA multi-hop QA.

    HotpotQA requires reasoning over 2+ Wikipedia articles to answer
    a question. This tests the graph's ability to:
    1. Retrieve relevant entities across multiple hops
    2. Chain evidence across documents
    3. Produce accurate final answers
    """

    def __init__(
        self,
        rag_pipeline=None,
        retriever=None,
        llm=None,
        max_hops: int = 3,
        retrieval_top_k: int = 5,
    ):
        """
        Args:
            rag_pipeline: A callable(query) -> answer string.
            retriever: A callable(query) -> List[str] of retrieved documents.
            llm: LLM instance for answer generation (if retriever-only mode).
            max_hops: Maximum reasoning hops to explore.
            retrieval_top_k: Top-K for retrieval evaluation.
        """
        self.rag_pipeline = rag_pipeline
        self.retriever = retriever
        self.llm = llm
        self.max_hops = max_hops
        self.retrieval_top_k = retrieval_top_k

    def evaluate_case(
        self,
        query: str,
        expected: List[str],
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a single HotpotQA case.

        Returns:
            Tuple of (f1_score, metadata_dict)
        """
        context = context or {}
        supporting_facts = context.get("supporting_facts", [])
        gold_titles = set(context.get("gold_titles", []))

        answer = ""
        retrieved_docs = []

        if self.rag_pipeline:
            answer = self.rag_pipeline(query)
        elif self.retriever and self.llm:
            retrieved_docs = self.retriever(query)
            # Synthesize answer from retrieved context
            ctx_text = "\n".join(retrieved_docs[:5])
            answer = self.llm.generate(
                f"Based on the following context, answer the question.\n"
                f"Context:\n{ctx_text}\n"
                f"Question: {query}\nAnswer:"
            )
        else:
            # No pipeline available — return 0
            return 0.0, {"answer": "", "status": "no_pipeline"}

        # Compute metrics against each expected answer
        best_f1 = 0.0
        best_em = 0.0
        best_answer = ""

        for exp in expected:
            f1 = compute_f1(answer, exp)
            em = compute_exact_match(answer, exp)
            if f1 > best_f1:
                best_f1 = f1
                best_answer = exp
            if em > best_em:
                best_em = em

        # Retrieval metrics (if retriever available)
        retrieval_recall = 0.0
        retrieval_precision = 0.0
        if self.retriever and retrieved_docs:
            retrieval_recall = compute_recall_at_k(
                retrieved_docs, expected, k=self.retrieval_top_k
            )
            retrieval_precision = compute_precision_at_k(
                retrieved_docs, expected, k=self.retrieval_top_k
            )

        # Supporting facts recall (if available)
        support_recall = 0.0
        if supporting_facts and retrieved_docs:
            support_titles = set(f[0] for f in supporting_facts)
            retrieved_titles = set()
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    retrieved_titles.add(doc.get("title", ""))
                elif isinstance(doc, str):
                    retrieved_titles.add(doc[:50])
            if support_titles:
                support_recall = len(
                    support_titles & retrieved_titles
                ) / len(support_titles)

        meta = {
            "answer": answer[:200],
            "expected": best_answer,
            "f1": best_f1,
            "em": best_em,
            "retrieval_recall@5": retrieval_recall,
            "retrieval_precision@5": retrieval_precision,
            "support_recall": support_recall,
            "n_expected": len(expected),
            "n_retrieved": len(retrieved_docs),
            "status": "ok",
        }
        if metadata:
            meta.update(metadata)

        return best_f1, meta


def create_hotpotqa_framework(
    data_path: Optional[str] = None,
    rag_pipeline=None,
    retriever=None,
    llm=None,
    max_cases: int = 100,
) -> BenchmarkFramework:
    """Create a pre-configured HotpotQA benchmark framework.

    Args:
        data_path: Path to HotpotQA JSON file. If None, uses synthetic cases.
        rag_pipeline: RAG pipeline callable.
        retriever: Retriever callable.
        llm: LLM for answer synthesis.
        max_cases: Maximum number of cases to load.

    Returns:
        Configured BenchmarkFramework.
    """
    evaluator = HotpotQAEvaluator(
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        llm=llm,
    )

    framework = BenchmarkFramework(
        name="hotpotqa",
        output_dir="benchmarks/reports/hotpotqa",
    )

    if data_path and os.path.exists(data_path):
        framework.add_cases_from_file(data_path)
        # Limit cases
        if len(framework.cases) > max_cases:
            framework.cases = framework.cases[:max_cases]
    else:
        # Add synthetic test cases for CI/development
        _add_synthetic_hotpotqa_cases(framework)

    return framework


def _add_synthetic_hotpotqa_cases(framework: BenchmarkFramework):
    """Add synthetic HotpotQA-style test cases for development."""
    synthetic_cases = [
        {
            "case_id": "synthetic_001",
            "query": "Which director directed both Inception and Interstellar?",
            "expected": ["Christopher Nolan"],
            "context": {
                "supporting_facts": [["Inception", 0], ["Interstellar", 0]],
                "gold_titles": ["Inception", "Interstellar"],
                "type": "bridge",
            },
        },
        {
            "case_id": "synthetic_002",
            "query": "What is the capital of the country where the Eiffel Tower is located?",
            "expected": ["Paris"],
            "context": {
                "supporting_facts": [["Eiffel Tower", 0], ["France", 0]],
                "gold_titles": ["Eiffel Tower", "France"],
                "type": "bridge",
            },
        },
        {
            "case_id": "synthetic_003",
            "query": "Who was the US president when the iPhone was first released?",
            "expected": ["George W. Bush", "George Bush"],
            "context": {
                "supporting_facts": [["iPhone", 0], ["George W. Bush", 0]],
                "gold_titles": ["iPhone", "George W. Bush"],
                "type": "bridge",
            },
        },
        {
            "case_id": "synthetic_004",
            "query": "In which city is the university where Albert Einstein worked located?",
            "expected": ["Princeton", "Berlin", "Zurich"],
            "context": {
                "supporting_facts": [
                    ["Albert Einstein", 0],
                    ["Princeton University", 0],
                ],
                "gold_titles": ["Albert Einstein", "Princeton University"],
                "type": "bridge",
            },
        },
        {
            "case_id": "synthetic_005",
            "query": "What language is primarily spoken in the country that hosted the 2016 Olympics?",
            "expected": ["Portuguese"],
            "context": {
                "supporting_facts": [["2016 Summer Olympics", 0], ["Brazil", 0]],
                "gold_titles": ["2016 Summer Olympics", "Brazil"],
                "type": "bridge",
            },
        },
    ]

    for case in synthetic_cases:
        framework.add_case(**case)


def run_hotpotqa_benchmark(
    data_path: Optional[str] = None,
    rag_pipeline=None,
    retriever=None,
    llm=None,
    max_cases: int = 100,
    iterations: int = 3,
) -> Dict[str, Any]:
    """Run the full HotpotQA benchmark.

    Returns:
        Summary dict with F1, EM, Recall@5, Precision@5, etc.
    """
    framework = create_hotpotqa_framework(
        data_path=data_path,
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        llm=llm,
        max_cases=max_cases,
    )

    evaluator = HotpotQAEvaluator(
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        llm=llm,
    )

    results = framework.run(evaluator.evaluate_case, iterations=iterations)
    aggregated = framework.aggregate(results)
    summary = framework.summary(aggregated)

    # Compute aggregate F1, EM, recall, precision from metadata
    all_f1 = []
    all_em = []
    all_recall = []
    all_precision = []

    for name, agg in aggregated.items():
        if agg.metadata:
            all_f1.append(agg.metadata.get("f1", 0))
            all_em.append(agg.metadata.get("em", 0))
            all_recall.append(agg.metadata.get("retrieval_recall@5", 0))
            all_precision.append(agg.metadata.get("retrieval_precision@5", 0))

    summary.update({
        "f1_score": sum(all_f1) / len(all_f1) if all_f1 else 0,
        "exact_match": sum(all_em) / len(all_em) if all_em else 0,
        "recall_at_5": sum(all_recall) / len(all_recall) if all_recall else 0,
        "precision_at_5": sum(all_precision) / len(all_precision) if all_precision else 0,
    })

    framework.save_results()
    return summary
