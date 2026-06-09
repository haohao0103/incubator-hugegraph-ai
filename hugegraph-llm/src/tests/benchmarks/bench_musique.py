"""MuSiQue Benchmark for HugeGraph GraphRAG.

Evaluates multi-hop reasoning on the MuSiQue dataset.
MuSiQue requires 2-4 hop compositional reasoning.

Reference: https://github.com/StonyBrookNLP/musique
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from tests.benchmarks.benchmark_framework import (
    BenchmarkFramework,
    compute_f1,
    compute_exact_match,
    AggregateResult,
)


class MuSiQueEvaluator:
    """Evaluator for MuSiQue multi-hop reasoning.

    MuSiQue focuses on compositional reasoning where questions require
    chaining multiple sub-questions to arrive at the final answer.
    """

    def __init__(
        self,
        rag_pipeline=None,
        retriever=None,
        llm=None,
        max_hops: int = 4,
        reasoning_top_k: int = 10,
    ):
        """
        Args:
            rag_pipeline: A callable(query) -> answer string.
            retriever: A callable(query) -> List[str] of retrieved documents.
            llm: LLM for answer generation.
            max_hops: Maximum reasoning depth.
            reasoning_top_k: Top-K for retrieval at each hop.
        """
        self.rag_pipeline = rag_pipeline
        self.retriever = retriever
        self.llm = llm
        self.max_hops = max_hops
        self.reasoning_top_k = reasoning_top_k

    def evaluate_case(
        self,
        query: str,
        expected: List[str],
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a single MuSiQue case.

        Returns:
            Tuple of (f1_score, metadata_dict).
        """
        context = context or {}
        question_decomposition = context.get("question_decomposition", [])
        hop_count = context.get("hop_count", len(question_decomposition))

        answer = ""
        retrieved_docs = []

        if self.rag_pipeline:
            answer = self.rag_pipeline(query)
        elif self.retriever and self.llm:
            retrieved_docs = self.retriever(query)
            ctx_text = "\n".join(retrieved_docs[:self.reasoning_top_k])
            answer = self.llm.generate(
                f"Answer the question based on the context. "
                f"Think step by step.\n"
                f"Context:\n{ctx_text}\n"
                f"Question: {query}\nAnswer:"
            )
        else:
            return 0.0, {
                "answer": "",
                "f1": 0.0,
                "em": 0.0,
                "hop_accuracy": 0.0,
                "status": "no_pipeline",
            }

        # Compute F1 and EM against best matching expected answer
        best_f1 = 0.0
        best_em = 0.0
        for exp in expected:
            f1 = compute_f1(answer, exp)
            em = compute_exact_match(answer, exp)
            best_f1 = max(best_f1, f1)
            best_em = max(best_em, em)

        # Hop-level accuracy (if decomposition available)
        hop_accuracy = 0.0
        if question_decomposition and retrieved_docs:
            correct_hops = 0
            for sub_q in question_decomposition:
                if isinstance(sub_q, str):
                    for doc in retrieved_docs:
                        if isinstance(doc, str) and sub_q.lower() in doc.lower():
                            correct_hops += 1
                            break
            hop_accuracy = correct_hops / len(question_decomposition)

        meta = {
            "answer": answer[:200],
            "f1": best_f1,
            "em": best_em,
            "hop_accuracy": hop_accuracy,
            "hop_count": hop_count,
            "n_retrieved": len(retrieved_docs),
            "status": "ok",
        }
        if metadata:
            meta.update(metadata)

        return best_f1, meta


def create_musique_framework(
    data_path: Optional[str] = None,
    rag_pipeline=None,
    retriever=None,
    llm=None,
    max_cases: int = 100,
) -> BenchmarkFramework:
    """Create a pre-configured MuSiQue benchmark framework."""
    evaluator = MuSiQueEvaluator(
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        llm=llm,
    )

    framework = BenchmarkFramework(
        name="musique",
        output_dir="benchmarks/reports/musique",
    )

    if data_path and os.path.exists(data_path):
        framework.add_cases_from_file(data_path)
        if len(framework.cases) > max_cases:
            framework.cases = framework.cases[:max_cases]
    else:
        _add_synthetic_musique_cases(framework)

    return framework


def _add_synthetic_musique_cases(framework: BenchmarkFramework):
    """Add synthetic MuSiQue-style compositional reasoning cases."""
    synthetic_cases = [
        {
            "case_id": "musique_001",
            "query": "What is the population of the capital of the country where the inventor of the telephone was born?",
            "expected": ["Edinburgh"],
            "context": {
                "question_decomposition": [
                    "Who invented the telephone?",
                    "Where was Alexander Graham Bell born?",
                    "What is the capital of Scotland?",
                    "What is the population of Edinburgh?",
                ],
                "hop_count": 4,
            },
        },
        {
            "case_id": "musique_002",
            "query": "Which award did the author of '1984' receive?",
            "expected": ["Prometheus Award", "retro Hugo"],
            "context": {
                "question_decomposition": [
                    "Who wrote '1984'?",
                    "What awards did George Orwell receive for '1984'?",
                ],
                "hop_count": 2,
            },
        },
        {
            "case_id": "musique_003",
            "query": "What is the GDP of the country where the tallest building in the world is located?",
            "expected": ["UAE"],
            "context": {
                "question_decomposition": [
                    "Where is the tallest building in the world?",
                    "What country is Dubai in?",
                    "What is the GDP of the UAE?",
                ],
                "hop_count": 3,
            },
        },
        {
            "case_id": "musique_004",
            "query": "What sport does the spouse of the founder of Microsoft play?",
            "expected": ["bridge"],
            "context": {
                "question_decomposition": [
                    "Who founded Microsoft?",
                    "Who is Bill Gates married to?",
                    "What sport does Melinda French Gates play?",
                ],
                "hop_count": 3,
            },
        },
        {
            "case_id": "musique_005",
            "query": "In what year was the university that Albert Einstein attended founded?",
            "expected": ["1855", "1848"],
            "context": {
                "question_decomposition": [
                    "Which university did Albert Einstein attend?",
                    "When was ETH Zurich founded?",
                ],
                "hop_count": 2,
            },
        },
    ]

    for case in synthetic_cases:
        framework.add_case(**case)


def run_musique_benchmark(
    data_path: Optional[str] = None,
    rag_pipeline=None,
    retriever=None,
    llm=None,
    max_cases: int = 100,
    iterations: int = 3,
) -> Dict[str, Any]:
    """Run the full MuSiQue benchmark."""
    framework = create_musique_framework(
        data_path=data_path,
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        llm=llm,
        max_cases=max_cases,
    )

    evaluator = MuSiQueEvaluator(
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        llm=llm,
    )

    results = framework.run(evaluator.evaluate_case, iterations=iterations)
    aggregated = framework.aggregate(results)
    summary = framework.summary(aggregated)

    all_f1 = []
    all_em = []
    all_hop_acc = []

    for name, agg in aggregated.items():
        if agg.metadata:
            all_f1.append(agg.metadata.get("f1", 0))
            all_em.append(agg.metadata.get("em", 0))
            all_hop_acc.append(agg.metadata.get("hop_accuracy", 0))

    summary.update({
        "f1_score": sum(all_f1) / len(all_f1) if all_f1 else 0,
        "exact_match": sum(all_em) / len(all_em) if all_em else 0,
        "hop_accuracy": sum(all_hop_acc) / len(all_hop_acc) if all_hop_acc else 0,
    })

    framework.save_results()
    return summary
