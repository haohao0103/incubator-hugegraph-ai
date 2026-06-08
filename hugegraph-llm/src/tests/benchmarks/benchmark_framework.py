"""HugeGraph GraphRAG Benchmark Framework.

Reusable evaluation framework for measuring GraphRAG performance across
multiple dimensions: answer quality, retrieval latency, index efficiency,
and end-to-end pipeline throughput.
"""

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class AggregateResult:
    """Aggregated results across multiple runs."""

    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    n_runs: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkFramework:
    """Reusable benchmark execution framework.

    Usage:
        framework = BenchmarkFramework(name="hotpotqa_eval")

        # Add test cases
        framework.add_case("simple_fact", query="Who founded Apple?",
                          expected=["Steve Jobs"], retriever=my_retriever)

        # Run with multiple iterations
        results = framework.run(iterations=3, warmup=1)

        # Aggregate statistics
        agg = framework.aggregate(results)
    """

    def __init__(self, name: str, output_dir: str = "benchmarks/reports"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.cases: List[Dict[str, Any]] = []
        self.results: List[List[BenchmarkResult]] = []

    def add_case(
        self,
        case_id: str,
        query: str,
        expected: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a benchmark test case."""
        self.cases.append({
            "case_id": case_id,
            "query": query,
            "expected": expected or [],
            "context": context or {},
            "metadata": metadata or {},
        })

    def add_cases_from_file(self, filepath: str, query_key: str = "question",
                            answer_key: str = "answer", id_key: str = "id"):
        """Load benchmark cases from a JSON/JSONL file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {filepath}")

        cases = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cases.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                cases = json.load(f)
                if isinstance(cases, dict):
                    cases = cases.get("data", cases.get("examples", []))

        for case in cases:
            self.add_case(
                case_id=str(case.get(id_key, len(self.cases))),
                query=case.get(query_key, ""),
                expected=self._extract_expected(case, answer_key),
                metadata={"source": str(filepath)},
            )

    def _extract_expected(self, case: Dict, answer_key: str) -> List[str]:
        """Extract expected answers from a case dict."""
        val = case.get(answer_key, "")
        if isinstance(val, list):
            return [str(v) for v in val]
        if isinstance(val, str):
            # Try to split by common delimiters
            for sep in [" | ", ",", ";"]:
                if sep in val:
                    return [a.strip() for a in val.split(sep) if a.strip()]
            return [val]
        return [str(val)]

    def run_single(
        self,
        case: Dict,
        eval_fn: Callable,
        **kwargs,
    ) -> BenchmarkResult:
        """Execute a single benchmark case."""
        start = time.perf_counter()
        try:
            score, meta = eval_fn(
                query=case["query"],
                expected=case["expected"],
                context=case.get("context", {}),
                metadata=case.get("metadata", {}),
                **kwargs,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return BenchmarkResult(
                name=case["case_id"],
                score=score,
                metadata=meta,
                duration_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return BenchmarkResult(
                name=case["case_id"],
                score=0.0,
                error=str(e),
                duration_ms=elapsed,
            )

    def run(
        self,
        eval_fn: Callable,
        iterations: int = 3,
        warmup: int = 1,
        **kwargs,
    ) -> List[List[BenchmarkResult]]:
        """Run all cases with multiple iterations.

        Args:
            eval_fn: Callable(query, expected, context, metadata, **kwargs)
                     -> Tuple[float, Dict]
            iterations: Number of iterations per case.
            warmup: Number of warmup iterations (not counted).
            **kwargs: Additional arguments passed to eval_fn.

        Returns:
            List of iteration results, each a list of BenchmarkResult.
        """
        self.results = []

        for i in range(warmup + iterations):
            iteration_results = []
            for case in self.cases:
                result = self.run_single(case, eval_fn, **kwargs)
                iteration_results.append(result)

            if i >= warmup:
                self.results.append(iteration_results)

        return self.results

    def aggregate(self, results: Optional[List[List[BenchmarkResult]]] = None
                  ) -> Dict[str, AggregateResult]:
        """Aggregate results across iterations with statistics."""
        if results is None:
            results = self.results
        if not results:
            return {}

        # Group by case name across iterations
        case_scores: Dict[str, List[float]] = {}
        case_durations: Dict[str, List[float]] = {}

        for iteration in results:
            for result in iteration:
                if result.name not in case_scores:
                    case_scores[result.name] = []
                    case_durations[result.name] = []
                case_scores[result.name].append(result.score)
                case_durations[result.name].append(result.duration_ms)

        aggregated = {}
        for name, scores in case_scores.items():
            if not scores:
                continue
            aggregated[name] = AggregateResult(
                name=name,
                mean=statistics.mean(scores),
                std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                min_val=min(scores),
                max_val=max(scores),
                median=statistics.median(scores),
                n_runs=len(scores),
                metadata={
                    "avg_duration_ms": statistics.mean(case_durations[name]),
                },
            )

        return aggregated

    def summary(self, aggregated: Optional[Dict[str, AggregateResult]] = None
                ) -> Dict[str, Any]:
        """Generate a summary of benchmark results."""
        if aggregated is None:
            aggregated = self.aggregate()

        if not aggregated:
            return {"benchmark": self.name, "status": "no results"}

        all_scores = [a.mean for a in aggregated.values()]
        all_durations = [a.metadata.get("avg_duration_ms", 0) for a in aggregated.values()]

        return {
            "benchmark": self.name,
            "total_cases": len(aggregated),
            "overall_mean": statistics.mean(all_scores),
            "overall_std": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            "total_duration_ms": sum(all_durations),
            "avg_duration_ms": statistics.mean(all_durations) if all_durations else 0.0,
            "cases": {
                name: {
                    "mean": a.mean,
                    "std": a.std,
                    "min": a.min_val,
                    "max": a.max_val,
                    "median": a.median,
                    "duration_ms": a.metadata.get("avg_duration_ms", 0),
                }
                for name, a in aggregated.items()
            },
        }

    def save_results(self, results_dir: Optional[str] = None):
        """Save results to JSON files."""
        output = Path(results_dir or self.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        summary = self.summary()

        # Save summary
        summary_path = output / f"{self.name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save detailed results
        if self.results:
            detailed = []
            for i, iteration in enumerate(self.results):
                for result in iteration:
                    detailed.append({
                        "iteration": i,
                        "case": result.name,
                        "score": result.score,
                        "duration_ms": result.duration_ms,
                        "error": result.error,
                        "metadata": result.metadata,
                    })
            detail_path = output / f"{self.name}_detailed.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(detailed, f, indent=2, ensure_ascii=False)

        return str(summary_path)


# --- Evaluation Metrics ---

def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def compute_recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int = 5,
) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 1.0

    retrieved_k = set(r.lower() for r in retrieved[:k])
    relevant_set = set(r.lower() for r in relevant)

    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int = 5,
) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0

    retrieved_k = [r.lower() for r in retrieved[:k]]
    relevant_set = set(r.lower() for r in relevant)

    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / k


def compute_mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    relevant_set = set(r.lower() for r in relevant)
    for i, r in enumerate(retrieved):
        if r.lower() in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def latency_percentiles(
    latencies: List[float],
) -> Dict[str, float]:
    """Compute P50/P95/P99 latency percentiles."""
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "p50": sorted_lat[int(n * 0.50)] if n > 0 else 0,
        "p95": sorted_lat[min(int(n * 0.95), n - 1)] if n > 0 else 0,
        "p99": sorted_lat[min(int(n * 0.99), n - 1)] if n > 0 else 0,
        "mean": statistics.mean(sorted_lat),
        "min": min(sorted_lat),
        "max": max(sorted_lat),
    }
