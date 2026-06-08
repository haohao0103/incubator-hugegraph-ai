"""Retrieval Latency Benchmark for HugeGraph GraphRAG.

Measures retrieval latency percentiles (P50/P95/P99) and throughput (QPS)
for different retrieval strategies.
"""

import os
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from tests.benchmarks.benchmark_framework import (
    latency_percentiles,
)


@dataclass
class LatencyProfile:
    """Latency profile for a retrieval strategy."""

    strategy_name: str
    n_queries: int = 0
    total_time_s: float = 0.0
    qps: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatencyBenchmark:
    """Benchmark for measuring retrieval latency and throughput.

    Tests each retrieval strategy:
    1. Vector-only search
    2. Graph traversal (Gremlin)
    3. Hybrid (vector + graph fusion)
    4. DRIFT search (multi-hop deep search)

    For each strategy, measures:
    - P50/P95/P99 latency
    - QPS (queries per second)
    - Error rate
    """

    def __init__(
        self,
        vector_search_fn: Optional[Callable] = None,
        graph_search_fn: Optional[Callable] = None,
        hybrid_search_fn: Optional[Callable] = None,
        drift_search_fn: Optional[Callable] = None,
    ):
        """
        Args:
            vector_search_fn: Callable(query, top_k) -> List[results]
            graph_search_fn: Callable(query, max_depth) -> List[results]
            hybrid_search_fn: Callable(query, top_k, max_depth) -> List[results]
            drift_search_fn: Callable(query) -> Dict (drift result)
        """
        self.strategies = {
            "vector": vector_search_fn,
            "graph": graph_search_fn,
            "hybrid": hybrid_search_fn,
            "drift": drift_search_fn,
        }

    def generate_queries(self, n: int = 100) -> List[str]:
        """Generate benchmark queries."""
        templates = [
            "Who is the CEO of {company}?",
            "What is the population of {city}?",
            "When was {entity} founded?",
            "What products does {company} make?",
            "Who invented {concept}?",
            "Where is {entity} located?",
            "What is the relationship between {a} and {b}?",
            "How does {concept} affect {entity}?",
            "What are the competitors of {company}?",
            "What awards has {person} received?",
        ]

        fillers = {
            "company": ["Google", "Apple", "Microsoft", "Amazon", "Tesla"],
            "city": ["Tokyo", "London", "New York", "Paris", "Berlin"],
            "entity": ["Python", "Linux", "Bitcoin", "the Internet", "GPS"],
            "concept": ["gravity", "relativity", "quantum computing", "AI"],
            "a": ["Albert Einstein", "Nikola Tesla", "Marie Curie"],
            "b": ["Thomas Edison", "Isaac Newton", "Charles Darwin"],
            "person": ["Elon Musk", "Tim Cook", "Satya Nadella"],
        }

        import random
        queries = []
        for i in range(n):
            template = templates[i % len(templates)]
            q = template
            for key, values in fillers.items():
                q = q.replace("{" + key + "}", random.choice(values))
            queries.append(q)
        return queries

    def benchmark_strategy(
        self,
        strategy_name: str,
        search_fn: Callable,
        queries: List[str],
        warmup: int = 5,
        **kwargs,
    ) -> LatencyProfile:
        """Benchmark a single retrieval strategy."""
        profile = LatencyProfile(strategy_name=strategy_name)
        latencies = []

        # Warmup
        for q in queries[:warmup]:
            try:
                search_fn(q, **kwargs)
            except Exception:
                pass

        # Measure
        start_total = time.perf_counter()
        for q in queries[warmup:]:
            start = time.perf_counter()
            try:
                search_fn(q, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
            except Exception:
                profile.errors += 1
        total_time = time.perf_counter() - start_total

        if latencies:
            percentiles = latency_percentiles(latencies)
            profile.p50_ms = percentiles["p50"]
            profile.p95_ms = percentiles["p95"]
            profile.p99_ms = percentiles["p99"]
            profile.mean_ms = percentiles["mean"]
            profile.min_ms = percentiles["min"]
            profile.max_ms = percentiles["max"]
            profile.latencies = latencies
            profile.qps = len(queries[warmup:]) / total_time if total_time > 0 else 0

        profile.n_queries = len(queries[warmup:])
        profile.total_time_s = total_time

        return profile

    def run_all(
        self,
        n_queries: int = 100,
        warmup: int = 5,
    ) -> Dict[str, LatencyProfile]:
        """Run benchmarks for all available strategies."""
        queries = self.generate_queries(n_queries)
        results = {}

        for name, fn in self.strategies.items():
            if fn is not None:
                results[name] = self.benchmark_strategy(
                    name, fn, queries, warmup=warmup
                )

        return results

    def run_synthetic(
        self,
        n_queries: int = 100,
    ) -> Dict[str, LatencyProfile]:
        """Run synthetic benchmarks without real services.

        Produces estimated latency profiles based on typical values.
        """
        import random

        results = {}

        # Vector search: typically 10-50ms
        vector_latencies = [
            random.gauss(25, 8) for _ in range(n_queries)
        ]
        results["vector"] = LatencyProfile(
            strategy_name="vector",
            n_queries=n_queries,
            latencies=vector_latencies,
            metadata={"mode": "synthetic"},
        )

        # Graph traversal: typically 50-200ms
        graph_latencies = [
            random.gauss(100, 40) for _ in range(n_queries)
        ]
        results["graph"] = LatencyProfile(
            strategy_name="graph",
            n_queries=n_queries,
            latencies=graph_latencies,
            metadata={"mode": "synthetic"},
        )

        # Hybrid: typically 100-300ms
        hybrid_latencies = [
            random.gauss(180, 60) for _ in range(n_queries)
        ]
        results["hybrid"] = LatencyProfile(
            strategy_name="hybrid",
            n_queries=n_queries,
            latencies=hybrid_latencies,
            metadata={"mode": "synthetic"},
        )

        # DRIFT: typically 500-2000ms (multi-hop)
        drift_latencies = [
            random.gauss(1200, 400) for _ in range(n_queries)
        ]
        results["drift"] = LatencyProfile(
            strategy_name="drift",
            n_queries=n_queries,
            latencies=drift_latencies,
            metadata={"mode": "synthetic"},
        )

        # Compute percentiles for all
        for name, profile in results.items():
            percentiles = latency_percentiles(profile.latencies)
            profile.p50_ms = percentiles["p50"]
            profile.p95_ms = percentiles["p95"]
            profile.p99_ms = percentiles["p99"]
            profile.mean_ms = percentiles["mean"]
            profile.min_ms = percentiles["min"]
            profile.max_ms = percentiles["max"]
            profile.total_time_s = sum(profile.latencies) / 1000
            profile.qps = (
                profile.n_queries / profile.total_time_s
                if profile.total_time_s > 0 else 0
            )

        return results


def run_latency_benchmark(
    n_queries: int = 100,
    vector_fn=None,
    graph_fn=None,
    hybrid_fn=None,
    drift_fn=None,
) -> Dict[str, Any]:
    """Run latency benchmark and return summary.

    Returns dict with per-strategy latency profiles.
    """
    bench = LatencyBenchmark(
        vector_search_fn=vector_fn,
        graph_search_fn=graph_fn,
        hybrid_search_fn=hybrid_fn,
        drift_search_fn=drift_fn,
    )

    # Use real functions if provided, otherwise synthetic
    if any([vector_fn, graph_fn, hybrid_fn, drift_fn]):
        results = bench.run_all(n_queries=n_queries)
    else:
        results = bench.run_synthetic(n_queries=n_queries)

    # Format as summary
    summary = {"benchmark": "latency", "n_queries": n_queries}
    for name, profile in results.items():
        summary[name] = {
            "qps": round(profile.qps, 2),
            "p50_ms": round(profile.p50_ms, 2),
            "p95_ms": round(profile.p95_ms, 2),
            "p99_ms": round(profile.p99_ms, 2),
            "mean_ms": round(profile.mean_ms, 2),
            "min_ms": round(profile.min_ms, 2),
            "max_ms": round(profile.max_ms, 2),
            "errors": profile.errors,
        }

    return summary
