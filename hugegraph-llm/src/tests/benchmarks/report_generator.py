"""Benchmark Report Generator for HugeGraph GraphRAG.

Generates Markdown reports from benchmark results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from tests.benchmarks.bench_hotpotqa import run_hotpotqa_benchmark
from tests.benchmarks.bench_musique import run_musique_benchmark
from tests.benchmarks.bench_index import run_index_benchmark
from tests.benchmarks.bench_latency import run_latency_benchmark


class BenchmarkReportGenerator:
    """Generate comprehensive benchmark reports.

    Collects results from all benchmark suites and produces
    a unified Markdown report with tables and analysis.
    """

    def __init__(self, output_dir: str = "benchmarks/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}

    def collect_results(
        self,
        hotpotqa_path: Optional[str] = None,
        musique_path: Optional[str] = None,
        n_docs: int = 100,
        n_queries: int = 100,
        rag_pipeline=None,
        retriever=None,
        llm=None,
    ):
        """Collect results from all benchmark suites."""
        print("Collecting benchmark results...")

        # HotpotQA
        try:
            self.results["hotpotqa"] = run_hotpotqa_benchmark(
                data_path=hotpotqa_path,
                rag_pipeline=rag_pipeline,
                retriever=retriever,
                llm=llm,
                max_cases=100,
            )
            print("  HotpotQA: done")
        except Exception as e:
            print(f"  HotpotQA: skipped ({e})")

        # MuSiQue
        try:
            self.results["musique"] = run_musique_benchmark(
                data_path=musique_path,
                rag_pipeline=rag_pipeline,
                retriever=retriever,
                llm=llm,
                max_cases=100,
            )
            print("  MuSiQue: done")
        except Exception as e:
            print(f"  MuSiQue: skipped ({e})")

        # Index Efficiency
        try:
            self.results["index"] = run_index_benchmark(n_docs=n_docs)
            print("  Index Efficiency: done")
        except Exception as e:
            print(f"  Index Efficiency: skipped ({e})")

        # Latency
        try:
            self.results["latency"] = run_latency_benchmark(n_queries=n_queries)
            print("  Latency: done")
        except Exception as e:
            print(f"  Latency: skipped ({e})")

    def generate_report(self) -> str:
        """Generate the full Markdown report."""
        lines = []
        lines.append("# HugeGraph GraphRAG Benchmark Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("- [Answer Quality](#answer-quality)")
        lines.append("- [Index Efficiency](#index-efficiency)")
        lines.append("- [Retrieval Latency](#retrieval-latency)")
        lines.append("- [Summary](#summary)")
        lines.append("")

        # Answer Quality
        lines.append("## Answer Quality")
        lines.append("")

        if "hotpotqa" in self.results:
            lines.append("### HotpotQA (Multi-hop QA)")
            lines.append("")
            self._render_hotpotqa_table(lines, self.results["hotpotqa"])

        if "musique" in self.results:
            lines.append("### MuSiQue (Compositional Reasoning)")
            lines.append("")
            self._render_musique_table(lines, self.results["musique"])

        # Index Efficiency
        lines.append("## Index Efficiency")
        lines.append("")

        if "index" in self.results:
            self._render_index_table(lines, self.results["index"])

        # Latency
        lines.append("## Retrieval Latency")
        lines.append("")

        if "latency" in self.results:
            self._render_latency_table(lines, self.results["latency"])

        # Summary
        lines.append("## Summary")
        lines.append("")
        self._render_summary(lines)

        # Performance Targets
        lines.append("### Performance Targets")
        lines.append("")
        lines.append("| Metric | Current | Target | Status |")
        lines.append("|--------|---------|--------|--------|")
        self._render_targets(lines)
        lines.append("")

        report = "\n".join(lines)

        # Save report
        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Save raw results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nReport saved to: {report_path}")
        print(f"Raw results saved to: {results_path}")

        return report

    def _render_hotpotqa_table(self, lines: List[str], data: Dict):
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| F1 Score | {data.get('f1_score', 'N/A'):.4f} |" if isinstance(data.get('f1_score'), (int, float)) else "| F1 Score | N/A |")
        lines.append(f"| Exact Match | {data.get('exact_match', 'N/A'):.4f} |" if isinstance(data.get('exact_match'), (int, float)) else "| Exact Match | N/A |")
        lines.append(f"| Recall@5 | {data.get('recall_at_5', 'N/A'):.4f} |" if isinstance(data.get('recall_at_5'), (int, float)) else "| Recall@5 | N/A |")
        lines.append(f"| Precision@5 | {data.get('precision_at_5', 'N/A'):.4f} |" if isinstance(data.get('precision_at_5'), (int, float)) else "| Precision@5 | N/A |")
        lines.append(f"| Total Cases | {data.get('total_cases', 'N/A')} |")
        lines.append(f"| Avg Duration | {data.get('avg_duration_ms', 0):.1f} ms |" if isinstance(data.get('avg_duration_ms'), (int, float)) else "")
        lines.append("")

    def _render_musique_table(self, lines: List[str], data: Dict):
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        f1 = data.get("f1_score", "N/A")
        em = data.get("exact_match", "N/A")
        ha = data.get("hop_accuracy", "N/A")
        lines.append(f"| F1 Score | {f1:.4f} |" if isinstance(f1, (int, float)) else "| F1 Score | N/A |")
        lines.append(f"| Exact Match | {em:.4f} |" if isinstance(em, (int, float)) else "| Exact Match | N/A |")
        lines.append(f"| Hop Accuracy | {ha:.4f} |" if isinstance(ha, (int, float)) else "| Hop Accuracy | N/A |")
        lines.append(f"| Total Cases | {data.get('total_cases', 'N/A')} |")
        lines.append("")

    def _render_index_table(self, lines: List[str], data: Dict):
        lines.append("| Phase | Time (s) |")
        lines.append("|-------|----------|")
        phases = [
            ("Chunk Split", "chunk_split_time_s"),
            ("Entity Extraction", "extraction_time_s"),
            ("Graph Build", "graph_build_time_s"),
            ("Vector Index", "vector_index_time_s"),
            ("Community Detection", "community_detect_time_s"),
            ("Community Reports", "community_report_time_s"),
        ]
        for label, key in phases:
            val = data.get(key, "N/A")
            lines.append(f"| {label} | {val:.2f} |" if isinstance(val, (int, float)) else f"| {label} | N/A |")

        lines.append(f"| **Total** | **{data.get('total_time_s', 0):.2f}** |")
        lines.append("")
        lines.append("| Resource | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Documents | {data.get('total_documents', 'N/A')} |")
        lines.append(f"| Chunks | {data.get('total_chunks', 'N/A')} |")
        lines.append(f"| Entities | {data.get('total_entities', 'N/A')} |")
        lines.append(f"| Relations | {data.get('total_relations', 'N/A')} |")
        lines.append(f"| Communities | {data.get('total_communities', 'N/A')} |")
        lines.append(f"| LLM Tokens | {data.get('llm_tokens_used', 'N/A'):,} |" if isinstance(data.get('llm_tokens_used'), (int, float)) else "| LLM Tokens | N/A |")
        lines.append(f"| Embedding Tokens | {data.get('embedding_tokens_used', 'N/A'):,} |" if isinstance(data.get('embedding_tokens_used'), (int, float)) else "| Embedding Tokens | N/A |")
        lines.append(f"| Storage | {data.get('total_storage_bytes', 0) / 1024:.1f} KB |" if isinstance(data.get('total_storage_bytes'), (int, float)) else "")
        lines.append("")

    def _render_latency_table(self, lines: List[str], data: Dict):
        lines.append("| Strategy | P50 (ms) | P95 (ms) | P99 (ms) | QPS |")
        lines.append("|----------|----------|----------|----------|-----|")
        for strategy in ["vector", "graph", "hybrid", "drift"]:
            if strategy in data:
                s = data[strategy]
                lines.append(
                    f"| {strategy} | {s.get('p50_ms', 0):.1f} | "
                    f"{s.get('p95_ms', 0):.1f} | {s.get('p99_ms', 0):.1f} | "
                    f"{s.get('qps', 0):.1f} |"
                )
        lines.append("")

    def _render_summary(self, lines: List[str]):
        if "hotpotqa" in self.results:
            f1 = self.results["hotpotqa"].get("f1_score", 0)
            lines.append(f"- **HotpotQA F1**: {f1:.4f}")
        if "musique" in self.results:
            f1 = self.results["musique"].get("f1_score", 0)
            lines.append(f"- **MuSiQue F1**: {f1:.4f}")
        if "latency" in self.results and "vector" in self.results["latency"]:
            p95 = self.results["latency"]["vector"].get("p95_ms", 0)
            lines.append(f"- **P95 Latency (vector)**: {p95:.1f} ms")
        if "index" in self.results:
            total = self.results["index"].get("total_time_s", 0)
            lines.append(f"- **Index Time**: {total:.2f} s")
        lines.append("")

    def _render_targets(self, lines: List[str]):
        targets = [
            ("HotpotQA F1", "hotpotqa", "f1_score", 0.75),
            ("MuSiQue F1", "musique", "f1_score", 0.38),
            ("P95 Latency", "latency", "vector.p95_ms", 500),
        ]

        for metric_name, suite, key_path, target in targets:
            keys = key_path.split(".")
            val = self.results.get(suite, {})
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k, "N/A")
                else:
                    val = "N/A"
                    break

            if isinstance(val, (int, float)):
                status = "PASS" if val >= target else "FAIL"
                lines.append(f"| {metric_name} | {val:.4f} | {target} | {status} |")
            else:
                lines.append(f"| {metric_name} | N/A | {target} | PENDING |")

    def generate_all(
        self,
        hotpotqa_path: Optional[str] = None,
        musique_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Collect results and generate report."""
        self.collect_results(
            hotpotqa_path=hotpotqa_path,
            musique_path=musique_path,
            **kwargs,
        )
        return self.generate_report()
