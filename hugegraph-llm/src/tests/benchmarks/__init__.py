"""HugeGraph GraphRAG Benchmarks.

Performance benchmarks for GraphRAG capabilities including:
- HotpotQA: Multi-hop QA evaluation (F1, EM, Recall@5)
- MuSiQue: Compositional reasoning evaluation
- Index Efficiency: Build time, token usage, storage
- Latency: P50/P95/P99 retrieval latency, QPS

Usage:
    python -m pytest src/tests/benchmarks/ -v
"""
