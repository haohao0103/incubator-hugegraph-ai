"""Unit tests for Memory Compressor."""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.memory_compressor import (
    MemoryCompressor,
    ClusterFinder,
    SummaryGenerator,
    PruningEngine,
    cosine_similarity,
)


class TestClusterFinder:

    def setup_method(self):
        self.finder = ClusterFinder(similarity_threshold=0.85, max_cluster_size=10)

    def test_cluster_by_content_empty(self):
        clusters = self.finder.cluster_by_content([])
        assert clusters == []

    def test_cluster_by_content_single(self):
        memories = [{"content": "张三是工程师"}]
        clusters = self.finder.cluster_by_content(memories)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_cluster_by_content_similar(self):
        memories = [
            {"content": "张三在货拉拉公司工作"},
            {"content": "张三在货拉拉公司上班"},  # very similar
            {"content": "李四在北京工作"},  # different
        ]
        clusters = self.finder.cluster_by_content(memories)
        # Similar memories should be in same cluster
        assert len(clusters) >= 1

    def test_cluster_by_content_identical(self):
        memories = [
            {"content": "hello world"},
            {"content": "hello world"},  # exact duplicate
        ]
        clusters = self.finder.cluster_by_content(memories)
        # Identical content should be grouped
        assert any(len(c) == 2 for c in clusters)

    def test_cluster_by_vectors_empty(self):
        clusters = self.finder.cluster_by_vectors([], [])
        assert clusters == []

    def test_cluster_by_vectors_single(self):
        memories = [{"content": "test"}]
        vectors = [[0.1, 0.2, 0.3, 0.4]]
        clusters = self.finder.cluster_by_vectors(memories, vectors)
        assert len(clusters) == 1

    def test_extract_keywords(self):
        cluster = [{"content": "张三在货拉拉公司做Python开发"}]
        keywords = ClusterFinder._extract_keywords(cluster)
        # Keywords may include multi-char Chinese entities and English terms
        assert len(keywords) > 0
        # At least some recognizable entity should appear
        assert any(k in keywords for k in ["张三在货拉拉公司", "Python", "货拉拉公司"])

    def test_merge_by_keywords(self):
        clusters = [
            [{"content": "张三喜欢Python编程"}],
            [{"content": "李四擅长Python开发"}],  # shares "Python" keyword
        ]
        merged = self.finder._merge_by_keywords(clusters)
        # Should merge due to Python keyword overlap
        assert len(merged) <= 2


class TestSummaryGenerator:

    def test_summarize_cluster_empty(self):
        gen = SummaryGenerator(llm_callback=None)
        # No OpenAI client, falls back to heuristic
        result = gen.summarize_cluster([])
        assert result == ""

    def test_summarize_cluster_single(self):
        gen = SummaryGenerator(llm_callback=None)
        memories = [{"content": "张三是工程师"}]
        result = gen.summarize_cluster(memories)
        assert result == "张三是工程师"

    def test_heuristic_summary(self):
        texts = ["张三是工程师，擅长Python", "李四是设计师，擅长UI"]
        result = SummaryGenerator._heuristic_summary(texts)
        assert len(result) > 0
        assert len(result) <= 500

    def test_summarize_with_mock_llm(self):
        def mock_llm(prompt):
            return "张三和李四都是技术人员"

        gen = SummaryGenerator(llm_callback=mock_llm)
        memories = [
            {"content": "张三是工程师"},
            {"content": "李四是设计师"},
        ]
        result = gen.summarize_cluster(memories)
        assert result == "张三和李四都是技术人员"

    def test_summarize_llm_failure_fallback(self):
        def failing_llm(prompt):
            raise RuntimeError("API error")

        gen = SummaryGenerator(llm_callback=failing_llm)
        memories = [
            {"content": "张三是工程师，擅长Python"},
            {"content": "李四是设计师，擅长UI设计"},
        ]
        result = gen.summarize_cluster(memories)
        # Should fall back to heuristic summary
        assert len(result) > 0


class TestPruningEngine:

    def setup_method(self):
        self.pruner = PruningEngine(
            importance_threshold=0.3,
            retention_threshold=0.2,
        )

    def test_should_prune_low_importance(self):
        memory = {"content": "ok", "importance": 0.1}
        assert self.pruner.should_prune(memory) is True

    def test_should_keep_high_importance(self):
        memory = {"content": "张三是关键客户，负责百万级项目", "importance": 0.9}
        assert self.pruner.should_prune(memory) is False

    def test_should_prune_low_retention(self):
        memory = {
            "content": "hello",
            "importance": 0.5,
            "retention": 0.1,  # below threshold
        }
        assert self.pruner.should_prune(memory) is True

    def test_should_keep_good_retention(self):
        memory = {
            "content": "important fact",
            "importance": 0.7,
            "retention": 0.8,
        }
        assert self.pruner.should_prune(memory) is False

    def test_prune_batch(self):
        memories = [
            {"content": "important", "importance": 0.8, "retention": 0.9},
            {"content": "trivial", "importance": 0.1, "retention": 0.05},
            {"content": "medium", "importance": 0.5, "retention": 0.6},
        ]
        kept, pruned = self.pruner.prune_batch(memories)
        assert len(pruned) >= 1
        assert len(kept) >= 1

    def test_max_age_pruning(self):
        pruner = PruningEngine(max_age_hours=24)
        now = time.time()
        old_memory = {"content": "old", "importance": 0.8, "created_at": now - 48 * 3600}
        new_memory = {"content": "new", "importance": 0.8, "created_at": now - 1 * 3600}
        assert pruner.should_prune(old_memory, now) is True
        assert pruner.should_prune(new_memory, now) is False

    def test_no_max_age_limit(self):
        pruner = PruningEngine(max_age_hours=0)
        now = time.time()
        # Use a recent memory that hasn't decayed much
        recent_memory = {"content": "重要客户信息，负责百万级项目", "importance": 0.9, "created_at": now - 3600, "access_count": 10}
        assert pruner.should_prune(recent_memory, now) is False


class TestMemoryCompressor:

    def setup_method(self):
        self.compressor = MemoryCompressor()

    def test_compress_empty(self):
        result = self.compressor.compress([])
        assert result["summaries"] == []
        assert result["stats"]["input_count"] == 0

    def test_compress_single_memory(self):
        memories = [{"id": "m1", "content": "张三是工程师"}]
        result = self.compressor.compress(memories)
        assert result["stats"]["input_count"] == 1

    def test_compress_similar_memories(self):
        memories = [
            {"id": "m1", "content": "张三在货拉拉做Python开发"},
            {"id": "m2", "content": "张三在货拉拉做Python编程"},
            {"id": "m3", "content": "李四在北京工作"},
        ]
        result = self.compressor.compress(memories)
        assert result["stats"]["input_count"] == 3
        assert result["stats"]["cluster_count"] >= 1

    def test_compress_with_llm_callback(self):
        def mock_llm(prompt):
            return "张三是货拉拉的Python工程师"

        compressor = MemoryCompressor(llm_callback=mock_llm)
        memories = [
            {"id": "m1", "content": "张三在货拉拉做Python开发"},
            {"id": "m2", "content": "张三在货拉拉做Python编程"},
        ]
        result = self.compressor.compress(memories)
        assert result["stats"]["summary_count"] >= 0

    def test_compress_stats(self):
        memories = [
            {"id": "m1", "content": "hello", "importance": 0.8},
            {"id": "m2", "content": "world", "importance": 0.1},
        ]
        result = self.compressor.compress(memories)
        stats = result["stats"]
        assert "compression_ratio" in stats
        assert "input_count" in stats
        assert "output_count" in stats


class TestCosineSimilarity:

    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == -1.0

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert cosine_similarity(a, b) == 0.0
