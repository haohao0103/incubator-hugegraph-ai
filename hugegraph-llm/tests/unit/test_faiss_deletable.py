"""Unit tests for FAISS Deletable Index."""

import pytest
import sys
import os
import time
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.faiss_deletable import FaissDeletableIndex


def mock_embed(text):
    """Generate deterministic mock embeddings."""
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    # Use dim=8 for speed
    vec = np.array([float(int(h[i:i+2], 16)) / 255.0 for i in range(0, 16, 2)], dtype=np.float32)
    vec = np.pad(vec, (0, 0))  # already 8 dim
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class TestFaissDeletableIndex:
    """Test real FaissDeletableIndex with mocked embedding model."""

    def setup_method(self):
        """Patch the model to avoid needing sentence-transformers."""
        with patch.object(FaissDeletableIndex, '_load_model'):
            self.index = FaissDeletableIndex(dim=8)
        # Mock the model and embed_text
        mock_model = MagicMock()
        mock_model.encode = lambda text, **kwargs: mock_embed(text)
        FaissDeletableIndex._model = mock_model

    def test_init(self):
        assert self.index.dim == 8
        assert self.index._next_id == 0

    def test_add_memory(self):
        int_id = self.index.add_memory("m1", "hello world")
        assert int_id == 0
        assert "m1" in self.index._str_to_int
        assert self.index.index.ntotal == 1

    def test_add_multiple(self):
        self.index.add_memory("m1", "hello")
        self.index.add_memory("m2", "world")
        assert self.index.index.ntotal == 2
        assert len(self.index._str_to_int) == 2

    def test_add_with_metadata(self):
        int_id = self.index.add_memory(
            "m1", "hello", created_at=1000.0, metadata={"user_id": "alice"}
        )
        meta = self.index._int_to_meta[int_id]
        assert meta["created_at"] == 1000.0
        assert meta["metadata"]["user_id"] == "alice"

    def test_remove_by_id(self):
        self.index.add_memory("m1", "hello")
        self.index.add_memory("m2", "world")
        result = self.index.remove_by_id("m1")
        assert result is True
        assert "m1" not in self.index._str_to_int
        assert 0 in self.index._tombstones

    def test_remove_by_id_then_search(self):
        self.index.add_memory("m1", "张三在货拉拉")
        self.index.add_memory("m2", "李四在北京")
        self.index.remove_by_id("m1")
        results = self.index.search("张三", top_k=5)
        assert not any(r["memory_id"] == "m1" for r in results)

    def test_remove_nonexistent(self):
        result = self.index.remove_by_id("nonexistent")
        assert result is False

    def test_remove_by_ids_batch(self):
        self.index.add_memory("m1", "hello")
        self.index.add_memory("m2", "world")
        self.index.add_memory("m3", "foo")
        n = self.index.remove_by_ids(["m1", "m2"])
        assert n == 2
        assert "m1" not in self.index._str_to_int
        assert "m2" not in self.index._str_to_int
        assert "m3" in self.index._str_to_int

    def test_remove_by_ids_empty(self):
        n = self.index.remove_by_ids([])
        assert n == 0

    def test_search_basic(self):
        self.index.add_memory("m1", "张三是工程师")
        self.index.add_memory("m2", "李四是设计师")
        results = self.index.search("张三", top_k=2)
        assert len(results) >= 1

    def test_search_with_ebbinghaus(self):
        self.index.add_memory("m1", "重要记忆")
        results = self.index.search("重要", top_k=1, ebbinghaus_weights={"m1": 0.5})
        if results:
            assert results[0]["retention"] == 0.5

    def test_search_empty_index(self):
        results = self.index.search("hello", top_k=5)
        assert results == []

    def test_get_stats(self):
        self.index.add_memory("m1", "hello")
        self.index.add_memory("m2", "world")
        stats = self.index.get_stats()
        assert stats["total_vectors"] == 2
        assert stats["total_known_ids"] == 2
        assert stats["tombstones"] == 0

    def test_get_stats_after_delete(self):
        self.index.add_memory("m1", "hello")
        self.index.remove_by_id("m1")
        stats = self.index.get_stats()
        assert stats["tombstones"] == 1

    def test_compact(self):
        self.index.add_memory("m1", "hello")
        self.index.add_memory("m2", "world")
        self.index.remove_by_id("m1")
        n = self.index.compact()
        assert n == 1
        stats = self.index.get_stats()
        assert stats["tombstones"] == 0

    def test_embed_text(self):
        vec = self.index.embed_text("hello")
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_faiss_indexidmap_basic(self):
        """Test that IndexIDMap supports add_with_ids and remove_ids."""
        import faiss
        base = faiss.IndexFlatIP(8)
        idx_map = faiss.IndexIDMap(base)
        vec = np.random.rand(1, 8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        idx_map.add_with_ids(vec, np.array([100], dtype=np.int64))
        assert idx_map.ntotal == 1
        # Remove by ID using IDSelectorBatch
        ids_np = np.array([100], dtype=np.int64)
        n = idx_map.remove_ids(faiss.IDSelectorBatch(ids_np.size, faiss.swig_ptr(ids_np)))
        assert n == 1
        assert idx_map.ntotal == 0
