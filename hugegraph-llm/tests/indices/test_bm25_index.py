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

"""Tests for BM25 keyword index."""

import os
import tempfile

import pytest

from hugegraph_llm.indices.keyword_index import BM25Index, tokenize


class TestTokenize:
    """Tests for the tokenize function."""

    def test_chinese_text(self):
        tokens = tokenize("司机运送实际车型")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_english_text(self):
        tokens = tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_mixed_text(self):
        tokens = tokenize("司机driver运送ship")
        # jieba should produce some tokens
        assert len(tokens) > 0

    def test_empty_string(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_punctuation_removed(self):
        tokens = tokenize("hello, world! test.")
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens
        assert "hello" in tokens

    def test_lowercase(self):
        tokens = tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens


class TestBM25Index:
    """Tests for BM25Index."""

    def _make_index(self) -> BM25Index:
        idx = BM25Index()
        idx.add_documents(
            [
                "司机宽表包含司机ID和物理车型信息",
                "订单宽表记录订单状态和完成时间",
                "司机到达发货地的时间记录在订单宽表",
                "物理车型对应车辆的实际使用车型",
            ],
            ids=["doc1", "doc2", "doc3", "doc4"],
        )
        return idx

    def test_add_documents(self):
        idx = self._make_index()
        assert idx.doc_count == 4

    def test_empty_index(self):
        idx = BM25Index()
        assert idx.doc_count == 0
        assert idx.search("anything") == []

    def test_basic_search(self):
        idx = self._make_index()
        results = idx.search("司机ID", top_k=2)
        assert len(results) > 0
        assert results[0]["score"] > 0
        assert results[0]["id"] == "doc1"  # doc1 has "司机ID"

    def test_search_top_k(self):
        idx = self._make_index()
        results = idx.search("司机", top_k=2)
        assert len(results) <= 2
        # Should return highest scoring docs first
        if len(results) == 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_search_returns_text(self):
        idx = self._make_index()
        results = idx.search("订单")
        assert len(results) > 0
        assert "text" in results[0]
        assert len(results[0]["text"]) > 0

    def test_search_min_score(self):
        idx = self._make_index()
        all_results = idx.search("司机", top_k=10, min_score=0.0)
        filtered = idx.search("司机", top_k=10, min_score=999.0)
        assert len(filtered) <= len(all_results)

    def test_search_with_props(self):
        idx = BM25Index()
        idx.add_documents(
            ["test document content"],
            ids=["d1"],
            props=[{"table": "orders"}],
        )
        results = idx.search("test")
        assert results[0]["prop"]["table"] == "orders"

    def test_no_query(self):
        idx = self._make_index()
        results = idx.search("")
        assert results == []

    def test_remove_documents(self):
        idx = self._make_index()
        removed = idx.remove({"doc1", "doc2"})
        assert removed == 2
        assert idx.doc_count == 2

    def test_remove_nonexistent(self):
        idx = self._make_index()
        removed = idx.remove({"nonexistent"})
        assert removed == 0

    def test_incremental_add(self):
        idx = BM25Index()
        idx.add_documents(["first document"], ids=["d1"])
        assert idx.doc_count == 1
        idx.add_documents(["second document"], ids=["d2"])
        assert idx.doc_count == 2
        results = idx.search("second")
        assert len(results) >= 1
        assert results[0]["id"] == "d2"

    def test_auto_ids(self):
        idx = BM25Index()
        idx.add_documents(["doc one", "doc two"])
        assert idx.doc_count == 2
        results = idx.search("one")
        assert len(results) >= 1
        assert "doc_0" in results[0]["id"]

    def test_persistence(self):
        idx = self._make_index()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch resource_path for this test
            import hugegraph_llm.indices.keyword_index as mod
            old_rp = mod.resource_path
            mod.resource_path = tmpdir
            try:
                idx.save_index_by_name("test_graph", "bm25")
                assert BM25Index.exist("test_graph", "bm25")

                loaded = BM25Index.from_name("test_graph", "bm25")
                assert loaded.doc_count == 4

                results = loaded.search("司机")
                assert len(results) > 0

                # Clean up
                assert BM25Index.clean("test_graph", "bm25")
                assert not BM25Index.exist("test_graph", "bm25")
            finally:
                mod.resource_path = old_rp

    def test_persistence_not_found(self):
        loaded = BM25Index.from_name("nonexistent_graph", "bm25")
        assert loaded.doc_count == 0

    def test_bm25_scoring_order(self):
        """More relevant docs should score higher."""
        idx = BM25Index()
        idx.add_documents(
            [
                "this document is about driver identification",
                "this document is about order processing",
                "driver identification is critical for safety",
            ],
            ids=["a", "b", "c"],
        )
        results = idx.search("driver identification")
        assert len(results) >= 2
        # docs a and c should rank higher than b
        ids = [r["id"] for r in results]
        assert "a" in ids or "c" in ids
        top_id = ids[0]
        assert top_id in ("a", "c")

    def test_chinese_search(self):
        idx = self._make_index()
        results = idx.search("物理车型")
        assert len(results) > 0
        # doc1 mentions 物理车型, doc4 mentions it too
        top_ids = [r["id"] for r in results]
        assert "doc1" in top_ids or "doc4" in top_ids

    def test_scores_are_positive(self):
        idx = self._make_index()
        results = idx.search("司机", top_k=10)
        for r in results:
            assert r["score"] >= 0


class TestBM25IndexQuery:
    """Tests for BM25IndexQuery operator."""

    def test_operator_run(self):
        from hugegraph_llm.operators.index_op.bm25_index_query import BM25IndexQuery

        # Create an index first
        idx = BM25Index()
        idx.add_documents(
            ["司机宽表包含物理车型", "订单宽表记录订单状态"],
            ids=["d1", "d2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            import hugegraph_llm.indices.keyword_index as mod
            old_rp = mod.resource_path
            mod.resource_path = tmpdir
            old_settings = None
            try:
                # Override huge_settings
                import hugegraph_llm.config as cfg
                old_settings = cfg.huge_settings.graph_name
                cfg.huge_settings.graph_name = "test_graph"
                idx.save_index_by_name("test_graph", "bm25")

                query_op = BM25IndexQuery(topk=3, min_score=0.0)
                context = query_op.run({"query": "物理车型"})
                assert "bm25_result" in context
                assert len(context["bm25_result"]) > 0
                assert context["bm25_result"][0]["id"] == "d1"
            finally:
                mod.resource_path = old_rp
                if old_settings:
                    cfg.huge_settings.graph_name = old_settings

    def test_empty_query(self):
        from hugegraph_llm.operators.index_op.bm25_index_query import BM25IndexQuery

        with tempfile.TemporaryDirectory() as tmpdir:
            import hugegraph_llm.config as cfg
            old_settings = cfg.huge_settings.graph_name
            cfg.huge_settings.graph_name = "test_empty"
            try:
                query_op = BM25IndexQuery(topk=5)
                context = query_op.run({"query": ""})
                assert context["bm25_result"] == []
            finally:
                cfg.huge_settings.graph_name = old_settings
