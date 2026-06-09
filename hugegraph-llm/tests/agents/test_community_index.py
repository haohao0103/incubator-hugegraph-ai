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

"""Tests for community index build and query operators."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.operators.index_op.build_community_index import (
    BuildCommunityIndex,
    CommunityIndexQuery,
)


# ── Test Helpers ───────────────────────────────────────────────


def _make_mock_embedding(dim=128):
    """Create a mock embedding model."""
    emb = MagicMock()
    emb.get_embedding_dim.return_value = dim
    emb.get_texts_embeddings.return_value = [[0.1] * dim]
    return emb


def _make_mock_vector_index():
    """Create a mock vector index class and instance."""
    index_instance = MagicMock()
    index_cls = MagicMock(return_value=index_instance)
    index_cls.from_name = MagicMock(return_value=index_instance)
    return index_cls, index_instance


def _make_reports(count=3):
    """Create sample community reports."""
    reports = []
    for i in range(count):
        reports.append({
            "community_id": f"C{i}",
            "level": 0,
            "title": f"Community {i}",
            "summary": f"Summary of community {i}.",
            "key_entities": [f"Entity{i}a", f"Entity{i}b"],
            "relationship_patterns": [f"pattern_{i}a"],
            "importance_score": 8.0 - i * 0.5,
        })
    return reports


# ── BuildCommunityIndex Tests ──────────────────────────────────


class TestBuildCommunityIndex:
    """Tests for the BuildCommunityIndex operator."""

    def test_init(self):
        """Test basic initialization."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        builder = BuildCommunityIndex(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        assert builder._vector_index_cls is mock_index_cls
        assert builder._embedding is mock_emb

    def test_run_empty_reports(self):
        """Test that run returns early with empty reports."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        builder = BuildCommunityIndex(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {"community_reports": []}
        result = builder.run(context)

        assert result["community_index_built"] is False
        assert result["community_index_count"] == 0

    def test_run_no_reports_key(self):
        """Test handling when community_reports key is missing."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        builder = BuildCommunityIndex(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {}
        result = builder.run(context)

        assert result["community_index_built"] is False
        assert result["community_index_count"] == 0

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_with_reports(self, mock_huge_settings):
        """Test building index with community reports."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, mock_index = _make_mock_vector_index()
        mock_emb = _make_mock_embedding(128)
        mock_emb.get_texts_embeddings.return_value = [[0.1] * 128, [0.2] * 128, [0.3] * 128]

        builder = BuildCommunityIndex(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        reports = _make_reports(3)
        context = {"community_reports": reports}
        result = builder.run(context)

        assert result["community_index_built"] is True
        assert result["community_index_count"] == 3

        # Verify embed was called with text representations
        mock_emb.get_texts_embeddings.assert_called_once()
        texts_arg = mock_emb.get_texts_embeddings.call_args[0][0]
        assert len(texts_arg) == 3
        for text in texts_arg:
            assert "Title:" in text
            assert "Summary:" in text

        # Verify index interactions
        mock_index_cls.from_name.assert_called_once_with(
            128, "test_graph", "communities"
        )
        mock_index.add.assert_called_once()
        mock_index.save_index_by_name.assert_called_once_with(
            "test_graph", "communities"
        )

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_report_text_content(self, mock_huge_settings):
        """Test the format of generated report text."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        builder = BuildCommunityIndex(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        reports = [
            {
                "community_id": "C0",
                "title": "Engineering",
                "summary": "A group of engineers.",
                "key_entities": ["Alice", "Bob"],
                "relationship_patterns": ["coworker"],
            }
        ]
        builder.run({"community_reports": reports})

        texts = mock_emb.get_texts_embeddings.call_args[0][0]
        assert "Title: Engineering" in texts[0]
        assert "Summary: A group of engineers." in texts[0]
        assert "Key Entities: Alice, Bob" in texts[0]
        assert "Patterns: coworker" in texts[0]

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_adds_json_as_property(self, mock_huge_settings):
        """Test that full report JSON is stored as property."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, mock_index = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        builder = BuildCommunityIndex(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        reports = _make_reports(1)
        builder.run({"community_reports": reports})

        # Check that add was called with JSON properties
        _, props = mock_index.add.call_args[0]
        assert len(props) == 1
        parsed = json.loads(props[0])
        assert parsed["community_id"] == "C0"
        assert parsed["title"] == "Community 0"


# ── CommunityIndexQuery Tests ──────────────────────────────────


class TestCommunityIndexQuery:
    """Tests for the CommunityIndexQuery operator."""

    def test_init(self):
        """Test basic initialization."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
            top_k=5,
        )
        assert querier._vector_index_cls is mock_index_cls
        assert querier._embedding is mock_emb
        assert querier._top_k == 5

    def test_init_default_top_k(self):
        """Test default top_k value."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        assert querier._top_k == 10

    def test_run_empty_query(self):
        """Test that empty query returns empty matches."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {"query": ""}
        result = querier.run(context)

        assert result["community_matches"] == []

    def test_run_no_query_key(self):
        """Test handling when query key is missing."""
        mock_index_cls, _ = _make_mock_vector_index()
        mock_emb = _make_mock_embedding()

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {}
        result = querier.run(context)

        assert result["community_matches"] == []

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_with_query(self, mock_huge_settings):
        """Test querying community index with a search query."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, mock_index = _make_mock_vector_index()
        mock_emb = _make_mock_embedding(128)
        mock_emb.get_texts_embeddings.return_value = [[0.5] * 128]

        # Mock search results - raw JSON strings
        report1 = json.dumps({
            "community_id": "C0",
            "title": "Engineering",
            "summary": "Engineers at Acme.",
            "key_entities": ["Alice", "Bob"],
            "relationship_patterns": ["coworker"],
            "importance_score": 9.0,
        })
        report2 = json.dumps({
            "community_id": "C1",
            "title": "External",
            "summary": "External team.",
            "key_entities": ["Dave"],
            "relationship_patterns": [],
            "importance_score": 5.0,
        })
        mock_index.search.return_value = [report1, report2]

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
            top_k=10,
        )
        context = {"query": "engineering team"}
        result = querier.run(context)

        assert len(result["community_matches"]) == 2
        assert result["community_matches"][0]["community_id"] == "C0"
        assert result["community_matches"][0]["title"] == "Engineering"
        assert result["community_matches"][1]["community_id"] == "C1"

        # Verify embedding and search were called correctly
        mock_emb.get_texts_embeddings.assert_called_once_with(["engineering team"])
        mock_index_cls.from_name.assert_called_once_with(
            128, "test_graph", "communities"
        )
        mock_index.search.assert_called_once()  # search was called

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_search_error_handling(self, mock_huge_settings):
        """Test graceful handling of index search errors."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, mock_index = _make_mock_vector_index()
        mock_index.search.side_effect = RuntimeError("Index read error")
        mock_emb = _make_mock_embedding(128)
        mock_emb.get_texts_embeddings.return_value = [[0.5] * 128]

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {"query": "test query"}
        result = querier.run(context)

        # Should return empty on error
        assert result["community_matches"] == []

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_invalid_json_in_results(self, mock_huge_settings):
        """Test handling of invalid JSON in index results."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, mock_index = _make_mock_vector_index()
        mock_index.search.return_value = ["valid json" * 100]  # invalid JSON
        mock_emb = _make_mock_embedding(128)
        mock_emb.get_texts_embeddings.return_value = [[0.5] * 128]

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {"query": "test"}
        result = querier.run(context)

        # Invalid JSON should be skipped
        assert result["community_matches"] == []

    @patch("hugegraph_llm.operators.index_op.build_community_index.huge_settings")
    def test_run_empty_search_results(self, mock_huge_settings):
        """Test handling of empty search results."""
        mock_huge_settings.graph_name = "test_graph"

        mock_index_cls, mock_index = _make_mock_vector_index()
        mock_index.search.return_value = []
        mock_emb = _make_mock_embedding(128)
        mock_emb.get_texts_embeddings.return_value = [[0.5] * 128]

        querier = CommunityIndexQuery(
            vector_index=mock_index_cls,
            embedding=mock_emb,
        )
        context = {"query": "nonexistent topic"}
        result = querier.run(context)

        assert result["community_matches"] == []
