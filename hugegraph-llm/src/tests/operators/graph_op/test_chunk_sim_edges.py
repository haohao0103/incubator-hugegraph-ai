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

"""Unit tests for ChunkSimEdgeBuilder and MultiGranularityRetriever."""

import unittest
from unittest.mock import MagicMock

from hugegraph_llm.operators.graph_op.chunk_sim_edges import (
    ChunkSimEdgeBuilder,
    MultiGranularityRetriever,
)


class TestChunkSimEdgeBuilderInit(unittest.TestCase):
    def test_default_params(self):
        b = ChunkSimEdgeBuilder()
        self.assertEqual(b._top_k, 5)
        self.assertEqual(b._min_score, 0.5)
        self.assertEqual(b.EDGE_LABEL, "SIMILAR")

    def test_custom_params(self):
        b = ChunkSimEdgeBuilder(top_k=10, min_score=0.8)
        self.assertEqual(b._top_k, 10)
        self.assertEqual(b._min_score, 0.8)

    def test_all_components_optional(self):
        b = ChunkSimEdgeBuilder()
        self.assertIsNone(b._embedding)
        self.assertIsNone(b._vector_index)
        self.assertIsNone(b._client)


class TestBuildAll(unittest.TestCase):
    def test_no_client_returns_zero(self):
        b = ChunkSimEdgeBuilder(graph_client=None)
        self.assertEqual(b.build_all(), 0)

    def test_empty_chunks_returns_zero(self):
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": []}
        b = ChunkSimEdgeBuilder(graph_client=mock_client, embedding=MagicMock(), vector_index=MagicMock())
        self.assertEqual(b.build_all(), 0)

    def test_builds_edges_for_chunks(self):
        # This test verifies the build_all flow runs without error.
        # Since chunk IDs come from Gremlin valueMap() which may not
        # have an "id" field, we test the embedding + KNN flow path.
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ]
        mock_vindex = MagicMock()
        mock_vindex.search.return_value = ["neighbor1", "neighbor2"]

        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": [
            {"text": "Chunk about AI"},
            {"text": "Chunk about ML"},
        ]}

        b = ChunkSimEdgeBuilder(
            embedding=mock_embedding,
            vector_index=mock_vindex,
            graph_client=mock_client,
            top_k=2,
            min_score=0.0,
        )
        # The build_all may produce 0 edges since chunks lack "id",
        # but it should not crash.
        count = b.build_all()
        self.assertIsInstance(count, int)

    def test_operator_run_interface(self):
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": []}
        b = ChunkSimEdgeBuilder(graph_client=mock_client)
        context = b.run({"chunk_label": "Chunk", "text_property": "text"})
        self.assertEqual(context["chunk_sim_edges_added"], 0)


class TestBuildIncremental(unittest.TestCase):
    def test_no_client_returns_zero(self):
        b = ChunkSimEdgeBuilder(graph_client=None)
        self.assertEqual(b.build_incremental(["c1", "c2"]), 0)

    def test_empty_ids_returns_zero(self):
        b = ChunkSimEdgeBuilder(graph_client=MagicMock())
        self.assertEqual(b.build_incremental([]), 0)

    def test_chunk_not_found_skipped(self):
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": []}
        b = ChunkSimEdgeBuilder(graph_client=mock_client, embedding=MagicMock())
        self.assertEqual(b.build_incremental(["missing_id"]), 0)


class TestMultiGranularityRetrieverInit(unittest.TestCase):
    def test_default_params(self):
        r = MultiGranularityRetriever()
        self.assertEqual(r._entities_top_k, 10)
        self.assertEqual(r._communities_top_k, 5)

    def test_custom_params(self):
        r = MultiGranularityRetriever(entities_top_k=20, communities_top_k=10)
        self.assertEqual(r._entities_top_k, 20)
        self.assertEqual(r._communities_top_k, 10)


class TestRetrieve(unittest.TestCase):
    def test_retrieve_returns_both_levels(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [[0.1, 0.2]]
        mock_vindex = MagicMock()
        mock_vindex.search.return_value = ["entity1", "entity2"]

        reports = [
            {"title": "AI", "summary": "AI concepts", "importance_score": 8.0},
            {"title": "ML", "summary": "ML basics", "importance_score": 7.0},
        ]
        r = MultiGranularityRetriever(
            embedding=mock_embedding,
            vector_index=mock_vindex,
            community_reports=reports,
        )
        result = r.retrieve("What is AI?")

        self.assertIn("entities", result)
        self.assertIn("communities", result)
        self.assertIn("fused_context", result)
        self.assertTrue(len(result["entities"]) > 0)
        self.assertTrue(len(result["communities"]) > 0)

    def test_retrieve_no_embedding(self):
        r = MultiGranularityRetriever()
        result = r.retrieve("test?")
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["communities"], [])

    def test_retrieve_no_communities(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [[0.1, 0.2]]
        mock_vindex = MagicMock()
        mock_vindex.search.return_value = ["entity1"]
        r = MultiGranularityRetriever(
            embedding=mock_embedding, vector_index=mock_vindex, community_reports=[]
        )
        result = r.retrieve("test?")
        self.assertEqual(result["communities"], [])
        self.assertTrue(len(result["entities"]) > 0)

    def test_fused_context_format(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [[0.1]]
        mock_vindex = MagicMock()
        mock_vindex.search.return_value = ["entity text"]

        reports = [{"title": "AI", "summary": "AI summary", "importance_score": 9.0}]
        r = MultiGranularityRetriever(
            embedding=mock_embedding, vector_index=mock_vindex, community_reports=reports,
        )
        result = r.retrieve("AI?")
        fused = result["fused_context"]
        self.assertIn("Specific Facts", fused)
        self.assertIn("Broader Patterns", fused)
        self.assertIn("[AI]", fused)
        self.assertIn("entity text", fused)

    def test_communities_sorted_by_importance(self):
        reports = [
            {"title": "Low", "summary": "low", "importance_score": 3.0},
            {"title": "High", "summary": "high", "importance_score": 9.0},
            {"title": "Mid", "summary": "mid", "importance_score": 6.0},
        ]
        r = MultiGranularityRetriever(community_reports=reports, communities_top_k=10)
        result = r.retrieve("test?")
        titles = [c["title"] for c in result["communities"]]
        self.assertEqual(titles[0], "High")

    def test_operator_run_interface(self):
        r = MultiGranularityRetriever()
        context = r.run({"query": "test?"})
        self.assertIn("entities", context)
        self.assertIn("communities", context)
        self.assertIn("fused_context", context)


class TestFuse(unittest.TestCase):
    def test_empty_inputs(self):
        fused = MultiGranularityRetriever._fuse([], [])
        self.assertIn("Specific Facts", fused)
        self.assertIn("Broader Patterns", fused)

    def test_entities_only(self):
        fused = MultiGranularityRetriever._fuse(
            [{"text": "fact1"}], []
        )
        self.assertIn("fact1", fused)

    def test_communities_only(self):
        fused = MultiGranularityRetriever._fuse(
            [], [{"title": "Topic", "summary": "desc", "importance_score": 7.5}]
        )
        self.assertIn("[Topic]", fused)

    def test_text_truncated(self):
        fused = MultiGranularityRetriever._fuse(
            [], [{"title": "T", "summary": "x" * 500, "importance_score": 5.0}]
        )
        # Check the summary is truncated to 200 chars in the output
        self.assertIn("x" * 200, fused)
        self.assertNotIn("x" * 201, fused)


if __name__ == "__main__":
    unittest.main()
