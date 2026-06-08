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

"""Unit tests for EntityResolution operator.

Tests cover:
- UnionFind data structure
- Exact match strategy
- Embedding similarity strategy
- LLM verification strategy
- Hybrid strategy
- Transitive merging
- Edge cases (empty input, single vertex, no duplicates, etc.)
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.operators.graph_op.entity_resolution import (
    EntityResolution,
    MergeCandidate,
    MergeResult,
    UnionFind,
)


# ---------------------------------------------------------------------------
# UnionFind tests
# ---------------------------------------------------------------------------

class TestUnionFind(unittest.TestCase):
    """Test UnionFind data structure."""

    def test_single_element(self):
        uf = UnionFind()
        uf.find("a")
        self.assertEqual(uf.groups(), [])

    def test_two_elements_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        self.assertEqual(uf.find("a"), uf.find("b"))
        groups = uf.groups()
        self.assertEqual(len(groups), 1)
        self.assertIn("a", groups[0])
        self.assertIn("b", groups[0])

    def test_transitive_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        # a-b-c should all be in the same group
        self.assertEqual(uf.find("a"), uf.find("c"))
        groups = uf.groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {"a", "b", "c"})

    def test_multiple_disjoint_sets(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        groups = uf.groups()
        self.assertEqual(len(groups), 2)
        group_set = [frozenset(g) for g in groups]
        self.assertIn(frozenset({"a", "b"}), group_set)
        self.assertIn(frozenset({"c", "d"}), group_set)

    def test_path_compression(self):
        uf = UnionFind()
        for i in range(100):
            uf.union(str(i), str(i + 1))
        # After path compression, all should have same root
        root = uf.find("0")
        for i in range(100):
            self.assertEqual(uf.find(str(i)), root)

    def test_self_union(self):
        uf = UnionFind()
        uf.union("a", "a")
        groups = uf.groups()
        self.assertEqual(groups, [])  # Single element -> no group


# ---------------------------------------------------------------------------
# MergeResult tests
# ---------------------------------------------------------------------------

class TestMergeResult(unittest.TestCase):
    """Test MergeResult data structure."""

    def test_empty_result(self):
        result = MergeResult()
        d = result.to_dict()
        self.assertEqual(d["merged_count"], 0)
        self.assertEqual(d["deprecated_vids"], [])
        self.assertEqual(d["edges_migrated"], 0)
        self.assertEqual(d["errors"], [])

    def test_populated_result(self):
        result = MergeResult(
            merged_pairs=[{"from_vid": "v1", "to_vid": "v2"}],
            merged_count=1,
            deprecated_vids=["v1"],
            edges_migrated=3,
            errors=[],
        )
        d = result.to_dict()
        self.assertEqual(len(d["merged_pairs"]), 1)
        self.assertEqual(d["merged_count"], 1)


# ---------------------------------------------------------------------------
# MergeCandidate tests
# ---------------------------------------------------------------------------

class TestMergeCandidate(unittest.TestCase):
    """Test MergeCandidate data structure."""

    def test_creation(self):
        c = MergeCandidate(
            from_vid="v1", from_label="Person",
            from_properties={"name": "Alice"},
            to_vid="v2", to_label="Person",
            to_properties={"name": "Alice", "age": "30"},
            strategy="exact_match", confidence=1.0,
        )
        self.assertEqual(c.from_vid, "v1")
        self.assertEqual(c.to_vid, "v2")
        self.assertEqual(c.strategy, "exact_match")
        self.assertEqual(c.confidence, 1.0)


# ---------------------------------------------------------------------------
# EntityResolution - strategy validation tests
# ---------------------------------------------------------------------------

class TestEntityResolutionInit(unittest.TestCase):
    """Test EntityResolution initialization."""

    def _make_client(self):
        return MagicMock()

    def test_valid_strategies(self):
        client = self._make_client()
        for strategy in ["exact_match", "embedding", "llm_verify", "hybrid"]:
            resolver = EntityResolution(client=client, strategy=strategy)
            self.assertEqual(resolver._strategy, strategy)

    def test_invalid_strategy_raises(self):
        client = self._make_client()
        with self.assertRaises(ValueError):
            EntityResolution(client=client, strategy="invalid")


# ---------------------------------------------------------------------------
# EntityResolution - exact match tests
# ---------------------------------------------------------------------------

class TestExactMatchStrategy(unittest.TestCase):
    """Test exact match strategy with mock client."""

    def _make_resolver(self, strategy="exact_match"):
        client = MagicMock()
        return EntityResolution(client=client, strategy=strategy), client

    def test_exact_match_finds_duplicates(self):
        resolver, client = self._make_resolver("exact_match")

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
            {"id": "v3", "label": "Person", "properties": {"name": "Bob"}, "degree": 1},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        # Should find one merge pair: v2 -> v1 (higher degree)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].from_vid, "v2")
        self.assertEqual(candidates[0].to_vid, "v1")
        self.assertEqual(candidates[0].confidence, 1.0)

    def test_exact_match_no_duplicates(self):
        resolver, client = self._make_resolver("exact_match")

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 1},
            {"id": "v2", "label": "Person", "properties": {"name": "Bob"}, "degree": 1},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        self.assertEqual(len(candidates), 0)

    def test_exact_match_different_labels_not_merged(self):
        resolver, client = self._make_resolver("exact_match")

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Apple"}, "degree": 1},
            {"id": "v2", "label": "Organization", "properties": {"name": "Apple"}, "degree": 1},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        self.assertEqual(len(candidates), 0)

    def test_exact_match_keeps_highest_degree(self):
        resolver, client = self._make_resolver("exact_match")

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 10},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v3", "label": "Person", "properties": {"name": "Alice"}, "degree": 20},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        # Should merge v1 and v2 into v3 (highest degree)
        self.assertEqual(len(candidates), 2)
        to_vids = {c.to_vid for c in candidates}
        self.assertEqual(to_vids, {"v3"})

    def test_exact_match_multiple_properties(self):
        resolver, client = self._make_resolver("exact_match")
        resolver._resolve_properties = ["name", "age"]

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice", "age": "25"}, "degree": 3},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice", "age": "25"}, "degree": 1},
            {"id": "v3", "label": "Person", "properties": {"name": "Alice", "age": "30"}, "degree": 2},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        # Only v1 and v2 match (both name=Alice, age=25)
        self.assertEqual(len(candidates), 1)


# ---------------------------------------------------------------------------
# EntityResolution - embedding strategy tests
# ---------------------------------------------------------------------------

class TestEmbeddingStrategy(unittest.TestCase):
    """Test embedding similarity strategy."""

    def _make_resolver(self):
        client = MagicMock()
        embedding = MagicMock()
        # Return identical embeddings for similar names
        embedding.get_text_embedding = MagicMock(side_effect=lambda text: [0.1, 0.2, 0.3, 0.4])
        resolver = EntityResolution(
            client=client, embedding=embedding,
            strategy="embedding", threshold=0.99,  # High threshold
        )
        return resolver, client, embedding

    def test_embedding_identical_text(self):
        resolver, client, embedding = self._make_resolver()

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        # Identical embeddings should produce cos_sim = 1.0 > 0.99
        self.assertEqual(len(candidates), 1)

    def test_embedding_different_text(self):
        resolver, client, embedding = self._make_resolver()
        # Return orthogonal embeddings
        embedding.get_text_embedding = MagicMock(side_effect=lambda text: {
            "Alice": [1.0, 0.0, 0.0],
            "Bob": [0.0, 1.0, 0.0],
        }.get(text, [0.0, 0.0, 0.0]))

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 1},
            {"id": "v2", "label": "Person", "properties": {"name": "Bob"}, "degree": 1},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        # Orthogonal vectors -> cos_sim = 0 < threshold
        self.assertEqual(len(candidates), 0)

    def test_embedding_no_model_raises_warning(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="embedding")
        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 1},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 1},
        ]
        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)
        # Should produce 0 candidates because embedding is None
        self.assertEqual(len(candidates), 0)


# ---------------------------------------------------------------------------
# EntityResolution - LLM verification tests
# ---------------------------------------------------------------------------

class TestLLMVerifyStrategy(unittest.TestCase):
    """Test LLM verification of candidates."""

    def test_llm_verify_confirms(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(return_value="[true]")

        resolver = EntityResolution(
            client=client, llm=llm, strategy="llm_verify",
        )

        candidates = [
            MergeCandidate(
                from_vid="v1", from_label="Person", from_properties={"name": "US"},
                to_vid="v2", to_label="Person", to_properties={"name": "United States"},
                strategy="exact_match", confidence=1.0,
            ),
        ]

        verified = resolver._verify_candidates(candidates)
        # exact_match candidates (confidence=1.0) should be passed through
        self.assertEqual(len(verified), 1)

    def test_llm_verify_with_embedding_candidates(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(return_value="[true, false]")

        resolver = EntityResolution(
            client=client, llm=llm, strategy="llm_verify",
        )

        candidates = [
            MergeCandidate(
                from_vid="v1", from_label="Person", from_properties={"name": "US"},
                to_vid="v2", to_label="Person", to_properties={"name": "United States"},
                strategy="embedding", confidence=0.85,
            ),
            MergeCandidate(
                from_vid="v3", from_label="Person", from_properties={"name": "Alice"},
                to_vid="v4", to_label="Person", to_properties={"name": "Alicia"},
                strategy="embedding", confidence=0.88,
            ),
        ]

        verified = resolver._verify_candidates(candidates)
        self.assertEqual(len(verified), 1)  # Only first confirmed
        self.assertEqual(verified[0].from_vid, "v1")

    def test_llm_verify_json_parse_error(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(return_value="not json")

        resolver = EntityResolution(
            client=client, llm=llm, strategy="llm_verify",
        )

        candidates = [
            MergeCandidate(
                from_vid="v1", from_label="Person", from_properties={"name": "US"},
                to_vid="v2", to_label="Person", to_properties={"name": "United States"},
                strategy="embedding", confidence=0.85,
            ),
        ]

        # Fail-open: accept all on parse error
        verified = resolver._verify_candidates(candidates)
        self.assertEqual(len(verified), 1)

    def test_llm_verify_no_llm(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="llm_verify")

        candidates = [
            MergeCandidate(
                from_vid="v1", from_label="Person", from_properties={"name": "US"},
                to_vid="v2", to_label="Person", to_properties={"name": "United States"},
                strategy="embedding", confidence=0.85,
            ),
        ]

        # Should return all candidates unchanged
        verified = resolver._verify_candidates(candidates)
        self.assertEqual(len(verified), 1)


# ---------------------------------------------------------------------------
# EntityResolution - hybrid strategy tests
# ---------------------------------------------------------------------------

class TestHybridStrategy(unittest.TestCase):
    """Test hybrid (all strategies combined) resolution."""

    def test_hybrid_combines_exact_and_embedding(self):
        client = MagicMock()
        embedding = MagicMock()
        embedding.get_text_embedding = MagicMock(side_effect=lambda text: [0.1, 0.2, 0.3, 0.4])

        resolver = EntityResolution(
            client=client, embedding=embedding, strategy="hybrid", threshold=0.99,
        )

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
        ]

        groups = resolver._group_by_label(vertices)
        candidates = resolver._find_candidates(groups)

        # Hybrid should find at least the exact match
        self.assertGreaterEqual(len(candidates), 1)


# ---------------------------------------------------------------------------
# EntityResolution - in-memory resolution tests
# ---------------------------------------------------------------------------

class TestInMemoryResolution(unittest.TestCase):
    """Test full in-memory resolution pipeline."""

    def test_resolve_empty_vertices(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="exact_match")
        result = resolver.run({"vertices": [], "schema": None})
        self.assertIn("resolution_result", result)
        self.assertEqual(result["resolution_result"]["merged_count"], 0)

    def test_resolve_single_vertex(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="exact_match")
        result = resolver.run({
            "vertices": [{"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 1}],
        })
        self.assertEqual(result["resolution_result"]["merged_count"], 0)

    def test_resolve_with_label_filter(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="exact_match")

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
            {"id": "v3", "label": "Organization", "properties": {"name": "Alice"}, "degree": 1},
        ]

        result = resolver.run({
            "vertices": vertices,
            "vertex_labels": ["Person"],  # Only resolve Person
        })

        self.assertEqual(result["resolution_result"]["merged_count"], 1)

    def test_resolve_preserves_audit_trail(self):
        client = MagicMock()
        # Mock graph operations
        client.gremlin.return_value.exec = MagicMock(return_value=[])

        resolver = EntityResolution(client=client, strategy="exact_match")
        # The merge phase calls graph client methods, but for in-memory
        # with no graph store, it only affects the context result
        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
        ]

        result = resolver.run({"vertices": vertices})
        self.assertIn("resolution_result", result)
        self.assertIn("deprecated_vids", result["resolution_result"])


# ---------------------------------------------------------------------------
# EntityResolution - graph store resolution tests (mocked)
# ---------------------------------------------------------------------------

class TestGraphStoreResolution(unittest.TestCase):
    """Test resolution from HugeGraph store with mocked client."""

    def _setup_mock_client(self, vertices=None):
        client = MagicMock()
        schema_mock = MagicMock()
        schema_mock.getVertexLabels.return_value = ["Person", "Organization"]
        client.schema.return_value = schema_mock

        if vertices is None:
            vertices = [
                {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
                {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
                {"id": "v3", "label": "Person", "properties": {"name": "Bob"}, "degree": 1},
            ]

        client.gremlin.return_value.exec = MagicMock(return_value={"data": vertices})
        return client

    def test_graph_store_resolution(self):
        client = self._setup_mock_client()
        resolver = EntityResolution(client=client, strategy="exact_match")
        result = resolver.run({})
        self.assertIn("resolution_result", result)

    def test_graph_store_no_labels(self):
        client = MagicMock()
        schema_mock = MagicMock()
        schema_mock.getVertexLabels.return_value = []
        client.schema.return_value = schema_mock
        resolver = EntityResolution(client=client, strategy="exact_match")
        result = resolver.run({})
        self.assertEqual(result["resolution_result"]["merged_count"], 0)


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        sim = EntityResolution._cosine_similarity([1, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(sim, 1.0)

    def test_orthogonal_vectors(self):
        sim = EntityResolution._cosine_similarity([1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(sim, 0.0)

    def test_opposite_vectors(self):
        sim = EntityResolution._cosine_similarity([1, 0, 0], [-1, 0, 0])
        self.assertAlmostEqual(sim, -1.0)

    def test_zero_vector(self):
        sim = EntityResolution._cosine_similarity([0, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(sim, 0.0)

    def test_similar_vectors(self):
        sim = EntityResolution._cosine_similarity([0.9, 0.1], [1.0, 0.0])
        self.assertGreater(sim, 0.99)


if __name__ == "__main__":
    unittest.main()
