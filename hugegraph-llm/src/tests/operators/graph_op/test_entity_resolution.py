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

import unittest
from unittest.mock import MagicMock

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
        self.assertEqual(d["synonym_edges"], 0)
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
            from_vid="v1",
            from_label="Person",
            from_properties={"name": "Alice"},
            to_vid="v2",
            to_label="Person",
            to_properties={"name": "Alice", "age": "30"},
            strategy="exact_match",
            confidence=1.0,
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
            client=client,
            embedding=embedding,
            strategy="embedding",
            threshold=0.99,  # High threshold
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
        embedding.get_text_embedding = MagicMock(
            side_effect=lambda text: {
                "Alice": [1.0, 0.0, 0.0],
                "Bob": [0.0, 1.0, 0.0],
            }.get(text, [0.0, 0.0, 0.0])
        )

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
            client=client,
            llm=llm,
            strategy="llm_verify",
        )

        candidates = [
            MergeCandidate(
                from_vid="v1",
                from_label="Person",
                from_properties={"name": "US"},
                to_vid="v2",
                to_label="Person",
                to_properties={"name": "United States"},
                strategy="exact_match",
                confidence=1.0,
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
            client=client,
            llm=llm,
            strategy="llm_verify",
        )

        candidates = [
            MergeCandidate(
                from_vid="v1",
                from_label="Person",
                from_properties={"name": "US"},
                to_vid="v2",
                to_label="Person",
                to_properties={"name": "United States"},
                strategy="embedding",
                confidence=0.85,
            ),
            MergeCandidate(
                from_vid="v3",
                from_label="Person",
                from_properties={"name": "Alice"},
                to_vid="v4",
                to_label="Person",
                to_properties={"name": "Alicia"},
                strategy="embedding",
                confidence=0.88,
            ),
        ]

        verified = resolver._verify_candidates(candidates)
        self.assertEqual(len(verified), 1)  # Only first confirmed
        self.assertEqual(verified[0].from_vid, "v1")

    def test_llm_verify_json_parse_error_fail_closed(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(return_value="not json")

        resolver = EntityResolution(
            client=client,
            llm=llm,
            strategy="llm_verify",
        )

        candidates = [
            MergeCandidate(
                from_vid="v1",
                from_label="Person",
                from_properties={"name": "US"},
                to_vid="v2",
                to_label="Person",
                to_properties={"name": "United States"},
                strategy="embedding",
                confidence=0.85,
            ),
        ]

        # Default fail-closed: reject the whole batch on unparseable response
        verified = resolver._verify_candidates(candidates)
        self.assertEqual(len(verified), 0)

    def test_llm_verify_json_parse_error_fail_open(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(return_value="not json")

        resolver = EntityResolution(
            client=client,
            llm=llm,
            strategy="llm_verify",
            llm_fail_open=True,
        )

        candidates = [
            MergeCandidate(
                from_vid="v1",
                from_label="Person",
                from_properties={"name": "US"},
                to_vid="v2",
                to_label="Person",
                to_properties={"name": "United States"},
                strategy="embedding",
                confidence=0.85,
            ),
        ]

        # Legacy fail-open (opt-in): accept all on parse error
        verified = resolver._verify_candidates(candidates)
        self.assertEqual(len(verified), 1)

    def test_llm_verify_no_llm(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="llm_verify")

        candidates = [
            MergeCandidate(
                from_vid="v1",
                from_label="Person",
                from_properties={"name": "US"},
                to_vid="v2",
                to_label="Person",
                to_properties={"name": "United States"},
                strategy="embedding",
                confidence=0.85,
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
            client=client,
            embedding=embedding,
            strategy="hybrid",
            threshold=0.99,
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
        result = resolver.run(
            {
                "vertices": [{"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 1}],
            }
        )
        self.assertEqual(result["resolution_result"]["merged_count"], 0)

    def test_resolve_with_label_filter(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="exact_match")

        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Alice"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Alice"}, "degree": 2},
            {"id": "v3", "label": "Organization", "properties": {"name": "Alice"}, "degree": 1},
        ]

        result = resolver.run(
            {
                "vertices": vertices,
                "vertex_labels": ["Person"],  # Only resolve Person
            }
        )

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


# ---------------------------------------------------------------------------
# Pure in-memory resolution (pre-commit) tests
# ---------------------------------------------------------------------------


class TestResolveInMemoryPure(unittest.TestCase):
    """Test resolve_in_memory_pure: dedupe + edge repointing without graph access."""

    def _make_resolver(self):
        return EntityResolution(client=MagicMock(), strategy="exact_match")

    def test_exact_match_dedups_and_repoints_edges(self):
        resolver = self._make_resolver()
        vertices = [
            {"id": "p:A", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "p:B", "label": "Person", "properties": {"name": "Alice"}},
        ]
        edges = [
            {"label": "knows", "outV": "p:A", "inV": "p:C", "properties": {}},
            {"label": "knows", "outV": "p:A", "inV": "p:D", "properties": {}},
            {"label": "knows", "outV": "p:E", "inV": "p:B", "properties": {}},
        ]

        out = resolver.resolve_in_memory_pure(vertices, edges)

        # p:A (degree 2) is kept, p:B (degree 1) is deprecated.
        self.assertEqual([v["id"] for v in out["vertices"]], ["p:A"])
        # The edge pointing at p:B is repointed to p:A.
        repointed = [e for e in out["edges"] if e["outV"] == "p:E"]
        self.assertEqual(len(repointed), 1)
        self.assertEqual(repointed[0]["inV"], "p:A")
        self.assertEqual(len(out["edges"]), 3)
        self.assertEqual(out["resolution_result"]["merged_count"], 1)
        self.assertEqual(out["resolution_result"]["deprecated_vids"], ["p:B"])
        self.assertEqual(out["resolution_result"]["edges_migrated"], 1)

    def test_no_duplicates_returns_input_unchanged(self):
        resolver = self._make_resolver()
        vertices = [
            {"id": "p:A", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "p:B", "label": "Person", "properties": {"name": "Bob"}},
        ]
        edges = [{"label": "knows", "outV": "p:A", "inV": "p:B", "properties": {}}]

        out = resolver.resolve_in_memory_pure(vertices, edges)

        self.assertEqual([v["id"] for v in out["vertices"]], ["p:A", "p:B"])
        self.assertEqual(out["edges"], edges)
        self.assertEqual(out["resolution_result"]["merged_count"], 0)

    def test_property_merge_keep_wins_and_fills_gaps(self):
        resolver = self._make_resolver()
        vertices = [
            {"id": "p:A", "label": "Person", "properties": {"name": "Alice", "age": "30"}},
            {"id": "p:B", "label": "Person", "properties": {"name": "Alice", "city": "Beijing"}},
        ]
        edges = [
            {"label": "knows", "outV": "p:A", "inV": "p:C", "properties": {}},
            {"label": "knows", "outV": "p:E", "inV": "p:B", "properties": {}},
        ]

        out = resolver.resolve_in_memory_pure(vertices, edges)

        self.assertEqual(len(out["vertices"]), 1)
        kept = out["vertices"][0]
        self.assertEqual(kept["properties"]["name"], "Alice")
        # Both properties survive; keeper value wins on conflict.
        self.assertIn("age", kept["properties"])
        self.assertIn("city", kept["properties"])

    def test_self_loop_dropped_after_merge(self):
        resolver = self._make_resolver()
        vertices = [
            {"id": "p:A", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "p:B", "label": "Person", "properties": {"name": "Alice"}},
        ]
        # This edge becomes a self-loop once p:B → p:A.
        edges = [{"label": "knows", "outV": "p:B", "inV": "p:A", "properties": {}}]

        out = resolver.resolve_in_memory_pure(vertices, edges)

        # Both endpoints collapse to p:A → self-loop dropped.
        self.assertEqual(out["edges"], [])
        self.assertEqual(out["resolution_result"]["merged_count"], 1)

    def test_transitive_merge(self):
        resolver = self._make_resolver()
        vertices = [
            {"id": "p:A", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "p:B", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "p:C", "label": "Person", "properties": {"name": "Alice"}},
        ]
        edges = [
            {"label": "knows", "outV": "p:A", "inV": "p:X", "properties": {}},
            {"label": "knows", "outV": "p:A", "inV": "p:Y", "properties": {}},
            {"label": "knows", "outV": "p:Z", "inV": "p:B", "properties": {}},
            {"label": "knows", "outV": "p:W", "inV": "p:C", "properties": {}},
        ]

        out = resolver.resolve_in_memory_pure(vertices, edges)

        # All three collapse into p:A (degree 2, the highest).
        self.assertEqual([v["id"] for v in out["vertices"]], ["p:A"])
        self.assertEqual(out["resolution_result"]["merged_count"], 2)
        self.assertEqual(set(out["resolution_result"]["deprecated_vids"]), {"p:B", "p:C"})

    def test_empty_vertices(self):
        resolver = self._make_resolver()
        out = resolver.resolve_in_memory_pure([], [])
        self.assertEqual(out["vertices"], [])
        self.assertEqual(out["edges"], [])
        self.assertEqual(out["resolution_result"]["merged_count"], 0)


# ---------------------------------------------------------------------------
# merge_mode validation
# ---------------------------------------------------------------------------


class TestInvalidMergeMode(unittest.TestCase):
    def test_invalid_merge_mode_raises(self):
        client = MagicMock()
        with self.assertRaises(ValueError):
            EntityResolution(client=client, merge_mode="bogus")


# ---------------------------------------------------------------------------
# synonym_edge mode (HippoRAG-style: keep both vertices, add synonym edge)
# ---------------------------------------------------------------------------


class TestSynonymEdgeMode(unittest.TestCase):
    def test_synonym_edge_keeps_vertices_and_adds_edge(self):
        client = MagicMock()
        resolver = EntityResolution(
            client=client, strategy="exact_match", merge_mode="synonym_edge"
        )
        # _merge_entities picks the highest-degree vertex as keeper; feed real
        # degree info so the v1(keep) / v2(duplicate) direction is deterministic.
        resolver._get_vertex_info = MagicMock(
            side_effect=lambda vid: {
                "v1": {"label": "Person", "properties": {"name": "Apple"}, "degree": 5},
                "v2": {"label": "Person", "properties": {"name": "Apple"}, "degree": 2},
            }[vid]
        )
        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Apple"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Apple"}, "degree": 2},
        ]
        result = resolver.run({"vertices": vertices})
        res = result["resolution_result"]

        # Both vertices preserved; a synonym edge is written instead of merge.
        self.assertEqual(res["merged_count"], 1)
        self.assertEqual(res["synonym_edges"], 1)
        self.assertEqual(res["deprecated_vids"], [])  # no deprecation in synonym mode

        add_edge_calls = client.graph.return_value.addEdge.call_args_list
        self.assertEqual(len(add_edge_calls), 1)
        args = add_edge_calls[0].args
        self.assertEqual(args[0], "SYNONYM_OF")
        self.assertEqual(args[1], "v2")  # from_vid (the duplicate)
        self.assertEqual(args[2], "v1")  # to_vid (the keeper)

    def test_synonym_edge_custom_label(self):
        client = MagicMock()
        resolver = EntityResolution(
            client=client,
            strategy="exact_match",
            merge_mode="synonym_edge",
            synonym_edge_label="ALIAS_OF",
        )
        vertices = [
            {"id": "v1", "label": "Person", "properties": {"name": "Apple"}, "degree": 5},
            {"id": "v2", "label": "Person", "properties": {"name": "Apple"}, "degree": 2},
        ]
        resolver.run({"vertices": vertices})
        args = client.graph.return_value.addEdge.call_args_list[0].args
        self.assertEqual(args[0], "ALIAS_OF")

    def test_synonym_edge_error_recorded(self):
        client = MagicMock()
        client.graph.return_value.addEdge = MagicMock(side_effect=RuntimeError("rejected"))
        resolver = EntityResolution(
            client=client, strategy="exact_match", merge_mode="synonym_edge"
        )
        candidates = [
            MergeCandidate(
                from_vid="v2",
                from_label="Person",
                from_properties={},
                to_vid="v1",
                to_label="Person",
                to_properties={},
                strategy="exact_match",
                confidence=1.0,
            )
        ]
        result = resolver._merge_entities(candidates, {})
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.synonym_edges, 0)


# ---------------------------------------------------------------------------
# ANN blocking (scalability: replace O(n^2) with blocks + optional ANN)
# ---------------------------------------------------------------------------


class TestANNBlocking(unittest.TestCase):
    def _resolver(self, **kw):
        return EntityResolution(client=MagicMock(), embedding=MagicMock(), strategy="embedding", **kw)

    def test_default_blocking_key_groups_by_first_token(self):
        r = self._resolver()
        v1 = {"id": "1", "label": "Person", "properties": {"name": "Apple Inc"}, "degree": 0}
        v2 = {"id": "2", "label": "Person", "properties": {"name": "Apple Corp"}, "degree": 0}
        v3 = {"id": "3", "label": "Person", "properties": {"name": "Banana"}, "degree": 0}
        self.assertEqual(r._default_blocking_key(v1), r._default_blocking_key(v2))
        self.assertNotEqual(r._default_blocking_key(v1), r._default_blocking_key(v3))
        # Empty / missing property -> stable sentinel
        self.assertEqual(r._default_blocking_key({"id": "x", "properties": {}}), "__none__")

    def test_blocking_groups_partitions(self):
        r = self._resolver()
        group = [
            {"id": "1", "label": "Person", "properties": {"name": "Apple Inc"}, "degree": 0},
            {"id": "2", "label": "Person", "properties": {"name": "Apple Corp"}, "degree": 0},
            {"id": "3", "label": "Person", "properties": {"name": "Banana"}, "degree": 0},
        ]
        blocks = r._blocking_groups(group)
        sizes = sorted(len(b) for b in blocks)
        self.assertEqual(sizes, [1, 2])

    def test_candidate_pairs_ann_retriever(self):
        r = self._resolver(ann_topk=2)
        r._ann_retriever = lambda texts, k: [[1, 2], [0, 2], [0, 1]]
        block = [
            {"id": "1", "properties": {"name": "A"}, "degree": 0},
            {"id": "2", "properties": {"name": "B"}, "degree": 0},
            {"id": "3", "properties": {"name": "C"}, "degree": 0},
        ]
        pairs = r._candidate_pairs_in_block(block)
        # Unique undirected pairs across the three neighbours lists.
        self.assertEqual(set(pairs), {(0, 1), (0, 2), (1, 2)})

    def test_candidate_pairs_ann_retriever_filters_out_of_range(self):
        r = self._resolver(ann_topk=2)
        r._ann_retriever = lambda texts, k: [[5, -1, 0], [0], [0]]  # bogus indices
        block = [
            {"id": "1", "properties": {"name": "A"}, "degree": 0},
            {"id": "2", "properties": {"name": "B"}, "degree": 0},
        ]
        pairs = r._candidate_pairs_in_block(block)
        self.assertEqual(pairs, [(0, 1)])

    def test_candidate_pairs_ann_retriever_exception_falls_back(self):
        r = self._resolver(ann_topk=2)
        r._ann_retriever = lambda texts, k: (_ for _ in ()).throw(RuntimeError("knn down"))
        block = [
            {"id": "1", "properties": {"name": "A"}, "degree": 0},
            {"id": "2", "properties": {"name": "B"}, "degree": 0},
        ]
        pairs = r._candidate_pairs_in_block(block)  # falls back to pairwise
        self.assertEqual(pairs, [(0, 1)])

    def test_candidate_pairs_fallback_pairwise(self):
        r = self._resolver()
        block = [
            {"id": "1", "properties": {"name": "A"}, "degree": 0},
            {"id": "2", "properties": {"name": "B"}, "degree": 0},
            {"id": "3", "properties": {"name": "C"}, "degree": 0},
        ]
        self.assertEqual(r._candidate_pairs_in_block(block), [(0, 1), (0, 2), (1, 2)])

    def test_embedding_with_ann_retriever(self):
        client = MagicMock()
        embedding = MagicMock()
        embedding.get_text_embedding = MagicMock(
            side_effect=lambda t: {"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [0.0, 1.0]}[t]
        )
        resolver = EntityResolution(
            client=client,
            embedding=embedding,
            strategy="embedding",
            threshold=0.5,
            blocking_key=lambda v: "x",  # force one block so ANN is consulted
        )
        resolver._ann_retriever = lambda texts, k: [[1], [0], []]  # A<->B neighbours, C isolated
        group = [
            {"id": "1", "label": "Person", "properties": {"name": "A"}, "degree": 5},
            {"id": "2", "label": "Person", "properties": {"name": "B"}, "degree": 1},
            {"id": "3", "label": "Person", "properties": {"name": "C"}, "degree": 1},
        ]
        candidates = resolver._embedding_candidates(group)
        # A and B share identical embeddings -> 1 candidate; C isolated.
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].from_vid, "2")
        self.assertEqual(candidates[0].to_vid, "1")


# ---------------------------------------------------------------------------
# LLM fail-closed (default) vs fail-open (opt-in)
# ---------------------------------------------------------------------------


class TestFailClosedVerify(unittest.TestCase):
    def _cands(self):
        return [
            MergeCandidate(
                from_vid="v1",
                from_label="Person",
                from_properties={},
                to_vid="v2",
                to_label="Person",
                to_properties={},
                strategy="embedding",
                confidence=0.85,
            )
        ]

    def test_verify_batch_exception_fail_closed(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=RuntimeError("llm 500"))
        resolver = EntityResolution(client=client, llm=llm, strategy="llm_verify")
        self.assertEqual(resolver._verify_batch(self._cands()), [])

    def test_verify_batch_exception_fail_open(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=RuntimeError("llm 500"))
        resolver = EntityResolution(
            client=client, llm=llm, strategy="llm_verify", llm_fail_open=True
        )
        self.assertEqual(resolver._verify_batch(self._cands()), self._cands())

    def test_parse_response_non_list_fail_closed(self):
        client = MagicMock()
        llm = MagicMock()
        resolver = EntityResolution(client=client, llm=llm, strategy="llm_verify")
        self.assertEqual(resolver._parse_verify_response("{'not':'a list'}", self._cands()), [])

    def test_parse_response_non_list_fail_open(self):
        client = MagicMock()
        llm = MagicMock()
        resolver = EntityResolution(
            client=client, llm=llm, strategy="llm_verify", llm_fail_open=True
        )
        self.assertEqual(
            resolver._parse_verify_response("{'not':'a list'}", self._cands()), self._cands()
        )

    def test_verify_candidates_no_llm_returns_unchanged(self):
        client = MagicMock()
        resolver = EntityResolution(client=client, strategy="hybrid")
        cands = self._cands()
        # No LLM -> warning + candidates returned unchanged (unverified).
        self.assertEqual(resolver._verify_candidates(cands), cands)


# ---------------------------------------------------------------------------
# Graph-store merge execution (migrate edges, mark deprecated, error paths)
# ---------------------------------------------------------------------------


class TestMergeEntitiesGraphStore(unittest.TestCase):
    def test_merge_migrates_edges_and_marks_deprecated(self):
        client = MagicMock()

        def exec_side_effect(groovy):
            if "outE" in groovy:
                return [{"label": "knows", "inV": "vX", "properties": {}}]
            if "inE" in groovy:
                return [{"label": "knows", "outV": "vY", "properties": {}}]
            if "bothE" in groovy:
                return []
            return []

        client.gremlin.return_value.exec = MagicMock(side_effect=exec_side_effect)
        resolver = EntityResolution(client=client, strategy="exact_match")
        resolver._get_vertex_info = MagicMock(
            side_effect=lambda vid: {
                "v1": {"label": "Person", "properties": {}, "degree": 5},
                "v2": {"label": "Person", "properties": {}, "degree": 2},
            }[vid]
        )
        candidates = [
            MergeCandidate(
                from_vid="v2",
                from_label="Person",
                from_properties={},
                to_vid="v1",
                to_label="Person",
                to_properties={},
                strategy="exact_match",
                confidence=1.0,
            )
        ]
        result = resolver._merge_entities(candidates, {})
        self.assertEqual(result.merged_count, 1)
        self.assertEqual(result.edges_migrated, 2)  # one outgoing + one incoming
        self.assertEqual(result.deprecated_vids, ["v2"])
        self.assertGreaterEqual(client.graph.return_value.addEdge.call_count, 2)

    def test_merge_mode_merge_default_migrates(self):
        # Same as above but explicitly merge_mode="merge" (default behaviour).
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(return_value=[])
        resolver = EntityResolution(client=client, strategy="exact_match", merge_mode="merge")
        resolver._get_vertex_info = MagicMock(
            side_effect=lambda vid: {
                "v1": {"label": "Person", "properties": {}, "degree": 5},
                "v2": {"label": "Person", "properties": {}, "degree": 2},
            }[vid]
        )
        candidates = [
            MergeCandidate(
                from_vid="v2",
                from_label="Person",
                from_properties={},
                to_vid="v1",
                to_label="Person",
                to_properties={},
                strategy="exact_match",
                confidence=1.0,
            )
        ]
        result = resolver._merge_entities(candidates, {})
        self.assertEqual(result.merged_count, 1)
        self.assertEqual(result.deprecated_vids, ["v2"])


# ---------------------------------------------------------------------------
# Helper / graph-store query edge branches
# ---------------------------------------------------------------------------


class TestGraphStoreHelpers(unittest.TestCase):
    def test_get_vertex_labels_dict_response(self):
        client = MagicMock()
        schema_mock = MagicMock()
        schema_mock.getVertexLabels.return_value = {
            "vertexlabels": [{"name": "Person"}, {"name": "Org"}]
        }
        client.schema.return_value = schema_mock
        resolver = EntityResolution(client=client)
        self.assertEqual(resolver._get_vertex_labels(None), ["Person", "Org"])

    def test_get_vertex_labels_exception_returns_empty(self):
        client = MagicMock()
        client.schema.return_value.getVertexLabels = MagicMock(side_effect=RuntimeError("boom"))
        resolver = EntityResolution(client=client)
        self.assertEqual(resolver._get_vertex_labels(None), [])

    def test_fetch_vertices_by_label_exception_returns_empty(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=RuntimeError("boom"))
        resolver = EntityResolution(client=client)
        self.assertEqual(resolver._fetch_vertices_by_label("Person"), [])

    def test_get_vertex_info_exception_returns_none(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=RuntimeError("boom"))
        resolver = EntityResolution(client=client)
        self.assertIsNone(resolver._get_vertex_info("v1"))

    def test_resolve_from_graph_empty_fetch(self):
        client = MagicMock()
        schema_mock = MagicMock()
        schema_mock.getVertexLabels.return_value = ["Person"]
        client.schema.return_value = schema_mock
        client.gremlin.return_value.exec = MagicMock(return_value={"data": []})
        resolver = EntityResolution(client=client, strategy="exact_match")
        result = resolver.run({})
        self.assertEqual(result["resolution_result"]["merged_count"], 0)


# ---------------------------------------------------------------------------
# Extra branch coverage (drive every remaining line to 100%)
# ---------------------------------------------------------------------------


class TestEntityResolutionEdgeCoverage(unittest.TestCase):
    """Cover defensive branches and edge paths for 100% statement coverage."""

    def _cand(self, from_vid="v1", to_vid="v2", confidence=0.85):
        return MergeCandidate(
            from_vid=from_vid,
            from_label="Person",
            from_properties={},
            to_vid=to_vid,
            to_label="Person",
            to_properties={},
            strategy="embedding",
            confidence=confidence,
        )

    # -- UnionFind rank rebalancing -----------------------------------------
    def test_union_rank_swap_triggers_rebalance(self):
        uf = UnionFind()
        uf.union("a", "b")  # a -> rank 1
        uf.union("c", "d")  # c -> rank 1
        uf.union("e", "f")  # e -> rank 1
        uf.union("c", "e")  # c -> rank 2
        # rank(a)=1 < rank(c)=2 -> the swap branch in union() executes
        uf.union("a", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))

    # -- embedding candidate discovery edge cases ---------------------------
    def test_embedding_single_vertex_group_returns_empty(self):
        r = EntityResolution(
            client=MagicMock(), embedding=MagicMock(), strategy="embedding", threshold=0.9
        )
        # A label group of one vertex cannot produce pairs (early return).
        self.assertEqual(
            r._embedding_candidates(
                [{"id": "1", "label": "Person", "properties": {"name": "A"}, "degree": 1}]
            ),
            [],
        )

    def test_embedding_empty_text_skipped(self):
        embedding = MagicMock()
        embedding.get_text_embedding = MagicMock(return_value=[1.0, 0.0])
        r = EntityResolution(
            client=MagicMock(), embedding=embedding, strategy="embedding", threshold=0.0
        )
        # Both vertices have empty resolve text -> land in the "__none__" block
        # but are skipped before any embedding call.
        group = [
            {"id": "1", "label": "Person", "properties": {}, "degree": 1},
            {"id": "2", "label": "Person", "properties": {}, "degree": 1},
        ]
        self.assertEqual(r._embedding_candidates(group), [])

    def test_candidate_pairs_single_vertex_block(self):
        r = EntityResolution(client=MagicMock(), embedding=MagicMock(), strategy="embedding")
        # n < 2 short-circuits before any pairwise / ANN work.
        self.assertEqual(r._candidate_pairs_in_block([{"id": "1", "properties": {}}]), [])

    def test_candidate_pairs_sampling_when_over_limit(self):
        r = EntityResolution(
            client=MagicMock(), embedding=MagicMock(), strategy="embedding", max_pairs_per_label=3
        )
        block = [
            {"id": str(i), "properties": {"name": f"N{i}"}, "degree": 0} for i in range(4)
        ]
        # C(4,2) = 6 > 3 -> falls into the sampling branch.
        pairs = r._candidate_pairs_in_block(block)
        self.assertEqual(len(pairs), 3)

    # -- LLM verify batching / parsing branches -----------------------------
    def test_verify_candidates_mid_loop_flush(self):
        client = MagicMock()
        llm = MagicMock()
        llm.generate = MagicMock(return_value="[true, true]")
        r = EntityResolution(client=client, llm=llm, strategy="llm_verify", batch_size=1)
        cands = [self._cand("v1", "v2", 0.8), self._cand("v3", "v4", 0.8)]
        verified = r._verify_candidates(cands)
        # each candidate flushes mid-loop (not only at the tail)
        self.assertEqual(len(verified), 2)

    def test_parse_response_markdown_codeblock(self):
        r = EntityResolution(client=MagicMock(), llm=MagicMock(), strategy="llm_verify")
        verified = r._parse_verify_response("```json\n[true]\n```", [self._cand()])
        self.assertEqual(len(verified), 1)

    def test_parse_response_valid_json_non_list_fail_closed(self):
        r = EntityResolution(client=MagicMock(), llm=MagicMock(), strategy="llm_verify")
        # response parses to a string (valid JSON, but not a list) -> reject
        self.assertEqual(r._parse_verify_response('"yes"', [self._cand()]), [])

    def test_parse_response_valid_json_non_list_fail_open(self):
        r = EntityResolution(
            client=MagicMock(), llm=MagicMock(), strategy="llm_verify", llm_fail_open=True
        )
        self.assertEqual(
            r._parse_verify_response('"yes"', [self._cand()]), [self._cand()]
        )

    # -- merge execution: outer except + migrate edge failures --------------
    def test_merge_entities_outer_except_records_error(self):
        client = MagicMock()
        r = EntityResolution(client=client, strategy="exact_match")
        r._get_vertex_info = MagicMock(
            side_effect=lambda vid: {
                "v1": {"label": "Person", "properties": {}, "degree": 5},
                "v2": {"label": "Person", "properties": {}, "degree": 2},
            }[vid]
        )
        # Force an unexpected failure inside the merge body -> outer except.
        r._migrate_edges = MagicMock(side_effect=ValueError("boom"))
        result = r._merge_entities(
            [self._cand("v2", "v1", 1.0)], {}
        )
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.merged_count, 0)

    def test_migrate_edges_outgoing_add_fails(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(
            side_effect=lambda g: [{"label": "knows", "inV": "vX", "properties": {}}]
            if "outE" in g
            else []
        )
        client.graph.return_value.addEdge = MagicMock(side_effect=RuntimeError("rejected"))
        r = EntityResolution(client=client, strategy="exact_match")
        self.assertEqual(r._migrate_edges("v2", "v1"), 0)  # outgoing add raised

    def test_migrate_edges_incoming_add_fails(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(
            side_effect=lambda g: [{"label": "knows", "outV": "vY", "properties": {}}]
            if "inE" in g
            else []
        )
        client.graph.return_value.addEdge = MagicMock(side_effect=RuntimeError("rejected"))
        r = EntityResolution(client=client, strategy="exact_match")
        self.assertEqual(r._migrate_edges("v2", "v1"), 0)  # incoming add raised

    # -- graph-store helper branch coverage --------------------------------
    def test_query_edges_both_direction(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(
            return_value=[{"id": "e1", "label": "knows"}]
        )
        r = EntityResolution(client=client)
        self.assertEqual(
            r._query_edges("v1", direction="both"), [{"id": "e1", "label": "knows"}]
        )

    def test_query_edges_exception_returns_empty(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=RuntimeError("boom"))
        r = EntityResolution(client=client)
        self.assertEqual(r._query_edges("v1", direction="out"), [])

    def test_delete_edges_exception_logged(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=RuntimeError("boom"))
        r = EntityResolution(client=client)
        r._delete_edges("v1")  # must not raise

    def test_mark_deprecated_exception_logged(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=RuntimeError("boom"))
        r = EntityResolution(client=client)
        r._mark_deprecated("v1", "v2")  # must not raise

    def test_fetch_vertices_by_label_returns_list(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(
            return_value=[{"id": "v1", "label": "Person", "properties": {}}]
        )
        r = EntityResolution(client=client)
        self.assertEqual(
            r._fetch_vertices_by_label("Person"),
            [{"id": "v1", "label": "Person", "properties": {}}],
        )

    def test_get_vertex_info_dict_response(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(
            return_value={"id": "v1", "label": "Person"}
        )
        r = EntityResolution(client=client)
        self.assertEqual(r._get_vertex_info("v1"), {"id": "v1", "label": "Person"})

    def test_get_vertex_info_data_list(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(
            return_value={"data": [{"id": "v1", "label": "Person"}]}
        )
        r = EntityResolution(client=client)
        self.assertEqual(r._get_vertex_info("v1"), {"id": "v1", "label": "Person"})

    # -- Gremlin id quoting (_g_id) -----------------------------------------
    def test_g_id_quoting(self):
        # numeric / numeric-string ids must stay unquoted; string ids quoted.
        self.assertEqual(EntityResolution._g_id(12345), "12345")
        self.assertEqual(EntityResolution._g_id("-7"), "-7")
        self.assertEqual(EntityResolution._g_id("abc"), "'abc'")
        self.assertEqual(EntityResolution._g_id("o'brien"), "'o\\'brien'")
        # bool falls through to quoted-string handling
        self.assertEqual(EntityResolution._g_id(True), "'True'")

    def test_get_vertex_labels_filtered(self):
        client = MagicMock()
        schema_mock = MagicMock()
        schema_mock.getVertexLabels.return_value = ["Person", "Org"]
        client.schema.return_value = schema_mock
        r = EntityResolution(client=client)
        # labels provided -> filtered return branch (line 1045)
        self.assertEqual(r._get_vertex_labels(["Person"]), ["Person"])

    def test_get_vertex_labels_vertexlabel_objects(self):
        # Newer hugegraph-python-client returns a list of VertexLabel objects.
        from types import SimpleNamespace

        client = MagicMock()
        schema_mock = MagicMock()
        schema_mock.getVertexLabels.return_value = [
            SimpleNamespace(name="Person"),
            SimpleNamespace(name="Org"),
        ]
        client.schema.return_value = schema_mock
        r = EntityResolution(client=client)
        self.assertEqual(r._get_vertex_labels(None), ["Person", "Org"])


if __name__ == "__main__":
    unittest.main()
