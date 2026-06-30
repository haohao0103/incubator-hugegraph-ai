# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not with this file except in compliance
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

"""
Tests for LightRAG-style GraphRAG operators:
- Incremental graph update (entity ID as primary key)
- Dual-level retrieval (entity-centric + relationship-centric)
- Simplified query planner (LOW/HIGH/HYBRID levels)
- Community detection, summary, DRIFT, evaluation (optional/deferred)
"""

import pytest

from hugegraph_llm.operators.graphrag_op.community_detection import CommunityDetector
from hugegraph_llm.operators.graphrag_op.drift_search import DriftSearch
from hugegraph_llm.operators.graphrag_op.dual_level_retrieval import (
    DualLevelRetriever,
    RetrievalLevel,
)
from hugegraph_llm.operators.graphrag_op.evaluation import (
    BenchmarkRunner,
    GraphRAGEvaluator,
)
from hugegraph_llm.operators.graphrag_op.incremental_update import IncrementalGraphUpdater
from hugegraph_llm.operators.graphrag_op.query_planner import (
    QueryIntent,
    QueryLevel,
    QueryPlanner,
)

# ========== Incremental Update (LightRAG Core) ==========


class TestIncrementalGraphUpdater:
    """Tests for the LightRAG-style IncrementalGraphUpdater."""

    def test_incremental_update_new_data(self):
        """Test basic incremental update with new vertices and edges."""
        context = {
            "vertices": [
                {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}},
                {"id": "1:Bob", "label": "person", "properties": {"name": "Bob"}},
            ],
            "edges": [
                {
                    "label": "knows",
                    "outV": "1:Alice",
                    "inV": "1:Bob",
                    "outVLabel": "person",
                    "inVLabel": "person",
                    "properties": {},
                },
            ],
        }
        updater = IncrementalGraphUpdater(graph_client=None)
        result = updater.run(context)

        assert "incremental_update_summary" in result
        summary = result["incremental_update_summary"]
        assert "new_vertices" in summary
        assert "new_edges" in summary
        assert "timestamp" in summary

    def test_empty_update(self):
        """Test that empty data is handled gracefully."""
        context = {"vertices": [], "edges": []}
        updater = IncrementalGraphUpdater()
        result = updater.run(context)

        assert "incremental_update_summary" in result
        assert result["incremental_update_summary"]["new_vertices"] == 0

    def test_entity_name_as_primary_key(self):
        """Test LightRAG-style: entity name is the primary key for dedup."""
        updater = IncrementalGraphUpdater()
        vertex = {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}}

        # _get_entity_name should extract name from properties
        name = updater._get_entity_name(vertex)
        assert name == "Alice"

    def test_entity_name_fallback_to_title(self):
        """Test entity name extraction falls back to title property."""
        updater = IncrementalGraphUpdater()
        vertex = {"id": "1:doc1", "label": "document", "properties": {"title": "My Doc"}}

        name = updater._get_entity_name(vertex)
        assert name == "My Doc"

    def test_entity_name_fallback_to_id(self):
        """Test entity name extraction falls back to vertex id."""
        updater = IncrementalGraphUpdater()
        vertex = {"id": "some_id", "label": "thing", "properties": {}}

        name = updater._get_entity_name(vertex)
        assert name == "some_id"

    def test_build_entity_name_map(self):
        """Test that entity name → ID mapping is built correctly."""
        updater = IncrementalGraphUpdater()
        vertices = [
            {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}},
            {"id": "1:Bob", "label": "person", "properties": {"name": "Bob"}},
        ]
        updater._build_entity_name_map(vertices)

        mapping = updater.get_entity_name_to_id()
        assert "Alice" in mapping
        assert "Bob" in mapping
        assert mapping["Alice"] == "1:Alice"

    def test_property_merging_scalar(self):
        """Test scalar property merge: new overrides old, old preserved if not in new."""
        updater = IncrementalGraphUpdater()
        updated_vertices = [
            {
                "id": "1:Alice",
                "label": "person",
                "properties": {"name": "Alice", "age": 30},
                "_existing_properties": {"name": "Alice", "occupation": "lawyer"},
            },
        ]
        merged = updater._merge_vertex_properties(updated_vertices)

        assert len(merged) == 1
        props = merged[0]["properties"]
        assert props["name"] == "Alice"
        assert props["age"] == 30
        assert props["occupation"] == "lawyer"  # Preserved from existing

    def test_property_merging_list(self):
        """Test list property merge: lists are unioned."""
        updater = IncrementalGraphUpdater()
        updated_vertices = [
            {
                "id": "1:Alice",
                "label": "person",
                "properties": {"tags": ["engineer", "ml"]},
                "_existing_properties": {"tags": ["engineer", "leader"]},
            },
        ]
        merged = updater._merge_vertex_properties(updated_vertices)

        tags = merged[0]["properties"]["tags"]
        assert "engineer" in tags
        assert "ml" in tags
        assert "leader" in tags

    def test_change_log(self):
        """Test that change log is maintained across updates."""
        updater = IncrementalGraphUpdater()
        context = {
            "vertices": [{"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}}],
            "edges": [],
        }
        updater.run(context)

        log = updater.get_change_log()
        assert len(log) == 1
        assert "new_vertices" in log[0]

    def test_entity_name_to_id_in_context(self):
        """Test that entity name → ID mapping is stored in context output."""
        context = {
            "vertices": [
                {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}},
            ],
            "edges": [],
        }
        updater = IncrementalGraphUpdater(graph_client=None)
        result = updater.run(context)

        assert "entity_name_to_id" in result
        assert "Alice" in result["entity_name_to_id"]

    def test_deduplication_vertices_without_client(self):
        """Test vertex deduplication without graph client (all treated as new)."""
        updater = IncrementalGraphUpdater()
        vertices = [
            {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}},
            {"id": "1:Bob", "label": "person", "properties": {"name": "Bob"}},
        ]
        new_v, updated_v = updater._deduplicate_vertices(vertices, schema=None)

        # Without graph_client, all vertices are treated as new
        assert len(new_v) == 2
        assert len(updated_v) == 0


# ========== Dual-Level Retrieval (LightRAG Core) ==========


class TestDualLevelRetriever:
    """Tests for the LightRAG-style DualLevelRetriever."""

    def test_retrieval_level_determination_low(self):
        """Test that specific entity questions map to LOW level."""
        retriever = DualLevelRetriever()
        level = retriever._determine_retrieval_level("Who is Alice?", ["Alice"])
        assert level == RetrievalLevel.LOW

    def test_retrieval_level_determination_high(self):
        """Test that abstract questions map to HIGH level."""
        retriever = DualLevelRetriever()
        level = retriever._determine_retrieval_level("How are AI and ML related?", ["AI", "ML"])
        assert level == RetrievalLevel.HIGH

    def test_retrieval_level_determination_hybrid(self):
        """Test that multi-keyword queries default to HYBRID."""
        retriever = DualLevelRetriever()
        level = retriever._determine_retrieval_level("Compare", ["Python", "Java", "Rust"])
        assert level == RetrievalLevel.HYBRID

    def test_retrieval_level_chinese_queries(self):
        """Test Chinese query level determination."""
        retriever = DualLevelRetriever()
        level_low = retriever._determine_retrieval_level("什么是机器学习？", ["机器学习"])
        assert level_low == RetrievalLevel.LOW

    def test_run_with_no_query(self):
        """Test graceful handling of missing query."""
        retriever = DualLevelRetriever()
        context = {"query": ""}
        result = retriever.run(context)

        # Should return context unchanged (no crash)
        assert isinstance(result, dict)

    def test_run_determines_retrieval_level(self):
        """Test that run() sets the retrieval_level in context."""
        retriever = DualLevelRetriever(graph_client=None)
        context = {"query": "What is Python?", "keywords": ["Python"]}
        result = retriever.run(context)

        assert "retrieval_level" in result
        assert result["retrieval_level"] in ("low", "high", "hybrid")

    def test_run_sets_graph_result(self):
        """Test that dual-level retrieval sets graph_result for downstream."""
        retriever = DualLevelRetriever(graph_client=None)
        context = {"query": "Who is Alice?", "keywords": ["Alice"]}
        result = retriever.run(context)

        assert "graph_result" in result
        assert "dual_level_results" in result

    def test_low_level_retrieval_without_client(self):
        """Test low-level retrieval without graph client."""
        retriever = DualLevelRetriever(graph_client=None)
        context = {
            "query": "Who is Alice?",
            "keywords": ["Alice"],
            "entity_name_to_id": {"Alice": "1:Alice"},
        }
        results = retriever._low_level_retrieval("Who is Alice?", ["Alice"], context)
        # Without graph_client, no results from graph traversal
        # But should not crash
        assert isinstance(results, list)

    def test_high_level_retrieval_without_client(self):
        """Test high-level retrieval without graph client."""
        retriever = DualLevelRetriever(graph_client=None)
        context = {
            "query": "How are Alice and Bob related?",
            "keywords": ["Alice", "Bob"],
            "entity_name_to_id": {"Alice": "1:Alice", "Bob": "1:Bob"},
        }
        results = retriever._high_level_retrieval("How are Alice and Bob related?", ["Alice", "Bob"], context)
        assert isinstance(results, list)

    def test_merge_results_deduplication(self):
        """Test that merged results are deduplicated."""
        retriever = DualLevelRetriever()
        low = ["Alice is a person", "Bob is a person"]
        high = ["Alice is a person", "Charlie is related"]
        merged = retriever._merge_results(low, high, RetrievalLevel.HYBRID)

        # Should not have duplicates
        assert len(merged) == len(set(r.strip().lower()[:100] for r in merged))

    def test_format_vertex_result(self):
        """Test vertex result formatting."""
        element_map = {"label": "person", "name": "Alice", "age": "30"}
        result = DualLevelRetriever._format_vertex_result(element_map)
        assert "person" in result
        assert "Alice" in result

    def test_format_edge_result(self):
        """Test edge result formatting."""
        element_map = {"label": "knows", "since": "2020"}
        result = DualLevelRetriever._format_edge_result(element_map)
        assert "knows" in result


# ========== Simplified Query Planner (LightRAG) ==========


class TestQueryPlanner:
    """Tests for the simplified LightRAG-style QueryPlanner."""

    def test_classify_specific_entity_intent(self):
        planner = QueryPlanner(llm=None)
        intent = planner._classify_intent("Who is Sarah?")
        assert intent == QueryIntent.SPECIFIC_ENTITY

    def test_classify_relationship_intent(self):
        planner = QueryPlanner(llm=None)
        intent = planner._classify_intent("How are Alice and Bob related?")
        assert intent == QueryIntent.RELATIONSHIP

    def test_classify_abstract_intent(self):
        planner = QueryPlanner(llm=None)
        intent = planner._classify_intent("Why did the stock market crash?")
        assert intent == QueryIntent.ABSTRACT

    def test_classify_chinese_specific_entity(self):
        planner = QueryPlanner(llm=None)
        intent = planner._classify_intent("什么是深度学习？")
        assert intent == QueryIntent.SPECIFIC_ENTITY

    def test_classify_chinese_relationship(self):
        planner = QueryPlanner(llm=None)
        intent = planner._classify_intent("如何关联 Alice 和 Bob？")
        assert intent == QueryIntent.RELATIONSHIP

    def test_intent_to_level_mapping_low(self):
        """Test that SPECIFIC_ENTITY maps to LOW level."""
        from hugegraph_llm.operators.graphrag_op.query_planner import INTENT_LEVEL_MAP

        assert INTENT_LEVEL_MAP[QueryIntent.SPECIFIC_ENTITY] == QueryLevel.LOW

    def test_intent_to_level_mapping_high(self):
        """Test that ABSTRACT maps to HIGH level."""
        from hugegraph_llm.operators.graphrag_op.query_planner import INTENT_LEVEL_MAP

        assert INTENT_LEVEL_MAP[QueryIntent.ABSTRACT] == QueryLevel.HIGH

    def test_intent_to_level_mapping_hybrid(self):
        """Test that RELATIONSHIP maps to HYBRID level."""
        from hugegraph_llm.operators.graphrag_op.query_planner import INTENT_LEVEL_MAP

        assert INTENT_LEVEL_MAP[QueryIntent.RELATIONSHIP] == QueryLevel.HYBRID

    def test_full_plan_generation(self):
        planner = QueryPlanner(llm=None)
        context = {"query": "Who is Alice?"}
        result = planner.run(context)

        assert "query_plan" in result
        plan = result["query_plan"]
        assert "retrieval_level" in plan
        assert "steps" in plan
        assert "parameters" in plan
        assert result["query_intent"] == "specific_entity"
        assert result["retrieval_level"] == "low"

    def test_full_plan_relationship_query(self):
        planner = QueryPlanner(llm=None)
        context = {"query": "How are Python and Java related?"}
        result = planner.run(context)

        assert result["query_intent"] == "relationship"
        assert result["retrieval_level"] == "hybrid"

    def test_full_plan_abstract_query(self):
        planner = QueryPlanner(llm=None)
        context = {"query": "Why did the market crash?"}
        result = planner.run(context)

        assert result["query_intent"] == "abstract"
        assert result["retrieval_level"] == "high"

    def test_no_community_dependency(self):
        """Test that the simplified planner does NOT depend on communities."""
        planner = QueryPlanner(llm=None)

        # All these should work WITHOUT community data
        queries = [
            "Who is Alice?",
            "How are X and Y related?",
            "Why did something happen?",
        ]
        for query in queries:
            result = planner.run({"query": query})
            assert "query_plan" in result
            assert "retrieval_level" in result

    def test_default_plan(self):
        """Test default plan for empty query."""
        planner = QueryPlanner(llm=None)
        plan = planner._default_plan()

        assert plan["retrieval_level"] == "hybrid"
        assert len(plan["steps"]) > 0


# ========== Community Detection (Optional/Deferred) ==========


class TestCommunityDetector:
    """Tests for the CommunityDetector class (optional, deferred)."""

    def test_detect_communities_from_vertices_edges(self):
        context = {
            "vertices": [
                {"id": "A", "label": "person", "properties": {"name": "Alice"}},
                {"id": "B", "label": "person", "properties": {"name": "Bob"}},
                {"id": "C", "label": "person", "properties": {"name": "Charlie"}},
                {"id": "D", "label": "person", "properties": {"name": "Diana"}},
                {"id": "E", "label": "person", "properties": {"name": "Eve"}},
            ],
            "edges": [
                {"outV": "A", "inV": "B", "label": "knows"},
                {"outV": "B", "inV": "C", "label": "knows"},
                {"outV": "C", "inV": "A", "label": "knows"},
                {"outV": "D", "inV": "E", "label": "knows"},
            ],
        }
        detector = CommunityDetector(algorithm="louvain", min_community_size=2)
        result = detector.run(context)

        assert "communities" in result
        assert "community_hierarchy" in result
        assert result["community_count"] > 0
        assert result["community_algorithm"] == "louvain"

    def test_empty_graph(self):
        context = {"vertices": [], "edges": []}
        detector = CommunityDetector()
        result = detector.run(context)

        assert result["communities"] == []
        assert result["community_count"] == 0


# ========== DRIFT Search (Optional/Deferred) ==========


class TestDriftSearch:
    """Tests for the DriftSearch class (optional, deferred)."""

    def test_drift_search_with_communities(self):
        context = {
            "query": "Who is Alice?",
            "community_summaries": [
                {
                    "community_id": "C0",
                    "title": "Alice's Community",
                    "summary": "A community about Alice who is a lawyer",
                    "key_entities": ["Alice", "Bob"],
                    "themes": ["law", "friendship"],
                }
            ],
            "communities": [["Alice", "Bob"]],
            "vertices": [
                {"id": "Alice", "label": "person", "properties": {"name": "Alice", "occupation": "lawyer"}},
            ],
            "graph_result": [],
        }
        searcher = DriftSearch()
        result = searcher.run(context)

        assert "drift_results" in result

    def test_drift_search_no_query(self):
        context = {"query": "", "community_summaries": []}
        searcher = DriftSearch()
        result = searcher.run(context)

        assert isinstance(result, dict)


# ========== Evaluation ==========


class TestGraphRAGEvaluator:
    """Tests for the GraphRAGEvaluator class."""

    def test_evaluate_with_heuristics(self):
        evaluator = GraphRAGEvaluator(llm=None)
        context = {
            "query": "Who is Alice?",
            "answer": "Alice is a software engineer who works at Google. She has 10 years of experience in machine learning.",
            "graph_result": ["Alice works at Google"],
            "vector_result": ["Alice is an engineer"],
        }
        result = evaluator.run(context)

        assert "evaluation_results" in result
        results = result["evaluation_results"]
        assert "overall" in results
        assert 0 <= results["overall"] <= 1

    def test_evaluate_empty_answer(self):
        evaluator = GraphRAGEvaluator(llm=None)
        context = {"query": "", "answer": ""}
        result = evaluator.run(context)

        assert "error" in result.get("evaluation_results", {})

    def test_faithfulness_evaluation(self):
        evaluator = GraphRAGEvaluator(llm=None)
        score = evaluator._evaluate_faithfulness(
            "What is AI?", "AI is artificial intelligence", ["AI stands for artificial intelligence"]
        )
        assert 0 <= score <= 1


class TestBenchmarkRunner:
    """Tests for the BenchmarkRunner class."""

    def test_benchmark_runner_init(self):
        runner = BenchmarkRunner()
        assert runner.get_benchmark_history() == []

    def test_benchmark_runner_no_func(self):
        runner = BenchmarkRunner()
        with pytest.raises(ValueError, match="No RAG function"):
            runner.run_benchmark([{"query": "test"}])

    def test_benchmark_runner_with_mock_func(self):
        def mock_rag(query):
            return {"query": query, "answer": "test answer", "graph_result": [], "vector_result": []}

        evaluator = GraphRAGEvaluator(llm=None)
        runner = BenchmarkRunner(evaluator=evaluator)
        result = runner.run_benchmark(
            [{"query": "What is AI?", "ground_truth": "AI is artificial intelligence"}],
            rag_func=mock_rag,
        )

        assert "total_cases" in result
        assert result["total_cases"] == 1


# ========== Integration (LightRAG Pipeline) ==========


class TestLightRAGIntegration:
    """Integration tests for the LightRAG-style pipeline."""

    def test_incremental_update_then_query(self):
        """Test the core LightRAG flow: incremental update → query planning → retrieval."""
        # Step 1: Incremental update (indexing)
        context = {
            "vertices": [
                {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}},
                {"id": "1:Bob", "label": "person", "properties": {"name": "Bob"}},
            ],
            "edges": [
                {
                    "label": "knows",
                    "outV": "1:Alice",
                    "inV": "1:Bob",
                    "properties": {},
                },
            ],
        }
        updater = IncrementalGraphUpdater(graph_client=None)
        context = updater.run(context)
        assert "incremental_update_summary" in context
        assert "entity_name_to_id" in context

        # Step 2: Query planning
        context["query"] = "Who is Alice?"
        planner = QueryPlanner(llm=None)
        context = planner.run(context)
        assert context["retrieval_level"] == "low"

        # Step 3: Dual-level retrieval
        context["keywords"] = ["Alice"]
        retriever = DualLevelRetriever(graph_client=None)
        context = retriever.run(context)
        assert "dual_level_results" in context
        assert "graph_result" in context

    def test_second_incremental_update(self):
        """Test that a second incremental update works correctly (append-only)."""
        # First update
        context1 = {
            "vertices": [
                {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}},
            ],
            "edges": [],
        }
        updater = IncrementalGraphUpdater(graph_client=None)
        result1 = updater.run(context1)
        assert result1["incremental_update_summary"]["new_vertices"] >= 1

        # Second update with new entity
        context2 = {
            "vertices": [
                {"id": "1:Charlie", "label": "person", "properties": {"name": "Charlie"}},
            ],
            "edges": [],
        }
        result2 = updater.run(context2)
        assert result2["incremental_update_summary"]["new_vertices"] >= 1

        # Verify change log has two entries
        assert len(updater.get_change_log()) == 2

    def test_entity_name_dedup_across_updates(self):
        """Test that entity name deduplication works across multiple updates."""
        updater = IncrementalGraphUpdater()

        # First update creates Alice
        v1 = {"id": "1:Alice", "label": "person", "properties": {"name": "Alice"}}
        updater._build_entity_name_map([v1])

        # Verify name mapping exists
        mapping = updater.get_entity_name_to_id()
        assert "Alice" in mapping
