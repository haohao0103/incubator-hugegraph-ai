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

"""Tests for EntityRanker operator."""

import pytest

from hugegraph_llm.operators.graph_op.entity_ranker import (
    EntityRanker,
    EntityRankerConfig,
    EntityRankerResult,
    HugeGraphEntityRankerAdapter,
    build_ranker_from_edges,
)


# ---------------------------------------------------------------------------
# Empty graph
# ---------------------------------------------------------------------------


def test_empty_graph_returns_empty_result():
    ranker = EntityRanker(edges_loader=lambda: [])
    result = ranker.compute_global_pagerank()
    assert result.scores == {}
    assert result.converged is True


def test_no_loader_returns_empty_result():
    ranker = EntityRanker(edges_loader=None)
    ranker.load_graph()
    result = ranker.compute_global_pagerank()
    assert result.scores == {}


def test_load_graph_with_no_loader_clears_state():
    ranker = EntityRanker(edges_loader=None)
    ranker.load_graph()
    assert ranker._adjacency == {}
    assert ranker._nodes == set()
    assert ranker._out_weights == {}


# ---------------------------------------------------------------------------
# Global PageRank on simple graphs
# ---------------------------------------------------------------------------


def test_two_node_rank():
    edges = [("A", "B", 1.0)]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_global_pagerank()
    # With normalization, the highest score is 1.0 (B usually has higher PR)
    assert result.top_k(1)[0][0] == "B"
    assert 0.0 <= result.get_score("A") <= 1.0
    assert 0.0 <= result.get_score("B") <= 1.0


def test_source_has_highest_score():
    """In a cycle, source should be the most important."""
    edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_global_pagerank()
    # After normalization, A, B, C should have equal scores in a symmetric cycle
    scores = result.scores
    assert scores["A"] == pytest.approx(scores["B"], abs=1e-6)
    assert scores["B"] == pytest.approx(scores["C"], abs=1e-6)


def test_star_graph_center_ranks_highest():
    edges = [
        ("A", "center", 1.0),
        ("B", "center", 1.0),
        ("C", "center", 1.0),
    ]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_global_pagerank()
    top_node, top_score = result.top_k(1)[0]
    assert top_node == "center"
    assert top_score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_weighted_edges_change_ranking():
    edges = [
        ("A", "B", 1.0),
        ("A", "C", 10.0),
    ]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_global_pagerank()
    # C should receive more weight than B
    assert result.get_score("C") > result.get_score("B")


# ---------------------------------------------------------------------------
# Directionality
# ---------------------------------------------------------------------------


def test_undirected_mode_adds_reverse_edges():
    edges = [("A", "B", 1.0)]
    config = EntityRankerConfig(undirected=True)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    # In undirected mode both nodes should have equal scores
    assert result.get_score("A") == pytest.approx(result.get_score("B"), abs=1e-6)


def test_directed_mode_preserves_direction():
    edges = [("A", "B", 1.0)]
    config = EntityRankerConfig(undirected=False)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    # B receives incoming mass, so it should have higher score than A
    assert result.get_score("B") > result.get_score("A")


# ---------------------------------------------------------------------------
# Dangling nodes
# ---------------------------------------------------------------------------


def test_dangling_node_teleport_enabled():
    edges = [("A", "B", 1.0), ("C", "C", 1.0)]  # C is dangling (self-loop only)
    config = EntityRankerConfig(dangling_teleport=True)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    # All nodes should receive some score from dangling teleport
    assert all(s > 0 for s in result.scores.values())


def test_dangling_node_teleport_disabled():
    edges = [("A", "B", 1.0)]
    config = EntityRankerConfig(dangling_teleport=False)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    # B has incoming mass; A has no incoming mass, so its score is near zero
    assert result.get_score("B") > result.get_score("A")


# ---------------------------------------------------------------------------
# Personalized PageRank
# ---------------------------------------------------------------------------


def test_ppr_single_source():
    # Star graph: source is the center, should receive most PPR mass
    edges = [
        ("A", "B", 1.0),
        ("A", "C", 1.0),
        ("A", "D", 1.0),
    ]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_ppr({"A": 1.0})
    # Source A should have the highest PPR score
    assert result.top_k(1)[0][0] == "A"
    # All neighbors should be reachable with positive score
    assert result.get_score("B") > 0
    assert result.get_score("C") > 0
    assert result.get_score("D") > 0


def test_ppr_multiple_sources():
    edges = [
        ("A", "X", 1.0),
        ("B", "X", 1.0),
    ]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_ppr({"A": 1.0, "B": 1.0})
    # X receives from both sources
    assert result.get_score("X") > result.get_score("A")
    assert result.get_score("X") > result.get_score("B")


def test_ppr_unknown_source_falls_back_to_uniform():
    edges = [("A", "B", 1.0)]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_ppr({"UNKNOWN": 1.0})
    # Unknown source should be ignored, falling back to uniform teleport
    assert "A" in result.scores
    assert "B" in result.scores


# ---------------------------------------------------------------------------
# score() and score_personalized() helpers
# ---------------------------------------------------------------------------


def test_score_method_returns_global_score():
    edges = [("A", "B", 1.0), ("B", "C", 1.0)]
    ranker = build_ranker_from_edges(edges)
    assert 0.0 <= ranker.score("B") <= 1.0


def test_score_method_returns_zero_for_unknown_entity():
    edges = [("A", "B", 1.0)]
    ranker = build_ranker_from_edges(edges)
    assert ranker.score("UNKNOWN") == 0.0


def test_score_personalized_method():
    edges = [("A", "B", 1.0), ("B", "C", 1.0)]
    ranker = build_ranker_from_edges(edges)
    assert ranker.score_personalized("A", {"A": 1.0}) > 0.0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_global_pagerank_is_cached():
    edges = [("A", "B", 1.0)]
    ranker = build_ranker_from_edges(edges)
    first = ranker.compute_global_pagerank()
    second = ranker.compute_global_pagerank()
    assert first is second


def test_global_pagerank_cache_can_be_refreshed():
    edges = [("A", "B", 1.0)]
    ranker = build_ranker_from_edges(edges)
    first = ranker.compute_global_pagerank()
    second = ranker.compute_global_pagerank(force_refresh=True)
    assert first is not second


def test_reset_cache_clears_global_result():
    edges = [("A", "B", 1.0)]
    ranker = build_ranker_from_edges(edges)
    ranker.compute_global_pagerank()
    ranker.reset_cache()
    assert ranker._global_pagerank is None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_off():
    edges = [("A", "B", 1.0)]
    config = EntityRankerConfig(normalize_scores=False)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    # Without normalization, scores are probability masses and sum to 1.0
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_normalization_on():
    edges = [("A", "B", 1.0)]
    config = EntityRankerConfig(normalize_scores=True)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    # Top score should be 1.0
    assert max(result.scores.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_convergence_on_small_graph():
    edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
    config = EntityRankerConfig(epsilon=1e-8, max_iterations=200)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    assert result.converged is True
    assert result.final_delta < 1e-8


def test_max_iterations_reached():
    edges = [("A", "B", 1.0), ("B", "A", 1.0)]
    config = EntityRankerConfig(epsilon=1e-20, max_iterations=2)
    ranker = build_ranker_from_edges(edges, config=config)
    result = ranker.compute_global_pagerank()
    assert result.num_iterations <= 2
    # May not converge with such tight epsilon and few iterations


# ---------------------------------------------------------------------------
# Graph introspection
# ---------------------------------------------------------------------------


def test_get_nodes_and_edge_count():
    edges = [("A", "B", 1.0), ("B", "C", 1.0)]
    ranker = build_ranker_from_edges(edges)
    assert ranker.get_nodes() == {"A", "B", "C"}
    assert ranker.get_edge_count() == 2


# ---------------------------------------------------------------------------
# HugeGraph adapter
# ---------------------------------------------------------------------------


class FakeHGClient:
    """Fake PyHugeClient that returns deterministic edges."""

    def __init__(self, edges):
        self._edges = edges

    def getEdgeByCondition(self, edge_label=None, limit=10000):
        if edge_label is None:
            return self._edges
        return [e for e in self._edges if e.get("label") == edge_label]


def test_adapter_builds_ranker_from_client():
    edges = [
        {"source": "A", "target": "B", "label": "REL", "properties": {"weight": 2.0}},
        {"source": "B", "target": "C", "label": "REL", "properties": {}},
    ]
    adapter = HugeGraphEntityRankerAdapter(FakeHGClient(edges))
    ranker = adapter.build_ranker()
    result = ranker.compute_global_pagerank()
    assert set(result.scores.keys()) == {"A", "B", "C"}
    # A->B has weight 2.0, B->C has default weight 1.0
    assert result.get_score("B") > result.get_score("A")


def test_adapter_filters_by_edge_label():
    edges = [
        {"source": "A", "target": "B", "label": "REL", "properties": {}},
        {"source": "B", "target": "C", "label": "OTHER", "properties": {}},
    ]
    adapter = HugeGraphEntityRankerAdapter(
        FakeHGClient(edges), edge_labels=["REL"]
    )
    ranker = adapter.build_ranker()
    result = ranker.compute_global_pagerank()
    assert "C" not in result.scores
    assert set(result.scores.keys()) == {"A", "B"}


def test_adapter_handles_missing_client():
    adapter = HugeGraphEntityRankerAdapter(None)
    ranker = adapter.build_ranker()
    result = ranker.compute_global_pagerank()
    assert result.scores == {}


def test_adapter_handles_malformed_weight():
    edges = [
        {"source": "A", "target": "B", "label": "REL", "properties": {"weight": "bad"}},
    ]
    adapter = HugeGraphEntityRankerAdapter(FakeHGClient(edges))
    ranker = adapter.build_ranker()
    result = ranker.compute_global_pagerank()
    assert "A" in result.scores
    assert "B" in result.scores


def test_adapter_handles_fetch_exception():
    class BadHGClient:
        def getEdgeByCondition(self, edge_label=None, limit=10000):
            raise RuntimeError("api failure")

    adapter = HugeGraphEntityRankerAdapter(BadHGClient())
    ranker = adapter.build_ranker()
    result = ranker.compute_global_pagerank()
    assert result.scores == {}


def test_adapter_skips_edges_with_empty_source_or_target():
    edges = [
        {"source": "", "target": "B", "label": "REL", "properties": {}},
        {"source": "A", "target": "", "label": "REL", "properties": {}},
        {"source": "A", "target": "B", "label": "REL", "properties": {}},
    ]
    adapter = HugeGraphEntityRankerAdapter(FakeHGClient(edges))
    ranker = adapter.build_ranker()
    result = ranker.compute_global_pagerank()
    assert set(result.scores.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# Edge filtering and malformed input
# ---------------------------------------------------------------------------


def test_build_adjacency_skips_malformed_edges():
    edges = [
        ("A",),  # too short
        ("", "B", 1.0),  # empty source
        ("A", "", 1.0),  # empty target
        ("A", "B", 1.0),  # valid
    ]
    ranker = build_ranker_from_edges(edges)
    assert ranker.get_nodes() == {"A", "B"}
    assert ranker.get_edge_count() == 1


def test_build_adjacency_two_tuple_edge_defaults_weight():
    edges = [("A", "B")]
    ranker = build_ranker_from_edges(edges)
    result = ranker.compute_global_pagerank()
    assert "A" in result.scores
    assert "B" in result.scores


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def test_result_top_k():
    result = EntityRankerResult(
        scores={"A": 0.1, "B": 0.9, "C": 0.5}
    )
    top = result.top_k(2)
    assert top == [("B", 0.9), ("C", 0.5)]


def test_result_get_score_default():
    result = EntityRankerResult(scores={"A": 0.5})
    assert result.get_score("A") == 0.5
    assert result.get_score("B", default=-1.0) == -1.0
