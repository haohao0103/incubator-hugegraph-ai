# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for hybrid scoring — mem0-style additive retrieval scoring."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.hybrid_scoring import (
    get_bm25_params,
    normalize_bm25,
    compute_entity_boosts,
    extract_query_entities_simple,
    score_and_rank,
)


# ── BM25 Normalization Tests ──────────────────────────────────


class TestGetBM25Params:

    def test_short_query(self):
        midpoint, steepness = get_bm25_params("张三")
        assert midpoint == 10.0
        assert steepness == 1.0

    def test_medium_query(self):
        midpoint, steepness = get_bm25_params("张三在货拉拉公司工作")
        # 8 chars → medium category (4 < 8 <= 12)
        assert midpoint == 15.0
        assert steepness == 0.7

    def test_long_query(self):
        midpoint, steepness = get_bm25_params("张三和李四是同事他们在货拉拉公司一起工作")
        # 20+ chars → long category (>12)
        assert midpoint == 20.0
        assert steepness == 0.5


class TestNormalizeBM25:

    def test_high_score(self):
        # High raw score → close to 1.0
        result = normalize_bm25(50.0, midpoint=10.0, steepness=1.0)
        assert result > 0.99

    def test_mid_score(self):
        # Score at midpoint → ~0.5
        result = normalize_bm25(10.0, midpoint=10.0, steepness=1.0)
        assert 0.49 <= result <= 0.51

    def test_low_score(self):
        # Low raw score → close to 0.0
        result = normalize_bm25(1.0, midpoint=10.0, steepness=1.0)
        assert result < 0.01

    def test_zero_score(self):
        result = normalize_bm25(0.0, midpoint=10.0, steepness=1.0)
        assert result == 0.0

    def test_negative_score(self):
        result = normalize_bm25(-5.0, midpoint=10.0, steepness=1.0)
        assert result == 0.0

    def test_steepness_effect(self):
        # Higher steepness → sharper transition
        gentle = normalize_bm25(12.0, midpoint=10.0, steepness=0.5)
        sharp = normalize_bm25(12.0, midpoint=10.0, steepness=2.0)
        assert sharp > gentle  # Steeper sigmoid amplifies deviation from midpoint


# ── Entity Boost Tests ────────────────────────────────────────


class TestComputeEntityBoosts:

    def test_matching_entities(self):
        query_entities = ["张三", "货拉拉"]
        memory_entities = {
            "mem1": ["张三", "货拉拉公司", "深圳市"],
            "mem2": ["李四", "滴滴公司"],
            "mem3": ["张三"],
        }
        boosts = compute_entity_boosts(query_entities, memory_entities)
        assert "mem1" in boosts
        # 张三 + 货拉拉 both match (case-insensitive)
        assert boosts["mem1"] >= 0.5
        assert "mem3" in boosts
        assert boosts["mem3"] == 0.5  # Only 张三 matches

    def test_no_matching_entities(self):
        query_entities = ["不存在的人"]
        memory_entities = {"mem1": ["张三"]}
        boosts = compute_entity_boosts(query_entities, memory_entities)
        assert len(boosts) == 0

    def test_custom_weight(self):
        query_entities = ["张三"]
        memory_entities = {"mem1": ["张三"]}
        boosts = compute_entity_boosts(query_entities, memory_entities, boost_weight=1.0)
        assert boosts["mem1"] == 1.0

    def test_empty_inputs(self):
        boosts = compute_entity_boosts([], {})
        assert len(boosts) == 0


# ── Query Entity Extraction Tests ─────────────────────────────


class TestExtractQueryEntitiesSimple:

    def test_chinese_org(self):
        entities = extract_query_entities_simple("张三在货拉拉公司工作")
        assert any("货拉拉" in e for e in entities) or any("张三" in e for e in entities)

    def test_english_name(self):
        entities = extract_query_entities_simple("John Smith went to Stanford University")
        assert "John Smith" in entities

    def test_stopwords_filtered(self):
        entities = extract_query_entities_simple("什么是什么")
        assert "什么" not in entities

    def test_empty_query(self):
        entities = extract_query_entities_simple("")
        assert entities == []


# ── score_and_rank Tests ──────────────────────────────────────


class TestScoreAndRank:

    def test_semantic_only(self):
        semantic_results = [
            {"id": "m1", "content": "张三在货拉拉工作", "score": 0.85, "metadata": {}},
            {"id": "m2", "content": "李四在滴滴工作", "score": 0.45, "metadata": {}},
            {"id": "m3", "content": "无关内容", "score": 0.05, "metadata": {}},
        ]
        results = score_and_rank(
            semantic_results=semantic_results,
            bm25_scores={},
            entity_boosts={},
            threshold=0.1,
            top_k=10,
        )
        # m3 should be filtered out (score < 0.1)
        assert len(results) == 2
        # Results should be sorted by score
        assert results[0]["score"] > results[1]["score"]

    def test_semantic_bm25_entity(self):
        semantic_results = [
            {"id": "m1", "content": "张三在货拉拉工作", "score": 0.7, "metadata": {}},
            {"id": "m2", "content": "其他内容", "score": 0.6, "metadata": {}},
        ]
        bm25_scores = {"m1": 30.0, "m2": 10.0}
        entity_boosts = {"m1": 0.5}

        results = score_and_rank(
            semantic_results=semantic_results,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=0.1,
            top_k=10,
        )
        # m1 should score higher due to BM25 + entity boost
        assert results[0]["id"] == "m1"

    def test_threshold_filtering(self):
        semantic_results = [
            {"id": "m1", "content": "低相关", "score": 0.05, "metadata": {}},
        ]
        results = score_and_rank(
            semantic_results=semantic_results,
            bm25_scores={},
            entity_boosts={},
            threshold=0.1,
        )
        assert len(results) == 0

    def test_top_k_limit(self):
        semantic_results = [
            {"id": f"m{i}", "content": f"content{i}", "score": 0.5, "metadata": {}}
            for i in range(20)
        ]
        results = score_and_rank(
            semantic_results=semantic_results,
            bm25_scores={},
            entity_boosts={},
            top_k=5,
        )
        assert len(results) == 5

    def test_explain_mode(self):
        semantic_results = [
            {"id": "m1", "content": "张三在货拉拉工作", "score": 0.7, "metadata": {}},
        ]
        bm25_scores = {"m1": 20.0}
        entity_boosts = {"m1": 0.5}

        results = score_and_rank(
            semantic_results=semantic_results,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            explain=True,
        )
        assert len(results) == 1
        assert "score_breakdown" in results[0]
        bd = results[0]["score_breakdown"]
        assert bd["semantic"] == 0.7
        assert bd["entity_boost"] == 0.5
        assert bd["bm25_normalized"] > 0

    def test_bm25_normalization_in_scoring(self):
        """Verify BM25 scores are normalized before combining."""
        semantic_results = [
            {"id": "m1", "content": "张三在货拉拉工作", "score": 0.7, "metadata": {}},
        ]
        # Very high raw BM25 score
        bm25_scores = {"m1": 100.0}

        results = score_and_rank(
            semantic_results=semantic_results,
            bm25_scores=bm25_scores,
            entity_boosts={},
        )
        # After sigmoid normalization, BM25 should be close to 1.0
        assert results[0]["score"] > 0.7  # Adding normalized BM25 pushes it higher
