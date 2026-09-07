# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BM25 and n-way RRF fusion."""

import pytest

from hugegraph_llm.semantic_layer.bm25 import BM25, tokenize
from hugegraph_llm.semantic_layer.fusion import rrf_fuse

CORPUS = [
    ("orders", "orders customer_id amount created_at order total"),
    ("customers", "customers customer_id name signup city"),
    ("products", "products product_id price category name"),
]


# -- tokenization ----------------------------------------------------------


def test_tokenize_lowercases():
    assert tokenize("Orders Table") == ["orders", "table"]


def test_tokenize_splits_cjk_per_character():
    assert tokenize("订单表") == ["订", "单", "表"]


def test_tokenize_ignores_punctuation():
    assert tokenize("a-b_c.d") == ["a", "b_c", "d"]


# -- bm25 ------------------------------------------------------------------


def test_bm25_finds_exact_identifier():
    bm = BM25().fit(CORPUS)
    hits = bm.search("customer_id", top_k=3)
    assert [h[0] for h in hits][:2] == ["orders", "customers"] or set(
        h[0] for h in hits[:2]
    ) == {"orders", "customers"}


def test_bm25_matches_snake_case_split():
    bm = BM25().fit(CORPUS)
    hits = bm.search("customer id", top_k=1)
    assert hits[0][0] in {"orders", "customers"}


def test_bm25_matches_cjk_query():
    bm = BM25().fit([("订单表", "订单表 订单 金额")])
    assert bm.search("订单", top_k=1)[0][0] == "订单表"


def test_bm25_scores_are_non_negative():
    """A term in every document must not produce a negative contribution."""
    bm = BM25().fit([("a", "common"), ("b", "common"), ("c", "common")])
    assert all(score >= 0 for _id, score in bm.search("common"))


def test_bm25_respects_top_k():
    bm = BM25().fit(CORPUS)
    assert len(bm.search("name", top_k=2)) <= 2


def test_bm25_empty_query_returns_nothing():
    assert BM25().fit(CORPUS).search("") == []


def test_bm25_unknown_term_returns_nothing():
    assert BM25().fit(CORPUS).search("zzzznotpresent") == []


def test_bm25_fit_replaces_previous_corpus():
    bm = BM25().fit(CORPUS)
    assert bm.size == 3
    bm.fit([("only", "one document")])
    assert bm.size == 1


# -- rrf -------------------------------------------------------------------


def test_rrf_fuses_two_lists():
    fused = rrf_fuse([[("a", 1.0), ("b", 0.5)], [("b", 9.0), ("a", 8.0)]])
    # ranks: a = 1,1 ; b = 2,1  -> a wins on the sum of reciprocal ranks
    assert [i for i, _ in fused] == ["a", "b"]


def test_rrf_handles_three_lists():
    fused = rrf_fuse([[("a", 1.0)], [("b", 1.0)], [("c", 1.0)]])
    assert {i for i, _ in fused} == {"a", "b", "c"}


def test_rrf_tie_broken_by_number_of_sources():
    """Two-source agreement beats a single source at equal fused score."""
    fused = rrf_fuse([[("both", 1.0)], [("both", 1.0), ("single", 1.0)]])
    assert [i for i, _ in fused][0] == "both"


def test_rrf_weights_shift_ranking():
    unweighted = rrf_fuse([[("a", 1.0)], [("b", 1.0)]])
    weighted = rrf_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[1.0, 5.0])
    assert dict(unweighted)["a"] == dict(unweighted)["b"]
    assert dict(weighted)["b"] > dict(weighted)["a"]


def test_rrf_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        rrf_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[1.0])


def test_rrf_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
