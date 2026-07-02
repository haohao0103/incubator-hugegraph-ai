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

"""Tests for MultimodalKGSearchRetriever."""

import pytest

from hugegraph_llm.operators.graph_op.kg_search_retriever import (
    KGSearchConfig,
    KGSearchRetriever,
    KGSearchResult,
    ScoredChunk,
    ScoredEntity,
)
from hugegraph_llm.operators.graph_op.multimodal_kg_search_retriever import (
    MultimodalItem,
    MultimodalKGSearchResult,
    MultimodalKGSearchRetriever,
    MultimodalQuery,
    MultimodalSearchResult,
    multimodal_kg_search,
)
from hugegraph_llm.operators.llm_op.query_rewrite import QueryRewriteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kg_retriever(seed_log=None):
    """Build a fake KGSearchRetriever whose traversal depends on seed IDs."""

    def traverse(entity_id, max_depth, max_fanout):
        if seed_log is not None:
            seed_log.append(entity_id)
        # hub has a neighbor, leaf has none
        if entity_id == "hub":
            return [("leaf", 1, "RELATED_TO")]
        if entity_id == "seed":
            return [("hub", 1, "RELATED_TO")]
        if entity_id == "image_seed":
            return [("cross_node", 1, "ASSOCIATED")]
        return []

    return KGSearchRetriever(
        graph_traversal_func=traverse,
        config=KGSearchConfig(max_depth=2, max_fanout=10),
    )


def make_text_router():
    """Fake router that returns the query text as a seed chunk."""
    class FakeRouter:
        def route(self, query):
            return [(query, 1.0)]
    return FakeRouter()


# ---------------------------------------------------------------------------
# MultimodalQuery
# ---------------------------------------------------------------------------


def test_query_text_only():
    q = MultimodalQuery(text="revenue trend")
    assert q.is_text_only is True
    assert q.has_image is False


def test_query_with_image():
    q = MultimodalQuery(text="revenue trend", image_base64="base64data")
    assert q.is_text_only is False
    assert q.has_image is True


def test_query_forced_image_mode():
    q = MultimodalQuery(text="revenue trend", query_type="image")
    assert q.is_text_only is False


# ---------------------------------------------------------------------------
# MultimodalItem / MultimodalSearchResult
# ---------------------------------------------------------------------------


def test_item_source_type_detection():
    img = MultimodalItem(id="desc_1", source_type="image")
    txt = MultimodalItem(id="txt_1", source_type="text")
    mixed = MultimodalItem(id="mix_1", source_type="mixed")
    assert img.is_image is True
    assert txt.is_text is True
    assert mixed.is_image and mixed.is_text


def test_search_result_seed_extraction():
    result = MultimodalSearchResult(
        query=MultimodalQuery(),
        query_mode="mixed",
        results=[
            MultimodalItem(id="desc_1", source_type="image", score=0.9),
            MultimodalItem(id="txt_1", source_type="text", score=0.8),
        ],
    )
    assert result.image_entity_ids() == ["desc_1"]
    assert result.text_chunk_ids() == ["txt_1"]


def test_search_result_top_k():
    result = MultimodalSearchResult(
        query=MultimodalQuery(),
        results=[
            MultimodalItem(id="a", score=0.1),
            MultimodalItem(id="b", score=0.9),
            MultimodalItem(id="c", score=0.5),
        ],
    )
    top = result.top_k(2)
    assert [item.id for item in top] == ["b", "c"]


# ---------------------------------------------------------------------------
# Text-only retrieval
# ---------------------------------------------------------------------------


def test_text_only_uses_baseline_kg_retriever():
    kg = make_kg_retriever()
    mm = MultimodalKGSearchRetriever(kg_retriever=kg)
    result = mm.retrieve(MultimodalQuery(text="seed"))
    assert result.text_result is result.cross_modal_result
    assert len(result.combined_entities) == 1
    assert result.combined_entities[0].entity_id == "hub"


def test_missing_image_func_falls_back_to_text():
    kg = make_kg_retriever()
    mm = MultimodalKGSearchRetriever(
        kg_retriever=kg,
        multimodal_search_func=None,
    )
    result = mm.retrieve(
        MultimodalQuery(text="seed", image_base64="img", query_type="image")
    )
    assert result.provenance["query_type"] == "text"


# ---------------------------------------------------------------------------
# Cross-modal retrieval
# ---------------------------------------------------------------------------


def test_image_seeds_inject_cross_modal_traversal():
    seed_log = []
    kg = make_kg_retriever(seed_log=seed_log)

    def mm_search(query):
        return MultimodalSearchResult(
            query=query,
            query_mode="mixed",
            results=[
                MultimodalItem(id="image_seed", label="ImageDescription", source_type="image"),
            ],
        )

    mm = MultimodalKGSearchRetriever(
        kg_retriever=kg,
        multimodal_search_func=mm_search,
    )
    result = mm.retrieve(MultimodalQuery(text="seed", image_base64="img"))

    assert "image_seed" in seed_log
    assert any(e.entity_id == "cross_node" for e in result.cross_modal_result.entities)
    assert result.provenance["query_type"] == "mixed"


def test_image_only_query():
    kg = make_kg_retriever()

    def mm_search(query):
        return MultimodalSearchResult(
            query=query,
            query_mode="image",
            results=[
                MultimodalItem(id="image_seed", source_type="image"),
            ],
        )

    mm = MultimodalKGSearchRetriever(
        kg_retriever=kg,
        multimodal_search_func=mm_search,
    )
    result = mm.search_by_image("base64", text_query="")
    assert result.provenance["query_type"] == "image"


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_merge_entities_keeps_higher_score():
    a = ScoredEntity(entity_id="e1", score=0.5)
    b = ScoredEntity(entity_id="e1", score=0.9)
    c = ScoredEntity(entity_id="e2", score=0.3)
    merged = MultimodalKGSearchRetriever._merge_entities([a], [b, c])
    assert len(merged) == 2
    assert merged[0].score == 0.9
    assert merged[0].entity_id == "e1"


def test_merge_chunks_keeps_higher_score():
    a = ScoredChunk(chunk_id="c1", score=0.4)
    b = ScoredChunk(chunk_id="c1", score=0.8)
    c = ScoredChunk(chunk_id="c2", score=0.2)
    merged = MultimodalKGSearchRetriever._merge_chunks([a], [b, c])
    assert len(merged) == 2
    assert merged[0].score == 0.8


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_result():
    kg = make_kg_retriever()
    mm = MultimodalKGSearchRetriever(kg_retriever=kg)
    result = mm.retrieve(MultimodalQuery())
    assert result == MultimodalKGSearchResult()


def test_no_multimodal_results_still_runs():
    kg = make_kg_retriever()

    def mm_search(query):
        return MultimodalSearchResult(query=query, query_mode="mixed", results=[])

    mm = MultimodalKGSearchRetriever(
        kg_retriever=kg,
        multimodal_search_func=mm_search,
    )
    result = mm.retrieve(MultimodalQuery(text="seed", image_base64="img"))
    assert result.cross_modal_seeds == []
    assert result.cross_modal_result == result.text_result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_multimodal_kg_search_convenience():
    kg = make_kg_retriever()

    def mm_search(query):
        return MultimodalSearchResult(
            query=query,
            query_mode="mixed",
            results=[MultimodalItem(id="image_seed", source_type="image")],
        )

    result = multimodal_kg_search(
        query_text="seed",
        image_base64="img",
        kg_retriever=kg,
        multimodal_search_func=mm_search,
    )
    assert result.provenance["query_type"] == "mixed"


def test_multimodal_kg_search_missing_dependencies():
    result = multimodal_kg_search("query")
    assert result == MultimodalKGSearchResult()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def test_result_to_dict():
    result = MultimodalKGSearchResult(
        text_result=KGSearchResult(
            entities=[ScoredEntity(entity_id="e1", score=0.5)],
        ),
        cross_modal_seeds=["image_seed"],
        combined_entities=[ScoredEntity(entity_id="e1", score=0.8)],
    )
    d = result.to_dict()
    assert d["text_result"]["entities"][0]["entity_id"] == "e1"
    assert d["cross_modal_seeds"] == ["image_seed"]
    assert d["combined_entities"] == ["e1"]
