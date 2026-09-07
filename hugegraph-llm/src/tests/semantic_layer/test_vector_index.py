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

"""Tests for vector indexing and weighted fusion.

The deterministic embedder here has no semantic content by design -- it
exercises plumbing and *mechanism* (thresholds that discriminate), never
retrieval quality. Quality claims require a real embedding endpoint, which
this environment currently lacks (401).
"""

import pytest

from hugegraph_llm.semantic_layer.indexer import (
    TERM_PREFIX,
    TABLE_PREFIX,
    DeterministicEmbedder,
    SemanticIndexer,
    embed_from_settings,
    table_document,
    term_document,
)
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    InMemorySemanticReader,
    SemanticProjection,
    TableRow,
    TermRow,
)
from hugegraph_llm.semantic_layer.retrieval import (
    RetrievalConfig,
    SemanticLayerRetriever,
    weighted_fuse,
)


def _proj():
    proj = SemanticProjection()
    proj.tables["orders"] = TableRow(
        name="orders", database="wh", schema="public",
        comment="All customer orders", row_count=1000,
    )
    proj.tables["customers"] = TableRow(
        name="customers", database="wh", schema="public",
        comment="Customer master data", row_count=200,
    )
    for col in (
        ColumnRow(name="order_id", table="orders", is_primary_key=True),
        ColumnRow(name="amount", table="orders", comment="Order total"),
        ColumnRow(name="customer_id", table="customers", is_primary_key=True),
    ):
        proj.columns[col.qualified] = col
    proj.references["orders.customer_id"] = ["customers.customer_id"]
    proj.reference_proven[("orders.customer_id", "customers.customer_id")] = True
    proj.terms["ARR"] = TermRow(name="ARR", description="Annual recurring revenue")
    proj.term_columns["ARR"] = ["orders.amount"]
    return proj


class _ArrayStore:
    """Minimal in-memory store honouring the SchemaVectorStore contract."""

    def __init__(self):
        self.vectors = {}

    def upsert(self, ids, vectors):
        for doc_id, vector in zip(ids, vectors):
            self.vectors[doc_id] = vector

    def search(self, query_vector, top_k):
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if not na or not nb:
                return 0.0
            return max(0.0, dot / (na * nb))

        scored = [
            (doc_id, cosine(query_vector, vec))
            for doc_id, vec in self.vectors.items()
        ]
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]


# -- document builders -----------------------------------------------------


def test_table_document_folds_columns_and_terms():
    doc = table_document("orders", "All orders", ["order_id", "amount"], ["营收"])
    assert "orders" in doc and "order_id" in doc and "营收" in doc


def test_term_document_includes_aliases():
    doc = term_document("ARR", "Annual recurring revenue", ["annual arr"])
    assert "annual arr" in doc


def test_document_prefixes_match_recall_resolution():
    """Indexer ids must be resolvable by _node_to_table -- no mapping layer."""
    assert TABLE_PREFIX == "table:" and TERM_PREFIX == "term:"


# -- indexer ----------------------------------------------------------------


def test_build_indexes_tables_and_terms():
    retriever = SemanticLayerRetriever(InMemorySemanticReader(_proj()))
    store = _ArrayStore()
    indexer = SemanticIndexer(retriever, store, DeterministicEmbedder(32))
    stats = indexer.build()
    assert stats.tables == 2 and stats.terms == 1
    assert stats.embedded == 3 and stats.failed == 0
    assert set(store.vectors) == {
        "table:orders", "table:customers", "term:ARR",
    }
    assert all(len(v) == 32 for v in store.vectors.values())


def test_build_empty_projection_is_safe():
    retriever = SemanticLayerRetriever(InMemorySemanticReader(SemanticProjection()))
    indexer = SemanticIndexer(retriever, _ArrayStore(), DeterministicEmbedder(8))
    stats = indexer.build()
    assert stats.documents == 0


def test_build_reports_failed_batches_without_raising():
    class BrokenEmbed:
        def __call__(self, text):
            raise RuntimeError("endpoint down")

    retriever = SemanticLayerRetriever(InMemorySemanticReader(_proj()))
    indexer = SemanticIndexer(retriever, _ArrayStore(), BrokenEmbed())
    stats = indexer.build()
    assert stats.failed == 3 and stats.embedded == 0


def test_build_refresh_invalidates_retriever_caches():
    retriever = SemanticLayerRetriever(InMemorySemanticReader(_proj()))
    first_bm25 = retriever.bm25
    SemanticIndexer(retriever, _ArrayStore(), DeterministicEmbedder(8)).build(
        refresh=True
    )
    assert retriever.bm25 is not first_bm25


def test_deterministic_embedder_is_stable():
    e = DeterministicEmbedder(16)
    assert e("orders table") == e("orders table")


def test_deterministic_embedder_is_normalised():
    vec = DeterministicEmbedder(16)("orders table")
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0)


# -- end-to-end: indexed retrieval uses the vector path ---------------------


def test_indexed_retriever_uses_vector_source():
    retriever = SemanticLayerRetriever(InMemorySemanticReader(_proj()))
    store = _ArrayStore()
    SemanticIndexer(retriever, store, DeterministicEmbedder(32)).build()
    retriever.vector_store = store
    retriever.embed = DeterministicEmbedder(32)

    result = retriever.retrieve("customer orders")
    assert "vector" in result.sources_used
    assert result.tables  # plumbing works end to end


# -- weighted fusion --------------------------------------------------------


def test_weighted_fuse_preserves_magnitude():
    """A strong cosine hit must outrank a weak one by more than rank order."""
    fused = weighted_fuse(
        [[("strong", 0.95), ("weak", 0.55)]], [1.0]
    )
    scores = dict(fused)
    assert scores["strong"] > scores["weak"]
    assert scores["weak"] == pytest.approx(0.55)


def test_weighted_fuse_normalises_unbounded_bm25():
    fused = weighted_fuse([[("a", 12.0), ("b", 6.0)]], [1.0], unbounded=(0,))
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(0.5)


def test_weighted_fuse_clamps_out_of_range_bounded_scores():
    """A source returning 3.0 for a 'cosine' gets clamped, not trusted."""
    fused = weighted_fuse([[("a", 3.0)]], [1.0])
    assert dict(fused)["a"] == pytest.approx(1.0)


def test_weighted_fuse_combines_sources():
    fused = weighted_fuse(
        [[("a", 0.8)], [("a", 5.0), ("b", 2.5)]], [1.0, 1.0], unbounded=(1,)
    )
    scores = dict(fused)
    # vector stays absolute (0.8); bm25 is max-normalised (5.0 -> 1.0).
    assert scores["a"] == pytest.approx(1.8)
    assert scores["b"] == pytest.approx(0.5)


def test_weighted_fuse_does_not_inflate_bounded_scores():
    """Normalising cosine by max would make a mediocre batch look strong."""
    fused = weighted_fuse(
        [[("mediocre_best", 0.55), ("mediocre", 0.40)]], [1.0]
    )
    scores = dict(fused)
    # 0.55 stays 0.55: the query simply has no strong hit.
    assert scores["mediocre_best"] == pytest.approx(0.55)
    assert scores["mediocre"] == pytest.approx(0.40)


def test_weighted_fuse_empty_source_contributes_nothing():
    fused = weighted_fuse([[], [("a", 3.0)]], [1.0, 1.0])
    assert dict(fused)["a"] == pytest.approx(1.0)


def test_weighted_fuse_weights_must_match():
    with pytest.raises(ValueError):
        weighted_fuse([[("a", 1.0)]], [1.0, 1.0])


def test_weighted_fuse_all_zero_scores():
    """A source whose max score is 0 must not divide by zero."""
    assert weighted_fuse([[("a", 0.0)]], [1.0]) == []


# -- the payoff: thresholds discriminate in weighted mode --------------------


class _ScoredStore:
    """Returns fixed scores, letting tests control the spread exactly."""

    def __init__(self, hits):
        self._hits = hits

    def search(self, vector, top_k):
        return self._hits[:top_k]


def _indexed_retriever_with(vector_hits):
    retriever = SemanticLayerRetriever(InMemorySemanticReader(_proj()))
    retriever.vector_store = _ScoredStore(vector_hits)
    retriever.embed = lambda text: [1.0]
    return retriever


def test_min_seed_ratio_prunes_weak_seeds_in_weighted_mode():
    """Vector: orders 0.95, customers 0.30. Floor 0.5 keeps only orders."""
    retriever = _indexed_retriever_with(
        [("table:orders", 0.95), ("table:customers", 0.30)]
    )
    cfg = RetrievalConfig(
        fusion_mode="weighted", top_k=5, hops=0, min_seed_ratio=0.5
    )
    result = retriever.retrieve("anything", config=cfg)
    assert result.seeds == ["orders"]


def test_min_seed_ratio_zero_keeps_all_in_weighted_mode():
    retriever = _indexed_retriever_with(
        [("table:orders", 0.95), ("table:customers", 0.30)]
    )
    cfg = RetrievalConfig(fusion_mode="weighted", top_k=5, hops=0)
    result = retriever.retrieve("anything", config=cfg)
    assert set(result.seeds) == {"orders", "customers"}


def test_min_seed_ratio_always_keeps_one_seed():
    """A question with only weak hits must not return nothing."""
    retriever = _indexed_retriever_with(
        [("table:orders", 0.4), ("table:customers", 0.1)]
    )
    cfg = RetrievalConfig(
        fusion_mode="weighted", top_k=5, hops=0, min_seed_ratio=0.9
    )
    result = retriever.retrieve("anything", config=cfg)
    assert result.seeds == ["orders"]


def test_min_score_ratio_prunes_expansion_in_weighted_mode():
    """Expansion score = seed * decay^hop; floor cuts the far branch."""
    retriever = _indexed_retriever_with([("table:customers", 1.0)])
    cfg = RetrievalConfig(
        fusion_mode="weighted",
        top_k=5,
        hops=2,
        hop_decay=0.5,
        min_score_ratio=0.6,  # hop1=0.5, hop2=0.25 -> both pruned
    )
    result = retriever.retrieve("anything", config=cfg)
    # customers is a seed (never filtered); orders was hop-1 only.
    assert "customers" in result.tables
    assert "orders" not in result.tables


def test_rrf_mode_still_ignores_thresholds():
    """Guard the documented behaviour: rrf + thresholds == rrf alone."""
    retriever = _indexed_retriever_with(
        [("table:orders", 0.95), ("table:customers", 0.30)]
    )
    base = RetrievalConfig(top_k=5, hops=0)
    floored = RetrievalConfig(
        fusion_mode="rrf", top_k=5, hops=0, min_seed_ratio=0.9
    )
    r1 = retriever.retrieve("q", config=base)
    r2 = retriever.retrieve("q", config=floored)
    assert r1.seeds == r2.seeds


def test_unknown_fusion_mode_raises():
    retriever = _indexed_retriever_with([])
    with pytest.raises(ValueError):
        retriever.retrieve("q", config=RetrievalConfig(fusion_mode="bogus"))


# -- settings adapter -------------------------------------------------------


def test_embed_from_settings_rejects_backend_without_text_embedding(monkeypatch):
    """A backend without get_text_embedding is a config error, not a silent
    adapter that fails per call."""
    import hugegraph_llm.semantic_layer.indexer as indexer_mod

    class _BadBackend:
        pass

    class _FakeEmbeddings:
        def get_embedding(self):
            return _BadBackend()

    monkeypatch.setattr(
        indexer_mod, "Embeddings", _FakeEmbeddings, raising=False
    )
    # The adapter imports inside the function, so patch the real module.
    import hugegraph_llm.models.embeddings.init_embedding as init_mod

    monkeypatch.setattr(
        init_mod, "Embeddings", _FakeEmbeddings
    )
    with pytest.raises(TypeError):
        embed_from_settings()


def test_embed_from_settings_adapts_get_text_embedding(monkeypatch):
    import hugegraph_llm.models.embeddings.init_embedding as init_mod

    class _Backend:
        def get_text_embedding(self, text):
            return [1, 2, 3]

    class _FakeEmbeddings:
        def get_embedding(self):
            return _Backend()

    monkeypatch.setattr(init_mod, "Embeddings", _FakeEmbeddings)
    embed = embed_from_settings()
    assert embed("hello") == [1.0, 2.0, 3.0]
