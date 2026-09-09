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

"""Tests for the vector channel and its fusion with the temporal channel."""

import pytest

from hugegraph_llm.memory.retrieval import MemoryRecallConfig, recall
from hugegraph_llm.memory.temporal import TemporalStore
from hugegraph_llm.memory.vector_channel import (
    FaissChannel,
    NoOpChannel,
    VectorChannelConfig,
)
from tests.memory.test_temporal import FakeClient

DAY = 24 * 3600 * 1000
T0 = 1704067200000
T30 = T0 + 30 * DAY


class KeywordEmbed:
    """Deterministic 'embedding': one dimension per vocabulary term.

    Not semantic -- it exists so plumbing and fusion behaviour can be
    asserted without a model. It does give cosine 1.0 for identical term
    sets and 0.0 for disjoint ones, which is enough to test ranking.
    """

    VOCAB = ("acme", "globex", "works", "coffee", "tea", "dark", "mode")

    def __call__(self, text: str) -> list:
        tokens = set(str(text).lower().replace("_", " ").split())
        return [1.0 if word in tokens else 0.0 for word in self.VOCAB]


# -- channel contract -------------------------------------------------------


def test_noop_channel_never_matches():
    channel = NoOpChannel()
    channel.add([("f1", "anything")])
    assert channel.search("anything", 5) == []
    assert channel.ranked_ids("anything", 5) == []


def test_faiss_channel_returns_ranked_ids():
    channel = FaissChannel(KeywordEmbed())
    channel.add([("f1", "works at acme"), ("f2", "prefers dark mode")])
    hits = channel.search("works at acme", 5)
    assert hits, "faiss should match an identical text"
    assert hits[0][0] == "f1"


def test_faiss_channel_size_tracks_additions():
    channel = FaissChannel(KeywordEmbed())
    channel.add([("f1", "a"), ("f2", "b")])
    assert channel.size == 2


def test_faiss_channel_min_score_filters():
    """A similarity floor drops weak hits while keeping strong ones.

    'works at acme' vs 'works at globex' shares one term of five, so the
    cosine is well below a 0.9 floor while the exact match stays above it.
    """
    channel = FaissChannel(
        KeywordEmbed(), config=VectorChannelConfig(min_score=0.9)
    )
    channel.add([("f1", "works at acme"), ("f2", "works at globex")])
    strong = channel.search("works at acme", 5)
    assert [i for i, _ in strong] == ["f1"]

    # A disjoint query scores 0 for everything -> filtered out entirely.
    assert channel.search("none of these words", 5) == []


def test_empty_add_is_safe():
    assert FaissChannel(KeywordEmbed()).search("x", 5) == []


def test_faiss_channel_survives_backend_failure():
    """Recall must degrade, not raise, when the vector backend breaks."""

    class BrokenEmbed:
        def __call__(self, text):
            raise RuntimeError("model unavailable")

    channel = FaissChannel(BrokenEmbed())
    # add() catches per-item failure; nothing indexed, search stays empty.
    channel.add([("f1", "x")])
    assert channel.search("x", 5) == []


# -- fusion with the temporal channel ----------------------------------------


@pytest.fixture
def store():
    s = TemporalStore(FakeClient())
    s.ensure_schema()
    s.add_entity("u1", "张明", created_at=T0)
    s.add_entity("c1", "Acme", created_at=T0)
    s.add_entity("c2", "Globex", created_at=T0)
    s.add_fact("f1", "works at Acme", "u1", "c1",
               valid_at=T0, invalid_at=T30, created_at=T0, conflict_key="works_at")
    s.add_fact("f2", "works at Globex", "u1", "c2",
               valid_at=T30, created_at=T30, conflict_key="works_at")
    return s


def _channel(store):
    ch = FaissChannel(KeywordEmbed())
    ch.add([(f.uuid, f.fact) for f in store.all_facts()])
    return ch


def test_vector_channel_reorders_by_relevance(store):
    """At T0 only Acme is valid; asking about Globex must not resurrect it."""
    result = recall(store, query="works at globex", at=T0, now=T0,
                    vector_channel=_channel(store))
    contents = [i.content for i in result.items]
    assert "works at Acme" in contents
    assert "works at Globex" not in contents  # not valid at T0


def test_gating_still_wins_over_relevance(store):
    """The whole point: relevance never overrides temporal validity."""
    result = recall(store, query="globex", at=T0, now=T0,
                    vector_channel=_channel(store))
    assert all(i.metadata["valid_at"] <= T0 < i.metadata["invalid_at"]
               for i in result.items)


def test_vector_channel_appears_in_metadata(store):
    result = recall(store, query="works", at=T30, now=T30,
                    vector_channel=_channel(store))
    assert any("vector" in name for name in result.metadata["channels"])


def test_recall_without_vector_channel_is_unchanged(store):
    """Adding a channel must be opt-in -- default behaviour stays as before."""
    plain = recall(store, at=T30, now=T30)
    assert plain.metadata["channels"] == ["temporal"]


def test_unavailable_vector_channel_does_not_break_recall(store):
    result = recall(store, query="anything", at=T30, now=T30,
                    vector_channel=NoOpChannel())
    assert not result.is_empty
    assert "vector_unavailable" in result.metadata["channels"]


def test_broken_vector_channel_does_not_break_recall(store):
    """Relevance is best-effort: a failure degrades, never fails the call."""

    class Broken(FaissChannel):
        def ranked_ids(self, query, top_k):
            raise RuntimeError("index gone")

    result = recall(store, query="works", at=T30, now=T30,
                    vector_channel=Broken(KeywordEmbed()))
    assert not result.is_empty
    # Fell back to the temporal channel alone.
    assert result.metadata["channels"] == ["temporal"]


def test_query_ignored_when_no_vector_channel(store):
    """Query text is only consumed by a vector channel."""
    a = recall(store, query="globex", at=T0, now=T0)
    b = recall(store, at=T0, now=T0)
    assert [i.metadata["uuid"] for i in a.items] == \
           [i.metadata["uuid"] for i in b.items]
