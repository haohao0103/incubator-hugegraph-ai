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

"""Tests for memory retrieval (temporal gating + decay + shared RRF)."""

import pytest

from hugegraph_llm.memory.retrieval import (
    DEFAULT_HALF_LIFE_MS,
    MemoryRecallConfig,
    TemporalRetriever,
    recall,
    time_decay_score,
)
from hugegraph_llm.memory.schema import OPEN
from hugegraph_llm.memory.temporal import Fact, TemporalStore
from tests.memory.test_temporal import FakeClient

DAY = 24 * 3600 * 1000
T0 = 1704067200000  # 2024-01-01
T30 = T0 + 30 * DAY
T45 = T0 + 45 * DAY
T60 = T0 + 60 * DAY
T90 = T0 + 90 * DAY


@pytest.fixture
def store():
    s = TemporalStore(FakeClient())
    s.ensure_schema()
    return s


def _facts(store):
    """Two consecutive employers plus one long-running preference."""
    store.add_entity("u1", "张明", created_at=T0)
    store.add_entity("c1", "Acme", created_at=T0)
    store.add_entity("c2", "Globex", created_at=T0)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T0, invalid_at=T30, created_at=T0, conflict_key="works_at")
    store.add_fact("f2", "works at Globex", "u1", "c2",
                   valid_at=T30, created_at=T30, conflict_key="works_at")
    store.add_fact("f3", "prefers dark mode", "u1", "u1",
                   valid_at=T0, created_at=T0)


# -- decay ------------------------------------------------------------------


def test_decay_is_one_for_a_fact_that_is_current():
    f = Fact(uuid="f", fact="x", valid_at=T0, invalid_at=OPEN)
    assert time_decay_score(f, T0) == pytest.approx(1.0)


def test_decay_halves_after_one_half_life():
    f = Fact(uuid="f", fact="x", valid_at=T0)
    assert time_decay_score(f, T30) == pytest.approx(0.5)


def test_decay_quarters_after_two_half_lives():
    f = Fact(uuid="f", fact="x", valid_at=T0)
    assert time_decay_score(f, T60) == pytest.approx(0.25)


def test_decay_uses_validity_not_knowledge_time():
    """A fact true for a year is old even if we only learned it yesterday."""
    f = Fact(uuid="f", fact="x", valid_at=T0, invalid_at=T90, created_at=T90)
    assert time_decay_score(f, T90) < 0.2


def test_decay_of_ended_fact_stops_at_invalid_at():
    """Once a fact stops being true, it does not keep decaying further."""
    f = Fact(uuid="f", fact="x", valid_at=T0, invalid_at=T30)
    assert time_decay_score(f, T30) == pytest.approx(time_decay_score(f, T90))


def test_zero_half_life_disables_decay():
    f = Fact(uuid="f", fact="x", valid_at=T0)
    assert time_decay_score(f, T90, half_life_ms=0) == 1.0


# -- temporal gating comes first --------------------------------------------


def test_recall_excludes_facts_not_valid_at_query_time(store):
    _facts(store)
    result = recall(store, at=T0, now=T0)
    texts = {i.content for i in result.items}
    assert "works at Acme" in texts
    assert "works at Globex" not in texts  # not yet true at T0


def test_recall_at_later_time_returns_the_newer_fact(store):
    _facts(store)
    result = recall(store, at=T60, now=T60)
    texts = {i.content for i in result.items}
    assert "works at Globex" in texts
    assert "works at Acme" not in texts  # superseded and no longer valid


def test_gating_precedes_scoring(store):
    """An expired fact must not consume rank budget.

    If scoring ran before gating, a stale-but-highly-decayed fact could
    still occupy a slot and push a current one out of top_k.
    """
    _facts(store)
    result = recall(store, at=T60, now=T60, config=MemoryRecallConfig(top_k=1))
    assert len(result.items) == 1
    assert result.items[0].content == "works at Globex"


def test_superseded_facts_hidden_by_default(store):
    _facts(store)
    result = recall(store, at=T60, now=T60)
    assert all(i.metadata["is_current"] for i in result.items)


def test_superseded_facts_can_be_included(store):
    """Distinguishes the two axes, which are easy to conflate.

    A fact can be *still true* (valid interval covers T) while we have
    *stopped believing it* (expired_at <= T). ``include_superseded``
    controls only the second: it reveals beliefs we have abandoned, never
    facts that were not true at T.
    """
    store.add_entity("u1", "张明", created_at=T0)
    store.add_entity("c1", "Acme", created_at=T0)
    store.add_entity("c2", "Globex", created_at=T0)
    # Valid through T60, but superseded at T30 when we learned he moved.
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T0, invalid_at=T60, created_at=T0, conflict_key="works_at")
    store.add_fact("f2", "works at Globex", "u1", "c2",
                   valid_at=T30, created_at=T30, conflict_key="works_at")

    hidden = recall(store, at=T45, now=T45)
    assert "works at Acme" not in {i.content for i in hidden.items}

    shown = recall(store, at=T45, now=T45,
                   config=MemoryRecallConfig(include_superseded=True))
    assert "works at Acme" in {i.content for i in shown.items}


# -- ranking ----------------------------------------------------------------


def test_newer_fact_ranks_above_older(store):
    store.add_entity("u1", "A", created_at=T0)
    store.add_entity("u2", "B", created_at=T0)
    store.add_fact("old", "likes tea", "u1", "u2", valid_at=T0, created_at=T0)
    store.add_fact("new", "likes coffee", "u1", "u2", valid_at=T90, created_at=T90)
    result = recall(store, at=T90, now=T90)
    assert result.items[0].content == "likes coffee"


def test_scores_are_populated(store):
    _facts(store)
    result = recall(store, at=T60, now=T60)
    assert all(i.score is not None for i in result.items)
    assert all(0 < i.score <= 1 for i in result.items)


def test_top_k_limits_results(store):
    _facts(store)
    result = recall(store, at=T60, now=T60, config=MemoryRecallConfig(top_k=1))
    assert len(result.items) <= 1


# -- composition with other channels ----------------------------------------


def test_extra_channel_participates_in_fusion(store):
    """A second channel can reorder results without new fusion code."""
    _facts(store)
    # Push the older preference to the top from another channel.
    result = recall(store, at=T60, now=T60,
                    extra_channels=[["f3"]])
    assert result.metadata["candidates"] >= 2
    ids = [i.metadata["uuid"] for i in result.items]
    assert "f3" in ids


def test_extra_channel_can_surface_otherwise_lower_ranked_fact(store):
    _facts(store)
    without = recall(store, at=T60, now=T60)
    with_extra = recall(store, at=T60, now=T60, extra_channels=[["f3"]])
    # The preference was valid the whole time; boosting it should not lose
    # the current employer, only reorder.
    assert {i.metadata["uuid"] for i in without.items} == \
           {i.metadata["uuid"] for i in with_extra.items}


# -- retriever contract -----------------------------------------------------


def test_retriever_implements_shared_contract(store):
    _facts(store)
    result = TemporalRetriever(store, now=T60).search("", at=T60)
    assert not result.is_empty
    assert result.metadata["__retriever"] == "TemporalRetriever"


def test_retriever_items_carry_temporal_metadata(store):
    _facts(store)
    result = TemporalRetriever(store, now=T60).search("", at=T60)
    meta = result.items[0].metadata
    assert {"uuid", "valid_at", "invalid_at", "is_current"} <= set(meta)


def test_empty_store_returns_empty_result():
    store = TemporalStore(FakeClient())
    result = recall(store, at=T0, now=T0)
    assert result.is_empty
