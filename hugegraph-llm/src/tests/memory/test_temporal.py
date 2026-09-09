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

"""Tests for the bi-temporal store. Offline: fake REST client."""

import pytest

from hugegraph_llm.memory.schema import OPEN, EdgeLabel, VertexLabel, build_schema_dict
from hugegraph_llm.memory.temporal import Fact, TemporalStore

# Epoch millis for readability.
T2024 = 1704067200000  # 2024-01-01
T2025 = 1735689600000  # 2025-01-01
T2026 = 1767225600000  # 2026-01-01


class FakeClient:
    """Minimal HugeGraph REST stand-in backed by dicts."""

    def __init__(self):
        self.vertices = {}   # vid -> {label, props}
        self.edges = {}      # eid -> {label, outV, inV, props}
        self._seq = 0
        self.schema_initialised = False
        self._propertykeys = set()
        self._vertexlabels = set()
        self._edgelabels = set()

    def init_schema(self, rebuild=True):
        self.schema_initialised = True

    def ensure_schema(self, schema):
        """Mirror the real client: create what is missing, report counts."""
        self.schema_initialised = True
        self.schema_seen = schema
        created = {"propertykeys": 0, "vertexlabels": 0, "edgelabels": 0}
        for key, singular in (
            ("propertykeys", "propertykeys"),
            ("vertexlabels", "vertexlabels"),
            ("edgelabels", "edgelabels"),
        ):
            for item in schema.get(key, []):
                if item["name"] not in getattr(self, f"_{singular}", set()):
                    created[key] += 1
        return created

    def upsert_vertex(self, label, vid, props):
        self.vertices[str(vid)] = {"label": label, "properties": dict(props)}

    def update_vertex(self, label, vid, props):
        self.vertices[str(vid)]["properties"].update(props)

    def get_vertex(self, vid):
        return self.vertices.get(str(vid))

    def get_vertices_by_label(self, label, limit=1000):
        return [
            {"id": vid, **data}
            for vid, data in self.vertices.items()
            if data["label"] == label
        ]

    def upsert_edge(self, label, src, tgt, slabel, tlabel, props):
        self._seq += 1
        eid = f"{label}-{src}->{tgt}-{self._seq}"
        self.edges[eid] = {
            "label": label, "outV": str(src), "inV": str(tgt),
            "properties": dict(props),
        }
        return eid

    def update_edge(self, edge_id, label, props):
        self.edges[edge_id]["properties"].update(props)

    def get_edges_of(self, vid, direction="OUT", limit=1000):
        vid = str(vid)
        return [
            {"id": eid, **edge}
            for eid, edge in self.edges.items()
            if edge["outV"] == vid
        ]


@pytest.fixture
def store():
    client = FakeClient()
    s = TemporalStore(client)
    s.ensure_schema()
    return s


def _two_entities(store):
    store.add_entity("u1", "张明", created_at=T2024)
    store.add_entity("c1", "Acme", created_at=T2024)
    store.add_entity("c2", "Globex", created_at=T2024)


# -- schema -----------------------------------------------------------------


def test_schema_is_built_with_bi_temporal_fields():
    schema = build_schema_dict()
    names = {pk["name"] for pk in schema["propertykeys"]}
    assert {"valid_at", "invalid_at", "created_at", "expired_at"} <= names


def test_temporal_fields_are_long():
    """Millis compare and index natively; text dates would not."""
    schema = build_schema_dict()
    types = {pk["name"]: pk["data_type"] for pk in schema["propertykeys"]}
    assert all(types[f] == "LONG" for f in
               ("valid_at", "invalid_at", "created_at", "expired_at"))


def test_schema_has_range_indexes_for_as_of():
    schema = build_schema_dict()
    fields = {(i["field"], i["index_type"]) for i in schema["indexes"]}
    assert ("valid_at", "RANGE") in fields
    assert ("invalid_at", "RANGE") in fields


def test_edges_declare_source_target_pairs():
    """HugeGraph locks one endpoint pair per edge label."""
    for name, source, target, _props in build_schema_dict()["edgelabels"]:
        assert source and target and name


# -- fact validity ----------------------------------------------------------


def test_fact_valid_at_time_is_half_open():
    f = Fact(uuid="f", fact="x", valid_at=100, invalid_at=200)
    assert f.valid_at_time(100)
    assert f.valid_at_time(199)
    assert not f.valid_at_time(200)   # end exclusive: periods may abut
    assert not f.valid_at_time(99)


def test_open_interval_is_sentinel_not_zero():
    f = Fact(uuid="f", fact="x", valid_at=0)
    assert f.invalid_at == OPEN and OPEN > 0
    assert f.valid_at_time(2**62)


# -- as_of ------------------------------------------------------------------


def test_as_of_returns_facts_true_at_that_time(store):
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    facts = store.as_of(T2024 + 1000)
    assert [f.fact for f in facts] == ["works at Acme"]


def test_as_of_excludes_fact_not_yet_known(store):
    """Learned in 2026, true in 2024 -> invisible when asking about 2024.

    This is the transaction-time axis: a fact we did not know at T cannot
    be part of "what we believed at T".
    """
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2026,
                   conflict_key="works_at")
    assert store.as_of(T2024 + 1000) == []


def test_as_of_after_invalid_at_returns_nothing(store):
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    assert store.as_of(T2025 + 1000) == []


# -- invalidation (the core of the design) -----------------------------------


def test_new_fact_expires_the_old_one(store):
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    expired = store.add_fact("f2", "works at Globex", "u1", "c2",
                             valid_at=T2025, created_at=T2025,
                             conflict_key="works_at")
    assert expired == ["f1"]


def test_superseded_fact_is_expired_not_deleted(store):
    """Deleting history would defeat the entire point of a temporal store."""
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    store.add_fact("f2", "works at Globex", "u1", "c2",
                   valid_at=T2025, created_at=T2025, conflict_key="works_at")

    all_facts = store.all_facts()
    assert len(all_facts) == 2  # both still present
    old = next(f for f in all_facts if f.uuid == "f1")
    assert old.expired_at == T2025
    assert not old.is_current


def test_as_of_returns_historical_answer_after_supersession(store):
    """The end-to-end property: history stays answerable."""
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    store.add_fact("f2", "works at Globex", "u1", "c2",
                   valid_at=T2025, created_at=T2025, conflict_key="works_at")

    assert [f.fact for f in store.as_of(T2024 + 1000)] == ["works at Acme"]
    assert [f.fact for f in store.as_of(T2025 + 1000)] == ["works at Globex"]


def test_history_of_shows_evolution(store):
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    store.add_fact("f2", "works at Globex", "u1", "c2",
                   valid_at=T2025, created_at=T2025, conflict_key="works_at")
    assert [f.fact for f in store.history_of("works_at")] == [
        "works at Acme", "works at Globex",
    ]


def test_facts_without_conflict_key_are_never_expired(store):
    """Not every fact contradicts its predecessor."""
    _two_entities(store)
    store.add_fact("f1", "likes coffee", "u1", "c1", valid_at=T2024, created_at=T2024)
    expired = store.add_fact("f2", "likes tea", "u1", "c1",
                             valid_at=T2024, created_at=T2024)
    assert expired == []
    assert len(store.all_facts()) == 2


def test_already_expired_fact_is_not_expired_again(store):
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024,
                   conflict_key="works_at")
    store.add_fact("f2", "works at Globex", "u1", "c2",
                   valid_at=T2025, created_at=T2025, conflict_key="works_at")
    expired = store.add_fact("f3", "works at Initech", "u1", "c1",
                             valid_at=T2026, created_at=T2026, conflict_key="works_at")
    assert expired == ["f2"]  # only the currently-believed one


# -- between ----------------------------------------------------------------


def test_between_finds_overlapping_facts(store):
    _two_entities(store)
    store.add_fact("f1", "works at Acme", "u1", "c1",
                   valid_at=T2024, invalid_at=T2025, created_at=T2024)
    assert [f.fact for f in store.between(T2024, T2025)] == ["works at Acme"]
    assert store.between(T2026, T2026 + 1000) == []


# -- state graph ------------------------------------------------------------


def test_current_state_is_the_one_without_end(store):
    store.add_entity("u1", "张明", created_at=T2024)
    store.add_state("s1", "active", "u1", valid_from=T2024, valid_to=T2025)
    store.add_state("s2", "suspended", "u1", valid_from=T2025)
    current = store.current_state("u1")
    assert current["name"] == "suspended"


def test_current_state_none_when_no_open_state(store):
    store.add_entity("u1", "张明", created_at=T2024)
    store.add_state("s1", "active", "u1", valid_from=T2024, valid_to=T2025)
    assert store.current_state("u1") is None


def test_close_state_ends_a_period(store):
    store.add_entity("u1", "张明", created_at=T2024)
    store.add_state("s1", "active", "u1", valid_from=T2024)
    store.close_state("s1", T2025)
    assert store.current_state("u1") is None


def test_transition_path_is_chronological(store):
    store.add_entity("u1", "张明", created_at=T2024)
    store.add_state("s2", "suspended", "u1", valid_from=T2025)
    store.add_state("s1", "active", "u1", valid_from=T2024, valid_to=T2025)
    assert store.transition_path("u1") == ["active", "suspended"]


def test_transition_path_is_per_entity(store):
    store.add_entity("u1", "A", created_at=T2024)
    store.add_entity("u2", "B", created_at=T2024)
    store.add_state("s1", "active", "u1", valid_from=T2024)
    store.add_state("s2", "trial", "u2", valid_from=T2024)
    assert store.transition_path("u1") == ["active"]
