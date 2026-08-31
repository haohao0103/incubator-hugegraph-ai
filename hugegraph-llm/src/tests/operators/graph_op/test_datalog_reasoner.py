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

"""Tests for the deterministic Datalog reasoning operator.

Covers constant/predicate normalisation, the semi-naive fixpoint engine,
query evaluation, graph loading, and the operator protocol.
"""

import pytest

from hugegraph_llm.operators.graph_op.datalog_reasoner import (
    BodyAtom,
    DatalogFact,
    DatalogReasoner,
    DatalogReasonerOp,
    DatalogRule,
    to_const,
    to_pred,
)


# ============================================================
# Normalisation
# ============================================================


class TestNormalisation:
    def test_to_pred_lowercases(self):
        assert to_pred("Used_Device") == "used_device"

    def test_to_pred_replaces_spaces(self):
        assert to_pred("used device") == "used_device"

    def test_to_pred_replaces_punctuation(self):
        assert to_pred("used-device.v2") == "used_device_v2"

    def test_to_pred_preserves_cjk(self):
        assert to_pred("司机") == "司机"

    def test_to_pred_empty_becomes_unknown(self):
        assert to_pred("!!!") == "unknown"

    def test_to_pred_leading_digit_prefixed(self):
        assert to_pred("2fast") == "n2fast"

    def test_to_pred_has_no_const_prefix(self):
        """Predicate names must match rule syntax verbatim."""
        assert to_pred("used_device") == "used_device"

    def test_to_const_adds_prefix(self):
        assert to_const("Microsoft") == "c_microsoft"

    def test_to_const_cjk(self):
        assert to_const("张三") == "c_张三"

    def test_to_const_digit(self):
        assert to_const("2fast") == "c_n2fast"

    def test_to_const_from_int(self):
        assert to_const(42) == "c_n42"


# ============================================================
# Data structures
# ============================================================


class TestDataStructures:
    def test_fact_str(self):
        f = DatalogFact("parent", ("c_tom", "c_bob"))
        assert str(f) == "parent(c_tom, c_bob)"

    def test_fact_hashable(self):
        a = DatalogFact("p", ("c_a",))
        b = DatalogFact("p", ("c_a",))
        assert a == b
        assert len({a, b}) == 1

    def test_body_atom_is_namedtuple(self):
        atom = BodyAtom("parent", ("X", "Y"))
        assert atom.predicate == "parent"
        assert atom.args == ("X", "Y")

    def test_rule_fields(self):
        rule = DatalogRule("anc", ("X", "Y"), [BodyAtom("par", ("X", "Z"))])
        assert rule.head_predicate == "anc"
        assert rule.head_args == ("X", "Y")
        assert len(rule.body) == 1


# ============================================================
# Fact input shapes
# ============================================================


class TestAddFact:
    def test_add_fact_instance(self):
        r = DatalogReasoner()
        r.add_fact(DatalogFact("p", ("c_a",)))
        assert DatalogFact("p", ("c_a",)) in r.facts

    def test_add_fact_string(self):
        r = DatalogReasoner()
        r.add_fact("parent(tom, bob)")
        assert DatalogFact("parent", ("tom", "bob")) in r.facts

    def test_add_fact_triple_dict(self):
        r = DatalogReasoner()
        r.add_fact({"subject": "Alice", "predicate": "knows", "object": "Bob"})
        assert DatalogFact("knows", ("c_alice", "c_bob")) in r.facts

    def test_add_fact_edge_dict(self):
        r = DatalogReasoner()
        r.add_fact({"source": "d01", "target": "dev_a", "type": "used_device"})
        assert DatalogFact("used_device", ("c_d01", "c_dev_a")) in r.facts

    def test_add_fact_edge_dict_default_type(self):
        r = DatalogReasoner()
        r.add_fact({"source": "a", "target": "b"})
        assert DatalogFact("connected_to", ("c_a", "c_b")) in r.facts

    def test_add_fact_edge_dict_variant_keys(self):
        r = DatalogReasoner()
        r.add_fact({"source_id": "a", "target_id": "b", "relation": "rel"})
        assert DatalogFact("rel", ("c_a", "c_b")) in r.facts

    def test_add_fact_edge_missing_endpoint_skipped(self):
        r = DatalogReasoner()
        r.add_fact({"source": "a", "type": "rel"})  # no target
        assert len(r.facts) == 0

    def test_add_fact_entity_dict(self):
        r = DatalogReasoner()
        r.add_fact({"id": "d01", "type": "driver"})
        assert DatalogFact("driver", ("c_d01",)) in r.facts

    def test_add_fact_entity_dict_default_type(self):
        r = DatalogReasoner()
        r.add_fact({"name": "x"})
        assert DatalogFact("entity", ("c_x",)) in r.facts

    def test_add_fact_unrecognised_skipped(self):
        r = DatalogReasoner()
        r.add_fact(["not", "supported"])
        assert len(r.facts) == 0

    def test_add_fact_duplicate_ignored(self):
        r = DatalogReasoner()
        r.add_fact("p(a)")
        r.add_fact("p(a)")
        assert len(r.facts) == 1

    def test_add_fact_empty_arg_raises(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_fact(DatalogFact("p", ("",)))

    def test_parse_fact_string_invalid(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_fact("not_a_fact")

    def test_parse_fact_string_empty_arg(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_fact("p(a, )")

    def test_parse_fact_string_variable_rejected(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_fact("p(Tom)")

    def test_add_fact_entity_empty_id_skipped(self):
        """An entity whose id is empty carries no information -> skipped."""
        r = DatalogReasoner()
        r.add_fact({"id": "", "type": "driver"})
        assert len(r.facts) == 0


# ============================================================
# Rule parsing
# ============================================================


class TestAddRule:
    def test_add_rule(self):
        r = DatalogReasoner()
        r.add_rule("anc(X, Y) :- par(X, Y).")
        assert len(r.rules) == 1
        assert r.rules[0].head_predicate == "anc"

    def test_add_rule_missing_colon_dash(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_rule("anc(X, Y) par(X, Y)")

    def test_add_rule_bad_head(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_rule("anc X Y :- par(X, Y)")

    def test_add_rule_no_body_atoms(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.add_rule("anc(X, Y) :- .")


# ============================================================
# Engine internals
# ============================================================


class TestEngineInternals:
    def test_clear(self):
        r = DatalogReasoner()
        r.add_fact("p(a)")
        r.add_rule("q(X) :- p(X).")
        r.clear()
        assert len(r.facts) == 0
        assert len(r.rules) == 0

    def test_is_variable(self):
        r = DatalogReasoner()
        assert r._is_variable("X") is True
        assert r._is_variable("x") is False
        assert r._is_variable("") is False

    def test_unify_arity_mismatch(self):
        r = DatalogReasoner()
        assert r._unify(("X",), ("a", "b"), {}) is None

    def test_unify_constant_mismatch(self):
        r = DatalogReasoner()
        assert r._unify(("a",), ("b",), {}) is None

    def test_unify_conflicting_binding(self):
        r = DatalogReasoner()
        assert r._unify(("X", "X"), ("a", "b"), {}) is None

    def test_unify_consistent_binding(self):
        r = DatalogReasoner()
        assert r._unify(("X", "X"), ("a", "a"), {}) == {"X": "a"}

    def test_instantiate_unbound_returns_none(self):
        r = DatalogReasoner()
        assert r._instantiate(("X",), {}) is None

    def test_instantiate_constants(self):
        r = DatalogReasoner()
        assert r._instantiate(("a", "b"), {}) == ("a", "b")

    def test_instantiate_fact_none(self):
        r = DatalogReasoner()
        assert r._instantiate_fact("p", ("X",), {}) is None

    def test_apply_rule_without_body(self):
        """A rule with no body atoms is a bare fact."""
        r = DatalogReasoner()
        rule = DatalogRule("p", ("c_a",), [])
        assert r._apply_rule(rule) == {DatalogFact("p", ("c_a",))}

    def test_apply_rule_naive(self):
        """Passing delta_index=None exercises the naive path."""
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        rule = DatalogRule("anc", ("X", "Y"), [BodyAtom("par", ("X", "Y"))])
        assert DatalogFact("anc", ("c_tom", "c_bob")) in r._apply_rule(rule, None)

    def test_apply_rule_bodyless_with_variable(self):
        """A bodyless rule whose head has an unbound variable yields nothing."""
        r = DatalogReasoner()
        rule = DatalogRule("p", ("X",), [])
        assert r._apply_rule(rule) == set()

    def test_apply_rule_head_variable_unbound(self):
        """A head variable absent from the body cannot be instantiated."""
        r = DatalogReasoner()
        r.add_fact("q(c_a)")
        rule = DatalogRule("p", ("X", "Y"), [BodyAtom("q", ("X",))])
        assert r._apply_rule(rule) == set()


# ============================================================
# Fixpoint evaluation
# ============================================================


class TestDeriveAll:
    def test_transitive_closure(self):
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        r.add_fact("par(c_bob, c_ann)")
        r.add_rule("anc(X, Y) :- par(X, Y).")
        r.add_rule("anc(X, Y) :- anc(X, Z), par(Z, Y).")
        facts = r.derive_all()
        assert "anc(c_tom, c_bob)" in facts
        assert "anc(c_tom, c_ann)" in facts

    def test_derive_all_idempotent(self):
        r = DatalogReasoner()
        r.add_fact("p(c_a)")
        r.derive_all()
        first = set(r.facts)
        r.derive_all()
        assert set(r.facts) == first

    def test_derive_all_no_rules(self):
        r = DatalogReasoner()
        r.add_fact("p(c_a)")
        assert r.derive_all() == ["p(c_a)"]

    def test_fixpoint_terminates_on_cycle(self):
        """Recursive rules over a cycle must still terminate."""
        r = DatalogReasoner()
        r.add_fact("edge(c_a, c_b)")
        r.add_fact("edge(c_b, c_a)")
        r.add_rule("reach(X, Y) :- edge(X, Y).")
        r.add_rule("reach(X, Y) :- reach(X, Z), edge(Z, Y).")
        facts = r.derive_all()
        assert "reach(c_a, c_a)" in facts


# ============================================================
# Query
# ============================================================

class TestQuery:
    def test_query_with_question_mark_var(self):
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        assert r.query("par(c_tom, ?Y)") == [{"Y": "c_bob"}]

    def test_query_with_uppercase_var(self):
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        assert r.query("par(c_tom, Y)") == [{"Y": "c_bob"}]

    def test_query_no_match(self):
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        assert r.query("zzz(c_tom, ?Y)") == []

    def test_query_constant_only_no_var(self):
        """A pattern with no variables yields no rows."""
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        assert r.query("par(c_tom, c_bob)") == []

    def test_query_with_bindings(self):
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        r.add_fact("par(c_tom, c_ann)")
        assert r.query("par(c_tom, ?Y)", {"Y": "c_ann"}) == [{"Y": "c_ann"}]

    def test_query_triggers_derivation(self):
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        r.add_fact("par(c_bob, c_ann)")
        r.add_rule("anc(X, Y) :- par(X, Y).")
        r.add_rule("anc(X, Y) :- anc(X, Z), par(Z, Y).")
        # Results are deterministic (sorted), not set-iteration dependent.
        assert r.query("anc(c_tom, ?Y)") == [{"Y": "c_ann"}, {"Y": "c_bob"}]

    def test_sorted_facts_is_deterministic(self):
        """Facts live in sets; sorted_facts() must give a stable order."""
        r = DatalogReasoner()
        r.add_fact("p(c_z)")
        r.add_fact("p(c_a)")
        r.add_fact("p(c_m)")
        assert [str(f) for f in r.sorted_facts()] == ["p(c_a)", "p(c_m)", "p(c_z)"]

    def test_derive_all_is_deterministic(self):
        r = DatalogReasoner()
        r.add_fact("p(c_z)")
        r.add_fact("p(c_a)")
        assert r.derive_all() == ["p(c_a)", "p(c_z)"]

    def test_query_invalid_syntax(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.query("not valid")

    def test_query_empty_var_name(self):
        r = DatalogReasoner()
        with pytest.raises(ValueError):
            r.query("p(?)")

    def test_query_dedupes(self):
        r = DatalogReasoner()
        r.add_fact("p(c_a)")
        r.add_rule("p(X) :- p(X).")
        assert r.query("p(?Y)") == [{"Y": "c_a"}]

    def test_query_mixed_bound_and_matched_vars(self):
        """One variable supplied via bindings, another solved by matching."""
        r = DatalogReasoner()
        r.add_fact("par(c_tom, c_bob)")
        assert r.query("par(?X, ?Y)", {"X": "c_tom"}) == [
            {"X": "c_tom", "Y": "c_bob"}
        ]


# ============================================================
# Graph loading
# ============================================================


class _StoreGraph:
    """Graph exposing the find_edges/find_nodes interface."""

    def find_edges(self):
        return [
            {"source": "d01", "target": "dev_a", "type": "used_device"},
            {"type": "broken"},  # missing endpoints -> skipped
        ]

    def find_nodes(self):
        return [{"id": "d01", "type": "driver"}]


class _NxGraph:
    """networkx-style graph yielding tuples."""

    def edges(self):
        return [
            ("d01", "dev_a", {"type": "used_device"}),
            ("d02", "dev_b"),  # 2-tuple, no data dict
        ]

    def nodes(self):
        return [("d01", {"type": "driver"}), ("d02", "not-a-dict")]


class _AttrGraph:
    """Graph with non-callable attributes."""

    edges = [{"source": "a", "target": "b", "type": "rel"}]
    nodes = {"a": {"id": "a", "type": "entity"}}


class _JunkGraph:
    """Graph containing elements that match none of the known shapes."""

    edges = ["junk-edge", 123]
    nodes = ["junk-node", 456]


class TestLoadFromGraph:
    def test_store_interface(self):
        r = DatalogReasoner()
        added = r.load_from_graph(_StoreGraph())
        assert added == 2  # 1 edge + 1 node (broken edge skipped)

    def test_networkx_style(self):
        r = DatalogReasoner()
        added = r.load_from_graph(_NxGraph())
        assert added == 4  # 2 edges + 2 nodes
        assert DatalogFact("used_device", ("c_d01", "c_dev_a")) in r.facts
        assert DatalogFact("driver", ("c_d01",)) in r.facts

    def test_attr_graph(self):
        r = DatalogReasoner()
        added = r.load_from_graph(_AttrGraph())
        assert added == 2

    def test_empty_object(self):
        r = DatalogReasoner()
        assert r.load_from_graph(object()) == 0

    def test_junk_elements_skipped(self):
        """Elements that are neither dicts nor recognised tuples are skipped."""
        r = DatalogReasoner()
        assert r.load_from_graph(_JunkGraph()) == 0
        assert len(r.facts) == 0

    def test_tuple_converters(self):
        assert DatalogReasoner._edge_tuple_to_dict(("a", "b")) == {
            "source": "a", "target": "b", "type": "connected_to"
        }
        assert DatalogReasoner._node_tuple_to_dict(("a", "not-dict")) == {
            "id": "a", "type": "entity"
        }


# ============================================================
# Operator
# ============================================================


class TestOperator:
    def test_run_full_pipeline(self):
        op = DatalogReasonerOp()
        ctx = {
            "entities": [{"id": "d01", "type": "driver"},
                         {"id": "d02", "type": "driver"}],
            "relations": [
                {"source": "d01", "target": "dev_a", "type": "used_device"},
                {"source": "d02", "target": "dev_a", "type": "used_device"},
            ],
            "datalog_rules": [
                "shared_device(X, Y) :- used_device(X, D), used_device(Y, D).",
            ],
        }
        op.run(ctx)
        assert ctx["datalog_stats"]["loaded"] == 4
        assert ctx["datalog_stats"]["rules"] == 1
        assert ctx["datalog_stats"]["derived"] > 0

    def test_run_transitive_closure(self):
        op = DatalogReasonerOp()
        ctx = {
            "relations": [
                {"source": "d01", "target": "dev_a", "type": "used_device"},
                {"source": "d02", "target": "dev_a", "type": "used_device"},
                {"source": "d02", "target": "dev_b", "type": "used_device"},
                {"source": "d03", "target": "dev_b", "type": "used_device"},
            ],
            "datalog_rules": [
                "shared_device(X, Y) :- used_device(X, D), used_device(Y, D).",
                "same_gang(X, Y) :- shared_device(X, Y).",
                "same_gang(X, Y) :- same_gang(X, Z), shared_device(Z, Y).",
            ],
        }
        op.run(ctx)
        gangs = [f for f in ctx["datalog_derived"] if f["predicate"] == "same_gang"]
        pairs = {tuple(sorted(g["args"])) for g in gangs}
        # d01 and d03 are linked transitively through d02
        assert ("d01", "d03") in pairs
        assert ("d01", "d02") in pairs

    def test_derived_excludes_loaded(self):
        op = DatalogReasonerOp()
        op.load_relations([{"source": "a", "target": "b", "type": "rel"}])
        op.add_rules(["same(X, Y) :- rel(X, Y)."])
        op._reasoner.derive_all()
        derived = op.derived_facts()
        predicates = {d["predicate"] for d in derived}
        assert "rel" not in predicates
        assert "same" in predicates

    def test_original_names_restored(self):
        op = DatalogReasonerOp()
        op.load_relations([{"source": "Microsoft", "target": "Azure", "type": "owns"}])
        op.add_rules(["owned(X, Y) :- owns(X, Y)."])
        op._reasoner.derive_all()
        derived = op.derived_facts()
        assert derived[0]["args"] == ["Microsoft", "Azure"]

    def test_load_entities_skips_missing_id(self):
        op = DatalogReasonerOp()
        assert op.load_entities([{"type": "driver"}]) == 0

    def test_load_relations_skips_missing_endpoint(self):
        op = DatalogReasonerOp()
        assert op.load_relations([{"source": "a", "type": "rel"}]) == 0

    def test_orig_unmapped_returns_input(self):
        op = DatalogReasonerOp()
        assert op._orig("c_unknown") == "c_unknown"

    def test_check_consistency_empty(self):
        op = DatalogReasonerOp()
        assert op.check_consistency([]) == []

    def test_check_consistency_flags(self):
        op = DatalogReasonerOp()
        ctx = {
            "relations": [{"source": "a", "target": "b", "type": "contradicts"}],
            "datalog_rules": [],
            "conflict_rules": ["contradicts"],
        }
        op.run(ctx)
        assert len(ctx["datalog_conflicts"]) == 1
        assert ctx["datalog_conflicts"][0]["predicate"] == "contradicts"

    def test_run_resets_state_between_calls(self):
        op = DatalogReasonerOp()
        ctx1 = {
            "relations": [{"source": "a", "target": "b", "type": "rel"}],
            "datalog_rules": ["q(X, Y) :- rel(X, Y)."],
        }
        op.run(ctx1)
        first_derived = len(ctx1["datalog_derived"])
        ctx2 = {"relations": [], "datalog_rules": []}
        op.run(ctx2)
        assert ctx2["datalog_stats"]["loaded"] == 0
        assert first_derived >= 1

    def test_inject_custom_reasoner(self):
        engine = DatalogReasoner()
        op = DatalogReasonerOp(reasoner=engine)
        op.load_relations([{"source": "a", "target": "b", "type": "rel"}])
        assert len(engine.facts) == 1
