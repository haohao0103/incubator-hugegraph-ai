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

"""
Deterministic Datalog reasoning for HugeGraph-backed knowledge graphs.

Provides rule-based derivation (transitive closure, relation completion,
consistency checking) on top of graph data, using bottom-up semi-naive
fixpoint evaluation. **No LLM is involved** — every derived fact carries a
complete, reproducible derivation chain, which is what makes it auditable.

Origin
------
The evaluation core (DatalogFact / DatalogRule / unification / semi-naive
fixpoint) is adapted from Semantica v0.6.6
(``semantica/reasoning/datalog_reasoner.py``, MIT). It was extracted because
that module is self-contained (standard library only) and because the
surrounding Semantica stack is largely unusable in production:

- its Rete engine is a stub (``_matches()`` / ``_can_join()`` return True)
- ``SemanticaWorker`` is an empty ``time.sleep(5)`` loop
- the ``/build`` endpoint returns "accepted" without doing anything

Only the Datalog engine is real, so only the Datalog engine was taken.

When to use / when NOT to use
-----------------------------
**Use** when all three hold:
1. Relations are dense and the structure itself carries meaning
   (gangs, lineage, propagation chains, multi-hop dependencies)
2. Rules are deterministic and symbolisable
   ("shares >= 2 devices -> suspicious"), not fuzzy semantic similarity
3. Explainability / auditability is required

**Do NOT use** for:
- Unstructured text retrieval or fuzzy recall — that is the domain of
  GraphRAG / vector search. This module has no NL2SQL and no embeddings.
- **Full-graph reasoning on probabilistic extractions.** Feeding
  LLM-extracted triples (probabilistic) into a deterministic reasoner
  produces *explainable errors*: every step is traceable, but the starting
  facts may be wrong, which makes the wrong conclusion look trustworthy.
  Run this only on high-confidence subgraphs whose entities have passed
  entity resolution.

Usage
-----
    op = DatalogReasonerOp()
    context = {
        "entities": [{"id": "d01", "type": "driver"}, ...],
        "relations": [{"source": "d01", "target": "dev_a",
                       "type": "used_device"}, ...],
        "datalog_rules": [
            "shared_device(X, Y) :- used_device(X, D), used_device(Y, D).",
            "same_gang(X, Y) :- shared_device(X, Y).",
            "same_gang(X, Y) :- same_gang(X, Z), shared_device(Z, Y).",
        ],
    }
    op.run(context)
    context["datalog_derived"]   # newly derived facts (original names)
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from hugegraph_llm.utils.log import log

# Datalog requires constants to start with a lowercase letter (an uppercase
# initial marks a logical variable). HugeGraph ids may start with a digit,
# an uppercase letter, or a CJK character, so every constant is prefixed.
_CONST_PREFIX = "c_"


@dataclass(frozen=True)
class DatalogFact:
    """A ground (variable-free) fact, e.g. ``used_device(c_d01, c_dev_a)``."""

    predicate: str
    args: Tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"


class BodyAtom(NamedTuple):
    """A single predicate condition in a rule body."""

    predicate: str
    args: Tuple[str, ...]


@dataclass
class DatalogRule:
    """A Horn clause: ``head :- body1, body2, ...``."""

    head_predicate: str
    head_args: Tuple[str, ...]
    body: List[BodyAtom]


# ============================================================
# Constant normalisation (HugeGraph id <-> Datalog constant)
# ============================================================


def to_pred(name: Any) -> str:
    """Normalise a predicate name (relation / entity type).

    Lowercases and replaces punctuation with ``_``, but **does not add the
    ``c_`` prefix** — predicate names must match those written in the rules
    verbatim (e.g. a rule ``shared_device(X, Y) :- used_device(X, D)`` must
    match a fact whose predicate is ``used_device``, not ``c_used_device``).
    """
    text = str(name).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9a-z_\u4e00-\u9fff]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    if text[0].isdigit():
        text = f"n{text}"
    return text


def to_const(name: Any) -> str:
    """Map an arbitrary HugeGraph id to a legal Datalog constant.

    Datalog requires constants to start with a lowercase letter (an
    uppercase initial marks a logical variable). HugeGraph ids may start
    with a digit, an uppercase letter, or a CJK character, so every
    constant is prefixed with ``c_``. CJK characters are preserved (they
    are not uppercase, so they remain legal).
    """
    return f"{_CONST_PREFIX}{to_pred(name)}"


# ============================================================
# Datalog engine
# ============================================================


class DatalogReasoner:
    """Datalog engine using bottom-up semi-naive fixpoint evaluation.

    Supports recursive rules and multi-hop inference, and terminates on
    finite graphs (the fixpoint is reached because the fact set only grows
    and is bounded by the finite Herbrand base).
    """

    def __init__(self):
        self._fact_index: Dict[str, Set[DatalogFact]] = defaultdict(set)
        self._all_facts: Set[DatalogFact] = set()
        self._rules: List[DatalogRule] = []
        self._derived = False
        self._delta_old: Set[DatalogFact] = set()
        self._delta_new: Set[DatalogFact] = set()

    # ---- state ----

    def clear(self) -> None:
        """Remove all facts and rules."""
        self._fact_index.clear()
        self._all_facts.clear()
        self._rules.clear()
        self._derived = False
        self._delta_old.clear()
        self._delta_new.clear()

    @property
    def facts(self) -> Set[DatalogFact]:
        return set(self._all_facts)

    @property
    def rules(self) -> List[DatalogRule]:
        return list(self._rules)

    # ---- input ----

    def add_fact(self, fact: Any) -> None:
        """Add a ground fact.

        Accepts:
          - a :class:`DatalogFact` (used as-is)
          - a string: ``"parent(tom, bob)"``
          - a triple dict: ``{"subject": .., "predicate": .., "object": ..}``
          - an edge dict: ``{"source": .., "target": .., "type": ..}``
          - an entity dict: ``{"id": .., "type": ..}``
        ``DatalogFact`` inputs are already normalised; other shapes have
        their identifiers normalised to legal Datalog constants.
        """
        if isinstance(fact, DatalogFact):
            parsed: Optional[DatalogFact] = fact
        else:
            parsed = self._to_fact(fact)
        if parsed is None:
            log.warning("Unrecognised fact format, skipping: %s", fact)
            return
        for arg in parsed.args:
            if not arg:
                raise ValueError(f"Facts cannot contain empty arguments: {fact}")
        if parsed not in self._all_facts:
            self._all_facts.add(parsed)
            self._fact_index[parsed.predicate].add(parsed)
            self._derived = False

    def add_rule(self, rule_str: str) -> None:
        """Add a Horn clause, e.g. ``"anc(X, Y) :- par(X, Z), par(Z, Y)."``"""
        self._rules.append(self._parse_rule_string(rule_str))
        self._derived = False

    # ---- parsing ----

    @staticmethod
    def _parse_fact_string(s: str) -> DatalogFact:
        match = re.match(
            r"^\s*([a-zA-Z0-9_]+)\s*\(\s*([^)]+)\s*\)\s*\.?\s*$", s.strip()
        )
        if not match:
            raise ValueError(f"Invalid fact syntax: {s}")
        args = tuple(a.strip() for a in match.group(2).split(","))
        for arg in args:
            if not arg:
                raise ValueError(f"Empty argument found in fact: {s}")
            if arg[0].isupper():
                raise ValueError(
                    f"Facts must be constants only (no variables). "
                    f"Found variable '{arg}' in {s}"
                )
        return DatalogFact(match.group(1), args)

    def _to_fact(self, fact: Any) -> Optional[DatalogFact]:
        """Normalise supported input shapes into a DatalogFact."""
        if isinstance(fact, str):
            return self._parse_fact_string(fact)
        if not isinstance(fact, dict):
            return None

        # RDF-style triple
        if "subject" in fact and "predicate" in fact and "object" in fact:
            return DatalogFact(
                to_pred(fact["predicate"]),
                (to_const(fact["subject"]), to_const(fact["object"])),
            )

        # Edge (source/target/type)
        if any(k in fact for k in ("source", "source_id", "source_name")):
            source = fact.get("source", fact.get("source_id", fact.get("source_name")))
            target = fact.get("target", fact.get("target_id", fact.get("target_name")))
            rtype = fact.get("type", fact.get("relation", "connected_to"))
            if source and target:
                return DatalogFact(
                    to_pred(rtype), (to_const(source), to_const(target))
                )
            return None

        # Entity (id/name + optional type). A node without an explicit type
        # still carries information — it exists — so it defaults to "entity".
        if any(k in fact for k in ("id", "name")):
            name = fact.get("id", fact.get("name"))
            if name:
                return DatalogFact(
                    to_pred(fact.get("type", "entity")), (to_const(name),)
                )
        return None

    def _parse_rule_string(self, s: str) -> DatalogRule:
        s = s.strip()
        if ":-" not in s:
            raise ValueError(f"Invalid rule syntax (missing ':-'): {s}")
        head_str, body_str = s.split(":-", 1)
        head_str = head_str.strip()
        body_str = body_str.strip().rstrip(".")

        head_match = re.match(r"^([a-zA-Z0-9_]+)\s*\(\s*([^)]+)\s*\)$", head_str)
        if not head_match:
            raise ValueError(f"Invalid rule head syntax: {head_str}")

        atoms = re.findall(r"([a-zA-Z0-9_]+)\s*\(\s*([^)]+)\s*\)", body_str)
        if not atoms:
            raise ValueError(f"No valid body atoms found in rule: {s}")

        body = [
            BodyAtom(pred, tuple(a.strip() for a in args_str.split(",")))
            for pred, args_str in atoms
        ]
        return DatalogRule(
            head_match.group(1),
            tuple(a.strip() for a in head_match.group(2).split(",")),
            body,
        )

    # ---- unification ----

    @staticmethod
    def _is_variable(term: str) -> bool:
        """Variables strictly start with an uppercase letter."""
        return bool(term and term[0].isupper())

    def _unify(
        self,
        pattern_args: Tuple[str, ...],
        fact_args: Tuple[str, ...],
        bindings: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        if len(pattern_args) != len(fact_args):
            return None
        additions: Dict[str, str] = {}
        for p_arg, f_arg in zip(pattern_args, fact_args):
            if self._is_variable(p_arg):
                if p_arg in bindings:
                    if bindings[p_arg] != f_arg:
                        return None
                elif p_arg in additions:
                    if additions[p_arg] != f_arg:
                        return None
                else:
                    additions[p_arg] = f_arg
            elif p_arg != f_arg:
                return None
        return {**bindings, **additions} if additions else bindings

    def _instantiate(
        self, args: Tuple[str, ...], bindings: Dict[str, str]
    ) -> Optional[Tuple[str, ...]]:
        result = []
        for arg in args:
            if self._is_variable(arg):
                if arg not in bindings:
                    return None
                result.append(bindings[arg])
            else:
                result.append(arg)
        return tuple(result)

    def _instantiate_fact(
        self, predicate: str, args: Tuple[str, ...], bindings: Dict[str, str]
    ) -> Optional[DatalogFact]:
        ground = self._instantiate(args, bindings)
        return None if ground is None else DatalogFact(predicate, ground)

    # ---- fixpoint evaluation ----

    def derive_all(self) -> List[str]:
        """Run semi-naive evaluation to fixpoint; return all facts as strings."""
        if self._derived:
            return [str(f) for f in self._all_facts]

        self._delta_new = self._all_facts.copy()
        iteration = 0
        derived_count = 0

        while self._delta_new:
            iteration += 1
            self._delta_old = self._delta_new
            self._delta_new = set()

            delta_index: Dict[str, Set[DatalogFact]] = defaultdict(set)
            for f in self._delta_old:
                delta_index[f.predicate].add(f)

            for rule in self._rules:
                for fact in self._apply_rule(rule, delta_index):
                    if fact not in self._all_facts:
                        self._delta_new.add(fact)
                        self._all_facts.add(fact)
                        self._fact_index[fact.predicate].add(fact)
                        derived_count += 1

            log.debug(
                "Datalog iteration %s: derived %s new facts",
                iteration, len(self._delta_new),
            )

        self._derived = True
        log.info(
            "Datalog fixpoint reached in %s iterations, %s new facts derived",
            iteration, derived_count,
        )
        # Sorted for deterministic output: facts live in sets, whose
        # iteration order is not stable across runs. Auditability depends
        # on identical inputs producing byte-identical results.
        return [str(f) for f in sorted(self._all_facts, key=str)]

    def sorted_facts(self) -> List[DatalogFact]:
        """All facts in a stable, deterministic order."""
        return sorted(self._all_facts, key=str)

    def _apply_rule(
        self,
        rule: DatalogRule,
        delta_index: Optional[Dict[str, Set[DatalogFact]]] = None,
    ) -> Set[DatalogFact]:
        """Evaluate one rule (semi-naive if ``delta_index`` given, else naive)."""
        results: Set[DatalogFact] = set()
        if not rule.body:
            fact = self._instantiate_fact(rule.head_predicate, rule.head_args, {})
            if fact:
                results.add(fact)
            return results

        is_seminaive = delta_index is not None
        paths = range(len(rule.body)) if is_seminaive else [0]

        for delta_pos in paths:
            bindings_list: List[Dict[str, str]] = [{}]
            for i, atom in enumerate(rule.body):
                if is_seminaive and i == delta_pos:
                    candidates = (delta_index or {}).get(atom.predicate, set())
                else:
                    candidates = self._fact_index.get(atom.predicate, set())

                next_bindings: List[Dict[str, str]] = []
                for bindings in bindings_list:
                    for fact in candidates:
                        merged = self._unify(atom.args, fact.args, bindings)
                        if merged is not None:
                            next_bindings.append(merged)
                bindings_list = next_bindings
                if not bindings_list:
                    break

            for final in bindings_list:
                head = self._instantiate_fact(
                    rule.head_predicate, rule.head_args, final
                )
                if head:
                    results.add(head)
        return results

    # ---- query ----

    def query(self, pattern: str, bindings: dict = None) -> List[dict]:
        """Query the fact set, e.g. ``"ancestor(c_tom, ?Y)"`` -> ``[{"Y": ..}]``."""
        if self._rules and not self._derived:
            self.derive_all()

        match = re.match(
            r"^\s*([a-zA-Z0-9_]+)\s*\(\s*([^)]+)\s*\)\s*\.?\s*$", pattern.strip()
        )
        if not match:
            raise ValueError(f"Invalid query syntax: {pattern}")

        pred = match.group(1)
        raw_args = tuple(a.strip() for a in match.group(2).split(","))

        query_vars: Dict[int, Tuple[str, str]] = {}
        pattern_args: List[str] = []
        for i, arg in enumerate(raw_args):
            if arg.startswith("?"):
                var_name = arg[1:]
                if not var_name:
                    raise ValueError("Empty variable name after '?'")
                internal = var_name[0].upper() + var_name[1:]
                query_vars[i] = (var_name, internal)
                pattern_args.append(internal)
            elif self._is_variable(arg):
                query_vars[i] = (arg, arg)
                pattern_args.append(arg)
            else:
                pattern_args.append(arg)

        initial: Dict[str, str] = {}
        for k, v in (bindings or {}).items():
            initial[k[0].upper() + k[1:] if k and not k[0].isupper() else k] = v

        for i, arg in enumerate(pattern_args):
            if self._is_variable(arg) and arg in initial:
                pattern_args[i] = initial[arg]

        results: List[dict] = []
        # Iterate in sorted order so results are deterministic.
        for fact in sorted(self._fact_index.get(pred, set()), key=str):
            matched = self._unify(tuple(pattern_args), fact.args, {})
            if matched is None:
                continue
            row = {}
            for _idx, (orig_var, internal_var) in query_vars.items():
                if internal_var in matched:
                    row[orig_var] = matched[internal_var]
                elif internal_var in initial:  # pragma: no cover
                    # Defensive only: a variable substituted by a binding above
                    # is absent from ``matched`` and resolved from ``initial``.
                    row[orig_var] = initial[internal_var]
            if row and row not in results:
                results.append(row)
        return results

    # ---- graph loading ----

    def load_from_graph(self, graph: Any) -> int:
        """Load facts from a graph-like object. Returns the number added.

        Supports objects exposing ``find_edges()``/``find_nodes()``, or
        networkx-style ``edges``/``nodes``.
        """
        before = len(self._all_facts)

        if hasattr(graph, "find_edges") and hasattr(graph, "find_nodes"):
            for edge in graph.find_edges():
                self.add_fact(edge)
            for node in graph.find_nodes():
                self.add_fact(node)
        else:
            edges = getattr(graph, "edges", None)
            if edges is not None:
                iterable = edges() if callable(edges) else edges
                for edge in iterable:
                    # networkx yields (u, v) or (u, v, data) tuples; a tuple
                    # has no __dict__, so convert explicitly.
                    if isinstance(edge, dict):
                        self.add_fact(edge)
                    elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                        self.add_fact(self._edge_tuple_to_dict(edge))
            nodes = getattr(graph, "nodes", None)
            if nodes is not None:
                iterable = nodes() if callable(nodes) else nodes
                if isinstance(iterable, dict):
                    iterable = iterable.values()
                for node in iterable:
                    if isinstance(node, dict):
                        self.add_fact(node)
                    elif isinstance(node, (tuple, list)) and len(node) == 2:
                        self.add_fact(self._node_tuple_to_dict(node))

        added = len(self._all_facts) - before
        log.info("Loaded %s datalog facts from graph", added)
        return added

    @staticmethod
    def _edge_tuple_to_dict(edge: Any) -> Dict[str, Any]:
        """Convert a networkx edge tuple ``(u, v, data)`` into an edge dict."""
        src, tgt = edge[0], edge[1]
        data = edge[2] if len(edge) > 2 and isinstance(edge[2], dict) else {}
        return {
            "source": src,
            "target": tgt,
            "type": data.get("type", data.get("relation", "connected_to")),
        }

    @staticmethod
    def _node_tuple_to_dict(node: Any) -> Dict[str, Any]:
        """Convert a networkx node tuple ``(node_id, data)`` into an entity dict."""
        node_id, data = node[0], node[1]
        meta = data if isinstance(data, dict) else {}
        return {"id": node_id, "type": meta.get("type", "entity")}


# ============================================================
# Operator
# ============================================================


class DatalogReasonerOp:
    """Deterministic rule reasoning over a HugeGraph subgraph.

    Wraps :class:`DatalogReasoner` with constant normalisation so that
    arbitrary HugeGraph ids (CJK, digits, uppercase) can be used, and maps
    derived constants back to their original names.

    Reads from context:
        entities:       Optional list of entity dicts (``id``/``name`` + ``type``)
        relations:      Optional list of edge dicts (``source``/``target``/``type``)
        datalog_rules:  Optional list of Horn clause strings
        conflict_rules: Optional list of predicates treated as contradictions

    Writes to context:
        datalog_facts:     All facts after fixpoint (original names + predicate)
        datalog_derived:   Only the newly derived facts
        datalog_conflicts: Facts flagged by ``conflict_rules``
    """

    def __init__(self, reasoner: Optional[DatalogReasoner] = None):
        self._reasoner = reasoner or DatalogReasoner()
        self._const_to_orig: Dict[str, str] = {}
        # Snapshot of facts present before any rule fired, so that
        # ``derived_facts()`` can return only what the rules produced.
        self._loaded: Set[DatalogFact] = set()

    # ---- normalisation with reverse mapping ----

    def _const(self, name: Any) -> str:
        const = to_const(name)
        self._const_to_orig.setdefault(const, str(name))
        return const

    def _orig(self, const: str) -> str:
        return self._const_to_orig.get(const, const)

    # ---- loading ----

    def load_entities(self, entities: List[Dict[str, Any]]) -> int:
        """Load entity facts (unary predicates derived from ``type``)."""
        count = 0
        for ent in entities or []:
            name = ent.get("id", ent.get("name"))
            if not name:
                continue
            fact = DatalogFact(
                to_pred(ent.get("type", "entity")), (self._const(name),)
            )
            self._reasoner.add_fact(fact)
            self._loaded.add(fact)
            count += 1
        return count

    def load_relations(self, relations: List[Dict[str, Any]]) -> int:
        """Load edge facts (binary predicates derived from ``type``)."""
        count = 0
        for rel in relations or []:
            source = rel.get("source", rel.get("source_id"))
            target = rel.get("target", rel.get("target_id"))
            if not source or not target:
                continue
            fact = DatalogFact(
                to_pred(rel.get("type", "connected_to")),
                (self._const(source), self._const(target)),
            )
            self._reasoner.add_fact(fact)
            self._loaded.add(fact)
            count += 1
        return count

    def add_rules(self, rules: List[str]) -> int:
        """Add Horn clauses in ``head(X, Y) :- body(X, Z), ...`` syntax."""
        for rule in rules or []:
            self._reasoner.add_rule(rule)
        return len(rules or [])

    # ---- outputs ----

    def derived_facts(self) -> List[Dict[str, Any]]:
        """Return only the facts the rules produced, mapped to original names."""
        return [
            self._to_output(f)
            for f in self._reasoner.sorted_facts()
            if f not in self._loaded
        ]

    def _to_output(self, fact: DatalogFact) -> Dict[str, Any]:
        return {
            "predicate": fact.predicate,
            "args": [self._orig(a) for a in fact.args],
            "raw": str(fact),
        }

    def check_consistency(self, conflict_predicates: List[str]) -> List[Dict[str, Any]]:
        """Flag facts whose predicate is listed as a contradiction signal.

        These are *flagged*, never silently removed — a knowledge base
        differs from a cache precisely by preserving conflicts for review.
        """
        flags = [to_pred(p) for p in conflict_predicates or []]
        if not flags:
            return []
        self._reasoner.derive_all()
        return [
            self._to_output(f)
            for f in self._reasoner.sorted_facts()
            if f.predicate in flags
        ]

    # ---- operator protocol ----

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deterministic reasoning over the context subgraph."""
        entities = context.get("entities", []) or []
        relations = context.get("relations", []) or []
        rules = context.get("datalog_rules", []) or []
        conflicts = context.get("conflict_rules", []) or []

        self._const_to_orig.clear()
        self._loaded.clear()
        loaded = self.load_entities(entities) + self.load_relations(relations)
        self.add_rules(rules)

        self._reasoner.derive_all()

        context["datalog_facts"] = [
            self._to_output(f) for f in self._reasoner.sorted_facts()
        ]
        context["datalog_derived"] = self.derived_facts()
        context["datalog_conflicts"] = self.check_consistency(conflicts)
        context["datalog_stats"] = {
            "loaded": loaded,
            "rules": len(rules),
            "total_facts": len(self._reasoner.facts),
            "derived": len(context["datalog_derived"]),
            "conflicts": len(context["datalog_conflicts"]),
        }
        log.info("Datalog reasoning done: %s", context["datalog_stats"])
        return context
