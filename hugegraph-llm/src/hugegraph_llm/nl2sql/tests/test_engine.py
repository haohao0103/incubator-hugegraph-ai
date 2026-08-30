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
Tests for the pluggable graph-engine layer.

The interesting part is ``FakeVermeerMaster``: instead of mocking
``VermeerEngine``'s own methods — which would only assert that the code calls
itself — it stands in for the *cluster*. It parses the load files the engine
actually wrote, honours the Vermeer behaviours that are easy to get wrong
(hard-coded ``,`` inside the property field, ``output.need_query``, sparse
community labels, ``float32`` max as "unreachable"), and answers with real
networkx results. So a regression in the load format or the param contract
fails here, not six months later against a real cluster.
"""

import json
import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pytest

from hugegraph_llm.nl2sql.engine import (
    EngineCapabilities,
    GraphEngine,
    LocalEngine,
    affinity_of,
)
from hugegraph_llm.nl2sql.engine.vermeer import (
    AFFINITY_PROP,
    FIELD_SEP,
    JOIN_COST_PROP,
    PROP_SEP,
    VermeerEngine,
    _compact_labels,
    _to_float,
)
from hugegraph_llm.nl2sql.engine.vermeer_client import (
    VermeerClient,
    VermeerError,
)
from hugegraph_llm.nl2sql.join_path.path_finder import JoinPathFinder
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker
from hugegraph_llm.nl2sql.pipeline import NL2SQLPipeline
from hugegraph_llm.nl2sql.schema_graph.backend import (
    shortest_join_path,
    steiner_join_tree,
    to_join_graph,
)
from hugegraph_llm.nl2sql.schema_graph.builder import SchemaGraphBuilder
from hugegraph_llm.nl2sql.schema_graph.model import SchemaGraph, Table
from hugegraph_llm.nl2sql.tests.test_nl2sql import _build_warehouse

_FLOAT32_MAX = 3.4028235e38


@contextmanager
def captured_warnings():
    """Collect records from the project logger.

    ``pytest``'s ``caplog`` hooks the root logger, and ``hugegraph_llm``'s
    ``llm`` logger does not propagate to it, so warnings would silently go
    unasserted. Attaching a sink to the real logger keeps the assertions honest
    without reconfiguring production logging.
    """
    records: List[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    sink = _Sink(level=logging.WARNING)
    logger = logging.getLogger("llm")
    logger.addHandler(sink)
    try:
        yield records
    finally:
        logger.removeHandler(sink)


@pytest.fixture
def warehouse():
    return _build_warehouse()


# ============================================================
# Fake Vermeer cluster
# ============================================================


def _parse_load(params: Dict[str, str], host: str) -> nx.DiGraph:
    """Parse the load files exactly the way a Vermeer worker would."""
    vertex_files = json.loads(params["load.vertex_files"])
    edge_files = json.loads(params["load.edge_files"])
    delimiter = params.get("load.delimiter", ",")

    schema: Dict[str, int] = {}
    if params.get("load.use_property") == "1":
        for name, spec in json.loads(params["load.edge_property"]).items():
            value_type, index = spec.split(",")
            assert value_type == "0", "only Float32 is exercised here"
            schema[name] = int(index)

    g = nx.DiGraph()
    with open(vertex_files[host], encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                g.add_node(line.split(delimiter)[0])
    with open(edge_files[host], encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(delimiter)
            props = {}
            if schema and len(fields) > 2:
                # Vermeer splits the property field on a hard-coded ",",
                # regardless of load.delimiter.
                pieces = fields[2].split(",")
                for name, index in schema.items():
                    if index < len(pieces):
                        props[name] = float(pieces[index])
            g.add_edge(fields[0], fields[1], **props)
    return g


def _undirected(g: nx.DiGraph, weight_prop: Optional[str] = None) -> nx.Graph:
    u = nx.Graph()
    u.add_nodes_from(g.nodes())
    for a, b, data in g.edges(data=True):
        w = 1.0 if weight_prop is None else float(data.get(weight_prop, 1.0))
        u.add_edge(a, b, weight=w)
    return u


def _run_algorithm(g: nx.DiGraph, params: Dict[str, str]) -> Dict[str, str]:
    algorithm = params["compute.algorithm"]

    if algorithm == "ppr":
        source = params["ppr.source"]
        if source not in g:
            return {}
        # Vermeer's ppr is unweighted and redistributes dangling mass to the
        # source; a source-only personalisation vector makes nx do the same.
        base = _undirected(g)
        personalization = {n: (1.0 if n == source else 0.0) for n in base}
        scores = nx.pagerank(
            base,
            alpha=float(params.get("ppr.damping", 0.85)),
            personalization=personalization,
        )
        return {node: repr(score) for node, score in scores.items()}

    if algorithm == "weighted_sssp":
        source = params["sssp.source"]
        prop = params.get("sssp.edge_weight_property", "")
        walk = nx.DiGraph()
        walk.add_nodes_from(g.nodes())
        for a, b, data in g.edges(data=True):
            w = float(data.get(prop, 0.0))
            if w <= 0.0:  # weighted_sssp skips non-positive weights
                continue
            walk.add_edge(a, b, weight=w)
        lengths = (
            nx.single_source_dijkstra_path_length(walk, source, weight="weight")
            if source in walk
            else {}
        )
        # Unreachable vertices keep their float32-max initial value.
        return {
            node: repr(float(lengths.get(node, _FLOAT32_MAX)))
            for node in g.nodes()
        }

    if algorithm in ("louvain_weighted", "leiden"):
        prefix = "louvain" if algorithm == "louvain_weighted" else "leiden"
        weight_prop = params.get(f"{prefix}.edge_weight_property")
        base = _undirected(g, weight_prop)
        groups = nx.community.louvain_communities(
            base,
            weight="weight",
            resolution=float(params.get(f"{prefix}.resolution", 1.0)),
            seed=42,
        )
        # Vermeer labels a community by an internal vertex index, so the ids
        # are sparse and never a tidy 0..k-1 range.
        out: Dict[str, str] = {}
        for i, group in enumerate(groups):
            for node in group:
                out[node] = str(7919 * (i + 3))
        return out

    raise AssertionError(f"unexpected algorithm {algorithm!r}")


class FakeVermeerMaster:
    """Stands in for a Vermeer master, faithfully enough to catch regressions."""

    def __init__(self, hosts=("10.0.0.1",)):
        self.hosts = list(hosts)
        self.load_params: Dict[str, Dict[str, str]] = {}
        self.compute_calls: List[Tuple[str, Dict[str, str]]] = []
        self.created: List[str] = []
        self.deleted: List[str] = []
        self.allocs = 0
        self.closed = False
        self.graphs: Dict[str, nx.DiGraph] = {}
        self._tasks: Dict[int, Tuple[str, Dict[str, str]]] = {}
        self._next_id = 1000

    # ---- surface used by VermeerEngine ----

    def worker_hosts(self, group: Optional[str] = None) -> List[str]:
        return list(self.hosts)

    def alloc_worker_group(self) -> None:
        self.allocs += 1

    def create_graph(self, name: str) -> None:
        self.created.append(name)

    def delete_graph(self, name: str) -> None:
        self.deleted.append(name)

    def close(self) -> None:
        self.closed = True

    def run_load(self, graph: str, params: Dict[str, str]):
        assert self.allocs > 0, "worker group must be allocated before a task"
        self.load_params[graph] = dict(params)
        self.graphs[graph] = _parse_load(params, self.hosts[0])
        return {"state": "loaded"}

    def submit_task(self, task_type: str, graph: str, params: Dict[str, str]) -> int:
        assert task_type == "compute"
        self._next_id += 1
        self._tasks[self._next_id] = (graph, dict(params))
        self.compute_calls.append((graph, dict(params)))
        return self._next_id

    def wait_task(self, task_id: int, done_states, timeout=None):
        return {"state": tuple(done_states)[0]}

    def compute_values(self, task_id: int, limit: int = 100000) -> Dict[str, str]:
        graph, params = self._tasks[task_id]
        if params.get("output.need_query") != "1":
            # The master frees the result set without this; returning nothing
            # is exactly what the caller would observe.
            return {}
        return _run_algorithm(self.graphs[graph], params)


@pytest.fixture
def vermeer(warehouse, tmp_path):
    master = FakeVermeerMaster()
    engine = VermeerEngine(
        warehouse,
        client=master,
        data_dir=str(tmp_path),
        data_hosts=master.hosts,
        graph_prefix="ut",
    )
    yield engine, master


class RecordingEngine(GraphEngine):
    """Minimal engine that proves the layers really delegate."""

    def __init__(self):
        self.ppr_calls: List[Tuple[Dict[str, float], float]] = []
        self.path_calls: List[Tuple[str, str]] = []
        self.steiner_calls: List[List[str]] = []
        self.community_calls: List[Tuple[float, str]] = []
        self.closed = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(name="recording",
                                  community_algorithms=("louvain", "leiden"))

    def personalized_pagerank(self, seeds, alpha=0.85):
        self.ppr_calls.append((dict(seeds), alpha))
        return {"table:dw.orders": 0.9, "column:dw.orders.amount": 0.1}

    def shortest_join_path(self, source, target):
        self.path_calls.append((source, target))
        return [source, "table:dw.users", target]

    def steiner_join_tree(self, terminals):
        self.steiner_calls.append(list(terminals))
        return list(terminals), 4.25

    def communities(self, resolution=1.0, algorithm="louvain"):
        self.community_calls.append((resolution, algorithm))
        return {"table:dw.orders": 5, "table:dw.users": 5, "table:dw.cities": 9}

    def close(self):
        self.closed = True


# ============================================================
# base: affinity + capabilities + ABC
# ============================================================


class TestAffinity:
    def test_inverts_cost(self):
        assert affinity_of(4.0) == pytest.approx(0.25)

    def test_cheaper_join_gets_higher_affinity(self):
        assert affinity_of(1.0) > affinity_of(3.0)

    def test_zero_cost_saturates(self):
        assert affinity_of(0.0) == 1e9

    def test_negative_cost_saturates(self):
        assert affinity_of(-1.0) == 1e9

    def test_infinite_cost_is_no_affinity(self):
        assert affinity_of(float("inf")) == 0.0

    def test_nan_is_no_affinity(self):
        assert affinity_of(float("nan")) == 0.0

    def test_garbage_is_no_affinity(self):
        assert affinity_of("not a number") == 0.0
        assert affinity_of(None) == 0.0

    def test_numeric_string_accepted(self):
        assert affinity_of("2") == pytest.approx(0.5)


class TestCapabilities:
    def test_supports_declared_algorithm(self):
        caps = EngineCapabilities(name="x", community_algorithms=("louvain",))
        assert caps.supports_community("louvain")
        assert not caps.supports_community("leiden")

    def test_frozen(self):
        caps = EngineCapabilities(name="x")
        with pytest.raises(Exception):
            caps.name = "y"

    def test_defaults_are_the_local_engine_shape(self):
        caps = EngineCapabilities(name="x")
        assert caps.distributed is False
        assert caps.weighted_ppr is True
        assert caps.weighted_paths is True


class TestGraphEngineABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            GraphEngine()

    def test_name_comes_from_capabilities(self):
        assert RecordingEngine().name == "recording"

    def test_repr_names_the_engine(self):
        assert "recording" in repr(RecordingEngine())

    def test_context_manager_closes(self):
        engine = RecordingEngine()
        with engine as same:
            assert same is engine
        assert engine.closed is True

    def test_close_is_a_noop_by_default(self, warehouse):
        LocalEngine(warehouse).close()  # must not raise


# ============================================================
# LocalEngine
# ============================================================


class TestLocalEngine:
    def test_capabilities(self, warehouse):
        caps = LocalEngine(warehouse).capabilities
        assert caps.name == "local"
        assert caps.distributed is False
        assert caps.weighted_ppr is True

    def test_ppr_scores_sum_to_one(self, warehouse):
        scores = LocalEngine(warehouse).personalized_pagerank(
            {"term:营收": 1.0})
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_ppr_favours_the_seed_neighbourhood(self, warehouse):
        scores = LocalEngine(warehouse).personalized_pagerank(
            {"term:营收": 1.0})
        assert scores["column:dw.orders.amount"] > scores["table:dw.cities"]

    def test_ppr_unknown_seed_is_ignored(self, warehouse):
        assert LocalEngine(warehouse).personalized_pagerank(
            {"term:nope": 1.0}) == {}

    def test_ppr_empty_graph(self):
        assert LocalEngine(SchemaGraph()).personalized_pagerank(
            {"a": 1.0}) == {}

    def test_ppr_zero_weight_seed_is_ignored(self, warehouse):
        assert LocalEngine(warehouse).personalized_pagerank(
            {"term:营收": 0.0}) == {}

    def test_ppr_alpha_changes_spread(self, warehouse):
        engine = LocalEngine(warehouse)
        near = engine.personalized_pagerank({"term:营收": 1.0}, alpha=0.2)
        far = engine.personalized_pagerank({"term:营收": 1.0}, alpha=0.95)
        assert near["term:营收"] > far["term:营收"]

    def test_shortest_path_matches_backend(self, warehouse):
        engine = LocalEngine(warehouse)
        expected = shortest_join_path(
            to_join_graph(warehouse), "table:dw.orders", "table:dw.cities")
        assert engine.shortest_join_path(
            "table:dw.orders", "table:dw.cities") == expected

    def test_steiner_matches_backend(self, warehouse):
        engine = LocalEngine(warehouse)
        terminals = ["table:dw.orders", "table:dw.users", "table:dw.cities"]
        assert engine.steiner_join_tree(terminals) == steiner_join_tree(
            to_join_graph(warehouse), terminals)

    def test_projections_exposed(self, warehouse):
        engine = LocalEngine(warehouse)
        assert engine.join_graph.number_of_nodes() == 4
        assert engine.graph.number_of_nodes() > 4

    def test_communities_label_every_table(self, warehouse):
        labels = LocalEngine(warehouse).communities()
        assert set(labels) == set(to_join_graph(warehouse).nodes())

    def test_community_ids_are_dense(self, warehouse):
        labels = LocalEngine(warehouse).communities()
        assert sorted(set(labels.values())) == list(range(len(set(labels.values()))))

    def test_communities_group_connected_tables(self, warehouse):
        labels = LocalEngine(warehouse).communities()
        assert labels["table:dw.orders"] == labels["table:dw.users"]

    def test_isolated_table_gets_its_own_community(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("lonely", "dw")])
        b.add_lineage("dw.a", "dw.b")
        labels = LocalEngine(b.build()).communities()
        assert labels["table:dw.lonely"] != labels["table:dw.a"]

    def test_communities_empty_graph(self):
        assert LocalEngine(SchemaGraph()).communities() == {}

    def test_communities_rejects_unknown_algorithm(self, warehouse):
        with pytest.raises(ValueError, match="louvain only"):
            LocalEngine(warehouse).communities(algorithm="leiden")


# ============================================================
# Steiner reachability fix
# ============================================================


class TestSteinerReachability:
    def _chain(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("c", "dw")])
        b.add_lineage("dw.a", "dw.b")
        b.add_lineage("dw.b", "dw.c")
        return b.build()

    def test_routes_through_an_intermediate_table(self):
        """a and c share no edge; b is a legitimate Steiner point."""
        nodes, cost = LocalEngine(self._chain()).steiner_join_tree(
            ["table:dw.a", "table:dw.c"])
        assert set(nodes) == {"table:dw.a", "table:dw.b", "table:dw.c"}
        assert cost > 0

    def test_still_rejects_a_genuinely_isolated_terminal(self):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("c", "dw")])
        b.add_lineage("dw.a", "dw.b")
        assert LocalEngine(b.build()).steiner_join_tree(
            ["table:dw.a", "table:dw.b", "table:dw.c"]) == ([], 0.0)

    def test_join_finder_connects_via_intermediate(self):
        path = JoinPathFinder(self._chain()).connect(["dw.a", "dw.c"])
        assert path is not None
        assert "dw.b" in path.tables


# ============================================================
# Engine injection into the NL2SQL layers
# ============================================================


class TestEngineInjection:
    def test_linker_defaults_to_local(self, warehouse):
        assert SchemaLinker(warehouse).engine.name == "local"

    def test_linker_uses_injected_engine(self, warehouse):
        engine = RecordingEngine()
        items = SchemaLinker(warehouse, engine=engine).link("营收", top_k=5)
        assert engine.ppr_calls, "linker did not delegate the PPR"
        assert [i.name for i in items] == ["orders", "amount"]

    def test_linker_passes_alpha_through(self, warehouse):
        engine = RecordingEngine()
        SchemaLinker(warehouse, alpha=0.5, engine=engine).link("营收")
        assert engine.ppr_calls[0][1] == 0.5

    def test_linker_seeds_reach_the_engine(self, warehouse):
        engine = RecordingEngine()
        SchemaLinker(warehouse, engine=engine).link("营收")
        assert "term:营收" in engine.ppr_calls[0][0]

    def test_path_finder_defaults_to_local(self, warehouse):
        assert JoinPathFinder(warehouse).engine.name == "local"

    def test_path_finder_uses_injected_engine(self, warehouse):
        engine = RecordingEngine()
        path = JoinPathFinder(warehouse, engine=engine).shortest_path(
            "dw.orders", "dw.cities")
        assert engine.path_calls == [("table:dw.orders", "table:dw.cities")]
        assert path.tables == ["dw.orders", "dw.users", "dw.cities"]

    def test_path_finder_keeps_local_join_keys(self, warehouse):
        """Key resolution reads the schema, so it works with any engine."""
        engine = RecordingEngine()
        path = JoinPathFinder(warehouse, engine=engine).shortest_path(
            "dw.orders", "dw.cities")
        assert path.steps[0].proven is True
        assert "user_id" in path.steps[0].to_sql()

    def test_connect_uses_injected_steiner(self, warehouse):
        engine = RecordingEngine()
        path = JoinPathFinder(warehouse, engine=engine).connect(
            ["dw.orders", "dw.users"])
        assert engine.steiner_calls == [["table:dw.orders", "table:dw.users"]]
        assert path.total_cost == 4.25

    def test_domains_group_by_label(self, warehouse):
        engine = RecordingEngine()
        domains = JoinPathFinder(warehouse, engine=engine).domains()
        assert domains == {"5": ["dw.orders", "dw.users"], "9": ["dw.cities"]}

    def test_domains_forward_arguments(self, warehouse):
        engine = RecordingEngine()
        JoinPathFinder(warehouse, engine=engine).domains(
            resolution=2.0, algorithm="leiden")
        assert engine.community_calls == [(2.0, "leiden")]

    def test_pipeline_defaults_to_local(self, warehouse):
        assert NL2SQLPipeline(warehouse).engine.name == "local"

    def test_pipeline_shares_one_engine(self, warehouse):
        engine = RecordingEngine()
        pipe = NL2SQLPipeline(warehouse, engine=engine)
        assert pipe._linker.engine is engine
        assert pipe._join_finder.engine is engine

    def test_pipeline_exposes_capabilities(self, warehouse):
        assert NL2SQLPipeline(warehouse).capabilities.name == "local"

    def test_pipeline_communities(self, warehouse):
        domains = NL2SQLPipeline(warehouse).communities()
        assert sum(len(v) for v in domains.values()) == 4

    def test_pipeline_domain_of(self, warehouse):
        pipe = NL2SQLPipeline(warehouse)
        assert pipe.domain_of("dw.orders") == pipe.domain_of("dw.users")

    def test_pipeline_domain_of_unknown_table(self, warehouse):
        assert NL2SQLPipeline(warehouse).domain_of("dw.nope") is None

    def test_pipeline_close_closes_the_engine(self, warehouse):
        engine = RecordingEngine()
        NL2SQLPipeline(warehouse, engine=engine).close()
        assert engine.closed is True

    def test_pipeline_context_manager(self, warehouse):
        engine = RecordingEngine()
        with NL2SQLPipeline(warehouse, engine=engine) as pipe:
            assert pipe.engine is engine
        assert engine.closed is True

    def test_pipeline_schema_context_uses_the_engine(self, warehouse):
        engine = RecordingEngine()
        pipe = NL2SQLPipeline(warehouse, engine=engine)
        assert "orders" in pipe.schema_context("营收")


# ============================================================
# VermeerEngine: load-file materialisation
# ============================================================


class TestVermeerMaterialisation:
    def test_graph_names_use_the_prefix(self, vermeer):
        engine, _ = vermeer
        assert engine.schema_graph_name == "ut_schema"
        assert engine.join_graph_name == "ut_join"

    def test_capabilities_declare_unweighted_ppr(self, vermeer):
        engine, _ = vermeer
        caps = engine.capabilities
        assert caps.name == "vermeer"
        assert caps.distributed is True
        # Vermeer's ppr operator takes no edge-weight parameter.
        assert caps.weighted_ppr is False
        assert caps.supports_community("leiden")

    def test_schema_edges_written_in_both_directions(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        lines = _read_lines(os.path.join(engine.data_dir, "schema_edges.csv"))
        pairs = {tuple(line.split(FIELD_SEP)) for line in lines}
        assert ("term:营收", "column:dw.orders.amount") in pairs
        assert ("column:dw.orders.amount", "term:营收") in pairs

    def test_schema_edges_deduplicated(self, vermeer):
        engine, _ = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        lines = _read_lines(os.path.join(engine.data_dir, "schema_edges.csv"))
        assert len(lines) == len(set(lines))

    def test_schema_edge_count_matches_networkx(self, vermeer, warehouse):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        loaded = master.graphs["ut_schema"]
        local = LocalEngine(warehouse).graph
        # Both directions of every undirected edge, self-loops excluded.
        expected = 2 * sum(1 for u, v in local.edges() if u != v)
        assert loaded.number_of_edges() == expected

    def test_every_schema_node_is_loaded(self, vermeer, warehouse):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        assert set(master.graphs["ut_schema"].nodes()) == set(warehouse.nodes)

    def test_join_edges_carry_cost_and_affinity(self, vermeer):
        engine, master = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.cities")
        loaded = master.graphs["ut_join"]
        data = loaded["table:dw.orders"]["table:dw.users"]
        assert data[JOIN_COST_PROP] > 0
        assert data[AFFINITY_PROP] == pytest.approx(
            affinity_of(data[JOIN_COST_PROP]))

    def test_join_costs_survive_the_round_trip(self, vermeer, warehouse):
        engine, master = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.cities")
        loaded = master.graphs["ut_join"]
        for u, v, data in to_join_graph(warehouse).edges(data=True):
            assert loaded[u][v][JOIN_COST_PROP] == pytest.approx(
                data["weight"], rel=1e-9)

    def test_field_separator_is_not_a_comma(self):
        """The property field is split on a hard-coded "," inside Vermeer."""
        assert FIELD_SEP != PROP_SEP
        assert PROP_SEP == ","

    def test_property_field_is_comma_separated(self, vermeer):
        engine, _ = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.users")
        lines = _read_lines(os.path.join(engine.data_dir, "join_edges.csv"))
        fields = lines[0].split(FIELD_SEP)
        assert len(fields) == 3
        assert len(fields[2].split(PROP_SEP)) == 2

    def test_load_params_declare_the_property_schema(self, vermeer):
        engine, master = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.users")
        params = master.load_params["ut_join"]
        assert params["load.use_property"] == "1"
        schema = json.loads(params["load.edge_property"])
        assert schema == {JOIN_COST_PROP: "0,0", AFFINITY_PROP: "0,1"}

    def test_schema_graph_declares_no_property(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        assert "load.use_property" not in master.load_params["ut_schema"]

    def test_load_files_are_ip_keyed_maps(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        files = json.loads(master.load_params["ut_schema"]["load.vertex_files"])
        assert list(files) == master.hosts
        assert files[master.hosts[0]].endswith("schema_vertices.csv")

    def test_load_requests_out_edges_and_degree(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        params = master.load_params["ut_schema"]
        assert params["load.use_outedge"] == "1"
        assert params["load.use_out_degree"] == "1"
        assert params["load.delimiter"] == FIELD_SEP

    def test_worker_group_allocated_once(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        engine.shortest_join_path("table:dw.orders", "table:dw.users")
        assert master.allocs == 1

    def test_graph_loaded_once_per_view(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        engine.personalized_pagerank({"term:收入": 1.0})
        assert master.created.count("ut_schema") == 1

    def test_hosts_auto_detected_when_absent(self, warehouse, tmp_path):
        master = FakeVermeerMaster(hosts=("192.168.1.9",))
        engine = VermeerEngine(warehouse, client=master,
                               data_dir=str(tmp_path), graph_prefix="auto")
        engine.personalized_pagerank({"term:营收": 1.0})
        files = json.loads(master.load_params["auto_schema"]["load.vertex_files"])
        assert list(files) == ["192.168.1.9"]

    def test_no_workers_is_an_error(self, warehouse, tmp_path):
        master = FakeVermeerMaster(hosts=())
        engine = VermeerEngine(warehouse, client=master,
                               data_dir=str(tmp_path))
        with pytest.raises(VermeerError, match="no workers"):
            engine.personalized_pagerank({"term:营收": 1.0})

    def test_tab_in_node_id_is_rejected(self, tmp_path):
        b = SchemaGraphBuilder()
        b.add_tables([Table("we\tird", "dw"), Table("ok", "dw")])
        b.add_lineage("dw.we\tird", "dw.ok")
        engine = VermeerEngine(b.build(), client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        with pytest.raises(ValueError, match="tab or newline"):
            engine.shortest_join_path("table:dw.we\tird", "table:dw.ok")

    def test_compute_always_requests_the_result_set(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        assert all(params["output.need_query"] == "1"
                   for _, params in master.compute_calls)


# ============================================================
# VermeerEngine: algorithms
# ============================================================


class TestVermeerPPR:
    def test_single_seed_scores_sum_to_one(self, vermeer):
        engine, _ = vermeer
        scores = engine.personalized_pagerank({"term:营收": 1.0})
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_single_seed_uses_one_task(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        assert len(master.compute_calls) == 1

    def test_multi_seed_runs_one_task_per_seed(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank(
            {"term:营收": 1.0, "table:dw.cities": 1.0})
        algorithms = [p["compute.algorithm"] for _, p in master.compute_calls]
        assert algorithms == ["ppr", "ppr"]

    def test_ppr_results_are_cached_per_seed(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        engine.personalized_pagerank({"term:营收": 1.0})
        assert len(master.compute_calls) == 1

    def test_alpha_is_part_of_the_cache_key(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0}, alpha=0.85)
        engine.personalized_pagerank({"term:营收": 1.0}, alpha=0.5)
        assert len(master.compute_calls) == 2
        assert master.compute_calls[1][1]["ppr.damping"] == "0.5"

    def test_multi_seed_is_the_weighted_sum_of_single_seeds(self, vermeer):
        """Vermeer's teleport term is affine, so the sum must hold."""
        engine, _ = vermeer
        a = engine.personalized_pagerank({"term:营收": 1.0})
        b = engine.personalized_pagerank({"table:dw.cities": 1.0})
        both = engine.personalized_pagerank(
            {"term:营收": 3.0, "table:dw.cities": 1.0})
        for node in both:
            expected = 0.75 * a.get(node, 0.0) + 0.25 * b.get(node, 0.0)
            assert both[node] == pytest.approx(expected, abs=2e-3)

    def test_seed_weights_shift_the_ranking(self, vermeer):
        engine, _ = vermeer
        revenue_heavy = engine.personalized_pagerank(
            {"term:营收": 9.0, "table:dw.cities": 1.0})
        city_heavy = engine.personalized_pagerank(
            {"term:营收": 1.0, "table:dw.cities": 9.0})
        assert (revenue_heavy["table:dw.cities"]
                < city_heavy["table:dw.cities"])

    def test_unknown_seed_short_circuits(self, vermeer):
        engine, master = vermeer
        assert engine.personalized_pagerank({"term:nope": 1.0}) == {}
        assert master.compute_calls == []

    def test_non_positive_seed_weight_ignored(self, vermeer):
        engine, _ = vermeer
        assert engine.personalized_pagerank({"term:营收": 0.0}) == {}

    def test_warehouse_has_no_dangling_nodes(self, vermeer):
        """The multi-seed sum is exact here — nothing is isolated."""
        engine, _ = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        assert engine._isolated_nodes == 0

    def test_isolated_node_triggers_the_approximation_warning(self, tmp_path):
        engine = VermeerEngine(_with_isolated_table(),
                               client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        with captured_warnings() as records:
            engine.personalized_pagerank(
                {"table:dw.a": 1.0, "table:dw.b": 1.0})
        assert engine._isolated_nodes == 1
        assert any("approximation" in r.getMessage() for r in records)

    def test_approximation_warning_fires_once(self, tmp_path):
        engine = VermeerEngine(_with_isolated_table(),
                               client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        seeds = {"table:dw.a": 1.0, "table:dw.b": 1.0}
        with captured_warnings() as records:
            engine.personalized_pagerank(seeds)
            engine.personalized_pagerank(seeds, alpha=0.5)
        hits = [r for r in records if "approximation" in r.getMessage()]
        assert len(hits) == 1

    def test_single_seed_does_not_warn(self, tmp_path):
        engine = VermeerEngine(_with_isolated_table(),
                               client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        with captured_warnings() as records:
            engine.personalized_pagerank({"table:dw.a": 1.0})
        assert not any("approximation" in r.getMessage() for r in records)

    def test_top_ranked_node_agrees_with_the_local_engine(self, vermeer,
                                                          warehouse):
        """Different weighting, same winner: the seed's own neighbourhood."""
        engine, _ = vermeer
        remote = engine.personalized_pagerank({"term:营收": 1.0})
        local = LocalEngine(warehouse).personalized_pagerank({"term:营收": 1.0})
        best = lambda s: max(s, key=s.get)  # noqa: E731
        assert best(remote) == best(local) == "term:营收"

    def test_missing_result_set_yields_nothing(self, warehouse, tmp_path):
        """A regression that drops output.need_query must not go unnoticed."""
        master = FakeVermeerMaster()
        engine = VermeerEngine(warehouse, client=master,
                               data_dir=str(tmp_path))
        original = master.submit_task

        def strip(task_type, graph, params):
            params = dict(params)
            params.pop("output.need_query", None)
            return original(task_type, graph, params)

        master.submit_task = strip
        assert engine.personalized_pagerank({"term:营收": 1.0}) == {}


class TestVermeerPaths:
    def test_shortest_path_matches_the_local_engine(self, vermeer, warehouse):
        engine, _ = vermeer
        assert engine.shortest_join_path(
            "table:dw.orders", "table:dw.cities"
        ) == LocalEngine(warehouse).shortest_join_path(
            "table:dw.orders", "table:dw.cities")

    def test_path_starts_and_ends_correctly(self, vermeer):
        engine, _ = vermeer
        path = engine.shortest_join_path("table:dw.cities", "table:dw.orders")
        assert path[0] == "table:dw.cities"
        assert path[-1] == "table:dw.orders"

    def test_same_node_needs_no_task(self, vermeer):
        engine, master = vermeer
        assert engine.shortest_join_path(
            "table:dw.orders", "table:dw.orders") == ["table:dw.orders"]
        assert master.compute_calls == []

    def test_same_unknown_node_is_none(self, vermeer):
        engine, _ = vermeer
        assert engine.shortest_join_path("table:nope", "table:nope") is None

    def test_unknown_endpoint_is_none(self, vermeer):
        engine, master = vermeer
        assert engine.shortest_join_path(
            "table:dw.orders", "table:nope") is None
        assert master.compute_calls == []

    def test_sssp_asks_for_the_cost_property(self, vermeer):
        engine, master = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.cities")
        params = master.compute_calls[0][1]
        assert params["compute.algorithm"] == "weighted_sssp"
        assert params["sssp.edge_weight_property"] == JOIN_COST_PROP
        assert params["sssp.source"] == "table:dw.orders"

    def test_distances_are_cached_per_source(self, vermeer):
        engine, master = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.cities")
        engine.shortest_join_path("table:dw.orders", "table:dw.users")
        assert len(master.compute_calls) == 1

    def test_unreachable_target_is_none(self, tmp_path):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("c", "dw")])
        b.add_lineage("dw.a", "dw.b")
        master = FakeVermeerMaster()
        engine = VermeerEngine(b.build(), client=master,
                               data_dir=str(tmp_path))
        assert engine.shortest_join_path("table:dw.a", "table:dw.c") is None

    def test_float32_max_is_treated_as_unreachable(self, vermeer):
        engine, _ = vermeer
        engine.shortest_join_path("table:dw.orders", "table:dw.cities")
        distances = engine._distances("table:dw.orders")
        assert all(d < 3.0e38 for d in distances.values())

    def test_falls_back_when_the_walk_cannot_be_rebuilt(self, vermeer):
        """A distance vector that no local edge explains must not be trusted."""
        engine, _ = vermeer
        engine._dist_cache["table:dw.orders"] = {
            "table:dw.orders": 0.0,
            "table:dw.cities": 0.5,  # no edge combination produces 0.5
        }
        path = engine.shortest_join_path("table:dw.orders", "table:dw.cities")
        assert path[0] == "table:dw.orders" and path[-1] == "table:dw.cities"

    def test_steiner_matches_the_local_engine(self, vermeer, warehouse):
        engine, _ = vermeer
        terminals = ["table:dw.orders", "table:dw.users", "table:dw.cities"]
        remote_nodes, remote_cost = engine.steiner_join_tree(terminals)
        local_nodes, local_cost = LocalEngine(warehouse).steiner_join_tree(
            terminals)
        assert set(remote_nodes) == set(local_nodes)
        assert remote_cost == pytest.approx(local_cost, rel=1e-4)

    def test_steiner_single_terminal(self, vermeer):
        engine, master = vermeer
        assert engine.steiner_join_tree(["table:dw.orders"]) == (
            ["table:dw.orders"], 0.0)
        assert master.compute_calls == []

    def test_steiner_no_known_terminal(self, vermeer):
        engine, _ = vermeer
        assert engine.steiner_join_tree(["table:nope"]) == ([], 0.0)

    def test_steiner_deduplicates_terminals(self, vermeer):
        engine, master = vermeer
        nodes, _ = engine.steiner_join_tree(
            ["table:dw.orders", "table:dw.orders", "table:dw.users"])
        assert nodes.count("table:dw.orders") == 1

    def test_steiner_disconnected_terminal(self, tmp_path):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("c", "dw")])
        b.add_lineage("dw.a", "dw.b")
        engine = VermeerEngine(b.build(), client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        assert engine.steiner_join_tree(
            ["table:dw.a", "table:dw.c"]) == ([], 0.0)

    def test_steiner_routes_through_an_intermediate(self, tmp_path):
        b = SchemaGraphBuilder()
        b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("c", "dw")])
        b.add_lineage("dw.a", "dw.b")
        b.add_lineage("dw.b", "dw.c")
        engine = VermeerEngine(b.build(), client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        nodes, cost = engine.steiner_join_tree(["table:dw.a", "table:dw.c"])
        assert set(nodes) == {"table:dw.a", "table:dw.b", "table:dw.c"}
        assert cost > 0


class TestVermeerCommunities:
    def test_louvain_alias_maps_to_the_weighted_operator(self, vermeer):
        engine, master = vermeer
        engine.communities()
        graph, params = master.compute_calls[0]
        assert graph == "ut_join"
        assert params["compute.algorithm"] == "louvain_weighted"

    def test_louvain_uses_affinity_not_cost(self, vermeer):
        engine, master = vermeer
        engine.communities()
        params = master.compute_calls[0][1]
        assert params["louvain.edge_weight_property"] == AFFINITY_PROP

    def test_resolution_is_forwarded(self, vermeer):
        engine, master = vermeer
        engine.communities(resolution=2.5)
        assert master.compute_calls[0][1]["louvain.resolution"] == "2.5"

    def test_leiden_runs_unweighted(self, vermeer):
        engine, master = vermeer
        engine.communities(algorithm="leiden")
        params = master.compute_calls[0][1]
        assert params["compute.algorithm"] == "leiden"
        assert "leiden.edge_weight_property" not in params

    def test_sparse_vermeer_labels_are_compacted(self, vermeer):
        engine, _ = vermeer
        labels = engine.communities()
        assert sorted(set(labels.values())) == list(
            range(len(set(labels.values()))))

    def test_connected_tables_share_a_community(self, vermeer):
        engine, _ = vermeer
        labels = engine.communities()
        assert labels["table:dw.orders"] == labels["table:dw.users"]

    def test_partition_matches_the_local_engine(self, vermeer, warehouse):
        engine, _ = vermeer
        remote = engine.communities()
        local = LocalEngine(warehouse).communities()
        assert _as_partition(remote) == _as_partition(local)

    def test_empty_join_graph_needs_no_task(self, tmp_path):
        master = FakeVermeerMaster()
        engine = VermeerEngine(SchemaGraph(), client=master,
                               data_dir=str(tmp_path))
        assert engine.communities() == {}
        assert master.compute_calls == []

    def test_unknown_algorithm_rejected(self, vermeer):
        engine, _ = vermeer
        with pytest.raises(ValueError, match="vermeer engine supports"):
            engine.communities(algorithm="label_propagation")


class TestVermeerLifecycle:
    def test_close_drops_loaded_graphs(self, vermeer):
        engine, master = vermeer
        engine.personalized_pagerank({"term:营收": 1.0})
        engine.communities()
        engine.close()
        assert set(master.deleted) == {"ut_schema", "ut_join"}
        assert master.closed is True

    def test_keep_graphs_leaves_them_alone(self, warehouse, tmp_path):
        master = FakeVermeerMaster()
        engine = VermeerEngine(warehouse, client=master,
                               data_dir=str(tmp_path), keep_graphs=True)
        engine.personalized_pagerank({"term:营收": 1.0})
        engine.close()
        assert master.deleted == []

    def test_owned_temp_dir_is_removed(self, warehouse):
        master = FakeVermeerMaster()
        engine = VermeerEngine(warehouse, client=master)
        data_dir = engine.data_dir
        engine.personalized_pagerank({"term:营收": 1.0})
        assert os.path.isdir(data_dir)
        engine.close()
        assert not os.path.isdir(data_dir)

    def test_supplied_data_dir_is_kept(self, warehouse, tmp_path):
        engine = VermeerEngine(warehouse, client=FakeVermeerMaster(),
                               data_dir=str(tmp_path))
        engine.close()
        assert os.path.isdir(str(tmp_path))

    def test_context_manager(self, warehouse, tmp_path):
        master = FakeVermeerMaster()
        with VermeerEngine(warehouse, client=master,
                           data_dir=str(tmp_path)) as engine:
            engine.personalized_pagerank({"term:营收": 1.0})
        assert master.closed is True

    def test_prefix_is_unique_by_default(self, warehouse, tmp_path):
        a = VermeerEngine(warehouse, client=FakeVermeerMaster(),
                          data_dir=str(tmp_path))
        b = VermeerEngine(warehouse, client=FakeVermeerMaster(),
                          data_dir=str(tmp_path))
        assert a.schema_graph_name != b.schema_graph_name


class TestVermeerHelpers:
    def test_to_float_parses(self):
        assert _to_float("1.5") == 1.5

    def test_to_float_rejects_garbage(self):
        assert _to_float("abc") is None
        assert _to_float(None) is None

    def test_to_float_rejects_nan(self):
        assert _to_float("nan") is None

    def test_compact_labels_renumber_from_zero(self):
        raw = {"b": "7919", "a": "7919", "c": "23757"}
        assert _compact_labels(raw) == {"a": 0, "b": 0, "c": 1}

    def test_compact_labels_ordered_by_smallest_member(self):
        raw = {"z": "1", "a": "999"}
        assert _compact_labels(raw) == {"a": 0, "z": 1}

    def test_compact_labels_skip_unparsable(self):
        assert _compact_labels({"a": "x", "b": "1"}) == {"b": 0}

    def test_compact_labels_accept_float_text(self):
        assert _compact_labels({"a": "3.0"}) == {"a": 0}


# ============================================================
# VermeerClient (HTTP contract, no cluster)
# ============================================================


class FakeResponse:
    def __init__(self, status_code=200, text="{}"):
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Records requests and replays scripted responses."""

    def __init__(self, responses=None):
        self.calls: List[Tuple[str, str, Optional[dict]]] = []
        self._responses = list(responses or [])
        self.trust_env = True
        self.closed = False
        self.raise_on_request: Optional[Exception] = None

    def request(self, method, url, json=None, timeout=None, proxies=None):
        self.calls.append((method, url, json))
        self.last_proxies = proxies
        if self.raise_on_request is not None:
            raise self.raise_on_request
        if not self._responses:
            return FakeResponse()
        nxt = self._responses.pop(0)
        return nxt() if callable(nxt) else nxt

    def close(self):
        self.closed = True


def _client(responses=None, **kwargs) -> Tuple[VermeerClient, FakeSession]:
    session = FakeSession(responses)
    return VermeerClient(session=session, poll_interval=0.0, **kwargs), session


class TestVermeerClientRequests:
    def test_default_session_ignores_env_proxies(self):
        client = VermeerClient()
        assert client._session.trust_env is False
        client.close()

    def test_proxies_disabled_per_request(self):
        client, session = _client()
        client.healthcheck()
        assert session.last_proxies == {"http": None, "https": None}

    def test_base_url_is_normalised(self):
        client, _ = _client(base_url="http://host:6688/")
        assert client.base_url == "http://host:6688"

    def test_space_default_is_the_builtin(self):
        client, _ = _client()
        assert client.space == "$DEFAULT"

    def test_http_error_raises(self):
        client, _ = _client([FakeResponse(500, "boom")])
        with pytest.raises(VermeerError, match="HTTP 500"):
            client.workers()

    def test_transport_error_raises(self):
        import requests

        client, session = _client()
        session.raise_on_request = requests.ConnectionError("refused")
        with pytest.raises(VermeerError, match="failed"):
            client.workers()

    def test_non_json_raises(self):
        client, _ = _client([FakeResponse(200, "<html>nope</html>")])
        with pytest.raises(VermeerError, match="non-JSON"):
            client.workers()

    def test_errcode_raises(self):
        client, _ = _client(
            [FakeResponse(200, '{"errcode": 1, "message": "bad graph"}')])
        with pytest.raises(VermeerError, match="bad graph"):
            client.workers()

    def test_empty_body_is_ok(self):
        client, _ = _client([FakeResponse(200, "   ")])
        assert client.workers() == []

    def test_json_list_body_is_wrapped(self):
        """A bare JSON array is surfaced under "data"."""
        client, _ = _client([FakeResponse(200, "[1, 2]")])
        assert client.workers() == [1, 2]

    def test_healthcheck_true(self):
        client, _ = _client([FakeResponse(200, '{"code": 200}')])
        assert client.healthcheck() is True

    def test_healthcheck_false_on_error(self):
        client, _ = _client([FakeResponse(503, "down")])
        assert client.healthcheck() is False

    def test_close_closes_the_session(self):
        client, session = _client()
        client.close()
        assert session.closed is True


class TestVermeerClientCluster:
    _WORKERS = json.dumps({"workers": [
        {"name": "w1", "ip_addr": "10.0.0.1", "group": "default"},
        {"name": "w2", "ip_addr": "10.0.0.2", "group": "default"},
        {"name": "w3", "ip_addr": "10.0.0.1", "group": "default"},
        {"name": "w4", "ip_addr": "10.9.9.9", "group": "other"},
    ]})

    def test_worker_hosts_deduplicated_and_filtered(self):
        client, _ = _client([FakeResponse(200, self._WORKERS)])
        assert client.worker_hosts() == ["10.0.0.1", "10.0.0.2"]

    def test_worker_hosts_explicit_group(self):
        client, _ = _client([FakeResponse(200, self._WORKERS)])
        assert client.worker_hosts(group="other") == ["10.9.9.9"]

    def test_worker_hosts_tolerates_go_field_names(self):
        body = json.dumps({"workers": [{"IpAddr": "10.1.1.1"}]})
        client, _ = _client([FakeResponse(200, body)])
        assert client.worker_hosts() == ["10.1.1.1"]

    def test_worker_hosts_skips_malformed_rows(self):
        body = json.dumps({"workers": ["nonsense", {"name": "no ip"}]})
        client, _ = _client([FakeResponse(200, body)])
        assert client.worker_hosts() == []

    def test_alloc_uses_group_and_space(self):
        client, session = _client(worker_group="grp", space="$SP")
        client.alloc_worker_group()
        assert session.calls[0][1].endswith("/admin/workers/alloc/grp/$SP")

    def test_graph_names(self):
        body = json.dumps({"graphs": [{"name": "g1"}, {"nope": 1}]})
        client, _ = _client([FakeResponse(200, body)])
        assert client.graph_names() == ["g1"]

    def test_delete_graph_swallows_errors(self):
        client, _ = _client([FakeResponse(404, "gone")])
        client.delete_graph("missing")  # must not raise

    def test_create_graph_posts_the_name(self):
        client, session = _client()
        client.create_graph("g")
        assert session.calls[0][2] == {"name": "g"}


class TestVermeerClientTasks:
    def test_submit_stringifies_params(self):
        client, session = _client(
            [FakeResponse(200, '{"task": {"id": 7}}')])
        assert client.submit_task("compute", "g", {"a": 1}) == 7
        assert session.calls[0][2]["params"] == {"a": "1"}

    def test_submit_without_id_raises(self):
        client, _ = _client([FakeResponse(200, '{"task": {}}')])
        with pytest.raises(VermeerError, match="no task id"):
            client.submit_task("load", "g", {})

    def test_wait_returns_on_done_state(self):
        client, _ = _client([
            FakeResponse(200, '{"task": {"state": "loading"}}'),
            FakeResponse(200, '{"task": {"state": "loaded"}}'),
        ])
        assert client.wait_task(1, ("loaded",))["state"] == "loaded"

    def test_wait_raises_on_error_state(self):
        client, _ = _client([
            FakeResponse(200, '{"task": {"state": "error", '
                              '"error_msg": "load failed"}}'),
        ])
        with pytest.raises(VermeerError, match="load failed"):
            client.wait_task(1, ("loaded",))

    def test_wait_times_out(self):
        client, _ = _client([
            FakeResponse(200, '{"task": {"state": "waiting"}}')
            for _ in range(4)
        ])
        with pytest.raises(VermeerError, match="after timeout"):
            client.wait_task(1, ("loaded",), timeout=-1.0)

    def test_task_state_helper(self):
        client, _ = _client([FakeResponse(200, '{"task": {"state": "idle"}}')])
        assert client.task_state(3) == "idle"

    def test_run_load_waits_for_loaded(self):
        client, _ = _client([
            FakeResponse(200, '{"task": {"id": 1}}'),
            FakeResponse(200, '{"task": {"state": "on_disk"}}'),
        ])
        assert client.run_load("g", {})["state"] == "on_disk"

    def test_run_compute_waits_for_complete(self):
        client, _ = _client([
            FakeResponse(200, '{"task": {"id": 2}}'),
            FakeResponse(200, '{"task": {"state": "complete"}}'),
        ])
        assert client.run_compute("g", {})["state"] == "complete"


class TestVermeerClientValues:
    def test_pagination_accumulates_pages(self):
        page1 = json.dumps({"vertices": [{"ID": "a", "Value": "1"}],
                            "cursor": 1})
        page2 = json.dumps({"vertices": [{"ID": "b", "Value": "2"}],
                            "cursor": 2})
        eof = json.dumps({"vertices": [], "message": "EOF", "cursor": 2})
        client, session = _client([
            FakeResponse(200, page1),
            FakeResponse(200, page2),
            FakeResponse(200, eof),
        ])
        assert client.compute_values(9) == {"a": "1", "b": "2"}
        assert "cursor=1" in session.calls[1][1]

    def test_lowercase_keys_tolerated(self):
        body = json.dumps({"vertices": [{"id": "a", "value": "1"}],
                           "message": "EOF"})
        client, _ = _client([FakeResponse(200, body)])
        assert client.compute_values(9) == {"a": "1"}

    def test_null_value_becomes_empty_string(self):
        body = json.dumps({"vertices": [{"ID": "a", "Value": None}],
                           "message": "EOF"})
        client, _ = _client([FakeResponse(200, body)])
        assert client.compute_values(9) == {"a": ""}

    def test_stops_when_cursor_does_not_advance(self):
        body = json.dumps({"vertices": [{"ID": "a", "Value": "1"}],
                           "cursor": 0})
        client, session = _client([FakeResponse(200, body)] * 5)
        assert client.compute_values(9) == {"a": "1"}
        assert len(session.calls) == 1

    def test_stops_on_empty_page(self):
        client, session = _client([FakeResponse(200, '{"vertices": []}')])
        assert client.compute_values(9) == {}
        assert len(session.calls) == 1

    def test_skips_malformed_rows(self):
        body = json.dumps({"vertices": ["x", {"Value": "1"},
                                        {"ID": "a", "Value": "2"}],
                           "message": "EOF"})
        client, _ = _client([FakeResponse(200, body)])
        assert client.compute_values(9) == {"a": "2"}

    def test_limit_is_clamped(self):
        client, session = _client([FakeResponse(200, '{"vertices": []}')])
        client.compute_values(9, limit=0)
        assert "limit=1" in session.calls[0][1]


# ============================================================
# helpers
# ============================================================


def _with_isolated_table() -> SchemaGraph:
    """a -- b, plus a table nothing references (a dangling vertex)."""
    b = SchemaGraphBuilder()
    b.add_tables([Table("a", "dw"), Table("b", "dw"), Table("lonely", "dw")])
    b.add_lineage("dw.a", "dw.b")
    return b.build()


def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def _as_partition(labels: Dict[str, int]) -> List[List[str]]:
    """Comparable form: sorted groups of members, ignoring label values."""
    groups: Dict[int, List[str]] = {}
    for node, cid in labels.items():
        groups.setdefault(cid, []).append(node)
    return sorted(sorted(members) for members in groups.values())
