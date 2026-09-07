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

"""Tests for the Ossie / OSI connector (no HugeGraph server required).

Export is tested by stubbing the graph *read* (``_read_vertices``) rather
than the whole client, so the spec-assembly logic is exercised without
needing a live server or a fake Gremlin interpreter.
"""

import pytest

from hugegraph_llm.semantic_layer.connectors.ossie import (
    TRUST_VENDOR,
    OssieConnector,
    _dialect_expressions,
    _is_query_source,
    _split_source,
    _trust_extension,
)
from hugegraph_llm.semantic_layer.connectors.ossie import load_spec as _real_load_spec
from tests.semantic_layer.fakes import FakeHugeGraph

# -- helpers ---------------------------------------------------------------


def _spec(**overrides):
    base = {
        "version": "0.1.1",
        "semantic_model": [{
            "name": "acme",
            "datasets": [{
                "name": "orders",
                "source": "wh.public.orders",
                "primary_key": ["order_id"],
                "description": "All orders",
                "ai_context": {"synonyms": ["sales", "purchases"]},
                "fields": [
                    {"name": "order_id",
                     "expression": {"dialects": [
                         {"dialect": "ANSI_SQL", "expression": "order_id"}]}},
                    {"name": "amount",
                     "description": "Order total",
                     "expression": {"dialects": [
                         {"dialect": "ANSI_SQL", "expression": "amount"}]}},
                    {"name": "created_at", "dimension": {"is_time": True}},
                    {"name": "status"},
                ],
            }],
            "relationships": [{
                "name": "orders_to_customers",
                "from": "orders",
                "to": "customers",
                "from_columns": ["customer_id"],
                "to_columns": ["customer_id"],
            }],
            "metrics": [{
                "name": "total_revenue",
                "description": "Sum of order amounts",
                "expression": {"dialects": [{
                    "dialect": "ANSI_SQL",
                    "expression": "SUM(orders.amount)",
                }]},
                "ai_context": {"synonyms": ["revenue"]},
            }],
        }],
    }
    base.update(overrides)
    return base


class _Connector(OssieConnector):
    """Connector whose spec comes from an in-memory dict, backed by a fake graph."""

    def __init__(self, spec, **kwargs):
        super().__init__(client=FakeHugeGraph(), **kwargs)
        self._spec = spec

    def extract(self):
        # Version checks live in the real extract(); replay them here so
        # tests still observe the warnings a file-backed run would produce.
        declared = self._spec.get("version")
        if declared is None:
            self.warnings.append("spec has no top-level 'version' field")
        elif str(declared) != self.version:
            self.warnings.append(
                f"spec version {declared!r} != expected {self.version!r}"
            )
        return [{"version": declared,
                 "model": m} for m in self._spec["semantic_model"]]


def _vertices_by_id(connector):
    return {vid: node for vid, node in connector._vertices}


# -- source parsing --------------------------------------------------------


def test_split_source_three_parts():
    assert _split_source("wh.public.orders") == ("wh", "public", "orders")


@pytest.mark.parametrize("bad", ["orders", "public.orders", ""])
def test_split_source_rejects_non_three_parts(bad):
    with pytest.raises(ValueError):
        _split_source(bad)


def test_query_source_is_detected():
    assert _is_query_source("SELECT * FROM t")
    assert not _is_query_source("wh.public.orders")


def test_dialect_expressions():
    raw = {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(x)"}]}
    assert _dialect_expressions(raw) == [("ANSI_SQL", "SUM(x)")]
    assert _dialect_expressions(None) == []


# -- ingest mapping --------------------------------------------------------


def test_dataset_maps_to_database_schema_table_column():
    conn = _Connector(_spec())
    conn.ingest()
    verts = _vertices_by_id(conn)
    assert conn.stats["tables"] == 1
    assert conn.stats["columns"] == 4
    assert "acme:table:orders" in verts
    assert "acme:db:wh" in verts
    assert "acme:schema:wh:public" in verts
    assert "acme:column:orders:amount" in verts


def test_ids_are_namespaced_by_model():
    """Two models must be able to coexist in one graph."""
    conn = _Connector(_spec())
    conn.ingest()
    assert all(vid.startswith("acme:") for vid, _n in conn._vertices)


def test_primary_key_flag_is_set():
    conn = _Connector(_spec())
    conn.ingest()
    col = _vertices_by_id(conn)["acme:column:orders:order_id"]
    assert col["properties"]["is_primary_key"] is True


def test_synonyms_become_business_terms_and_are_tagged():
    conn = _Connector(_spec())
    conn.ingest()
    verts = _vertices_by_id(conn)
    assert conn.stats["terms"] == 3  # sales, purchases, revenue
    assert "acme:term:sales" in verts
    tagged = [e for e in conn._edges if e[0] == "TABLE_TAGGED_WITH"]
    assert any(e[2] == "acme:term:sales" for e in tagged)


def test_metric_synonyms_are_linked_not_orphaned():
    """A term with no edge answers nothing; the link is the whole point."""
    conn = _Connector(_spec())
    conn.ingest()
    tagged = [e for e in conn._edges if e[0] == "METRIC_TAGGED_WITH"]
    assert any(e[1] == "acme:metric:total_revenue" for e in tagged)
    assert any(e[2] == "acme:term:revenue" for e in tagged)


def test_synonyms_merge_on_name():
    """The same synonym on two datasets must not create two terms."""
    spec = _spec()
    spec["semantic_model"][0]["datasets"].append({
        "name": "refunds",
        "source": "wh.public.refunds",
        "ai_context": {"synonyms": ["sales"]},
        "fields": [],
    })
    conn = _Connector(spec)
    conn.ingest()
    assert conn.stats["terms"] == 3  # sales still counted once


# -- tri-state time dimension ---------------------------------------------


def test_is_time_true_is_stored():
    conn = _Connector(_spec())
    conn.ingest()
    col = _vertices_by_id(conn)["acme:column:orders:created_at"]
    assert col["properties"]["is_time_dimension"] is True


def test_absent_dimension_stays_absent():
    """'unknown' and 'false' are different; absence must not become False."""
    conn = _Connector(_spec())
    conn.ingest()
    col = _vertices_by_id(conn)["acme:column:orders:status"]
    assert "is_time_dimension" not in col["properties"]


def test_is_time_false_is_stored_as_false():
    spec = _spec()
    spec["semantic_model"][0]["datasets"][0]["fields"].append(
        {"name": "flag", "dimension": {"is_time": False}}
    )
    conn = _Connector(spec)
    conn.ingest()
    col = _vertices_by_id(conn)["acme:column:orders:flag"]
    assert col["properties"]["is_time_dimension"] is False


# -- joins -----------------------------------------------------------------


def test_relationship_becomes_join_with_ordered_columns():
    conn = _Connector(_spec())
    conn.ingest()
    join = _vertices_by_id(conn)["acme:join:orders_to_customers"]
    assert join["properties"]["from_columns"] == ["customer_id"]
    assert join["properties"]["to_columns"] == ["customer_id"]
    assert join["properties"]["proven"] is True


def test_composite_join_preserves_column_order():
    spec = _spec()
    spec["semantic_model"][0]["relationships"] = [{
        "name": "composite",
        "from": "a",
        "to": "b",
        "from_columns": ["k1", "k2"],
        "to_columns": ["k1", "k2"],
    }]
    conn = _Connector(spec)
    conn.ingest()
    join = _vertices_by_id(conn)["acme:join:composite"]
    assert join["properties"]["from_columns"] == ["k1", "k2"]


def test_join_emits_positional_references_only_for_known_columns():
    """The spec's `to: customers` table is absent here, so no edge is emitted."""
    conn = _Connector(_spec())
    conn.ingest()
    refs = [e for e in conn._edges if e[0] == "REFERENCES"]
    assert refs == []


def test_join_emits_references_when_both_columns_exist():
    spec = _spec()
    spec["semantic_model"][0]["datasets"].append({
        "name": "customers",
        "source": "wh.public.customers",
        "fields": [{"name": "customer_id"}],
    })
    spec["semantic_model"][0]["datasets"][0]["fields"].append(
        {"name": "customer_id"}
    )
    conn = _Connector(spec)
    conn.ingest()
    refs = [e for e in conn._edges if e[0] == "REFERENCES"]
    assert refs == [(
        "REFERENCES",
        "acme:column:orders:customer_id",
        "acme:column:customers:customer_id",
        {"proven": True},
    )]


# -- metrics ---------------------------------------------------------------


def test_metric_expression_and_dialect_are_stored():
    conn = _Connector(_spec())
    conn.ingest()
    metric = _vertices_by_id(conn)["acme:metric:total_revenue"]
    assert metric["properties"]["expression"] == "SUM(orders.amount)"
    assert metric["properties"]["dialect"] == "ANSI_SQL"


def test_metric_links_to_columns_named_in_expression():
    conn = _Connector(_spec())
    conn.ingest()
    edges = [e for e in conn._edges if e[0] == "HAS_EXPRESSION"]
    assert edges
    assert edges[0][1] == "acme:metric:total_revenue"
    assert edges[0][2] == "acme:column:orders:amount"


# -- trust extension -------------------------------------------------------


def test_trust_extension_is_empty_without_trust_fields():
    assert _trust_extension({"properties": {"name": "x"}}) is None


def test_trust_extension_carries_confidence_and_freshness():
    ext = _trust_extension({"properties": {
        "name": "x", "confidence": 0.9, "freshness_ts": 123,
    }})
    assert ext["vendor_name"] == TRUST_VENDOR
    assert ext["data"]["confidence"] == 0.9


# -- export ----------------------------------------------------------------


def _ingested(spec=None):
    """Run a full ingest against the fake graph and return the connector."""
    conn = _Connector(spec or _spec())
    conn.ingest()
    return conn


def test_ingest_writes_to_graph():
    conn = _ingested()
    graph = conn.client
    assert graph.count("Table") == 1
    assert graph.count("Column") == 4
    assert graph.count("Metric") == 1
    assert graph.edge_count("HAS_COLUMN") == 4


def test_round_trip_preserves_datasets_fields_and_metrics():
    """Ingest then export must recover the model, not just run without error."""
    conn = _ingested()
    model = conn.export("acme")["semantic_model"][0]

    assert len(model["datasets"]) == 1
    dataset = model["datasets"][0]
    assert dataset["name"] == "orders"
    assert dataset["source"] == "wh.public.orders"
    assert dataset["description"] == "All orders"
    assert [f["name"] for f in dataset["fields"]] == [
        "order_id", "amount", "created_at", "status",
    ]
    assert dataset["primary_key"] == ["order_id"]

    assert len(model["relationships"]) == 1
    assert model["relationships"][0]["from_columns"] == ["customer_id"]

    assert len(model["metrics"]) == 1
    assert model["metrics"][0]["name"] == "total_revenue"
    assert model["metrics"][0]["expression"]["dialects"][0]["expression"] == (
        "SUM(orders.amount)"
    )


def test_round_trip_preserves_tri_state_time_dimension():
    spec = _spec()
    spec["semantic_model"][0]["datasets"][0]["fields"].append(
        {"name": "flag", "dimension": {"is_time": False}}
    )
    conn = _ingested(spec)
    fields = conn.export("acme")["semantic_model"][0]["datasets"][0]["fields"]
    by_name = {f["name"]: f for f in fields}
    assert by_name["created_at"]["dimension"] == {"is_time": True}
    assert by_name["flag"]["dimension"] == {"is_time": False}
    assert "dimension" not in by_name["status"]  # unknown stays unknown


def test_round_trip_carries_trust_extension():
    conn = _ingested()
    dataset = conn.export("acme")["semantic_model"][0]["datasets"][0]
    # Ingest stamps source_system; it must survive as a vendor extension.
    ext = dataset["custom_extensions"][0]
    assert ext["vendor_name"] == TRUST_VENDOR
    assert ext["data"]["source_system"] == "ossie:acme"


def test_export_only_includes_named_model():
    conn = _ingested()
    other = _Connector(
        {"version": "0.1.1", "semantic_model": [{
            "name": "other",
            "datasets": [{"name": "u", "source": "d.s.u", "fields": []}],
            "relationships": [], "metrics": [],
        }]}
    )
    other.client = conn.client  # share one graph
    other.ingest()

    model = conn.export("acme")["semantic_model"][0]
    assert [d["name"] for d in model["datasets"]] == ["orders"]


# -- version handling ------------------------------------------------------


def test_version_mismatch_records_warning(monkeypatch):
    """Drive the real extract() so the version guard itself is covered."""
    import hugegraph_llm.semantic_layer.connectors.ossie as ossie_mod

    monkeypatch.setattr(ossie_mod, "load_spec", lambda _src: _spec(version="9.9.9"))
    conn = OssieConnector(client=FakeHugeGraph(), spec_source="any.yaml")
    conn.extract()
    assert any("9.9.9" in w for w in conn.warnings)


def test_missing_version_records_warning(monkeypatch):
    import hugegraph_llm.semantic_layer.connectors.ossie as ossie_mod

    spec = _spec()
    del spec["version"]
    monkeypatch.setattr(ossie_mod, "load_spec", lambda _src: spec)
    conn = OssieConnector(client=FakeHugeGraph(), spec_source="any.yaml")
    conn.extract()
    assert any("no top-level 'version'" in w for w in conn.warnings)


def test_load_spec_rejects_non_mapping(tmp_path, monkeypatch):
    import hugegraph_llm.semantic_layer.connectors.ossie as ossie_mod

    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ossie_mod.load_spec(str(spec_file))


def test_model_without_name_is_rejected():
    conn = _Connector({"version": "0.1.1", "semantic_model": [{}]})
    with pytest.raises(ValueError):
        conn.ingest()
