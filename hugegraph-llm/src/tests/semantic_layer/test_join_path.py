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

"""Tests for join-path resolution."""

import pytest

from hugegraph_llm.semantic_layer.join_path import (
    JoinStep,
    find_join_path,
    render_join_path,
)
from hugegraph_llm.semantic_layer.readers import SemanticProjection


def _proj():
    """orders -> customers -> regions, plus an isolated table."""
    proj = SemanticProjection()
    for name in ("orders", "customers", "regions", "island"):
        proj.tables[name] = type("T", (), {})()  # placeholder, replaced below
    from hugegraph_llm.semantic_layer.readers import TableRow
    for name in ("orders", "customers", "regions", "island"):
        proj.tables[name] = TableRow(name=name)
    proj.references["orders.customer_id"] = ["customers.customer_id"]
    proj.references["customers.region_id"] = ["regions.region_id"]
    proj.reference_proven[("orders.customer_id", "customers.customer_id")] = True
    proj.reference_proven[("customers.region_id", "regions.region_id")] = True
    return proj


def test_direct_join_path():
    path = find_join_path(_proj(), "orders", "customers")
    assert path.found
    assert len(path.steps) == 1
    assert path.steps[0].to_sql() == "orders.customer_id = customers.customer_id"
    assert path.all_proven


def test_multi_hop_path_is_ordered():
    path = find_join_path(_proj(), "orders", "regions")
    assert path.found
    assert path.tables == ["orders", "customers", "regions"]
    assert len(path.steps) == 2


def test_path_is_symmetric():
    proj = _proj()
    forward = find_join_path(proj, "orders", "regions")
    backward = find_join_path(proj, "regions", "orders")
    assert forward.found and backward.found
    assert backward.tables == ["regions", "customers", "orders"]


def test_no_path_returns_not_found():
    path = find_join_path(_proj(), "orders", "island")
    assert not path.found
    assert path.steps == []


def test_same_table_has_no_steps():
    path = find_join_path(_proj(), "orders", "orders")
    assert not path.found


def test_unknown_table_returns_not_found():
    path = find_join_path(_proj(), "orders", "does_not_exist")
    assert not path.found


def test_unproven_step_renders_as_comment():
    """An inferred join must never become a silent ON clause."""
    step = JoinStep("orders", "user_id", "users", "user_id", proven=False)
    assert step.to_sql().startswith("/*")
    assert "orders.user_id = users.user_id" not in step.to_sql()


def test_unproven_path_is_flagged():
    proj = _proj()
    proj.references["orders.region_code"] = ["regions.code"]
    proj.reference_proven[("orders.region_code", "regions.code")] = False
    # The proven 2-hop route (cost 2) beats the unproven 1-hop (cost 4).
    path = find_join_path(proj, "orders", "regions")
    assert path.found
    assert path.all_proven
    assert len(path.steps) == 2


def test_unproven_used_when_no_alternative():
    proj = _proj()
    proj.references["island.code"] = ["regions.code"]
    proj.reference_proven[("island.code", "regions.code")] = False
    path = find_join_path(proj, "island", "regions")
    assert path.found
    assert not path.all_proven


def test_to_dict_shape():
    payload = find_join_path(_proj(), "orders", "customers").to_dict()
    assert payload["found"] is True
    assert payload["tables"] == ["orders", "customers"]
    assert payload["steps"][0]["sql"] == "orders.customer_id = customers.customer_id"
    assert payload["steps"][0]["proven"] is True


def test_render_warns_when_no_path():
    text = render_join_path(find_join_path(_proj(), "orders", "island"))
    assert "No column-level join path" in text
    assert "Do not invent" in text


def test_render_warns_on_unproven():
    proj = _proj()
    proj.references["island.code"] = ["regions.code"]
    proj.reference_proven[("island.code", "regions.code")] = False
    text = render_join_path(find_join_path(proj, "island", "regions"))
    assert "unproven join" in text
    assert "Confirm before using" in text


def test_lineage_only_relation_does_not_produce_a_path():
    """LINEAGE says 'related', not 'joinable' -- it must not yield an ON clause."""
    proj = _proj()
    proj.lineage["orders"] = ["island"]
    assert not find_join_path(proj, "orders", "island").found
