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

"""Unit tests for incremental indexing utilities.

Tests cover:
- persist_community_assignments
- clear_stale_community_assignments
- find_affected_communities
- get_community_vertices
- get_community_edges
- Gremlin response parsers
"""

import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.operators.graph_op.incremental_utils import (
    _parse_gremlin_count,
    _parse_gremlin_list,
    _parse_gremlin_vertex_list,
    clear_stale_community_assignments,
    find_affected_communities,
    get_community_edges,
    get_community_vertices,
    persist_community_assignments,
)


# ---------------------------------------------------------------------------
# Gremlin response parser tests
# ---------------------------------------------------------------------------

class TestParseGremlinCount(unittest.TestCase):
    """Test Gremlin count response parser."""

    def test_dict_with_result(self):
        resp = {"data": [{"result": 42}]}
        self.assertEqual(_parse_gremlin_count(resp), 42)

    def test_dict_with_count(self):
        resp = {"data": [{"count": 10}]}
        self.assertEqual(_parse_gremlin_count(resp), 10)

    def test_dict_with_nested_list(self):
        resp = {"data": [[5]]}
        self.assertEqual(_parse_gremlin_count(resp), 5)

    def test_empty_dict(self):
        self.assertEqual(_parse_gremlin_count({}), 0)

    def test_int_response(self):
        self.assertEqual(_parse_gremlin_count(7), 7)

    def test_float_response(self):
        self.assertEqual(_parse_gremlin_count(3.14), 3)

    def test_zero_response(self):
        self.assertEqual(_parse_gremlin_count(0), 0)


class TestParseGremlinList(unittest.TestCase):
    """Test Gremlin list response parser."""

    def test_dict_with_result(self):
        resp = {"data": [{"result": "C1"}]}
        self.assertEqual(_parse_gremlin_list(resp), ["C1"])

    def test_dict_with_nested_list(self):
        resp = {"data": [["C1", "C2", "C3"]]}
        result = _parse_gremlin_list(resp)
        self.assertEqual(len(result), 3)
        self.assertIn("C1", result)

    def test_empty_dict(self):
        self.assertEqual(_parse_gremlin_list({}), [])

    def test_list_response(self):
        self.assertEqual(_parse_gremlin_list(["A", "B"]), ["A", "B"])

    def test_list_with_none(self):
        self.assertEqual(_parse_gremlin_list(["A", None, "B"]), ["A", "B"])


class TestParseGremlinVertexList(unittest.TestCase):
    """Test Gremlin vertex list response parser."""

    def test_dict_with_single_vertex(self):
        resp = {"data": [{"id": "v1", "label": "Person"}]}
        result = _parse_gremlin_vertex_list(resp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "v1")

    def test_dict_with_list_of_vertices(self):
        resp = {"data": [[
            {"id": "v1", "label": "Person"},
            {"id": "v2", "label": "Org"},
        ]]}
        result = _parse_gremlin_vertex_list(resp)
        self.assertEqual(len(result), 2)

    def test_empty(self):
        self.assertEqual(_parse_gremlin_vertex_list({}), [])


# ---------------------------------------------------------------------------
# persist_community_assignments tests
# ---------------------------------------------------------------------------

class TestPersistCommunityAssignments(unittest.TestCase):
    """Test community assignment persistence."""

    def _make_client(self):
        client = MagicMock()
        schema = MagicMock()
        schema.propertyKey.return_value.ifNotExist.return_value.asText.return_value.create.return_value = None
        client.schema.return_value = schema
        client.gremlin.return_value.exec = MagicMock(return_value={"data": [{"result": 2}]})
        return client

    def test_persist_basic(self):
        client = self._make_client()
        communities = [
            {"id": "C1", "vertices": ["v1", "v2"]},
            {"id": "C2", "vertices": ["v3", "v4", "v5"]},
        ]
        result = persist_community_assignments(client, communities)
        self.assertEqual(result["updated_count"], 4)  # 2 + 2 batches
        self.assertEqual(len(result["errors"]), 0)

    def test_persist_empty(self):
        client = self._make_client()
        result = persist_community_assignments(client, [])
        self.assertEqual(result["updated_count"], 0)

    def test_persist_no_id(self):
        client = self._make_client()
        communities = [{"id": "", "vertices": ["v1"]}]
        result = persist_community_assignments(client, communities)
        self.assertEqual(result["updated_count"], 0)

    def test_persist_large_batch(self):
        client = MagicMock()
        schema = MagicMock()
        client.schema.return_value = schema
        # Mock 2 batches
        client.gremlin.return_value.exec = MagicMock(return_value={"data": [{"result": 100}]})
        vertices = [f"v{i}" for i in range(200)]
        communities = [{"id": "C1", "vertices": vertices}]
        result = persist_community_assignments(client, communities, batch_size=100)
        self.assertEqual(result["updated_count"], 200)


# ---------------------------------------------------------------------------
# clear_stale_community_assignments tests
# ---------------------------------------------------------------------------

class TestClearStaleAssignments(unittest.TestCase):
    """Test clearing stale community assignments."""

    def test_clear_all(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=[
            {"data": [{"result": 10}]},  # total count
            {"data": [{"result": 10}]},  # cleared count
        ])
        cleared = clear_stale_community_assignments(client, keep_community_ids=None)
        self.assertEqual(cleared, 10)

    def test_clear_with_keep_set(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=[
            {"data": [{"result": 10}]},  # total count
            {"data": [{"result": 5}]},   # cleared count
        ])
        cleared = clear_stale_community_assignments(client, keep_community_ids={"C1", "C2"})
        self.assertEqual(cleared, 5)

    def test_clear_error_fallback(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=[
            Exception("connection error"),
        ])
        cleared = clear_stale_community_assignments(client)
        self.assertEqual(cleared, 0)


# ---------------------------------------------------------------------------
# find_affected_communities tests
# ---------------------------------------------------------------------------

class TestFindAffectedCommunities(unittest.TestCase):
    """Test affected community detection."""

    def test_empty_vertices(self):
        client = MagicMock()
        affected = find_affected_communities(client, [])
        self.assertEqual(affected, set())

    def test_single_vertex(self):
        client = MagicMock()
        # Mock: 1-hop neighbors have community C1, self has community C2
        client.gremlin.return_value.exec = MagicMock(side_effect=[
            {"data": [["C1", "C3"]]},  # neighbor communities
            {"data": [["C2"]]},       # self community
        ])
        affected = find_affected_communities(client, ["v_new_1"])
        self.assertEqual(affected, {"C1", "C2", "C3"})

    def test_multiple_vertices_batched(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(side_effect=[
            {"data": [["C1"]]},       # batch 1 neighbors
            {"data": [["C2"]]},       # batch 2 neighbors
            {"data": [["C1", "C2"]]},  # self communities
        ])
        affected = find_affected_communities(client, ["v1", "v2", "v3"])
        self.assertIn("C1", affected)
        self.assertIn("C2", affected)

    def test_no_community_assigned(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(return_value={"data": [[]]})
        affected = find_affected_communities(client, ["v_isolated"])
        self.assertEqual(affected, set())


# ---------------------------------------------------------------------------
# get_community_vertices tests
# ---------------------------------------------------------------------------

class TestGetCommunityVertices(unittest.TestCase):
    """Test fetching community vertices."""

    def test_empty_set(self):
        client = MagicMock()
        result = get_community_vertices(client, set())
        self.assertEqual(result, {})

    def test_with_data(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(return_value={
            "data": [[
                {"id": "v1", "label": "Person", "community_id": "C1"},
                {"id": "v2", "label": "Person", "community_id": "C1"},
                {"id": "v3", "label": "Org", "community_id": "C2"},
            ]]
        })
        result = get_community_vertices(client, {"C1", "C2"})
        self.assertIn("C1", result)
        self.assertIn("C2", result)
        self.assertEqual(len(result["C1"]), 2)
        self.assertEqual(len(result["C2"]), 1)


# ---------------------------------------------------------------------------
# get_community_edges tests
# ---------------------------------------------------------------------------

class TestGetCommunityEdges(unittest.TestCase):
    """Test fetching community edges."""

    def test_empty_set(self):
        client = MagicMock()
        result = get_community_edges(client, set())
        self.assertEqual(result, [])

    def test_with_edges(self):
        client = MagicMock()
        client.gremlin.return_value.exec = MagicMock(return_value={
            "data": [[
                {"label": "knows", "outV": "v1", "inV": "v2"},
            ]]
        })
        result = get_community_edges(client, {"v1", "v2"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], "knows")


if __name__ == "__main__":
    unittest.main()
