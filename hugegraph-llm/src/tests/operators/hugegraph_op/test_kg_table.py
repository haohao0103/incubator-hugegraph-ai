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

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest
from pyhugegraph.utils.exceptions import NotFoundError, ServerError

from hugegraph_llm.operators.hugegraph_op.kg_table import (
    _DEFAULT_SCHEMA,
    KgTable,
    cap_str,
    sanitize_label,
)

pytestmark = [pytest.mark.unit]


class TestKgTableHelpers(unittest.TestCase):
    def test_sanitize_label_keeps_plain_name(self):
        self.assertEqual(sanitize_label("entities"), "entities")

    def test_sanitize_label_replaces_invalid_chars(self):
        self.assertEqual(sanitize_label("my table!"), "my_table")

    def test_sanitize_label_prefixes_digit_start(self):
        self.assertEqual(sanitize_label("1abc"), "ns_1abc")

    def test_sanitize_label_handles_empty(self):
        self.assertEqual(sanitize_label("---"), "default")

    def test_cap_str_truncates_long_text(self):
        self.assertEqual(len(cap_str("a" * 40000)), 32000)

    def test_cap_str_keeps_short_text(self):
        self.assertEqual(cap_str("hello"), "hello")

    def test_cap_str_none_to_empty(self):
        self.assertEqual(cap_str(None), "")

    def test_cap_str_non_string_coerced(self):
        self.assertEqual(cap_str(123), "123")


class TestKgTable(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_graph = MagicMock()
        self.mock_client.graph.return_value = self.mock_graph
        # MagicMock getters are truthy, so make schema lookups report "absent"
        # to exercise the real create path of the idempotent layer.
        self.mock_schema = self.mock_client.schema.return_value
        self.mock_schema.getPropertyKey.return_value = None
        self.mock_schema.getVertexLabel.return_value = None
        self.mock_schema.getIndexLabel.return_value = None
        # Chain the schema builders back to themselves so call chains are
        # observable on a single mock.
        for schema_method in ("propertyKey", "vertexLabel", "edgeLabel", "indexLabel"):
            builder = MagicMock()
            for method in (
                "asText",
                "useCustomizeStringId",
                "properties",
                "nullableKeys",
                "ifNotExist",
                "create",
            ):
                getattr(builder, method).return_value = builder
            setattr(self.mock_schema, schema_method, MagicMock(return_value=builder))
            self._builders = getattr(self, "_builders", {})
            self._builders[schema_method] = builder

    def _vertex(self, row_id, props):
        return MagicMock(id=f"entities:{row_id}", label="entities", properties=props)

    # -- init ---------------------------------------------------------------

    def test_init_defaults(self):
        table = KgTable(self.mock_client, "entities")
        self.assertEqual(table.table_name, "entities")
        self.assertEqual(table.label, "entities")
        self.assertEqual(table.schema, dict(_DEFAULT_SCHEMA))
        self.assertEqual(table._page_size, 500)

    def test_init_custom_schema_and_prefix(self):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT", "age": "INT"},
            page_size=10,
            label_prefix="kg_",
        )
        self.assertEqual(table.label, "kg_entities")
        self.assertEqual(table.schema, {"name": "TEXT", "age": "INT"})
        self.assertEqual(table._page_size, 10)

    def test_init_clamps_page_size(self):
        table = KgTable(self.mock_client, "entities", page_size=0)
        self.assertEqual(table._page_size, 1)

    def test_init_sanitizes_label(self):
        table = KgTable(self.mock_client, "my entities!")
        self.assertEqual(table.label, "my_entities")

    # -- id helpers -----------------------------------------------------------

    def test_vid_builds_prefixed_id(self):
        table = KgTable(self.mock_client, "entities")
        self.assertEqual(table._vid(7), "entities:7")
        self.assertEqual(table._vid("abc"), "entities:abc")

    def test_row_id_strips_prefix(self):
        table = KgTable(self.mock_client, "entities")
        self.assertEqual(table._row_id("entities:7"), "7")
        self.assertEqual(table._row_id("other:7"), "other:7")

    # -- schema ---------------------------------------------------------------

    def test_ensure_schema_creates_once(self):
        table = KgTable(
            self.mock_client, "entities", schema={"name": "TEXT", "status": "TEXT"}
        )
        table._ensure_schema()
        self.assertTrue(table._schema_ensured)
        self.assertEqual(self.mock_schema.propertyKey.call_count, 2)
        vl_builder = self._builders["vertexLabel"]
        vl_builder.useCustomizeStringId.assert_called_once_with()
        vl_builder.nullableKeys.assert_called_once_with("name", "status")

        # second call is a no-op
        table._ensure_schema()
        self.assertEqual(self.mock_schema.propertyKey.call_count, 2)

    # -- upsert ---------------------------------------------------------------

    def test_upsert_writes_vertex_in_schema_mode(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.addVertex.return_value = MagicMock(id="entities:1")
        vid = table.upsert({"id": 1, "name": "Tom", "extra": "dropped"})
        self.assertEqual(vid, "entities:1")
        self.mock_graph.addVertex.assert_called_once_with(
            "entities", {"name": "Tom"}, id="entities:1"
        )

    def test_upsert_missing_id_raises(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        with self.assertRaises(ValueError):
            table.upsert({"name": "Tom"})

    def test_upsert_json_serializes_list_and_dict(self):
        table = KgTable(
            self.mock_client, "entities", schema={"tags": "TEXT", "meta": "TEXT"}
        )
        self.mock_graph.addVertex.return_value = MagicMock(id="entities:1")
        table.upsert({"id": 1, "tags": ["a", "b"], "meta": {"k": "v"}})
        args, _ = self.mock_graph.addVertex.call_args
        props = args[1]
        self.assertEqual(props["tags"], json.dumps(["a", "b"]))
        self.assertEqual(props["meta"], json.dumps({"k": "v"}))

    def test_upsert_caps_long_text(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.addVertex.return_value = MagicMock(id="entities:1")
        table.upsert({"id": 1, "name": "x" * 40000})
        args, _ = self.mock_graph.addVertex.call_args
        self.assertEqual(len(args[1]["name"]), 32000)

    def test_upsert_kv_mode_serializes_whole_row(self):
        table = KgTable(self.mock_client, "kv")  # schema omitted -> value TEXT
        self.mock_graph.addVertex.return_value = MagicMock(id="kv:1")
        table.upsert({"id": 1, "name": "Tom", "age": 30})
        self.mock_graph.addVertex.assert_called_once_with(
            "kv", {"value": json.dumps({"name": "Tom", "age": 30})}, id="kv:1"
        )

    def test_upsert_keeps_scalar_values(self):
        table = KgTable(
            self.mock_client, "entities", schema={"age": "INT", "active": "BOOLEAN"}
        )
        self.mock_graph.addVertex.return_value = MagicMock(id="entities:1")
        table.upsert({"id": 1, "age": 30, "active": True})
        args, _ = self.mock_graph.addVertex.call_args
        self.assertEqual(args[1], {"age": 30, "active": True})

    def test_upsert_raises_when_add_vertex_fails(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.addVertex.return_value = None
        with self.assertRaises(RuntimeError):
            table.upsert({"id": 1, "name": "Tom"})

    def test_upsert_many_writes_all_rows(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.addVertex.return_value = MagicMock(id="entities:1")
        count = table.upsert_many([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        self.assertEqual(count, 2)
        self.assertEqual(self.mock_graph.addVertex.call_count, 2)

    # -- delete / clear -------------------------------------------------------

    def test_delete_returns_true_on_success(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.removeVertexById.return_value = {"deleted": 1}
        self.assertTrue(table.delete(7))
        self.mock_graph.removeVertexById.assert_called_once_with("entities:7")

    def test_delete_returns_false_when_missing(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.removeVertexById.return_value = None
        self.assertFalse(table.delete(7))

    def test_delete_returns_false_on_not_found_error(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.removeVertexById.side_effect = NotFoundError("404")
        self.assertFalse(table.delete(7))

    def test_delete_returns_false_on_server_error(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.removeVertexById.side_effect = ServerError("No such vertex")
        self.assertFalse(table.delete(7))

    def test_clear_deletes_all_rows(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexByPage.return_value = (
            [self._vertex(1, {"name": "a"}), self._vertex(2, {"name": "b"})],
            None,
        )
        self.mock_graph.removeVertexById.return_value = {"deleted": 1}
        self.assertEqual(table.clear(), 2)
        self.assertEqual(self.mock_graph.removeVertexById.call_count, 2)

    def test_clear_counts_only_successful_deletes(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexByPage.return_value = (
            [self._vertex(1, {"name": "a"}), self._vertex(2, {"name": "b"})],
            None,
        )
        self.mock_graph.removeVertexById.side_effect = [{"deleted": 1}, None]
        self.assertEqual(table.clear(), 1)
        self.assertEqual(self.mock_graph.removeVertexById.call_count, 2)

    # -- read -----------------------------------------------------------------

    def test_get_returns_flattened_row(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexById.return_value = self._vertex(7, {"name": "Tom"})
        row = table.get(7)
        self.assertEqual(row, {"id": "7", "name": "Tom"})

    def test_flatten_drops_unknown_props(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        vertex = self._vertex(7, {"name": "Tom", "unknown_col": 42})
        row = table._flatten(vertex)
        self.assertEqual(row, {"id": "7", "name": "Tom"})

    def test_get_returns_none_when_absent(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexById.return_value = None
        self.assertIsNone(table.get(7))

    def test_get_returns_none_on_not_found_error(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexById.side_effect = NotFoundError("404")
        self.assertIsNone(table.get(7))

    def test_has_true_and_false(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexById.return_value = self._vertex(7, {"name": "Tom"})
        self.assertTrue(table.has(7))
        self.mock_graph.getVertexById.return_value = None
        self.assertFalse(table.has(8))

    def test_has_false_on_not_found_error(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexById.side_effect = NotFoundError("404")
        self.assertFalse(table.has(7))

    def test_length_counts_single_page(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.getVertexByPage.return_value = (
            [self._vertex(1, {}), self._vertex(2, {})],
            None,
        )
        self.assertEqual(table.length(), 2)

    def test_length_pages_until_empty(self):
        table = KgTable(self.mock_client, "entities", page_size=2)
        self.mock_graph.getVertexByPage.side_effect = [
            ([self._vertex(1, {}), self._vertex(2, {})], "p2"),
            ([self._vertex(3, {})], None),
        ]
        self.assertEqual(table.length(), 3)

    def test_length_empty_table(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.getVertexByPage.return_value = ([], None)
        self.assertEqual(table.length(), 0)

    def test_list_rows_single_page(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexByPage.return_value = (
            [self._vertex(1, {"name": "a"})],
            None,
        )
        rows = table.list_rows()
        self.assertEqual(rows, [{"id": "1", "name": "a"}])

    def test_list_rows_paginates_and_honors_limit(self):
        table = KgTable(self.mock_client, "entities", page_size=2)
        self.mock_graph.getVertexByPage.side_effect = [
            ([self._vertex(1, {}), self._vertex(2, {})], "p2"),
            ([self._vertex(3, {})], None),
        ]
        rows = table.list_rows(limit=3)
        self.assertEqual([r["id"] for r in rows], ["1", "2", "3"])

    def test_list_rows_passes_properties_filter(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.getVertexByPage.return_value = ([], None)
        table.list_rows(properties={"status": "processed"})
        self.mock_graph.getVertexByPage.assert_called_once_with(
            "entities", 500, page=None, properties={"status": "processed"}
        )

    def test_list_rows_empty_table(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.getVertexByPage.return_value = ([], None)
        self.assertEqual(table.list_rows(), [])

    def test_list_rows_limit_zero_breaks_immediately(self):
        table = KgTable(self.mock_client, "entities")
        rows = table.list_rows(limit=0)
        self.assertEqual(rows, [])
        self.mock_graph.getVertexByPage.assert_not_called()

    def test_iter_rows_yields_all(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.mock_graph.getVertexByPage.return_value = (
            [self._vertex(1, {"name": "a"}), self._vertex(2, {"name": "b"})],
            None,
        )
        rows = list(table.iter_rows())
        self.assertEqual([r["id"] for r in rows], ["1", "2"])

    # -- internal helpers -----------------------------------------------------

    def test_page_vertices_degrades_on_error(self):
        table = KgTable(self.mock_client, "entities")
        self.mock_graph.getVertexByPage.side_effect = RuntimeError("boom")
        self.assertEqual(table._page_vertices(None), ([], None))

    def test_serialize_props_drops_unknown_columns(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        props = table._serialize_props({"id": 1, "name": "Tom", "extra": "x"})
        self.assertEqual(props, {"name": "Tom"})

    def test_serialize_props_kv_mode(self):
        table = KgTable(self.mock_client, "kv")
        props = table._serialize_props({"id": 1, "a": 1, "b": "x"})
        self.assertEqual(props, {"value": json.dumps({"a": 1, "b": "x"})})


    # -- index_fields ----------------------------------------------------------

    def test_init_default_no_index_fields(self):
        table = KgTable(self.mock_client, "entities", schema={"name": "TEXT"})
        self.assertEqual(table.index_fields, [])

    def test_init_index_fields(self):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT", "status": "TEXT"},
            index_fields=["name"],
        )
        self.assertEqual(table.index_fields, ["name"])

    def test_ensure_schema_creates_secondary_indexes(self):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT", "status": "TEXT"},
            index_fields=["name", "status"],
        )
        table._ensure_schema()
        self.assertEqual(self.mock_schema.indexLabel.call_count, 2)
        self.mock_schema.indexLabel.assert_any_call("entities_name")
        self.mock_schema.indexLabel.assert_any_call("entities_status")
        index_builder = self.mock_schema.indexLabel.return_value
        index_builder.secondary.assert_called()
        table._ensure_schema()
        self.assertEqual(self.mock_schema.indexLabel.call_count, 2)

    def test_ensure_schema_skips_index_field_not_in_schema(self):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT"},
            index_fields=["name", "ghost"],
        )
        table._ensure_schema()
        self.assertEqual(self.mock_schema.indexLabel.call_count, 1)
        self.mock_schema.indexLabel.assert_called_once_with("entities_name")

    @patch("pyhugegraph.api.rebuild.RebuildManager")
    def test_rebuild_indexes_triggers_tasks(self, mock_rebuild_cls):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT", "status": "TEXT"},
            index_fields=["name", "status"],
        )
        mock_rebuild = MagicMock()
        mock_rebuild.rebuild_indexlabels.side_effect = [
            {"task_id": 10},
            {"task_id": 11},
        ]
        mock_rebuild_cls.return_value = mock_rebuild

        tasks = table.rebuild_indexes()

        self.assertEqual(tasks, {"entities_name": 10, "entities_status": 11})
        mock_rebuild.rebuild_indexlabels.assert_any_call("entities_name")
        mock_rebuild.rebuild_indexlabels.assert_any_call("entities_status")

    @patch("pyhugegraph.api.rebuild.RebuildManager")
    def test_rebuild_indexes_skips_field_not_in_schema(self, mock_rebuild_cls):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT"},
            index_fields=["ghost"],
        )
        mock_rebuild = MagicMock()
        mock_rebuild_cls.return_value = mock_rebuild
        tasks = table.rebuild_indexes()
        self.assertEqual(tasks, {})
        mock_rebuild.rebuild_indexlabels.assert_not_called()

    @patch("pyhugegraph.api.rebuild.RebuildManager")
    def test_rebuild_indexes_handles_non_dict_and_errors(self, mock_rebuild_cls):
        table = KgTable(
            self.mock_client,
            "entities",
            schema={"name": "TEXT"},
            index_fields=["name"],
        )
        mock_rebuild = MagicMock()
        mock_rebuild.rebuild_indexlabels.return_value = None  # no task id
        mock_rebuild_cls.return_value = mock_rebuild
        tasks = table.rebuild_indexes()
        self.assertEqual(tasks, {"entities_name": None})

        mock_rebuild.rebuild_indexlabels.side_effect = RuntimeError("rebuild down")
        tasks = table.rebuild_indexes()
        self.assertEqual(tasks, {})

    def test_kv_mode_index_field_skipped(self):
        table = KgTable(self.mock_client, "kv", index_fields=["name"])
        table._ensure_schema()
        self.assertEqual(self.mock_schema.indexLabel.call_count, 0)


if __name__ == "__main__":
    unittest.main()
