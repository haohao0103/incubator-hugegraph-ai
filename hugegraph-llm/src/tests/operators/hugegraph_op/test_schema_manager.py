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
from urllib.parse import quote

import pytest
from requests.exceptions import RequestException

from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager

pytestmark = [pytest.mark.unit]


class TestSchemaManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Setup mock client
        self.mock_client = MagicMock()
        self.mock_schema = MagicMock()
        self.mock_client.schema.return_value = self.mock_schema

        # Create SchemaManager instance
        self.graph_name = "test_graph"
        with patch("hugegraph_llm.operators.hugegraph_op.schema_manager.PyHugeClient") as mock_client_class:
            mock_client_class.return_value = self.mock_client
            self.schema_manager = SchemaManager(self.graph_name)

        # Sample schema data for testing
        self.sample_schema = {
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["name", "age"],
                    "primary_keys": ["name"],
                    "nullable_keys": [],
                    "index_labels": [],
                },
                {
                    "id": 2,
                    "name": "software",
                    "properties": ["name", "lang"],
                    "primary_keys": ["name"],
                    "nullable_keys": [],
                    "index_labels": [],
                },
            ],
            "edgelabels": [
                {
                    "id": 3,
                    "name": "created",
                    "source_label": "person",
                    "target_label": "software",
                    "frequency": "SINGLE",
                    "properties": ["weight"],
                    "sort_keys": [],
                    "nullable_keys": [],
                    "index_labels": [],
                },
                {
                    "id": 4,
                    "name": "knows",
                    "source_label": "person",
                    "target_label": "person",
                    "frequency": "SINGLE",
                    "properties": ["weight"],
                    "sort_keys": [],
                    "nullable_keys": [],
                    "index_labels": [],
                },
            ],
        }

    def test_init(self):
        """Test initialization of SchemaManager class."""
        self.assertEqual(self.schema_manager.graph_name, self.graph_name)
        self.assertEqual(self.schema_manager.client, self.mock_client)
        self.assertEqual(self.schema_manager.schema, self.mock_schema)

    @patch("hugegraph_llm.operators.hugegraph_op.schema_manager.PyHugeClient")
    @patch("hugegraph_llm.operators.hugegraph_op.schema_manager.huge_settings")
    def test_init_uses_request_local_connection_settings(self, mock_settings, mock_client_class):
        mock_settings.graph_url = "default:8080"
        mock_settings.graph_user = "default_user"
        mock_settings.graph_pwd = "default_pwd"
        mock_settings.graph_space = "default_space"

        SchemaManager(
            "custom_graph",
            connection={
                "url": "10.0.0.1:8080",
                "user": "admin",
                "pwd": "secret",
                "graphspace": "space_a",
            },
        )

        mock_client_class.assert_called_once_with(
            url="10.0.0.1:8080",
            graph="custom_graph",
            user="admin",
            pwd="secret",
            graphspace="space_a",
        )

    @patch("hugegraph_llm.operators.hugegraph_op.schema_manager.PyHugeClient")
    @patch("hugegraph_llm.operators.hugegraph_op.schema_manager.huge_settings")
    def test_init_request_config_does_not_inherit_global_graphspace(self, mock_settings, mock_client_class):
        mock_settings.graph_url = "default:8080"
        mock_settings.graph_user = "default_user"
        mock_settings.graph_pwd = "default_pwd"
        mock_settings.graph_space = "global_space"

        SchemaManager(
            "custom_graph",
            connection={
                "url": "10.0.0.1:8080",
                "user": "admin",
                "pwd": "secret",
                "graphspace": None,
            },
        )

        _, kwargs = mock_client_class.call_args
        assert kwargs["graphspace"] is None
        assert kwargs["url"] == "10.0.0.1:8080"

    @patch("hugegraph_llm.operators.hugegraph_op.schema_manager.PyHugeClient")
    @patch("hugegraph_llm.operators.hugegraph_op.schema_manager.huge_settings")
    def test_init_falls_back_to_globals_without_connection(self, mock_settings, mock_client_class):
        mock_settings.graph_url = "default:8080"
        mock_settings.graph_user = "default_user"
        mock_settings.graph_pwd = "default_pwd"
        mock_settings.graph_space = "global_space"

        SchemaManager("custom_graph")

        mock_client_class.assert_called_once_with(
            url="default:8080",
            graph="custom_graph",
            user="default_user",
            pwd="default_pwd",
            graphspace="global_space",
        )

    def test_simple_schema_with_full_schema(self):
        """Test simple_schema method with a full schema."""
        # Call the method
        simple_schema = self.schema_manager.simple_schema(self.sample_schema)

        # Verify the result
        self.assertIn("vertexlabels", simple_schema)
        self.assertIn("edgelabels", simple_schema)

        # Check vertex labels
        self.assertEqual(len(simple_schema["vertexlabels"]), 2)
        for vertex in simple_schema["vertexlabels"]:
            self.assertIn("id", vertex)
            self.assertIn("name", vertex)
            self.assertIn("properties", vertex)
            self.assertNotIn("primary_keys", vertex)
            self.assertNotIn("nullable_keys", vertex)
            self.assertNotIn("index_labels", vertex)

        # Check edge labels
        self.assertEqual(len(simple_schema["edgelabels"]), 2)
        for edge in simple_schema["edgelabels"]:
            self.assertIn("name", edge)
            self.assertIn("source_label", edge)
            self.assertIn("target_label", edge)
            self.assertIn("properties", edge)
            self.assertNotIn("id", edge)
            self.assertNotIn("frequency", edge)
            self.assertNotIn("sort_keys", edge)
            self.assertNotIn("nullable_keys", edge)
            self.assertNotIn("index_labels", edge)

    def test_simple_schema_with_empty_schema(self):
        """Test simple_schema method with an empty schema."""
        empty_schema = {}
        simple_schema = self.schema_manager.simple_schema(empty_schema)
        self.assertEqual(simple_schema, {})

    def test_simple_schema_with_partial_schema(self):
        """Test simple_schema method with a partial schema."""
        partial_schema = {"vertexlabels": [{"id": 1, "name": "person", "properties": ["name", "age"]}]}
        simple_schema = self.schema_manager.simple_schema(partial_schema)
        self.assertIn("vertexlabels", simple_schema)
        self.assertNotIn("edgelabels", simple_schema)
        self.assertEqual(len(simple_schema["vertexlabels"]), 1)

    def test_run_with_valid_schema(self):
        """Test run method with a valid schema."""
        # Setup mock to return the sample schema
        self.mock_schema.getSchema.return_value = self.sample_schema

        # Call the run method
        context = {}
        result = self.schema_manager.run(context)

        # Verify the result
        self.assertIn("schema", result)
        self.assertIn("simple_schema", result)
        self.assertEqual(result["schema"], self.sample_schema)

    def test_run_with_empty_schema(self):
        """Test run method with an empty schema."""
        # Setup mock to return empty schema
        empty_schema = {"vertexlabels": [], "edgelabels": []}
        self.mock_schema.getSchema.return_value = empty_schema

        # Call the run method and expect an exception
        with self.assertRaises(Exception) as cm:
            self.schema_manager.run({})

        # Verify the exception message
        self.assertIn(f"Cannot get {self.graph_name}'s schema from HugeGraph!", str(cm.exception))

    def test_run_with_existing_context(self):
        """Test run method with an existing context."""
        # Setup mock to return the sample schema
        self.mock_schema.getSchema.return_value = self.sample_schema

        # Call the run method with an existing context
        existing_context = {"existing_key": "existing_value"}
        result = self.schema_manager.run(existing_context)

        # Verify the result
        self.assertIn("existing_key", result)
        self.assertEqual(result["existing_key"], "existing_value")
        self.assertIn("schema", result)
        self.assertIn("simple_schema", result)

    def test_run_with_none_context(self):
        """Test run method with None context."""
        # Setup mock to return the sample schema
        self.mock_schema.getSchema.return_value = self.sample_schema

        # Call the run method with None context
        result = self.schema_manager.run(None)

        # Verify the result
        self.assertIn("schema", result)
        self.assertIn("simple_schema", result)

    def test_run_with_connection_error(self):
        """Test run method when the server connection fails."""
        self.mock_schema.getSchema.side_effect = RequestException("connection refused")
        with self.assertRaises(ValueError) as cm:
            self.schema_manager.run({})
        self.assertIn(f"Failed to connect to HugeGraph to get schema '{self.graph_name}'", str(cm.exception))


class TestSchemaManagerIdempotentSchema(unittest.TestCase):
    """Coverage for the idempotent schema layer (generalized from the
    MS-GraphRAG HugeGraph provider: exists-probe before create,
    nullable-everything labels, JSON-string-literal vertex id encoding)."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_schema = MagicMock()
        self.mock_client.schema.return_value = self.mock_schema
        with patch("hugegraph_llm.operators.hugegraph_op.schema_manager.PyHugeClient") as mock_client_class:
            mock_client_class.return_value = self.mock_client
            self.sm = SchemaManager("test_graph")

    def _absent(self):
        """Make every schema getter report "not found" (MagicMock is truthy)."""
        self.mock_schema.getPropertyKey.return_value = None
        self.mock_schema.getVertexLabel.return_value = None
        self.mock_schema.getEdgeLabel.return_value = None
        self.mock_schema.getIndexLabel.return_value = None

    def _builder(self, schema_method):
        builder = MagicMock()
        schema_method.return_value = builder
        for name in (
            "asText",
            "asInt",
            "asLong",
            "asDouble",
            "asDate",
            "valueSingle",
            "valueList",
            "valueSet",
            "properties",
            "nullableKeys",
            "primaryKeys",
            "usePrimaryKeyId",
            "useCustomizeStringId",
            "useCustomizeNumberId",
            "useAutomaticId",
            "sourceLabel",
            "targetLabel",
            "onV",
            "onE",
            "by",
            "secondary",
            "range",
            "search",
            "shard",
            "unique",
            "ifNotExist",
        ):
            getattr(builder, name).return_value = builder
        builder.create.return_value = None
        return builder

    # -- encode_vertex_id ----------------------------------------------------

    def test_encode_vertex_id_plain(self):
        self.assertEqual(
            SchemaManager.encode_vertex_id("Apple"), quote(json.dumps("Apple"), safe="")
        )

    def test_encode_vertex_id_cjk_and_spaces(self):
        vid = "Tom Hanks 张三"
        self.assertEqual(
            SchemaManager.encode_vertex_id(vid),
            quote(json.dumps(vid, ensure_ascii=False), safe=""),
        )

    def test_encode_vertex_id_number(self):
        self.assertEqual(
            SchemaManager.encode_vertex_id(123), quote(json.dumps("123"), safe="")
        )

    def test_encode_vertex_id_none(self):
        self.assertEqual(
            SchemaManager.encode_vertex_id(None), quote(json.dumps("None"), safe="")
        )

    # -- exists --------------------------------------------------------------

    def test_exists_true_for_all_kinds(self):
        kinds = {
            "propertykeys": "getPropertyKey",
            "vertexlabels": "getVertexLabel",
            "edgelabels": "getEdgeLabel",
            "indexlabels": "getIndexLabel",
        }
        for kind, getter in kinds.items():
            with self.subTest(kind=kind):
                getattr(self.mock_schema, getter).return_value = object()
                self.assertTrue(self.sm.exists(kind, "x"))
                getattr(self.mock_schema, getter).assert_called_once_with("x")

    def test_exists_false_for_all_kinds(self):
        self._absent()
        for kind in ("propertykeys", "vertexlabels", "edgelabels", "indexlabels"):
            with self.subTest(kind=kind):
                self.assertFalse(self.sm.exists(kind, "x"))

    def test_exists_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            self.sm.exists("bogus_kind", "x")

    def test_exists_getter_exception_returns_false(self):
        self.mock_schema.getPropertyKey.side_effect = RuntimeError("boom")
        self.assertFalse(self.sm.exists("propertykeys", "x"))

    # -- create_property_key -------------------------------------------------

    def test_create_property_key_new(self):
        self._absent()
        builder = self._builder(self.mock_schema.propertyKey)
        self.assertTrue(self.sm.create_property_key("name", "TEXT", "SINGLE"))
        self.mock_schema.propertyKey.assert_called_once_with("name")
        builder.asText.assert_called_once_with()
        builder.valueSingle.assert_called_once_with()
        builder.ifNotExist.assert_called_once_with()
        builder.create.assert_called_once_with()

    def test_create_property_key_already_exists(self):
        self.mock_schema.getPropertyKey.return_value = object()
        self.assertFalse(self.sm.create_property_key("name", "TEXT"))
        self.mock_schema.propertyKey.assert_not_called()

    def test_create_property_key_invalid_data_type(self):
        self._absent()
        self.assertFalse(self.sm.create_property_key("x", "NOT_A_TYPE"))
        self.mock_schema.propertyKey.assert_not_called()

    def test_create_property_key_invalid_cardinality(self):
        self._absent()
        self.assertFalse(self.sm.create_property_key("x", "TEXT", "BOGUS"))
        self.mock_schema.propertyKey.assert_not_called()

    def test_create_property_key_data_type_mappings(self):
        cases = {
            "INT": "asInt",
            "LONG": "asLong",
            "FLOAT": "asDouble",
            "DOUBLE": "asDouble",
            "BYTE": "asInt",
            "BLOB": "asText",
            "DATE": "asDate",
            "UUID": "asText",
        }
        for data_type, method in cases.items():
            with self.subTest(data_type=data_type):
                self._absent()
                builder = self._builder(self.mock_schema.propertyKey)
                self.assertTrue(self.sm.create_property_key(f"p_{data_type}", data_type))
                getattr(builder, method).assert_called_once_with()
                builder.valueSingle.assert_called_once_with()

    def test_create_property_key_boolean_no_type_setter(self):
        self._absent()
        builder = self._builder(self.mock_schema.propertyKey)
        self.assertTrue(self.sm.create_property_key("flag", "BOOLEAN"))
        builder.asText.assert_not_called()
        builder.create.assert_called_once_with()

    def test_create_property_key_cardinality_list_set(self):
        for cardinality, method in (("LIST", "valueList"), ("SET", "valueSet")):
            with self.subTest(cardinality=cardinality):
                self._absent()
                builder = self._builder(self.mock_schema.propertyKey)
                self.assertTrue(self.sm.create_property_key("hobbies", "TEXT", cardinality))
                getattr(builder, method).assert_called_once_with()

    # -- create_vertex_label -------------------------------------------------

    def test_create_vertex_label_primary_key_default_nullable(self):
        self._absent()
        builder = self._builder(self.mock_schema.vertexLabel)
        self.assertTrue(
            self.sm.create_vertex_label(
                "person", ["name", "age"], id_strategy="PRIMARY_KEY", primary_keys=["name"]
            )
        )
        builder.usePrimaryKeyId.assert_called_once_with()
        builder.primaryKeys.assert_called_once_with("name")
        builder.nullableKeys.assert_called_once_with("name", "age")
        builder.create.assert_called_once_with()

    def test_create_vertex_label_primary_key_missing_pk(self):
        self._absent()
        self.assertFalse(self.sm.create_vertex_label("person", ["name"], primary_keys=[]))
        self.mock_schema.vertexLabel.assert_not_called()

    def test_create_vertex_label_customize_string(self):
        self._absent()
        builder = self._builder(self.mock_schema.vertexLabel)
        self.assertTrue(
            self.sm.create_vertex_label("person", ["name"], id_strategy="CUSTOMIZE_STRING")
        )
        builder.useCustomizeStringId.assert_called_once_with()
        builder.usePrimaryKeyId.assert_not_called()

    def test_create_vertex_label_customize_number(self):
        self._absent()
        builder = self._builder(self.mock_schema.vertexLabel)
        self.assertTrue(
            self.sm.create_vertex_label("person", ["name"], id_strategy="CUSTOMIZE_NUMBER")
        )
        builder.useCustomizeNumberId.assert_called_once_with()

    def test_create_vertex_label_automatic(self):
        for strategy in ("AUTOMATIC", "AUTO"):
            with self.subTest(strategy=strategy):
                self._absent()
                builder = self._builder(self.mock_schema.vertexLabel)
                self.assertTrue(
                    self.sm.create_vertex_label("person", ["name"], id_strategy=strategy)
                )
                builder.useAutomaticId.assert_called_once_with()

    def test_create_vertex_label_unknown_strategy(self):
        self._absent()
        self.assertFalse(self.sm.create_vertex_label("person", ["name"], id_strategy="BOGUS"))
        self.mock_schema.vertexLabel.assert_not_called()

    def test_create_vertex_label_explicit_nullable_keys(self):
        self._absent()
        builder = self._builder(self.mock_schema.vertexLabel)
        self.assertTrue(
            self.sm.create_vertex_label(
                "person",
                ["name", "age"],
                id_strategy="PRIMARY_KEY",
                primary_keys=["name"],
                nullable_keys=["age"],
            )
        )
        builder.nullableKeys.assert_called_once_with("age")

    def test_create_vertex_label_already_exists(self):
        self.mock_schema.getVertexLabel.return_value = object()
        self.assertFalse(self.sm.create_vertex_label("person", ["name"]))
        self.mock_schema.vertexLabel.assert_not_called()

    # -- create_edge_label ---------------------------------------------------

    def test_create_edge_label_new(self):
        self._absent()
        builder = self._builder(self.mock_schema.edgeLabel)
        self.assertTrue(self.sm.create_edge_label("knows", "person", "person", ["weight"]))
        builder.sourceLabel.assert_called_once_with("person")
        builder.targetLabel.assert_called_once_with("person")
        builder.properties.assert_called_once_with("weight")
        builder.nullableKeys.assert_called_once_with("weight")
        builder.create.assert_called_once_with()

    def test_create_edge_label_already_exists(self):
        self.mock_schema.getEdgeLabel.return_value = object()
        self.assertFalse(self.sm.create_edge_label("knows", "person", "person", ["weight"]))
        self.mock_schema.edgeLabel.assert_not_called()

    # -- create_index_label --------------------------------------------------

    def test_create_index_label_secondary_on_vertex(self):
        self._absent()
        builder = self._builder(self.mock_schema.indexLabel)
        self.assertTrue(self.sm.create_index_label("personByName", "person", "name"))
        builder.onV.assert_called_once_with("person")
        builder.onE.assert_not_called()
        builder.by.assert_called_once_with("name")
        builder.secondary.assert_called_once_with()
        builder.create.assert_called_once_with()

    def test_create_index_label_on_edge(self):
        self._absent()
        builder = self._builder(self.mock_schema.indexLabel)
        self.assertTrue(
            self.sm.create_index_label("edgeByName", "edge", "name", on="edge")
        )
        builder.onE.assert_called_once_with("edge")

    def test_create_index_label_types(self):
        for index_type, method in (
            ("RANGE", "range"),
            ("SEARCH", "search"),
            ("SHARD", "shard"),
            ("UNIQUE", "unique"),
        ):
            with self.subTest(index_type=index_type):
                self._absent()
                builder = self._builder(self.mock_schema.indexLabel)
                self.assertTrue(
                    self.sm.create_index_label("i", "person", "age", index_type=index_type)
                )
                getattr(builder, method).assert_called_once_with()

    def test_create_index_label_unknown_type(self):
        self._absent()
        self.assertFalse(self.sm.create_index_label("i", "person", "name", index_type="BOGUS"))
        self.mock_schema.indexLabel.assert_not_called()

    def test_create_index_label_already_exists(self):
        self.mock_schema.getIndexLabel.return_value = object()
        self.assertFalse(self.sm.create_index_label("personByName", "person", "name"))
        self.mock_schema.indexLabel.assert_not_called()

    # -- ensure_schema -------------------------------------------------------

    @staticmethod
    def _sample_schema():
        return {
            "propertykeys": [
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "age", "data_type": "INT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "name": "person",
                    "properties": ["id", "name", "age"],
                    "primary_keys": ["id"],
                    "id_strategy": "PRIMARY_KEY",
                },
                {
                    "name": "movie",
                    "properties": ["title", "name"],
                    "primary_keys": ["title"],
                    "id_strategy": "PRIMARY_KEY",
                },
            ],
            "edgelabels": [
                {
                    "name": "acted_in",
                    "properties": ["role"],
                    "source_label": "person",
                    "target_label": "movie",
                }
            ],
        }

    def test_ensure_schema_creates_everything(self):
        self._absent()
        for method in ("propertyKey", "vertexLabel", "edgeLabel", "indexLabel"):
            self._builder(getattr(self.mock_schema, method))
        summary = self.sm.ensure_schema(self._sample_schema())
        # 2 props + 2 vertex labels + 1 edge label + 2 name indexes
        # (person.name and movie.name are not primary keys)
        self.assertEqual(
            summary,
            {"property_keys": 2, "vertex_labels": 2, "edge_labels": 1, "index_labels": 2},
        )
        self.assertEqual(self.mock_schema.indexLabel.call_count, 2)
        self.mock_schema.indexLabel.assert_any_call("personByName")
        self.mock_schema.indexLabel.assert_any_call("movieByName")

    def test_ensure_schema_all_exist_skips(self):
        self.mock_schema.getPropertyKey.return_value = object()
        self.mock_schema.getVertexLabel.return_value = object()
        self.mock_schema.getEdgeLabel.return_value = object()
        self.mock_schema.getIndexLabel.return_value = object()
        summary = self.sm.ensure_schema(self._sample_schema())
        self.assertEqual(
            summary,
            {"property_keys": 0, "vertex_labels": 0, "edge_labels": 0, "index_labels": 0},
        )
        self.mock_schema.propertyKey.assert_not_called()
        self.mock_schema.vertexLabel.assert_not_called()
        self.mock_schema.edgeLabel.assert_not_called()
        self.mock_schema.indexLabel.assert_not_called()

    def test_ensure_schema_name_pk_skips_index(self):
        self._absent()
        self.mock_schema.getSchema.return_value = {
            "vertexlabels": [{"name": "movie", "primary_keys": ["title", "name"]}]
        }
        for method in ("propertyKey", "vertexLabel", "edgeLabel", "indexLabel"):
            self._builder(getattr(self.mock_schema, method))
        summary = self.sm.ensure_schema(self._sample_schema())
        # server says movie.name is a PK -> no movie index; person.name is not a
        # PK (input PK=[id]) -> person index still built
        self.assertEqual(summary["index_labels"], 1)
        self.mock_schema.indexLabel.assert_called_once_with("personByName")

    def test_ensure_schema_server_schema_failure_falls_back_to_input_pk(self):
        self._absent()
        self.mock_schema.getSchema.side_effect = RuntimeError("server down")
        for method in ("propertyKey", "vertexLabel", "edgeLabel", "indexLabel"):
            self._builder(getattr(self.mock_schema, method))
        summary = self.sm.ensure_schema(self._sample_schema())
        # fallback to input primary_keys: person PK=[id], movie PK=[title]
        # -> both name indexes are still built
        self.assertEqual(summary["index_labels"], 2)

    def test_ensure_schema_benign_index_exception_tolerated(self):
        self._absent()
        for method in ("propertyKey", "vertexLabel", "edgeLabel"):
            self._builder(getattr(self.mock_schema, method))
        index_builder = self._builder(self.mock_schema.indexLabel)
        index_builder.create.side_effect = RuntimeError("No need to build index")
        summary = self.sm.ensure_schema(self._sample_schema())
        self.assertEqual(summary["index_labels"], 0)

    def test_ensure_schema_other_index_exception_reraises(self):
        self._absent()
        for method in ("propertyKey", "vertexLabel", "edgeLabel"):
            self._builder(getattr(self.mock_schema, method))
        index_builder = self._builder(self.mock_schema.indexLabel)
        index_builder.create.side_effect = RuntimeError("real failure")
        with self.assertRaises(RuntimeError):
            self.sm.ensure_schema(self._sample_schema())

    def test_ensure_schema_invalid_property_skipped(self):
        schema = self._sample_schema()
        schema["propertykeys"].append({"name": "bad", "data_type": "NOT_A_TYPE"})
        self._absent()
        for method in ("propertyKey", "vertexLabel", "edgeLabel", "indexLabel"):
            self._builder(getattr(self.mock_schema, method))
        summary = self.sm.ensure_schema(schema)
        self.assertEqual(summary["property_keys"], 2)

    def test_ensure_schema_vertex_without_name_property(self):
        schema = self._sample_schema()
        schema["vertexlabels"].append(
            {"name": "tag", "properties": ["id"], "primary_keys": ["id"]}
        )
        self._absent()
        for method in ("propertyKey", "vertexLabel", "edgeLabel", "indexLabel"):
            self._builder(getattr(self.mock_schema, method))
        summary = self.sm.ensure_schema(schema)
        # tag has no `name` property -> no secondary index for it; person/movie
        # still get theirs (this also covers the no-name branch of the loop)
        self.assertEqual(summary["index_labels"], 2)

    def test_ensure_schema_empty(self):
        self._absent()
        summary = self.sm.ensure_schema({})
        self.assertEqual(
            summary,
            {"property_keys": 0, "vertex_labels": 0, "edge_labels": 0, "index_labels": 0},
        )

    @staticmethod
    def _idx(name, base_value, fields):
        idx = MagicMock()
        idx.name = name
        idx.baseType = "VERTEX_LABEL"
        idx.baseValue = base_value
        idx.fields = fields
        idx.indexType = "SECONDARY"
        return idx

    def test_list_indexes_all(self):
        self.mock_schema.getIndexLabels.return_value = [
            self._idx("entities_name", "entities", ["name"]),
            self._idx("entities_status", "entities", ["status"]),
        ]
        indexes = self.sm.list_indexes()
        self.assertEqual(len(indexes), 2)
        self.assertEqual(indexes[0]["name"], "entities_name")
        self.assertEqual(indexes[0]["fields"], ["name"])
        self.assertEqual(indexes[0]["index_type"], "SECONDARY")

    def test_list_indexes_filtered_by_label(self):
        self.mock_schema.getIndexLabels.return_value = [
            self._idx("entities_name", "entities", ["name"]),
            self._idx("documents_title", "documents", ["title"]),
        ]
        indexes = self.sm.list_indexes(base_label="entities")
        self.assertEqual(len(indexes), 1)
        self.assertEqual(indexes[0]["name"], "entities_name")

    def test_list_indexes_empty_and_error(self):
        self.mock_schema.getIndexLabels.return_value = None
        self.assertEqual(self.sm.list_indexes(), [])
        self.mock_schema.getIndexLabels.side_effect = RequestException("down")
        self.assertEqual(self.sm.list_indexes(), [])

    def test_get_index_info_found_and_missing(self):
        self.mock_schema.getIndexLabel.return_value = self._idx("entities_name", "entities", ["name"])
        info = self.sm.get_index_info("entities_name")
        self.assertEqual(info["base_value"], "entities")
        self.mock_schema.getIndexLabel.return_value = None
        self.assertIsNone(self.sm.get_index_info("nope"))
        self.mock_schema.getIndexLabel.side_effect = RequestException("down")
        self.assertIsNone(self.sm.get_index_info("nope"))

    def test_probe_capabilities_success(self):
        self.mock_schema.getSchema.return_value = {"vertexlabels": [{"name": "x"}]}
        caps = self.sm.probe_capabilities()
        self.assertTrue(caps["graph_reachable"])
        self.assertTrue(caps["schema_readable"])

    def test_probe_capabilities_empty_schema(self):
        self.mock_schema.getSchema.return_value = {}
        caps = self.sm.probe_capabilities()
        self.assertTrue(caps["graph_reachable"])
        self.assertFalse(caps["schema_readable"])

    def test_probe_capabilities_connection_failure(self):
        self.mock_schema.getSchema.side_effect = RequestException("down")
        caps = self.sm.probe_capabilities()
        self.assertFalse(caps["graph_reachable"])
        self.assertFalse(caps["schema_readable"])

    # -- client injection ----------------------------------------------------

    def test_init_with_injected_client_reuses_it(self):
        with patch(
            "hugegraph_llm.operators.hugegraph_op.schema_manager.PyHugeClient"
        ) as mock_client_class:
            sm = SchemaManager("g2", client=self.mock_client)
            mock_client_class.assert_not_called()
            self.assertIs(sm.client, self.mock_client)
            self.assertIs(sm.schema, self.mock_schema)


if __name__ == "__main__":
    unittest.main()
