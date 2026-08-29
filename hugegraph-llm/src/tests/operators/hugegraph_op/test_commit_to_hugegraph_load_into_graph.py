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

# pylint: disable=protected-access,no-member
import unittest
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import Commit2Graph
from hugegraph_llm.operators.llm_op.property_graph_extract import PropertyGraphExtract

pytestmark = [pytest.mark.unit]

# FIXME: cover failure branches where vertex type errors stop edge writes and
# surface an explicit import failure.


class TestCommit2GraphLoadIntoGraph(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_schema = MagicMock()
        self.mock_client.schema.return_value = self.mock_schema

        with patch(
            "hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.PyHugeClient", return_value=self.mock_client
        ):
            self.commit2graph = Commit2Graph()

        self.schema = {
            "propertykeys": [
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "age", "data_type": "INT", "cardinality": "SINGLE"},
                {"name": "title", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "year", "data_type": "INT", "cardinality": "SINGLE"},
                {"name": "role", "data_type": "TEXT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["name", "age"],
                    "primary_keys": ["name"],
                    "nullable_keys": ["age"],
                    "id_strategy": "PRIMARY_KEY",
                },
                {
                    "id": 2,
                    "name": "movie",
                    "properties": ["title", "year"],
                    "primary_keys": ["title"],
                    "nullable_keys": ["year"],
                    "id_strategy": "PRIMARY_KEY",
                },
            ],
            "edgelabels": [
                {"name": "acted_in", "properties": ["role"], "source_label": "person", "target_label": "movie"}
            ],
        }

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._check_property_data_type")
    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph(self, mock_handle_graph_creation, mock_check_property_data_type):
        """Test load_into_graph method."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")
        mock_check_property_data_type.return_value = True

        vertices = [
            {"label": "person", "properties": {"name": "Tom Hanks", "age": 67}},
            {"label": "movie", "properties": {"title": "Forrest Gump", "year": 1994}},
        ]
        edges = [
            {
                "label": "acted_in",
                "properties": {"role": "Forrest Gump"},
                "outV": "person:Tom Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        self.commit2graph.load_into_graph(vertices, edges, self.schema)

        self.assertEqual(mock_handle_graph_creation.call_count, 3)

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_with_data_type_validation_success(self, mock_handle_graph_creation):
        """Test load_into_graph method with successful data type validation."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [
            {"label": "person", "properties": {"name": "Tom Hanks", "age": 67}},
            {"label": "movie", "properties": {"title": "Forrest Gump", "year": 1994}},
        ]
        edges = [
            {
                "label": "acted_in",
                "properties": {"role": "Forrest Gump"},
                "outV": "person:Tom Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        self.commit2graph.load_into_graph(vertices, edges, self.schema)

        self.assertEqual(mock_handle_graph_creation.call_count, 3)

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_maps_llm_vertex_ids_to_created_vertex_ids(self, mock_handle_graph_creation):
        """Test edges use server-created vertex ids when LLM ids differ."""
        mock_handle_graph_creation.side_effect = [
            MagicMock(id="1:Tom Hanks"),
            MagicMock(id="2:Forrest Gump"),
            MagicMock(id="edge_id"),
        ]

        vertices = [
            {"id": "person:Tom Hanks", "label": "person", "properties": {"name": "Tom Hanks", "age": 67}},
            {"id": "movie:Forrest Gump", "label": "movie", "properties": {"title": "Forrest Gump", "year": 1994}},
        ]
        edges = [
            {
                "label": "acted_in",
                "properties": {"role": "Forrest Gump"},
                "outV": "person:Tom Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        self.commit2graph.load_into_graph(vertices, edges, self.schema)

        self.assertEqual(vertices[0]["id"], "1:Tom Hanks")
        self.assertEqual(vertices[1]["id"], "2:Forrest Gump")
        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addEdge,
            "acted_in",
            "1:Tom Hanks",
            "2:Forrest Gump",
            {"role": "Forrest Gump"},
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_maps_multiple_primary_keys_to_created_vertex_ids(self, mock_handle_graph_creation):
        mock_handle_graph_creation.side_effect = [
            MagicMock(id="1:Tom!Hanks"),
            MagicMock(id="2:Forrest Gump"),
            MagicMock(id="edge_id"),
        ]
        schema = {
            "propertykeys": [
                {"name": "first", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "last", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "title", "data_type": "TEXT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["first", "last"],
                    "primary_keys": ["first", "last"],
                    "nullable_keys": [],
                    "id_strategy": "PRIMARY_KEY",
                },
                {
                    "id": 2,
                    "name": "movie",
                    "properties": ["title"],
                    "primary_keys": ["title"],
                    "nullable_keys": [],
                    "id_strategy": "PRIMARY_KEY",
                },
            ],
            "edgelabels": [{"name": "acted_in", "properties": [], "source_label": "person", "target_label": "movie"}],
        }
        vertices = [
            {"label": "person", "properties": {"first": "Tom", "last": "Hanks"}},
            {"label": "movie", "properties": {"title": "Forrest Gump"}},
        ]
        edges = [
            {
                "label": "acted_in",
                "properties": {},
                "outV": "person:Tom!Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        self.commit2graph.load_into_graph(vertices, edges, schema)

        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addEdge,
            "acted_in",
            "1:Tom!Hanks",
            "2:Forrest Gump",
            {},
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_uses_explicit_customize_string_ids(self, mock_handle_graph_creation):
        """Test custom string ids are passed to HugeGraph when schema requires them."""
        # The authoritative id strategy lives on the SERVER (labels may have been
        # created earlier with a different strategy), so mock the server schema.
        self.mock_schema.getSchema.return_value = {
            "vertexlabels": [
                {"name": "person", "id_strategy": "CUSTOMIZE_STRING", "properties": ["name"]},
                {"name": "movie", "id_strategy": "CUSTOMIZE_STRING", "properties": ["title"]},
            ]
        }
        mock_handle_graph_creation.side_effect = [
            MagicMock(id="Tom Hanks"),
            MagicMock(id="Forrest Gump"),
            MagicMock(id="edge_id"),
        ]
        schema = {
            "propertykeys": [
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "title", "data_type": "TEXT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "id": 7,
                    "name": "person",
                    "id_strategy": "CUSTOMIZE_STRING",
                    "primary_keys": ["name"],
                    "properties": ["name"],
                    "nullable_keys": [],
                },
                {
                    "id": 8,
                    "name": "movie",
                    "id_strategy": "CUSTOMIZE_STRING",
                    "primary_keys": ["title"],
                    "properties": ["title"],
                    "nullable_keys": [],
                },
            ],
            "edgelabels": [{"name": "acted_in", "properties": [], "source_label": "person", "target_label": "movie"}],
        }
        vertices = [
            {"id": "Tom Hanks", "label": "person", "properties": {"name": "Tom Hanks"}},
            {"id": "Forrest Gump", "label": "movie", "properties": {"title": "Forrest Gump"}},
        ]
        edges = [{"label": "acted_in", "properties": {}, "outV": "Tom Hanks", "inV": "Forrest Gump"}]

        self.commit2graph.load_into_graph(vertices, edges, schema)

        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addVertex,
            "person",
            {"name": "Tom Hanks"},
            id="Tom Hanks",
        )
        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addVertex,
            "movie",
            {"title": "Forrest Gump"},
            id="Forrest Gump",
        )
        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addEdge,
            "acted_in",
            "Tom Hanks",
            "Forrest Gump",
            {},
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_accepts_normalized_extraction_without_item_type(self, mock_handle_graph_creation):
        """Test normalized LLM output without type fields can be committed."""
        mock_handle_graph_creation.side_effect = [
            MagicMock(id="1:Tom Hanks"),
            MagicMock(id="2:Forrest Gump"),
            MagicMock(id="edge_id"),
        ]
        llm_output = """{
            "vertices": [
            {
                "id": "person:Tom Hanks",
                "label": "person",
                "properties": {
                    "name": "Tom Hanks",
                    "age": 67
                }
            },
            {
                "id": "movie:Forrest Gump",
                "label": "movie",
                "properties": {
                    "title": "Forrest Gump",
                    "year": 1994
                }
            }
            ],
            "edges": [
            {
                "label": "acted_in",
                "outV": "person:Tom Hanks",
                "outVLabel": "person",
                "inV": "movie:Forrest Gump",
                "inVLabel": "movie",
                "properties": {
                    "role": "Forrest Gump"
                }
            }
            ]
        }"""

        items = PropertyGraphExtract(llm=MagicMock())._extract_and_filter_label(self.schema, llm_output)
        vertices = [item for item in items if item["type"] == "vertex"]
        edges = [item for item in items if item["type"] == "edge"]
        self.assertEqual(edges[0]["outV"], "1:Tom Hanks")
        self.assertEqual(edges[0]["inV"], "2:Forrest Gump")

        self.commit2graph.load_into_graph(vertices, edges, self.schema)

        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addEdge,
            "acted_in",
            "1:Tom Hanks",
            "2:Forrest Gump",
            {"role": "Forrest Gump"},
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_property_graph_extract_run_preserves_typed_values_for_commit(self, mock_handle_graph_creation):
        """Test extracted typed properties survive the full extraction-to-commit path."""
        mock_handle_graph_creation.side_effect = [
            MagicMock(id="1:Tom Hanks"),
            MagicMock(id="2:Forrest Gump"),
            MagicMock(id="edge_id"),
        ]
        llm = MagicMock()
        llm.generate.return_value = """{
            "vertices": [
            {
                "label": "person",
                "properties": {
                    "name": "Tom Hanks",
                    "age": 67
                }
            },
            {
                "label": "movie",
                "properties": {
                    "title": "Forrest Gump",
                    "year": 1994
                }
            }
            ],
            "edges": [
            {
                "label": "acted_in",
                "properties": {
                    "role": "Forrest Gump"
                },
                "source": {
                    "label": "person",
                    "properties": {
                        "name": "Tom Hanks"
                    }
                },
                "target": {
                    "label": "movie",
                    "properties": {
                        "title": "Forrest Gump"
                    }
                }
            }
            ]
        }"""
        context = PropertyGraphExtract(llm=llm, example_prompt=None).run(
            {
                "schema": self.schema,
                "chunks": ["Tom Hanks acted in Forrest Gump."],
            }
        )

        self.assertIsInstance(context["vertices"][0]["properties"]["age"], int)
        self.assertIsInstance(context["vertices"][1]["properties"]["year"], int)

        self.commit2graph.load_into_graph(context["vertices"], context["edges"], self.schema)

        self.assertEqual(mock_handle_graph_creation.call_count, 3)

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_raises_explicit_error_when_vertex_creation_fails(self, mock_handle_graph_creation):
        """Test failed vertex creation is reported before edge creation."""
        mock_handle_graph_creation.return_value = None

        vertices = [{"label": "person", "properties": {"name": "Tom Hanks", "age": 67}}]
        edges = [
            {
                "label": "acted_in",
                "properties": {"role": "Forrest Gump"},
                "outV": "person:Tom Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        with self.assertRaisesRegex(ValueError, "Failed to create vertex"):
            self.commit2graph.load_into_graph(vertices, edges, self.schema)

        mock_handle_graph_creation.assert_called_once()

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_with_data_type_validation_failure(self, mock_handle_graph_creation):
        """Test load_into_graph method with data type validation failure."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [
            {"label": "person", "properties": {"name": "Tom Hanks", "age": "67"}},
            {"label": "movie", "properties": {"title": "Forrest Gump", "year": "1994"}},
        ]
        edges = [
            {
                "label": "acted_in",
                "properties": {"role": "Forrest Gump"},
                "outV": "person:Tom Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        self.commit2graph.load_into_graph(vertices, edges, self.schema)

        self.assertEqual(mock_handle_graph_creation.call_count, 1)

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_server_schema_fetch_failure_uses_input_schema(self, mock_handle_graph_creation):
        """Server schema fetch failure must degrade to the input schema props."""
        self.mock_schema.getSchema.side_effect = RuntimeError("server down")
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [{"label": "person", "properties": {"name": "Tom Hanks", "age": 67}}]
        self.commit2graph.load_into_graph(vertices, [], self.schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom Hanks", "age": 67}
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_unknown_vertex_label_skipped(self, mock_handle_graph_creation):
        """Vertices whose label is not in the schema are skipped."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [
            {"label": "person", "properties": {"name": "Tom Hanks", "age": 67}},
            {"label": "alien", "properties": {"name": "ET"}},
        ]
        self.commit2graph.load_into_graph(vertices, [], self.schema)

        mock_handle_graph_creation.assert_called_once()
        mock_handle_graph_creation.assert_any_call(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom Hanks", "age": 67}
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_single_pk_missing_skips_vertex(self, mock_handle_graph_creation):
        """A single missing primary key skips the vertex."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [{"label": "person", "properties": {"age": 67}}]  # name (PK) missing
        self.commit2graph.load_into_graph(vertices, [], self.schema)

        mock_handle_graph_creation.assert_not_called()

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_multi_pk_missing_uses_default(self, mock_handle_graph_creation):
        """With multiple primary keys, a missing one is filled with a default."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")
        schema = {
            "propertykeys": [
                {"name": "id", "data_type": "INT", "cardinality": "SINGLE"},
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["id", "name"],
                    "primary_keys": ["id", "name"],
                    "nullable_keys": [],
                    "id_strategy": "PRIMARY_KEY",
                }
            ],
            "edgelabels": [],
        }
        vertices = [{"label": "person", "properties": {"name": "Tom"}}]  # id missing -> default 0

        self.commit2graph.load_into_graph(vertices, [], schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom", "id": 0}
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_strips_extra_properties(self, mock_handle_graph_creation):
        """Properties not present in the SERVER schema are stripped."""
        self.mock_schema.getSchema.return_value = {
            "vertexlabels": [{"name": "person", "properties": ["name"], "id_strategy": "PRIMARY_KEY"}]
        }
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [{"label": "person", "properties": {"name": "Tom", "hobby": "acting"}}]
        self.commit2graph.load_into_graph(vertices, [], self.schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom"}
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_sets_default_for_non_nullable_property(self, mock_handle_graph_creation):
        """A missing non-nullable property is filled with its type default."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")
        # schema: person nullable_keys=["name"], so "age" (INT) must be set
        schema = {
            "propertykeys": [
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
                {"name": "age", "data_type": "INT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["name", "age"],
                    "primary_keys": ["name"],
                    "nullable_keys": ["name"],
                    "id_strategy": "PRIMARY_KEY",
                }
            ],
            "edgelabels": [],
        }
        vertices = [{"label": "person", "properties": {"name": "Tom"}}]  # age missing -> default 0

        self.commit2graph.load_into_graph(vertices, [], schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom", "age": 0}
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_customize_string_derives_id_from_pk(self, mock_handle_graph_creation):
        """CUSTOMIZE_STRING without explicit id derives one from primary keys."""
        self.mock_schema.getSchema.return_value = {
            "vertexlabels": [
                {"name": "person", "id_strategy": "CUSTOMIZE_STRING", "properties": ["name"]}
            ]
        }
        mock_handle_graph_creation.return_value = MagicMock(id="person:Tom Hanks")

        vertices = [{"label": "person", "properties": {"name": "Tom Hanks"}}]  # no explicit id
        self.commit2graph.load_into_graph(vertices, [], self.schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex,
            "person",
            {"name": "Tom Hanks"},
            id="person:Tom Hanks",
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_customize_string_no_id_no_pk_skips(self, mock_handle_graph_creation):
        """CUSTOMIZE_STRING with neither id nor primary keys skips the vertex."""
        self.mock_schema.getSchema.return_value = {
            "vertexlabels": [
                {"name": "person", "id_strategy": "CUSTOMIZE_STRING", "properties": ["name"]}
            ]
        }
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")
        schema = {
            "propertykeys": [{"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"}],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["name"],
                    "primary_keys": [],
                    "nullable_keys": [],
                    "id_strategy": "CUSTOMIZE_STRING",
                }
            ],
            "edgelabels": [],
        }

        vertices = [{"label": "person", "properties": {"name": "Tom"}}]  # no id, no PK
        self.commit2graph.load_into_graph(vertices, [], schema)

        mock_handle_graph_creation.assert_not_called()

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_unknown_edge_label_skipped(self, mock_handle_graph_creation):
        """Edges whose label is not in the schema are skipped."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")

        vertices = [
            {"label": "person", "properties": {"name": "Tom Hanks", "age": 67}},
            {"label": "movie", "properties": {"title": "Forrest Gump", "year": 1994}},
        ]
        edges = [
            {
                "label": "directed_by",
                "properties": {},
                "outV": "person:Tom Hanks",
                "inV": "movie:Forrest Gump",
            }
        ]

        self.commit2graph.load_into_graph(vertices, edges, self.schema)

        # only the two vertices are created, the unknown edge is skipped
        self.assertEqual(mock_handle_graph_creation.call_count, 2)

    # -- _create_provenance_links ---------------------------------------------

    def test_create_provenance_links_no_metadata_returns_zero(self):
        self.assertEqual(self.commit2graph._create_provenance_links({}), 0)

    def test_create_provenance_links_no_vertices_returns_zero(self):
        data = {"chunk_metadata": [{"doc_name": "d1", "text": "hello"}], "vertices": []}
        self.assertEqual(self.commit2graph._create_provenance_links(data), 0)

    @patch("hugegraph_llm.operators.hugegraph_op.provenance_manager.ProvenanceManager")
    def test_create_provenance_links_links_matching_entities(self, mock_pm_class):
        mock_pm = MagicMock()
        mock_pm.create_document.return_value = "doc1"
        mock_pm.create_chunk.return_value = "chunk0"
        mock_pm.link_entity_to_chunk.return_value = True
        mock_pm_class.return_value = mock_pm

        data = {
            "chunk_metadata": [
                {"doc_name": "d1", "doc_source": "src", "text": "Tom Hanks stars in the movie"}
            ],
            "vertices": [
                {"id": "v1", "properties": {"name": "Tom Hanks"}},
                {"id": "v2", "properties": {"name": "Nobody"}},
            ],
        }

        count = self.commit2graph._create_provenance_links(data)

        mock_pm.init_schema.assert_called_once_with()
        mock_pm.create_document.assert_called_once_with("d1", "src")
        mock_pm.create_chunk.assert_called_once_with("doc1", "Tom Hanks stars in the movie", 0)
        # only v1's name appears in the chunk text
        self.assertEqual(count, 1)

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_multi_pk_missing_list_defaults_to_empty_list(
        self, mock_handle_graph_creation
    ):
        """A missing LIST-cardinality primary key is filled with an empty list."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")
        schema = {
            "propertykeys": [
                {"name": "tags", "data_type": "TEXT", "cardinality": "LIST"},
                {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
            ],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["tags", "name"],
                    "primary_keys": ["tags", "name"],
                    "nullable_keys": [],
                    "id_strategy": "PRIMARY_KEY",
                }
            ],
            "edgelabels": [],
        }
        vertices = [{"label": "person", "properties": {"name": "Tom"}}]  # tags missing -> []

        self.commit2graph.load_into_graph(vertices, [], schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom", "tags": []}
        )

    @patch("hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph.Commit2Graph._handle_graph_creation")
    def test_load_into_graph_no_mapping_id(self, mock_handle_graph_creation):
        """Vertices without id or primary keys create with no mapping entry."""
        mock_handle_graph_creation.return_value = MagicMock(id="vertex_id")
        schema = {
            "propertykeys": [{"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"}],
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "properties": ["name"],
                    "primary_keys": [],
                    "nullable_keys": [],
                    "id_strategy": "PRIMARY_KEY",
                }
            ],
            "edgelabels": [],
        }
        vertices = [{"label": "person", "properties": {"name": "Tom"}}]  # no id, no PK

        self.commit2graph.load_into_graph(vertices, [], schema)

        mock_handle_graph_creation.assert_called_once_with(
            self.commit2graph.client.graph().addVertex, "person", {"name": "Tom"}
        )
        self.assertEqual(vertices[0]["id"], "vertex_id")

    @patch("hugegraph_llm.operators.hugegraph_op.provenance_manager.ProvenanceManager")
    def test_create_provenance_links_skips_when_link_fails(self, mock_pm_class):
        mock_pm = MagicMock()
        mock_pm.create_document.return_value = "doc1"
        mock_pm.create_chunk.return_value = "chunk0"
        mock_pm.link_entity_to_chunk.return_value = False  # server refuses the link
        mock_pm_class.return_value = mock_pm

        data = {
            "chunk_metadata": [{"doc_name": "d1", "text": "Tom Hanks stars"}],
            "vertices": [{"id": "v1", "properties": {"name": "Tom Hanks"}}],
        }

        count = self.commit2graph._create_provenance_links(data)

        mock_pm.link_entity_to_chunk.assert_called_once_with("v1", "chunk0")
        self.assertEqual(count, 0)

    @patch("hugegraph_llm.operators.hugegraph_op.provenance_manager.ProvenanceManager")
    def test_create_provenance_links_exception_returns_zero(self, mock_pm_class):
        mock_pm_class.side_effect = RuntimeError("provenance unavailable")
        data = {
            "chunk_metadata": [{"doc_name": "d1", "text": "hello"}],
            "vertices": [{"id": "v1", "properties": {"name": "x"}}],
        }
        self.assertEqual(self.commit2graph._create_provenance_links(data), 0)
