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

import unittest
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.operators.hugegraph_op.kg_table import KgTable
from hugegraph_llm.operators.hugegraph_op.kg_table_provider import KgTableProvider

pytestmark = [pytest.mark.unit]


class TestKgTableProvider(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.schema.return_value = MagicMock()

    def test_init_with_injected_client_reuses_it(self):
        with patch(
            "hugegraph_llm.operators.hugegraph_op.kg_table_provider.SchemaManager"
        ) as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm.client = self.mock_client
            mock_sm.graph_name = "g"
            mock_sm_class.return_value = mock_sm

            provider = KgTableProvider(self.mock_client, graph_name="g")

            mock_sm_class.assert_called_once_with("g", client=self.mock_client)
            self.assertIs(provider.client, self.mock_client)
            self.assertEqual(provider.namespace, "")

    @patch("hugegraph_llm.operators.hugegraph_op.kg_table_provider.huge_settings")
    @patch("hugegraph_llm.operators.hugegraph_op.kg_table_provider.SchemaManager")
    def test_init_uses_global_settings_without_client(self, mock_sm_class, mock_settings):
        mock_settings.graph_name = "default_graph"
        mock_sm = MagicMock()
        mock_sm.client = MagicMock()
        mock_sm_class.return_value = mock_sm

        provider = KgTableProvider()

        mock_sm_class.assert_called_once_with("default_graph", connection=None)
        self.assertIs(provider.client, mock_sm.client)

    @patch("hugegraph_llm.operators.hugegraph_op.kg_table_provider.huge_settings")
    @patch("hugegraph_llm.operators.hugegraph_op.kg_table_provider.SchemaManager")
    def test_init_with_explicit_connection(self, mock_sm_class, mock_settings):
        mock_settings.graph_name = "default_graph"
        mock_sm = MagicMock()
        mock_sm.client = MagicMock()
        mock_sm_class.return_value = mock_sm

        KgTableProvider(connection={"url": "10.0.0.1:8080", "user": "admin"})

        mock_sm_class.assert_called_once_with(
            "default_graph", connection={"url": "10.0.0.1:8080", "user": "admin"}
        )

    def test_table_creates_and_caches(self):
        provider = KgTableProvider(self.mock_client)
        t1 = provider.table("entities", schema={"name": "TEXT"})
        self.assertIsInstance(t1, KgTable)
        self.assertEqual(t1.label, "entities")
        # cached: same object for the same namespaced name
        self.assertIs(provider.table("entities"), t1)
        # different table name -> new instance
        self.assertIsNot(provider.table("documents"), t1)

    def test_table_namespace_prefixes_label(self):
        provider = KgTableProvider(self.mock_client, namespace="delta")
        table = provider.table("entities")
        self.assertEqual(table.table_name, "delta_entities")
        self.assertEqual(table.label, "delta_entities")

    def test_table_forwards_kwargs(self):
        provider = KgTableProvider(self.mock_client)
        table = provider.table("entities", page_size=7, label_prefix="kg_")
        self.assertEqual(table._page_size, 7)
        self.assertEqual(table.label, "kg_entities")

    def test_table_uses_namespace_plus_label_prefix(self):
        provider = KgTableProvider(self.mock_client, namespace="delta", label_prefix="kg_")
        table = provider.table("entities")
        self.assertEqual(table.table_name, "delta_entities")
        self.assertEqual(table.label, "kg_delta_entities")

    def test_child_combines_namespace_and_shares_client(self):
        provider = KgTableProvider(self.mock_client, namespace="main")
        child = provider.child("delta")
        self.assertEqual(child.namespace, "main_delta")
        self.assertIs(child.client, provider.client)
        table = child.table("entities")
        self.assertEqual(table.table_name, "main_delta_entities")

    def test_child_without_parent_namespace(self):
        provider = KgTableProvider(self.mock_client)
        child = provider.child("delta")
        self.assertEqual(child.namespace, "delta")
        self.assertIs(child.client, provider.client)

    def test_namespaced_helper(self):
        provider = KgTableProvider(self.mock_client)
        self.assertEqual(provider._namespaced("entities"), "entities")
        ns_provider = KgTableProvider(self.mock_client, namespace="delta")
        self.assertEqual(ns_provider._namespaced("entities"), "delta_entities")

    def test_namespaced_tables_returns_copy(self):
        provider = KgTableProvider(self.mock_client)
        provider.table("entities")
        provider.table("documents")
        snapshot = provider.namespaced_tables()
        self.assertEqual(set(snapshot.keys()), {"entities", "documents"})
        self.assertIs(snapshot["entities"], provider.table("entities"))


if __name__ == "__main__":
    unittest.main()
