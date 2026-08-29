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

from hugegraph_llm.operators.graph_op.incremental_merge import (
    DeltaMergeCoordinator,
    EntityMergeInput,
    IncrementalMergePipeline,
    RelationMergeInput,
    SourceIdsConfig,
    SourceIdStrategy,
    SourceIdsManager,
    merge_entity_descriptions,
    vote_entity_type,
)

pytestmark = [pytest.mark.unit]


class TestSourceIdsManager(unittest.TestCase):
    def test_merge_dedup_preserves_order(self):
        mgr = SourceIdsManager()
        merged = mgr.merge(existing_ids=["d1", "d2"], new_ids=["d2", "d3"])
        self.assertEqual(merged, ["d1", "d2", "d3"])

    def test_merge_none_inputs(self):
        mgr = SourceIdsManager()
        self.assertEqual(mgr.merge(), [])
        self.assertEqual(mgr.merge(existing_ids=["d1"]), ["d1"])
        self.assertEqual(mgr.merge(new_ids=["d1"]), ["d1"])

    def test_merge_keep_strategy_truncates_head(self):
        mgr = SourceIdsManager(config=SourceIdsConfig(max_source_ids=2, strategy=SourceIdStrategy.KEEP))
        merged = mgr.merge(existing_ids=["d1", "d2", "d3"], new_ids=["d4"])
        self.assertEqual(merged, ["d1", "d2"])

    def test_merge_fifo_strategy_truncates_tail(self):
        mgr = SourceIdsManager(config=SourceIdsConfig(max_source_ids=2, strategy=SourceIdStrategy.FIFO))
        merged = mgr.merge(existing_ids=["d1", "d2", "d3"], new_ids=["d4"])
        self.assertEqual(merged, ["d3", "d4"])

    def test_remove_ids(self):
        mgr = SourceIdsManager()
        self.assertEqual(mgr.remove(["d1", "d2", "d3"], ["d2"]), ["d1", "d3"])

    def test_merge_unknown_strategy_keeps_all(self):
        # dataclass does not validate the enum; an unknown strategy skips
        # both truncation branches and keeps every id.
        mgr = SourceIdsManager(
            config=SourceIdsConfig(max_source_ids=2, strategy="bogus")  # type: ignore[arg-type]
        )
        merged = mgr.merge(existing_ids=["d1", "d2", "d3"])
        self.assertEqual(merged, ["d1", "d2", "d3"])


class TestVoteEntityType(unittest.TestCase):
    def test_votes_existing_and_new(self):
        self.assertEqual(vote_entity_type(existing_type="Person", new_types=["Person", "Person"]), "Person")

    def test_votes_majority_wins(self):
        self.assertEqual(vote_entity_type(existing_type="Person", new_types=["Company", "Company"]), "Company")

    def test_all_types_overrides(self):
        self.assertEqual(
            vote_entity_type(existing_type="X", new_types=["Y"], all_types=["A", "A", "B"]),
            "A",
        )

    def test_empty_types_returns_empty(self):
        self.assertEqual(vote_entity_type(), "")
        self.assertEqual(vote_entity_type(existing_type="", new_types=["", None]), "")


class TestMergeEntityDescriptions(unittest.TestCase):
    def test_uses_injected_merger(self):
        mock_merger = MagicMock()
        mock_merger.merge.return_value = "merged"
        result = merge_entity_descriptions(
            existing_description="old",
            new_descriptions=["new"],
            merger=mock_merger,
        )
        self.assertEqual(result, "merged")
        mock_merger.merge.assert_called_once_with(["old", "new"])

    def test_single_description_no_llm(self):
        # DescriptionMerger Level 1: single description returned directly
        self.assertEqual(merge_entity_descriptions(new_descriptions=["only"]), "only")

    def test_empty_descriptions(self):
        self.assertEqual(merge_entity_descriptions(), "")

    def test_existing_only(self):
        self.assertEqual(merge_entity_descriptions(existing_description="old"), "old")


class TestIncrementalMergePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = IncrementalMergePipeline()

    def test_merge_entity_changed(self):
        out = self.pipeline.merge_entity(
            EntityMergeInput(
                entity_name="e1",
                existing_type="Person",
                existing_description="old desc",
                existing_source_ids=["d1"],
                new_type="Person",
                new_description="new desc",
                new_source_ids=["d2"],
            )
        )
        self.assertEqual(out.entity_name, "e1")
        self.assertEqual(out.merged_type, "Person")
        self.assertIn("old desc", out.merged_description)
        self.assertEqual(out.merged_source_ids, ["d1", "d2"])
        self.assertTrue(out.changed)

    def test_merge_entity_unchanged(self):
        out = self.pipeline.merge_entity(
            EntityMergeInput(
                entity_name="e1",
                existing_type="Person",
                existing_description="desc",
                existing_source_ids=["d1"],
            )
        )
        self.assertFalse(out.changed)

    def test_merge_relation_changed(self):
        out = self.pipeline.merge_relation(
            RelationMergeInput(
                source_entity="a",
                target_entity="b",
                relation_label="knows",
                existing_description="old",
                existing_source_ids=["d1"],
                new_description="new",
                new_source_ids=["d2"],
            )
        )
        self.assertEqual(out.source_entity, "a")
        self.assertEqual(out.target_entity, "b")
        self.assertEqual(out.relation_label, "knows")
        self.assertEqual(out.merged_source_ids, ["d1", "d2"])
        self.assertTrue(out.changed)

    def test_merge_relation_unchanged(self):
        out = self.pipeline.merge_relation(
            RelationMergeInput(
                source_entity="a",
                target_entity="b",
                relation_label="knows",
                existing_description="desc",
                existing_source_ids=["d1"],
            )
        )
        self.assertFalse(out.changed)


class TestDeltaMergeCoordinator(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_delta = MagicMock()
        self.mock_provider.child.return_value = self.mock_delta
        self.coord = DeltaMergeCoordinator(provider=self.mock_provider)

    def test_init_uses_provider_child(self):
        self.mock_provider.child.assert_called_once_with("delta")
        self.assertIs(self.coord.provider, self.mock_provider)
        self.assertIs(self.coord.delta_provider, self.mock_delta)

    @patch("hugegraph_llm.operators.hugegraph_op.kg_table_provider.KgTableProvider")
    def test_init_builds_provider_from_client(self, mock_provider_cls):
        mock_provider_cls.return_value.child.return_value = MagicMock()
        client = object()
        DeltaMergeCoordinator(client=client)
        mock_provider_cls.assert_called_once()
        _, kwargs = mock_provider_cls.call_args
        self.assertIs(kwargs["client"], client)

    def test_table_registers_schema_and_returns_delta(self):
        mock_table = MagicMock()
        self.mock_delta.table.return_value = mock_table
        table = self.coord.table("entities", schema={"name": "TEXT"})
        self.assertIs(table, mock_table)
        self.mock_delta.table.assert_called_once_with("entities", schema={"name": "TEXT"})
        self.assertEqual(self.coord.pending_tables(), ["entities"])

    def test_table_keeps_first_schema(self):
        self.coord.table("entities", schema={"name": "TEXT"})
        self.coord.table("entities", schema={"other": "INT"})
        # pending schema registry keeps the first registration
        self.assertEqual(self.coord.pending_tables(), ["entities"])

    def test_write_rows_delegates_to_delta(self):
        mock_table = MagicMock()
        mock_table.upsert_many.return_value = 3
        self.mock_delta.table.return_value = mock_table
        count = self.coord.write_rows("entities", [{"id": "1"}, {"id": "2"}, {"id": "3"}])
        self.assertEqual(count, 3)
        mock_table.upsert_many.assert_called_once_with([{"id": "1"}, {"id": "2"}, {"id": "3"}])

    def test_list_delta(self):
        mock_table = MagicMock()
        mock_table.list_rows.return_value = [{"id": "1"}]
        self.mock_delta.table.return_value = mock_table
        self.assertEqual(self.coord.list_delta("entities"), [{"id": "1"}])

    def test_has_delta(self):
        mock_table = MagicMock()
        mock_table.has.return_value = True
        self.mock_delta.table.return_value = mock_table
        self.assertTrue(self.coord.has_delta("entities", "e1"))
        mock_table.has.assert_called_once_with("e1")

    def test_commit_single_table(self):
        self.coord.table("entities", schema={"name": "TEXT"})
        self.coord.table("documents", schema={"title": "TEXT"})
        delta_entities = MagicMock()
        delta_entities.list_rows.return_value = [{"id": "1", "name": "a"}]
        main_entities = MagicMock()
        self.mock_delta.table.side_effect = lambda name, schema=None: delta_entities if name == "entities" else MagicMock()
        self.mock_provider.table.side_effect = lambda name, schema=None: main_entities if name == "entities" else MagicMock()

        merged = self.coord.commit("entities")

        self.assertEqual(merged, 1)
        main_entities.upsert.assert_called_once_with({"id": "1", "name": "a"})
        delta_entities.clear.assert_called_once()

    def test_commit_all_tables(self):
        self.coord.table("entities", schema={"name": "TEXT"})
        self.coord.table("documents", schema={"title": "TEXT"})
        # every registered table commits: 2 rows total
        delta_entities = MagicMock()
        delta_entities.list_rows.return_value = [{"id": "1", "name": "a"}]
        delta_documents = MagicMock()
        delta_documents.list_rows.return_value = [{"id": "d1", "title": "t"}]
        self.mock_delta.table.side_effect = lambda name, schema=None: (
            delta_entities if name == "entities" else delta_documents
        )
        main_entities = MagicMock()
        main_documents = MagicMock()
        self.mock_provider.table.side_effect = lambda name, schema=None: (
            main_entities if name == "entities" else main_documents
        )

        merged = self.coord.commit()

        self.assertEqual(merged, 2)
        main_entities.upsert.assert_called_once_with({"id": "1", "name": "a"})
        main_documents.upsert.assert_called_once_with({"id": "d1", "title": "t"})
        delta_entities.clear.assert_called_once()
        delta_documents.clear.assert_called_once()

    def test_rollback_single_table(self):
        self.coord.table("entities", schema={"name": "TEXT"})
        delta_entities = MagicMock()
        delta_entities.clear.return_value = 2
        self.mock_delta.table.return_value = delta_entities

        cleared = self.coord.rollback("entities")

        self.assertEqual(cleared, 2)
        delta_entities.clear.assert_called_once()

    def test_rollback_all_tables(self):
        self.coord.table("entities", schema={"name": "TEXT"})
        self.coord.table("documents", schema={"title": "TEXT"})
        delta_entities = MagicMock()
        delta_entities.clear.return_value = 1
        delta_documents = MagicMock()
        delta_documents.clear.return_value = 3
        self.mock_delta.table.side_effect = lambda name, schema=None: (
            delta_entities if name == "entities" else delta_documents
        )

        cleared = self.coord.rollback()

        self.assertEqual(cleared, 4)

    def test_pending_tables_empty(self):
        self.assertEqual(self.coord.pending_tables(), [])


if __name__ == "__main__":
    unittest.main()
