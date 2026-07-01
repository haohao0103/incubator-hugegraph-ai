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

"""Tests for document deletion + rebuild pipeline.

Inspired by LightRAG's ``adelete_by_doc_id`` test coverage.
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch

from hugegraph_llm.operators.graph_op.doc_deletion import (
    DeletionStatus,
    DeletionResult,
    DeletionConfig,
    DependencyAnalysis,
    DocumentDeletionPipeline,
    subtract_source_ids,
    analyze_graph_dependencies,
    GRAPH_FIELD_SEP,
)
from hugegraph_llm.operators.graph_op.storage_interfaces import (
    DocStatus,
    DocStatusRecord,
    InMemoryDocStatusStorage,
)


# ═══════════════════════════════════════════════════════════════════════
# subtract_source_ids tests
# ═══════════════════════════════════════════════════════════════════════


class TestSubtractSourceIds:
    """Tests for source ID subtraction logic."""

    def test_basic_subtraction(self):
        existing = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
        to_remove = ["chunk_2", "chunk_4"]
        result = subtract_source_ids(existing, to_remove)
        assert result == ["chunk_1", "chunk_3"]

    def test_remove_all(self):
        existing = ["chunk_1", "chunk_2"]
        to_remove = ["chunk_1", "chunk_2"]
        result = subtract_source_ids(existing, to_remove)
        assert result == []

    def test_remove_none(self):
        existing = ["chunk_1", "chunk_2"]
        to_remove = ["chunk_5"]
        result = subtract_source_ids(existing, to_remove)
        assert result == ["chunk_1", "chunk_2"]

    def test_preserves_order(self):
        existing = ["c3", "c1", "c5", "c2", "c4"]
        to_remove = ["c1", "c4"]
        result = subtract_source_ids(existing, to_remove)
        assert result == ["c3", "c5", "c2"]

    def test_empty_existing(self):
        result = subtract_source_ids([], ["chunk_1"])
        assert result == []

    def test_empty_to_remove(self):
        existing = ["chunk_1", "chunk_2"]
        result = subtract_source_ids(existing, [])
        assert result == ["chunk_1", "chunk_2"]

    def test_both_empty(self):
        result = subtract_source_ids([], [])
        assert result == []

    def test_duplicates_in_existing(self):
        existing = ["chunk_1", "chunk_1", "chunk_2"]
        to_remove = ["chunk_1"]
        result = subtract_source_ids(existing, to_remove)
        assert result == ["chunk_2"]

    def test_duplicates_in_to_remove(self):
        existing = ["chunk_1", "chunk_2"]
        to_remove = ["chunk_1", "chunk_1"]
        result = subtract_source_ids(existing, to_remove)
        assert result == ["chunk_2"]


# ═══════════════════════════════════════════════════════════════════════
# analyze_graph_dependencies tests
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeDependencies:
    """Tests for dependency analysis logic."""

    def test_entity_exclusive_to_doc(self):
        """Entity only referenced by deleted doc → DELETE."""
        entity_map = {"Alice": ["doc1_chunk1", "doc1_chunk2"]}
        relation_map = {}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_chunk1", "doc1_chunk2"]
        )
        assert "Alice" in analysis.entities_to_delete
        assert len(analysis.entities_to_rebuild) == 0

    def test_entity_shared_across_docs(self):
        """Entity referenced by multiple docs → REBUILD."""
        entity_map = {"Alice": ["doc1_chunk1", "doc2_chunk3"]}
        relation_map = {}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_chunk1"]
        )
        assert "Alice" not in analysis.entities_to_delete
        assert "Alice" in analysis.entities_to_rebuild
        assert analysis.entities_to_rebuild["Alice"] == ["doc2_chunk3"]

    def test_entity_not_affected(self):
        """Entity not referencing any deleted chunks → UNTOUCHED."""
        entity_map = {"Bob": ["doc2_chunk3", "doc2_chunk4"]}
        relation_map = {}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_chunk1"]
        )
        assert "Bob" not in analysis.entities_to_delete
        assert "Bob" not in analysis.entities_to_rebuild
        assert analysis.untouched_entities == 1

    def test_relation_exclusive_to_doc(self):
        """Relation only from deleted doc → DELETE."""
        entity_map = {}
        relation_map = {("Alice", "Bob"): ["doc1_chunk1"]}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_chunk1"]
        )
        assert ("Alice", "Bob") in analysis.relations_to_delete
        assert len(analysis.relations_to_rebuild) == 0

    def test_relation_shared_across_docs(self):
        """Relation from multiple docs → REBUILD."""
        entity_map = {}
        relation_map = {("Alice", "Bob"): ["doc1_chunk1", "doc2_chunk3"]}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_chunk1"]
        )
        assert ("Alice", "Bob") not in analysis.relations_to_delete
        assert ("Alice", "Bob") in analysis.relations_to_rebuild
        assert analysis.relations_to_rebuild[("Alice", "Bob")] == ["doc2_chunk3"]

    def test_empty_maps(self):
        """No entities or relations → empty analysis."""
        analysis = analyze_graph_dependencies({}, {}, ["doc1_chunk1"])
        assert len(analysis.entities_to_delete) == 0
        assert len(analysis.entities_to_rebuild) == 0
        assert len(analysis.relations_to_delete) == 0
        assert len(analysis.relations_to_rebuild) == 0

    def test_entity_with_no_source_ids(self):
        """Entity with empty source_ids → DELETE (orphan)."""
        entity_map = {"Ghost": []}
        relation_map = {}
        analysis = analyze_graph_dependencies(entity_map, relation_map, ["doc1_chunk1"])
        assert "Ghost" in analysis.entities_to_delete

    def test_mixed_entities(self):
        """Mix of exclusive, shared, and untouched entities."""
        entity_map = {
            "Exclusive": ["doc1_c1"],
            "Shared": ["doc1_c1", "doc2_c2"],
            "Untouched": ["doc3_c3"],
        }
        relation_map = {}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_c1"]
        )
        assert "Exclusive" in analysis.entities_to_delete
        assert "Shared" in analysis.entities_to_rebuild
        assert "Untouched" not in analysis.entities_to_delete
        assert "Untouched" not in analysis.entities_to_rebuild
        assert analysis.untouched_entities == 1

    def test_relation_not_affected(self):
        """Relation not referencing deleted chunks → UNTOUCHED."""
        entity_map = {}
        relation_map = {("A", "B"): ["doc2_c1"]}
        analysis = analyze_graph_dependencies(
            entity_map, relation_map, ["doc1_c1"]
        )
        assert analysis.untouched_relations == 1


# ═══════════════════════════════════════════════════════════════════════
# DocumentDeletionPipeline tests
# ═══════════════════════════════════════════════════════════════════════


class TestDocumentDeletionPipeline:
    """Tests for the full deletion pipeline."""

    def _make_doc_status_store(self, records=None):
        """Create an in-memory doc status store with pre-loaded records."""
        store = InMemoryDocStatusStorage()
        if records:
            for r in records:
                store.upsert(r)
        return store

    def test_doc_not_found(self):
        """Deleting a nonexistent doc → NOT_FOUND."""
        store = self._make_doc_status_store()
        pipeline = DocumentDeletionPipeline(doc_status_store=store)
        result = pipeline.delete("nonexistent_doc")
        assert result.status == DeletionStatus.NOT_FOUND
        assert result.status_code == 404

    def test_doc_with_no_chunks(self):
        """Document with no chunk_ids → simple cleanup, SUCCESS."""
        store = self._make_doc_status_store([
            DocStatusRecord(doc_id="doc1", file_path="a.txt", status=DocStatus.PROCESSED)
        ])
        full_doc_data = {"doc_id": "doc1", "chunk_ids": []}

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda doc_id: full_doc_data,
        )
        result = pipeline.delete("doc1")
        assert result.status == DeletionStatus.SUCCESS
        assert result.chunks_deleted == 0

    def test_doc_with_chunks_basic_deletion(self):
        """Document with chunks → delete chunks + cleanup."""
        store = self._make_doc_status_store([
            DocStatusRecord(
                doc_id="doc1", file_path="a.txt",
                status=DocStatus.PROCESSED, chunks_count=2,
            )
        ])
        full_doc_data = {
            "doc_id": "doc1",
            "chunk_ids": ["doc1_c1", "doc1_c2"],
            "entity_names": ["Alice", "Bob"],
            "relation_pairs": [("Alice", "Bob")],
        }

        deleted_chunks = []
        deleted_entities = []
        deleted_relations = []

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda doc_id: full_doc_data,
            get_entity_source_ids=lambda name: ["doc1_c1"] if name == "Alice" else ["doc2_c1"],
            get_relation_source_ids=lambda key: ["doc1_c1"],
            delete_entity_from_graph=lambda name: (deleted_entities.append(name), True)[1],
            delete_relation_from_graph=lambda key: (deleted_relations.append(key), True)[1],
            delete_chunks_from_vector=lambda ids: (deleted_chunks.extend(ids), len(ids))[1],
            delete_chunks_from_bm25=lambda ids: len(ids),
            delete_chunks_from_graph=lambda ids: len(ids),
            delete_full_doc=lambda doc_id: True,
        )
        result = pipeline.delete("doc1")

        assert result.status == DeletionStatus.SUCCESS
        assert result.chunks_deleted == 2
        assert "Alice" in deleted_entities  # exclusive → deleted
        assert len(deleted_chunks) == 2
        # Check doc_status was deleted
        assert store.get("doc1") is None

    def test_shared_entity_rebuild(self):
        """Entity shared across docs → rebuild (update description)."""
        store = self._make_doc_status_store([
            DocStatusRecord(doc_id="doc1", file_path="a.txt", status=DocStatus.PROCESSED)
        ])
        full_doc_data = {
            "doc_id": "doc1",
            "chunk_ids": ["doc1_c1"],
            "entity_names": ["Alice"],
            "relation_pairs": [],
        }

        updated_entities = {}

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda doc_id: full_doc_data,
            get_entity_source_ids=lambda name: ["doc1_c1", "doc2_c1"] if name == "Alice" else [],
            update_entity_description=lambda name, desc, ids: (updated_entities.update({name: ids}), True)[1],
            delete_chunks_from_vector=lambda ids: len(ids),
            delete_full_doc=lambda doc_id: True,
        )
        result = pipeline.delete("doc1")
        assert result.status == DeletionStatus.SUCCESS
        assert "Alice" in updated_entities
        assert updated_entities["Alice"] == ["doc2_c1"]
        assert result.entities_rebuilt == 1

    def test_partial_deletion_due_to_limit(self):
        """Too many rebuilds → PARTIAL status when hitting max limit."""
        store = self._make_doc_status_store([
            DocStatusRecord(doc_id="doc1", file_path="a.txt", status=DocStatus.PROCESSED)
        ])
        # 3 shared entities, but max_rebuild=2
        full_doc_data = {
            "doc_id": "doc1",
            "chunk_ids": ["doc1_c1"],
            "entity_names": ["A", "B", "C"],
            "relation_pairs": [],
        }

        updated_entities = {}

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            config=DeletionConfig(max_rebuild_descriptions=2),
            get_full_doc=lambda doc_id: full_doc_data,
            get_entity_source_ids=lambda name: ["doc1_c1", "doc2_c1"],
            update_entity_description=lambda name, desc, ids: (updated_entities.update({name: ids}), True)[1],
            delete_chunks_from_vector=lambda ids: len(ids),
            delete_full_doc=lambda doc_id: True,
        )
        result = pipeline.delete("doc1")
        assert result.status == DeletionStatus.PARTIAL
        assert result.entities_rebuilt == 2
        assert len(updated_entities) == 2  # only 2 out of 3 rebuilt

    def test_no_dependency_functions(self):
        """No dependency functions → skip analysis, just delete chunks + doc."""
        store = self._make_doc_status_store([
            DocStatusRecord(doc_id="doc1", file_path="a.txt", status=DocStatus.PROCESSED)
        ])
        full_doc_data = {"doc_id": "doc1", "chunk_ids": ["c1", "c2"]}

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda doc_id: full_doc_data,
            delete_chunks_from_vector=lambda ids: len(ids),
            delete_full_doc=lambda doc_id: True,
        )
        result = pipeline.delete("doc1")
        assert result.status == DeletionStatus.SUCCESS
        assert result.chunks_deleted == 2
        assert result.entities_deleted == 0
        assert result.entities_rebuilt == 0

    def test_deletion_status_values(self):
        """Test DeletionStatus enum values."""
        assert DeletionStatus.SUCCESS.value == "success"
        assert DeletionStatus.NOT_FOUND.value == "not_found"
        assert DeletionStatus.NOT_ALLOWED.value == "not_allowed"
        assert DeletionStatus.PARTIAL.value == "partial"
        assert DeletionStatus.FAIL.value == "fail"

    def test_deletion_result_defaults(self):
        """Test DeletionResult default values."""
        result = DeletionResult(status=DeletionStatus.SUCCESS, doc_id="test")
        assert result.message == ""
        assert result.status_code == 200
        assert result.file_path is None
        assert result.chunks_deleted == 0
        assert result.entities_deleted == 0
        assert result.entities_rebuilt == 0
        assert result.relations_deleted == 0
        assert result.relations_rebuilt == 0

    def test_graph_field_sep_constant(self):
        """GRAPH_FIELD_SEP should be newline (LightRAG pattern)."""
        assert GRAPH_FIELD_SEP == "\n"

    def test_pipeline_with_all_stages(self):
        """Full pipeline: entities + relations + chunks + doc cleanup."""
        store = self._make_doc_status_store([
            DocStatusRecord(doc_id="doc1", file_path="file.txt", status=DocStatus.PROCESSED)
        ])
        full_doc_data = {
            "doc_id": "doc1",
            "chunk_ids": ["d1_c1", "d1_c2"],
            "entity_names": ["E_exclusive", "E_shared"],
            "relation_pairs": [["R_exclusive_src", "R_exclusive_tgt"], ["R_shared_src", "R_shared_tgt"]],
        }

        deleted_entities = []
        deleted_relations = []
        updated_entities = {}
        updated_relations = {}
        deleted_chunks_count = 0
        deleted_docs = []

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda did: full_doc_data,
            get_entity_source_ids=lambda name: (
                ["d1_c1"] if name == "E_exclusive"
                else ["d1_c1", "d2_c1"] if name == "E_shared"
                else []
            ),
            get_relation_source_ids=lambda key: (
                ["d1_c2"] if key == ("R_exclusive_src", "R_exclusive_tgt")
                else ["d1_c2", "d2_c2"] if key == ("R_shared_src", "R_shared_tgt")
                else []
            ),
            delete_entity_from_graph=lambda name: (deleted_entities.append(name), True)[1],
            delete_relation_from_graph=lambda key: (deleted_relations.append(key), True)[1],
            update_entity_description=lambda name, desc, ids: (updated_entities.update({name: ids}), True)[1],
            update_relation_description=lambda key, desc, ids: (updated_relations.update({str(key): ids}), True)[1],
            delete_chunks_from_vector=lambda ids: (deleted_chunks_count := len(ids), len(ids))[1],
            delete_chunks_from_bm25=lambda ids: len(ids),
            delete_chunks_from_graph=lambda ids: len(ids),
            delete_full_doc=lambda did: (deleted_docs.append(did), True)[1],
        )

        result = pipeline.delete("doc1")

        assert result.status == DeletionStatus.SUCCESS
        assert result.chunks_deleted == 2
        assert "E_exclusive" in deleted_entities
        assert "E_shared" not in deleted_entities
        assert "E_shared" in updated_entities
        assert ("R_exclusive_src", "R_exclusive_tgt") in deleted_relations
        assert "('R_shared_src', 'R_shared_tgt')" in updated_relations  # rebuilt with remaining IDs
        assert "doc1" in deleted_docs


# ═══════════════════════════════════════════════════════════════════════
# Integration-style test
# ═══════════════════════════════════════════════════════════════════════


class TestDocDeletionIntegration:
    """Integration tests combining dependency analysis + pipeline."""

    def test_multi_doc_scenario(self):
        """Simulate: doc1 deleted → some entities shared with doc2 → rebuild."""
        # Pre-seed doc2 in the status store
        store = InMemoryDocStatusStorage()
        store.upsert(DocStatusRecord(doc_id="doc1", file_path="f1.txt", status=DocStatus.PROCESSED))
        store.upsert(DocStatusRecord(doc_id="doc2", file_path="f2.txt", status=DocStatus.PROCESSED))

        # doc1 has entities: E1 (exclusive), E2 (shared with doc2)
        full_doc_data = {
            "doc_id": "doc1",
            "chunk_ids": ["d1_c1", "d1_c2"],
            "entity_names": ["E1", "E2"],
            "relation_pairs": [],
        }

        rebuild_log = []

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda did: full_doc_data if did == "doc1" else None,
            get_entity_source_ids=lambda name: (
                ["d1_c1"] if name == "E1" else ["d1_c1", "d2_c1"]
            ),
            delete_entity_from_graph=lambda name: True,
            update_entity_description=lambda name, desc, ids: (rebuild_log.append((name, ids)), True)[1],
            delete_chunks_from_vector=lambda ids: len(ids),
            delete_full_doc=lambda did: True,
        )

        result = pipeline.delete("doc1")
        assert result.entities_deleted == 1  # E1
        assert result.entities_rebuilt == 1  # E2 → ["d2_c1"]

        # doc2 should still exist in status store
        assert store.get("doc2") is not None
        # doc1 should be gone
        assert store.get("doc1") is None

    def test_failed_doc_deletion(self):
        """Deleting a FAILED document should succeed (cleanup)."""
        store = InMemoryDocStatusStorage()
        store.upsert(DocStatusRecord(doc_id="failed_doc", file_path="bad.txt", status=DocStatus.FAILED))

        pipeline = DocumentDeletionPipeline(
            doc_status_store=store,
            get_full_doc=lambda did: {"doc_id": did, "chunk_ids": []},
        )
        result = pipeline.delete("failed_doc")
        assert result.status == DeletionStatus.SUCCESS
