# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Tests for Reference Citation module."""

import pytest

from hugegraph_llm.operators.llm_op.reference_citation import (
    ReferenceEntry,
    ReferenceIdGenerator,
    ReferenceList,
    ReferenceCitationBuilder,
)


# ── Test fixtures ──

def _make_chunks():
    """Create test chunks with file_path."""
    return [
        {"file_path": "/docs/physics.pdf", "text": "Quantum mechanics describes particles.", "content": "Quantum mechanics describes particles."},
        {"file_path": "/docs/physics.pdf", "text": "Wave-particle duality is key.", "content": "Wave-particle duality is key."},
        {"file_path": "/docs/chemistry.pdf", "text": "Chemical bonds form between atoms.", "content": "Chemical bonds form between atoms."},
        {"file_path": "/docs/biology.pdf", "text": "DNA carries genetic information.", "content": "DNA carries genetic information."},
        {"file_path": "/docs/biology.pdf", "text": "Cells are the basic unit of life.", "content": "Cells are the basic unit of life."},
        {"file_path": "/docs/biology.pdf", "text": "Mitosis divides cells.", "content": "Mitosis divides cells."},
    ]


def _make_provenance_records():
    """Create mock ProvenanceRecords."""
    class MockRecord:
        def __init__(self, doc_path, chunk_text):
            self.document_path = doc_path
            self.file_path = doc_path
            self.chunk_text = chunk_text

    return [
        MockRecord("/data/report1.pdf", "Key finding A"),
        MockRecord("/data/report2.pdf", "Key finding B"),
        MockRecord("/data/report1.pdf", "Additional evidence C"),
    ]


# ── ReferenceEntry tests ──

class TestReferenceEntry:
    def test_to_dict(self):
        entry = ReferenceEntry(reference_id="1", file_path="/docs/test.pdf")
        d = entry.to_dict()
        assert d == {"reference_id": "1", "file_path": "/docs/test.pdf"}

    def test_to_citation_str_with_text(self):
        entry = ReferenceEntry(
            reference_id="2", file_path="/docs/long.pdf",
            chunk_text="A very long text that should be truncated",
        )
        s = entry.to_citation_str(max_text_len=20)
        assert "[2]" in s
        assert "/docs/long.pdf" in s
        assert "…" in s

    def test_to_citation_str_without_text(self):
        entry = ReferenceEntry(reference_id="3", file_path="/docs/short.pdf")
        s = entry.to_citation_str()
        assert s == "[3] /docs/short.pdf"


# ── ReferenceList tests ──

class TestReferenceList:
    def test_format_reference_section_empty(self):
        ref_list = ReferenceList()
        assert ref_list.format_reference_section() == ""

    def test_format_reference_section_with_entries(self):
        entries = [
            ReferenceEntry(reference_id="1", file_path="/a.pdf", chunk_text="Text A"),
            ReferenceEntry(reference_id="2", file_path="/b.pdf"),
        ]
        ref_list = ReferenceList(entries=entries)
        section = ref_list.format_reference_section()
        assert "## References" in section
        assert "[1]" in section
        assert "[2]" in section

    def test_to_dict_list(self):
        entries = [
            ReferenceEntry(reference_id="1", file_path="/a.pdf"),
            ReferenceEntry(reference_id="2", file_path="/b.pdf"),
        ]
        ref_list = ReferenceList(entries=entries)
        dicts = ref_list.to_dict_list()
        assert len(dicts) == 2
        assert dicts[0]["reference_id"] == "1"


# ── ReferenceIdGenerator tests ──

class TestReferenceIdGenerator:
    def test_generate_empty_chunks(self):
        gen = ReferenceIdGenerator()
        ref_list, updated = gen.generate_from_chunks([])
        assert ref_list.entries == []
        assert updated == []

    def test_generate_frequency_sorting(self):
        """Most frequent file_path gets reference_id=1."""
        gen = ReferenceIdGenerator()
        chunks = _make_chunks()
        ref_list, updated = gen.generate_from_chunks(chunks)

        # biology.pdf has 3 occurrences → should be ref_id=1
        # physics.pdf has 2 occurrences → should be ref_id=2
        # chemistry.pdf has 1 occurrence → should be ref_id=3
        assert ref_list.entries[0].file_path == "/docs/biology.pdf"
        assert ref_list.entries[0].reference_id == "1"
        assert ref_list.entries[0].frequency == 3

        assert ref_list.entries[1].file_path == "/docs/physics.pdf"
        assert ref_list.entries[1].reference_id == "2"

        assert ref_list.entries[2].file_path == "/docs/chemistry.pdf"
        assert ref_list.entries[2].reference_id == "3"

    def test_generate_injects_reference_id_into_chunks(self):
        gen = ReferenceIdGenerator()
        chunks = _make_chunks()
        _, updated = gen.generate_from_chunks(chunks)

        # All biology chunks should have reference_id="1"
        bio_chunks = [c for c in updated if c["file_path"] == "/docs/biology.pdf"]
        assert all(c["reference_id"] == "1" for c in bio_chunks)

        # All physics chunks should have reference_id="2"
        phys_chunks = [c for c in updated if c["file_path"] == "/docs/physics.pdf"]
        assert all(c["reference_id"] == "2" for c in phys_chunks)

    def test_generate_unknown_source_marker(self):
        gen = ReferenceIdGenerator(unknown_source_marker="unknown_source")
        chunks = [
            {"file_path": "unknown_source", "text": "No source"},
            {"file_path": "/docs/known.pdf", "text": "Known source"},
        ]
        ref_list, updated = gen.generate_from_chunks(chunks)

        # unknown_source should not appear in reference list
        assert len(ref_list.entries) == 1
        assert ref_list.entries[0].file_path == "/docs/known.pdf"

        # unknown_source chunk should have empty reference_id
        assert updated[0]["reference_id"] == ""

    def test_generate_empty_file_path(self):
        gen = ReferenceIdGenerator()
        chunks = [
            {"file_path": "", "text": "No file path"},
            {"file_path": "/docs/valid.pdf", "text": "Valid"},
        ]
        ref_list, updated = gen.generate_from_chunks(chunks)
        assert len(ref_list.entries) == 1
        assert updated[0]["reference_id"] == ""

    def test_generate_preserves_original_chunk_data(self):
        gen = ReferenceIdGenerator()
        chunks = [{"file_path": "/a.pdf", "text": "Content", "extra": "data"}]
        _, updated = gen.generate_from_chunks(chunks)
        assert updated[0]["extra"] == "data"
        assert updated[0]["text"] == "Content"

    def test_generate_id_to_file_path_mapping(self):
        gen = ReferenceIdGenerator()
        chunks = _make_chunks()
        ref_list, _ = gen.generate_from_chunks(chunks)
        assert "1" in ref_list.id_to_file_path
        assert ref_list.id_to_file_path["1"] == "/docs/biology.pdf"


# ── ReferenceIdGenerator.from_records tests ──

class TestReferenceIdGeneratorFromRecords:
    def test_generate_from_records(self):
        gen = ReferenceIdGenerator()
        records = _make_provenance_records()
        ref_list = gen.generate_from_records(records)

        # report1.pdf has 2 occurrences → ref_id=1
        assert ref_list.entries[0].file_path == "/data/report1.pdf"
        assert ref_list.entries[0].reference_id == "1"
        assert ref_list.entries[0].frequency == 2

    def test_generate_from_empty_records(self):
        gen = ReferenceIdGenerator()
        ref_list = gen.generate_from_records([])
        assert ref_list.entries == []

    def test_generate_from_records_text_truncation(self):
        gen = ReferenceIdGenerator()
        records = _make_provenance_records()
        ref_list = gen.generate_from_records(records, max_text_len=100)
        for entry in ref_list.entries:
            assert len(entry.chunk_text) <= 100


# ── ReferenceCitationBuilder tests ──

class TestReferenceCitationBuilder:
    def test_build_prompt_with_references(self):
        builder = ReferenceCitationBuilder()
        entries = [
            ReferenceEntry(reference_id="1", file_path="/docs/a.pdf"),
            ReferenceEntry(reference_id="2", file_path="/docs/b.pdf"),
        ]
        ref_list = ReferenceList(entries=entries)
        prompt = builder.build_prompt_with_references("Answer this: {query}", ref_list)
        assert "[1]" in prompt
        assert "/docs/a.pdf" in prompt
        assert "Reference Document List" in prompt

    def test_build_prompt_empty_references(self):
        builder = ReferenceCitationBuilder()
        ref_list = ReferenceList()
        prompt = builder.build_prompt_with_references("Base prompt", ref_list)
        assert prompt == "Base prompt"

    def test_build_answer_with_references(self):
        builder = ReferenceCitationBuilder()
        entries = [
            ReferenceEntry(reference_id="1", file_path="/docs/a.pdf", chunk_text="Key fact"),
            ReferenceEntry(reference_id="2", file_path="/docs/b.pdf", chunk_text="Support"),
        ]
        ref_list = ReferenceList(entries=entries)
        answer = builder.build_answer_with_references(
            "According to [1], quantum physics is complex.",
            ref_list,
        )
        assert "## References" in answer
        assert "[1]" in answer
        assert "/docs/a.pdf" in answer

    def test_build_answer_no_double_append(self):
        builder = ReferenceCitationBuilder()
        ref_list = ReferenceList(entries=[
            ReferenceEntry(reference_id="1", file_path="/a.pdf"),
        ])
        # Answer already has ## References
        answer = builder.build_answer_with_references(
            "Answer\n\n## References\n[1] existing",
            ref_list,
        )
        # Should not double-append
        assert answer.count("## References") == 1

    def test_build_answer_empty_answer(self):
        builder = ReferenceCitationBuilder()
        ref_list = ReferenceList(entries=[
            ReferenceEntry(reference_id="1", file_path="/a.pdf"),
        ])
        assert builder.build_answer_with_references("", ref_list) == ""

    def test_extract_cited_reference_ids(self):
        builder = ReferenceCitationBuilder()
        ids = builder.extract_cited_reference_ids("According to [1] and [3], ...")
        assert ids == ["1", "3"]

    def test_extract_cited_reference_ids_no_citations(self):
        builder = ReferenceCitationBuilder()
        ids = builder.extract_cited_reference_ids("No citations here")
        assert ids == []

    def test_invalid_citation_warning(self):
        """Builder should warn about invalid reference IDs."""
        builder = ReferenceCitationBuilder()
        ref_list = ReferenceList(
            entries=[ReferenceEntry(reference_id="1", file_path="/a.pdf")],
            id_to_file_path={"1": "/a.pdf"},
        )
        # Answer cites [5] which doesn't exist in ref_list
        answer = builder.build_answer_with_references(
            "According to [1] and [5]...",
            ref_list,
        )
        # Should still append references
        assert "## References" in answer
