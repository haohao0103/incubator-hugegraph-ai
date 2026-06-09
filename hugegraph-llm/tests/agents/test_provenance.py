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

"""Tests for the text provenance and citation system."""

from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.operators.hugegraph_op.provenance_manager import (
    PROVENANCE_VERTEX_LABELS,
    PROVENANCE_EDGE_LABELS,
    PROVENANCE_PROPERTY_KEYS,
    ProvenanceManager,
    ProvenanceRecord,
    create_provenance_manager,
)
from hugegraph_llm.operators.llm_op.provenance_answer import ProvenanceAnswerSynthesize


# ── ProvenanceRecord Tests ────────────────────────────────────


class TestProvenanceRecord:
    """Tests for the ProvenanceRecord dataclass."""

    def test_to_citation(self):
        record = ProvenanceRecord(
            entity_id="1:Sarah",
            chunk_id="CHUNK:abc123",
            chunk_text="Sarah is a 30-year-old software engineer living in San Francisco.",
            chunk_index=2,
            document_name="report.pdf",
            document_source="/data/report.pdf",
        )
        citation = record.to_citation(max_text_len=100)
        assert "Sarah is a 30-year-old" in citation
        assert "report.pdf" in citation or "/data/report.pdf" in citation

    def test_to_citation_truncation(self):
        record = ProvenanceRecord(
            entity_id="1:X",
            chunk_id="CHUNK:xyz",
            chunk_text="A" * 500,
            chunk_index=0,
            document_name="doc.txt",
        )
        citation = record.to_citation(max_text_len=50)
        assert len(citation) < 200  # Should be truncated
        assert "..." in citation

    def test_to_citation_no_source(self):
        record = ProvenanceRecord(
            entity_id="1:X",
            chunk_id="CHUNK:xyz",
            chunk_text="Some text.",
            chunk_index=0,
            document_name="doc.txt",
            document_source="",
        )
        citation = record.to_citation()
        assert "doc.txt" in citation  # Falls back to doc name


# ── ProvenanceManager Tests ───────────────────────────────────


class TestProvenanceManager:
    """Tests for the ProvenanceManager."""

    def _make_mock_client(self):
        """Create a mock HugeGraph client."""
        client = MagicMock()
        client.schema.return_value = MagicMock()
        client.graph.return_value = MagicMock()
        client.gremlin.return_value = MagicMock()
        return client

    def test_document_id_prefix(self):
        """Test that document IDs use the DOC: prefix."""
        assert ProvenanceManager.DOC_PREFIX == "DOC:"
        assert ProvenanceManager.CHUNK_PREFIX == "CHUNK:"

    def test_create_document_id_format(self):
        """Test document creation returns correct ID format."""
        pm = ProvenanceManager(client=self._make_mock_client())
        # Override init_schema to no-op
        pm._initialized = True

        doc_id = pm.DOC_PREFIX + "test.pdf"
        assert doc_id.startswith("DOC:")
        assert "test.pdf" in doc_id

    def test_create_chunk_stable_id(self):
        """Test that chunk IDs are stable (hash-based)."""
        import hashlib

        text = "This is a test chunk."
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        expected_id = f"CHUNK:{text_hash}"

        assert expected_id.startswith("CHUNK:")
        assert len(text_hash) == 16  # MD5 first 16 chars

    def test_provenance_schema_labels(self):
        """Test that provenance schema labels are correctly defined."""
        assert "Document" in PROVENANCE_VERTEX_LABELS
        assert "Chunk" in PROVENANCE_VERTEX_LABELS
        assert "CONTAINS_CHUNK" in PROVENANCE_EDGE_LABELS
        assert "EXTRACTED_FROM" in PROVENANCE_EDGE_LABELS

        doc_config = PROVENANCE_VERTEX_LABELS["Document"]
        assert "name" in doc_config["properties"]
        assert "source" in doc_config["properties"]

    def test_property_keys_defined(self):
        """Test that all required property keys are defined."""
        key_names = [pk["name"] for pk in PROVENANCE_PROPERTY_KEYS]
        assert "name" in key_names
        assert "text" in key_names
        assert "index" in key_names
        assert "extraction_type" in key_names

    def test_get_provenance_with_mock(self):
        """Test provenance query with mocked Gremlin result."""
        client = self._make_mock_client()
        pm = ProvenanceManager(client=client)
        pm._initialized = True

        # Mock Gremlin exec to return provenance data
        mock_response = {
            "data": [
                {
                    "chunk": {
                        "id": "CHUNK:abc",
                        "text": "Sarah is a software engineer.",
                        "index": 2,
                    },
                    "doc": {
                        "name": "report.pdf",
                        "source": "/data/report.pdf",
                    },
                }
            ]
        }
        client.gremlin.return_value.exec.return_value = mock_response

        records = pm.get_provenance("1:Sarah")
        assert len(records) == 1
        assert records[0].entity_id == "1:Sarah"
        assert "Sarah" in records[0].chunk_text
        assert records[0].document_name == "report.pdf"

    def test_get_provenance_empty(self):
        """Test provenance query with no results."""
        client = self._make_mock_client()
        pm = ProvenanceManager(client=client)
        pm._initialized = True
        client.gremlin.return_value.exec.return_value = {"data": []}

        records = pm.get_provenance("nonexistent")
        assert records == []

    def test_get_provenance_for_answer(self):
        """Test batch provenance query for multiple entities."""
        client = self._make_mock_client()
        pm = ProvenanceManager(client=client)
        pm._initialized = True

        mock_response = {
            "data": [
                {
                    "chunk": {
                        "id": "CHUNK:1", "text": "Text 1", "index": 0
                    },
                    "doc": {"name": "doc1", "source": "s1"},
                }
            ]
        }
        client.gremlin.return_value.exec.return_value = mock_response

        result = pm.get_provenance_for_answer(
            ["1:Alice", "2:Bob"], max_per_entity=1
        )
        assert isinstance(result, dict)
        assert len(result) == 2  # Both entities have provenance

    def test_create_provenance_manager(self):
        """Test factory function."""
        client = self._make_mock_client()
        pm = create_provenance_manager(client=client)
        assert isinstance(pm, ProvenanceManager)
        assert pm._initialized is True  # Schema should be initialized


# ── ProvenanceAnswerSynthesize Tests ──────────────────────────


class TestProvenanceAnswerSynthesize:
    """Tests for provenance-aware answer synthesis."""

    def test_run_without_llm_or_pm(self):
        """Test that run works without LLM or provenance manager."""
        synth = ProvenanceAnswerSynthesize()
        context = {
            "query": "Who is Sarah?",
            "graph_result": "Sarah is a person.",
            "match_vids": ["1:Sarah"],
        }
        result = synth.run(context)
        assert "answer" in result
        assert result["citations"] == []  # No PM = no citations

    def test_run_with_citations(self):
        """Test that citations are appended when provenance data exists."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Sarah is a software engineer."

        mock_pm = MagicMock()
        mock_pm.get_provenance_for_answer.return_value = {
            "1:Sarah": [
                ProvenanceRecord(
                    entity_id="1:Sarah",
                    chunk_id="CHUNK:abc",
                    chunk_text="Sarah, a 30-year-old engineer...",
                    chunk_index=0,
                    document_name="report.pdf",
                )
            ]
        }

        synth = ProvenanceAnswerSynthesize(llm=mock_llm, provenance_manager=mock_pm)
        context = {
            "query": "Who is Sarah?",
            "graph_result": "Sarah is a person.",
            "match_vids": ["1:Sarah"],
        }
        result = synth.run(context)

        assert "Sarah is a software engineer" in result["answer"]
        assert "## 来源" in result["answer"]  # Citations header
        assert len(result["citations"]) == 1
        assert "Sarah" in result["citations"][0]

    def test_run_deduplicates_citations(self):
        """Test that duplicate citations are removed."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Answer about Alice and Bob."

        mock_pm = MagicMock()
        mock_pm.get_provenance_for_answer.return_value = {
            "1:Alice": [
                ProvenanceRecord(
                    entity_id="1:Alice",
                    chunk_id="CHUNK:1",
                    chunk_text="Same text appears twice.",
                    chunk_index=0,
                    document_name="doc.txt",
                )
            ],
            "2:Bob": [
                ProvenanceRecord(
                    entity_id="2:Bob",
                    chunk_id="CHUNK:1",  # Same chunk
                    chunk_text="Same text appears twice.",  # Same text
                    chunk_index=0,
                    document_name="doc.txt",
                )
            ],
        }

        synth = ProvenanceAnswerSynthesize(
            llm=mock_llm, provenance_manager=mock_pm, max_citations=10
        )
        context = {
            "query": "Who are Alice and Bob?",
            "match_vids": ["1:Alice", "2:Bob"],
        }
        result = synth.run(context)

        # Should deduplicate since both entities point to same text
        assert len(result["citations"]) <= 1

    def test_run_respects_max_citations(self):
        """Test that max_citations limit is respected."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Answer."

        records = {}
        for i in range(10):
            eid = f"1:Entity{i}"
            records[eid] = [
                ProvenanceRecord(
                    entity_id=eid,
                    chunk_id=f"CHUNK:{i}",
                    chunk_text=f"Unique text {i}",
                    chunk_index=i,
                    document_name=f"doc{i}.txt",
                )
            ]

        mock_pm = MagicMock()
        mock_pm.get_provenance_for_answer.return_value = records

        synth = ProvenanceAnswerSynthesize(
            llm=mock_llm, provenance_manager=mock_pm, max_citations=3
        )
        context = {
            "query": "Tell me about all entities.",
            "match_vids": [f"1:Entity{i}" for i in range(10)],
        }
        result = synth.run(context)

        assert len(result["citations"]) <= 3

    def test_run_with_no_match_vids(self):
        """Test that answer is generated even without matched vertex IDs."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "A generic answer."

        synth = ProvenanceAnswerSynthesize(llm=mock_llm)
        context = {
            "query": "What is the meaning of life?",
            "vector_result": "42 is the answer.",
            "match_vids": [],
        }
        result = synth.run(context)
        assert "A generic answer" in result["answer"]
        assert result["citations"] == []  # No entities to trace

    def test_run_with_llm_error(self):
        """Test graceful handling of LLM errors."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")

        synth = ProvenanceAnswerSynthesize(llm=mock_llm)
        result = synth.run({"query": "test", "match_vids": []})
        assert result["answer"] == ""
        assert result["citations"] == []
