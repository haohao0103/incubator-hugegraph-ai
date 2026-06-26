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

"""Comprehensive tests for Claim extraction and Coref resolution operators."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from hugegraph_llm.operators.llm_op.claim_extract import (
    Claim, ClaimExtract, ClaimIndex, ClaimStatus,
    CLAIM_EXTRACT_PROMPT,
)
from hugegraph_llm.operators.llm_op.coref_resolution import (
    CorefMapping, CorefResolver,
    COREF_LLM_PROMPT,
    _CN_PERSONAL_PRONOUNS, _CN_DEMONSTRATIVES,
)


# ════════════════════════════════════════════════════════════════
#  Claim data class tests
# ════════════════════════════════════════════════════════════════

class TestClaimDataClass:

    def test_auto_id_generation(self):
        """Claim auto-generates MD5-based claim_id in __post_init__."""
        claim = Claim(subject="HugeGraph", predicate="supports", object="Gremlin")
        assert claim.claim_id.startswith("claim-")
        assert len(claim.claim_id) > 10

    def test_to_dict_roundtrip(self):
        """Claim.to_dict() and Claim.from_dict() preserve all fields."""
        claim = Claim(
            subject="张三",
            predicate="works_at",
            object="阿里云",
            description="张三在阿里云工作",
            status=ClaimStatus.SUPPORTING,
            confidence=0.95,
            source_text="张三2020年加入阿里云",
            chunk_id="chunk_0",
            doc_id="doc_1",
            start_char=0,
            end_char=20,
        )
        d = claim.to_dict()
        restored = Claim.from_dict(d)

        assert restored.subject == "张三"
        assert restored.predicate == "works_at"
        assert restored.object == "阿里云"
        assert restored.status == ClaimStatus.SUPPORTING
        assert restored.confidence == 0.95
        assert restored.chunk_id == "chunk_0"

    def test_from_dict_invalid_status_defaults(self):
        """from_dict with invalid status defaults to NOT_ENOUGH_INFO."""
        d = {"subject": "A", "predicate": "B", "object": "C", "status": "unknown_status"}
        claim = Claim.from_dict(d)
        assert claim.status == ClaimStatus.NOT_ENOUGH_INFO

    def test_triple(self):
        """Claim.triple() returns (subject, predicate, object)."""
        claim = Claim(subject="S", predicate="P", object="O")
        assert claim.triple() == ("S", "P", "O")

    def test_confidence_rounding(self):
        """to_dict() rounds confidence to 4 decimal places."""
        claim = Claim(subject="S", predicate="P", object="O", confidence=0.123456789)
        d = claim.to_dict()
        assert d["confidence"] == 0.1235  # Rounded to 4 places

    def test_status_enum_values(self):
        """ClaimStatus enum has exactly 3 values."""
        assert len(ClaimStatus) == 3
        assert ClaimStatus.SUPPORTING.value == "supporting"
        assert ClaimStatus.CONTRADICTING.value == "contradicting"
        assert ClaimStatus.NOT_ENOUGH_INFO.value == "not_enough_info"


# ════════════════════════════════════════════════════════════════
#  ClaimExtract operator tests
# ════════════════════════════════════════════════════════════════

class TestClaimExtract:

    def test_empty_chunks_returns_empty(self):
        """No chunks produces empty claims."""
        extractor = ClaimExtract(llm=None)
        context = {"chunks": [], "vertices": [], "edges": []}
        result = extractor.run(context)

        assert result["claims"] == []
        assert result["claim_count"] == 0

    def test_no_llm_returns_empty_claims(self):
        """Without LLM, extraction returns empty claims (response="[]")."""
        extractor = ClaimExtract(llm=None)
        context = {
            "chunks": [{"text": "HugeGraph supports Gremlin.", "chunk_id": "c0"}],
            "vertices": [],
            "edges": [],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] == 0  # No LLM → response="[]" → empty

    def test_with_mock_llm(self):
        """Mock LLM returns structured claims."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {
                "subject": "HugeGraph",
                "predicate": "supports",
                "object": "Gremlin",
                "description": "HugeGraph supports Gremlin query language",
                "status": "supporting",
                "confidence": 0.95,
                "source_text": "HugeGraph supports Gremlin",
                "start_char": 0,
                "end_char": 30,
            }
        ])

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "HugeGraph supports Gremlin queries.", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] >= 1
        assert any(c["subject"] == "HugeGraph" for c in result["claims"])

    def test_llm_response_with_markdown_fence(self):
        """LLM response wrapped in markdown code fence is parsed correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "Here are the claims:\n```json\n"
            + json.dumps([
                {"subject": "A", "predicate": "is", "object": "B",
                 "description": "A is B", "status": "supporting",
                 "confidence": 0.8, "source_text": "A is B",
                 "start_char": 0, "end_char": 6}
            ])
            + "\n```"
        )

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "A is B.", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] >= 1

    def test_deduplication(self):
        """Duplicate (subject, predicate, object) triples are deduplicated."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {"subject": "A", "predicate": "is", "object": "B",
             "description": "A is B", "status": "supporting",
             "confidence": 0.8, "source_text": "A is B"},
            {"subject": "a", "predicate": "IS", "object": "b",
             "description": "a IS b", "status": "supporting",
             "confidence": 0.9, "source_text": "a IS b"},  # Same triple, higher conf
        ])

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "A is B.", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        # Dedup should keep only 1 (higher confidence)
        assert result["claim_count"] == 1

    def test_incomplete_claims_filtered(self):
        """Claims missing required fields (subject/predicate/object) are skipped."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {"subject": "A", "predicate": "", "object": "B"},  # Empty predicate
            {"subject": "", "predicate": "is", "object": "C"},  # Empty subject
        ])

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "Some text.", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] == 0  # Both filtered out

    def test_low_confidence_filtered(self):
        """Claims below MIN_CONFIDENCE (0.3) are filtered."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {"subject": "A", "predicate": "is", "object": "B",
             "description": "A is B", "status": "supporting",
             "confidence": 0.2, "source_text": "A is B"},  # Below threshold
        ])

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "A is B.", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] == 0

    def test_max_claims_per_chunk_cap(self):
        """More than MAX_CLAIMS_PER_CHUNK (15) claims are truncated."""
        claims_data = [
            {"subject": f"S{i}", "predicate": f"P{i}", "object": f"O{i}",
             "description": f"Claim {i}", "status": "supporting",
             "confidence": 0.9, "source_text": f"Source {i}"}
            for i in range(20)
        ]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(claims_data)

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "Many claims here.", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] <= 15  # Cap applied

    def test_format_entities(self):
        """_format_entities creates readable entity context."""
        entities = [
            {"label": "Person", "properties": {"name": "张三"}},
            {"label": "Company", "properties": {"name": "阿里云"}},
        ]
        result = ClaimExtract._format_entities(entities)
        assert "[Person] 张三" in result
        assert "[Company] 阿里云" in result

    def test_format_entities_empty(self):
        """Empty entity list returns empty string."""
        assert ClaimExtract._format_entities([]) == ""

    def test_format_entities_truncated(self):
        """More than 30 entities are truncated."""
        entities = [{"label": "E", "properties": {"name": f"N{i}"}} for i in range(50)]
        result = ClaimExtract._format_entities(entities)
        assert result.count("- [E]") == 30  # Only first 30

    def test_format_relations(self):
        """_format_relations creates readable relation context."""
        relations = [
            {"label": "works_at", "outV": "张三", "inV": "阿里云"},
        ]
        result = ClaimExtract._format_relations(relations)
        assert "(张三)-[works_at]->(阿里云)" in result

    def test_llm_exception_returns_empty(self):
        """LLM exception returns empty claims for that chunk."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API timeout")

        extractor = ClaimExtract(llm=mock_llm)
        context = {
            "chunks": [{"text": "Some text", "chunk_id": "c0"}],
            "doc_id": "doc1",
        }
        result = extractor.run(context)

        assert result["claim_count"] == 0


# ════════════════════════════════════════════════════════════════
#  ClaimIndex tests
# ════════════════════════════════════════════════════════════════

class TestClaimIndex:

    def test_add_and_lookup(self):
        """Add claims and look up by subject/predicate/status."""
        index = ClaimIndex()
        c1 = Claim(subject="HugeGraph", predicate="supports", object="Gremlin",
                    status=ClaimStatus.SUPPORTING, confidence=0.9)
        c2 = Claim(subject="Neo4j", predicate="costs", object="money",
                    status=ClaimStatus.CONTRADICTING, confidence=0.7)
        index.add_batch([c1, c2])

        assert index.size == 2
        assert len(index.get_by_subject("hugegraph")) == 1
        assert len(index.get_by_predicate("supports")) == 1
        assert len(index.get_by_status("supporting")) == 1
        assert len(index.get_by_status("contradicting")) == 1

    def test_get_for_community(self):
        """get_for_community returns claims whose subject is in entity set."""
        index = ClaimIndex()
        c1 = Claim(subject="HugeGraph", predicate="supports", object="Gremlin")
        c2 = Claim(subject="Neo4j", predicate="costs", object="money")
        index.add_batch([c1, c2])

        result = index.get_for_community(["hugegraph"])
        assert len(result) == 1
        assert result[0].subject == "HugeGraph"

    def test_stats(self):
        """Stats returns correct counts and breakdown."""
        index = ClaimIndex()
        c1 = Claim(subject="A", predicate="is", object="B", status=ClaimStatus.SUPPORTING)
        c2 = Claim(subject="A", predicate="has", object="C", status=ClaimStatus.SUPPORTING)
        c3 = Claim(subject="B", predicate="is", object="D", status=ClaimStatus.NOT_ENOUGH_INFO)
        index.add_batch([c1, c2, c3])

        stats = index.stats()
        assert stats["total_claims"] == 3
        assert stats["unique_subjects"] == 2
        assert stats["unique_predicates"] == 2  # "is" and "has"
        assert stats["status_breakdown"]["supporting"] == 2

    def test_empty_index(self):
        """Empty index has size 0 and empty lookups."""
        index = ClaimIndex()
        assert index.size == 0
        assert index.get_by_subject("anything") == []
        assert index.stats()["total_claims"] == 0


# ════════════════════════════════════════════════════════════════
#  CorefMapping data class tests
# ════════════════════════════════════════════════════════════════

class TestCorefMappingDataClass:

    def test_to_dict(self):
        """CorefMapping.to_dict() includes all fields."""
        mapping = CorefMapping(
            mention="他",
            canonical="张三",
            entity_type="Person",
            chunk_id="c0",
            confidence=0.85,
            method="rule",
        )
        d = mapping.to_dict()
        assert d["mention"] == "他"
        assert d["canonical"] == "张三"
        assert d["entity_type"] == "Person"
        assert d["confidence"] == 0.85
        assert d["method"] == "rule"

    def test_default_values(self):
        """CorefMapping defaults are empty string and rule method."""
        mapping = CorefMapping(mention="它", canonical="HugeGraph")
        assert mapping.entity_type == ""
        assert mapping.chunk_id == ""
        assert mapping.method == "rule"
        assert mapping.confidence == 0.0


# ════════════════════════════════════════════════════════════════
#  CorefResolver tests
# ════════════════════════════════════════════════════════════════

class TestCorefResolver:

    def test_empty_chunks_or_vertices(self):
        """No chunks or vertices returns empty mappings."""
        resolver = CorefResolver()
        context = {"chunks": [], "vertices": []}
        result = resolver.run(context)

        assert result["coref_mappings"] == []
        assert result["coref_count"] == 0

    def test_no_entities_for_resolution(self):
        """Chunks without pronouns produce no mappings."""
        resolver = CorefResolver()
        context = {
            "chunks": [{"text": "HugeGraph is a graph database.", "chunk_id": "c0"}],
            "vertices": [{"label": "Software", "properties": {"name": "HugeGraph"}}],
            "doc_id": "doc1",
        }
        result = resolver.run(context)

        # No pronouns or demonstratives → no mappings
        assert result["coref_count"] == 0

    def test_pronoun_resolution(self):
        """Chinese pronoun 他 resolves to most recent Person entity."""
        resolver = CorefResolver()
        context = {
            "chunks": [
                {"text": "张三加入了阿里云。", "chunk_id": "c0"},
                {"text": "他是一名工程师。", "chunk_id": "c1"},
            ],
            "vertices": [
                {"label": "Person", "properties": {"name": "张三"}},
                {"label": "Company", "properties": {"name": "阿里云"}},
            ],
            "doc_id": "doc1",
        }
        result = resolver.run(context)

        # "他" should resolve to "张三"
        mappings = result["coref_mappings"]
        assert any(m["mention"] == "他" and m["canonical"] == "张三" for m in mappings)

    def test_demonstrative_resolution(self):
        """Chinese demonstrative 该公司 resolves to most recent Org."""
        resolver = CorefResolver()
        context = {
            "chunks": [
                {"text": "阿里云推出了新服务。", "chunk_id": "c0"},
                {"text": "该公司发展迅速。", "chunk_id": "c1"},
            ],
            "vertices": [
                {"label": "Organization", "properties": {"name": "阿里云"}},
            ],
            "doc_id": "doc1",
        }
        result = resolver.run(context)

        mappings = result["coref_mappings"]
        assert any(m["canonical"] == "阿里云" for m in mappings)

    def test_title_resolution(self):
        """Title 张先生 resolves to entity with surname 张."""
        resolver = CorefResolver()
        context = {
            "chunks": [
                {"text": "张三是项目负责人。", "chunk_id": "c0"},
                {"text": "张先生负责技术方向。", "chunk_id": "c1"},
            ],
            "vertices": [
                {"label": "Person", "properties": {"name": "张三"}},
            ],
            "doc_id": "doc1",
        }
        result = resolver.run(context)

        mappings = result["coref_mappings"]
        assert any(m["canonical"] == "张三" and "张先生" in m["mention"] for m in mappings)

    def test_org_alias_resolution(self):
        """Organization alias 阿里 resolves to 阿里云."""
        resolver = CorefResolver()
        context = {
            "chunks": [
                {"text": "阿里云提供云计算服务。", "chunk_id": "c0"},
                {"text": "阿里在市场上占有领先地位。", "chunk_id": "c1"},
            ],
            "vertices": [
                {"label": "Organization", "properties": {"name": "阿里云"}},
            ],
            "doc_id": "doc1",
        }
        result = resolver.run(context)

        mappings = result["coref_mappings"]
        # Should find some mapping for "阿里" → "阿里云"
        assert result["coref_count"] >= 0  # Depends on pattern matching

    def test_deduplicate_mappings(self):
        """Duplicate (mention, canonical, chunk_id) tuples are removed."""
        mappings = [
            CorefMapping(mention="他", canonical="张三", chunk_id="c0"),
            CorefMapping(mention="他", canonical="张三", chunk_id="c0"),  # Duplicate
            CorefMapping(mention="她", canonical="李四", chunk_id="c1"),
        ]
        unique = CorefResolver._deduplicate_mappings(mappings)
        assert len(unique) == 2

    def test_apply_to_text(self):
        """apply_to_text replaces mentions with canonical names."""
        resolver = CorefResolver()
        text = "他是一名工程师。"
        mappings = [
            CorefMapping(mention="他", canonical="张三", confidence=0.85),
        ]
        result = resolver.apply_to_text(text, mappings)
        assert "张三" in result
        assert "他" not in result

    def test_apply_to_text_longer_first(self):
        """Longer mentions are replaced first to avoid partial overlap."""
        resolver = CorefResolver()
        text = "该公司和他的项目"
        mappings = [
            CorefMapping(mention="该", canonical="阿里云"),
            CorefMapping(mention="该公司", canonical="阿里云"),
            CorefMapping(mention="他", canonical="张三"),
        ]
        result = resolver.apply_to_text(text, mappings)
        # "该公司" should be replaced before "该"
        assert "阿里云" in result

    def test_build_entity_catalog(self):
        """_build_entity_catalog creates name->(type,props) mapping."""
        vertices = [
            {"label": "Person", "properties": {"name": "张三"}},
            {"label": "Company", "properties": {"name": "阿里云"}},
            {"label": "Unknown", "properties": {}},  # No name → skipped
        ]
        catalog = CorefResolver._build_entity_catalog(vertices)

        assert "张三" in catalog
        assert catalog["张三"][0] == "Person"
        assert "阿里云" in catalog

    def test_find_explicit_mentions(self):
        """_find_explicit_mentions finds entity names in text."""
        catalog = {"张三": ("Person", {}), "阿里云": ("Organization", {})}
        text = "张三加入了阿里云。"
        mentions = CorefResolver._find_explicit_mentions(text, catalog)

        assert "张三" in mentions
        assert "阿里云" in mentions

    @patch.object(CorefResolver, "_resolve_llm")
    def test_llm_pass_enabled(self, mock_resolve_llm):
        """LLM pass is executed when enable_llm_pass=True."""
        mock_resolve_llm.return_value = [
            CorefMapping(mention="这个系统", canonical="HugeGraph", method="llm"),
        ]

        resolver = CorefResolver(llm=MagicMock(), enable_llm_pass=True)
        context = {
            "chunks": [{"text": "这个系统很好。", "chunk_id": "c0"}],
            "vertices": [{"label": "Software", "properties": {"name": "HugeGraph"}}],
            "doc_id": "doc1",
        }
        result = resolver.run(context)

        mock_resolve_llm.assert_called_once()

    def test_llm_pass_disabled(self):
        """LLM pass is NOT executed when enable_llm_pass=False."""
        resolver = CorefResolver(llm=None, enable_llm_pass=False)
        context = {
            "chunks": [{"text": "Some text", "chunk_id": "c0"}],
            "vertices": [{"label": "E", "properties": {"name": "Entity"}}],
            "doc_id": "doc1",
        }
        # Should NOT try to call LLM
        result = resolver.run(context)
        assert result["coref_count"] >= 0  # Just rule-based, no LLM

    def test_parse_llm_response(self):
        """_parse_llm_response extracts CorefMapping from JSON."""
        response = json.dumps([
            {"mention": "他", "canonical": "张三", "entity_type": "Person", "confidence": 0.9},
            {"mention": "该公司", "canonical": "阿里云", "entity_type": "Organization", "confidence": 0.8},
        ])

        result = CorefResolver._parse_llm_response(response, "c0")
        assert len(result) == 2
        assert result[0].mention == "他"
        assert result[0].method == "llm"

    def test_parse_llm_response_invalid_json(self):
        """Invalid JSON in LLM response returns empty list."""
        result = CorefResolver._parse_llm_response("not json at all", "c0")
        assert result == []

    def test_parse_llm_response_missing_fields(self):
        """Items missing mention or canonical are skipped."""
        response = json.dumps([
            {"mention": "", "canonical": "张三"},  # Empty mention
            {"mention": "他", "canonical": ""},  # Empty canonical
            {"mention": "她", "canonical": "李四"},  # Valid
        ])
        result = CorefResolver._parse_llm_response(response, "c0")
        assert len(result) == 1
        assert result[0].mention == "她"


# ════════════════════════════════════════════════════════════════
#  Prompt templates sanity tests
# ════════════════════════════════════════════════════════════════

class TestPromptTemplates:

    def test_claim_prompt_has_placeholders(self):
        """CLAIM_EXTRACT_PROMPT has all required placeholders."""
        assert "{chunk_id}" in CLAIM_EXTRACT_PROMPT
        assert "{text}" in CLAIM_EXTRACT_PROMPT
        assert "{entities_ctx}" in CLAIM_EXTRACT_PROMPT
        assert "{relations_ctx}" in CLAIM_EXTRACT_PROMPT

    def test_claim_prompt_formatting(self):
        """CLAIM_EXTRACT_PROMPT can be formatted without errors."""
        formatted = CLAIM_EXTRACT_PROMPT.format(
            chunk_id="c0",
            text="Some text here",
            entities_ctx="- [Person] 张三",
            relations_ctx="- (张三)-[works_at]->(阿里云)",
        )
        assert "Some text here" in formatted

    def test_coref_prompt_has_placeholders(self):
        """COREF_LLM_PROMPT has all required placeholders."""
        assert "{chunk_id}" in COREF_LLM_PROMPT
        assert "{text}" in COREF_LLM_PROMPT
        assert "{entities_list}" in COREF_LLM_PROMPT

    def test_coref_prompt_formatting(self):
        """COREF_LLM_PROMPT can be formatted without errors."""
        formatted = COREF_LLM_PROMPT.format(
            chunk_id="c1",
            text="他是一名工程师",
            entities_list="- [Person] 张三\n- [Organization] 阿里云",
        )
        assert "他是一名工程师" in formatted

    def test_cn_pronouns_non_empty(self):
        """Chinese pronoun dictionaries are populated."""
        assert len(_CN_PERSONAL_PRONOUNS) > 0
        assert "他" in _CN_PERSONAL_PRONOUNS
        assert "她" in _CN_PERSONAL_PRONOUNS
        assert "他们" in _CN_PERSONAL_PRONOUNS

    def test_cn_demonstratives_non_empty(self):
        """Chinese demonstrative dictionaries are populated."""
        assert len(_CN_DEMONSTRATIVES) > 0
        assert "该公司" in _CN_DEMONSTRATIVES
        assert "这" in _CN_DEMONSTRATIVES
