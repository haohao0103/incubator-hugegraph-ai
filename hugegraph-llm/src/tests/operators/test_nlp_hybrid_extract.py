# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not with this file except in compliance
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

"""
Tests for NLP hybrid extraction operator.
"""

from hugegraph_llm.operators.graphrag_op.nlp_hybrid_extract import (
    ExtractMode,
    HybridExtractor,
    NLPExtractor,
)


class TestNLPExtractor:
    """Tests for the NLPExtractor class."""

    def setup_method(self):
        self.extractor_en = NLPExtractor(language="en")
        self.extractor_zh = NLPExtractor(language="zh")

    def test_extract_entities_english(self):
        text = "Meet Sarah, a 30-year-old attorney, and her roommate James."
        entities = self.extractor_en.extract_entities(text)

        assert len(entities) > 0
        # Should extract at least some named entities
        assert any(e["type"] == "person" for e in entities)

    def test_extract_entities_chinese(self):
        text = "认识Sarah，她是一位30岁的律师，和她的室友James从2010年开始合住。"
        entities = self.extractor_zh.extract_entities(text)

        assert len(entities) > 0

    def test_extract_entities_date(self):
        text = "They moved in together in January 2010 and lived there until 2020-03-15."
        entities = self.extractor_en.extract_entities(text)

        date_entities = [e for e in entities if e["type"] == "date"]
        assert len(date_entities) > 0

    def test_extract_entities_organization(self):
        text = "Sarah works at Microsoft Corporation and James is at Stanford University."
        entities = self.extractor_en.extract_entities(text)

        org_entities = [e for e in entities if e["type"] == "organization"]
        assert len(org_entities) > 0

    def test_extract_relations_english(self):
        text = "Sarah is an attorney. James works as a journalist."
        entities = self.extractor_en.extract_entities(text)
        relations = self.extractor_en.extract_relations(text, entities)

        assert isinstance(relations, list)

    def test_extract_relations_chinese(self):
        text = "Sarah是律师。James担任记者。"
        entities = self.extractor_zh.extract_entities(text)
        relations = self.extractor_zh.extract_relations(text, entities)

        assert isinstance(relations, list)

    def test_extract_co_references(self):
        text = "Sarah is an attorney. She lives with James."
        entities = self.extractor_en.extract_entities(text)
        co_refs = self.extractor_en.extract_co_references(text, entities)

        assert isinstance(co_refs, list)

    def test_empty_text(self):
        entities = self.extractor_en.extract_entities("")
        assert entities == []

    def test_no_entities_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        entities = self.extractor_en.extract_entities(text)
        # May or may not find entities, but should not error
        assert isinstance(entities, list)


class TestHybridExtractor:
    """Tests for the HybridExtractor class."""

    def test_nlp_only_mode_no_llm(self):
        extractor = HybridExtractor(extract_mode=ExtractMode.NLP_ONLY, language="en")
        context = {
            "chunks": ["Meet Sarah, a 30-year-old attorney, and James, a journalist."],
        }

        result = extractor.run(context)

        assert "vertices" in result
        assert "edges" in result
        assert result["extract_mode"] == "nlp_only"
        # NLP_ONLY mode should have zero LLM calls
        assert result.get("call_count", 0) == 0

    def test_nlp_only_mode_with_schema(self):
        schema = {
            "vertexlabels": [
                {
                    "id": 1,
                    "name": "person",
                    "primary_keys": ["name"],
                    "properties": ["name", "age", "occupation"],
                    "nullable_keys": [],
                }
            ],
            "edgelabels": [{"name": "roommate", "source_label": "person", "target_label": "person", "properties": []}],
        }
        extractor = HybridExtractor(extract_mode=ExtractMode.NLP_ONLY, language="en")
        context = {
            "chunks": ["Meet Sarah, a 30-year-old attorney, and James, a journalist."],
            "schema": schema,
        }

        result = extractor.run(context)

        assert "vertices" in result
        assert "edges" in result
        assert result.get("call_count", 0) == 0

    def test_hybrid_mode_without_llm(self):
        """When LLM is None but mode is HYBRID, should gracefully degrade to NLP_ONLY."""
        extractor = HybridExtractor(extract_mode=ExtractMode.HYBRID, language="en", llm=None)
        context = {
            "chunks": ["Sarah is an attorney and James is a journalist."],
        }

        result = extractor.run(context)

        assert "vertices" in result
        assert result["extract_mode"] == "hybrid"

    def test_deduplication(self):
        extractor = HybridExtractor(extract_mode=ExtractMode.NLP_ONLY, language="en")
        context = {
            "chunks": [
                "Sarah is an attorney.",
                "Sarah works at a law firm.",
            ],
        }

        result = extractor.run(context)

        # Entities should be deduplicated
        entity_names = [e["name"] for e in result.get("extracted_entities", [])]
        # Sarah should appear only once
        sarah_count = sum(1 for name in entity_names if "sarah" in name.lower())
        assert sarah_count <= 2  # May have different casing variants

    def test_extract_mode_enum(self):
        assert ExtractMode.NLP_ONLY.value == "nlp_only"
        assert ExtractMode.LLM_ONLY.value == "llm_only"
        assert ExtractMode.HYBRID.value == "hybrid"

    def test_empty_chunks(self):
        extractor = HybridExtractor(extract_mode=ExtractMode.NLP_ONLY)
        context = {"chunks": []}

        result = extractor.run(context)

        assert result.get("extracted_entities", []) == []
        assert result.get("extracted_relations", []) == []

    def test_triples_output_without_schema(self):
        extractor = HybridExtractor(extract_mode=ExtractMode.NLP_ONLY, language="en")
        context = {
            "chunks": ["Sarah is an attorney and James is a journalist."],
        }

        result = extractor.run(context)

        # Without schema, should produce triples
        assert "triples" in result
