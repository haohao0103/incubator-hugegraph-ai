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
# "ASIS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for EDC + Guided schema pipeline — GraphRAGSchemaConfig,
KGSchemaDefineOperator, KGSchemaCanonicalizeOperator,
GuidedExtractOperator, EDCPipelineOrchestrator."""

import json
import math
import re
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.operators.graphrag_op.graphrag_schema_config import (
    CanonicalizeStrategy,
    DefineTriggerPolicy,
    GraphRAGSchemaConfig,
    SchemaMode,
)
from hugegraph_llm.operators.graphrag_op.kg_schema_define import (
    KGSchemaDefineOperator,
)
from hugegraph_llm.operators.graphrag_op.kg_schema_canonicalize import (
    KGSchemaCanonicalizeOperator,
)
from hugegraph_llm.operators.graphrag_op.guided_extract import (
    GuidedEntity,
    GuidedExtractOperator,
    GuidedExtractResponse,
    GuidedResponseModelBuilder,
    GuidedRelation,
)
from hugegraph_llm.operators.graphrag_op.edc_pipeline import (
    EDCPipelineOrchestrator,
)


# ============================================================
# Mock LLM
# ============================================================

class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        self.call_count += 1
        return '{"description": "mock entity", "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": true}], "parent_types": ["Entity"]}'


def simple_embed_func(text: str) -> List[float]:
    """Simple embedding function for testing — returns a deterministic vector."""
    # Create a simple hash-based vector for testing
    dim = 10
    vector = [0.0] * dim
    for i, char in enumerate(text[:dim]):
        vector[i] = ord(char) / 128.0
    # Normalize
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


# ============================================================
# GraphRAGSchemaConfig Tests
# ============================================================

class TestGraphRAGSchemaConfig:
    """Tests for GraphRAGSchemaConfig dataclass."""

    def test_default_config(self):
        config = GraphRAGSchemaConfig()
        assert config.mode == SchemaMode.EVOLVING
        assert config.canonicalize_strategy == CanonicalizeStrategy.EMBEDDING_SIM
        assert config.define_trigger_policy == DefineTriggerPolicy.NEW_TYPES_ONLY
        assert config.canonicalize_similarity_threshold == 0.85
        assert config.canonicalize_suggest_threshold == 0.70
        assert config.preserve_raw_type is True

    def test_guided_mode_config(self):
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.GUIDED,
            relationship_graph_types=["person", "device", "ip_address"],
        )
        assert config.mode == SchemaMode.GUIDED
        assert len(config.relationship_graph_types) == 3

    def test_validate_guided_without_types(self):
        config = GraphRAGSchemaConfig(mode=SchemaMode.GUIDED)
        issues = config.validate()
        assert len(issues) > 0
        assert "requires relationship_graph_types" in issues[0]

    def test_validate_guided_with_types(self):
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.GUIDED,
            relationship_graph_types=["person", "device"],
        )
        issues = config.validate()
        assert len(issues) == 0

    def test_validate_embedding_sim_without_embeddings(self):
        config = GraphRAGSchemaConfig(
            canonicalize_strategy=CanonicalizeStrategy.EMBEDDING_SIM,
        )
        issues = config.validate()
        assert len(issues) > 0
        assert "EMBEDDING_SIM" in issues[0]

    def test_validate_threshold_values(self):
        config = GraphRAGSchemaConfig(
            canonicalize_similarity_threshold=0.60,
            canonicalize_suggest_threshold=0.80,
            relationship_graph_types=["person", "device"],  # satisfy EMBEDDING_SIM check
        )
        issues = config.validate()
        threshold_issues = [i for i in issues if "should be >= " in i]
        assert len(threshold_issues) > 0

    def test_to_dict(self):
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.EVOLVING,
            relationship_graph_types=["person", "device"],
        )
        d = config.to_dict()
        assert d["mode"] == "evolving"
        assert d["relationship_graph_types_count"] == 2

    def test_from_dict(self):
        d = {
            "mode": "guided",
            "canonicalize_strategy": "exact_match",
            "relationship_graph_types": ["person", "device"],
        }
        config = GraphRAGSchemaConfig.from_dict(d)
        assert config.mode == SchemaMode.GUIDED
        assert config.canonicalize_strategy == CanonicalizeStrategy.EXACT_MATCH
        assert len(config.relationship_graph_types) == 2

    def test_merge_from_context(self):
        config = GraphRAGSchemaConfig()
        context = {
            "graph_rag_schema_mode": "guided",
            "relationship_graph_types": ["person", "company"],
            "known_type_registry": {"person": {"description": "A human being"}},
        }
        config.merge_from_context(context)
        assert config.mode == SchemaMode.GUIDED
        assert len(config.relationship_graph_types) == 2
        assert "person" in config.known_type_registry

    def test_run_operator_protocol(self):
        config = GraphRAGSchemaConfig(
            relationship_graph_types=["person", "device"],
        )
        context = {}
        result = config.run(context)
        assert "graph_rag_schema_config" in result
        assert result["graph_rag_schema_mode"] == "evolving"
        assert result["graph_rag_schema_config"] is config


# ============================================================
# KGSchemaDefineOperator Tests
# ============================================================

class TestKGSchemaDefineOperator:
    """Tests for EDC Define phase operator."""

    def test_skip_in_guided_mode(self):
        llm = MockLLM()
        config = GraphRAGSchemaConfig(mode=SchemaMode.GUIDED)
        op = KGSchemaDefineOperator(llm=llm, config=config)
        context = {"extracted_entities": [{"name": "Alice", "type": "person"}]}
        result = op.run(context)
        assert result["type_definitions"] == {}
        assert result["define_call_count"] == 0

    def test_no_new_types_skips_define(self):
        llm = MockLLM()
        config = GraphRAGSchemaConfig(
            known_type_registry={"person": {"description": "A human"}},
        )
        op = KGSchemaDefineOperator(llm=llm, config=config)
        context = {
            "extracted_entities": [{"name": "Alice", "type": "person"}],
            "known_type_registry": {"person": {"description": "A human"}},
        }
        result = op.run(context)
        assert result["define_call_count"] == 0

    def test_define_single_new_entity_type(self):
        llm_response = json.dumps({
            "description": "A corporation or business entity",
            "properties": [
                {"name": "name", "type": "string", "cardinality": "single", "required": True},
                {"name": "founded_year", "type": "integer", "cardinality": "optional", "required": False},
            ],
            "parent_types": ["Entity", "organization"],
        })
        llm = MockLLM(responses=[llm_response])
        config = GraphRAGSchemaConfig()
        op = KGSchemaDefineOperator(llm=llm, config=config)

        context = {
            "extracted_entities": [
                {"name": "摩拜单车", "type": "company"},
            ],
            "chunks": ["摩拜单车是一家共享单车公司"],
        }
        result = op.run(context)
        assert "company" in result["type_definitions"]
        assert result["define_call_count"] == 1
        assert result["type_definitions"]["company"]["description"] == "A corporation or business entity"

    def test_define_batch_types(self):
        # Create 7 new types (above the batch threshold of 5)
        definitions = {
            "definitions": [
                {
                    "type_name": "corporation",
                    "type_category": "entity",
                    "description": "A large business entity",
                    "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
                    "parent_types": ["Entity"],
                },
                {
                    "type_name": "startup",
                    "type_category": "entity",
                    "description": "A newly founded business",
                    "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
                    "parent_types": ["Entity"],
                },
                {
                    "type_name": "founded_by",
                    "type_category": "relation",
                    "description": "Entity founded by another entity",
                    "source_types": ["corporation", "startup"],
                    "target_types": ["person"],
                    "properties": [],
                },
                {
                    "type_name": "product",
                    "type_category": "entity",
                    "description": "A product or service",
                    "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
                    "parent_types": ["Entity"],
                },
                {
                    "type_name": "technology",
                    "type_category": "entity",
                    "description": "A technology or technical concept",
                    "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
                    "parent_types": ["Entity"],
                },
                {
                    "type_name": "invests_in",
                    "type_category": "relation",
                    "description": "Investment relation",
                    "source_types": ["person", "corporation"],
                    "target_types": ["corporation", "startup"],
                    "properties": [],
                },
                {
                    "type_name": "located_in",
                    "type_category": "relation",
                    "description": "Geographic location relation",
                    "source_types": ["Entity"],
                    "target_types": ["city", "country"],
                    "properties": [],
                },
            ]
        }
        llm = MockLLM(responses=[json.dumps(definitions)])
        config = GraphRAGSchemaConfig()
        op = KGSchemaDefineOperator(llm=llm, config=config)

        context = {
            "extracted_entities": [
                {"name": "摩拜", "type": "corporation"},
                {"name": "ofo", "type": "startup"},
                {"name": "bike", "type": "product"},
                {"name": "AI", "type": "technology"},
            ],
            "extracted_relations": [
                {"subject": "胡玮炜", "predicate": "founded_by", "object": "摩拜"},
                {"subject": "腾讯", "predicate": "invests_in", "object": "摩拜"},
                {"subject": "摩拜", "predicate": "located_in", "object": "北京"},
            ],
            "chunks": [],
        }
        result = op.run(context)
        assert "corporation" in result["type_definitions"]
        assert "startup" in result["type_definitions"]
        assert "founded_by" in result["type_definitions"]
        assert result["define_call_count"] == 1  # Batch uses 1 call

    def test_define_trigger_threshold_policy(self):
        llm = MockLLM(responses=['{"description": "mock", "properties": [], "parent_types": []}'])
        config = GraphRAGSchemaConfig(
            define_trigger_policy=DefineTriggerPolicy.THRESHOLD,
            define_threshold_ratio=0.50,
            known_type_registry={"person": {"description": "A human"}},
        )
        op = KGSchemaDefineOperator(llm=llm, config=config)

        # 1 new type out of 2 total = 50% → exactly at threshold, should trigger
        context = {
            "extracted_entities": [
                {"name": "Alice", "type": "person"},
                {"name": "摩拜", "type": "company"},
            ],
            "known_type_registry": {"person": {"description": "A human"}},
        }
        result = op.run(context)
        assert result["define_call_count"] >= 1

    def test_define_trigger_threshold_below_ratio(self):
        """THRESHOLD policy: only trigger Define when new_type ratio exceeds threshold.

        Note: ratio is computed on UNIQUE types, not on entity instances.
        e.g. 10 entities but only 2 unique types → ratio is based on 2 types.
        """
        llm = MockLLM()
        config = GraphRAGSchemaConfig(
            define_trigger_policy=DefineTriggerPolicy.THRESHOLD,
            define_threshold_ratio=0.50,
            known_type_registry={
                "person": {"description": "A human"},
                "device": {"description": "A device"},
                "location": {"description": "A location"},
                "date": {"description": "A date"},
                "organization": {"description": "An org"},
            },
        )
        op = KGSchemaDefineOperator(llm=llm, config=config)

        # 5 known types + 1 new type ("corporation") = 6 total unique types
        # new ratio = 1/6 ≈ 16.7% → below 50% threshold
        entities = [
            {"name": "Alice", "type": "person"},
            {"name": "Phone", "type": "device"},
            {"name": "北京", "type": "location"},
            {"name": "2024", "type": "date"},
            {"name": "腾讯", "type": "organization"},
            {"name": "摩拜", "type": "corporation"},  # new type
        ]
        context = {
            "extracted_entities": entities,
            "known_type_registry": config.known_type_registry,
        }
        result = op.run(context)
        # 1/6 = 16.7% < 50% → should NOT trigger Define
        assert result["type_definitions"] == {}

    def test_define_fallback_on_llm_failure(self):
        llm = MockLLM(responses=None)  # Will raise AttributeError
        # Actually MockLLM won't raise, it returns default. Let me make it raise.
        failing_llm = MagicMock()
        failing_llm.generate.side_effect = RuntimeError("LLM unavailable")

        config = GraphRAGSchemaConfig()
        op = KGSchemaDefineOperator(llm=failing_llm, config=config)

        context = {
            "extracted_entities": [{"name": "摩拜", "type": "company"}],
        }
        result = op.run(context)
        # Should have minimal definition as fallback
        assert "company" in result["type_definitions"]
        assert result["type_definitions"]["company"]["type_category"] == "entity"

    def test_is_relation_type_heuristic(self):
        op = KGSchemaDefineOperator(llm=MockLLM())
        # Should detect relation patterns
        assert op._is_relation_type("founded_by") is True
        assert op._is_relation_type("works_at") is True
        assert op._is_relation_type("located_in") is True
        assert op._is_relation_type("belongs_to") is True
        assert op._is_relation_type("has_chunk") is True
        # Should NOT flag entity types
        assert op._is_relation_type("person") is False
        assert op._is_relation_type("company") is False
        assert op._is_relation_type("device") is False

    def test_registry_update_across_runs(self):
        """Verify that the known_type_registry accumulates across runs."""
        llm_response1 = json.dumps({
            "description": "A corporation",
            "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
            "parent_types": ["Entity"],
        })
        llm_response2 = json.dumps({
            "description": "A city",
            "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
            "parent_types": ["Entity", "location"],
        })

        # Run 1: Define "company"
        llm1 = MockLLM(responses=[llm_response1])
        config = GraphRAGSchemaConfig()
        op1 = KGSchemaDefineOperator(llm=llm1, config=config)
        context1 = {
            "extracted_entities": [{"name": "摩拜", "type": "company"}],
        }
        result1 = op1.run(context1)
        assert "company" in result1["known_type_registry"]

        # Run 2: Define "city" with accumulated registry
        llm2 = MockLLM(responses=[llm_response2])
        config2 = GraphRAGSchemaConfig(
            known_type_registry=result1["known_type_registry"],
        )
        op2 = KGSchemaDefineOperator(llm=llm2, config=config2)
        context2 = {
            "extracted_entities": [
                {"name": "Alice", "type": "person"},
                {"name": "北京", "type": "city"},
            ],
            "known_type_registry": result1["known_type_registry"],
        }
        result2 = op2.run(context2)
        # "company" should still be in registry, "city" should be newly defined
        assert "company" in result2["known_type_registry"]
        assert "city" in result2["known_type_registry"]


# ============================================================
# KGSchemaCanonicalizeOperator Tests
# ============================================================

class TestKGSchemaCanonicalizeOperator:
    """Tests for EDC Canonicalize phase operator."""

    def test_skip_in_guided_mode(self):
        config = GraphRAGSchemaConfig(mode=SchemaMode.GUIDED)
        op = KGSchemaCanonicalizeOperator(config=config)
        context = {"raw_types": ["company", "person"]}
        result = op.run(context)
        assert result["canonicalized_types"] == {}

    def test_exact_match_strategy(self):
        config = GraphRAGSchemaConfig(
            canonicalize_strategy=CanonicalizeStrategy.EXACT_MATCH,
            relationship_graph_types=["person", "device", "ip_address", "company"],
        )
        op = KGSchemaCanonicalizeOperator(config=config)
        context = {
            "raw_types": ["person", "company", "corporation", "IP_Address"],
            "relationship_graph_types": ["person", "device", "ip_address", "company"],
        }
        result = op.run(context)
        ct = result["canonicalized_types"]
        assert ct["person"] == "person"  # Exact match
        assert ct["company"] == "company"  # Exact match
        assert ct["corporation"] == "corporation"  # No match, keep as-is
        assert ct["IP_Address"] == "ip_address"  # Case-insensitive match

    def test_embedding_similarity_strategy(self):
        # Pre-compute embeddings for relationship graph types
        rel_types = ["person", "device", "ip_address", "company"]
        rel_embeddings = {t: simple_embed_func(t) for t in rel_types}

        config = GraphRAGSchemaConfig(
            canonicalize_strategy=CanonicalizeStrategy.EMBEDDING_SIM,
            canonicalize_similarity_threshold=0.85,
            canonicalize_suggest_threshold=0.70,
            relationship_graph_types=rel_types,
            relationship_graph_type_embeddings=rel_embeddings,
        )
        op = KGSchemaCanonicalizeOperator(
            embed_func=simple_embed_func,
            config=config,
        )

        # Type definitions from Define phase
        type_definitions = {
            "corporation": {"description": "A corporation or business entity"},
            "individual": {"description": "A human individual, person"},
        }

        context = {
            "raw_types": ["corporation", "individual", "device"],
            "type_definitions": type_definitions,
            "relationship_graph_types": rel_types,
            "relationship_graph_type_embeddings": rel_embeddings,
        }
        result = op.run(context)

        # "device" should match exactly with "device" (same embedding)
        # "corporation" and "individual" depend on embedding similarity
        assert "corporation" in result["canonicalized_types"]
        assert "individual" in result["canonicalized_types"]
        assert "device" in result["canonicalized_types"]

    def test_embedding_similarity_no_embeddings_fallback(self):
        """When no embeddings are available, fall back to EXACT_MATCH."""
        config = GraphRAGSchemaConfig(
            canonicalize_strategy=CanonicalizeStrategy.EMBEDDING_SIM,
            relationship_graph_types=["person", "device"],
        )
        op = KGSchemaCanonicalizeOperator(config=config)
        context = {
            "raw_types": ["person", "corporation"],
            "relationship_graph_types": ["person", "device"],
        }
        result = op.run(context)
        # Should fall back to EXACT_MATCH
        assert result["canonicalized_types"]["person"] == "person"
        assert result["canonicalized_types"]["corporation"] == "corporation"

    def test_llm_classify_strategy(self):
        llm_response = json.dumps({
            "classifications": [
                {"new_type": "corporation", "classified_as": "company", "confidence": 0.95, "reason": "Corporation is a type of company"},
                {"new_type": "individual", "classified_as": "person", "confidence": 0.90, "reason": "Individual refers to a person"},
                {"new_type": "novel_type", "classified_as": "NEW", "confidence": 0.30, "reason": "No clear match"},
            ]
        })
        llm = MockLLM(responses=[llm_response])
        config = GraphRAGSchemaConfig(
            canonicalize_strategy=CanonicalizeStrategy.LLM_CLASSIFY,
            relationship_graph_types=["person", "device", "company", "ip_address"],
        )
        op = KGSchemaCanonicalizeOperator(llm=llm, config=config)

        context = {
            "raw_types": ["corporation", "individual", "novel_type"],
            "relationship_graph_types": ["person", "device", "company", "ip_address"],
            "type_definitions": {
                "corporation": {"description": "A corporation"},
                "individual": {"description": "An individual person"},
                "novel_type": {"description": "Something novel"},
            },
        }
        result = op.run(context)
        ct = result["canonicalized_types"]
        assert ct["corporation"] == "company"  # LLM classified as company
        assert ct["individual"] == "person"  # LLM classified as person
        assert ct["novel_type"] == "novel_type"  # LLM said NEW, keep as-is

    def test_cosine_similarity(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        assert KGSchemaCanonicalizeOperator._cosine_similarity(vec_a, vec_b) == 1.0

        vec_c = [0.0, 1.0, 0.0]
        assert KGSchemaCanonicalizeOperator._cosine_similarity(vec_a, vec_c) == 0.0

        vec_d = [1.0, 1.0, 0.0]
        expected = 1.0 / math.sqrt(2.0)
        assert abs(KGSchemaCanonicalizeOperator._cosine_similarity(vec_a, vec_d) - expected) < 0.001

    def test_apply_canonicalization_to_entities(self):
        config = GraphRAGSchemaConfig(preserve_raw_type=True)
        op = KGSchemaCanonicalizeOperator(config=config)

        canonicalized = {
            "corporation": "company",
            "person": "person",
        }
        context = {
            "extracted_entities": [
                {"name": "摩拜", "type": "corporation"},
                {"name": "Alice", "type": "person"},
            ],
            "vertices": [
                {"label": "corporation", "properties": {"name": "摩拜"}},
            ],
        }
        op._apply_canonicalization_to_entities(context, canonicalized)

        # "corporation" should be renamed to "company" with raw_type preserved
        assert context["extracted_entities"][0]["type"] == "company"
        assert context["extracted_entities"][0]["raw_type"] == "corporation"
        # "person" stays the same
        assert context["extracted_entities"][1]["type"] == "person"

        # Vertex label should also be updated
        assert context["vertices"][0]["label"] == "company"
        assert context["vertices"][0]["raw_label"] == "corporation"

    def test_no_relationship_types_warning(self):
        config = GraphRAGSchemaConfig()
        op = KGSchemaCanonicalizeOperator(config=config)
        context = {"raw_types": ["company", "person"]}
        result = op.run(context)
        # Without rel_types, each type maps to itself
        assert result["canonicalized_types"]["company"] == "company"


# ============================================================
# GuidedExtractOperator & ResponseModel Tests
# ============================================================

class TestGuidedResponseModelBuilder:
    """Tests for Pydantic ResponseModel builder."""

    def test_build_with_vertex_types(self):
        builder = GuidedResponseModelBuilder(
            vertex_types=["person", "device", "ip_address"],
            edge_types=["uses", "owns", "located_at"],
        )
        model = builder.build()
        # Model should have entities and relations fields
        assert hasattr(model, '__fields__') or hasattr(model, 'model_fields')

    def test_build_property_schema(self):
        schema_dict = {
            "vertexlabels": [
                {
                    "name": "person",
                    "properties": ["name", "age", "occupation"],
                    "primary_keys": ["name"],
                },
            ],
            "edgelabels": [
                {
                    "name": "uses",
                    "source_label": "person",
                    "target_label": "device",
                    "properties": ["since"],
                },
            ],
        }
        builder = GuidedResponseModelBuilder(
            vertex_types=["person", "device"],
            edge_types=["uses"],
            schema_dict=schema_dict,
        )
        props_map = builder._build_vertex_properties_map()
        assert "person" in props_map
        assert "name" in props_map["person"]

    def test_guided_entity_validation(self):
        """Test that GuidedEntity validates label."""
        entity = GuidedEntity(label="person", name="Alice")
        assert entity.label == "person"
        assert entity.name == "Alice"

    def test_guided_entity_empty_label_fails(self):
        with pytest.raises(Exception):
            GuidedEntity(label="", name="Alice")

    def test_guided_relation_validation(self):
        rel = GuidedRelation(
            label="uses",
            source_label="person",
            source_name="Alice",
            target_label="device",
            target_name="Phone",
        )
        assert rel.label == "uses"


class TestGuidedExtractOperator:
    """Tests for guided mode extraction operator."""

    def test_skip_in_evolving_mode(self):
        config = GraphRAGSchemaConfig(mode=SchemaMode.EVOLVING)
        op = GuidedExtractOperator(llm=MockLLM(), config=config)
        context = {"chunks": ["some text"]}
        result = op.run(context)
        # Should skip (not GUIDED mode)
        assert "vertices" not in result or len(result.get("vertices", [])) == 0

    def test_guided_extraction_with_schema(self):
        llm_response = json.dumps({
            "entities": [
                {"label": "person", "name": "Alice", "properties": {"occupation": "engineer"}},
                {"label": "device", "name": "Phone", "properties": {}},
            ],
            "relations": [
                {"label": "uses", "source_label": "person", "source_name": "Alice",
                 "target_label": "device", "target_name": "Phone", "properties": {}},
            ],
        })
        llm = MockLLM(responses=[llm_response])
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.GUIDED,
            relationship_graph_types=["person", "device", "ip_address"],
        )

        schema_dict = {
            "vertexlabels": [
                {"name": "person", "properties": ["name", "occupation"], "primary_keys": ["name"],
                 "id_strategy": "PRIMARY_KEY", "id": "1"},
                {"name": "device", "properties": ["name", "type"], "primary_keys": ["name"],
                 "id_strategy": "PRIMARY_KEY", "id": "2"},
            ],
            "edgelabels": [
                {"name": "uses", "source_label": "person", "target_label": "device",
                 "properties": ["since"]},
            ],
        }

        op = GuidedExtractOperator(llm=llm, config=config)
        context = {
            "chunks": ["Alice uses a Phone for work"],
            "schema": schema_dict,
            "relationship_graph_types": ["person", "device", "ip_address"],
        }
        result = op.run(context)
        assert len(result["vertices"]) >= 1
        assert len(result["edges"]) >= 1
        assert result["call_count"] == 1

    def test_guided_extraction_deduplication(self):
        llm_response = json.dumps({
            "entities": [
                {"label": "person", "name": "Alice", "properties": {}},
                {"label": "person", "name": "Alice", "properties": {"age": 30}},
            ],
            "relations": [],
        })
        llm = MockLLM(responses=[llm_response])
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.GUIDED,
            relationship_graph_types=["person"],
        )
        op = GuidedExtractOperator(llm=llm, config=config)
        context = {
            "chunks": ["Alice is 30 years old"],
            "schema": {
                "vertexlabels": [
                    {"name": "person", "properties": ["name", "age"], "primary_keys": ["name"],
                     "id_strategy": "PRIMARY_KEY", "id": "1"},
                ],
                "edgelabels": [],
            },
        }
        result = op.run(context)
        # Deduplication should reduce 2 identical entities to 1
        assert len(result["vertices"]) == 1


# ============================================================
# EDCPipelineOrchestrator Tests
# ============================================================

class TestEDCPipelineOrchestrator:
    """Tests for the full EDC + Guided pipeline orchestrator."""

    def test_evolving_mode_full_pipeline(self):
        """Test the full EDC pipeline: Config → Extract post-process → Define → Canonicalize."""
        define_response = json.dumps({
            "description": "A business corporation",
            "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
            "parent_types": ["Entity", "organization"],
        })

        llm = MockLLM(responses=[define_response])
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.EVOLVING,
            canonicalize_strategy=CanonicalizeStrategy.EXACT_MATCH,
            relationship_graph_types=["person", "device", "company", "ip_address"],
        )

        # Pre-compute embeddings for canonicalize
        rel_embeddings = {t: simple_embed_func(t) for t in config.relationship_graph_types}

        orchestrator = EDCPipelineOrchestrator(
            llm=llm,
            embed_func=simple_embed_func,
            config=config,
        )

        context = {
            "chunks": ["摩拜单车是一家共享单车公司"],
            "extracted_entities": [
                {"name": "摩拜单车", "type": "company"},
                {"name": "胡玮炜", "type": "person"},
            ],
            "vertices": [
                {"label": "company", "properties": {"name": "摩拜单车"}},
                {"label": "person", "properties": {"name": "胡玮炜"}},
            ],
            "relationship_graph_types": config.relationship_graph_types,
            "relationship_graph_type_embeddings": rel_embeddings,
        }

        result = orchestrator.run(context)

        # Verify pipeline outputs
        assert "raw_types" in result
        assert "company" in result["raw_types"]
        assert "person" in result["raw_types"]

        # Define should have been called for "person" and "company" (both new)
        # In EXACT_MATCH canonicalize: "person" → "person", "company" → "company"
        assert "canonicalized_types" in result
        assert result["canonicalized_types"]["company"] == "company"

    def test_guided_mode_pipeline(self):
        """Test the Guided mode pipeline."""
        llm_response = json.dumps({
            "entities": [
                {"label": "person", "name": "Alice", "properties": {}},
            ],
            "relations": [],
        })
        llm = MockLLM(responses=[llm_response])
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.GUIDED,
            relationship_graph_types=["person", "device", "ip_address"],
        )

        orchestrator = EDCPipelineOrchestrator(llm=llm, config=config)

        context = {
            "chunks": ["Alice uses a device"],
            "schema": {
                "vertexlabels": [
                    {"name": "person", "properties": ["name"], "primary_keys": ["name"],
                     "id_strategy": "PRIMARY_KEY", "id": "1"},
                ],
                "edgelabels": [],
            },
        }
        result = orchestrator.run(context)
        assert result["graph_rag_schema_mode"] == "guided"

    def test_extract_post_process_preserves_raw_types(self):
        """Verify that post-processing preserves raw_type alongside normalized labels."""
        config = GraphRAGSchemaConfig(mode=SchemaMode.EVOLVING)
        orchestrator = EDCPipelineOrchestrator(llm=MockLLM(), config=config)

        context = {
            "extracted_entities": [
                {"name": "Alice", "type": "person"},
                {"name": "摩拜", "type": "company"},
            ],
            "vertices": [
                {"label": "person", "properties": {"name": "Alice"}},
            ],
            "edges": [
                {"label": "uses"},
            ],
        }
        result = orchestrator._extract_post_process(context)

        assert "raw_types" in result
        assert "person" in result["raw_types"]
        assert "company" in result["raw_types"]

        # raw_type should be added to entities
        assert result["extracted_entities"][0]["raw_type"] == "person"
        assert result["vertices"][0]["raw_label"] == "person"


# ============================================================
# SchemaMode / CanonicalizeStrategy Enum Tests
# ============================================================

class TestSchemaEnums:
    """Tests for schema-related enums."""

    def test_schema_mode_values(self):
        assert SchemaMode.EVOLVING.value == "evolving"
        assert SchemaMode.GUIDED.value == "guided"

    def test_canonicalize_strategy_values(self):
        assert CanonicalizeStrategy.EMBEDDING_SIM.value == "embedding_sim"
        assert CanonicalizeStrategy.EXACT_MATCH.value == "exact_match"
        assert CanonicalizeStrategy.LLM_CLASSIFY.value == "llm_classify"

    def test_define_trigger_policy_values(self):
        assert DefineTriggerPolicy.NEW_TYPES_ONLY.value == "new_types_only"
        assert DefineTriggerPolicy.ALWAYS.value == "always"
        assert DefineTriggerPolicy.THRESHOLD.value == "threshold"

    def test_schema_mode_from_string(self):
        assert SchemaMode("evolving") == SchemaMode.EVOLVING
        assert SchemaMode("guided") == SchemaMode.GUIDED

    def test_invalid_schema_mode_raises(self):
        with pytest.raises(ValueError):
            SchemaMode("invalid_mode")


# ============================================================
# Cross-Integration Tests
# ============================================================

class TestEDCIntegration:
    """Integration tests spanning multiple EDC components."""

    def test_evolving_pipeline_with_known_registry(self):
        """Test that a populated registry skips Define for known types."""
        define_response = json.dumps({
            "description": "A novel concept",
            "properties": [{"name": "name", "type": "string", "cardinality": "single", "required": True}],
            "parent_types": ["Entity"],
        })
        llm = MockLLM(responses=[define_response])
        config = GraphRAGSchemaConfig(
            mode=SchemaMode.EVOLVING,
            canonicalize_strategy=CanonicalizeStrategy.EXACT_MATCH,
            known_type_registry={
                "person": {"description": "A human being", "type_category": "entity"},
                "device": {"description": "A physical device", "type_category": "entity"},
            },
            relationship_graph_types=["person", "device", "ip_address"],
        )

        orchestrator = EDCPipelineOrchestrator(llm=llm, config=config)

        context = {
            "chunks": ["Alice uses a device. A novel concept appeared."],
            "extracted_entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Phone", "type": "device"},
                {"name": "QuantumComputing", "type": "novel_concept"},
            ],
            "known_type_registry": config.known_type_registry,
            "relationship_graph_types": config.relationship_graph_types,
        }
        result = orchestrator.run(context)

        # "person" and "device" should NOT trigger Define (already known)
        # "novel_concept" should trigger Define
        assert "novel_concept" in result.get("type_definitions", {})

        # Canonicalize: "person" → "person" (exact match)
        # "novel_concept" → "novel_concept" (no exact match in rel_types)
        ct = result["canonicalized_types"]
        assert ct.get("person") == "person"
        assert ct.get("novel_concept") == "novel_concept"
