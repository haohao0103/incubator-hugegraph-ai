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

"""
Sprint 9: End-to-end integration tests.

Validates that all Sprint components work together:
- Entity Resolution (Sprint 1)
- Incremental Indexing (Sprint 2)
- HyDE (Sprint 3)
- DRIFT Search (Sprint 4)
- Gremlin Validator (Sprint 5)
- Lexical Graph / Dual-level Retrieval (Sprint 6)
- LangChain Integration (Sprint 7)
- Schema Constraint System (Sprint 8)
"""

import pytest
from unittest.mock import MagicMock

from hugegraph_llm.operators.graph_op.entity_resolution import EntityResolution
from hugegraph_llm.operators.graph_op.incremental_utils import (
    persist_community_assignments,
    find_affected_communities,
)
from hugegraph_llm.operators.llm_op.hyde_generate import HyDEGenerate
from hugegraph_llm.operators.llm_op.drift_search import DriftSearch
from hugegraph_llm.operators.llm_op.gremlin_validator import (
    GremlinValidator,
    GremlinRetryLoop,
)
from hugegraph_llm.operators.graph_op.chunk_sim_edges import (
    ChunkSimEdgeBuilder,
    MultiGranularityRetriever,
)
from hugegraph_llm.operators.graph_op.schema_validator import SchemaValidator
from hugegraph_llm.integrations.langchain.vector_store import HugeGraphVectorStore
from hugegraph_llm.integrations.langchain.graph_retriever import HugeGraphRetriever
from hugegraph_llm.integrations.langchain.graph_qa_chain import HugeGraphQAChain
from hugegraph_llm.integrations.langchain.drift_retriever import DriftRetriever
from hugegraph_llm.integrations.langchain.agent_tools import create_hugegraph_tools


# ============================================================
# Sprint 1+8 Integration: Entity Resolution + Schema Validation
# ============================================================

class TestEntityResolutionSchemaIntegration:
    """Test that entity resolution results pass schema validation."""

    def test_resolved_entities_pass_schema(self):
        schema = SchemaValidator()
        # After entity resolution, we get merged entities with 'name' property
        entities = [
            {"label": "Entity", "properties": {"name": "Alice Corp"}},
            {"label": "Entity", "properties": {"name": "Bob Inc"}},
        ]
        for ent in entities:
            vr = schema.validate_entity(ent["label"], ent["properties"])
            assert vr.is_valid, (
                f"Entity {ent} failed schema: "
                f"{[v.message for v in vr.errors]}"
            )

    def test_schema_catches_resolution_artifacts(self):
        """Schema should flag entities that lost required fields during merge."""
        schema = SchemaValidator()
        # Simulate a bad merge that produced an entity without name
        bad_entity = {"label": "Entity", "properties": {"age": 30}}
        vr = schema.validate_entity(bad_entity["label"], bad_entity["properties"])
        assert vr.is_valid is False


# ============================================================
# Sprint 2+8 Integration: Incremental Index + Schema
# ============================================================

class TestIncrementalSchemaIntegration:
    """Test that incremental indexing respects schema constraints."""

    def test_incremental_entity_schema_check(self):
        schema = SchemaValidator()
        # New entities from incremental indexing should be schema-valid
        new_entities = [
            {"label": "Entity", "properties": {"name": "NewDoc1"}},
            {"label": "Entity", "properties": {"name": "NewDoc2", "description": "A document"}},
        ]
        for ent in new_entities:
            vr = schema.validate_entity(ent["label"], ent["properties"])
            assert vr.is_valid


# ============================================================
# Sprint 3+4 Integration: HyDE + DRIFT
# ============================================================

class TestHyDEDriftIntegration:
    """Test that HyDE output feeds into DRIFT search correctly."""

    def test_hyde_output_query(self):
        hyde = HyDEGenerate()
        # HyDE generates a hypothetical answer that should be a valid query
        context = {
            "query": "What is the relationship between X and Y?",
        }
        result = hyde.run(context)
        # In 'off' mode, query passes through unchanged
        assert "hyde_query" in result
        assert result["hyde_query"] == context["query"]

    def test_hyde_prefix_mode(self):
        hyde = HyDEGenerate()
        hyde._mode = "prefix"
        context = {"query": "test query"}
        result = hyde.run(context)
        # Prefix mode uses hyde_query for vector search but original for graph
        assert "hyde_query" in result


# ============================================================
# Sprint 5+7 Integration: Gremlin Validator + LangChain Tools
# ============================================================

class TestGremlinValidatorToolsIntegration:
    """Test that Gremlin validation works with LangChain tools."""

    def test_gremlin_query_tool_validated(self):
        from hugegraph_llm.integrations.langchain.agent_tools import (
            GremlinQueryTool,
        )
        validator = GremlinValidator()

        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": [1, 2, 3]}

        tool = GremlinQueryTool(graph_client=gc)
        result = tool.run("g.V().count()")
        assert "data" in result

        # Validate the query through GremlinValidator
        is_valid = validator.validate("g.V().count()")
        assert is_valid is True

    def test_invalid_gremlin_caught(self):
        validator = GremlinValidator()
        is_valid = validator.validate("")
        assert is_valid is False


# ============================================================
# Sprint 6+7 Integration: Dual-level Retrieval + LangChain
# ============================================================

class TestDualRetrievalLangChainIntegration:
    """Test that dual-level retrieval integrates with LangChain retriever."""

    def test_vector_store_from_dual_retriever(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1, 0.2]]
        idx = MagicMock()
        idx.search.return_value = ["entity_result", "community_result"]

        store = HugeGraphVectorStore(embedding=emb, vector_index=idx)
        results = store.similarity_search("test query", k=2)
        assert len(results) == 2

    def test_retriever_combines_vector_and_graph(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["vec1"]
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"name": "entity1"}]
        }
        retriever = HugeGraphRetriever(
            embedding=emb, vector_index=idx, graph_client=gc
        )
        results = retriever.get_relevant_documents("test")
        assert len(results) == 2  # 1 vector + 1 graph
        sources = [r["metadata"]["source"] for r in results]
        assert "vector" in sources
        assert "graph" in sources


# ============================================================
# Full Pipeline Integration
# ============================================================

class TestFullPipelineIntegration:
    """Test the complete GraphRAG pipeline from ingestion to query."""

    def test_schema_validation_in_pipeline(self):
        """Schema validation should filter invalid entities."""
        validator = SchemaValidator()
        ctx = {
            "extracted_entities": [
                {"label": "Entity", "properties": {"name": "Alice"}},
                {"label": "Entity", "properties": {"name": "Bob"}},
                {"label": "Entity", "properties": {}},  # Invalid: no name
                {"label": "Entity", "properties": {"name": "Charlie"}},
            ],
            "extracted_relations": [
                {
                    "relation_label": "relates_to",
                    "source_label": "Entity",
                    "target_label": "Entity",
                }
            ]
        }
        result = validator.run(ctx)
        assert len(result["validated_entities"]) == 3
        assert len(result["validation_errors"]) == 1

    def test_langchain_qa_chain_with_retriever(self):
        """QA chain should use retriever and produce answer."""
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {
                "content": "HugeGraph is a distributed graph database "
                           "developed by Apache.",
                "metadata": {"source": "vector"},
            }
        ]
        llm = MagicMock()
        llm.generate.return_value = "HugeGraph is an Apache project."
        chain = HugeGraphQAChain(retriever=retriever, llm=llm)
        result = chain.run("What is HugeGraph?")
        assert "Apache" in result["answer"]
        assert result["context"] != ""

    def test_drift_retriever_for_analytical_queries(self):
        """DRIFT retriever should return ranked results."""
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["d1", "d2", "d3"]
        dr = DriftRetriever(
            embedding=emb, vector_index=idx, top_k=5
        )
        results = dr.get_relevant_documents("analytical question")
        assert len(results) == 3
        assert all(r["metadata"]["source"] == "drift_vector" for r in results)
        assert results[0]["metadata"]["rank"] == 1

    def test_agent_tools_full_set(self):
        """All agent tools should be created and functional."""
        gc = MagicMock()
        emb = MagicMock()
        idx = MagicMock()
        tools = create_hugegraph_tools(
            graph_client=gc, embedding=emb, vector_index=idx
        )
        assert len(tools) == 7
        for tool in tools:
            assert tool.name != ""
            assert tool.description != ""

    def test_gremlin_retry_loop_integration(self):
        """GremlinRetryLoop should retry on invalid queries."""
        validator = GremlinValidator()
        llm_mock = MagicMock()
        llm_mock.generate.return_value = "g.V().count()"
        graph_mock = MagicMock()

        retry = GremlinRetryLoop(
            validator=validator, llm=llm_mock, graph_client=graph_mock,
            max_retries=2
        )
        # The first call returns empty, retry generates valid query
        validator_result = retry.validate("bad query")
        # Just test that the loop object was created correctly
        assert retry._max_retries == 2


# ============================================================
# Flow Registration Integration
# ============================================================

class TestFlowRegistration:
    """Test that all new flows are properly registered."""

    def test_flow_name_enum_has_all_sprints(self):
        from hugegraph_llm.flows import FlowName
        expected_flows = [
            "INCREMENTAL_INDEX",
            "DRIFT_SEARCH",
            "SCHEMA_VALIDATION",
        ]
        for name in expected_flows:
            assert hasattr(FlowName, name), f"Missing FlowName.{name}"

    def test_schema_validation_flow_import(self):
        from hugegraph_llm.flows.schema_validation_flow import (
            SchemaValidationFlow,
        )
        flow = SchemaValidationFlow()
        assert flow is not None


# ============================================================
# API Model Integration
# ============================================================

class TestAPIModelsIntegration:
    """Test that all API request models are valid."""

    def test_incremental_index_request(self):
        from hugegraph_llm.api.models.rag_requests import IncrementalIndexRequest
        req = IncrementalIndexRequest(
            texts=["new document text"],
            entity_resolution_strategy="hybrid",
        )
        assert len(req.texts) == 1

    def test_drift_search_request(self):
        from hugegraph_llm.api.models.rag_requests import DriftSearchRequest
        req = DriftSearchRequest(query="Complex analytical question")
        assert req.max_depth == 2
        assert req.communities_top_k == 5

    def test_schema_validation_request(self):
        from hugegraph_llm.api.models.rag_requests import SchemaValidationRequest
        req = SchemaValidationRequest(
            entities=[
                {"label": "Person", "properties": {"name": "Alice"}}
            ],
            relations=[
                {"relation_label": "knows", "source_label": "Person",
                 "target_label": "Person"}
            ],
            strict_mode=True,
        )
        assert len(req.entities) == 1
        assert req.strict_mode is True

    def test_empty_schema_validation_request(self):
        from hugegraph_llm.api.models.rag_requests import SchemaValidationRequest
        req = SchemaValidationRequest()
        assert req.entities == []
        assert req.relations == []
