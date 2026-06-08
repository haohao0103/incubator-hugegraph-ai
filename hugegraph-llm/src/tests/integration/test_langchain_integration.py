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

"""Tests for Sprint 7 LangChain integration components."""

import pytest
from unittest.mock import MagicMock, patch

from hugegraph_llm.integrations.langchain.vector_store import HugeGraphVectorStore
from hugegraph_llm.integrations.langchain.graph_retriever import HugeGraphRetriever
from hugegraph_llm.integrations.langchain.graph_qa_chain import HugeGraphQAChain
from hugegraph_llm.integrations.langchain.drift_retriever import DriftRetriever
from hugegraph_llm.integrations.langchain.agent_tools import (
    GremlinQueryTool,
    VectorSearchTool,
    EntitySearchTool,
    CommunityInfoTool,
    SchemaInfoTool,
    PathFindTool,
    NeighborExploreTool,
    create_hugegraph_tools,
)


# ============================================================
# HugeGraphVectorStore Tests
# ============================================================

class TestHugeGraphVectorStore:
    """Tests for HugeGraphVectorStore."""

    def test_init_defaults(self):
        store = HugeGraphVectorStore()
        assert store._embedding is None
        assert store._vector_index is None
        assert store._graph_name == "hugegraph"
        assert store._top_k == 5

    def test_init_with_params(self):
        emb = MagicMock()
        idx = MagicMock()
        store = HugeGraphVectorStore(
            embedding=emb, vector_index=idx, graph_name="test", top_k=10
        )
        assert store._embedding is emb
        assert store._vector_index is idx
        assert store._graph_name == "test"
        assert store._top_k == 10

    def test_add_texts_no_embedding(self):
        store = HugeGraphVectorStore()
        ids = store.add_texts(["hello", "world"])
        assert ids == ["doc_0", "doc_1"]

    def test_add_texts_with_embedding(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        idx = MagicMock()
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx)
        ids = store.add_texts(["hello", "world"], metadatas=[{"k": "v1"}, {"k": "v2"}])
        assert ids == ["doc_0", "doc_1"]
        assert emb.get_texts_embeddings.call_count == 1
        assert idx.add_with_ids.call_count == 2

    def test_add_texts_with_properties(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.add_properties = MagicMock()
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx)
        store.add_texts(["hello"], metadatas=[{"k": "v"}])
        idx.add_properties.assert_called_once_with("doc_0", {"k": "v"})

    def test_add_texts_error_returns_empty(self):
        emb = MagicMock()
        emb.get_texts_embeddings.side_effect = RuntimeError("fail")
        idx = MagicMock()
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx)
        ids = store.add_texts(["hello"])
        assert ids == []

    def test_similarity_search_no_config(self):
        store = HugeGraphVectorStore()
        results = store.similarity_search("hello")
        assert results == []

    def test_similarity_search_success(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1, 0.2]]
        idx = MagicMock()
        idx.search.return_value = ["result1", "result2"]
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx, top_k=5)
        results = store.similarity_search("hello")
        assert len(results) == 2
        assert results[0]["content"] == "result1"
        assert results[0]["metadata"] == {}

    def test_similarity_search_custom_k(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["r1"]
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx, top_k=5)
        store.similarity_search("hello", k=1)
        idx.search.assert_called_once_with([0.1], 1)

    def test_similarity_search_non_list_result(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = "not a list"
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx)
        results = store.similarity_search("hello")
        assert results == []

    def test_similarity_search_error(self):
        emb = MagicMock()
        emb.get_texts_embeddings.side_effect = RuntimeError("fail")
        idx = MagicMock()
        store = HugeGraphVectorStore(embedding=emb, vector_index=idx)
        results = store.similarity_search("hello")
        assert results == []

    def test_from_texts(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1], [0.2]]
        idx = MagicMock()
        store = HugeGraphVectorStore.from_texts(
            ["a", "b"], embedding=emb, vector_index=idx
        )
        assert isinstance(store, HugeGraphVectorStore)
        assert store._embedding is emb
        assert idx.add_with_ids.call_count == 2

    def test_dis_threshold_stored(self):
        store = HugeGraphVectorStore(dis_threshold=3.5)
        assert store._dis_threshold == 3.5


# ============================================================
# HugeGraphRetriever Tests
# ============================================================

class TestHugeGraphRetriever:
    """Tests for HugeGraphRetriever."""

    def test_init_defaults(self):
        r = HugeGraphRetriever()
        assert r._top_k == 5
        assert r._graph_ratio == 0.5

    def test_get_relevant_documents_no_config(self):
        r = HugeGraphRetriever()
        results = r.get_relevant_documents("hello")
        assert results == []

    def test_vector_only_search(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["vec1", "vec2"]
        r = HugeGraphRetriever(embedding=emb, vector_index=idx, graph_ratio=0.5)
        results = r.get_relevant_documents("hello", k=4)
        assert len(results) == 2
        assert all(d["metadata"]["source"] == "vector" for d in results)

    def test_vector_and_graph_search(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["vec1"]
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"name": "Entity1"}]
        }
        r = HugeGraphRetriever(
            embedding=emb, vector_index=idx, graph_client=gc, top_k=4
        )
        results = r.get_relevant_documents("hello")
        assert len(results) == 2
        assert results[0]["metadata"]["source"] == "vector"
        assert results[1]["metadata"]["source"] == "graph"

    def test_vector_error_fallback_to_graph(self):
        emb = MagicMock()
        emb.get_texts_embeddings.side_effect = RuntimeError("fail")
        idx = MagicMock()
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": ["g1"]}
        r = HugeGraphRetriever(
            embedding=emb, vector_index=idx, graph_client=gc
        )
        results = r.get_relevant_documents("hello")
        # vector failed, results empty so graph skipped
        assert results == []

    def test_graph_error_no_crash(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["vec1"]
        gc = MagicMock()
        gc.gremlin.side_effect = RuntimeError("graph down")
        r = HugeGraphRetriever(
            embedding=emb, vector_index=idx, graph_client=gc, top_k=5
        )
        results = r.get_relevant_documents("hello")
        # vector works, graph fails, returns vector results only
        assert len(results) == 1

    def test_custom_k_param(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["r1"]
        r = HugeGraphRetriever(embedding=emb, vector_index=idx)
        r.get_relevant_documents("hello", k=1)
        idx.search.assert_called_once_with([0.1], 1)


# ============================================================
# HugeGraphQAChain Tests
# ============================================================

class TestHugeGraphQAChain:
    """Tests for HugeGraphQAChain."""

    def test_init_defaults(self):
        chain = HugeGraphQAChain()
        assert chain._retriever is None
        assert chain._llm is None
        assert chain._max_context_length == 4000
        assert chain._include_sources is True

    def test_run_no_retriever_no_context(self):
        chain = HugeGraphQAChain()
        result = chain.run("What is X?")
        assert "No relevant context" in result["answer"]
        assert result["sources"] == []

    def test_run_with_retriever_no_llm(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {"content": "X is a database", "metadata": {"source": "graph"}}
        ]
        chain = HugeGraphQAChain(retriever=retriever)
        result = chain.run("What is X?")
        assert result["answer"] == "No relevant context found for this question."
        assert result["sources"] == ["graph"]

    def test_run_with_retriever_and_llm(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {"content": "HugeGraph is a graph database", "metadata": {"source": "vector"}}
        ]
        llm = MagicMock()
        llm.generate.return_value = "HugeGraph is a distributed graph database."
        chain = HugeGraphQAChain(retriever=retriever, llm=llm)
        result = chain.run("What is HugeGraph?")
        assert "graph database" in result["answer"]
        assert result["sources"] == ["vector"]
        assert "HugeGraph is a graph database" in result["context"]

    def test_run_llm_error_returns_error_message(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {"content": "test content", "metadata": {"source": "test"}}
        ]
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM fail")
        chain = HugeGraphQAChain(retriever=retriever, llm=llm)
        result = chain.run("test?")
        assert "Error" in result["answer"]

    def test_run_retriever_error_empty_context(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.side_effect = RuntimeError("fail")
        llm = MagicMock()
        chain = HugeGraphQAChain(retriever=retriever, llm=llm)
        result = chain.run("test?")
        assert "No relevant context" in result["answer"]

    def test_context_truncation(self):
        retriever = MagicMock()
        long_text = "x" * 5000
        retriever.get_relevant_documents.return_value = [
            {"content": long_text, "metadata": {"source": "s1"}},
            {"content": long_text, "metadata": {"source": "s2"}},
        ]
        llm = MagicMock()
        llm.generate.return_value = "answer"
        chain = HugeGraphQAChain(
            retriever=retriever, llm=llm, max_context_length=2000
        )
        result = chain.run("test?")
        assert len(result["context"]) <= 2500  # some slack for formatting

    def test_summarize_with_llm(self):
        llm = MagicMock()
        llm.generate.return_value = "Short summary."
        chain = HugeGraphQAChain(llm=llm)
        summary = chain.summarize("Q?", "Long context text here.")
        assert summary == "Short summary."

    def test_summarize_no_llm(self):
        chain = HugeGraphQAChain()
        summary = chain.summarize("Q?", "A" * 1000)
        assert len(summary) == 500

    def test_summarize_error_fallback(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("fail")
        chain = HugeGraphQAChain(llm=llm)
        summary = chain.summarize("Q?", "Short text")
        assert summary == "Short text"

    def test_custom_prompt_template(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {"content": "data", "metadata": {"source": "s"}}
        ]
        llm = MagicMock()
        llm.generate.return_value = "ans"
        chain = HugeGraphQAChain(
            retriever=retriever, llm=llm,
            prompt_template="CTX: {context}\nQ: {question}"
        )
        chain.run("test?")
        prompt_arg = llm.generate.call_args[1]["prompt"]
        assert "CTX:" in prompt_arg

    def test_include_sources_false(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {"content": "data", "metadata": {"source": "s"}}
        ]
        chain = HugeGraphQAChain(retriever=retriever, include_sources=False)
        result = chain.run("test?")
        assert "sources" not in result

    def test_multiple_documents_context(self):
        retriever = MagicMock()
        retriever.get_relevant_documents.return_value = [
            {"content": "doc1", "metadata": {"source": "v1"}},
            {"content": "doc2", "metadata": {"source": "g1"}},
            {"content": "doc3", "metadata": {"source": "v2"}},
        ]
        chain = HugeGraphQAChain(retriever=retriever)
        result = chain.run("test?")
        assert "[1]" in result["context"]
        assert "[3]" in result["context"]


# ============================================================
# DriftRetriever Tests
# ============================================================

class TestDriftRetriever:
    """Tests for DriftRetriever."""

    def test_init_defaults(self):
        dr = DriftRetriever()
        assert dr._top_k == 5
        assert dr._max_parallel == 3

    def test_no_config_returns_empty(self):
        dr = DriftRetriever()
        results = dr.get_relevant_documents("hello")
        assert results == []

    def test_vector_search_only(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["drift_result_1", "drift_result_2"]
        dr = DriftRetriever(embedding=emb, vector_index=idx)
        results = dr.get_relevant_documents("hello")
        assert len(results) == 2
        assert all(r["metadata"]["source"] == "drift_vector" for r in results)

    def test_vector_and_graph_expansion(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["v1", "v2"]
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"name": "ent1"}]
        }
        dr = DriftRetriever(
            embedding=emb, vector_index=idx, graph_client=gc, top_k=4
        )
        results = dr.get_relevant_documents("hello")
        assert len(results) == 3  # 2 vector + 1 graph
        assert results[0]["metadata"]["source"] == "drift_vector"
        assert results[2]["metadata"]["source"] == "drift_graph"

    def test_results_capped_at_top_k(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["r1", "r2", "r3", "r4", "r5"]
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": ["g1", "g2"]}
        dr = DriftRetriever(
            embedding=emb, vector_index=idx, graph_client=gc, top_k=3
        )
        results = dr.get_relevant_documents("hello")
        assert len(results) == 3

    def test_graph_error_no_crash(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["v1"]
        gc = MagicMock()
        gc.gremlin.side_effect = RuntimeError("fail")
        dr = DriftRetriever(
            embedding=emb, vector_index=idx, graph_client=gc
        )
        results = dr.get_relevant_documents("hello")
        assert len(results) == 1  # vector only

    def test_vector_error_returns_empty(self):
        emb = MagicMock()
        emb.get_texts_embeddings.side_effect = RuntimeError("fail")
        idx = MagicMock()
        dr = DriftRetriever(embedding=emb, vector_index=idx)
        results = dr.get_relevant_documents("hello")
        assert results == []

    def test_rank_metadata(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["r1", "r2"]
        dr = DriftRetriever(embedding=emb, vector_index=idx)
        results = dr.get_relevant_documents("hello")
        assert results[0]["metadata"]["rank"] == 1
        assert results[1]["metadata"]["rank"] == 2


# ============================================================
# Agent Tools Tests
# ============================================================

class TestGremlinQueryTool:
    def test_run_success(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": [1, 2, 3]}
        tool = GremlinQueryTool(graph_client=gc)
        result = tool.run("g.V().count()")
        assert "data" in result

    def test_run_error(self):
        gc = MagicMock()
        gc.gremlin.side_effect = RuntimeError("timeout")
        tool = GremlinQueryTool(graph_client=gc)
        result = tool.run("g.V().count()")
        assert "error" in result.lower()

    def test_name_and_description(self):
        tool = GremlinQueryTool()
        assert tool.name == "gremlin_query"
        assert "Gremlin" in tool.description


class TestVectorSearchTool:
    def test_no_config(self):
        tool = VectorSearchTool()
        result = tool.run("hello")
        assert "not configured" in result.lower()

    def test_success(self):
        emb = MagicMock()
        emb.get_texts_embeddings.return_value = [[0.1]]
        idx = MagicMock()
        idx.search.return_value = ["res1"]
        tool = VectorSearchTool(embedding=emb, vector_index=idx)
        result = tool.run("hello", top_k=3)
        assert "res1" in result

    def test_error(self):
        emb = MagicMock()
        emb.get_texts_embeddings.side_effect = RuntimeError("fail")
        idx = MagicMock()
        tool = VectorSearchTool(embedding=emb, vector_index=idx)
        result = tool.run("hello")
        assert "error" in result.lower()

    def test_name(self):
        assert VectorSearchTool.name == "vector_search"


class TestEntitySearchTool:
    def test_success(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"name": "Entity1"}]
        }
        tool = EntitySearchTool(graph_client=gc)
        result = tool.run("Entity1")
        assert "Entity1" in result

    def test_custom_label(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": []}
        tool = EntitySearchTool(graph_client=gc)
        tool.run("test", label="Person")
        call_arg = gc.gremlin.call_args[0][0]
        assert 'Person' in call_arg

    def test_sql_injection_safe(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": []}
        tool = EntitySearchTool(graph_client=gc)
        tool.run("test'or 1=1")
        call_arg = gc.gremlin.call_args[0][0]
        assert "or 1=1" not in call_arg
        assert "\\'" in call_arg

    def test_error(self):
        gc = MagicMock()
        gc.gremlin.side_effect = RuntimeError("fail")
        tool = EntitySearchTool(graph_client=gc)
        result = tool.run("test")
        assert "error" in result.lower()

    def test_name(self):
        assert EntitySearchTool.name == "entity_search"


class TestCommunityInfoTool:
    def test_success(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"community_id": "c1", "summary": "test"}]
        }
        tool = CommunityInfoTool(graph_client=gc)
        result = tool.run("c1")
        assert "c1" in result

    def test_name(self):
        assert CommunityInfoTool.name == "community_info"


class TestSchemaInfoTool:
    def test_success(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": ["Entity", "Relation"]}
        tool = SchemaInfoTool(graph_client=gc)
        result = tool.run("")
        assert "vertex_labels" in result
        assert "edge_labels" in result

    def test_empty_query_default(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": []}
        tool = SchemaInfoTool(graph_client=gc)
        tool.run()  # no args
        assert gc.gremlin.call_count == 2

    def test_name(self):
        assert SchemaInfoTool.name == "schema_info"


class TestPathFindTool:
    def test_success(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"path": "A->B->C"}]
        }
        tool = PathFindTool(graph_client=gc)
        result = tool.run("Alice|Bob")
        assert "Alice" in result

    def test_bad_format(self):
        tool = PathFindTool()
        result = tool.run("single_entity")
        assert "Format" in result

    def test_sql_injection_safe(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": []}
        tool = PathFindTool(graph_client=gc)
        tool.run("A'; drop--|B")
        call_arg = gc.gremlin.call_args[0][0]
        assert "drop" not in call_arg

    def test_custom_max_depth(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": []}
        tool = PathFindTool(graph_client=gc)
        tool.run("A|B", max_depth=3)
        call_arg = gc.gremlin.call_args[0][0]
        assert "gt(3)" in call_arg

    def test_error(self):
        gc = MagicMock()
        gc.gremlin.side_effect = RuntimeError("fail")
        tool = PathFindTool(graph_client=gc)
        result = tool.run("A|B")
        assert "error" in result.lower()

    def test_name(self):
        assert PathFindTool.name == "path_find"


class TestNeighborExploreTool:
    def test_success(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {
            "data": [{"name": "N1"}, {"name": "N2"}]
        }
        tool = NeighborExploreTool(graph_client=gc)
        result = tool.run("Alice")
        assert "Alice" in result

    def test_custom_depth(self):
        gc = MagicMock()
        gc.gremlin.return_value.exec.return_value = {"data": []}
        tool = NeighborExploreTool(graph_client=gc)
        tool.run("Alice", depth=2)
        call_arg = gc.gremlin.call_args[0][0]
        assert "times(2)" in call_arg

    def test_name(self):
        assert NeighborExploreTool.name == "neighbor_explore"


class TestCreateHugegraphTools:
    def test_creates_all_tools(self):
        gc = MagicMock()
        emb = MagicMock()
        idx = MagicMock()
        tools = create_hugegraph_tools(
            graph_client=gc, embedding=emb, vector_index=idx
        )
        assert len(tools) == 7

    def test_tool_names(self):
        gc = MagicMock()
        tools = create_hugegraph_tools(graph_client=gc)
        names = [t.name for t in tools]
        assert "gremlin_query" in names
        assert "vector_search" in names
        assert "entity_search" in names
        assert "community_info" in names
        assert "schema_info" in names
        assert "path_find" in names
        assert "neighbor_explore" in names

    def test_no_graph_client(self):
        tools = create_hugegraph_tools()
        assert len(tools) == 7  # all tools created, just with None client

    def test_vector_search_gets_embedding(self):
        gc = MagicMock()
        emb = MagicMock()
        idx = MagicMock()
        tools = create_hugegraph_tools(
            graph_client=gc, embedding=emb, vector_index=idx
        )
        vs_tool = [t for t in tools if t.name == "vector_search"][0]
        assert vs_tool._embedding is emb
        assert vs_tool._vector_index is idx
