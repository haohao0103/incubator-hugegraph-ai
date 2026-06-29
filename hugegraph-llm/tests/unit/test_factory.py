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

"""Tests for hugegraph_llm.engines.memory.factory."""

import pytest

from hugegraph_llm.engines.memory import (
    LLMFactory,
    EmbedderFactory,
    VectorStoreFactory,
    RerankerFactory,
    GraphStoreFactory,
    llm_factory,
    embedder_factory,
    vector_store_factory,
    reranker_factory,
    graph_store_factory,
    register_llm,
    register_embedder,
    register_vector_store,
    register_reranker,
    register_graph_store,
    get_default_vector_store,
    get_default_reranker,
)


class DummyLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyEmbedder:
    pass


class DummyVectorStore:
    pass


class DummyReranker:
    pass


class DummyGraphStore:
    pass


class TestRegistry:
    def test_register_and_create(self):
        factory = LLMFactory()
        factory.register("dummy", DummyLLM)
        instance = factory.create("dummy", api_key="x")
        assert isinstance(instance, DummyLLM)
        assert instance.kwargs == {"api_key": "x"}

    def test_create_unknown(self):
        factory = LLMFactory()
        with pytest.raises(ValueError):
            factory.create("unknown")

    def test_register_decorators(self):
        register_llm("dummy_llm")(DummyLLM)
        assert llm_factory.create("dummy_llm") is not None

        register_embedder("dummy_embedder")(DummyEmbedder)
        assert embedder_factory.create("dummy_embedder") is not None

        register_vector_store("dummy_vs")(DummyVectorStore)
        assert vector_store_factory.create("dummy_vs") is not None

        register_reranker("dummy_reranker")(DummyReranker)
        assert reranker_factory.create("dummy_reranker") is not None

        register_graph_store("dummy_graph")(DummyGraphStore)
        assert graph_store_factory.create("dummy_graph") is not None

    def test_case_insensitive(self):
        factory = LLMFactory()
        factory.register("Dummy", DummyLLM)
        instance = factory.create("DUMMY")
        assert isinstance(instance, DummyLLM)


class TestDefaults:
    def test_get_default_vector_store(self):
        # Default is faiss -> returns None (lazy init)
        store = get_default_vector_store()
        assert store is None

    def test_get_default_reranker_disabled(self):
        from hugegraph_llm.config.memory_config import memory_settings
        old = memory_settings.rerank_enabled
        memory_settings.rerank_enabled = False
        try:
            assert get_default_reranker() is None
        finally:
            memory_settings.rerank_enabled = old
