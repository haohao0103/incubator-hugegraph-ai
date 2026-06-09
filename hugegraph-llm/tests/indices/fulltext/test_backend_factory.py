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

import os
import tempfile
import pytest

from hugegraph_llm.indices.fulltext.base import FullTextBase
from hugegraph_llm.indices.fulltext.bm25_fulltext import BM25FullTextBackend


class TestBM25FullTextBackend:
    """Test the refactored BM25FullTextBackend."""

    def test_implements_base(self):
        assert issubclass(BM25FullTextBackend, FullTextBase)

    def test_add_and_search(self):
        backend = BM25FullTextBackend()
        backend.add_documents(
            ["Python programming language", "Java programming language"],
            ids=["d1", "d2"],
        )
        assert backend.doc_count == 2

        results = backend.search("Python")
        assert len(results) >= 1
        assert results[0]["id"] == "d1"
        assert results[0]["score"] > 0

    def test_remove(self):
        backend = BM25FullTextBackend()
        backend.add_documents(["doc one", "doc two", "doc three"], ids=["a", "b", "c"])
        assert backend.doc_count == 3
        removed = backend.remove(["a", "c"])
        assert removed == 2
        assert backend.doc_count == 1

    def test_backward_compat_alias(self):
        """BM25Index from keyword_index.py should be the same class."""
        from hugegraph_llm.indices.keyword_index import BM25Index

        assert BM25Index is BM25FullTextBackend

    def test_backward_compat_import(self):
        """Old import path still works."""
        from hugegraph_llm.indices.keyword_index import tokenize

        tokens = tokenize("HugeGraph 知识图谱")
        assert "hugegraph" in tokens
        assert len(tokens) > 0


class TestOceanBaseFTSBackend:
    """Test OceanBase FTS backend (without real connection)."""

    def test_import(self):
        from hugegraph_llm.indices.fulltext.oceanbase_fulltext import (
            OceanBaseFTSBackend,
        )

        assert issubclass(OceanBaseFTSBackend, FullTextBase)

    def test_create_instance(self):
        from hugegraph_llm.indices.fulltext.oceanbase_fulltext import (
            OceanBaseFTSBackend,
        )

        fts = OceanBaseFTSBackend(dsn="test://localhost")
        assert fts._dsn == "test://localhost"
        assert fts._table == "rag_chunks"

    def test_search_fails_without_connection(self):
        from hugegraph_llm.indices.fulltext.oceanbase_fulltext import (
            OceanBaseFTSBackend,
        )

        fts = OceanBaseFTSBackend(dsn="test://localhost")
        with pytest.raises(ImportError):
            fts.search("test query")


class TestOceanBaseVectorStore:
    """Test OceanBase vector backend (without real connection)."""

    def test_import(self):
        from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
            OceanBaseVectorStore,
        )

        assert issubclass(OceanBaseVectorStore, object)

    def test_create_instance(self):
        from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
            OceanBaseVectorStore,
        )

        store = OceanBaseVectorStore(
            dsn="test://localhost",
            embed_dim=768,
            index_type="hnsw",
        )
        assert store._embed_dim == 768
        assert store._space_type == "cosinesimil"

    def test_invalid_distance_metric(self):
        from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
            OceanBaseVectorStore,
        )

        with pytest.raises(ValueError):
            OceanBaseVectorStore(dsn="test", distance_metric="invalid")

    def test_search_fails_without_connection(self):
        from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
            OceanBaseVectorStore,
        )

        store = OceanBaseVectorStore(dsn="test://localhost", embed_dim=768)
        with pytest.raises(ImportError):
            store.search([0.1] * 768)


class TestBackendFactory:
    """Test the storage backend factory."""

    def setup_method(self):
        self._env_backup = {}
        for key in ["VECTOR_BACKEND", "FULLTEXT_BACKEND", "OCEANBASE_DSN"]:
            if key in os.environ:
                self._env_backup[key] = os.environ[key]
                del os.environ[key]

    def teardown_method(self):
        os.environ.update(self._env_backup)

    def test_default_vector_backend_is_faiss(self):
        from hugegraph_llm.indices.backend_factory import create_vector_store

        store = create_vector_store(embed_dim=128)
        assert type(store).__name__ == "FaissVectorIndex"

    def test_default_fulltext_backend_is_bm25(self):
        from hugegraph_llm.indices.backend_factory import create_fulltext_store

        store = create_fulltext_store()
        assert isinstance(store, BM25FullTextBackend)

    def test_oceanbase_vector_requires_dsn(self):
        from hugegraph_llm.indices.backend_factory import create_vector_store

        os.environ["VECTOR_BACKEND"] = "oceanbase"
        with pytest.raises(ValueError, match="OCEANBASE_DSN"):
            create_vector_store(embed_dim=768)

    def test_oceanbase_fts_requires_dsn(self):
        from hugegraph_llm.indices.backend_factory import create_fulltext_store

        os.environ["FULLTEXT_BACKEND"] = "oceanbase"
        with pytest.raises(ValueError, match="OCEANBASE_DSN"):
            create_fulltext_store()

    def test_oceanbase_vector_with_dsn_param(self):
        from hugegraph_llm.indices.backend_factory import create_vector_store
        from hugegraph_llm.indices.vector_index.oceanbase_vector_store import (
            OceanBaseVectorStore,
        )

        os.environ["VECTOR_BACKEND"] = "oceanbase"
        store = create_vector_store(embed_dim=768, dsn="ob://test:pass@host/db")
        assert isinstance(store, OceanBaseVectorStore)

    def test_oceanbase_fts_with_dsn_param(self):
        from hugegraph_llm.indices.backend_factory import create_fulltext_store
        from hugegraph_llm.indices.fulltext.oceanbase_fulltext import (
            OceanBaseFTSBackend,
        )

        os.environ["FULLTEXT_BACKEND"] = "oceanbase"
        store = create_fulltext_store(dsn="ob://test:pass@host/db")
        assert isinstance(store, OceanBaseFTSBackend)

    def test_unknown_vector_backend(self):
        from hugegraph_llm.indices.backend_factory import create_vector_store

        os.environ["VECTOR_BACKEND"] = "unknown"
        with pytest.raises(ValueError, match="Unknown vector backend"):
            create_vector_store()

    def test_unknown_fulltext_backend(self):
        from hugegraph_llm.indices.backend_factory import create_fulltext_store

        os.environ["FULLTEXT_BACKEND"] = "unknown"
        with pytest.raises(ValueError, match="Unknown full-text backend"):
            create_fulltext_store()
