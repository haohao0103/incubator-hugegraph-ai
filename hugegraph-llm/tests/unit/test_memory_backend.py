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

"""Unit tests for hugegraph_llm.poc.memory_backend helpers and FaissMemoryIndex."""

import json
import os
import tempfile
import uuid
from unittest import mock

import numpy as np
import pytest

from hugegraph_llm.poc.memory_backend import (
    FaissMemoryIndex,
    _extract_json_from_response,
    _normalize_keys,
    get_metadata_db,
    init_metadata_db,
)


class MockMessage:
    def __init__(self, content="", reasoning_content=""):
        self.content = content
        self.reasoning_content = reasoning_content


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, content=""):
        self.choices = [MockChoice(MockMessage(content=content))]


class MockSentenceTransformer:
    """Deterministic mock sentence transformer that returns a unique 384-dim vector per text."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, text, **kwargs):
        # Deterministic vector based on text hash, normalized to unit length
        h = hash(text) & 0xFFFFFFFF
        np.random.seed(h)
        vec = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Patch FaissMemoryIndex to use a deterministic mock embedding model."""
    old_model = FaissMemoryIndex._model
    FaissMemoryIndex._model = MockSentenceTransformer(dim=384)
    yield
    FaissMemoryIndex._model = old_model


def test_extract_json_from_markdown():
    content = json.dumps({"entities": [{"name": "Alice", "type": "person"}]})
    response = MockResponse(f'```json\n{content}\n```')
    result = _extract_json_from_response(response)
    assert result["entities"][0]["name"] == "Alice"


def test_extract_json_from_plain():
    content = json.dumps({"entities": [{"name": "Bob", "type": "person"}]})
    response = MockResponse(content)
    result = _extract_json_from_response(response)
    assert result["entities"][0]["name"] == "Bob"


def test_extract_json_from_regex_fallback():
    response = MockResponse('some text before "name": "Carol", "type": "person" after')
    result = _extract_json_from_response(response)
    assert result["entities"][0]["name"] == "Carol"


def test_extract_json_reasoning_content():
    response = MockResponse()
    response.choices[0].message.content = ""
    response.choices[0].message.reasoning_content = '{"entities": []}'
    result = _extract_json_from_response(response)
    assert result["entities"] == []


def test_normalize_keys():
    raw = {
        "entities": [{"name": "Alice", "type": "Person"}, {"entity": "Bob", "category": "person"}],
        "relationships": [{"source": "Alice", "relationship": "works_at", "target": "Tencent"}],
    }
    result = _normalize_keys(raw)
    assert result["entities"] == [
        {"name": "Alice", "type": "person"},
        {"name": "Bob", "type": "person"},
    ]
    assert result["relationships"] == [
        {"source": "Alice", "relationship": "works_at", "target": "Tencent"},
    ]


def test_normalize_keys_skips_self_references():
    raw = {
        "entities": [{"name": "我", "type": "person"}, {"name": "Alice", "type": "person"}],
    }
    result = _normalize_keys(raw)
    assert len(result["entities"]) == 1


class TestFaissMemoryIndex:
    def test_add_and_search(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello world", 123456.0)
        results = idx.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0]["memory_id"] == "m1"

    def test_search_with_weights(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello", 123456.0)
        idx.add_memory("m2", "world", 123456.0)
        results = idx.search("hello", top_k=5, ebbinghaus_weights={"m1": 1.0, "m2": 0.5})
        assert results[0]["memory_id"] == "m1"

    def test_save_and_load(self, mock_sentence_transformer):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "faiss.index")
            idx = FaissMemoryIndex(dim=384, index_path=index_path)
            idx.add_memory("m1", "hello", 123456.0)
            idx.save()

            idx2 = FaissMemoryIndex(dim=384, index_path=index_path)
            idx2.load()
            results = idx2.search("hello", top_k=5)
            assert len(results) == 1

    def test_delete_memory(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello", 123456.0)
        idx.add_memory("m2", "world", 123456.0)
        idx.delete_memory("m1")
        results = idx.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0]["memory_id"] == "m2"

    def test_clear(self, mock_sentence_transformer):
        idx = FaissMemoryIndex(dim=384)
        idx.add_memory("m1", "hello", 123456.0)
        idx.clear()
        assert idx.index.ntotal == 0


class TestMetadataDb:
    def test_init_metadata_db_creates_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MEMORY_DB_PATH"] = os.path.join(tmpdir, "meta.db")
            try:
                init_metadata_db()
                db = get_metadata_db()
                tables = db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                names = {row[0] for row in tables}
                assert "memories" in names
                assert "personas" in names
                db.close()
            finally:
                del os.environ["MEMORY_DB_PATH"]

    def test_metadata_db_migration_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MEMORY_DB_PATH"] = os.path.join(tmpdir, "meta.db")
            try:
                init_metadata_db()
                db = get_metadata_db()
                info = db.execute("PRAGMA table_info(memories)").fetchall()
                columns = {row[1] for row in info}
                assert "scope" in columns
                assert "privacy" in columns
                assert "importance" in columns
                assert "metadata" in columns
                db.close()
            finally:
                del os.environ["MEMORY_DB_PATH"]


def test_memory_backend_import():
    """Ensure memory_backend imports correctly after all modifications."""
    from hugegraph_llm.poc import memory_backend
    assert hasattr(memory_backend, "MemoryPipelineBackend")
    assert hasattr(memory_backend, "HugeGraphMemoryClient")
    assert hasattr(memory_backend, "FaissMemoryIndex")
