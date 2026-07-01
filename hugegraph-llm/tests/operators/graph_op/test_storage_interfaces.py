# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Tests for Storage Interface stubs."""

import json
import os
import tempfile
import time
import pytest

from hugegraph_llm.operators.graph_op.storage_interfaces import (
    BaseKVStorage,
    JsonFileKVStorage,
    InMemoryKVStorage,
    KVStorageFactory,
    DocStatus,
    DocStatusRecord,
    BaseDocStatusStorage,
    SQLiteDocStatusStorage,
    InMemoryDocStatusStorage,
    DocStatusFactory,
)


# ── KV Storage tests ──

class TestJsonFileKVStorage:
    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir, namespace="test_kv")
            kv.set("key1", "value1")
            assert kv.get("key1") == "value1"

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            assert kv.get("missing") is None

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            kv.set("key1", "value1")
            assert kv.delete("key1") is True
            assert kv.get("key1") is None
            assert kv.delete("key1") is False  # Already deleted

    def test_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            kv.set("a", "1")
            kv.set("b", "2")
            kv.set("c", "3")
            assert sorted(kv.keys()) == ["a", "b", "c"]

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            kv.set("a", "1")
            kv.set("b", "2")
            kv.clear()
            assert kv.size() == 0

    def test_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            assert kv.size() == 0
            kv.set("a", "1")
            assert kv.size() == 1
            kv.set("b", "2")
            assert kv.size() == 2

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            kv.set("key1", "old_value")
            kv.set("key1", "new_value")
            assert kv.get("key1") == "new_value"

    def test_persistence(self):
        """Data persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kv1 = JsonFileKVStorage(working_dir=tmpdir, namespace="persist")
            kv1.set("key1", "value1")

            kv2 = JsonFileKVStorage(working_dir=tmpdir, namespace="persist")
            assert kv2.get("key1") == "value1"

    def test_unicode_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = JsonFileKVStorage(working_dir=tmpdir)
            kv.set("中文", "值是中文")
            assert kv.get("中文") == "值是中文"


class TestInMemoryKVStorage:
    def test_set_and_get(self):
        kv = InMemoryKVStorage()
        kv.set("key1", "value1")
        assert kv.get("key1") == "value1"

    def test_delete(self):
        kv = InMemoryKVStorage()
        kv.set("key1", "value1")
        assert kv.delete("key1") is True
        assert kv.get("key1") is None

    def test_keys(self):
        kv = InMemoryKVStorage()
        kv.set("a", "1")
        kv.set("b", "2")
        assert sorted(kv.keys()) == ["a", "b"]

    def test_clear(self):
        kv = InMemoryKVStorage()
        kv.set("a", "1")
        kv.clear()
        assert kv.size() == 0

    def test_size(self):
        kv = InMemoryKVStorage()
        kv.set("a", "1")
        assert kv.size() == 1


class TestKVStorageFactory:
    def test_create_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = KVStorageFactory.create("json_file", working_dir=tmpdir)
            assert isinstance(kv, JsonFileKVStorage)

    def test_create_memory(self):
        kv = KVStorageFactory.create("memory")
        assert isinstance(kv, InMemoryKVStorage)

    def test_create_default_is_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = KVStorageFactory.create(**{"working_dir": tmpdir})
            assert isinstance(kv, JsonFileKVStorage)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown KV backend"):
            KVStorageFactory.create("unknown")

    def test_create_redis_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should fall back to json_file
            kv = KVStorageFactory.create("redis", working_dir=tmpdir)
            assert isinstance(kv, JsonFileKVStorage)

    def test_create_mongodb_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kv = KVStorageFactory.create("mongodb", working_dir=tmpdir)
            assert isinstance(kv, JsonFileKVStorage)


# ── Doc Status tests ──

class TestDocStatusEnum:
    def test_status_values(self):
        assert DocStatus.PENDING.value == "PENDING"
        assert DocStatus.PROCESSED.value == "PROCESSED"
        assert DocStatus.FAILED.value == "FAILED"


class TestDocStatusRecord:
    def test_default_timestamps(self):
        rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
        assert rec.created_at > 0
        assert rec.updated_at > 0

    def test_custom_timestamps(self):
        rec = DocStatusRecord(
            doc_id="doc1", file_path="/test.pdf",
            created_at=100.0, updated_at=200.0,
        )
        assert rec.created_at == 100.0
        assert rec.updated_at == 200.0

    def test_default_counts(self):
        rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
        assert rec.chunks_count == 0
        assert rec.entities_count == 0
        assert rec.relations_count == 0


class TestSQLiteDocStatusStorage:
    def test_upsert_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
            ds.upsert(rec)
            result = ds.get("doc1")
            assert result is not None
            assert result.doc_id == "doc1"
            assert result.file_path == "/test.pdf"
            assert result.status == DocStatus.PENDING
            ds.close()

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            assert ds.get("missing") is None
            ds.close()

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
            ds.upsert(rec)
            assert ds.delete("doc1") is True
            assert ds.get("doc1") is None
            ds.close()

    def test_upsert_updates_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
            ds.upsert(rec)

            # Update status to PROCESSED
            rec.status = DocStatus.PROCESSED
            rec.chunks_count = 10
            ds.upsert(rec)

            result = ds.get("doc1")
            assert result.status == DocStatus.PROCESSED
            assert result.chunks_count == 10
            ds.close()

    def test_get_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
            ds.upsert(DocStatusRecord(
                doc_id="d2", file_path="/b.pdf", status=DocStatus.PROCESSED,
            ))
            ds.upsert(DocStatusRecord(
                doc_id="d3", file_path="/c.pdf", status=DocStatus.PROCESSED,
            ))

            pending = ds.get_by_status(DocStatus.PENDING)
            processed = ds.get_by_status(DocStatus.PROCESSED)
            assert len(pending) == 1
            assert len(processed) == 2
            ds.close()

    def test_get_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
            ds.upsert(DocStatusRecord(
                doc_id="d2", file_path="/b.pdf", status=DocStatus.PROCESSED,
            ))
            pending = ds.get_pending()
            assert len(pending) == 1
            ds.close()

    def test_count_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
            ds.upsert(DocStatusRecord(
                doc_id="d2", file_path="/b.pdf", status=DocStatus.PROCESSED,
            ))
            counts = ds.count_by_status()
            assert counts.get("PENDING") == 1
            assert counts.get("PROCESSED") == 1
            ds.close()

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
            ds.clear()
            assert ds.size() == 0
            ds.close()

    def test_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = SQLiteDocStatusStorage(working_dir=tmpdir)
            assert ds.size() == 0
            ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
            assert ds.size() == 1
            ds.close()


class TestInMemoryDocStatusStorage:
    def test_upsert_and_get(self):
        ds = InMemoryDocStatusStorage()
        rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
        ds.upsert(rec)
        result = ds.get("doc1")
        assert result is not None
        assert result.doc_id == "doc1"

    def test_delete(self):
        ds = InMemoryDocStatusStorage()
        rec = DocStatusRecord(doc_id="doc1", file_path="/test.pdf")
        ds.upsert(rec)
        assert ds.delete("doc1") is True
        assert ds.get("doc1") is None

    def test_get_by_status(self):
        ds = InMemoryDocStatusStorage()
        ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
        ds.upsert(DocStatusRecord(doc_id="d2", file_path="/b.pdf", status=DocStatus.PROCESSED))
        assert len(ds.get_by_status(DocStatus.PENDING)) == 1
        assert len(ds.get_by_status(DocStatus.PROCESSED)) == 1

    def test_count_by_status(self):
        ds = InMemoryDocStatusStorage()
        ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
        counts = ds.count_by_status()
        assert counts.get("PENDING") == 1

    def test_clear_and_size(self):
        ds = InMemoryDocStatusStorage()
        ds.upsert(DocStatusRecord(doc_id="d1", file_path="/a.pdf"))
        ds.clear()
        assert ds.size() == 0


class TestDocStatusFactory:
    def test_create_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocStatusFactory.create("sqlite", working_dir=tmpdir)
            assert isinstance(ds, SQLiteDocStatusStorage)
            ds.close()

    def test_create_memory(self):
        ds = DocStatusFactory.create("memory")
        assert isinstance(ds, InMemoryDocStatusStorage)

    def test_create_default_is_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocStatusFactory.create(**{"working_dir": tmpdir})
            assert isinstance(ds, SQLiteDocStatusStorage)
            ds.close()

    def test_create_json_file_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DocStatusFactory.create("json_file", working_dir=tmpdir)
            assert isinstance(ds, SQLiteDocStatusStorage)  # Falls back to sqlite
            ds.close()

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown DocStatus backend"):
            DocStatusFactory.create("unknown")

    def test_create_postgresql_not_implemented(self):
        with pytest.raises(NotImplementedError):
            DocStatusFactory.create("postgresql")
