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

"""Tests for document loaders."""

import csv
import json
import os
import tempfile

import pytest

from hugegraph_llm.document.document_loader import (
    CsvLoader,
    Document,
    JsonLoader,
    MetadataLoader,
    TextLoader,
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_auto_id(self):
        doc = Document(content="test")
        assert doc.doc_id.startswith("doc_")
        assert len(doc.doc_id) > 4

    def test_custom_id(self):
        doc = Document(content="test", doc_id="custom_id")
        assert doc.doc_id == "custom_id"

    def test_metadata_default(self):
        doc = Document(content="test")
        assert doc.metadata == {}

    def test_post_init_auto_id(self):
        doc = Document(content="test")
        assert doc.doc_id.startswith("doc_")


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_texts(self):
        loader = TextLoader()
        docs = loader.load(["doc one", "doc two", "doc three"])
        assert len(docs) == 3
        assert docs[0].content == "doc one"
        assert docs[0].metadata["source_type"] == "text"

    def test_skip_empty(self):
        loader = TextLoader()
        docs = loader.load(["", "   ", "valid"])
        assert len(docs) == 1
        assert docs[0].content == "valid"

    def test_empty_input(self):
        loader = TextLoader()
        docs = loader.load([])
        assert docs == []


class TestJsonLoader:
    """Tests for JsonLoader."""

    def test_load_json_array(self):
        data = [
            {"name": "orders", "comment": "订单表", "columns": [
                {"name": "driver_id", "type": "bigint", "comment": "司机ID"},
                {"name": "status", "type": "varchar", "comment": "状态"},
            ]},
            {"name": "drivers", "comment": "司机表", "columns": [
                {"name": "id", "type": "bigint"},
            ]},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            path = f.name

        try:
            loader = JsonLoader()
            docs = loader.load_file(path)
            assert len(docs) == 2
            # First doc should have formatted content
            assert "Table: orders" in docs[0].content
            assert "Comment: 订单表" in docs[0].content
            assert "driver_id (bigint): 司机ID" in docs[0].content
            assert "status (varchar): 状态" in docs[0].content
        finally:
            os.unlink(path)

    def test_load_json_with_content_field(self):
        data = [
            {"id": "1", "text": "content one", "extra": "ignored"},
            {"id": "2", "text": "content two", "extra": "also ignored"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            path = f.name

        try:
            loader = JsonLoader(content_field="text", id_field="id")
            docs = loader.load_file(path)
            assert len(docs) == 2
            assert docs[0].content == "content one"
            assert docs[0].doc_id == "1"
            assert docs[1].doc_id == "2"
        finally:
            os.unlink(path)

    def test_load_single_object(self):
        data = {"name": "single_table", "comment": "test"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            path = f.name

        try:
            loader = JsonLoader()
            docs = loader.load_file(path)
            assert len(docs) == 1
            assert "Table: single_table" in docs[0].content
        finally:
            os.unlink(path)

    def test_metadata_format_with_db(self):
        data = {
            "name": "orders",
            "database": "dw",
            "comment": "订单宽表",
            "owner": "data_team",
            "columns": [
                {"name": "order_id", "type": "bigint", "comment": "订单ID"},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            path = f.name

        try:
            loader = JsonLoader()
            docs = loader.load_file(path)
            content = docs[0].content
            assert "dw.orders" in content
            assert "Owner: data_team" in content
            assert "订单ID" in content
        finally:
            os.unlink(path)


class TestCsvLoader:
    """Tests for CsvLoader."""

    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["name", "type", "comment"])
            writer.writeheader()
            writer.writerow({"name": "driver_id", "type": "bigint", "comment": "司机ID"})
            writer.writerow({"name": "order_status", "type": "varchar", "comment": "订单状态"})
            f.flush()
            path = f.name

        try:
            loader = CsvLoader()
            docs = loader.load_file(path)
            assert len(docs) == 2
            assert "driver_id" in docs[0].content
            assert "司机ID" in docs[0].content
        finally:
            os.unlink(path)

    def test_load_csv_with_text_columns(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["table_name", "column_name", "comment", "owner"]
            )
            writer.writeheader()
            writer.writerow({
                "table_name": "orders",
                "column_name": "driver_id",
                "comment": "司机ID",
                "owner": "team_a",
            })
            f.flush()
            path = f.name

        try:
            loader = CsvLoader(text_columns=["column_name", "comment"])
            docs = loader.load_file(path)
            assert "column_name: driver_id" in docs[0].content
            assert "comment: 司机ID" in docs[0].content
            # owner should NOT be in content (filtered by text_columns)
            assert "team_a" not in docs[0].content
        finally:
            os.unlink(path)

    def test_load_csv_with_id_column(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["id", "term", "alias"])
            writer.writeheader()
            writer.writerow({"id": "t1", "term": "物理车型", "alias": "实际车型"})
            f.flush()
            path = f.name

        try:
            loader = CsvLoader(id_column="id")
            docs = loader.load_file(path)
            assert docs[0].doc_id == "t1"
        finally:
            os.unlink(path)

    def test_load_tsv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["key", "value"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"key": "name", "value": "orders"})
            f.flush()
            path = f.name

        try:
            loader = CsvLoader(delimiter="\t")
            docs = loader.load_file(path)
            assert len(docs) == 1
            assert "orders" in docs[0].content
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["a", "b"])
            writer.writeheader()
            f.flush()
            path = f.name

        try:
            loader = CsvLoader()
            docs = loader.load_file(path)
            assert len(docs) == 0  # header only, no data rows
        finally:
            os.unlink(path)


class TestMetadataLoader:
    """Tests for MetadataLoader."""

    def test_from_local_json(self):
        data = [
            {
                "name": "orders",
                "database": "dw",
                "comment": "订单宽表",
                "columns": [
                    {"name": "order_id", "type": "bigint", "comment": "订单ID"},
                ],
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            path = f.name

        try:
            docs = MetadataLoader.from_local_json(path, domain="order")
            assert len(docs) == 1
            assert docs[0].metadata["source_type"] == "metadata_json"
            assert docs[0].metadata["domain"] == "order"
            assert "dw.orders" in docs[0].content
        finally:
            os.unlink(path)

    def test_no_base_url_returns_empty(self):
        loader = MetadataLoader(base_url="")
        docs = loader.load_domain("order")
        assert docs == []

    def test_domain_filter(self):
        loader = MetadataLoader(
            base_url="",
            domain_filter=["order"],
        )
        docs = loader.load(["driver", "order"])
        assert docs == []
        # Even with base_url empty, driver is filtered out
        # and order would try to fetch but base_url is empty -> empty

    def test_format_table_doc_static(self):
        table_info = {
            "name": "test_table",
            "database": "test_db",
            "comment": "Test table",
            "owner": "team",
        }
        columns = [
            {"name": "col1", "data_type": "varchar", "comment": "Column 1", "is_nullable": "YES"},
            {"name": "col2", "data_type": "bigint", "comment": "", "is_nullable": "NO"},
        ]
        text = MetadataLoader._format_table_doc(table_info, columns)
        assert "test_db.test_table" in text
        assert "Comment: Test table" in text
        assert "col1 (varchar, nullable): Column 1" in text
        assert "col2 (bigint, required)" in text
