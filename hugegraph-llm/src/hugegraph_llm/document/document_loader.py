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

"""Document loaders for knowledge graph ingestion.

Provides a unified interface for loading documents from multiple
data sources into the HugeGraph-AI pipeline. Currently supports:

- **TextLoader**: Direct text string input
- **JsonLoader**: Structured JSON files (tables, entities, metadata)
- **CsvLoader**: CSV/TSV tabular files (metadata, dictionaries)
- **MetadataLoader**: Metadata platform adapter for database schema import

All loaders produce a ``Document`` dataclass with content, metadata,
and source tracking. Documents can be fed directly into the
ChunkSplit pipeline.

Usage::

    from hugegraph_llm.document.document_loader import (
        TextLoader, JsonLoader, CsvLoader, MetadataLoader,
    )

    # Simple text
    docs = TextLoader().load(["doc1 text", "doc2 text"])

    # JSON file
    docs = JsonLoader().load_file("metadata/tables.json")

    # CSV file
    docs = CsvLoader().load_file("metadata/columns.csv")

    # Metadata platform
    loader = MetadataLoader(base_url="https://metadata.internal/api")
    docs = loader.load_domain("order")
"""

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Document:
    """A loaded document with content and metadata.

    Attributes:
        content: The document text content.
        metadata: Key-value metadata (source, title, type, etc.).
        doc_id: Unique identifier. Auto-generated if not provided.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"doc_{id(self):x}"


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, sources: List[Any]) -> List[Document]:
        """Load documents from the given sources.

        Args:
            sources: Source-specific input (texts, file paths, URLs, etc.).

        Returns:
            List of Document objects.
        """

    def load_file(self, filepath: str) -> List[Document]:
        """Load from a single file.

        Args:
            filepath: Path to the file.

        Returns:
            List of Document objects.
        """
        return self.load([filepath])


class TextLoader(BaseLoader):
    """Load plain text documents.

    Each input string becomes a separate Document.
    """

    def load(self, sources: List[str]) -> List[Document]:
        """Load text strings as documents.

        Args:
            sources: List of text strings.

        Returns:
            List of Document objects, one per input string.
        """
        docs = []
        for i, text in enumerate(sources):
            if not text or not text.strip():
                continue
            docs.append(
                Document(
                    content=text.strip(),
                    metadata={"source_type": "text", "index": i},
                )
            )
        log.debug("TextLoader loaded %d documents", len(docs))
        return docs


class JsonLoader(BaseLoader):
    """Load structured JSON data as documents.

    Supports two modes:
    1. **Array mode**: JSON file is an array of objects, each becomes a Document.
    2. **Nested mode**: JSON file has nested structure, extracts text from
       specified content fields.

    For metadata/table schema JSON, each table definition becomes a Document
    with its fields formatted as text content.
    """

    def __init__(
        self,
        content_field: Optional[str] = None,
        id_field: Optional[str] = None,
    ):
        """Initialize JSON loader.

        Args:
            content_field: Key to extract text content from each JSON object.
                          If None, uses entire object as JSON string.
            id_field: Key to extract document ID from each object.
        """
        self.content_field = content_field
        self.id_field = id_field

    def load(self, sources: List[str]) -> List[Document]:
        """Load JSON files.

        Args:
            sources: List of file paths to JSON files.

        Returns:
            List of Document objects.
        """
        docs = []
        for filepath in sources:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_docs = self._parse_data(data, filepath)
                docs.extend(file_docs)
                log.debug(
                    "JsonLoader loaded %d docs from %s",
                    len(file_docs),
                    filepath,
                )
            except Exception as e:
                log.error("Failed to load JSON from %s: %s", filepath, e)
        return docs

    def _parse_data(
        self, data: Any, source_path: str
    ) -> List[Document]:
        """Parse JSON data into documents."""
        if isinstance(data, list):
            return [self._object_to_doc(obj, source_path, i) for i, obj in enumerate(data)]
        return [self._object_to_doc(data, source_path, 0)]

    def _object_to_doc(
        self, obj: Dict, source_path: str, index: int
    ) -> Document:
        """Convert a JSON object to a Document."""
        doc_id = ""
        if self.id_field and self.id_field in obj:
            doc_id = str(obj[self.id_field])

        if self.content_field and self.content_field in obj:
            content = str(obj[self.content_field])
        else:
            content = self._format_metadata_doc(obj)

        return Document(
            content=content,
            metadata={
                "source_type": "json",
                "source_path": source_path,
                "index": index,
                "original_keys": list(obj.keys()),
            },
            doc_id=doc_id,
        )

    @staticmethod
    def _format_metadata_doc(obj: Dict) -> str:
        """Format a metadata object (table/column definition) as readable text.

        Produces text like:
        "Table: orders (dw.orders)
         Comment: Order fact table
         Columns:
           - driver_id (bigint): Driver ID
           - order_status (varchar): Order status
           ..."
        """
        lines = []

        table_name = obj.get("name", obj.get("table_name", obj.get("tableName", "unknown")))
        db = obj.get("database", obj.get("schema", obj.get("db", "")))
        if db:
            lines.append(f"Table: {table_name} ({db}.{table_name})")
        else:
            lines.append(f"Table: {table_name}")

        comment = obj.get("comment") or obj.get("description", "")
        if comment:
            lines.append(f"Comment: {comment}")

        owner = obj.get("owner", "")
        if owner:
            lines.append(f"Owner: {owner}")

        columns = obj.get("columns", obj.get("fields", obj.get("column_list", [])))
        if columns:
            lines.append("Columns:")
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", col.get("column_name", "?"))
                    col_type = col.get("type", col.get("data_type", ""))
                    col_comment = col.get("comment", col.get("description", ""))
                    if col_comment:
                        lines.append(f"  - {col_name} ({col_type}): {col_comment}")
                    else:
                        lines.append(f"  - {col_name} ({col_type})")
                elif isinstance(col, str):
                    lines.append(f"  - {col}")

        return "\n".join(lines)


class CsvLoader(BaseLoader):
    """Load CSV/TSV files as documents.

    Each row becomes a separate Document. Column headers are included
    in the content as field names.
    """

    def __init__(
        self,
        delimiter: str = ",",
        text_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None,
    ):
        """Initialize CSV loader.

        Args:
            delimiter: Column delimiter (default: comma).
            text_columns: Columns to include as content. If None, all columns.
            id_column: Column to use as document ID.
        """
        self.delimiter = delimiter
        self.text_columns = text_columns
        self.id_column = id_column

    def load(self, sources: List[str]) -> List[Document]:
        """Load CSV files.

        Args:
            sources: List of file paths to CSV/TSV files.

        Returns:
            List of Document objects.
        """
        docs = []
        for filepath in sources:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=self.delimiter)
                    for i, row in enumerate(reader):
                        content = self._row_to_text(row)
                        if not content.strip():
                            continue
                        doc_id = ""
                        if self.id_column and self.id_column in row:
                            doc_id = str(row[self.id_column])
                        docs.append(
                            Document(
                                content=content,
                                metadata={
                                    "source_type": "csv",
                                    "source_path": filepath,
                                    "index": i,
                                },
                                doc_id=doc_id,
                            )
                        )
                log.debug(
                    "CsvLoader loaded %d docs from %s", len(docs), filepath
                )
            except Exception as e:
                log.error("Failed to load CSV from %s: %s", filepath, e)
        return docs

    def _row_to_text(self, row: Dict[str, str]) -> str:
        """Convert a CSV row to text content."""
        if self.text_columns:
            parts = []
            for col in self.text_columns:
                if col in row and row[col].strip():
                    parts.append(f"{col}: {row[col]}")
            return "\n".join(parts)
        # All columns
        return "\n".join(
            f"{k}: {v}" for k, v in row.items() if v.strip()
        )


class MetadataLoader(BaseLoader):
    """Load metadata from a metadata management platform.

    Designed for enterprise metadata retrieval scenarios (like
    the Huolala/Huolala case study). Connects to a metadata
    platform API to fetch table/column definitions and converts
    them into Documents suitable for the ChunkSplit pipeline.

    The loader:
    1. Fetches table list from the API
    2. For each table, fetches column details
    3. Formats each table as a Document with structured content

    Usage::

        loader = MetadataLoader(
            base_url="https://metadata.internal/api/v1",
            auth_token=os.environ["METADATA_TOKEN"],
        )
        docs = loader.load_domain("order")
    """

    def __init__(
        self,
        base_url: str = "",
        auth_token: str = "",
        timeout: int = 30,
        domain_filter: Optional[List[str]] = None,
    ):
        """Initialize metadata loader.

        Args:
            base_url: Base URL of the metadata platform API.
            auth_token: Authentication token for API access.
            timeout: HTTP request timeout in seconds.
            domain_filter: Optional list of domains to filter.
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.domain_filter = domain_filter

    def load(self, sources: List[str]) -> List[Document]:
        """Load metadata for specified domains.

        Args:
            sources: List of domain names (e.g., ["order", "driver"]).

        Returns:
            List of Document objects, one per table.
        """
        docs = []
        for domain in sources:
            if self.domain_filter and domain not in self.domain_filter:
                continue
            try:
                domain_docs = self._load_domain(domain)
                docs.extend(domain_docs)
            except Exception as e:
                log.error("Failed to load domain '%s': %s", domain, e)
        log.debug("MetadataLoader loaded %d documents", len(docs))
        return docs

    def load_domain(self, domain: str) -> List[Document]:
        """Load metadata for a single domain.

        Args:
            domain: Domain name.

        Returns:
            List of Documents.
        """
        return self.load([domain])

    def _load_domain(self, domain: str) -> List[Document]:
        """Fetch and parse metadata for a domain."""
        if not self.base_url:
            log.warning(
                "MetadataLoader: base_url not configured, "
                "returning empty. Set base_url to connect to metadata platform."
            )
            return []

        tables = self._fetch_tables(domain)
        docs = []
        for table_info in tables:
            table_name = table_info.get("name", table_info.get("table_name", ""))
            if not table_name:
                continue
            columns = self._fetch_columns(table_name)
            content = self._format_table_doc(table_info, columns)
            docs.append(
                Document(
                    content=content,
                    metadata={
                        "source_type": "metadata_platform",
                        "domain": domain,
                        "table_name": table_name,
                        "column_count": len(columns),
                    },
                    doc_id=f"table:{domain}.{table_name}",
                )
            )
        return docs

    def _fetch_tables(self, domain: str) -> List[Dict]:
        """Fetch table list from metadata platform API."""
        return self._api_get(f"/tables?domain={domain}")

    def _fetch_columns(self, table_name: str) -> List[Dict]:
        """Fetch column details for a table."""
        return self._api_get(f"/tables/{table_name}/columns")

    def _api_get(self, path: str) -> Any:
        """Make an API GET request."""
        try:
            import urllib.request

            url = f"{self.base_url}{path}"
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            log.warning("API request to %s failed: %s", path, e)
            return []

    @staticmethod
    def _format_table_doc(table_info: Dict, columns: List[Dict]) -> str:
        """Format a table's metadata as readable text."""
        lines = []
        name = table_info.get("name", table_info.get("table_name", "unknown"))
        db = table_info.get("database", table_info.get("db", ""))
        if db:
            lines.append(f"Table: {name} ({db}.{name})")
        else:
            lines.append(f"Table: {name}")

        comment = table_info.get("comment", table_info.get("description", ""))
        if comment:
            lines.append(f"Comment: {comment}")

        owner = table_info.get("owner", "")
        if owner:
            lines.append(f"Owner: {owner}")

        if columns:
            lines.append("Columns:")
            for col in columns:
                col_name = col.get("name", col.get("column_name", "?"))
                col_type = col.get("type", col.get("data_type", ""))
                col_comment = col.get("comment", col.get("description", ""))
                nullable = col.get("nullable", col.get("is_nullable", ""))
                null_label = "nullable" if nullable == "YES" else "required" if nullable == "NO" else ""
                type_str = f"{col_type}, {null_label}" if null_label else col_type
                if col_comment:
                    lines.append(f"  - {col_name} ({type_str}): {col_comment}")
                else:
                    lines.append(f"  - {col_name} ({type_str})")

        return "\n".join(lines)

    @staticmethod
    def from_local_json(filepath: str, domain: str = "") -> List[Document]:
        """Load metadata from a local JSON file.

        Convenience method for testing or when metadata is exported
        as JSON from the platform.

        Args:
            filepath: Path to JSON file containing table metadata.
            domain: Optional domain tag for metadata filtering.

        Returns:
            List of Documents.
        """
        loader = JsonLoader()
        docs = loader.load_file(filepath)
        # Tag with metadata source type
        for doc in docs:
            doc.metadata["source_type"] = "metadata_json"
            if domain:
                doc.metadata["domain"] = domain
        return docs
