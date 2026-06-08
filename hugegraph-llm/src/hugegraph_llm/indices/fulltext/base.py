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

"""Abstract base class for full-text search backends.

Provides a unified interface for pluggable full-text retrieval
implementations (BM25 local file, OceanBase FTS, Elasticsearch, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class FullTextBase(ABC):
    """Abstract interface for full-text search backends.

    All implementations must support adding documents, searching with
    relevance scoring, removing documents, and persistence lifecycle.
    """

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        props: Optional[List[Any]] = None,
    ) -> None:
        """Add documents to the full-text index.

        Args:
            texts: List of document texts.
            ids: Optional document IDs. Auto-generated if None.
            props: Optional metadata for each document.
        """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search the index for relevant documents.

        Args:
            query: Query text.
            top_k: Maximum results.
            min_score: Minimum relevance score threshold.

        Returns:
            List of dicts with keys: id, text, score, prop.
            Sorted by score descending.
        """

    @abstractmethod
    def remove(self, doc_ids: Union[set, List[str]]) -> int:
        """Remove documents by ID.

        Args:
            doc_ids: IDs of documents to remove.

        Returns:
            Number of documents removed.
        """

    @property
    @abstractmethod
    def doc_count(self) -> int:
        """Number of indexed documents."""

    @abstractmethod
    def save_index_by_name(self, *name: str) -> None:
        """Persist the index to storage."""

    @classmethod
    @abstractmethod
    def from_name(cls, *name: str) -> "FullTextBase":
        """Load an index from storage."""

    @staticmethod
    @abstractmethod
    def exist(*name: str) -> bool:
        """Check if a saved index exists."""

    @staticmethod
    @abstractmethod
    def clean(*name: str) -> bool:
        """Delete saved index files."""
