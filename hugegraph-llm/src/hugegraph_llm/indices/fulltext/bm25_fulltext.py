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

"""BM25 fulltext keyword index for AI memory retrieval.

Uses rank_bm25 for scoring and jieba for Chinese/English tokenization.
Persisted to disk via pickle so the index survives server restarts.
"""

import os
import pickle
import re
from typing import Dict, List, Optional

from hugegraph_llm.utils.log import log


class BM25FullTextBackend:
    """BM25 fulltext backend backed by rank_bm25.BM25Okapi."""

    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
            self._BM25Okapi = BM25Okapi
        except ImportError as exc:
            raise ImportError("rank_bm25 is required for BM25FullTextBackend") from exc

        try:
            import jieba
            self._jieba = jieba
        except ImportError as exc:
            raise ImportError("jieba is required for BM25FullTextBackend") from exc

        self._docs: List[str] = []
        self._ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._model: Optional[object] = None
        self._rebuild_needed = True

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using jieba, keeping CJK characters and alphanumerics."""
        text = str(text)
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        tokens = [t.strip().lower() for t in self._jieba.lcut(text) if len(t.strip()) > 1]
        return tokens

    def _ensure_built(self):
        if self._rebuild_needed and self._docs:
            self._model = self._BM25Okapi([self._tokenize(d) for d in self._docs])
            self._rebuild_needed = False

    def add_documents(self, texts: List[str], ids: List[str]) -> None:
        """Add documents to the BM25 index.

        Args:
            texts: List of document texts.
            ids: List of document IDs (same length as texts).
        """
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")

        for text, doc_id in zip(texts, ids):
            if doc_id in self._id_to_index:
                idx = self._id_to_index[doc_id]
                self._docs[idx] = text
            else:
                idx = len(self._docs)
                self._docs.append(text)
                self._ids.append(doc_id)
                self._id_to_index[doc_id] = idx

        self._rebuild_needed = True
        self._ensure_built()

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        if doc_id not in self._id_to_index:
            return False
        idx = self._id_to_index[doc_id]
        self._docs.pop(idx)
        self._ids.pop(idx)
        self._id_to_index.pop(doc_id)
        # Rebuild index mapping
        self._id_to_index = {doc_id: i for i, doc_id in enumerate(self._ids)}
        self._rebuild_needed = True
        self._ensure_built()
        return True

    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[Dict[str, object]]:
        """Search the BM25 index.

        Returns:
            List of {"id": doc_id, "score": bm25_score} sorted by score descending.
        """
        self._ensure_built()
        if not self._model or not self._docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._model.get_scores(query_tokens)
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score < min_score:
                break
            results.append({"id": self._ids[idx], "score": score})
        return results

    def save_index_by_name(self, directory: str, name: str) -> None:
        """Persist the index to disk."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "docs": self._docs,
                "ids": self._ids,
            }, f)
        log.debug("BM25 index saved to %s", path)

    @classmethod
    def from_name(cls, directory: str, name: str):
        """Load a persisted index from disk."""
        instance = cls()
        path = os.path.join(directory, f"{name}.pkl")
        if not os.path.exists(path):
            log.warning("BM25 index not found at %s, starting empty", path)
            return instance

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance._docs = data.get("docs", [])
        instance._ids = data.get("ids", [])
        instance._id_to_index = {doc_id: i for i, doc_id in enumerate(instance._ids)}
        instance._rebuild_needed = True
        instance._ensure_built()
        log.info("BM25 index loaded from %s (%d docs)", path, len(instance._docs))
        return instance
