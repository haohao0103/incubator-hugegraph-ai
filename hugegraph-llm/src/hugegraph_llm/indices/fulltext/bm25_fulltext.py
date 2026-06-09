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

"""BM25 full-text search backend (local file storage).

Implements BM25Okapi ranking with jieba Chinese tokenization.
Data is persisted as JSON files on local disk.

This is the default backend for development/testing. For production,
use OceanBaseFTSBackend or ElasticsearchBackend instead.
"""

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import jieba

from hugegraph_llm.config import resource_path
from hugegraph_llm.indices.fulltext.base import FullTextBase

log = logging.getLogger(__name__)

BM25_INDEX_FILE = "bm25_index.json"
BM25_DOCS_FILE = "bm25_docs.json"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> List[str]:
    """Tokenize text using jieba for Chinese and whitespace for others."""
    if not text or not text.strip():
        return []
    raw_tokens = jieba.lcut(text)
    cleaned = []
    for tok in raw_tokens:
        tok = tok.strip().lower()
        if tok and re.match(r"^[\w\u4e00-\u9fff]+$", tok):
            cleaned.append(tok)
    return cleaned


class BM25FullTextBackend(FullTextBase):
    """Local file-based BM25 full-text search.

    Suitable for development, testing, and small-to-medium knowledge bases
    (tens of thousands of chunks). For production, prefer OceanBaseFTSBackend.

    Persistence: JSON files at ``{resource_path}/{graph_name}/bm25/``.
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = k1
        self.b = b
        self._docs: Dict[str, List[str]] = {}
        self._raw_docs: Dict[str, str] = {}
        self._properties: Dict[str, Any] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._dirty = True

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        props: Optional[List[Any]] = None,
    ) -> None:
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        if props is None:
            props = [None] * len(texts)
        for text, doc_id, prop in zip(texts, ids, props):
            self._docs[doc_id] = tokenize(text)
            self._raw_docs[doc_id] = text
            if prop is not None:
                self._properties[doc_id] = prop
        self._dirty = True

    def remove(self, doc_ids: Union[set, List[str]]) -> int:
        id_set = set(doc_ids)
        removed = 0
        for doc_id in id_set:
            if doc_id in self._docs:
                del self._docs[doc_id]
                del self._raw_docs[doc_id]
                self._properties.pop(doc_id, None)
                removed += 1
        if removed > 0:
            self._dirty = True
        return removed

    def _ensure_idf(self) -> None:
        if not self._dirty:
            return
        N = len(self._docs)
        if N == 0:
            self._idf = {}
            self._avgdl = 0.0
            self._dirty = False
            return
        df: Counter = Counter()
        total_len = 0
        for tokens in self._docs.values():
            total_len += len(tokens)
            for tok in set(tokens):
                df[tok] += 1
        self._idf = {}
        for tok, freq in df.items():
            self._idf[tok] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
        self._avgdl = total_len / N
        self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self._ensure_idf()
        if not self._docs:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: Dict[str, float] = {}
        for doc_id, doc_tokens in self._docs.items():
            score = self._score_doc(query_tokens, doc_tokens)
            if score >= min_score:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            results.append({
                "id": doc_id,
                "text": self._raw_docs.get(doc_id, ""),
                "score": round(score, 4),
                "prop": self._properties.get(doc_id),
            })
        return results

    def _score_doc(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        tf_map: Counter = Counter(doc_tokens)
        score = 0.0
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            idf = self._idf.get(tok, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / max(self._avgdl, 1e-8)
            )
            score += idf * numerator / max(denominator, 1e-8)
        return score

    def save_index_by_name(self, *name: str) -> None:
        self._ensure_idf()
        dir_path = os.path.join(resource_path, *name)
        os.makedirs(dir_path, exist_ok=True)
        index_data = {
            "k1": self.k1,
            "b": self.b,
            "idf": self._idf,
            "avgdl": self._avgdl,
            "docs": self._docs,
            "raw_docs": self._raw_docs,
        }
        with open(os.path.join(dir_path, BM25_INDEX_FILE), "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)
        with open(os.path.join(dir_path, BM25_DOCS_FILE), "w", encoding="utf-8") as f:
            json.dump(self._properties, f, ensure_ascii=False, default=str)

    @classmethod
    def from_name(cls, *name: str) -> "BM25FullTextBackend":
        dir_path = os.path.join(resource_path, *name)
        index_file = os.path.join(dir_path, BM25_INDEX_FILE)
        docs_file = os.path.join(dir_path, BM25_DOCS_FILE)
        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            return cls()
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            instance = cls(k1=data.get("k1", DEFAULT_K1), b=data.get("b", DEFAULT_B))
            instance._idf = data.get("idf", {})
            instance._avgdl = data.get("avgdl", 0.0)
            instance._docs = data.get("docs", {})
            instance._raw_docs = data.get("raw_docs", {})
            with open(docs_file, "r", encoding="utf-8") as f:
                instance._properties = {str(k): v for k, v in json.load(f).items()}
            instance._dirty = False
            return instance
        except Exception as e:
            log.warning("Failed to load BM25 index: %s", e)
            return cls()

    @staticmethod
    def exist(*name: str) -> bool:
        dir_path = os.path.join(resource_path, *name)
        return os.path.exists(os.path.join(dir_path, BM25_INDEX_FILE)) and os.path.exists(
            os.path.join(dir_path, BM25_DOCS_FILE)
        )

    @staticmethod
    def clean(*name: str) -> bool:
        dir_path = os.path.join(resource_path, *name)
        deleted = False
        for fname in [BM25_INDEX_FILE, BM25_DOCS_FILE]:
            fpath = os.path.join(dir_path, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                deleted = True
        return deleted
