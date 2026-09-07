# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process BM25 over the semantic layer corpus.

Why this exists at all: HugeGraph 1.7 has no full-text index (``SEARCH`` is
declared in ``IndexType`` but marked "not supported now"), and
``HugeGraphMCPServer._bm25_search`` is a ``NotImplementedError`` placeholder
expecting an externally built index. Lexical recall is the half of hybrid
search that catches exact identifiers a vector model smooths away, so it is
implemented here over documents read from the graph.

Metadata corpora are small (hundreds to low thousands of documents), so a
plain Python implementation is faster than standing up a search service.
"""

import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["BM25", "tokenize"]

#: Splits on non-alphanumerics. CJK has no spaces, so consecutive runs of
#: CJK are also emitted as single-character tokens -- coarse, but enough for
#: Chinese table and column names to be matchable.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """Lower-cased tokens, with CJK split per character."""
    if not text:
        return []
    out: List[str] = []
    for match in _TOKEN_RE.finditer(text):
        piece = match.group(0)
        if _CJK_RE.fullmatch(piece):
            out.append(piece)
        else:
            out.append(piece.lower())
    return out


def _maybe_split_camel(token: str) -> List[str]:
    """``customerId`` -> ``['customer', 'id']``.

    Warehouse identifiers are predominantly snake_case or camelCase, and
    BM25 over the raw token would miss a query for "customer id".

    CJK is returned as-is: the camel-case pattern matches ASCII only, so
    without this branch every Chinese token would be silently discarded and
    lexical recall over Chinese metadata would find nothing.
    """
    if _CJK_RE.search(token):
        return [token]
    if "_" in token:
        return [p for p in token.split("_") if p]
    if token.islower() or token.isupper() or token.isdigit():
        return [token]
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+", token)
    return [p.lower() for p in parts if p]


class BM25:
    """Okapi BM25 over an in-memory corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._avg_len = 0.0

    # -- indexing -----------------------------------------------------------

    def fit(self, documents: Iterable[Tuple[str, str]]) -> "BM25":
        """Build the index from ``(doc_id, text)`` pairs. Rebuilds it."""
        self._doc_ids = []
        self._doc_tokens = []
        self._tf = []
        self._df = {}
        for doc_id, text in documents:
            tokens: List[str] = []
            for token in tokenize(text):
                tokens.extend(_maybe_split_camel(token))
            self._doc_ids.append(doc_id)
            self._doc_tokens.append(tokens)
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._tf.append(counts)
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1
        total = sum(len(t) for t in self._doc_tokens)
        self._avg_len = (total / len(self._doc_tokens)) if self._doc_tokens else 0.0
        return self

    # -- querying -----------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return ``(doc_id, score)`` best first. Scores are non-negative."""
        if top_k <= 0 or not self._doc_ids:
            return []
        query_tokens: List[str] = []
        for token in tokenize(query):
            query_tokens.extend(_maybe_split_camel(token))
        if not query_tokens:
            return []

        n_docs = len(self._doc_ids)
        scored: List[Tuple[str, float]] = []
        for idx, counts in enumerate(self._tf):
            doc_len = len(self._doc_tokens[idx])
            score = 0.0
            for token in set(query_tokens):
                freq = counts.get(token)
                if not freq:
                    continue
                df = self._df[token]
                # Standard BM25 IDF, floored at 0 so a term present in every
                # document cannot push a score negative.
                idf = max(0.0, math.log(1 + (n_docs - df + 0.5) / (df + 0.5)))
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / self._avg_len if self._avg_len else 1)
                )
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((self._doc_ids[idx], score))
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def document_terms(self, doc_id: str) -> Sequence[str]:
        """Tokens of an indexed document, for debugging."""
        try:
            return self._doc_tokens[self._doc_ids.index(doc_id)]
        except ValueError:
            return []
