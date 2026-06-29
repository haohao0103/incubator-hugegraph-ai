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

"""
Sparse vector hybrid retrieval for AI memory.

Two backends:
  - "bm25_tokens": jieba tokenization + TF-IDF-like sparse weights,
                   cheap and no extra model download.
  - "splade": SPLADE-v2/former (requires transformers, heavy).

The sparse representation is a dict {term: weight} that can be merged with
a dense vector score via weighted sum.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.utils.log import log


def get_sparse_index():
    """Factory."""
    if not memory_settings.sparse_enabled:
        return None
    if memory_settings.sparse_backend == "splade":
        return SpladeSparseIndex()
    return BM25TokensSparseIndex()


class BaseSparseIndex:
    def encode(self, text: str) -> Dict[str, float]:
        raise NotImplementedError

    def score(self, query_sparse: Dict[str, float], doc_sparse: Dict[str, float]) -> float:
        """Dot product between sparse vectors."""
        score = 0.0
        for term, qweight in query_sparse.items():
            score += qweight * doc_sparse.get(term, 0.0)
        return score


class BM25TokensSparseIndex(BaseSparseIndex):
    """Lightweight sparse vector based on jieba + term frequency."""

    def __init__(self):
        try:
            import jieba
            self.jieba = jieba
        except ImportError as exc:
            raise ImportError("jieba is required for bm25_tokens sparse index") from exc
        self._df: Dict[str, int] = Counter()
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        tokens = [t.strip().lower() for t in self.jieba.lcut(text) if len(t.strip()) > 1]
        return tokens

    def encode(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        max_tf = max(tf.values())
        sparse = {}
        for term, count in tf.items():
            idf = math.log(1 + (self._doc_count + 1) / (self._df.get(term, 0) + 1))
            sparse[term] = round((count / max_tf) * idf, 4)
        return sparse

    def add_document(self, text: str):
        tokens = self._tokenize(text)
        self._doc_count += 1
        for term in set(tokens):
            self._df[term] += 1


class SpladeSparseIndex(BaseSparseIndex):
    """SPLADE sparse encoder (requires transformers)."""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for SPLADE sparse index"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> Dict[str, float]:
        import torch

        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**tokens).logits
        relu = torch.relu(output)
        max_over_seq = torch.max(relu, dim=1).values.squeeze()
        cols = max_over_seq.nonzero(as_tuple=True)[0]
        weights = max_over_seq[cols].tolist()
        ids = cols.tolist()
        sparse = {}
        for idx, weight in zip(ids, weights):
            token = self.tokenizer.decode([idx])
            if token and token not in self.tokenizer.all_special_tokens:
                sparse[token] = round(weight, 4)
        return sparse


def fuse_dense_sparse(
    dense_scores: List[Tuple[str, float]],
    docs: List[Tuple[str, str]],
    query: str,
    sparse_weight: float = None,
) -> List[Tuple[str, float]]:
    """
    Merge dense retrieval scores with sparse vector scores.

    Args:
        dense_scores: list of (doc_id, dense_score)
        docs: list of (doc_id, doc_text)
        query: raw query text
        sparse_weight: weight for sparse score (default from config)

    Returns:
        list of (doc_id, fused_score) sorted descending
    """
    sparse_weight = sparse_weight if sparse_weight is not None else memory_settings.sparse_weight
    sparse_index = get_sparse_index()
    if sparse_index is None or sparse_weight <= 0:
        return sorted(dense_scores, key=lambda x: x[1], reverse=True)

    query_sparse = sparse_index.encode(query)
    doc_sparse_map = {doc_id: sparse_index.encode(text) for doc_id, text in docs}

    # Normalize dense scores
    if dense_scores:
        max_dense = max(s for _, s in dense_scores)
        min_dense = min(s for _, s in dense_scores)
        dense_range = max_dense - min_dense if max_dense > min_dense else 1.0
    else:
        dense_range = 1.0

    fused = []
    for doc_id, dscore in dense_scores:
        norm_dense = (dscore - min_dense) / dense_range if dense_range else 0.0
        sscore = sparse_index.score(query_sparse, doc_sparse_map.get(doc_id, {}))
        fused_score = (1 - sparse_weight) * norm_dense + sparse_weight * sscore
        fused.append((doc_id, round(fused_score, 6)))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
