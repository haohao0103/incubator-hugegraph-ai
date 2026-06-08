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

"""BM25 keyword search operator.

Provides full-text retrieval using BM25Okapi scoring, complementary
to vector search. Designed to be used alongside VectorIndexQuery in
hybrid retrieval pipelines.

Usage::

    from hugegraph_llm.operators.index_op.bm25_index_query import BM25IndexQuery

    bm25_query = BM25IndexQuery(topk=5, min_score=1.0)
    context = bm25_query.run({"query": "user question text"})
    # context["bm25_result"] = [{"id": ..., "text": ..., "score": ..., "prop": ...}]
"""

import logging
from typing import Any, Dict

from hugegraph_llm.config import huge_settings
from hugegraph_llm.indices.keyword_index import BM25Index

log = logging.getLogger(__name__)


class BM25IndexQuery:
    """BM25 full-text search operator.

    Loads a persisted BM25 index and searches it with BM25Okapi scoring.
    Results are written to ``context["bm25_result"]`` as a list of dicts
    with keys: ``id``, ``text``, ``score``, ``prop``.

    Follows the same naming convention as VectorIndexQuery:
    index stored at ``{resource_path}/{graph_name}/bm25/``.
    """

    def __init__(self, topk: int = 5, min_score: float = 0.0):
        """Initialize BM25 query operator.

        Args:
            topk: Maximum number of results to return.
            min_score: Minimum BM25 score threshold (0.0 = no filter).
        """
        self.topk = topk
        self.min_score = min_score
        self._index = BM25Index.from_name(huge_settings.graph_name, "bm25")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute BM25 search.

        Args:
            context: Must contain ``"query"`` key with the search text.

        Returns:
            Updated context with ``"bm25_result"`` key containing
            a list of scored result dicts.
        """
        query = context.get("query", "")
        if not query:
            context["bm25_result"] = []
            return context

        try:
            results = self._index.search(
                query, top_k=self.topk, min_score=self.min_score
            )
            context["bm25_result"] = results
            log.debug(
                "BM25 search returned %d results for query: %s",
                len(results),
                query[:50],
            )
        except Exception as e:
            log.error("BM25 search failed: %s", e)
            context["bm25_result"] = []

        return context
