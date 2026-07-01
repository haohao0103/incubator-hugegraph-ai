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

"""Reciprocal Rank Fusion (RRF) for combining multi-channel retrieval results.

RRF is a simple, parameter-free rank fusion method that combines ranked
lists from different retrieval channels (vector, keyword, graph) into a
single ranking.  For each document *d*:

    score(d) = sum over channels c of  1 / (k + rank_c(d))

where *k* is a tuning constant (default 60) and *rank_c(d)* is the 1-based
rank of *d* in channel *c* (documents not present in a channel contribute 0).

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log

DEFAULT_K = 60


class ReciprocalRankFusion:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    k : int
        RRF damping constant.  Higher values smooth differences between
        top-ranked and lower-ranked items.  Default 60 (the value
        recommended in the original paper).
    """

    def __init__(self, k: int = DEFAULT_K):
        self._k = k

    def fuse(
        self,
        ranked_lists: List[List[Dict[str, Any]]],
        id_key: str = "id",
        score_key: str = "score",
    ) -> List[Dict[str, Any]]:
        """Fuse multiple ranked lists into a single ranked list.

        Parameters
        ----------
        ranked_lists:
            A list of ranked result lists.  Each inner list is a list of
            dicts, each containing at least *id_key*.  The list is assumed
            to be sorted by relevance (most relevant first).
        id_key:
            Key used to identify a document across channels.  Default ``"id"``.
        score_key:
            Key used to store the fused RRF score in the output.  Default
            ``"score"``.

        Returns
        -------
        list of dict
            Fused results sorted by RRF score descending.  Each dict
            contains *id_key*, *score_key*, and any other keys from the
            first occurrence of the document.
        """
        rrf_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}

        for channel_idx, ranked in enumerate(ranked_lists):
            for rank, item in enumerate(ranked, start=1):
                doc_id = str(item.get(id_key, ""))
                if not doc_id:
                    continue
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._k + rank)
                if doc_id not in doc_data:
                    doc_data[doc_id] = dict(item)

        fused = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = dict(doc_data[doc_id])
            entry[score_key] = round(score, 6)
            fused.append(entry)

        log.debug("RRF fused %d channels -> %d unique docs", len(ranked_lists), len(fused))
        return fused


def fuse_results_with_scores(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = DEFAULT_K,
    id_key: str = "id",
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """Convenience function: create an RRF instance and fuse in one call."""
    return ReciprocalRankFusion(k=k).fuse(ranked_lists, id_key=id_key, score_key=score_key)
