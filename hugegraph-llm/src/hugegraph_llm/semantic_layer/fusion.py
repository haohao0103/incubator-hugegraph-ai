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

"""Reciprocal Rank Fusion over an arbitrary number of ranked lists.

``HugeGraphMCPServer._rrf_fusion`` fuses exactly two lists, which suits a
vector+BM25 pair but not the three-way recall this layer uses (vector,
business-term, BM25) -- and a fourth way (query-log co-occurrence) is
planned. Rather than nesting pairwise fusions (order-dependent and hard to
reason about), candidate lists are fused in one pass.

RRF is used instead of score normalisation because the three sources emit
incomparable scores: cosine similarity in [0,1], unbounded BM25, and a
binary-ish term match. Only the *ranks* are comparable.
"""

from typing import Dict, List, Sequence, Tuple

__all__ = ["rrf_fuse", "DEFAULT_RRF_K"]

DEFAULT_RRF_K = 60


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    k: int = DEFAULT_RRF_K,
    *,
    weights: Sequence[float] | None = None,
) -> List[Tuple[str, float]]:
    """Fuse ranked lists into one ranking.

    :param ranked_lists: lists of ``(id, score)``, each best-first.
    :param k: RRF damping; larger flattens the contribution of top ranks.
    :param weights: optional per-list multiplier, same length as
        ``ranked_lists``. A trusted source (for example an exact business-term
        match) can be given more influence without rescaling its scores.
    :returns: ``(id, fused_score)`` best-first.
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")

    scores: Dict[str, float] = {}
    evidence: Dict[str, int] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else weights[list_index]
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
            evidence[doc_id] = evidence.get(doc_id, 0) + 1

    # Ties are broken by how many sources agreed: a table found by both the
    # vector index and the business-term lookup is a stronger candidate than
    # one found by a single source at the same fused score.
    ordered = sorted(
        scores.items(), key=lambda item: (-item[1], -evidence[item[0]], item[0])
    )
    return ordered
