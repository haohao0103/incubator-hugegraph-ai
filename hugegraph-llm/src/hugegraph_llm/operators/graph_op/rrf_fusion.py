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

"""Reciprocal Rank Fusion (RRF) for multi-channel result merging.

When multiple retrieval channels (vector, keyword, graph traversal, etc.)
return ranked result lists, RRF provides a robust, score-free method to
merge them into a single ranked list.

Reference:
    Cormack, Gordon V., et al. "Reciprocal rank fusion outperforms
    condorcet and individual rank learning methods." SIGIR 2009.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence


class RRFResults:
    """Immutable container for RRF-fused results."""

    __slots__ = ("items", "scores")

    def __init__(self, items: List[Any], scores: Dict[Any, float]):
        self.items = items
        self.scores = scores

    def top_k(self, k: int) -> List[Any]:
        return self.items[:k]

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"RRFResults(n={len(self.items)})"


class ReciprocalRankFusion:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Usage::

        rrf = ReciprocalRankFusion(k=60, min_score=0.0)
        fused = rrf.fuse([
            ("vector", ["doc_a", "doc_b", "doc_c"]),
            ("keyword", ["doc_b", "doc_d", "doc_a"]),
            ("graph", ["doc_c", "doc_e"]),
        ])
        print(fused.top_k(3))  # merged top 3
    """

    def __init__(self, k: int = 60, min_score: float = 0.0):
        """Initialize RRF.

        Args:
            k: RRF constant (default 60). Larger *k* dampens the
               effect of rank position differences. Typical values
               range from 20-100.
            min_score: Minimum RRF score to include in results.
        """
        self._k = k
        self._min_score = min_score

    def fuse(
        self,
        ranked_lists: Sequence,
        ids: Optional[Sequence] = None,
    ) -> RRFResults:
        """Fuse multiple ranked lists into a single ranking.

        Args:
            ranked_lists: An iterable of ``(channel_name, [item_id, ...])``
                tuples or just ``[item_id, ...]`` lists.  If tuples,
                *channel_name* is used for traceability in metadata.
            ids: Optional sequence of (channel_name, [item_id, ...]) tuples
                that will be matched to *ranked_lists* by index.

        Returns:
            An ``RRFResults`` object with fused ranking.
        """
        scores: Dict[Any, float] = defaultdict(float)
        # Track which channels contributed each item (for metadata)
        channel_map: Dict[Any, List[str]] = defaultdict(list)

        for entry in ranked_lists:
            if isinstance(entry, tuple) and len(entry) == 2:
                channel, items = entry
            else:
                channel = "default"
                items = entry

            for rank, item in enumerate(items, start=1):
                scores[item] += 1.0 / (rank + self._k)
                channel_map[item].append(channel)

        # Filter by minimum score and sort descending
        sorted_items = sorted(
            ((score, item) for item, score in scores.items() if score >= self._min_score),
            key=lambda x: x[0],
            reverse=True,
        )

        final_items = [item for _, item in sorted_items]
        final_scores = {item: score for score, item in sorted_items}

        return RRFResults(final_items, final_scores)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def fuse_results(
    *ranked_lists: List[Any],
    k: int = 60,
    min_score: float = 0.0,
) -> List[Any]:
    """Quick-fuse multiple ranked lists and return the top items.

    >>> fuse_results(["a", "b", "c"], ["b", "d", "a"])
    ['b', 'a', 'c', 'd']
    """
    rrf = ReciprocalRankFusion(k=k, min_score=min_score)
    return rrf.fuse(ranked_lists).items


def fuse_results_with_scores(
    *ranked_lists: List[Any],
    k: int = 60,
    min_score: float = 0.0,
) -> List[tuple]:
    """Quick-fuse and return ``(item, score)`` pairs sorted by score desc."""
    rrf = ReciprocalRankFusion(k=k, min_score=min_score)
    result = rrf.fuse(ranked_lists)
    return [(item, result.scores[item]) for item in result.items]
