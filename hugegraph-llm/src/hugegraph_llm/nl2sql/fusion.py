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

"""Result-level fusion for multi-recall schema linking (RRF / weighted score).

Seed-level merging (``SchemaLinker.link_multi``) blends *seeds* before one PPR.
Result-level fusion instead ranks each recall path independently (lexical /
vector / BM25) and fuses the *rankings* — the standard RRF trick. Borrowed
from the parallel NL2SQL demo branch's ``kg_multi_retrieval``; pure functions,
no external deps.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def rrf_fuse(
    lists: Sequence[Sequence[T]],
    key_fn,
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[T, float]]:
    """Reciprocal-rank fusion: each path contributes ``w / (k + rank)``.

    ``key_fn(item)`` maps an item to its identity (e.g. node_id). Items keep
    the highest-scoring source instance; the returned score is the fused one.
    """
    weights = weights or {}
    acc: Dict[object, float] = defaultdict(float)
    best: Dict[object, T] = {}
    for path in lists:
        w = float(weights.get(getattr(path[0], "source", ""), 1.0)) if path else 1.0
        for rank, item in enumerate(path):
            key = key_fn(item)
            acc[key] += w / (k + rank + 1)
            if key not in best:
                best[key] = item
    ordered = sorted(acc.items(), key=lambda kv: -kv[1])
    return [(best[key], score) for key, score in ordered]


def score_fuse(
    lists: Sequence[Sequence[T]],
    key_fn,
    score_fn,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[T, float]]:
    """Weighted score-sum fusion across recall paths.

    Use when all paths share a comparable score scale; deterministic ties are
    resolved by summing, unlike RRF which flattens to ~1/(k+rank).
    """
    weights = weights or {}
    acc: Dict[object, float] = defaultdict(float)
    best: Dict[object, T] = {}
    for path in lists:
        w = float(weights.get(getattr(path[0], "source", ""), 1.0)) if path else 1.0
        for item in path:
            key = key_fn(item)
            acc[key] += w * score_fn(item)
            if key not in best:
                best[key] = item
    ordered = sorted(acc.items(), key=lambda kv: -kv[1])
    return [(best[key], score) for key, score in ordered]
