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

"""Memory retrieval: temporal filtering plus time-decay scoring.

This sits **on the general retrieval stack**, not beside it:

* :class:`KGRetriever` (``operators/graph_op/kg_retriever_base.py``) is the
  shared retriever contract; :class:`TemporalRetriever` is one
  implementation of it, like any other channel.
* :class:`ReciprocalRankFusion` (``operators/graph_op/rrf_fusion.py``) does
  the merging. It is domain-agnostic -- it only knows ranked id lists -- so
  memory does not need its own fusion.

What memory uniquely contributes is the **temporal dimension**, and that is
all this module implements:

1. **Temporal gating** -- only facts valid (and known) at the query time
   are candidates at all. This is a *filter*, not a score: a fact that was
   not true at T is wrong to surface no matter how well it matches the
   question.
2. **Time decay** -- among surviving facts, older ones score lower. Memory
   is expected to fade; a recency signal is what makes it behave that way.

Note the ordering: gating happens **before** scoring. Scoring first and
then filtering would let an ancient, expired fact consume rank budget and
push a current one out of ``top_k``.

The decay is exponential with a configurable half-life, matching the
behaviour validated in the earlier temporal-KG PoC.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

from hugegraph_llm.memory.schema import OPEN
from hugegraph_llm.memory.temporal import Fact, TemporalStore
from hugegraph_llm.operators.graph_op.kg_retriever_base import (
    KGRetriever,
    RetrieverResult,
    RetrieverResultItem,
)
from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion

logger = logging.getLogger(__name__)

__all__ = [
    "TemporalRetriever",
    "MemoryRecallConfig",
    "recall",
    "time_decay_score",
]

#: Decay constants. A half-life of 30 days means a 30-day-old memory scores
#: 0.5, 60 days 0.25 -- aggressive enough that stale preferences stop
#: dominating, gentle enough that last month's context still counts.
DEFAULT_HALF_LIFE_MS = 30 * 24 * 3600 * 1000


class MemoryRecallConfig:
    """Knobs for one recall call."""

    def __init__(
        self,
        *,
        top_k: int = 10,
        fusion_k: int = 60,
        half_life_ms: int = DEFAULT_HALF_LIFE_MS,
        include_superseded: bool = False,
    ) -> None:
        self.top_k = top_k
        self.fusion_k = fusion_k
        self.half_life_ms = half_life_ms
        self.include_superseded = include_superseded


def time_decay_score(
    fact: Fact, now: int, half_life_ms: int = DEFAULT_HALF_LIFE_MS
) -> float:
    """Exponential recency weight in (0, 1].

    Measures how long the fact has *been true*, not how long we have known
    it: a fact that held for a year is old context even if we only learned
    it yesterday, and one still valid today is current regardless of when
    it was recorded.
    """
    if half_life_ms <= 0:
        return 1.0
    # Latest moment the fact was relevant: still valid -> now, else the
    # moment it stopped being true.
    reference = now if fact.invalid_at == OPEN else min(fact.invalid_at, now)
    age = max(0, reference - fact.valid_at)
    return float(math.pow(0.5, age / half_life_ms))


class TemporalRetriever(KGRetriever):
    """Recall memories valid at a point in time, ranked by recency.

    Implements the shared :class:`KGRetriever` contract
    (``search(query, **kwargs) -> RetrieverResult``), so it can be composed
    with any other channel through the shared RRF implementation.
    """

    def __init__(self, store: TemporalStore, now: Optional[int] = None) -> None:
        self.store = store
        self.now = now

    # -- KGRetriever contract ----------------------------------------------

    def get_search_results(self, query: str, **kwargs: Any) -> List[Fact]:
        """Raw temporal retrieval. ``query`` is the as-of timestamp context.

        Keyword args: ``at`` (as-of time, default now), ``include_superseded``.
        """
        at = kwargs.get("at", self.now)
        include = kwargs.get("include_superseded", False)
        facts = self.store.as_of(at, include_superseded=include) if at is not None \
            else self.store.all_facts()
        return facts

    def _result_items(self, raw: Any) -> List[Fact]:
        return list(raw or [])

    def get_result_formatter(self):
        def formatter(fact: Fact) -> RetrieverResultItem:
            return RetrieverResultItem(
                content=fact.fact,
                metadata={
                    "uuid": fact.uuid,
                    "name": fact.name,
                    "source_uuid": fact.source_uuid,
                    "target_uuid": fact.target_uuid,
                    "valid_at": fact.valid_at,
                    "invalid_at": fact.invalid_at,
                    "expired_at": fact.expired_at,
                    "is_current": fact.is_current,
                },
                score=None,
            )

        return formatter

    def _result_metadata(self, raw: Any) -> Dict[str, Any]:
        return {"count": len(list(raw or []))}


def recall(
    store: TemporalStore,
    *,
    at: Optional[int] = None,
    now: Optional[int] = None,
    config: Optional[MemoryRecallConfig] = None,
    extra_channels: Optional[Sequence[Sequence[str]]] = None,
) -> RetrieverResult:
    """Recall memories as of ``at``, ranked by recency.

    :param at: as-of time; None means "current beliefs".
    :param now: reference time for decay; defaults to ``at``.
    :param extra_channels: additional ranked fact-id lists from other
        retrievers (a vector channel, for example). Each is fused with the
        temporal channel through the shared RRF implementation, so adding a
        channel needs no new fusion code.
    """
    cfg = config or MemoryRecallConfig()
    reference = now if now is not None else at
    if reference is None:
        import time

        reference = int(time.time() * 1000)

    retrieved = TemporalRetriever(store, now=reference).search(
        "", at=at, include_superseded=cfg.include_superseded
    )

    # Decay score, then rank. Gating already happened in as_of(); scoring
    # afterwards keeps expired facts from consuming rank budget.
    fact_index = {f.uuid: f for f in store.all_facts()}
    decayed: Dict[str, float] = {}
    for item in retrieved.items:
        match = fact_index.get(item.metadata.get("uuid"))
        if match is None:
            continue
        decayed[match.uuid] = time_decay_score(
            match, reference, cfg.half_life_ms
        )

    temporal_ranking = [
        uuid for uuid, _score in sorted(decayed.items(), key=lambda kv: -kv[1])
    ]

    # Every channel is a named (channel, ranked_ids) pair -- RRF accepts
    # bare lists too, but unnamed channels are untraceable in metadata and
    # a mixed list breaks the caller's own unpacking.
    channels: List[Any] = [("temporal", temporal_ranking)]
    for index, channel in enumerate(extra_channels or []):
        channels.append((f"extra_{index}", list(channel)))

    fused = ReciprocalRankFusion(k=cfg.fusion_k).fuse(channels)

    by_uuid = {
        item.metadata["uuid"]: item for item in retrieved.items
    }
    items: List[RetrieverResultItem] = []
    for uuid in fused.top_k(cfg.top_k):
        item = by_uuid.get(uuid)
        if item is None:
            continue
        item.score = decayed.get(uuid)
        items.append(item)

    return RetrieverResult(
        items=items,
        metadata={
            "as_of": at,
            "decay_reference": reference,
            "half_life_ms": cfg.half_life_ms,
            "candidates": len(decayed),
            "channels": [name for name, _ranked in channels],
        },
    )
