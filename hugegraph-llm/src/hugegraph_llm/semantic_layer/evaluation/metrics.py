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

"""Per-case retrieval metrics.

Four things are measured, and one is deliberately not:

* **table recall** -- P@k / R@k against the tables the gold SQL needs.
* **token cost** -- tokens spent, versus a full-schema baseline.
* **join soundness** -- whether the retrieved tables can actually be joined
  using declared foreign keys. This is the proxy for "did we prevent a
  hallucinated JOIN": a set with an unreachable member invites invention,
  whatever the model does next.
* **term recall** -- whether business-term recall fired and resolved the
  question to a table.

**Not measured: execution accuracy.** Without a warehouse the gold SQL
cannot be run, so any "correctness" figure would be invented. Reporting a
metric that cannot be reproduced is precisely the failure this milestone
exists to avoid, so the field is left out rather than filled in.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from hugegraph_llm.semantic_layer.evaluation.dataset import EvalCase
from hugegraph_llm.semantic_layer.join_path import find_join_path
from hugegraph_llm.semantic_layer.readers import SemanticProjection

__all__ = [
    "CaseResult",
    "MetricSummary",
    "RetrievalEvaluator",
    "precision_at_k",
    "recall_at_k",
]


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the first ``k`` retrieved tables that are relevant.

    Divides by ``k`` even when fewer tables were returned -- standard P@k.
    Dividing by ``len(retrieved)`` instead would score a single-table answer
    as 1.0 and make the metric incomparable across cases with different
    retrieval sizes; ``excess_ratio`` covers that view properly.
    """
    if k <= 0:
        return 0.0
    top = list(retrieved[:k])
    hits = sum(1 for table in top if table in gold)
    return hits / k


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold tables present in the first ``k`` retrieved."""
    if not gold:
        return 0.0
    top = set(list(retrieved[:k]))
    hits = sum(1 for table in gold if table in top)
    return hits / len(gold)


@dataclass
class CaseResult:
    """What one question produced."""

    question: str
    source: str
    gold_tables: List[str]
    retrieved_tables: List[str]
    multi_table: bool

    # recall
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    recall_full: float = 0.0
    #: True when every gold table was retrieved.
    all_gold_found: bool = False

    # cost
    tokens_used: int = 0
    baseline_tokens: int = 0
    token_saving: float = 0.0
    #: ``len(retrieved) / len(gold)`` -- noise tables per gold table.
    #: This, not P@5, is the honest precision measure here: gold sets are
    #: small (ACME averages 1.56 tables), so P@5's fixed 5 slots cap it at
    #: ~0.31 even for a perfect retriever, making 0.29 look like a problem
    #: when it is actually 94% of the ceiling. An excess ratio of 1.0 means
    #: surgical retrieval; 8.8 means the model wades through eight noise
    #: tables per relevant one.
    excess_ratio: float = 0.0

    # join soundness
    retrieved_joinable: bool = False
    gold_joinable: bool = False
    unreachable_tables: List[str] = field(default_factory=list)

    # term recall
    term_recall_fired: bool = False
    sources_used: List[str] = field(default_factory=list)

    # executability (present when a warehouse executor was supplied)
    #: gold SQL ran without error against the warehouse.
    gold_executable: Optional[bool] = None

    @property
    def is_success(self) -> bool:
        """Every gold table retrieved, and the set is joinable."""
        return self.all_gold_found and self.retrieved_joinable

    def to_dict(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "source": self.source,
            "multi_table": self.multi_table,
            "gold_tables": self.gold_tables,
            "retrieved_tables": self.retrieved_tables,
            "precision@5": round(self.precision_at_5, 3),
            "recall@5": round(self.recall_at_5, 3),
            "recall_full": round(self.recall_full, 3),
            "all_gold_found": self.all_gold_found,
            "excess_ratio": round(self.excess_ratio, 2),
            "tokens_used": self.tokens_used,
            "baseline_tokens": self.baseline_tokens,
            "token_saving": round(self.token_saving, 3),
            "retrieved_joinable": self.retrieved_joinable,
            "gold_joinable": self.gold_joinable,
            "term_recall_fired": self.term_recall_fired,
            "sources_used": self.sources_used,
            "success": self.is_success,
        }


@dataclass
class MetricSummary:
    """Averages over a set of :class:`CaseResult`."""

    count: int = 0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    recall_full: float = 0.0
    all_gold_found_rate: float = 0.0
    mean_tokens: float = 0.0
    mean_baseline_tokens: float = 0.0
    token_saving: float = 0.0
    #: Mean of per-case ``retrieved/gold``. Unlike P@5 this is not capped by
    #: the gold set size, so it can actually distinguish good retrieval from
    #: over-retrieval. 1.0 is perfect; lower is impossible.
    mean_excess_ratio: float = 0.0
    retrieved_joinable_rate: float = 0.0
    gold_joinable_rate: float = 0.0
    term_recall_rate: float = 0.0
    success_rate: float = 0.0
    #: Fraction of gold SQL that executed cleanly on the warehouse.
    #: None when no executor was supplied.
    gold_executable_rate: Optional[float] = None

    @classmethod
    def from_results(cls, results: Sequence[CaseResult]) -> "MetricSummary":
        if not results:
            return cls()
        n = len(results)
        total_tokens = sum(r.tokens_used for r in results)
        total_baseline = sum(r.baseline_tokens for r in results)
        return cls(
            count=n,
            precision_at_5=sum(r.precision_at_5 for r in results) / n,
            recall_at_5=sum(r.recall_at_5 for r in results) / n,
            recall_full=sum(r.recall_full for r in results) / n,
            all_gold_found_rate=sum(1 for r in results if r.all_gold_found) / n,
            mean_tokens=total_tokens / n,
            mean_baseline_tokens=total_baseline / n,
            token_saving=(
                1 - total_tokens / total_baseline if total_baseline else 0.0
            ),
            mean_excess_ratio=sum(r.excess_ratio for r in results) / n,
            retrieved_joinable_rate=sum(
                1 for r in results if r.retrieved_joinable
            ) / n,
            gold_joinable_rate=sum(1 for r in results if r.gold_joinable) / n,
            term_recall_rate=sum(1 for r in results if r.term_recall_fired) / n,
            success_rate=sum(1 for r in results if r.is_success) / n,
            gold_executable_rate=(
                sum(1 for r in results if r.gold_executable) / n
                if any(r.gold_executable is not None for r in results)
                else None
            ),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "precision@5": round(self.precision_at_5, 3),
            "recall@5": round(self.recall_at_5, 3),
            "recall_full": round(self.recall_full, 3),
            "all_gold_found_rate": round(self.all_gold_found_rate, 3),
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_baseline_tokens": round(self.mean_baseline_tokens, 1),
            "token_saving": round(self.token_saving, 3),
            "mean_excess_ratio": round(self.mean_excess_ratio, 2),
            "retrieved_joinable_rate": round(self.retrieved_joinable_rate, 3),
            "gold_joinable_rate": round(self.gold_joinable_rate, 3),
            "term_recall_rate": round(self.term_recall_rate, 3),
            "success_rate": round(self.success_rate, 3),
            "gold_executable_rate": (
                round(self.gold_executable_rate, 3)
                if self.gold_executable_rate is not None
                else None
            ),
        }


class RetrievalEvaluator:
    """Scores retrieval results against a dataset.

    :param projection: supplies join paths and table names.
    :param baseline_tokens: cost of injecting the *whole* schema, used as
        the denominator for token saving. Measured once, not per case.
    """

    def __init__(
        self,
        projection: SemanticProjection,
        baseline_tokens: int = 0,
        k: int = 5,
    ) -> None:
        self.projection = projection
        self.baseline_tokens = baseline_tokens
        self.k = k

    def score(
        self,
        case: EvalCase,
        retrieved: Sequence[str],
        *,
        tokens_used: int = 0,
        term_recall_fired: bool = False,
        sources_used: Optional[List[str]] = None,
    ) -> CaseResult:
        """Score one retrieval against its case."""
        gold = [t for t in case.gold_tables if t in self.projection.tables]
        retrieved_list = list(retrieved)

        result = CaseResult(
            question=case.question,
            source=case.source,
            gold_tables=gold,
            retrieved_tables=retrieved_list,
            multi_table=case.multi_table,
            precision_at_5=precision_at_k(retrieved_list, gold, self.k),
            recall_at_5=recall_at_k(retrieved_list, gold, self.k),
            recall_full=recall_at_k(retrieved_list, gold, len(retrieved_list)),
            all_gold_found=bool(gold) and all(t in retrieved_list for t in gold),
            tokens_used=tokens_used,
            baseline_tokens=self.baseline_tokens,
            token_saving=(
                1 - tokens_used / self.baseline_tokens
                if self.baseline_tokens
                else 0.0
            ),
            term_recall_fired=term_recall_fired,
            sources_used=list(sources_used or []),
        )

        result.retrieved_joinable, result.unreachable_tables = _joinable(
            self.projection, retrieved_list
        )
        result.gold_joinable, _ = _joinable(self.projection, gold)
        result.excess_ratio = (
            len(retrieved_list) / len(gold) if gold else float(len(retrieved_list))
        )
        return result


def _joinable(
    projection: SemanticProjection, tables: Sequence[str]
) -> tuple[bool, List[str]]:
    """Are all ``tables`` mutually reachable through declared joins?

    A single table is trivially joinable. For more, every member must be
    reachable from the first via :func:`find_join_path`, which traverses
    ``REFERENCES`` only -- ``LINEAGE`` and ``CO_OCCUR`` name no columns and
    so cannot yield a real ``ON`` clause.
    """
    known = [t for t in tables if t in projection.tables]
    if len(known) <= 1:
        return True, []
    root = known[0]
    unreachable = [
        t for t in known[1:] if not find_join_path(projection, root, t).found
    ]
    return (not unreachable), unreachable
