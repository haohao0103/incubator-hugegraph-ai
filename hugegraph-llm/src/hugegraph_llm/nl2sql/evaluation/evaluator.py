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

"""
Clause-level evaluation for Text2SQL.

Exact-match on whole SQL is too coarse to guide engineering: a query can be
logically identical yet differ in whitespace, alias naming, or predicate
order, and it can also be *coincidentally* identical while reasoning wrong.

Following SuperSonic's evaluation approach, SQL is decomposed into clauses
(select / where / group by / order by / limit) and precision, recall and F1
are computed per clause over normalised token sets. This makes a schema-layer
change measurable: "did fixing the metric definition improve the select
clause without regressing the where clause?"

Execution accuracy is supported optionally via an injected executor, so that
semantic equivalence (different SQL, same result set) can be credited.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from hugegraph_llm.utils.log import log

# Each clause is captured up to the next clause keyword or end of statement.
CLAUSE_PATTERNS: Dict[str, str] = {
    "select": r"select\s+(.*?)\s+from\s",
    "from": r"\bfrom\s+(.*?)(?:\s+where\s|\s+group\s+by\s|\s+order\s+by\s|\s+limit\s|$)",
    "where": r"\bwhere\s+(.*?)(?:\s+group\s+by\s|\s+order\s+by\s|\s+limit\s|$)",
    "group_by": r"\bgroup\s+by\s+(.*?)(?:\s+having\s|\s+order\s+by\s|\s+limit\s|$)",
    "order_by": r"\border\s+by\s+(.*?)(?:\s+limit\s|$)",
    "limit": r"\blimit\s+(\d+)",
}

# Tokens that carry no meaning for comparison.
_STOPWORDS = {"as", "and", "or", "the", "a", "an", "on", "in", "is", "null"}


@dataclass
class ClauseScore:
    """Precision / recall / F1 for a single clause."""

    clause: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matched: bool = False


@dataclass
class SQLScore:
    """Aggregated evaluation result for one predicted/gold pair."""

    clauses: Dict[str, ClauseScore] = field(default_factory=dict)
    exact_match: bool = False
    execution_match: Optional[bool] = None

    @property
    def mean_f1(self) -> float:
        scores = [c.f1 for c in self.clauses.values()]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class AggregateScore:
    """Averaged metrics over a batch of pairs."""

    count: int = 0
    exact_match_rate: float = 0.0
    execution_accuracy: Optional[float] = None
    mean_f1: float = 0.0
    per_clause: Dict[str, float] = field(default_factory=dict)


class SQLEvaluator:
    """Clause-level Text2SQL evaluation."""

    def __init__(self, executor: Optional[Callable[[str], List]] = None):
        """
        :param executor: Optional callable running a SQL string and returning
                         its rows. When supplied, execution accuracy is also
                         computed (different SQL, same result set = correct).
        """
        self._executor = executor

    def evaluate(self, predicted: str, gold: str) -> SQLScore:
        """Score one predicted SQL against its gold reference."""
        score = SQLScore()
        pred_norm, gold_norm = _normalise(predicted), _normalise(gold)
        score.exact_match = pred_norm == gold_norm

        for clause, pattern in CLAUSE_PATTERNS.items():
            p_text = _extract_clause(pred_norm, pattern)
            g_text = _extract_clause(gold_norm, pattern)

            if not p_text and not g_text:
                continue  # clause absent from both: nothing to compare

            p_tokens, g_tokens = _tokenize(p_text), _tokenize(g_text)
            if not p_tokens and not g_tokens:
                precision = recall = 1.0
            elif not p_tokens or not g_tokens:
                precision = recall = 0.0
            else:
                overlap = len(p_tokens & g_tokens)
                precision = overlap / len(p_tokens)
                recall = overlap / len(g_tokens)

            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            score.clauses[clause] = ClauseScore(
                clause=clause,
                precision=precision,
                recall=recall,
                f1=f1,
                matched=(p_text == g_text),
            )

        if self._executor is not None:
            score.execution_match = self._execution_match(predicted, gold)
        return score

    def evaluate_batch(
        self, pairs: List[tuple]
    ) -> AggregateScore:
        """Average metrics over ``[(predicted, gold), ...]`` pairs."""
        if not pairs:
            return AggregateScore()
        scores = [self.evaluate(p, g) for p, g in pairs]
        total = len(scores)
        exact = sum(1 for s in scores if s.exact_match)
        exec_results = [s.execution_match for s in scores
                        if s.execution_match is not None]
        exec_acc = (
            sum(1 for r in exec_results if r) / len(exec_results)
            if exec_results else None
        )

        per_clause: Dict[str, float] = {}
        all_clauses = {c for s in scores for c in s.clauses}
        for clause in all_clauses:
            values = [s.clauses[clause].f1 for s in scores
                      if clause in s.clauses]
            per_clause[clause] = sum(values) / len(values) if values else 0.0

        return AggregateScore(
            count=total,
            exact_match_rate=exact / total,
            execution_accuracy=exec_acc,
            mean_f1=sum(s.mean_f1 for s in scores) / total,
            per_clause=per_clause,
        )

    def _execution_match(self, predicted: str, gold: str) -> bool:
        try:
            return self._executor(predicted) == self._executor(gold)
        except Exception as exc:  # noqa: BLE001
            log.warning("execution comparison failed: %s", exc)
            return False


def _normalise(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    text = re.sub(r"\s+", " ", str(sql or "")).strip()
    return text.rstrip(";").strip().lower()


def _extract_clause(sql: str, pattern: str) -> str:
    match = re.search(pattern, sql, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _tokenize(clause: str) -> set:
    """Split a clause into comparable tokens, dropping noise words."""
    if not clause:
        return set()
    text = clause.lower()
    text = re.sub(r"[(),`'\"]", " ", text)
    tokens = {t for t in re.split(r"\s+", text) if t}
    return {t for t in tokens if t not in _STOPWORDS}
