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

"""Evaluation dataset: questions, gold SQL, and the tables they need.

**Gold tables are derived from the gold SQL, not annotated by hand.** Hand
labelling "which tables does this question need" is slow, and it drifts: a
label saying ``[orders, customers]`` while the SQL joins three tables is
worse than no label, because it quietly rewards under-retrieval. Extracting
them from the SQL keeps the two in sync by construction.

Each case carries ``source`` so a report can distinguish *vocabulary the
model actually knows* (``term`` / ``schema``) from *paraphrases* (``free``).
That distinction matters: it is the difference between "the semantic layer
works" and "the semantic layer works when you happen to use its exact
wording", and collapsing them is how a benchmark ends up overselling.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "EvalCase",
    "EvalDataset",
    "extract_tables",
    "load_dataset",
    "DatasetError",
]

#: ``FROM <t>`` and ``JOIN <t>`` -- the two places a table can appear.
#: Backtick quoting (StarRocks/MySQL dialect, e.g. ``JOIN `order` o``) is
#: unwrapped; without it, reserved-word table names would silently vanish
#: from gold-table extraction.
_TABLE_RE = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?", re.IGNORECASE
)

_SQL_WORDS = {
    "select", "where", "group", "by", "having", "limit", "on",
    "and", "or", "as", "inner", "left", "right", "outer", "full", "cross",
    "union", "all", "distinct", "case", "when", "then", "else", "end",
    "with", "values", "insert", "update", "delete", "set", "not", "null",
    "is", "in", "exists", "between", "like", "asc", "desc", "count", "sum",
    "avg", "min", "max", "coalesce", "cast", "date", "interval", "extract",
}
# NOTE: "order" is deliberately NOT a keyword here. The regex only captures
# identifiers directly after FROM/JOIN, where "ORDER BY" cannot occur, and
# `order` is an extremely common real table name (this sample domain has one).


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


@dataclass
class EvalCase:
    """One evaluation question."""

    question: str
    gold_sql: str
    #: Derived from ``gold_sql`` unless explicitly overridden.
    gold_tables: List[str] = field(default_factory=list)
    #: "term" (business vocabulary), "schema" (table/column names),
    #: or "free" (paraphrase, no shared vocabulary).
    source: str = "free"
    #: Extra credit: terms a good retriever should have matched.
    expected_terms: List[str] = field(default_factory=list)
    #: Questions spanning more than one join are tracked separately.
    multi_table: bool = False

    def __post_init__(self) -> None:
        if not self.gold_tables:
            self.gold_tables = extract_tables(self.gold_sql)
        if not self.multi_table:
            self.multi_table = len(self.gold_tables) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "gold_sql": self.gold_sql,
            "source": self.source,
            "expected_terms": list(self.expected_terms),
        }


@dataclass
class EvalDataset:
    """A named set of cases."""

    name: str
    cases: List[EvalCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def by_source(self, source: str) -> List[EvalCase]:
        return [c for c in self.cases if c.source == source]

    @property
    def multi_table_cases(self) -> List[EvalCase]:
        return [c for c in self.cases if c.multi_table]

    def summary(self) -> Dict[str, int]:
        return {
            "cases": len(self.cases),
            "multi_table": len(self.multi_table_cases),
            "by_source": {
                source: len(self.by_source(source))
                for source in ("term", "schema", "free")
                if self.by_source(source)
            },
        }


def extract_tables(sql: str) -> List[str]:
    """Extract table names referenced by a SQL string.

    Only ``FROM`` / ``JOIN`` positions are considered; identifiers in a
    subquery's SELECT list or a CTE name are not tables. Obvious SQL keywords
    that can follow those words are filtered out.
    """
    if not sql:
        return []
    found: List[str] = []
    seen: Set[str] = set()
    for match in _TABLE_RE.finditer(sql):
        name = match.group(1)
        if name.lower() in _SQL_WORDS:
            continue
        if name in seen:
            continue
        seen.add(name)
        found.append(name)
    return found


def load_dataset(path: str) -> EvalDataset:
    """Load a dataset from JSON or JSONL.

    JSON form: ``{"name": ..., "cases": [{...}, ...]}``
    JSONL form: one case object per line (name taken from the file stem).
    """
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"dataset not found: {path}")
    text = source.read_text(encoding="utf-8")

    if source.suffix.lower() == ".jsonl":
        name = source.stem
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            name, cases = source.stem, payload
        else:
            name = payload.get("name", source.stem)
            cases = payload.get("cases", [])

    return EvalDataset(name=name, cases=[_build_case(c) for c in cases])


def _build_case(raw: Dict[str, Any]) -> EvalCase:
    question = raw.get("question")
    gold_sql = raw.get("gold_sql")
    if not question or not gold_sql:
        raise DatasetError(f"case missing 'question' or 'gold_sql': {raw}")
    return EvalCase(
        question=question,
        gold_sql=gold_sql,
        gold_tables=raw.get("gold_tables") or extract_tables(gold_sql),
        source=raw.get("source", "free"),
        expected_terms=raw.get("expected_terms", []),
        multi_table=raw.get("multi_table", False),
    )
