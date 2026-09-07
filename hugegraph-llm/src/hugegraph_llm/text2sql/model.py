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

"""Semantic-layer data model for Text2SQL.

These dataclasses describe the *database semantic layer* — the bridge between
natural-language questions and physical SQL — and deliberately mirror the
proven modelling idioms of:

- **WrenAI MDL** / **dbt Semantic Layer + MetricFlow**: tables are described by
  dimensions (groupable columns), measures (aggregatable columns), metrics
  (named aggregations with a defined 口径), and relationships (joins).
- **Vanna**: the ``QueryPattern`` records "DDL + documentation + verified SQL"
  as the retrieval source for few-shot SQL generation.

The model is the deterministic source of truth.  It is produced from DDL + a
data dictionary (or a metadata platform) and only *enriched* (never invented)
by an LLM.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Column:
    """A physical column, annotated with its semantic role."""

    name: str
    data_type: str = "TEXT"
    description: str = ""
    # Semantic role: dimension | measure | time | partition | pk | fk | value
    role: str = "dimension"
    # For calculated columns (e.g. ``amount * 0.9``); empty for physical columns.
    expression: str = ""


@dataclass
class Table:
    """A physical table/view with semantic annotations."""

    name: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    domain: str = ""
    columns: List[Column] = field(default_factory=list)
    # Time / partition columns: critical for SQL correctness (avoid full scans).
    time_column: str = ""
    partition_column: str = ""
    timezone: str = "Asia/Shanghai"


@dataclass
class Join:
    """A join path between two tables with its correctness-critical metadata."""

    from_table: str
    to_table: str
    on_condition: str  # e.g. "order.user_id = user.id"
    join_type: str = "LEFT JOIN"
    # "none" | "one_to_many" | "many_to_many" — drives DISTINCT / dedup warnings.
    fanout_risk: str = "none"


@dataclass
class Value:
    """A code -> meaning mapping for a column (e.g. status 2 = 已支付)."""

    column: str  # fully-qualified column ref, e.g. "order.status"
    code: str
    meaning: str


@dataclass
class Term:
    """A business term with its synonyms and the columns/metrics it maps to.

    ``aliases`` carry the synonym list (e.g. GMV / 成交额 / 支付金额).  The term
    is the *canonical* node; resolution normalizes a mention to ``name``.
    """

    name: str
    aliases: List[str] = field(default_factory=list)
    column_refs: List[str] = field(default_factory=list)  # "order_detail.amount"
    metric_refs: List[str] = field(default_factory=list)  # "gmv"


@dataclass
class Filter:
    """A single, auditable 口径 filter condition.

    Structured (not a SQL string) so each condition can be verified against the
    ``column`` / ``value`` vertices and reconstructed deterministically.
    """

    column: str  # fully-qualified column ref, e.g. "order.status"
    operator: str = "IN"  # IN | EQ | NE | GT | GTE | LT | LTE | BETWEEN | LIKE
    values: List[str] = field(default_factory=list)  # e.g. ["2", "3"]


@dataclass
class Metric:
    """A named metric with an explicit, reviewable 口径 (definition).

    Mirrors MetricFlow: a metric = ``agg_func(measure)`` over ``dimensions``
    with ``time_granularity`` and structured ``filters``.  The LLM must read
    this definition verbatim — never recompute it.
    """

    name: str
    description: str = ""
    measure: str = ""  # the column to aggregate, e.g. "order_detail.amount"
    agg_func: str = "SUM"  # SUM | COUNT | AVG | COUNT_DISTINCT | MAX | MIN
    dimensions: List[str] = field(default_factory=list)  # groupable columns
    time_granularity: str = ""  # day | month | quarter | year
    time_column: str = ""  # e.g. "order.pay_time"
    filters: List[Filter] = field(default_factory=list)  # auditable WHERE clauses
    dedup: bool = False  # dedup required when joining a fanout table


@dataclass
class QueryPattern:
    """A verified question -> SQL example (Vanna's "SQL few-shot" source)."""

    question: str
    sql: str
    tables: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)


@dataclass
class SemanticModel:
    """A complete semantic layer for one business domain."""

    name: str
    domain: str = ""
    tables: List[Table] = field(default_factory=list)
    joins: List[Join] = field(default_factory=list)
    terms: List[Term] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    query_patterns: List[QueryPattern] = field(default_factory=list)

    def table(self, name: str) -> Optional[Table]:
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def metric(self, name: str) -> Optional[Metric]:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
