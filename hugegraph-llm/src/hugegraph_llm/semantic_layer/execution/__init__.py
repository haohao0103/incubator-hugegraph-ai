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

"""Warehouse execution for NL2SQL (the leg the semantic layer never had).

The semantic layer closes NL2SQL's hard half -- recall, join constraints,
calibers. This module closes the execution leg with the smallest honest
implementation: a **SQLite warehouse generated from the semantic layer's
own projection**, so schema and metadata can never drift apart.

Why generated, and why SQLite:

* No warehouse exists in this environment, but "generate SQL" is only half
  of NL2SQL -- without somewhere to run it, neither correctness nor even
  executability can be measured. M4's ``not_measured`` explicitly names
  this.
* Building the warehouse *from the projection* (types from ``ColumnRow``,
  foreign keys from ``REFERENCES``, row factories from column names) means
  the gold SQL in an evaluation set -- written against the same model --
  is testable against the warehouse by construction. That turns
  ``execution_accuracy`` from unmeasurable into measurable the moment an
  LLM produces candidate SQL.

**Scope, stated plainly**: this is a *sample warehouse* for closing the
loop and validating evaluation. It is not a production executor -- no
concurrency, no credentials, real deployments point
:class:`SqliteExecutor`-shaped adapters at StarRocks/MySQL/BigQuery
instead. The BigQuery DDL that ships with the ACME model
(``neocarta/datasets/acme-dataset.sql``) is a future source of real seed
data; it needs a dialect converter (OPTIONS() stripping, STRUCT/ARRAY to
JSON, DATE/TIMESTAMP literals) that is deliberately not written yet.
"""

from hugegraph_llm.semantic_layer.execution.executor import (
    ExecutionResult,
    SqliteExecutor,
)
from hugegraph_llm.semantic_layer.execution.synthetic import (
    SyntheticWarehouse,
    build_sqlite_warehouse,
)

__all__ = [
    "ExecutionResult",
    "SqliteExecutor",
    "SyntheticWarehouse",
    "build_sqlite_warehouse",
]
