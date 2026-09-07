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

"""Text2SQL semantic-layer graph for HugeGraph.

The package models the database semantic layer — tables, columns, joins,
business terms, metrics, enum values and verified SQL patterns — as a
schema-constrained knowledge graph, then provides deterministic graph-traversal
operators that ground LLM SQL generation.
"""

from hugegraph_llm.text2sql.model import (
    Column,
    Filter,
    Join,
    Metric,
    QueryPattern,
    SemanticModel,
    Table,
    Term,
    Value,
)
from hugegraph_llm.text2sql.pipeline import Text2SQLPipeline, Text2SQLResult
from hugegraph_llm.text2sql.retrieval import (
    JoinStep,
    MetricResolution,
    SemanticGraph,
    TermIndex,
    build_sql_prompt,
    find_join_path,
    render_sql_filter,
    resolve_metric,
    schema_link,
)
from hugegraph_llm.text2sql.schema import Text2SQLSchema
from hugegraph_llm.text2sql.seed import seed_graph

__all__ = [
    "Column",
    "Filter",
    "Join",
    "Metric",
    "QueryPattern",
    "SemanticModel",
    "Table",
    "Term",
    "Value",
    "Text2SQLSchema",
    "seed_graph",
    "SemanticGraph",
    "TermIndex",
    "schema_link",
    "find_join_path",
    "JoinStep",
    "resolve_metric",
    "MetricResolution",
    "render_sql_filter",
    "build_sql_prompt",
    "Text2SQLPipeline",
    "Text2SQLResult",
]
