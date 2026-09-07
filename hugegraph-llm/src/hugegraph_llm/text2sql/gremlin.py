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

"""Gremlin mappings for the Text2SQL semantic-layer graph.

The in-memory operators in :mod:`hugegraph_llm.text2sql.retrieval` map 1:1 to
these HugeGraph traversals.  Each function returns a Gremlin string so the
schema is provably queryable against a real HugeGraph instance.
"""


def _q(value: str) -> str:
    """Quote a Gremlin string literal, escaping embedded single quotes."""
    return "'" + value.replace("'", "\\'") + "'"


def term_to_columns(term_name: str) -> str:
    """Schema linking: resolve a business term to its mapped columns (full_name)."""
    return f"g.V().hasLabel('term').has('name', {_q(term_name)}).out('maps_to').values('full_name')"


def term_to_metrics(term_name: str) -> str:
    """Schema linking: resolve a business term to its mapped metrics."""
    return f"g.V().hasLabel('term').has('name', {_q(term_name)}).out('maps_to_metric').values('name')"


def metric_value_map(metric_name: str) -> str:
    """Metric resolution: read the 口径 verbatim from the metric vertex."""
    return (
        f"g.V().hasLabel('metric').has('name', {_q(metric_name)})"
        ".valueMap('agg_func','measure','time_granularity','time_column','dedup','dimensions')"
    )


def metric_filters(metric_name: str) -> str:
    """Audit: read a metric's structured filter conditions."""
    return (
        f"g.V().hasLabel('metric').has('name', {_q(metric_name)})"
        ".out('has_filter').valueMap('column','operator','values')"
    )


def join_path(from_table: str, to_table: str) -> str:
    """Join path finding: shortest path over ``joins`` edges (treated undirected)."""
    return (
        f"g.V().hasLabel('table').has('name', {_q(from_table)})"
        ".repeat(both('joins').simplePath())"
        f".until(hasLabel('table').has('name', {_q(to_table)}))"
        ".path().by('name')"
    )


def column_values(column_full_name: str) -> str:
    """Enum resolution: code -> meaning for a column's discrete values."""
    return (
        f"g.V().hasLabel('column').has('full_name', {_q(column_full_name)}).out('has_value').valueMap('code','meaning')"
    )


def few_shot_by_metric(metric_name: str) -> str:
    """Few-shot retrieval: verified SQL patterns that use a given metric."""
    return (
        "g.V().hasLabel('query_pattern')"
        f".where(out('uses_metric').has('name', {_q(metric_name)}))"
        ".valueMap('sql','description')"
    )
