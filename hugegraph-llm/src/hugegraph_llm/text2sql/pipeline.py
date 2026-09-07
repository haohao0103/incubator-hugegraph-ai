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

"""End-to-end orchestration: question -> semantic graph context -> SQL prompt.

This is the "executable" entry point that chains the five retrieval operators.
It is pure and LLM-agnostic: ``generate`` returns the assembled prompt when no
LLM is injected, or the LLM's SQL when one is.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hugegraph_llm.text2sql.model import SemanticModel
from hugegraph_llm.text2sql.retrieval import (
    JoinStep,
    MetricResolution,
    SemanticGraph,
    TermIndex,
    build_sql_prompt,
    find_join_path,
    resolve_metric,
    schema_link,
)
from hugegraph_llm.text2sql.seed import seed_graph


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _table_of_column(column_id: str) -> str:
    return column_id.split(".", 1)[0]


def _append_table(tables: List[str], table: str) -> None:
    if table and table not in tables:
        tables.append(table)


@dataclass
class Text2SQLResult:
    """The deterministic retrieval context plus the generated prompt/SQL."""

    question: str
    resolved_terms: List[str] = field(default_factory=list)
    linked_columns: List[str] = field(default_factory=list)
    linked_metrics: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    join_paths: List[JoinStep] = field(default_factory=list)
    metric_definitions: List[MetricResolution] = field(default_factory=list)
    prompt: str = ""
    sql: Optional[str] = None


class Text2SQLPipeline:
    """Chain term resolution -> schema linking -> join path -> metric -> prompt."""

    def __init__(self, model: SemanticModel, llm=None, db_type: str = "StarRocks"):
        self.model = model
        self.graph = SemanticGraph(seed_graph(model))
        self.term_index = TermIndex(model.terms)
        self.llm = llm
        self.db_type = db_type

    def plan(self, question: str) -> Text2SQLResult:
        terms = self.term_index.find_in_text(question)

        columns: List[str] = []
        metrics: List[str] = []
        for term in terms:
            link = schema_link(self.graph, term)
            columns.extend(link["columns"])
            metrics.extend(link["metrics"])
        columns = _dedupe(columns)
        metrics = _dedupe(metrics)

        tables: List[str] = []
        for column in columns:
            _append_table(tables, _table_of_column(column))
        for metric_name in metrics:
            resolved = resolve_metric(self.graph, metric_name)
            if resolved:
                col_refs = [resolved.measure, resolved.time_column] + list(resolved.dimensions)
                col_refs += [filter_.column for filter_ in resolved.filters]
                for col_ref in col_refs:
                    if col_ref:
                        _append_table(tables, _table_of_column(col_ref))

        join_paths: List[JoinStep] = []
        if len(tables) > 1:
            for table in tables[1:]:
                join_paths.extend(find_join_path(self.graph, tables[0], table))

        metric_definitions = [r for r in (resolve_metric(self.graph, m) for m in metrics) if r is not None]

        few_shot = self._few_shot(tables, metrics)
        prompt = build_sql_prompt(
            self.graph,
            question,
            tables,
            metrics,
            join_paths,
            few_shot,
            self.db_type,
        )

        return Text2SQLResult(
            question=question,
            resolved_terms=terms,
            linked_columns=columns,
            linked_metrics=metrics,
            tables=tables,
            join_paths=join_paths,
            metric_definitions=metric_definitions,
            prompt=prompt,
        )

    def generate(self, question: str) -> str:
        """Return generated SQL when an LLM is injected, else the assembled prompt."""
        result = self.plan(question)
        if self.llm is None:
            return result.prompt
        result.sql = self.llm.generate(prompt=result.prompt)
        return result.sql

    def answer(self, question: str) -> Text2SQLResult:
        """Run the full pipeline and return the result with ``sql`` populated.

        When an LLM is injected, ``sql`` holds the generated SQL; otherwise it is
        left ``None`` and ``prompt`` carries the assembled context.
        """
        result = self.plan(question)
        if self.llm is not None:
            result.sql = self.llm.generate(prompt=result.prompt)
        return result

    def _few_shot(self, tables: List[str], metrics: List[str]) -> List[str]:
        """Retrieve verified SQL patterns that reference the involved tables/metrics."""
        examples: List[str] = []
        table_set = set(tables)
        metric_set = set(metrics)
        for vertex in self.graph.vertices.values():
            if vertex["label"] != "query_pattern":
                continue
            used_tables = set()
            for edge in self.graph.out_edges(vertex["id"], "uses_table"):
                table = self.graph.vertex(edge["inV"])
                if table:
                    used_tables.add(table.get("properties", {}).get("name", ""))
            used_metrics = set()
            for edge in self.graph.out_edges(vertex["id"], "uses_metric"):
                metric = self.graph.vertex(edge["inV"])
                if metric:
                    used_metrics.add(metric.get("properties", {}).get("name", ""))
            if table_set & used_tables or metric_set & used_metrics:
                props = vertex.get("properties", {})
                examples.append(f"{props.get('description', '')} -> {props.get('sql', '')}")
        return examples
