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

"""Question -> semantic-layer context -> SQL prompt.

This is a thin adapter, not a second retrieval stack. Everything it does
delegates to the graph-native semantic layer:

* table/column recall, business-term resolution and budgeted context
  come from :class:`~semantic_layer.retrieval.SemanticLayerRetriever` (M2);
* join paths come from :func:`~semantic_layer.join_path.find_join_path`
  over ``REFERENCES`` edges, proven steps rendered as real ON conditions
  and inferred ones as comments (M3 semantics);
* metric definitions come from ``Metric`` vertices read into the
  projection -- the 口径 text is rendered verbatim, never recomputed;
* few-shot examples come from ``Query`` vertices -- the same store M6
  feedback writes, so verified usage accumulates into better prompts.

What was deleted with the old in-code stack: its own term index, schema
linker, join finder, DDL renderer, graph writer and seed loader. The
former ``QueryPattern``/``Value`` structures survive only as graph data
(Query vertices, column comments).

Pure and LLM-agnostic: ``generate`` returns the assembled prompt when no
LLM is injected, or the LLM's SQL when one is.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hugegraph_llm.semantic_layer.context import TableContext
from hugegraph_llm.semantic_layer.join_path import JoinPath, find_join_path
from hugegraph_llm.semantic_layer.readers import (
    InMemorySemanticReader,
    SemanticGraphReader,
)
from hugegraph_llm.semantic_layer.retrieval import (
    RetrievalConfig,
    SemanticLayerRetriever,
)

__all__ = ["Text2SQLResult", "Text2SQLPipeline"]


@dataclass
class Text2SQLResult:
    """The deterministic retrieval context plus the generated prompt/SQL."""

    question: str
    resolved_terms: List[str] = field(default_factory=list)
    linked_columns: List[str] = field(default_factory=list)
    linked_metrics: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    join_paths: List[JoinPath] = field(default_factory=list)
    metric_definitions: List[str] = field(default_factory=list)
    prompt: str = ""
    sql: Optional[str] = None


class Text2SQLPipeline:
    """Assemble a SQL-generation prompt from the graph-native semantic layer."""

    def __init__(
        self,
        retriever: SemanticLayerRetriever,
        llm=None,
        db_type: str = "StarRocks",
        max_tokens: int = 4000,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.db_type = db_type
        self.max_tokens = max_tokens

    @classmethod
    def for_projection(
        cls,
        projection,
        llm=None,
        db_type: str = "StarRocks",
        max_tokens: int = 4000,
    ) -> "Text2SQLPipeline":
        """Build over an in-memory projection (tests, offline demos)."""
        return cls(
            SemanticLayerRetriever(InMemorySemanticReader(projection)),
            llm=llm, db_type=db_type, max_tokens=max_tokens,
        )

    # -- planning -----------------------------------------------------------

    def plan(self, question: str) -> Text2SQLResult:
        projection = self.retriever.reader.projection()
        retrieval = self.retriever.retrieve(question, max_tokens=self.max_tokens)

        tables = retrieval.tables
        columns = self._columns_of(projection, tables)

        # Metrics are term-driven (a term like GMV names its metric), which
        # keeps the 口径 section focused instead of listing every metric
        # whose columns happen to sit in a retrieved table.
        metrics: List[str] = []
        for term in retrieval.matched_terms:
            for metric in projection.term_metrics.get(term, []):
                if metric in projection.metrics and metric not in metrics:
                    metrics.append(metric)

        join_paths: List[JoinPath] = []
        if len(tables) > 1:
            join_paths = [
                find_join_path(projection, tables[0], table)
                for table in tables[1:]
            ]
            join_paths = [p for p in join_paths if p.found]

        metric_definitions = [
            self._render_metric(projection.metrics[m]) for m in metrics
        ]
        few_shot = self._few_shot(projection, tables, metrics)

        prompt = self._build_prompt(
            contexts=retrieval.budget.contexts,
            question=question,
            metric_definitions=metric_definitions,
            join_paths=join_paths,
            few_shot=few_shot,
        )

        return Text2SQLResult(
            question=question,
            resolved_terms=retrieval.matched_terms,
            linked_columns=columns,
            linked_metrics=metrics,
            tables=tables,
            join_paths=join_paths,
            metric_definitions=metric_definitions,
            prompt=prompt,
        )

    def generate(self, question: str) -> str:
        """Return generated SQL when an LLM is injected, else the prompt."""
        result = self.plan(question)
        if self.llm is None:
            return result.prompt
        result.sql = self.llm.generate(prompt=result.prompt)
        return result.sql

    def answer(self, question: str) -> Text2SQLResult:
        """Run the pipeline; ``sql`` is set only when an LLM is configured."""
        result = self.plan(question)
        if self.llm is not None:
            result.sql = self.llm.generate(prompt=result.prompt)
        return result

    # -- pieces -------------------------------------------------------------

    @staticmethod
    def _columns_of(projection, tables: List[str]) -> List[str]:
        out: List[str] = []
        for table in tables:
            for col in projection.columns_of(table):
                out.append(col.qualified)
        return out

    @staticmethod
    def _render_metric(metric) -> str:
        """One line per metric: name, 口径 expression, description."""
        line = f"{metric.name} = {metric.expression}" if metric.expression \
            else metric.name
        if metric.description:
            line += f" -- {metric.description}"
        return line

    @staticmethod
    def _few_shot(projection, tables: List[str], metrics: List[str]) -> List[str]:
        """Verified (question -> SQL) pairs touching the retrieved tables.

        Reads the same ``Query`` vertices that M6 feedback writes, so every
        confirmed answer becomes a future few-shot example.
        """
        table_set = set(tables)
        examples = []
        for pattern in projection.query_patterns:
            if table_set & set(pattern.tables):
                examples.append(f"{pattern.question} -> {pattern.sql}")
        return examples

    def _build_prompt(
        self,
        contexts: List[TableContext],
        question: str,
        metric_definitions: List[str],
        join_paths: List[JoinPath],
        few_shot: List[str],
    ) -> str:
        sections: List[str] = [
            f"You are an expert {self.db_type} SQL engineer. "
            "Generate ONLY a valid, executable SQL query.",
            "",
            "# Database schema (retrieved context)",
        ]
        sections.extend(ctx.render("full") for ctx in contexts)

        if metric_definitions:
            sections += [
                "",
                "# Metric definitions (口径 — follow EXACTLY, never recompute)",
                *metric_definitions,
            ]

        if join_paths:
            join_lines = [
                f"- {path.tables[0]} JOIN {path.tables[-1]}: {path.to_sql()}"
                for path in join_paths
            ]
            sections += [
                "",
                "# Join paths (use these exact ON conditions)",
                *join_lines,
            ]

        if few_shot:
            sections += [
                "",
                "# Few-shot examples (question -> verified SQL)",
                *few_shot,
            ]

        sections += ["", "# Question", question, "", "SQL:"]
        return "\n".join(sections)
