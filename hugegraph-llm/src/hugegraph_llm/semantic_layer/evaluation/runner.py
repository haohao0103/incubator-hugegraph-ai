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

"""Run the retrieval evaluation end to end.

The baseline is **full-schema injection** -- dump every table and column
into the prompt, which is what a Text2SQL system without a semantic layer
does. That is the honest comparison for "does the semantic layer pay for
itself": not against another retrieval system, but against not retrieving
at all.

Both sides are measured with the same token estimator on the same graph, so
the saving figure is a property of the approach rather than of the metric.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hugegraph_llm.semantic_layer.context import estimate_tokens
from hugegraph_llm.semantic_layer.evaluation.dataset import EvalDataset, load_dataset
from hugegraph_llm.semantic_layer.evaluation.metrics import (
    CaseResult,
    MetricSummary,
    RetrievalEvaluator,
)
from hugegraph_llm.semantic_layer.readers import SemanticGraphReader
from hugegraph_llm.semantic_layer.retrieval import (
    RetrievalConfig,
    SemanticLayerRetriever,
)

__all__ = ["EvaluationReport", "evaluate", "measure_baseline"]

#: Cases are grouped so a regression is attributable: a drop confined to
#: "free" wording means the model's vocabulary coverage shrank, not that
#: retrieval broke.
_SOURCE_GROUPS = ("term", "schema", "free")


@dataclass
class EvaluationReport:
    """Everything a CI job or a human needs to judge one run."""

    dataset: str
    graph: str
    overall: MetricSummary
    by_source: Dict[str, MetricSummary] = field(default_factory=dict)
    multi_table: Optional[MetricSummary] = None
    results: List[CaseResult] = field(default_factory=list)
    baseline_tokens: int = 0
    elapsed_ms: float = 0.0
    #: Metrics this harness cannot measure, and why.
    not_measured: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "dataset": self.dataset,
            "graph": self.graph,
            "baseline_tokens": self.baseline_tokens,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "overall": self.overall.to_dict(),
            "by_source": {
                name: summary.to_dict()
                for name, summary in self.by_source.items()
            },
            "not_measured": self.not_measured,
        }
        if self.multi_table is not None:
            payload["multi_table"] = self.multi_table.to_dict()
        payload["cases"] = [r.to_dict() for r in self.results]
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def render(self) -> str:
        """Human-readable summary."""
        o = self.overall
        lines = [
            f"Semantic layer retrieval evaluation",
            f"  dataset : {self.dataset} ({o.count} cases)",
            f"  graph   : {self.graph}",
            f"  elapsed : {self.elapsed_ms:.0f}ms",
            "",
            f"  {'metric':28s} value",
            f"  {'-' * 28} -----",
            f"  {'precision@5':28s} {o.precision_at_5:.3f}",
            f"  {'recall@5':28s} {o.recall_at_5:.3f}",
            f"  {'recall (all retrieved)':28s} {o.recall_full:.3f}",
            f"  {'excess tables per gold':28s} {o.mean_excess_ratio:.2f}x",
            f"  {'all gold tables found':28s} {o.all_gold_found_rate:.3f}",
            f"  {'retrieved set joinable':28s} {o.retrieved_joinable_rate:.3f}",
            f"  {'gold set joinable':28s} {o.gold_joinable_rate:.3f}",
            f"  {'business-term recall fired':28s} {o.term_recall_rate:.3f}",
            f"  {'success (gold + joinable)':28s} {o.success_rate:.3f}",
            "",
            f"  {'mean tokens (retrieved)':28s} {o.mean_tokens:.0f}",
            f"  {'mean tokens (full schema)':28s} {o.mean_baseline_tokens:.0f}",
            f"  {'token saving':28s} {o.token_saving * 100:.1f}%",
        ]
        if self.multi_table is not None and self.multi_table.count:
            m = self.multi_table
            lines += [
                "",
                f"  multi-table subset ({m.count} cases)",
                f"  {'recall@5':28s} {m.recall_at_5:.3f}",
                f"  {'retrieved set joinable':28s} {m.retrieved_joinable_rate:.3f}",
                f"  {'success (gold + joinable)':28s} {m.success_rate:.3f}",
            ]
        if self.by_source:
            lines += ["", "  by question source"]
            for name in _SOURCE_GROUPS:
                if name not in self.by_source:
                    continue
                s = self.by_source[name]
                lines.append(
                    f"    {name:8s} n={s.count:<3d} recall@5={s.recall_at_5:.3f} "
                    f"success={s.success_rate:.3f} "
                    f"term_recall={s.term_recall_rate:.3f}"
                )
        if self.not_measured:
            lines += ["", "  not measured"]
            for name, why in self.not_measured.items():
                lines.append(f"    {name}: {why}")
        return "\n".join(lines)


def measure_baseline(
    reader: SemanticGraphReader, retriever: SemanticLayerRetriever
) -> int:
    """Cost of injecting the entire schema, in tokens.

    This is the "no semantic layer" number: every table, every column,
    full detail. Measured once per run so it is comparable across cases.
    """
    projection = reader.projection()
    parts = [
        retriever._build_context(projection, name, 0.0, [], True).render("full")
        for name in projection.tables
    ]
    return estimate_tokens("\n".join(parts))


def evaluate(
    reader: SemanticGraphReader,
    dataset_path: str,
    *,
    graph_name: str = "",
    config: Optional[RetrievalConfig] = None,
    max_tokens: int = 4000,
    vector_store: Any = None,
    embed: Any = None,
    executor: Any = None,
) -> EvaluationReport:
    """Run every case in a dataset and return the report.

    Each case is retrieved, scored, and compared against the full-schema
    baseline. Grouping by ``source`` happens afterwards so a regression can
    be attributed to vocabulary coverage rather than to retrieval itself.

    :param executor: optional warehouse executor (e.g.
        :class:`~semantic_layer.execution.SqliteExecutor`). When supplied,
        every case's gold SQL is executed against it -- a validity check on
        the evaluation set itself. Full execution accuracy (comparing
        generated SQL's results against gold) additionally needs an LLM.
    """
    cfg = config or RetrievalConfig()
    retriever = SemanticLayerRetriever(
        reader, vector_store=vector_store, embed=embed, config=cfg
    )
    dataset: EvalDataset = load_dataset(dataset_path)
    projection = reader.projection()

    baseline = measure_baseline(reader, retriever)
    evaluator = RetrievalEvaluator(projection, baseline_tokens=baseline)

    results: List[CaseResult] = []
    start = time.time()
    for case in dataset:
        result = retriever.retrieve(case.question, max_tokens=max_tokens)
        scored = evaluator.score(
            case,
            result.tables,
            tokens_used=result.budget.used_tokens,
            term_recall_fired="business_term" in (result.sources_used or []),
            sources_used=list(result.sources_used or []),
        )
        if executor is not None:
            scored.gold_executable = executor.execute(case.gold_sql).ok
        results.append(scored)
    elapsed_ms = (time.time() - start) * 1000

    by_source = {
        source: MetricSummary.from_results(
            [r for r in results if r.source == source]
        )
        for source in _SOURCE_GROUPS
        if any(r.source == source for r in results)
    }
    multi = [r for r in results if r.multi_table]

    return EvaluationReport(
        dataset=dataset.name,
        graph=graph_name or "(in-memory)",
        overall=MetricSummary.from_results(results),
        by_source=by_source,
        multi_table=MetricSummary.from_results(multi) if multi else None,
        results=results,
        baseline_tokens=baseline,
        elapsed_ms=elapsed_ms,
        not_measured=_not_measured(executor is not None),
    )


def _not_measured(has_executor: bool) -> Dict[str, str]:
    if has_executor:
        execution = (
            "warehouse executor supplied, but comparing generated SQL "
            "against gold additionally needs a working LLM endpoint"
        )
    else:
        execution = (
            "no warehouse executor supplied; pass one to enable gold-SQL "
            "executability checks"
        )
    return {
        "execution_accuracy": execution,
        "end_to_end_sql_correctness": (
            "depends on the generating model, not on retrieval"
        ),
    }


def write_report(report: EvaluationReport, path: str) -> None:
    """Persist a run so results can be diffed between commits."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.to_json(), encoding="utf-8")
