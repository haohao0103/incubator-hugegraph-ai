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

"""Tests for the evaluation harness."""

import json

import pytest

from hugegraph_llm.semantic_layer.evaluation.dataset import (
    DatasetError,
    EvalCase,
    EvalDataset,
    extract_tables,
    load_dataset,
)
from hugegraph_llm.semantic_layer.evaluation.metrics import (
    MetricSummary,
    RetrievalEvaluator,
    precision_at_k,
    recall_at_k,
)
from hugegraph_llm.semantic_layer.evaluation.runner import (
    EvaluationReport,
    evaluate,
    measure_baseline,
)
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    InMemorySemanticReader,
    SemanticProjection,
    TableRow,
)
from hugegraph_llm.semantic_layer.retrieval import SemanticLayerRetriever

# -- table extraction ------------------------------------------------------


def test_extract_tables_simple():
    assert extract_tables("SELECT * FROM orders") == ["orders"]


def test_extract_tables_with_join():
    sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
    assert extract_tables(sql) == ["orders", "customers"]


def test_extract_tables_deduplicates():
    assert extract_tables("SELECT * FROM orders UNION SELECT * FROM orders") == [
        "orders"
    ]


def test_extract_tables_skips_subquery_alias():
    """``FROM (SELECT 1) x`` has no real table; the alias is not one either.

    Returning ``["x"]`` would create a gold table that matches nothing in the
    catalogue and silently drag recall down.
    """
    assert extract_tables("SELECT COUNT(*) FROM (SELECT 1) x") == []


def test_extract_tables_handles_cte():
    sql = "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent JOIN customers c ON 1=1"
    # "recent" is a CTE, not a table -- but it is indistinguishable from one
    # without a SQL parser, so it is returned and filtered later against the
    # projection (see test_gold_tables_outside_projection_are_dropped).
    assert "orders" in extract_tables(sql)


def test_extract_tables_ignores_sql_keywords():
    assert extract_tables("SELECT * FROM orders WHERE 1=1") == ["orders"]
    assert extract_tables("SELECT * FROM orders GROUP BY status") == ["orders"]


def test_extract_tables_empty():
    assert extract_tables("") == []


# -- dataset ---------------------------------------------------------------


def test_case_derives_gold_tables_from_sql():
    case = EvalCase(question="q", gold_sql="SELECT * FROM a JOIN b ON a.id = b.id")
    assert case.gold_tables == ["a", "b"]
    assert case.multi_table is True


def test_single_table_case_is_not_multi():
    case = EvalCase(question="q", gold_sql="SELECT * FROM a")
    assert case.multi_table is False


def test_explicit_gold_tables_win():
    case = EvalCase(question="q", gold_sql="SELECT * FROM a", gold_tables=["z"])
    assert case.gold_tables == ["z"]


def test_dataset_groups_by_source():
    ds = EvalDataset(name="d", cases=[
        EvalCase("a", "SELECT * FROM t", source="term"),
        EvalCase("b", "SELECT * FROM t", source="free"),
        EvalCase("c", "SELECT * FROM t JOIN u", source="free"),
    ])
    assert len(ds.by_source("free")) == 2
    assert len(ds.multi_table_cases) == 1
    assert ds.summary()["cases"] == 3


def test_load_dataset_json(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({
        "name": "unit",
        "cases": [{"question": "q", "gold_sql": "SELECT * FROM a", "source": "term"}],
    }), encoding="utf-8")
    ds = load_dataset(str(path))
    assert ds.name == "unit" and len(ds) == 1


def test_load_dataset_jsonl(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"question": "q1", "gold_sql": "SELECT * FROM a"}\n'
        '{"question": "q2", "gold_sql": "SELECT * FROM b"}\n',
        encoding="utf-8",
    )
    ds = load_dataset(str(path))
    assert ds.name == "ds"
    assert len(ds) == 2


def test_load_dataset_missing_file():
    with pytest.raises(DatasetError):
        load_dataset("/nonexistent/dataset.json")


def test_load_dataset_rejects_case_without_sql(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cases": [{"question": "q"}]}), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(str(path))


# -- metric primitives -----------------------------------------------------


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], 3) == pytest.approx(2 / 3)


def test_precision_at_k_uses_only_first_k():
    assert precision_at_k(["a", "x"], ["a"], 1) == 1.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "c"], 5) == 0.5


def test_recall_no_gold_is_zero():
    """Avoid a spurious 1.0 when a case has no gold tables."""
    assert recall_at_k(["a"], [], 5) == 0.0


def test_empty_retrieval_scores_zero():
    assert precision_at_k([], ["a"], 5) == 0.0


# -- projection for join tests ---------------------------------------------


def _proj():
    proj = SemanticProjection()
    for name in ("orders", "customers", "products", "island"):
        proj.tables[name] = TableRow(name=name)
    for col in (
        ColumnRow(name="order_id", table="orders", is_primary_key=True),
        ColumnRow(name="customer_id", table="orders", is_foreign_key=True),
        ColumnRow(name="product_id", table="orders", is_foreign_key=True),
        ColumnRow(name="customer_id", table="customers", is_primary_key=True),
        ColumnRow(name="product_id", table="products", is_primary_key=True),
        ColumnRow(name="code", table="island", is_primary_key=True),
    ):
        proj.columns[col.qualified] = col
    proj.references["orders.customer_id"] = ["customers.customer_id"]
    proj.references["orders.product_id"] = ["products.product_id"]
    proj.reference_proven[("orders.customer_id", "customers.customer_id")] = True
    proj.reference_proven[("orders.product_id", "products.product_id")] = True
    return proj


def _evaluator(baseline=1000):
    return RetrievalEvaluator(_proj(), baseline_tokens=baseline)


def test_gold_tables_outside_projection_are_dropped():
    case = EvalCase("q", "SELECT * FROM orders JOIN ghost ON 1=1")
    result = _evaluator().score(case, ["orders"])
    assert result.gold_tables == ["orders"]


def test_all_gold_found():
    case = EvalCase("q", "SELECT * FROM orders JOIN customers ON 1=1")
    result = _evaluator().score(case, ["orders", "customers"])
    assert result.all_gold_found is True


def test_missing_gold_table_fails():
    case = EvalCase("q", "SELECT * FROM orders JOIN customers ON 1=1")
    result = _evaluator().score(case, ["orders"])
    assert result.all_gold_found is False


def test_joinable_retrieved_set():
    case = EvalCase("q", "SELECT * FROM orders JOIN customers ON 1=1")
    result = _evaluator().score(case, ["orders", "customers"])
    assert result.retrieved_joinable is True
    assert result.unreachable_tables == []


def test_unreachable_member_breaks_joinability():
    """An unreachable table is what invites a hallucinated join."""
    case = EvalCase("q", "SELECT * FROM orders")
    result = _evaluator().score(case, ["orders", "island"])
    assert result.retrieved_joinable is False
    assert result.unreachable_tables == ["island"]


def test_single_table_is_joinable():
    result = _evaluator().score(EvalCase("q", "SELECT * FROM island"), ["island"])
    assert result.retrieved_joinable is True


def test_token_saving():
    result = _evaluator(baseline=1000).score(
        EvalCase("q", "SELECT * FROM orders"), ["orders"], tokens_used=250
    )
    assert result.token_saving == pytest.approx(0.75)


def test_token_saving_without_baseline_is_zero():
    result = _evaluator(baseline=0).score(
        EvalCase("q", "SELECT * FROM orders"), ["orders"], tokens_used=250
    )
    assert result.token_saving == 0.0


def test_success_requires_both():
    case = EvalCase("q", "SELECT * FROM orders JOIN customers ON 1=1")
    # all gold present, but island makes the set unjoinable
    result = _evaluator().score(case, ["orders", "customers", "island"])
    assert result.all_gold_found is True
    assert result.retrieved_joinable is False
    assert result.is_success is False


def test_summary_averages():
    case = EvalCase("q", "SELECT * FROM orders")
    ev = _evaluator(baseline=1000)
    results = [
        ev.score(case, ["orders"], tokens_used=100),
        ev.score(case, ["orders", "customers"], tokens_used=300),
    ]
    summary = MetricSummary.from_results(results)
    assert summary.count == 2
    assert summary.mean_tokens == 200
    assert summary.token_saving == pytest.approx(0.8)


def test_summary_of_nothing_is_safe():
    assert MetricSummary.from_results([]).count == 0


# -- runner ----------------------------------------------------------------


def _dataset_file(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"question": "orders", "gold_sql": "SELECT * FROM orders", "source": "schema"}\n'
        '{"question": "orders with customers", '
        '"gold_sql": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id", '
        '"source": "schema"}\n',
        encoding="utf-8",
    )
    return str(path)


def test_evaluate_runs_all_cases(tmp_path):
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    assert report.overall.count == 2
    assert len(report.results) == 2


def test_evaluate_reports_baseline(tmp_path):
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    assert report.baseline_tokens > 0
    assert report.overall.mean_baseline_tokens == report.baseline_tokens


def test_evaluate_marks_unmeasurable_metrics(tmp_path):
    """Execution accuracy cannot be measured without a warehouse."""
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    assert "execution_accuracy" in report.not_measured


def test_report_render_includes_not_measured(tmp_path):
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    assert "not measured" in report.render()


def test_report_is_json_serialisable(tmp_path):
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    payload = json.loads(report.to_json())
    assert payload["overall"]["count"] == 2


def test_measure_baseline_is_positive():
    reader = InMemorySemanticReader(_proj())
    baseline = measure_baseline(reader, SemanticLayerRetriever(reader))
    assert baseline > 0


def test_baseline_exceeds_retrieval(tmp_path):
    """The whole point: retrieving costs less than dumping everything."""
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    assert report.overall.mean_tokens < report.overall.mean_baseline_tokens
    assert report.overall.token_saving > 0


def test_grouping_by_source(tmp_path):
    report = evaluate(InMemorySemanticReader(_proj()), _dataset_file(tmp_path))
    assert "schema" in report.by_source
    assert report.by_source["schema"].count == 2
