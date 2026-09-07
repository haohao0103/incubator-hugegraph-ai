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

"""Tests for the semantic-layer-backed Text2SQL pipeline."""

from unittest.mock import MagicMock

import pytest

from hugegraph_llm.text2sql.orders import (
    ORDER_METRICS,
    ORDER_QUERIES,
    OrderDomainConnector,
    ensure_order_domain,
    orders_projection,
)
from hugegraph_llm.text2sql.pipeline import Text2SQLPipeline, Text2SQLResult
from tests.semantic_layer.fakes import FakeHugeGraph


@pytest.fixture
def pipeline():
    return Text2SQLPipeline.for_projection(orders_projection())


@pytest.fixture
def llm():
    stub = MagicMock()
    stub.generate.return_value = "SELECT SUM(od.amount) AS gmv"
    return stub


# -- retrieval via the semantic layer ---------------------------------------


def test_plan_resolves_terms_through_graph(pipeline):
    """GMV is a BusinessTerm vertex; recall runs through the M2 retriever.

    Tables include the M2 connected expansion (user/driver are FK-reachable
    from the seeds) -- semantic_layer's recall-first behaviour is the
    source of truth here; the budgeter degrades extra tables if needed.
    """
    result = pipeline.plan("上个月成交额是多少")
    assert "GMV" in result.resolved_terms
    assert {"order_detail", "order"} <= set(result.tables)
    assert set(result.tables) <= {"order", "order_detail", "user", "driver"}


def test_plan_metrics_are_term_driven(pipeline):
    result = pipeline.plan("上个月成交额是多少")
    assert result.linked_metrics == ["gmv"]
    assert result.metric_definitions == [
        f"gmv = {ORDER_METRICS[0]['expression']} -- {ORDER_METRICS[0]['description']}"
    ]


def test_plan_finds_join_path_over_references(pipeline):
    result = pipeline.plan("上个月成交额是多少")
    direct = next(
        p for p in result.join_paths if p.tables == ["order_detail", "order"]
    )
    assert direct.found and direct.all_proven
    # Declared FK renders as a real ON condition.
    assert direct.steps[0].to_sql() == "order_detail.order_id = order.id"


def test_plan_empty_for_unknown_question(pipeline):
    """A question sharing nothing with the corpus retrieves nothing.

    Note: the query must avoid CJK characters entirely -- CJK is tokenised
    per character (a documented BM25 coarseness), so a single coincidental
    character such as 完 (inside 已完成) can produce a weak match. Real
    CJK recall quality is the M4-tuning track's job, not this test's.
    """
    result = pipeline.plan("qqzzxx wuv")
    assert result.tables == []
    assert result.linked_metrics == []
    assert result.join_paths == []


# -- prompt assembly --------------------------------------------------------


def test_prompt_contains_schema_context(pipeline):
    prompt = pipeline.plan("上个月成交额是多少").prompt
    assert "[order_detail]" in prompt
    assert "[order]" in prompt
    # Value dictionary is folded into the column comment and renders verbatim.
    assert "1=待支付" in prompt


def test_prompt_contains_metric_caliber(pipeline):
    prompt = pipeline.plan("上个月成交额是多少").prompt
    assert "口径" in prompt
    assert "SUM(order_detail.amount) WHERE order.status IN (2, 3)" in prompt


def test_prompt_contains_few_shot_from_query_vertices(pipeline):
    """Few-shot comes from Query vertices -- the M6 feedback store."""
    prompt = pipeline.plan("上个月成交额是多少").prompt
    assert "上个月 GMV 是多少" in prompt
    assert ORDER_QUERIES[0]["sql"] in prompt


def test_prompt_unproven_join_renders_as_comment(pipeline):
    """An inferred FK must not look like declared integrity."""
    proj = orders_projection()
    # Replace the proven FK with an inferred one.
    proj.references["order_detail.order_id"] = ["order.id"]
    proj.reference_proven[("order_detail.order_id", "order.id")] = False
    prompt = Text2SQLPipeline.for_projection(proj).plan(
        "上个月成交额是多少"
    ).prompt
    assert "/* unproven join" in prompt


def test_prompt_always_carries_the_question(pipeline):
    prompt = pipeline.plan("qqzzxx wuv").prompt
    assert "qqzzxx wuv" in prompt
    assert prompt.rstrip().endswith("SQL:")


# -- generation -------------------------------------------------------------


def test_generate_without_llm_returns_prompt(pipeline):
    out = pipeline.generate("上个月成交额是多少")
    assert out == pipeline.plan("上个月成交额是多少").prompt


def test_generate_with_llm_returns_sql(pipeline, llm):
    pipeline.llm = llm
    out = pipeline.generate("上个月成交额是多少")
    assert out == "SELECT SUM(od.amount) AS gmv"


def test_answer_populates_sql_only_with_llm(pipeline, llm):
    without = pipeline.answer("上个月成交额是多少")
    assert without.sql is None and without.prompt

    pipeline.llm = llm
    with_llm = pipeline.answer("上个月成交额是多少")
    assert with_llm.sql == "SELECT SUM(od.amount) AS gmv"


def test_result_dataclass_contract(pipeline):
    """The API layer depends on these exact field names."""
    result = pipeline.plan("订单量")
    assert isinstance(result, Text2SQLResult)
    assert {"question", "resolved_terms", "linked_columns", "linked_metrics",
            "tables", "join_paths", "metric_definitions", "prompt",
            "sql"} <= set(result.__dataclass_fields__)


# -- seeding through the M1 connector contract -------------------------------


def _seeded_graph():
    graph = FakeHugeGraph()
    connector = OrderDomainConnector(graph)
    summary = connector.ingest()
    return graph, connector, summary


def test_connector_seeds_semantic_layer_schema():
    """Seed lands in M0 labels/conventions -- one schema stack, not two."""
    graph, _connector, summary = _seeded_graph()
    assert summary["loaded"] > 0
    assert graph.count("Table") == 4
    assert graph.count("Column") == 17
    assert graph.count("BusinessTerm") == 5
    assert graph.count("Metric") == 2
    assert graph.count("Query") == 2
    assert graph.edge_count("REFERENCES") == 3
    assert graph.edge_count("TERM_MAPS") == 5
    assert graph.edge_count("HAS_EXPRESSION") == 2
    assert graph.edge_count("USES_TABLE") == 4


def test_seeded_graph_reads_back_as_the_same_projection():
    """Projection fixture and graph seed must agree -- one source of truth.

    Read back through the real GremlinSemanticReader over the fake's
    Gremlin surface: the same code path a live server takes.
    """
    from hugegraph_llm.semantic_layer.readers import GremlinSemanticReader

    graph, _connector, _summary = _seeded_graph()
    seeded = GremlinSemanticReader(graph).projection()
    expected = orders_projection()

    assert set(seeded.tables) == set(expected.tables)
    assert set(seeded.columns) == set(expected.columns)
    assert set(seeded.terms) == set(expected.terms)
    assert set(seeded.metrics) == set(expected.metrics)
    assert seeded.references == expected.references
    assert seeded.term_columns == expected.term_columns
    assert seeded.term_metrics == expected.term_metrics
    assert seeded.metric_columns == expected.metric_columns
    assert {(p.question, p.sql) for p in seeded.query_patterns} == {
        (p.question, p.sql) for p in expected.query_patterns
    }
    for pattern in seeded.query_patterns:
        assert set(pattern.tables) <= set(expected.tables)


def test_ensure_order_domain_is_guards_and_seeds_once():
    graph, _connector, _summary = _seeded_graph()
    # Already seeded -> no second pass.
    assert ensure_order_domain(graph) is False
    # Empty graph -> seeds.
    assert ensure_order_domain(FakeHugeGraph()) is True
