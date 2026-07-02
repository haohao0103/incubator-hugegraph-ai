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

"""Tests for QueryRewrite operator."""

import pytest

from hugegraph_llm.operators.llm_op.query_rewrite import (
    QueryRewrite,
    QueryRewriteConfig,
    QueryRewriteResult,
)


class FakeLLM:
    """Fake LLM that returns deterministic JSON responses."""

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response.format(query=prompt) if "{query}" in self.response else self.response


class BadLLM:
    """LLM that always raises an exception."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("llm failure")


class NoGenerateLLM:
    """LLM with no supported interface."""

    def ask(self, prompt: str) -> str:
        return prompt


# ---------------------------------------------------------------------------
# QueryRewriteResult
# ---------------------------------------------------------------------------


def test_result_defaults():
    result = QueryRewriteResult()
    assert result.original_query == ""
    assert result.needs_rewrite is False
    assert result.sub_queries == []
    assert result.executable_queries == []


def test_result_executable_queries_original():
    result = QueryRewriteResult(original_query="What is X?")
    assert result.executable_queries == ["What is X?"]


def test_result_executable_queries_sub_queries():
    result = QueryRewriteResult(
        original_query="complex query",
        needs_rewrite=True,
        sub_queries=["q1", "q2"],
    )
    assert result.executable_queries == ["q1", "q2"]


def test_result_to_dict():
    result = QueryRewriteResult(
        original_query="test",
        needs_rewrite=True,
        sub_queries=["a", "b"],
        reasoning="r",
        extraction_method="llm",
    )
    d = result.to_dict()
    assert d["original_query"] == "test"
    assert d["sub_queries"] == ["a", "b"]


# ---------------------------------------------------------------------------
# LLM-based rewrite
# ---------------------------------------------------------------------------


def test_llm_rewrite_complex():
    response = (
        '{"needs_rewrite": true, "sub_queries": ["Who is X?", "Who is Y?"], "reasoning": "test"}'
    )
    rewriter = QueryRewrite(llm=FakeLLM(response))
    result = rewriter.extract("What is the relationship between X and Y?")
    assert result.needs_rewrite is True
    assert result.sub_queries == ["Who is X?", "Who is Y?"]
    assert result.extraction_method == "llm"


def test_llm_rewrite_simple():
    response = (
        '{"needs_rewrite": false, "sub_queries": [], "reasoning": "simple"}'
    )
    rewriter = QueryRewrite(llm=FakeLLM(response))
    result = rewriter.extract("What is the capital of France?")
    assert result.needs_rewrite is False
    assert result.sub_queries == []


def test_llm_rewrite_with_code_fence():
    response = "```json\n{\"needs_rewrite\": true, \"sub_queries\": [\"q1\"], \"reasoning\": \"r\"}\n```"
    rewriter = QueryRewrite(llm=FakeLLM(response))
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.needs_rewrite is True
    assert result.sub_queries == ["q1"]


def test_llm_rewrite_dedup_and_limit():
    response = (
        '{"needs_rewrite": true, "sub_queries": ["q1", "q1", "q2", "q3", "q4", "q5"], "reasoning": "r"}'
    )
    config = QueryRewriteConfig(max_sub_queries=3)
    rewriter = QueryRewrite(llm=FakeLLM(response), config=config)
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.sub_queries == ["q1", "q2", "q3"]


def test_llm_rewrite_empty_sub_queries_means_no_rewrite():
    response = (
        '{"needs_rewrite": true, "sub_queries": [], "reasoning": "r"}'
    )
    rewriter = QueryRewrite(llm=FakeLLM(response))
    result = rewriter.extract("question")
    assert result.needs_rewrite is False


def test_llm_rewrite_malformed_json_falls_back():
    rewriter = QueryRewrite(llm=BadLLM(), config=QueryRewriteConfig(fallback_to_heuristic=True))
    result = rewriter.extract("What is the relationship between X and Y?")
    assert result.extraction_method == "heuristic"


def test_llm_rewrite_no_fallback():
    config = QueryRewriteConfig(fallback_to_heuristic=False, llm_max_retries=1)
    rewriter = QueryRewrite(llm=BadLLM(), config=config)
    result = rewriter.extract("What is the relationship between X and Y?")
    assert result.needs_rewrite is False


def test_unsupported_llm_interface_no_fallback():
    config = QueryRewriteConfig(fallback_to_heuristic=False, llm_max_retries=1)
    rewriter = QueryRewrite(llm=NoGenerateLLM(), config=config)
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.needs_rewrite is False
    assert "Unsupported LLM interface" in result.reasoning
    assert result.extraction_method == "heuristic"


def test_llm_rewrite_chat_interface():
    class ChatLLM:
        def chat(self, prompt: str) -> str:
            return '{"needs_rewrite": true, "sub_queries": ["q1"], "reasoning": "r"}'

    rewriter = QueryRewrite(llm=ChatLLM())
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.sub_queries == ["q1"]


def test_llm_rewrite_completion_interface():
    class CompletionLLM:
        def completion(self, prompt: str) -> str:
            return '{"needs_rewrite": true, "sub_queries": ["q1"], "reasoning": "r"}'

    rewriter = QueryRewrite(llm=CompletionLLM())
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.sub_queries == ["q1"]


def test_llm_rewrite_invalid_json_falls_back():
    class BadJsonLLM:
        def generate(self, prompt: str) -> str:
            return "not json at all"

    rewriter = QueryRewrite(llm=BadJsonLLM(), config=QueryRewriteConfig(fallback_to_heuristic=True, llm_max_retries=1))
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.extraction_method == "heuristic"


def test_llm_rewrite_sub_queries_not_list():
    response = '{"needs_rewrite": true, "sub_queries": "not a list", "reasoning": "r"}'
    rewriter = QueryRewrite(llm=FakeLLM(response), config=QueryRewriteConfig(fallback_to_heuristic=True, llm_max_retries=1))
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.extraction_method == "heuristic"


def test_llm_rewrite_mixed_sub_query_types():
    response = '{"needs_rewrite": true, "sub_queries": ["q1", 123, "q2"], "reasoning": "r"}'
    rewriter = QueryRewrite(llm=FakeLLM(response))
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.sub_queries == ["q1", "q2"]


def test_extract_whitespace_only_query():
    rewriter = QueryRewrite(llm=None)
    result = rewriter.extract("   ")
    assert result == QueryRewriteResult()



# ---------------------------------------------------------------------------
# Heuristic rewrite
# ---------------------------------------------------------------------------


def test_heuristic_simple_query():
    rewriter = QueryRewrite(llm=None)
    result = rewriter.extract("What is the capital of France?")
    assert result.needs_rewrite is False
    assert result.extraction_method == "heuristic"


def test_heuristic_split_by_and():
    rewriter = QueryRewrite(llm=None)
    result = rewriter.extract("What is the relationship between X and Y?")
    assert result.needs_rewrite is True
    # Each clause is capitalized and punctuation stripped
    assert any("x" in sq.lower() for sq in result.sub_queries)
    assert any("y" in sq.lower() for sq in result.sub_queries)


def test_heuristic_no_marker():
    config = QueryRewriteConfig(simple_query_threshold=10)
    rewriter = QueryRewrite(llm=None, config=config)
    result = rewriter.extract("France?")
    assert result.needs_rewrite is False


def test_heuristic_fallback_disabled():
    config = QueryRewriteConfig(fallback_to_heuristic=False)
    rewriter = QueryRewrite(llm=None, config=config)
    result = rewriter.extract("long complex query with multiple entities and relationships")
    assert result.needs_rewrite is False
    assert result.reasoning == "No LLM available and heuristic fallback disabled."


# ---------------------------------------------------------------------------
# Operator protocol
# ---------------------------------------------------------------------------


def test_run_operator_protocol():
    response = '{"needs_rewrite": true, "sub_queries": ["q1"], "reasoning": "r"}'
    rewriter = QueryRewrite(llm=FakeLLM(response))
    ctx = {"query": "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"}
    result = rewriter.run(ctx)
    assert "query_rewrite" in result
    assert result["query_rewrite"].needs_rewrite is True


def test_run_empty_query():
    rewriter = QueryRewrite(llm=None)
    ctx = {"query": ""}
    result = rewriter.run(ctx)
    assert result["query_rewrite"].needs_rewrite is False


def test_run_missing_query():
    rewriter = QueryRewrite(llm=None)
    ctx = {}
    result = rewriter.run(ctx)
    assert result["query_rewrite"].needs_rewrite is False


# ---------------------------------------------------------------------------
# Custom template
# ---------------------------------------------------------------------------


def test_custom_template():
    template = "Rewrite: {query}"
    llm = FakeLLM('{"needs_rewrite": true, "sub_queries": ["custom"], "reasoning": "r"}')
    rewriter = QueryRewrite(llm=llm, rewrite_template=template)
    result = rewriter.extract(
        "What is the relationship between the CEO of Apple and the company that designed the A17 chip?"
    )
    assert result.sub_queries == ["custom"]
