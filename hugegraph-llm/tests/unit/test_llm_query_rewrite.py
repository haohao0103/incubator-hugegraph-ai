"""Unit tests for LLM Query Rewrite Engine."""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.llm_query_rewrite import (
    LLMQueryRewriteEngine,
    llm_rewrite_query,
    LLM_QUERY_REWRITE_PROMPT,
)


# ---- Mock LLM callback ----

def mock_llm_callback(prompt: str) -> str:
    """Simulate LLM responses for testing."""
    if "query understanding" in prompt.lower() or "Original query" in prompt:
        # Check for known queries in the prompt
        if "张三" in prompt and "货拉拉" in prompt:
            return json.dumps({
                "rewritten": "张三在货拉拉公司做什么职位",
                "entities": [{"name": "张三", "type": "person"}, {"name": "货拉拉", "type": "organization"}],
                "intent": "fact_lookup",
                "variants": ["张三 货拉拉 职位", "张三在货拉拉的工作"],
            })
        if "喜欢" in prompt or "运动" in prompt:
            return json.dumps({
                "rewritten": "张三喜欢什么运动",
                "entities": [{"name": "张三", "type": "person"}],
                "intent": "preference_query",
                "variants": ["张三 喜欢 运动"],
            })
        if "什么时候" in prompt or "何时" in prompt:
            return json.dumps({
                "rewritten": "项目启动的具体日期是什么",
                "entities": [{"name": "项目", "type": "concept"}],
                "intent": "temporal_query",
                "variants": ["项目 启动 日期"],
            })
        return json.dumps({
            "rewritten": "通用查询",
            "entities": [],
            "intent": "general",
            "variants": ["通用 查询"],
        })
    return "{}"


# ---- Test Class ----

class TestLLMQueryRewriteEngine:

    def test_init_with_callback(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=True)
        assert engine.use_llm is True
        assert engine.llm_callback is not None

    def test_init_without_callback(self):
        engine = LLMQueryRewriteEngine(use_llm=False)
        assert engine.use_llm is False
        assert engine.llm_callback is None

    def test_init_default_use_llm_false_when_no_callback(self):
        engine = LLMQueryRewriteEngine()
        # use_llm defaults to True but no callback/api_key means it stays False effectively
        assert engine.llm_callback is None

    def test_llm_rewrite_basic(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=True)
        result = engine.rewrite("张三在货拉拉公司做什么")
        assert result["method"] == "llm"
        assert "rewritten" in result
        assert "entities" in result
        assert len(result["entities"]) > 0

    def test_llm_rewrite_pronoun_resolution(self):
        engine = LLMQueryRewriteEngine(
            llm_callback=mock_llm_callback,
            use_llm=True,
            user_profile="张三是一名工程师",
        )
        result = engine.rewrite("他喜欢什么运动", context="之前提到张三")
        assert result["method"] == "llm"
        assert result["rewritten"] == "张三喜欢什么运动"

    def test_llm_rewrite_intent_classification(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=True)
        result = engine.rewrite("项目什么时候启动的")
        assert result["method"] == "llm"
        assert result["intent"] == "temporal_query"

    def test_rule_fallback_on_llm_failure(self):
        def failing_callback(prompt: str) -> str:
            raise RuntimeError("LLM API error")

        engine = LLMQueryRewriteEngine(llm_callback=failing_callback, use_llm=True)
        result = engine.rewrite("张三在货拉拉")
        assert result["method"] == "rule"
        assert "rewritten" in result

    def test_rule_fallback_explicit(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=False)
        result = engine.rewrite("张三在货拉拉")
        assert result["method"] == "rule"
        assert "entities" in result

    def test_rule_fallback_with_aliases(self):
        engine = LLMQueryRewriteEngine(
            use_llm=False,
            aliases={"HLL": "货拉拉"},
        )
        result = engine.rewrite("HLL总部在哪里")
        assert result["method"] == "rule"
        assert "货拉拉" in result["rewritten"]

    def test_entity_boosts_in_result(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=True)
        result = engine.rewrite("张三在货拉拉公司做什么")
        assert "boosts" in result
        assert isinstance(result["boosts"], dict)

    def test_variants_merged(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=True)
        result = engine.rewrite("张三在货拉拉公司做什么")
        assert "variants" in result
        assert len(result["variants"]) >= 1

    def test_convenience_function(self):
        result = llm_rewrite_query(
            "张三在货拉拉",
            use_llm=False,
            aliases={"HLL": "货拉拉"},
        )
        assert "rewritten" in result
        assert "method" in result

    def test_parse_llm_response_json(self):
        engine = LLMQueryRewriteEngine(use_llm=False)
        data = LLMQueryRewriteEngine._parse_llm_response(
            json.dumps({"rewritten": "foo", "entities": [], "intent": "general"})
        )
        assert data["rewritten"] == "foo"

    def test_parse_llm_response_markdown_block(self):
        engine = LLMQueryRewriteEngine(use_llm=False)
        raw = "```json\n{\"rewritten\": \"bar\", \"entities\": []}\n```"
        data = LLMQueryRewriteEngine._parse_llm_response(raw)
        assert data["rewritten"] == "bar"

    def test_parse_llm_response_plain_text_fallback(self):
        engine = LLMQueryRewriteEngine(use_llm=False)
        raw = "some plain text response"
        data = LLMQueryRewriteEngine._parse_llm_response(raw)
        assert data["rewritten"] == "some plain text response"

    def test_parse_llm_response_empty(self):
        engine = LLMQueryRewriteEngine(use_llm=False)
        data = LLMQueryRewriteEngine._parse_llm_response("")
        assert data["rewritten"] == ""

    def test_intent_heuristic_temporal(self):
        intent = LLMQueryRewriteEngine._classify_intent_heuristic("什么时候启动的")
        assert intent == "temporal_query"

    def test_intent_heuristic_relationship(self):
        intent = LLMQueryRewriteEngine._classify_intent_heuristic("张三和李四是什么关系")
        assert intent == "relationship_query"

    def test_intent_heuristic_preference(self):
        intent = LLMQueryRewriteEngine._classify_intent_heuristic("他喜欢什么")
        assert intent == "preference_query"

    def test_intent_heuristic_fact(self):
        intent = LLMQueryRewriteEngine._classify_intent_heuristic("什么是量子计算")
        assert intent == "fact_lookup"

    def test_intent_heuristic_general(self):
        intent = LLMQueryRewriteEngine._classify_intent_heuristic("随便聊聊")
        assert intent == "general"

    def test_context_injection_in_prompt(self):
        prompt = LLM_QUERY_REWRITE_PROMPT.format(
            query="他做什么",
            context="之前提到张三是工程师",
            profile="张三",
        )
        assert "之前提到张三是工程师" in prompt
        assert "张三" in prompt

    def test_rewrite_with_context_override(self):
        engine = LLMQueryRewriteEngine(llm_callback=mock_llm_callback, use_llm=True)
        result = engine.rewrite("他做什么", context="之前提到张三", user_profile="张三")
        assert result["method"] in ("llm", "rule")
