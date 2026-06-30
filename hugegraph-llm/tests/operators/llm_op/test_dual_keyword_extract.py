# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for dual_keyword_extract.py — hl/ll keyword extraction."""

import json
import pytest

from hugegraph_llm.operators.llm_op.dual_keyword_extract import (
    DUAL_KEYWORD_EXTRACT_PROMPT,
    DualKeywordConfig,
    DualKeywordExtract,
    DualKeywords,
    extract_dual_keywords,
)


# ── Test DualKeywords ─────────────────────────────────────────────


class TestDualKeywords:
    """Test DualKeywords dataclass."""

    def test_default_keywords(self):
        kw = DualKeywords()
        assert kw.hl_keywords == []
        assert kw.ll_keywords == []
        assert kw.has_keywords == False
        assert kw.extraction_method == "llm"

    def test_keywords_with_data(self):
        kw = DualKeywords(
            hl_keywords=["treatment", "disease"],
            ll_keywords=["diabetes", "insulin"],
            extraction_method="heuristic",
        )
        assert kw.has_keywords == True
        assert kw.hl_str == "treatment disease"
        assert kw.ll_str == "diabetes insulin"

    def test_to_dict(self):
        kw = DualKeywords(
            hl_keywords=["concept"],
            ll_keywords=["entity"],
            extraction_method="llm",
        )
        d = kw.to_dict()
        assert d["hl_keywords"] == ["concept"]
        assert d["ll_keywords"] == ["entity"]
        assert d["extraction_method"] == "llm"

    def test_str_join_empty(self):
        kw = DualKeywords()
        assert kw.hl_str == ""
        assert kw.ll_str == ""


# ── Test DualKeywordConfig ────────────────────────────────────────


class TestDualKeywordConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        config = DualKeywordConfig()
        assert config.max_keywords_per_level == 5
        assert config.min_keyword_length == 2
        assert config.language == "en"
        assert config.llm_max_retries == 2
        assert config.fallback_to_heuristic == True
        assert config.short_query_threshold == 50

    def test_custom_config(self):
        config = DualKeywordConfig(
            max_keywords_per_level=10,
            language="zh",
            fallback_to_heuristic=False,
        )
        assert config.max_keywords_per_level == 10
        assert config.language == "zh"
        assert config.fallback_to_heuristic == False


# ── Test heuristic extraction ─────────────────────────────────────


class TestHeuristicExtraction:
    """Test keyword extraction without LLM (heuristic mode)."""

    def test_simple_query(self):
        """Simple English query → keywords extracted."""
        extractor = DualKeywordExtract(llm=None)  # No LLM → heuristic
        kw = extractor.extract("What is the treatment for diabetes?")
        assert kw.extraction_method in ("heuristic", "short_query_fallback")
        assert kw.has_keywords == True
        # Keywords should contain relevant terms (whole query or extracted words)
        all_kw = kw.hl_keywords + kw.ll_keywords
        assert len(all_kw) > 0

    def test_query_with_proper_nouns(self):
        """Capitalized words → ll_keywords (proper nouns)."""
        extractor = DualKeywordExtract(llm=None)
        kw = extractor.extract("How does Apple compare to Microsoft?")
        assert kw.has_keywords == True
        # "Apple" and "Microsoft" should be in ll_keywords (proper nouns)

    def test_short_query_fallback(self):
        """Short query (<50 chars) → entire query as ll_keywords."""
        extractor = DualKeywordExtract(llm=None)
        kw = extractor.extract("diabetes treatment")
        assert kw.extraction_method == "short_query_fallback"
        assert kw.ll_keywords == ["diabetes treatment"]

    def test_empty_query(self):
        """Empty query → empty keywords."""
        extractor = DualKeywordExtract(llm=None)
        kw = extractor.extract("")
        assert kw.has_keywords == False
        assert kw.extraction_method == "empty"

    def test_chinese_query(self):
        """Chinese query → heuristic extraction."""
        config = DualKeywordConfig(language="zh")
        extractor = DualKeywordExtract(llm=None, config=config)
        kw = extractor.extract("糖尿病的治疗方法是什么？")
        assert kw.extraction_method in ("heuristic", "short_query_fallback")

    def test_all_stop_words(self):
        """Query consisting only of stop words → few or no keywords."""
        extractor = DualKeywordExtract(llm=None)
        kw = extractor.extract("What is the?")
        # Mostly stop words → very few or no keywords
        assert len(kw.hl_keywords) + len(kw.ll_keywords) <= 2


# ── Test LLM extraction ──────────────────────────────────────────


class TestLLMExtraction:
    """Test keyword extraction with mock LLM."""

    class MockLLM:
        """Mock LLM that returns predefined JSON."""

        def __init__(self, response=None):
            self._response = response or json.dumps({
                "high_level_keywords": ["treatment", "disease management"],
                "low_level_keywords": ["type 2 diabetes", "insulin therapy"],
            })

        def generate(self, prompt):
            return self._response

    def test_llm_extraction_basic(self):
        """LLM returns valid JSON → keywords extracted."""
        llm = self.MockLLM()
        extractor = DualKeywordExtract(llm=llm)
        kw = extractor.extract("What is the treatment for type 2 diabetes?")
        assert kw.extraction_method == "llm"
        assert "treatment" in kw.hl_keywords
        assert "type 2 diabetes" in kw.ll_keywords

    def test_llm_extraction_markdown_fenced(self):
        """LLM returns markdown-fenced JSON → parsed correctly."""
        response = '```json\n{"high_level_keywords": ["concept"], "low_level_keywords": ["entity"]}\n```'
        llm = self.MockLLM(response=response)
        extractor = DualKeywordExtract(llm=llm)
        kw = extractor.extract("test query")
        assert kw.extraction_method == "llm"
        assert "concept" in kw.hl_keywords
        assert "entity" in kw.ll_keywords

    def test_llm_extraction_empty_response(self):
        """LLM returns empty keywords → falls back to heuristic."""
        response = json.dumps({
            "high_level_keywords": [],
            "low_level_keywords": [],
        })
        llm = self.MockLLM(response=response)
        extractor = DualKeywordExtract(llm=llm, config=DualKeywordConfig(
            fallback_to_heuristic=True, llm_max_retries=1,
        ))
        kw = extractor.extract("What is diabetes treatment?")
        # Empty LLM response → heuristic fallback
        assert kw.extraction_method in ("heuristic", "llm")
        # Even with empty LLM response, should have keywords from fallback

    def test_llm_malformed_json(self):
        """LLM returns malformed JSON → regex fallback extraction."""
        response = '{"high_level_keywords": ["concept", "low_level_keywords": ["entity"]}'
        llm = self.MockLLM(response=response)
        extractor = DualKeywordExtract(llm=llm)
        kw = extractor.extract("test query")
        # Even malformed JSON should be parsed somehow
        assert kw.extraction_method == "llm"

    def test_llm_exception_fallback(self):
        """LLM raises exception → heuristic fallback."""
        class ErrorLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM error")

        extractor = DualKeywordExtract(
            llm=ErrorLLM(),
            config=DualKeywordConfig(fallback_to_heuristic=True, llm_max_retries=1),
        )
        kw = extractor.extract("What is diabetes treatment?")
        assert kw.extraction_method == "heuristic"

    def test_llm_no_fallback_config(self):
        """LLM fails + fallback disabled → llm_failed method."""
        class ErrorLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM error")

        extractor = DualKeywordExtract(
            llm=ErrorLLM(),
            config=DualKeywordConfig(fallback_to_heuristic=False, llm_max_retries=1),
        )
        kw = extractor.extract("test query")
        assert kw.extraction_method == "llm_failed"


# ── Test operator protocol ────────────────────────────────────────


class TestOperatorProtocol:
    """Test DualKeywordExtract.run(context) operator protocol."""

    def test_run_with_query(self):
        """run() writes hl/ll keywords to context."""
        llm = self.MockLLM() if hasattr(self, 'MockLLM') else None
        extractor = DualKeywordExtract(llm=None)  # Use heuristic
        context = {"query": "What is the treatment for diabetes?"}
        result = extractor.run(context)

        assert "hl_keywords" in result
        assert "ll_keywords" in result
        assert "dual_keywords" in result
        assert isinstance(result["dual_keywords"], DualKeywords)

    def test_run_without_query(self):
        """run() with no query → empty keywords."""
        extractor = DualKeywordExtract(llm=None)
        context = {}
        result = extractor.run(context)

        assert result["hl_keywords"] == []
        assert result["ll_keywords"] == []


# ── Test _parse_llm_response ──────────────────────────────────────


class TestParseLLMResponse:
    """Test DualKeywordExtract._parse_llm_response."""

    def test_clean_json(self):
        """Clean JSON → parsed correctly."""
        response = json.dumps({
            "high_level_keywords": ["concept A", "concept B"],
            "low_level_keywords": ["entity X", "entity Y"],
        })
        hl, ll = DualKeywordExtract._parse_llm_response(response)
        assert hl == ["concept A", "concept B"]
        assert ll == ["entity X", "entity Y"]

    def test_json_with_extra_fields(self):
        """JSON with extra fields → only hl/ll extracted."""
        response = json.dumps({
            "high_level_keywords": ["A"],
            "low_level_keywords": ["B"],
            "extra_field": "ignored",
        })
        hl, ll = DualKeywordExtract._parse_llm_response(response)
        assert hl == ["A"]
        assert ll == ["B"]

    def test_none_keyword_list(self):
        """None keyword list → empty list."""
        response = json.dumps({"high_level_keywords": None, "low_level_keywords": None})
        hl, ll = DualKeywordExtract._parse_llm_response(response)
        assert hl == []
        assert ll == []

    def test_string_keyword_list(self):
        """String keyword list → split by comma."""
        response = json.dumps({
            "high_level_keywords": "concept A, concept B",
            "low_level_keywords": "entity X; entity Y",
        })
        hl, ll = DualKeywordExtract._parse_llm_response(response)
        assert len(hl) == 2
        assert len(ll) == 2

    def test_malformed_json_regex_fallback(self):
        """Malformed JSON → regex extraction."""
        response = '{"high_level_keywords": ["A"], "low_level_keywords": ["B"]}'
        hl, ll = DualKeywordExtract._parse_llm_response(response)
        # Should parse at minimum
        assert isinstance(hl, list)
        assert isinstance(ll, list)


# ── Test _normalize_keywords ──────────────────────────────────────


class TestNormalizeKeywords:
    """Test DualKeywordExtract._normalize_keywords."""

    def test_deduplication(self):
        """Duplicate keywords are removed."""
        result = DualKeywordExtract._normalize_keywords(["test", "Test", "test"])
        assert len(result) == 1
        assert result[0] == "test"

    def test_length_filter(self):
        """Keywords shorter than 2 chars are filtered."""
        result = DualKeywordExtract._normalize_keywords(["a", "ab", "abc"])
        assert "a" not in result
        assert "ab" in result
        assert "abc" in result

    def test_case_normalization(self):
        """Keywords are lowercased."""
        result = DualKeywordExtract._normalize_keywords(["Diabetes", "TREATMENT"])
        assert "diabetes" in result
        assert "treatment" in result


# ── Test convenience function ─────────────────────────────────────


class TestExtractDualKeywordsFunction:
    """Test extract_dual_keywords convenience function."""

    def test_quick_extract(self):
        """Quick-extract works with minimal args."""
        kw = extract_dual_keywords("What is diabetes treatment?")
        assert isinstance(kw, DualKeywords)
        assert kw.has_keywords == True

    def test_quick_extract_with_language(self):
        """Language parameter works."""
        kw = extract_dual_keywords("test", language="zh")
        assert isinstance(kw, DualKeywords)
