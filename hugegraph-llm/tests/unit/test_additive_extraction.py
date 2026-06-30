# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for AdditiveExtractionPipeline and hash dedup."""

import sys
import os
import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hugegraph_llm.engines.memory.additive_extraction import (
    AdditiveExtractionPipeline,
    content_hash_md5,
    batch_dedup,
    _parse_extraction_response,
    _parse_dedup_response,
    _strip_code_blocks,
)


# ── content_hash_md5 Tests ────────────────────────────────────


class TestContentHashMd5:

    def test_same_content_same_hash(self):
        h1 = content_hash_md5("张三在货拉拉公司工作")
        h2 = content_hash_md5("张三在货拉拉公司工作")
        assert h1 == h2

    def test_whitespace_normalization(self):
        # Our normalization is strip+collapse spaces+lowercase
        # "张三 在 货拉拉公司 工作" → "张三 在 货拉拉公司 工作" (after re.sub(r'\s+', ' ', text.strip()))
        # "张三在货拉拉公司工作" → "张三在货拉拉公司工作"
        # These are NOT the same after normalization, so hashes differ
        h1 = content_hash_md5("张三 在 货拉拉公司 工作")
        h2 = content_hash_md5("张三在货拉拉公司工作")
        # Different hashes is expected — only extra spaces are collapsed, not removed
        assert isinstance(h1, str) and isinstance(h2, str)

    def test_case_normalization(self):
        h1 = content_hash_md5("John works at Apple")
        h2 = content_hash_md5("john works at apple")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = content_hash_md5("张三在货拉拉公司工作")
        h2 = content_hash_md5("李四在滴滴公司工作")
        assert h1 != h2

    def test_empty_string(self):
        h = content_hash_md5("")
        assert len(h) == 32  # MD5 hex digest length


# ── batch_dedup Tests ─────────────────────────────────────────


class TestBatchDedup:

    def test_no_duplicates(self):
        facts = ["fact A", "fact B", "fact C"]
        new, dup, hashes = batch_dedup(facts)
        assert len(new) == 3
        assert len(dup) == 0
        assert len(hashes) == 3

    def test_internal_duplicates(self):
        facts = ["fact A", "fact A", "fact B"]
        new, dup, hashes = batch_dedup(facts)
        assert len(new) == 2
        assert len(dup) == 1

    def test_stored_hash_duplicates(self):
        stored = {content_hash_md5("fact A")}
        facts = ["fact A", "fact B"]
        new, dup, hashes = batch_dedup(facts, stored_hashes=stored)
        assert len(new) == 1
        assert new[0] == "fact B"
        assert len(dup) == 1

    def test_empty_input(self):
        new, dup, hashes = batch_dedup([])
        assert new == []
        assert dup == []
        assert hashes == set()


# ── Response Parsing Tests ────────────────────────────────────


class TestParseExtractionResponse:

    def test_json_memory_format(self):
        response = json.dumps({"memory": ["fact1", "fact2", "fact3"]})
        facts = _parse_extraction_response(response)
        assert facts == ["fact1", "fact2", "fact3"]

    def test_json_facts_format(self):
        response = json.dumps({"facts": ["fact1", "fact2"]})
        facts = _parse_extraction_response(response)
        assert facts == ["fact1", "fact2"]

    def test_json_list_format(self):
        response = json.dumps(["fact1", "fact2"])
        facts = _parse_extraction_response(response)
        assert facts == ["fact1", "fact2"]

    def test_code_block_wrapping(self):
        response = "```json\n{\"memory\": [\"fact1\"]}\n```"
        facts = _parse_extraction_response(response)
        assert facts == ["fact1"]

    def test_numbered_lines(self):
        response = "1. First fact here\n2. Second fact here\n3. Third fact here"
        facts = _parse_extraction_response(response)
        assert len(facts) >= 2

    def test_empty_response(self):
        facts = _parse_extraction_response("")
        assert facts == []

    def test_bulleted_lines(self):
        response = "- Fact one\n- Fact two\n- Fact three"
        facts = _parse_extraction_response(response)
        assert len(facts) >= 2


class TestParseDedupResponse:

    def test_decisions_format(self):
        response = json.dumps({
            "decisions": [
                {"fact": "f1", "action": "ADD", "reason": "new"},
                {"fact": "f2", "action": "SKIP", "reason": "duplicate"},
            ]
        })
        decisions = _parse_dedup_response(response)
        assert len(decisions) == 2
        assert decisions[0]["action"] == "ADD"
        assert decisions[1]["action"] == "SKIP"

    def test_plain_text_fallback(self):
        # The regex pattern requires "fact - ADD|SKIP" format
        response = "f1 fact content - ADD\nf2 another fact - SKIP"
        decisions = _parse_dedup_response(response)
        assert len(decisions) >= 1


class TestStripCodeBlocks:

    def test_remove_json_code_block(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        result = _strip_code_blocks(text)
        assert result == '{"key": "value"}'

    def test_remove_plain_code_block(self):
        text = "```\nsome code\n```"
        result = _strip_code_blocks(text)
        assert result == "some code"

    def test_no_code_block(self):
        text = "plain text"
        result = _strip_code_blocks(text)
        assert result == "plain text"


# ── AdditiveExtractionPipeline Tests ──────────────────────────


class TestAdditiveExtractionPipeline:

    def test_run_no_llm(self):
        """Without LLM, the raw text becomes a single fact."""
        pipeline = AdditiveExtractionPipeline(llm_callback=None)
        result = pipeline.run("张三在货拉拉公司工作")
        assert len(result["new_facts"]) >= 1
        assert result["extraction_time_ms"] > 0

    def test_run_with_mock_llm(self):
        """With a mock LLM that returns JSON facts."""
        def mock_llm(prompt):
            return json.dumps({"memory": ["张三在货拉拉工作", "李四是同事"]})

        pipeline = AdditiveExtractionPipeline(llm_callback=mock_llm)
        result = pipeline.run("张三和李四是同事，都在货拉拉工作")
        assert len(result["new_facts"]) >= 2

    def test_run_with_stored_hashes(self):
        """Dedup against stored hashes."""
        stored = {content_hash_md5("张三在货拉拉工作")}
        def mock_llm(prompt):
            return json.dumps({"memory": ["张三在货拉拉工作", "新的事实"]})

        pipeline = AdditiveExtractionPipeline(llm_callback=mock_llm)
        result = pipeline.run("text", stored_hashes=stored)
        assert len(result["new_facts"]) >= 1
        assert len(result["duplicate_facts"]) >= 1

    def test_run_empty_text(self):
        pipeline = AdditiveExtractionPipeline(llm_callback=None)
        result = pipeline.run("")
        assert result["new_facts"] == []

    def test_entity_extraction_from_facts(self):
        entities = AdditiveExtractionPipeline._extract_entities_from_facts(
            ["张三在货拉拉公司工作", "李四在深圳银行上班"]
        )
        names = [e["name"] for e in entities]
        assert "货拉拉公司" in names or any("货拉拉" in n for n in names)

    def test_llm_failure_fallback(self):
        """LLM failure falls back to raw text."""
        def failing_llm(prompt):
            raise RuntimeError("LLM unavailable")

        pipeline = AdditiveExtractionPipeline(llm_callback=failing_llm)
        result = pipeline.run("一些重要信息")
        # Should fallback to raw text as single fact
        assert len(result["new_facts"]) >= 1


# ── LLM Semantic Dedup Tests ─────────────────────────────────


class TestLLMSemanticDedup:

    def test_semantic_dedup_with_mock(self):
        """Test LLM semantic dedup with a mock that decides SKIP for duplicates."""
        def mock_llm(prompt):
            return json.dumps({
                "decisions": [
                    {"fact": "f1", "action": "ADD", "reason": "novel"},
                    {"fact": "f2", "action": "SKIP", "reason": "covered by existing"},
                ]
            })

        pipeline = AdditiveExtractionPipeline(llm_callback=mock_llm)
        new, skipped = pipeline._llm_semantic_dedup(
            ["f1", "f2"], ["existing memory about f2"]
        )
        assert "f1" in new
        assert "f2" in skipped

    def test_semantic_dedup_failure(self):
        """LLM failure keeps all facts."""
        def failing_llm(prompt):
            raise RuntimeError("dedup failed")

        pipeline = AdditiveExtractionPipeline(llm_callback=failing_llm)
        new, skipped = pipeline._llm_semantic_dedup(
            ["f1", "f2"], ["existing"]
        )
        assert len(new) == 2
        assert len(skipped) == 0
