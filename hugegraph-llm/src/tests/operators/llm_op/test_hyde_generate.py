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

"""Unit tests for HyDE (Hypothetical Document Embeddings) query enhancement."""

import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.operators.llm_op.hyde_generate import (
    DEFAULT_HYDE_PROMPT,
    DEFAULT_HYDE_PROMPT_CN,
    HyDEGenerate,
)


class TestHyDEGenerateInit(unittest.TestCase):
    """Test HyDEGenerate initialization."""

    def test_default_mode_is_prefix(self):
        enhancer = HyDEGenerate()
        self.assertEqual(enhancer.mode, "prefix")

    def test_custom_mode(self):
        enhancer = HyDEGenerate(mode="full")
        self.assertEqual(enhancer.mode, "full")

    def test_off_mode(self):
        enhancer = HyDEGenerate(mode="off")
        self.assertEqual(enhancer.mode, "off")

    def test_custom_max_query_length(self):
        enhancer = HyDEGenerate(max_query_length=200)
        self.assertEqual(enhancer._max_query_length, 200)

    def test_custom_prompt_template(self):
        custom = "Custom prompt: {query}"
        enhancer = HyDEGenerate(prompt_template=custom)
        self.assertEqual(enhancer._prompt_template, custom)

    def test_custom_prompt_template_cn(self):
        custom_cn = "自定义提示：{query}"
        enhancer = HyDEGenerate(prompt_template_cn=custom_cn)
        self.assertEqual(enhancer._prompt_template_cn, custom_cn)

    def test_llm_can_be_none(self):
        enhancer = HyDEGenerate(llm=None)
        self.assertIsNone(enhancer._llm)

    def test_llm_can_be_provided(self):
        mock_llm = MagicMock()
        enhancer = HyDEGenerate(llm=mock_llm)
        self.assertEqual(enhancer._llm, mock_llm)


class TestShouldEnhance(unittest.TestCase):
    """Test the _should_enhance decision logic."""

    def test_off_mode_never_enhances(self):
        enhancer = HyDEGenerate(mode="off")
        self.assertFalse(enhancer._should_enhance("short query"))

    def test_empty_query_skips(self):
        enhancer = HyDEGenerate(mode="prefix")
        self.assertFalse(enhancer._should_enhance(""))

    def test_none_query_skips(self):
        enhancer = HyDEGenerate(mode="prefix")
        self.assertFalse(enhancer._should_enhance(None))

    def test_whitespace_only_skips(self):
        enhancer = HyDEGenerate(mode="prefix")
        self.assertFalse(enhancer._should_enhance("   "))

    def test_short_query_enhances(self):
        enhancer = HyDEGenerate(mode="prefix", max_query_length=100)
        self.assertTrue(enhancer._should_enhance("What is HugeGraph?"))

    def test_long_query_skips(self):
        enhancer = HyDEGenerate(mode="prefix", max_query_length=10)
        long_query = "a" * 11
        self.assertFalse(enhancer._should_enhance(long_query))

    def test_exact_length_boundary_enhances(self):
        enhancer = HyDEGenerate(mode="prefix", max_query_length=10)
        self.assertTrue(enhancer._should_enhance("a" * 10))

    def test_boundary_length_plus_one_skips(self):
        enhancer = HyDEGenerate(mode="prefix", max_query_length=10)
        self.assertFalse(enhancer._should_enhance("a" * 11))

    def test_prefix_mode_enhances_short_query(self):
        enhancer = HyDEGenerate(mode="prefix")
        self.assertTrue(enhancer._should_enhance("short"))

    def test_full_mode_enhances_short_query(self):
        enhancer = HyDEGenerate(mode="full")
        self.assertTrue(enhancer._should_enhance("short"))


class TestGenerateHypothetical(unittest.TestCase):
    """Test the hypothetical passage generation."""

    def test_calls_llm_with_prompt(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "HugeGraph is a graph database."
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer._generate_hypothetical("What is HugeGraph?")
        self.assertEqual(result, "HugeGraph is a graph database.")
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        self.assertIn("What is HugeGraph?", call_args.kwargs.get("prompt", "") or
                       (call_args.args[0] if call_args.args else ""))

    def test_uses_cn_prompt_for_chinese(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "HugeGraph是一个图数据库。"
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        enhancer._generate_hypothetical("HugeGraph是什么？", language="cn")
        call_args = mock_llm.generate.call_args
        prompt_used = call_args.kwargs.get("prompt", "") or (call_args.args[0] if call_args.args else "")
        self.assertIn("问题", prompt_used)

    def test_uses_en_prompt_by_default(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test"
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        enhancer._generate_hypothetical("test query", language="en")
        call_args = mock_llm.generate.call_args
        prompt_used = call_args.kwargs.get("prompt", "") or (call_args.args[0] if call_args.args else "")
        self.assertIn("Question:", prompt_used)

    def test_returns_empty_on_llm_failure(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer._generate_hypothetical("test")
        self.assertEqual(result, "")

    def test_strips_whitespace(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "  passage with spaces  "
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer._generate_hypothetical("test")
        self.assertEqual(result, "passage with spaces")

    def test_handles_empty_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer._generate_hypothetical("test")
        self.assertEqual(result, "")


class TestEnhance(unittest.TestCase):
    """Test the enhance() method — the main public API."""

    def test_off_mode_returns_original(self):
        enhancer = HyDEGenerate(mode="off")
        query = "What is graph?"
        result = enhancer.enhance(query)
        self.assertEqual(result, query)

    def test_prefix_mode_combines_query_and_passage(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Graph is a data structure."
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer.enhance("What is graph?")
        self.assertIn("What is graph?", result)
        self.assertIn("Graph is a data structure.", result)

    def test_full_mode_returns_only_passage(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Graph is a data structure."
        enhancer = HyDEGenerate(llm=mock_llm, mode="full")

        result = enhancer.enhance("What is graph?")
        self.assertEqual(result, "Graph is a data structure.")

    def test_long_query_returns_original(self):
        mock_llm = MagicMock()
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix", max_query_length=10)

        long_query = "a" * 20
        result = enhancer.enhance(long_query)
        self.assertEqual(result, long_query)
        mock_llm.generate.assert_not_called()

    def test_empty_query_returns_empty(self):
        enhancer = HyDEGenerate(mode="prefix")
        result = enhancer.enhance("")
        self.assertEqual(result, "")

    def test_fallback_on_empty_hypothetical(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""
        enhancer = HyDEGenerate(llm=mock_llm, mode="full")

        query = "short query"
        result = enhancer.enhance(query)
        self.assertEqual(result, query)

    def test_fallback_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API down")
        enhancer = HyDEGenerate(llm=mock_llm, mode="full")

        query = "short query"
        result = enhancer.enhance(query)
        self.assertEqual(result, query)

    def test_cn_language_uses_cn_prompt(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "图数据库是..."
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer.enhance("什么是图数据库？", language="cn")
        self.assertIn("图数据库是...", result)

    def test_chinese_short_query_enhanced(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "HugeGraph是一个图数据库。"
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        result = enhancer.enhance("HugeGraph")
        self.assertIn("HugeGraph", result)
        self.assertIn("HugeGraph是一个图数据库。", result)


class TestRunOperator(unittest.TestCase):
    """Test the run(context) operator interface."""

    def test_run_sets_hyde_query(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hypothetical answer."
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        context = {"query": "test?", "call_count": 0}
        result = enhancer.run(context)

        self.assertIn("hyde_query", result)
        self.assertIn("test?", result["hyde_query"])
        self.assertIn("Hypothetical answer.", result["hyde_query"])

    def test_run_sets_hyde_applied_true(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hypothetical answer."
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        context = {"query": "test?"}
        result = enhancer.run(context)

        self.assertTrue(result["hyde_applied"])

    def test_run_sets_hyde_applied_false_when_off(self):
        enhancer = HyDEGenerate(mode="off")
        context = {"query": "test?"}
        result = enhancer.run(context)

        self.assertFalse(result["hyde_applied"])

    def test_run_preserves_original_query(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hypothetical."
        enhancer = HyDEGenerate(llm=mock_llm, mode="full")

        context = {"query": "original?"}
        result = enhancer.run(context)

        self.assertEqual(result["original_query"], "original?")

    def test_run_increments_call_count_when_enhanced(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hypothetical."
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        context = {"query": "test?", "call_count": 5}
        result = enhancer.run(context)

        self.assertEqual(result["call_count"], 6)

    def test_run_does_not_increment_call_count_when_skipped(self):
        enhancer = HyDEGenerate(mode="off")

        context = {"query": "test?", "call_count": 5}
        result = enhancer.run(context)

        self.assertEqual(result["call_count"], 5)

    def test_run_with_missing_query(self):
        enhancer = HyDEGenerate(mode="prefix")
        context = {}
        result = enhancer.run(context)

        self.assertFalse(result["hyde_applied"])
        self.assertEqual(result["hyde_query"], "")

    def test_run_full_mode_hyde_query_is_passage_only(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Only the passage."
        enhancer = HyDEGenerate(llm=mock_llm, mode="full")

        context = {"query": "original?"}
        result = enhancer.run(context)

        self.assertEqual(result["hyde_query"], "Only the passage.")

    def test_run_with_language_context(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "CN answer"
        enhancer = HyDEGenerate(llm=mock_llm, mode="prefix")

        context = {"query": "问题?", "language": "cn"}
        result = enhancer.run(context)

        call_args = mock_llm.generate.call_args
        prompt_used = call_args.kwargs.get("prompt", "") or (call_args.args[0] if call_args.args else "")
        self.assertIn("问题", prompt_used)


if __name__ == "__main__":
    unittest.main()
