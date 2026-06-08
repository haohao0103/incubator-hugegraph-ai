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

"""Tests for Token Budget controller."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from hugegraph_llm.operators.graph_op.token_budget import (
    TokenBudget,
    TokenBudgetConfig,
    _estimate_tokens,
)


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_estimate_tokens(""), 0)

    def test_latin_text(self):
        # ~4 chars per token
        text = "Hello world this is a test"
        est = _estimate_tokens(text)
        self.assertGreater(est, 0)
        self.assertLess(est, len(text))  # tokens < chars

    def test_cjk_text(self):
        # ~1.5 chars per token (CJK is denser)
        text = "中华人民共和国"
        est = _estimate_tokens(text)
        self.assertGreater(est, 0)
        # CJK should estimate more tokens per char than Latin
        latin = "abcdefg" * 2
        est_latin = _estimate_tokens(latin)
        # 14 Latin chars should give fewer tokens than 7 CJK chars
        self.assertGreater(est, est_latin)

    def test_mixed_text(self):
        text = "Apache HugeGraph是一个图数据库"
        est = _estimate_tokens(text)
        self.assertGreater(est, 0)


class TestTokenBudgetConfig(unittest.TestCase):
    def test_defaults(self):
        config = TokenBudgetConfig()
        self.assertEqual(config.max_total_tokens, 4096)
        self.assertEqual(config.reserve_for_prompt, 300)

    def test_effective_total(self):
        config = TokenBudgetConfig(max_total_tokens=4096, reserve_for_prompt=300)
        self.assertEqual(config.effective_total(), 3796)

    def test_effective_total_floor(self):
        config = TokenBudgetConfig(max_total_tokens=100, reserve_for_prompt=500)
        self.assertEqual(config.effective_total(), 0)


class TestTokenBudget(unittest.TestCase):
    def setUp(self):
        self.budget = TokenBudget(TokenBudgetConfig(
            max_total_tokens=200,
            max_entity_tokens=80,
            max_relation_tokens=60,
            max_community_tokens=40,
            max_chunk_tokens=10,
            reserve_for_prompt=10,
        ))

    def test_add_within_budget(self):
        ok = self.budget.add("entity", "Entity: Test", estimated_tokens=5)
        self.assertTrue(ok)
        self.assertEqual(self.budget.total_used, 5)

    def test_add_exceeds_category(self):
        # Fill entity budget
        self.budget.add("entity", "E1", estimated_tokens=50)
        self.budget.add("entity", "E2", estimated_tokens=30)
        # This should fail (50+30+30 > 80)
        ok = self.budget.add("entity", "E3", estimated_tokens=30)
        self.assertFalse(ok)
        self.assertEqual(self.budget.total_used, 80)

    def test_add_exceeds_total(self):
        self.budget.add("entity", "E1", estimated_tokens=80)
        self.budget.add("relation", "R1", estimated_tokens=60)
        # Total 140 + anything more should be capped at effective_total=190
        ok = self.budget.add("entity", "E2", estimated_tokens=60)
        self.assertFalse(ok)

    def test_add_truncated_fits(self):
        text = "Short text"
        result = self.budget.add_truncated("entity", text, estimated_tokens=5)
        self.assertEqual(result, text)

    def test_add_truncated_truncates(self):
        # Fill up most of entity budget
        self.budget.add("entity", "Existing entity content", estimated_tokens=75)
        # Try to add 10 tokens worth, but only 5 remain
        result = self.budget.add_truncated("entity", "New entity content here", estimated_tokens=10)
        self.assertTrue(len(result) > 0)
        self.assertTrue(len(result) < len("New entity content here"))

    def test_add_truncated_no_budget(self):
        self.budget.add("entity", "X" * 80, estimated_tokens=80)
        result = self.budget.add_truncated("entity", "More", estimated_tokens=5)
        self.assertEqual(result, "")

    def test_build_context(self):
        self.budget.add("entity", "Entity A", estimated_tokens=5)
        self.budget.add("entity", "Entity B", estimated_tokens=5)
        self.budget.add("relation", "A->B", estimated_tokens=3)
        context = self.budget.build_context()
        # Community section should be empty (no entries)
        self.assertIn("Entity A", context)
        self.assertIn("Entity B", context)
        self.assertIn("A->B", context)

    def test_build_context_priority_order(self):
        self.budget.add("chunk", "chunk text", estimated_tokens=5)
        self.budget.add("entity", "entity text", estimated_tokens=5)
        self.budget.add("community", "community text", estimated_tokens=5)
        self.budget.add("relation", "relation text", estimated_tokens=5)
        context = self.budget.build_context()
        # Community should appear before entity in output
        comm_idx = context.index("community text")
        ent_idx = context.index("entity text")
        self.assertLess(comm_idx, ent_idx)

    def test_summary(self):
        self.budget.add("entity", "E1", estimated_tokens=10)
        self.budget.add("relation", "R1", estimated_tokens=5)
        s = self.budget.summary()
        self.assertEqual(s["total_used"], 15)
        self.assertIn("entity", s["by_category"])
        self.assertIn("relation", s["by_category"])
        self.assertEqual(s["by_category"]["entity"]["entries"], 1)

    def test_remaining(self):
        self.budget.add("entity", "E1", estimated_tokens=10)
        self.assertGreater(self.budget.remaining, 0)
        self.assertLessEqual(self.budget.remaining, 190)

    def test_reset(self):
        self.budget.add("entity", "E1", estimated_tokens=50)
        self.budget.reset()
        self.assertEqual(self.budget.total_used, 0)
        self.assertEqual(self.budget.remaining, 190)

    def test_unknown_category(self):
        # Unknown category should use effective_total as limit
        ok = self.budget.add("unknown", "test", estimated_tokens=5)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
