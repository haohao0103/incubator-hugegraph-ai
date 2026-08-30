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

"""Tests for query understanding (dual keywords + synonym expansion)."""

import unittest
from typing import Any, Dict

from hugegraph_llm.operators.graph_op.kg_query_understanding import (
    QueryIntent,
    QueryUnderstanding,
    QueryUnderstandingConfig,
    _NOISE_TAILS,
)
from hugegraph_llm.operators.graph_op.kg_term_graph import KgTermGraph


class _FakeLLM:
    """Deterministic LLM double returning a canned dual-keyword JSON."""

    def __init__(self, hl=None, ll=None, fail: bool = False) -> None:
        self._hl = hl or []
        self._ll = ll or []
        self._fail = fail

    def generate(self, prompt: str) -> str:
        if self._fail:
            raise RuntimeError("llm down")
        return (
            '{"high_level_keywords": ' + str(self._hl) +
            ', "low_level_keywords": ' + str(self._ll) + "}"
        ).replace("'", '"')


def _terms_graph() -> KgTermGraph:
    return KgTermGraph.from_jargon_map(
        {"客单价": "arpu", "平均每单成交金额": "arpu", "大单": "order_total"}
    )


class TestQueryUnderstanding(unittest.TestCase):
    def setUp(self) -> None:
        self.terms = _terms_graph()

    def test_heuristic_fallback_expands_synonyms(self):
        # force the heuristic path (no LLM, threshold below question length)
        qu = QueryUnderstanding(
            term_graph=self.terms,
            config=QueryUnderstandingConfig(short_query_threshold=5),
        )
        intent = qu.understand("客单价是多少")
        self.assertEqual(intent.extraction_method, "heuristic")
        self.assertIn("arpu", intent.expanded_terms)  # synonym expanded
        hits = dict(intent.synonym_hits)
        self.assertEqual(hits.get("客单价"), "arpu")

    def test_llm_path_uses_hl_ll_split(self):
        qu = QueryUnderstanding(
            llm=_FakeLLM(hl=["订单"], ll=["客单价"]),
            term_graph=self.terms,
        )
        intent = qu.understand("客单价是多少")
        self.assertEqual(intent.extraction_method, "llm")
        self.assertIn("客单价", intent.ll_keywords)
        self.assertIn("订单", intent.hl_keywords)
        # entities come first, then hl, then the expanded canonical
        self.assertEqual(intent.expanded_terms[0], "客单价")
        self.assertIn("arpu", intent.expanded_terms)

    def test_llm_failure_falls_back_to_heuristic(self):
        qu = QueryUnderstanding(
            llm=_FakeLLM(fail=True),
            term_graph=self.terms,
            config=QueryUnderstandingConfig(),
        )
        intent = qu.understand("客单价是多少")
        self.assertEqual(intent.extraction_method, "heuristic")

    def test_local_context_contains_entities_and_canonicals(self):
        qu = QueryUnderstanding(llm=_FakeLLM(hl=["订单"], ll=["客单价"]), term_graph=self.terms)
        intent = qu.understand("客单价是多少")
        local = intent.local_context
        self.assertIn("客单价", local)
        self.assertIn("arpu", local)

    def test_global_context_contains_themes(self):
        qu = QueryUnderstanding(llm=_FakeLLM(hl=["订单"], ll=["客单价"]), term_graph=self.terms)
        intent = qu.understand("客单价是多少")
        self.assertEqual(intent.global_context, "订单")

    def test_local_context_dedupes_canonical_already_present(self):
        # direct construction: synonym hit whose canonical is already in
        # ll_keywords must not duplicate it in local_context
        intent = QueryIntent(
            question="客单价是多少",
            ll_keywords=["arpu"],
            synonym_hits=[("客单价", "arpu")],
        )
        self.assertEqual(intent.local_context.count("arpu"), 1)

    def test_term_graph_accessor(self):
        qu = QueryUnderstanding(term_graph=self.terms)
        self.assertIs(qu.term_graph, self.terms)

    def test_duplicate_and_empty_ll_keywords_deduped(self):
        # direct inject of dirty keywords bypassing the LLM normalizer, to
        # exercise the dedupe/empty guard inside understand()
        from hugegraph_llm.operators.llm_op.dual_keyword_extract import DualKeywords
        from unittest import mock

        qu = QueryUnderstanding(
            term_graph=self.terms,
            config=QueryUnderstandingConfig(short_query_threshold=5),
        )
        with mock.patch.object(
            qu._extractor,
            "extract",
            return_value=DualKeywords(
                hl_keywords=["", "订单", "订单"],
                ll_keywords=["", "客单价", "客单价"],
            ),
        ):
            intent = qu.understand("客单价是多少")
        # ll_keywords keep the raw extractor output (trace); the retrieval
        # terms are what get deduplicated/cleaned
        self.assertNotIn("", intent.expanded_terms)
        self.assertEqual(intent.expanded_terms.count("客单价"), 1)
        self.assertEqual(intent.expanded_terms.count(""), 0)

    def test_cjk_block_adds_bigrams(self):
        # a >4-char CJK block contributes its 2-grams so the lexical path can
        # still score against definitions that only share substrings
        from hugegraph_llm.operators.llm_op.dual_keyword_extract import DualKeywords
        from unittest import mock

        qu = QueryUnderstanding(
            term_graph=self.terms,
            config=QueryUnderstandingConfig(short_query_threshold=5),
        )
        with mock.patch.object(
            qu._extractor,
            "extract",
            return_value=DualKeywords(ll_keywords=["每个城市的订单总额"]),
        ):
            intent = qu.understand("每个城市的订单总额")
        grams = intent.expanded_terms
        self.assertIn("订单", grams)
        self.assertIn("总额", grams)
        self.assertIn("城市", grams)
        # the full block stays as the primary term
        self.assertEqual(grams[0], "每个城市的订单总额")

    def test_empty_question(self):
        qu = QueryUnderstanding(term_graph=self.terms)
        intent = qu.understand("")
        self.assertFalse(intent.has_terms)

    def test_noise_tails_filtered(self):
        # "多少" must not survive as a retrieval term
        qu = QueryUnderstanding(term_graph=self.terms)
        intent = qu.understand("客单价是多少")
        self.assertNotIn("多少", intent.expanded_terms)

    def test_include_hl_in_terms_off(self):
        qu = QueryUnderstanding(
            llm=_FakeLLM(hl=["订单"], ll=["客单价"]),
            term_graph=self.terms,
            config=QueryUnderstandingConfig(include_hl_in_terms=False),
        )
        intent = qu.understand("客单价是多少")
        self.assertNotIn("订单", intent.expanded_terms)
        self.assertIn("客单价", intent.expanded_terms)

    def test_expand_with_synonyms_off(self):
        qu = QueryUnderstanding(
            term_graph=self.terms,
            config=QueryUnderstandingConfig(expand_with_synonyms=False),
        )
        intent = qu.understand("客单价是多少")
        self.assertNotIn("arpu", intent.expanded_terms)

    def test_short_query_uses_whole_question(self):
        qu = QueryUnderstanding(
            term_graph=self.terms,
            config=QueryUnderstandingConfig(short_query_threshold=50),
        )
        intent = qu.understand("客单价是多少")
        # short-query fallback keeps the whole question; the interrogative
        # tail is stripped so the remaining entity term is clean
        self.assertEqual(intent.extraction_method, "short_query_fallback")
        self.assertIn("客单价是", intent.ll_keywords)

    def test_intent_to_dict(self):
        qu = QueryUnderstanding(llm=_FakeLLM(hl=["订单"], ll=["客单价"]), term_graph=self.terms)
        d = qu.understand("客单价是多少").to_dict()
        self.assertEqual(d["question"], "客单价是多少")
        self.assertIn("local_context", d)
        self.assertIn("global_context", d)
        self.assertIn("extraction_method", d)

    def test_noise_tail_constant_is_a_set(self):
        self.assertIsInstance(_NOISE_TAILS, set)


if __name__ == "__main__":
    unittest.main()
