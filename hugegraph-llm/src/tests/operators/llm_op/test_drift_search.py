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

"""Unit tests for DRIFT search operator."""

import json
import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.operators.llm_op.drift_search import (
    DEEP_FOLLOW_UP_PROMPT,
    DEEP_FOLLOW_UP_PROMPT_CN,
    DriftSearch,
    FINDING_SYNTHESIZE_PROMPT,
    HYDE_PROMPT,
    HYDE_PROMPT_CN,
    PRIMER_PROMPT,
    REDUCE_PROMPT,
)


def _make_llm(responses):
    """Create a mock LLM that returns responses in order."""
    mock = MagicMock()
    mock.generate.side_effect = responses
    return mock


SAMPLE_COMMUNITIES = [
    {
        "community_id": "C1",
        "title": "Supply Chain Risk",
        "summary": "Supply chain vulnerabilities and dependencies.",
        "key_entities": ["Supplier A", "Material X"],
        "importance_score": 8.5,
        "relationship_patterns": ["supplies", "depends_on"],
    },
    {
        "community_id": "C2",
        "title": "Financial Controls",
        "summary": "Financial risk management and controls.",
        "key_entities": ["Bank B", "Currency Risk"],
        "importance_score": 7.0,
        "relationship_patterns": ["manages", "monitors"],
    },
    {
        "community_id": "C3",
        "title": "Regulatory Compliance",
        "summary": "Regulatory requirements and compliance status.",
        "key_entities": ["Regulator R", "Standard S"],
        "importance_score": 6.0,
        "relationship_patterns": ["enforces", "complies_with"],
    },
]


class TestDriftSearchInit(unittest.TestCase):
    """Test DriftSearch initialization."""

    def test_default_params(self):
        ds = DriftSearch()
        self.assertEqual(ds._max_local_depth, 2)
        self.assertEqual(ds._communities_top_k, 5)
        self.assertEqual(ds._local_search_top_k, 10)
        self.assertEqual(ds._language, "en")

    def test_custom_params(self):
        ds = DriftSearch(
            max_local_depth=3,
            communities_top_k=10,
            language="cn",
        )
        self.assertEqual(ds._max_local_depth, 3)
        self.assertEqual(ds._communities_top_k, 10)
        self.assertEqual(ds._language, "cn")

    def test_max_depth_clamped_upper(self):
        ds = DriftSearch(max_local_depth=10)
        self.assertEqual(ds._max_local_depth, 3)

    def test_max_depth_clamped_lower(self):
        ds = DriftSearch(max_local_depth=0)
        self.assertEqual(ds._max_local_depth, 1)

    def test_llm_can_be_none(self):
        ds = DriftSearch(llm=None)
        self.assertIsNone(ds._llm)

    def test_llm_can_be_provided(self):
        mock_llm = MagicMock()
        ds = DriftSearch(llm=mock_llm)
        self.assertEqual(ds._llm, mock_llm)


class TestStep1HyDE(unittest.TestCase):
    """Test Step 1: HyDE generation."""

    def test_generates_hyde_passage(self):
        mock_llm = _make_llm(["HugeGraph is a distributed graph database."])
        ds = DriftSearch(llm=mock_llm)
        result = ds._step1_hyde("What is HugeGraph?")
        self.assertEqual(result, "HugeGraph is a distributed graph database.")

    def test_returns_empty_on_llm_failure(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")
        ds = DriftSearch(llm=mock_llm)
        result = ds._step1_hyde("test?")
        self.assertEqual(result, "")

    def test_strips_whitespace(self):
        mock_llm = _make_llm(["  passage  "])
        ds = DriftSearch(llm=mock_llm)
        result = ds._step1_hyde("test?")
        self.assertEqual(result, "passage")

    def test_uses_cn_prompt_for_chinese(self):
        mock_llm = _make_llm(["CN passage"])
        ds = DriftSearch(llm=mock_llm, language="cn")
        ds._step1_hyde("测试?")
        call_args = mock_llm.generate.call_args
        prompt_used = call_args.kwargs.get("prompt", "")
        self.assertIn("问题", prompt_used)


class TestStep2CommunityMatch(unittest.TestCase):
    """Test Step 2: Community matching."""

    def test_returns_top_k_by_importance(self):
        ds = DriftSearch(communities_top_k=2)
        result = ds._step2_match_communities("test query", SAMPLE_COMMUNITIES)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["community_id"], "C1")

    def test_returns_empty_for_no_communities(self):
        ds = DriftSearch()
        result = ds._step2_match_communities("test", [])
        self.assertEqual(result, [])

    def test_returns_all_when_fewer_than_top_k(self):
        ds = DriftSearch(communities_top_k=10)
        result = ds._step2_match_communities("test", SAMPLE_COMMUNITIES)
        self.assertEqual(len(result), 3)

    def test_sorts_by_importance_descending(self):
        ds = DriftSearch(communities_top_k=10)
        result = ds._step2_match_communities("test", SAMPLE_COMMUNITIES)
        scores = [r["importance_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_uses_vector_similarity_when_available(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ]
        ds = DriftSearch(
            embedding=mock_embedding,
            vector_index=MagicMock(),
            communities_top_k=2,
        )
        result = ds._step2_match_communities("query", SAMPLE_COMMUNITIES)
        self.assertEqual(len(result), 2)


class TestStep3Primer(unittest.TestCase):
    """Test Step 3: Primer generation."""

    def test_generates_primer_with_json(self):
        mock_llm = _make_llm([
            json.dumps({
                "initial_answer": "Initial analysis here.",
                "follow_up_queries": ["What are the risks?", "How to mitigate?"],
            })
        ])
        ds = DriftSearch(llm=mock_llm)
        result = ds._step3_primer("What are the supply chain risks?", SAMPLE_COMMUNITIES[:2])
        self.assertEqual(result["initial_answer"], "Initial analysis here.")
        self.assertEqual(len(result["follow_up_queries"]), 2)

    def test_fallback_on_invalid_json(self):
        mock_llm = _make_llm(["not json at all"])
        ds = DriftSearch(llm=mock_llm)
        result = ds._step3_primer("test?", SAMPLE_COMMUNITIES[:2])
        self.assertIn("initial_answer", result)
        self.assertIsInstance(result["follow_up_queries"], list)

    def test_fallback_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API down")
        ds = DriftSearch(llm=mock_llm)
        result = ds._step3_primer("test?", SAMPLE_COMMUNITIES[:2])
        self.assertEqual(result["initial_answer"], "")
        self.assertEqual(result["follow_up_queries"], ["test?"])

    def test_extracts_json_from_code_block(self):
        mock_llm = _make_llm([
            '```json\n{"initial_answer": "analysis", "follow_up_queries": ["q1"]}\n```'
        ])
        ds = DriftSearch(llm=mock_llm)
        result = ds._step3_primer("test?", SAMPLE_COMMUNITIES[:1])
        self.assertEqual(result["initial_answer"], "analysis")

    def test_uses_cn_prompt_for_chinese(self):
        mock_llm = _make_llm([json.dumps({
            "initial_answer": "初始分析",
            "follow_up_queries": ["子问题1"],
        })])
        ds = DriftSearch(llm=mock_llm, language="cn")
        ds._step3_primer("测试?", SAMPLE_COMMUNITIES[:1])
        call_args = mock_llm.generate.call_args
        prompt_used = call_args.kwargs.get("prompt", "")
        self.assertIn("用户问题", prompt_used)


class TestStep4LocalSearch(unittest.TestCase):
    """Test Step 4: Parallel Local Search."""

    def test_returns_findings_for_each_query(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_index = MagicMock()
        mock_vector_index.search.return_value = ["chunk1", "chunk2"]

        mock_llm = _make_llm([
            "Finding 1",  # finding synthesize for query 1
            "Finding 2",  # finding synthesize for query 2
        ])
        ds = DriftSearch(
            llm=mock_llm,
            embedding=mock_embedding,
            vector_index=mock_vector_index,
        )
        findings = ds._step4_parallel_local_search(["q1", "q2"])
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0]["sub_query"], "q1")
        self.assertEqual(findings[1]["sub_query"], "q2")

    def test_returns_empty_for_no_queries(self):
        ds = DriftSearch()
        findings = ds._step4_parallel_local_search([])
        self.assertEqual(findings, [])

    def test_handles_search_failure_gracefully(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.side_effect = Exception("embedding error")
        ds = DriftSearch(embedding=mock_embedding, vector_index=MagicMock())
        findings = ds._step4_parallel_local_search(["q1"])
        # Even when vector search fails, fallback findings are produced
        self.assertEqual(len(findings), 1)
        self.assertIn("No relevant information", findings[0]["finding"])

    def test_no_vector_search_when_no_embedding(self):
        ds = DriftSearch()
        results = ds._local_vector_search("test?")
        self.assertEqual(results, [])

    def test_finding_synthesize_fallback(self):
        ds = DriftSearch()
        result = ds._local_synthesize("q?", [])
        self.assertIn("No relevant information", result)


class TestDeepFollowUps(unittest.TestCase):
    """Test generating deep follow-up questions."""

    def test_generates_follow_ups_from_findings(self):
        mock_llm = _make_llm(['["new q1", "new q2", "new q3"]'])
        ds = DriftSearch(llm=mock_llm)
        findings = [{"finding": f"Finding {i}"} for i in range(5)]
        result = ds._generate_deep_follow_ups("original?", findings)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "new q1")

    def test_returns_empty_on_parse_failure(self):
        mock_llm = _make_llm(["not json"])
        ds = DriftSearch(llm=mock_llm)
        result = ds._generate_deep_follow_ups("q?", [])
        self.assertEqual(result, [])

    def test_limits_to_3_follow_ups(self):
        mock_llm = _make_llm(['["q1", "q2", "q3", "q4", "q5"]'])
        ds = DriftSearch(llm=mock_llm)
        result = ds._generate_deep_follow_ups("q?", [])
        self.assertTrue(len(result) <= 3)


class TestStep5Reduce(unittest.TestCase):
    """Test Step 5: Reduce / final synthesis."""

    def test_synthesizes_answer(self):
        mock_llm = _make_llm(["Final comprehensive answer."])
        ds = DriftSearch(llm=mock_llm)
        findings = [
            {"sub_query": "q1", "finding": "Finding 1 details"},
            {"sub_query": "q2", "finding": "Finding 2 details"},
        ]
        result = ds._step5_reduce("What is the risk?", "Initial analysis.", findings)
        self.assertEqual(result, "Final comprehensive answer.")

    def test_returns_initial_answer_when_no_findings(self):
        ds = DriftSearch()
        result = ds._step5_reduce("query?", "Initial answer.", [])
        self.assertEqual(result, "Initial answer.")

    def test_returns_fallback_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        ds = DriftSearch(llm=mock_llm)
        findings = [{"sub_query": "q1", "finding": "Detail 1"}]
        result = ds._step5_reduce("query?", "Initial.", findings)
        self.assertIn("Initial.", result)

    def test_uses_cn_prompt_for_chinese(self):
        mock_llm = _make_llm(["最终答案"])
        ds = DriftSearch(llm=mock_llm, language="cn")
        ds._step5_reduce("问题?", "初始", [{"sub_query": "q1", "finding": "发现1"}])
        call_args = mock_llm.generate.call_args
        prompt_used = call_args.kwargs.get("prompt", "")
        self.assertIn("用户问题", prompt_used)


class TestFullRun(unittest.TestCase):
    """Test the complete DRIFT run() pipeline."""

    def test_full_pipeline_with_community_reports(self):
        """Test complete 5-step pipeline."""
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_index = MagicMock()
        mock_vector_index.search.return_value = ["chunk_result"]

        # Step 1: HyDE
        # Step 3: Primer
        # Step 4: finding synthesize (depth 1, 2 queries)
        # Step 4: deep follow-ups (depth 1 → 2)
        # Step 4: finding synthesize (depth 2, 2 queries)
        # Step 5: Reduce
        mock_llm = _make_llm([
            "HyDE passage about supply chain.",        # Step 1
            json.dumps({                                # Step 3
                "initial_answer": "Supply chain has risks.",
                "follow_up_queries": ["Risk 1?", "Risk 2?"],
            }),
            "Finding for risk 1",                      # Step 4 depth 1
            "Finding for risk 2",                      # Step 4 depth 1
            json.dumps(["Deep q1", "Deep q2"]),       # Step 4 depth 1 → 2
            "Deep finding 1",                          # Step 4 depth 2
            "Deep finding 2",                          # Step 4 depth 2
            "Final comprehensive answer.",              # Step 5
        ])

        ds = DriftSearch(
            llm=mock_llm,
            embedding=mock_embedding,
            vector_index=mock_vector_index,
            max_local_depth=2,
            communities_top_k=2,
        )
        context = {
            "query": "What are the key supply chain risks?",
            "community_reports": SAMPLE_COMMUNITIES,
            "call_count": 0,
        }
        result = ds.run(context)

        self.assertEqual(result["drift_answer"], "Final comprehensive answer.")
        self.assertEqual(result["drift_communities_used"], 2)
        self.assertEqual(result["drift_depth_reached"], 2)
        self.assertTrue(len(result["drift_findings"]) >= 2)
        self.assertEqual(result["call_count"], 8)

    def test_full_pipeline_depth_1(self):
        """Test pipeline with depth=1 (single iteration)."""
        mock_llm = _make_llm([
            "HyDE passage",
            json.dumps({
                "initial_answer": "Initial.",
                "follow_up_queries": ["q1?"],
            }),
            "Finding 1",
            "Final answer.",
        ])
        ds = DriftSearch(llm=mock_llm, max_local_depth=1, communities_top_k=1)
        context = {
            "query": "test?",
            "community_reports": SAMPLE_COMMUNITIES,
            "call_count": 0,
        }
        result = ds.run(context)
        self.assertEqual(result["drift_depth_reached"], 1)
        self.assertEqual(len(result["drift_findings"]), 1)

    def test_empty_query_returns_empty(self):
        ds = DriftSearch()
        context = {"query": "", "community_reports": SAMPLE_COMMUNITIES}
        result = ds.run(context)
        self.assertEqual(result["drift_answer"], "")

    def test_no_community_reports(self):
        mock_embedding = MagicMock()
        mock_embedding.get_texts_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_index = MagicMock()
        mock_vector_index.search.return_value = []

        mock_llm = _make_llm([
            "HyDE passage",
            json.dumps({"initial_answer": "No info", "follow_up_queries": []}),
            "Final answer.",
        ])
        ds = DriftSearch(llm=mock_llm, embedding=mock_embedding, vector_index=mock_vector_index)
        context = {
            "query": "test?",
            "community_reports": [],
            "call_count": 0,
        }
        result = ds.run(context)
        self.assertEqual(result["drift_communities_used"], 0)
        self.assertIn("drift_answer", result)

    def test_hyde_failure_continues_pipeline(self):
        """Pipeline continues even if HyDE step fails."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            Exception("HyDE fails"),                      # Step 1 fails
            json.dumps({                                     # Step 3
                "initial_answer": "Analysis.",
                "follow_up_queries": ["q1?"],
            }),
            "Finding 1",
            "Final answer.",
        ]
        ds = DriftSearch(llm=mock_llm, max_local_depth=1)
        context = {
            "query": "test?",
            "community_reports": SAMPLE_COMMUNITIES,
            "call_count": 0,
        }
        result = ds.run(context)
        # Should still produce an answer
        self.assertIsNotNone(result["drift_answer"])

    def test_primer_failure_returns_default_queries(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "HyDE passage",
            Exception("Primer fails"),
        ]
        ds = DriftSearch(llm=mock_llm, max_local_depth=1)
        context = {
            "query": "test?",
            "community_reports": SAMPLE_COMMUNITIES,
            "call_count": 0,
        }
        result = ds.run(context)
        # Should fall back gracefully
        self.assertIn("drift_answer", result)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity helper."""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(DriftSearch._cosine_similarity(vec, vec), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(
            DriftSearch._cosine_similarity([1, 0, 0], [0, 1, 0]), 0.0
        )

    def test_empty_vectors(self):
        self.assertEqual(DriftSearch._cosine_similarity([], []), 0.0)

    def test_unequal_length(self):
        self.assertEqual(DriftSearch._cosine_similarity([1, 2], [1, 2, 3]), 0.0)

    def test_none_vectors(self):
        self.assertEqual(DriftSearch._cosine_similarity(None, None), 0.0)

    def test_known_values(self):
        import math
        sim = DriftSearch._cosine_similarity([1, 1], [1, 0])
        expected = 1 / math.sqrt(2)
        self.assertAlmostEqual(sim, expected, places=4)


class TestJsonParsing(unittest.TestCase):
    """Test JSON parsing helpers."""

    def test_parse_primer_direct_json(self):
        response = '{"initial_answer": "test", "follow_up_queries": ["q1"]}'
        result = DriftSearch._parse_primer_json(response)
        self.assertEqual(result["initial_answer"], "test")

    def test_parse_primer_code_block(self):
        response = '```json\n{"initial_answer": "test", "follow_up_queries": ["q1"]}\n```'
        result = DriftSearch._parse_primer_json(response)
        self.assertEqual(result["initial_answer"], "test")

    def test_parse_primer_embedded_json(self):
        response = 'Here is the result: {"initial_answer": "test", "follow_up_queries": ["q1"]}'
        result = DriftSearch._parse_primer_json(response)
        self.assertEqual(result["initial_answer"], "test")

    def test_parse_primer_invalid_fallback(self):
        response = "No JSON here at all"
        result = DriftSearch._parse_primer_json(response)
        self.assertIn("initial_answer", result)
        self.assertEqual(result["follow_up_queries"], [])

    def test_parse_json_array_direct(self):
        result = DriftSearch._parse_json_array('["a", "b", "c"]')
        self.assertEqual(result, ["a", "b", "c"])

    def test_parse_json_array_code_block(self):
        result = DriftSearch._parse_json_array('```json\n["a", "b"]\n```')
        self.assertEqual(result, ["a", "b"])

    def test_parse_json_array_invalid(self):
        result = DriftSearch._parse_json_array("not json")
        self.assertEqual(result, [])


class TestCommunityToText(unittest.TestCase):
    """Test _community_to_text helper."""

    def test_basic_conversion(self):
        report = {
            "title": "Test",
            "summary": "Summary text",
            "key_entities": ["A", "B"],
            "relationship_patterns": ["rel1"],
        }
        text = DriftSearch._community_to_text(report)
        self.assertIn("Test", text)
        self.assertIn("Summary text", text)
        self.assertIn("A", text)
        self.assertIn("B", text)
        self.assertIn("rel1", text)

    def test_missing_fields(self):
        report = {"title": "Test"}
        text = DriftSearch._community_to_text(report)
        self.assertIn("Test", text)


if __name__ == "__main__":
    unittest.main()
