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

"""Tests for RRF fusion algorithm."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from hugegraph_llm.operators.graph_op.rrf_fusion import (
    ReciprocalRankFusion,
    RRFResults,
    fuse_results,
    fuse_results_with_scores,
)


class TestRRFResults(unittest.TestCase):
    def test_top_k(self):
        r = RRFResults(["a", "b", "c"], {"a": 0.5, "b": 0.3, "c": 0.1})
        self.assertEqual(r.top_k(2), ["a", "b"])
        self.assertEqual(r.top_k(5), ["a", "b", "c"])
        self.assertEqual(r.top_k(0), [])

    def test_len(self):
        r = RRFResults(["a", "b"], {})
        self.assertEqual(len(r), 2)
        r2 = RRFResults([], {})
        self.assertEqual(len(r2), 0)

    def test_repr(self):
        r = RRFResults(["a", "b"], {"a": 0.5, "b": 0.3})
        self.assertIn("n=2", repr(r))


class TestReciprocalRankFusion(unittest.TestCase):
    def test_basic_fusion_two_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse([
            ("vector", ["a", "b", "c"]),
            ("keyword", ["b", "d", "a"]),
        ])
        # 'b' should rank highest (rank 2 in both lists)
        self.assertEqual(result.items[0], "b")
        self.assertIn("a", result.items)
        self.assertIn("c", result.items)
        self.assertIn("d", result.items)

    def test_basic_fusion_three_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse([
            ("vector", ["a", "b", "c"]),
            ("keyword", ["b", "d", "a"]),
            ("graph", ["c", "e", "b"]),
        ])
        # 'b' appears in all 3 lists, should rank highest
        self.assertEqual(result.items[0], "b")

    def test_min_score_filter(self):
        rrf = ReciprocalRankFusion(k=60, min_score=0.02)
        result = rrf.fuse([
            ("vector", list(range(1, 101))),  # 100 items
            ("keyword", list(range(50, 150))),
        ])
        # Items with very low combined scores should be filtered
        self.assertTrue(len(result) < 200)

    def test_scores_descending(self):
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse([
            ["a", "b", "c"],
            ["c", "b", "a"],
        ])
        scores = [result.scores[item] for item in result.items]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_channel_tracking(self):
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse([
            ("vector", ["a", "b"]),
            ("keyword", ["b", "c"]),
        ])
        # Item 'b' appears in both channels
        self.assertEqual(len(result.items), 3)

    def test_empty_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse([])
        self.assertEqual(len(result), 0)

    def test_single_list(self):
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse([("only", ["x", "y", "z"])])
        self.assertEqual(result.items, ["x", "y", "z"])

    def test_k_parameter_effect(self):
        # Small k amplifies rank differences
        rrf_small = ReciprocalRankFusion(k=1)
        rrf_large = ReciprocalRankFusion(k=1000)
        r1 = rrf_small.fuse([["a", "b", "c"], ["c", "b", "a"]])
        r2 = rrf_large.fuse([["a", "b", "c"], ["c", "b", "a"]])
        # With small k, the top-ranked items should get much higher scores
        self.assertNotEqual(r1.scores["a"], r2.scores["a"])


class TestConvenienceFunctions(unittest.TestCase):
    def test_fuse_results(self):
        result = fuse_results(["a", "b"], ["b", "c"])
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "b")  # 'b' in both lists

    def test_fuse_results_with_scores(self):
        result = fuse_results_with_scores(["a", "b"], ["b", "c"])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)  # (item, score)
        # First item should be 'b' with highest score
        self.assertEqual(result[0][0], "b")


if __name__ == "__main__":
    unittest.main()
