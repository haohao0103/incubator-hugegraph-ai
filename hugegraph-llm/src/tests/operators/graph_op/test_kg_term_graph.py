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

"""Tests for the independent term/synonym layer (KgTermGraph)."""

import os
import tempfile
import unittest

from hugegraph_llm.operators.graph_op.kg_jargon_map import DEFAULT_JARGON
from hugegraph_llm.operators.graph_op.kg_term_graph import KgTermGraph, TermNode


class TestKgTermGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = KgTermGraph.from_jargon_map(
            {
                "客单价": "arpu",
                "平均每单成交金额": "arpu",
                "完单量": "completed_order",
                "完单": "completed_order",
                "单量": "order_count",
                "司机": "driver",
            }
        )

    def test_lookup_exact_alias(self):
        self.assertEqual(self.graph.lookup("客单价"), "arpu")
        self.assertEqual(self.graph.lookup("完单量"), "completed_order")

    def test_lookup_canonical_resolves_to_itself(self):
        self.assertEqual(self.graph.lookup("arpu"), "arpu")
        self.assertEqual(self.graph.lookup("order_count"), "order_count")

    def test_lookup_unknown_returns_none(self):
        self.assertIsNone(self.graph.lookup("不存在的词"))

    def test_canonical_aliases_longest_first(self):
        aliases = self.graph.canonical_aliases("arpu")
        self.assertIn("客单价", aliases)
        self.assertIn("平均每单成交金额", aliases)
        # longest alias first for greedy matching
        self.assertEqual(aliases[0], "平均每单成交金额")

    def test_match_longest_alias_wins(self):
        hits = self.graph.match("本月完单量是多少")
        pairs = dict(hits)
        self.assertEqual(pairs.get("完单量"), "completed_order")
        self.assertNotIn("完单", pairs)  # 完单量 matched first, same canonical

    def test_match_no_hit(self):
        self.assertEqual(self.graph.match("完全不相关"), [])

    def test_expand_terms_appends_canonical(self):
        out = self.graph.expand_terms(["客单", "单量"])
        self.assertIn("order_count", out)
        # 客单 is not an exact alias (客单价 is) -> no arpu appended
        self.assertNotIn("arpu", out)

    def test_expand_question_returns_canonicals(self):
        out = self.graph.expand_question("客单价是多少")
        self.assertIn("arpu", out)
        self.assertNotIn("completed_order", out)

    def test_add_term_and_alias(self):
        g = KgTermGraph()
        g.add_term(TermNode(canonical="new_user", aliases=["拉新", "新客"]))
        g.add_alias("new_user", "获客")
        self.assertEqual(g.lookup("拉新"), "new_user")
        self.assertEqual(g.lookup("获客"), "new_user")
        self.assertEqual(g.num_terms, 1)
        self.assertEqual(g.num_aliases, 3)

    def test_constructor_accepts_term_nodes(self):
        g = KgTermGraph(terms=[TermNode(canonical="arpu", aliases=["客单价"])])
        self.assertEqual(g.lookup("客单价"), "arpu")

    def test_from_jargon_map_skips_empty_and_self_aliases(self):
        g = KgTermGraph.from_jargon_map({"": "x", "y": "", "arpu": "arpu", "客单价": "arpu"})
        # only 客单价->arpu survives as a real alias edge
        self.assertEqual(g.num_terms, 1)
        self.assertEqual(g.num_aliases, 1)

    def test_add_term_duplicate_alias_not_repeated(self):
        g = KgTermGraph()
        g.add_term(TermNode(canonical="x", aliases=["a"]))
        g.add_term(TermNode(canonical="x", aliases=["a"]))
        self.assertEqual(g.num_aliases, 1)

    def test_add_alias_ignores_invalid_and_duplicate(self):
        g = KgTermGraph()
        g.add_alias("", "x")
        g.add_alias("x", "x")
        g.add_alias("x", "y")
        g.add_alias("x", "y")
        self.assertEqual(g.num_aliases, 1)

    def test_lookup_empty_returns_none(self):
        self.assertIsNone(self.graph.lookup(""))

    def test_match_empty_returns_empty(self):
        self.assertEqual(self.graph.match(""), [])

    def test_add_term_ignores_empty(self):
        g = KgTermGraph()
        g.add_term(TermNode(canonical="", aliases=["x"]))
        g.add_term(TermNode(canonical="ok", aliases=[""]))
        self.assertEqual(g.num_terms, 1)

    def test_default_has_domain_vocabulary(self):
        g = KgTermGraph.default()
        self.assertGreater(g.num_terms, 5)
        self.assertEqual(g.lookup("客单价"), "arpu")  # DEFAULT_JARGON parity

    def test_to_jargon_map_roundtrip(self):
        flat = self.graph.to_jargon_map()
        self.assertEqual(flat["客单价"], "arpu")
        rebuilt = KgTermGraph.from_jargon_map(flat)
        self.assertEqual(rebuilt.lookup("完单量"), "completed_order")

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "terms.json")
            self.graph.save(path)
            loaded = KgTermGraph.load(path)
        self.assertEqual(loaded.lookup("客单价"), "arpu")
        self.assertEqual(loaded.lookup("司机"), "driver")

    def test_save_bare_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)  # relative path -> empty dirname branch
            try:
                self.graph.save("terms.json")
                self.assertTrue(os.path.exists("terms.json"))
            finally:
                os.chdir(cwd)

    def test_default_jargon_parity(self):
        # the term graph must cover every slang in the curated jargon map
        flat = KgTermGraph.default().to_jargon_map()
        self.assertEqual(set(flat.keys()), set(DEFAULT_JARGON.keys()))


if __name__ == "__main__":
    unittest.main()
