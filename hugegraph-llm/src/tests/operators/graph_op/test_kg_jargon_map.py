"""Tests for KgJargonMap (business-slang -> canonical for NL2SQL)."""

import unittest
from typing import Any, Dict

from hugegraph_llm.operators.graph_op.kg_jargon_map import KgJargonMap, DEFAULT_JARGON
from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker


def _graph_with_new_user() -> Dict[str, Any]:
    return {
        "vertices": {
            "Table": [{"name": "user"}],
            "Field": [{"name": "user.new_user_cnt", "comment": "新用户数"}],
            "Metric": [
                {"name": "new_user", "definition": "count of first-time registrations", "formula": "COUNT(user.new_user_cnt)"}
            ],
        },
        "edges": {
            "hasColumn": [("user", "user.new_user_cnt")],
            "computedFromField": [("new_user", "user.new_user_cnt")],
        },
    }


class TestLookup(unittest.TestCase):
    def setUp(self):
        self.j = KgJargonMap()

    def test_exact_slant(self):
        self.assertEqual(self.j.lookup("拉新"), "new_user")
        self.assertEqual(self.j.lookup("GTV"), "gmv")

    def test_missing(self):
        self.assertIsNone(self.j.lookup("不存在的黑话"))

    def test_empty(self):
        self.assertIsNone(self.j.lookup(""))


class TestMatch(unittest.TestCase):
    def setUp(self):
        self.j = KgJargonMap()

    def test_substring_hits(self):
        hits = self.j.match("拉新用户数怎么看")
        self.assertIn(("拉新", "new_user"), hits)

    def test_longer_slant_wins_order(self):
        # both 完单 and 完单量 are substrings of 完单量; the longer must come first
        hits = self.j.match("昨日完单量")
        self.assertEqual(hits[0], ("完单量", "completed_order"))

    def test_empty_text(self):
        self.assertEqual(self.j.match(""), [])


class TestExpandTerms(unittest.TestCase):
    def setUp(self):
        self.j = KgJargonMap()

    def test_appends_canonical(self):
        out = self.j.expand_terms(["拉新", "用户"])
        self.assertIn("new_user", out)

    def test_dedup_when_canonical_present(self):
        out = self.j.expand_terms(["new_user", "拉新"])
        # "拉新" -> "new_user" already in the list, must not be duplicated
        self.assertEqual(out, ["new_user", "拉新"])

    def test_non_slant_unchanged(self):
        out = self.j.expand_terms(["订单", "金额"])
        self.assertEqual(out, ["订单", "金额"])


class TestInitAndExport(unittest.TestCase):
    def test_default_non_empty(self):
        self.assertGreater(len(KgJargonMap()._map), 10)

    def test_extra_overrides_default(self):
        j = KgJargonMap(extra={"拉新": "acq_user"})
        self.assertEqual(j.lookup("拉新"), "acq_user")
        # other defaults still present
        self.assertEqual(j.lookup("客诉"), "complaint")

    def test_extra_skips_empty(self):
        j = KgJargonMap(extra={"": "x", "y": ""})
        self.assertNotIn("", j._map)
        self.assertNotIn("y", j._map)

    def test_to_synonym_map_is_copy(self):
        j = KgJargonMap()
        sm = j.to_synonym_map()
        sm["拉新"] = "mutated"
        self.assertEqual(j.lookup("拉新"), "new_user")  # original untouched

    def test_size(self):
        self.assertEqual(KgJargonMap().size, len(DEFAULT_JARGON))


class TestLinkerIntegration(unittest.TestCase):
    """Prove jargon actually improves schema linking (no client needed)."""

    def _link(self, question: str, with_jargon: bool):
        synonyms = KgJargonMap().to_synonym_map() if with_jargon else None
        linker = KgSchemaLinker(synonyms=synonyms)
        return linker.link(question, _graph_with_new_user())

    def test_jargon_links_metric(self):
        ctx = self._link("拉新用户数", with_jargon=True)
        names = [m["name"] for m in ctx.metrics]
        self.assertIn("new_user", names)

    def test_without_jargon_metric_misses(self):
        # without the slang bridge, "new_user" is never a term, so the metric
        # (whose name is the canonical English) is not linked.
        ctx = self._link("拉新用户数", with_jargon=False)
        names = [m["name"] for m in ctx.metrics]
        self.assertNotIn("new_user", names)

    def test_jargon_links_field_too(self):
        ctx = self._link("拉新用户数", with_jargon=True)
        fnames = [f["name"] for f in ctx.fields]
        self.assertIn("user.new_user_cnt", fnames)


if __name__ == "__main__":
    unittest.main()
