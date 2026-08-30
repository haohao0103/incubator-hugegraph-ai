"""Tests for KgMetricAuthority (authoritative metric selection for NL2SQL)."""

import unittest
from typing import Any, Dict, List, Optional

from hugegraph_llm.operators.graph_op.kg_metric_authority import (
    KgMetricAuthority,
    authority_score,
    SOURCE_RANK,
)


def _m(name: str, **kw) -> Dict[str, Any]:
    d = {"name": name}
    d.update(kw)
    return d


class TestAuthorityScore(unittest.TestCase):
    def test_authoritative_dominates(self):
        s_auth, auth = authority_score(_m("a", authoritative="true", priority=0))
        s_plain, _ = authority_score(_m("b", priority=999))
        self.assertTrue(auth)
        self.assertGreater(s_auth, s_plain)

    def test_priority_adds(self):
        base, _ = authority_score(_m("a"))
        with_prio, _ = authority_score(_m("a", priority=7))
        self.assertEqual(with_prio - base, 7)

    def test_source_bonus(self):
        off, _ = authority_score(_m("a", source="official_dw"))
        ud, _ = authority_score(_m("a", source="user_defined"))
        self.assertEqual(off - ud, SOURCE_RANK["official_dw"] - SOURCE_RANK["user_defined"])

    def test_missing_source_default(self):
        s, _ = authority_score(_m("a"))
        self.assertGreaterEqual(s, 5)  # DEFAULT_SOURCE_BONUS

    def test_invalid_priority_treated_zero(self):
        s, _ = authority_score(_m("a", priority="not-a-number"))
        base, _ = authority_score(_m("a"))
        self.assertEqual(s, base)

    def test_is_true_variants(self):
        for v in ("true", "1", "yes", "True"):
            self.assertTrue(authority_score(_m("a", authoritative=v))[1])
        for v in ("false", "0", "no", "", None):
            self.assertFalse(authority_score(_m("a", authoritative=v))[1])


class TestSelect(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(KgMetricAuthority.select([]))  # type: ignore[arg-type]

    def test_single_returned(self):
        m = _m("only")
        self.assertIs(KgMetricAuthority.select([m]), m)

    def test_authoritative_wins(self):
        plain = _m("plain", priority=50)
        auth = _m("auth", authoritative="true", priority=0)
        self.assertEqual(KgMetricAuthority.select([plain, auth])["name"], "auth")

    def test_source_ranking_when_no_authoritative(self):
        off = _m("off", source="official_dw")
        ud = _m("ud", source="user_defined")
        self.assertEqual(KgMetricAuthority.select([ud, off])["name"], "off")

    def test_tie_breaks_on_smaller_name(self):
        # equal score (both non-auth, priority 0, default source)
        a = _m("alpha")
        b = _m("beta")
        self.assertEqual(KgMetricAuthority.select([b, a])["name"], "alpha")


class TestResolveByName(unittest.TestCase):
    def setUp(self):
        metrics = [
            _m("gmv", authoritative="false", source="user_defined", priority=1),
            _m("gmv", authoritative="true", source="official_dw", priority=2),
            _m("gmv", authoritative="false", source="temp", priority=9),
        ]
        self.auth = KgMetricAuthority(graph_data={"vertices": {"Metric": metrics}})

    def test_picks_authoritative_among_same_name(self):
        chosen = self.auth.resolve_by_name("gmv")
        self.assertEqual(chosen["source"], "official_dw")
        self.assertTrue(chosen.get("authoritative"))

    def test_no_match(self):
        self.assertIsNone(self.auth.resolve_by_name("ghost"))

    def test_ranked_sorted(self):
        ranked = self.auth.ranked("gmv")
        scores = [authority_score(m)[0] for m in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # top is the authoritative one
        self.assertTrue(ranked[0].get("authoritative"))


class TestInitSources(unittest.TestCase):
    def test_init_from_graph_data(self):
        a = KgMetricAuthority(graph_data={"vertices": {"Metric": [_m("x")]}})
        self.assertEqual(len(a._metrics), 1)

    def test_init_from_client(self):
        class _FakeClient:
            def gremlin(self):
                return self

            def exec(self, q):
                if "hasLabel('Metric')" in q:
                    return {"data": [{"id": "1:m1", "label": "Metric", "name": "m1",
                                      "authoritative": "true", "priority": "5",
                                      "source": "official_dw"}]}
                return {"data": []}

        a = KgMetricAuthority(client=_FakeClient())
        self.assertEqual(len(a._metrics), 1)
        self.assertEqual(a.resolve_by_name("m1")["name"], "m1")

    def test_init_neither_empty(self):
        a = KgMetricAuthority()
        self.assertEqual(a._metrics, [])
        self.assertIsNone(a.resolve_by_name("x"))


if __name__ == "__main__":
    unittest.main()
