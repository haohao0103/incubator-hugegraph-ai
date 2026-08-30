"""Metric authority resolution for NL2SQL over the metadata KG.

When schema linking returns several candidate metrics for one question, or the
same metric name carries conflicting definitions across domains (KgRuleEngine
rule D1), NL2SQL must pick *one* authoritative definition -- otherwise the LLM
may emit a SQL that computes the number with the wrong formula.

Authority is a per-Metric attribute (added to the vertex, seeded by the data
governance team), ranked as:

* ``authoritative`` (boolean) -- an explicit "this is the canonical definition"
  flag; dominates every other signal (it is what governance stamps);
* ``priority`` (int) -- a numeric tie-breaker when several are authoritative
  or none is;
* ``source`` (string) -- the originating system, ranked best->worst
  (official_dw > governance > bi/mart > temp > user_defined).

The selector is a pure, deterministic function over the :data:`GraphData`
shape (no LLM call), so it is testable and auditable.

Typical use::

    auth = KgMetricAuthority(graph_data)
    chosen = auth.resolve_by_name("gmv")        # disambiguate a D1 conflict
    chosen = auth.select(linker.metrics)         # pick among linked metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.operators.graph_op.kg_rule_engine import (
    KgRuleEngine,
    GraphData,
)

# Source systems ranked best -> worst. A better source earns a higher bonus so
# that, all else equal, the official warehouse definition wins over a temp BI
# calc or a user-defined ad-hoc metric.
SOURCE_RANK: Dict[str, int] = {
    "official_dw": 50,
    "governance": 40,
    "bi": 30,
    "mart": 30,
    "temp": 10,
    "user_defined": 0,
}
DEFAULT_SOURCE_BONUS = 5  # missing / unknown source still allowed, but low
AUTHORITATIVE_BONUS = 1000  # dominates every priority/source combination


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def authority_score(metric: Dict[str, Any]) -> Tuple[int, bool]:
    """Return (score, is_authoritative) for a metric dict.

    Higher score wins. ``authoritative`` dominates; then priority; then the
    source-system bonus.
    """
    is_auth = _is_true(metric.get("authoritative"))
    priority = _as_int(metric.get("priority", 0))
    source = str(metric.get("source") or "").strip().lower()
    bonus = SOURCE_RANK.get(source, DEFAULT_SOURCE_BONUS)
    score = (AUTHORITATIVE_BONUS if is_auth else 0) + priority + bonus
    return score, is_auth


class KgMetricAuthority:
    """Select the authoritative metric among candidates / name conflicts."""

    def __init__(
        self,
        graph_data: Optional[GraphData] = None,
        client: Optional[Any] = None,
        graph_name: Optional[str] = None,
    ) -> None:
        self._graph_name = graph_name
        if graph_data is not None:
            self._metrics = list(graph_data.get("vertices", {}).get("Metric", []))
        elif client is not None:
            self._metrics = list(
                KgRuleEngine(client, graph_name).load_graph()
                .get("vertices", {}).get("Metric", [])
            )
        else:
            self._metrics = []

    # -- public API ----------------------------------------------------------

    @staticmethod
    def select(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the most authoritative metric, or None when there are none.

        Ties on score are broken by lexicographically smaller ``name`` (stable,
        deterministic). A single candidate is returned as-is.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        best: Optional[Dict[str, Any]] = None
        best_score = -1
        for m in sorted(candidates, key=lambda x: str(x.get("name") or "")):
            score, _ = authority_score(m)
            if score > best_score:
                best_score = score
                best = m
        return best

    def resolve_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Disambiguate a same-name conflict (KgRuleEngine D1).

        Returns the authoritative metric among all vertices sharing ``name``.
        """
        matches = [m for m in self._metrics if m.get("name") == name]
        return self.select(matches)

    def ranked(self, name: str) -> List[Dict[str, Any]]:
        """All metrics sharing ``name``, sorted most->least authoritative."""
        matches = [m for m in self._metrics if m.get("name") == name]
        return sorted(
            matches,
            key=lambda m: (-authority_score(m)[0], str(m.get("name") or "")),
        )
