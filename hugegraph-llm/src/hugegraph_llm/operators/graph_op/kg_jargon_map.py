"""Business-jargon map for NL2SQL over the Huolala data warehouse.

Warehouse users (analysts, operations, business partners) speak in slang:
"拉新", "客诉", "完单", "GTV", "单量", "妥投" ... while the metadata graph
stores canonical, usually English, identifiers: ``new_user``, ``complaint``,
``completed_order``, ``gmv``, ``order_count``, ``delivered``. A literal
schema-linking pass never connects "拉新用户数" to the ``new_user`` metric.

This module is the deterministic bridge. It is a curated slang->canonical map
(the logistics/ride-hailing domain vocabulary) plus helpers to feed it into the
existing :class:`KgSchemaLinker` through its synonym mechanism (no linker
changes, no LLM call). The map is the single source of truth that keeps NL2SQL
robust to how people actually phrase questions.

Typical use::

    jargon = KgJargonMap()
    linker = KgSchemaLinker(client, synonyms=jargon.to_synonym_map())
    # "拉新用户数" now links to the new_user metric/field
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# slang (as users say it) -> canonical identifier in the metadata graph.
# Curated for the Huolala logistics/ride-hailing domain; extend freely.
DEFAULT_JARGON: Dict[str, str] = {
    # user / growth
    "拉新": "new_user",
    "新客": "new_user",
    "获客": "acquisition",
    "活跃": "active",
    "留存": "retention",
    "复购": "repurchase",
    "渗透": "penetration",
    "沉睡": "dormant",
    # orders / fulfilment
    "单量": "order_count",
    "完单": "completed_order",
    "完单量": "completed_order",
    "妥投": "delivered",
    "在途": "in_transit",
    "拒收": "rejected",
    "取消单": "cancelled_order",
    "取消": "cancelled",
    "履约": "fulfillment",
    "接单": "accepted_order",
    "成单": "completed_order",
    # money
    "营收": "revenue",
    "收入": "revenue",
    "成交额": "gmv",
    "GTV": "gmv",
    "GMV": "gmv",
    "客单价": "arpu",
    "补贴": "subsidy",
    "优惠": "coupon",
    "优惠券": "coupon",
    "充值": "recharge",
    "提现": "withdraw",
    "账单": "bill",
    # complaint / service
    "客诉": "complaint",
    "投诉": "complaint",
    "差评": "bad_review",
    "赔付": "compensation",
    # roles
    "司机": "driver",
    "货主": "shipper",
    "商户": "merchant",
    "运力": "capacity",
    # marketing
    "营销": "marketing",
    "活动": "campaign",
}


class KgJargonMap:
    """Curated slang -> canonical map for NL2SQL schema linking."""

    def __init__(self, extra: Optional[Dict[str, str]] = None) -> None:
        self._map: Dict[str, str] = dict(DEFAULT_JARGON)
        if extra:
            self._map.update({k: v for k, v in extra.items() if k and v})

    # -- queries -------------------------------------------------------------

    def lookup(self, slang: str) -> Optional[str]:
        """Return the canonical identifier for an exact slang term, or None."""
        if not slang:
            return None
        return self._map.get(slang)

    def match(self, text: str) -> List[Tuple[str, str]]:
        """Return every (slang, canonical) hit found as a substring of ``text``.

        Longer slang entries are tested first so "完单量" wins over "完单" when
        both could match.
        """
        if not text:
            return []
        hits: List[Tuple[str, str]] = []
        for slang in sorted(self._map, key=len, reverse=True):
            if slang in text:
                hits.append((slang, self._map[slang]))
        return hits

    def expand_terms(self, terms: Sequence[str]) -> List[str]:
        """Append the canonical form of any slang term (deduplicated, order-kept)."""
        out: List[str] = list(terms)
        for term in terms:
            canonical = self._map.get(term)
            if canonical and canonical.lower() not in {t.lower() for t in out}:
                out.append(canonical.lower())
        return out

    def to_synonym_map(self) -> Dict[str, str]:
        """Feed into :class:`KgSchemaLinker` (same alias->canonical shape)."""
        return dict(self._map)

    @property
    def size(self) -> int:
        return len(self._map)
