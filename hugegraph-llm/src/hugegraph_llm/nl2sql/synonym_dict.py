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

"""Business-jargon -> canonical-identifier dictionary (synonym data source).

Curated for the logistics / ride-hailing domain (originally from
``kg_jargon_map`` on the parallel NL2SQL demo branch): warehouse users say
"拉新" / "完单" / "GTV" while the metadata graph stores ``new_user`` /
``completed_order`` / ``gmv``. A literal linker pass never connects the two.

The dictionary is fed into :class:`SchemaLinker` as extra match texts, so a
question containing slang reaches the canonical node — no LLM call, no graph
change. It is the data payload behind the P0-2 synonym layer.
"""

from typing import Dict, List, Optional

# slang (as users say it) -> canonical identifier in the metadata graph.
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


class JargonMap:
    """Curated slang -> canonical map, with longest-match lookup."""

    def __init__(self, extra: Optional[Dict[str, str]] = None) -> None:
        self._map: Dict[str, str] = dict(DEFAULT_JARGON)
        if extra:
            self._map.update({k: v for k, v in extra.items() if k and v})
        self._terms = sorted(self._map, key=len, reverse=True)

    def lookup(self, slang: str) -> Optional[str]:
        """Canonical identifier for an exact slang term, or ``None``."""
        return self._map.get(slang)

    def match(self, text: str) -> List[tuple]:
        """All ``(alias, canonical)`` pairs whose alias appears in ``text``.

        Longest match first, case-insensitive for Latin terms.
        """
        lower = text.lower()
        hits: List[tuple] = []
        for term in self._terms:
            t = term.lower()
            if t and t in lower:
                hits.append((term, self._map[term]))
        return hits

    def expand(self, text: str) -> List[str]:
        """Canonical identifiers for every slang term found in ``text``.

        Longest match first so "完单量" wins over "完单". Case-insensitive for
        Latin terms (GTV == gtv).
        """
        hits: List[str] = []
        lower = text.lower()
        rest = lower
        for term in self._terms:
            t = term.lower()
            if t and t in rest:
                canon = self._map[term]
                if canon not in hits:
                    hits.append(canon)
                rest = rest.replace(t, " ")
        return hits
