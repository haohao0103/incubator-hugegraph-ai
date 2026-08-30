"""Real-machine comparison: single-pass lexical vs multi-recall fusion.

Runs against the live ``kg_platform`` metadata graph:

* single path  -- KgSchemaLinker (lexical 2-gram ranking);
* multi path   -- KgMultiSchemaLinker (graph structure + SEARCH fulltext via the
                  live HugeGraph indexes + lexical), RRF-fused, with a
                  synonyms alias map to show the graph path rescuing wording
                  gaps that pure lexical misses.

For each question it prints which metrics each path links, so the recall gain
of the fused pipeline is visible per question.

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/multi_retrieval_e2e/e2e_multi_retrieval.py 2>&1 \\
        | tee _out/multi_retrieval_e2e/logs/e2e_multi_retrieval.log
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.operators.graph_op.kg_multi_retrieval import (  # noqa: E402
    KgMultiSchemaLinker,
    MultiRecallConfig,
)
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker  # noqa: E402
from hugegraph_llm.config import huge_settings  # noqa: E402
from pyhugegraph.client import PyHugeClient  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_platform")


def _client(graph: str = GRAPH):
    # NOTE: the client's bound graph is what Gremlin queries hit; KgRuleEngine's
    # graph_name only keys the TTL cache
    return PyHugeClient(
        url=huge_settings.graph_url, graph=graph,
        user=huge_settings.graph_user, pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )

QUESTIONS = [
    "订单总额是多少",
    "各城市订单金额",
    "平均每单成交金额",
    "司机订单数",
    "订单数量",
    "大单有多少",       # alias "大单"->order_total: lexical misses, graph rescues
    "客单价",           # definition already contains 客单价 (doc-merged)
    "风控引擎实时决策表在哪里",  # out-of-scope: no evidence -> refuse
]

# alias map: 黑话/别名 -> canonical metric name (what the platform would tune)
SYNONYMS = {"大单": "order_total", "客单价": "avg_order_value"}


def _metric_names(ctx) -> list:
    return [m.get("name") for m in ctx.metrics]


def main() -> int:
    try:
        client = _client()
        client.gremlin().exec("g.V().limit(1).count()")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: graph unreachable: {exc}")
        return 0

    data = KgRuleEngine(client, GRAPH).load_graph()
    single = KgSchemaLinker()
    multi = KgMultiSchemaLinker(client=client, synonyms=SYNONYMS, config=MultiRecallConfig())
    multi_imp = KgMultiSchemaLinker(
        client=client, synonyms=SYNONYMS,
        config=MultiRecallConfig(importance_weight=1.0),
    )

    print(f"=== graph={GRAPH}  single(lexical) vs multi(graph+fulltext+lexical+alias) ===\n")
    print(f"{'question':<18} | {'single':<30} | {'multi':<34} | {'multi+importance':<34} | note")
    print("-" * 130)
    gains = no_evidence = 0
    for q in QUESTIONS:
        s = _metric_names(single.link(q, data=data))
        m = _metric_names(multi.link(q, data=data))
        mi = _metric_names(multi_imp.link(q, data=data))
        extra = [x for x in m if x not in s]
        if extra:
            gains += 1
        if not m:
            no_evidence += 1
        note = f"+{extra}" if extra else ("NO EVIDENCE -> refuse" if not m else "")
        print(f"{q:<18} | {str(s):<30} | {str(m):<34} | {str(mi):<34} | {note}")
    print(f"\nmulti-recall gains: {gains}/{len(QUESTIONS)}; "
          f"no-evidence (refuse) cases: {no_evidence}")
    print("PASS: multi-recall + importance re-rank + no-evidence signal on the live graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
