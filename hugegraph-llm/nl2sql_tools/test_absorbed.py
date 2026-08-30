"""Smoke tests for absorbed capabilities from the parallel nl2sql-demo branch:
jargon map, RRF/score fusion, metric authority, SQL voter/validator.

Run: PYTHONPATH=incubator-hugegraph-ai/hugegraph-llm/src python scripts/test_absorbed.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from p2_corpus import build_warehouse_schema  # noqa: E402
from p2_embedder import make_embedder  # noqa: E402

from hugegraph_llm.nl2sql.fusion import rrf_fuse, score_fuse  # noqa: E402
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from hugegraph_llm.nl2sql.metric_authority import authority_score, resolve_metric  # noqa: E402
from hugegraph_llm.nl2sql.sql_ops import SqlValidator, SqlVoter  # noqa: E402
from hugegraph_llm.nl2sql.synonym_dict import JargonMap  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and cond
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


def main():
    schema = build_warehouse_schema()

    # ---- 1. jargon map ----
    jm = JargonMap()
    check("jargon expand 优惠券/活动", set(jm.expand("优惠券相关的活动")) >= {"coupon", "campaign"},
          str(jm.expand("优惠券相关的活动")))
    check("jargon longest match 完单量", jm.expand("完单量") == ["completed_order"],
          str(jm.expand("完单量")))
    check("jargon latin case-insensitive", "gmv" in jm.expand("GTV 是多少"), str(jm.expand("GTV")))

    # ---- 2. RRF / score fusion ----
    class It:
        def __init__(self, nid, score):
            self.node_id, self.score = nid, score

    r1 = [It("a", 3.0), It("b", 2.0)]
    r2 = [It("b", 3.0), It("c", 2.0)]
    fused = rrf_fuse([r1, r2], key_fn=lambda i: i.node_id)
    check("rrf top1 is b (rank1 in both)", fused[0][0].node_id == "b",
          str([(x.node_id, s) for x, s in fused]))
    sf = score_fuse([r1, r2], key_fn=lambda i: i.node_id, score_fn=lambda i: i.score)
    check("score-fuse b=5 > a=3", sf[0][0].node_id == "b",
          str([(x.node_id, s) for x, s in sf]))

    # ---- 3. metric authority ----
    low = authority_score({"priority": 1, "source": "query_log"})
    gov = authority_score({"authoritative": True, "source": "governance"})
    check("authoritative dominates", gov[0] > low[0] and gov[1], f"{low} vs {gov}")
    win = resolve_metric([("m_a", {"priority": 2}), ("m_b", {"priority": 1})])
    check("resolve picks higher priority", win[0] == "m_a", str(win))
    win2 = resolve_metric([("m_a", {"priority": 9}), ("m_b", {"authoritative": True})])
    check("resolve picks authoritative over high priority", win2[0] == "m_b", str(win2))

    # ---- 4. SQL validator / voter ----
    val = SqlValidator(schema)
    rep = val.validate("SELECT gmv FROM dw.orders JOIN dw.payments ON dw.orders.order_id = dw.payments.order_id")
    check("validator accepts real columns", rep["valid"], str(rep))
    bad = val.validate("SELECT nonexistent_col FROM nosuch_table")
    check("validator rejects unknown table/col", not bad["valid"],
          f"tables={bad['unknown_tables']} cols={bad['unknown_columns']}")
    voter = SqlVoter(schema)
    ranked = voter.vote(
        ["SELECT gmv FROM dw.orders",
         "SELECT junk FROM nope",
         "SELECT gmv, COUNT(*) FROM dw.orders GROUP BY gmv"],
        linked_ids=["table:dw.orders"], metric_agg="count",
    )
    check("voter ranks valid first", ranked[0][1] > ranked[1][1],
          str([(s[:40], round(sc, 1)) for s, sc, _ in ranked]))

    # ---- 5. linker integration: jargon + fusion ----
    embedder = make_embedder()
    lk = SchemaLinker(schema, embedder=embedder, top_k_vector=3, vector_weight=0.5,
                      fusion="rrf")
    items = lk.link("GMV 和支付总额是多少", top_k=10)
    ids = [i.node_id for i in items]
    check("jargon+rrf: gmv surfaced", any("gmv" in nid for nid in ids),
          str([x.replace("column:dw.", "").replace("table:dw.", "T:") for x in ids[:6]]))
    items2 = lk.link("客户数", top_k=10)
    check("rrf link non-empty on semantic q", len(items2) > 0, str(len(items2)))

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
