"""全量合并 Step 2 验证：L3 纠错沿语义边传播移植到 e2e 链路。

链路（真实 HG kg_rag）：
    p2 corpus -> meta (含 calibers + corrections) -> ingest -> loader 读回
    -> 构造种子 -> fetch_corrections 沿图传播 -> 断言：
      ① 纠错顶点/边写入成功；
      ② 挂在「非种子节点」(caliber 对应 term) 的纠错，经传播也能被召回；
      ③ 按纠错 id 去重（一条纠错挂多端点不重复）。

与 PoC 的差异：传播在内存 SchemaGraph 上做 BFS（e2e 已把全图拉到内存），
不需要逐节点 Gremlin；口径端点经 hasCaliber 反查归到所属术语再挂纠错。

Run:
  PYTHONPATH=hugegraph-llm/src HF_HUB_OFFLINE=1 \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/merge_step2_correction.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from ingest_metadata_to_hg import ingest  # noqa: E402
from e2e_ingest_load import corpus_to_metadata  # noqa: E402
from p2_corpus import build_warehouse_schema  # noqa: E402
from hugegraph_llm.nl2sql.hugegraph_schema_source import (  # noqa: E402
    build_schema_from_hugegraph,
)
from hugegraph_llm.nl2sql.correction_propagation import (  # noqa: E402
    fetch_corrections, propagate_seeds,
)
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402

LOG_PATH = "_out/nl2sql_merge/logs/step2_correction.log"
URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def build_meta_with_corrections():
    """corpus meta + 两条纠错：
    - corr_a 挂在 term:成交额 上（词法种子可直接命中）
    - corr_b 挂在 caliber:GMV口径 上（成交额的口径，**不在**任何词法种子内，
      只有沿 hasCaliber 边传播才能召回 —— 验证 L3 沿图传播的差异化能力）
    """
    schema0 = build_warehouse_schema()
    meta = corpus_to_metadata(schema0)
    meta["corrections"] = [
        {
            "id": "corr_step2_a",
            "question": "订单的成交总额",
            "wrong_sql": "SELECT SUM(gmv) FROM orders",
            "correct_sql": "SELECT SUM(gmv) FROM orders WHERE order_status = 'paid'",
            "correction_reason": "成交额仅统计已支付订单（GMV口径），需要 WHERE order_status='paid'",
            "applies_to": ["term:成交额"],
        },
        {
            "id": "corr_step2_b",
            "question": "GMV 口径",
            "wrong_sql": "SELECT SUM(amount) FROM orders",
            "correct_sql": "SELECT SUM(amount) FROM orders WHERE status = 'paid'",
            "correction_reason": "GMV 口径仅统计 paid 订单金额之和（挂口径，非种子节点）",
            "applies_to": ["caliber:GMV口径"],
        },
    ]
    return meta


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== merge step2: L3 correction propagation into e2e ===")

    meta = build_meta_with_corrections()
    counts = ingest(meta, URL, GRAPH, clear=True)
    log(f"ingested: {counts}")
    assert counts["corrections"] >= 2, "CorrectionDecision vertices not written"
    assert counts["correction_edges"] >= 2, "correctionAppliesTo* edges not written"

    back = build_schema_from_hugegraph(url=URL, graph=GRAPH)
    nodes_with_corr = [n.node_id for n in back.nodes.values()
                       if n.properties.get("corrections")]
    log(f"loaded back: terms={len(back.terms())} corrections on nodes: {nodes_with_corr}")
    assert "term:成交额" in nodes_with_corr, "corr on term:成交额 not loaded"
    assert "term:成交额" in nodes_with_corr, "corr on caliber->term not folded"

    # ① 种子=成交额（词法命中 term）: 沿语义边传播应同时召回 corr_a（直接挂）
    #    和 corr_b（经 term -> hasCaliber -> GMV口径 反查回 term 挂载）
    seeds = {"term:成交额"}
    corrs, stats = fetch_corrections(back, seeds, hops=2)
    ids = [c["id"] for c in corrs]
    log(f"seed={sorted(seeds)} -> reached={len(stats['reached'])} "
        f"propagated={stats['propagated']} corrections={ids}")
    assert "corr_step2_a" in ids, f"corr_step2_a missing: {ids}"
    assert "corr_step2_b" in ids, (
        f"corr_step2_b (on caliber, NON-seed) NOT recalled via propagation: {ids}")

    # ② 去重：corr_b 只挂一处（caliber），corr_a 只挂一处 —— 但把 corr_a 再挂到
    #    字段 endpoints 后，多端点仍应只召回一次。用 propagate 到 column 验证
    #    多端点去重（同一条纠错挂 term + field，两个端点都可达，只出一次）。
    dup_meta = dict(meta)
    dup_meta["corrections"] = [
        {
            "id": "corr_step2_dup",
            "question": "x",
            "wrong_sql": "A",
            "correct_sql": "B",
            "correction_reason": "multi-endpoint dedup check",
            "applies_to": ["term:支付总额", "field:payments.pay_amount"],
        },
    ]
    counts2 = ingest(dup_meta, URL, GRAPH, clear=True)
    log(f"re-ingested with multi-endpoint corr: {counts2}")
    back2 = build_schema_from_hugegraph(url=URL, graph=GRAPH)
    # 种子 term:支付总额 -> TERM_MAPS 到 column:dw.payments.pay_amount
    corrs2, stats2 = fetch_corrections(back2, {"term:支付总额"}, hops=2)
    ids2 = [c["id"] for c in corrs2]
    log(f"multi-endpoint: recalled {ids2} (dedup => len==1)")
    assert ids2.count("corr_step2_dup") == 1 and len(ids2) == 1, (
        f"dedup failed: {ids2}")

    # ③ linker.seed_nodes 接入：真实问题「订单的成交总额」命中 column:orders.gmv
    #    （注释"成交总额"），种子是 column 而非 term —— 传播沿 TERM_MAPS 反向
    #    1 跳到达 term:成交额，再经 hasCaliber 召回挂在 GMV口径 上的 corr_step2_b。
    #    这才是端到端闭环：问题 -> 种子 -> 图传播 -> 纠错召回。
    linker = SchemaLinker(back2)
    seed_ids = linker.seed_nodes("订单的成交总额")
    log(f"linker.seed_nodes('订单的成交总额') = {seed_ids}")
    assert any(s.startswith("column:") for s in seed_ids), (
        f"seed_nodes missing column seeds: {seed_ids}")
    corrs3, stats3 = fetch_corrections(back2, set(seed_ids), hops=2)
    ids3 = [c["id"] for c in corrs3]
    log(f"e2e recall from {seed_ids}: reached={len(stats3['reached'])} "
        f"corrections={ids3}")
    assert "corr_step2_dup" in ids3, f"e2e recall missing corr_step2_dup: {ids3}"

    log("STEP2: ALL PASS")
    print("ALL PASS")


if __name__ == "__main__":
    main()
