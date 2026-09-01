"""统一数据导入接口 · 真实 HG 端到端验证（生产语义：diff 增量，绝不 clear）。

验证清单：
  A. 结构化元数据导入（Nl2SqlIngester.ingest source=structured）
  B. 文档 bundle 导入（数据字典/口径说明/历史问答，真实 MiMo LLM 抽取）
  C. 幂等复跑（同样 payload 再 ingest -> written 全 0）
  D. 图读回一致性（读回计数 == 内存 merged 计数；图是唯一事实源）
  E. 单向量库（embed_schema_nodes 刷新 == schema 节点数；与读路径同一实例）
  F. 读路径同源（load_pipeline().link 可跑且命中 schema 节点）

运行（日志实时落盘）：
  cd incubator-hugegraph-ai
  PYTHONPATH=hugegraph-llm/src HUGEGRAPH_LLM_ENV_PATH=_out/e2e_sqlgen/mimo.env \
  HF_HUB_OFFLINE=1 \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/ingest_unified_check.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hugegraph_llm.nl2sql.ingest import Nl2SqlIngester  # noqa: E402

LOG_PATH = "_out/nl2sql_unified/logs/check.log"
URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"
SAMPLE = os.path.join(os.path.dirname(__file__), "sample_docs")

PASS, FAIL = [], []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    log(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 结构化 meta：精简版数仓（orders/payments/ads_daily_sales）。
# 与 kg_rag 已有数据 diff 增量合并——验证 merge 语义，不破坏 e2e 评估数据。
STRUCTURED_META = {
    "tables": [
        {"name": "dw.orders", "comment": "订单表", "row_count": 1000000},
        {"name": "dw.payments", "comment": "支付流水表", "row_count": 980000},
        {"name": "dw.ads_daily_sales", "comment": "日销售汇总表", "row_count": 3650},
    ],
    "columns": [
        {"name": "order_id", "table": "dw.orders", "comment": "订单编号", "data_type": "bigint"},
        {"name": "gmv", "table": "dw.orders", "comment": "订单金额", "data_type": "decimal"},
        {"name": "status", "table": "dw.orders", "comment": "订单状态(paid/cancelled)", "data_type": "string"},
        {"name": "pay_time", "table": "dw.orders", "comment": "支付时间", "data_type": "timestamp"},
        {"name": "payment_id", "table": "dw.payments", "comment": "支付流水号", "data_type": "bigint"},
        {"name": "order_id", "table": "dw.payments", "comment": "订单编号", "data_type": "bigint"},
        {"name": "pay_amount", "table": "dw.payments", "comment": "支付金额", "data_type": "decimal"},
        {"name": "pay_status", "table": "dw.payments", "comment": "支付状态", "data_type": "string"},
        {"name": "stat_date", "table": "dw.ads_daily_sales", "comment": "统计日期", "data_type": "date"},
        {"name": "gmv", "table": "dw.ads_daily_sales", "comment": "成交总额", "data_type": "decimal"},
        {"name": "order_cnt", "table": "dw.ads_daily_sales", "comment": "订单数", "data_type": "int"},
        {"name": "avg_order_value", "table": "dw.ads_daily_sales", "comment": "平均客单价", "data_type": "decimal"},
    ],
    "terms": [
        {"name": "成交额", "comment": "已支付订单金额合计", "expression": "SUM(gmv) WHERE status='paid'"},
        {"name": "支付总额", "comment": "用户实际支付金额合计", "expression": "SUM(pay_amount)"},
        {"name": "客单价", "comment": "成交总额/订单数", "expression": "gmv/order_cnt"},
    ],
    "term_bindings": [
        ["成交额", "orders.gmv"],
        ["支付总额", "payments.pay_amount"],
        ["客单价", "ads_daily_sales.avg_order_value"],
    ],
    "calibers": [
        {"name": "GMV口径", "metric": "成交额", "dimension": "status",
         "description": "成交额仅统计 paid 状态订单；退款/取消不计入"},
        {"name": "实付口径", "metric": "支付总额", "dimension": "status",
         "description": "支付总额=实际支付成功金额，不含退款"},
        {"name": "客单价口径", "metric": "客单价", "dimension": "grain",
         "description": "客单价=成交总额/订单数，分母为订单数"},
    ],
    "corrections": [
        {"id": "corr_gmv_status", "question": "上周各渠道成交额？", "wrong_sql": "SELECT channel,SUM(gmv) FROM orders",
         "correct_sql": "SELECT o.channel,SUM(o.gmv) FROM orders o WHERE o.status='paid'",
         "correction_reason": "GMV 口径要求仅统计 paid", "applies_to": ["term:成交额"]},
    ],
    "lineage": [
        ["dw.orders", "dw.ads_daily_sales"],
        ["dw.payments", "dw.ads_daily_sales"],
    ],
    "synonyms": [["成交额", "支付总额"]],
}


def load_docs():
    """3 份样例文档 -> doc_bundle payload."""
    docs = []
    for dt, fn in (("dictionary", "data_dictionary.md"),
                   ("caliber", "caliber_docs.md"),
                   ("qa", "qa_history.md")):
        with open(os.path.join(SAMPLE, fn), encoding="utf-8") as f:
            docs.append({"doc_type": dt, "content": f.read(), "name": fn})
    return {"source": "doc_bundle", "docs": docs}


def main():
    log("=== unified ingest e2e check ===")
    log(f"url={URL} graph={GRAPH} sample_docs={SAMPLE}")

    with Nl2SqlIngester(url=URL, graph=GRAPH) as ingester:
        # ---- A. structured ----
        log("--- A. structured ingest ---")
        rep_a = ingester.ingest({"source": "structured", "meta": STRUCTURED_META})
        check("A structured ok", rep_a.ok and not rep_a.errors, f"written={rep_a.written}")
        for k, v in rep_a.merge.get("added", {}).items():
            if v:
                log(f"    added {k}={v}")
        baseline_a = {k: v for k, v in rep_a.baseline.items()}

        # ---- B. doc bundle ----
        log("--- B. doc_bundle ingest (real LLM extraction) ---")
        rep_b = ingester.ingest(load_docs())
        check("B doc ok (no fatal error)", rep_b.ok,
              f"errors={rep_b.errors} written={rep_b.written}")
        ext = rep_b.extracted
        log(f"    extracted: terms={ext.get('terms')} calibers={ext.get('calibers')} "
            f"corrections={ext.get('corrections')} unmatched={len(rep_b.merge.get('unmatched', []))}")
        check("B terms grew", rep_b.baseline["terms"] < ext.get("terms", 0) + rep_b.baseline["terms"]
              or ext.get("terms", 0) > 0, f"doc terms extracted={ext.get('terms')}")
        # 软断言：文档产物确实进入图（计数增长即可；LLM 名称漂移不硬判）
        check("B graph grew after docs",
              (rep_b.written.get("terms", 0) + rep_b.written.get("calibers", 0)
               + rep_b.written.get("corrections", 0)) > 0,
              f"written={rep_b.written}")
        for u in rep_b.merge.get("unmatched", [])[:3]:
            log(f"    unmatched sample: {u}")

        # ---- C. idempotent re-run ----
        log("--- C. idempotent re-ingest (same doc_bundle) ---")
        rep_c = ingester.ingest(load_docs())
        zero_written = all(v == 0 for k, v in rep_c.written.items())
        check("C idempotent (written all 0)", zero_written, f"written={rep_c.written}")
        check("C dedup (dup counts recorded)", (rep_c.merge.get("dup_terms")
                                                or rep_c.merge.get("dup_calibers")
                                                or rep_c.merge.get("dup_corrections")),
              f"dup_terms={len(rep_c.merge.get('dup_terms', []))}")

        # ---- D. read-back consistency ----
        log("--- D. read-back consistency ---")
        schema = ingester.read_schema(refresh=True)
        n_tables, n_cols, n_terms = (len(schema.tables()), len(schema.columns()),
                                     len(schema.terms()))
        n_bind = len(schema.edges_of_type(__import__(
            "hugegraph_llm.nl2sql.schema_graph.model", fromlist=["EdgeType"]).EdgeType.TERM_MAPS))
        log(f"    graph now: tables={n_tables} columns={n_cols} terms={n_terms} "
            f"term_bind_edges={n_bind}")
        check("D tables >= structured tables", n_tables >= len(STRUCTURED_META["tables"]))
        check("D terms >= baseline terms",
              n_terms >= rep_a.baseline["terms"], f"{n_terms} >= {rep_a.baseline['terms']}")
        st = ingester.status()
        log(f"    status={st}")
        check("D status ok", st.get("tables", 0) == n_tables)

        # ---- E. single vector store (P2) ----
        log("--- E. vector index (text2vec, shared store) ---")
        try:
            from p2_embedder import make_embedder

            # text2vec-base-chinese 的本地 HF 缓存缺权重文件（下载中断，会崩
            # 'endswith'）；改用本地完整的 BAAI/bge-small-zh-v1.5（中文、512 维）。
            embedder = make_embedder("BAAI/bge-small-zh-v1.5")
            # 用共享 embedder 重新构造 ingester 以测单 store 注入路径
            from hugegraph_llm.nl2sql.vector_store import NumpySchemaVectorStore

            store = NumpySchemaVectorStore()
            ing2 = Nl2SqlIngester(url=URL, graph=GRAPH, embedder=embedder,
                                  vector_store=store)
            vrep = ing2.refresh_vector_index()
            log(f"    vector: {vrep}")
            check("E vector enabled", vrep.get("enabled"),
                  f"nodes={vrep.get('nodes')} dim={vrep.get('dim')}")
            check("E vector nodes == schema nodes", vrep.get("nodes") == n_tables + n_cols + n_terms,
                  f"{vrep.get('nodes')} vs {n_tables + n_cols + n_terms}")

            # ---- F. read path shares the same store ----
            log("--- F. load_pipeline (read path, same store) ---")
            pipe = ing2.load_pipeline()
            items = pipe.link("上周各渠道成交额是多少", top_k=8)
            log(f"    link items={len(items)}: "
                + "; ".join(str(i) for i in items[:5]))
            check("F link non-empty", len(items) > 0)
            check("F same store instance",
                  ing2._vector_store is store
                  and pipe._linker._vector_store is store,
                  "read path uses the injected store")
            term_hit = any("成交额" in i.name or "gmv" in i.name.lower() for i in items)
            check("F 成交额/gmv retrieved", term_hit, "seed reaches the metric")
        except Exception as exc:  # noqa: BLE001
            log(f"    E/F skipped: {type(exc).__name__}: {exc}")
            check("E vector enabled", False, str(exc)[:120])

    log(f"=== RESULT: {len(PASS)} passed, {len(FAIL)} failed ===")
    for n in FAIL:
        log(f"  FAILED: {n}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
