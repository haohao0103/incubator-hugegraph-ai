"""全量合并 Step 4 前置冒烟：L3 纠错数据进 ingest -> loader 读回折叠。

链路（真实 HG kg_rag）：
    p2 corpus (CORRECTIONS) -> corpus_to_metadata (meta.corrections=3)
    -> ingest (写 3 CorrectionDecision 顶点 + 3 correctionAppliesTo* 边)
    -> build_schema_from_hugegraph (loader 折叠到 term/column properties)
    -> 断言折叠正确（供 fetch_corrections 沿图传播召回）。

Run:
  PYTHONPATH=hugegraph-llm/src HF_HUB_OFFLINE=1 \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/merge_step4_prep.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from ingest_metadata_to_hg import ingest  # noqa: E402
from e2e_ingest_load import corpus_to_metadata  # noqa: E402
from p2_corpus import build_warehouse_schema, CORRECTIONS  # noqa: E402
from hugegraph_llm.nl2sql.hugegraph_schema_source import (  # noqa: E402
    build_schema_from_hugegraph,
)

LOG_PATH = "_out/nl2sql_merge/logs/step4_prep.log"
URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== merge step4 prep: corrections into ingest -> loader fold ===")

    meta = corpus_to_metadata(build_warehouse_schema())
    corrs = meta.get("corrections", [])
    log(f"meta corrections={len(corrs)} (expect {len(CORRECTIONS)})")
    assert len(corrs) == len(CORRECTIONS) >= 3, f"corrections missing: {corrs}"

    counts = ingest(meta, URL, GRAPH, clear=True)
    log(f"ingested: corrections={counts.get('corrections')} "
        f"correction_edges={counts.get('correction_edges')}")
    assert counts.get("corrections") == 3, "CorrectionDecision vertices not written"
    assert counts.get("correction_edges") == 3, "correctionAppliesTo edges not written"

    back = build_schema_from_hugegraph(url=URL, graph=GRAPH)
    folded = sum(
        1 for t in back.terms() if t.properties.get("corrections")
    )
    n_corr_edges = (
        len(back.edges_of_type("correctionAppliesToTerm"))
        if hasattr(back, "edges_of_type") else 0
    )
    log(f"loaded back: terms_with_corrections={folded} "
        f"correctionAppliesToTerm_edges={n_corr_edges}")

    # 断言 1: term 成交额 折叠到 corr_gmv_status
    t = next((x for x in back.terms() if x.name == "成交额"), None)
    assert t is not None, "term 成交额 missing"
    t_corrs = t.properties.get("corrections", [])
    ids = {c.get("id") for c in t_corrs}
    log(f"term 成交额 corrections={sorted(ids)}")
    assert "corr_gmv_status" in ids, f"corr_gmv_status not folded onto 成交额: {ids}"

    # 断言 2: caliber 端点经 hasCaliber 反查归到 term（GMV口径 -> 成交额 -> corr_gmv_caliber）
    # loader 的 _annotate_corrections 把 caliber 纠错挂到所属 term 上
    assert "corr_gmv_caliber" in ids, f"corr_gmv_caliber (caliber endpoint) not on 成交额: {ids}"

    # 断言 3: column 端点折叠到 field（corr_pay_amount -> payments.pay_amount）
    cols_with = [
        c for c in back.columns() if c.properties.get("corrections")
    ]
    pay = next((c for c in cols_with if c.name == "pay_amount"), None)
    if pay:
        pay_ids = {cc.get("id") for cc in pay.properties.get("corrections", [])}
        log(f"column payments.pay_amount corrections={sorted(pay_ids)}")
        assert "corr_pay_amount" in pay_ids, f"corr_pay_amount not on pay_amount: {pay_ids}"
    else:
        log("column pay_amount: no corrections folded (check edge direction)")
        assert False, "corr_pay_amount (field endpoint) not folded"

    log("STEP4_PREP: ALL PASS")
    print("ALL PASS")


if __name__ == "__main__":
    main()
