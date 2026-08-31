"""全量合并 Step 1 验证：术语/口径本体进 ingest。

链路（真实 HG kg_rag）：
    p2 corpus -> corpus_to_metadata (含 calibers) -> ingest (写 Caliber 顶点 +
    hasCaliber 边) -> build_schema_from_hugegraph (读回) -> 断言 caliber 挂到 term。

关键点：
    * Caliber 顶点用 PRIMARY_KEY(name) 体系，与 Table/Field/Metric 一致；
    * hasCaliber 边标签 Metric->Caliber，与 PoC 的 CUSTOMIZE_STRING 体系隔离；
    * ingest 的 _ensure_vertex_label 带自愈：同名标签 id 策略不匹配时 drop 重建。

Run:
  PYTHONPATH=hugegraph-llm/src HF_HUB_OFFLINE=1 \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/merge_step1_caliber.py
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

LOG_PATH = "_out/nl2sql_merge/logs/step1_caliber.log"
URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"
EXPECTED_CALIBERS = {"成交额": "GMV口径", "支付总额": "实付口径",
                     "客单价": "客单价口径", "营收": "营收口径"}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== merge step1: caliber ontology into ingest ===")

    schema0 = build_warehouse_schema()
    meta = corpus_to_metadata(schema0)
    n_cal = len(meta.get("calibers", []))
    log(f"corpus: tables={len(meta['tables'])} cols={len(meta['columns'])} "
        f"terms={len(meta['terms'])} calibers={n_cal}")
    assert n_cal >= 4, f"expected >=4 calibers from corpus, got {n_cal}"

    counts = ingest(meta, URL, GRAPH, clear=True)
    log(f"ingested: {counts}")
    assert counts["calibers"] >= 4, "Caliber vertices not written"
    assert counts["has_caliber_edges"] >= 4, "hasCaliber edges not written"

    back = build_schema_from_hugegraph(url=URL, graph=GRAPH)
    log(f"loaded back: tables={len(back.tables())} cols={len(back.columns())} "
        f"terms={len(back.terms())} calibers_in_terms={sum(1 for t in back.terms() if t.properties.get('calibers'))}")

    # 逐个断言：每个 term 的 caliber name/description 都正确挂载
    failed = []
    for term_name, cal_name in EXPECTED_CALIBERS.items():
        terms = [t for t in back.terms() if t.name == term_name]
        if not terms:
            failed.append(f"{term_name}: term missing")
            continue
        cals = terms[0].properties.get("calibers", [])
        if not any(c.get("name") == cal_name for c in cals):
            failed.append(f"{term_name}: caliber {cal_name} missing, got {[c.get('name') for c in cals]}")
        else:
            c = next(c for c in cals if c.get("name") == cal_name)
            log(f"  PASS {term_name} -> {cal_name}: {c.get('description', '')[:40]}...")

    if failed:
        for f in failed:
            log(f"  FAIL {f}")
        log("STEP1: FAIL")
        sys.exit(1)
    log("STEP1: ALL PASS")
    print("ALL PASS" if not failed else "FAIL")


if __name__ == "__main__":
    main()
