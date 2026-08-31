# -*- coding: utf-8 -*-
"""恢复 kg_rag 的 e2e 数仓 schema（被 semantica PoC 的误删 hasColumn 破坏）。

只跑 ingest（写 HG），不调 LLM。用法：
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \\
      _out/semantica_poc/restore_kg_rag.py
"""
import os
import sys
import time

REPO = "/Users/mac/Desktop/apache-code/hg-rag-hmsgraphrag/incubator-hugegraph-ai"
sys.path.insert(0, os.path.join(REPO, "hugegraph-llm/nl2sql_tools"))
sys.path.insert(0, os.path.join(REPO, "hugegraph-llm/src"))

LOG_PATH = os.path.join(REPO, "_out/semantica_poc/logs/restore_kg_rag.log")
URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"


def log(msg):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def ingest_with_retry(meta, tries=4):
    last = None
    for i in range(tries):
        try:
            counts = ingest(meta, URL, GRAPH, clear=True)
            log(f"ingest ok: {counts}")
            return counts
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            if "locked" in msg:
                log(f"lock hit (try {i+1}/{tries}); sleep 6s")
                time.sleep(6)
                continue
            if "already existed master" in msg:
                log(f"zombie master hit (try {i+1}/{tries}); sleep 12s")
                time.sleep(12)
                continue
            raise
    raise last


if __name__ == "__main__":
    from p2_corpus import build_warehouse_schema  # noqa: E402
    from e2e_ingest_load import corpus_to_metadata  # noqa: E402
    from ingest_metadata_to_hg import ingest  # noqa: E402

    log("=== restore kg_rag e2e schema ===")
    schema0 = build_warehouse_schema()
    meta = corpus_to_metadata(schema0)
    log(f"corpus: tables={len(meta['tables'])} cols={len(meta['columns'])} "
        f"terms={len(meta['terms'])}")
    ingest_with_retry(meta)

    # 校验：hasColumn 边标签是否回来
    import requests  # noqa: E402
    r = requests.get(f"{URL}/graphs/{GRAPH}/schema/edgelabels", timeout=15)
    names = [e["name"] for e in r.json().get("edgelabels", [])]
    log(f"edgelabels after restore: {sorted(names)}")
    assert "hasColumn" in names, "hasColumn NOT restored!"
    log("DONE: hasColumn restored OK")
