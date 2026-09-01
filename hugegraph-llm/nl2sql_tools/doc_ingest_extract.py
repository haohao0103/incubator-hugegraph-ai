"""nl2sql 文档通道 CLI —— 薄壳。

抽取核心已迁入 src 包：``hugegraph_llm.nl2sql.doc_extract``（NL 文档 ->
语义元数据，产物与 meta.json 同构，字段绑定只落到真实列，unmatched 进
report 不静默丢弃）。

生产入口请用统一接口（结构化 + 文档同一入口、单图单向量库）：

    python -m hugegraph_llm.nl2sql.ingest --source doc --doc-type dictionary --doc x.md
    python -m hugegraph_llm.nl2sql.ingest --source doc_bundle \
        --docs dictionary:a.md,caliber:b.md,qa:c.md

本脚本仅保留旧 CLI 形态（--docs '类型:路径' --meta --out），内部转调
``doc_extract`` 的抽取 + merge，供既有流程复用。

Usage:
  PYTHONPATH=hugegraph-llm/src HF_HUB_OFFLINE=1 \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/doc_ingest_extract.py \
      --docs dictionary:sample_docs/data_dictionary.md \
             caliber:sample_docs/caliber_docs.md \
             qa:sample_docs/qa_history.md \
      --meta _out/sql_miner/meta.json \
      --out _out/nl2sql_doc/merged_meta.json
"""
import argparse
import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hugegraph_llm.config import llm_settings  # noqa: E402
from hugegraph_llm.models.llms.init_llm import LLMs  # noqa: E402
from hugegraph_llm.nl2sql.doc_extract import (  # noqa: E402
    extract_document,
    merge_into_meta,
    normalize_caliber,
    normalize_dictionary,
    normalize_qa,
)

LOG_PATH = "_out/nl2sql_doc/logs/doc_extract.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", nargs="+", required=True,
                    help="'类型:路径'，如 dictionary:data_dictionary.md")
    ap.add_argument("--meta", required=True, help="既有 SchemaMetadata JSON")
    ap.add_argument("--out", required=True, help="merge 后的 JSON 输出")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== doc channel: NL documents -> semantic metadata (thin CLI) ===")

    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)

    llm = LLMs().get_chat_llm()
    log(f"llm type={type(llm).__name__} model={llm_settings.openai_chat_language_model}")

    extracted = {"terms": [], "term_bindings": [], "calibers": [], "corrections": []}
    for spec in args.docs:
        doc_type, _, path = spec.partition(":")
        if not os.path.exists(path):
            log(f"  SKIP missing doc: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        log(f"extracting [{doc_type}] {path} ({len(text)} chars)")
        items = extract_document(text, doc_type, llm)
        part = {
            "dictionary": normalize_dictionary,
            "caliber": normalize_caliber,
            "qa": normalize_qa,
        }[doc_type](items)
        for k, v in part.items():
            extracted[k].extend(v)

    merged, report = merge_into_meta(meta, extracted)
    log(f"extracted totals: terms={len(extracted['terms'])} "
        f"calibers={len(extracted['calibers'])} "
        f"corrections={len(extracted['corrections'])} "
        f"unmatched={len(report['unmatched'])} dup_terms={len(report['dup_terms'])}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out} "
        f"(terms={len(merged['terms'])} bindings={len(merged['term_bindings'])} "
        f"calibers={len(merged['calibers'])} corrections={len(merged['corrections'])})")
    log("DONE")


if __name__ == "__main__":
    main()
