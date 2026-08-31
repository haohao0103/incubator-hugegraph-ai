"""End-to-end NL2SQL link against a real HugeGraph + real MiMo LLM.

Full chain in ONE process (avoids HG's cross-graph read/write leakage that makes
inter-process reads of kg_rag flaky):

    corpus -> ingest to kg_rag (HG) -> build_schema_from_hugegraph (real HG read)
           -> SchemaLinker (with cross-encoder reranker) -> MiMo generates SQL
           -> SqlValidator checks references -> metrics + samples.

Usage:
  PYTHONPATH=hugegraph-llm/src HF_HUB_OFFLINE=1 \\
  python hugegraph-llm/nl2sql_tools/e2e_sql_gen.py

  (reranker model/candidate_k/alpha are hardcoded in main(): BAAI/bge-reranker-base,
   candidate_k=10, alpha=0.3. The model must already be in the local HF cache.)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from p2_corpus import build_warehouse_schema, WAREHOUSE_QUESTIONS  # noqa: E402
from e2e_ingest_load import corpus_to_metadata  # noqa: E402
from ingest_metadata_to_hg import ingest  # noqa: E402
from hugegraph_llm.nl2sql.hugegraph_schema_source import (  # noqa: E402
    build_schema_from_hugegraph,
)
from hugegraph_llm.nl2sql.rerank import CrossEncoderReranker  # noqa: E402
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from hugegraph_llm.nl2sql.correction_propagation import fetch_corrections  # noqa: E402
from hugegraph_llm.nl2sql.sql_ops import SqlValidator  # noqa: E402
from hugegraph_llm.models.llms.init_llm import LLMs  # noqa: E402
from hugegraph_llm.config import llm_settings  # noqa: E402

URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"
TOPK = 10
LOG_PATH = "_out/e2e_sqlgen/logs/e2e_sqlgen.log"
OUT_JSON = "_out/e2e_sqlgen/e2e_sqlgen.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def ingest_with_retry(meta, tries=4):
    """ingest clears kg_rag first. Two transient HG defects can block the clear:
      - a stale hstore edge-delete lock  -> 'locked'  (retry after 6s)
      - a zombie master registration from a sibling graph
        (e.g. 'DEFAULT-kg_enrich_live/server-1') that expires after ~10s
        -> 'Already existed master' (retry after 12s)
    We surface any *other* error immediately."""
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
                log(f"zombie master hit (try {i+1}/{tries}); sleep 12s "
                    f"(sibling graph registration expires)")
                time.sleep(12)
                continue
            raise
    raise last


def core_of(node_id: str) -> str:
    """'column:dw.payments.pay_amount' -> 'payments.pay_amount'."""
    s = node_id.split(":", 1)[-1]
    return s[3:] if s.startswith("dw.") else s


def node_display(schema, node_id):
    n = schema.nodes.get(node_id)
    if not n:
        return node_id
    p = n.properties
    t = n.node_type.value
    if t == "table":
        return f"{n.name} (表) {p.get('comment', '')}"
    if t == "column":
        return f"{p.get('table', '')}.{n.name} ({p.get('data_type', '')}) {p.get('comment', '')}"
    if t == "term":
        return f"{n.name} (指标) {p.get('comment', '')}"
    return node_id


def build_prompt(question, items, schema, corrections=None, corr_stats=None):
    lines = ["你是一个数仓 Text2SQL 助手。只依据下面召回的表和字段生成 SQL。",
             f"问题：{question}", "召回的 schema（表/字段）："]
    seen = set()
    for it in items:
        disp = node_display(schema, it.node_id)
        if disp not in seen:
            seen.add(disp)
            lines.append(f"  - {disp}")

    # 口径约束：从召回列反查绑定术语（TERM_MAPS 边），注入该术语的 caliber。
    # 语义层差异化能力——LLM 只凭字段注释不知道「GMV 只统计 paid」，口径给了它。
    item_ids = {it.node_id for it in items}
    calibers = []
    for e in schema.edges:
        if e.edge_type.value != "term_maps" or e.target not in item_ids:
            continue
        t = schema.nodes.get(e.source)
        if t:
            calibers.extend(t.properties.get("calibers", []) or [])
    if calibers:
        lines.append("【口径约束（必须严格遵守）】")
        for c in calibers:
            lines.append(f"  - {c.get('name', '')}：{c.get('description', '')}")

    # L3 历史纠错：沿语义边传播召回（同义词/指标链/口径可达也召回），避免重犯。
    if corrections:
        lines.append("【历史纠错（重要，避免重犯）】")
        for c in corrections:
            lines.append(f"  - 曾被纠正：错误SQL={c.get('wrong_sql', '')}")
            lines.append(f"    正确SQL={c.get('correct_sql', '')}")
            lines.append(f"    原因={c.get('correction_reason', '')}")

    lines.append("要求：仅使用上述字段；输出 SQL 本身，不要解释、不要 markdown 代码块。")
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== e2e: ingest -> HG read -> link(rerank) -> MiMo SQL ===")

    # 1) write schema to HG
    schema0 = build_warehouse_schema()
    meta = corpus_to_metadata(schema0)
    log(f"corpus: tables={len(meta['tables'])} cols={len(meta['columns'])} "
        f"terms={len(meta['terms'])}")
    ingest_with_retry(meta)

    # 2) read schema back FROM HG (real read path)
    schema = build_schema_from_hugegraph(url=URL, graph=GRAPH)
    log(f"schema from HG: tables={len(schema.tables())} "
        f"cols={len(schema.columns())} terms={len(schema.terms())} "
        f"edges={len(schema.edges)}")
    if not schema.tables():
        log("ERROR: kg_rag returned empty schema")
        raise SystemExit(1)

    # 3) linker + cross-encoder reranker (the feature under test).
    # The ~1.1GB cross-encoder is loaded ONCE (probe on .available), used across
    # all link() calls in PHASE A, then released so it is never resident alongside
    # the MiMo LLM client + schema graph — on this dev box holding both at once
    # triggers an OOM kill (137).
    rer = CrossEncoderReranker(model_name="BAAI/bge-reranker-base",
                               candidate_k=10, alpha=0.3)
    log(f"reranker configured (load-once): model={rer.model_name} "
        f"available_probe={rer.available}")
    linker = SchemaLinker(schema, reranker=rer)
    validator = SqlValidator(schema)
    llm = LLMs().get_chat_llm()
    log(f"llm type={type(llm).__name__} model={llm_settings.openai_chat_language_model} "
        f"base={llm_settings.openai_chat_api_base}")

    # 4) PHASE A — link every question with the reranker (model resident).
    mrr_sum, hits, n = 0.0, 0, 0
    linked = []  # (question_dict, items, rank)
    for qi, q in enumerate(WAREHOUSE_QUESTIONS, 1):
        qq = q["q"]
        gold = set(q.get("gold", []))
        items = linker.link(qq, top_k=TOPK)
        ids = [i.node_id for i in items]
        rank = None
        for idx, nid in enumerate(ids, 1):
            if core_of(nid) in gold:
                rank = idx
                break
        if rank:
            mrr_sum += 1.0 / rank
            hits += 1
        n += 1
        linked.append((q, items, rank))
        log(f"[{qi:2d}/{len(WAREHOUSE_QUESTIONS)}][link][{q['category']:<8}] "
            f"{qq:<14} rank={rank} top={core_of(ids[0]) if ids else '-'}")

    # Free the ~1.1GB cross-encoder before the LLM phase (avoid OOM on this box).
    rer.release()

    # 4) PHASE B — real MiMo SQL generation + validation (model released).
    #    L3 纠错召回：对每个问题的 seed 沿语义边传播，把可达纠错注入 prompt。
    sql_valid = 0
    per_q = []
    for qi, (q, items, rank) in enumerate(linked, 1):
        qq = q["q"]
        corrs, cstats = fetch_corrections(schema, set(linker.seed_nodes(qq)))
        prompt = build_prompt(qq, items, schema, corrections=corrs,
                              corr_stats=cstats)
        try:
            gen = llm.generate(prompt=prompt)
        except Exception as e:  # noqa: BLE001
            gen = ""
            log(f"  LLM FAIL: {type(e).__name__} {str(e)[:120]}")
        rep = validator.validate(gen)
        if rep.get("valid"):
            sql_valid += 1
        per_q.append({
            "q": qq, "category": q["category"], "rank": rank,
            "gold": q.get("gold"), "sql": gen, "valid": rep.get("valid"),
            "linked": [core_of(i.node_id) for i in items[:5]],
            "n_corrections": len(corrs),
            "corr_propagated": len(cstats.get("propagated", [])),
        })
        log(f"[{qi:2d}/{len(WAREHOUSE_QUESTIONS)}][sql ][{q['category']:<8}] "
            f"{qq:<14} rank={rank} valid={rep.get('valid')} "
            f"corr={len(corrs)}(prop {len(cstats.get('propagated', []))}) "
            f"sql={gen[:70]}")

    summary = {
        "n": n,
        "retrieval_recall@%d" % TOPK: hits / n,
        "retrieval_mrr": mrr_sum / n,
        "sql_valid_rate": sql_valid / n if n else 0.0,
    }
    log("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_q}, f,
                  ensure_ascii=False, indent=2)
    log("wrote " + OUT_JSON)
    log("DONE")


if __name__ == "__main__":
    main()
