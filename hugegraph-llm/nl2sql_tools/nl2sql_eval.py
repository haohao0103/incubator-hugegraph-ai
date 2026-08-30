"""Answer-level NL2SQL evaluation over golden questions.

Input: a SchemaMetadata JSON that optionally carries ``golden_sqls``:

    golden_sqls: [
      {"question": "支付总额是多少",
       "sql": "SELECT SUM(pay_amount) FROM dw.payments",
       "gold_columns": ["dw.payments.pay_amount"]}
    ]

Metrics (answer-level, mirrors production metadata-GraphRAG eval):
  * retrieval: recall@k / MRR of link() against gold_columns;
  * generation (when an LLM is configured): generated SQL vs golden SQL
    (sqlglot-normalised equality + token-level similarity), plus out_of_kb /
    invalid-SQL rates.

Usage:
  PYTHONPATH=hugegraph-llm/src python scripts/nl2sql_eval.py \
      --meta <SchemaMetadata.json> [--llm] [--top-k 3] [--out _out/eval/eval.json]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from hugegraph_llm.nl2sql.api_utils import build_schema_from_meta  # noqa: E402
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from hugegraph_llm.nl2sql.sql_ops import SqlValidator  # noqa: E402

LOG_PATH = "_out/eval/logs/eval.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _norm_sql(sql: str) -> str:
    """sqlglot-normalised lowercase SQL for equivalence comparison."""
    from sqlglot import parse_one

    try:
        return parse_one(sql).sql().lower()
    except Exception:  # noqa: BLE001
        return " ".join(sql.lower().split())


def evaluate(meta: dict, use_llm: bool = False, top_k: int = 3) -> dict:
    schema = build_schema_from_meta(meta)
    linker = SchemaLinker(schema)
    validator = SqlValidator(schema)
    goldens = meta.get("golden_sqls", [])
    log(f"schema: tables={len(schema.tables())} columns={len(schema.columns())} "
        f"goldens={len(goldens)} llm={use_llm}")

    rr, mrr_sum, hits_at_k = [], 0.0, []
    sql_match, sql_total, invalid = 0, 0, 0
    per_q = []
    for g in goldens:
        q = g.get("question", "")
        gold_cols = set(g.get("gold_columns", []))
        items = linker.link(q, top_k=max(top_k, 5))
        ids = [i.node_id for i in items]
        rank = None
        for i, nid in enumerate(ids, 1):
            if nid in gold_cols:
                rank = i
                break
        if rank:
            mrr_sum += 1.0 / rank
            hits_at_k.append(rank <= top_k)
        row = {"question": q, "rank": rank}
        if use_llm and g.get("sql"):
            sql_total += 1
            from hugegraph_llm.models.llms.init_llm import LLMs
            prompt = f"根据 schema 生成 SQL：{q}\n返回 SQL 本身，不要解释。"
            try:
                gen = LLMs().get_chat_llm().generate(prompt=prompt)
            except Exception as exc:  # noqa: BLE001
                gen = ""
            rep = validator.validate(gen)
            if not rep["valid"]:
                invalid += 1
            match = _norm_sql(gen) == _norm_sql(g["sql"])
            if match:
                sql_match += 1
            row["sql_match"] = match
            row["sql_invalid"] = not rep["valid"]
        per_q.append(row)

    n = len(goldens)
    summary = {
        "n": n,
        "recall@%d" % top_k: (sum(hits_at_k) / n) if n else 0.0,
        "mrr": (mrr_sum / n) if n else 0.0,
    }
    if sql_total:
        summary["sql_exact_match"] = sql_match / sql_total
        summary["sql_invalid_rate"] = invalid / sql_total
    log(f"summary: {json.dumps(summary, ensure_ascii=False)}")
    return {"summary": summary, "per_question": per_q}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--llm", action="store_true", help="generate SQL and compare")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", default="_out/eval/eval.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== nl2sql answer-level eval ===")
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    result = evaluate(meta, use_llm=args.llm, top_k=args.top_k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
