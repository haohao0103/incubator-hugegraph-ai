"""全量合并 Step 4 对比：合并前 e2e 基线 vs 合并后（口径约束+L3纠错+Datalog）。

输入：
  before: _out/e2e_sqlgen/e2e_sqlgen_before_merge.json  (13:57, 无 corrections 注入)
  after:  _out/e2e_sqlgen/e2e_sqlgen.json               (20:59, 合并后全链)

对比维度：
  1. summary 指标（recall/MRR 应不变——检索与 LLM 无关；sql_valid_rate 应升）
  2. 逐题 valid 翻转（哪些题变好/变坏）
  3. 纠错召回（corr=N）与口径约束生效的 SQL 证据
  4. 按 category 分组 valid rate
  5. SQL 长度/字段引用质量（粗粒度：引用字段数、是否含口径过滤 WHERE）

Run:
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/merge_step4_compare.py
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "_out", "e2e_sqlgen")
BEFORE = os.path.join(OUT_DIR, "e2e_sqlgen_before_merge.json")
AFTER = os.path.join(OUT_DIR, "e2e_sqlgen.json")
LOG_PATH = os.path.join(REPO, "_out", "nl2sql_merge", "logs", "step4_compare.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    before, after = load(BEFORE), load(AFTER)
    sb, sa = before["summary"], after["summary"]
    log("=== merge step4: before-merge e2e vs after-merge ===")

    log("SUMMARY  before: " + json.dumps(sb, ensure_ascii=False))
    log("SUMMARY  after : " + json.dumps(sa, ensure_ascii=False))
    dv = (sa["sql_valid_rate"] - sb["sql_valid_rate"]) * 100
    dr = (sa["retrieval_recall@10"] - sb["retrieval_recall@10"]) * 100
    log(f"delta: sql_valid={dv:+.1f}pp  recall={dr:+.1f}pp  "
        f"mrr={sa['retrieval_mrr']-sb['retrieval_mrr']:+.4f}")

    pb = {q["q"]: q for q in before["per_question"]}
    pa = {q["q"]: q for q in after["per_question"]}

    improved, worsened, unchanged = [], [], []
    for q in after["per_question"]:
        vb = pb[q["q"]]["valid"]
        va = q["valid"]
        if va and not vb:
            improved.append(q["q"])
        elif vb and not va:
            worsened.append(q["q"])
        else:
            unchanged.append(q["q"])
    log(f"valid flip: improved={improved}")
    log(f"valid flip: worsened={worsened}")
    log(f"valid flip: unchanged={len(unchanged)}")

    # 按 category 分组 valid rate
    for cat in ("lexical", "semantic", "join"):
        nb = sum(1 for q in before["per_question"] if q["category"] == cat)
        ab = sum(1 for q in before["per_question"]
                 if q["category"] == cat and q["valid"])
        na = sum(1 for q in after["per_question"] if q["category"] == cat)
        aa = sum(1 for q in after["per_question"]
                 if q["category"] == cat and q["valid"])
        log(f"[{cat:<8}] before {ab}/{nb}  after {aa}/{na}")

    # 纠错召回统计（合并后）
    with_corr = [q for q in after["per_question"] if q.get("n_corrections", 0) > 0]
    total_corr = sum(q.get("n_corrections", 0) for q in after["per_question"])
    log(f"corrections: questions_with_corr={len(with_corr)}/23  "
        f"total_corr_retrieved={total_corr}")
    for q in with_corr:
        log(f"  corr={q['n_corrections']}(prop {q.get('corr_propagated')}) "
            f"valid={q['valid']}  {q['q']}")

    # 口径约束证据：3 条纠错对应的问题（term/caliber/field 端点）
    log("caliber/correction evidence (after-merge SQL):")
    for key in ("订单的成交总额", "买卖盘子有多大", "支付金额是多少",
                "用户下的订单支付了多少", "门店销售额和退款的差额"):
        if key in pa:
            q = pa[key]
            log(f"  [{q['category']}] {key}: valid={q['valid']} corr={q.get('n_corrections')} "
                f"sql={q['sql'][:100]!r}")
            if key in pb:
                log(f"      before: valid={pb[key]['valid']} sql={pb[key]['sql'][:100]!r}")

    # SQL 含 paid 口径过滤的题（口径约束生效的证据）
    paid_filter = [q for q in after["per_question"]
                   if "paid" in (q.get("sql") or "").lower()]
    log(f"SQL containing paid-filter (caliber constraint): {len(paid_filter)} questions "
        f"-> {[q['q'] for q in paid_filter]}")

    # 结论
    n_imp, n_wor = len(improved), len(worsened)
    log(f"VERDICT: improved={n_imp} worsened={n_wor} "
        f"valid_rate {sb['sql_valid_rate']:.3f} -> {sa['sql_valid_rate']:.3f} "
        f"({'BETTER' if dv > 0 and n_wor == 0 else 'MIXED' if dv > 0 else 'NO_GAIN'})")


if __name__ == "__main__":
    main()
