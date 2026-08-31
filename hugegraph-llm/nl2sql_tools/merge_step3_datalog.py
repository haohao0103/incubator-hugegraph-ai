"""全量合并 Step 3 验证：Datalog 接线进 ingest（血缘闭包/口径一致性校验）。

链路（纯确定性推理，不连 HG、无 LLM）：
    p2 corpus -> corpus_to_metadata (lineage + calibers + term_bindings)
    -> validate_metadata_rules (DatalogReasonerOp 半朴素 fixpoint)
    -> 断言血缘闭包 / 口径沿血缘传播 / 共合并目标 / 冲突与悬挂检查。

验证的价值点（对比"合并前 e2e"）：
    * 血缘闭包：lineage 只有直接边，upstream*/downstream* 由规则推导（可审计）；
    * 口径传播：ads_daily_sales 的 GMV口径/实付口径 不是直接定义，而是从
      orders/payments 沿 lineage 继承——汇总表指标自动继承明细口径；
    * co_dest：orders 与 payments 汇入同一汇总表 ads_daily_sales
      （join 路径发现前置，供上层 NL2SQL 选择合并路径）。

Run:
  PYTHONPATH=hugegraph-llm/src HF_HUB_OFFLINE=1 \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
      hugegraph-llm/nl2sql_tools/merge_step3_datalog.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from ingest_metadata_to_hg import validate_metadata_rules  # noqa: E402
from e2e_ingest_load import corpus_to_metadata  # noqa: E402
from p2_corpus import build_warehouse_schema  # noqa: E402

LOG_PATH = "_out/nl2sql_merge/logs/step3_datalog.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== merge step3: datalog validation wired into ingest ===")

    meta = corpus_to_metadata(build_warehouse_schema())
    log(f"corpus: lineage={len(meta.get('lineage', []))} "
        f"calibers={len(meta.get('calibers', []))} "
        f"terms={len(meta['terms'])} bindings={len(meta['term_bindings'])}")

    report = validate_metadata_rules(meta)
    assert not report.get("skipped"), f"validation skipped: {report.get('reason')}"

    lg, ca, co, cf, ig = (report["lineage"], report["calibers"],
                          report["co_dest"], report["conflicts"],
                          report["integrity"])
    log(f"lineage edges={lg['edges']} upstream_tables={sorted(lg['upstream'])}")
    log(f"  upstream:  {lg['upstream']}")
    log(f"  downstream:{lg['downstream']}")
    log(f"calibers direct:    {ca['direct']}")
    log(f"calibers inherited: {ca['inherited']}")
    log(f"co_dest (A,B,D): {co}")
    log(f"conflicts: {cf}")
    log(f"integrity: dangling_terms={ig['dangling_terms']} "
        f"dangling_calibers={ig['dangling_calibers']} "
        f"no_caliber_terms={ig['no_caliber_terms']}")
    log(f"stats: {report['stats']}")

    failed = []

    def check(name, cond, detail=""):
        if cond:
            log(f"  PASS {name}")
        else:
            failed.append(name)
            log(f"  FAIL {name} {detail}")

    # --- 1. lineage transitive closure ---
    check("lineage edges == 5", lg["edges"] == 5, f"got {lg['edges']}")
    check("upstream(ads_daily_sales) = {orders,payments,order_items}",
          lg["upstream"].get("ads_daily_sales") == ["order_items", "orders", "payments"],
          f"got {lg['upstream'].get('ads_daily_sales')}")
    check("upstream(ads_user_profile) = {orders,users}",
          lg["upstream"].get("ads_user_profile") == ["orders", "users"],
          f"got {lg['upstream'].get('ads_user_profile')}")
    check("no transitive upstream (no deeper lineage)",
          set(lg["upstream"]) == {"ads_daily_sales", "ads_user_profile"})
    check("downstream(orders) = {ads_daily_sales, ads_user_profile}",
          lg["downstream"].get("orders") == ["ads_daily_sales", "ads_user_profile"],
          f"got {lg['downstream'].get('orders')}")
    check("downstream(payments) = {ads_daily_sales}",
          lg["downstream"].get("payments") == ["ads_daily_sales"],
          f"got {lg['downstream'].get('payments')}")

    # --- 2. caliber propagation along lineage ---
    check("direct: orders={GMV口径}",
          ca["direct"].get("orders") == ["GMV口径"], f"got {ca['direct'].get('orders')}")
    check("direct: payments={实付口径}",
          ca["direct"].get("payments") == ["实付口径"], f"got {ca['direct'].get('payments')}")
    check("direct: ads_daily_sales={营收口径,客单价口径}",
          ca["direct"].get("ads_daily_sales") == ["客单价口径", "营收口径"],
          f"got {ca['direct'].get('ads_daily_sales')}")
    check("inherited: ads_daily_sales={GMV口径,实付口径} (from orders/payments)",
          ca["inherited"].get("ads_daily_sales") == ["GMV口径", "实付口径"],
          f"got {ca['inherited'].get('ads_daily_sales')}")
    check("inherited: ads_user_profile={GMV口径} (from orders)",
          ca["inherited"].get("ads_user_profile") == ["GMV口径"],
          f"got {ca['inherited'].get('ads_user_profile')}")

    # --- 3. common merge target (join path discovery) ---
    check("co_dest(orders, payments, ads_daily_sales)",
          ("orders", "payments", "ads_daily_sales") in co, f"got {co}")
    check("co_dest no self-pairs",
          all(a != b for a, b, _ in co))

    # --- 4. caliber conflicts ---
    check("no caliber conflicts", cf == {}, f"got {cf}")

    # --- 5. integrity (term binding completeness) ---
    check("no dangling terms (every term has computedFromField)",
          ig["dangling_terms"] == [], f"got {ig['dangling_terms']}")
    check("no dangling calibers (every caliber metric has a binding)",
          ig["dangling_calibers"] == [], f"got {ig['dangling_calibers']}")
    check("caliber coverage advisory: 7 terms without caliber",
          len(ig["no_caliber_terms"]) == 7, f"got {ig['no_caliber_terms']}")

    if failed:
        log(f"STEP3: FAIL ({len(failed)}) {failed}")
        print(f"FAIL {failed}")
        sys.exit(1)
    log("STEP3: ALL PASS")
    print("ALL PASS")


if __name__ == "__main__":
    main()
