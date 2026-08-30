"""Platform retrieval evaluation: 准确率 / 知识召回率 / TopK / MRR.

Metrics aligned with the 货拉拉 metadata-GraphRAG practice
(准确率 78% / 知识召回率 91% / TopK 90% / MRR 0.73):

- 知识召回率 (Recall@k): expected entities (tables/fields/metrics) hit in
  the retrieved context / total expected entities.
- TopK 命中率: fraction of questions where ALL expected entities appear in
  the retrieved top-k (k = max_tables + max_metrics + max_fields).
- MRR: mean reciprocal rank of the first expected entity hit.
- 准确率 (answer proxy): fraction of questions where the top-1 retrieved
  entity belongs to the expected set. Honest note: this is a retrieval
  top-1 proxy, NOT end-to-end SQL accuracy (that needs the platform's real
  eval set with golden SQL).

Questions flagged ``expect_refusal`` are checked for no_evidence instead of
hits (the 货拉拉 '不存在也答' badcase guard).

Usage::

    python _out/platform_eval/eval_retrieval.py            # live kg_platform
    python _out/platform_eval/eval_retrieval.py --set path  # custom eval set
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logging.disable(logging.CRITICAL)

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
sys.path.insert(0, _ROOT)

from hugegraph_llm.config import huge_settings  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_multi_retrieval import (  # noqa: E402
    KgMultiSchemaLinker,
    MultiRecallConfig,
)
from hugegraph_llm.operators.graph_op.kg_query_understanding import (  # noqa: E402
    QueryUnderstanding,
    QueryUnderstandingConfig,
)
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_term_graph import KgTermGraph  # noqa: E402

DEFAULT_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")

# reference targets from the 货拉拉 practice
TARGETS = {"准确率": 0.78, "知识召回率": 0.91, "TopK命中率": 0.90, "MRR": 0.73}


def _client(graph: str):
    from hugegraph_llm.utils.hugegraph_utils import get_hg_client

    client = get_hg_client()
    # get_hg_client is bound to huge_settings.graph_name; re-point if needed
    if huge_settings.graph_name != graph:
        from pyhugegraph.client import PyHugeClient

        return PyHugeClient(
            url=huge_settings.graph_url, graph=graph,
            user=huge_settings.graph_user, pwd=huge_settings.graph_pwd,
            graphspace=huge_settings.graph_space,
        )
    return client


def _linked_names(ctx) -> Dict[str, set]:
    return {
        "tables": {t.get("name") for t in ctx.tables},
        "fields": {f.get("name") for f in ctx.fields},
        "metrics": {m.get("name") for m in ctx.metrics},
    }


def evaluate(
    questions: List[Dict[str, Any]],
    synonyms: Dict[str, str],
    graph: str,
    live: bool = True,
    importance_weight: float = 0.5,
    intent_weight: float = 0.8,
    use_llm: bool = False,
    distractors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    client = _client(graph) if live else None
    llm = None
    if use_llm:
        from hugegraph_llm.models.llms.init_llm import LLMs

        llm = LLMs().get_chat_llm()  # falls back to heuristic inside
    linker = KgMultiSchemaLinker(
        client=client,
        synonyms=None,
        config=MultiRecallConfig(
            importance_weight=importance_weight,
            intent_weight=intent_weight,
        ),
        query_understanding=QueryUnderstanding(
            llm=llm,
            term_graph=KgTermGraph.from_jargon_map(synonyms),
            config=QueryUnderstandingConfig(short_query_threshold=8),
        ),
    )
    data = KgRuleEngine(client, graph).load_graph() if client else None

    rows: List[Dict[str, Any]] = []
    hit_entities = 0
    total_entities = 0
    topk_ok = 0
    refusal_ok = 0
    refusal_total = 0
    top1_ok = 0
    mrr_sum = 0.0
    q_count = 0
    answer_total = 0
    answer_top1 = 0

    # deterministic answer-level probe: for questions with a golden SQL,
    # vote over [golden + 2 distractors] and count golden landing at top-1.
    # This measures the validator+voter's preference for the correct SQL,
    # NOT end-to-end generation accuracy (that needs the platform's real set).
    distractor_sqls = [d.get("sql") for d in (distractors or []) if d.get("sql")]

    for item in questions:
        q = item["question"]
        expected = item.get("expected", {})
        expect_refusal = bool(item.get("expect_refusal"))

        start = time.monotonic()
        ctx = linker.link(q, data=data)
        latency_ms = (time.monotonic() - start) * 1000
        linked = _linked_names(ctx)

        if expect_refusal:
            refusal_total += 1
            ok = ctx.empty
            refusal_ok += int(ok)
            rows.append({
                "question": q, "type": "refusal", "ok": ok,
                "empty": ctx.empty, "latency_ms": round(latency_ms, 1),
            })
            continue

        q_count += 1
        all_expected = (
            list(expected.get("tables", []))
            + list(expected.get("fields", []))
            + list(expected.get("metrics", []))
        )
        total_entities += len(all_expected)
        hits = [e for e in all_expected if e in (
            linked["tables"] | linked["fields"] | linked["metrics"]
        )]
        hit_entities += len(hits)

        # top-k = all budgeted slots; question is a TopK hit when every
        # expected entity is inside the retrieved context
        ok_topk = len(hits) == len(all_expected)
        topk_ok += int(ok_topk)

        # MRR over expected entities by their best rank in a fused ranking
        fused = _fused_rank(linker, q, data, all_expected)
        if fused:
            mrr_sum += 1.0 / fused
        top1 = fused == 1
        top1_ok += int(top1)

        rows.append({
            "question": q, "type": "hit",
            "expected": all_expected,
            "hit": hits, "ok": ok_topk, "top1": top1,
            "mrr_rank": fused or None,
            "tables": sorted(linked["tables"]), "metrics": sorted(linked["metrics"]),
            "latency_ms": round(latency_ms, 1),
        })

        # answer-level: vote [golden + 2 distractors], golden at top-1?
        golden = item.get("expected_sql")
        if golden and distractor_sqls:
            answer_total += 1
            candidates = [golden] + distractor_sqls[:2]
            try:
                from hugegraph_llm.operators.graph_op.kg_sql_voter import KgSqlVoter

                voter = KgSqlVoter(graph_data=data)
                votes = voter.vote(candidates)
                if votes and votes[0].sql.strip() == golden.strip():
                    answer_top1 += 1
                rows[-1]["answer_top1"] = votes[0].sql.strip() == golden.strip()
            except Exception as exc:  # pragma: no cover - voter guard
                rows[-1]["answer_top1"] = None

    recall = hit_entities / total_entities if total_entities else 0.0
    topk_rate = topk_ok / q_count if q_count else 0.0
    mrr = mrr_sum / q_count if q_count else 0.0
    acc = top1_ok / q_count if q_count else 0.0
    refusal_rate = refusal_ok / refusal_total if refusal_total else 0.0

    metrics = {
        "准确率(top-1代理)": round(acc, 4),
        "知识召回率": round(recall, 4),
        "TopK命中率": round(topk_rate, 4),
        "MRR": round(mrr, 4),
        "拒答正确率": round(refusal_rate, 4),
        "答案级top-1(投票)": round(answer_top1 / answer_total, 4) if answer_total else None,
        "平均延迟ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 1) if rows else 0.0,
    }
    return {"metrics": metrics, "rows": rows}


def _fused_rank(linker, question: str, data, expected: List[str]) -> Optional[int]:
    """Best (minimum) rank of any expected entity in the fused ranking."""
    if not expected:
        return None
    ctx = linker.link(question, data=data)
    ranked = [name for _label, name in ctx.ranking] if ctx.ranking else (
        [t.get("name") for t in ctx.tables]
        + [m.get("name") for m in ctx.metrics]
        + [f.get("name") for f in ctx.fields]
    )
    for i, name in enumerate(ranked, 1):
        if name in expected:
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Platform retrieval evaluation")
    parser.add_argument("--set", default=DEFAULT_SET)
    parser.add_argument("--graph", default="kg_platform")
    parser.add_argument("--offline", action="store_true",
                        help="run against in-memory graph data (no server)")
    parser.add_argument("--importance", type=float, default=0.5)
    parser.add_argument("--intent-weight", type=float, default=0.8)
    parser.add_argument("--llm", action="store_true",
                        help="use the live LLM for query understanding "
                             "(falls back to heuristic on flake)")
    args = parser.parse_args()

    with open(args.set, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not args.offline:
        try:
            _client(args.graph).gremlin().exec("g.V().limit(1).count()")
        except Exception as exc:  # pragma: no cover
            print(f"SKIP: graph {args.graph} unreachable: {exc}")
            return 0

    result = evaluate(
        payload["questions"],
        payload.get("synonyms", {}),
        args.graph,
        live=not args.offline,
        importance_weight=args.importance,
        intent_weight=args.intent_weight,
        use_llm=args.llm,
        distractors=payload.get("sql_distractors"),
    )
    metrics, rows = result["metrics"], result["rows"]

    print(f"=== 平台检索评估 graph={args.graph} (questions={len(rows)}) ===")
    print(f"\n{'指标':<16}{'本系统':>10}{'对标(货拉拉)':>14}{'达标':>8}")
    print("-" * 50)
    for name, target in TARGETS.items():
        key = {"准确率": "准确率(top-1代理)", "知识召回率": "知识召回率",
               "TopK命中率": "TopK命中率", "MRR": "MRR"}[name]
        val = metrics[key]
        ok = "✓" if val >= target else "✗"
        print(f"{name:<16}{val:>10.2%}{target:>14.2%}{ok:>8}")

    print(f"{'拒答正确率':<16}{metrics['拒答正确率']:>10.2%}{'—':>14}")
    if metrics["答案级top-1(投票)"] is not None:
        print(f"{'答案级top-1(投票)':<16}{metrics['答案级top-1(投票)']:>10.2%}{'78%目标':>14}")
    print(f"{'平均延迟':<16}{metrics['平均延迟ms']:>10.1f}ms{'—':>14}")

    print("\n=== 逐题明细 ===")
    for r in rows:
        if r["type"] == "refusal":
            print(f"  [拒答] {'✓' if r['ok'] else '✗'} {r['question']}  ({r['latency_ms']}ms)")
        else:
            flag = "✓" if r["ok"] else "✗"
            miss = sorted(set(r["expected"]) - set(r["hit"]))
            ans = r.get("answer_top1")
            ans_flag = "✓" if ans else ("✗" if ans is False else "-")
            print(f"  [{flag}] {r['question']}  top1={r['top1']} "
                  f"mrr_rank={r['mrr_rank']}  ans={ans_flag}  miss={miss or '-'}  ({r['latency_ms']}ms)")

    print(f"\nnote: 准确率是检索 top-1 代理指标（非端到端 SQL 准确率），"
          f"待平台真实评估集（含 golden SQL）后升级为答案级。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
