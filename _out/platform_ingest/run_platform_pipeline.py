"""End-to-end platform pipeline demo (灌库 -> 检索 -> 文档补口径 -> 重灌 -> 命中).

Simulates tomorrow's platform onboarding with sample data:

1. ingest the platform catalog+metrics JSON into ``kg_platform`` (reset);
2. retrieval smoke 1: a Chinese question for a metric that has NO 口径 in the
   structured data (only a name) -- expected to MISS;
3. Feishu-doc extraction: read the sample 口径 doc, LLM-extract the glossary,
   validate, merge the authoritative definition/formula back into the payload;
4. re-ingest with the merged payload (reset);
5. retrieval smoke 2: the same Chinese questions -- now expected to HIT the
   doc-completed metrics.

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/platform_ingest/run_platform_pipeline.py 2>&1 \\
        | tee _out/platform_ingest/logs/run_platform_pipeline.log
"""

from __future__ import annotations

import json
import logging
import os
import sys

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.config import huge_settings  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker  # noqa: E402
from pyhugegraph.client import PyHugeClient  # noqa: E402

from doc_extract.doc_metric_extractor import (  # noqa: E402
    extract_metrics_from_text,
    merge_glossary,
    validate_glossary,
)
from ingest_adapter import (  # noqa: E402
    ingest_platform,
    normalize_catalog,
    normalize_metrics,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_platform")


def _client(graph: str = GRAPH):
    return PyHugeClient(
        url=huge_settings.graph_url, graph=graph,
        user=huge_settings.graph_user, pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )


def _smoke(question: str) -> list:
    data = KgRuleEngine(_client(), GRAPH).load_graph()
    ctx = KgSchemaLinker().link(question, data=data)
    return [m.get("name") for m in ctx.metrics]


def main() -> int:
    catalog_payload = normalize_catalog(
        json.load(open(os.path.join(_HERE, "sample_data/platform_catalog.json"), encoding="utf-8"))
    )
    metric_payload = normalize_metrics(
        json.load(open(os.path.join(_HERE, "sample_data/platform_metrics.json"), encoding="utf-8"))
    )

    print(f"=== 1) ingest platform data into '{GRAPH}' (reset) ===")
    counts = ingest_platform(
        catalog_payload, metric_payload, graph=GRAPH, domain="platform", reset=True
    )
    print(f"  ingested {counts['vertices']} vertices / {counts['edges']} edges")

    print("\n=== 2) retrieval smoke 1 (口径 missing in structured data) ===")
    for q in ("客单价是多少", "司机订单数"):
        hit = _smoke(q)
        print(f"  {q!r:>12} -> metrics={hit}")
    before_hit = "avg_order_value" in _smoke("客单价是多少")

    print("\n=== 3) Feishu-doc -> 口径 glossary -> merge ===")
    doc_text = open(
        os.path.join(_HERE, "sample_docs/指标口径文档.md"), encoding="utf-8"
    ).read()
    glossary = extract_metrics_from_text(doc_text)
    ok, issues = validate_glossary(glossary)
    print(f"  extracted {len(glossary)} metric entries; valid={ok}")
    if issues:
        print(f"  issues: {issues[:5]}")
    if not glossary:
        print("  WARN: LLM extraction returned nothing (endpoint flake?) -- "
              "pipeline will show no improvement")
    merged, stats = merge_glossary(metric_payload, glossary)
    print(f"  merge stats: {stats}")
    for m in merged["metrics"]:
        print(f"    {m['name']:<18} def={m['definition'][:18]!r:<22} formula={m['formula'][:40]!r}")

    print(f"\n=== 4) re-ingest with merged 口径 (reset) ===")
    counts = ingest_platform(
        catalog_payload, merged, graph=GRAPH, domain="platform", reset=True
    )
    print(f"  ingested {counts['vertices']} vertices / {counts['edges']} edges")

    print("\n=== 5) retrieval smoke 2 (口径 completed by doc) ===")
    after_hit = False
    for q in ("客单价是多少", "司机订单数"):
        hit = _smoke(q)
        print(f"  {q!r:>12} -> metrics={hit}")
        after_hit = after_hit or "avg_order_value" in hit or "driver_order_cnt" in hit

    print("\n=== result ===")
    print(f"  '客单价' hit before doc: {before_hit} -> after doc: "
          f"{'avg_order_value' in _smoke('客单价是多少')}")
    print(f"  doc-completed metric linked by Chinese question: "
          f"{'YES' if after_hit else 'NO'}")
    if after_hit:
        print("PASS: Feishu-doc 口径 extraction -> merge -> retrieval loop works")
        return 0
    print("FAIL: doc-completed metric still not linked (check LLM output / merge)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
