"""Feishu-doc -> metric glossary extraction layer (骨架).

The upstream platform keeps authoritative metric 口径 in Feishu docs while the
structured catalog/metrics JSON may be incomplete (e.g. a metric with a name
but no formula). This module:

1. defines the **glossary contract** -- a JSON structure that is isomorphic to
   the ``metric_json`` payload consumed by ``unified_convert``, so a glossary
   can be merged straight back into the ingest payload (see
   ``DOC_EXTRACT_SPEC.md`` for the full interface spec);
2. extracts -- split a document into blocks, ask an LLM (glm-5.3, injectable)
   to emit glossary entries in JSON, tolerate fenced blocks / noise;
3. merges  -- fill the authoritative 口径 (definition/formula) into the
   platform metric payload, top up source tables/fields, report conflicts.

The pipeline is::

    Feishu doc text -> split_doc_blocks -> extract_metrics_from_text
        -> validate_glossary -> merge_glossary(platform_metrics, glossary)
        -> ingest_adapter.ingest_platform(...)

Run the end-to-end demo via ``../run_platform_pipeline.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# 1) Glossary contract (metrics) -- isomorphic to unified_convert metric payload
# ---------------------------------------------------------------------------
GLOSSARY_KEYS = (
    "name", "definition", "formula", "source_tables", "source_fields", "depends_on",
)

_EXTRACT_PROMPT = """你是数仓指标口径整理助手。从下面的文档片段中，抽取所有**业务指标口径**。

只输出一个 JSON 数组，不要输出任何其他文字。数组元素格式严格为：
{{
  "name": "指标英文名(如 order_total)",
  "definition": "指标中文口径描述，必须以文档中该指标的中文名开头(如 '客单价：平均每单成交金额…')",
  "formula": "口径公式(如 SUM(order.amount))",
  "source_tables": ["来源表名"],
  "source_fields": ["来源字段(表.字段)"],
  "depends_on": ["依赖指标名"]
}}

只抽取有明确"指标名 + 口径/公式"的内容；不确定的字段填空字符串或空数组。
文档片段：
---BEGIN---
{block}
---END---"""


def validate_glossary(glossary: Any) -> Tuple[bool, List[str]]:
    """Check a glossary against the contract. Accepts a bare list of metrics
    (what ``extract_metrics_from_text`` returns) or the wrapped dict form.
    Returns (ok, issues)."""
    issues: List[str] = []
    if isinstance(glossary, dict):
        metrics = glossary.get("metrics")
    else:
        metrics = glossary
    if not isinstance(metrics, list):
        return False, ["glossary must be {\"metrics\": [...]} or a list of metrics"]
    seen: set = set()
    for i, m in enumerate(metrics):
        if not isinstance(m, dict):
            issues.append(f"metrics[{i}] is not an object")
            continue
        name = str(m.get("name", "")).strip()
        if not name:
            issues.append(f"metrics[{i}] missing 'name'")
        elif name in seen:
            issues.append(f"duplicate metric name {name!r}")
        seen.add(name)
        for key in GLOSSARY_KEYS:
            if key not in m:
                issues.append(f"metrics[{i}] missing key {key!r}")
    return len(issues) == 0, issues


def split_doc_blocks(text: str) -> List[str]:
    """Split a document into self-contained chunks (blank-line / heading aware)."""
    blocks: List[str] = []
    cur: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:  # blank line -> block boundary
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if len(b) >= 8]


def _parse_llm_json(raw: str) -> List[Dict[str, Any]]:
    """Parse the LLM output: bare JSON array or a ```json fenced block."""
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # find the first '[' ... last ']' span as a fallback
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [
        {k: (m.get(k) or ([] if k in ("source_tables", "source_fields", "depends_on") else ""))
         for k in GLOSSARY_KEYS}
        for m in data if isinstance(m, dict)
    ]


def _default_generate(prompt: str) -> str:
    """Real glm-5.3 via the project's LLM role (only chat-capable model)."""
    from hugegraph_llm.models.llms.init_llm import LLMs

    return str(LLMs().get_text2gql_llm().generate(prompt=prompt))


def extract_metrics_from_text(
    text: str,
    generate: Optional[Callable[[str], str]] = None,
    max_blocks: int = 20,
) -> List[Dict[str, Any]]:
    """Extract metric glossary entries from a document (LLM, failure-tolerant)."""
    generate = generate or _default_generate
    out: Dict[str, Dict[str, Any]] = {}
    for block in split_doc_blocks(text)[:max_blocks]:
        try:
            raw = generate(_EXTRACT_PROMPT.format(block=block))
            for entry in _parse_llm_json(raw):
                name = str(entry.get("name", "")).strip()
                if name:
                    out.setdefault(name, entry)
        except Exception as exc:  # noqa: BLE001 - LLM flake -> skip block
            logging.getLogger(__name__).warning("extract block failed: %s", exc)
    return list(out.values())


def merge_glossary(
    metric_payload: Dict[str, Any],
    glossary: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Merge authoritative 口径 from the glossary into the platform metric payload.

    Glossary wins for definition/formula (docs are the authoritative 口径
    source); source tables/fields are only filled in when empty. Returns
    (merged_payload, {"filled": n, "conflicts": n}).
    """
    by_name = {str(m.get("name", "")): m for m in metric_payload.get("metrics", [])}
    filled = conflicts = 0
    for g in glossary:
        name = str(g.get("name", "")).strip()
        target = by_name.get(name)
        if target is None:
            continue  # glossary-only metrics are not added by default
        for key in ("definition", "formula"):
            val = str(g.get(key, "")).strip()
            if val:
                if str(target.get(key, "")).strip() and str(target[key]) != val:
                    conflicts += 1
                target[key] = val
                filled += 1
        for key in ("source_tables", "source_fields", "depends_on"):
            vals = list(g.get(key) or [])
            if vals and not list(target.get(key) or []):
                target[key] = vals
                filled += 1
    return metric_payload, {"filled": filled, "conflicts": conflicts}
