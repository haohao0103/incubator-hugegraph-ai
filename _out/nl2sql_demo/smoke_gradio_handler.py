"""Smoke-test the Gradio nl2sql handler without launching the UI.

Calls ``_query_handler`` (the exact function wired to the "查询 Query" button)
against the live ``kg-rag`` graph:

1. deterministic path  -- pasted candidates, no LLM (fast, asserted);
2. golden-feedback path-- store_golden=True, expects the golden_feedback stage
   (the demo seed added golden records for these questions);
3. live path           -- no candidates -> real glm-5.3 generation (guarded,
   never asserted on content);
4. markdown renderer   -- non-nl2sql route fallback.

Run from the repo root::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/nl2sql_demo/smoke_gradio_handler.py 2>&1 \\
        | tee _out/nl2sql_demo/logs/smoke_gradio.log
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.demo.rag_demo.unified_io_block import (  # noqa: E402
    EXAMPLE_NL2SQL_CANDIDATES,
    _query_handler,
    _render_query_markdown,
)

GOLDEN_ANSWER = (
    "SELECT city, SUM(order.amount) AS order_amount FROM order "
    "GROUP BY city ORDER BY order_amount DESC"
)


def main() -> int:
    # 1) deterministic candidates -> the group-by SQL must win (it is
    #    metric-caliber-correct; the payment SUM is unrelated; amnt is invalid)
    md, raw = _query_handler("各城市订单总额", "nl2sql", "", 5, EXAMPLE_NL2SQL_CANDIDATES, False)
    assert "nl2sql" in md and "sql_voting" in md, md[:200]
    assert GOLDEN_ANSWER in md, "expected group-by winner in markdown"
    assert '"' + GOLDEN_ANSWER + '"' in raw, "expected winner in raw JSON"
    print("PASS[1] deterministic candidates -> group-by SQL wins")
    print(md)
    print("----")

    # 2) golden feedback loop: store_golden=True -> golden_feedback stage shows
    md2, raw2 = _query_handler("各城市订单总额", "nl2sql", "", 5, EXAMPLE_NL2SQL_CANDIDATES, True)
    assert "golden_feedback" in md2, "expected golden_feedback stage when store_golden=True"
    assert "stored" in md2, "expected stored payload in golden_feedback"
    print("PASS[2] golden feedback stage present when store_golden=True")
    print(md2)
    print("----")

    # 3) live LLM path (guarded, no content assertion)
    try:
        md3, raw3 = _query_handler("各城市订单总额", "nl2sql", "", 5, "", False)
        assert isinstance(md3, str) and isinstance(raw3, str)
        print("PASS[3] live path returned; source line present:", "llm" in raw3 or "provided" in raw3)
        print(md3[:600])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"NOTE[3] live path degraded gracefully: {exc}")
    print("----")

    # 4) non-nl2sql markdown fallback renderer
    out = _render_query_markdown({"route": "precise", "answer": "gremlin result"})
    assert "precise" in out and "gremlin result" in out
    print("PASS[4] non-nl2sql markdown fallback renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
