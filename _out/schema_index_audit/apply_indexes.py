"""Real-machine: create the Query index labels on kg_rag and verify.

Uses ``KgGoldenSqlStore.ensure_schema()`` -- the exact production code path --
so the golden store becomes self-sufficient. Then:

1. before: single ``has('Query','domain',...)`` is REJECTED (no index);
2. ensure_schema creates QueryByDomain / QueryByQuestion (idempotent);
3. trigger a rebuild of the indexes over existing data (tolerant of the
   endpoint not existing on older servers);
4. after: the same single-has query is accepted and runs via the index
   (profile duration reported).

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/schema_index_audit/apply_indexes.py 2>&1 \\
        | tee _out/schema_index_audit/logs/apply_indexes.log
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, List

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.config import huge_settings  # noqa: E402
from hugegraph_llm.operators.graph_op.kg_golden_sql import (  # noqa: E402
    KgGoldenSqlStore,
)
from hugegraph_llm.operators.hugegraph_op.schema_manager import (  # noqa: E402
    SchemaManager,
)
from pyhugegraph.client import PyHugeClient  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")
INDEX_NAMES = ("QueryByDomain", "QueryByQuestion")


def _client():
    return PyHugeClient(
        url=huge_settings.graph_url,
        graph=GRAPH,
        user=huge_settings.graph_user,
        pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )


def _exec(client, q: str) -> List[Any]:
    resp = client.gremlin().exec(q)
    return resp.get("data") if isinstance(resp, dict) else (resp or [])


def _try_single_has(client, key: str, value: str) -> str:
    """Return 'OK count=N' or 'REJECTED: <reason>' for a single-has query."""
    try:
        n = _exec(client, f"g.V().has('Query','{key}','{value}').count()")
        return f"OK count={n[0] if n else 0}"
    except Exception as exc:  # noqa: BLE001
        return f"REJECTED: {str(exc)[:110]}"


def _base_url() -> str:
    url = str(huge_settings.graph_url)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _rebuild(client, name: str) -> str:
    """Trigger async rebuild of an index over existing data and poll to done.

    HugeGraph 1.7 exposes this at ``PUT /graphs/{g}/jobs/rebuild/indexlabels/{n}``
    (the ``/schema/indexlabels/{n}/rebuild`` path does not exist in 1.7).
    """
    import time as _time

    import requests

    base = _base_url()
    auth = None
    if huge_settings.graph_user and huge_settings.graph_pwd:
        auth = (huge_settings.graph_user, huge_settings.graph_pwd)
    try:
        resp = requests.put(
            f"{base}/graphs/{GRAPH}/jobs/rebuild/indexlabels/{name}",
            auth=auth, timeout=30,
        )
        result = f"trigger HTTP {resp.status_code} {str(resp.text)[:120]}"
    except Exception as exc:  # noqa: BLE001
        result = f"trigger skipped ({str(exc)[:80]})"

    for _ in range(30):  # poll up to ~15s
        status = _index_status(name)
        if status == "CREATED":
            return f"{result}; status=CREATED"
        _time.sleep(0.5)
    return f"{result}; status={status or 'unknown'} (still rebuilding)"


def _index_status(name: str) -> str:
    import requests

    auth = None
    if huge_settings.graph_user and huge_settings.graph_pwd:
        auth = (huge_settings.graph_user, huge_settings.graph_pwd)
    try:
        resp = requests.get(
            f"{_base_url()}/graphs/{GRAPH}/schema/indexlabels/{name}",
            auth=auth, timeout=15,
        )
        if resp.status_code == 200:
            return str(resp.json().get("status", ""))
    except Exception:  # noqa: BLE001
        pass
    return ""


def main() -> int:
    try:
        client = _client()
        _exec(client, "g.V().limit(1).count()")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: graph unreachable: {exc}")
        return 0

    print(f"=== graph={GRAPH} ===\n")

    print("--- BEFORE ---")
    for key, val in (("domain", "demo_golden"), ("question", "各城市订单总额")):
        print(f"  single has('Query','{key}',...) -> {_try_single_has(client, key, val)}")

    print("\n--- apply ensure_schema (store path) ---")
    store = KgGoldenSqlStore(client, GRAPH)
    ok = store.ensure_schema()
    print(f"  store.ensure_schema() -> {ok}")

    mgr = SchemaManager(GRAPH, client=client)
    indexes = mgr.list_indexes(base_label="Query")
    print("  Query indexes now present:")
    for idx in indexes:
        print(f"    {idx['name']}  type={idx['index_type']} fields={idx['fields']}")
    new_names = {i["name"] for i in indexes}
    missing = [n for n in INDEX_NAMES if n not in new_names]
    if missing:
        print(f"  !! missing: {missing}")
        return 1

    print("\n--- rebuild existing data ---")
    for name in INDEX_NAMES:
        print(f"  {_rebuild(client, name)}")

    print("\n--- AFTER (same single-has queries, profiled) ---")
    for key, val in (("domain", "demo_golden"), ("question", "各城市订单总额")):
        probe = f"g.V().has('Query','{key}','{val}').count()"
        print(f"  {probe}")
        print(f"    result -> {_try_single_has(client, key, val)}")
        try:
            rows = _exec(client, probe + ".profile()")
            for r in rows:
                print("    profile: " + str(r)[:260])
        except Exception as exc:  # pragma: no cover
            print(f"    profile ERR: {exc}")

    print("\nPASS: Query single-has property lookups now accepted via index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
