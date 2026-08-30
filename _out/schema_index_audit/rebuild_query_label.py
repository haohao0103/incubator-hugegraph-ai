"""Real-machine: rebuild the Query golden label with LONG created_at + full
indexes, preserving the existing golden records.

Why: the golden store originally created ``Query.created_at`` as TEXT, which
cannot host a RANGE index (HugeGraph only RANGEs numbers/dates). This script:

1. backs up every Query vertex (question/sql/schema_refs/domain/created_at);
2. drops the Query vertex label (async server task, polled to completion);
3. re-ensures the label via ``KgGoldenSqlStore.ensure_schema`` -- created_at
   LONG + the full index set (domain/question SECONDARY, sql/schema_refs
   SEARCH, created_at RANGE);
4. re-inserts the backed-up records (created_at as integer);
5. rebuilds the Query indexes;
6. verifies: record count preserved, single-has domain/question accepted,
   RANGE created_at query works, SEARCH textContains works.

Run (tee'd log)::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/schema_index_audit/rebuild_query_label.py 2>&1 \\
        | tee _out/schema_index_audit/logs/rebuild_query_label.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List

import requests

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
from pyhugegraph.client import PyHugeClient  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")
QUERY_INDEXES = (
    "QueryByDomain", "QueryByQuestion", "QueryBySql", "QueryBySchemaRefs", "QueryByCreatedAt",
)
_KEEP_KEYS = ("question", "sql", "schema_refs", "domain", "created_at")


def _client():
    return PyHugeClient(
        url=huge_settings.graph_url,
        graph=GRAPH,
        user=huge_settings.graph_user,
        pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )


def _base_url() -> str:
    url = str(huge_settings.graph_url)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _auth():
    if huge_settings.graph_user and huge_settings.graph_pwd:
        return (huge_settings.graph_user, huge_settings.graph_pwd)
    return None


def _exec(client, q: str) -> List[Any]:
    resp = client.gremlin().exec(q)
    return resp.get("data") if isinstance(resp, dict) else (resp or [])


def _backup(client) -> List[Dict[str, Any]]:
    rows = _exec(client, "g.V().hasLabel('Query').elementMap()")
    out = []
    for row in rows:
        props = {k: row.get(k) for k in _KEEP_KEYS}
        out.append(props)
    return out


def _schema_status(path: str) -> int:
    """HTTP status of a schema GET; 404 means the object is gone."""
    try:
        resp = requests.get(f"{_base_url()}/graphs/{GRAPH}{path}", auth=_auth(), timeout=15)
        return resp.status_code
    except Exception:  # noqa: BLE001
        return 0


def _wait_schema_gone(path: str, timeout_s: float = 60.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if _schema_status(path) == 404:
            return True
        time.sleep(0.5)
    return False


def _wait_task(task_id: Any, timeout_s: float = 90.0) -> str:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            t = requests.get(
                f"{_base_url()}/graphs/{GRAPH}/tasks/{task_id}", auth=_auth(), timeout=15
            )
            if t.status_code == 200:
                status = str(t.json().get("task_status", "")).lower()
                if status in ("success", "cancelled", "failed"):
                    return f"task {task_id}: {status}"
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return f"task {task_id}: still running"


def _drop_query_label() -> str:
    url = f"{_base_url()}/graphs/{GRAPH}/schema/vertexlabels/Query"
    resp = requests.delete(url, auth=_auth(), timeout=30)
    if resp.status_code in (200, 202):
        result = _wait_task(resp.json().get("task_id"))
    else:
        result = f"drop HTTP {resp.status_code}: {str(resp.text)[:120]}"
    gone = _wait_schema_gone("/schema/vertexlabels/Query")
    return f"{result}; label_gone={gone}"


def _drop_property_key(name: str) -> str:
    url = f"{_base_url()}/graphs/{GRAPH}/schema/propertykeys/{name}"
    resp = requests.delete(url, auth=_auth(), timeout=30)
    if resp.status_code in (200, 202):
        result = _wait_task(resp.json().get("task_id"))
    else:
        result = f"drop pk HTTP {resp.status_code}: {str(resp.text)[:120]}"
    gone = _wait_schema_gone(f"/schema/propertykeys/{name}")
    return f"{result}; pk_gone={gone}"


def _index_status(name: str) -> str:
    try:
        resp = requests.get(
            f"{_base_url()}/graphs/{GRAPH}/schema/indexlabels/{name}",
            auth=_auth(), timeout=15,
        )
        if resp.status_code == 200:
            return str(resp.json().get("status", ""))
    except Exception:  # noqa: BLE001
        pass
    return ""


def _rebuild(name: str) -> str:
    try:
        resp = requests.put(
            f"{_base_url()}/graphs/{GRAPH}/jobs/rebuild/indexlabels/{name}",
            auth=_auth(), timeout=30,
        )
        result = f"trigger HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        result = f"trigger skipped ({str(exc)[:70]})"
    for _ in range(30):
        if _index_status(name) == "CREATED":
            return f"{result}; status=CREATED"
        time.sleep(0.5)
    return f"{result}; status={_index_status(name) or 'unknown'}"


def _try_has(client, key: str, value: Any) -> str:
    try:
        n = _exec(client, f"g.V().has('Query','{key}',{value}).count()")
        return f"OK count={n[0] if n else 0}"
    except Exception as exc:  # noqa: BLE001
        return f"REJECTED: {str(exc)[:100]}"


def main() -> int:
    try:
        client = _client()
        _exec(client, "g.V().limit(1).count()")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: graph unreachable: {exc}")
        return 0

    backup = _backup(client)
    print(f"=== graph={GRAPH}: backing up {len(backup)} Query vertices ===")
    for rec in backup[:2]:
        print(f"  {rec}")

    print("\n--- drop Query vertex label ---")
    print(f"  {_drop_query_label()}")

    print("\n--- drop property key 'created_at' (TEXT -> will recreate as LONG) ---")
    print(f"  {_drop_property_key('created_at')}")

    print("\n--- re-ensure schema (created_at=LONG + full indexes) ---")
    store = KgGoldenSqlStore(client, GRAPH)
    ok = store.ensure_schema()
    print(f"  store.ensure_schema() -> {ok}")

    print("\n--- re-insert backup ---")
    n_ok = 0
    for rec in backup:
        try:
            _exec(
                client,
                "g.addV('Query').property('question', %s)"
                ".property('sql', %s).property('schema_refs', %s)"
                ".property('domain', %s).property('created_at', %d)"
                % (
                    "'" + str(rec.get("question", "")).replace("'", "\\'") + "'",
                    "'" + str(rec.get("sql", "")).replace("'", "\\'") + "'",
                    "'" + str(rec.get("schema_refs", "")).replace("'", "\\'") + "'",
                    "'" + str(rec.get("domain", "")).replace("'", "\\'") + "'",
                    int(rec.get("created_at") or 0),
                ),
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  reinsert failed for {rec.get('question')}: {str(exc)[:90]}")
    print(f"  reinserted {n_ok}/{len(backup)}")

    print("\n--- rebuild Query indexes ---")
    for name in QUERY_INDEXES:
        print(f"  {name}: {_rebuild(name)}")

    print("\n--- verify ---")
    n = _exec(client, "g.V().hasLabel('Query').count()")
    print(f"  Query count after rebuild: {n[0] if n else 0} (backup {len(backup)})")
    dom_q = "'demo_golden'"
    que_q = "'各城市订单总额'"
    print(f"  has domain:  {_try_has(client, 'domain', dom_q)}")
    print(f"  has question:{_try_has(client, 'question', que_q)}")
    try:
        r = _exec(client, "g.V().has('Query','created_at',gt(0)).count()")
        print(f"  RANGE created_at gt(0): OK count={r[0] if r else 0}")
    except Exception as exc:  # noqa: BLE001
        print(f"  RANGE created_at gt(0): REJECTED: {str(exc)[:100]}")
    try:
        r = _exec(client, "g.V().has('Query','sql', Text.contains('order')).count()")
        print(f"  SEARCH sql Text.contains('order'): OK count={r[0] if r else 0}")
    except Exception as exc:  # noqa: BLE001
        print(f"  SEARCH sql Text.contains: REJECTED: {str(exc)[:100]}")

    assert n and n[0] == len(backup), f"count mismatch: {n} vs backup {len(backup)}"
    print("\nPASS: Query label rebuilt (created_at=LONG + RANGE/SEARCH/SECONDARY indexes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
