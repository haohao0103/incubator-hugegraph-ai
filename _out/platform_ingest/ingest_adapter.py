"""Platform data -> HugeGraph metadata KG ingestion adapter (骨架).

The upstream platform delivers warehouse metadata as JSON/CSV (tables, fields,
metrics) plus Feishu docs. This adapter is the "structured data" path:

1. normalize  -- accepts loose/aliased field names (中文列名 tolerated) and
                 produces the canonical payloads that
                 ``unified_convert.convert_catalog_to_graph`` /
                 ``convert_metric_to_graph`` already consume;
2. ingest     -- builds (or resets) a target graph, ensures the KG_SCHEMA with
                 the full index set, converts the payloads and writes the
                 vertices/edges via Gremlin (name is the PRIMARY KEY, so it is
                 auto-indexed; property filters are index-accelerated).

It is intentionally decoupled from the running Gradio/HTTP service so a demo
or a staging graph (e.g. ``kg_platform``) can be populated without touching
the ``kg_rag`` demo slice.

CLI::

    /Users/mac/.workbuddy/binaries/python/envs/hg-e2e/bin/python \\
        _out/platform_ingest/ingest_adapter.py \\
        --catalog sample_data/platform_catalog.json \\
        --metrics sample_data/platform_metrics.json \\
        --graph kg_platform --domain platform --reset

Exit codes: 0 ok, 1 data/CLI error, 2 graph unreachable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.api.unified_convert import (  # noqa: E402
    KG_SCHEMA,
    convert_catalog_to_graph,
    convert_metric_to_graph,
)
from hugegraph_llm.config import huge_settings  # noqa: E402
from hugegraph_llm.operators.hugegraph_op.schema_manager import (  # noqa: E402
    SchemaManager,
)
from pyhugegraph.client import PyHugeClient  # noqa: E402

# ---------------------------------------------------------------------------
# 1) normalize: alias-tolerant field name mapping
# ---------------------------------------------------------------------------
_TABLE_NAME = ("name", "table_name", "表名", "库表名")
_TABLE_COMMENT = ("comment", "表注释", "注释", "描述")
_FIELD_NAME = ("name", "field_name", "column_name", "字段名", "列名")
_FIELD_COMMENT = ("comment", "字段注释", "注释", "描述")
_FIELD_TYPE = ("type", "data_type", "数据类型")
_METRIC_NAME = ("name", "metric_name", "指标名")
_METRIC_DEFINITION = ("definition", "定义", "口径", "指标说明")
_METRIC_FORMULA = ("formula", "公式", "口径公式")
_METRIC_TABLES = ("source_tables", "来源表", "源表")
_METRIC_FIELDS = ("source_fields", "来源字段", "源字段")
_METRIC_DEPENDS = ("depends_on", "依赖指标", "依赖")


def _pick(obj: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def normalize_catalog(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Any reasonable platform catalog shape -> canonical convert payload."""
    tables = []
    for t in raw.get("tables") or raw.get("表") or []:
        fields = []
        for f in t.get("fields") or t.get("字段") or []:
            fields.append(
                {
                    "name": str(_pick(f, *_FIELD_NAME)),
                    "comment": str(_pick(f, *_FIELD_COMMENT)),
                    "type": str(_pick(f, *_FIELD_TYPE)),
                }
            )
        tables.append(
            {
                "name": str(_pick(t, *_TABLE_NAME)),
                "comment": str(_pick(t, *_TABLE_COMMENT)),
                "fields": fields,
            }
        )
    return {"tables": tables}


def normalize_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    metrics = []
    for m in raw.get("metrics") or raw.get("指标") or []:
        metrics.append(
            {
                "name": str(_pick(m, *_METRIC_NAME)),
                "definition": str(_pick(m, *_METRIC_DEFINITION)),
                "formula": str(_pick(m, *_METRIC_FORMULA)),
                "source_tables": list(_pick(m, *_METRIC_TABLES, default=[]) or []),
                "source_fields": list(_pick(m, *_METRIC_FIELDS, default=[]) or []),
                "depends_on": list(_pick(m, *_METRIC_DEPENDS, default=[]) or []),
            }
        )
    return {"metrics": metrics}


# ---------------------------------------------------------------------------
# 2) ingest: build graph + ensure schema + write vertices/edges
# ---------------------------------------------------------------------------
_HSTORE_BODY = {
    "gremlin.graph": "org.apache.hugegraph.HugeFactory",
    "backend": "hstore",
    "serializer": "binary",
    "store": None,  # filled per graph
    "pd.peers": "127.0.0.1:8686",
}


def _base_url() -> str:
    url = str(huge_settings.graph_url)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _auth():
    if huge_settings.graph_user and huge_settings.graph_pwd:
        return (huge_settings.graph_user, huge_settings.graph_pwd)
    return None


def _quote(value: Any) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _wait_schema_gone(
    base: str, auth: Any, graph: str, path: str, timeout_s: float = 60.0
) -> bool:
    import time as _time

    import requests

    end = _time.time() + timeout_s
    while _time.time() < end:
        try:
            if requests.get(f"{base}/graphs/{graph}{path}", auth=auth, timeout=10).status_code == 404:
                return True
        except Exception:  # noqa: BLE001
            pass
        _time.sleep(0.5)
    return False


def _wait_task(base: str, auth: Any, graph: str, task_id: Any, timeout_s: float = 60.0) -> str:
    import time as _time

    import requests

    end = _time.time() + timeout_s
    while _time.time() < end:
        try:
            t = requests.get(f"{base}/graphs/{graph}/tasks/{task_id}", auth=auth, timeout=10)
            if t.status_code == 200:
                status = str(t.json().get("task_status", "")).lower()
                if status in ("success", "cancelled", "failed"):
                    return status
        except Exception:  # noqa: BLE001
            pass
        _time.sleep(0.5)
    return "running"


def _drop_kg_schema(client: PyHugeClient, graph: str) -> None:
    """Drop the KG edge/vertex labels so ensure_schema can rebuild them fresh.
    Edge labels first (vertex labels are referenced by them)."""
    import requests

    base, auth = _base_url(), _auth()
    for el in ("hasColumn", "computedFrom", "computedFromField", "dependsOn"):
        r = requests.delete(f"{base}/graphs/{graph}/schema/edgelabels/{el}", auth=auth, timeout=30)
        if r.status_code in (200, 202):
            _wait_task(base, auth, graph, r.json().get("task_id"))
        _wait_schema_gone(base, auth, graph, f"/schema/edgelabels/{el}")
    for vl in ("Metric", "Field", "Table", "Query"):
        r = requests.delete(f"{base}/graphs/{graph}/schema/vertexlabels/{vl}", auth=auth, timeout=30)
        if r.status_code in (200, 202):
            _wait_task(base, auth, graph, r.json().get("task_id"))
        _wait_schema_gone(base, auth, graph, f"/schema/vertexlabels/{vl}")


def ensure_graph(client: PyHugeClient, graph: str, reset: bool = False) -> None:
    """Create the graph if missing (hstore backend). ``reset`` clears the data
    and drops the KG schema labels so a subsequent ensure_schema rebuilds them
    with the current KG_SCHEMA (e.g. newly added properties). Drop graph is
    async and unreliable, so we never delete the graph itself."""
    import requests

    base, auth = _base_url(), _auth()
    body = dict(_HSTORE_BODY, store=graph, name=graph)

    resp = requests.post(f"{base}/graphs/{graph}", json=body, auth=auth, timeout=30)
    if resp.status_code in (200, 201):
        exists = False
    elif resp.status_code == 400 and "conflicts with existed graph" in str(resp.text):
        exists = True  # already exists; server reports 400, not 409
    else:
        raise RuntimeError(
            f"create graph {graph} failed: HTTP {resp.status_code} {resp.text[:160]}"
        )
    if reset and exists:
        client.graphs().clear_graph_all_data()
        _drop_kg_schema(client, graph)
        print(f"  graph '{graph}' data+schema reset")


def write_vertices(client: PyHugeClient, vertices: List[Dict[str, Any]]) -> int:
    for v in vertices:
        label = v["label"]
        props = v.get("properties", {})
        p_str = "".join(f".property('{k}', {_quote(val)})" for k, val in props.items())
        client.gremlin().exec(f"g.addV('{label}'){p_str}")
    return len(vertices)


def write_edges(client: PyHugeClient, edges: List[Dict[str, Any]]) -> int:
    written = 0
    for e in edges:
        label = e["label"]
        out_label, out_name = e["outV"].split(":", 1)
        in_label, in_name = e["inV"].split(":", 1)
        try:
            client.gremlin().exec(
                f"g.V().has('{out_label}','name',{_quote(out_name)}).as('s')"
                f".V().has('{in_label}','name',{_quote(in_name)})"
                f".addE('{label}').from('s')"
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 - dangling refs are skipped
            print(f"  skip edge {label} {out_name}->{in_name}: {str(exc)[:80]}")
    return written


def ingest_platform(
    catalog_payload: Dict[str, Any],
    metric_payload: Dict[str, Any],
    graph: str,
    domain: str = "default",
    source: str = "platform",
    reset: bool = False,
    client: Optional[PyHugeClient] = None,
) -> Dict[str, int]:
    """Full local ingest: ensure graph/schema, convert, write. Returns counts."""
    client = client or PyHugeClient(
        url=huge_settings.graph_url, graph=graph,
        user=huge_settings.graph_user, pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )
    ensure_graph(client, graph, reset=reset)

    mgr = SchemaManager(graph, client=client)
    summary = mgr.ensure_schema(KG_SCHEMA)
    print(f"  schema ensured: {summary}")

    cat = convert_catalog_to_graph(catalog_payload, domain=domain, source=source)
    met = convert_metric_to_graph(metric_payload, domain=domain, source=source)
    vertices = cat["vertices"] + met["vertices"]
    edges = cat["edges"] + met["edges"]

    nv = write_vertices(client, vertices)
    ne = write_edges(client, edges)
    return {"vertices": nv, "edges": ne}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="platform catalog JSON (tables+fields)")
    parser.add_argument("--metrics", required=True, help="platform metrics JSON")
    parser.add_argument("--graph", default="kg_platform", help="target graph name (no hyphens)")
    parser.add_argument("--domain", default="platform")
    parser.add_argument("--reset", action="store_true", help="clear the graph before ingest")
    args = parser.parse_args()

    try:
        with open(args.catalog, encoding="utf-8") as f:
            catalog_payload = normalize_catalog(json.load(f))
        with open(args.metrics, encoding="utf-8") as f:
            metric_payload = normalize_metrics(json.load(f))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR reading inputs: {exc}")
        return 1

    try:
        counts = ingest_platform(
            catalog_payload, metric_payload,
            graph=args.graph, domain=args.domain, reset=args.reset,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR ingest: {exc}")
        return 2

    print(f"PASS: ingested into '{args.graph}' "
          f"({counts['vertices']} vertices / {counts['edges']} edges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
