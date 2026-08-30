"""Ingest SchemaMetadata JSON into the HugeGraph warehouse-metadata KG.

This is the **persistent** write path (the production counterpart of the
in-memory ``/nl2sql/reload``): it writes table / field / metric / query
vertices and the edges the NL2SQL loader reads, so the knowledge lives in the
graph, not in process memory.

Write contract (must match ``nl2sql/hugegraph_schema_source.py`` loader):

    Table  vertex  label="Table",  name=<bare table name>       (PRIMARY_KEY)
    Field  vertex  label="Field",  name="<table>.<column>"      (PRIMARY_KEY)
    Metric vertex  label="Metric", name=<term>, formula, definition
    Query  vertex  label="Query",  schema_refs="t1;t2;..."      (co-occurrence)
    hasColumn edge         Table -> Field
    computedFromField edge Metric -> Field   (term <-> column binding; loader reads this)

Loader reads: Table/Field/Metric/Query vertices + computedFromField edges +
Query.schema_refs (co-occurrence). FKs are inferred by the loader from ``*_id``
column names (declared FK / lineage edges are NOT read yet -- v1 gap).

Usage:
  python scripts/ingest_metadata_to_hg.py --meta _out/sql_miner/meta.json \
      --url http://127.0.0.1:8081 --graph kg_rag [--clear]
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from urllib.error import HTTPError, URLError

LOG_PATH = "_out/hg_ingest/logs/ingest.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _request(url: str, method: str, payload=None, timeout: int = 60):
    """Proxy-free HTTP helper (localhost must bypass the env proxy)."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Accept-Encoding": "gzip"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        log(f"HTTP {exc.code} on {method} {url}: {raw[:400]}")
        raise


def _clear_graph(url: str, graph: str) -> None:
    """Wipe the whole graph's data (keeps schema labels).

    HG 1.7 PRIMARY_KEY vertices only accept their internal numeric id on
    DELETE (the REST query returns the display form '<label_id>:<name>'),
    and per-label delete / gremlin drop are unsupported on this instance, so
    the reliable reset is the official whole-graph clear. Only safe for a
    graph dedicated to warehouse metadata (kg_rag).
    """
    from urllib.parse import quote

    confirm = quote("I'm sure to delete all data")
    clear_url = f"{url.rstrip('/')}/graphs/{graph}/clear?confirm_message={confirm}"
    r = _request(clear_url, "DELETE")
    log(f"cleared graph {graph} (data wiped, schema kept): {r}")


def _fetch_ids(base: str, label: str) -> dict:
    """name -> vertex id for one label (PRIMARY_KEY ids are '<label_id>:<name>')."""
    r = _request(f"{base}/vertices?label={label}&limit=10000", "GET")
    return {v["properties"]["name"]: v["id"] for v in r.get("vertices", [])}


def _ensure_edge_label(url: str, graph: str, name: str, source: str, target: str):
    """Create an edge label if missing (idempotent)."""
    schema_base = f"{url.rstrip('/')}/graphs/{graph}/schema"
    labels = _request(f"{schema_base}/edgelabels", "GET").get("edgelabels", [])
    if any(lbl.get("name") == name for lbl in labels):
        return
    _request(f"{schema_base}/edgelabels", "POST", {
        "name": name, "source_label": source, "target_label": target,
        "frequency": "SINGLE", "sort_keys": [], "nullable_keys": [],
        "properties": [],
    })
    log(f"created edge label {name} ({source}->{target})")


def bare_table(full: str) -> str:
    """'dw.orders' -> 'orders' (kg_rag stores bare table names)."""
    return full.split(".")[-1] if "." in full else full


def ingest(meta: dict, url: str, graph: str, clear: bool = False) -> dict:
    """Write SchemaMetadata into the HugeGraph KG. Returns write counts."""
    base = f"{url.rstrip('/')}/graphs/{graph}/graph"

    if clear:
        _clear_graph(url, graph)

    tables = meta.get("tables", [])
    columns = meta.get("columns", [])
    terms = meta.get("terms", [])
    term_bindings = meta.get("term_bindings", [])
    query_logs = meta.get("query_logs", [])

    table_names = {bare_table(t["name"]) for t in tables}
    col_records = []
    for c in columns:
        col_records.append({
            "table": bare_table(c["table"]),
            "column": c["name"],
            "comment": c.get("comment", ""),
            "data_type": c.get("data_type", ""),
        })

    v_payload = []
    for t in tables:
        props = {"name": bare_table(t["name"])}
        if t.get("comment"):
            props["comment"] = t["comment"]
        v_payload.append({"label": "Table", "properties": props})
    for c in col_records:
        props = {"name": f"{c['table']}.{c['column']}", "type": c["data_type"]}
        if c["comment"]:
            props["comment"] = c["comment"]
        v_payload.append({"label": "Field", "properties": props})
    for tm in terms:
        props = {"name": tm["name"]}
        if tm.get("comment"):
            props["definition"] = tm["comment"]
        if tm.get("expression"):
            props["formula"] = tm["expression"]
        if tm.get("aliases"):
            props["aliases"] = ";".join(tm["aliases"])
        v_payload.append({"label": "Metric", "properties": props})
    for q in query_logs:
        tables_in_q = sorted({bare_table(x) for x in q if x in table_names})
        if len(tables_in_q) >= 2:
            v_payload.append({"label": "Query",
                              "properties": {"schema_refs": ";".join(tables_in_q)}})

    if v_payload:
        _request(f"{base}/vertices/batch", "POST", v_payload)
    n_q = len(v_payload) - len(tables) - len(col_records) - len(terms)

    # PRIMARY_KEY vertex ids are '<label_id>:<name>'; re-read to build edges.
    table_ids = _fetch_ids(base, "Table")
    field_ids = _fetch_ids(base, "Field")
    metric_ids = _fetch_ids(base, "Metric")

    e_payload = []
    for c in col_records:
        e_payload.append({
            "label": "hasColumn", "outV": table_ids.get(c["table"]),
            "outVLabel": "Table",
            "inV": field_ids.get(f"{c['table']}.{c['column']}"),
            "inVLabel": "Field", "properties": {},
        })
    term_table = {}
    for tb in term_bindings:
        if len(tb) == 2:
            term_table.setdefault(tb[0], []).append(tb[1])
    for tname, cols in term_table.items():
        for full_col in cols:
            tbl, _, col = full_col.rpartition(".")
            e_payload.append({
                "label": "computedFromField", "outV": metric_ids.get(tname),
                "outVLabel": "Metric",
                "inV": field_ids.get(f"{bare_table(tbl)}.{col}"),
                "inVLabel": "Field", "properties": {},
            })
    # drop any edge whose endpoint id could not be resolved (safety)
    e_payload = [e for e in e_payload if e["outV"] and e["inV"]]
    if e_payload:
        _request(f"{base}/edges/batch", "POST", e_payload)

    # ---- lineage (Table -> Table, upstream -> downstream) ----
    lineage = meta.get("lineage", [])
    if lineage:
        _ensure_edge_label(url, graph, "lineage", "Table", "Table")
        lg_payload = []
        for pair in lineage:
            if len(pair) == 2:
                lg_payload.append({
                    "label": "lineage", "outV": table_ids.get(bare_table(pair[0])),
                    "outVLabel": "Table",
                    "inV": table_ids.get(bare_table(pair[1])),
                    "inVLabel": "Table", "properties": {},
                })
        lg_payload = [e for e in lg_payload if e["outV"] and e["inV"]]
        if lg_payload:
            _request(f"{base}/edges/batch", "POST", lg_payload)

    # ---- synonyms (Metric <-> Metric, same meaning) ----
    synonyms = meta.get("synonyms", [])
    if synonyms:
        _ensure_edge_label(url, graph, "synonym", "Metric", "Metric")
        syn_payload = []
        for pair in synonyms:
            if len(pair) == 2:
                syn_payload.append({
                    "label": "synonym", "outV": metric_ids.get(pair[0]),
                    "outVLabel": "Metric",
                    "inV": metric_ids.get(pair[1]),
                    "inVLabel": "Metric", "properties": {},
                })
        syn_payload = [e for e in syn_payload if e["outV"] and e["inV"]]
        if syn_payload:
            _request(f"{base}/edges/batch", "POST", syn_payload)

    counts = {
        "tables": len(tables), "columns": len(col_records), "terms": len(terms),
        "queries": n_q,
        "has_column_edges": len(col_records),
        "term_bind_edges": sum(1 for e in e_payload if e["label"] == "computedFromField"),
        "lineage_edges": len(lineage),
        "synonym_edges": len(synonyms),
    }
    log(f"ingested: {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="SchemaMetadata JSON file")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--graph", default="kg_rag")
    ap.add_argument("--clear", action="store_true",
                    help="wipe Table/Field/Metric/Query vertices first")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log(f"=== hg ingest {args.graph} from {args.meta} ===")
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    ingest(meta, args.url, args.graph, clear=args.clear)
    log("ingest done (verify via loader next)")


if __name__ == "__main__":
    main()
