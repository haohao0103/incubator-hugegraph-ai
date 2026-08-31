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
    Caliber vertex label="Caliber", name=<caliber>, dimension, description
    CorrectionDecision vertex label="CorrectionDecision", name=<corr_id>,
        question, wrong_sql, correct_sql, correction_reason        (PRIMARY_KEY)
    hasColumn edge         Table -> Field
    computedFromField edge Metric -> Field   (term <-> column binding; loader reads this)
    hasCaliber edge        Metric -> Caliber (term <-> caliber; loader reads this)
    correctionAppliesTo* edge CorrectionDecision -> Metric|Field|Caliber
        (L3 correction provenance; loader reads this)

Loader reads: Table/Field/Metric/Query vertices + computedFromField edges +
Query.schema_refs (co-occurrence). FKs are inferred by the loader from ``*_id``
column names (declared FK / lineage edges are NOT read yet -- v1 gap).

After the writes, ``ingest()`` runs a **deterministic Datalog validation stage**
(``validate_metadata_rules``): lineage transitive closure, caliber propagation
along lineage, caliber-conflict and term-binding-integrity checks. The inputs
are the metadata facts themselves (no LLM, no probabilistic extractions), so
every derived fact is auditable. The report is logged, never blocks ingestion.

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


def _ensure_property_key(url: str, graph: str, name: str, data_type: str = "TEXT"):
    """Create a property key if missing (idempotent)."""
    schema_base = f"{url.rstrip('/')}/graphs/{graph}/schema"
    keys = _request(f"{schema_base}/propertykeys", "GET").get("propertykeys", [])
    if any(k.get("name") == name for k in keys):
        return
    _request(f"{schema_base}/propertykeys", "POST", {
        "name": name, "data_type": data_type,
        "cardinality": "SINGLE", "aggregate_type": "NONE",
    })
    log(f"created property key {name}")


def _ensure_vertex_label(url: str, graph: str, name: str, props: list,
                         pk: list = None, nullable: list = None):
    """Create a vertex label if missing (idempotent).

    ``pk=None`` => AUTOMATIC id strategy (used for the ``Query`` label, which
    carries only ``schema_refs`` and no primary-key property). Optional props
    are listed in ``nullable`` so vertices that omit them still write cleanly.

    Self-heal: if a label with this name already exists but with a *different*
    id strategy (e.g. a ``CUSTOMIZE_STRING`` ``Caliber`` left over from an
    earlier PoC on the same graph), we drop it first — otherwise vertex writes
    fail with "Can't customize vertex id when id strategy is 'PRIMARY_KEY'".
    """
    schema_base = f"{url.rstrip('/')}/graphs/{graph}/schema"
    labels = _request(f"{schema_base}/vertexlabels", "GET").get("vertexlabels", [])
    target_strategy = "PRIMARY_KEY" if pk else "AUTOMATIC"
    for l in labels:
        if l.get("name") != name:
            continue
        if l.get("id_strategy") != target_strategy:
            log(f"vertex label {name}: id_strategy mismatch "
                f"({l.get('id_strategy')} != {target_strategy}); dropping & rebuilding")
            _request(f"{schema_base}/vertexlabels/{name}", "DELETE")
            # drop is async on this HG; poll until the label is actually gone
            for _ in range(30):
                cur = _request(f"{schema_base}/vertexlabels", "GET").get("vertexlabels", [])
                if not any(x.get("name") == name for x in cur):
                    break
                time.sleep(1)
        return
    body = {"name": name, "properties": props,
            "nullable_keys": list(nullable or [])}
    if pk:
        body["id_strategy"] = "PRIMARY_KEY"
        body["primary_keys"] = pk
    else:
        body["id_strategy"] = "AUTOMATIC"
    _request(f"{schema_base}/vertexlabels", "POST", body)
    log(f"created vertex label {name}")


def _ensure_schema(url: str, graph: str):
    """Idempotently create the warehouse-metadata KG schema.

    Mirrors the loader contract in ``nl2sql/hugegraph_schema_source.py``:
    Table/Field/Metric (PRIMARY_KEY on ``name``) + Query (AUTOMATIC) and the
    four edge labels. ``ingest()`` assumes these labels exist, so we bootstrap
    them here to make the e2e reproducible on a fresh graph.
    """
    for pk, dt in [("name", "TEXT"), ("comment", "TEXT"), ("row_count", "LONG"),
                   ("type", "TEXT"), ("definition", "TEXT"),
                   ("formula", "TEXT"), ("aliases", "TEXT"),
                   ("schema_refs", "TEXT"), ("dimension", "TEXT"),
                   ("description", "TEXT"), ("question", "TEXT"),
                   ("wrong_sql", "TEXT"), ("correct_sql", "TEXT"),
                   ("correction_reason", "TEXT")]:
        _ensure_property_key(url, graph, pk, dt)
    _ensure_vertex_label(url, graph, "Table",
                         ["name", "comment", "row_count"], pk=["name"],
                         nullable=["comment", "row_count"])
    _ensure_vertex_label(url, graph, "Field",
                         ["name", "type", "comment"], pk=["name"],
                         nullable=["type", "comment"])
    _ensure_vertex_label(url, graph, "Metric",
                         ["name", "definition", "formula", "aliases"], pk=["name"],
                         nullable=["definition", "formula", "aliases"])
    _ensure_vertex_label(url, graph, "Query", ["schema_refs"], pk=None,
                         nullable=["schema_refs"])
    _ensure_vertex_label(url, graph, "Caliber",
                         ["name", "dimension", "description"], pk=["name"],
                         nullable=["dimension", "description"])
    _ensure_vertex_label(
        url, graph, "CorrectionDecision",
        ["name", "question", "wrong_sql", "correct_sql", "correction_reason"],
        pk=["name"], nullable=["question", "wrong_sql", "correct_sql",
                               "correction_reason"])
    _ensure_edge_label(url, graph, "hasColumn", "Table", "Field")
    _ensure_edge_label(url, graph, "computedFromField", "Metric", "Field")
    _ensure_edge_label(url, graph, "hasCaliber", "Metric", "Caliber")
    _ensure_edge_label(url, graph, "correctionAppliesToTerm", "CorrectionDecision", "Metric")
    _ensure_edge_label(url, graph, "correctionAppliesToField", "CorrectionDecision", "Field")
    _ensure_edge_label(url, graph, "correctionAppliesToCaliber", "CorrectionDecision", "Caliber")
    _ensure_edge_label(url, graph, "lineage", "Table", "Table")
    _ensure_edge_label(url, graph, "synonym", "Metric", "Metric")
    log(f"schema ensured for graph {graph}")


def bare_table(full: str) -> str:
    """'dw.orders' -> 'orders' (kg_rag stores bare table names)."""
    return full.split(".")[-1] if "." in full else full


# ============================================================
# Deterministic Datalog validation stage (Step 3 of the merge)
# ============================================================
# Runs *after* the metadata writes. The facts are the metadata itself
# (deterministic, no LLM), so every derived fact is auditable. The engine is
# ``hugegraph_llm.operators.graph_op.datalog_reasoner`` (semi-naive fixpoint).
#
# Rule direction conventions:
#   lineage(U, D)      = U is an upstream of D (D is derived from U)
#   upstream(D, U)     = D's upstream table is U
#   downstream(U, D)   = U's downstream table is D
#   table_has_caliber(T, C)  = table T directly defines caliber C on some term
#   table_caliber(T, C)      = T carries caliber C (direct or inherited)
DATALOG_RULES = [
    # --- lineage transitive closure ---
    "upstream(D, U) :- lineage(U, D).",
    "upstream(D, U) :- upstream(D, X), lineage(U, X).",
    "downstream(U, D) :- lineage(U, D).",
    "downstream(U, D) :- lineage(U, X), downstream(X, D).",
    # --- common merge target (two detail tables feeding the same summary
    #     table -> join path discovery, e.g. orders & payments -> ads_daily_sales)
    "co_dest(A, B, D) :- downstream(A, D), downstream(B, D).",
    # --- caliber propagation: a summary table inherits its upstream detail
    #     tables' calibers (ads_daily_sales.gmv keeps orders' GMV caliber) ---
    "table_caliber(T, C) :- table_has_caliber(T, C).",
    "table_caliber(T, C) :- table_caliber(P, C), lineage(P, T).",
    # --- caliber conflict candidates (same metric bound to 2+ calibers;
    #     engine has no inequality, so self-pairs C1==C2 are filtered in Python)
    "metric_caliber(M, C) :- metric_has_caliber(M, C).",
    "metric_caliber_pair(M, C1, C2) :- metric_caliber(M, C1), metric_caliber(M, C2).",
]


def validate_metadata_rules(meta: dict) -> dict:
    """Run the Datalog rule set over metadata facts; return a validation report.

    Report keys:
      lineage:    upstream / downstream closure maps (per table -> sorted list)
      calibers:   direct + propagated per-table caliber map
      co_dest:    common merge-target triples (A, B, D), self-pairs dropped
      conflicts:  metric -> [caliber pairs] where a metric carries >1 caliber
      integrity:  dangling_terms (term with no computedFromField binding),
                  dangling_calibers (caliber whose metric has no binding),
                  no_caliber_terms (terms without any caliber; advisory)
      stats:      engine stats (loaded facts / rules / derived / conflicts)

    Import of the Datalog engine is deferred so the ingest write path keeps
    working even if ``hugegraph_llm`` is not importable (validation degrades to
    a warning instead of blocking ingestion).
    """
    try:
        from hugegraph_llm.operators.graph_op.datalog_reasoner import DatalogReasonerOp
    except Exception as e:  # noqa: BLE001 -- validation is best-effort
        log(f"datalog validation unavailable (skip): {type(e).__name__}: {e}")
        return {"skipped": True, "reason": str(e)}

    calibers = meta.get("calibers", []) or []
    term_bindings = meta.get("term_bindings", []) or []
    terms = meta.get("terms", []) or []

    # term -> binding column (full name "dw.orders.gmv") / -> bare table
    bind_col = {tb[0]: tb[1] for tb in term_bindings if len(tb) == 2}
    table_of_term = {
        tname: bare_table(full.rpartition(".")[0])
        for tname, full in bind_col.items()
    }

    relations = []
    for pair in meta.get("lineage", []) or []:
        if len(pair) == 2:
            relations.append({"source": bare_table(pair[0]),
                              "target": bare_table(pair[1]),
                              "type": "lineage"})
    for c in calibers:
        tname = c.get("metric", "")
        cname = c.get("name", "")
        if not (cname and tname):
            continue
        # metric-level fact (caliber conflicts) + table-level fact (propagation)
        if tname in table_of_term:
            relations.append({"source": tname, "target": cname,
                              "type": "metric_has_caliber"})
            relations.append({"source": table_of_term[tname], "target": cname,
                              "type": "table_has_caliber"})

    ctx = DatalogReasonerOp().run({
        "entities": [],
        "relations": relations,
        "datalog_rules": DATALOG_RULES,
        "conflict_rules": [],
    })
    facts = ctx["datalog_facts"]  # [{"predicate", "args", "raw"}], orig names

    def by_pred(pred: str):
        return sorted({tuple(f["args"]) for f in facts if f["predicate"] == pred})

    # upstream/downstream closure maps
    upstream_map, downstream_map = {}, {}
    for d, u in by_pred("upstream"):
        upstream_map.setdefault(d, set()).add(u)
    for u, d in by_pred("downstream"):
        downstream_map.setdefault(u, set()).add(d)

    # caliber propagation map (direct + inherited)
    caliber_map = {}
    for t, c in by_pred("table_caliber"):
        caliber_map.setdefault(t, set()).add(c)

    # common merge targets, drop self-pairs (A == B)
    co_dest = sorted({(a, b, d) for a, b, d in by_pred("co_dest") if a != b})

    # caliber conflicts per metric, drop self-pairs (C1 == C2)
    conflicts = {}
    for m, c1, c2 in by_pred("metric_caliber_pair"):
        if c1 == c2:
            continue
        conflicts.setdefault(m, set()).add((c1, c2))

    # integrity: term bindings & caliber dangling + caliber coverage (advisory)
    bound_terms = set(bind_col)
    all_terms = {t.get("name", "") for t in terms}
    dangling_terms = sorted(all_terms - bound_terms)
    calib_metrics = {c.get("metric", "") for c in calibers if c.get("name")}
    dangling_calibers = sorted(calib_metrics - bound_terms)
    no_caliber_terms = sorted(all_terms - calib_metrics)

    report = {
        "skipped": False,
        "lineage": {
            "edges": len([r for r in relations if r["type"] == "lineage"]),
            "upstream": {k: sorted(v) for k, v in upstream_map.items()},
            "downstream": {k: sorted(v) for k, v in downstream_map.items()},
        },
        "calibers": {},
        "co_dest": co_dest,
        "conflicts": {k: sorted(v) for k, v in conflicts.items()},
        "integrity": {
            "dangling_terms": dangling_terms,
            "dangling_calibers": dangling_calibers,
            "no_caliber_terms": no_caliber_terms,
        },
        "stats": ctx["datalog_stats"],
    }
    # split per-table caliber map into direct vs inherited-by-lineage.
    # A table can be BOTH: ads_daily_sales directly defines 营收/客单价 calibers
    # and inherits GMV/实付 from orders/payments — so the split must be done on
    # (table, caliber) pairs, not per table.
    direct_pairs = {
        (r["source"], r["target"])
        for r in relations if r["type"] == "table_has_caliber"
    }
    all_pairs = {(t, c) for t, cs in caliber_map.items() for c in cs}
    inherited_pairs = all_pairs - direct_pairs

    def group(pairs):
        out = {}
        for t, c in sorted(pairs):
            out.setdefault(t, []).append(c)
        return out

    report["calibers"]["direct"] = group(direct_pairs)
    report["calibers"]["inherited"] = group(inherited_pairs)
    return report


def _log_validation(report: dict) -> None:
    """Log the validation report as a compact summary (never blocks ingest)."""
    if report.get("skipped"):
        log(f"validation skipped: {report.get('reason')}")
        return
    lg = report["lineage"]
    ca = report["calibers"]
    co = report["co_dest"]
    cf = report["conflicts"]
    ig = report["integrity"]
    log("datalog validation: "
        f"lineage={lg['edges']} upstream_tables={len(lg['upstream'])} "
        f"downstream_tables={len(lg['downstream'])} "
        f"caliber_tables={len(ca['direct'])} inherited={len(ca['inherited'])} "
        f"co_dest={len(co)} conflicts={sum(len(v) for v in cf.values())} "
        f"dangling_terms={ig['dangling_terms']} "
        f"dangling_calibers={ig['dangling_calibers']} "
        f"no_caliber_terms={len(ig['no_caliber_terms'])}")


def ingest(meta: dict, url: str, graph: str, clear: bool = False,
           diff: bool = False) -> dict:
    """Write SchemaMetadata into the HugeGraph KG. Returns write counts.

    ``diff=True`` enables incremental sync: only tables/columns/terms that are
    not already in the graph are written (PRIMARY_KEY upsert makes re-runs
    idempotent; diff skips the already-present ones entirely). Existing
    vertices are left untouched, so deleted metadata is *not* removed.
    """
    base = f"{url.rstrip('/')}/graphs/{graph}/graph"

    # Idempotent schema bootstrap: the warehouse-metadata KG needs its vertex
    # labels (Table/Field/Metric/Query) + property keys + the four edge labels
    # before any data can be written. ``ingest`` previously assumed they
    # existed, which broke on a fresh graph (HTTP 400 "Undefined vertex label").
    _ensure_schema(url, graph)

    if clear:
        _clear_graph(url, graph)

    tables = meta.get("tables", [])
    columns = meta.get("columns", [])
    terms = meta.get("terms", [])
    term_bindings = meta.get("term_bindings", [])
    query_logs = meta.get("query_logs", [])
    calibers = meta.get("calibers", [])
    corrections = meta.get("corrections", [])

    # incremental diff: existing names per label (only when diff=True)
    existing = None
    if diff:
        existing = {
            "Table": set(_fetch_ids(base, "Table")),
            "Field": set(_fetch_ids(base, "Field")),
            "Metric": set(_fetch_ids(base, "Metric")),
            "Caliber": set(_fetch_ids(base, "Caliber")),
            "CorrectionDecision": set(_fetch_ids(base, "CorrectionDecision")),
        }

    def _new_table(name):
        return not (existing and name in existing["Table"])

    def _new_field(fname):
        return not (existing and fname in existing["Field"])

    def _new_term(name):
        return not (existing and name in existing["Metric"])

    def _new_caliber(name):
        return not (existing and name in existing["Caliber"])

    def _new_correction(name):
        return not (existing and name in existing["CorrectionDecision"])

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
        if not _new_table(bare_table(t["name"])):
            continue
        # HG vertex labels declare non-null properties, so always send a value
        # (empty string / 0) for every declared property.
        props = {"name": bare_table(t["name"]),
                 "comment": t.get("comment") or "",
                 "row_count": int(t.get("row_count") or 0)}
        v_payload.append({"label": "Table", "properties": props})
    for c in col_records:
        fname = f"{c['table']}.{c['column']}"
        if not _new_field(fname):
            continue
        props = {"name": fname,
                 "type": c["data_type"] or "",
                 "comment": c["comment"] or ""}
        v_payload.append({"label": "Field", "properties": props})
    for tm in terms:
        if not _new_term(tm["name"]):
            continue
        aliases = tm.get("aliases") or []
        props = {"name": tm["name"],
                 "definition": tm.get("comment") or "",
                 "formula": tm.get("expression") or "",
                 "aliases": ";".join(aliases) if aliases else ""}
        v_payload.append({"label": "Metric", "properties": props})
    for c in calibers:
        if not _new_caliber(c.get("name", "")):
            continue
        props = {"name": c.get("name", ""),
                 "dimension": c.get("dimension", "") or "",
                 "description": c.get("description", "") or ""}
        if props["name"]:
            v_payload.append({"label": "Caliber", "properties": props})
    for corr in corrections:
        cid = corr.get("id") or corr.get("name")
        if not cid or not _new_correction(cid):
            continue
        props = {"name": cid,
                 "question": corr.get("question", "") or "",
                 "wrong_sql": corr.get("wrong_sql", "") or "",
                 "correct_sql": corr.get("correct_sql", "") or "",
                 "correction_reason": corr.get("correction_reason", "") or ""}
        v_payload.append({"label": "CorrectionDecision", "properties": props})
    for q in (query_logs if not diff else []):
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
    caliber_ids = _fetch_ids(base, "Caliber")

    # In diff mode, only build edges for the vertices just written (existing
    # vertices already have their edges from a previous run).
    new_field_names = {
        f"{c['table']}.{c['column']}"
        for c in col_records
        if _new_field(f"{c['table']}.{c['column']}")
    }
    new_term_names = {tm["name"] for tm in terms if _new_term(tm["name"])}
    new_caliber_names = {c.get("name", "") for c in calibers
                         if _new_caliber(c.get("name", ""))}

    e_payload = []
    for c in col_records:
        fname = f"{c['table']}.{c['column']}"
        if diff and fname not in new_field_names:
            continue
        e_payload.append({
            "label": "hasColumn", "outV": table_ids.get(c["table"]),
            "outVLabel": "Table",
            "inV": field_ids.get(fname),
            "inVLabel": "Field", "properties": {},
        })
    term_table = {}
    for tb in term_bindings:
        if len(tb) == 2:
            term_table.setdefault(tb[0], []).append(tb[1])
    for tname, cols in term_table.items():
        for full_col in cols:
            tbl, _, col = full_col.rpartition(".")
            fname = f"{bare_table(tbl)}.{col}"
            if diff and (fname not in new_field_names
                         and tname not in new_term_names):
                continue
            e_payload.append({
                "label": "computedFromField", "outV": metric_ids.get(tname),
                "outVLabel": "Metric",
                "inV": field_ids.get(fname),
                "inVLabel": "Field", "properties": {},
            })
    # drop any edge whose endpoint id could not be resolved (safety)
    e_payload = [e for e in e_payload if e["outV"] and e["inV"]]
    if e_payload:
        _request(f"{base}/edges/batch", "POST", e_payload)

    # ---- hasCaliber (Metric -> Caliber, term <-> caliber binding) ----
    has_caliber_edges = 0
    hc_payload = []
    for c in calibers:
        cname, mname = c.get("name", ""), c.get("metric", "")
        if diff and (cname not in new_caliber_names
                     and mname not in new_term_names):
            continue
        if cname and mname:
            hc_payload.append({
                "label": "hasCaliber", "outV": metric_ids.get(mname),
                "outVLabel": "Metric",
                "inV": caliber_ids.get(cname),
                "inVLabel": "Caliber", "properties": {},
            })
    hc_payload = [e for e in hc_payload if e["outV"] and e["inV"]]
    if hc_payload:
        _request(f"{base}/edges/batch", "POST", hc_payload)
        has_caliber_edges = len(hc_payload)

    # ---- correctionAppliesTo* (CorrectionDecision -> Metric|Field|Caliber) ----
    # L3 纠错 provenance：纠错挂到相关术语/字段/口径。召回时沿语义边传播，
    # 同义词/指标链/口径可达节点上的纠错也能被召回（见 e2e_sql_gen）。
    corr_ids = _fetch_ids(base, "CorrectionDecision")
    correction_edges = 0
    for corr in corrections:
        cid = corr.get("id") or corr.get("name")
        if not cid:
            continue
        for target in corr.get("applies_to", []) or []:
            # target 形如 "term:成交额" / "field:payments.pay_amount" /
            # "caliber:GMV口径"，映射到对应端点 id
            kind, _, name = target.partition(":")
            if kind == "term":
                inV, inLbl = metric_ids.get(name), "Metric"
                edge_label = "correctionAppliesToTerm"
            elif kind == "caliber":
                inV, inLbl = caliber_ids.get(name), "Caliber"
                edge_label = "correctionAppliesToCaliber"
            elif kind == "field":
                tbl, _, col = name.rpartition(".")
                fname = f"{bare_table(tbl)}.{col}"
                inV, inLbl = field_ids.get(fname), "Field"
                edge_label = "correctionAppliesToField"
            else:
                continue
            if not inV:
                continue
            try:
                _request(f"{base}/edges", "POST", {
                    "label": edge_label, "outV": corr_ids.get(cid),
                    "outVLabel": "CorrectionDecision",
                    "inV": inV, "inVLabel": inLbl, "properties": {},
                })
                correction_edges += 1
            except Exception as e:  # noqa: BLE001 -- duplicate edge: skip
                log(f"correction edge skip ({target}): {type(e).__name__}")

    # ---- lineage (Table -> Table, upstream -> downstream); skip in diff mode
    # (existing endpoints already carry their edges from a previous run)
    lineage = meta.get("lineage", [])
    if lineage and not diff:
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

    # ---- synonyms (Metric <-> Metric); skip in diff mode ----
    synonyms = meta.get("synonyms", [])
    if synonyms and not diff:
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

    n_written = {"Table": 0, "Field": 0, "Metric": 0, "Query": 0,
                 "Caliber": 0, "CorrectionDecision": 0}
    for v in v_payload:
        n_written[v["label"]] = n_written.get(v["label"], 0) + 1
    counts = {
        "tables": n_written.get("Table", 0),
        "columns": n_written.get("Field", 0),
        "terms": n_written.get("Metric", 0),
        "queries": n_written.get("Query", 0),
        "calibers": n_written.get("Caliber", 0),
        "corrections": n_written.get("CorrectionDecision", 0),
        "has_column_edges": sum(1 for e in e_payload if e["label"] == "hasColumn"),
        "term_bind_edges": sum(1 for e in e_payload if e["label"] == "computedFromField"),
        "has_caliber_edges": has_caliber_edges,
        "correction_edges": correction_edges,
        "lineage_edges": len(lg_payload) if "lg_payload" in dir() and diff is False else 0,
        "synonym_edges": len(syn_payload) if "syn_payload" in dir() and diff is False else 0,
    }
    log(f"ingested: {counts}")

    # ---- deterministic Datalog validation stage (Step 3) ----
    # Facts = the metadata just written (no LLM). Report is logged, never
    # blocks ingestion; a hard failure here means the rules themselves are buggy.
    try:
        v_report = validate_metadata_rules(meta)
        _log_validation(v_report)
    except Exception as e:  # noqa: BLE001 -- validation must not break ingest
        log(f"datalog validation failed (non-blocking): {type(e).__name__}: {e}")

    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="SchemaMetadata JSON file")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--graph", default="kg_rag")
    ap.add_argument("--clear", action="store_true",
                    help="wipe the whole graph data first (kg_rag-dedicated)")
    ap.add_argument("--diff", action="store_true",
                    help="incremental: only write tables/columns/terms not yet in the graph")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log(f"=== hg ingest {args.graph} from {args.meta} ===")
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    ingest(meta, args.url, args.graph, clear=args.clear, diff=args.diff)
    log("ingest done (verify via loader next)")


if __name__ == "__main__":
    main()
