# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ingest a warehouse Schema Graph from a live HugeGraph KG.

This is the **production path** for NL2SQL: instead of shipping a static JSON
file, the schema is pulled deterministically from the HugeGraph KG that the
platform already maintains (the ``kg_rag`` graph at ``http://127.0.0.1:8081``
is the clean warehouse-metadata KG; see ``resources/example_warehouse.json`` for
the file fallback).

The HugeGraph KG model we map from (verified against the live ``kg_rag`` graph):

    vertex   Table   {name, comment}                     -> SchemaGraph Table
    vertex   Field   {name="table.column", comment, type}-> SchemaGraph Column
    vertex   Metric  {name, formula, definition}         -> SchemaGraph Term
    vertex   Query   {schema_refs="t1;t2;col.x;..."}     -> co-occurrence mining
    vertex   Caliber {name, dimension, description}      -> Term.properties["calibers"]
    vertex   CorrectionDecision {name, question, wrong_sql, correct_sql,
        correction_reason}                                -> node.properties["corrections"]
    edge     hasColumn          Table   -> Field         (structural ownership)
    edge     computedFromField  Metric  -> Field         (term -> column binding)
    edge     hasCaliber         Metric  -> Caliber       (term -> caliber binding)
    edge     correctionAppliesTo*  CorrectionDecision -> Metric|Field|Caliber
        (L3 correction provenance; folded into node.properties["corrections"])
    edge     computedFrom       Metric  -> Table         (metric source table)

Foreign keys are usually *not* declared in a warehouse catalog, so — unless the
KG provides them — we infer weak FKs from shared ``*_id`` column names across
tables. That keeps ``join_path`` useful even on a KG that only ships ownership.

Usage
-----
    from hugegraph_llm.nl2sql.hugegraph_schema_source import (
        build_schema_from_hugegraph,
    )
    schema = build_schema_from_hugegraph(
        url="http://127.0.0.1:8081", graph="kg_rag"
    )

The returned :class:`SchemaGraph` is exactly what ``SchemaGraphBuilder().build()``
produces, so it drops into ``NL2SQLPipeline`` unchanged.
"""

import gzip
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from hugegraph_llm.utils.log import log

from .schema_graph.builder import SchemaGraphBuilder
from .schema_graph.model import Column, SchemaGraph, Table, Term

# HugeGraph may sit behind an HTTP proxy in dev; localhost must bypass it.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _hg_get(url: str, timeout: int = 30) -> dict:
    """GET a HugeGraph REST URL, handling gzip transparently.

    HugeGraph compresses responses when the client advertises gzip; we always
    decompress so the caller gets plain JSON regardless.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":  # gzip magic bytes
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _collect(endpoint: str, kind: str, label: str, limit: int, timeout: int) -> List[dict]:
    """Page through ``/graph/{vertices|edges}`` for one label, with retries.

    Uses the ``page`` token when the server returns one; a guard caps runaway
    pagination. A single large ``limit`` is enough for typical schemas.
    Transient HTTP failures retry up to ``_HG_RETRIES`` times with backoff.
    """
    out: List[dict] = []
    page: Optional[str] = None
    guard = 0
    while True:
        url = f"{endpoint}/{kind}?label={label}&limit={limit}"
        if page:
            url += f"&page={page}"
        data = _hg_get_with_retry(url, timeout)
        out.extend(data.get(kind, []))
        page = data.get("page")
        if not page:
            break
        guard += 1
        if guard > 1000:
            log.warning("hg schema paging guard hit for label=%s", label)
            break
    return out


_HG_RETRIES = int(os.getenv("NL2SQL_HG_RETRIES", "3"))
_HG_RETRY_BACKOFF = [0.5, 1.0, 2.0]


def _hg_get_with_retry(url: str, timeout: int) -> dict:
    last: Optional[Exception] = None
    for attempt in range(_HG_RETRIES):
        try:
            return _hg_get(url, timeout)
        except Exception as exc:  # noqa: BLE001 -- transient network failure
            last = exc
            if attempt < _HG_RETRIES - 1:
                time.sleep(_HG_RETRY_BACKOFF[attempt % len(_HG_RETRY_BACKOFF)])
    raise last  # type: ignore[misc]


@dataclass
class HugeGraphSchemaMapping:
    """Configurable HugeGraph label/property -> Schema Graph role mapping.

    Defaults are tuned for the clean ``kg_rag`` graph. Override for other
    graphs (e.g. the messier ``hugegraph`` graph) without touching the loader.
    """

    table_label: str = "Table"
    field_label: str = "Field"
    metric_label: str = "Metric"
    query_label: str = "Query"
    caliber_label: str = "Caliber"
    correction_label: str = "CorrectionDecision"
    has_column_edge: str = "hasColumn"
    computed_from_field_edge: str = "computedFromField"
    has_caliber_edge: str = "hasCaliber"
    correction_to_term_edge: str = "correctionAppliesToTerm"
    correction_to_field_edge: str = "correctionAppliesToField"
    correction_to_caliber_edge: str = "correctionAppliesToCaliber"
    lineage_edge: str = "lineage"          # Table -> Table upstream->downstream
    synonym_edge: str = "synonym"          # Metric <-> Metric same-meaning terms

    table_name_prop: str = "name"
    table_comment_prop: str = "comment"
    table_rowcount_prop: str = "row_count"
    field_name_prop: str = "name"
    field_comment_prop: str = "comment"
    field_type_prop: str = "type"
    metric_name_prop: str = "name"
    metric_formula_prop: str = "formula"
    metric_definition_prop: str = "definition"
    metric_aliases_prop: str = "aliases"
    query_refs_prop: str = "schema_refs"
    caliber_name_prop: str = "name"
    caliber_dimension_prop: str = "dimension"
    caliber_description_prop: str = "description"
    correction_name_prop: str = "name"
    correction_question_prop: str = "question"
    correction_wrong_sql_prop: str = "wrong_sql"
    correction_correct_sql_prop: str = "correct_sql"
    correction_reason_prop: str = "correction_reason"


def build_schema_from_hugegraph(
    url: str = "http://127.0.0.1:8081",
    graph: str = "kg_rag",
    mapping: Optional[HugeGraphSchemaMapping] = None,
    infer_foreign_keys: bool = True,
    timeout: int = 30,
    limit: int = 500,
) -> SchemaGraph:
    """Build a :class:`SchemaGraph` from a live HugeGraph KG.

    :param url: HugeGraph REST base, e.g. ``http://127.0.0.1:8081``.
    :param graph: graph name, e.g. ``kg_rag``.
    :param mapping: label/property mapping; defaults to ``kg_rag``.
    :param infer_foreign_keys: when no FK edge exists, infer weak FKs from
        shared ``*_id`` column names across tables.
    :param timeout: per-request HTTP timeout (seconds).
    :param limit: page size when paginating vertices/edges.
    """
    m = mapping or HugeGraphSchemaMapping()
    base = f"{url.rstrip('/')}/graphs/{graph}/graph"

    table_vs = _collect(base, "vertices", m.table_label, limit, timeout)
    field_vs = _collect(base, "vertices", m.field_label, limit, timeout)
    metric_vs = _collect(base, "vertices", m.metric_label, limit, timeout)
    query_vs = _collect(base, "vertices", m.query_label, limit, timeout)
    caliber_vs = _collect(base, "vertices", m.caliber_label, limit, timeout)
    correction_vs = _collect(base, "vertices", m.correction_label, limit, timeout)
    cf_edges = _collect(base, "edges", m.computed_from_field_edge, limit, timeout)
    hc_edges = _collect(base, "edges", m.has_caliber_edge, limit, timeout)
    lg_edges = _collect(base, "edges", m.lineage_edge, limit, timeout)
    syn_edges = _collect(base, "edges", m.synonym_edge, limit, timeout)
    corr_edges = (
        _collect(base, "edges", m.correction_to_term_edge, limit, timeout)
        + _collect(base, "edges", m.correction_to_field_edge, limit, timeout)
        + _collect(base, "edges", m.correction_to_caliber_edge, limit, timeout)
    )

    log.info(
        "hg schema pull %s/%s: %s tables, %s fields, %s metrics, %s queries, "
        "%s calibers, %s corrections, %s term-edges, %s caliber-edges, "
        "%s lineage, %s synonym, %s correction-edges",
        url, graph, len(table_vs), len(field_vs), len(metric_vs), len(query_vs),
        len(caliber_vs), len(correction_vs), len(cf_edges), len(hc_edges),
        len(lg_edges), len(syn_edges), len(corr_edges),
    )

    b = SchemaGraphBuilder()

    # ---- Tables ----
    table_names: set = set()
    table_by_id: Dict[str, str] = {}
    for v in table_vs:
        props = v.get("properties", {}) or {}
        name = props.get(m.table_name_prop)
        if not name:
            continue
        table_names.add(name)
        table_by_id[v.get("id")] = name
        try:
            row_count = int(props.get(m.table_rowcount_prop, 0) or 0)
        except (TypeError, ValueError):
            row_count = 0
        b.add_table(Table(name=name, comment=props.get(m.table_comment_prop, ""),
                          row_count=row_count))

    # ---- Lineage (Table -> Table, upstream -> downstream) ----
    lineage_n = 0
    for e in lg_edges:
        up = table_by_id.get(e.get("outV"))
        down = table_by_id.get(e.get("inV"))
        if up and down and up != down:
            b.add_lineage(up, down)
            lineage_n += 1
    log.info("hg schema: loaded %s lineage edges", lineage_n)

    # ---- Columns (Field vertex: name is "table.column") ----
    field_by_id: Dict[str, tuple] = {}
    for v in field_vs:
        props = v.get("properties", {}) or {}
        fname = props.get(m.field_name_prop)
        if not fname or "." not in fname:
            # Skip malformed fields (e.g. a Field stored without a dotted name).
            log.debug("hg schema: skipping field %r (no dotted name)", fname)
            continue
        tbl, col = fname.split(".", 1)
        field_by_id[v.get("id")] = (tbl, col, fname)
        b.add_column(Column(
            name=col,
            table=tbl,
            data_type=props.get(m.field_type_prop, "") or "",
            comment=props.get(m.field_comment_prop, ""),
        ))

    # ---- Metrics -> Terms ----
    metric_by_id: Dict[str, str] = {}
    for v in metric_vs:
        props = v.get("properties", {}) or {}
        mname = props.get(m.metric_name_prop)
        if not mname:
            continue
        metric_by_id[v.get("id")] = mname
        aliases = props.get(m.metric_aliases_prop, "") or []
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(";") if a.strip()]
        b.add_term(Term(
            name=mname,
            aliases=list(aliases),
            expression=props.get(m.metric_formula_prop, "") or "",
            comment=props.get(m.metric_definition_prop, "") or "",
        ))

    # ---- Term -> Column bindings (computedFromField edges) ----
    bound = 0
    for e in cf_edges:
        mname = metric_by_id.get(e.get("outV"))
        finfo = field_by_id.get(e.get("inV"))
        if mname and finfo:
            b.bind_term(mname, finfo[2])  # full column name "table.column"
            bound += 1
    log.info("hg schema: bound %s term-column pairs", bound)

    # ---- Calibers: Metric -> Caliber edges -> Term.properties["calibers"] ----
    # 口径是语义层核心：同一指标在不同维度/约束下口径不同（如 GMV 仅统计 paid）。
    # 挂到 term.properties["calibers"] 后，linker/prompt 组装可注入「口径约束」。
    caliber_by_id: Dict[str, dict] = {}
    for v in caliber_vs:
        props = v.get("properties", {}) or {}
        cname = props.get(m.caliber_name_prop)
        if not cname:
            continue
        caliber_by_id[v.get("id")] = {
            "name": cname,
            "dimension": props.get(m.caliber_dimension_prop, "") or "",
            "description": props.get(m.caliber_description_prop, "") or "",
        }
    term_calibers: Dict[str, list] = {}
    for e in hc_edges:
        mname = metric_by_id.get(e.get("outV"))
        cal = caliber_by_id.get(e.get("inV"))
        if mname and cal:
            term_calibers.setdefault(mname, []).append(cal)
    if term_calibers:
        for node in b._terms.values():
            cals = term_calibers.get(node.name)
            if cals:
                node.properties["calibers"] = cals
        log.info("hg schema: attached calibers to %s terms", len(term_calibers))

    # ---- Co-occurrence from historical queries (CO_OCCUR edges) ----
    query_sets: List[List[str]] = []
    for v in query_vs:
        refs = (v.get("properties", {}) or {}).get(m.query_refs_prop, "")
        if not refs:
            continue
        toks = [t.strip() for t in str(refs).split(";") if t.strip()]
        # table names are tokens WITHOUT a dot; keep only known tables
        tbls = {t for t in toks if "." not in t and t in table_names}
        if len(tbls) >= 2:
            query_sets.append(sorted(tbls))
    if query_sets:
        b.add_query_logs(query_sets)
        log.info("hg schema: mined %s query co-occurrence sets", len(query_sets))

    # ---- Optional FK inference (no declared FK in most catalogs) ----
    if infer_foreign_keys:
        _infer_foreign_keys(b)

    schema = b.build()
    _annotate_synonyms(schema, syn_edges, metric_by_id)
    _annotate_corrections(schema, correction_vs, corr_edges, metric_by_id,
                          field_by_id, caliber_by_id, hc_edges, m)
    return schema


def _annotate_corrections(
    schema: SchemaGraph,
    correction_vs: List[dict],
    corr_edges: List[dict],
    metric_by_id: Dict[str, str],
    field_by_id: Dict[str, tuple],
    caliber_by_id: Dict[str, dict],
    hc_edges: List[dict],
    m: HugeGraphSchemaMapping,
) -> None:
    """Fold CorrectionDecision vertices + edges into node.properties["corrections"].

    L3 纠错决策（带 provenance）：纠错内容在 CorrectionDecision 顶点，边
    (correctionAppliesTo*) 定位它挂到哪个术语/字段/口径。折叠到目标节点属性后，
    检索侧沿语义边传播时，可达节点的纠错一起召回（同义词/指标链/口径可达也召回）。
    口径端点先经 hasCaliber 边反查归到所属术语，再挂到该术语节点。

    挂载结构: node.properties["corrections"] = [ {id, question, wrong_sql,
    correct_sql, correction_reason}, ... ] —— 与 PoC fetch_corrections 的
    返回结构一致，按 id 去重由检索侧负责。
    """
    if not corr_edges:
        return
    # CorrectionDecision 顶点 -> 纠错内容
    corr_by_id: Dict[str, dict] = {}
    for v in correction_vs:
        p = v.get("properties", {}) or {}
        cid = p.get(m.correction_name_prop)
        if not cid:
            continue
        corr_by_id[v.get("id")] = {
            "id": cid,
            "question": p.get(m.correction_question_prop, "") or "",
            "wrong_sql": p.get(m.correction_wrong_sql_prop, "") or "",
            "correct_sql": p.get(m.correction_correct_sql_prop, "") or "",
            "correction_reason": p.get(m.correction_reason_prop, "") or "",
        }
    if not corr_by_id:
        return
    # 口径 -> 所属术语（hasCaliber: Metric -> Caliber 的反向）
    caliber_to_term: Dict[str, str] = {}
    for e in hc_edges:
        mname = metric_by_id.get(e.get("outV"))
        cname = next((c.get("name") for cid_, c in caliber_by_id.items()
                      if cid_ == e.get("inV")), None)
        if mname and cname:
            caliber_to_term[cname] = mname
    attached = 0
    for e in corr_edges:
        corr = corr_by_id.get(e.get("outV"))
        if not corr:
            continue
        target_id = _correction_target_node_id(
            e, metric_by_id, field_by_id, caliber_by_id, caliber_to_term)
        if not target_id or target_id not in schema.nodes:
            continue
        schema.nodes[target_id].properties.setdefault(
            "corrections", []).append(corr)
        attached += 1
    log.info("hg schema: attached %s corrections to schema nodes", attached)


def _correction_target_node_id(
    e: dict,
    metric_by_id: Dict[str, str],
    field_by_id: Dict[str, tuple],
    caliber_by_id: Dict[str, dict],
    caliber_to_term: Dict[str, str],
) -> Optional[str]:
    """Map a correction edge to the SchemaGraph node_id it attaches to.

    Edge labels: correctionAppliesToTerm (-> Metric), correctionAppliesToField
    (-> Field), correctionAppliesToCaliber (-> Caliber, re-mapped to the term
    that owns the caliber via hasCaliber).
    """
    label = e.get("label", "")
    inV = e.get("inV")
    if label == "correctionAppliesToTerm":
        mname = metric_by_id.get(inV)
        return f"term:{mname}" if mname else None
    if label == "correctionAppliesToField":
        finfo = field_by_id.get(inV)
        if not finfo:
            return None
        return f"column:{finfo[2]}"
    if label == "correctionAppliesToCaliber":
        cname = next((c.get("name") for cid_, c in caliber_by_id.items()
                      if cid_ == inV), None)
        if cname and cname in caliber_to_term:
            return f"term:{caliber_to_term[cname]}"
    return None


def _annotate_synonyms(
    schema: SchemaGraph,
    syn_edges: List[dict],
    metric_by_id: Dict[str, str],
) -> None:
    """Fold Metric<->Metric synonym edges into term node properties.

    ``node.properties["synonyms"]`` lists same-meaning term names; the linker
    expands them when a term seed hits, so "实际车型" reaches "物理车型".
    """
    if not syn_edges:
        return
    syn_map: Dict[str, set] = {}
    for e in syn_edges:
        a = metric_by_id.get(e.get("outV"))
        b_ = metric_by_id.get(e.get("inV"))
        if a and b_ and a != b_:
            syn_map.setdefault(a, set()).add(b_)
            syn_map.setdefault(b_, set()).add(a)
    for node in schema.terms():
        syns = syn_map.get(node.name)
        if syns:
            node.properties["synonyms"] = sorted(syns)


def _infer_foreign_keys(b: SchemaGraphBuilder) -> None:
    """Heuristic: shared ``*_id`` column name across two tables => foreign key.

    Conservative on purpose — only ``id`` / ``*_id`` columns are treated as
    join keys, because those are the columns that actually connect tables in a
    warehouse. Declared FKs (via an HG edge) should still be the source of
    truth; this only fills gaps.
    """
    by_name: Dict[str, List[str]] = {}
    for full, col in b._columns.items():
        if col.name == "id" or col.name.endswith("_id"):
            by_name.setdefault(col.name, []).append(full)

    added = 0
    for fulls in by_name.values():
        if len(fulls) < 2:
            continue
        for i in range(len(fulls)):
            for j in range(i + 1, len(fulls)):
                a, c = fulls[i], fulls[j]
                if b._columns[a].table == b._columns[c].table:
                    continue
                b.add_foreign_key(a, c)
                added += 1
    if added:
        log.info("hg schema: inferred %s foreign keys from shared *_id columns", added)
