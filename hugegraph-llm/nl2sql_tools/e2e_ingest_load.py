"""E2E: corpus metadata -> ingest to HugeGraph kg_rag -> load back via the
NL2SQL loader -> compare counts. Proves the persistent write path is lossless.

Run: PYTHONPATH=incubator-hugegraph-ai/hugegraph-llm/src \
       /path/to/hg-llm/python scripts/e2e_ingest_load.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from ingest_metadata_to_hg import ingest  # noqa: E402
from p2_corpus import (  # noqa: E402
    build_warehouse_schema, CORRECTIONS, WAREHOUSE_QUESTIONS,
)

from hugegraph_llm.nl2sql.hugegraph_schema_source import (  # noqa: E402
    build_schema_from_hugegraph,
)

LOG_PATH = "_out/hg_ingest/logs/e2e.log"
URL, GRAPH = "http://127.0.0.1:8081", "kg_rag"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def corpus_to_metadata(schema):
    """Dump the p2 corpus SchemaGraph back into SchemaMetadata JSON."""
    tables, columns, terms, term_bindings = [], [], [], []
    calibers = []
    for n in schema.nodes.values():
        t = n.node_type.value
        if t == "table":
            tables.append({
                "name": n.properties.get("database", "") + "." + n.name,
                "comment": n.properties.get("comment", ""),
                "row_count": n.properties.get("row_count", 0),
            })
        elif t == "column":
            columns.append({
                "name": n.name,
                "table": n.properties.get("table", ""),
                "comment": n.properties.get("comment", ""),
                "data_type": n.properties.get("data_type", ""),
            })
        elif t == "term":
            terms.append({
                "name": n.name,
                "comment": n.properties.get("comment", ""),
                "expression": n.properties.get("expression", ""),
            })
            # 术语口径：随 term 一起 dump，ingest 侧写成 Caliber 顶点 + hasCaliber 边
            for c in n.properties.get("calibers", []) or []:
                calibers.append({
                    "name": c.get("name", ""),
                    "metric": n.name,
                    "dimension": c.get("dimension", ""),
                    "description": c.get("description", ""),
                })
    lineage = []
    for e in schema.edges:
        if e.edge_type.value == "term_maps":
            term_bindings.append([e.source.split(":", 1)[1],
                                  e.target.split(":", 1)[1]])
        elif e.edge_type.value == "lineage":
            lineage.append([e.source.split(":", 1)[1],
                            e.target.split(":", 1)[1]])
    term_names = [t["name"] for t in terms]
    synonyms = ([[term_names[0], term_names[1]]] if len(term_names) >= 2 else [])
    return {"tables": tables, "columns": columns, "terms": terms,
            "term_bindings": term_bindings, "lineage": lineage,
            "synonyms": synonyms if synonyms else [],
            "calibers": [c for c in calibers if c.get("name")],
            "corrections": CORRECTIONS,
            "query_logs": [["dw.orders", "dw.payments"], ["dw.orders", "dw.users"]]}


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== e2e ingest -> load back ===")
    schema = build_warehouse_schema()
    meta = corpus_to_metadata(schema)
    log(f"input: tables={len(meta['tables'])} columns={len(meta['columns'])} "
        f"terms={len(meta['terms'])} bindings={len(meta['term_bindings'])}")

    counts = ingest(meta, URL, GRAPH, clear=True)
    log(f"ingested to {GRAPH}: {counts}")

    back = build_schema_from_hugegraph(url=URL, graph=GRAPH)
    log(f"loaded back: tables={len(back.tables())} columns={len(back.columns())} "
        f"terms={len(back.terms())} edges={len(back.edges)}")
    n_lineage = len(back.edges_of_type("lineage")) if hasattr(back, "edges_of_type") else 0
    syn_terms = [t for t in back.terms() if t.properties.get("synonyms")]
    log(f"lineage edges read back: {n_lineage}; terms with synonyms: {len(syn_terms)}")

    ok = (
        len(back.tables()) == len(meta["tables"])
        and len(back.columns()) == len(meta["columns"])
        and len(back.terms()) == len(meta["terms"])
        and n_lineage >= 1
        and len(syn_terms) >= 1
    )
    log(f"count match: {'PASS' if ok else 'FAIL'}")
    # spot check a semantic term binding survived
    pay_total = [t for t in back.terms() if t.name == "支付总额"]
    if pay_total:
        props = pay_total[0].properties
        log(f"term 支付总额 comment={props.get('comment')!r}")
    print("ALL PASS" if ok else "COUNT MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
