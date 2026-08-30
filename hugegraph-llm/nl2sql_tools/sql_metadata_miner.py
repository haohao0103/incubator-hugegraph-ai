"""Mine SchemaMetadata (lineage / proven JOIN keys / co-occurrence) from SQL scripts.

Parses the data warehouse's production task SQL (Hive / SparkSQL / Trino / MySQL
etc.) with sqlglot and emits the three optional fields of ``SchemaMetadata`` so
they can be POSTed straight to ``/nl2sql/reload``:

  lineage       [[upstream_table, downstream_table], ...]   from INSERT/CTAS targets
  foreign_keys  [[child_col, parent_col], ...]              from equality join keys
  query_logs    [[t1, t2, ...], ...]                        one entry per statement

The platform only needs to hand over its task SQL text; nothing is hand-curated.

Usage:
  python scripts/sql_metadata_miner.py --dir /path/to/sql --out _out/sql_miner/meta.json
  python scripts/sql_metadata_miner.py --file a.sql,b.sql --dialect hive --out meta.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

from sqlglot import exp, parse

LOG_PATH = "_out/sql_miner/logs/miner.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def table_full_name(tbl: exp.Table) -> str:
    """catalog.db.name -> 'catalog.db.name'; db.name -> 'db.name'."""
    parts = [p for p in (tbl.catalog, tbl.db, tbl.name) if p]
    return ".".join(parts)


def _cte_names(node) -> set:
    """CTE aliases attached to a node (WITH x AS ... may live on the statement
    or on the inner SELECT, depending on how sqlglot models the construct)."""
    names = set()
    ctes = getattr(node, "ctes", None)
    if isinstance(ctes, (list, tuple)):
        for cte in ctes:
            if isinstance(cte, exp.CTE) and cte.alias:
                names.add(cte.alias)
    return names


def _collect_reads(stmt, cte_names) -> list:
    """Physical tables read by a statement (excluding CTE / derived aliases)."""
    if isinstance(stmt, exp.Insert):
        sel = stmt.expression
    elif isinstance(stmt, exp.Create):
        sel = stmt.expression
    else:
        sel = stmt
    if not isinstance(sel, exp.Select):
        return []
    tables = []
    for t in sel.find_all(exp.Table):
        name = t.name
        if name in cte_names:
            continue
        if isinstance(t.parent, exp.Subquery):
            continue  # derived table alias, not a physical source
        full = table_full_name(t)
        if full:
            tables.append(full)
    return tables


def _alias_map(sel, cte_names) -> dict:
    """alias -> full physical table name for column resolution."""
    mapping = {}
    for t in sel.find_all(exp.Table):
        if t.name in cte_names:
            continue
        full = table_full_name(t)
        if t.alias:
            mapping[t.alias] = full
        elif full:
            mapping[t.name] = full
    return mapping


def _col_full(col, alias_map) -> str:
    """Resolve a column expression to 'db.table.column' when possible."""
    parts = []
    node = col
    while isinstance(node, exp.Dot):
        parts.append(node.expression.name)
        node = node.this
    if isinstance(node, exp.Column):
        parts.append(node.name)
        qual = node.table or ""
        if qual:
            base = alias_map.get(qual, qual)
            parts.append(base)
    elif isinstance(node, exp.Identifier):
        parts.append(node.name)
    if len(parts) < 2:
        return ""
    return ".".join(reversed(parts))


def _join_keys(sel, alias_map) -> list:
    """Equality conditions between columns of different tables -> join keys."""
    keys = []
    known = set(alias_map.values())
    for eq in sel.find_all(exp.EQ):
        l, r = eq.this, eq.expression
        if not isinstance(l, (exp.Column, exp.Dot)) or not isinstance(
            r, (exp.Column, exp.Dot)
        ):
            continue
        lf, rf = _col_full(l, alias_map), _col_full(r, alias_map)
        if lf and rf and lf != rf:
            l_tbl, r_tbl = lf.rsplit(".", 1)[0], rf.rsplit(".", 1)[0]
            if (l_tbl != r_tbl and l_tbl in known and r_tbl in known):
                keys.append(tuple(sorted((lf, rf))))
    # USING(col) shorthand is rare in production Hive/Spark SQL; the EQ-based
    # extraction above covers the mainstream join-key forms.
    return keys


def mine_file(path: str, dialect: str):
    """Return (lineage, fk, cooc, stat) for one SQL file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        sql = f.read()
    lineage, fk, cooc = [], [], []
    seen = set()
    n_parsed = n_statements = 0
    try:
        statements = parse(sql, read=dialect)
    except Exception as exc:  # pragma: no cover - defensive
        log(f"  parse error {path}: {exc}")
        return [], [], [], {"errors": 1}
    for stmt in statements:
        n_statements += 1
        digest = hashlib.sha1(
            (stmt.sql() or "").strip().encode("utf-8")
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if not isinstance(stmt, (exp.Select, exp.Insert, exp.Create,
                                 exp.Merge, exp.Update)):
            continue
        n_parsed += 1
        sel = stmt.expression if isinstance(
            stmt, (exp.Insert, exp.Create)) else stmt
        cte_names = _cte_names(stmt) | _cte_names(sel)
        reads = _collect_reads(stmt, cte_names)
        if not reads:
            continue
        # write target (downstream)
        target = None
        if isinstance(stmt, exp.Insert) and isinstance(stmt.this, exp.Table):
            target = table_full_name(stmt.this)
        elif isinstance(stmt, exp.Create) and isinstance(stmt.this, exp.Schema):
            target = table_full_name(stmt.this)
        elif isinstance(stmt, exp.Merge) and isinstance(stmt.this, exp.Table):
            target = table_full_name(stmt.this)
        if target:
            for r in reads:
                if r != target:
                    lineage.append([r, target])
        # join keys
        if isinstance(sel, exp.Select):
            fk.extend(_join_keys(sel, _alias_map(sel, cte_names)))
        # co-occurrence
        uniq = sorted(set(reads))
        if len(uniq) >= 2:
            cooc.append(uniq)
    return lineage, fk, cooc, {
        "statements": n_statements, "parsed": n_parsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="directory scanned recursively for *.sql")
    ap.add_argument("--file", help="comma-separated SQL files")
    ap.add_argument("--dialect", default=None,
                    help="sqlglot dialect (hive/spark/trino/mysql/...); auto if unset")
    ap.add_argument("--out", default="_out/sql_miner/meta.json",
                    help="output SchemaMetadata JSON")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== sql metadata miner ===")
    files = []
    if args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "**", "*.sql"),
                                 recursive=True))
    if args.file:
        files.extend(p.strip() for p in args.file.split(",") if p.strip())
    files = list(dict.fromkeys(files))  # dedup, keep order
    log(f"files={len(files)} dialect={args.dialect or 'auto'}")

    lineage, fk, cooc = [], [], []
    fk_counter = Counter()
    stats = defaultdict(int)
    for path in files:
        try:
            lg, fk_li, co, st = mine_file(path, args.dialect)
        except Exception as exc:  # pragma: no cover
            log(f"  FAILED {path}: {exc}")
            stats["failed"] += 1
            continue
        lineage.extend(lg)
        for k in fk_li:
            fk_counter[k] += 1
        cooc.extend(co)
        for k_, v in st.items():
            stats[k_] += v
        stats["files_ok"] += 1

    # dedup lineage / join keys; keep join keys with their frequency
    meta = {
        "lineage": [list(x) for x in dict.fromkeys(map(tuple, lineage))],
        "foreign_keys": [list(k) for k, _ in fk_counter.most_common()],
        "query_logs": cooc,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out}: lineage={len(meta['lineage'])} "
        f"foreign_keys={len(meta['foreign_keys'])} "
        f"query_logs={len(meta['query_logs'])}")
    log(f"stats: {dict(stats)}")
    log("DONE")


if __name__ == "__main__":
    main()
