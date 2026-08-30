# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Metadata quality gate: validate SchemaMetadata before it enters the graph.

Catches the failures that silently poison a schema graph:

* duplicate table / column names (the graph would merge them by PRIMARY_KEY);
* columns without a comment (the single biggest lexical/semantic recall killer);
* columns whose table does not exist, term bindings to missing columns,
  FK / lineage edges whose endpoints are missing;
* the same metric name carrying conflicting definitions (口径冲突).

Severity: ``error`` blocks ingestion, ``warning`` is reported but tolerated.
Pure function over the ``SchemaMetadata`` dict shape, no external deps.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class QualityIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _bare(name: str) -> str:
    """'dw.orders' -> 'orders' (kg_rag stores bare table names)."""
    return name.split(".")[-1] if "." in name else name


def validate_metadata(meta: dict) -> List[QualityIssue]:
    """Validate a SchemaMetadata dict; returns a list of issues (possibly empty)."""
    issues: List[QualityIssue] = []
    tables = meta.get("tables", [])
    columns = meta.get("columns", [])
    terms = meta.get("terms", [])
    term_bindings = meta.get("term_bindings", [])
    foreign_keys = meta.get("foreign_keys", [])
    lineage = meta.get("lineage", [])

    # ---- tables: duplicates ----
    table_names: set = set()
    dup_tables: set = set()
    for t in tables:
        n = t.get("name")
        if not n:
            issues.append(QualityIssue("error", "TABLE_NO_NAME",
                                       "表缺少 name 字段"))
            continue
        if n in table_names:
            dup_tables.add(n)
        table_names.add(n)
    for n in sorted(dup_tables):
        issues.append(QualityIssue("error", "DUP_TABLE",
                                   f"重复表名（会按 PRIMARY_KEY 合并）: {n}"))

    # ---- columns: dup / no comment / orphan ----
    col_keys: set = set()
    dup_cols: set = set()
    empty_comment: List[str] = []
    orphan_cols: List[str] = []
    for c in columns:
        tbl, name = c.get("table", ""), c.get("name")
        if not name:
            issues.append(QualityIssue("error", "COL_NO_NAME",
                                       "字段缺少 name 字段"))
            continue
        if not tbl:
            issues.append(QualityIssue("error", "COL_NO_TABLE",
                                       f"字段缺少 table 归属: {name}"))
            continue
        key = f"{_bare(tbl)}.{name}"
        if key in col_keys:
            dup_cols.add(key)
        col_keys.add(key)
        if not (c.get("comment") or "").strip():
            empty_comment.append(key)
        if _bare(tbl) not in table_names:
            orphan_cols.append(key)
    for k in sorted(dup_cols):
        issues.append(QualityIssue("error", "DUP_COLUMN",
                                   f"重复字段: {k}"))
    for k in sorted(empty_comment):
        issues.append(QualityIssue("warning", "COL_NO_COMMENT",
                                   f"字段无中文注释（建议 enrich_column_comments 补齐）: {k}"))
    for k in sorted(orphan_cols):
        issues.append(QualityIssue("error", "COL_ORPHAN",
                                   f"字段归属的表不存在: {k}"))

    # ---- terms: conflicting 口径 ----
    term_def: Dict[str, str] = {}
    for t in terms:
        n = t.get("name")
        if not n:
            continue
        d = str(t.get("comment") or "")
        if n in term_def and term_def[n] != d:
            issues.append(QualityIssue("error", "TERM_CONFLICT",
                                       f"同名指标口径冲突: {n}"))
        term_def.setdefault(n, d)

    # ---- term bindings -> missing column ----
    for tb in term_bindings:
        if len(tb) == 2 and tb[1] not in col_keys:
            issues.append(QualityIssue(
                "error", "BIND_MISSING_COLUMN",
                f"指标绑定列不存在: {tb[0]} -> {tb[1]}"))

    # ---- FK / lineage endpoints ----
    for fk in foreign_keys:
        if len(fk) == 2 and (fk[0] not in col_keys or fk[1] not in col_keys):
            issues.append(QualityIssue(
                "warning", "FK_ENDPOINT_MISSING",
                f"外键端点缺失（将被跳过）: {fk[0]} -> {fk[1]}"))
    for lg in lineage:
        if len(lg) == 2 and (_bare(lg[0]) not in table_names
                             or _bare(lg[1]) not in table_names):
            issues.append(QualityIssue(
                "warning", "LINEAGE_ENDPOINT_MISSING",
                f"血缘端点缺失（将被跳过）: {lg[0]} -> {lg[1]}"))

    return issues


def summarize(meta: dict) -> Dict[str, object]:
    """One-line quality summary for API responses."""
    issues = validate_metadata(meta)
    errors = [i for i in issues if i.severity == "error"]
    return {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": sum(1 for i in issues if i.severity == "warning"),
        "issues": [i.to_dict() for i in issues],
    }
