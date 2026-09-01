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

"""Natural-language documents -> semantic metadata (NL2SQL doc channel).

The NL2SQL semantic layer accepts four input kinds; this module powers the
fourth one — business documents:

  1. structured metadata   meta.json                 -> nl2sql.ingest_to_hg
  2. production SQL        sql_metadata_miner.py     -> lineage/fk/query_logs
  3. column comment enrich enrich_column_comments.py
  4. NL documents          THIS MODULE              -> terms/bindings/calibers/corrections
       (data dictionary / caliber docs / QA history incl. wrong SQL)

The extraction target is **business semantics** (term definitions, caliber
constraints, historical corrections) — deliberately different from the
GraphRAG document pipeline's generic entity/relation triples.

Document types (``doc_type``):
  dictionary  data dictionary  -> [{name, definition, aliases[], related_column}]
  caliber     caliber docs     -> [{metric, name(caliber), dimension, description,
                                    related_column}]
  qa          QA history       -> [{question, wrong_sql, correct_sql,
                                    correction_reason, applies_to: [...]}]

Output is SchemaMetadata-homogeneous:
  {"terms": [...], "term_bindings": [[term, "t.col"]...],
   "calibers": [{name, metric, dimension, description}...],
   "corrections": [{id, question, wrong_sql, correct_sql, correction_reason,
                    applies_to: [...]}...]}

Binding rule: ``related_column`` may be a Chinese name or a physical name; a
binding is only produced when it resolves to a real column in the target meta.
Unmatched references land in ``report["unmatched"]`` — never silently dropped.

Callers: :class:`~hugegraph_llm.nl2sql.ingest.Nl2SqlIngester` (production
entry point) and the standalone CLI ``nl2sql_tools/doc_ingest_extract.py``.
"""

import json
import re
from typing import List, Optional

from hugegraph_llm.utils.log import log

DOC_TYPE_PROMPTS = {
    "dictionary": (
        "你是数仓语义建模专家。从下面的【数据字典】文档中抽取业务术语定义。\n"
        "输出一个 JSON 数组（不要输出其他内容），每个元素：\n"
        "{\"name\": \"术语名\", \"definition\": \"一句话业务定义\", "
        "\"aliases\": [\"同义词\"], \"related_column\": \"文档中提到的物理字段名或中文描述，无则空串\"}\n"
    ),
    "caliber": (
        "你是指标口径治理专家。从下面的【口径说明文档】中抽取指标口径。\n"
        "输出一个 JSON 数组（不要输出其他内容），每个元素：\n"
        "{\"metric\": \"指标/术语名\", \"name\": \"口径名\", "
        "\"dimension\": \"约束维度(status/grain/time/...)\", "
        "\"description\": \"口径的完整计算约束描述\", "
        "\"related_column\": \"口径适用的物理字段，无则空串\"}\n"
    ),
    "qa": (
        "你是 NL2SQL 纠错专家。从下面的【历史问答记录】中抽取纠错样本。\n"
        "输出一个 JSON 数组（不要输出其他内容），每个元素：\n"
        "{\"question\": \"用户问题\", \"wrong_sql\": \"错误 SQL\", "
        "\"correct_sql\": \"正确 SQL\", \"correction_reason\": \"纠错原因\", "
        "\"applies_to\": [\"term:术语名\" 或 \"field:表.字段\"]}\n"
    ),
}


def _extract_json_block(text: str):
    """Pull the first JSON array/object out of an LLM reply (fence-tolerant)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    for start, end in ((text.find("["), text.rfind("]")),
                       (text.find("{"), text.rfind("}"))):
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON block found in LLM output: {text[:200]}")


def extract_document(text: str, doc_type: str, llm, retries: int = 2) -> list:
    """Extract one document into a list of typed semantic-metadata items.

    :param llm: a callable with ``generate(prompt=...) -> str`` (the
                ``hugegraph_llm`` chat-LLM contract).
    :param retries: number of retries after a parse/LLM failure.
    :raises: the last error when every attempt fails (caller decides whether
             the failure is fatal for the whole batch).
    """
    if doc_type not in DOC_TYPE_PROMPTS:
        raise ValueError(f"unknown doc_type: {doc_type}")
    prompt = DOC_TYPE_PROMPTS[doc_type] + "\n【文档内容】\n" + text
    last_err = None
    for i in range(retries + 1):
        try:
            gen = llm.generate(prompt=prompt)
            items = _extract_json_block(gen)
            if isinstance(items, dict):  # single-object fallback
                items = [items]
            if not isinstance(items, list):
                raise ValueError(f"not a list: {type(items)}")
            log.info("doc_extract[%s] ok: %s items (attempt %s)",
                     doc_type, len(items), i + 1)
            return items
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("doc_extract[%s] attempt %s failed: %s: %s",
                        doc_type, i + 1, type(e).__name__, str(e)[:100])
    raise last_err


def extract_documents(docs, llm, retries: int = 2) -> dict:
    """Extract a batch of ``{doc_type, content, name?}`` dicts.

    Each document is extracted independently: a single failure raises for that
    document only and is returned under ``errors``, so a bundle of documents
    keeps the successfully extracted parts (partial success is explicit, never
    silent).
    """
    out: dict = {"terms": [], "term_bindings": [], "calibers": [],
                 "corrections": [], "errors": []}
    for doc in docs:
        doc_type = doc.get("doc_type")
        content = doc.get("content") or ""
        name = doc.get("name") or doc_type
        if doc_type not in DOC_TYPE_PROMPTS:
            out["errors"].append({"name": name, "error": f"unknown doc_type {doc_type}"})
            continue
        if not content.strip():
            out["errors"].append({"name": name, "error": "empty content"})
            continue
        try:
            items = extract_document(content, doc_type, llm, retries=retries)
        except Exception as e:  # noqa: BLE001
            out["errors"].append({"name": name, "error": f"{type(e).__name__}: {e}"})
            continue
        part = {
            "dictionary": normalize_dictionary,
            "caliber": normalize_caliber,
            "qa": normalize_qa,
        }[doc_type](items)
        for k, v in part.items():
            out[k].extend(v)
        log.info("doc_extract[%s] normalized: terms=%s calibers=%s corrections=%s",
                 name, len(part.get("terms", [])), len(part.get("calibers", [])),
                 len(part.get("corrections", [])))
    return out


# ---------------------------------------------------------------------------
# Normalisation (-> SchemaMetadata-homogeneous)
# ---------------------------------------------------------------------------


def normalize_dictionary(items: list) -> dict:
    terms, bindings = [], []
    for it in items:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        aliases = [str(a).strip() for a in (it.get("aliases") or []) if str(a).strip()]
        terms.append({
            "name": name,
            "comment": (it.get("definition") or "").strip(),
            "expression": "",
            "aliases": aliases,
        })
        col = (it.get("related_column") or "").strip()
        if col:
            bindings.append([name, col])
    return {"terms": terms, "term_bindings": bindings}


def normalize_caliber(items: list) -> dict:
    calibers = []
    for it in items:
        name = (it.get("name") or "").strip()
        metric = (it.get("metric") or "").strip()
        if not (name and metric):
            continue
        calibers.append({
            "name": name,
            "metric": metric,
            "dimension": (it.get("dimension") or "").strip(),
            "description": (it.get("description") or "").strip(),
            "related_column": (it.get("related_column") or "").strip(),
        })
    return {"calibers": calibers}


def normalize_qa(items: list, start_id: int = 1) -> dict:
    corrections = []
    for i, it in enumerate(items):
        q = (it.get("question") or "").strip()
        ws = (it.get("wrong_sql") or "").strip()
        cs = (it.get("correct_sql") or "").strip()
        if not (q and ws and cs):
            continue
        applies = []
        for a in (it.get("applies_to") or []):
            a = str(a).strip()
            if a:
                applies.append(a)
        # 稳定 id：基于纠错内容指纹，而非抽取序号——LLM 对同一文档两次抽取
        # 顺序/条数可能漂移，指纹 id 保证同一条纠错重复导入时 diff 幂等。
        import hashlib

        digest = hashlib.sha1(f"{q}\x00{cs}".encode("utf-8")).hexdigest()[:10]
        corrections.append({
            "id": f"doc_qa_{digest}",
            "question": q,
            "wrong_sql": ws,
            "correct_sql": cs,
            "correction_reason": (it.get("correction_reason") or "").strip(),
            "applies_to": applies,
        })
    return {"corrections": corrections}


# ---------------------------------------------------------------------------
# Merge into an existing meta (bind only to real columns)
# ---------------------------------------------------------------------------


def _column_index(meta: dict):
    """Resolve every column surface a document may reference to a Column dict.

    Accepts ``name``, ``table.column`` (bare) and ``db.table.column`` so a
    document written for humans (中文名 / 物理名 / 全限定名) binds when possible.
    """
    col_names = set()
    col_lookup = {}
    for c in meta.get("columns", []):
        tbl = c["table"].split(".")[-1]
        full = f"{tbl}.{c['name']}"
        for surface in (c["name"], full, f"dw.{full}"):
            col_names.add(surface)
            col_lookup.setdefault(surface, c)
    return col_names, col_lookup


def merge_into_meta(meta: dict, extracted: dict) -> tuple:
    """Merge doc-extraction output into an existing meta dict (in place on a
    shallow copy). Existing entries win on name collisions; new ones append.

    Term bindings are produced **only** when ``related_column`` resolves to a
    real column of the target meta; everything else is reported under
    ``report["unmatched"]`` so no information is silently dropped.

    Returns ``(merged_meta, report)`` where report carries:
      unmatched:  [{type, term, column} | {type, caliber, column}]
      dup_terms / dup_calibers / dup_corrections: collisions skipped.
    """
    out = dict(meta)
    report: dict = {"unmatched": []}

    col_names, col_lookup = _column_index(meta)
    existing_terms = {t["name"] for t in meta.get("terms", [])}
    existing_bindings = {tuple(b) for b in meta.get("term_bindings", [])}

    # 1) terms + term_bindings
    terms = list(meta.get("terms", []))
    bindings = [list(b) for b in meta.get("term_bindings", [])]
    dup_terms: List[str] = []
    for t in extracted.get("terms", []):
        if t["name"] in existing_terms:
            dup_terms.append(t["name"])
            continue
        terms.append(t)
        existing_terms.add(t["name"])
    for name, col in extracted.get("term_bindings", []):
        col = col.strip()
        if col in col_names:
            if (name, col) not in existing_bindings:
                bindings.append([name, col])
                existing_bindings.add((name, col))
        else:
            report["unmatched"].append(
                {"type": "term_binding", "term": name, "column": col})
            log.info("doc_merge unmatched term binding: %s -> %s", name, col)

    # 2) calibers（已有口径名跳过）；related_column 匹配到真实列时顺带生成
    #    term_binding（口径文档提到字段 => 术语-字段绑定，口径约束才能沿图传播）
    calibers = list(meta.get("calibers", []))
    existing_cals = {c.get("name") for c in calibers}
    dup_calibers: List[str] = []
    for c in extracted.get("calibers", []):
        if c["name"] in existing_cals:
            dup_calibers.append(c["name"])
            continue
        rc = (c.get("related_column") or "").strip()
        if rc and rc in col_names:
            metric = c["metric"]
            if (metric, rc) not in existing_bindings:
                bindings.append([metric, rc])
                existing_bindings.add((metric, rc))
        elif rc:
            report["unmatched"].append(
                {"type": "caliber_column", "caliber": c["name"], "column": rc})
            log.info("doc_merge unmatched caliber column: %s -> %s", c["name"], rc)
        calibers.append({
            "name": c["name"], "metric": c["metric"],
            "dimension": c.get("dimension", ""),
            "description": c.get("description", ""),
        })
        existing_cals.add(c["name"])

    # 3) corrections（按 id 去重）
    corrections = list(meta.get("corrections", []))
    existing_ids = {c.get("id") for c in corrections}
    dup_corrections: List[str] = []
    for c in extracted.get("corrections", []):
        if c["id"] in existing_ids:
            dup_corrections.append(c["id"])
            continue
        corrections.append(c)
        existing_ids.add(c["id"])

    out["terms"] = terms
    out["term_bindings"] = bindings
    out["calibers"] = calibers
    out["corrections"] = corrections
    report["dup_terms"] = dup_terms
    report["dup_calibers"] = dup_calibers
    report["dup_corrections"] = dup_corrections
    return out, report


__all__ = [
    "DOC_TYPE_PROMPTS", "extract_document", "extract_documents",
    "merge_into_meta", "normalize_caliber", "normalize_dictionary",
    "normalize_qa",
]
