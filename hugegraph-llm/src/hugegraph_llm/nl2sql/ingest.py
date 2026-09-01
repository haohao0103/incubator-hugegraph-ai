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

"""Unified, production-grade NL2SQL metadata ingestion entry point.

One interface for every input the NL2SQL semantic layer accepts. The caller
declares the input kind with a single flag (``source``); everything else —
parsing, LLM extraction, merging with what the graph already holds, diff
writes to the HugeGraph KG, the P2 vector-index refresh — is internal:

    payload = {"source": "structured", "meta": {...}}            # tables/columns/terms/...
    payload = {"source": "doc", "doc_type": "dictionary",        # data dictionary
               "content": "..."}
    payload = {"source": "doc", "doc_type": "caliber", "content": "..."}   # caliber docs
    payload = {"source": "doc", "doc_type": "qa", "content": "..."}        # QA history (incl. wrong SQL)
    payload = {"source": "doc_bundle", "docs": [{"doc_type": ..., "content": ...}, ...]}

Consistency guarantees (production contract)
---------------------------------------------
* **Single source of truth**: one graph (default ``kg_rag``). Every ingest
  reads the current graph state back first (:func:`schema_graph_to_meta`) and
  merges the new input into it, so structured metadata and documents can never
  drift into two inconsistent copies.
* **Single vector store**: one :class:`SchemaVectorStore` instance is injected
  at construction and shared by the write path (index refresh after each
  ingest) and the read path (:meth:`Nl2SqlIngester.load_pipeline`). Node
  surfaces are embedded through :func:`~hugegraph_llm.nl2sql.vector_store.
  embed_schema_nodes`, the same function the linker uses, so the vector index
  is always rebuilt from the same text and the same store.
* **Idempotent**: writes run in diff mode (PRIMARY_KEY upsert); re-ingesting
  the same payload adds zero vertices/edges.
* **Explicit failure modes**: structural payload errors raise ``ValueError``
  before any write; a doc whose LLM extraction fails raises when it produced
  nothing (no silent half-writes); in a ``doc_bundle``, failures are per-doc —
  the successful parts still land and every failure is reported.
* **No silent drops**: document terms/calibers that cannot be bound to a real
  column land in ``report["unmatched"]``, never silently discarded.

CLI
---
    python -m hugegraph_llm.nl2sql.ingest --source structured --meta meta.json
    python -m hugegraph_llm.nl2sql.ingest --source doc --doc-type dictionary --doc data_dictionary.md
    python -m hugegraph_llm.nl2sql.ingest --source doc_bundle \
        --docs dictionary:data_dictionary.md,caliber:caliber_docs.md,qa:qa_history.md
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from hugegraph_llm.utils.log import log as hg_log

from . import ingest_to_hg
from .doc_extract import (
    extract_document,
    normalize_caliber,
    normalize_dictionary,
    normalize_qa,
)
from .hugegraph_schema_source import build_schema_from_hugegraph
from .schema_graph.model import EdgeType, NodeType
from .vector_store import SchemaVectorStore, embed_schema_nodes

_NORMALIZERS = {
    "dictionary": normalize_dictionary,
    "caliber": normalize_caliber,
    "qa": normalize_qa,
}

LOG_PATH = os.environ.get(
    "HUGEGRAPH_NL2SQL_UNIFIED_LOG",
    "_out/nl2sql_unified/logs/ingest.log",
)

_SUPPORTED_DOC_TYPES = ("dictionary", "caliber", "qa")

# payload fields that must be lists when present (structured meta)
_LIST_FIELDS = ("tables", "columns", "terms", "term_bindings", "calibers",
                "corrections", "lineage", "synonyms", "query_logs")


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------


def validate_payload(payload: dict) -> None:
    """Structural validation; raises ``ValueError`` with a precise message."""
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    source = payload.get("source")
    if source not in ("structured", "doc", "doc_bundle"):
        raise ValueError(
            "payload['source'] must be one of 'structured' | 'doc' | "
            f"'doc_bundle', got {source!r}"
        )
    if source == "structured":
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("structured payload requires payload['meta'] as a dict")
        for k in _LIST_FIELDS:
            if k in meta and not isinstance(meta[k], list):
                raise ValueError(f"meta['{k}'] must be a list, got {type(meta[k]).__name__}")
        for k in ("tables", "columns"):
            if k in meta:
                for i, item in enumerate(meta[k]):
                    if not isinstance(item, dict) or not item.get("name"):
                        raise ValueError(f"meta['{k}'][{i}] must be a dict with 'name'")
    elif source == "doc":
        doc_type = payload.get("doc_type")
        if doc_type not in _SUPPORTED_DOC_TYPES:
            raise ValueError(
                "doc payload requires doc_type in " + "|".join(_SUPPORTED_DOC_TYPES)
                + f", got {doc_type!r}")
        if not str(payload.get("content") or "").strip():
            raise ValueError("doc payload requires non-empty 'content'")
    else:  # doc_bundle
        docs = payload.get("docs")
        if not isinstance(docs, list) or not docs:
            raise ValueError("doc_bundle payload requires payload['docs'] as a non-empty list")
        for i, d in enumerate(docs):
            if not isinstance(d, dict) or d.get("doc_type") not in _SUPPORTED_DOC_TYPES:
                raise ValueError(
                    f"docs[{i}] must be a dict with doc_type in "
                    + "|".join(_SUPPORTED_DOC_TYPES))
            if not str(d.get("content") or "").strip():
                raise ValueError(f"docs[{i}] requires non-empty 'content'")


# ---------------------------------------------------------------------------
# SchemaGraph -> SchemaMetadata (lossless read-back)
# ---------------------------------------------------------------------------


def schema_graph_to_meta(schema) -> dict:
    """Round-trip a SchemaGraph read back from the graph into meta.json shape.

    Mirrors ``corpus_to_metadata`` but fixes the read-back field mapping:
    ingest writes Metric.definition <- term.comment, Metric.formula <-
    term.expression, and the loader restores them onto ``term.properties``, so
    the export reads those exact keys. Calibers and corrections (folded onto
    term/column properties by the loader) are expanded back into
    ``meta["calibers"]`` / ``meta["corrections"]`` with their ``metric`` /
    ``applies_to`` provenance reconstructed, making the read-back lossless.
    """
    tables, columns, terms, term_bindings, lineage = [], [], [], [], []
    calibers, corrections = [], []

    def _corr(c: dict, applies: str) -> dict:
        out = {
            "id": c.get("id"),
            "question": c.get("question", ""),
            "wrong_sql": c.get("wrong_sql", ""),
            "correct_sql": c.get("correct_sql", ""),
            "correction_reason": c.get("correction_reason", ""),
        }
        applies_to = [a for a in (c.get("applies_to") or []) if a]
        if not applies_to:
            applies_to = [applies]
        out["applies_to"] = applies_to
        return out

    for n in schema.nodes.values():
        t = n.node_type.value
        if t == "table":
            tables.append({
                "name": n.name,
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
            for c in n.properties.get("corrections", []) or []:
                corrections.append(_corr(c, f"field:{n.name}"))
        elif t == "term":
            terms.append({
                "name": n.name,
                "comment": n.properties.get("comment", ""),
                "expression": n.properties.get("expression", ""),
                "aliases": list(n.properties.get("aliases", []) or []),
            })
            for c in n.properties.get("calibers", []) or []:
                calibers.append({
                    "name": c.get("name", ""),
                    "metric": n.name,
                    "dimension": c.get("dimension", ""),
                    "description": c.get("description", ""),
                })
            for c in n.properties.get("corrections", []) or []:
                corrections.append(_corr(c, f"term:{n.name}"))

    for e in schema.edges:
        if e.edge_type == EdgeType.TERM_MAPS:
            term_bindings.append([e.source.split(":", 1)[1],
                                  e.target.split(":", 1)[1]])
        elif e.edge_type == EdgeType.LINEAGE:
            lineage.append([e.source.split(":", 1)[1],
                            e.target.split(":", 1)[1]])

    # dedupe corrections by id (a correction can hang off several nodes)
    seen, uniq = set(), []
    for c in corrections:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(c)

    # synonym edges are folded onto term.properties["synonyms"] by the loader
    synonyms = []
    names = {t["name"] for t in terms}
    for t in terms:
        sn = schema.nodes.get(f"term:{t['name']}")
        if sn is None:
            continue
        for s in sn.properties.get("synonyms", []) or []:
            if s in names and (t["name"], s) not in synonyms:
                synonyms.append([t["name"], s])

    return {
        "tables": tables,
        "columns": columns,
        "terms": terms,
        "term_bindings": term_bindings,
        "lineage": lineage,
        "synonyms": synonyms,
        "calibers": [c for c in calibers if c.get("name")],
        "corrections": uniq,
        "query_logs": [],
    }


# ---------------------------------------------------------------------------
# Structured merge (new input wins on name collision; graph state preserved)
# ---------------------------------------------------------------------------


def merge_structured(baseline: dict, incoming: dict,
                     force_update: bool = False) -> tuple:
    """Merge structured metadata into the graph's current state.

    The graph is the source of truth, so by default a name already present is
    kept as-is (document-derived semantics are never wiped by an incomplete
    schema snapshot). ``force_update=True`` overwrites colliding entries with
    the incoming values. Pairwise collections (term_bindings / lineage /
    synonyms) merge as deduped unions.

    Returns ``(merged_meta, report)``; ``report["added"]`` counts per section.
    """
    out: dict = {}
    for k in _LIST_FIELDS:
        out[k] = list(baseline.get(k, []) or [])
    report: dict = {"added": {}}

    def _merge_items(key: str, name_fn: Callable[[dict], str]) -> int:
        existing = {name_fn(x) for x in out.get(key, [])}
        added = 0
        for item in incoming.get(key, []) or []:
            name = name_fn(item)
            if name in existing:
                if force_update:
                    out[key] = [item if name_fn(x) == name else x for x in out[key]]
                continue
            out[key].append(item)
            existing.add(name)
            added += 1
        return added

    for key, name_fn in (
        ("tables", lambda x: x["name"]),
        ("columns", lambda x: f"{x['table']}.{x['name']}"),
        ("terms", lambda x: x["name"]),
        ("calibers", lambda x: x.get("name") or ""),
        ("corrections", lambda x: x.get("id") or x.get("name") or ""),
    ):
        report["added"][key] = _merge_items(key, name_fn)

    for key in ("term_bindings", "lineage", "synonyms"):
        existing = {tuple(x) for x in out.get(key, [])}
        added = 0
        for pair in incoming.get(key, []) or []:
            if len(pair) != 2:
                continue
            if tuple(pair) not in existing:
                out[key].append(list(pair))
                existing.add(tuple(pair))
                added += 1
        report["added"][key] = added

    # query_logs are ephemeral co-occurrence hints: union without dedupe
    report["added"]["query_logs"] = len(incoming.get("query_logs", []) or [])
    out["query_logs"] = list(out.get("query_logs", [])) + list(
        incoming.get("query_logs", []) or [])
    return out, report


# ---------------------------------------------------------------------------
# The unified ingester
# ---------------------------------------------------------------------------


@dataclass
class IngestReport:
    """Structured outcome of one :meth:`Nl2SqlIngester.ingest` call."""

    ok: bool
    source: str
    baseline: dict = field(default_factory=dict)   # graph state before merge
    extracted: dict = field(default_factory=dict)  # doc channel only
    merge: dict = field(default_factory=dict)      # added / unmatched / dup
    written: dict = field(default_factory=dict)    # ingest write counts
    validation: dict = field(default_factory=dict)  # Datalog report
    vector: dict = field(default_factory=dict)      # index refresh outcome
    errors: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "source": self.source,
            "baseline": self.baseline, "extracted": self.extracted,
            "merge": self.merge, "written": self.written,
            "validation": self.validation, "vector": self.vector,
            "errors": self.errors, "elapsed_ms": self.elapsed_ms,
        }


class Nl2SqlIngester:
    """Unified entry point for every NL2SQL metadata input.

    Construct once per process (single graph + single vector store), then call
    :meth:`ingest` for structured metadata and/or documents. The read path
    (:meth:`load_pipeline`) shares the same vector store and embedder, so the
    graph and the vector index can never diverge.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8081",
        graph: str = "kg_rag",
        llm: Optional[Callable] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        vector_store: Optional[SchemaVectorStore] = None,
        timeout: int = 60,
        cache_extractions: bool = True,
    ):
        self._url = url
        self._graph = graph
        self._llm = llm
        self._embedder = embedder
        self._vector_store = vector_store  # None -> in-process numpy at first use
        self._timeout = timeout
        self._schema = None      # cached SchemaGraph read-back
        self._meta: Optional[dict] = None  # cached merged meta (graph is truth)
        self._vector_disabled = False
        # LLM extraction cache: keyed by sha1(doc_type + content). Extraction is
        # expensive AND non-deterministic; caching the normalised product makes
        # re-ingesting the same document idempotent instead of re-rolling the
        # LLM dice (names drift run-to-run, which would silently add duplicates).
        self._cache_extractions = cache_extractions
        self._extract_cache: Dict[str, dict] = {}

    # ---- graph read-back ----

    def read_schema(self, refresh: bool = False):
        """The current SchemaGraph, read back from the graph (cached)."""
        if self._schema is None or refresh:
            self._schema = build_schema_from_hugegraph(
                url=self._url, graph=self._graph, timeout=self._timeout)
        return self._schema

    def _read_baseline(self) -> dict:
        """Current graph state as meta dict; empty meta on a fresh graph."""
        try:
            schema = self.read_schema()
            return schema_graph_to_meta(schema)
        except Exception as exc:  # noqa: BLE001 -- graph not ready yet?
            hg_log.warning("read-back baseline failed (%s); starting from empty meta", exc)
            return {}

    # ---- vector index ----

    def _ensure_store(self) -> SchemaVectorStore:
        if self._vector_store is None:
            from .vector_store import NumpySchemaVectorStore
            self._vector_store = NumpySchemaVectorStore()
        return self._vector_store

    def refresh_vector_index(self) -> dict:
        """Re-embed the whole schema into the single shared vector store.

        Called after every ingest (only when an embedder is configured).
        Failures degrade to lexical linking and are reported, never fatal.
        """
        if self._embedder is None:
            return {"enabled": False, "reason": "no embedder configured"}
        if self._vector_disabled:
            return {"enabled": False, "reason": "vector disabled by earlier failure"}
        try:
            schema = self.read_schema()
            store = self._ensure_store()
            n, dim = embed_schema_nodes(schema, self._embedder, store)
            _log(f"vector index refreshed: nodes={n} dim={dim}")
            return {"enabled": True, "nodes": n, "dim": dim}
        except Exception as exc:  # noqa: BLE001 -- degrade to lexical
            self._vector_disabled = True
            hg_log.warning("vector refresh failed; P2 disabled: %s", exc)
            return {"enabled": False, "error": str(exc)}

    # ---- read path (same graph + same vector store) ----

    def load_pipeline(self, **kwargs):
        """A ready-to-use :class:`NL2SQLPipeline` sharing this ingester's
        graph, embedder and vector store — the read side of the same chain."""
        from .pipeline import NL2SQLPipeline

        schema = self.read_schema()
        return NL2SQLPipeline(
            schema,
            embedder=self._embedder,
            vector_store=self._vector_store,
            **kwargs,
        )

    def status(self) -> dict:
        """Current graph state (vertices/edges) + vector index state."""
        try:
            schema = self.read_schema(refresh=True)
            n_lineage = len(schema.edges_of_type(EdgeType.LINEAGE))
            n_bind = len(schema.edges_of_type(EdgeType.TERM_MAPS))
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return {
            "graph": self._graph,
            "tables": len(schema.tables()),
            "columns": len(schema.columns()),
            "terms": len(schema.terms()),
            "lineage_edges": n_lineage,
            "term_bind_edges": n_bind,
            "vector_embedder": self._embedder is not None,
        }

    # ---- the unified write path ----

    def ingest(self, payload: dict) -> IngestReport:
        t0 = time.perf_counter()
        validate_payload(payload)
        source = payload["source"]
        _log(f"=== ingest source={source} graph={self._graph} ===")

        baseline = self._read_baseline()
        _log(f"baseline: tables={len(baseline['tables'])} "
             f"columns={len(baseline['columns'])} terms={len(baseline['terms'])} "
             f"calibers={len(baseline['calibers'])} "
             f"corrections={len(baseline['corrections'])}")

        errors: List[str] = []
        extracted: dict = {}

        if source == "structured":
            merged, merge_report = merge_structured(
                baseline, payload["meta"],
                force_update=bool(payload.get("force_update", False)))
        else:
            if source == "doc":
                docs = [{"doc_type": payload["doc_type"],
                         "content": payload["content"],
                         "name": payload.get("name") or payload["doc_type"]}]
            else:
                docs = payload["docs"]
            extracted = self._extract_with_cache(docs)
            for e in extracted.get("errors", []):
                errors.append(f"doc[{e['name']}] {e['error']}")
                _log(f"  doc extraction error: {errors[-1]}")
            # no successful extraction at all -> explicit failure, no half-write
            n_ok = (len(extracted.get("terms", []))
                    + len(extracted.get("calibers", []))
                    + len(extracted.get("corrections", [])))
            if n_ok == 0:
                raise RuntimeError(
                    "no terms/calibers/corrections extracted from document(s): "
                    + "; ".join(errors) if errors else "empty extraction")
            from .doc_extract import merge_into_meta

            merged, merge_report = merge_into_meta(baseline, extracted)

        self._report_merge(merge_report)
        # a no-op merge (nothing new at all) is a legit idempotent re-ingest,
        # but a *changed* payload that adds nothing is worth flagging
        added_total = sum(v for v in merge_report.get("added", {}).values())
        if added_total == 0:
            _log("merge added nothing (already present) — idempotent re-ingest")

        # diff write: PRIMARY_KEY upsert, idempotent, never drops graph state
        counts = ingest_to_hg.ingest(merged, self._url, self._graph, diff=True)
        self._meta = merged

        v_report = ingest_to_hg.validate_metadata_rules(merged)
        ingest_to_hg._log_validation(v_report)

        vector = self.refresh_vector_index()

        report = IngestReport(
            ok=True,
            source=source,
            baseline={k: len(v) for k, v in baseline.items()
                      if isinstance(v, list)},
            extracted={k: (len(v) if isinstance(v, list) else v)
                       for k, v in extracted.items() if k != "errors"},
            merge=merge_report,
            written=counts,
            validation=v_report,
            vector=vector,
            errors=errors,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
        _log(f"done: written={counts} errors={len(errors)} "
             f"elapsed={report.elapsed_ms:.0f}ms")
        return report

    def _get_llm(self) -> Callable:
        if self._llm is None:
            from hugegraph_llm.models.llms.init_llm import LLMs

            self._llm = LLMs().get_chat_llm()
        return self._llm

    def _extract_with_cache(self, docs: List[dict]) -> dict:
        """Per-document LLM extraction with a content-keyed cache.

        Each document is extracted independently (a failure only fails that
        document and is reported); on cache hit the previous normalised product
        is reused, making re-ingests deterministic. Returns the same shape as
        ``doc_extract.extract_documents`` (``terms/term_bindings/calibers/
        corrections/errors``).
        """
        out: dict = {"terms": [], "term_bindings": [], "calibers": [],
                     "corrections": [], "errors": []}
        for doc in docs:
            doc_type = doc.get("doc_type")
            content = doc.get("content") or ""
            name = doc.get("name") or doc_type
            if doc_type not in _NORMALIZERS:
                out["errors"].append({"name": name,
                                      "error": f"unknown doc_type {doc_type}"})
                continue
            if not content.strip():
                out["errors"].append({"name": name, "error": "empty content"})
                continue
            key = hashlib.sha1(f"{doc_type}\x00{content}".encode("utf-8")).hexdigest()
            if self._cache_extractions and key in self._extract_cache:
                part = self._extract_cache[key]
                _log(f"  doc cache hit [{name}] (reuse {len(part.get('terms', []))}"
                     f" terms / {len(part.get('calibers', []))} calibers / "
                     f"{len(part.get('corrections', []))} corrections)")
            else:
                try:
                    items = extract_document(content, doc_type, self._get_llm())
                except Exception as e:  # noqa: BLE001
                    out["errors"].append({"name": name,
                                          "error": f"{type(e).__name__}: {e}"})
                    continue
                part = _NORMALIZERS[doc_type](items)
                if self._cache_extractions:
                    self._extract_cache[key] = part
                _log(f"  doc extracted [{name}]: {len(part.get('terms', []))} terms"
                     f" / {len(part.get('calibers', []))} calibers / "
                     f"{len(part.get('corrections', []))} corrections")
            for k, v in part.items():
                out[k].extend(v)
        return out

    @staticmethod
    def _report_merge(merge_report: dict) -> None:
        added = merge_report.get("added", {})
        _log("merge added: " + " ".join(
            f"{k}={v}" for k, v in added.items() if v))
        for u in merge_report.get("unmatched", []):
            _log(f"  unmatched {u['type']}: {u.get('term') or u.get('caliber')} "
                 f"-> {u.get('column')}")
        for k in ("dup_terms", "dup_calibers", "dup_corrections"):
            dups = merge_report.get(k, [])
            if dups:
                _log(f"  skipped duplicates ({k}): {len(dups)}")

    def close(self) -> None:
        """Release resources (vector store / engine) if they own any."""
        self._vector_store = None
        self._schema = None

    def __enter__(self) -> "Nl2SqlIngester":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_embedder(spec: str) -> Callable:
    """Load an embedder factory from ``module:attr`` (e.g. the nl2sql_tools
    ``p2_embedder:make_embedder``) and call it."""
    module, _, attr = spec.partition(":")
    import importlib

    mod = importlib.import_module(module)
    factory = getattr(mod, attr or "make_embedder")
    return factory()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Unified NL2SQL metadata ingestion (structured + docs).")
    ap.add_argument("--source", choices=("structured", "doc", "doc_bundle"),
                    required=True, help="input kind (the only flag a caller needs)")
    ap.add_argument("--meta", help="meta.json path (--source structured)")
    ap.add_argument("--doc-type", choices=_SUPPORTED_DOC_TYPES,
                    help="document type (--source doc)")
    ap.add_argument("--doc", help="document file path (--source doc)")
    ap.add_argument("--name", help="document name for logs (--source doc)")
    ap.add_argument("--docs", help="comma-separated 'type:path' list (--source doc_bundle)")
    ap.add_argument("--force-update", action="store_true",
                    help="structured: overwrite colliding table/column/term entries")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--graph", default="kg_rag")
    ap.add_argument("--embed", default=None,
                    help="embedder factory 'module:attr' to enable the P2 "
                         "vector channel, e.g. nl2sql_tools.p2_embedder:make_embedder")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args(argv)

    if args.source == "structured" and not args.meta:
        ap.error("--source structured requires --meta")
    if args.source == "doc" and not (args.doc and args.doc_type):
        ap.error("--source doc requires --doc and --doc-type")
    if args.source == "doc_bundle" and not args.docs:
        ap.error("--source doc_bundle requires --docs")

    embedder = None
    if args.embed:
        try:
            embedder = _load_embedder(args.embed)
        except Exception as exc:  # noqa: BLE001
            ap.error(f"--embed failed to load {args.embed!r}: {exc}")

    if args.source == "structured":
        with open(args.meta, encoding="utf-8") as f:
            payload = {"source": "structured", "meta": json.load(f)}
        if args.force_update:
            payload["force_update"] = True
    elif args.source == "doc":
        with open(args.doc, encoding="utf-8") as f:
            payload = {"source": "doc", "doc_type": args.doc_type,
                       "content": f.read()}
        if args.name:
            payload["name"] = args.name
    else:
        docs = []
        for spec in args.docs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            doc_type, _, path = spec.partition(":")
            with open(path, encoding="utf-8") as f:
                docs.append({"doc_type": doc_type, "content": f.read(),
                             "name": path})
        payload = {"source": "doc_bundle", "docs": docs}

    with Nl2SqlIngester(url=args.url, graph=args.graph, embedder=embedder,
                        timeout=args.timeout) as ingester:
        report = ingester.ingest(payload)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok and not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
