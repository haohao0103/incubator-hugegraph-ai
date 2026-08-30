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
NL2SQL HTTP API -- expose the graph-enhanced Text2SQL pipeline to the platform.

The pipeline (``hugegraph_llm.nl2sql``) is usable whole or in parts:
``link`` (L1 schema linking), ``join_path`` (L2 join discovery),
``communities`` (L3 subject domains), ``schema_context`` (narrowed prompt
context) and ``run`` (full pipeline -> SQL). This module mounts those four
entry points as HTTP endpoints on the shared ``api_auth`` router, so the
upstream platform can call them directly instead of importing the library.

Schema source
-------------
A schema is built deterministically from warehouse metadata via
``SchemaGraphBuilder``. This module ships a file-based loader (see
``resources/example_warehouse.json``) so the API is runnable and testable
immediately. **The production path is to ingest the warehouse schema from the
HugeGraph KG (the ``hg-rag-hmsgraphrag`` anchor case) -- that loader is the
natural follow-up and should populate the same ``SchemaGraphBuilder``.**

Engine selection
-----------------
The pipeline is built once (lazily) and cached. It uses ``VermeerEngine`` when
a Vermeer cluster is reachable on ``VERMEER_MASTER`` (default
``http://127.0.0.1:6688``), otherwise it falls back to the in-process
``LocalEngine``. The public API is identical either way.

LLM
---
``/nl2sql/run`` needs an LLM. It is obtained from the global ``llm_settings``
singleton (``LLMs().get_chat_llm()``), the same chat model the rest of
``hugegraph-llm`` uses.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.nl2sql.engine import (
    LocalEngine,
    VermeerClient,
    VermeerEngine,
)
from hugegraph_llm.nl2sql.hugegraph_schema_source import build_schema_from_hugegraph
from hugegraph_llm.nl2sql.linking.schema_linker import LinkedItem
from hugegraph_llm.nl2sql.llm_gateway import (
    LLMGateway,
    LLMGatewayError,
    LLMGatewayOpenError,
)
from hugegraph_llm.nl2sql.pipeline import NL2SQLPipeline
from hugegraph_llm.nl2sql.schema_graph.builder import SchemaGraphBuilder
from hugegraph_llm.nl2sql.schema_graph.model import Column, SchemaGraph, Table, Term
from hugegraph_llm.utils.log import log

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent                       # .../hugegraph_llm/api
_PKG_DIR = _THIS_DIR.parent                                       # .../hugegraph_llm
_DEFAULT_SCHEMA_PATH = _PKG_DIR / "nl2sql" / "resources" / "example_warehouse.json"
_VERMEER_MASTER = os.getenv("VERMEER_MASTER", "http://127.0.0.1:6688")

# Production path: when set, the default pipeline is built from a live HugeGraph
# KG instead of the bundled file. e.g. NL2SQL_HG_GRAPH=kg_rag (with
# NL2SQL_HG_URL defaulting to http://127.0.0.1:8081).
_HG_GRAPH_ENV = os.getenv("NL2SQL_HG_GRAPH")
_HG_URL_ENV = os.getenv("NL2SQL_HG_URL", "http://127.0.0.1:8081")

# --- production hardening knobs (all optional, all env-gated) ---------------
# LLM gateway: retry / timeout / circuit breaker around the chat endpoint.
_LLM_MAX_RETRIES = int(os.getenv("NL2SQL_LLM_MAX_RETRIES", "3"))
_LLM_TIMEOUT_S = float(os.getenv("NL2SQL_LLM_TIMEOUT_S", "30"))
_LLM_CIRCUIT_THRESHOLD = int(os.getenv("NL2SQL_LLM_CIRCUIT_THRESHOLD", "5"))
_LLM_CIRCUIT_RESET_S = float(os.getenv("NL2SQL_LLM_CIRCUIT_RESET_S", "60"))

# Schema cache: rebuild from a local snapshot instead of re-pulling HugeGraph.
# NL2SQL_SCHEMA_CACHE=<path> enables disk caching; NL2SQL_SCHEMA_TTL_S bounds
# its freshness, and the pipeline is lazily re-pulled once past it.
_SCHEMA_CACHE = os.getenv("NL2SQL_SCHEMA_CACHE") or None
_SCHEMA_TTL_S = int(os.getenv("NL2SQL_SCHEMA_TTL_S", "3600"))

# Result cache: LRU with TTL for link / schema_context (high-frequency Qs).
_RESULT_CACHE_SIZE = int(os.getenv("NL2SQL_RESULT_CACHE_SIZE", "256"))
_RESULT_CACHE_TTL_S = float(os.getenv("NL2SQL_RESULT_CACHE_TTL_S", "60"))

# Supported SQL dialects for /nl2sql/run output (sqlglot write dialects).
_DIALECTS = {"hive", "trino", "spark", "mysql", "duckdb", "clickhouse"}

# --- LLM gateway (module-level, shared by run + keyword extraction) ---------
_LLM_GATEWAY = LLMGateway(
    llm_factory=LLMs().get_chat_llm,
    max_retries=_LLM_MAX_RETRIES,
    timeout_s=_LLM_TIMEOUT_S,
    circuit_fail_threshold=_LLM_CIRCUIT_THRESHOLD,
    circuit_reset_s=_LLM_CIRCUIT_RESET_S,
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class LinkRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, description="Override default retrieval size")
    min_score: Optional[float] = Field(
        default=None,
        description="Floor on the top result's score; below it the question is "
                    "treated as out of the knowledge base (out_of_kb=true)",
    )


class JoinPathRequest(BaseModel):
    source: str
    target: str


class CommunitiesRequest(BaseModel):
    resolution: float = 1.0
    algorithm: str = "louvain"


class SchemaContextRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    include_joins: bool = False
    include_global: bool = Field(
        default=False,
        description="Append same-subject-domain sibling tables (Global context)",
    )
    min_score: Optional[float] = None


class RunRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    include_joins: bool = True
    dialect: Optional[str] = Field(
        default=None,
        description="Transpile the generated SQL to a target dialect via sqlglot: "
                    "hive/trino/spark/mysql/duckdb/clickhouse (default: leave as generated)",
    )


class SchemaMetadata(BaseModel):
    """Deterministic warehouse metadata, mirrors SchemaGraphBuilder inputs."""

    tables: List[dict] = Field(default_factory=list)
    columns: List[dict] = Field(default_factory=list)
    foreign_keys: List[List[str]] = Field(default_factory=list)
    lineage: List[List[str]] = Field(default_factory=list)
    query_logs: List[List[str]] = Field(default_factory=list)
    terms: List[dict] = Field(default_factory=list)
    term_bindings: List[List[str]] = Field(default_factory=list)


class ReloadRequest(BaseModel):
    metadata: SchemaMetadata


class HgLoadRequest(BaseModel):
    """Pull the schema from a live HugeGraph KG (the production schema source)."""

    url: str = Field(default="http://127.0.0.1:8081", description="HugeGraph REST base")
    graph: str = Field(default="kg_rag", description="Graph name holding warehouse metadata")
    infer_foreign_keys: bool = Field(
        default=True,
        description="Infer weak FKs from shared *_id column names when none are declared",
    )
    use_embedding: bool = Field(
        default=False,
        description="Enable P2 semantic seed recall (requires NL2SQL_EMBEDDING env to point at a configured backend)",
    )


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def build_schema(meta: dict) -> SchemaGraph:
    """Populate a SchemaGraphBuilder from a metadata dict (file or request)."""
    b = SchemaGraphBuilder()
    for t in meta.get("tables", []):
        b.add_table(
            Table(
                name=t["name"],
                database=t.get("database", ""),
                comment=t.get("comment", ""),
                is_fact=bool(t.get("is_fact", False)),
            )
        )
    for c in meta.get("columns", []):
        b.add_column(
            Column(
                name=c["name"],
                table=c["table"],
                data_type=c.get("data_type", ""),
                comment=c.get("comment", ""),
            )
        )
    for fk in meta.get("foreign_keys", []):
        if len(fk) == 2:
            b.add_foreign_key(fk[0], fk[1])
    for lg in meta.get("lineage", []):
        if len(lg) == 2:
            b.add_lineage(lg[0], lg[1])
    if meta.get("query_logs"):
        b.add_query_logs(meta["query_logs"])
    for tm in meta.get("terms", []):
        b.add_term(Term(name=tm["name"], comment=tm.get("comment", "")))
    for tb in meta.get("term_bindings", []):
        if len(tb) == 2:
            b.bind_term(tb[0], tb[1])
    return b.build()


# ---------------------------------------------------------------------------
# Engine + pipeline (lazily built, cached)
# ---------------------------------------------------------------------------


def _make_engine(schema: SchemaGraph):
    """Vermeer when reachable, else in-process LocalEngine."""
    try:
        client = VermeerClient(base_url=_VERMEER_MASTER)
        if client.healthcheck():
            log.info("nl2sql api: using VermeerEngine at %s", _VERMEER_MASTER)
            return VermeerEngine(schema, client=client)
    except Exception as exc:  # noqa: BLE001 -- fall back gracefully
        log.warning("nl2sql api: vermeer unreachable, LocalEngine fallback: %s", exc)
    return LocalEngine(schema)


def _make_embedder() -> Optional[Callable[[str], List[float]]]:
    """Build a semantic embedder for P2 linking, gated by ``NL2SQL_EMBEDDING``.

    Reuses the project's configured embedding backend (OpenAI / Ollama /
    LiteLLM) from ``llm_settings``. Returns ``None`` when the env flag is unset
    or the backend is unavailable, so P2 degrades cleanly to lexical linking.
    """
    if not os.getenv("NL2SQL_EMBEDDING"):
        return None
    try:
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings

        inst = Embeddings().get_embedding()
        return lambda text: inst.get_text_embedding(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("nl2sql api: embedding unavailable, P2 disabled: %s", exc)
        return None


_PIPELINE: Optional[NL2SQLPipeline] = None
_PIPELINE_TS: float = 0.0
_LOCK = threading.Lock()


def _build_pipeline() -> NL2SQLPipeline:
    """Construct the pipeline from the default schema source (file or HG)."""
    if _HG_GRAPH_ENV:
        log.info("nl2sql api: default schema from HugeGraph %s/%s",
                 _HG_URL_ENV, _HG_GRAPH_ENV)
        schema = build_schema_from_hugegraph(
            url=_HG_URL_ENV, graph=_HG_GRAPH_ENV,
            cache_path=_SCHEMA_CACHE, cache_ttl_s=_SCHEMA_TTL_S,
        )
    else:
        meta = json.loads(_DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
        schema = build_schema(meta)
    return NL2SQLPipeline(
        schema,
        engine=_make_engine(schema),
        embedder=_make_embedder(),
        keyword_extractor=_make_keyword_extractor(),
    )


def get_pipeline() -> NL2SQLPipeline:
    """Return the cached pipeline, rebuilding when the schema TTL has passed.

    The default schema source is the bundled file unless ``NL2SQL_HG_GRAPH`` is
    set, in which case it is pulled live from the named HugeGraph KG (the
    production path). Rebuilds are lazy (checked on call) and lock-guarded.
    """
    global _PIPELINE, _PIPELINE_TS
    now = time.monotonic()
    if _PIPELINE is None or now - _PIPELINE_TS > _SCHEMA_TTL_S:
        with _LOCK:
            now = time.monotonic()
            if _PIPELINE is None or now - _PIPELINE_TS > _SCHEMA_TTL_S:
                try:
                    _PIPELINE = _build_pipeline()
                    _PIPELINE_TS = time.monotonic()
                except Exception as exc:  # noqa: BLE001 - keep serving stale
                    log.warning("nl2sql api: schema rebuild failed, serving stale: %s", exc)
    return _PIPELINE


def _make_keyword_extractor() -> Optional[Callable[[str], List[str]]]:
    """LLM keyword pre-extraction for linking, gated by ``NL2SQL_KEYWORD_LLM``.

    Returns ``None`` when the env flag is unset or the LLM is unavailable, so
    the pipeline falls back to linking on the raw question only.
    """
    if not os.getenv("NL2SQL_KEYWORD_LLM"):
        return None

    def extract(question: str) -> List[str]:
        prompt = (
            "你是数仓元数据检索助手。从用户问题里提取用于检索库表/字段的关键词"
            "（表名、字段名、业务术语、口径词均可）。只输出关键词，用逗号分隔，"
            "不要任何解释或标点修饰。\n问题：" + str(question)
        )
        try:
            raw = _llm_callable(prompt)
        except LLMGatewayOpenError:
            log.warning("nl2sql api: keyword LLM circuit open, lexical-only linking")
            return []
        kws = [k.strip() for k in raw.replace("，", ",").split(",") if k.strip()]
        return kws[:8]  # cap keyword count

    return extract


def _llm_callable(prompt: str) -> str:
    """Chat call through the gateway (retry / timeout / circuit breaker)."""
    return _LLM_GATEWAY(prompt)


# ---------------------------------------------------------------------------
# Result cache (LRU + TTL) and health reporting
# ---------------------------------------------------------------------------


class _ResultCache:
    """Thread-safe LRU with TTL for endpoint results (high-frequency questions)."""

    def __init__(self, maxsize: int, ttl_s: float) -> None:
        self._data: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl = ttl_s

    def get(self, key: tuple) -> Optional[dict]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, val = item
            if time.monotonic() - ts > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return val

    def put(self, key: tuple, val: dict) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), val)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._data), "maxsize": self._maxsize,
                    "ttl_s": self._ttl}


_RESULT_CACHE = _ResultCache(_RESULT_CACHE_SIZE, _RESULT_CACHE_TTL_S)


def nl2sql_health() -> dict:
    """Dependency + liveness state for the app's /healthz endpoint."""
    pipe = get_pipeline()
    return {
        "status": "ok",
        "schema_source": _HG_GRAPH_ENV or "bundled-file",
        "schema_cache": _SCHEMA_CACHE,
        "schema_ttl_s": _SCHEMA_TTL_S,
        "schema_age_s": round(time.monotonic() - _PIPELINE_TS, 1) if _PIPELINE else None,
        "engine": pipe.capabilities.name if pipe else None,
        "tables": len(pipe._schema.tables()) if pipe else None,
        "llm_gateway": _LLM_GATEWAY.state(),
        "result_cache": _RESULT_CACHE.stats(),
    }


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _item_to_dict(it: LinkedItem) -> dict:
    return {
        "node_type": it.node_type,
        "name": it.name,
        "score": it.score,
        "table": it.table,
        "properties": it.properties,
    }


def _path_to_dict(path) -> Optional[dict]:
    if path is None:
        return None
    return {
        "tables": path.tables,
        "total_cost": path.total_cost,
        "all_proven": path.all_proven,
        "steps": [
            {
                "left_table": s.left_table,
                "right_table": s.right_table,
                "left_column": s.left_column,
                "right_column": s.right_column,
                "edge_type": s.edge_type,
                "cost": s.cost,
                "proven": s.proven,
                "on_clause": s.to_sql(),
            }
            for s in path.steps
        ],
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_router = APIRouter()


@_router.post("/nl2sql/link")
def nl2sql_link(req: LinkRequest):
    """L1: top-k linked tables/columns for a natural-language question.

    With ``min_score`` set, a question whose best hit scores below the floor
    (or that matches nothing at all) is flagged ``out_of_kb`` so the caller
    can refuse to answer instead of hallucinating a table.
    """
    cache_key = ("link", req.question, req.top_k, req.min_score)
    cached = _RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    items = get_pipeline().link(req.question, top_k=req.top_k)
    best = max((i.score for i in items), default=0.0)
    out_of_kb = bool(not items or (req.min_score is not None and best < req.min_score))
    resp = {"question": req.question,
            "items": [_item_to_dict(i) for i in items]}
    if out_of_kb:
        resp["out_of_kb"] = True
        resp["message"] = "问题超出当前知识库范围，建议人工确认后再查询"
    _RESULT_CACHE.put(cache_key, resp)
    return resp


@_router.post("/nl2sql/join_path")
def nl2sql_join_path(req: JoinPathRequest):
    """L2: shortest join path (with proven ON clauses) between two tables."""
    path = get_pipeline().join_path(req.source, req.target)
    return {"source": req.source, "target": req.target, "path": _path_to_dict(path)}


@_router.post("/nl2sql/communities")
def nl2sql_communities(req: CommunitiesRequest):
    """L3: partition the warehouse into subject domains."""
    return {
        "domains": get_pipeline().communities(
            resolution=req.resolution, algorithm=req.algorithm
        )
    }


@_router.post("/nl2sql/schema_context")
def nl2sql_schema_context(req: SchemaContextRequest):
    """Narrowed, flat schema string for a prompt (the SuperSonic-style view).

    ``include_global`` appends same-subject-domain sibling tables; with
    ``min_score`` set, out-of-knowledge-base questions are flagged explicitly.
    """
    pipe = get_pipeline()
    cache_key = ("schema_context", req.question, req.top_k, req.include_joins,
                 req.include_global, req.min_score)
    cached = _RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ctx = pipe.schema_context(
        req.question, top_k=req.top_k, include_joins=req.include_joins,
        include_global=req.include_global,
    )
    resp = {"question": req.question, "schema_context": ctx}
    if not ctx:
        resp["out_of_kb"] = True
        resp["message"] = "问题超出当前知识库范围，建议人工确认后再查询"
    elif req.min_score is not None:
        best = max((i.score for i in pipe.link(req.question, top_k=req.top_k)),
                   default=0.0)
        if best < req.min_score:
            resp["out_of_kb"] = True
            resp["message"] = "问题超出当前知识库范围，建议人工确认后再查询"
    _RESULT_CACHE.put(cache_key, resp)
    return resp


@_router.post("/nl2sql/run")
def nl2sql_run(req: RunRequest):
    """Full pipeline: question -> narrowed context -> SQL via the configured LLM.

    The LLM call goes through the gateway: transient failures are retried with
    backoff, and if the circuit breaker is open the endpoint degrades to a 503
    with the lexical schema context (so the platform can still build a prompt).
    """
    if req.dialect and req.dialect not in _DIALECTS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported dialect {req.dialect!r}; choose from {sorted(_DIALECTS)}",
        )
    base = get_pipeline()
    pipe = NL2SQLPipeline(
        base._schema, llm=_llm_callable, engine=base._engine
    )
    try:
        result = pipe.run(req.question, top_k=req.top_k, include_joins=req.include_joins)
    except LLMGatewayOpenError as exc:
        raise HTTPException(status_code=503, detail=f"LLM temporarily unavailable: {exc}")
    except LLMGatewayError as exc:
        raise HTTPException(status_code=502, detail=f"LLM gateway error: {exc}")
    sql = result.sql
    if sql and req.dialect:
        try:
            import sqlglot

            sql = sqlglot.transpile(sql, write=req.dialect)[0]
        except Exception as exc:  # noqa: BLE001 - non-fatal, keep original SQL
            log.warning("nl2sql api: dialect transpile %s failed: %s", req.dialect, exc)
    return {
        "question": result.question,
        "sql": sql,
        "schema_context": result.schema_context,
        "tables": result.tables,
    }


@_router.post("/nl2sql/reload")
def nl2sql_reload(req: ReloadRequest):
    """Rebuild the cached pipeline from inline metadata (engine auto-selected)."""
    global _PIPELINE, _PIPELINE_TS
    meta = req.metadata.model_dump()
    schema = build_schema(meta)
    with _LOCK:
        _PIPELINE = NL2SQLPipeline(
            schema, engine=_make_engine(schema),
            keyword_extractor=_make_keyword_extractor(),
        )
        _PIPELINE_TS = time.monotonic()
    return {
        "status": "reloaded",
        "engine": _PIPELINE.capabilities.name,
        "tables": len(schema.tables()),
    }


@_router.post("/nl2sql/load_hugegraph")
def nl2sql_load_hugegraph(req: HgLoadRequest):
    """Rebuild the cached pipeline from a live HugeGraph KG (production path).

    Swaps the in-memory pipeline: the schema is ingested deterministically from
    the warehouse metadata KG, then the engine is auto-selected (Vermeer when
    reachable, else in-process LocalEngine). All other NL2SQL endpoints
    (/link, /join_path, /communities, /schema_context, /run) then operate on
    the KG-derived schema.
    """
    global _PIPELINE, _PIPELINE_TS
    schema = build_schema_from_hugegraph(
        url=req.url, graph=req.graph, infer_foreign_keys=req.infer_foreign_keys,
        cache_path=_SCHEMA_CACHE, cache_ttl_s=_SCHEMA_TTL_S,
    )
    embedder = _make_embedder() if req.use_embedding else None
    with _LOCK:
        _PIPELINE = NL2SQLPipeline(
            schema, engine=_make_engine(schema), embedder=embedder,
            keyword_extractor=_make_keyword_extractor(),
        )
        _PIPELINE_TS = time.monotonic()
    return {
        "status": "loaded",
        "source": f"hugegraph:{req.graph}",
        "engine": _PIPELINE.capabilities.name,
        "embedding": embedder is not None,
        "tables": len(schema.tables()),
        "columns": len(schema.columns()),
        "terms": len(schema.terms()),
        "edges": len(schema.edges),
    }


def nl2sql_http_api(router: APIRouter) -> None:
    """Register all NL2SQL routes onto the shared ``api_auth`` router."""
    router.include_router(_router)
