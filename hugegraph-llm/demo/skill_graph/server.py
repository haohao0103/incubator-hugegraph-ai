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

"""Skill Graph 可视化 HTTP 服务（部署到 STG 后用浏览器直接访问）。

路由::

    GET /               当前 active build 的交互式可视化页面
    GET /api/stats      当前 build 的元信息（build_id / 节点边数 / embedder 指纹）
    GET /healthz        存活检查

设计要点::

* 页面渲染读取全图（运维可视化的预期行为），走带完整性校验的
  ``read_build``；快照按 (deployment, build_id) 进程内 LRU 缓存，
  发布/回滚后下一次请求自动切到新版本。
* 无 active pointer（从未 publish）或读出的构建不完整 → 503 + 明确
  错误信息，绝不渲染一个看似正常的空白页面。
* ``?refresh=1`` 绕过缓存强制重读。

部署（STG, 无系统 python3 的 CentOS 7）::

    # 1) 代码（server.py + 7 个 skill_graph 模块，namespace package，
    #    不带各级 __init__.py，避免拉起 heavy 的主包依赖链）
    /opt/skill-graph-viz/
      ├── skill_graph_server.py
      └── hugegraph_llm/engines/skill_graph/{schema,errors,manifest,viz,
                                             hg_store,ob_store,service_loader}.py

    # 2) 运行时 + 依赖（仅 flask / pyhugegraph / pymysql）
    /opt/miniconda3/envs/skillgraph/bin/python

    # 3) 配置（systemd EnvironmentFile, chmod 600）
    SKILL_GRAPH_HG_URL=http://10.129.33.121:8080
    SKILL_GRAPH_HG_GRAPHSPACE=dataleap
    SKILL_GRAPH_HG_GRAPH=skillgraph
    OCEANBASE_HOST=... OCEANBASE_USER=... OCEANBASE_PASSWORD=...
    OCEANBASE_DATABASE=dataleap_graph_rag

    # 4) systemd: skill-graph-viz.service → 0.0.0.0:8766
    # 访问: http://10.129.33.121:8766/

本地调试::

    cd incubator-hugegraph-ai/hugegraph-llm
    python demo/skill_graph/server.py --port 8766
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# demo 目录无包环境：把 repo 的 src 加入 sys.path（本地开发）；
# STG 部署时该目录不存在，代码部署到 /opt/skill-graph-viz 并按
# namespace package 导入，两种方式都能跑。
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from flask import Flask, Response, jsonify, request  # noqa: E402

from hugegraph_llm.engines.skill_graph.errors import GraphReadError  # noqa: E402
from hugegraph_llm.engines.skill_graph.service_loader import (  # noqa: E402
    ServiceGraphCache,
    load_graph_from_services,
)
from hugegraph_llm.engines.skill_graph.viz import highlight_payload, render_html  # noqa: E402

log = logging.getLogger("skill_graph_viz")

app = Flask(__name__)

# 运维页面按 (deployment, build) 缓存整图快照；两份容量保证回滚后前一版
# 仍然热着，指针切换后下一次请求自动加载新构建。
_cache = ServiceGraphCache(max_entries=2)
_page_cache: Dict[str, Any] = {"key": None, "html": None, "ts": 0.0}


def _load_env() -> None:
    """Load .env from the repo root (local runs). STG uses systemd env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def _hg_kwargs() -> Dict[str, Any]:
    return {
        "hg_url": os.environ.get("SKILL_GRAPH_HG_URL"),
        "hg_user": os.environ.get("SKILL_GRAPH_HG_USER", "admin"),
        "hg_pwd": os.environ.get("SKILL_GRAPH_HG_PWD", "admin"),
        "hg_graphspace": os.environ.get("SKILL_GRAPH_HG_GRAPHSPACE"),
        "hg_graph": os.environ.get("SKILL_GRAPH_HG_GRAPH"),
    }


def _ob_kwargs() -> Dict[str, Any]:
    return {
        "ob_host": os.environ.get("OCEANBASE_HOST"),
        "ob_port": int(os.environ.get("OCEANBASE_PORT", "3306")),
        "ob_user": os.environ.get("OCEANBASE_USER"),
        "ob_password": os.environ.get("OCEANBASE_PASSWORD"),
        "ob_database": os.environ.get("OCEANBASE_DATABASE"),
    }


def _graph(force_refresh: bool):
    return load_graph_from_services(cache=_cache, force_refresh=force_refresh, **_hg_kwargs(), **_ob_kwargs())


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "ts": time.time()})


@app.route("/favicon.ico")
def favicon():
    # No icon asset is shipped; answer 204 so browsers do not log a 404 error
    # in the console on every visit (noise for E2E/ops alike).
    return Response(status=204)


@app.route("/api/stats")
def api_stats():
    try:
        graph = _graph(force_refresh=False)
    except GraphReadError as exc:
        return jsonify({"error": str(exc)}), 503
    manifest = getattr(graph, "manifest", None)
    stats = graph.stats()
    payload: Dict[str, Any] = {
        "build_id": getattr(manifest, "build_id", None),
        "ob_table": getattr(manifest, "ob_table", None),
        "embedder_id": getattr(manifest, "embedder_id", None),
        "embedder_dim": getattr(manifest, "embedder_dim", None),
        "content_hash": getattr(manifest, "content_hash", None),
        "activated_at": getattr(manifest, "activated_at", None),
        "skills": stats.get("skills"),
        "categories": stats.get("categories"),
        "nodes": stats.get("nodes"),
        "edges": stats.get("edges"),
    }
    return jsonify(payload)


@app.route("/api/query")
def api_query():
    """Run a live dependency-aware retrieval and return the highlight payload.

    This is the demo hook for the visualization page: the browser calls it,
    gets ``{highlight, value}`` (the same protocol a server-rendered result
    uses) and re-highlights the force graph in place — showing seed hits vs.
    graph-added prerequisites on the real production query path.
    """
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "missing query parameter: q"}), 400
    try:
        result = _retrieve(query)
    except Exception as exc:  # noqa: BLE001 - surfaced to the browser verbatim
        log.exception("query failed: %s", query)
        return jsonify({"error": str(exc)}), 502
    payload = highlight_payload(result)
    payload["degraded"] = getattr(result, "degraded", False)
    payload["warnings"] = list(getattr(result, "warnings", []) or [])
    return jsonify(payload)


def _retrieve(query: str):
    """Query-path retrieval: seeds from OceanBase, expansion in HugeGraph."""
    from hugegraph_llm.engines.skill_graph.retriever import ServiceSkillRetriever

    retriever = ServiceSkillRetriever(
        embedder=_query_embedder(),
        hg_store=None,
        ob_host=os.environ.get("OCEANBASE_HOST"),
        ob_port=int(os.environ.get("OCEANBASE_PORT", "3306")),
        ob_user=os.environ.get("OCEANBASE_USER"),
        ob_password=os.environ.get("OCEANBASE_PASSWORD"),
        ob_database=os.environ.get("OCEANBASE_DATABASE"),
        **_hg_kwargs(),
    )
    try:
        return retriever.retrieve(query, top_k_seeds=5, token_budget=4000)
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()


_embedder_lock = threading.Lock()
_embedder_instance = None


def _query_embedder():
    """Lazily build the query embedder once (HTTP call per query stays, init once)."""
    global _embedder_instance
    with _embedder_lock:
        if _embedder_instance is None:
            from hugegraph_llm.engines.skill_graph.embedder import resolve_embedder

            # strict: production build vectors are 1536-d; a silent hashing
            # fallback would compare different vector spaces and return noise.
            _embedder_instance = resolve_embedder(strict=True)
        return _embedder_instance


@app.route("/")
def index():
    refresh = os.environ.get("SKILL_GRAPH_FORCE_REFRESH", "") == "1"
    try:
        graph = _graph(force_refresh=refresh)
    except GraphReadError as exc:
        # Never render a blank-but-200 page: an unpublished or broken build
        # must be loudly visible to whoever opens the URL.
        log.error("cannot serve visualization: %s", exc)
        body = (
            "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<title>Skill Graph — 不可用</title></head>"
            "<body style='font-family:sans-serif;background:#0f1620;color:#e8eef5;"
            "padding:40px;line-height:1.8'>"
            "<h1>Skill Graph 暂不可用</h1>"
            f"<p>当前 HugeGraph 中没有可用的已发布构建（active pointer）：</p>"
            f"<pre style='background:#16202b;padding:12px;border-radius:8px;"
            f"white-space:pre-wrap'>{str(exc)}</pre>"
            "<p>先执行 publish 发布一个构建，然后刷新本页。</p></body></html>"
        )
        return Response(body, status=503, mimetype="text/html")

    key = f"{getattr(getattr(graph, 'manifest', None), 'build_id', '')}|{id(graph)}"
    now = time.time()
    if _page_cache["key"] != key or refresh:
        _page_cache["key"] = key
        _page_cache["html"] = render_html(graph, title="Skill Graph — STG")
        _page_cache["ts"] = now
        log.info("rendered page for %s", key)
    return Response(_page_cache["html"], mimetype="text/html")


def main() -> int:
    parser = argparse.ArgumentParser(description="Skill Graph visualization server")
    parser.add_argument("--host", default=os.environ.get("SKILL_GRAPH_VIZ_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SKILL_GRAPH_VIZ_PORT", "8766")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _load_env()
    print(
        json.dumps(
            {
                "msg": "skill graph visualization server",
                "hg_url": os.environ.get("SKILL_GRAPH_HG_URL"),
                "graphspace": os.environ.get("SKILL_GRAPH_HG_GRAPHSPACE"),
                "graph": os.environ.get("SKILL_GRAPH_HG_GRAPH"),
                "listen": f"http://{args.host}:{args.port}/",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
