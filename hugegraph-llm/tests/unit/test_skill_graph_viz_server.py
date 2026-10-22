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

"""Unit tests for the skill-graph visualization HTTP server.

The E2E suite (``demo/skill_graph/test_viz_e2e.py``) drives a real browser
against the STG deployment; these tests pin the server's own behaviour with
fakes: fail-fast 503 when no active build exists, page/stats rendering from a
pointered snapshot, and the favicon no-op.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from hugegraph_llm.engines.skill_graph.errors import GraphReadError
from hugegraph_llm.engines.skill_graph.graph_builder import GraphBuilder
from hugegraph_llm.engines.skill_graph.manifest import build_manifest
from hugegraph_llm.engines.skill_graph.schema import SkillGraph

_SERVER = Path(__file__).resolve().parents[2] / "demo" / "skill_graph" / "server.py"


def _load_server_module():
    """Import demo/skill_graph/server.py without the demo package layout."""
    spec = importlib.util.spec_from_file_location("skill_graph_viz_server", _SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load_server_module()
    module._page_cache.update({"key": None, "html": None, "ts": 0.0})
    module._cache.clear()
    yield module
    sys.modules.pop("skill_graph_viz_server", None)


def _built_graph() -> SkillGraph:
    graph = SkillGraph()
    builder = GraphBuilder(embedder=None, similarity_threshold=0.0)
    graph, _ = builder.build(graph)
    graph.manifest = build_manifest(
        graph,
        embedder_id="hash:test",
        embedder_dim=8,
        ob_table="skill_graph_skill_semantic__test",
    )
    return graph


def _patch_loader(monkeypatch: pytest.MonkeyPatch, server: Any, fn: Callable[..., Any]) -> None:
    monkeypatch.setattr(server, "load_graph_from_services", fn)


class TestVizServer:
    def test_index_fails_fast_with_503_when_no_active_build(self, server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(**_: Any) -> Any:
            raise GraphReadError("no active build pointer; publish a build first")

        _patch_loader(monkeypatch, server, boom)
        client = server.app.test_client()

        resp = client.get("/")
        assert resp.status_code == 503
        assert "不可用" in resp.get_data(as_text=True)
        # The message is echoed in the page so ops sees the actual cause.
        assert "publish a build first" in resp.get_data(as_text=True)

    def test_index_and_stats_serve_the_active_build(self, server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        graph = _built_graph()

        def fake_load(**_: Any) -> Any:
            return graph

        _patch_loader(monkeypatch, server, fake_load)
        client = server.app.test_client()

        page = client.get("/")
        assert page.status_code == 200
        text = page.get_data(as_text=True)
        assert "Skill Graph" in text
        # The rendered page carries the graph payload for the browser JS.
        assert '"nodes"' in text

        stats = client.get("/api/stats")
        assert stats.status_code == 200
        body = stats.get_json()
        assert body["build_id"] == graph.manifest.build_id
        assert body["nodes"] == graph.stats()["nodes"]

    def test_healthz_and_favicon_do_not_touch_backends(self, server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(**_: Any) -> Any:
            raise AssertionError("health endpoints must not load the graph")

        _patch_loader(monkeypatch, server, boom)
        client = server.app.test_client()

        assert client.get("/healthz").status_code == 200
        assert client.get("/favicon.ico").status_code == 204

    def test_stats_reports_503_on_graph_error(self, server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(**_: Any) -> Any:
            raise GraphReadError("graphspace unreachable")

        _patch_loader(monkeypatch, server, boom)
        client = server.app.test_client()

        resp = client.get("/api/stats")
        assert resp.status_code == 503
        assert resp.get_json() == {"error": "graphspace unreachable"}

    def test_env_kwargs_read_service_config(self, server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILL_GRAPH_HG_URL", "http://hg:8080")
        monkeypatch.setenv("SKILL_GRAPH_HG_GRAPHSPACE", "gs")
        monkeypatch.setenv("SKILL_GRAPH_HG_GRAPH", "g")
        monkeypatch.setenv("OCEANBASE_HOST", "ob")
        monkeypatch.setenv("OCEANBASE_PORT", "3306")
        monkeypatch.setenv("OCEANBASE_USER", "u")
        monkeypatch.setenv("OCEANBASE_PASSWORD", "p")
        monkeypatch.setenv("OCEANBASE_DATABASE", "db")
        assert server._hg_kwargs() == {
            "hg_url": "http://hg:8080",
            "hg_user": "admin",
            "hg_pwd": "admin",
            "hg_graphspace": "gs",
            "hg_graph": "g",
        }
        assert server._ob_kwargs() == {
            "ob_host": "ob",
            "ob_port": 3306,
            "ob_user": "u",
            "ob_password": "p",
            "ob_database": "db",
        }


class TestPointerOnDeployedService:
    """Smoke check that /api/stats on the deployed STG service is coherent.

    Skipped when the STG host is unreachable (offline dev machines); the real
    E2E run executes the full browser flow against this deployment.
    """

    def test_stg_stats_has_active_build(self) -> None:
        import urllib.request

        url = "http://10.129.33.121:8766/api/stats"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                body = json.loads(resp.read().decode())
        except OSError:
            pytest.skip("STG visualization service unreachable from this machine")
        assert resp.status == 200
        assert body["build_id"]
        assert body["nodes"] > 0
