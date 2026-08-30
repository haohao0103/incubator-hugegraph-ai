"""API-layer contract tests for the NL2SQL router (7 endpoints + healthz +
error model + metadata validation). Production-readiness P0.

Run with a patched pipeline (small in-memory schema, no embedder) so the
tests are fast and hermetic.
"""
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from hugegraph_llm.api import nl2sql_api
from hugegraph_llm.api.nl2sql_api import nl2sql_http_api
from hugegraph_llm.nl2sql.pipeline import NL2SQLPipeline

pytestmark = pytest.mark.contract

SMALL_META = {
    "tables": [
        {"name": "orders", "database": "dw", "comment": "订单表", "is_fact": True},
        {"name": "payments", "database": "dw", "comment": "支付表", "is_fact": True},
    ],
    "columns": [
        {"name": "order_id", "table": "dw.orders", "data_type": "bigint",
         "comment": "订单编号"},
        {"name": "gmv", "table": "dw.orders", "data_type": "decimal",
         "comment": "成交总额"},
        {"name": "pay_amount", "table": "dw.payments", "data_type": "decimal",
         "comment": "支付金额"},
        {"name": "order_id", "table": "dw.payments", "data_type": "bigint",
         "comment": "订单编号"},
    ],
    "foreign_keys": [["dw.orders.order_id", "dw.payments.order_id"]],
    "terms": [{"name": "支付总额", "comment": "支付金额汇总"}],
    "term_bindings": [["支付总额", "dw.payments.pay_amount"]],
}


def _make_client(monkeypatch):
    schema = nl2sql_api.build_schema(SMALL_META)
    pipe = NL2SQLPipeline(schema)
    monkeypatch.setattr(nl2sql_api, "get_pipeline", lambda: pipe)
    router = APIRouter()
    nl2sql_http_api(router)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_healthz_ok(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.get("/nl2sql/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "dependencies" in body and "engine" in body["dependencies"]
    assert body["dependencies"]["engine"]["name"] == "local"


def test_link_returns_items(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/link", json={"question": "支付总额是多少"})
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    names = {i["name"] for i in body["items"]}
    assert "pay_amount" in names  # term binding surfaces the metric column


def test_link_out_of_kb_flag(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/link", json={"question": "今天天气如何",
                                     "min_score": 0.99})
    assert r.status_code == 200
    body = r.json()
    assert body.get("out_of_kb") is True
    assert "message" in body


def test_link_bad_request(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/link", json={})
    assert r.status_code == 422  # FastAPI validation


def test_join_path(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/join_path",
               json={"source": "dw.orders", "target": "dw.payments"})
    assert r.status_code == 200
    path = r.json()["path"]
    assert path is not None and path["all_proven"] is True


def test_communities(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/communities", json={})
    assert r.status_code == 200
    assert isinstance(r.json()["domains"], dict)


def test_schema_context_include_global(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/schema_context",
               json={"question": "支付总额", "include_global": True})
    assert r.status_code == 200
    ctx = r.json()["schema_context"]
    assert "payments" in ctx


def test_schema_context_out_of_kb(monkeypatch):
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/schema_context", json={"question": "今天天气如何"})
    assert r.status_code == 200
    assert r.json().get("out_of_kb") is True


def test_run_llm_unavailable_error_model(monkeypatch):
    c = _make_client(monkeypatch)

    def _boom(_prompt):
        from hugegraph_llm.api.nl2sql_api import NL2SQL_ERR_LLM, _err
        raise _err(NL2SQL_ERR_LLM, "LLM not configured")

    monkeypatch.setattr(nl2sql_api, "_llm_callable", _boom)
    r = c.post("/nl2sql/run", json={"question": "支付总额是多少"})
    assert r.status_code == 503
    # FastAPI wraps HTTPException detail under the standard "detail" key
    assert r.json()["detail"]["error"]["code"] == "NL2SQL_LLM_UNAVAILABLE"


def test_validate_metadata_flags_issues(monkeypatch):
    c = _make_client(monkeypatch)
    bad = {
        "tables": [{"name": "orders", "comment": ""}],
        "columns": [
            {"name": "id", "table": "dw.orders"},
            {"name": "id", "table": "dw.orders"},  # duplicate
            {"name": "x", "table": "dw.nosuch"},   # orphan
        ],
        "terms": [{"name": "t", "comment": "a"}, {"name": "t", "comment": "b"}],
    }
    r = c.post("/nl2sql/validate", json={"metadata": bad})
    assert r.status_code == 200
    body = r.json()["metadata"]
    assert body["valid"] is False
    codes = {i["code"] for i in body["issues"]}
    assert {"DUP_COLUMN", "COL_ORPHAN", "TERM_CONFLICT",
            "COL_NO_COMMENT"} <= codes


def test_load_hugegraph_path(monkeypatch):
    from hugegraph_llm.nl2sql.schema_graph.model import SchemaGraph

    def _fake_loader(url, graph, infer_foreign_keys=True, **_):
        return nl2sql_api.build_schema(SMALL_META)

    monkeypatch.setattr(nl2sql_api, "build_schema_from_hugegraph", _fake_loader)
    c = _make_client(monkeypatch)
    r = c.post("/nl2sql/load_hugegraph",
               json={"url": "http://127.0.0.1:8081", "graph": "kg_rag"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "loaded"
    assert body["tables"] == 2 and body["prebuilt"] is True
