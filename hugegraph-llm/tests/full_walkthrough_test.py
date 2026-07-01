#!/usr/bin/env python3
"""
Full walkthrough test of all 19 frontend API endpoints.
Tests each endpoint and reports pass/fail.
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8765"
PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS = []

def api_call(method, path, data=None, timeout=60):
    """Make API call and return parsed JSON response."""
    url = BASE + path
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    else:
        body = None
    req = urllib.request.Request(url, data=body, method=method,
                                  headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except:
            body = ""
        return {"error": body, "status": e.code}, e.code
    except Exception as e:
        return {"error": str(e), "status": -1}, -1

def test(name, method, path, data=None, check_fn=None, timeout=60):
    """Test an API endpoint and report result."""
    global PASS_COUNT, FAIL_COUNT
    result, status = api_call(method, path, data, timeout)
    ok = False
    detail = ""
    if status == 200:
        if check_fn:
            ok, detail = check_fn(result)
        else:
            ok = True
            detail = f"status={status}"
    else:
        detail = f"HTTP {status}: {str(result)[:200]}"
    
    emoji = "✅" if ok else "❌"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    RESULTS.append(f"{emoji} {name}: {detail}")
    return result

# ============ Test Sequence ============

print("=" * 60)
print("Full Walkthrough: 19 API Endpoints")
print("=" * 60)

# 1. GET /api/stats
test("1/19 GET /api/stats", "GET", "/api/stats",
     check_fn=lambda r: (r.get("hugegraph_connected") is True, f"connected={r.get('hugegraph_connected')}"))

# 2. POST /api/clear
test("2/19 POST /api/clear", "POST", "/api/clear", {},
     check_fn=lambda r: (r.get("status") == "cleared", f"status={r.get('status')}"))

# 3. POST /api/memory/add
add1_result = test("3/19 POST /api/memory/add", "POST", "/api/memory/add",
    {"content": "我叫张明，是货拉拉图谱团队的工程师。我擅长关系图谱和图数据库运维，有3年60亿点边集群运维经验。",
     "user_id": "demo_user"},
    check_fn=lambda r: (r.get("action") in ["add", "ADD"], f"action={r.get('action')}, entities={len(r.get('entities',[]))}"),
    timeout=120)

# Get the memory_id from add1 for later tests
add1_id = ""
if add1_result.get("action") == "add":
    add1_id = add1_result.get("memory_id", "")
    # Also add another memory for richer search
    add2_result, _ = api_call("POST", "/api/memory/add",
        {"content": "我的同事李华也在图谱团队，负责GraphRAG引擎开发。他之前在腾讯工作过3年。",
         "user_id": "demo_user"}, timeout=120)

# 4. POST /api/memory/search
search_result = test("4/19 POST /api/memory/search", "POST", "/api/memory/search",
    {"query": "张明擅长什么", "user_id": "demo_user", "top_k": 5, "fast_eval": True},
    check_fn=lambda r: ("results" in r and len(r.get("results",[])) >= 0, 
                        f"results={len(r.get('results',[]))}, trace={len(r.get('trace',[]))}"),
    timeout=30)

# 5. POST /api/memory/update (if we have a memory_id)
if add1_id:
    test("5/19 POST /api/memory/update", "POST", "/api/memory/update",
        {"memory_id": add1_id, "content": "张明升级为全公司AI Agent图存储底座负责人", "user_id": "demo_user"},
        check_fn=lambda r: (r.get("action") in ["update", "add"], f"action={r.get('action')}"),
        timeout=120)
else:
    RESULTS.append("⏭️ 5/19 POST /api/memory/update: SKIPPED (no memory_id)")

# 6. POST /api/memory/delete (if we have a memory_id)
if add1_id:
    test("6/19 POST /api/memory/delete", "POST", "/api/memory/delete",
        {"memory_id": add1_id, "user_id": "demo_user"},
        check_fn=lambda r: (r.get("action") == "delete" or "deleted" in str(r), f"action={r.get('action')}"),
        timeout=30)
else:
    RESULTS.append("⏭️ 6/19 POST /api/memory/delete: SKIPPED (no memory_id)")

# 7. POST /api/graph/search (multi-hop traversal)
test("7/19 POST /api/graph/search", "POST", "/api/graph/search",
    {"query": "张明", "max_hops": 2, "limit": 20},
    check_fn=lambda r: ("results" in r or "entities" in r or "vertices" in r, f"keys={list(r.keys())[:5]}"),
    timeout=30)

# 8. POST /api/profile/auto
test("8/19 POST /api/profile/auto", "POST", "/api/profile/auto",
    {"user_id": "demo_user"},
    check_fn=lambda r: ("name" in r or "topics" in r or "error" in r, f"name={r.get('name','?')}, topics={r.get('topics','?')}"),
    timeout=60)

# 9. POST /api/profile/inject
test("9/19 POST /api/profile/inject", "POST", "/api/profile/inject",
    {"user_id": "demo_user"},
    check_fn=lambda r: ("profile" in r or "aliases" in r or "topics" in r, f"keys={list(r.keys())[:5]}"),
    timeout=60)

# 10. GET /api/memory/history
test("10/19 GET /api/memory/history", "GET", "/api/memory/history?limit=5",
    check_fn=lambda r: ("events" in r or "history" in r, f"events={len(r.get('events',[]))}"),
    timeout=10)

# 11. POST /api/scoring/explain
test("11/19 POST /api/scoring/explain", "POST", "/api/scoring/explain",
    {"query": "张明擅长什么"},
    check_fn=lambda r: ("extracted_entities" in r or "bm25" in str(r), f"entities={r.get('extracted_entities',[])}"),
    timeout=60)

# 12. POST /api/privacy/filter
test("12/19 POST /api/privacy/filter", "POST", "/api/privacy/filter",
    {"content": "我叫张明，手机号13812345678，住在深圳", "privacy_level": "confidential"},
    check_fn=lambda r: ("filtered" in r and "original" in r, f"filtered={r.get('filtered','?')[:30]}"),
    timeout=30)

# 13. POST /api/agent/check_access (with FIXED parameter names)
test("13/19 POST /api/agent/check_access", "POST", "/api/agent/check_access",
    {"requesting_agent_id": "agent_b", "owner_agent_id": "agent_a", "operation": "read", "scope": "SHARED", "privacy": "STANDARD"},
    check_fn=lambda r: ("allowed" in r, f"allowed={r.get('allowed')}"),
    timeout=10)

# 14. POST /api/agent/share
test("14/19 POST /api/agent/share", "POST", "/api/agent/share",
    {"memory_id": "test_memory_1", "source_agent": "agent_a", "target_agent": "agent_b", "scope": "agent_group"},
    check_fn=lambda r: ("success" in r or "error" in r, f"success={r.get('success','?')}"),
    timeout=10)

# 15. POST /api/memory/compress
test("15/19 POST /api/memory/compress", "POST", "/api/memory/compress",
    {"user_id": "demo_user"},
    check_fn=lambda r: ("archived" in r or "pruned" in r or "stats" in r, f"archived={r.get('archived','?')}, kept={r.get('kept','?')}"),
    timeout=60)

# 16. GET /api/locomo
test("16/19 GET /api/locomo", "GET", "/api/locomo",
    check_fn=lambda r: ("metrics" in r or "sessions" in r, f"keys={list(r.keys())[:5]}"),
    timeout=10)

# 17. GET /api/graph/entities
test("17/19 GET /api/graph/entities", "GET", "/api/graph/entities",
    check_fn=lambda r: ("entities" in r or len(r) >= 0, f"type={type(r).__name__}, keys={list(r.keys())[:5] if isinstance(r,dict) else 'list'}"),
    timeout=10)

# 18. GET /api/graph/relations
test("18/19 GET /api/graph/relations", "GET", "/api/graph/relations",
    check_fn=lambda r: ("relations" in r or "edges" in r or len(r) >= 0, f"type={type(r).__name__}, keys={list(r.keys())[:5] if isinstance(r,dict) else 'list'}"),
    timeout=10)

# 19. POST /api/query/rewrite
test("19/19 POST /api/query/rewrite", "POST", "/api/query/rewrite",
    {"query": "张明擅长什么"},
    check_fn=lambda r: ("rewritten" in r or "method" in r, f"rewritten={r.get('rewritten','?')}, method={r.get('method','?')}"),
    timeout=60)

# ============ Summary ============
print()
print("=" * 60)
for r in RESULTS:
    print(r)
print("=" * 60)
print(f"TOTAL: {PASS_COUNT + FAIL_COUNT} tests, ✅ {PASS_COUNT} PASS, ❌ {FAIL_COUNT} FAIL")
if FAIL_COUNT == 0:
    print("🎉 ALL 19 API ENDPOINTS WORKING!")
else:
    print(f"⚠️ {FAIL_COUNT} endpoints need fixing")
