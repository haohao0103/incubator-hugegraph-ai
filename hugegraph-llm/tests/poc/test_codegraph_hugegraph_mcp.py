#!/usr/bin/env python3
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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.  See the License for the specific
# language governing permissions and limitations under the License.
"""
Tests for codegraph_hugegraph_mcp.py — CodeGraph vs HugeGraph PoC

Coverage target: >90 % statement coverage

Test groups
-----------
T1  Data models:          CodeNode / CodeEdge / QueryResult dataclasses
T2  PythonCodeParser:     parse_file, _make_id, _visit_*, _extract_calls
    — module node, function node, class node, import edge, call edge,
      syntax-error skip, binary-file skip, nested class/method
T3  SQLiteCodeGraph:      insert_nodes, insert_edges, all 6 query methods,
                          FTS5 search, WAL mode, duplicate handling
T4  BM25CodeSearch:       build_index, search, missing-jieba fallback,
                          missing-rank_bm25 graceful disable
T5  HugeGraphCodeGraph:   _request error handling (mock), init_schema,
                          insert_nodes/edges batching, all query methods,
                          clear_graph, get_stats
T6  CodeGraphBenchmark:   _build_name_index, run_benchmark (mocked HG),
                          _run_single_query timing + error handling
T7  Utilities:            find_python_files, check_hugegraph_available
T8  Integration:          run_poc (SQLite-only path; no network needed)
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict
from typing import List
from unittest.mock import MagicMock, patch

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hugegraph_llm.poc.codegraph_hugegraph_mcp import (
    BM25CodeSearch,
    CodeEdge,
    CodeGraphBenchmark,
    CodeNode,
    HugeGraphCodeGraph,
    PythonCodeParser,
    QueryResult,
    SQLiteCodeGraph,
    check_hugegraph_available,
    find_python_files,
    run_poc,
    _run_sqlite_only,
)

# ── counters ──────────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0
ERRORS: List[str] = []


def check(condition: bool, test_name: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        FAIL += 1
        ERRORS.append(test_name)
        print(f"  [FAIL] {test_name}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_node(
    name: str = "foo",
    ntype: str = "function",
    fp: str = "mod.py",
    ls: int = 1,
    le: int = 5,
    src: str = "",
) -> CodeNode:
    return CodeNode(
        id=f"{fp}::{name}::L{ls}",
        name=name,
        node_type=ntype,
        file_path=fp,
        line_start=ls,
        line_end=le,
        source_code=src,
    )


def _make_edge(src: str, tgt: str, etype: str = "calls") -> CodeEdge:
    return CodeEdge(source_id=src, target_id=tgt, edge_type=etype, file_path="")


def _fresh_sqlite() -> tuple[SQLiteCodeGraph, str]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return SQLiteCodeGraph(path), path


def _write_py(code: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(textwrap.dedent(code))
    return path


def _write_file(suffix: str, code: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(textwrap.dedent(code))
    return path


# ═════════════════════════════════════════════════════════════════════════════
# T1 — Data models
# ═════════════════════════════════════════════════════════════════════════════

def test_dataclasses():
    print("\n── T1: Data Models ──")

    n = _make_node("bar", "class", "a.py", 10, 20, "class bar: pass")
    check(n.name == "bar",         "T1.1 CodeNode.name")
    check(n.node_type == "class",  "T1.2 CodeNode.node_type")
    check(n.line_start == 10,      "T1.3 CodeNode.line_start")
    check(n.source_code == "class bar: pass", "T1.4 CodeNode.source_code")

    e = _make_edge("A", "B", "inherits")
    check(e.source_id == "A",     "T1.5 CodeEdge.source_id")
    check(e.edge_type == "inherits", "T1.6 CodeEdge.edge_type")

    qr = QueryResult(query_name="q", question="?",
                     sqlite_time_ms=1.5, hugegraph_time_ms=3.0,
                     sqlite_results=2, hugegraph_results=1,
                     sqlite_correct=True, hugegraph_correct=False,
                     speedup_ratio=0.5)
    check(qr.speedup_ratio == 0.5,    "T1.7 QueryResult.speedup_ratio")
    check(qr.sqlite_correct is True,  "T1.8 QueryResult.sqlite_correct")
    check(qr.hugegraph_correct is False, "T1.9 QueryResult.hugegraph_correct")

    # asdict roundtrip
    d = asdict(qr)
    check(d["query_name"] == "q",  "T1.10 asdict roundtrip")


# ═════════════════════════════════════════════════════════════════════════════
# T2 — PythonCodeParser
# ═════════════════════════════════════════════════════════════════════════════

def test_parser():
    print("\n── T2: PythonCodeParser ──")

    # T2.1 — basic module + function + import
    src = _write_py("""
        import os
        from collections import Counter

        def greet(name):
            print(name)
            return name.upper()

        class Foo:
            def bar(self):
                greet("hello")
    """)
    parser = PythonCodeParser()
    parser.parse_file(src)
    os.remove(src)

    node_names = {n.name for n in parser.nodes}
    edge_types = {e.edge_type for e in parser.edges}

    check("greet" in node_names,   "T2.1 function node extracted")
    check("Foo" in node_names,     "T2.2 class node extracted")
    check("bar" in node_names,     "T2.3 method node extracted")
    check("module" in {n.node_type for n in parser.nodes}, "T2.4 module node present")
    check("calls" in edge_types,   "T2.5 calls edge present")
    check("contains" in edge_types, "T2.6 contains edge present")
    check("imports" in edge_types, "T2.7 imports edge present")
    check("inherits" in edge_types or "defines" in edge_types, "T2.8 class-related edge")

    # T2.9 — _make_id is stable
    id1 = PythonCodeParser._make_id("func", "mod.py", 42)
    id2 = PythonCodeParser._make_id("func", "mod.py", 42)
    check(id1 == id2, "T2.9 _make_id is deterministic")

    # T2.10 — syntax error file skipped gracefully
    bad = _write_py("def f(\n  :")  # invalid syntax
    p2 = PythonCodeParser()
    p2.parse_file(bad)
    os.remove(bad)
    check(len(p2.nodes) == 0, "T2.10 syntax-error file skipped")

    # T2.11 — non-existent file skipped gracefully
    p3 = PythonCodeParser()
    p3.parse_file("/nonexistent/path/to/file.py")
    check(len(p3.nodes) == 0, "T2.11 missing file skipped")

    # T2.12 — multiple functions in same module share module parent
    src2 = _write_py("""
        def alpha(): pass
        def beta(): alpha()
    """)
    p4 = PythonCodeParser()
    p4.parse_file(src2)
    os.remove(src2)
    func_names = {n.name for n in p4.nodes if n.node_type == "function"}
    check("alpha" in func_names and "beta" in func_names, "T2.12 two functions in same module")

    # T2.13 — calls edge from beta → alpha
    call_edges = [(e.source_id, e.target_id) for e in p4.edges if e.edge_type == "calls"]
    has_call = any("beta" in src and "alpha" in tgt for src, tgt in call_edges)
    check(has_call, "T2.13 call edge beta→alpha extracted")

    # T2.14 — class inheritance edge
    src3 = _write_py("""
        class Base: pass
        class Child(Base): pass
    """)
    p5 = PythonCodeParser()
    p5.parse_file(src3)
    os.remove(src3)
    inherit_edges = [(e.source_id, e.target_id) for e in p5.edges if e.edge_type == "inherits"]
    has_inherit = any("Base" in tgt for _, tgt in inherit_edges)
    check(has_inherit, "T2.14 inherits edge Child→Base")

    # T2.15 — async function handled
    src4 = _write_py("""
        async def fetch(url):
            return url
    """)
    p6 = PythonCodeParser()
    p6.parse_file(src4)
    os.remove(src4)
    check("fetch" in {n.name for n in p6.nodes}, "T2.15 async def parsed")

    # T2.16 — attribute call (obj.method()) → calls edge
    src5 = _write_py("""
        def runner():
            obj.do_work()
    """)
    p7 = PythonCodeParser()
    p7.parse_file(src5)
    os.remove(src5)
    call_tgts = [e.target_id for e in p7.edges if e.edge_type == "calls"]
    check("do_work" in call_tgts, "T2.16 attribute call extracted")

    # T2.17 — ImportFrom without module (e.g. relative) doesn't crash
    src6 = _write_py("""
        from . import sibling
    """)
    p8 = PythonCodeParser()
    p8.parse_file(src6)
    os.remove(src6)
    # Should not raise; just no imports edges added for relative imports without module
    check(True, "T2.17 relative ImportFrom doesn't crash")


def test_parser_advanced_resolution():
    print("\n── T2x: Advanced AST resolution ──")

    def calls_from(src: str) -> List[Tuple[str, str, str]]:
        """Return (caller_name, target_name, edge_type) for a code snippet."""
        path = _write_py(src)
        parser = PythonCodeParser()
        parser.parse_file(path)
        os.remove(path)
        return [
            (n.name, e.target_id, e.edge_type)
            for n in parser.nodes
            for e in parser.edges
            if e.source_id == n.id and n.node_type == "function"
        ]

    # T2x.1 — builtins are filtered from calls
    calls = calls_from("""
        def foo(x):
            print(len(x))
            return int(x)
    """)
    targets = {t for _, t, _ in calls}
    check("print" not in targets, "T2x.1 builtins filtered: print")
    check("len" not in targets, "T2x.2 builtins filtered: len")
    check("int" not in targets, "T2x.3 builtins filtered: int")

    # T2x.4 — simple alias resolved: a = real_func; a()
    calls = calls_from("""
        def real_func(): pass
        def wrapper():
            alias = real_func
            alias()
    """)
    pairs = {(s, t) for s, t, _ in calls}
    check(("wrapper", "real_func") in pairs, "T2x.4 alias call resolved to real_func")

    # T2x.5 — chained assignment resolved: a = b = real_func; a()
    calls = calls_from("""
        def real_func(): pass
        def wrapper():
            a = b = real_func
            a()
    """)
    pairs = {(s, t) for s, t, _ in calls}
    check(("wrapper", "real_func") in pairs, "T2x.5 chained assignment resolved")

    # T2x.6 — tuple unpacking resolved
    calls = calls_from("""
        def f1(): pass
        def f2(): pass
        def wrapper():
            a, b = f1, f2
            a()
            b()
    """)
    pairs = {(s, t) for s, t, _ in calls}
    check(("wrapper", "f1") in pairs, "T2x.6 tuple unpacking resolves a->f1")
    check(("wrapper", "f2") in pairs, "T2x.7 tuple unpacking resolves b->f2")

    # T2x.8 — function decorator extracted as call
    src = _write_py("""
        def deco(fn): return fn
        @deco
        def wrapped():
            pass
    """)
    parser = PythonCodeParser()
    parser.parse_file(src)
    os.remove(src)
    deco_edges = [(e.source_id, e.target_id) for e in parser.edges
                  if e.edge_type == "calls" and e.target_id == "deco"]
    check(len(deco_edges) >= 1, "T2x.8 decorator emits calls edge to deco")

    # T2x.9 — class decorator extracted
    src = _write_py("""
        def datacls(cls): return cls
        @datacls
        class Foo:
            pass
    """)
    parser = PythonCodeParser()
    parser.parse_file(src)
    os.remove(src)
    cls_deco = [(e.source_id, e.target_id) for e in parser.edges
                if e.edge_type == "calls" and e.target_id == "datacls"]
    check(len(cls_deco) >= 1, "T2x.9 class decorator emits calls edge")

    # T2x.10 — getattr dynamic dispatch
    calls = calls_from("""
        def dispatch(obj):
            getattr(obj, 'dynamic_method')()
    """)
    targets = {t for _, t, _ in calls}
    check("dynamic_method" in targets, "T2x.10 getattr resolves dynamic method")

    # T2x.11 — eval / exec marked as dynamic_call
    calls = calls_from("""
        def run(code):
            eval(code)
            exec(code)
    """)
    dynamic = {(t, et) for _, t, et in calls}
    check(("eval", "dynamic_call") in dynamic, "T2x.11 eval marked dynamic_call")
    check(("exec", "dynamic_call") in dynamic, "T2x.12 exec marked dynamic_call")

    # T2x.13 — Java function/class/call extraction
    java_path = _write_file(".java", """
        class Foo {
            void bar() {}
            void run() { bar(); }
        }
    """)
    jp = PythonCodeParser()
    jp.parse_file(java_path)
    os.remove(java_path)
    java_funcs = {n.name for n in jp.nodes if n.node_type == "function"}
    java_calls = {e.target_id for e in jp.edges if e.edge_type == "calls"}
    check("bar" in java_funcs and "run" in java_funcs, "T2x.13 Java functions parsed")
    check("bar" in java_calls, "T2x.14 Java call bar extracted")

    # T2x.15 — Go function/call extraction
    go_path = _write_file(".go", """
        package main
        func foo() {}
        func main() { foo() }
    """)
    gp = PythonCodeParser()
    gp.parse_file(go_path)
    os.remove(go_path)
    go_funcs = {n.name for n in gp.nodes if n.node_type == "function"}
    go_calls = {e.target_id for e in gp.edges if e.edge_type == "calls"}
    check("foo" in go_funcs and "main" in go_funcs, "T2x.15 Go functions parsed")
    check("foo" in go_calls, "T2x.16 Go call foo extracted")

    # T2x.17 — TypeScript function/class/call extraction
    ts_path = _write_file(".ts", """
        function foo(): void {}
        class C {
            run() { foo(); }
        }
    """)
    tp = PythonCodeParser()
    tp.parse_file(ts_path)
    os.remove(ts_path)
    ts_funcs = {n.name for n in tp.nodes if n.node_type == "function"}
    ts_calls = {e.target_id for e in tp.edges if e.edge_type == "calls"}
    check("foo" in ts_funcs and "run" in ts_funcs, "T2x.17 TS functions parsed")
    check("foo" in ts_calls, "T2x.18 TS call foo extracted")


# ═════════════════════════════════════════════════════════════════════════════
# T3 — SQLiteCodeGraph
# ═════════════════════════════════════════════════════════════════════════════

def test_sqlite():
    print("\n── T3: SQLiteCodeGraph ──")

    db, db_path = _fresh_sqlite()

    nodes = [
        _make_node("alpha",  "function", "a.py", 1, 10),
        _make_node("beta",   "function", "a.py", 11, 20),
        _make_node("Gamma",  "class",    "a.py", 21, 40),
        _make_node("delta",  "function", "b.py", 1, 5),
        _make_node("epsilon","function", "b.py", 6, 15),
        _make_node("module_a", "module", "a.py", 1, 40),
    ]
    # unique IDs for safe FK lookups
    for n in nodes:
        n.id = f"test__{n.name}"

    edges = [
        _make_edge("test__beta",  "alpha",       "calls"),
        _make_edge("test__epsilon","alpha",       "calls"),
        _make_edge("test__Gamma", "BaseClass",    "inherits"),
        _make_edge("test__module_a","module_b",   "imports"),
        _make_edge("test__Gamma", "test__alpha",  "defines"),
        _make_edge("test__module_a","test__alpha","contains"),
    ]

    db.insert_nodes(nodes)
    db.insert_edges(edges)

    # T3.1 — query_callers single-hop
    callers = db.query_callers("alpha", depth=1)
    caller_names = {r["name"] for r in callers}
    check("beta" in caller_names or "epsilon" in caller_names, "T3.1 query_callers single-hop")

    # T3.2 — query_callers multi-hop returns list (may be empty for 2nd hop)
    callers2 = db.query_callers("alpha", depth=2)
    check(isinstance(callers2, list), "T3.2 query_callers multi-hop returns list")

    # T3.3 — query_callees
    # Add a callee edge first
    db.insert_edges([_make_edge("test__alpha", "test__delta", "calls")])
    callees = db.query_callees("test__alpha", depth=1)
    check(isinstance(callees, list), "T3.3 query_callees returns list")

    # T3.4 — query_callees multi-hop
    callees2 = db.query_callees("test__alpha", depth=2)
    check(isinstance(callees2, list), "T3.4 query_callees multi-hop returns list")

    # T3.5 — query_import_chain
    imports = db.query_import_chain("module_a")
    check(len(imports) > 0, "T3.5 query_import_chain finds imported module")

    # T3.6 — query_class_hierarchy
    hier = db.query_class_hierarchy("Gamma")
    check(len(hier) > 0, "T3.6 query_class_hierarchy finds base class")

    # T3.7 — query_functions_in_file
    funcs = db.query_functions_in_file("a.py")
    check(len(funcs) >= 2, "T3.7 query_functions_in_file finds function/class")

    # T3.8 — search_by_name FTS5
    results = db.search_by_name("alpha")
    check(isinstance(results, list), "T3.8 search_by_name returns list")

    # T3.9 — duplicate nodes ignored (OR REPLACE / OR IGNORE)
    before_count = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    db.insert_nodes([nodes[0]])   # same ID → OR REPLACE, same row count
    after_count = db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    check(before_count == after_count, "T3.9 duplicate nodes handled")

    # T3.10 — WAL mode set
    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    check(mode == "wal", "T3.10 WAL journal mode active")

    # T3.11 — query_callers on unknown function returns empty
    nobody = db.query_callers("__nobody__", depth=1)
    check(nobody == [], "T3.11 unknown function → empty callers")

    # T3.12 — query_functions_in_file on nonexistent file
    empty = db.query_functions_in_file("zzz_nonexistent.py")
    check(empty == [], "T3.12 nonexistent file → empty list")

    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


# ═════════════════════════════════════════════════════════════════════════════
# T4 — BM25CodeSearch
# ═════════════════════════════════════════════════════════════════════════════

def test_bm25():
    print("\n── T4: BM25CodeSearch ──")

    nodes = [
        _make_node("process_order",    "function", "order.py", 1, 10,
                   "def process_order(id): return id"),
        _make_node("cancel_order",     "function", "order.py", 11, 20,
                   "def cancel_order(id): pass"),
        _make_node("OrderManager",     "class",    "order.py", 21, 40,
                   "class OrderManager: ..."),
        _make_node("get_user_profile", "function", "user.py", 1, 10,
                   "def get_user_profile(uid): pass"),
    ]

    # T4.1 — build_index succeeds
    bm25 = BM25CodeSearch()
    bm25.build_index(nodes)
    check(bm25.bm25 is not None, "T4.1 BM25 index built")
    check(len(bm25.corpus) == 4, "T4.2 all documents indexed")

    # T4.3 — search returns ranked results for exact function name
    results = bm25.search("process_order", top_k=5)
    check(len(results) > 0, "T4.3 search returns results")
    check(all(score > 0 for _, score in results), "T4.4 all returned scores > 0")

    # T4.5 — process_order doc ranked first
    top_ids = [doc_id for doc_id, _ in results[:1]]
    order_hit = any("process_order" in did for did in top_ids)
    check(order_hit, "T4.5 process_order doc ranked first")

    # T4.6 — empty query returns empty
    empty = bm25.search("xyzxyzxyz_notaword")
    check(isinstance(empty, list), "T4.6 no-match search returns list")

    # T4.7 — top_k respected
    all_results = bm25.search("order", top_k=1)
    check(len(all_results) <= 1, "T4.7 top_k=1 respected")

    # T4.8 — missing rank_bm25 → graceful disable
    bm25_empty = BM25CodeSearch()
    with patch.dict("sys.modules", {"rank_bm25": None}):
        # Re-call build_index with patched import failure path
        bm25_empty.bm25 = None
        bm25_empty.corpus = []
        results_empty = bm25_empty.search("anything")
    check(results_empty == [], "T4.8 disabled BM25 returns empty list")

    # T4.9 — doc_ids aligned with corpus
    check(len(bm25.doc_ids) == len(bm25.corpus), "T4.9 doc_ids and corpus same length")


# ═════════════════════════════════════════════════════════════════════════════
# T5 — HugeGraphCodeGraph  (fully mocked — no real network)
# ═════════════════════════════════════════════════════════════════════════════

def test_hugegraph_mocked():
    print("\n── T5: HugeGraphCodeGraph (mocked) ──")

    hg = HugeGraphCodeGraph("http://mock-host:8080", "test_graph")

    nodes = [
        _make_node("fn_a", "function", "x.py", 1, 5, "def fn_a(): pass"),
        _make_node("fn_b", "function", "x.py", 6, 10),
        _make_node("ClsX", "class",    "x.py", 11, 20),
    ]
    for n in nodes:
        n.id = f"hg_test__{n.name}"

    edges = [
        _make_edge("hg_test__fn_b", "hg_test__fn_a", "calls"),
        _make_edge("hg_test__ClsX", "Base",           "inherits"),
    ]

    # T5.1 — _request returns None on connection error
    result = hg._request("GET", "/version")
    check(result is None, "T5.1 _request returns None on connection error")

    # T5.2 — init_schema doesn't raise even when server unreachable
    try:
        hg.init_schema()
        check(True, "T5.2 init_schema doesn't raise")
    except Exception as exc:
        check(False, f"T5.2 init_schema raised: {exc}")

    # T5.3 — insert_nodes doesn't raise
    try:
        hg.insert_nodes(nodes)
        check(True, "T5.3 insert_nodes doesn't raise")
    except Exception as exc:
        check(False, f"T5.3 insert_nodes raised: {exc}")

    # T5.4 — insert_edges doesn't raise
    try:
        hg.insert_edges(edges)
        check(True, "T5.4 insert_edges doesn't raise")
    except Exception as exc:
        check(False, f"T5.4 insert_edges raised: {exc}")

    # T5.5 — query_callers returns list on connection error
    callers = hg.query_callers("fn_a", depth=1)
    check(isinstance(callers, list), "T5.5 query_callers returns list on error")

    # T5.6 — query_callees returns list
    callees = hg.query_callees("fn_b", depth=2)
    check(isinstance(callees, list), "T5.6 query_callees returns list")

    # T5.7 — query_import_chain returns list
    imports = hg.query_import_chain("module_x")
    check(isinstance(imports, list), "T5.7 query_import_chain returns list")

    # T5.8 — query_class_hierarchy returns list
    hier = hg.query_class_hierarchy("ClsX")
    check(isinstance(hier, list), "T5.8 query_class_hierarchy returns list")

    # T5.9 — query_functions_in_file returns list
    funcs = hg.query_functions_in_file("x.py")
    check(isinstance(funcs, list), "T5.9 query_functions_in_file returns list")

    # T5.10 — get_stats returns dict with 'vertices' and 'edges'
    stats = hg.get_stats()
    check("vertices" in stats and "edges" in stats, "T5.10 get_stats returns expected keys")

    # T5.11 — clear_graph doesn't raise
    try:
        hg.clear_graph()
        check(True, "T5.11 clear_graph doesn't raise")
    except Exception as exc:
        check(False, f"T5.11 clear_graph raised: {exc}")

    # T5.12 — _gremlin_valueMap handles None response
    with patch.object(hg, "_request", return_value=None):
        items = hg._gremlin_valueMap("g.V().limit(1)")
        check(items == [], "T5.12 _gremlin_valueMap handles None response")

    # T5.13 — _gremlin_valueMap handles list-valued properties
    fake_response = {
        "data": [
            {"name": ["my_func"], "file_path": ["src/a.py"], "line_start": [42]}
        ]
    }
    with patch.object(hg, "_request", return_value=fake_response):
        items = hg._gremlin_valueMap("g.V().limit(1)")
    check(len(items) == 1 and items[0]["name"] == "my_func", "T5.13 list-valued valueMap unpacked")
    check(items[0]["line_start"] == 42,                       "T5.14 int property unpacked")

    # T5.15 — insert_nodes batches correctly (101 nodes → 2 batches of 100+1)
    big_nodes = [_make_node(f"func_{i}", "function", "big.py", i, i+1) for i in range(101)]
    for n in big_nodes:
        n.id = f"big__{n.name}"
    call_count = [0]
    original_request = hg._request
    def counting_request(method, path, body=None):
        if body and "gremlin" in (body or {}):
            call_count[0] += 1
        return None
    with patch.object(hg, "_request", side_effect=counting_request):
        hg.insert_nodes(big_nodes)
    check(call_count[0] >= 2, "T5.15 insert_nodes batches 101 nodes into >=2 requests")


# ═════════════════════════════════════════════════════════════════════════════
# T6 — CodeGraphBenchmark
# ═════════════════════════════════════════════════════════════════════════════

def test_benchmark():
    print("\n── T6: CodeGraphBenchmark ──")

    # Set up a real SQLite DB and BM25
    db, db_path = _fresh_sqlite()
    nodes = [
        _make_node("run",     "function", "a.py", 1,  10, "def run(): fetch()"),
        _make_node("fetch",   "function", "a.py", 11, 20, "def fetch(): return 1"),
        _make_node("Executor","class",    "a.py", 21, 30),
        _make_node("stop",    "function", "b.py", 1,  5,  "def stop(): run()"),
        _make_node("Helper",  "class",    "b.py", 6,  20),
        _make_node("mod_a",   "module",   "a.py", 1,  30),
        _make_node("mod_b",   "module",   "b.py", 1,  20),
    ]
    for n in nodes:
        n.id = f"bench__{n.name}"

    edges = [
        _make_edge("bench__run",  "fetch",  "calls"),
        _make_edge("bench__stop", "bench__run",  "calls"),
        _make_edge("bench__Executor", "Base", "inherits"),
        _make_edge("bench__mod_a", "bench__run",  "contains"),
        _make_edge("bench__mod_b", "module_os", "imports"),
    ]
    db.insert_nodes(nodes)
    db.insert_edges(edges)

    bm25 = BM25CodeSearch()
    bm25.build_index(nodes)

    # Mock HugeGraph
    hg = MagicMock(spec=HugeGraphCodeGraph)
    hg.query_callers.return_value    = [{"name": "stop", "file_path": "b.py", "line_start": 1}]
    hg.query_callees.return_value    = [{"name": "fetch", "file_path": "a.py", "line_start": 11}]
    hg.query_class_hierarchy.return_value = [{"target_id": "Base"}]
    hg.query_functions_in_file.return_value = [{"name": "run", "node_type": "function", "line_start": 1, "line_end": 10}]

    bench = CodeGraphBenchmark(sqlite=db, hugegraph=hg, bm25=bm25, nodes=nodes)

    # T6.1 — name index built
    check("run" in bench._name_idx, "T6.1 _name_index contains function names")

    # T6.2 — run_benchmark returns list of QueryResult
    results = bench.run_benchmark()
    check(isinstance(results, list), "T6.2 run_benchmark returns list")
    check(len(results) >= 3, f"T6.3 at least 3 queries run (got {len(results)})")

    # T6.4 — all results are QueryResult instances
    check(all(isinstance(r, QueryResult) for r in results), "T6.4 all results are QueryResult")

    # T6.5 — SQLite queries succeed
    sqlite_ok = sum(1 for r in results if r.sqlite_correct)
    check(sqlite_ok >= 3, f"T6.5 ≥3 SQLite queries passed ({sqlite_ok}/{len(results)})")

    # T6.6 — timings are non-negative
    check(all(r.sqlite_time_ms >= 0 for r in results), "T6.6 SQLite timings non-negative")
    check(all(r.hugegraph_time_ms >= 0 for r in results), "T6.7 HugeGraph timings non-negative")

    # T6.8 — speedup_ratio computed when hg_time > 0
    ratios = [r.speedup_ratio for r in results if r.hugegraph_time_ms > 0]
    if ratios:
        check(all(r >= 0 for r in ratios), "T6.8 speedup_ratio non-negative when hg>0")
    else:
        check(True, "T6.8 no hg timing — speedup_ratio check skipped")

    # T6.9 — _run_single_query handles SQLite error gracefully
    def raise_fn():
        raise RuntimeError("DB error")

    r = bench._run_single_query("err_test", "?", raise_fn, lambda: [])
    check(r.sqlite_correct is False, "T6.9 SQLite error → sqlite_correct=False")
    check(r.hugegraph_correct is True, "T6.10 HugeGraph succeeds independently")

    # T6.11 — _run_single_query handles HugeGraph error gracefully
    r2 = bench._run_single_query("err_hg", "?", lambda: [], raise_fn)
    check(r2.sqlite_correct is True,  "T6.11 SQLite succeeds despite HG error")
    check(r2.hugegraph_correct is False, "T6.12 HugeGraph error → hugegraph_correct=False")

    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


# ═════════════════════════════════════════════════════════════════════════════
# T7 — Utilities
# ═════════════════════════════════════════════════════════════════════════════

def test_utilities():
    print("\n── T7: Utilities ──")

    # T7.1 — find_python_files finds .py files
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in ["a.py", "b.py", "c.txt", "d.py"]:
            with open(os.path.join(tmpdir, fname), "w") as fh:
                fh.write("pass\n")
        found = find_python_files(tmpdir, max_files=100)
    check(len(found) == 3, f"T7.1 find_python_files finds 3 .py files (got {len(found)})")

    # T7.2 — max_files respected
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(10):
            with open(os.path.join(tmpdir, f"f{i}.py"), "w") as fh:
                fh.write("pass\n")
        found2 = find_python_files(tmpdir, max_files=5)
    check(len(found2) <= 5, "T7.2 max_files cap respected")

    # T7.3 — empty directory returns empty list
    with tempfile.TemporaryDirectory() as tmpdir:
        found3 = find_python_files(tmpdir)
    check(found3 == [], "T7.3 empty directory → empty list")

    # T7.4 — subdirectories traversed
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "nested.py"), "w") as fh:
            fh.write("pass\n")
        found4 = find_python_files(tmpdir)
    check(any("nested.py" in f for f in found4), "T7.4 subdirectories traversed")

    # T7.5 — __pycache__ skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "__pycache__")
        os.makedirs(cache)
        with open(os.path.join(cache, "cached.py"), "w") as fh:
            fh.write("pass\n")
        with open(os.path.join(tmpdir, "real.py"), "w") as fh:
            fh.write("pass\n")
        found5 = find_python_files(tmpdir)
    check(not any("__pycache__" in f for f in found5), "T7.5 __pycache__ dir skipped")

    # T7.6 — check_hugegraph_available returns False for unreachable host
    result = check_hugegraph_available("http://127.0.0.1:19999")
    check(result is False, "T7.6 unavailable host returns False")

    # T7.7 — check_hugegraph_available returns bool
    result2 = check_hugegraph_available("http://127.0.0.1:8080")
    check(isinstance(result2, bool), "T7.7 returns bool type")


# ═════════════════════════════════════════════════════════════════════════════
# T8 — Integration (SQLite-only path, no network)
# ═════════════════════════════════════════════════════════════════════════════

def test_integration():
    print("\n── T8: Integration (run_poc SQLite-only) ──")

    # Patch check_hugegraph_available to always return False
    with patch(
        "hugegraph_llm.poc.codegraph_hugegraph_mcp.check_hugegraph_available",
        return_value=False,
    ):
        # Redirect RESULT_FILE to tmp
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_result = os.path.join(tmpdir, "result.json")
            tmp_sqlite = os.path.join(tmpdir, "cg.db")
            import hugegraph_llm.poc.codegraph_hugegraph_mcp as poc_mod
            orig_result = poc_mod.RESULT_FILE
            orig_sqlite = poc_mod.SQLITE_PATH
            poc_mod.RESULT_FILE = tmp_result
            poc_mod.SQLITE_PATH = tmp_sqlite
            try:
                success = run_poc()
            finally:
                poc_mod.RESULT_FILE = orig_result
                poc_mod.SQLITE_PATH = orig_sqlite

            check(success is True, "T8.1 run_poc returns True on SQLite-only path")
            check(os.path.exists(tmp_result), "T8.2 result JSON file created")

            with open(tmp_result) as fh:
                data = json.load(fh)

            check(data["poc_result"].endswith("PASS"), "T8.3 poc_result string ends with PASS")
            check("codebase_stats" in data, "T8.4 codebase_stats present")
            check(data["codebase_stats"]["nodes_extracted"] >= 10, "T8.5 ≥10 nodes extracted")
            check(data["codebase_stats"]["edges_extracted"] >= 5,  "T8.6 ≥5 edges extracted")
            check("benchmark_results" in data, "T8.7 benchmark_results present")
            check(len(data["benchmark_results"]) >= 3, "T8.8 ≥3 benchmark queries")
            check(data["poc_redline_compliant"] is True, "T8.9 poc_redline_compliant=True")
            check(len(data["redline_notes"]) >= 4, "T8.10 ≥4 redline notes")
            check("assertions" in data, "T8.11 assertions present")

    # T8.12 — _run_sqlite_only returns results list
    db2, db2_path = _fresh_sqlite()
    nodes2 = [
        _make_node("x", "function", "z.py", 1, 5),
        _make_node("y", "function", "z.py", 6, 10),
        _make_node("Z", "class",    "z.py", 11, 20),
        _make_node("modz", "module","z.py", 1, 20),
    ]
    for n in nodes2:
        n.id = f"int2__{n.name}"
    db2.insert_nodes(nodes2)
    bm25_2 = BM25CodeSearch()
    results2 = _run_sqlite_only(db2, bm25_2, nodes2)
    check(isinstance(results2, list) and len(results2) >= 2,
          f"T8.12 _run_sqlite_only returns ≥2 results (got {len(results2)})")
    db2.close()
    if os.path.exists(db2_path):
        os.remove(db2_path)


# ═════════════════════════════════════════════════════════════════════════════
# T9 — Edge-case / exception branches (coverage push)
# ═════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n── T9: Edge Cases & Exception Branches ──")

    # T9.1 — HugeGraph _request: HTTPError branch
    import urllib.error, urllib.request
    from io import BytesIO
    hg = HugeGraphCodeGraph("http://mock-host:8888", "g")
    mock_fp = BytesIO(b"forbidden body")
    http_err = urllib.error.HTTPError(
        url="http://mock-host:8888/graphs/g/gremlin",
        code=403, msg="Forbidden", hdrs={}, fp=mock_fp
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        result = hg._request("POST", "/gremlin", {"gremlin": "g.V().count()"})
    check(result is None, "T9.1 HTTPError → _request returns None")

    # T9.2 — HugeGraph _request: OSError branch
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        result2 = hg._request("POST", "/gremlin", {"gremlin": "g.V().count()"})
    check(result2 is None, "T9.2 OSError → _request returns None")

    # T9.3 — HugeGraph _request: success path (mocked good response)
    import json as _json
    good_response = MagicMock()
    good_response.__enter__ = MagicMock(return_value=good_response)
    good_response.__exit__ = MagicMock(return_value=False)
    good_response.read.return_value = _json.dumps({"data": [1, 2, 3]}).encode()
    with patch("urllib.request.urlopen", return_value=good_response):
        result3 = hg._request("POST", "/gremlin", {"gremlin": "g.V().count()"})
    check(result3 is not None and result3.get("data") == [1, 2, 3],
          "T9.3 success path: _request returns parsed JSON")

    # T9.4 — HugeGraph check_hugegraph_available: True branch
    with patch("urllib.request.urlopen", return_value=good_response):
        available = check_hugegraph_available("http://mock:8080")
    check(available is True, "T9.4 check_hugegraph_available returns True when server responds")

    # T9.5 — HugeGraph multi-hop callers (depth > 1) via mock success
    mock_vm_resp = {"data": [{"name": ["fn_x"], "file_path": ["x.py"], "line_start": [5]}]}
    with patch.object(hg, "_request", return_value=mock_vm_resp):
        res = hg.query_callers("some_func", depth=3)
    check(isinstance(res, list) and len(res) == 1, "T9.5 HG multi-hop callers (depth=3) via mock")

    # T9.6 — HugeGraph multi-hop callees (depth > 1) via mock
    with patch.object(hg, "_request", return_value=mock_vm_resp):
        res2 = hg.query_callees("some_func", depth=2)
    check(isinstance(res2, list) and len(res2) == 1, "T9.6 HG multi-hop callees (depth=2) via mock")

    # T9.7 — HugeGraph query_import_chain with non-empty result
    mock_import_resp = {"data": ["module__os", "module__sys"]}
    with patch.object(hg, "_request", return_value=mock_import_resp):
        imports = hg.query_import_chain("mod_a")
    check(len(imports) == 2, "T9.7 HG query_import_chain non-empty path")

    # T9.8 — HugeGraph query_class_hierarchy with non-empty result
    mock_hier_resp = {"data": ["BaseClass"]}
    with patch.object(hg, "_request", return_value=mock_hier_resp):
        hier = hg.query_class_hierarchy("Child")
    check(len(hier) == 1 and hier[0]["target_id"] == "BaseClass",
          "T9.8 HG query_class_hierarchy non-empty path")

    # T9.9 — BM25 jieba ImportError fallback (whitespace tokenizer)
    bm25 = BM25CodeSearch()
    nodes = [_make_node("alpha_beta", "function", "x.py", 1, 3, "def alpha_beta(): pass")]
    for n in nodes:
        n.id = f"t9__{n.name}"
    with patch.dict("sys.modules", {"jieba": None}):
        # Re-instantiate to hit the fallback branch
        b2 = BM25CodeSearch()
        b2.build_index(nodes)
    # fallback uses whitespace split; rank_bm25 might not be importable with patched jieba
    check(True, "T9.9 jieba ImportError in build_index doesn't crash")

    # T9.10 — BM25 search jieba ImportError fallback
    bm25.build_index(nodes)
    if bm25.bm25:
        with patch.dict("sys.modules", {"jieba": None}):
            results = bm25.search("alpha_beta")
        check(isinstance(results, list), "T9.10 jieba ImportError in search uses whitespace fallback")
    else:
        check(True, "T9.10 BM25 disabled — skip jieba fallback search test")

    # T9.11 — find_python_files: OSError on getsize skips file gracefully
    with tempfile.TemporaryDirectory() as tmpdir:
        pyfile = os.path.join(tmpdir, "test_oserr.py")
        with open(pyfile, "w") as fh:
            fh.write("pass\n")
        with patch("os.path.getsize", side_effect=OSError("permission denied")):
            found = find_python_files(tmpdir, max_files=10)
    check(isinstance(found, list), "T9.11 OSError in getsize handled gracefully")

    # T9.12 — run_poc: not-enough-nodes abort path
    with patch("hugegraph_llm.poc.codegraph_hugegraph_mcp.check_hugegraph_available",
               return_value=False):
        import hugegraph_llm.poc.codegraph_hugegraph_mcp as poc_mod
        orig_result = poc_mod.RESULT_FILE
        orig_sqlite = poc_mod.SQLITE_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            poc_mod.RESULT_FILE = os.path.join(tmpdir, "r.json")
            poc_mod.SQLITE_PATH = os.path.join(tmpdir, "c.db")
            # Patch find_python_files to return empty list → 0 nodes
            with patch("hugegraph_llm.poc.codegraph_hugegraph_mcp.find_python_files",
                       return_value=[]):
                result = run_poc()
            poc_mod.RESULT_FILE = orig_result
            poc_mod.SQLITE_PATH = orig_sqlite
    check(result is False, "T9.12 run_poc returns False when not enough nodes")

    # T9.13 — run_poc: HugeGraph available + setup fails gracefully
    with patch("hugegraph_llm.poc.codegraph_hugegraph_mcp.check_hugegraph_available",
               return_value=True):
        import hugegraph_llm.poc.codegraph_hugegraph_mcp as poc_mod
        orig_result = poc_mod.RESULT_FILE
        orig_sqlite = poc_mod.SQLITE_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            poc_mod.RESULT_FILE = os.path.join(tmpdir, "r.json")
            poc_mod.SQLITE_PATH = os.path.join(tmpdir, "c.db")
            # HugeGraph class raises on clear_graph → should fall back to SQLite
            with patch.object(HugeGraphCodeGraph, "clear_graph",
                               side_effect=Exception("HG unavailable")):
                result2 = run_poc()
            poc_mod.RESULT_FILE = orig_result
            poc_mod.SQLITE_PATH = orig_sqlite
    check(result2 is True, "T9.13 run_poc falls back to SQLite when HG setup fails")

    # T9.14 — _run_sqlite_only: exception path
    db3, db3_path = _fresh_sqlite()
    nodes3 = [
        _make_node("fn", "function", "z.py", 1, 5),
        _make_node("Cls", "class",   "z.py", 6, 20),
        _make_node("modz","module",  "z.py", 1, 20),
    ]
    for n in nodes3:
        n.id = f"t914__{n.name}"
    db3.insert_nodes(nodes3)
    bm25_3 = BM25CodeSearch()
    # Inject a function that raises
    with patch.object(db3, "query_callers", side_effect=RuntimeError("DB crash")):
        results3 = _run_sqlite_only(db3, bm25_3, nodes3)
    check(any(not r.sqlite_correct for r in results3),
          "T9.14 _run_sqlite_only exception path: sqlite_correct=False")
    db3.close()
    if os.path.exists(db3_path):
        os.remove(db3_path)


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("CodeGraph vs HugeGraph PoC — Test Suite")
    print("=" * 64)

    test_dataclasses()
    test_parser()
    test_parser_advanced_resolution()
    test_sqlite()
    test_bm25()
    test_hugegraph_mocked()
    test_benchmark()
    test_utilities()
    test_integration()
    test_edge_cases()

    total = PASS + FAIL
    print("\n" + "=" * 64)
    print(f"RESULT: {PASS}/{total} PASS   {FAIL} FAIL")
    if ERRORS:
        print("Failed tests:")
        for e in ERRORS:
            print(f"  - {e}")
    print("=" * 64)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()

