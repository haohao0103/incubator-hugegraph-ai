"""Tests for QueryRouter: Global vs Local search classification.

Run: PYTHONPATH=src python tests/operators/test_query_router.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hugegraph_llm.operators.rag_op.query_router import QueryRouter, RouteResult
from hugegraph_llm.operators.rag_op.e2e_rag_pipeline import E2ERAGPipeline, PipelineConfig


def test_global_main_business_chinese():
    router = QueryRouter()
    r = router.classify("腾讯公司的主要业务是什么？")
    assert r.query_type == "global", f"Expected global, got {r.query_type}"
    assert r.confidence >= 0.7, f"Expected confidence >= 0.7, got {r.confidence}"
    print(f"PASS: global_main_business_chinese (type={r.query_type}, conf={r.confidence})")


def test_global_summary():
    router = QueryRouter()
    r = router.classify("总结一下人工智能领域的发展趋势")
    assert r.query_type == "global", f"Expected global, got {r.query_type}"
    assert r.confidence >= 0.7
    print(f"PASS: global_summary (type={r.query_type}, conf={r.confidence})")


def test_global_compare():
    router = QueryRouter()
    r = router.classify("Compare Alibaba and Tencent")
    assert r.query_type == "global"
    assert r.confidence >= 0.9
    print(f"PASS: global_compare (type={r.query_type}, conf={r.confidence})")


def test_local_who_chinese():
    router = QueryRouter()
    r = router.classify("张三在哪家公司工作？")
    assert r.query_type == "local", f"Expected local, got {r.query_type}"
    assert r.confidence >= 0.7
    print(f"PASS: local_who_chinese (type={r.query_type}, conf={r.confidence})")


def test_local_who_english():
    router = QueryRouter()
    r = router.classify("Who founded Alibaba?")
    assert r.query_type == "local"
    assert r.confidence >= 0.9
    print(f"PASS: local_who_english (type={r.query_type}, conf={r.confidence})")


def test_local_where():
    router = QueryRouter()
    r = router.classify("Where is Tencent headquartered?")
    assert r.query_type == "local"
    assert r.confidence >= 0.9
    print(f"PASS: local_where (type={r.query_type}, conf={r.confidence})")


def test_local_relationship():
    router = QueryRouter()
    r = router.classify("李四和王五是什么关系？")
    assert r.query_type == "local"
    assert r.confidence >= 0.9
    print(f"PASS: local_relationship (type={r.query_type}, conf={r.confidence})")


def test_ambiguous_defaults_global():
    router = QueryRouter()
    r = router.classify("告诉我关于腾讯的信息")
    assert r.query_type == "global"
    assert r.method == "default"
    print(f"PASS: ambiguous_defaults_global (type={r.query_type}, method={r.method})")


def test_pipeline_routes_global():
    pipeline = E2ERAGPipeline(config=PipelineConfig())
    result = pipeline.query("腾讯公司的主要业务是什么？")
    assert result.data["scope"] == "global"
    assert result.data["mode"] == "global_search"
    print(f"PASS: pipeline_routes_global (scope={result.data['scope']}, mode={result.data['mode']})")


def test_pipeline_routes_local():
    pipeline = E2ERAGPipeline(config=PipelineConfig())
    result = pipeline.query("张三在哪家公司工作？")
    assert result.data["scope"] == "local"
    assert result.data["route"]["query_type"] == "local"
    print(f"PASS: pipeline_routes_local (scope={result.data['scope']})")


def test_pipeline_forced_global_mode():
    pipeline = E2ERAGPipeline(config=PipelineConfig())
    result = pipeline.query("Who is Jack Ma?", mode="global")
    assert result.data["scope"] == "global"
    assert result.data["route"]["method"] == "forced"
    print(f"PASS: pipeline_forced_global_mode (scope={result.data['scope']})")


def test_pipeline_forced_local_mode():
    pipeline = E2ERAGPipeline(config=PipelineConfig())
    result = pipeline.query("What are the main themes?", mode="local")
    assert result.data["scope"] == "local"
    assert result.data["route"]["method"] == "forced"
    print(f"PASS: pipeline_forced_local_mode (scope={result.data['scope']})")


if __name__ == "__main__":
    tests = [
        test_global_main_business_chinese,
        test_global_summary,
        test_global_compare,
        test_local_who_chinese,
        test_local_who_english,
        test_local_where,
        test_local_relationship,
        test_ambiguous_defaults_global,
        test_pipeline_routes_global,
        test_pipeline_routes_local,
        test_pipeline_forced_global_mode,
        test_pipeline_forced_local_mode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1

    print()
    print(f"{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
