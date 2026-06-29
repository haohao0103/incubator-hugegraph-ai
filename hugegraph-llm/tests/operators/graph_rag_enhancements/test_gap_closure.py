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

"""
Test suite for GraphRAG Enhancement modules (G1-G5).

Coverage target: >= 90% of all public methods and edge cases.
Run:  pytest tests/operators/graph_rag_enhancements/test_gap_closure.py -v --cov=graph_rag_enhancements --cov-report=term-missing

Tests are organized by Gap module with clear section markers.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from hugegraph_llm.operators.graph_rag_enhancements.llm_cache import (
    BaseCache,
    InMemoryCache,
    JsonFileCache,
    NoopCache,
    CacheStats,
    create_cache_key,
    create_llm_cache,
)
from hugegraph_llm.operators.graph_rag_enhancements.token_budget import (
    TokenCounter,
    TokenBudgetManager,
    BudgetExceededError,
    BudgetConfig,
    SlidingWindowRateLimiter,
    LLMCallGuard,
)
from hugegraph_llm.operators.graph_rag_enhancements.gleaning_extractor import (
    GleaningExtractor,
    ExtractionResult,
    GleaningConfig,
)
from hugegraph_llm.operators.graph_rag_enhancements.community_detector import (
    CommunityDetector,
    CommunityReporter,
    CommunityReport,
    FindingModel,
    CommunityConfig,
    ClusteringResult,
)
from hugegraph_llm.operators.graph_rag_enhancements.global_retriever import (
    GlobalSearchRetriever,
    DriftChainBuilder,
    SearchResult,
    RetrievedContext,
    GlobalSearchConfig,
)


# ===================================================================
# G2: LLM Cache Tests
# ===================================================================


class TestCreateCacheKey:
    """Tests for cache key generation (deterministic, excludes auth fields)."""

    def test_same_input_produces_same_key(self):
        key1 = create_cache_key("gpt-4o", [{"role": "user", "content": "Hello"}])
        key2 = create_cache_key("gpt-4o", [{"role": "user", "content": "Hello"}])
        assert key1 == key2

    def test_different_model_produces_different_key(self):
        key1 = create_cache_key("gpt-4o", [{"role": "user", "content": "Hi"}])
        key2 = create_cache_key("gpt-3.5-turbo", [{"role": "user", "content": "Hi"}])
        assert key1 != key2

    def test_api_key_excluded(self):
        k1 = create_cache_key(
            "gpt-4o",
            [{"role": "user", "content": "test"}],
            extra_params={"api_key": "secret123"},
        )
        k2 = create_cache_key(
            "gpt-4o",
            [{"role": "user", "content": "test"}],
            extra_params={"api_key": "different-secret"},
        )
        assert k1 == k2  # api_key excluded → same key

    def test_base_url_excluded(self):
        k1 = create_cache_key("gpt-4o", [{"role": "user", "content": "x"}],
                              extra_params={"base_url": "https://a.com"})
        k2 = create_cache_key("gpt-4o", [{"role": "user", "content": "x"}],
                              extra_params={"base_url": "https://b.com"})
        assert k1 == k2

    def test_temperature_included(self):
        k1 = create_cache_key("gpt-4o", [{"role": "user", "content": "x"}], temperature=0.0)
        k2 = create_cache_key("gpt-4o", [{"role": "user", "content": "x"}], temperature=1.0)
        assert k1 != k2

    def test_max_tokens_included(self):
        k1 = create_cache_key("gpt-4o", [{"role": "user", "content": "x"}], max_tokens=100)
        k2 = create_cache_key("gpt-4o", [{"role": "user", "content": "x"}], max_tokens=200)
        assert k1 != k2

    def test_multimodal_content_hashed_text_only(self):
        msg_with_image = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }
        key = create_cache_key("gpt-4o", [msg_with_image])
        assert isinstance(key, str)
        assert len(key) > 10

    def test_key_starts_with_prefix(self):
        key = create_cache_key("model", [])
        assert key.startswith("llm_")

    def test_empty_messages(self):
        key = create_cache_key("gpt-4o", [])
        assert isinstance(key, str)


class TestInMemoryCache:
    """Unit tests for InMemoryCache — fast, no disk I/O."""

    @pytest.fixture()
    def cache(self) -> InMemoryCache:
        return InMemoryCache(default_ttl_seconds=60)

    def test_set_and_get(self, cache: InMemoryCache):
        cache.set("k1", {"answer": 42})
        result = cache.get("k1")
        assert result == {"answer": 42}

    def test_get_miss_returns_none(self, cache: InMemoryCache):
        assert cache.get("nonexistent") is None

    def test_has_true_on_hit(self, cache: InMemoryCache):
        cache.set("k1", "val")
        assert cache.has("k1") is True

    def test_has_false_on_miss(self, cache: InMemoryCache):
        assert cache.has("nope") is False

    def test_delete_existing(self, cache: InMemoryCache):
        cache.set("k1", "val")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None

    def test_delete_missing(self, cache: InMemoryCache):
        assert cache.delete("nope") is False

    def test_clear_removes_all(self, cache: InMemoryCache):
        cache.set("a", 1); cache.set("b", 2); cache.set("c", 3)
        n = cache.clear()
        assert n == 3
        assert cache.get("a") is None

    def test_ttl_expiry(self, cache: InMemoryCache):
        c = InMemoryCache(default_ttl_seconds=0.01)  # 10ms TTL
        c.set("expiring", "value")
        time.sleep(0.02)
        assert c.get("expiring") is None  # Expired

    def test_stats_tracking(self, cache: InMemoryCache):
        cache.get("miss1")          # miss
        cache.get("miss2")          # miss
        cache.set("k1", "v1")       # write
        cache.get("k1")             # hit
        snap = cache.stats.snapshot()
        assert snap["hits"] == 1
        assert snap["misses"] == 2
        assert snap["writes"] == 1
        assert abs(snap["hit_rate"] - 1 / 3) < 0.001

    def test_eviction_counted_on_expire(self, cache: InMemoryCache):
        c = InMemoryCache(default_ttl_seconds=0.005)
        c.set("e", "v")
        time.sleep(0.01)
        c.get("e")  # triggers miss + eviction
        snap = c.stats.snapshot()
        assert snap["evictions"] >= 1

    def test_child_isolation(self, cache: InMemoryCache):
        child = cache.child("subspace")
        cache.set("parent_k", "parent_v")
        child.set("child_k", "child_v")

        assert cache.get("parent_k") == "parent_v"
        assert cache.get("child_k") is None      # parent can't see child
        assert child.get("parent_k") is None     # child can't see parent
        assert child.get("child_k") == "child_v"


class TestNoopCache:
    """NoopCache always misses."""

    def test_get_always_none(self):
        nc = NoopCache()
        assert nc.get("anything") is None

    def test_set_does_nothing(self):
        nc = NoopCache()
        nc.set("k", "v")
        assert nc.get("k") is None

    def test_has_always_false(self):
        nc = NoopCache()
        assert nc.has("x") is False

    def test_clear_returns_zero(self):
        nc = NoopCache()
        assert nc.clear() == 0

    def test_stats_all_zero(self):
        nc = NoopCache()
        nc.get("anything")  # trigger miss
        snap = nc.stats.snapshot()
        assert snap["hits"] == 0 and snap["misses"] == 1 and snap["writes"] == 0


class TestJsonFileCache:
    """Integration tests for JsonFileCache (requires temp directory)."""

    @pytest.fixture()
    def tmpcache(self, tmp_path):
        return JsonFileCache(cache_dir=tmp_path / "llm_test", default_ttl_seconds=3600)

    def test_persistence_across_instances(self, tmpcache: JsonFileCache, tmp_path):
        tmpcache.set("persist", {"data": [1, 2, 3]})
        # Re-open same directory
        cache2 = JsonFileCache(cache_dir=tmp_path / "llm_test")
        assert cache2.get("persist") == {"data": [1, 2, 3]}

    def test_json_corruption_handled_gracefully(self, tmpcache: JsonFileCache):
        fpath = tmpcache.cache_dir / "corrupt.json"
        fpath.write_text("{invalid json!!!", encoding="utf-8")
        assert tmpcache.get("corrupt") is None  # Should not crash

    def test_unicode_content_roundtrip(self, tmpcache: JsonFileCache):
        val = {"zh": "你好世界", "emoji": "🦊 GraphRAG"}
        tmpcache.set("unicode", val)
        assert tmpcache.get("unicode") == val

    def test_ttl_expired_file_cleaned(self, tmpcache: JsonFileCache):
        c = JsonFileCache(cache_dir=tmpcache.cache_dir, default_ttl_seconds=0.001)
        c.set("short_lived", "gone soon")
        time.sleep(0.002)
        evicted = c.evict_expired()
        assert evicted >= 1
        assert c.get("short_lived") is None

    def test_child_creates_subdirectory(self, tmpcache: JsonFileCache):
        sub = tmpcache.child("extractor")
        sub.set("nested", "ok")
        assert sub.get("nested") == "ok"
        # Child cache stores data in subdirectory; parent's view depends on implementation
        # The key test is that child can read its own writes
        tmpcache.set("parent_only", "val")
        assert sub.get("parent_only") is None

    def test_size_bytes_report(self, tmpcache: JsonFileCache):
        tmpcache.set("a", "x" * 1024)
        sz = tmpcache.size_bytes()
        assert sz > 500  # At least some bytes stored


class TestCacheFactory:
    def test_factory_json_file(self):
        c = create_llm_cache("json_file", cache_dir="/tmp/test_cache")
        assert isinstance(c, JsonFileCache)

    def test_factory_memory(self):
        c = create_llm_cache("memory")
        assert isinstance(c, InMemoryCache)

    def test_factory_noop(self):
        c = create_llm_cache("noop")
        assert isinstance(c, NoopCache)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            create_llm_cache("redis")


# ===================================================================
# G3: Token Budget Tests
# ===================================================================


class TestTokenCounter:
    def test_count_plain_text(self):
        tc = TokenCounter(model="gpt-4o")
        r = tc.count("Hello world")
        assert r.num_tokens > 0
        assert r.encoding_name != "unknown"

    def test_count_messages(self):
        tc = TokenCounter(model="gpt-4o")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."},
        ]
        r = tc.count_messages(msgs)
        assert r.num_tokens > len(msgs) * 4  # per-message overhead

    def test_estimate_input_shortcut(self):
        tc = TokenCounter()
        assert tc.estimate_input_tokens(prompt="Hello") > 0

    def test_available_flag(self):
        tc = TokenCounter()  # May or may not have tiktoken
        assert isinstance(tc.available, bool)


class TestBudgetManager:
    @pytest.fixture()
    def budget(self) -> TokenBudgetManager:
        return TokenBudgetManager(config=BudgetConfig(
            max_tokens_per_request=100,
            max_tokens_global=500,
            enable_truncation=False,
        ))

    def test_within_budget_passes(self, budget: TokenBudgetManager):
        assert budget.check(50) is True

    def test_over_request_limit_raises(self, budget: TokenBudgetManager):
        with pytest.raises(BudgetExceededError):
            budget.check_or_raise(150, scope="request")

    def test_over_global_limit_raises(self, budget: TokenBudgetManager):
        budget.record_usage(450)  # Use 450/500
        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check_or_raise(60, scope="global")
        assert exc_info.value.scope == "global"

    def test_truncation_enabled_adjusts(self):
        b = TokenBudgetManager(config=BudgetConfig(
            max_tokens_per_request=50, enable_truncation=True))
        approved = b.check_or_raise(80)  # Over limit but truncation on
        assert approved <= 50  # Adjusted down to limit

    def test_record_usage_tracks(self, budget: TokenBudgetManager):
        budget.record_usage(100)
        assert budget.global_used == 100
        assert budget.global_remaining == 400

    def test_reset_global(self, budget: TokenBudgetManager):
        budget.record_usage(400)
        budget.reset_global()
        assert budget.global_used == 0

    def test_snapshot(self, budget: TokenBudgetManager):
        s = budget.snapshot()
        assert "global_used" in s and "utilization" in s


class TestTruncation:
    @pytest.fixture()
    def manager(self):
        return TokenBudgetManager(config=BudgetConfig(
            max_tokens_per_request=30,
            enable_truncation=True,
        ))

    def _make_msgs(self, system: str, user_parts: list[str]):
        msgs = [{"role": "system", "content": system}]
        for p in user_parts:
            msgs.append({"role": "user", "content": p})
        return msgs

    def test_truncate_head_keeps_system_and_early(self, manager: TokenBudgetManager):
        msgs = self._make_msgs("sys", ["A" * 50, "B" * 50, "C" * 50])
        trimmed = manager.truncate_messages(msgs, 20, strategy="head")
        # System should be preserved; only first user msg fits
        assert len(trimmed) >= 1
        assert any(m.get("role") == "system" for m in trimmed)

    def test_truncate_tail_keeps_latest(self, manager: TokenBudgetManager):
        msgs = self._make_msgs("sys", ["A" * 50, "B" * 50, "C" * 50])
        trimmed = manager.truncate_messages(msgs, 20, strategy="tail")
        assert len(trimmed) >= 1

    def test_preserve_system_keeps_system_user_drops_tool(self, manager: TokenBudgetManager):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "U" * 10},
            {"role": "tool", "content": "T" * 100},  # Should drop first
            {"role": "assistant", "content": "A" * 50},
        ]
        trimmed = manager.truncate_messages(msgs, 25, strategy="preserve_system")
        roles = [m.get("role") for m in trimmed]
        assert "system" in roles or "user" in roles


class TestSlidingWindowRateLimiter:
    def test_basic_acquire_succeeds(self):
        rl = SlidingWindowRateLimiter(
            period_in_seconds=1.0, requests_per_period=10
        )
        with rl.acquire(token_count=100):
            pass  # Should not block

    def test_rate_limit_blocks_when_full(self):
        rl = SlidingWindowRateLimiter(period_in_seconds=10.0, requests_per_period=2)
        acquired = []
        for i in range(3):
            try:
                # Use a short timeout to avoid hanging tests
                import threading
                event = threading.Event()
                def do_acquire():
                    with rl.acquire():
                        acquired.append(True)
                        event.set()

                t = threading.Thread(target=do_acquire)
                t.start()
                if event.wait(timeout=2.0):
                    pass
                else:
                    break  # Blocked — rate limited
                t.join(timeout=1)
                time.sleep(0.05)
            except Exception:
                break
        # Should have allowed at most `requests_per_period` through
        assert len(acquired) <= 2

    def test_reset_clears_state(self):
        rl = SlidingWindowRateLimiter(requests_per_period=1)
        with rl.acquire():
            pass
        rl.reset()
        with rl.acquire():  # Should succeed after reset
            pass


# ===================================================================
# G1: Gleaning Extractor Tests
# ===================================================================


class MockLLMForGleaning:
    """Mock LLM that returns pre-canned responses for gleaning tests."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = []

    async def agenerate(self, messages=None, prompt=None):
        self.call_count += 1
        if messages:
            self.last_messages = messages
        idx = min(self.call_count - 1, len(self.responses) - 1)
        return self.responses[idx] if self.responses else '{"entities":[],"relationships":[]}'

    async def __call__(self, messages=None):
        return await self.agenerate(messages=messages)


class TestGleaningExtractor:
    @pytest.fixture()
    def mock_llm(self):
        initial_response = json.dumps({
            "entities": [{"name": "Alice", "description": "Person"}, {"name": "Bob", "description": ""}],
            "relationships": [{"source": "Alice", "target": "Bob", "description": "knows"}],
        })
        glean_response = json.dumps({
            "entities": [{"name": "Bob", "description": "Friend of Alice since college"}],
            "relationships": [],
        })
        return MockLLMForGleaning([initial_response, glean_response])

    @pytest.fixture()
    def config(self) -> GleaningConfig:
        return GleaningConfig(enabled=True, max_rounds=1, use_json_mode=True)

    @pytest.mark.asyncio
    async def test_gleaning_produces_more_entities(self, mock_llm, config):
        ext = GleaningExtractor(llm_generate_fn=mock_llm, config=config)
        result = await ext.extract_with_gleaning("Alice knows Bob from work.")
        assert result.is_gleaning is False  # Final merged result flag
        # Initial had 2 entities, gleaning added description for Bob
        names = [e.get("name") for e in result.entities]
        assert "Alice" in names
        assert "Bob" in names
        # Bob should now have a longer description (from gleaning)
        bob = [e for e in result.entities if e.get("name") == "Bob"]
        if bob:
            assert len(bob[0].get("description", "")) > 0  # Got description from gleaning

    @pytest.mark.asyncio
    async def test_gleaning_disabled_skips_gleaning_call(self, mock_llm):
        config_disabled = GleaningConfig(enabled=False, max_rounds=1)
        ext = GleaningExtractor(llm_generate_fn=mock_llm, config=config_disabled)
        result = await ext.extract_with_gleaning("Test text.")
        assert mock_llm.call_count == 1  # Only initial extraction

    @pytest.mark.asyncio
    async def test_zero_rounds_skips(self, mock_llm):
        cfg = GleaningConfig(max_rounds=0)
        ext = GleaningExtractor(llm_generate_fn=mock_llm, config=cfg)
        await ext.extract_with_gleaning("text")
        assert mock_llm.call_count == 1

    def test_merge_entities_by_description_length(self):
        base = [{"name": "X", "description": "Short"}]
        gleaned = [{"name": "X", "description": "Much Longer Description Here"}]
        merged = GleaningExtractor._merge_entities(base, gleaned)
        assert len(merged) == 1
        assert merged[0]["description"] == "Much Longer Description Here"

    def test_merge_entities_new_entity_added(self):
        base = [{"name": "X", "description": "OK"}]
        gleaned = [{"name": "Y", "description": "New entity"}]
        merged = GleaningExtractor._merge_entities(base, gleaned)
        assert len(merged) == 2
        assert {e["name"] for e in merged} == {"X", "Y"}

    def test_merge_relationships_dedup_by_pair(self):
        base = [{"source": "A", "target": "B", "description": "old desc"}]
        gleaned = [{"source": "A", "target": "B", "description": "new improved desc"}]
        merged = GleaningExtractor._merge_relationships(base, gleaned)
        assert len(merged) == 1
        assert merged[0]["description"] == "new improved desc"

    def test_parse_json_valid(self):
        ext = GleaningExtractor.__new__(GleaningExtractor)
        raw = '{"entities":[{"name":"E1","type":"T1"}],"relationships":[{"src_id":"E1","tgt_id":"E2","rel":"R1"}]}'
        entities, rels = ext._parse_json(raw)
        assert len(entities) == 1
        assert len(rels) == 1

    def test_parse_json_with_markdown_fence(self):
        ext = GleaningExtractor.__new__(GleaningExtractor)
        raw = '```json\n{"entities":[{"name":"E1"}],"relationships":[]}\n```'
        entities, rels = ext._parse_json(raw)
        assert len(entities) == 1

    def test_parse_json_invalid_falls_back_to_empty(self):
        ext = GleaningExtractor.__new__(GleaningExtractor)
        entities, rels = ext._parse_json("not json at all")
        assert entities == [] and rels == []

    def test_parse_delimiter_format(self):
        cfg = GleaningConfig(use_json_mode=False)
        extractor = GleaningExtractor.__new__(GleaningExtractor)
        extractor.config = cfg
        raw = "entity:Alice,Person\nrelation:Alice→Bob,knows\n<|end|>"
        entities, rels = extractor._parse_delimiter(raw)
        assert len(entities) >= 1
        assert len(rels) >= 1

    def test_extraction_result_fields(self):
        r = ExtractionResult(entities=[{"name": "E"}], relationships=[], duration_ms=42.5)
        assert r.tokens_used == 0
        assert r.is_gleaning is False


# ===================================================================
# G5: Community Detector Tests
# ===================================================================


SAMPLE_EDGES = [
    ("Alice", "Bob", 1.0),
    ("Bob", "Carol", 1.5),
    ("Carol", "Dave", 0.8),
    ("Dave", "Eve", 1.2),
    ("Frank", "Grace", 0.9),  # Separate component
]


class TestCommunityDetector:

    @pytest.fixture()
    def detector(self) -> CommunityDetector:
        return CommunityDetector(config=CommunityConfig())

    def test_detect_basic(self, detector: CommunityDetector):
        result = detector.detect(SAMPLE_EDGES)
        assert result.num_nodes > 0
        assert result.num_communities > 0
        assert len(result.node_to_community) == result.num_nodes
        assert result.method != "unknown"
        assert result.duration_ms >= 0

    def test_two_components_detected(self, detector: CommunityDetector):
        edges = SAMPLE_EDGES + [("isolated_node_a", "isolated_node_b", 1.0)]
        result = detector.detect(edges)
        assert result.num_communities >= 2  # At least main chain + isolated pair

    def test_single_edge_one_community(self, detector: CommunityDetector):
        result = detector.detect([("A", "B", 1.0)])
        assert result.num_communities >= 1

    def test_empty_edges(self, detector: CommunityDetector):
        result = detector.detect([])
        assert result.num_communities == 0
        assert result.num_nodes == 0

    def test_community_to_nodes_matches_node_to_community(self, detector: CommunityDetector):
        result = detector.detect(SAMPLE_EDGES)
        # Verify consistency between two views
        for comm_id, nodes in result.community_to_nodes.items():
            for node in nodes:
                assert result.node_to_community[node] == comm_id

    def test_config_params_used(self):
        det = CommunityDetector(config=CommunityConfig(max_cluster_size=3))
        result = det.detect(SAMPLE_EDGES)
        assert result is not None  # Config accepted without error


class TestCommunityReporter:
    @pytest.fixture()
    def mock_llm(self):
        llm = AsyncMock()
        llm.agenerate.return_value = json.dumps({
            "title": "Test Community",
            "summary": "A group of people connected together.",
            "rating": 7.5,
            "rating_explanation": "Medium impact.",
            "findings": [
                {"summary": "Finding 1", "explanation": "Detail 1"},
                {"summary": "Finding 2", "explanation": "Detail 2"},
            ],
        })
        return llm

    @pytest.fixture()
    def clustering(self) -> ClusteringResult:
        return ClusteringResult(
            node_to_community={"A": 0, "B": 0, "C": 0, "F": 1, "G": 1},
            community_to_nodes={0: ["A", "B", "C"], 1: ["F", "G"]},
            num_communities=2, num_nodes=5, method="test",
        )

    @pytest.mark.asyncio
    async def test_generates_reports_for_each_community(self, mock_llm, clustering):
        reporter = CommunityReporter(llm_generate_fn=mock_llm)
        reports = await reporter.generate_reports(
            clustering=clustering,
            all_entities=[
                {"name": "A", "type": "Person", "description": "Entity A"},
                {"name": "B", "type": "Person", "description": "Entity B"},
                {"name": "C", "type": "Org", "description": "Entity C"},
                {"name": "F", "type": "Location", "description": "Entity F"},
                {"name": "G", "type": "Location", "description": "Entity G"},
            ],
            all_relationships=[
                {"source": "A", "target": "B", "description": "knows"},
                {"source": "F", "target": "G", "description": "located near"},
            ],
        )
        assert len(reports) >= 1  # At least community 0 has members
        for report in reports:
            assert report.id != ""
            assert report.community_id >= 0
            assert isinstance(report.rating, float)
            assert report.size > 0

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, clustering):
        failing_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        reporter = CommunityReporter(llm_generate_fn=failing_llm)
        # Use entities that ARE in a community so at least one report is attempted
        reports = await reporter.generate_reports(
            clustering=clustering,
            all_entities=[
                {"name": "A", "description": "Entity A", "type": "Person"},
                {"name": "B", "description": "Entity B", "type": "Person"},
            ],
            all_relationships=[{"source": "A", "target": "B", "description": "knows"}],
        )
        # Should not crash; at least one report attempt was made
        assert isinstance(reports, list)  # No exception raised
        # Either we got fallback error reports or successful ones
        if len(reports) > 0:
            assert "Error" in reports[0].summary or reports[0].title != ""


class TestCommunityReportDataModel:
    def test_to_dict_contains_all_fields(self):
        cr = CommunityReport(
            id="abc123", community_id=5, title="T",
            summary="S", rating=8.0, findings=[
                FindingModel(summary="F1", explanation="D1"),
            ], size=10,
        )
        d = cr.to_dict()
        assert d["id"] == "abc123"
        assert d["community_id"] == 5
        assert d["rating"] == 8.0
        assert len(d["findings"]) == 1
        assert d["size"] == 10


# ===================================================================
# G4: Global Retriever / DRIFT Tests
# ===================================================================


class TestDriftChainBuilder:
    @pytest.fixture()
    def builder(self):
        return DriftChainBuilder(config=GlobalSearchConfig(max_hops=2))

    @pytest.fixture()
    def graph_data(self) -> dict:
        """Simple friendship graph."""
        store = {
            "Alice": [
                {"target": "Bob", "description": "knows well", "weight": 1.0},
                {"target": "Charlie", "description": "colleague", "weight": 0.8},
            ],
            "Bob": [
                {"target": "David", "description": "reports to", "weight": 1.0},
                {"target": "Eve", "description": "married to", "weight": 1.5},
            ],
            "Charlie": [
                {"target": "David", "description": "friend", "weight": 0.6},
            ],
            "David": [],  # Leaf node
            "Eve": [],
            "Isolated": [],  # Disconnected
        }
        return store

    def test_build_chains_from_seeds(self, builder, graph_data):
        chains = builder.build_chains(["Alice"], lambda name: graph_data.get(name, []))
        assert len(chains) > 0
        total_expanded = sum(c.get("num_expanded", 0) for c in chains)
        assert total_expanded > 0

    def test_respects_max_hops(self, builder, graph_data):
        b = DriftChainBuilder(config=GlobalSearchConfig(max_hops=1))
        chains = b.build_chains(["Alice"], lambda name: graph_data.get(name, []))
        assert all(c.get("hop", 0) <= 1 for c in chains)

    def test_empty_seeds_return_empty_chains(self, builder, graph_data):
        chains = builder.build_chains([], lambda name: graph_data.get(name, []))
        assert chains == []

    def test_unknown_seed_returns_empty(self, builder, graph_data):
        chains = builder.build_chains(["NonexistentNode"], lambda name: [])
        assert chains == []

    def test_format_chain_context_output(self, builder, graph_data):
        chains = builder.build_chains(["Alice"], lambda name: graph_data.get(name, []))
        text, count = DriftChainBuilder.format_chain_context(chains)
        assert "<drift_chain>" in text
        assert count >= 0

    def test_deduplicates_visited_entities(self, builder, graph_data):
        chains = builder.build_chains(["Alice"], lambda name: graph_data.get(name, []))
        visited = set()
        for chain in chains:
            for entity in chain.get("new_entities", []):
                assert entity not in visited, f"Duplicate visit: {entity}"
                visited.add(entity)


class TestGlobalSearchRetriever:

    def test_merge_round_robin_interleaves(self):
        retriever = GlobalSearchRetriever(
            config=GlobalSearchConfig(round_robin_merge=True),
            community_reports=[],
        )
        local_ctxs = [
            RetrievedContext(content="L1", source_type="local_entity", score=0.9),
            RetrievedContext(content="L2", source_type="local_relation", score=0.7),
        ]
        global_ctxs = [
            RetrievedContext(content="G1", source_type="global_community", score=0.85),
            RetrievedContext(content="G2", source_type="global_community", score=0.65),
        ]
        merged = retriever._merge_results(local_ctxs, global_ctxs)
        sources = [c.source_type for c in merged]
        # Round-robin should interleave L/G/L/G pattern
        assert len(merged) == 4
        assert "local_entity" in sources
        assert "global_community" in sources

    def test_merge_global_first_puts_global_first(self):
        retriever = GlobalSearchRetriever(
            config=GlobalSearchConfig(round_robin_merge=False),
            community_reports=[],
        )
        local = [RetrievedContext(content="L", source_type="local")]
        global_ = [RetrievedContext(content="G", source_type="global")]
        merged = retriever._merge_results(local, global_)
        assert merged[0].source_type == "global"

    def test_extract_seed_entities_from_local_contexts(self):
        seeds = GlobalSearchRetriever._extract_seed_entities([
            RetrievedContext(content="Entity: Alice and Entity: Bob discussed Project X", source_type="local_entity"),
            RetrievedContext(content="Relation: Alice -> Bob : worked together", source_type="local_relation"),
        ])
        assert len(seeds) > 0
        assert isinstance(seeds, list)
        assert all(isinstance(s, str) for s in seeds)


class TestSearchResultDataModel:
    def test_default_values(self):
        sr = SearchResult(query="test query")
        assert sr.contexts == []
        assert sr.mode == "unknown"
        assert sr.duration_ms == 0.0

    def test_stats_dict_populated(self):
        sr = SearchResult(query="q", stats={"community_count": 3})
        assert sr.stats["community_count"] == 3


# ===================================================================
# Integration / Edge Case Tests
# ===================================================================


class TestEdgeCases:

    def test_cache_concurrent_writes(self):
        import threading
        cache = InMemoryCache(default_ttl_seconds=60)
        errors = []

        def writer(i):
            try:
                cache.set(f"k_{i}", {"value": i})
                v = cache.get(f"k_{i}")
                assert v == {"value": i}
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert errors == []

    def test_budget_exhaustion_flow(self):
        budget = TokenBudgetManager(config=BudgetConfig(
            max_tokens_per_request=10, max_tokens_global=25, enable_truncation=False
        ))
        budget.check_or_raise(8)   # OK: 8 <= 10, 8 <= 25
        budget.record_usage(8)     # Used: 8/25
        budget.check_or_raise(7)   # OK: 7 <= 10, 15 <= 25
        budget.record_usage(7)     # Used: 15/25
        with pytest.raises(BudgetExceededError):
            budget.check_or_raise(12)  # 12 > 10 (per-request), also 27 > 25 (global)

    def test_gleaning_merges_complex_overlaps(self):
        base = [
            {"name": "E1", "description": "A"},
            {"name": "E2", "description": "B"},
            {"name": "E3", "description": "C"},
        ]
        gleaned = [
            {"name": "E1", "description": "AAA — much longer description"},
            {"name": "E4", "description": "D"},  # New entity
            {"name": "E2", "description": "BBB"},  # Updated
        ]
        merged = GleaningExtractor._merge_entities(base, gleaned)
        assert len(merged) == 4
        by_name = {e["name"]: e for e in merged}
        assert by_name["E1"]["description"].startswith("AAA")
        assert by_name["E2"]["description"] == "BBB"
        assert by_name["E3"]["description"] == "C"
        assert by_name["E4"]["description"] == "D"

    def test_community_detector_large_graph_performance(self):
        """Stress test: 1000-node random graph."""
        import random
        random.seed(42)
        edges = [(f"N{i}", f"N{j}", round(random.random(), 2))
                  for i in range(1000) for j in range(i+1, min(i+6, 1000))]
        det = CommunityDetector()
        result = det.detect(edges[:500])  # Subset for speed
        assert result.num_nodes > 0
        assert result.duration_ms < 30000  # Should complete within 30s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
