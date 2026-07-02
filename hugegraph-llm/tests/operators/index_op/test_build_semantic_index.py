# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for BuildSemanticIndex vid_embed_strategy: 4 entity→vector text formats.

Strategies tested:
  - fastrag (default): [{TYPE}] name\\n[DESCRIPTION] {properties}
  - lightrag: {name}\\n{description}
  - ms_graphrag: {title}:{description}
  - hipporag: {name} only

Covers:
  - Each strategy's static formatter output format
  - Property-to-description conversion with filtering
  - Truncation logic
  - Exclude props config
  - Fallback behavior when vertex_details missing
  - _format_entity_text dispatch
"""

from hugegraph_llm.operators.index_op.build_semantic_index import (
    VALID_STRATEGIES,
    BuildSemanticIndex,
)

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestStrategyFormatters:
    """Test each strategy's static formatter in isolation."""

    def test_fastrag_with_description(self):
        result = BuildSemanticIndex._format_fastrag("Person", "张三", "货拉拉CEO,住朝阳区")
        assert result == "[PERSON] 张三\n[DESCRIPTION] 货拉拉CEO,住朝阳区"

    def test_fastrag_without_description(self):
        result = BuildSemanticIndex._format_fastrag("Company", "货拉拉", "")
        assert result == "[COMPANY] 货拉拉"
        # No [DESCRIPTION] line when empty

    def test_lightrag_with_description(self):
        result = BuildSemanticIndex._format_lightrag("张三", "货拉拉CEO,35岁")
        assert result == "张三\n货拉拉CEO,35岁"

    def test_lightrag_without_description(self):
        result = BuildSemanticIndex._format_lightrag("张三", "")
        assert result == "张三"

    def test_ms_graphrag_with_description(self):
        result = BuildSemanticIndex._format_ms_graphrag("张三", "货拉拉CEO")
        assert result == "张三:货拉拉CEO"

    def test_ms_graphrag_without_description(self):
        result = BuildSemanticIndex._format_ms_graphrag("张三", "")
        assert result == "张三"

    def test_hipporag_always_name_only(self):
        result = BuildSemanticIndex._format_hipporag("Person:1:张三", "张三")
        assert result == "张三"


class TestPropertyToDescription:
    """Test property dict → description string conversion."""

    def _make_index(self, **overrides) -> "BuildSemanticIndex":
        """Create a BuildSemanticIndex instance with mocked dependencies."""
        idx = object.__new__(BuildSemanticIndex)
        idx.strategy = overrides.get("strategy", "fastrag")
        idx.max_chars = overrides.get("max_chars", 512)
        idx.exclude_props = set(overrides.get("exclude_props", []))
        return idx

    def test_basic_properties_to_desc(self):
        idx = self._make_index()
        props = {"name": "张三", "age": 35, "city": "北京"}
        result = idx._properties_to_description(props)
        assert "age: 35" in result
        assert "city: 北京" in result
        assert "name: 张三" in result

    def test_empty_properties(self):
        idx = self._make_index()
        result = idx._properties_to_description({})
        assert result == ""

    def test_exclude_props_filters_out(self):
        idx = self._make_index(exclude_props=["id", "created_at"])
        props = {"name": "张三", "id": 42, "created_at": "2024-01-01", "city": "北京"}
        result = idx._properties_to_description(props)
        assert "id:" not in result
        assert "created_at:" not in result
        assert "city: 北京" in result

    def test_none_and_empty_values_filtered(self):
        idx = self._make_index()
        props = {"name": "张三", "notes": None, "bio": ""}
        result = idx._properties_to_description(props)
        assert "notes:" not in result
        assert "bio:" not in result
        assert "name: 张三" in result

    def test_sorted_by_key(self):
        idx = self._make_index()
        props = {"z_key": "z", "a_key": "a", "m_key": "m"}
        result = idx._properties_to_description(props)
        keys_pos = [result.find(k) for k in ["a_key:", "m_key:", "z_key:"]]
        assert keys_pos == sorted(keys_pos), f"Properties should be sorted, got: {result}"

    def test_truncation_within_budget(self):
        idx = self._make_index(max_chars=50)
        long_text = idx._properties_to_description({"a": "x" * 100})
        assert len(long_text) <= 50


class TestTruncation:
    """Test max_chars truncation."""

    def _make_index(self, max_chars=512):
        idx = object.__new__(BuildSemanticIndex)
        idx.max_chars = max_chars
        idx.exclude_props = set()
        return idx

    def test_short_text_unchanged(self):
        idx = self._make_index(max_chars=100)
        assert idx._truncate("hello") == "hello"

    def test_exact_fit(self):
        idx = self._make_index(max_chars=5)
        assert idx._truncate("hello") == "hello"

    def test_long_text_truncated(self):
        idx = self._make_index(max_chars=10)
        result = idx._truncate("hello world! this is too long")
        assert len(result) <= 10

    def test_midword_split_avoided(self):
        idx = self._make_index(max_chars=16)
        result = idx._truncate("hello world foo bar")
        # Should split at space near budget, not mid-word
        last_space = result.rfind(" ")
        if last_space > 16 * 0.8:
            assert len(result) == last_space, f"Should split at space: '{result}'"


class TestFormatEntityTextDispatch:
    """Test _format_entity_text routes to correct strategy."""

    def _make_idx_with_strategy(self, strategy="fastrag"):
        idx = object.__new__(BuildSemanticIndex)
        idx.strategy = strategy
        idx.max_chars = 512
        idx.exclude_props = set()
        return idx

    @pytest.fixture
    def sample_detail(self):
        return {
            "vid": "Person:1:张三",
            "label": "Person",
            "properties": {
                "name": "张三",
                "age": 35,
                "city": "朝阳区",
                "company": "货拉拉",
            },
        }

    def test_dispatch_fastrag_enriched(self, sample_detail):
        idx = self._make_idx_with_strategy("fastrag")
        result = idx._format_entity_text(sample_detail, "Person:1:张三", "张三")
        assert "[PERSON]" in result.upper()
        assert "张三" in result
        assert "[DESCRIPTION]" in result
        assert "朝阳区" in result  # from properties

    def test_dispatch_lightrag_enriched(self, sample_detail):
        idx = self._make_idx_with_strategy("lightrag")
        result = idx._format_entity_text(sample_detail, "Person:1:张三", "张三")
        assert result.startswith("张三\n")
        assert "朝阳区" in result

    def test_dispatch_ms_graphrag_enriched(self, sample_detail):
        idx = self._make_idx_with_strategy("ms_graphrag")
        result = idx._format_entity_text(sample_detail, "Person:1:张三", "张三")
        assert "张三:" in result
        assert "朝阳区" in result

    def test_dispatch_hipporag_ignores_details(self, sample_detail):
        idx = self._make_idx_with_strategy("hipporag")
        result = idx._format_entity_text(sample_detail, "Person:1:张三", "张三")
        assert result == "张三"
        assert "朝阳区" not in result

    def test_fastrag_no_detail_fallback_guess_label(self):
        idx = self._make_idx_with_strategy("fastrag")
        result = idx._format_entity_text(None, "Company:42:货拉拉", "货拉拉")
        assert "[COMPANY]" in result.upper()

    def test_unknown_strategy_falls_back_to_fastrag(self, sample_detail):
        idx = self._make_idx_with_strategy("nonexistent_strategy")
        result = idx._format_entity_text(sample_detail, "Person:1:张三", "张三")
        # Should still produce output (fallback to fastrag)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExtractNames:
    """Test legacy name extraction helper."""

    def test_pk_vid_format(self):
        idx = object.__new__(BuildSemanticIndex)
        names = idx._extract_names(["Person:1:张三", "Company:2:货拉拉", "Order:99"])
        assert names == ["张三", "货拉拉", "99"]

    def test_non_pk_vid_passthrough(self):
        idx = object.__new__(BuildSemanticIndex)
        vids = ["some_random_id", "another_id"]
        names = idx._extract_names(vids)
        assert names == vids


class TestDetailLookup:
    """Test detail lookup builder."""

    def test_build_lookup(self):
        details = [
            {"vid": "A:1:x", "label": "A", "props": {}},
            {"vid": "B:2:y", "label": "B", "props": {}},
        ]
        idx = object.__new__(BuildSemanticIndex)
        lookup = idx._build_detail_lookup(details)
        assert lookup["A:1:x"]["label"] == "A"
        assert lookup["B:2:y"]["label"] == "B"

    def test_empty_details(self):
        idx = object.__new__(BuildSemanticIndex)
        lookup = idx._build_detail_lookup([])
        assert lookup == {}


class TestGuessLabelFromVid:
    """Test VID → label fallback guesser."""

    def test_standard_vid(self):
        assert BuildSemanticIndex._guess_label_from_vid("Person:1:张三") == "Person"

    def test_underscore_vid(self):
        assert BuildSemanticIndex._guess_label_from_vid("my_label_123:name") == "my_label"

    def test_no_colon_fallback(self):
        assert BuildSemanticIndex._guess_label_from_vid("plain_id") == "ENTITY"


class TestValidStrategies:
    """Test VALID_STRATEGIES constant covers all implemented strategies."""

    def test_all_four_strategies_present(self):
        expected = {"fastrag", "lightrag", "ms_graphrag", "hipporag"}
        assert expected == VALID_STRATEGIES

    def test_strategies_match_formatter_methods(self):
        # Every valid strategy must have a corresponding _format_* method
        idx = object.__new__(BuildSemanticIndex)
        for strategy in VALID_STRATEGIES:
            method_name = f"_format_{strategy}" if strategy != "ms_graphrag" else "_format_ms_graphrag"
            assert hasattr(idx, method_name), f"Missing formatter for strategy '{strategy}'"
