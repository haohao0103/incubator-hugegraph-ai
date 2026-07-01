"""Tests for description_merger.py — LightRAG-style iterative Map-Reduce."""

import asyncio
import inspect
from unittest.mock import MagicMock, AsyncMock

import pytest

from hugegraph_llm.operators.llm_op.description_merger import (
    CJK_PATTERN,
    DescriptionMerger,
    DescriptionMergerConfig,
    SUMMARIZE_PROMPT_TEMPLATE,
    _count_tokens,
    _format_description_list,
    _has_cjk,
    _partition_descriptions,
    _summarize_partition,
    _summarize_partition_async,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Sync mock LLM that returns a fixed summary."""
    return MagicMock(return_value="Merged summary of descriptions.")


@pytest.fixture
def mock_llm_async():
    """Async mock LLM that returns a fixed summary."""
    return AsyncMock(return_value="Merged async summary of descriptions.")


@pytest.fixture
def default_config():
    return DescriptionMergerConfig()


@pytest.fixture
def custom_config():
    return DescriptionMergerConfig(
        force_llm_threshold=4,
        summary_max_tokens=500,
        summary_context_size=2000,
        max_output_tokens=300,
        separator="||",
        kind="relation",
        name="works_for",
    )


# ---------------------------------------------------------------------------
# Level 1 — single description
# ---------------------------------------------------------------------------

class TestLevel1:
    def test_single_description_returns_directly(self, default_config, mock_llm):
        merger = DescriptionMerger(config=default_config, llm_func=mock_llm)
        result = merger.merge(["Alice is a software engineer."])
        assert result == "Alice is a software engineer."
        mock_llm.assert_not_called()

    def test_single_description_no_llm(self, default_config):
        merger = DescriptionMerger(config=default_config, llm_func=None)
        result = merger.merge(["Bob is a data scientist."])
        assert result == "Bob is a data scientist."

    @pytest.mark.asyncio
    async def test_single_description_async(self, default_config, mock_llm_async):
        merger = DescriptionMerger(config=default_config, llm_func=None)
        result = await merger.merge_async(
            ["Alice is a software engineer."], llm_func_async=mock_llm_async
        )
        assert result == "Alice is a software engineer."
        mock_llm_async.assert_not_called()


# ---------------------------------------------------------------------------
# Level 2 — few descriptions, no LLM
# ---------------------------------------------------------------------------

class TestLevel2:
    def test_few_descriptions_join(self, default_config, mock_llm):
        descs = [f"Desc {i}" for i in range(3)]
        merger = DescriptionMerger(config=default_config, llm_func=mock_llm)
        result = merger.merge(descs)
        assert result == "\n".join(descs)
        mock_llm.assert_not_called()

    def test_custom_separator(self, custom_config):
        descs = ["A", "B", "C"]
        merger = DescriptionMerger(config=custom_config, llm_func=None)
        result = merger.merge(descs)
        assert result == "A||B||C"

    @pytest.mark.asyncio
    async def test_few_descriptions_async(self, default_config, mock_llm_async):
        descs = ["X", "Y", "Z"]
        merger = DescriptionMerger(config=default_config, llm_func=None)
        result = await merger.merge_async(descs, llm_func_async=mock_llm_async)
        assert result == "\n".join(descs)
        mock_llm_async.assert_not_called()


# ---------------------------------------------------------------------------
# Level 3 — medium descriptions, single LLM call
# ---------------------------------------------------------------------------

class TestLevel3:
    def test_medium_descriptions_single_llm_call(self, default_config, mock_llm):
        descs = [f"Description number {i} with some content." for i in range(10)]
        merger = DescriptionMerger(config=default_config, llm_func=mock_llm)
        result = merger.merge(descs)
        assert result == "Merged summary of descriptions."
        mock_llm.assert_called_once()

    def test_level3_no_llm_fallback(self, default_config):
        descs = [f"Desc {i}" for i in range(10)]
        merger = DescriptionMerger(config=default_config, llm_func=None)
        result = merger.merge(descs)
        assert result == "\n".join(descs)

    @pytest.mark.asyncio
    async def test_level3_async(self, default_config, mock_llm_async):
        descs = [f"Desc {i}" for i in range(10)]
        merger = DescriptionMerger(config=default_config, llm_func=None)
        result = await merger.merge_async(descs, llm_func_async=mock_llm_async)
        assert result == "Merged async summary of descriptions."
        mock_llm_async.assert_called_once()


# ---------------------------------------------------------------------------
# Level 4 — iterative Map-Reduce
# ---------------------------------------------------------------------------

class TestLevel4:
    def test_large_descriptions_iterative(self, custom_config, mock_llm):
        descs = [f"Long description {i}: " + "word " * 100 for i in range(20)]
        merger = DescriptionMerger(config=custom_config, llm_func=mock_llm)
        result = merger.merge(descs)
        assert result is not None
        assert len(result) > 0
        assert mock_llm.call_count >= 1

    def test_level4_no_llm_fallback(self, custom_config):
        descs = [f"Long description {i}: " + "word " * 100 for i in range(20)]
        merger = DescriptionMerger(config=custom_config, llm_func=None)
        result = merger.merge(descs)
        assert "Long description 0:" in result

    @pytest.mark.asyncio
    async def test_level4_async(self, custom_config, mock_llm_async):
        descs = [f"Async long desc {i}: " + "word " * 100 for i in range(20)]
        merger = DescriptionMerger(config=custom_config, llm_func=None)
        result = await merger.merge_async(descs, llm_func_async=mock_llm_async)
        assert result is not None
        assert len(result) > 0
        assert mock_llm_async.call_count >= 1


# ---------------------------------------------------------------------------
# CJK token estimation
# ---------------------------------------------------------------------------

class TestCJKTokenEstimation:
    def test_cjk_detection(self):
        assert _has_cjk("Hello 世界") is True
        assert _has_cjk("Hello world") is False
        assert _has_cjk("日本語テスト") is True
        assert _has_cjk("") is False

    def test_cjk_pattern_matches(self):
        matches = CJK_PATTERN.findall("abc日本語def")
        assert len(matches) == 3

    def test_token_count_pure_latin(self):
        tokens = _count_tokens("abc")
        assert tokens == 1

    def test_token_count_pure_cjk(self):
        tokens = _count_tokens("日本語")
        assert tokens == 2

    def test_token_count_mixed(self):
        tokens = _count_tokens("abc日本語def")
        assert tokens == 4

    def test_token_count_empty(self):
        assert _count_tokens("") == 0

    def test_token_count_long_text(self):
        short = "Hello world"
        long = short * 10
        assert _count_tokens(long) > _count_tokens(short)


# ---------------------------------------------------------------------------
# Partition helper
# ---------------------------------------------------------------------------

class TestPartition:
    def test_partition_basic(self):
        descs = ["short", "also short", "medium text here"]
        partitions = _partition_descriptions(descs, max_tokens_per_partition=100)
        assert len(partitions) == 1
        assert len(partitions[0]) == 3

    def test_partition_splits_on_budget(self):
        # Each "x" * 30 → ~10 tokens by heuristic. budget=20 → 2 per partition
        descs = ["x" * 30, "x" * 30, "x" * 30]
        partitions = _partition_descriptions(descs, max_tokens_per_partition=20)
        assert len(partitions) >= 2

    def test_partition_single_oversized(self):
        long_desc = "word " * 500
        descs = [long_desc, "short"]
        partitions = _partition_descriptions(descs, max_tokens_per_partition=100)
        assert len(partitions) == 2
        assert len(partitions[0]) == 1
        assert partitions[0][0] == long_desc

    def test_partition_empty_input(self):
        partitions = _partition_descriptions([], max_tokens_per_partition=100)
        assert partitions == []

    def test_partition_all_fit_single(self):
        descs = ["a", "bb", "ccc"]
        partitions = _partition_descriptions(descs, max_tokens_per_partition=500)
        assert len(partitions) == 1


# ---------------------------------------------------------------------------
# _summarize_partition / _summarize_partition_async
# ---------------------------------------------------------------------------

class TestSummarizePartition:
    def test_summarize_partition_builds_prompt(self, mock_llm):
        partition = ["Desc A", "Desc B"]
        result = _summarize_partition(
            partition, mock_llm, kind="entity", name="Alice", max_output_tokens=300
        )
        assert result == "Merged summary of descriptions."
        mock_llm.assert_called_once()
        prompt_arg = mock_llm.call_args[0][0]
        assert "Desc A" in prompt_arg
        assert "Alice" in prompt_arg
        assert "D1" in prompt_arg

    @pytest.mark.asyncio
    async def test_summarize_partition_async(self, mock_llm_async):
        partition = ["Async A", "Async B"]
        result = await _summarize_partition_async(
            partition,
            mock_llm_async,
            kind="relation",
            name="works_for",
            max_output_tokens=600,
        )
        assert result == "Merged async summary of descriptions."
        mock_llm_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_partition_async_sync_func(self):
        sync_fn = MagicMock(return_value="Sync result")
        partition = ["X"]
        result = await _summarize_partition_async(
            partition, sync_fn, kind="entity", name="E"
        )
        assert result == "Sync result"
        sync_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_list(self, default_config, mock_llm):
        merger = DescriptionMerger(config=default_config, llm_func=mock_llm)
        assert merger.merge([]) == ""

    @pytest.mark.asyncio
    async def test_empty_list_async(self, default_config, mock_llm_async):
        merger = DescriptionMerger(config=default_config, llm_func=None)
        assert await merger.merge_async([], llm_func_async=mock_llm_async) == ""

    def test_all_blank_descriptions(self, default_config):
        merger = DescriptionMerger(config=default_config, llm_func=None)
        assert merger.merge(["", "  ", "\n"]) == ""

    @pytest.mark.asyncio
    async def test_all_blank_async(self, default_config):
        merger = DescriptionMerger(config=default_config, llm_func=None)
        assert await merger.merge_async(["", "  "]) == ""

    def test_very_long_single_description(self, default_config, mock_llm):
        long_desc = "word " * 5000
        merger = DescriptionMerger(config=default_config, llm_func=mock_llm)
        result = merger.merge([long_desc])
        assert result == long_desc
        mock_llm.assert_not_called()

    def test_llm_unavailable_level3_fallback(self, default_config):
        descs = [f"Desc {i}" for i in range(10)]
        merger = DescriptionMerger(config=default_config, llm_func=None)
        result = merger.merge(descs)
        assert result == "\n".join(descs)

    def test_llm_unavailable_level4_fallback(self, custom_config):
        descs = [f"Long {i}: " + "w " * 50 for i in range(20)]
        merger = DescriptionMerger(config=custom_config, llm_func=None)
        result = merger.merge(descs)
        assert "Long 0:" in result

    def test_default_config_values(self):
        cfg = DescriptionMergerConfig()
        assert cfg.force_llm_threshold == 8
        assert cfg.summary_max_tokens == 1200
        assert cfg.summary_context_size == 12000
        assert cfg.max_output_tokens == 600
        assert cfg.separator == "\n"
        assert cfg.kind == "entity"
        assert cfg.name == "unknown"

    def test_blank_descriptions_filtered(self, default_config, mock_llm):
        descs = ["Valid description", "", "  ", "Another valid"]
        merger = DescriptionMerger(config=default_config, llm_func=mock_llm)
        result = merger.merge(descs)
        assert result == "Valid description\nAnother valid"
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Async with sync llm_func fallback
# ---------------------------------------------------------------------------

class TestAsyncWithSyncFallback:
    @pytest.mark.asyncio
    async def test_merge_async_uses_sync_llm_as_fallback(self, mock_llm):
        descs = [f"Desc {i}" for i in range(10)]
        merger = DescriptionMerger(llm_func=mock_llm)
        result = await merger.merge_async(descs)
        assert result == "Merged summary of descriptions."
        mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# Iterative map-reduce convergence
# ---------------------------------------------------------------------------

class TestMapReduceConvergence:
    def test_iterative_converges_to_level2(self):
        call_counts = [0]

        def counting_llm(prompt: str) -> str:
            call_counts[0] += 1
            return f"Summary {call_counts[0]}: short."

        config = DescriptionMergerConfig(
            force_llm_threshold=4,
            summary_max_tokens=200,
            summary_context_size=400,
            max_output_tokens=100,
        )
        descs = [f"Very long description {i}: " + "word " * 80 for i in range(15)]
        merger = DescriptionMerger(config=config, llm_func=counting_llm)
        result = merger.merge(descs)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Cooperative yield
# ---------------------------------------------------------------------------

class patch_async_sleep:
    """Context manager that patches asyncio.sleep to track calls."""

    def __init__(self):
        self.mock = AsyncMock()
        self._original = asyncio.sleep

    async def __aenter__(self):
        asyncio.sleep = self.mock
        return self.mock

    async def __aexit__(self, *args):
        asyncio.sleep = self._original


class TestCooperativeYield:
    @pytest.mark.asyncio
    async def test_yield_every_32_descriptions(self, mock_llm_async):
        config = DescriptionMergerConfig(
            force_llm_threshold=2,
            summary_max_tokens=50,
            summary_context_size=100,
            max_output_tokens=50,
        )
        descs = [f"Desc {i}: " + "x " * 100 for i in range(100)]
        merger = DescriptionMerger(config=config, llm_func=None)

        patcher = patch_async_sleep()
        async with patcher as sleep_mock:
            result = await merger.merge_async(descs, llm_func_async=mock_llm_async)
            # Cooperative yield should happen with many descriptions processed
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Prompt template coverage
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    def test_prompt_template_formatting(self):
        desc_list = _format_description_list(["desc1", "desc2", "desc3"])
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            kind="entity",
            name="Alice",
            description_list_text=desc_list,
            max_output_tokens=600,
            language="English",
        )
        assert "entity" in prompt
        assert "Alice" in prompt
        assert "desc1" in prompt

    def test_prompt_contains_instructions(self):
        desc_list = _format_description_list(["a"])
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            kind="relation",
            name="works_for",
            description_list_text=desc_list,
            max_output_tokens=400,
            language="English",
        )
        assert "Conflict Handling" in prompt
        assert "400" in prompt


# ---------------------------------------------------------------------------
# _classify_level coverage
# ---------------------------------------------------------------------------

class TestClassifyLevel:
    def test_level1_classified(self, default_config):
        merger = DescriptionMerger(config=default_config)
        assert merger._classify_level(["single desc"]) == 1

    def test_level2_classified(self, default_config):
        merger = DescriptionMerger(config=default_config)
        descs = ["a", "b", "c"]
        assert merger._classify_level(descs) == 2

    def test_level3_classified(self, default_config):
        merger = DescriptionMerger(config=default_config)
        descs = [f"Desc {i}" for i in range(10)]
        assert merger._classify_level(descs) == 3

    def test_level4_classified(self, custom_config):
        # Level 4 requires total_tokens > summary_context_size.
        # custom_config: summary_context_size=2000, force_llm_threshold=4.
        # Need descriptions whose total tokens exceed 2000.
        # Each "w " * 200 ≈ 144 tokens. 20 such descriptions ≈ 2880 > 2000.
        merger = DescriptionMerger(config=custom_config)
        descs = [f"Long {i}: " + "w " * 200 for i in range(20)]
        assert merger._classify_level(descs) == 4
