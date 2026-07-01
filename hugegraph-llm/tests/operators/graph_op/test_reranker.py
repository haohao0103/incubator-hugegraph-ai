"""Tests for reranker module — IdentityReranker, CrossEncoderReranker, LLMReranker, RerankerFactory."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hugegraph_llm.operators.graph_op.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    IdentityReranker,
    LLMReranker,
    RerankResult,
    RerankerFactory,
)


# ---------------------------------------------------------------------------
# IdentityReranker
# ---------------------------------------------------------------------------


class TestIdentityReranker:
    """Returns original order with score=1.0."""

    @pytest.mark.asyncio
    async def test_returns_original_order(self):
        reranker = IdentityReranker()
        docs = ["doc_a", "doc_b", "doc_c"]
        results = await reranker.rerank("query", docs)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.index == i
            assert r.text == docs[i]
            assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_top_k_limit(self):
        reranker = IdentityReranker()
        docs = ["d1", "d2", "d3", "d4", "d5"]
        results = await reranker.rerank("q", docs, top_k=3)
        assert len(results) == 3
        assert results[0].text == "d1"

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        reranker = IdentityReranker()
        results = await reranker.rerank("q", [])
        assert results == []


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """CrossEncoder with mock and fallback."""

    @pytest.mark.asyncio
    async def test_fallback_when_not_installed(self):
        reranker = CrossEncoderReranker()
        with patch(
            "hugegraph_llm.operators.graph_op.reranker.CrossEncoderReranker._try_load_model",
            return_value=False,
        ):
            reranker._available = False
            results = await reranker.rerank("q", ["d1", "d2"])
            # Falls back to IdentityReranker
            assert len(results) == 2
            assert all(r.score == 1.0 for r in results)

    @pytest.mark.asyncio
    async def test_rerank_with_mock_model(self):
        reranker = CrossEncoderReranker()
        # Mock the CrossEncoder class
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.6]

        with patch(
            "hugegraph_llm.operators.graph_op.reranker.CrossEncoderReranker._try_load_model",
            return_value=True,
        ):
            reranker._available = True
            reranker._model = mock_model

            results = await reranker.rerank("query", ["d1", "d2", "d3"], top_k=2)
            assert len(results) == 2
            # Should be sorted by score descending
            assert results[0].score == 0.9
            assert results[0].text == "d1"
            assert results[1].score == 0.6
            assert results[1].text == "d3"

    @pytest.mark.asyncio
    async def test_model_name_configurable(self):
        reranker = CrossEncoderReranker(model_name="custom-model")
        assert reranker._model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_lazy_model_loading(self):
        reranker = CrossEncoderReranker()
        assert reranker._available is None
        assert reranker._model is None

        # Simulate successful loading
        mock_model = MagicMock()
        with patch(
            "hugegraph_llm.operators.graph_op.reranker.CrossEncoderReranker._try_load_model",
            return_value=True,
        ) as mock_load:
            reranker._available = True
            reranker._model = mock_model
            results = await reranker.rerank("q", ["d1"])
            # Model predict should have been called via asyncio.to_thread
            # (mocked at the _try_load level, model is pre-set)

    @pytest.mark.asyncio
    async def test_import_error_fallback(self):
        """When sentence_transformers is not installed, fallback gracefully."""
        reranker = CrossEncoderReranker()
        # _try_load_model will fail because sentence_transformers is likely not installed
        # in test env; if it happens to be installed, patch the import
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = reranker._try_load_model()
            assert result is False
            assert reranker._available is False


# ---------------------------------------------------------------------------
# LLMReranker
# ---------------------------------------------------------------------------


class TestLLMReranker:
    """LLM-based reranker with mock functions."""

    def test_parse_score_numeric(self):
        assert LLMReranker._parse_score("7") == 7.0
        assert LLMReranker._parse_score("8.5") == 8.5

    def test_parse_score_with_text(self):
        assert LLMReranker._parse_score("Relevance score: 9.2") == 9.2
        assert LLMReranker._parse_score("I'd rate this 6 out of 10") == 6.0

    def test_parse_score_no_number(self):
        assert LLMReranker._parse_score("no score here") == 0.0

    def test_parse_score_empty(self):
        assert LLMReranker._parse_score("") == 0.0

    @pytest.mark.asyncio
    async def test_rerank_with_sync_llm_func(self):
        def mock_llm(prompt: str) -> str:
            # Return different scores based on document index
            if "d1" in prompt:
                return "9"
            elif "d2" in prompt:
                return "3"
            elif "d3" in prompt:
                return "7"
            return "0"

        reranker = LLMReranker(llm_func=mock_llm)
        results = await reranker.rerank("q", ["d1", "d2", "d3"], top_k=2)
        assert len(results) == 2
        # Sorted by score descending
        assert results[0].text == "d1"
        assert results[0].score == 9.0
        assert results[1].text == "d3"
        assert results[1].score == 7.0

    @pytest.mark.asyncio
    async def test_rerank_with_async_llm_func(self):
        async def mock_llm_async(prompt: str) -> str:
            if "d1" in prompt:
                return "8.5"
            elif "d2" in prompt:
                return "2.0"
            return "5"

        reranker = LLMReranker(llm_func_async=mock_llm_async)
        results = await reranker.rerank("q", ["d1", "d2", "d3"], top_k=2)
        assert len(results) == 2
        assert results[0].score == 8.5

    @pytest.mark.asyncio
    async def test_rerank_no_llm_func_fallback(self):
        reranker = LLMReranker()
        results = await reranker.rerank("q", ["d1", "d2"])
        # Falls back to IdentityReranker
        assert all(r.score == 1.0 for r in results)

    @pytest.mark.asyncio
    async def test_async_prefers_llm_func_async(self):
        """When both sync and async are provided, async should be used."""
        sync_fn = MagicMock(return_value="1")
        async_fn = AsyncMock(return_value="10")

        reranker = LLMReranker(llm_func=sync_fn, llm_func_async=async_fn)
        results = await reranker.rerank("q", ["d1"])
        assert results[0].score == 10.0
        sync_fn.assert_not_called()


# ---------------------------------------------------------------------------
# RerankerFactory
# ---------------------------------------------------------------------------


class TestRerankerFactory:
    """Factory creation and fallback."""

    def test_create_identity(self):
        r = RerankerFactory.create("identity")
        assert isinstance(r, IdentityReranker)

    def test_create_default_is_identity(self):
        r = RerankerFactory.create()
        assert isinstance(r, IdentityReranker)

    def test_create_unknown_type_fallback(self):
        r = RerankerFactory.create("nonexistent_type")
        assert isinstance(r, IdentityReranker)

    def test_create_llm(self):
        mock_fn = lambda x: "5"
        r = RerankerFactory.create("llm", llm_func=mock_fn)
        assert isinstance(r, LLMReranker)

    def test_create_cross_encoder_fallback_if_not_installed(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            r = RerankerFactory.create("cross_encoder")
            # Should fall back to IdentityReranker
            assert isinstance(r, IdentityReranker)

    def test_create_cross_encoder_with_mock(self):
        with patch(
            "hugegraph_llm.operators.graph_op.reranker.CrossEncoderReranker._try_load_model",
            return_value=True,
        ):
            r = RerankerFactory.create("cross_encoder")
            assert isinstance(r, CrossEncoderReranker)
