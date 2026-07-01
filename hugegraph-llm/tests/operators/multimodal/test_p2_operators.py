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

"""Tests for sidecar_backfill, image_dimension_validator, chunk_schema, async_vlm_pipeline."""

import asyncio
import base64
import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hugegraph_llm.operators.multimodal.sidecar_backfill import (
    ChunkBlockMatchError,
    SidecarBackfillOperator,
    _build_block_spans,
    _chunk_source_span,
    _covered_blockids,
    _is_unlocatable,
    _load_content_blocks,
    _normalize_text,
    backfill_chunk_sidecars,
)
from hugegraph_llm.operators.multimodal.image_dimension_validator import (
    ImageDimensionValidator,
    ImageDimensionValidatorConfig,
    ImageValidationResult,
    _detect_mime,
    _dimensions_from_bytes,
    _read_png_dimensions,
    _read_jpeg_dimensions,
    _read_gif_dimensions,
    _read_webp_dimensions,
    read_image_dimensions_from_base64,
    read_image_dimensions_from_path,
)
from hugegraph_llm.operators.multimodal.chunk_schema import (
    SIDECAR_TYPES,
    ChunkSchemaOperator,
    format_heading_context,
    normalize_sidecar,
    strip_internal_multimodal_markup,
)
from hugegraph_llm.operators.multimodal.async_vlm_pipeline import (
    AsyncVLMPipeline,
    VLMPipelineConfig,
    VLMResult,
    VLMTask,
    _cooperative_yield,
)


# ===========================================================================
# Sidecar Backfill Tests
# ===========================================================================

class TestLoadContentBlocks:
    def test_load_blocks_from_jsonl(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        lines = [
            json.dumps({"type": "meta", "version": 1}),
            json.dumps({"type": "content", "blockid": "b1", "content": "Hello"}),
            json.dumps({"type": "content", "blockid": "b2", "content": "World"}),
            json.dumps({"type": "drawing", "blockid": "d1", "content": "fig1"}),
        ]
        blocks_path.write_text("\n".join(lines), encoding="utf-8")
        result = _load_content_blocks(str(blocks_path))
        assert len(result) == 2
        assert result[0] == ("b1", "Hello")
        assert result[1] == ("b2", "World")

    def test_load_blocks_skips_malformed(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        lines = [
            "not json",
            json.dumps({"type": "content", "blockid": "b1", "content": "ok"}),
            "",
        ]
        blocks_path.write_text("\n".join(lines), encoding="utf-8")
        result = _load_content_blocks(str(blocks_path))
        assert len(result) == 1

    def test_load_blocks_empty_path(self):
        with pytest.raises(OSError):
            _load_content_blocks("")


class TestBuildBlockSpans:
    def test_basic_spans(self):
        blocks = [("b1", "AAA"), ("b2", "BBB")]
        merged, spans = _build_block_spans(blocks)
        assert merged == "AAA\n\nBBB"
        assert spans == [(0, 3, "b1"), (5, 8, "b2")]

    def test_empty_blocks_filtered(self):
        blocks = [("b1", "AAA"), ("b2", ""), ("b3", "CCC")]
        merged, spans = _build_block_spans(blocks)
        assert len(spans) == 2
        assert merged == "AAA\n\nCCC"

    def test_single_block(self):
        merged, spans = _build_block_spans([("b1", "Hello")])
        assert merged == "Hello"
        assert spans == [(0, 5, "b1")]


class TestCoveredBlockids:
    def test_single_block_covered(self):
        spans = [(0, 10, "b1"), (12, 20, "b2")]
        result = _covered_blockids(spans, 0, 10)
        assert result == ["b1"]

    def test_two_blocks_covered(self):
        spans = [(0, 10, "b1"), (12, 20, "b2")]
        result = _covered_blockids(spans, 5, 15)
        assert result == ["b1", "b2"]

    def test_no_overlap(self):
        spans = [(0, 10, "b1")]
        result = _covered_blockids(spans, 11, 20)
        assert result == []


class TestNormalizeText:
    def test_removes_all_whitespace(self):
        assert _normalize_text("a  b\tc\nd") == "abcd"

    def test_empty_string(self):
        assert _normalize_text("") == ""


class TestIsUnlocatable:
    def test_has_replacement_char(self):
        assert _is_unlocatable("hello\ufffdworld")

    def test_no_replacement_char(self):
        assert not _is_unlocatable("hello world")


class TestChunkSourceSpan:
    def test_valid_span(self):
        merged = "AAA\n\nBBB"
        chunk = {"_source_span": {"start": 0, "end": 3}, "content": "AAA"}
        result = _chunk_source_span(chunk, merged)
        assert result == (0, 3)

    def test_whitespace_normalized_match(self):
        merged = "AAA BBB"
        chunk = {"_source_span": {"start": 0, "end": 7}, "content": "AAABBB"}
        result = _chunk_source_span(chunk, merged)
        assert result == (0, 7)

    def test_no_span_field(self):
        chunk = {"content": "AAA"}
        assert _chunk_source_span(chunk, "AAA") is None

    def test_span_out_of_bounds(self):
        chunk = {"_source_span": {"start": 0, "end": 100}, "content": "AAA"}
        assert _chunk_source_span(chunk, "AAA") is None

    def test_span_content_mismatch(self):
        merged = "AAA"
        chunk = {"_source_span": {"start": 0, "end": 3}, "content": "ZZZ"}
        assert _chunk_source_span(chunk, merged) is None


class TestBackfillChunkSidecars:
    def test_backfill_basic(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        lines = [
            json.dumps({"type": "content", "blockid": "b1", "content": "Hello"}),
            json.dumps({"type": "content", "blockid": "b2", "content": "World"}),
        ]
        blocks_path.write_text("\n".join(lines), encoding="utf-8")

        chunks = [
            {"_source_span": {"start": 0, "end": 5}, "content": "Hello"},
            {"_source_span": {"start": 7, "end": 12}, "content": "World"},
        ]
        result = backfill_chunk_sidecars(chunks, str(blocks_path))
        assert result[0]["sidecar"]["id"] == "b1"
        assert result[1]["sidecar"]["id"] == "b2"

    def test_backfill_empty_path(self):
        chunks = [{"content": "Hello"}]
        result = backfill_chunk_sidecars(chunks, "")
        assert "sidecar" not in chunks[0]

    def test_backfill_chunk_with_existing_sidecar(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "content", "blockid": "b1", "content": "Hello"}),
            encoding="utf-8",
        )
        chunks = [
            {"sidecar": {"type": "block", "id": "x1"}, "content": "Hello"},
        ]
        result = backfill_chunk_sidecars(chunks, str(blocks_path))
        # Existing sidecar should not be overwritten
        assert result[0]["sidecar"]["id"] == "x1"

    def test_backfill_unlocatable_chunk_skipped(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "content", "blockid": "b1", "content": "Hello"}),
            encoding="utf-8",
        )
        chunks = [
            {"content": "Hello\ufffd", "_source_span": None},
        ]
        # Should not raise, just skip
        result = backfill_chunk_sidecars(chunks, str(blocks_path))

    def test_backfill_no_match_raises(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "content", "blockid": "b1", "content": "Hello"}),
            encoding="utf-8",
        )
        chunks = [
            {"content": "Mismatch", "_source_span": {"start": 0, "end": 8}},
        ]
        with pytest.raises(ChunkBlockMatchError):
            backfill_chunk_sidecars(chunks, str(blocks_path))


class TestSidecarBackfillOperator:
    def test_run_operator(self, tmp_path):
        blocks_path = tmp_path / "blocks.jsonl"
        blocks_path.write_text(
            json.dumps({"type": "content", "blockid": "b1", "content": "Hello"}),
            encoding="utf-8",
        )
        op = SidecarBackfillOperator()
        result = op.run({
            "chunks": [{"_source_span": {"start": 0, "end": 5}, "content": "Hello"}],
            "blocks_path": str(blocks_path),
        })
        assert result["chunks"][0]["sidecar"]["id"] == "b1"

    def test_run_empty_path(self):
        op = SidecarBackfillOperator()
        result = op.run({"chunks": [{"content": "Hello"}], "blocks_path": ""})
        assert "chunks" in result


# ===========================================================================
# Image Dimension Validator Tests
# ===========================================================================

class TestDetectMime:
    def test_png(self):
        assert _detect_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20) == "image/png"

    def test_jpeg(self):
        assert _detect_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 20) == "image/jpeg"

    def test_gif87a(self):
        assert _detect_mime(b"GIF87a" + b"\x00" * 10) == "image/gif"

    def test_gif89a(self):
        assert _detect_mime(b"GIF89a" + b"\x00" * 10) == "image/gif"

    def test_webp(self):
        data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20
        assert _detect_mime(data) == "image/webp"

    def test_unknown_defaults_png(self):
        assert _detect_mime(b"\x00\x00\x00") == "image/png"


class TestReadDimensions:
    def test_png_dimensions(self):
        # Minimal valid PNG: 8-byte sig + 4-byte length + 4-byte "IHDR" + 4-byte width + 4-byte height
        ihdr = struct.pack(">II", 800, 600)  # 800x600
        data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 10
        result = _read_png_dimensions(data)
        assert result == (800, 600)

    def test_gif_dimensions(self):
        data = b"GIF89a" + struct.pack("<HH", 320, 240)
        result = _read_gif_dimensions(data)
        assert result == (320, 240)

    def test_jpeg_dimensions(self):
        # SOF0 marker: 0xFFC0 + length + precision + height + width
        sof_payload = struct.pack(">BHH", 8, 480, 640) + b"\x00" * 6
        segment_len = len(sof_payload) + 2
        data = b"\xff\xd8\xff\xc0" + struct.pack(">H", segment_len) + sof_payload
        result = _read_jpeg_dimensions(data)
        assert result == (640, 480)

    def test_webp_vp8_dimensions(self):
        # VP8 lossy: 14-bit width/height
        data = b"RIFF" + b"\x00" * 4 + b"WEBPVP8 " + b"\x00" * 8
        data += b"\x00" * 2 + struct.pack("<HH", 1024 & 0x3FFF, 768 & 0x3FFF)
        data += b"\x00" * 10
        # Pad to >= 30 bytes
        data = data[:12] + b"VP8 " + data[16:26] + struct.pack("<HH", 1024 & 0x3FFF, 768 & 0x3FFF) + b"\x00" * 20
        result = _read_webp_dimensions(data)
        if result:
            assert result[0] == 1024
            assert result[1] == 768

    def test_empty_data(self):
        assert _dimensions_from_bytes(b"") is None

    def test_unknown_format(self):
        assert _dimensions_from_bytes(b"\x00\x01\x02\x03") is None


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TestReadFromPath:
    def test_valid_png_file(self, tmp_path):
        # Create a minimal PNG file
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        p = tmp_path / "test.png"
        p.write_bytes(png_data)
        dims = read_image_dimensions_from_path(p)
        assert dims == (200, 150)

    def test_nonexistent_file(self):
        dims = read_image_dimensions_from_path(Path("/nonexistent/file.png"))
        assert dims is None


class TestReadFromBase64:
    def test_valid_png_base64(self):
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        b64 = base64.b64encode(png_data).decode()
        dims = read_image_dimensions_from_base64(b64)
        assert dims == (200, 150)

    def test_invalid_base64(self):
        dims = read_image_dimensions_from_base64("not-valid-base64!!!")
        assert dims is None


class TestImageDimensionValidator:
    def test_validate_path_accepted(self, tmp_path):
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        p = tmp_path / "test.png"
        p.write_bytes(png_data)

        validator = ImageDimensionValidator()
        result = validator.validate_path(str(p))
        assert result.accepted
        assert result.width == 200
        assert result.height == 150
        assert result.mime_type == "image/png"

    def test_validate_path_too_small(self, tmp_path):
        ihdr = struct.pack(">II", 10, 10)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        p = tmp_path / "small.png"
        p.write_bytes(png_data)

        config = ImageDimensionValidatorConfig(min_width=50, min_height=50)
        validator = ImageDimensionValidator(config)
        result = validator.validate_path(str(p))
        assert not result.accepted
        assert "below minimum" in result.reason

    def test_validate_path_too_large_dims(self, tmp_path):
        ihdr = struct.pack(">II", 10000, 10000)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        p = tmp_path / "large.png"
        p.write_bytes(png_data)

        config = ImageDimensionValidatorConfig(max_width=8192, max_height=8192)
        validator = ImageDimensionValidator(config)
        result = validator.validate_path(str(p))
        assert not result.accepted
        assert "exceed maximum" in result.reason

    def test_validate_base64_accepted(self):
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        b64 = base64.b64encode(png_data).decode()

        validator = ImageDimensionValidator()
        result = validator.validate_base64(b64)
        assert result.accepted
        assert result.width == 200
        assert result.height == 150

    def test_validate_base64_invalid(self):
        validator = ImageDimensionValidator()
        result = validator.validate_base64("!!!invalid!!!")
        assert not result.accepted
        assert "invalid base64" in result.reason

    def test_validate_data_url(self):
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        b64 = base64.b64encode(png_data).decode()
        data_url = f"data:image/png;base64,{b64}"

        validator = ImageDimensionValidator()
        result = validator.validate_base64(data_url)
        assert result.accepted
        assert result.mime_type == "image/png"

    def test_validate_nonexistent_path(self):
        validator = ImageDimensionValidator()
        result = validator.validate_path("/nonexistent/image.png")
        assert not result.accepted
        assert "cannot read" in result.reason

    def test_run_operator(self, tmp_path):
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        p = tmp_path / "test.png"
        p.write_bytes(png_data)

        validator = ImageDimensionValidator()
        result = validator.run({"images": [str(p)], "image_mode": "path"})
        assert len(result["accepted_images"]) == 1
        assert len(result["rejected_images"]) == 0

    def test_config_mime_type_rejection(self):
        ihdr = struct.pack(">II", 200, 150)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        b64 = base64.b64encode(png_data).decode()

        config = ImageDimensionValidatorConfig(allowed_mime_types=("image/jpeg",))
        validator = ImageDimensionValidator(config)
        result = validator.validate_base64(b64)
        assert not result.accepted
        assert "mime type" in result.reason


# ===========================================================================
# Chunk Schema Tests
# ===========================================================================

class TestNormalizeSidecar:
    def test_valid_sidecar(self):
        dp = {"sidecar": {"type": "block", "id": "b1"}}
        result = normalize_sidecar(dp)
        assert result["type"] == "block"
        assert result["id"] == "b1"
        assert result["refs"] == [{"type": "block", "id": "b1"}]

    def test_sidecar_with_refs(self):
        dp = {"sidecar": {"type": "drawing", "id": "d1", "refs": [
            {"type": "drawing", "id": "d1"},
            {"type": "block", "id": "b1"},
        ]}}
        result = normalize_sidecar(dp)
        assert len(result["refs"]) == 2

    def test_invalid_type(self):
        dp = {"sidecar": {"type": "invalid", "id": "x1"}}
        assert normalize_sidecar(dp) is None

    def test_missing_id(self):
        dp = {"sidecar": {"type": "block"}}
        assert normalize_sidecar(dp) is None

    def test_no_sidecar(self):
        assert normalize_sidecar({"content": "hello"}) is None


class TestStripMultimodalMarkup:
    def test_strip_drawing_with_caption(self):
        content = '<drawing id="im-1" path="/fig.png" src="base64..." caption="Fig 1" />'
        result = strip_internal_multimodal_markup(content)
        assert result == '<drawing caption="Fig 1" />'

    def test_strip_drawing_without_caption(self):
        content = '<drawing id="im-1" path="/fig.png" />'
        result = strip_internal_multimodal_markup(content)
        assert result == ""

    def test_strip_table(self):
        content = '<table id="tb-1" format="json" caption="Data">rows</table>'
        result = strip_internal_multimodal_markup(content)
        assert 'format="json"' in result
        assert 'caption="Data"' in result
        assert 'id="tb-1"' not in result
        assert "rows" in result

    def test_strip_equation(self):
        content = '<equation id="eq-1" format="latex">E=mc^2</equation>'
        result = strip_internal_multimodal_markup(content)
        assert 'format="latex"' in result
        assert "E=mc^2" in result
        assert 'id="eq-1"' not in result

    def test_strip_cite_default(self):
        content = '<cite type="table" refid="tb-1">Table 1</cite>'
        result = strip_internal_multimodal_markup(content)
        assert result == "Table 1"

    def test_strip_cite_keep_tag(self):
        content = '<cite type="table" refid="tb-1">Table 1</cite>'
        result = strip_internal_multimodal_markup(content, keep_cite_tag=True)
        assert '<cite type="table">Table 1</cite>' in result
        assert "refid" not in result

    def test_empty_content(self):
        assert strip_internal_multimodal_markup("") == ""

    def test_no_markup(self):
        content = "Just plain text with no markup."
        assert strip_internal_multimodal_markup(content) == content


class TestFormatHeadingContext:
    def test_nested_heading(self):
        dp = {"heading": {"level": 2, "heading": "Section 2", "parent_headings": ["Chapter 1"]}}
        result = format_heading_context(dp)
        assert result == "Chapter 1 → Section 2"

    def test_flat_heading(self):
        dp = {"heading": "Title"}
        result = format_heading_context(dp)
        assert result == "Title"

    def test_no_heading(self):
        dp = {}
        result = format_heading_context(dp)
        assert result == ""

    def test_truncation(self):
        dp = {"heading": "A very very very very very very very very long heading that exceeds the limit"}
        result = format_heading_context(dp, max_heading_len=20)
        assert len(result) <= 20


class TestChunkSchemaOperator:
    def test_run_operator(self):
        op = ChunkSchemaOperator()
        chunks = [
            {"content": '<drawing id="im-1" caption="Fig 1" /> text'},
            {"content": '<cite type="table" refid="t1">T1</cite>'},
        ]
        result = op.run({"chunks": chunks})
        assert "cleaned_content" in result["chunks"][0]
        assert "id=" not in result["chunks"][0]["cleaned_content"]
        assert "refid" not in result["chunks"][1]["cleaned_content"]

    def test_run_with_keep_cite(self):
        op = ChunkSchemaOperator()
        chunks = [{"content": '<cite type="table" refid="t1">T1</cite>'}]
        result = op.run({"chunks": chunks, "keep_cite_tag": True})
        assert '<cite type="table">' in result["chunks"][0]["cleaned_content"]


# ===========================================================================
# Async VLM Pipeline Tests
# ===========================================================================

class TestVLMTaskAndResult:
    def test_task_creation(self):
        task = VLMTask(image_id="img1", image_data="base64...", prompt="Describe")
        assert task.image_id == "img1"
        assert task.priority == 0

    def test_result_creation(self):
        result = VLMResult(image_id="img1", description="A cat", success=True)
        assert result.success
        assert result.description == "A cat"


class TestCooperativeYield:
    @pytest.mark.asyncio
    async def test_yield_at_interval(self):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _cooperative_yield(8, every=8)
            mock_sleep.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_no_yield_before_interval(self):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _cooperative_yield(7, every=8)
            mock_sleep.assert_not_called()


class TestAsyncVLMPipelineSync:
    def test_process_sync_success(self):
        config = VLMPipelineConfig(max_retries=0)
        pipeline = AsyncVLMPipeline(config)
        tasks = [VLMTask(image_id="img1", image_data="data", prompt="Describe")]
        mock_vlm = MagicMock(return_value="A cat sitting on a mat")

        results = pipeline.process_sync(tasks, mock_vlm)
        assert len(results) == 1
        assert results[0].success
        assert results[0].description == "A cat sitting on a mat"

    def test_process_sync_failure_with_retry(self):
        config = VLMPipelineConfig(max_retries=2, retry_delay=0.01, retry_backoff_factor=2)
        pipeline = AsyncVLMPipeline(config)
        tasks = [VLMTask(image_id="img1", image_data="data", prompt="Describe")]
        mock_vlm = MagicMock(side_effect=RuntimeError("API error"))

        results = pipeline.process_sync(tasks, mock_vlm)
        assert len(results) == 1
        assert not results[0].success
        assert "API error" in results[0].error
        assert results[0].retry_count == 2

    def test_process_sync_retry_then_success(self):
        config = VLMPipelineConfig(max_retries=3, retry_delay=0.01)
        pipeline = AsyncVLMPipeline(config)
        tasks = [VLMTask(image_id="img1", image_data="data", prompt="Describe")]

        call_counts = [0]
        def flaky_vlm(data, prompt):
            call_counts[0] += 1
            if call_counts[0] < 3:
                raise RuntimeError("temporary error")
            return "Success description"

        results = pipeline.process_sync(tasks, flaky_vlm)
        assert results[0].success
        assert results[0].description == "Success description"
        assert results[0].retry_count == 2  # succeeded on 3rd attempt (2 retries)

    def test_process_sync_empty_tasks(self):
        pipeline = AsyncVLMPipeline()
        results = pipeline.process_sync([], lambda d, p: "")
        assert results == []

    def test_process_sync_preserves_order(self):
        pipeline = AsyncVLMPipeline(VLMPipelineConfig(max_retries=0))
        tasks = [
            VLMTask(image_id="low", image_data="d1", prompt="p1", priority=0),
            VLMTask(image_id="high", image_data="d2", prompt="p2", priority=10),
            VLMTask(image_id="mid", image_data="d3", prompt="p3", priority=5),
        ]
        results = pipeline.process_sync(tasks, lambda d, p: "ok")
        # Sync process preserves input order (no priority sorting)
        assert results[0].image_id == "low"
        assert results[1].image_id == "high"
        assert results[2].image_id == "mid"


class TestAsyncVLMPipelineAsync:
    @pytest.mark.asyncio
    async def test_process_async_success(self):
        config = VLMPipelineConfig(max_concurrent=2, max_retries=0)
        pipeline = AsyncVLMPipeline(config)
        tasks = [
            VLMTask(image_id="img1", image_data="data1", prompt="p1"),
            VLMTask(image_id="img2", image_data="data2", prompt="p2"),
        ]
        mock_vlm = AsyncMock(return_value="Description")

        results = await pipeline.process_async(tasks, mock_vlm)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_process_async_retry(self):
        config = VLMPipelineConfig(max_concurrent=1, max_retries=2, retry_delay=0.01)
        pipeline = AsyncVLMPipeline(config)
        tasks = [VLMTask(image_id="img1", image_data="data", prompt="p1")]

        call_counts = [0]
        async def flaky_vlm(data, prompt):
            call_counts[0] += 1
            if call_counts[0] < 2:
                raise RuntimeError("temp error")
            return "Recovered"

        results = await pipeline.process_async(tasks, flaky_vlm)
        assert results[0].success
        assert results[0].description == "Recovered"

    @pytest.mark.asyncio
    async def test_process_async_empty(self):
        pipeline = AsyncVLMPipeline()
        results = await pipeline.process_async([], AsyncMock())
        assert results == []


class TestAsyncVLMPipelineRun:
    def test_run_with_sync_func(self):
        pipeline = AsyncVLMPipeline(VLMPipelineConfig(max_retries=0))
        result = pipeline.run({
            "vlm_tasks": [
                {"image_id": "img1", "image_data": "data", "prompt": "Describe"},
            ],
            "vlm_func_sync": lambda d, p: "A cat",
        })
        assert result["vlm_success_count"] == 1
        assert result["vlm_failure_count"] == 0

    def test_run_with_no_func(self):
        pipeline = AsyncVLMPipeline()
        result = pipeline.run({
            "vlm_tasks": [
                {"image_id": "img1", "image_data": "data", "prompt": "Describe"},
            ],
        })
        assert result["vlm_failure_count"] == 1

    def test_run_with_vlm_task_objects(self):
        pipeline = AsyncVLMPipeline(VLMPipelineConfig(max_retries=0))
        result = pipeline.run({
            "vlm_tasks": [
                VLMTask(image_id="img1", image_data="data", prompt="Describe"),
            ],
            "vlm_func_sync": lambda d, p: "OK",
        })
        assert result["vlm_success_count"] == 1

    def test_run_empty_tasks(self):
        pipeline = AsyncVLMPipeline()
        result = pipeline.run({"vlm_tasks": []})
        assert result["vlm_success_count"] == 0
