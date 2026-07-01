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

"""Sidecar backfill — attach sidecar refs to chunks after splitting.

Adapted from LightRAG ``sidecar/backfill.py``. After a document has been parsed
into the sidecar format (blocks.jsonl exists) and then chunked, each chunk
carries a ``_source_span`` recording the half-open [start, end) char offsets of
its content within the merged text. This module maps those spans back to the
original content blocks and attaches a ``sidecar`` field to each chunk.

This is a HG-AI operator: call ``run(context)`` where context must contain
``chunks`` (list of dicts) and ``blocks_path`` (path to blocks.jsonl).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hugegraph_llm.operators.multimodal.chunk_schema import (
    SIDECAR_TYPES,
    normalize_sidecar,
)

_BLOCK_SEPARATOR = "\n\n"

_REPLACEMENT_CHAR = "\ufffd"


class ChunkBlockMatchError(Exception):
    """Raised when a chunk cannot be matched to any block in the source."""

    def __init__(
        self,
        chunk_order_index: int,
        chunk_preview: str,
        blocks_path: str,
    ) -> None:
        self.chunk_order_index = chunk_order_index
        self.chunk_preview = chunk_preview[:200]
        self.blocks_path = blocks_path
        super().__init__(
            f"chunk #{chunk_order_index} cannot match any block in {blocks_path}: "
            f"preview={chunk_preview[:80]}..."
        )


def _is_unlocatable(body: str) -> bool:
    """True when a chunk carries U+FFFD from a multi-byte token boundary split."""
    return _REPLACEMENT_CHAR in body


def _load_content_blocks(blocks_path: str) -> list[tuple[str, str]]:
    """Read ``type == "content"`` rows from a blocks.jsonl file in order."""
    blocks: list[tuple[str, str]] = []
    with Path(blocks_path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("type") != "content":
                continue
            content = obj.get("content", "")
            if not isinstance(content, str):
                continue
            blockid = str(obj.get("blockid") or "").strip()
            blocks.append((blockid, content))
    return blocks


def _build_block_spans(
    blocks: list[tuple[str, str]],
) -> tuple[str, list[tuple[int, int, str]]]:
    """Reconstruct the merged text and each block's char span."""
    spans: list[tuple[int, int, str]] = []
    parts: list[str] = []
    cursor = 0
    for blockid, content in blocks:
        if not content.strip():
            continue
        if parts:
            cursor += len(_BLOCK_SEPARATOR)
        start = cursor
        end = start + len(content)
        spans.append((start, end, blockid))
        parts.append(content)
        cursor = end
    return _BLOCK_SEPARATOR.join(parts), spans


def _normalize_text(text: str) -> str:
    """Whitespace-stripped form (every whitespace char removed)."""
    return "".join(text.split())


def _covered_blockids(
    spans: list[tuple[int, int, str]], o_start: int, o_end: int
) -> list[str]:
    """Blockids whose span overlaps [o_start, o_end), deduped."""
    covered: list[str] = []
    seen: set[str] = set()
    for start, end, blockid in spans:
        if start < o_end and o_start < end and blockid and blockid not in seen:
            seen.add(blockid)
            covered.append(blockid)
    return covered


def _chunk_source_span(
    chunk: dict[str, Any],
    merged: str,
) -> tuple[int, int] | None:
    """Resolve a chunk's _source_span to (start, end) char offsets."""
    span = chunk.get("_source_span")
    if not isinstance(span, dict):
        return None
    try:
        start = int(span["start"])
        end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end <= start or end > len(merged):
        return None
    body = chunk.get("content", "")
    if not isinstance(body, str):
        return None
    source_text = merged[start:end]
    if source_text != body and _normalize_text(source_text) != _normalize_text(body):
        return None
    return start, end


def backfill_chunk_sidecars(
    chunks: list[dict[str, Any]],
    blocks_path: str,
) -> list[dict[str, Any]]:
    """Attach a sidecar to each chunk via its _source_span, in place.

    Returns the same list (mutated in place for convenience).
    No-op when blocks_path is empty/unreadable.
    """
    if not blocks_path:
        return chunks

    try:
        blocks = _load_content_blocks(blocks_path)
    except OSError:
        return chunks

    merged, spans = _build_block_spans(blocks)
    if not spans:
        return chunks

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if normalize_sidecar(chunk) is not None:
            continue
        body = chunk.get("content", "")
        if not isinstance(body, str) or not body.strip():
            continue

        source_span = _chunk_source_span(chunk, merged)
        if source_span is None:
            if _is_unlocatable(body):
                continue
            raise ChunkBlockMatchError(
                chunk_order_index=int(chunk.get("chunk_order_index", -1)),
                chunk_preview=body,
                blocks_path=blocks_path,
            )

        o_start, o_end = source_span
        covered = _covered_blockids(spans, o_start, o_end)
        if not covered:
            raise ChunkBlockMatchError(
                chunk_order_index=int(chunk.get("chunk_order_index", -1)),
                chunk_preview=body,
                blocks_path=blocks_path,
            )

        chunk["sidecar"] = {
            "type": "block",
            "id": covered[0],
            "refs": [{"type": "block", "id": bid} for bid in covered],
        }

    return chunks


class SidecarBackfillOperator:
    """HG-AI operator for sidecar backfill.

    Usage::

        op = SidecarBackfillOperator()
        result = op.run({"chunks": [...], "blocks_path": "/path/to/blocks.jsonl"})
        # result["chunks"] now have sidecar fields
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        chunks = context.get("chunks", [])
        blocks_path = context.get("blocks_path", "")
        context["chunks"] = backfill_chunk_sidecars(chunks, blocks_path)
        return context
