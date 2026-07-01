# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Surrounding context enrichment for multimodal sidecar analysis.

Adapted from LightRAG multimodal_context.py. For each sidecar entry,
this module builds leading/trailing text from the same block row,
truncated to a token budget, to provide context for VLM analysis.

Operator protocol: run(context) -> context
  Enriches context["multimodal_sidecars"] items with "surrounding" field.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

DEFAULT_SURROUNDING_MAX_TOKENS = 2000

# Default separator cascade (adapted from LightRAG)
DEFAULT_R_SEPARATORS = [
    "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "",
]

# Multimodal tag regex (atomic units for truncation)
_MM_TAG_RE = re.compile(
    r"<drawing\b[^>]*/>"
    r"|<table\b[^>]*>.*?</table>"
    r"|<equation\b[^>]*>.*?</equation>",
    re.DOTALL,
)


# ============================================================================
# Public: build_surrounding
# ============================================================================

def build_surrounding(
    *,
    kind: str,
    block_content: str,
    target_start: int,
    target_end: int,
    max_tokens: int = DEFAULT_SURROUNDING_MAX_TOKENS,
    separators: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Compute {"leading": ..., "trailing": ...} for one sidecar entry.

    Args:
        kind: "drawings" / "tables" / "equations"
        block_content: Full block text containing the target tag.
        target_start: Char offset of the target tag start.
        target_end: Char offset of the target tag end.
        max_tokens: Per-half token budget.
        separators: Recursive separator cascade for truncation.

    Returns:
        {"leading": str, "trailing": str}
    """
    if not block_content or max_tokens <= 0:
        return {"leading": "", "trailing": ""}

    separators = separators or DEFAULT_R_SEPARATORS
    leading_src = block_content[:target_start]
    trailing_src = block_content[target_end:]

    # For tables kind, strip other table tags from surrounding
    if kind == "tables":
        leading_src = _remove_table_tags(leading_src)
        trailing_src = _remove_table_tags(trailing_src)

    # Strip parser-internal markers (id, path, src, refid)
    leading_src = _strip_internal_markers(leading_src)
    trailing_src = _strip_internal_markers(trailing_src)

    leading = _build_leading_text(leading_src, max_tokens, separators)
    trailing = _build_trailing_text(trailing_src, max_tokens, separators)

    return {"leading": leading, "trailing": trailing}


# ============================================================================
# SurroundingContextEnricher Operator
# ============================================================================

class SurroundingContextEnricher:
    """HG-AI operator that enriches sidecar items with surrounding context.

    Reads blocks_content_by_id from context, finds each sidecar item's
    target tag span, and adds "surrounding" field with leading/trailing text.

    Usage:
        enricher = SurroundingContextEnricher(max_tokens=2000)
        context = enricher.run(context)
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_SURROUNDING_MAX_TOKENS,
        separators: Optional[List[str]] = None,
    ):
        self.max_tokens = max_tokens
        self.separators = separators or DEFAULT_R_SEPARATORS

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich multimodal sidecar items with surrounding context.

        Args:
            context: Must contain "multimodal_sidecars" dict and
                     "blocks_content_by_id" dict.

        Returns:
            context with enriched sidecar items.
        """
        sidecars = context.get("multimodal_sidecars", {})
        blocks_content = context.get("blocks_content_by_id", {})

        counts = {"drawings": 0, "tables": 0, "equations": 0}

        for root_key in ("drawings", "tables", "equations"):
            items = sidecars.get(root_key, {})
            if not items:
                continue

            for item_id, item in items.items():
                if not isinstance(item, dict):
                    continue

                blockid = item.get("blockid")
                if not blockid:
                    continue

                block_content = blocks_content.get(blockid)
                if not block_content:
                    continue

                # Find target tag span
                span = _find_target_span(root_key, item_id, block_content)
                if span is None:
                    log.debug("%s/%s: id not found in block %s",
                              root_key, item_id, blockid)
                    continue

                surrounding = build_surrounding(
                    kind=root_key,
                    block_content=block_content,
                    target_start=span[0],
                    target_end=span[1],
                    max_tokens=self.max_tokens,
                    separators=self.separators,
                )
                item["surrounding"] = surrounding
                counts[root_key] += 1

        context["surrounding_enrichment_counts"] = counts
        return context


# ============================================================================
# Target tag locators
# ============================================================================

def _find_target_span(
    kind: str, item_id: str, block_content: str,
) -> Optional[Tuple[int, int]]:
    """Locate the target multimodal marker with given id in block_content."""
    esc = re.escape(item_id)
    if kind == "drawings":
        pattern = re.compile(
            rf'<drawing\b[^>]*?\bid\s*=\s*"{esc}"[^>]*?/>', re.DOTALL)
    elif kind == "tables":
        pattern = re.compile(
            rf'<table\b[^>]*?\bid\s*=\s*"{esc}"[^>]*?>.*?</table>', re.DOTALL)
    elif kind == "equations":
        pattern = re.compile(
            rf'<equation\b[^>]*?\bid\s*=\s*"{esc}"[^>]*?>.*?</equation>', re.DOTALL)
    else:
        return None

    match = pattern.search(block_content)
    if not match:
        return None
    return match.start(), match.end()


# ============================================================================
# Internal helpers for text truncation
# ============================================================================

def _remove_table_tags(text: str) -> str:
    """Strip all table tags from text (for tables kind surrounding)."""
    return re.sub(r"<table\b[^>]*>.*?</table>", "", text, flags=re.DOTALL)


_INTERNAL_ATTR_RE = re.compile(
    r'\b(id|path|src|refid)\s*=\s*"[^"]*"',
)


def _strip_internal_markers(text: str) -> str:
    """Strip parser-internal id/path/src/refid attributes."""
    return _INTERNAL_ATTR_RE.sub("", text)


def _atomize(text: str) -> List[Tuple[str, str]]:
    """Split text into (kind, content) atoms.

    kind ∈ {"text", "drawing", "equation", "table"}.
    """
    atoms: List[Tuple[str, str]] = []
    pos = 0
    for match in _MM_TAG_RE.finditer(text):
        if match.start() > pos:
            atoms.append(("text", text[pos:match.start()]))
        tag_text = match.group(0)
        if tag_text.startswith("<drawing"):
            kind = "drawing"
        elif tag_text.startswith("<table"):
            kind = "table"
        else:
            kind = "equation"
        atoms.append((kind, tag_text))
        pos = match.end()
    if pos < len(text):
        atoms.append(("text", text[pos:]))
    return atoms


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English, ~2 for CJK."""
    if not text:
        return 0
    # Quick estimate: chars / 4 (rough tokenization)
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f]', text))
    non_cjk_len = len(text) - cjk_count
    return int(non_cjk_len / 4 + cjk_count * 1.5) + 1


def _split_text_segment(text: str, separators: List[str]) -> Tuple[List[str], int]:
    """Split text using first separator that produces >1 pieces."""
    if not text:
        return [text], len(separators)
    for idx, sep in enumerate(separators):
        if not sep:
            continue
        if sep in text:
            parts = text.split(sep)
            assembled: List[str] = []
            for j, part in enumerate(parts):
                if j < len(parts) - 1:
                    assembled.append(part + sep)
                else:
                    if part:
                        assembled.append(part)
            if len(assembled) > 1:
                return assembled, idx
    return [text], len(separators)


def _build_leading_text(
    source: str,
    max_tokens: int,
    separators: List[str],
) -> str:
    """Build leading half: suffix of source within budget."""
    if not source or max_tokens <= 0:
        return ""
    atoms = _atomize(source)
    accumulated = ""

    for atom_idx in range(len(atoms) - 1, -1, -1):
        atom_kind, atom_text = atoms[atom_idx]
        if not atom_text:
            continue
        candidate = atom_text + accumulated
        if _estimate_tokens(candidate) <= max_tokens:
            accumulated = candidate
            continue

        if atom_kind in {"drawing", "equation", "table"}:
            # Cannot split atomic tags; stop
            break

        # Plain text: try to add suffix of segments
        segments, sep_idx = _split_text_segment(atom_text, separators)
        buf = ""
        for i in range(len(segments) - 1, -1, -1):
            candidate = segments[i] + buf + accumulated
            if _estimate_tokens(candidate) <= max_tokens:
                buf = segments[i] + buf
                continue
            if buf:
                break  # Already added some segments
            # Char-level fallback: take longest suffix that fits
            remaining = max_tokens - _estimate_tokens(accumulated)
            if remaining <= 0:
                break
            # Simple char trim from head
            char_budget = int(remaining * 4)  # rough chars from tokens
            trimmed = segments[i][-char_budget:] if char_budget < len(segments[i]) else segments[i]
            buf = trimmed
            break
        accumulated = buf + accumulated
        break

    return accumulated


def _build_trailing_text(
    source: str,
    max_tokens: int,
    separators: List[str],
) -> str:
    """Build trailing half: prefix of source within budget."""
    if not source or max_tokens <= 0:
        return ""
    atoms = _atomize(source)
    accumulated = ""

    for atom_kind, atom_text in atoms:
        if not atom_text:
            continue
        candidate = accumulated + atom_text
        if _estimate_tokens(candidate) <= max_tokens:
            accumulated = candidate
            continue

        if atom_kind in {"drawing", "equation", "table"}:
            break

        segments, sep_idx = _split_text_segment(atom_text, separators)
        buf = ""
        for seg in segments:
            candidate = accumulated + buf + seg
            if _estimate_tokens(candidate) <= max_tokens:
                buf = buf + seg
                continue
            if buf:
                break
            remaining = max_tokens - _estimate_tokens(accumulated)
            if remaining <= 0:
                break
            char_budget = int(remaining * 4)
            trimmed = seg[:char_budget] if char_budget < len(seg) else seg
            buf = trimmed
            break
        accumulated = accumulated + buf
        break

    return accumulated
