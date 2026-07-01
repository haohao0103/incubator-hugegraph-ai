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

"""Chunk schema helpers — multimodal markup stripping + sidecar normalization.

Adapted from LightRAG ``chunk_schema.py``. Three responsibilities:
1. ``normalize_sidecar()`` — validate and normalize a chunk's sidecar payload
2. ``strip_internal_multimodal_markup()`` — remove parser-internal identifiers
   from chunk content before sending to the entity-extraction LLM
3. ``format_heading_context()`` — join heading chain for extraction context

Only the entity-extraction prompt should receive the cleaned form; callers must
NOT mutate the stored chunk content so query-time citations still resolve.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Sidecar type whitelist
SIDECAR_TYPES = frozenset({"block", "drawing", "table", "equation"})

# Heading breadcrumb separator
HEADING_BREADCRUMB_SEP = " → "

# Regex patterns for multimodal markup stripping
_CITE_RE = re.compile(
    r"<cite\b[^>]*>(.*?)</cite>",
    flags=re.IGNORECASE | re.DOTALL,
)
_CITE_REFID_ATTR_RE = re.compile(
    r'\s+refid\s*=\s*"[^"]*"',
    flags=re.IGNORECASE,
)
_DRAWING_RE = re.compile(
    r"<drawing\b([^>]*)/>",
    flags=re.IGNORECASE,
)
_EQUATION_RE = re.compile(
    r"<equation\b([^>]*)>(.*?)</equation>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TABLE_RE = re.compile(
    r"<table\b([^>]*)>(.*?)</table>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ATTR_RE = re.compile(
    r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"',
)

# Unicode control/format categories to strip from headings
_HEADING_STRIP_CATEGORIES = frozenset({"Cc", "Cf"})
_HEADING_WHITESPACE_RE = re.compile(r"\s+")


def normalize_sidecar(dp: dict[str, Any]) -> dict[str, Any] | None:
    """Return the canonical sidecar dict or None when absent/invalid.

    Output shape::

        {"type": <block|drawing|table|equation>,
         "id":   <primary source id>,
         "refs": [{"type": ..., "id": ...}, ...]}
    """
    sidecar = dp.get("sidecar")
    if not isinstance(sidecar, dict):
        return None
    sidecar_type = str(sidecar.get("type") or "").strip()
    sidecar_id = str(sidecar.get("id") or "").strip()
    if sidecar_type not in SIDECAR_TYPES or not sidecar_id:
        return None

    refs_raw = sidecar.get("refs")
    refs: list[dict[str, str]] = []
    if isinstance(refs_raw, list):
        for entry in refs_raw:
            if not isinstance(entry, dict):
                continue
            ref_type = str(entry.get("type") or "").strip()
            ref_id = str(entry.get("id") or "").strip()
            if ref_type in SIDECAR_TYPES and ref_id:
                refs.append({"type": ref_type, "id": ref_id})
    if not refs:
        refs = [{"type": sidecar_type, "id": sidecar_id}]

    return {"type": sidecar_type, "id": sidecar_id, "refs": refs}


def _attrs_to_dict(attr_string: str) -> dict[str, str]:
    return {
        match.group(1).lower(): match.group(2)
        for match in _ATTR_RE.finditer(attr_string)
    }


def _format_attrs(pairs: list[tuple[str, str]]) -> str:
    return "".join(f' {k}="{v}"' for k, v in pairs if v)


def _replace_drawing(match: re.Match[str]) -> str:
    attrs = _attrs_to_dict(match.group(1))
    caption = attrs.get("caption", "")
    if not caption.strip():
        return ""
    return f"<drawing{_format_attrs([('caption', caption)])} />"


def _replace_equation(match: re.Match[str]) -> str:
    attrs = _attrs_to_dict(match.group(1))
    body = match.group(2)
    keep: list[tuple[str, str]] = []
    fmt = attrs.get("format", "")
    if fmt:
        keep.append(("format", fmt))
    caption = attrs.get("caption", "")
    if caption.strip():
        keep.append(("caption", caption))
    return f"<equation{_format_attrs(keep)}>{body}</equation>"


def _replace_table(match: re.Match[str]) -> str:
    attrs = _attrs_to_dict(match.group(1))
    body = match.group(2)
    keep: list[tuple[str, str]] = []
    fmt = attrs.get("format", "")
    if fmt:
        keep.append(("format", fmt))
    caption = attrs.get("caption", "")
    if caption.strip():
        keep.append(("caption", caption))
    return f"<table{_format_attrs(keep)}>{body}</table>"


def strip_internal_multimodal_markup(
    content: str, *, keep_cite_tag: bool = False
) -> str:
    """Strip parser-internal identifiers from chunk content string.

    Only the entity-extraction prompt should receive the cleaned form;
    callers must NOT mutate the stored chunk content.

    Transformations:
    - ``<drawing id="..." path="..." src="..." caption="Fig 1" />``
        → ``<drawing caption="Fig 1" />`` (drop entire tag when no caption)
    - ``<table id="..." format="json" caption="...">rows</table>``
        → ``<table format="json" caption="...">rows</table>``
    - ``<equation id="..." format="latex">...</equation>``
        → ``<equation format="latex">...</equation>``
    - Cite tag: ``<cite type="..." refid="...">text</cite>`` → ``text``
      (or ``<cite type="...">text</cite>`` when keep_cite_tag=True)
    """
    if not content:
        return content
    if keep_cite_tag:
        cleaned = _CITE_REFID_ATTR_RE.sub("", content)
    else:
        cleaned = _CITE_RE.sub(lambda m: m.group(1), content)
    cleaned = _DRAWING_RE.sub(_replace_drawing, cleaned)
    cleaned = _TABLE_RE.sub(_replace_table, cleaned)
    cleaned = _EQUATION_RE.sub(_replace_equation, cleaned)
    return cleaned


def _clean_heading_text(text: str) -> str:
    """Flatten a heading into one clean line for the LLM."""
    text = text.replace("→", " ")
    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch) not in _HEADING_STRIP_CATEGORIES
        or ch in "\t\n\r\f\v"
    )
    return _HEADING_WHITESPACE_RE.sub(" ", text).strip()


def format_heading_context(dp: dict[str, Any], *, max_heading_len: int = 80) -> str:
    """Join a chunk's heading chain into ``h1 → h2 → h3``."""
    heading = dp.get("heading")
    if isinstance(heading, dict):
        chain = list(heading.get("parent_headings") or [])
        h_text = str(heading.get("heading") or "").strip()
        if h_text:
            chain.append(h_text)
    elif heading:
        chain = [str(heading)]
        parents = dp.get("parent_headings") or []
        if isinstance(parents, list):
            chain = [str(p) for p in parents if str(p).strip()] + chain
    else:
        return ""

    cleaned = [_clean_heading_text(h) for h in chain if _clean_heading_text(h)]
    capped = [h if len(h) <= max_heading_len else h[:max_heading_len - 1] + "…" for h in cleaned]
    return HEADING_BREADCRUMB_SEP.join(capped)


class ChunkSchemaOperator:
    """HG-AI operator: strip multimodal markup from chunk content.

    Usage::

        op = ChunkSchemaOperator()
        result = op.run({"chunks": [{"content": "..."}, ...]})
        # result["chunks"] have cleaned content in "cleaned_content" field
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        chunks = context.get("chunks", [])
        keep_cite = context.get("keep_cite_tag", False)
        for chunk in chunks:
            if isinstance(chunk, dict) and "content" in chunk:
                chunk["cleaned_content"] = strip_internal_multimodal_markup(
                    chunk["content"], keep_cite_tag=keep_cite
                )
        return context
