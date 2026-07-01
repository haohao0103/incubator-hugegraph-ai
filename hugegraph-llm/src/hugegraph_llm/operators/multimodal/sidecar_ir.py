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

"""Intermediate representation (IR) for sidecar document parsing.

Adapted from LightRAG sidecar/ir.py. Parser engines produce an IRDoc;
write_sidecar() turns it into the spec-compliant parsed/ directory.

Placeholder convention:
- {{TBL:k}} — table with placeholder_key k
- {{IMG:k}} — drawing with placeholder_key k
- {{EQ:k}}  — block-level equation
- {{EQI:k}} — inline equation (no id, not in equations.json)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class IRPosition:
    """Block-level position info.

    type values: "paraid" (docx), "bbox" (pdf), "heading" (md), "absolute" (text).
    """
    type: str
    anchor: Any = None
    range: Optional[List] = None
    charspan: Optional[List[int]] = None
    origin: Optional[str] = None

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.anchor is not None:
            out["anchor"] = self.anchor
        if self.range is not None:
            out["range"] = list(self.range)
        if self.charspan is not None:
            out["charspan"] = list(self.charspan)
        if self.origin is not None:
            out["origin"] = self.origin
        return out


@dataclass
class IRTable:
    """Table sidecar item.

    rows (preferred) or html describes the body. Format is chosen by
    which payload the adapter populated.
    """
    placeholder_key: str
    rows: Optional[List[List[str]]] = None
    html: Optional[str] = None
    num_rows: int = 0
    num_cols: int = 0
    caption: str = ""
    footnotes: List[str] = field(default_factory=list)
    table_header: Optional[Any] = None
    self_ref: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    body_override: Optional[str] = None


@dataclass
class IRDrawing:
    """Drawing (image) sidecar item. asset_ref points to an AssetSpec."""
    placeholder_key: str
    asset_ref: str
    fmt: str = ""
    caption: str = ""
    footnotes: List[str] = field(default_factory=list)
    src: str = ""
    self_ref: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    path_override: Optional[str] = None


@dataclass
class IREquation:
    """Equation sidecar item. is_block=False => inline, no id."""
    placeholder_key: str
    latex: str
    is_block: bool = True
    caption: str = ""
    footnotes: List[str] = field(default_factory=list)
    self_ref: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class IRBlock:
    """One content block with placeholder tokens."""
    content_template: str
    heading: str = ""
    level: int = 0
    parent_headings: List[str] = field(default_factory=list)
    session_type: str = "body"
    table_slice: str = "none"
    table_header: Optional[str] = None
    positions: List[IRPosition] = field(default_factory=list)
    tables: List[IRTable] = field(default_factory=list)
    drawings: List[IRDrawing] = field(default_factory=list)
    equations: List[IREquation] = field(default_factory=list)


@dataclass
class AssetSpec:
    """Describes one file that lands in assets directory.

    source may be:
    - Path to an existing file (writer copies it)
    - bytes payload (writer dumps it)
    - None (file already in place)
    """
    ref: str
    suggested_name: str
    source: Any = None  # Path | bytes | None


@dataclass
class IRDoc:
    """Top-level IR — input to write_sidecar()."""
    document_name: str
    document_format: str
    doc_title: str
    split_option: dict[str, Any] = field(default_factory=dict)
    blocks: List[IRBlock] = field(default_factory=list)
    assets: List[AssetSpec] = field(default_factory=list)
    bbox_attributes: Optional[dict[str, Any]] = None
