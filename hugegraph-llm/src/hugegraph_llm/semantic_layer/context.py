# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The unit of context handed to a generating model.

Field names match neocarta's ``TableContext`` contract so an MCP client (or
an existing prompt template) written against that contract keeps working
when the backend is swapped from Neo4j to HugeGraph.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "ColumnContext",
    "TableContext",
    "RenderLevel",
    "FULL",
    "KEYS_ONLY",
    "NAME_ONLY",
    "estimate_tokens",
]

#: Detail levels, richest first. Lower levels are fallbacks used when the
#: token budget is tight -- see ``semantic_layer.budget``.
FULL = "full"
KEYS_ONLY = "keys_only"
NAME_ONLY = "name_only"

RenderLevel = str

#: Trust thresholds. A node below these is still returned, but flagged: an
#: agent that knows a definition is stale can ask, one that never sees it
#: will present it as fact.
LOW_CONFIDENCE = 0.5


@dataclass
class ColumnContext:
    """One column of a table."""

    name: str
    data_type: str = ""
    comment: str = ""
    #: "primary" / "foreign" / ""
    key_type: str = ""
    #: Referenced columns, as ``table.column`` -- the only join evidence a
    #: generator should trust.
    references: List[str] = field(default_factory=list)
    sample_values: List[str] = field(default_factory=list)
    is_time_dimension: bool = False
    confidence: float = 1.0

    def to_dict(self, level: RenderLevel = FULL) -> Dict[str, Any]:
        if level == NAME_ONLY:
            return {}
        if level == KEYS_ONLY:
            # Keys and join targets only: enough to write an ON clause. A
            # column that is neither is of no use at this level, so it is
            # dropped entirely rather than rendered as a bare name -- the
            # caller counts columns and must see that the view is reduced.
            if not self.key_type and not self.references:
                return {}
            out: Dict[str, Any] = {"name": self.name}
            if self.key_type:
                out["key_type"] = self.key_type
            if self.references:
                out["references"] = list(self.references)
            return out
        out: Dict[str, Any] = {"name": self.name}
        if self.data_type:
            out["data_type"] = self.data_type
        if self.comment:
            out["comment"] = self.comment
        if self.key_type:
            out["key_type"] = self.key_type
        if self.references:
            out["references"] = list(self.references)
        if self.is_time_dimension:
            out["is_time_dimension"] = True
        if self.sample_values:
            out["sample_values"] = list(self.sample_values)
        return out


@dataclass
class TableContext:
    """A table plus the columns the retriever decided are relevant."""

    table_name: str
    table_description: str = ""
    database_name: str = ""
    schema_name: str = ""
    columns: List[ColumnContext] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)

    # -- retrieval metadata (not part of the neocarta contract) --
    score: float = 0.0
    #: True when the table was recalled directly, not reached by expansion.
    is_seed: bool = False
    row_count: int = 0
    confidence: float = 1.0
    freshness_ts: Optional[int] = None
    #: Human-readable reasons this table was selected; surfaced for debugging
    #: and for the model to see why a table is in context.
    reasons: List[str] = field(default_factory=list)

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def is_stale(self) -> bool:
        """True when the definition should not be trusted without checking."""
        return self.confidence < LOW_CONFIDENCE

    def to_dict(self, level: RenderLevel = FULL) -> Dict[str, Any]:
        """Render for a prompt.

        ``num_columns`` always reports the *selected* column count for the
        level, so a model can tell that it is seeing a truncated view rather
        than assuming the table really has two columns.
        """
        columns = [c.to_dict(level) for c in self.columns]
        columns = [c for c in columns if c]
        out: Dict[str, Any] = {
            "table_name": self.table_name,
            "table_description": self.table_description,
            "database_name": self.database_name,
            "schema_name": self.schema_name,
            "columns": columns,
            "num_columns": len(columns),
            "primary_key": list(self.primary_key),
        }
        if self.is_stale:
            out["warning"] = (
                f"low confidence ({self.confidence:.2f}); verify before use"
            )
        if level == FULL and self.reasons:
            out["_reasons"] = list(self.reasons)
        return out

    def render(self, level: RenderLevel = FULL) -> str:
        """Compact text form, used for token estimation and for prompts."""
        data = self.to_dict(level)
        lines = [f"[{data['table_name']}]"]
        if data["table_description"]:
            lines.append(f"  {data['table_description']}")
        if data.get("warning"):
            lines.append(f"  ! {data['warning']}")
        for col in data["columns"]:
            bits = [col["name"]]
            if col.get("data_type"):
                bits.append(col["data_type"])
            if col.get("key_type"):
                bits.append(col["key_type"].upper())
            if col.get("comment"):
                bits.append(col["comment"])
            if col.get("references"):
                bits.append("-> " + ", ".join(col["references"]))
            lines.append("  - " + " ".join(bits))
        return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token count.

    Latin text averages ~4 characters per token; CJK is closer to one token
    per character. Using a flat ``len/4`` would badly undercount Chinese
    metadata, which is exactly what this stack carries (``订单表``), and an
    underestimated budget is how a prompt silently overflows.
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        if "一" <= ch <= "鿿":  # CJK Unified Ideographs
            cjk += 1
    return cjk + max(0, len(text) - cjk) // 4
