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

"""Find a join path between two tables.

This is the tool that attacks hallucinated JOINs at the source: instead of
asking a model to invent an ``ON`` clause, the agent asks for the path and
gets real column pairs plus a flag saying whether each step is *declared*
integrity or merely inferred.

**Only ``REFERENCES`` edges are traversed.** ``LINEAGE`` and ``CO_OCCUR``
say two tables are related, but they do not name the columns, so a path
built from them cannot be rendered as SQL. Returning such a path would be
worse than returning nothing: it looks authoritative and is unusable. When
no column-level path exists the caller is told so, and can decide to ask the
user instead of guessing.

Costs follow :data:`~semantic_layer.enums.EDGE_JOIN_COST`, with an extra
penalty for ``proven=false`` so a declared foreign key always wins over an
inferred ``*_id`` match of the same length.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hugegraph_llm.semantic_layer.readers import SemanticProjection

__all__ = [
    "JoinStep",
    "JoinPath",
    "find_join_path",
    "render_join_path",
    "UNPROVEN_PENALTY",
]

#: Extra cost for an inferred (``proven=false``) foreign key.
UNPROVEN_PENALTY = 4.0

#: Cost of one declared foreign-key hop.
PROVEN_COST = 1.0


@dataclass
class JoinStep:
    """One hop: ``left_column = right_column``."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    #: True when a declared relationship backs this step.
    proven: bool = False

    @property
    def left_qualified(self) -> str:
        return f"{self.left_table}.{self.left_column}"

    @property
    def right_qualified(self) -> str:
        return f"{self.right_table}.{self.right_column}"

    def to_sql(self) -> str:
        """Render the ON condition, or a comment when not proven.

        An unproven join is never rendered as a real condition -- a silent
        guess here becomes a wrong answer downstream, whereas a comment
        prompts the caller to confirm.
        """
        if self.proven:
            return f"{self.left_qualified} = {self.right_qualified}"
        return (
            f"/* unproven join: {self.left_qualified} <-> "
            f"{self.right_qualified} */"
        )


@dataclass
class JoinPath:
    """An ordered sequence of hops from ``left`` to ``right``."""

    left: str
    right: str
    steps: List[JoinStep] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.steps)

    @property
    def all_proven(self) -> bool:
        return all(step.proven for step in self.steps)

    @property
    def tables(self) -> List[str]:
        if not self.steps:
            return [self.left] if self.left == self.right else []
        return [self.left] + [s.right_table for s in self.steps]

    def to_sql(self) -> str:
        """Render every step; unproven ones become comments."""
        return "\n".join(step.to_sql() for step in self.steps)

    def to_dict(self) -> Dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "found": self.found,
            "all_proven": self.all_proven,
            "tables": self.tables,
            "steps": [
                {
                    "left": s.left_qualified,
                    "right": s.right_qualified,
                    "proven": s.proven,
                    "sql": s.to_sql(),
                }
                for s in self.steps
            ],
        }


def _column_edges(proj: SemanticProjection) -> Dict[str, List[Tuple[str, str, str, bool]]]:
    """Symmetric table adjacency carrying the columns that join them.

    Returns ``table -> [(other_table, from_col, to_col, proven), ...]``.
    """
    edges: Dict[str, List[Tuple[str, str, str, bool]]] = {}
    for source, targets in proj.references.items():
        left_table, left_col = source.split(".", 1)
        for target in targets:
            right_table, right_col = target.split(".", 1)
            if left_table == right_table:
                continue
            proven = proj.reference_proven.get((source, target), False)
            edges.setdefault(left_table, []).append(
                (right_table, source, target, proven)
            )
            edges.setdefault(right_table, []).append(
                (left_table, target, source, proven)
            )
    return edges


def find_join_path(
    proj: SemanticProjection, left: str, right: str
) -> JoinPath:
    """Shortest weighted path between two tables, preferring proven steps.

    Dijkstra with a small vertex count, so a linear-scan queue is plenty and
    avoids depending on an external graph engine for a per-query call.
    """
    path = JoinPath(left=left, right=right)
    if left not in proj.tables or right not in proj.tables:
        return path
    if left == right:
        return path

    edges = _column_edges(proj)
    dist: Dict[str, float] = {left: 0.0}
    prev: Dict[str, Tuple[str, JoinStep]] = {}
    visited: set = set()

    while True:
        current = None
        best = float("inf")
        for table, d in dist.items():
            if table not in visited and d < best:
                current, best = table, d
        if current is None:
            break
        if current == right:
            break
        visited.add(current)
        for other, from_col, to_col, proven in edges.get(current, []):
            cost = PROVEN_COST if proven else UNPROVEN_PENALTY
            candidate = dist[current] + cost
            if candidate < dist.get(other, float("inf")):
                dist[other] = candidate
                from_table, from_name = from_col.split(".", 1)
                to_table, to_name = to_col.split(".", 1)
                prev[other] = (
                    current,
                    JoinStep(
                        left_table=from_table,
                        left_column=from_name,
                        right_table=to_table,
                        right_column=to_name,
                        proven=proven,
                    ),
                )

    if right not in dist:
        return path

    # Walk back, then reverse so steps read left -> right.
    steps: List[JoinStep] = []
    node = right
    while node != left:
        if node not in prev:
            return JoinPath(left=left, right=right)
        node, step = prev[node]
        steps.append(step)
    steps.reverse()
    path.steps = steps
    return path


def render_join_path(path: JoinPath) -> str:
    """Prompt-ready rendering, honest about unproven steps."""
    if not path.found:
        return (
            f"No column-level join path between {path.left} and {path.right}. "
            "Do not invent a join; ask the user or pick another table."
        )
    lines = [f"Join path: {' -> '.join(path.tables)}"]
    lines.append(f"all_proven: {path.all_proven}")
    lines.append("")
    lines.append(path.to_sql())
    if not path.all_proven:
        lines.append("")
        lines.append(
            "Some steps are inferred, not declared. Confirm before using."
        )
    return "\n".join(lines)
