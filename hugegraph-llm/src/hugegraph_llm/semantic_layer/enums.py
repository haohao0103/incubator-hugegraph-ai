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

"""Controlled vocabulary for the HugeGraph-backed semantic layer.

Every connector, retriever and MCP tool speaks in these labels, so they are
declared once here instead of being re-typed as string literals across the
code base.

Design notes that are easy to get wrong on HugeGraph:

* An edge label locks a single ``source_label -> target_label`` pair at first
  creation. A later edge between different endpoint types is rejected, and
  widening the pair requires deleting the label and migrating existing edges.
  That is why the business-term tag is split into
  :attr:`EdgeLabel.TABLE_TAGGED_WITH` and :attr:`EdgeLabel.COLUMN_TAGGED_WITH`
  rather than one shared ``TAGGED_WITH`` label.
* :data:`EDGE_JOIN_COST` mirrors ``nl2sql.schema_graph.model`` so join-path
  behaviour stays identical whether the path is searched over the local
  Schema Graph or over HugeGraph.
"""

from enum import Enum
from typing import Dict

__all__ = [
    "VertexLabel",
    "EdgeLabel",
    "EDGE_JOIN_COST",
    "NON_JOINABLE_EDGES",
]


class VertexLabel(str, Enum):
    """Vertex labels of the semantic layer graph."""

    DATABASE = "Database"
    SCHEMA = "Schema"
    TABLE = "Table"
    COLUMN = "Column"
    BUSINESS_TERM = "BusinessTerm"
    METRIC = "Metric"
    JOIN = "Join"
    QUERY = "Query"
    DOMAIN = "Domain"

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)


class EdgeLabel(str, Enum):
    """Edge labels of the semantic layer graph.

    Each label maps to exactly one endpoint pair; see the module docstring
    for why ``TAGGED_WITH`` is split in two.
    """

    HAS_SCHEMA = "HAS_SCHEMA"
    HAS_TABLE = "HAS_TABLE"
    HAS_COLUMN = "HAS_COLUMN"
    REFERENCES = "REFERENCES"
    LINEAGE = "LINEAGE"
    CO_OCCUR = "CO_OCCUR"
    TABLE_TAGGED_WITH = "TABLE_TAGGED_WITH"
    COLUMN_TAGGED_WITH = "COLUMN_TAGGED_WITH"
    METRIC_TAGGED_WITH = "METRIC_TAGGED_WITH"
    TERM_MAPS = "TERM_MAPS"
    HAS_EXPRESSION = "HAS_EXPRESSION"
    SYNONYM = "SYNONYM"
    USES_TABLE = "USES_TABLE"
    USES_COLUMN = "USES_COLUMN"

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)


#: Cost of traversing an edge when building a join path. Lower is stronger.
#: A declared foreign key is always joinable; co-occurrence is only a hint
#: mined from query logs, so it carries the heaviest penalty.
EDGE_JOIN_COST: Dict[EdgeLabel, float] = {
    EdgeLabel.HAS_COLUMN: 0.5,
    EdgeLabel.REFERENCES: 1.0,
    EdgeLabel.LINEAGE: 1.5,
    EdgeLabel.CO_OCCUR: 3.0,
    EdgeLabel.TERM_MAPS: float("inf"),
    EdgeLabel.TABLE_TAGGED_WITH: float("inf"),
    EdgeLabel.COLUMN_TAGGED_WITH: float("inf"),
    EdgeLabel.METRIC_TAGGED_WITH: float("inf"),
}

#: Edges that must never be traversed when connecting two tables.
NON_JOINABLE_EDGES = frozenset(
    label for label, cost in EDGE_JOIN_COST.items() if cost == float("inf")
)
