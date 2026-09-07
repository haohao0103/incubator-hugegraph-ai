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

"""Graph-native semantic layer on HugeGraph.

Organised by the three layers the layer serves plus the governance layer
that makes it trustworthy:

    modelling   schema_def / bootstrap / connectors   (milestones M0, M1)
    retrieval   candidate generation, subgraph pruning, token budgeting (M2)
    serving     MCP tools                                              (M3)
    governance  confidence / lineage / freshness / provenance

Retrieval and serving live in later milestones; the modules here are the
foundation they build on.
"""

from hugegraph_llm.semantic_layer.enums import (
    EDGE_JOIN_COST,
    NON_JOINABLE_EDGES,
    EdgeLabel,
    VertexLabel,
)
from hugegraph_llm.semantic_layer.schema_def import (
    EDGE_LABELS,
    INDEX_LABELS,
    PROPERTY_KEYS,
    VERTEX_LABELS,
    build_schema_dict,
    coerce_value,
    validate,
)

__all__ = [
    "EdgeLabel",
    "VertexLabel",
    "EDGE_JOIN_COST",
    "NON_JOINABLE_EDGES",
    "PROPERTY_KEYS",
    "VERTEX_LABELS",
    "EDGE_LABELS",
    "INDEX_LABELS",
    "build_schema_dict",
    "coerce_value",
    "validate",
]
