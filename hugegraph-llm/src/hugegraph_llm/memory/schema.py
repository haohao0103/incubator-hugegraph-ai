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

"""Bi-temporal graph schema for agent memory, on HugeGraph.

Model (from Graphiti, whose semantics are proven by its own test suite):

    Entity   -- a user, object or concept
    Episode  -- a raw event (conversation turn, observation); the provenance
                anchor a memory can be traced back to
    State    -- a status an entity was in over a period (requirement ③:
                Graphiti has no state model, so this is ours)
    RELATES_TO -- fact between two entities, carrying the bi-temporal fields
    MENTIONS   -- episode -> entity, provenance
    TRANSITION -- state -> state, the flow of a status over time

**Two time axes**, because "when was this true" and "when did we find out"
are different questions and a memory system needs both:

    Valid Time        valid_at / invalid_at     when the fact held
    Transaction Time  created_at / expired_at   when we learned / superseded it

**Never delete a contradicted fact.** When new information conflicts, the
old edge gets ``expired_at`` set instead of being removed. That is what
makes "what did we believe on date X?" answerable -- the whole point of a
temporal memory, and the reason a plain upsert-based store cannot do it.

**Open intervals use a sentinel, not null.** An interval with no end is
stored as :data:`OPEN` (``Long.MAX_VALUE``) rather than left empty:
HugeGraph's ``has()`` filter requires an index, and null values are not
indexed, so "still valid" rows would silently drop out of every query.
This is the same tri-state lesson the semantic layer hit with
``is_time_dimension``.

All temporal fields are **epoch millis (LONG)**. Dates as text would sort
lexically only if strictly ISO-normalised, and range comparisons on strings
invite subtle bugs; millis compare and index natively.
"""

from typing import Any, Dict, List, Tuple

__all__ = [
    "OPEN",
    "VERTEX_LABELS",
    "EDGE_LABELS",
    "PROPERTY_KEYS",
    "INDEX_LABELS",
    "build_schema_dict",
    "VertexLabel",
    "EdgeLabel",
]

#: Sentinel for an interval with no end ("still valid", "not superseded").
#: Chosen to be the largest LONG so range filters treat it as +infinity.
OPEN = 2**63 - 1


class VertexLabel:
    Entity = "Entity"
    Episode = "Episode"
    State = "State"


class EdgeLabel:
    RELATES_TO = "RELATES_TO"
    MENTIONS = "MENTIONS"
    TRANSITION = "TRANSITION"


# (name, data_type, cardinality)
PROPERTY_KEYS: List[Tuple[str, str, str]] = [
    # identity
    ("uuid", "TEXT", "SINGLE"),
    ("name", "TEXT", "SINGLE"),
    ("summary", "TEXT", "SINGLE"),
    ("content", "TEXT", "SINGLE"),
    ("source", "TEXT", "SINGLE"),
    ("fact", "TEXT", "SINGLE"),
    # -- valid time: when the fact was true --
    ("valid_at", "LONG", "SINGLE"),
    ("invalid_at", "LONG", "SINGLE"),
    # -- transaction time: when we learned / superseded it --
    ("created_at", "LONG", "SINGLE"),
    ("expired_at", "LONG", "SINGLE"),
    # -- state graph (requirement ③) --
    ("entity_uuid", "TEXT", "SINGLE"),
    ("valid_from", "LONG", "SINGLE"),
    ("valid_to", "LONG", "SINGLE"),
    ("from_state", "TEXT", "SINGLE"),
    ("to_state", "TEXT", "SINGLE"),
    ("trigger_event", "TEXT", "SINGLE"),
]

#: vertex label -> property names
VERTEX_LABELS: Dict[str, List[str]] = {
    VertexLabel.Entity: ["uuid", "name", "summary", "created_at"],
    VertexLabel.Episode: ["uuid", "content", "source", "created_at", "valid_at"],
    VertexLabel.State: ["uuid", "name", "entity_uuid", "valid_from", "valid_to"],
}

#: edge label -> (source_label, target_label, property names)
EDGE_LABELS: List[Tuple[str, str, str, List[str]]] = [
    (
        EdgeLabel.RELATES_TO,
        VertexLabel.Entity,
        VertexLabel.Entity,
        ["uuid", "fact", "name", "valid_at", "invalid_at", "created_at", "expired_at"],
    ),
    (
        EdgeLabel.MENTIONS,
        VertexLabel.Episode,
        VertexLabel.Entity,
        ["created_at"],
    ),
    (
        EdgeLabel.TRANSITION,
        VertexLabel.State,
        VertexLabel.State,
        ["from_state", "to_state", "valid_at", "trigger_event"],
    ),
]

#: Extra indexes. Only SECONDARY and RANGE exist on HugeGraph 1.7 -- there is
#: no full-text or vector index, which is why retrieval scoring happens in
#: the service layer.
INDEX_LABELS: List[Dict[str, str]] = [
    {
        "name": "relatesByValidAt",
        "base_label": EdgeLabel.RELATES_TO,
        "field": "valid_at",
        "index_type": "RANGE",
        "on": "edge",
    },
    {
        "name": "relatesByInvalidAt",
        "base_label": EdgeLabel.RELATES_TO,
        "field": "invalid_at",
        "index_type": "RANGE",
        "on": "edge",
    },
    {
        "name": "relatesByExpiredAt",
        "base_label": EdgeLabel.RELATES_TO,
        "field": "expired_at",
        "index_type": "RANGE",
        "on": "edge",
    },
    {
        "name": "stateByValidTo",
        "base_label": VertexLabel.State,
        "field": "valid_to",
        "index_type": "RANGE",
        "on": "vertex",
    },
    {
        "name": "entityByName",
        "base_label": VertexLabel.Entity,
        "field": "name",
        "index_type": "SECONDARY",
        "on": "vertex",
    },
]


def build_schema_dict() -> Dict[str, Any]:
    """Schema payload for :meth:`HugeGraphClient.init_schema`."""
    propertykeys = [
        {"name": name, "data_type": dtype, "cardinality": card}
        for name, dtype, card in PROPERTY_KEYS
    ]
    vertexlabels = [
        {
            "name": label,
            "properties": list(props),
            "id_strategy": "CUSTOMIZE_STRING",
            # Everything optional: an episode may arrive without a summary,
            # a state without an end. HugeGraph rejects undeclared nulls
            # unless the key is listed here.
            "nullable_keys": list(props),
        }
        for label, props in VERTEX_LABELS.items()
    ]
    edgelabels = [
        {
            "name": name,
            "source_label": source,
            "target_label": target,
            "properties": list(props),
        }
        for name, source, target, props in EDGE_LABELS
    ]
    return {
        "propertykeys": propertykeys,
        "vertexlabels": vertexlabels,
        "edgelabels": edgelabels,
        "indexes": [dict(idx) for idx in INDEX_LABELS],
    }
