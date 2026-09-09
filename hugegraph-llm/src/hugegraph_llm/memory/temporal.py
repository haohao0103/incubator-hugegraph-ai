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

"""Bi-temporal operations over the memory graph.

This is the layer that turns "memory" into "memory over time":

* :meth:`TemporalStore.add_fact` records a fact **and invalidates any
  contradicting fact** rather than overwriting it -- the single design
  decision that makes history recoverable.
* :meth:`TemporalStore.as_of` answers "what did we believe at time T?"
* :meth:`TemporalStore.between` answers "what changed over an interval?"
* :meth:`TemporalStore.current_state` / :meth:`~.transition_path` serve
  requirement ③ (state graph), reusing plain graph traversal.

Implementation notes that matter:

* **REST-based**, matching the rest of this module (Gremlin's script engine
  is missing under some JDK configurations; REST has no such dependency).
* **Filtering is done in the service layer.** HugeGraph has no composite
  AS-OF index, so a query fetches candidates by one indexed bound
  (``valid_at <= t``) and applies the remaining predicates in Python.
  Correctness first; if a real corpus makes this too slow, the fix is a
  server-side interval index, not a semantic change.
* **Intervals are half-open** ``[valid_at, invalid_at)``: a fact starting
  at T and one ending at T do not overlap, so consecutive periods can
  abut without an ambiguous instant.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.memory.schema import OPEN, EdgeLabel, VertexLabel

__all__ = ["Fact", "TemporalStore"]


@dataclass
class Fact:
    """One recalled fact, in the shape callers consume."""

    uuid: str
    fact: str
    #: Conflict key: facts sharing one supersede each other over time.
    name: str = ""
    source_uuid: str = ""
    target_uuid: str = ""
    valid_at: int = 0
    invalid_at: int = OPEN
    created_at: int = 0
    expired_at: int = OPEN

    @property
    def is_current(self) -> bool:
        """Still believed, i.e. not superseded by later information."""
        return self.expired_at == OPEN

    def valid_at_time(self, t: int) -> bool:
        """Was this fact true at ``t``? Half-open interval."""
        return self.valid_at <= t < self.invalid_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "fact": self.fact,
            "name": self.name,
            "source_uuid": self.source_uuid,
            "target_uuid": self.target_uuid,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "created_at": self.created_at,
            "expired_at": self.expired_at,
            "is_current": self.is_current,
        }


class TemporalStore:
    """Bi-temporal memory operations on top of :class:`HugeGraphClient`."""

    def __init__(self, client: Any) -> None:
        self.client = client

    # -- schema -------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the memory schema. Idempotent.

        Uses the client's dict-driven ``ensure_schema`` rather than its
        baked-in ``init_schema``: the latter declares every property as TEXT,
        which cannot support range queries over epoch-millis timestamps.
        """
        from hugegraph_llm.memory.schema import build_schema_dict

        return self.client.ensure_schema(build_schema_dict())

    # -- writing ------------------------------------------------------------

    def add_entity(self, uuid: str, name: str, summary: str = "", created_at: int = 0) -> None:
        self.client.upsert_vertex(
            VertexLabel.Entity,
            uuid,
            {"uuid": uuid, "name": name, "summary": summary, "created_at": created_at},
        )

    def add_episode(
        self, uuid: str, content: str, source: str = "", valid_at: int = 0, created_at: int = 0
    ) -> None:
        """Record a raw event -- the provenance anchor for later facts."""
        self.client.upsert_vertex(
            VertexLabel.Episode,
            uuid,
            {
                "uuid": uuid,
                "content": content,
                "source": source,
                "valid_at": valid_at,
                "created_at": created_at,
            },
        )

    def add_fact(
        self,
        uuid: str,
        fact: str,
        source_uuid: str,
        target_uuid: str,
        *,
        valid_at: int = 0,
        invalid_at: int = OPEN,
        created_at: int = 0,
        conflict_key: Optional[str] = None,
    ) -> List[str]:
        """Record a fact, expiring any fact it contradicts.

        :param conflict_key: identity of the *kind* of fact (for example
            ``"works_at"``). Facts sharing a key are mutually exclusive: the
            new one supersedes the old, which is expired rather than
            deleted so history stays queryable. Without a key, nothing is
            expired -- not every fact contradicts its predecessors.
        :returns: uuids of facts that were expired by this one.
        """
        self.client.upsert_edge(
            EdgeLabel.RELATES_TO,
            source_uuid,
            target_uuid,
            VertexLabel.Entity,
            VertexLabel.Entity,
            {
                "uuid": uuid,
                "fact": fact,
                "name": conflict_key or "",
                "valid_at": valid_at,
                "invalid_at": invalid_at,
                "created_at": created_at,
                "expired_at": OPEN,
            },
        )
        if not conflict_key:
            return []
        return self.expire_conflicts(
            source_uuid, conflict_key, superseded_at=created_at, keep=uuid
        )

    def expire_conflicts(
        self,
        source_uuid: str,
        conflict_key: str,
        superseded_at: int,
        keep: str,
    ) -> List[str]:
        """Expire superseded facts of the same kind, keeping ``keep``.

        **Matched by subject + predicate, not by object.** "u1 works at
        Acme" (u1 -> c1) and "u1 works at Globex" (u1 -> c2) contradict each
        other even though their targets differ; matching on the target pair
        would miss the conflict and leave two mutually exclusive facts both
        marked current.

        The superseded edge keeps its ``invalid_at`` (the fact stopped being
        true then) and gains ``expired_at`` (when we stopped believing it).
        Deleting it would destroy the very history this design exists for.
        """
        expired: List[str] = []
        for edge in self._edges_from(source_uuid):
            props = edge.get("properties") or {}
            if (props.get("name") or "") != conflict_key:
                continue
            if (props.get("uuid") or "") == keep:
                continue
            if (props.get("expired_at") or OPEN) != OPEN:
                continue  # already superseded
            edge_id = edge.get("id")
            if not edge_id:
                continue
            self.client.update_edge(
                edge_id, EdgeLabel.RELATES_TO, {"expired_at": superseded_at}
            )
            expired.append(str(props.get("uuid") or ""))
        return expired

    # -- reading ------------------------------------------------------------

    def as_of(self, t: int, *, include_superseded: bool = False) -> List[Fact]:
        """Facts believed to be true at ``t``.

        Two independent conditions, and both are needed:

        * the fact was **true** at ``t`` (valid-time containment), and
        * we still **believed** it at ``t`` -- a fact learned after ``t``
          was not yet known, so it must not appear in a historical view
          even if it was true then.

        :param include_superseded: also return facts that were later
            invalidated. Off by default: callers usually want the state of
            belief, not every record.
        """
        facts = self.all_facts()
        out = []
        for fact in facts:
            if not fact.valid_at_time(t):
                continue
            if fact.created_at > t:
                continue  # not known yet at t
            if not include_superseded and (0 <= fact.expired_at <= t):
                continue  # already superseded by then
            out.append(fact)
        return out

    def between(self, start: int, end: int) -> List[Fact]:
        """Facts that held at any point in ``[start, end)``."""
        return [f for f in self.all_facts() if f.valid_at < end and f.invalid_at > start]

    def all_facts(self) -> List[Fact]:
        """Every RELATES_TO edge as a :class:`Fact`."""
        facts: List[Fact] = []
        for edge in self._all_edges(EdgeLabel.RELATES_TO):
            props = edge.get("properties") or {}
            facts.append(
                Fact(
                    uuid=str(props.get("uuid") or edge.get("id") or ""),
                    fact=str(props.get("fact") or ""),
                    name=str(props.get("name") or ""),
                    source_uuid=str(edge.get("outV") or ""),
                    target_uuid=str(edge.get("inV") or ""),
                    valid_at=int(props.get("valid_at") or 0),
                    invalid_at=int(props.get("invalid_at") or OPEN),
                    created_at=int(props.get("created_at") or 0),
                    expired_at=int(props.get("expired_at") or OPEN),
                )
            )
        return facts

    def history_of(self, conflict_key: str) -> List[Fact]:
        """Chronological evolution of one kind of fact (requirement ①)."""
        facts = [f for f in self.all_facts() if f.name == conflict_key]
        return sorted(facts, key=lambda f: (f.valid_at, f.created_at))

    # -- state graph (requirement ③) ----------------------------------------

    def add_state(
        self,
        uuid: str,
        name: str,
        entity_uuid: str,
        valid_from: int,
        valid_to: int = OPEN,
    ) -> None:
        self.client.upsert_vertex(
            VertexLabel.State,
            uuid,
            {
                "uuid": uuid,
                "name": name,
                "entity_uuid": entity_uuid,
                "valid_from": valid_from,
                "valid_to": valid_to,
            },
        )

    def close_state(self, uuid: str, valid_to: int) -> None:
        """End a state period -- the usual way a transition is recorded."""
        self.client.update_vertex(
            VertexLabel.State, uuid, {"valid_to": valid_to}
        )

    def current_state(self, entity_uuid: str) -> Optional[Dict[str, Any]]:
        """The state an entity is in now, i.e. the one with no end.

        Returns None when the entity has no open state -- which is a
        legitimate answer (never entered, or exited and not re-entered), not
        an error a caller should have to catch.
        """
        for vertex in self._all_vertices(VertexLabel.State):
            props = vertex.get("properties") or {}
            if (props.get("entity_uuid") or "") != entity_uuid:
                continue
            if int(props.get("valid_to") or OPEN) == OPEN:
                return dict(props)
        return None

    def transition_path(self, entity_uuid: str) -> List[str]:
        """Chronological state transitions of an entity (requirement ③)."""
        states = []
        for vertex in self._all_vertices(VertexLabel.State):
            props = vertex.get("properties") or {}
            if (props.get("entity_uuid") or "") != entity_uuid:
                continue
            states.append((int(props.get("valid_from") or 0), str(props.get("name") or "")))
        return [name for _ts, name in sorted(states)]

    # -- internals ----------------------------------------------------------

    def _edges_from(self, source_uuid: str) -> List[Dict[str, Any]]:
        """Facts whose subject is ``source_uuid``."""
        out = []
        for edge in self._all_edges(EdgeLabel.RELATES_TO):
            if str(edge.get("outV")) == source_uuid:
                out.append(edge)
        return out

    def _all_edges(self, label: str) -> List[Dict[str, Any]]:
        """All edges of a label.

        HugeGraph's REST API exposes edges per vertex; scanning the source
        vertices is the only label-wide read available without Gremlin.
        """
        edges: List[Dict[str, Any]] = []
        seen = set()
        for vertex in self._all_vertices(VertexLabel.Entity):
            vid = vertex.get("id")
            for edge in self.client.get_edges_of(str(vid), direction="OUT") or []:
                edge_id = edge.get("id")
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                if (edge.get("label") or label) == label:
                    edges.append(edge)
        return edges

    def _all_vertices(self, label: str) -> List[Dict[str, Any]]:
        return self.client.get_vertices_by_label(label) or []
