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

"""Write query feedback back into the semantic layer (milestone M6).

Retrieval quality so far comes from static metadata. This closes the loop:
when a question is answered successfully, the tables it actually needed are
recorded, and future retrieval can prefer tables that keep showing up in
real usage.

Data model (labels already exist in ``schema_def``)::

    Query   {name, content, exec_count, last_seen_ts, schema_refs}
    USES_TABLE  Query -> Table   {use_count}
    CO_OCCUR    Table -> Table   {weight}   one edge per co-occurrence

**CO_OCCUR is event-per-edge, not a running weight.** Each co-occurrence
appends one edge with ``weight=1`` instead of looking up an edge and
incrementing its property. Rationale: HugeGraph allows parallel edges and
the client's ``appendEdge`` puts the edge id (which contains ``>`` and
``:``) straight into the URL path, so read-modify-write is both racy and
fragile. Readers already traverse ``both()`` and aggregate, so parallel
edges cost nothing on the read side; the weight of a pair is its edge
count. The linear edge growth is acceptable at metadata scale (a query
over 8 tables adds 28 edges).

**Idempotency.** A submission's identity is ``sha1(question + sql)``. The
same pair again bumps the Query's ``exec_count`` and refreshes
``last_seen_ts`` but writes no new edges -- one question, one vote.

**Endpoints are resolved through the projection's ``table_vids``**, not by
rebuilding ids from a naming convention: Ossie namespaces its vertex ids
per model (``acme:table:orders``), so ``table:orders`` would be a vertex
that does not exist, and HugeGraph rejects the edge only after the caller
has already assumed the write worked.

Tables are also filtered to those present in the projection: an edge whose
endpoints do not exist fails server-side, far from the cause.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.semantic_layer.evaluation.dataset import extract_tables
from hugegraph_llm.semantic_layer.readers import SemanticGraphReader
from hugegraph_llm.semantic_layer.schema_def import coerce_value
from hugegraph_llm.utils.log import log

__all__ = ["FeedbackRecorder", "FeedbackResult"]


@dataclass
class FeedbackResult:
    """What one :meth:`FeedbackRecorder.record` call did."""

    question: str
    query_id: str
    #: Tables credited, after filtering to those in the projection.
    tables: List[str] = field(default_factory=list)
    #: Tables requested (or SQL-extracted) that the graph does not know.
    unknown_tables: List[str] = field(default_factory=list)
    #: True when this (question, sql) was already recorded -- counters were
    #: bumped but no new edges were written.
    deduplicated: bool = False
    exec_count: int = 1
    co_occurrence_edges: int = 0
    #: True when at least one table was credited.
    ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "tables": self.tables,
            "unknown_tables": self.unknown_tables,
            "deduplicated": self.deduplicated,
            "exec_count": self.exec_count,
            "co_occurrence_edges": self.co_occurrence_edges,
            "ok": self.ok,
        }


class FeedbackRecorder:
    """Records successful query usage back into the graph.

    After recording, call :meth:`invalidate` (or pass ``refresh=True`` on
    the next projection read) or the new co-occurrence edges stay invisible
    to retrieval in this process.
    """

    def __init__(self, client: PyHugeClient, reader: SemanticGraphReader) -> None:
        self.client = client
        self.reader = reader

    def invalidate(self) -> None:
        """Drop the reader's cached projection so feedback becomes visible."""
        self.reader.invalidate()

    def record(
        self,
        question: str,
        sql: str,
        *,
        tables: Optional[Sequence[str]] = None,
    ) -> FeedbackResult:
        """Record one successful execution.

        :param tables: tables the query used. When omitted they are
            extracted from the SQL (``FROM``/``JOIN`` positions), which
            keeps the call honest: the caller cannot easily claim credit
            for tables the query never touches.
        """
        projection = self.reader.projection()
        resolved, unknown = self._resolve_tables(sql, tables, projection)

        result = FeedbackResult(
            question=question,
            query_id=_query_id(question, sql),
            tables=resolved,
            unknown_tables=unknown,
        )
        if not resolved:
            log.warning(
                "feedback: no known tables for question %r; nothing recorded",
                question[:60],
            )
            return result

        now = int(time.time())
        existing = self._load_existing(result.query_id)
        if existing is not None:
            self._touch(result.query_id, existing, now)
            result.deduplicated = True
            # The feedback IS recorded (first time); this call only bumped
            # the counter. ok=False here would make a caller that reports
            # success-by-ok treat a perfectly good repeat as a failure.
            result.ok = True
            result.exec_count = int(existing.get("exec_count") or 0) + 1
            return result

        self._create_query(result, sql, now)
        self._record_uses(result)
        self._record_co_occurrence(result)
        result.ok = True
        return result

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _resolve_tables(
        sql: str,
        tables: Optional[Sequence[str]],
        projection: Any,
    ) -> tuple[List[str], List[str]]:
        candidates = list(tables) if tables else extract_tables(sql)
        known: List[str] = []
        unknown: List[str] = []
        for name in candidates:
            (known if name in projection.tables else unknown).append(name)
        return known, unknown

    def _load_existing(self, query_id: str) -> Optional[Dict[str, Any]]:
        try:
            vertex = self.client.graph().getVertexById(query_id)
        except Exception:  # noqa: BLE001 - absence surfaces as an error here
            return None
        if vertex is None:
            return None
        return dict(vertex.properties or {})

    def _touch(self, query_id: str, existing: Dict[str, Any], now: int) -> None:
        """Bump exec_count on a repeat submission; edges stay as they were."""
        try:
            self.client.graph().appendVertex(
                query_id,
                {
                    "exec_count": int(existing.get("exec_count") or 0) + 1,
                    "last_seen_ts": now,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("feedback: could not update query counters: %s", exc)

    def _create_query(
        self, result: FeedbackResult, sql: str, now: int
    ) -> None:
        self.client.graph().addVertex(
            VertexLabel.QUERY.value,
            _clean({
                "name": result.question[:120],
                "content": sql,
                "exec_count": 1,
                "last_seen_ts": now,
                "schema_refs": result.tables,
            }),
            id=result.query_id,
        )

    def _record_uses(self, result: FeedbackResult) -> None:
        projection = self.reader.projection()
        for table in result.tables:
            vid = projection.table_vids.get(table, f"table:{table}")
            try:
                self.client.graph().addEdge(
                    EdgeLabel.USES_TABLE.value,
                    result.query_id,
                    vid,
                    {"use_count": 1},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback: USES_TABLE %s failed: %s", table, exc)

    def _record_co_occurrence(self, result: FeedbackResult) -> None:
        """One weight-1 edge per pair; the pair's weight is its edge count."""
        projection = self.reader.projection()
        tables = result.tables
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                out_v = projection.table_vids.get(tables[i], f"table:{tables[i]}")
                in_v = projection.table_vids.get(tables[j], f"table:{tables[j]}")
                try:
                    self.client.graph().addEdge(
                        EdgeLabel.CO_OCCUR.value,
                        out_v,
                        in_v,
                        {"weight": 1},
                    )
                    result.co_occurrence_edges += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "feedback: CO_OCCUR %s-%s failed: %s", tables[i], tables[j], exc
                    )


def _query_id(question: str, sql: str) -> str:
    digest = hashlib.sha1(f"{question}\n{sql}".encode("utf-8")).hexdigest()[:16]
    return f"query:{digest}"


def _clean(props: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in props.items():
        coerced = coerce_value(key, value)
        if coerced is None:
            continue
        cleaned[key] = coerced
    return cleaned
