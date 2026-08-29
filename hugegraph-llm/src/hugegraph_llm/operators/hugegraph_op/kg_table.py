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
"""Logical table abstraction backed by HugeGraph vertices.

Generalized from the MS-GraphRAG HugeGraph provider's table layer
(``graphrag_storage/hugegraph/table.py``): every logical table maps to a
HugeGraph vertex label, each row is a vertex with a ``CUSTOMIZE_STRING`` id
of the form ``"<table>:<row_id>"`` so ids stay unique across tables
(HugeGraph vertex ids are global within a graph).

Two modes:
  * ``schema`` given: rows are stored as typed properties; unknown columns
    are dropped, list/dict values are JSON-serialized, TEXT values are capped.
  * ``schema`` omitted: a best-effort KV mode using a single ``value`` TEXT
    property holding the whole row (minus the id) as JSON.

Schema (propertykeys + vertexlabel) is created lazily and idempotently on
first write via the :class:`~hugegraph_llm.operators.hugegraph_op.schema_manager.SchemaManager`
idempotent layer (exists-probe before create, nullable-everything labels).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional

from pyhugegraph.client import PyHugeClient
from pyhugegraph.utils.exceptions import NotFoundError, ServerError

from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.log import log

_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_]+")

# Hard TEXT cap (HugeGraph TEXT has no hard limit, but we keep a sane upper
# bound so requests do not bloat).
_TEXT_CAP = 32000

# Best-effort single-property schema when the caller does not provide one.
_DEFAULT_SCHEMA = {"value": "TEXT"}


def sanitize_label(name: str) -> str:
    """Make a string safe to use as a HugeGraph label name.

    HugeGraph labels must start with a letter and contain only alphanumeric
    characters and underscores.
    """
    cleaned = _LABEL_SAFE.sub("_", name).strip("_")
    if not cleaned:
        cleaned = "default"
    if not re.match(r"[A-Za-z_]", cleaned[0]):
        cleaned = f"ns_{cleaned}"
    return cleaned


def cap_str(value: Any, limit: int = _TEXT_CAP) -> str:
    """Cap a value to ``limit`` characters (None becomes an empty string)."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return text[:limit] if len(text) > limit else text


class KgTable:
    """A logical table mapped to a HugeGraph vertex label.

    Rows are dicts that must contain an ``id`` field (the logical row id).
    """

    def __init__(
        self,
        client: PyHugeClient,
        table_name: str,
        *,
        schema: Optional[Dict[str, str]] = None,
        page_size: int = 500,
        label_prefix: str = "",
    ) -> None:
        self._client = client
        self._table_name = table_name
        self._schema = dict(schema or {}) or dict(_DEFAULT_SCHEMA)
        self._page_size = max(1, int(page_size))
        self._label = f"{label_prefix}{sanitize_label(table_name)}"
        self._schema_ensured = False
        self._schema_manager = SchemaManager(table_name, client=client)

    @property
    def label(self) -> str:
        """The HugeGraph vertex label backing this table."""
        return self._label

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def schema(self) -> Dict[str, str]:
        """The column -> HugeGraph-type mapping used by this table."""
        return dict(self._schema)

    # -- schema ----------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create propertykeys + vertexlabel idempotently (once per instance)."""
        if self._schema_ensured:
            return
        properties = list(self._schema.keys())
        for prop_name, data_type in self._schema.items():
            self._schema_manager.create_property_key(prop_name, data_type)
        self._schema_manager.create_vertex_label(
            self._label,
            properties,
            id_strategy="CUSTOMIZE_STRING",
            nullable_keys=properties,
        )
        self._schema_ensured = True

    def _known_props(self) -> set:
        return set(self._schema.keys())

    # -- id helpers ------------------------------------------------------------

    def _vid(self, row_id: Any) -> str:
        """Build a unique vertex id: ``"<table>:<row_id>"``."""
        return f"{self._table_name}:{row_id}"

    def _row_id(self, vid: Any) -> str:
        """Inverse of :meth:`_vid`."""
        prefix = f"{self._table_name}:"
        if isinstance(vid, str) and vid.startswith(prefix):
            return vid[len(prefix):]
        return str(vid)

    # -- write -----------------------------------------------------------------

    def upsert(self, row: Dict[str, Any]) -> str:
        """Upsert a row as a vertex. Returns the HugeGraph vertex id.

        In schema mode unknown columns are dropped; in KV mode the whole row
        (minus ``id``) is JSON-serialized into the ``value`` property.
        """
        self._ensure_schema()
        row_id = row.get("id")
        if row_id is None:
            raise ValueError(f"row missing 'id' field for table {self._table_name!r}")
        vid = self._vid(row_id)
        props = self._serialize_props(row)
        result = self._client.graph().addVertex(self._label, props, id=vid)
        if result is None:
            raise RuntimeError(f"addVertex failed for {self._label} id={vid}")
        return vid

    def upsert_many(self, rows: List[Dict[str, Any]]) -> int:
        """Upsert a batch of rows; returns the number of rows written."""
        count = 0
        for row in rows:
            self.upsert(row)
            count += 1
        return count

    def delete(self, row_id: Any) -> bool:
        """Delete the row's vertex; returns True if a vertex was removed.

        HugeGraph 1.7.x raises on absent ids: GET -> 404 ``NotFoundError``,
        DELETE -> 400 ``ServerError`` ("No such vertex"); both are treated as
        "nothing to delete" -> False.
        """
        try:
            response = self._client.graph().removeVertexById(self._vid(row_id))
        except (NotFoundError, ServerError) as exc:
            log.warning("KgTable %r delete %r: %s", self._table_name, row_id, exc)
            return False
        return response is not None

    def clear(self) -> int:
        """Delete every vertex of this table; returns the number removed."""
        count = 0
        for row in self.iter_rows():
            if self.delete(self._row_id(row["id"])):
                count += 1
        return count

    # -- read ------------------------------------------------------------------

    def get(self, row_id: Any) -> Optional[Dict[str, Any]]:
        """Fetch one row by logical id, or None if absent."""
        try:
            vertex = self._client.graph().getVertexById(self._vid(row_id))
        except NotFoundError:
            return None
        if vertex is None:
            return None
        return self._flatten(vertex)

    def has(self, row_id: Any) -> bool:
        """Check whether a row exists."""
        try:
            return self._client.graph().getVertexById(self._vid(row_id)) is not None
        except NotFoundError:
            return False

    def length(self) -> int:
        """Count rows by paging through the label."""
        count = 0
        page: Optional[str] = None
        while True:
            vertices, next_page = self._page_vertices(page)
            if not vertices:
                break
            count += len(vertices)
            if not next_page or len(vertices) < self._page_size:
                break
            page = next_page
        return count

    def list_rows(
        self,
        limit: Optional[int] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List rows (flattened), optionally limited and/or property-filtered.

        ``properties`` must be a dict of secondary-indexed keys (e.g.
        ``{"status": "processed"}``); HugeGraph rejects filters on
        non-indexed properties.
        """
        rows: List[Dict[str, Any]] = []
        page: Optional[str] = None
        while True:
            if limit is not None:
                remaining = limit - len(rows)
                if remaining <= 0:
                    break
                batch = min(self._page_size, remaining)
            else:
                batch = self._page_size
            vertices, next_page = self._page_vertices(page, properties=properties)
            if not vertices:
                break
            for vertex in vertices:
                rows.append(self._flatten(vertex))
            if not next_page or len(vertices) < batch:
                break
            page = next_page
        return rows

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        """Stream rows (pages are fetched lazily under the hood)."""
        yield from self.list_rows()

    # -- internal helpers ------------------------------------------------------

    def _page_vertices(
        self, page: Optional[str], properties: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """One page of raw vertices; server/network errors degrade to empty."""
        try:
            return self._client.graph().getVertexByPage(
                self._label, self._page_size, page=page, properties=properties
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("KgTable %r page fetch failed: %s", self._table_name, exc)
            return [], None

    def _flatten(self, vertex: Any) -> Dict[str, Any]:
        """Normalize a VertexData to a row dict (id stripped of table prefix)."""
        row: Dict[str, Any] = {"id": self._row_id(str(vertex.id))}
        props = vertex.properties or {}
        known = self._known_props()
        for key, value in props.items():
            if key in known:
                row[key] = value
        return row

    def _serialize_props(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Map a row to HugeGraph properties for this table's schema.

        * schema mode: keep known columns only, JSON-serialize list/dict
          values, cap TEXT values.
        * KV mode (single ``value`` column): serialize the whole row minus id.
        """
        known = self._known_props()
        props: Dict[str, Any] = {}
        if self._schema == _DEFAULT_SCHEMA and "value" in known:
            payload = {k: v for k, v in row.items() if k != "id"}
            props["value"] = cap_str(json.dumps(payload, ensure_ascii=False))
            return props
        for key, value in row.items():
            if key == "id":
                continue
            if key not in known:
                log.debug("dropping unknown property %r on table %r", key, self._table_name)
                continue
            if isinstance(value, (list, dict)):
                props[key] = cap_str(json.dumps(value, ensure_ascii=False))
            elif isinstance(value, str):
                props[key] = cap_str(value)
            else:
                props[key] = value
        return props
