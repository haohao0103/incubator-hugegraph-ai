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

"""Paging helpers that work around a pyhugegraph request bug.

``GraphManager.getVertexByPage`` / ``getEdgeByPage`` build their query string
like this::

    if page:
        para += f"&page={page}"
    else:
        para += "&page"          # <-- bare, empty parameter

On the first call ``page`` is ``None``, so the request goes out as
``?label=Column&page&limit=500``. HugeGraph 1.7 answers that with **500**,
which means any connector paging over a label with more than one page worth
of data fails as soon as the dataset grows past a toy example (12 tables
worked; 323 columns did not).

Until that is fixed upstream, reads go through Gremlin with ``range()``
paging instead. ``elementMap()`` is used rather than ``valueMap(true)``
because it returns single values instead of single-element lists, matching
the shape of the REST response the rest of the code expects.
"""

from typing import Any, Dict, Iterator, Optional, Tuple

__all__ = ["iter_vertices", "iter_edges", "PAGE_SIZE"]

PAGE_SIZE = 500


def _exec(client: Any, script: str) -> list:
    """Run a Gremlin script and normalise the response to a list."""
    response = client.gremlin().exec(script)
    if isinstance(response, dict):
        return response.get("data") or []
    return response or []


#: Keys ``elementMap()`` adds that are not properties.
_RESERVED = ("id", "label", "IN", "OUT")


def _properties(element: Dict[str, Any]) -> Dict[str, Any]:
    """Strip ``elementMap`` bookkeeping keys.

    Values are returned exactly as the server sends them: SINGLE properties
    as scalars, LIST properties as lists. Do **not** collapse single-element
    lists here -- a LIST property holding one value (``from_columns:
    ["customer_id"]``) would silently become a string, and any caller doing
    ``list(value)`` afterwards would explode it into ``["c","u","s",...]``.
    """
    return {
        key: value
        for key, value in element.items()
        if key not in _RESERVED
    }


def iter_vertices(
    client: Any, label: Optional[str] = None, page_size: int = PAGE_SIZE
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(vertex_id, properties)`` for every vertex, optionally filtered."""
    where = f".hasLabel('{label}')" if label else ""
    offset = 0
    while True:
        script = (
            f"g.V(){where}.range({offset}, {offset + page_size}).elementMap()"
        )
        batch = _exec(client, script)
        if not batch:
            return
        for element in batch:
            if not isinstance(element, dict):
                continue
            yield str(element.get("id")), _properties(element)
        if len(batch) < page_size:
            return
        offset += page_size


def iter_edges(
    client: Any, label: Optional[str] = None, page_size: int = PAGE_SIZE
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Yield ``(out_vertex_id, in_vertex_id, properties)`` for every edge.

    Endpoint ids are projected separately so callers can resolve them
    without a second query.
    """
    where = f".hasLabel('{label}')" if label else ""
    offset = 0
    while True:
        script = (
            f"g.E(){where}.range({offset}, {offset + page_size})"
            ".project('outV','inV','props')"
            ".by(outV().id()).by(inV().id()).by(elementMap())"
        )
        batch = _exec(client, script)
        if not batch:
            return
        for element in batch:
            if not isinstance(element, dict):
                continue
            yield (
                str(element.get("outV")),
                str(element.get("inV")),
                _properties(element.get("props") or {}),
            )
        if len(batch) < page_size:
            return
        offset += page_size
