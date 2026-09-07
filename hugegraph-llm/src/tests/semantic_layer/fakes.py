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

"""In-memory HugeGraph stand-in for connector tests.

Connectors now page through Gremlin (see ``semantic_layer.paging``), so a
fake has to answer both the write API (``graph().addVertex/addEdge``) and
the read path (``gremlin().exec``). Keeping one fake for both means
connector tests exercise a real ingest -> export round trip with no server.

Only the shapes the connectors actually emit are interpreted; anything else
returns an empty result rather than pretending to be a Gremlin interpreter.
"""

import re
from typing import Any, Dict, List, Tuple

__all__ = ["FakeHugeGraph"]

_LABEL_RE = re.compile(r"hasLabel\('([^']+)'\)")
_RANGE_RE = re.compile(r"range\((\d+),\s*(\d+)\)")


class FakeHugeGraph:
    """Minimal in-memory graph with the slice of the API connectors use."""

    def __init__(self, vertices=None, edges=None):
        # vid -> {"label": str, "properties": dict}
        self.vertices: Dict[str, Dict[str, Any]] = {
            vid: {"label": data[0], "properties": dict(data[1])}
            for vid, data in (vertices or {}).items()
        }
        # (label, out_v, in_v, properties)
        self.edges: List[Tuple[str, str, str, Dict[str, Any]]] = [
            (e["label"], e["outV"], e["inV"], dict(e.get("properties") or {}))
            for e in (edges or [])
        ]

    # -- client surface ----------------------------------------------------

    def graph(self):
        return self

    def gremlin(self):
        return self

    # -- writes ------------------------------------------------------------

    def addVertex(self, label, properties, id=None):  # noqa: A002 - matches client
        vid = id or f"auto:{len(self.vertices)}"
        self.vertices[vid] = {"label": label, "properties": dict(properties or {})}
        return vid

    def addEdge(self, edge_label, out_id, in_id, properties):
        if out_id not in self.vertices or in_id not in self.vertices:
            raise ValueError(f"Invalid vertex id: {out_id} -> {in_id}")
        self.edges.append(
            (edge_label, str(out_id), str(in_id), dict(properties or {}))
        )
        return f"edge:{len(self.edges)}"

    def addVertices(self, input_data):
        return [self.addVertex(*item) for item in input_data]

    def addEdges(self, input_data):
        return [self.addEdge(*item) for item in input_data]

    # -- reads (Gremlin) ---------------------------------------------------

    def exec(self, script: str) -> Dict[str, Any]:
        """Interpret the handful of scripts ``paging`` emits."""
        if "g.V()" in script:
            return {"data": self._exec_vertices(script)}
        if "g.E()" in script:
            return {"data": self._exec_edges(script)}
        return {"data": []}

    def _exec_vertices(self, script: str) -> List[Dict[str, Any]]:
        label_match = _LABEL_RE.search(script)
        range_match = _RANGE_RE.search(script)
        start, end = (int(range_match.group(1)), int(range_match.group(2))) \
            if range_match else (0, len(self.vertices))

        rows = []
        for vid, data in self.vertices.items():
            if label_match and data["label"] != label_match.group(1):
                continue
            # elementMap() shape: id/label alongside plain scalar properties.
            row: Dict[str, Any] = {"id": vid, "label": data["label"]}
            row.update(data["properties"])
            rows.append(row)
        return rows[start:end]

    def _exec_edges(self, script: str) -> List[Dict[str, Any]]:
        label_match = _LABEL_RE.search(script)
        range_match = _RANGE_RE.search(script)
        start, end = (int(range_match.group(1)), int(range_match.group(2))) \
            if range_match else (0, len(self.edges))

        rows = []
        for label, out_v, in_v, props in self.edges:
            if label_match and label != label_match.group(1):
                continue
            rows.append({"outV": out_v, "inV": in_v, "props": dict(props)})
        return rows[start:end]

    # -- assertions helpers ------------------------------------------------

    def count(self, label: str) -> int:
        return sum(1 for v in self.vertices.values() if v["label"] == label)

    def edge_count(self, label: str) -> int:
        return sum(1 for e in self.edges if e[0] == label)
