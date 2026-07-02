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


from typing import Any, Dict, List, Optional, Union

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.utils.log import log

# Default limits for graph data fetching (configurable via HugeGraphConfig)
DEFAULT_VERTEX_LIMIT = 10000
DEFAULT_EDGE_LIMIT = 200


class FetchGraphData:
    def __init__(self, graph: PyHugeClient, v_limit: int = DEFAULT_VERTEX_LIMIT,
                 e_limit: int = DEFAULT_EDGE_LIMIT):
        self.graph = graph
        self.v_limit = v_limit
        self.e_limit = e_limit

    def _fetch_all_vertices(self) -> List[str]:
        """Fetch vertex IDs via REST API, paginated by label.

        Returns a flat list of VID strings. Kept for backward compatibility.
        For enriched data with properties+label, use _fetch_all_vertices_detail().
        """
        schema = self.graph.schema()
        vertex_labels = schema.getVertexLabels()
        all_vids: List[str] = []
        for vl in vertex_labels:
            label = vl.get("name") if isinstance(vl, dict) else getattr(vl, "name", str(vl))
            page = ""
            fetched = 0
            while fetched < self.v_limit:
                batch_size = min(500, self.v_limit - fetched)
                try:
                    vertices, next_page = self.graph.graph().getVertexByPage(
                        label=label, limit=batch_size, page=page if page else None
                    )
                except Exception as e:
                    log.warning("FetchGraphData: failed to fetch vertices for label '%s': %s", label, e)
                    break
                if not vertices:
                    break
                for v in vertices:
                    all_vids.append(v.id)
                fetched += len(vertices)
                if not next_page or len(vertices) < batch_size:
                    break
                page = next_page
        return all_vids

    def _fetch_all_vertices_detail(self) -> List[Dict[str, Any]]:
        """Fetch vertex IDs + properties + label, paginated.

        Returns a list of dicts:
        [
          {"vid": "Person:张三", "label": "Person", "properties": {"name": "张三", "age": 35, ...}},
          ...
        ]
        Used by BuildSemanticIndex for rich entity text construction (vid_embed_strategy).
        """
        schema = self.graph.schema()
        vertex_labels = schema.getVertexLabels()
        all_details: List[Dict[str, Any]] = []

        for vl in vertex_labels:
            label = vl.get("name") if isinstance(vl, dict) else getattr(vl, "name", str(vl))
            page = ""
            fetched = 0
            while fetched < self.v_limit:
                batch_size = min(200, self.v_limit - fetched)  # Smaller batch for property payload
                try:
                    vertices, next_page = self.graph.graph().getVertexByPage(
                        label=label, limit=batch_size, page=page if page else None
                    )
                except Exception as e:
                    log.warning("FetchGraphData: failed to fetch vertex details for label '%s': %s", label, e)
                    break
                if not vertices:
                    break
                for v in vertices:
                    props: Dict[str, Any] = {}
                    if hasattr(v, "properties") and v.properties:
                        try:
                            props = dict(v.properties) if hasattr(v.properties, "items") else {}
                        except Exception:
                            props = {}
                    all_details.append({
                        "vid": v.id,
                        "label": label,
                        "properties": props,
                    })
                fetched += len(vertices)
                if not next_page or len(vertices) < batch_size:
                    break
                page = next_page
        return all_details

    def _fetch_all_edges(self) -> List[str]:
        """Fetch edge IDs via REST API, paginated."""
        all_eids: List[str] = []
        page = ""
        fetched = 0
        while fetched < self.e_limit:
            batch_size = min(500, self.e_limit - fetched)
            try:
                edges, next_page = self.graph.graph().getEdgeByPage(
                    limit=batch_size, page=page if page else None
                )
            except Exception as e:
                log.warning("FetchGraphData: failed to fetch edges: %s", e)
                break
            if not edges:
                break
            for e in edges:
                all_eids.append(e.id)
            fetched += len(edges)
            if not next_page or len(edges) < batch_size:
                break
            page = next_page
        return all_eids

    def run(self, graph_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if graph_summary is None:
            graph_summary = {}

        try:
            schema = self.graph.schema()
            vertex_labels = schema.getVertexLabels()
            edge_labels = schema.getEdgeLabels()
            graph_summary["vertex_num"] = len(vertex_labels)
            graph_summary["edge_num"] = len(edge_labels)
        except Exception as e:
            log.warning("FetchGraphData: failed to get schema: %s", e)
            graph_summary["vertex_num"] = 0
            graph_summary["edge_num"] = 0

        try:
            # Always fetch flat VID list (backward compat for SemanticIdQuery etc.)
            vertices = self._fetch_all_vertices()
            graph_summary["vertices"] = vertices
        except Exception as e:
            log.error("FetchGraphData: failed to fetch vertices: %s", e)
            graph_summary["vertices"] = []

        try:
            # Also fetch enriched details (vid + label + properties) for BuildSemanticIndex
            vertex_details = self._fetch_all_vertices_detail()
            graph_summary["vertex_details"] = vertex_details
        except Exception as e:
            log.error("FetchGraphData: failed to fetch vertex details: %s", e)
            graph_summary["vertex_details"] = []

        try:
            edges = self._fetch_all_edges()
            graph_summary["edges"] = edges
        except Exception as e:
            log.error("FetchGraphData: failed to fetch edges: %s", e)
            graph_summary["edges"] = []

        graph_summary["note"] = f"Only ≤{self.v_limit} VIDs and ≤ {self.e_limit} EIDs for brief overview."
        return graph_summary
