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

"""
Incremental graph update mechanism (LightRAG-style).

Core principle: Entity name as primary key enables append-only updates.
New documents can be indexed without rebuilding the entire graph:

1. Entity ID = entity name (dedup by name, not by generated ID)
2. New entities: match existing → merge properties; not found → create
3. New edges: connect to existing entities by name resolution
4. No global restructuring needed (unlike community-based approaches)

This is the key differentiator from Microsoft GraphRAG/LazyGraphRAG,
which require full rebuilds because their community structure is
a global property that changes when new data is added.

Reference: LightRAG (https://github.com/HKUDS/LightRAG)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.utils.log import log


class IncrementalGraphUpdater:
    """
    Manages incremental updates to the knowledge graph.

    LightRAG-style approach: entity name as primary key enables
    append-only incremental updates without full graph rebuild.

    Key design decisions:
    - Entity ID is derived from entity name (not auto-generated)
    - Deduplication is name-based, not ID-based
    - Property merge: new values override, old values preserved if not overridden
    - Edge dedup: based on (source_name, edge_label, target_name) triplet
    - No dependency on community structure (can be added optionally later)
    """

    def __init__(self, graph_client: Optional[Any] = None):
        self.graph_client = graph_client
        self._change_log: List[Dict[str, Any]] = []
        self._entity_name_to_id: Dict[str, str] = {}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run incremental graph update.

        Args:
            context: Dict containing 'vertices', 'edges', and optionally 'schema'.

        Returns:
            Updated context with incremental update metadata and
            name-to-ID mapping for downstream consumption.
        """
        vertices = context.get("vertices", [])
        edges = context.get("edges", [])
        schema = context.get("schema")

        if not vertices and not edges:
            log.info("No new vertices or edges to add incrementally")
            context["incremental_update_summary"] = {
                "new_vertices": 0,
                "updated_vertices": 0,
                "new_edges": 0,
                "updated_edges": 0,
                "timestamp": datetime.now().isoformat(),
            }
            return context

        # Phase 1: Build entity name → ID mapping from existing graph
        self._build_entity_name_map(vertices)

        # Phase 2: Resolve and deduplicate vertices by entity name
        new_vertices, updated_vertices = self._deduplicate_vertices(vertices, schema)

        # Phase 3: Resolve and deduplicate edges by name triplet
        new_edges, updated_edges = self._deduplicate_edges(edges)

        # Phase 4: Merge properties for existing entities
        merged_vertices = self._merge_vertex_properties(updated_vertices)

        # Phase 5: Commit changes to graph (if client available)
        if self.graph_client:
            self._commit_incremental(new_vertices, new_edges, merged_vertices, updated_edges, schema)

        # Phase 6: Record change log and update context
        change_summary = {
            "new_vertices": len(new_vertices),
            "updated_vertices": len(merged_vertices),
            "new_edges": len(new_edges),
            "updated_edges": len(updated_edges),
            "timestamp": datetime.now().isoformat(),
        }
        self._change_log.append(change_summary)

        # Store resolved vertex IDs in context for downstream consumers
        context["incremental_update_summary"] = change_summary
        context["new_vertices"] = new_vertices
        context["new_edges"] = new_edges
        context["merged_vertices"] = merged_vertices
        context["updated_edges"] = updated_edges
        context["entity_name_to_id"] = dict(self._entity_name_to_id)
        log.info(
            "Incremental update: %d new vertices, %d updated, %d new edges, %d updated",
            len(new_vertices),
            len(merged_vertices),
            len(new_edges),
            len(updated_edges),
        )
        return context

    def _build_entity_name_map(self, vertices: List[Dict[str, Any]]) -> None:
        """
        Build mapping from entity name to vertex ID.

        Uses entity name (from properties) as the natural key.
        This is the LightRAG core insight: entity name IS the identity.
        """
        for vertex in vertices:
            name = self._get_entity_name(vertex)
            vid = self._get_vertex_id(vertex)
            if name:
                self._entity_name_to_id[name] = vid
                # Also store lowercase for case-insensitive matching
                self._entity_name_to_id[name.lower()] = vid

    def _deduplicate_vertices(
        self, vertices: List[Dict[str, Any]], schema: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separate new vertices from those that already exist.

        Deduplication strategy (LightRAG-style):
        - Primary: entity name match (case-insensitive)
        - Secondary: vertex ID match (for schema-based ID)
        - If match found and properties differ → merge needed
        - If no match → new entity
        """
        new_vertices = []
        updated_vertices = []

        # Build existing entity name index from graph if client available
        existing_name_index: Dict[str, Dict[str, Any]] = {}
        if self.graph_client:
            existing_name_index = self._build_existing_entity_index()

        for vertex in vertices:
            entity_name = self._get_entity_name(vertex)
            vertex_id = self._get_vertex_id(vertex)

            # Check if entity already exists
            existing = None
            if entity_name and entity_name.lower() in existing_name_index:
                existing = existing_name_index[entity_name.lower()]
            elif vertex_id:
                existing = self._fetch_existing_vertex(vertex_id)

            if existing is None:
                new_vertices.append(vertex)
                # Register new entity name → ID mapping
                if entity_name:
                    self._entity_name_to_id[entity_name] = vertex_id
            else:
                # Entity exists — check if properties changed
                if self._properties_differ(vertex.get("properties", {}), existing.get("properties", {})):
                    vertex["_existing_properties"] = existing.get("properties", {})
                    vertex["_existing_id"] = existing.get("id", vertex_id)
                    updated_vertices.append(vertex)
                    # Update name → existing ID mapping
                    if entity_name:
                        self._entity_name_to_id[entity_name] = existing.get("id", vertex_id)
                else:
                    log.debug("Entity '%s' unchanged, skipping", entity_name or vertex_id)
                    # Still update name → ID mapping
                    if entity_name:
                        self._entity_name_to_id[entity_name] = existing.get("id", vertex_id)

        return new_vertices, updated_vertices

    def _deduplicate_edges(self, edges: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separate new edges from those that already exist.

        Edge deduplication strategy:
        - Key = (source_name, edge_label, target_name)
        - If both endpoints resolve to existing vertices → check if edge exists
        - New edge → append; existing with changed props → update
        """
        new_edges = []
        updated_edges = []

        for edge in edges:
            edge_key = self._get_edge_key(edge)
            existing = self._fetch_existing_edge(edge_key)

            if existing is None:
                # Resolve edge endpoints by entity name
                edge = self._resolve_edge_endpoints(edge)
                if edge is not None:
                    new_edges.append(edge)
            else:
                if self._properties_differ(edge.get("properties", {}), existing.get("properties", {})):
                    edge["_existing_properties"] = existing.get("properties", {})
                    updated_edges.append(edge)
                else:
                    log.debug("Edge '%s' unchanged, skipping", edge_key)

        return new_edges, updated_edges

    def _resolve_edge_endpoints(self, edge: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve edge endpoint vertex IDs from entity names.

        If an endpoint is referenced by name rather than ID,
        look up the ID from the entity_name_to_id map.
        """
        out_v = edge.get("outV", "")
        in_v = edge.get("inV", "")

        # Try to resolve by name if outV/inV look like names
        if out_v and out_v not in self._entity_name_to_id:
            resolved = self._entity_name_to_id.get(out_v.lower())
            if resolved:
                edge["outV"] = resolved

        if in_v and in_v not in self._entity_name_to_id:
            resolved = self._entity_name_to_id.get(in_v.lower())
            if resolved:
                edge["inV"] = resolved

        return edge

    def _merge_vertex_properties(self, updated_vertices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge new vertex properties with existing ones.

        Merge strategy (LightRAG-style):
        - New values override existing values (fresh data takes precedence)
        - Existing values not present in new data are preserved
        - List/set properties are unioned rather than overwritten
        """
        merged = []
        for vertex in updated_vertices:
            existing_props = vertex.pop("_existing_properties", {})
            existing_id = vertex.pop("_existing_id", None)
            new_props = vertex.get("properties", {})

            # Use existing ID to ensure update targets the right vertex
            if existing_id:
                vertex["id"] = existing_id

            # Smart merge: union list properties, override scalar properties
            merged_props = {}
            all_keys = set(existing_props.keys()) | set(new_props.keys())
            for key in all_keys:
                old_val = existing_props.get(key)
                new_val = new_props.get(key)

                if new_val is None:
                    merged_props[key] = old_val
                elif old_val is None:
                    merged_props[key] = new_val
                elif isinstance(old_val, list) and isinstance(new_val, list):
                    # Union lists (deduplicated)
                    union_list = list(set(str(x) for x in old_val) | set(str(x) for x in new_val))
                    merged_props[key] = union_list
                else:
                    # New value overrides
                    merged_props[key] = new_val

            vertex["properties"] = merged_props
            merged.append(vertex)

        return merged

    def _build_existing_entity_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Build an index of existing entities by name from the graph.

        Queries the graph for all vertices and indexes them by name property.
        This enables fast name-based deduplication.
        """
        index: Dict[str, Dict[str, Any]] = {}
        if not self.graph_client:
            return index

        try:
            result = self.graph_client.gremlin().exec(gremlin="g.V().has('name').elementMap()")
            for item in result.get("data", []):
                vertex = self._gremlin_result_to_vertex(item)
                name = vertex.get("properties", {}).get("name", "")
                if name:
                    index[name.lower()] = vertex
                    # Also update name → ID mapping
                    vid = vertex.get("id", "")
                    if vid:
                        self._entity_name_to_id[name] = vid
                        self._entity_name_to_id[name.lower()] = vid
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Failed to build entity index: %s", e)

        return index

    def _commit_incremental(
        self,
        new_vertices: List[Dict[str, Any]],
        new_edges: List[Dict[str, Any]],
        merged_vertices: List[Dict[str, Any]],
        updated_edges: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]],
    ) -> None:
        """
        Commit incremental changes to the HugeGraph database.

        Uses upsert semantics: create if not exists, update if exists.
        """
        try:
            from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import Commit2Graph

            committer = Commit2Graph()

            # Commit new vertices and edges
            if new_vertices or new_edges:
                data = {"vertices": new_vertices, "edges": new_edges, "schema": schema}
                committer.run(data)
                log.info("Committed %d new vertices and %d new edges", len(new_vertices), len(new_edges))

            # Commit merged (updated) vertices
            if merged_vertices:
                data = {"vertices": merged_vertices, "edges": [], "schema": schema}
                committer.run(data)
                log.info("Committed %d merged vertices", len(merged_vertices))

            # Commit updated edges
            if updated_edges:
                data = {"vertices": [], "edges": updated_edges, "schema": schema}
                committer.run(data)
                log.info("Committed %d updated edges", len(updated_edges))

        except Exception as e:  # pylint: disable=broad-except
            log.error("Failed to commit incremental updates: %s", e)

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the change log for auditability."""
        return list(self._change_log)

    def get_entity_name_to_id(self) -> Dict[str, str]:
        """Return the entity name → vertex ID mapping."""
        return dict(self._entity_name_to_id)

    # --- Helper methods ---

    @staticmethod
    def _get_entity_name(vertex: Dict[str, Any]) -> str:
        """Extract entity name from a vertex dict (LightRAG-style identity)."""
        # Priority: name property > title property > id
        props = vertex.get("properties", {})
        name = props.get("name", "")
        if not name:
            name = props.get("title", "")
        if not name:
            name = str(vertex.get("id", ""))
        return name

    @staticmethod
    def _get_vertex_id(vertex: Dict[str, Any]) -> str:
        """Extract or generate a vertex ID from a vertex dict."""
        vid = vertex.get("id")
        if vid:
            return str(vid)
        # LightRAG-style: use entity name as ID basis
        label = vertex.get("label", "unknown")
        name = vertex.get("properties", {}).get("name", "")
        if name:
            return f"{label}:{name}"
        return f"{label}:unknown"

    @staticmethod
    def _get_edge_key(edge: Dict[str, Any]) -> str:
        """Generate a unique key for an edge based on (source, label, target)."""
        label = edge.get("label", "unknown")
        out_v = edge.get("outV", "")
        in_v = edge.get("inV", "")
        return f"{out_v}-{label}-{in_v}"

    def _fetch_existing_vertex(self, vertex_id: str) -> Optional[Dict[str, Any]]:
        """Fetch an existing vertex from the graph by ID."""
        if not self.graph_client:
            return None
        try:
            result = self.graph_client.gremlin().exec(gremlin=f"g.V('{vertex_id}').elementMap()")
            data = result.get("data", [])
            if data:
                return self._gremlin_result_to_vertex(data[0])
        except Exception as e:  # pylint: disable=broad-except
            log.debug("Vertex '%s' not found or error: %s", vertex_id, e)
        return None

    def _fetch_existing_edge(self, edge_key: str) -> Optional[Dict[str, Any]]:
        """Fetch an existing edge from the graph by key."""
        # Edge lookup by composite key is complex;
        # for now, treat all edges as potentially new
        return None

    @staticmethod
    def _gremlin_result_to_vertex(gremlin_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Gremlin elementMap result to vertex dict."""
        props = {}
        for k, v in gremlin_result.items():
            if k not in ("id", "label", "~type", "~id"):
                props[k] = v
        return {
            "id": str(gremlin_result.get("id", "")),
            "label": gremlin_result.get("label", ""),
            "properties": props,
        }

    @staticmethod
    def _properties_differ(props_a: Dict[str, Any], props_b: Dict[str, Any]) -> bool:
        """Check if two property dicts differ."""
        return props_a != props_b
