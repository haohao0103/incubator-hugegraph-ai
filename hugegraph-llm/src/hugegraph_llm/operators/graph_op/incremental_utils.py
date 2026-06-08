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

"""Incremental indexing utilities for HugeGraph-AI GraphRAG.

Provides:
- persist_community_assignments: Write community_id as vertex property
- find_affected_communities: Detect which communities are impacted by new vertices
- rebuild_affected_communities: Partially rebuild community reports for affected areas

These utilities enable Sprint 2 (Incremental Indexing) by allowing
new documents to be indexed without full graph reconstruction.
"""

from typing import Any, Dict, List, Optional, Set

from hugegraph_llm.utils.log import log


# ── Community assignment persistence ──────────────────────────


def persist_community_assignments(
    client: Any,
    communities: List[Dict[str, Any]],
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Write community_id as a vertex property in HugeGraph.

    After community detection, this persists the community assignments
    so that incremental indexing can later query which community
    a vertex belongs to.

    Args:
        client: HugeGraph PyHugeClient instance.
        communities: List of community dicts from CommunityDetect.run().
            Each dict must have "id" and "vertices" (list of vertex IDs).
        batch_size: Number of vertices to update per Gremlin batch.

    Returns:
        Dict with "updated_count", "errors", "skipped".
    """
    result = {"updated_count": 0, "errors": [], "skipped": 0}

    # Ensure community_id property key exists on all vertex labels
    try:
        _ensure_community_id_property(client)
    except Exception as e:
        log.warning("Failed to ensure community_id property: %s", e)
        result["errors"].append(f"property_setup: {e}")

    for community in communities:
        comm_id = community.get("id", "")
        if not comm_id:
            continue

        vertices = community.get("vertices", [])
        if not vertices:
            continue

        # Batch update: set community_id property on each vertex
        for i in range(0, len(vertices), batch_size):
            batch = vertices[i : i + batch_size]
            vid_list = ", ".join(f"'{vid}'" for vid in batch)
            groovy = f"""
            g.V({vid_list}).has('community_id', neq('{comm_id}'))
                .property('community_id', '{comm_id}').count()
            """
            try:
                resp = client.gremlin().exec(groovy)
                count = _parse_gremlin_count(resp)
                result["updated_count"] += count
            except Exception as e:
                log.warning(
                    "Failed to persist community_id=%s for batch (offset=%d): %s",
                    comm_id, i, e,
                )
                result["errors"].append(f"batch_{comm_id}_{i}: {e}")
                result["skipped"] += len(batch)

    log.info(
        "Persisted community assignments: %d vertices updated, %d errors, %d skipped",
        result["updated_count"],
        len(result["errors"]),
        result["skipped"],
    )
    return result


def _ensure_community_id_property(client: Any) -> None:
    """Ensure community_id property key exists on all vertex labels.

    Creates the property key if it does not exist. Uses ifNotExist()
    for idempotency.
    """
    schema = client.schema()

    # Create property key (if not exists)
    try:
        schema.propertyKey("community_id").ifNotExist().asText().create()
    except Exception as e:
        # May already exist — that's fine
        log.debug("propertyKey 'community_id' create skipped: %s", e)


def clear_stale_community_assignments(
    client: Any,
    keep_community_ids: Optional[Set[str]] = None,
) -> int:
    """Remove community_id from vertices whose community no longer exists.

    After a full community detection run, some vertices may have been
    reassigned. This clears stale assignments before the new ones
    are written.

    Args:
        client: HugeGraph PyHugeClient instance.
        keep_community_ids: Set of valid community IDs to preserve.
            If None, clears ALL community_id assignments.

    Returns:
        Number of vertices updated.
    """
    groovy = "g.V().has('community_id').count()"
    try:
        total_resp = client.gremlin().exec(groovy)
        total = _parse_gremlin_count(total_resp)
    except Exception as e:
        log.warning("Failed to count vertices with community_id: %s", e)
        total = 0

    if keep_community_ids is not None and keep_community_ids:
        # Only clear vertices whose community_id is NOT in keep set
        allowed_list = ", ".join(f"'{cid}'" for cid in keep_community_ids)
        groovy = f"""
        g.V().has('community_id', without('{allowed_list}'))
            .property('community_id', '').count()
        """
    else:
        groovy = "g.V().has('community_id').property('community_id', '').count()"

    try:
        resp = client.gremlin().exec(groovy)
        cleared = _parse_gremlin_count(resp)
        log.info("Cleared %d/%d stale community assignments", cleared, total)
        return cleared
    except Exception as e:
        log.warning("Failed to clear stale community assignments: %s", e)
        return 0


# ── Affected community detection ─────────────────────────────


def find_affected_communities(
    client: Any,
    new_vertex_ids: List[str],
    hop: int = 1,
) -> Set[str]:
    """Find communities affected by newly added vertices.

    Logic:
    1. From each new vertex, traverse N hops to find all neighbors
    2. Collect community_id from all neighbors
    3. Include any community that contains at least one neighbor

    Args:
        client: HugeGraph PyHugeClient instance.
        new_vertex_ids: List of newly added vertex IDs.
        hop: Number of hops to traverse (default: 1).

    Returns:
        Set of affected community IDs.
    """
    if not new_vertex_ids:
        return set()

    affected: Set[str] = set()

    # Batch vertices into groups to avoid overly long Gremlin queries
    batch_size = 100
    for i in range(0, len(new_vertex_ids), batch_size):
        batch = new_vertex_ids[i : i + batch_size]
        vid_list = ", ".join(f"'{vid}'" for vid in batch)

        # Find neighbors within N hops that have community_id
        hop_chain = ".both()" * hop
        groovy = f"""
        g.V({vid_list}){hop_chain}
            .has('community_id')
            .values('community_id')
            .dedup()
            .toList()
        """
        try:
            resp = client.gremlin().exec(groovy)
            community_ids = _parse_gremlin_list(resp)
            affected.update(community_ids)
        except Exception as e:
            log.warning("Failed to detect affected communities for batch (offset=%d): %s", i, e)

    # Also check if any new vertex itself has been assigned to a community
    new_vid_list = ", ".join(f"'{vid}'" for vid in new_vertex_ids)
    groovy = f"""
    g.V({new_vid_list}).has('community_id')
        .values('community_id').dedup().toList()
    """
    try:
        resp = client.gremlin().exec(groovy)
        self_communities = _parse_gremlin_list(resp)
        affected.update(self_communities)
    except Exception as e:
        log.warning("Failed to detect self-community assignments: %s", e)

    log.info(
        "Detected %d affected communities from %d new vertices (hop=%d)",
        len(affected), len(new_vertex_ids), hop,
    )
    return affected


# ── Affected community partial rebuild ────────────────────────


def get_community_vertices(
    client: Any,
    community_ids: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all vertices belonging to the given communities.

    Args:
        client: HugeGraph PyHugeClient instance.
        community_ids: Set of community IDs to fetch.

    Returns:
        Dict mapping community_id -> list of vertex dicts.
    """
    if not community_ids:
        return {}

    result: Dict[str, List[Dict[str, Any]]] = {}

    cid_list = ", ".join(f"'{cid}'" for cid in community_ids)
    groovy = f"""
    g.V().has('community_id', within({cid_list}))
        .project('id', 'label', 'community_id')
        .by(id()).by(label()).by(values('community_id'))
        .toList()
    """
    try:
        resp = client.gremlin().exec(groovy)
        vertices = _parse_gremlin_vertex_list(resp)

        for v in vertices:
            cid = v.get("community_id", "")
            if cid:
                result.setdefault(cid, []).append(v)
    except Exception as e:
        log.error("Failed to fetch community vertices: %s", e)

    return result


def get_community_edges(
    client: Any,
    vertex_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Fetch all edges between the given vertices.

    Args:
        client: HugeGraph PyHugeClient instance.
        vertex_ids: Set of vertex IDs.

    Returns:
        List of edge dicts with outV, inV, label, properties.
    """
    if not vertex_ids:
        return []

    edges = []
    batch_size = 100

    vid_list = list(vertex_ids)
    for i in range(0, len(vid_list), batch_size):
        batch = vid_list[i : i + batch_size]
        id_list = ", ".join(f"'{vid}'" for vid in batch)
        groovy = f"""
        g.V({id_list}).bothE()
            .where(otherV().hasId(within({id_list})))
            .project('label', 'outV', 'inV', 'properties')
            .by(label())
            .by(inV().id())
            .by(outV().id())
            .by(valueMap().by(unfold()))
            .limit({batch_size * 50})
            .toList()
        """
        try:
            resp = client.gremlin().exec(groovy)
            edge_list = _parse_gremlin_vertex_list(resp)
            edges.extend(edge_list)
        except Exception as e:
            log.warning("Failed to fetch community edges for batch (offset=%d): %s", i, e)

    return edges


# ── Gremlin response parsers ──────────────────────────────────


def _parse_gremlin_count(resp: Any) -> int:
    """Parse a Gremlin count response into an integer."""
    if isinstance(resp, dict):
        data = resp.get("data", [])
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return int(item.get("result", item.get("count", 0)))
            if isinstance(item, list):
                # Nested list: unwrap first element
                item = item[0] if item else 0
            return int(item) if item else 0
        if isinstance(data, (int, float)):
            return int(data)
        return 0
    if isinstance(resp, (int, float)):
        return int(resp)
    return 0


def _parse_gremlin_list(resp: Any) -> List[str]:
    """Parse a Gremlin list response into a list of strings."""
    if isinstance(resp, dict):
        data = resp.get("data", [])
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return [str(item.get("result", item.get("value", "")))]
            if isinstance(item, list):
                return [str(v) for v in item]
        if isinstance(data, list):
            return [str(v) for v in data if v is not None]
    if isinstance(resp, list):
        return [str(v) for v in resp if v is not None]
    return []


def _parse_gremlin_vertex_list(resp: Any) -> List[Dict[str, Any]]:
    """Parse a Gremlin project() response into a list of dicts."""
    if isinstance(resp, dict):
        data = resp.get("data", [])
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return [item]
            if isinstance(item, list):
                return [v for v in item if isinstance(v, dict)]
        return [v for v in data if isinstance(v, dict)]
    if isinstance(resp, list):
        return [v for v in resp if isinstance(v, dict)]
    return []
