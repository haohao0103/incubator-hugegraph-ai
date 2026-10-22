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

"""Service-backed graph loading - the production source of truth.

Why this module exists
----------------------
The offline JSON file is a *build artifact*: convenient for development, CI
and fully-offline runs, but it is NOT an authority multiple services/processes
can share. The production architecture is **services as the source of truth**:

* **HugeGraph** - authoritative graph *structure*: skill/category vertices and
  the five typed edge relations.
* **OceanBase** - authoritative *content + vectors*: each skill's full body
  text (``content`` column) and its embedding (HNSW-indexed ``embedding``
  column).
* ``skill_graph.json`` - an optional export of a build, used offline. Nothing
  in the online path needs it.

:func:`load_graph_from_services` assembles the in-memory
:class:`~...schema.SkillGraph` consumed by the retriever **directly from the
two backing stores** - vertices + edges from HugeGraph via Gremlin, then
embeddings + bodies hydrated from OceanBase. The result is a stateless query
service: any process, any replica, no shared file, no drift.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from hugegraph_llm.engines.skill_graph.errors import GraphReadError
from hugegraph_llm.engines.skill_graph.schema import (
    EdgeType,
    NodeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
)

log = logging.getLogger(__name__)

# Gremlin label filters mirroring the write path (hg_store.HugeGraphSkillStore).
_VERTEX_LABELS = "skill','category"
_EDGE_LABELS = "','".join(e.value for e in EdgeType)


class ServiceGraphCache:
    """Reuse an assembled snapshot until the active build pointer moves.

    Fetching the graph structure from HugeGraph and hydrating every vector/body
    from OceanBase costs O(graph) per query. In a long-running service the build
    changes only on publish/rollback, so the assembled graph can be reused.

    Correctness comes from the pointer check, not from a TTL: every call still
    reads the single ``pointer::active`` vertex (one tiny Gremlin query). When
    the pointer changes, the next call loads the new snapshot and the cache
    keeps it under its build id. A snapshot is immutable, so a graph cached
    before a rollback remains valid if the operator rolls forward again.

    Thread-safe: concurrent requests may assemble the same snapshot twice on a
    cold cache (harmless, last one wins) but never observe a partial graph.
    """

    def __init__(self, max_entries: int = 2):
        self.max_entries = max(1, int(max_entries))
        self._entries: "OrderedDict[str, SkillGraph]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, build_id: str) -> Optional[SkillGraph]:
        with self._lock:
            graph = self._entries.get(build_id)
            if graph is not None:
                self._entries.move_to_end(build_id)
            return graph

    def put(self, build_id: str, graph: SkillGraph) -> None:
        with self._lock:
            self._entries[build_id] = graph
            self._entries.move_to_end(build_id)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _parse_embedding(raw: Any) -> Optional[List[float]]:
    """OceanBase VECTOR columns come back as ``"[v1,v2,...]"`` strings."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    text = str(raw).strip()
    if not text or text == "NULL":
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except ValueError:
        pass
    return [float(x) for x in text.strip("[] ").split(",") if x.strip()]


def graph_from_gremlin_data(
    vertex_data: List[Dict[str, Any]],
    edge_data: List[Dict[str, Any]],
    id_prefix: Optional[str] = None,
) -> SkillGraph:
    """Assemble a :class:`SkillGraph` from HugeGraph Gremlin result dicts.

    Pure function (no I/O) so it is directly unit-testable. Accepts the
    ``result['data']`` list returned by ``PyHugeClient.gremlin().exec()``.
    Unknown vertex/edge labels are skipped rather than raising, so schema
    evolution on the server never breaks the loader.

    ``id_prefix`` strips a build-scoped id prefix (``1a2b3c4d:``) so callers see
    stable node ids - the ids OceanBase rows are keyed by.
    """
    def _stable(raw: Any) -> str:
        value = str(raw)
        if id_prefix and value.startswith(id_prefix):
            return value[len(id_prefix):]
        return value

    nodes: List[SkillNode] = []
    for vertex in vertex_data:
        props = vertex.get("properties") or {}
        raw_type = props.get("node_type") or vertex.get("label") or NodeType.SKILL.value
        try:
            node_type = NodeType(raw_type)
        except ValueError:
            continue
        tags = [t for t in str(props.get("tags") or "").split(",") if t]
        nodes.append(
            SkillNode(
                id=_stable(vertex["id"]),
                name=str(props.get("node_name") or _stable(vertex["id"]).split("::", 1)[-1]),
                description=str(props.get("description") or ""),
                category=str(props.get("category") or ""),
                tags=tags,
                node_type=node_type,
                source_path=str(props.get("source_path")) if props.get("source_path") else None,
            )
        )
    graph = SkillGraph(nodes=nodes)
    for edge in edge_data:
        try:
            edge_type = EdgeType(edge.get("label"))
        except ValueError:
            continue
        props = edge.get("properties") or {}
        graph.add_edge(
            SkillEdge(
                source=_stable(edge["outV"]),
                target=_stable(edge["inV"]),
                edge_type=edge_type,
                weight=float(props.get("weight", 1.0) or 1.0),
                provenance=str(props.get("provenance", "explicit")),
            )
        )
    return graph


def _fetch_legacy(client: Any) -> SkillGraph:
    """Read pre-versioning data (unversioned ids), for migration only."""
    vertices = client.gremlin().exec(f"g.V().hasLabel('{_VERTEX_LABELS}')")
    edges = client.gremlin().exec(f"g.E().hasLabel('{_EDGE_LABELS}')")
    return graph_from_gremlin_data(
        (vertices or {}).get("data", []),
        (edges or {}).get("data", []),
    )


def _open_hg_store(
    url: Optional[str] = None,
    user: str = "admin",
    pwd: str = "admin",
    graphspace: Optional[str] = None,
    graph_name: Optional[str] = None,
) -> Any:
    """Open a read-only HugeGraph store (no schema creation on the query path)."""
    from hugegraph_llm.engines.skill_graph.hg_store import (
        HugeGraphConfig,
        HugeGraphSkillStore,
    )

    config = HugeGraphConfig.from_env(
        url=url, user=user, pwd=pwd, graphspace=graphspace, graph=graph_name
    )
    # Read paths must not create schema: that needs write permission and would
    # mutate the cluster just to serve a query. Only ``publish`` creates labels.
    return HugeGraphSkillStore(config, create_schema=False)


def fetch_graph_from_hugegraph(
    url: Optional[str] = None,
    user: str = "admin",
    pwd: str = "admin",
    graphspace: Optional[str] = None,
    graph_name: Optional[str] = None,
    allow_legacy: bool = True,
    store: Optional[Any] = None,
    pointer: Optional[Any] = None,
) -> SkillGraph:
    """Pull the authoritative graph structure from HugeGraph.

    Resolves the ``pointer::active`` vertex and reads exactly that build. If no
    pointer exists (pre-versioning deployment), falls back to the legacy
    unversioned vertices unless ``allow_legacy=False`` - production should run
    with a pointer so reads are snapshot-consistent.

    ``store`` / ``pointer`` let a caller that already resolved the pointer reuse
    both, so a cache hit costs one tiny Gremlin query and a cold load does not
    resolve the pointer twice.
    """
    if store is None:
        store = _open_hg_store(url, user, pwd, graphspace, graph_name)
    if pointer is None:
        pointer = store.active_pointer()
    if pointer is not None:
        return store.read_build(pointer.build_id)
    if not allow_legacy:
        raise GraphReadError(
            "HugeGraph has no active skill-graph build pointer; publish a build first"
        )
    log.warning(
        "HugeGraph has no active build pointer; reading LEGACY unversioned data. "
        "Publish a build to enable atomic reads/rollbacks."
    )
    graph = _fetch_legacy(store.client)
    log.info("fetched legacy skill graph from HugeGraph: %s", graph.stats())
    return graph


def hydrate_from_oceanbase(
    graph: SkillGraph,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    table: Optional[str] = None,
) -> int:
    """Fill each node's ``embedding`` and ``body`` from OceanBase rows.

    Returns the number of nodes hydrated. Nodes present in HugeGraph but
    missing from OceanBase (e.g. category hubs, or a stale write) keep their
    in-memory defaults - the retriever degrades gracefully to the lexical
    channel for them.

    The table defaults to the one recorded in the build manifest
    (``<base>__<build>``). This is what keeps a reader on a consistent
    snapshot: graph structure comes from the active build and vectors must come
    from *that same build's* table, never from whichever table happens to be
    the current default.
    """
    from hugegraph_llm.engines.skill_graph.ob_store import OceanBaseSkillStore

    manifest = getattr(graph, "manifest", None)
    manifest_table = getattr(manifest, "ob_table", None) if manifest is not None else None
    if table is None:
        table = manifest_table
    elif manifest_table and table != manifest_table:
        raise GraphReadError(
            f"OceanBase table {table!r} does not match the active build's table "
            f"{manifest_table!r}; vectors from another build are not a consistent snapshot"
        )

    store = OceanBaseSkillStore(
        host=host, port=port, user=user, password=password, database=database, table=table
    )
    try:
        with store.connection.cursor() as cur:
            cur.execute(
                f"SELECT node_id, embedding, content FROM `{store.table}` WHERE is_valid = 1"
            )
            rows = cur.fetchall()
    finally:
        store.close()

    hydrated = 0
    for node_id, embedding, content in rows:
        node = graph.nodes.get(str(node_id))
        if node is None:
            continue
        node.embedding = _parse_embedding(embedding)
        if content:
            node.body = str(content)
        hydrated += 1
    missing = sum(
        1
        for node in graph.nodes.values()
        if node.node_type == NodeType.SKILL and not node.embedding
    )
    if missing:
        # Per-node degradation is allowed (the lexical channel still finds the
        # skill) but must be visible: a missing vector is a data problem, not a
        # silent switch of search quality.
        log.warning(
            "%d skill(s) have no embedding in OceanBase table %s; "
            "they are searchable lexically only",
            missing,
            store.table,
        )
    log.info("hydrated %d/%d nodes from OceanBase", hydrated, len(graph.nodes))
    return hydrated


#: Process-wide snapshot cache. Two entries keep the previous build warm, so a
#: rollback (or rolling forward again) does not re-read the whole graph.
_DEFAULT_CACHE = ServiceGraphCache(max_entries=2)


def _cache_key(store: Any, pointer: Any, ob_host: Optional[str], ob_database: Optional[str]) -> str:
    """Identify a snapshot by deployment target *and* build.

    Build ids are content hashes, so two deployments can legitimately share one
    while holding different data (different clusters/databases). Keying on the
    build id alone would serve one deployment's graph to another.
    """
    config = store.config
    parts = [
        config.url,
        config.graphspace or "",
        config.graph,
        ob_host or "",
        ob_database or "",
        pointer.build_id,
    ]
    return "|".join(str(p) for p in parts)


def load_graph_from_services(
    hg_url: Optional[str] = None,
    hg_user: str = "admin",
    hg_pwd: str = "admin",
    hg_graphspace: Optional[str] = None,
    hg_graph: Optional[str] = None,
    ob_host: Optional[str] = None,
    ob_port: Optional[int] = None,
    ob_user: Optional[str] = None,
    ob_password: Optional[str] = None,
    ob_database: Optional[str] = None,
    ob_table: Optional[str] = None,
    cache: Optional[ServiceGraphCache] = None,
    force_refresh: bool = False,
) -> SkillGraph:
    """Assemble the retrieval-ready graph from HugeGraph + OceanBase.

    Production entry point - no local JSON file involved. Structure comes from
    HugeGraph (authority), embeddings and bodies from OceanBase (authority).

    The assembled graph is cached per ``(deployment, build)``: every call still
    reads ``pointer::active`` (a single tiny Gremlin query), and only a pointer
    change triggers the O(graph) reload. ``cache=None`` uses the process-wide
    cache; pass an explicit instance to isolate tenants/tests, or
    ``force_refresh=True`` to bypass a hit. Legacy (pointerless) deployments are
    not cached, since they have no snapshot identity.
    """
    snapshot_cache = _DEFAULT_CACHE if cache is None else cache
    store = _open_hg_store(hg_url, hg_user, hg_pwd, hg_graphspace, hg_graph)
    pointer = store.active_pointer()

    key: Optional[str] = None
    if pointer is not None:
        key = _cache_key(store, pointer, ob_host, ob_database)
        if not force_refresh:
            cached = snapshot_cache.get(key)
            if cached is not None:
                log.debug("serving skill graph build %s from snapshot cache", pointer.build_id)
                return cached

    graph = fetch_graph_from_hugegraph(
        url=hg_url,
        user=hg_user,
        pwd=hg_pwd,
        graphspace=hg_graphspace,
        graph_name=hg_graph,
        store=store,
        pointer=pointer,
    )
    hydrate_from_oceanbase(
        graph,
        host=ob_host, port=ob_port, user=ob_user, password=ob_password,
        database=ob_database, table=ob_table,
    )
    if key is not None:
        snapshot_cache.put(key, graph)
        log.info(
            "loaded skill graph build %s from services: %s", pointer.build_id, graph.stats()
        )
    return graph