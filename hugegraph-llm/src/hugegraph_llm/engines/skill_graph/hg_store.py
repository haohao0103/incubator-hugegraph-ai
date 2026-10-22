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

"""Versioned HugeGraph storage and the active-build pointer.

Why versioned ids instead of stable ids
---------------------------------------
Publishing a new build must not mutate the graph readers are using. HugeGraph
cannot move a vertex (and ``addVertex`` on an existing id is a conflict), so
each build writes its vertices under **build-scoped ids**::

    <short-build-id>:skill::rag-pipeline
    <short-build-id>:category::generation

A build therefore owns a disjoint subgraph. Readers resolve the active build
through the ``pointer::active`` vertex and only read ids carrying that prefix.
The switch is one property update - the commit point - and rollback is the
inverse update. No reader ever observes a half-written graph.

This avoids an in-place schema migration: ``skill``/``category`` labels stay as
they are, and old unversioned data can coexist during the transition.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hugegraph_llm.engines.skill_graph.errors import (
    BuildValidationError,
    GraphReadError,
    GraphWriteError,
    PublishError,
)
from hugegraph_llm.engines.skill_graph.manifest import (
    ActivePointer,
    BuildManifest,
    build_scoped_table,
    content_hash,
    short_build_id,
)
from hugegraph_llm.engines.skill_graph.schema import EdgeType, NodeType, SkillGraph

log = logging.getLogger(__name__)

DEFAULT_GRAPH = "skillgraph"
META_LABEL = "graph_meta"
POINTER_ID = "pointer::active"
_BATCH = 100


@dataclass(frozen=True)
class HugeGraphConfig:
    url: str = "http://127.0.0.1:8080"
    user: str = "admin"
    pwd: str = "admin"
    graphspace: Optional[str] = None
    graph: str = DEFAULT_GRAPH

    @classmethod
    def from_env(
        cls,
        url: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        graphspace: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> "HugeGraphConfig":
        return cls(
            url=url or os.environ.get("SKILL_GRAPH_HG_URL", "http://127.0.0.1:8080"),
            user=user or os.environ.get("SKILL_GRAPH_HG_USER", "admin"),
            pwd=pwd or os.environ.get("SKILL_GRAPH_HG_PWD", "admin"),
            graphspace=graphspace if graphspace is not None
            else (os.environ.get("SKILL_GRAPH_HG_GRAPHSPACE") or None),
            graph=graph or os.environ.get("SKILL_GRAPH_HG_GRAPH", DEFAULT_GRAPH),
        )


def versioned_id(build_id: str, node_id: str) -> str:
    """Build-scoped vertex id (the only place this prefix rule lives)."""
    return f"{short_build_id(build_id)}:{node_id}"


def _prop(props: Dict[str, Any], key: str) -> Any:
    """Normalise HugeGraph vertex properties (str, [obj] or {value: ...})."""
    value = props.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("value")
    return value


def _gremlin_ids(ids: Sequence[str]) -> str:
    return ",".join(json.dumps(i) for i in ids)


_EDGE_LABELS = {e.value for e in EdgeType}


def _validated_edge_labels(edge_types: Optional[Sequence[Any]]) -> List[str]:
    """Edge labels are interpolated into Gremlin; accept only known ontology names."""
    labels: List[str] = []
    for value in edge_types or (EdgeType.DEPENDS_ON, EdgeType.COMPOSES_WITH):
        label = value.value if isinstance(value, EdgeType) else str(value)
        if label not in _EDGE_LABELS:
            raise ValueError(f"unknown edge type {label!r}")
        labels.append(label)
    return labels


class HugeGraphSkillStore:
    """Versioned graph writer/reader with an atomic active-build pointer."""

    def __init__(self, config: Optional[HugeGraphConfig] = None, create_schema: bool = True):
        self.config = config or HugeGraphConfig.from_env()
        self._client = None
        if create_schema:
            self.ensure_schema()

    @property
    def client(self):
        if self._client is None:
            from pyhugegraph.client import PyHugeClient

            self._client = PyHugeClient(
                url=self.config.url,
                user=self.config.user,
                pwd=self.config.pwd,
                graph=self.config.graph,
                graphspace=self.config.graphspace,
            )
        return self._client

    def _gremlin(self, script: str) -> Dict[str, Any]:
        result = self.client.gremlin().exec(script)
        if not isinstance(result, dict):
            raise GraphReadError(f"HugeGraph returned an invalid Gremlin payload: {result!r}")
        return result

    # ── schema ───────────────────────────────────────────────
    def ensure_schema(self) -> None:
        """Create the structural labels if absent (idempotent, never destructive)."""
        schema = self.client.schema()
        for pk, ptype in (
            ("node_name", "TEXT"),
            ("description", "TEXT"),
            ("category", "TEXT"),
            ("tags", "TEXT"),
            ("node_type", "TEXT"),
            ("source_path", "TEXT"),
            ("token_estimate", "INT"),
            ("weight", "DOUBLE"),
            ("provenance", "TEXT"),
            ("kind", "TEXT"),
            ("payload", "TEXT"),
            ("created_at", "TEXT"),
        ):
            getattr(schema.propertyKey(pk), f"as{ptype.capitalize()}")().ifNotExist().create()

        node_props = [
            "node_name",
            "description",
            "category",
            "tags",
            "node_type",
            "source_path",
            "token_estimate",
        ]
        for label in ("skill", "category"):
            schema.vertexLabel(label).properties(*node_props).useCustomizeStringId().ifNotExist().create()
        # Metadata/pointer vertices are small JSON documents; a dedicated label
        # keeps them out of every skill query.
        schema.vertexLabel(META_LABEL).properties("kind", "payload", "created_at") \
            .useCustomizeStringId().ifNotExist().create()

        for label, (src, tgt) in {
            "depends_on": ("skill", "skill"),
            "composes_with": ("skill", "skill"),
            "similar_to": ("skill", "skill"),
            "conflicts_with": ("skill", "skill"),
            "belongs_to": ("skill", "category"),
        }.items():
            try:
                schema.edgeLabel(label).properties("weight", "provenance") \
                    .sourceLabel(src).targetLabel(tgt).ifNotExist().create()
            except Exception as exc:  # label may pre-exist with another shape
                log.warning("edge label %s not (re)created: %s", label, exc)
        log.info("ensured HugeGraph schema on %s/%s", self.config.url, self.config.graph)

    # ── build writes ─────────────────────────────────────────
    def write_build(self, graph: SkillGraph, build_id: str) -> Dict[str, int]:
        """Write the whole graph under ``build_id`` (no pointer change).

        Any failed vertex/edge aborts the publish: a build that silently lost
        vertices would otherwise become ``active`` and serve wrong results.
        """
        g = self.client.graph()
        failures: List[str] = []
        vertex_ids: List[str] = []

        for node in graph.nodes.values():
            label = "skill" if node.node_type == NodeType.SKILL else "category"
            vid = versioned_id(build_id, node.id)
            props = {
                "node_name": node.name,
                "description": node.description,
                "category": node.category,
                "tags": ",".join(node.tags),
                "node_type": node.node_type.value,
                "source_path": node.source_path or "",
                "token_estimate": node.token_estimate(),
            }
            try:
                g.addVertex(label, props, vid)
                vertex_ids.append(vid)
            except Exception as exc:
                failures.append(f"vertex {vid}: {exc}")

        edge_count = 0
        for edge in graph.edges:
            try:
                g.addEdge(
                    edge.edge_type.value,
                    versioned_id(build_id, edge.source),
                    versioned_id(build_id, edge.target),
                    {"weight": float(edge.weight), "provenance": edge.provenance},
                )
                edge_count += 1
            except Exception as exc:
                failures.append(f"edge {edge.key()}: {exc}")

        if failures:
            # Remove the partially written subgraph now: the build has no
            # manifest yet, so clear_build() cannot find it and a retry of the
            # same build id would collide with the orphaned vertices.
            for vid in vertex_ids:
                try:
                    g.removeVertexById(vid)
                except Exception as cleanup_exc:  # pragma: no cover - best effort
                    log.warning("failed to clean up vertex %s: %s", vid, cleanup_exc)
            preview = "; ".join(failures[:5])
            raise GraphWriteError(
                f"{len(failures)} write(s) failed for build {build_id} ({preview}"
                f"{'...' if len(failures) > 5 else ''}). The build was NOT completed and "
                "its partial vertices were removed; retry the publish."
            )
        return {"vertices": len(vertex_ids), "edges": edge_count, "vertex_ids": vertex_ids}

    def write_manifest(self, manifest: BuildManifest, vertex_ids: Sequence[str]) -> None:
        """Persist the manifest last: its presence certifies a complete build."""
        payload = manifest.to_dict()
        payload["vertex_ids"] = list(vertex_ids)
        self._put_meta(f"meta::{short_build_id(manifest.build_id)}", "build", payload)

    def _put_meta(self, vertex_id: str, kind: str, payload: Dict[str, Any]) -> None:
        g = self.client.graph()
        props = {
            "kind": kind,
            "payload": json.dumps(payload, ensure_ascii=False),
            "created_at": str(payload.get("created_at") or payload.get("activated_at") or ""),
        }
        try:
            g.addVertex(META_LABEL, props, vertex_id)
        except Exception:
            # Exists already: overwrite the pointer/manifest in one update.
            try:
                g.appendVertex(vertex_id, props)
            except Exception as exc:
                raise GraphWriteError(f"failed to update {vertex_id}: {exc}") from exc

    # ── build reads ──────────────────────────────────────────
    def read_manifest(self, build_id: str) -> Optional[BuildManifest]:
        data = self._read_meta(f"meta::{short_build_id(build_id)}")
        if data is None:
            return None
        data.pop("vertex_ids", None)
        return BuildManifest.from_dict(data).validate()

    def read_build(self, build_id: str) -> SkillGraph:
        """Read exactly one build's subgraph (ids carry its prefix)."""
        from hugegraph_llm.engines.skill_graph.service_loader import graph_from_gremlin_data

        meta = self._read_meta(f"meta::{short_build_id(build_id)}")
        if meta is None:
            raise GraphReadError(
                f"build {build_id} is not present in HugeGraph (no manifest vertex); "
                "publish it first or check the build id"
            )
        manifest = BuildManifest.from_dict({k: v for k, v in meta.items() if k != "vertex_ids"}).validate()
        ids = list(meta.get("vertex_ids") or [])
        vertices: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_edges = set()
        for start in range(0, len(ids), _BATCH):
            chunk = ids[start : start + _BATCH]
            if not chunk:
                continue
            vertices.extend(self._gremlin(f"g.V({_gremlin_ids(chunk)})").get("data", []))
            for edge in self._gremlin(f"g.V({_gremlin_ids(chunk)}).bothE()").get("data", []):
                key = edge.get("id") or (edge.get("label"), edge.get("outV"), edge.get("inV"))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(edge)

        # Strip the build prefix so callers see stable ids - the ids OceanBase
        # rows are keyed by. Without this, hydration silently matches nothing.
        graph = graph_from_gremlin_data(
            vertices, edges, id_prefix=f"{short_build_id(build_id)}:"
        )
        graph.manifest = manifest
        if graph.stats()["nodes"] != manifest.skills + manifest.categories:
            raise GraphReadError(
                f"build {build_id} is incomplete: manifest expects "
                f"{manifest.skills + manifest.categories} vertices but HugeGraph returned "
                f"{graph.stats()['nodes']}"
            )
        if graph.stats()["edges"] != manifest.edges:
            raise GraphReadError(
                f"build {build_id} is incomplete: manifest expects "
                f"{manifest.edges} edges but HugeGraph returned {graph.stats()['edges']}"
            )
        return graph

    def expand_prerequisites(
        self,
        seed_ids: Sequence[str],
        build_id: str,
        max_depth: int = 2,
        max_nodes: int = 200,
        max_edges: Optional[int] = None,
        edge_types: Optional[Sequence[Any]] = None,
    ) -> SkillGraph:
        """Read the prerequisite closure of ``seed_ids`` only (bounded subgraph).

        Seeds are the *stable* ids returned by OceanBase. Expansion follows
        outgoing edges (``source -> prerequisite``) breadth-first and stops at
        ``max_depth`` / ``max_nodes``, then reads up to ``max_edges`` edges whose
        both endpoints are inside that candidate set. Query cost is therefore
        O(candidates), independent of the total graph size - unlike
        :meth:`read_build`, which materialises a whole build.

        The returned graph contains stable ids (the build prefix is stripped),
        so it can be joined with OceanBase rows directly.
        """
        from hugegraph_llm.engines.skill_graph.service_loader import graph_from_gremlin_data

        if not seed_ids:
            return SkillGraph()
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        cap = max(int(max_nodes), len(seed_ids))
        edge_cap = int(max_edges) if max_edges is not None else cap * 4

        labels = _validated_edge_labels(edge_types)
        label_args = _gremlin_ids(labels)
        prefix = f"{short_build_id(build_id)}:"

        frontier = list(dict.fromkeys(versioned_id(build_id, sid) for sid in seed_ids))
        vertices: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(frontier), _BATCH):
            chunk = frontier[start : start + _BATCH]
            for vertex in self._gremlin(f"g.V({_gremlin_ids(chunk)})").get("data", []):
                vertices[str(vertex.get("id"))] = vertex
        seen = set(vertices)

        for _ in range(max_depth):
            room = cap - len(seen)
            if not frontier or room <= 0:
                break
            reached = self._gremlin(
                f"g.V({_gremlin_ids(frontier)}).out({label_args}).dedup().limit({room})"
            ).get("data", [])
            nxt: List[str] = []
            for vertex in reached:
                vid = str(vertex.get("id"))
                if vid in seen:
                    continue
                seen.add(vid)
                vertices[vid] = vertex
                nxt.append(vid)
            frontier = nxt

        # Induced edges among the candidate set keep PPR accurate without
        # reading any edge that points outside the closure.
        edges: Dict[Any, Dict[str, Any]] = {}
        ordered_ids = list(seen)
        for start in range(0, len(ordered_ids), _BATCH):
            remaining = edge_cap - len(edges)
            if remaining <= 0:
                break
            chunk = ordered_ids[start : start + _BATCH]
            data = self._gremlin(
                f"g.V({_gremlin_ids(chunk)}).bothE({label_args}).limit({remaining})"
            ).get("data", [])
            for edge in data:
                out_v, in_v = str(edge.get("outV")), str(edge.get("inV"))
                if out_v not in seen or in_v not in seen:
                    continue
                key = edge.get("id") or (edge.get("label"), out_v, in_v)
                edges.setdefault(key, edge)

        graph = graph_from_gremlin_data(
            list(vertices.values()), list(edges.values()), id_prefix=prefix
        )
        log.info(
            "expanded %d seed(s) of build %s to %d node(s)/%d edge(s) (depth<=%d)",
            len(seed_ids),
            build_id,
            len(graph.nodes),
            len(graph.edges),
            max_depth,
        )
        return graph

    def _read_meta(self, vertex_id: str) -> Optional[Dict[str, Any]]:
        data = self._gremlin(f"g.V({json.dumps(vertex_id)})").get("data", [])
        if not data:
            return None
        props = data[0].get("properties") or {}
        raw = _prop(props, "payload")
        if not raw:
            return None
        try:
            payload = json.loads(str(raw))
        except ValueError as exc:
            raise GraphReadError(f"corrupt metadata payload on {vertex_id}: {exc}") from exc
        return payload if isinstance(payload, dict) else None

    # ── active pointer ───────────────────────────────────────
    def activate(self, pointer: ActivePointer) -> None:
        """The commit point: readers switch to this build atomically."""
        pointer.validate()
        self._put_meta(POINTER_ID, "pointer", pointer.to_dict())
        log.info("activated skill-graph build %s", pointer.build_id)

    def active_pointer(self) -> Optional[ActivePointer]:
        data = self._read_meta(POINTER_ID)
        if data is None:
            return None
        return ActivePointer.from_dict(data).validate()

    def deactivate(self) -> None:
        """Remove the pointer without touching build data (maintenance only)."""
        try:
            self.client.graph().removeVertexById(POINTER_ID)
        except Exception as exc:
            raise GraphWriteError(f"failed to remove {POINTER_ID}: {exc}") from exc

    # ── maintenance ──────────────────────────────────────────
    def list_builds(self) -> List[Dict[str, Any]]:
        data = self._gremlin(
            f"g.V().hasLabel('{META_LABEL}').has('kind', 'build')"
        ).get("data", [])
        builds: List[Dict[str, Any]] = []
        for vertex in data:
            props = vertex.get("properties") or {}
            raw = _prop(props, "payload")
            try:
                payload = json.loads(str(raw)) if raw else {}
            except ValueError:
                payload = {}
            builds.append(payload)
        return builds

    def clear_build(self, build_id: str) -> int:
        """Delete a non-active build and its manifest. Refuses the active one."""
        active = self.active_pointer()
        if active is not None and active.build_id == build_id:
            raise BuildValidationError(
                f"refusing to delete the active build {build_id}; roll back to another build first"
            )
        meta = self._read_meta(f"meta::{short_build_id(build_id)}")
        if meta is None:
            return 0
        ids: Iterable[str] = meta.get("vertex_ids") or []
        deleted = 0
        failures: List[str] = []
        for vid in ids:
            try:
                self.client.graph().removeVertexById(vid)
                deleted += 1
            except Exception as exc:
                failures.append(f"{vid}: {exc}")
        try:
            self.client.graph().removeVertexById(f"meta::{short_build_id(build_id)}")
        except Exception as exc:
            failures.append(f"manifest: {exc}")
        if failures:
            raise GraphWriteError(
                f"failed to fully delete build {build_id}: {'; '.join(failures[:5])}"
            )
        log.info("deleted build %s (%d vertices)", build_id, deleted)
        return deleted


def publish_graph(
    graph: SkillGraph,
    *,
    hg_config: Optional[HugeGraphConfig] = None,
    ob_host: Optional[str] = None,
    ob_port: Optional[int] = None,
    ob_user: Optional[str] = None,
    ob_password: Optional[str] = None,
    ob_database: Optional[str] = None,
    ob_table_base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Write a built graph to both services and activate it atomically.

    The graph must already carry a validated manifest (see
    :class:`~...graph_builder.GraphBuilder`). The content hash is re-checked
    here so a graph mutated after the build cannot be published under a stale
    fingerprint.
    """
    from hugegraph_llm.engines.skill_graph.ob_store import (
        DEFAULT_TABLE,
        OceanBaseSkillStore,
        write_to_oceanbase,
    )

    manifest = getattr(graph, "manifest", None)
    if manifest is None:
        raise BuildValidationError(
            "graph has no build manifest; publish requires a graph produced by GraphBuilder "
            "(the JSON artifact alone cannot be published safely)"
        )
    manifest.validate()

    actual_hash = content_hash(graph)
    if actual_hash != manifest.content_hash:
        raise BuildValidationError(
            "graph content changed after the build (manifest hash mismatch); rebuild before publishing"
        )
    dims = {len(n.embedding) for n in graph.nodes.values() if n.embedding}
    if len(dims) > 1:
        raise BuildValidationError(
            f"graph mixes embedding dimensions {sorted(dims)}; rebuild it from a single embedder"
        )
    if dims and dims != {manifest.embedder_dim}:
        raise BuildValidationError(
            f"graph vectors are {sorted(dims)}-d but the manifest records {manifest.embedder_dim}-d; "
            "the artifact does not match its build identity"
        )

    store = HugeGraphSkillStore(hg_config)
    previous = store.active_pointer()
    if previous is not None and previous.build_id != manifest.build_id:
        # The store, not the artifact's lineage, is the truth about what was
        # live before this publish; rollback-with-no-args relies on it.
        manifest.previous_build = previous.build_id
    ob_table = build_scoped_table(ob_table_base or DEFAULT_TABLE, manifest.build_id)
    manifest.ob_table = ob_table

    if dry_run:
        return {
            "dry_run": True,
            "build_id": manifest.build_id,
            "previous_build": previous.build_id if previous else None,
            "ob_table": ob_table,
            "stats": graph.stats(),
        }

    try:
        written = store.write_build(graph, manifest.build_id)
        store.write_manifest(manifest, written["vertex_ids"])

        rows = write_to_oceanbase(
            graph,
            host=ob_host,
            port=ob_port,
            user=ob_user,
            password=ob_password,
            database=ob_database,
            table=ob_table,
            dim=manifest.embedder_dim,
        )
        expected_rows = sum(1 for n in graph.nodes.values() if n.embedding)
        if rows["rows"] != expected_rows:
            raise BuildValidationError(
                f"OceanBase wrote {rows['rows']} rows but {expected_rows} nodes have embeddings"
            )

        # Read back from the authority before switching traffic to it.
        store.read_build(manifest.build_id)
        ob_check = OceanBaseSkillStore(
            host=ob_host, port=ob_port, user=ob_user, password=ob_password,
            database=ob_database, table=ob_table, dim=manifest.embedder_dim,
        )
        try:
            ob_rows = ob_check.count()
        finally:
            ob_check.close()
        if ob_rows != expected_rows:
            raise BuildValidationError(
                f"OceanBase read-back returned {ob_rows} rows, expected {expected_rows}"
            )
    except Exception as exc:
        try:
            store.clear_build(manifest.build_id)
        except Exception as cleanup_exc:  # pragma: no cover - best effort
            log.warning("failed to clean up incomplete build %s: %s", manifest.build_id, cleanup_exc)
        if isinstance(exc, (BuildValidationError, GraphWriteError)):
            raise PublishError(f"publish of {manifest.build_id} failed: {exc}") from exc
        raise PublishError(f"publish of {manifest.build_id} failed: {exc}") from exc

    pointer = ActivePointer(
        build_id=manifest.build_id,
        embedder_id=manifest.embedder_id,
        embedder_dim=manifest.embedder_dim,
        content_hash=manifest.content_hash,
        ob_table=ob_table,
    )
    try:
        store.activate(pointer)
    except Exception as exc:
        raise PublishError(
            f"build {manifest.build_id} was written and validated but activation failed; "
            f"the previous build {previous.build_id if previous else '(none)'} is still active: {exc}"
        ) from exc

    return {
        "build_id": manifest.build_id,
        "previous_build": previous.build_id if previous else None,
        "ob_table": ob_table,
        "hugegraph": {"vertices": written["vertices"], "edges": written["edges"]},
        "oceanbase": rows,
        "activated": True,
    }


def rollback(
    build_id: Optional[str] = None,
    *,
    hg_config: Optional[HugeGraphConfig] = None,
) -> Dict[str, Any]:
    """Point readers back to ``build_id`` (default: the build before the active one)."""
    store = HugeGraphSkillStore(hg_config)
    active = store.active_pointer()
    if active is None:
        raise PublishError("there is no active build to roll back from")

    target_id = build_id
    if target_id is None:
        target_manifest = store.read_manifest(active.build_id)
        target_id = target_manifest.previous_build if target_manifest else None
        if not target_id:
            raise PublishError(
                f"active build {active.build_id} has no previous_build; pass --build-id explicitly"
            )
    manifest = store.read_manifest(target_id)
    if manifest is None:
        raise PublishError(f"build {target_id} does not exist in HugeGraph")

    pointer = ActivePointer(
        build_id=manifest.build_id,
        embedder_id=manifest.embedder_id,
        embedder_dim=manifest.embedder_dim,
        content_hash=manifest.content_hash,
        ob_table=manifest.ob_table,
    )
    store.activate(pointer)
    return {
        "rolled_back_from": active.build_id,
        "rolled_back_to": manifest.build_id,
        "ob_table": manifest.ob_table,
    }