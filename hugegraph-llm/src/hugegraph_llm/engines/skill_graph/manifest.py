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

"""Build manifest and active-build pointer.

Why a manifest
--------------
A skill graph is only trustworthy if a reader can answer three questions
before using it:

  1. *Which embedder produced these vectors?* Two 1536-d models share a
     dimension but not a vector space; mixing them yields plausible-looking
     garbage. The fingerprint (``embedder_id`` + ``embedder_dim``) makes that
     mix detectable instead of silent.
  2. *Which build is this?* ``build_id`` lets HugeGraph, OceanBase and the JSON
     artifact be correlated, and lets operators roll back by moving a pointer.
  3. *Is this the exact content I validated?* ``content_hash`` covers every
     node's text and every edge, so a half-written build is detectable.

Publishing model
----------------
A build is written entirely under its own ``build_id`` (HugeGraph vertices/edges
carry the id, OceanBase rows live in a build-scoped table). Nothing is visible
to readers until :class:`ActivePointer` is updated - that single write is the
commit point. Rollback is another pointer write.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from hugegraph_llm.engines.skill_graph.errors import BuildValidationError

SCHEMA_VERSION = "2.0"
MANIFEST_FIELD = "manifest"

_BUILD_ID_RE = re.compile(r"^[0-9TZ]+-[0-9a-f]{8}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_build_id() -> str:
    """Sortable build id: ``20261020T101500Z-1a2b3c4d``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def short_build_id(build_id: str) -> str:
    """8-char suffix used in resource names (``...__1a2b3c4d``).

    The id reaches Gremlin vertex ids and OceanBase table names, so it is
    validated here as a single choke point: a caller-supplied (``rollback
    --build-id``) or corrupted manifest id must fail fast rather than be
    interpolated into a query/identifier.
    """
    if not _BUILD_ID_RE.match(build_id or ""):
        raise BuildValidationError(
            f"invalid build id {build_id!r}; expected '<UTC timestamp>-<8 hex chars>'"
        )
    return build_id.rsplit("-", 1)[-1]


def build_scoped_table(base_table: str, build_id: str) -> str:
    """OceanBase table name for one build.

    Build-scoped tables make publish atomic without destructive updates: a new
    build never touches the live rows, and rollback only needs the pointer.
    """
    return f"{base_table}__{short_build_id(build_id)}"


def _update(h: "hashlib._Hash", value: Any) -> None:
    h.update(str(value).encode("utf-8"))
    h.update(b"\x00")


def content_hash(graph: Any) -> str:
    """Deterministic sha256 over all node text and typed edges.

    Embeddings are intentionally excluded: they are derived from the text plus
    the embedder identity, which is fingerprinted separately. Including them
    would make the hash depend on float formatting rather than content.
    """
    h = hashlib.sha256()
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        _update(h, node.id)
        _update(h, node.name)
        _update(h, node.description)
        _update(h, node.category)
        _update(h, ",".join(node.tags))
        _update(h, node.body)
        _update(h, node.node_type.value)
    for edge in sorted(graph.edges, key=lambda e: e.key()):
        _update(h, "|".join(edge.key()))
        _update(h, f"{edge.weight:.6f}")
        _update(h, edge.provenance)
    return h.hexdigest()


@dataclass
class BuildManifest:
    """Everything needed to verify and roll back one build."""

    build_id: str
    embedder_id: str
    embedder_dim: int
    content_hash: str
    created_at: str = field(default_factory=_utc_now)
    schema_version: str = SCHEMA_VERSION
    skills: int = 0
    categories: int = 0
    edges: int = 0
    ob_table: Optional[str] = None
    previous_build: Optional[str] = None
    mode: str = "full"  # full | incremental
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "BuildManifest":
        if not self.build_id:
            raise BuildValidationError("manifest.build_id is empty")
        short_build_id(self.build_id)  # rejects malformed/injected ids
        if not self.embedder_id:
            raise BuildValidationError(
                "manifest.embedder_id is empty; the embedder must expose a stable identity"
            )
        if not self.embedder_dim or self.embedder_dim <= 0:
            raise BuildValidationError(
                f"manifest.embedder_dim is invalid: {self.embedder_dim!r}"
            )
        if not self.content_hash:
            raise BuildValidationError("manifest.content_hash is empty")
        if self.schema_version != SCHEMA_VERSION:
            raise BuildValidationError(
                f"build schema {self.schema_version!r} is not supported by this engine "
                f"(expected {SCHEMA_VERSION!r}); rebuild the graph"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_id": self.build_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "embedder_id": self.embedder_id,
            "embedder_dim": self.embedder_dim,
            "content_hash": self.content_hash,
            "skills": self.skills,
            "categories": self.categories,
            "edges": self.edges,
            "ob_table": self.ob_table,
            "previous_build": self.previous_build,
            "mode": self.mode,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildManifest":
        known = {
            "build_id",
            "created_at",
            "schema_version",
            "embedder_id",
            "embedder_dim",
            "content_hash",
            "skills",
            "categories",
            "edges",
            "ob_table",
            "previous_build",
            "mode",
            "extra",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class ActivePointer:
    """The single commit point readers resolve against."""

    build_id: str
    embedder_id: str
    embedder_dim: int
    content_hash: str
    ob_table: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    activated_at: str = field(default_factory=_utc_now)

    def validate(self) -> "ActivePointer":
        if not self.build_id:
            raise BuildValidationError("active pointer has no build_id")
        short_build_id(self.build_id)  # rejects malformed/injected ids
        if not self.embedder_id or not self.embedder_dim:
            raise BuildValidationError(
                "active pointer is missing the embedder fingerprint; refusing to read a graph "
                "whose vectors cannot be verified"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_id": self.build_id,
            "embedder_id": self.embedder_id,
            "embedder_dim": self.embedder_dim,
            "content_hash": self.content_hash,
            "ob_table": self.ob_table,
            "schema_version": self.schema_version,
            "activated_at": self.activated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivePointer":
        return cls(
            build_id=str(data.get("build_id") or ""),
            embedder_id=str(data.get("embedder_id") or ""),
            embedder_dim=int(data.get("embedder_dim") or 0),
            content_hash=str(data.get("content_hash") or ""),
            ob_table=data.get("ob_table"),
            schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
            activated_at=str(data.get("activated_at") or _utc_now()),
        )


def build_manifest(
    graph: Any,
    embedder_id: str,
    embedder_dim: int,
    *,
    build_id: Optional[str] = None,
    ob_table: Optional[str] = None,
    previous_build: Optional[str] = None,
    mode: str = "full",
) -> BuildManifest:
    """Create and validate a manifest for ``graph``."""
    stats = graph.stats()
    manifest = BuildManifest(
        build_id=build_id or new_build_id(),
        embedder_id=embedder_id,
        embedder_dim=embedder_dim,
        content_hash=content_hash(graph),
        skills=stats["skills"],
        categories=stats["categories"],
        edges=stats["edges"],
        ob_table=ob_table,
        previous_build=previous_build,
        mode=mode,
    )
    return manifest.validate()