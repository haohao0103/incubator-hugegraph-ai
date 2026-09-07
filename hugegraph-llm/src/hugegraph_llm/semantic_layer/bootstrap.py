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

"""Idempotent bootstrap of the semantic layer graph.

Milestone M0: create every property key, vertex label, edge label and index
in one pass, on a freshly started HugeGraph instance or on an existing one.

Why edge labels must be created in a single pass: HugeGraph locks an edge
label to its first ``source_label -> target_label`` pair. Adding a missing
pair later means deleting the label and migrating existing edges, so the
full edge set is declared in ``schema_def.EDGE_LABELS`` and created here
before any data is ingested.
"""

from typing import Any, Dict, List, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.operators.hugegraph_op.schema_manager import (
    EDGE_LABELS as SCHEMA_KIND_EDGE_LABELS,
)
from hugegraph_llm.operators.hugegraph_op.schema_manager import (
    VERTEX_LABELS as SCHEMA_KIND_VERTEX_LABELS,
)
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.semantic_layer import schema_def
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.utils.log import log

__all__ = ["SemanticLayerBootstrap", "bootstrap_semantic_layer"]


class SemanticLayerBootstrap:
    """Creates (and verifies) the semantic layer schema on HugeGraph."""

    def __init__(
        self,
        graph_name: str,
        *,
        connection: Optional[Dict[str, Any]] = None,
        client: Optional[PyHugeClient] = None,
    ) -> None:
        self.graph_name = graph_name
        definition_error = schema_def.validate()
        if definition_error:
            raise ValueError(f"Invalid semantic layer schema: {definition_error}")
        self.manager = SchemaManager(
            graph_name, connection=connection, client=client
        )

    # -- create -------------------------------------------------------------

    def ensure(self) -> Dict[str, int]:
        """Idempotently create the whole schema. Returns created-object counts."""
        summary = self.manager.ensure_schema(schema_def.build_schema_dict())
        log.info(
            "semantic layer schema ensured on '%s': %s", self.graph_name, summary
        )
        return summary

    # -- verify -------------------------------------------------------------

    def verify(self) -> Dict[str, Any]:
        """Check the live graph against the declared schema.

        Returns a report rather than raising, so bootstrap can be inspected
        in CI without a server::

            {"ok": True, "missing_vertex_labels": [], ...}

        Any missing object means the graph was created by an older definition
        (or partially) and must be re-bootstrapped before ingestion.
        """
        report: Dict[str, Any] = {
            "graph": self.graph_name,
            "missing_vertex_labels": [],
            "missing_edge_labels": [],
            "missing_indexes": [],
            "edge_endpoint_mismatch": [],
            "missing_vertex_properties": [],
        }
        live: Optional[Dict[str, Any]]
        try:
            live = self.manager.get_schema_cached(refresh=True)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            report["ok"] = False
            report["error"] = f"cannot read schema: {exc}"
            return report

        if not live:
            report["ok"] = False
            report["error"] = "empty schema (server unreachable or graph missing)"
            return report

        live_vertex = {vl["name"] for vl in live.get("vertexlabels", [])}
        live_edge = {el["name"] for el in live.get("edgelabels", [])}
        endpoints = {
            el["name"]: (el.get("source_label"), el.get("target_label"))
            for el in live.get("edgelabels", [])
        }

        report["missing_vertex_labels"] = sorted(
            label.value for label in VertexLabel if label.value not in live_vertex
        )
        report["missing_edge_labels"] = sorted(
            label.value for label in EdgeLabel if label.value not in live_edge
        )

        for name, source, target, _props in schema_def.EDGE_LABELS:
            actual = endpoints.get(name.value)
            if actual is None:
                continue  # already reported as missing
            expected = (source.value, target.value)
            if actual != expected:
                report["edge_endpoint_mismatch"].append(
                    f"{name.value}: expected {expected[0]}->{expected[1]}, "
                    f"found {actual[0]}->{actual[1]}"
                )

        # A label that exists but lacks a declared property is the nastiest
        # failure mode: bootstrap reports success, then every write of that
        # property fails at runtime with "Invalid property". Property keys are
        # cheap to check here and expensive to discover in production.
        live_vprops = {
            vl["name"]: set(vl.get("properties") or [])
            for vl in live.get("vertexlabels", [])
        }
        for label, declared in schema_def.VERTEX_LABELS.items():
            actual = live_vprops.get(label.value)
            if actual is None:
                continue  # already reported as a missing label
            gap = sorted(set(declared) - actual)
            if gap:
                report["missing_vertex_properties"].append(
                    f"{label.value}: {gap} (label exists; add via label append "
                    f"or recreate the graph)"
                )

        live_indexes = {idx["name"] for idx in self.manager.list_indexes()}
        report["missing_indexes"] = [
            idx["name"] for idx in schema_def.INDEX_LABELS
            if idx["name"] not in live_indexes
        ]

        report["ok"] = not (
            report["missing_vertex_labels"]
            or report["missing_edge_labels"]
            or report["missing_indexes"]
            or report["edge_endpoint_mismatch"]
            or report["missing_vertex_properties"]
        )
        return report

    def ensure_and_verify(self) -> Dict[str, Any]:
        """Create then verify; the normal entry point for scripts and tests."""
        report: Dict[str, Any] = {"created": self.ensure()}
        report.update(self.verify())
        return report


def bootstrap_semantic_layer(
    graph_name: str,
    *,
    connection: Optional[Dict[str, Any]] = None,
    client: Optional[PyHugeClient] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: ensure + verify in one call."""
    return SemanticLayerBootstrap(
        graph_name, connection=connection, client=client
    ).ensure_and_verify()


def graph_summary(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Summarise a live schema dict as ``{vertex_labels, edge_labels}``."""
    return {
        "vertex_labels": sorted(
            vl["name"] for vl in schema.get(SCHEMA_KIND_VERTEX_LABELS, [])
        ),
        "edge_labels": sorted(
            el["name"] for el in schema.get(SCHEMA_KIND_EDGE_LABELS, [])
        ),
    }
