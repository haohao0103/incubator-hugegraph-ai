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

"""Unified ingest API (POST /api/v1/ingest).

Single external entry point for importing: Feishu wiki docs, table/field
catalog metadata (CSV/JDBC), and metric definitions into the *same* KG graph
(``kg-rag``).  Routing is by ``source_type``; the two shared post-steps
(entity resolution + vector index) are toggled via ``options``.

See docs/UNIFIED_INGEST_QUERY_API_SPEC.md for the design.
"""

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from hugegraph_llm.api.models.unified_requests import (
    UnifiedIngestRequest,
    UnifiedIngestResponse,
)
from hugegraph_llm.api.unified_convert import (
    KG_SCHEMA_JSON,
    build_index_texts,
    convert_catalog_to_graph,
    convert_metric_to_graph,
    detect_source_type,
)
from hugegraph_llm.config import huge_settings
from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.feishu_ingest import FeishuIngestService
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.utils.log import log


def unified_ingest(req: UnifiedIngestRequest) -> UnifiedIngestResponse:
    """Core ingest logic shared by the HTTP route and the Gradio tab."""
    scheduler = SchedulerSingleton.get_instance()

    source_type = req.source_type
    if source_type == "auto":
        source_type = detect_source_type(req.payload)

    domain = req.domain or "default"
    entity_resolution = bool(req.options.get("entity_resolution", True))
    build_vector_index = bool(req.options.get("build_vector_index", True))

    details: Dict[str, Any] = {}
    vertex_count, edge_count = 0, 0

    if source_type == "feishu":
        # payload carries FeishuIngestService.ingest_wiki kwargs
        # (space_id, app_id, app_secret, mode, graph_schema, build_graph, ...)
        payload = dict(req.payload)
        payload.setdefault("graph_schema", huge_settings.graph_name)
        payload.setdefault("build_graph", True)
        # FeishuIngestService already builds the vector index internally when
        # build_index=True, so avoid a duplicate build below.
        payload.setdefault("build_index", build_vector_index)
        result = FeishuIngestService.ingest_wiki(**payload)
        vertex_count, edge_count = result.vertex_count, result.edge_count
        details["doc_count"] = result.doc_count
        details["text_count"] = result.text_count
        build_vector_index = False

    elif source_type in ("catalog_csv", "jdbc", "metric_json"):
        if source_type == "metric_json":
            graph = convert_metric_to_graph(req.payload, domain=domain, source=source_type)
        else:
            graph = convert_catalog_to_graph(req.payload, domain=domain, source=source_type)

        vertices = graph["vertices"]
        edges = graph["edges"]
        vertex_count, edge_count = len(vertices), len(edges)
        details["vertices"] = vertex_count
        details["edges"] = edge_count

        scheduler.schedule_flow(
            FlowName.IMPORT_GRAPH_DATA,
            {"vertices": vertices, "edges": edges},
            KG_SCHEMA_JSON,
        )

        if build_vector_index:
            texts = build_index_texts(vertices, edges)
            if texts:
                scheduler.schedule_flow(FlowName.BUILD_VECTOR_INDEX, texts)
                details["index_texts"] = len(texts)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported source_type: {source_type}",
        )

    # Shared post-step: cross-type entity resolution (links Table/Field/Metric
    # that denote the same real-world thing). Optional for safety/perf.
    if entity_resolution:
        try:
            res = scheduler.schedule_flow(
                FlowName.ENTITY_RESOLUTION,
                vertex_labels=["Table", "Field", "Metric"],
                strategy="hybrid",
            )
            details["entity_resolution"] = json.loads(res) if isinstance(res, str) else res
        except Exception as exc:  # noqa: BLE001 - surface as detail, don't hard-fail ingest
            log.error("Entity resolution failed during unified ingest: %s", exc)
            details["entity_resolution_error"] = str(exc)

    return UnifiedIngestResponse(
        graph=huge_settings.graph_name,
        status="succeeded",
        source_type=source_type,
        domain=domain,
        vertex_count=vertex_count,
        edge_count=edge_count,
        details=details,
    )


def unified_ingest_http_api(router: APIRouter):
    @router.post(
        "/api/v1/ingest",
        status_code=status.HTTP_200_OK,
        response_model=UnifiedIngestResponse,
    )
    def unified_ingest_api(req: UnifiedIngestRequest):
        try:
            return unified_ingest(req)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("unified_ingest error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
