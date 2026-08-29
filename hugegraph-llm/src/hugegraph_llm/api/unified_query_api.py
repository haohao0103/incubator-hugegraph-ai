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

"""Unified query API (POST /api/v1/query).

One query box for the user. ``mode``:
  * ``auto``    -> if the question hits structured nodes (Table/Field/Metric),
                   route to precise Text2Gremlin; otherwise hybrid graph+vector
                   RAG.
  * ``precise`` -> Text2GremlinFlow (exact Gremlin + execution result).
  * ``semantic``-> vector-only RAG.
  * ``hybrid``  -> RAGGraphVectorFlow (graph traversal + vector recall).

All query flows read the KG graph name from ``huge_settings.graph_name``; no
full schema JSON is needed (the flows fetch it from the graph server).

See docs/UNIFIED_INGEST_QUERY_API_SPEC.md for the design.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status

from hugegraph_llm.api.models.unified_requests import (
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from hugegraph_llm.config import huge_settings
from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.utils.log import log


def _text2gremlin(question: str) -> Dict[str, Any]:
    scheduler = SchedulerSingleton.get_instance()
    return scheduler.schedule_flow(
        FlowName.TEXT2GREMLIN,
        question,
        2,  # example_num
        huge_settings.graph_name,  # schema_input (graph name, not full schema)
        None,  # gremlin_prompt_input
        # Request the generated gremlin AND its execution result so that
        # _format_precise() can return the actual data as the answer.
        ["template_gremlin", "raw_gremlin", "template_execution_result", "raw_execution_result"],
    )


def _graphrag(question: str, top_k: int, graph_search: bool, vector_only: bool) -> Dict[str, Any]:
    scheduler = SchedulerSingleton.get_instance()
    return scheduler.schedule_flow(
        FlowName.RAG_GRAPH_VECTOR,
        query=question,
        vector_search=True,
        graph_search=graph_search,
        raw_answer=False,
        vector_only_answer=vector_only,
        graph_only_answer=False,
        graph_vector_answer=(not vector_only),
        topk_return_results=top_k,
    )


def _format_precise(tg: Dict[str, Any], domain: Optional[str]) -> UnifiedQueryResponse:
    answer = tg.get("raw_execution_result") or tg.get("template_execution_result") or ""
    return UnifiedQueryResponse(
        answer=answer,
        route="precise",
        citations=[],
        subgraph={"match_result": tg.get("match_result", [])},
        raw=tg,
    )


def unified_query(req: UnifiedQueryRequest) -> UnifiedQueryResponse:
    """Core query logic shared by the HTTP route and the Gradio tab."""
    if not req.question or not str(req.question).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty.",
        )

    mode = req.mode
    domain = req.domain
    top_k = req.top_k or 5

    # auto: probe structured match via Text2Gremlin's match_result, then decide.
    if mode == "auto":
        tg = _text2gremlin(req.question)
        match = tg.get("match_result") if isinstance(tg, dict) else []
        if match:
            return _format_precise(tg, domain)
        mode = "hybrid"

    if mode == "precise":
        return _format_precise(_text2gremlin(req.question), domain)

    # semantic / hybrid -> RAGGraphVectorFlow
    vector_only = mode == "semantic"
    res = _graphrag(req.question, top_k, graph_search=(not vector_only), vector_only=vector_only)
    answer = res.get("graph_vector_answer") or res.get("vector_only_answer") or ""
    route = "semantic" if vector_only else "graphrag"
    return UnifiedQueryResponse(
        answer=answer,
        route=route,
        citations=[],
        subgraph={},
        raw=res,
    )


def unified_query_http_api(router: APIRouter):
    @router.post(
        "/api/v1/query",
        status_code=status.HTTP_200_OK,
        response_model=UnifiedQueryResponse,
    )
    def unified_query_api(req: UnifiedQueryRequest):
        try:
            return unified_query(req)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("unified_query error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
