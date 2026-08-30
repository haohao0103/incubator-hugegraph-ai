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
  * ``nl2sql``  -> KG-aware NL2SQL pipeline (P0 schema-linking + validation,
                   P1 jargon/authority/lineage/voting). Generates candidate SQL
                   from the metadata KG, deterministically validates and votes,
                   then optionally calls the LLM for generation.

All query flows read the KG graph name from ``huge_settings.graph_name``; no
full schema JSON is needed (the flows fetch it from the graph server).

See docs/UNIFIED_INGEST_QUERY_API_SPEC.md for the design.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status

from hugegraph_llm.api.models.unified_requests import (
    QueryStage,
    QueryStageBuilder,
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


def _format_precise(
    tg: Dict[str, Any], domain: Optional[str], question: Optional[str] = None
) -> UnifiedQueryResponse:
    answer = tg.get("raw_execution_result") or tg.get("template_execution_result") or ""
    stages: List[QueryStage] = [
        QueryStageBuilder.make(
            "text2gremlin",
            output={
                "template_gremlin": tg.get("template_gremlin", ""),
                "raw_gremlin": tg.get("raw_gremlin", ""),
                "match_result": tg.get("match_result", []),
            },
            input={"question": question, "domain": domain},
        ),
        QueryStageBuilder.make(
            "graph_execution",
            output={
                "template_execution_result": tg.get("template_execution_result", []),
                "raw_execution_result": tg.get("raw_execution_result", []),
            },
        ),
    ]
    return UnifiedQueryResponse(
        answer=answer,
        route="precise",
        citations=[],
        subgraph={"match_result": tg.get("match_result", [])},
        raw=tg,
        stages=stages,
    )


def _apply_fallback(resp: UnifiedQueryResponse, fallback: Optional[str]) -> UnifiedQueryResponse:
    """Return ``fallback`` verbatim when the query produced no answer.

    Generalized from neo4j-graphrag-python's ``response_fallback``: an empty
    context (no answer) short-circuits to a caller-provided message, saving
    an LLM call and avoiding hallucination.
    """
    if fallback and not resp.answer:
        resp.answer = fallback
    return resp


def _run_nl2sql(req: UnifiedQueryRequest, fallback: Optional[str]) -> UnifiedQueryResponse:
    """Route the question through the KG-aware NL2SQL pipeline (P0 + P1).

    Loads the metadata KG from the configured graph server and composes the
    schema-linking, deterministic SQL validation/voting, and the metric
    authority / lineage / golden-SQL audits. Candidate generation falls back to
    the project's LLM role (glm-5.3) when reachable, but every other step is
    LLM-free and the pipeline degrades gracefully (empty answer -> fallback).
    """
    from hugegraph_llm.operators.graph_op.kg_nl2sql_pipeline import KgNL2SQLPipeline
    from hugegraph_llm.utils.hugegraph_utils import get_hg_client

    client = get_hg_client()
    pipe = KgNL2SQLPipeline(
        question=req.question,
        client=client,
        domain=req.domain,
    )
    resp = pipe.run()
    # live generation may yield no valid SQL (LLM down / flaky endpoint);
    # surface the caller's fallback instead of an empty answer.
    return _apply_fallback(resp, fallback)


# default refusal copy when schema retrieval finds no evidence (the 货拉拉
# badcase '不存在也答' guard): the platform should not generate SQL from nothing
DEFAULT_NO_EVIDENCE_MSG = "未找到相关元数据，可能需加工"


def _run_schema_retrieval(req: UnifiedQueryRequest) -> UnifiedQueryResponse:
    """Schema retrieval mode: question -> relevant tables/fields/metrics.

    Uses the multi-recall linker (graph structure + fulltext + lexical,
    optionally query-understanding for dual keywords + synonym expansion).
    When nothing is linked, returns ``no_evidence=True`` with the refusal
    copy instead of a fabricated answer.
    """
    from hugegraph_llm.operators.graph_op.kg_multi_retrieval import (
        KgMultiSchemaLinker,
        MultiRecallConfig,
    )
    from hugegraph_llm.operators.graph_op.kg_query_understanding import QueryUnderstanding
    from hugegraph_llm.operators.graph_op.kg_rule_engine import KgRuleEngine
    from hugegraph_llm.utils.hugegraph_utils import get_hg_client

    client = get_hg_client()
    config = MultiRecallConfig()
    retriever_config = dict(req.retriever_config or {})
    if "importance_weight" in retriever_config:
        config.importance_weight = float(retriever_config["importance_weight"])

    linker = KgMultiSchemaLinker(
        client=client,
        config=config,
        # query understanding: LLM keyword extraction when reachable, heuristic
        # fallback otherwise; synonym expansion via the default term graph
        query_understanding=QueryUnderstanding(),
    )
    data = KgRuleEngine(client, huge_settings.graph_name).load_graph()
    ctx = linker.link(req.question, data=data)

    if ctx.empty:
        message = req.response_fallback or DEFAULT_NO_EVIDENCE_MSG
        return UnifiedQueryResponse(
            answer=message,
            route="schema",
            citations=[],
            subgraph={"no_evidence": True},
            raw={"no_evidence": True, "intent": ctx.query_intent},
            stages=[
                QueryStageBuilder.make(
                    "schema_retrieval",
                    output={"no_evidence": True, "message": message},
                    input={"question": req.question, "domain": req.domain},
                )
            ],
            no_evidence=True,
        )

    linked = {
        "tables": [t.get("name") for t in ctx.tables],
        "fields": [f.get("name") for f in ctx.fields],
        "metrics": [m.get("name") for m in ctx.metrics],
    }
    lines = ["相关元数据:"]
    if ctx.tables:
        lines.append("表: " + ", ".join(t.get("name", "") for t in ctx.tables))
    if ctx.metrics:
        lines.append(
            "指标: " + ", ".join(
                f"{m.get('name')}({m.get('definition') or m.get('formula') or ''})"
                for m in ctx.metrics
            )
        )
    if ctx.fields:
        lines.append("字段: " + ", ".join(f.get("name", "") for f in ctx.fields))
    if ctx.evidence:
        lines.append("口径证据: " + "; ".join(ctx.evidence[:5]))

    return UnifiedQueryResponse(
        answer="\n".join(lines),
        route="schema",
        citations=[],
        subgraph=linked,
        raw={"linked": linked, "evidence": ctx.evidence[:12], "intent": ctx.query_intent},
        stages=[
            QueryStageBuilder.make(
                "query_understanding",
                output=ctx.query_intent,
                input={"question": req.question},
            ),
            QueryStageBuilder.make(
                "schema_retrieval",
                output=linked,
                input={"question": req.question, "domain": req.domain},
            ),
        ],
        no_evidence=False,
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
    retriever_config = dict(req.retriever_config or {})
    top_k = int(retriever_config.get("top_k") or req.top_k or 5)
    fallback = req.response_fallback

    # auto: probe structured match via Text2Gremlin's match_result, then decide.
    if mode == "auto":
        tg = _text2gremlin(req.question)
        match = tg.get("match_result") if isinstance(tg, dict) else []
        if match:
            return _apply_fallback(
                _format_precise(tg, domain, req.question), fallback
            )
        mode = "hybrid"

    if mode == "precise":
        return _apply_fallback(
            _format_precise(_text2gremlin(req.question), domain, req.question),
            fallback,
        )

    if mode == "nl2sql":
        return _run_nl2sql(req, fallback)

    if mode == "schema":
        return _run_schema_retrieval(req)

    # semantic / hybrid -> RAGGraphVectorFlow (retriever_config forwarded)
    vector_only = bool(retriever_config.get("vector_only", mode == "semantic"))
    graph_search = bool(retriever_config.get("graph_search", not vector_only))
    res = _graphrag(req.question, top_k, graph_search=graph_search, vector_only=vector_only)
    answer = res.get("graph_vector_answer") or res.get("vector_only_answer") or ""
    route = "semantic" if vector_only else "graphrag"
    stages: List[QueryStage] = []
    if graph_search:
        stages.append(
            QueryStageBuilder.make(
                "graph_execution",
                output={
                    "graph_only_answer": res.get("graph_only_answer", ""),
                    "query_intent": res.get("query_intent", ""),
                },
                input={"question": req.question, "graph_search": True},
            )
        )
    stages.append(
        QueryStageBuilder.make(
            "vector_recall",
            output={
                "vector_only_answer": res.get("vector_only_answer", ""),
                "retrieval_level": res.get("retrieval_level", ""),
                "top_k": top_k,
            },
            input={"question": req.question, "top_k": top_k},
        )
    )
    return _apply_fallback(
        UnifiedQueryResponse(
            answer=answer,
            route=route,
            citations=[],
            subgraph={},
            raw=res,
            stages=stages,
        ),
        fallback,
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
