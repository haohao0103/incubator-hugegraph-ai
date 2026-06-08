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

import json

from fastapi import APIRouter, HTTPException, status

from hugegraph_llm.api.exceptions.rag_exceptions import generate_response
from hugegraph_llm.api.models.rag_requests import (
    AgentRequest,
    CommunityBuildRequest,
    GlobalSearchRequest,
    GraphConfigRequest,
    GraphRAGRequest,
    GraphRAGSearchMode,
    GraphRAGSearchRequest,
    GremlinGenerateRequest,
    IncrementalIndexRequest,
    DriftSearchRequest,
    LLMConfigRequest,
    RAGRequest,
    RerankerConfigRequest,
    SchemaValidationRequest,
)
from hugegraph_llm.api.models.rag_response import RAGResponse
from hugegraph_llm.config import huge_settings, llm_settings, prompt
from hugegraph_llm.utils.graph_index_utils import get_vertex_details
from hugegraph_llm.utils.log import log


def _enrich_with_provenance(result: tuple) -> list:
    """Try to add source citations to the answer using ProvenanceManager."""
    try:
        from hugegraph_llm.operators.hugegraph_op.provenance_manager import ProvenanceManager

        pm = ProvenanceManager()
        citations = []
        # Collect entity IDs from match_vids if available
        match_vids = getattr(result, "match_vids", None) or []
        if isinstance(result, dict):
            match_vids = result.get("match_vids", [])
        if match_vids:
            records = pm.get_provenance_for_answer(match_vids, max_per_entity=1)
            seen = set()
            for recs in records.values():
                for rec in recs:
                    key = rec.chunk_text[:100]
                    if key not in seen:
                        seen.add(key)
                        citations.append(rec.to_citation(max_text_len=200))
            return citations[:5]
    except Exception as e:
        log.debug("Provenance enrichment skipped: %s", e)
    return []


# pylint: disable=too-many-statements
def rag_http_api(
    router: APIRouter,
    rag_answer_func,
    graph_rag_recall_func,
    apply_graph_conf,
    apply_llm_conf,
    apply_embedding_conf,
    apply_reranker_conf,
    gremlin_generate_selective_func,
    agent_answer_func=None,
    community_build_func=None,
    global_search_func=None,
    graph_rag_search_func=None,
    incremental_index_func=None,
    drift_search_func=None,
):
    @router.post("/rag", status_code=status.HTTP_200_OK)
    def rag_answer_api(req: RAGRequest):
        set_graph_config(req)

        # Basic parameter validation: empty query => 400
        if not req.query or not str(req.query).strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must not be empty.",
            )

        result = rag_answer_func(
            text=req.query,
            raw_answer=req.raw_answer,
            vector_only_answer=req.vector_only,
            graph_only_answer=req.graph_only,
            graph_vector_answer=req.graph_vector_answer,
            graph_ratio=req.graph_ratio,
            rerank_method=req.rerank_method,
            near_neighbor_first=req.near_neighbor_first,
            gremlin_tmpl_num=req.gremlin_tmpl_num,
            max_graph_items=req.max_graph_items,
            topk_return_results=req.topk_return_results,
            vector_dis_threshold=req.vector_dis_threshold,
            topk_per_keyword=req.topk_per_keyword,
            # Keep prompt params in the end
            custom_related_information=req.custom_priority_info,
            answer_prompt=req.answer_prompt or prompt.answer_prompt,
            keywords_extract_prompt=req.keywords_extract_prompt or prompt.keywords_extract_prompt,
            gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
        )
        # Enrich with provenance citations if requested
        citations = []
        if req.include_provenance and result:
            citations = _enrich_with_provenance(result)

        # Build response
        response = {
            "query": req.query,
            **{
                key: value
                for key, value in zip(
                    ["raw_answer", "vector_only", "graph_only", "graph_vector_answer"],
                    result,
                )
                if getattr(req, key)
            },
        }
        if citations:
            for key in list(response.keys()):
                if key.endswith("_answer") and response[key]:
                    response[key] = f"{response[key]}\n\n## 来源\n" + "\n".join(
                        f"{i}. {c}" for i, c in enumerate(citations, 1)
                    )
        return response

    def set_graph_config(req):
        if req.client_config:
            huge_settings.graph_url = req.client_config.url
            huge_settings.graph_name = req.client_config.graph
            huge_settings.graph_user = req.client_config.user
            huge_settings.graph_pwd = req.client_config.pwd
            huge_settings.graph_space = req.client_config.gs

    @router.post("/rag/graph", status_code=status.HTTP_200_OK)
    def graph_rag_recall_api(req: GraphRAGRequest):
        try:
            set_graph_config(req)

            # Basic parameter validation: empty query => 400
            if not req.query or not str(req.query).strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query must not be empty.",
                )

            result = graph_rag_recall_func(
                query=req.query,
                max_graph_items=req.max_graph_items,
                topk_return_results=req.topk_return_results,
                vector_dis_threshold=req.vector_dis_threshold,
                topk_per_keyword=req.topk_per_keyword,
                gremlin_tmpl_num=req.gremlin_tmpl_num,
                rerank_method=req.rerank_method,
                near_neighbor_first=req.near_neighbor_first,
                custom_related_information=req.custom_priority_info,
                gremlin_prompt=req.gremlin_prompt or prompt.gremlin_generate_prompt,
                get_vertex_only=req.get_vertex_only,
            )

            if req.get_vertex_only:
                vertex_details = get_vertex_details(result["match_vids"], result)
                if vertex_details:
                    result["match_vids"] = vertex_details

            if isinstance(result, dict):
                params = [
                    "query",
                    "keywords",
                    "match_vids",
                    "graph_result_flag",
                    "gremlin",
                    "graph_result",
                    "vertex_degree_list",
                ]
                user_result = {key: result[key] for key in params if key in result}
                return {"graph_recall": user_result}
            return {"graph_recall": json.dumps(result)}

        except TypeError as e:
            log.error("TypeError in graph_rag_recall_api: %s", e)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        except Exception as e:
            log.error("Unexpected error occurred: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred.",
            ) from e

    @router.post("/config/graph", status_code=status.HTTP_201_CREATED)
    def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        res = apply_graph_conf(req.url, req.graph, req.user, req.pwd, req.gs, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    # TODO: restructure the implement of llm to three types, like "/config/chat_llm"
    @router.post("/config/llm", status_code=status.HTTP_201_CREATED)
    def llm_config_api(req: LLMConfigRequest):
        llm_settings.llm_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_llm_conf(
                req.api_key,
                req.api_base,
                req.language_model,
                req.max_tokens,
                origin_call="http",
            )
        else:
            res = apply_llm_conf(req.host, req.port, req.language_model, None, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/embedding", status_code=status.HTTP_201_CREATED)
    def embedding_config_api(req: LLMConfigRequest):
        llm_settings.embedding_type = req.llm_type

        if req.llm_type == "openai":
            res = apply_embedding_conf(req.api_key, req.api_base, req.language_model, origin_call="http")
        else:
            res = apply_embedding_conf(req.host, req.port, req.language_model, origin_call="http")
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/config/rerank", status_code=status.HTTP_201_CREATED)
    def rerank_config_api(req: RerankerConfigRequest):
        llm_settings.reranker_type = req.reranker_type

        if req.reranker_type == "cohere":
            res = apply_reranker_conf(req.api_key, req.reranker_model, req.cohere_base_url, origin_call="http")
        elif req.reranker_type == "siliconflow":
            res = apply_reranker_conf(req.api_key, req.reranker_model, None, origin_call="http")
        else:
            res = status.HTTP_501_NOT_IMPLEMENTED
        return generate_response(RAGResponse(status_code=res, message="Missing Value"))

    @router.post("/text2gremlin", status_code=status.HTTP_200_OK)
    def text2gremlin_api(req: GremlinGenerateRequest):
        try:
            set_graph_config(req)

            # Basic parameter validation: empty query => 400
            if not req.query or not str(req.query).strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query must not be empty.",
                )

            output_types_str_list = None
            if req.output_types:
                output_types_str_list = [ot.value for ot in req.output_types]

            response_dict = gremlin_generate_selective_func(
                inp=req.query,
                example_num=req.example_num,
                schema_input=huge_settings.graph_name,
                gremlin_prompt_input=req.gremlin_prompt,
                requested_outputs=output_types_str_list,
            )
            return response_dict
        except HTTPException as e:
            raise e
        except Exception as e:
            log.error("Error in text2gremlin_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during Gremlin generation.",
            ) from e

    @router.post("/agent", status_code=status.HTTP_200_OK)
    def agent_answer_api(req: AgentRequest):
        """Agent-based multi-step graph reasoning endpoint.

        For complex queries, runs a ReAct (Reasoning + Acting) loop where
        the LLM selects and executes tools to explore the knowledge graph.
        Simple queries are routed to existing fast RAG flows.

        Returns the final answer along with the full reasoning trace.
        """
        try:
            set_graph_config(req)

            # Validate query
            if not req.query or not str(req.query).strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query must not be empty.",
                )

            if agent_answer_func is None:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Agent function is not configured. Set up agent LLM and ToolRegistry.",
                )

            result = agent_answer_func(
                query=req.query,
                max_steps=req.max_steps,
                tools_filter=req.tools_filter,
                stream=req.stream,
                verbose=req.verbose,
            )

            # Handle routing case (simple query)
            if isinstance(result, dict) and result.get("is_simple_query"):
                return {
                    "status_code": 200,
                    "message": "Query routed to fast RAG flow",
                    "is_simple_query": True,
                    "simple_flow_used": result.get("simple_flow_used", "graph_only"),
                    "hint": "Re-run with /rag endpoint for direct answer",
                }

            return {
                "query": req.query,
                "answer": result.get("answer", ""),
                "trace": result.get("trace", []),
                "total_steps": result.get("total_steps", 0),
                "status_code": result.get("status_code", 200),
                "message": result.get("message", "Agent execution completed"),
            }

        except HTTPException as e:
            raise e
        except ValueError as e:
            log.error("ValueError in agent_answer_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except Exception as e:
            log.error("Unexpected error in agent_answer_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during agent execution.",
            ) from e

    @router.post("/community/build", status_code=status.HTTP_201_CREATED)
    def community_build_api(req: CommunityBuildRequest):
        """Build community detection index for the knowledge graph.

        This offline endpoint triggers:
        1. Community detection (Leiden/Louvain) on the graph
        2. LLM-based community report generation
        3. Vector index construction for community-level retrieval

        After building, the /rag/global endpoint can answer
        macro-level questions about the entire graph.
        """
        try:
            set_graph_config(req)

            if community_build_func is None:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Community build function is not configured.",
                )

            result = community_build_func(
                graph_name=req.graph_name or huge_settings.graph_name,
                algorithm=req.algorithm,
                max_levels=req.max_levels,
            )

            return {
                "status_code": result.get("status_code", 200),
                "message": result.get("message", "Community detection completed"),
                "community_count": result.get("community_count", 0),
                "report_count": result.get("report_count", 0),
                "index_built": result.get("index_built", False),
            }

        except HTTPException as e:
            raise e
        except Exception as e:
            log.error("Error in community_build_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Community detection failed.",
            ) from e

    @router.post("/rag/incremental", status_code=status.HTTP_201_CREATED)
    def incremental_index_api(req: IncrementalIndexRequest):
        """Incremental document indexing for the knowledge graph.

        Indexes new documents without full graph reconstruction.
        Only processes new content and updates affected portions:

        1. Chunk splitting and entity/relation extraction
        2. Entity resolution (merge new entities with existing)
        3. Append new vertices/edges to HugeGraph
        4. Detect affected communities from new vertices
        5. Regenerate community reports for affected communities
        6. Incrementally add new chunk vectors to FAISS index

        Much faster than full re-indexing for small batches of documents.
        """
        try:
            set_graph_config(req)

            if incremental_index_func is None:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Incremental index function is not configured.",
                )

            result = incremental_index_func(
                texts=req.texts,
                graph_name=req.graph_name or huge_settings.graph_name,
                entity_resolution_strategy=req.entity_resolution_strategy,
                community_hop=req.community_hop,
            )

            return {
                "status_code": result.get("status_code", 201),
                "message": result.get("message", "Incremental indexing completed"),
                "vertices_added": result.get("vertices_added", 0),
                "edges_added": result.get("edges_added", 0),
                "entities_merged": result.get("entities_merged", 0),
                "affected_communities": result.get("affected_communities", 0),
                "vectors_added": result.get("vectors_added", 0),
                "community_reports_updated": result.get("community_reports_updated", 0),
            }

        except HTTPException as e:
            raise e
        except Exception as e:
            log.error("Error in incremental_index_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Incremental indexing failed.",
            ) from e

    @router.post("/rag/drift", status_code=status.HTTP_200_OK)
    def drift_search_api(req: DriftSearchRequest):
        """DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search.

        A 5-step deep retrieval strategy combining Global Search breadth
        with Local Search depth:

        1. HyDE: Generate hypothetical answer for the query
        2. Community Match: Find top-K relevant communities
        3. Primer: Initial analysis + follow-up sub-questions
        4. Parallel Local Search: Iterative deep fact retrieval
        5. Reduce: Synthesize comprehensive answer

        Best for complex analytical questions that require both overview
        understanding and specific factual details.
        """
        try:
            set_graph_config(req)

            if drift_search_func is None:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="DRIFT search function is not configured.",
                )

            result = drift_search_func(
                query=req.query,
                graph_name=req.graph_name or huge_settings.graph_name,
                max_depth=req.max_depth,
                communities_top_k=req.communities_top_k,
                language=req.language,
            )

            return {
                "status_code": result.get("status_code", 200),
                "drift_answer": result.get("drift_answer", ""),
                "communities_used": result.get("drift_communities_used", 0),
                "depth_reached": result.get("drift_depth_reached", 0),
                "findings_count": len(result.get("drift_findings", [])),
                "call_count": result.get("call_count", 0),
            }

        except HTTPException as e:
            raise e
        except Exception as e:
            log.error("Error in drift_search_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="DRIFT search failed.",
            ) from e

    @router.post("/rag/schema/validate", status_code=status.HTTP_200_OK)
    def schema_validate_api(req: SchemaValidationRequest):
        """Validate entities and relations against the schema.

        Checks property types, required fields, relation constraints,
        and cardinality rules. Returns validation report with violations
        and suggested fixes.

        Usage:
            POST /rag/schema/validate
            {
                "entities": [{"label": "Person", "properties": {"name": "Alice", "age": 30}}],
                "relations": [{"relation_label": "works_at", "source_label": "Person", "target_label": "Company"}],
                "strict_mode": false
            }
        """
        try:
            from hugegraph_llm.operators.graph_op.schema_validator import (
                SchemaValidator,
            )

            validator = SchemaValidator(strict_mode=req.strict_mode)

            # Validate entities
            entity_results = []
            for i, ent in enumerate(req.entities):
                label = ent.get("label", "Entity")
                props = ent.get("properties", ent)
                vr = validator.validate_entity(label, props)
                entity_results.append({
                    "index": i,
                    "label": label,
                    "is_valid": vr.is_valid,
                    "errors": [v.message for v in vr.errors],
                    "warnings": [v.message for v in vr.warnings],
                })

            # Validate relations
            relation_results = []
            for i, rel in enumerate(req.relations):
                rel_label = rel.get("relation_label", rel.get("label", ""))
                src = rel.get("source_label", rel.get("source", ""))
                tgt = rel.get("target_label", rel.get("target", ""))
                props = rel.get("properties", {})
                vr = validator.validate_relation(rel_label, src, tgt, props)
                relation_results.append({
                    "index": i,
                    "relation_label": rel_label,
                    "is_valid": vr.is_valid,
                    "errors": [v.message for v in vr.errors],
                    "warnings": [v.message for v in vr.warnings],
                })

            total_errors = sum(
                len(r["errors"]) for r in entity_results + relation_results
            )
            total_warnings = sum(
                len(r["warnings"]) for r in entity_results + relation_results
            )

            return {
                "schema_version": validator._schema.version,
                "total_entities": len(req.entities),
                "total_relations": len(req.relations),
                "valid_entities": sum(1 for r in entity_results if r["is_valid"]),
                "valid_relations": sum(
                    1 for r in relation_results if r["is_valid"]
                ),
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "entity_results": entity_results,
                "relation_results": relation_results,
            }

        except Exception as e:
            log.error("Error in schema_validate_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Schema validation failed.",
            ) from e

    @router.post("/rag/global", status_code=status.HTTP_200_OK)
    def global_search_api(req: GlobalSearchRequest):
        """Macro-level Global Search over community reports.

        Answers broad, thematic questions about the entire knowledge
        graph using MapReduce over pre-computed community summaries.

        Requires community reports to have been built first via
        POST /community/build.
        """
        try:
            if not req.query or not str(req.query).strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query must not be empty.",
                )

            if global_search_func is None:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Global search function is not configured.",
                )

            result = global_search_func(query=req.query)

            return {
                "query": req.query,
                "answer": result.get("answer", ""),
                "communities_used": result.get("communities_used", 0),
                "map_findings": result.get("map_findings", []),
            }

        except HTTPException as e:
            raise e
        except Exception as e:
            log.error("Error in global_search_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Global search failed.",
            ) from e

    @router.post("/rag/graph/search", status_code=status.HTTP_200_OK)
    def graph_rag_search_api(req: GraphRAGSearchRequest):
        """Direct graph RAG search operations endpoint.

        Executes one of the supported graph-level operations
        (graph_traverse, semantic_id_lookup, text2gremlin, schema_lookup)
        without requiring the full agent loop.

        This endpoint exposes the same graph tools that the ReAct agent
        uses internally, making them available for direct API access.
        """
        try:
            set_graph_config(req)

            if graph_rag_search_func is None:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Graph RAG search function is not configured.",
                )

            # Validate required parameters per mode
            if req.mode == GraphRAGSearchMode.GRAPH_TRAVERSE:
                if not req.vertex_ids:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="vertex_ids is required for graph_traverse mode.",
                    )
            elif req.mode == GraphRAGSearchMode.SEMANTIC_ID_LOOKUP:
                if not req.keywords:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="keywords is required for semantic_id_lookup mode.",
                    )
            elif req.mode == GraphRAGSearchMode.TEXT2GREMLIN:
                if not req.query or not str(req.query).strip():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="query is required for text2gremlin mode.",
                    )

            result = graph_rag_search_func(
                mode=req.mode.value,
                query=req.query,
                vertex_ids=req.vertex_ids,
                max_depth=req.max_depth,
                max_items=req.max_items,
                keywords=req.keywords,
                gremlin_example_num=req.gremlin_example_num,
            )

            return {
                "mode": req.mode.value,
                "query": req.query,
                "result": result,
            }

        except HTTPException as e:
            raise e
        except ValueError as e:
            log.error("ValueError in graph_rag_search_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except Exception as e:
            log.error("Unexpected error in graph_rag_search_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Graph RAG search operation failed.",
            ) from e
