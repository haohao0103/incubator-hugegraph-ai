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

"""Pydantic request/response models for the unified ingest / query API.

Design goal (see docs/UNIFIED_INGEST_QUERY_API_SPEC.md): one external entry
point for ingest and one for query. Internally the backend routes by
``source_type`` / ``mode``; all data lands in the same KG graph (``kg-rag``)
using vertex/edge labels to discriminate document vs catalog vs metric data.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UnifiedIngestRequest(BaseModel):
    """Unified ingest request.

    ``source_type`` selects the ingest path:
      * ``feishu``       -> FeishuIngestService (Wiki docs -> graph + vector)
      * ``catalog_csv``  -> table/field metadata -> Table/Field + hasColumn
      * ``jdbc``         -> same as catalog_csv (tabular metadata)
      * ``metric_json``  -> metric definitions -> Metric + computedFrom/dependsOn
      * ``auto``         -> sniff ``payload`` keys to pick catalog vs metric

    ``payload`` is a source-specific dict (see the examples in
    demo/rag_demo/unified_io_block.py). ``options`` toggles the two shared
    post-steps: entity resolution and vector index build.
    """

    source_type: str = Field(..., description="feishu | catalog_csv | jdbc | metric_json | auto")
    payload: Dict[str, Any] = Field(default_factory=dict, description="source-specific payload")
    domain: Optional[str] = Field(None, description="business domain used for vertex isolation")
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="entity_resolution / build_vector_index toggles",
    )


class UnifiedIngestResponse(BaseModel):
    graph: str
    status: str
    source_type: str
    domain: Optional[str] = None
    vertex_count: int = 0
    edge_count: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)


class UnifiedQueryRequest(BaseModel):
    """Unified query request.

    ``mode``:
      * ``auto``    -> route to precise (Text2Gremlin) when the question hits
                       structured nodes, else hybrid (graph + vector RAG)
      * ``precise`` -> Text2Gremlin (exact Gremlin + execution)
      * ``semantic``-> vector-only RAG
      * ``hybrid``  -> graph + vector RAG (default fallback for auto)
    """

    question: str
    mode: str = Field("auto", description="auto | precise | semantic | hybrid")
    domain: Optional[str] = Field(None, description="optional domain filter")
    top_k: int = Field(5, description="max number of returned results")
    response_fallback: Optional[str] = Field(
        None,
        description="returned verbatim instead of an empty answer when the "
        "query yields no context (saves an LLM call, prevents hallucination)",
    )
    retriever_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="retriever-level overrides transparently forwarded to the "
        "query flow, e.g. {'top_k': 10}",
    )


class QueryStage(BaseModel):
    """One observable step of a query pipeline (unified stages contract).

    ``stage`` names follow the pipeline: ``text2gremlin``,
    ``graph_execution``, ``vector_recall``, ``reasoning`` (optional,
    Semantica-style deterministic inference), ``sql_generation`` (stage-2
    NL2SQL). Stages are appended in execution order; optional stages are
    simply absent when not executed.
    """

    stage: str
    input: Optional[Dict[str, Any]] = Field(default=None, description="stage input (question/schema/...)")
    output: Dict[str, Any] = Field(default_factory=dict, description="stage output (gremlin/data/chunks/...)")


class UnifiedQueryResponse(BaseModel):
    """Unified query response.

    ``stages`` is the observability contract for upstream platforms / agents:
    the execution pipeline is returned stage by stage (each stage carries its
    ``input`` and ``output``), so a caller can audit, debug or re-assemble
    the answer itself - e.g. take the generated Gremlin / the queried graph
    data / the recalled vector chunks to build its own SQL (NL2SQL stage-1
    integration), or verify the reasoning chain. ``raw`` keeps the full
    backend payload for backward compatibility.
    """

    answer: str = ""
    route: str = ""
    citations: List[Any] = Field(default_factory=list)
    subgraph: Dict[str, Any] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)
    stages: List[QueryStage] = Field(default_factory=list)


class QueryStageBuilder:
    """Small helper for building QueryStage rows consistently."""

    @staticmethod
    def make(stage: str, output: Dict[str, Any], input: Optional[Dict[str, Any]] = None) -> QueryStage:
        return QueryStage(stage=stage, input=input, output=output)
