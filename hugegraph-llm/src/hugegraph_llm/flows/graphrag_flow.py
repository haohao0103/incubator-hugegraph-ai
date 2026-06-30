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

"""
LightRAG-style GraphRAG query flow.

Replaces the previous Microsoft GraphRAG-style flow that required:
- Community detection
- Hierarchical community summaries
- DRIFT search (depends on communities)

With a simpler, production-ready LightRAG flow:
1. Query Planning → determine retrieval level (LOW/HIGH/HYBRID)
2. Keyword Extraction + Semantic ID Query (vector side)
3. Schema + Graph Query (graph side)
4. Dual-Level Retrieval (entity-centric + relationship-centric)
5. Merge Rerank
6. Answer Synthesize

No community detection dependency. No global restructuring.
This is the pragmatic "Plan A" approach for quick production landing.
"""

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.common_node.merge_rerank_node import MergeRerankNode
from hugegraph_llm.nodes.graphrag_node.dual_level_retrieval_node import DualLevelRetrievalNode
from hugegraph_llm.nodes.graphrag_node.query_planner_node import QueryPlannerNode
from hugegraph_llm.nodes.hugegraph_node.graph_query_node import GraphQueryNode
from hugegraph_llm.nodes.hugegraph_node.schema import SchemaNode
from hugegraph_llm.nodes.index_node.semantic_id_query_node import SemanticIdQueryNode
from hugegraph_llm.nodes.llm_node.answer_synthesize_node import AnswerSynthesizeNode
from hugegraph_llm.nodes.llm_node.keyword_extract_node import KeywordExtractNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


# pylint: disable=arguments-differ,keyword-arg-before-vararg
class GraphRAGFlow(BaseFlow):
    """
    LightRAG-style GraphRAG query flow.

    Uses dual-level retrieval (entity-centric + relationship-centric)
    instead of community-detection-based DRIFT search.

    Pipeline stages:
    1. Query Planning → determine level (LOW/HIGH/HYBRID)
    2. Keyword Extraction + Semantic ID Query (vector side)
    3. Schema + Graph Query (graph side)
    4. Dual-Level Retrieval (merge entity + relationship results)
    5. Merge Rerank
    6. Answer Synthesize
    """

    def prepare(
        self,
        prepared_input: WkFlowInput,
        query: str,
        vector_search: bool = True,
        graph_search: bool = True,
        raw_answer: bool = False,
        vector_only_answer: bool = False,
        graph_only_answer: bool = False,
        graph_vector_answer: bool = True,
        rerank_method: str = "bleu",
        near_neighbor_first: bool = False,
        custom_related_information: str = "",
        answer_prompt: str = None,
        keywords_extract_prompt: str = None,
        gremlin_tmpl_num: int = -1,
        gremlin_prompt: str = None,
        max_graph_items: int = None,
        topk_return_results: int = None,
        vector_dis_threshold: float = None,
        topk_per_keyword: int = None,
        **kwargs,
    ):
        prepared_input.query = query
        prepared_input.vector_search = vector_search
        prepared_input.graph_search = graph_search
        prepared_input.raw_answer = raw_answer
        prepared_input.vector_only_answer = vector_only_answer
        prepared_input.graph_only_answer = graph_only_answer
        prepared_input.graph_vector_answer = graph_vector_answer
        prepared_input.rerank_method = rerank_method
        prepared_input.near_neighbor_first = near_neighbor_first
        prepared_input.custom_related_information = custom_related_information
        prepared_input.answer_prompt = answer_prompt
        prepared_input.keywords_extract_prompt = keywords_extract_prompt
        prepared_input.gremlin_tmpl_num = gremlin_tmpl_num
        prepared_input.gremlin_prompt = gremlin_prompt
        prepared_input.max_graph_items = max_graph_items
        prepared_input.topk_return_results = topk_return_results
        prepared_input.vector_dis_threshold = vector_dis_threshold
        prepared_input.topk_per_keyword = topk_per_keyword

    def build_flow(self, **kwargs):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        self.prepare(prepared_input, **kwargs)

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Stage 1: Query Planning (simplified, no community dependency)
        query_planner_node = QueryPlannerNode()

        # Stage 2: Vector side
        keyword_extract_node = KeywordExtractNode("graphrag_keyword")
        semantic_id_query_node = SemanticIdQueryNode({keyword_extract_node}, "graphrag_semantic")

        # Stage 3: Graph side
        schema_node = SchemaNode()
        graph_query_node = GraphQueryNode("graphrag_graph")

        # Stage 4: Dual-level retrieval (replaces DRIFT + community detection)
        dual_level_retrieval_node = DualLevelRetrievalNode()

        # Stage 5: Merge rerank
        merge_rerank_node = MergeRerankNode()

        # Stage 6: Answer synthesis
        answer_synthesize_node = AnswerSynthesizeNode()

        # Register pipeline
        pipeline.registerGElement(query_planner_node, set(), "query_planner")
        pipeline.registerGElement(keyword_extract_node, set(), "keyword_extract")
        pipeline.registerGElement(semantic_id_query_node, {keyword_extract_node}, "semantic_query")
        pipeline.registerGElement(schema_node, set(), "schema")
        pipeline.registerGElement(graph_query_node, {schema_node, semantic_id_query_node}, "graph_query")
        pipeline.registerGElement(dual_level_retrieval_node, {graph_query_node}, "dual_level_retrieval")
        pipeline.registerGElement(merge_rerank_node, {dual_level_retrieval_node}, "merge_rerank")
        pipeline.registerGElement(answer_synthesize_node, {merge_rerank_node}, "answer")

        log.info("GraphRAGFlow pipeline built successfully (LightRAG-style)")
        return pipeline

    def post_deal(self, pipeline=None, **kwargs):
        if pipeline is None:
            return {"error": "No pipeline provided"}
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        log.info("GraphRAGFlow post processing success")
        return {
            "graph_vector_answer": res.get("graph_vector_answer", ""),
            "graph_only_answer": res.get("graph_only_answer", ""),
            "vector_only_answer": res.get("vector_only_answer", ""),
            "query_intent": res.get("query_intent", ""),
            "retrieval_level": res.get("retrieval_level", ""),
        }
