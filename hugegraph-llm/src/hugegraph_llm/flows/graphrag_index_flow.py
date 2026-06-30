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
LightRAG-style GraphRAG index building flow.

Core pipeline:
1. Chunk split → 2. LLM extraction → 3. Incremental update → 4. Commit to graph

Key differences from the previous Microsoft GraphRAG-style flow:
- Incremental update is the CORE step (not community detection)
- Entity name as primary key enables append-only updates
- No mandatory community detection / hierarchical summaries
- Community detection is available as an OPTIONAL post-processing step

This is the "Plan A" approach for quick production landing,
following the LightRAG pattern validated by Huolala (货拉拉元初团队).
"""

import json

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.document_node.chunk_split import ChunkSplitNode
from hugegraph_llm.nodes.graphrag_node.incremental_update_node import IncrementalUpdateNode
from hugegraph_llm.nodes.hugegraph_node.commit_to_hugegraph import Commit2GraphNode
from hugegraph_llm.nodes.hugegraph_node.schema import SchemaNode
from hugegraph_llm.nodes.llm_node.extract_info import ExtractNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


# pylint: disable=arguments-differ,keyword-arg-before-vararg
class GraphRAGIndexFlow(BaseFlow):
    """
    LightRAG-style GraphRAG index building flow.

    Builds a GraphRAG index from documents using LLM extraction and
    incremental graph updates. Entity name as primary key enables
    append-only indexing without full graph rebuild.
    """

    def prepare(
        self,
        prepared_input: WkFlowInput,
        schema,
        texts,
        language: str = "zh",
        incremental: bool = True,
        **kwargs,
    ):
        prepared_input.texts = texts
        prepared_input.language = language
        prepared_input.split_type = "document"
        prepared_input.schema = schema
        prepared_input.extract_type = "property_graph"
        setattr(prepared_input, "incremental", incremental)

    def build_flow(self, schema, texts, language="zh", incremental=True, **kwargs):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        self.prepare(prepared_input, schema, texts, language, incremental)

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Build pipeline stages (LightRAG-style: simple and effective)
        schema_node = SchemaNode()
        chunk_split_node = ChunkSplitNode()
        extract_node = ExtractNode()

        # Incremental update is the CORE step
        incremental_update_node = IncrementalUpdateNode()

        # Commit to graph
        commit_node = Commit2GraphNode()

        # Register pipeline: schema & chunk_split are independent,
        # then extract → incremental_update → commit
        pipeline.registerGElement(schema_node, set(), "schema_node")
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(
            extract_node,
            {schema_node, chunk_split_node},
            "llm_extract",
        )
        pipeline.registerGElement(
            incremental_update_node,
            {extract_node},
            "incremental_update",
        )
        pipeline.registerGElement(
            commit_node,
            {incremental_update_node},
            "commit_to_graph",
        )

        log.info("GraphRAGIndexFlow pipeline built successfully (LightRAG-style)")
        return pipeline

    def post_deal(self, pipeline=None, **kwargs):
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        vertices = res.get("vertices", [])
        edges = res.get("edges", [])
        update_summary = res.get("incremental_update_summary", {})

        result = {
            "vertices_count": len(vertices) if vertices else 0,
            "edges_count": len(edges) if edges else 0,
            "incremental_update": update_summary,
        }

        log.info("GraphRAG index built: %s", json.dumps(result, ensure_ascii=False))
        return json.dumps(result, ensure_ascii=False, indent=2)
