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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multimodal GraphRAG index building flow.

Pipeline stages:
1. MultimodalExtract (PDF → images + text blocks)
2. VLMDescribe (images → structured descriptions)
3. ChunkSplit (text → chunks)
4. Schema (load graph schema)
5. Extract (chunks → entities + relations via LLM)
6. MultimodalKGBuild (images + descriptions → multimodal KG vertices/edges)
7. IncrementalUpdate (merge entities/edges with dedup)
8. Commit2Graph (write everything to HugeGraph)

When PDF is provided, stages 1-2-6 run; when plain text is provided,
they are skipped. This enables a single flow to handle both
text-only and multimodal indexing.
"""

import json

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.nodes.document_node.chunk_split import ChunkSplitNode
from hugegraph_llm.nodes.graphrag_node.incremental_update_node import IncrementalUpdateNode
from hugegraph_llm.nodes.hugegraph_node.commit_to_hugegraph import Commit2GraphNode
from hugegraph_llm.nodes.hugegraph_node.schema import SchemaNode
from hugegraph_llm.nodes.llm_node.extract_info import ExtractNode
from hugegraph_llm.nodes.multimodal_node.multimodal_extract_node import MultimodalExtractNode
from hugegraph_llm.nodes.multimodal_node.vlm_describe_node import VLMDescribeNode
from hugegraph_llm.nodes.multimodal_node.multimodal_kg_build_node import MultimodalKGBuildNode
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class MultimodalRAGIndexFlow(BaseFlow):
    """
    Multimodal GraphRAG index building flow.

    Extends the LightRAG-style GraphRAGIndexFlow with three additional
    multimodal stages: PDF extraction, VLM description, and multimodal
    KG construction. These stages are always registered but only produce
    output when a PDF file path is provided via WkFlowInput.
    """

    def prepare(
        self,
        prepared_input: WkFlowInput,
        schema,
        texts,
        language: str = "zh",
        incremental: bool = True,
        pdf_file_path: str = None,
        pdf_max_pages: int = 5,
        vlm_provider: str = "xiaomimo",
        vlm_max_images: int = 10,
        multimodal_kg_name: str = "multimodal_poc",
        **kwargs,
    ):
        prepared_input.texts = texts
        prepared_input.language = language
        prepared_input.split_type = "document"
        prepared_input.schema = schema
        prepared_input.extract_type = "property_graph"
        setattr(prepared_input, "incremental", incremental)
        setattr(prepared_input, "pdf_file_path", pdf_file_path)
        setattr(prepared_input, "pdf_max_pages", pdf_max_pages)
        setattr(prepared_input, "vlm_provider", vlm_provider)
        setattr(prepared_input, "vlm_max_images", vlm_max_images)
        setattr(prepared_input, "multimodal_kg_name", multimodal_kg_name)

    def build_flow(
        self,
        schema,
        texts,
        language="zh",
        incremental=True,
        pdf_file_path=None,
        pdf_max_pages=5,
        vlm_provider="xiaomimo",
        vlm_max_images=10,
        multimodal_kg_name="multimodal_poc",
        **kwargs,
    ):
        pipeline = GPipeline()
        prepared_input = WkFlowInput()
        self.prepare(
            prepared_input,
            schema,
            texts,
            language,
            incremental,
            pdf_file_path,
            pdf_max_pages,
            vlm_provider,
            vlm_max_images,
            multimodal_kg_name,
        )

        pipeline.createGParam(prepared_input, "wkflow_input")
        pipeline.createGParam(WkFlowState(), "wkflow_state")

        # Multimodal stages (1-3)
        mm_extract_node = MultimodalExtractNode()
        vlm_describe_node = VLMDescribeNode()
        mm_kg_build_node = MultimodalKGBuildNode()

        # Standard GraphRAG stages
        schema_node = SchemaNode()
        chunk_split_node = ChunkSplitNode()
        extract_node = ExtractNode()
        incremental_update_node = IncrementalUpdateNode()
        commit_node = Commit2GraphNode()

        # Register pipeline with multimodal stages integrated:
        #
        #   mm_extract ──→ vlm_describe ──→ mm_kg_build ──────────────┐
        #   schema ──────────────────────────────────────────────────────┤
        #   chunk_split ─────────────────────────────────────────────────┤
        #                                                                  ├─→ incremental_update ──→ commit
        #   extract ──────────────────────────────────────────────────────┘
        #
        # Multimodal nodes run in sequence; standard nodes run independently.
        # incremental_update and commit wait for ALL predecessors.

        pipeline.registerGElement(mm_extract_node, set(), "mm_extract")
        pipeline.registerGElement(vlm_describe_node, {mm_extract_node}, "vlm_describe")
        pipeline.registerGElement(schema_node, set(), "schema_node")
        pipeline.registerGElement(chunk_split_node, set(), "chunk_split")
        pipeline.registerGElement(
            extract_node,
            {schema_node, chunk_split_node},
            "llm_extract",
        )
        pipeline.registerGElement(
            mm_kg_build_node,
            {vlm_describe_node},
            "mm_kg_build",
        )
        pipeline.registerGElement(
            incremental_update_node,
            {extract_node, mm_kg_build_node},
            "incremental_update",
        )
        pipeline.registerGElement(
            commit_node,
            {incremental_update_node},
            "commit_to_graph",
        )

        log.info("MultimodalRAGIndexFlow pipeline built successfully")
        return pipeline

    def post_deal(self, pipeline=None, **kwargs):
        res = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
        vertices = res.get("vertices", [])
        edges = res.get("edges", [])
        update_summary = res.get("incremental_update_summary", {})

        # Multimodal-specific results
        mm_extracted = res.get("multimodal_extracted", False)
        total_images = res.get("total_images", 0)
        vlm_descriptions = res.get("vlm_descriptions", [])
        mm_kg_built = res.get("multimodal_kg_built", False)
        mm_kg_stats = res.get("multimodal_kg_stats", {})

        result = {
            "vertices_count": len(vertices) if vertices else 0,
            "edges_count": len(edges) if edges else 0,
            "incremental_update": update_summary,
            "multimodal": {
                "extracted": mm_extracted,
                "total_images": total_images,
                "vlm_descriptions_count": len(vlm_descriptions) if vlm_descriptions else 0,
                "kg_built": mm_kg_built,
                "kg_stats": mm_kg_stats,
            },
        }

        log.info("Multimodal RAG index built: %s", json.dumps(result, ensure_ascii=False))
        return json.dumps(result, ensure_ascii=False, indent=2)
