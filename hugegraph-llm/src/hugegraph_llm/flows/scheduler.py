#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import threading
from typing import Any, Dict

from pycgraph import GPipeline, GPipelineManager

from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.agent_flow import AgentFlow
from hugegraph_llm.flows.community_flow import CommunityDetectionFlow, GlobalSearchFlow
from hugegraph_llm.flows.provenance_flow import ProvenanceAwareKGFlow
from hugegraph_llm.flows.build_example_index import BuildExampleIndexFlow
from hugegraph_llm.flows.build_schema import BuildSchemaFlow
from hugegraph_llm.flows.build_vector_index import BuildVectorIndexFlow
from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.flows.get_graph_index_info import GetGraphIndexInfoFlow
from hugegraph_llm.flows.graph_extract import GraphExtractFlow
from hugegraph_llm.flows.import_graph_data import ImportGraphDataFlow
from hugegraph_llm.flows.incremental_index_flow import IncrementalIndexFlow
from hugegraph_llm.flows.drift_flow import DriftFlow
from hugegraph_llm.flows.prompt_generate import PromptGenerateFlow
from hugegraph_llm.flows.rag_flow_graph_only import RAGGraphOnlyFlow
from hugegraph_llm.flows.rag_flow_graph_vector import RAGGraphVectorFlow
from hugegraph_llm.flows.rag_flow_raw import RAGRawFlow
from hugegraph_llm.flows.rag_flow_vector_only import RAGVectorOnlyFlow
from hugegraph_llm.flows.text2gremlin import Text2GremlinFlow
from hugegraph_llm.flows.update_vid_embeddings import UpdateVidEmbeddingsFlow
from hugegraph_llm.state.ai_state import WkFlowInput
from hugegraph_llm.utils.log import log


class Scheduler:
    pipeline_pool: Dict[str, Any]
    max_pipeline: int

    def __init__(self, max_pipeline: int = 10):
        self.pipeline_pool = {}
        # pipeline_pool act as a manager of GPipelineManager which used for pipeline management
        self.pipeline_pool[FlowName.BUILD_VECTOR_INDEX] = {
            "manager": GPipelineManager(),
            "flow": BuildVectorIndexFlow(),
        }
        self.pipeline_pool[FlowName.GRAPH_EXTRACT] = {
            "manager": GPipelineManager(),
            "flow": GraphExtractFlow(),
        }
        self.pipeline_pool[FlowName.IMPORT_GRAPH_DATA] = {
            "manager": GPipelineManager(),
            "flow": ImportGraphDataFlow(),
        }
        self.pipeline_pool[FlowName.UPDATE_VID_EMBEDDINGS] = {
            "manager": GPipelineManager(),
            "flow": UpdateVidEmbeddingsFlow(),
        }
        self.pipeline_pool[FlowName.GET_GRAPH_INDEX_INFO] = {
            "manager": GPipelineManager(),
            "flow": GetGraphIndexInfoFlow(),
        }
        self.pipeline_pool[FlowName.BUILD_SCHEMA] = {
            "manager": GPipelineManager(),
            "flow": BuildSchemaFlow(),
        }
        self.pipeline_pool[FlowName.PROMPT_GENERATE] = {
            "manager": GPipelineManager(),
            "flow": PromptGenerateFlow(),
        }
        self.pipeline_pool[FlowName.TEXT2GREMLIN] = {
            "manager": GPipelineManager(),
            "flow": Text2GremlinFlow(),
        }
        # New split rag pipelines
        self.pipeline_pool[FlowName.RAG_RAW] = {
            "manager": GPipelineManager(),
            "flow": RAGRawFlow(),
        }
        self.pipeline_pool[FlowName.RAG_VECTOR_ONLY] = {
            "manager": GPipelineManager(),
            "flow": RAGVectorOnlyFlow(),
        }
        self.pipeline_pool[FlowName.RAG_GRAPH_ONLY] = {
            "manager": GPipelineManager(),
            "flow": RAGGraphOnlyFlow(),
        }
        self.pipeline_pool[FlowName.RAG_GRAPH_VECTOR] = {
            "manager": GPipelineManager(),
            "flow": RAGGraphVectorFlow(),
        }
        self.pipeline_pool[FlowName.BUILD_EXAMPLES_INDEX] = {
            "manager": GPipelineManager(),
            "flow": BuildExampleIndexFlow(),
        }
        self.pipeline_pool[FlowName.AGENT] = {
            "manager": GPipelineManager(),
            "flow": AgentFlow(),
        }
        self.pipeline_pool[FlowName.COMMUNITY_DETECT] = {
            "manager": GPipelineManager(),
            "flow": CommunityDetectionFlow(),
        }
        self.pipeline_pool[FlowName.GLOBAL_SEARCH] = {
            "manager": GPipelineManager(),
            "flow": GlobalSearchFlow(),
        }
        self.pipeline_pool[FlowName.PROVENANCE_KG_BUILD] = {
            "manager": GPipelineManager(),
            "flow": ProvenanceAwareKGFlow(),
        }
        self.pipeline_pool[FlowName.INCREMENTAL_INDEX] = {
            "manager": GPipelineManager(),
            "flow": IncrementalIndexFlow(),
        }
        self.pipeline_pool[FlowName.DRIFT_SEARCH] = {
            "manager": GPipelineManager(),
            "flow": DriftFlow(),
        }
        self.max_pipeline = max_pipeline

    def agentic_flow(self, tool_registry=None, llm=None, **kwargs):
        """Execute the agentic workflow for multi-step graph reasoning.

        This replaces the stub implementation with a fully functional
        ReAct agent loop using the ToolRegistry and LLM.

        Args:
            tool_registry: ToolRegistry with registered tools.
            llm: LLM instance for agent reasoning.
            **kwargs: Additional parameters (query, max_steps, etc.).

        Returns:
            Dict with agent answer and execution trace.
        """
        if FlowName.AGENT not in self.pipeline_pool:
            raise ValueError("Agent flow is not registered in the pipeline pool.")

        flow_entry = self.pipeline_pool[FlowName.AGENT]
        flow: AgentFlow = flow_entry["flow"]

        # Configure the agent flow with provided dependencies
        if tool_registry is not None:
            flow.set_tool_registry(tool_registry)
        if llm is not None:
            flow.set_llm(llm)
        if "max_steps" in kwargs:
            flow.set_max_steps(kwargs["max_steps"])

        # Delegate to the standard schedule_flow mechanism
        return self.schedule_flow(FlowName.AGENT, **kwargs)

    def schedule_flow(self, flow_name: str, *args, **kwargs):
        if flow_name not in self.pipeline_pool:
            raise ValueError(f"Unsupported workflow {flow_name}")
        manager: GPipelineManager = self.pipeline_pool[flow_name]["manager"]
        flow: BaseFlow = self.pipeline_pool[flow_name]["flow"]
        pipeline: GPipeline = manager.fetch()
        if pipeline is None:
            # call coresponding flow_func to create new workflow
            pipeline = flow.build_flow(*args, **kwargs)
            status = pipeline.init()
            if status.isErr():
                error_msg = f"Error in flow init: {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            status = pipeline.run()
            if status.isErr():
                manager.add(pipeline)
                error_msg = f"Error in flow execution: {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            res = flow.post_deal(pipeline)
            manager.add(pipeline)
            return res
        try:
            # fetch pipeline & prepare input for flow
            prepared_input = pipeline.getGParamWithNoEmpty("wkflow_input")
            flow.prepare(prepared_input, *args, **kwargs)
            status = pipeline.run()
            if status.isErr():
                error_msg = f"Error in flow execution {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            res = flow.post_deal(pipeline)
        finally:
            manager.release(pipeline)
        return res

    async def schedule_stream_flow(self, flow_name: str, *args, **kwargs):
        if flow_name not in self.pipeline_pool:
            raise ValueError(f"Unsupported workflow {flow_name}")
        manager: GPipelineManager = self.pipeline_pool[flow_name]["manager"]
        flow: BaseFlow = self.pipeline_pool[flow_name]["flow"]
        pipeline: GPipeline = manager.fetch()
        if pipeline is None:
            # call coresponding flow_func to create new workflow
            pipeline = flow.build_flow(*args, **kwargs)
            pipeline.getGParamWithNoEmpty("wkflow_input").stream = True
            status = pipeline.init()
            if status.isErr():
                error_msg = f"Error in flow init: {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            status = pipeline.run()
            if status.isErr():
                manager.add(pipeline)
                error_msg = f"Error in flow execution: {status.getInfo()}"
                log.error(error_msg)
                raise RuntimeError(error_msg)
            async for res in flow.post_deal_stream(pipeline):
                yield res
            manager.add(pipeline)
        try:
            # fetch pipeline & prepare input for flow
            prepared_input: WkFlowInput = pipeline.getGParamWithNoEmpty("wkflow_input")
            prepared_input.stream = True
            flow.prepare(prepared_input, *args, **kwargs)
            status = pipeline.run()
            if status.isErr():
                raise RuntimeError(f"Error in flow execution {status.getInfo()}")
            async for res in flow.post_deal_stream(pipeline):
                yield res
        finally:
            manager.release(pipeline)


class SchedulerSingleton:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = Scheduler()
        return cls._instance
