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

"""E2E RAG pipeline node.

Unified entry node that accepts a stage parameter to determine whether
to execute build, query, or refresh workflows.
"""

from typing import Any, Dict

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.rag_op.e2e_rag_pipeline import (
    E2ERAGPipeline,
    PipelineConfig,
    PipelineResult,
)


class E2ERAGNode(BaseNode):
    """Execute E2E RAG pipeline (build / query / refresh / assess).

    The stage is determined by data_json["stage"].
    """

    context: None
    wk_input: None

    def __init__(
        self,
        llm=None,
        embedding=None,
        graph_client=None,
        vector_index_cls=None,
        config=None,
    ):
        super().__init__()
        self._llm = llm
        self._embedding = embedding
        self._graph_client = graph_client
        self._vector_index_cls = vector_index_cls
        self._config = config or PipelineConfig()
        self._pipeline = None

    def node_init(self):
        self._pipeline = E2ERAGPipeline(
            llm=self._llm,
            embedding=self._embedding,
            graph_client=self._graph_client,
            vector_index_cls=self._vector_index_cls,
            config=self._config,
        )

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        stage = data_json.get("stage", "query")
        result: PipelineResult

        if stage == "build":
            documents = data_json.get("documents", [])
            options = data_json.get("options", {})
            result = self._pipeline.build(documents, options)
        elif stage == "query":
            question = data_json.get("question", "")
            context = data_json.get("context", None)
            mode = data_json.get("mode", "auto")
            result = self._pipeline.query(question, context, mode)
        elif stage == "refresh":
            scope = data_json.get("scope", "stale")
            options = data_json.get("options", {})
            result = self._pipeline.refresh(scope, options)
        elif stage == "assess":
            result = self._pipeline.assess()
        else:
            return {"error": f"Unknown stage: {stage}"}

        return result.to_dict()
