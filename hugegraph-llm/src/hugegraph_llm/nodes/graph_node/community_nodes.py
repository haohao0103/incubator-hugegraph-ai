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

"""Community detection and global search pipeline nodes."""

from typing import Any, Dict

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.graph_op.community_detect import CommunityDetect
from hugegraph_llm.operators.index_op.build_community_index import (
    BuildCommunityIndex,
    CommunityIndexQuery,
)
from hugegraph_llm.operators.llm_op.community_report import CommunityReportGenerate
from hugegraph_llm.operators.llm_op.global_search import GlobalSearch
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class CommunityDetectNode(BaseNode):
    """Detect communities in the knowledge graph."""

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, client: Any = None, algorithm: str = "leiden", max_levels: int = 2):
        super().__init__()
        self._client = client
        self._algorithm = algorithm
        self._max_levels = max_levels

    def node_init(self):
        self._detector = CommunityDetect(
            client=self._client,
            algorithm=self._algorithm,
            max_levels=self._max_levels,
        )

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self._detector.run(data_json)


class CommunityReportNode(BaseNode):
    """Generate LLM reports for detected communities."""

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, llm: Any = None):
        super().__init__()
        self._llm = llm

    def node_init(self):
        self._reporter = CommunityReportGenerate(llm=self._llm)

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self._reporter.run(data_json)


class BuildCommunityIndexNode(BaseNode):
    """Build vector index for community reports."""

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, vector_index_cls: Any = None, embedding: Any = None):
        super().__init__()
        self._vector_index_cls = vector_index_cls
        self._embedding = embedding

    def node_init(self):
        self._builder = BuildCommunityIndex(
            vector_index=self._vector_index_cls,
            embedding=self._embedding,
        )

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self._builder.run(data_json)


class CommunityIndexQueryNode(BaseNode):
    """Query the community index for relevant reports."""

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, vector_index_cls: Any = None, embedding: Any = None, top_k: int = 10):
        super().__init__()
        self._vector_index_cls = vector_index_cls
        self._embedding = embedding
        self._top_k = top_k

    def node_init(self):
        self._querier = CommunityIndexQuery(
            vector_index=self._vector_index_cls,
            embedding=self._embedding,
            top_k=self._top_k,
        )

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self._querier.run(data_json)


class GlobalSearchNode(BaseNode):
    """Execute MapReduce Global Search over community reports."""

    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def __init__(self, llm: Any = None):
        super().__init__()
        self._llm = llm

    def node_init(self):
        self._searcher = GlobalSearch(llm=self._llm)

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self._searcher.run(data_json)
