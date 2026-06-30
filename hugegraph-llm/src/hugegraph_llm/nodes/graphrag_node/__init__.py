# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not with this file except in compliance
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
GraphRAG-specific nodes for the pipeline architecture (LightRAG-style).

Core nodes:
- IncrementalUpdateNode: Append-only incremental graph updates
- DualLevelRetrievalNode: Entity-centric + relationship-centric retrieval
- QueryPlannerNode: Simplified query level planning

Optional/deferred nodes (available for future use):
- CommunityDetectionNode: Louvain/Leiden community detection
- CommunitySummaryNode: Hierarchical community summaries
- DriftSearchNode: Hybrid global+local search
- EvaluationNode: Multi-dimensional evaluation
- NLPExtractNode: NLP-based low-cost extraction
"""

from hugegraph_llm.nodes.graphrag_node.community_detection_node import CommunityDetectionNode
from hugegraph_llm.nodes.graphrag_node.community_summary_node import CommunitySummaryNode
from hugegraph_llm.nodes.graphrag_node.drift_search_node import DriftSearchNode
from hugegraph_llm.nodes.graphrag_node.dual_level_retrieval_node import DualLevelRetrievalNode
from hugegraph_llm.nodes.graphrag_node.evaluation_node import EvaluationNode
from hugegraph_llm.nodes.graphrag_node.incremental_update_node import IncrementalUpdateNode
from hugegraph_llm.nodes.graphrag_node.nlp_extract_node import NLPExtractNode
from hugegraph_llm.nodes.graphrag_node.query_planner_node import QueryPlannerNode

__all__ = [
    "CommunityDetectionNode",
    "CommunitySummaryNode",
    "DriftSearchNode",
    "DualLevelRetrievalNode",
    "EvaluationNode",
    "IncrementalUpdateNode",
    "NLPExtractNode",
    "QueryPlannerNode",
]
