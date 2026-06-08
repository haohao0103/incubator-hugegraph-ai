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

from typing import Optional

from .models import BaseConfig


class HugeGraphConfig(BaseConfig):
    """HugeGraph settings"""

    # graph server config
    graph_url: str = "127.0.0.1:8080"
    graph_name: str = "hugegraph"
    graph_user: str = "admin"
    graph_pwd: str = "xxx"
    graph_space: Optional[str] = None

    # graph query config
    limit_property: str = "False"
    max_graph_path: int = 10
    max_graph_items: int = 30
    edge_limit_pre_label: int = 8

    # vector config
    vector_dis_threshold: float = 0.9
    topk_per_keyword: int = 1

    # rerank config
    topk_return_results: int = 20

    # Community detection config
    community_detection_algorithm: str = "leiden"  # "leiden" or "louvain"
    max_community_levels: int = 2  # Hierarchical levels (1=flat)
    min_community_size: int = 3  # Minimum community size
    community_resolution: float = 1.0  # Modularity resolution
    max_community_reports: int = 100  # Max communities to generate reports for

    # Provenance config
    enable_provenance: bool = False  # Enable Document->Chunk->Entity provenance tracking

    # Entity Resolution config (Sprint 1)
    entity_resolution_threshold: float = 0.85  # Cosine similarity threshold
    entity_resolution_batch_size: int = 50  # LLM verify batch size
    entity_resolution_strategy: str = "hybrid"  # exact_match | embedding | llm_verify | hybrid

    # HyDE config (Sprint 3)
    enable_hyde: bool = False  # Enable HyDE query enhancement
    hyde_mode: str = "prefix"  # off | prefix | full
    hyde_max_query_length: int = 100  # Skip HyDE for queries longer than this
