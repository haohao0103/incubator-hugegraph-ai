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
GraphRAG Enhancements — P0 Gap Closure (对标 LightRAG v1.5.4 + MS GraphRAG)

5 modules that close the capability gap identified in GRAPHRAG_FRAMEWORK_GAP_ANALYSIS.md:

| Gap | Module | Source | Benefit |
|-----|--------|--------|---------|
| G1 | gleaning_extractor | LightRAG | KG quality +20-30% |
| G2 | llm_cache | MS GraphRAG | LLM cost -40-60% |
| G3 | token_budget | MS GraphRAG | OOM protection |
| G4 | global_retriever | MS+LR | Complex QA reasoning |
| G5 | community_detector | MS GraphRAG | Global understanding |
"""

from hugegraph_llm.operators.graph_rag_enhancements.llm_cache import (
    BaseCache,
    InMemoryCache,
    JsonFileCache,
    NoopCache,
    CacheStats,
    create_cache_key,
    create_llm_cache,
)
from hugegraph_llm.operators.graph_rag_enhancements.token_budget import (
    TokenCounter,
    TokenBudgetManager,
    BudgetExceededError,
    BudgetConfig,
    SlidingWindowRateLimiter,
    LLMCallGuard,
)
from hugegraph_llm.operators.graph_rag_enhancements.gleaning_extractor import (
    GleaningExtractor,
    ExtractionResult,
    GleaningConfig,
)
from hugegraph_llm.operators.graph_rag_enhancements.community_detector import (
    CommunityDetector,
    CommunityReporter,
    CommunityReport,
    FindingModel,
    CommunityConfig,
    ClusteringResult,
)
from hugegraph_llm.operators.graph_rag_enhancements.global_retriever import (
    GlobalSearchRetriever,
    DriftChainBuilder,
    SearchResult,
    RetrievedContext,
    GlobalSearchConfig,
)

__all__ = [
    # G2: LLM Cache
    "BaseCache", "InMemoryCache", "JsonFileCache", "NoopCache",
    "CacheStats", "create_cache_key", "create_llm_cache",
    # G3: Token Budget
    "TokenCounter", "TokenBudgetManager", "BudgetExceededError",
    "BudgetConfig", "SlidingWindowRateLimiter", "LLMCallGuard",
    # G1: Gleaning
    "GleaningExtractor", "ExtractionResult", "GleaningConfig",
    # G5: Community Detection
    "CommunityDetector", "CommunityReporter", "CommunityReport",
    "FindingModel", "CommunityConfig", "ClusteringResult",
    # G4: Global/DRIFT Retrieval
    "GlobalSearchRetriever", "DriftChainBuilder",
    "SearchResult", "RetrievedContext", "GlobalSearchConfig",
]
