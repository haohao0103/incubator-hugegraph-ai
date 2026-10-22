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

"""Skill Graph engine — dependency-aware skill retrieval for HugeGraph-AI.

Layering (mirrors Graph-of-Skills / SkillNet / SkillDAG):

  - L0 ontology & schema  : :mod:`schema`
  - L1 offline construction: :mod:`skill_loader` + :mod:`graph_builder` + :mod:`embedder`
  - L2 online retrieval    : :mod:`retriever`
  - L3 integration & eval  : :mod:`cli`

Public API::

    from hugegraph_llm.engines.skill_graph import (
        SkillGraph, load_skills, GraphBuilder, SkillRetriever,
    )

    graph = load_skills("path/to/skills")
    graph, report = GraphBuilder().build(graph)

    retriever = SkillRetriever(graph)
    result = retriever.retrieve("build a RAG pipeline", token_budget=2000)
    print(result.to_context())
"""

from hugegraph_llm.engines.skill_graph.edge_inducer import (
    InductionReport,
    LLMEdgeInducer,
)
from hugegraph_llm.engines.skill_graph.embedder import (
    Embedder,
    HashingEmbedder,
    mean_vector,
    resolve_embedder,
)
from hugegraph_llm.engines.skill_graph.graph_builder import (
    BuildReport,
    GraphBuilder,
    write_to_hugegraph,
)
from hugegraph_llm.engines.skill_graph.retriever import (
    RetrievalResult,
    RetrievedSkill,
    ServiceSkillRetriever,
    SkillRetriever,
)
from hugegraph_llm.engines.skill_graph.schema import (
    SYMMETRIC_EDGE_TYPES,
    EdgeType,
    NodeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
    category_id,
    cosine,
    skill_id,
    slugify,
)
from hugegraph_llm.engines.skill_graph.service_loader import (
    load_graph_from_services,
)
from hugegraph_llm.engines.skill_graph.skill_loader import (
    load_skill_file,
    load_skills,
    parse_front_matter,
)

__all__ = [
    # schema / ontology
    "EdgeType",
    "NodeType",
    "SkillNode",
    "SkillEdge",
    "SkillGraph",
    "SYMMETRIC_EDGE_TYPES",
    "skill_id",
    "category_id",
    "slugify",
    "cosine",
    # loader
    "load_skills",
    "load_skill_file",
    "parse_front_matter",
    # embedding
    "Embedder",
    "HashingEmbedder",
    "resolve_embedder",
    "mean_vector",
    # builder
    "GraphBuilder",
    "BuildReport",
    "write_to_hugegraph",
    # typed edge induction (SkillDAG recipe)
    "LLMEdgeInducer",
    "InductionReport",
    # retriever
    "SkillRetriever",
    "ServiceSkillRetriever",
    "RetrievalResult",
    "RetrievedSkill",
    # service-backed loading (production source of truth)
    "load_graph_from_services",
]