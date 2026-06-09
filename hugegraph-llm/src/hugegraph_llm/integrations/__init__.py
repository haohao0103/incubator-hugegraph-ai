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
LangChain integration for HugeGraph GraphRAG.

Provides standard LangChain interfaces:
- HugeGraphVectorStore: LangChain VectorStore
- HugeGraphRetriever: LangChain BaseRetriever
- HugeGraphQAChain: GraphRAG question-answering chain
- DriftRetriever: DRIFT search as a LangChain retriever
- Agent Tools: 7 HugeGraph tools for LangChain agents

Installation: pip install -e ".[langchain]"
"""

from hugegraph_llm.integrations.langchain.graph_retriever import HugeGraphRetriever
from hugegraph_llm.integrations.langchain.graph_qa_chain import HugeGraphQAChain
from hugegraph_llm.integrations.langchain.vector_store import HugeGraphVectorStore
from hugegraph_llm.integrations.langchain.drift_retriever import DriftRetriever
from hugegraph_llm.integrations.langchain.agent_tools import (
    create_hugegraph_tools,
)

__all__ = [
    "HugeGraphVectorStore",
    "HugeGraphRetriever",
    "HugeGraphQAChain",
    "DriftRetriever",
    "create_hugegraph_tools",
]
