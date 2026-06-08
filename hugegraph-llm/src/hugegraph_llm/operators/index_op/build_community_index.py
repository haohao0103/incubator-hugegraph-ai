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

"""Build community report vector index for semantic retrieval.

Indexes community report summaries into a FAISS vector store so
that Global Search can efficiently find communities relevant to a query.
"""

import json
import os
from typing import Any, Dict, List

from hugegraph_llm.config import huge_settings
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.log import log


class BuildCommunityIndex:
    """Build a vector index of community report summaries.

    Embeds each community report's summary text and stores it
    in a FAISS index for efficient semantic retrieval during
    Global Search.

    Index naming follows existing pattern:
        {resource_path}/{graph_name}/communities/

    Usage:
        builder = BuildCommunityIndex(
            vector_index=FaissVectorIndex,
            embedding=embedding_model,
        )
        context = builder.run(context)
    """

    def __init__(
        self,
        vector_index: type[VectorStoreBase],
        embedding: BaseEmbedding,
    ):
        """Initialize the community index builder.

        Args:
            vector_index: Vector store class (FAISS/Milvus/Qdrant).
            embedding: Embedding model instance.
        """
        self._vector_index_cls = vector_index
        self._embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the community report vector index.

        Reads from context:
            community_reports: List of community report dicts.

        Writes to context:
            community_index_built: Whether the index was successfully built.
            community_index_count: Number of reports indexed.
        """
        reports = context.get("community_reports", [])
        if not reports:
            log.warning("No community reports to index.")
            context["community_index_built"] = False
            context["community_index_count"] = 0
            return context

        embed_dim = self._embedding.get_embedding_dim()
        graph_name = huge_settings.graph_name

        # Build text representations for each report
        report_texts = []
        report_data = []  # Store full report as property
        for report in reports:
            text = (
                f"Title: {report.get('title', '')}\n"
                f"Summary: {report.get('summary', '')}\n"
                f"Key Entities: {', '.join(report.get('key_entities', []))}\n"
                f"Patterns: {'; '.join(report.get('relationship_patterns', []))}"
            )
            report_texts.append(text)
            report_data.append(json.dumps(report, ensure_ascii=False))

        # Batch embed
        embeddings = self._embedding.get_texts_embeddings(report_texts)

        # Build FAISS index
        index = self._vector_index_cls.from_name(
            embed_dim, graph_name, "communities"
        )
        index.add(embeddings, report_data)
        index.save_index_by_name(graph_name, "communities")

        context["community_index_built"] = True
        context["community_index_count"] = len(reports)
        log.info(
            "Built community index: %d reports indexed for graph '%s'",
            len(reports),
            graph_name,
        )
        return context


class CommunityIndexQuery:
    """Query the community report vector index for relevant communities."""

    def __init__(
        self,
        vector_index: type[VectorStoreBase],
        embedding: BaseEmbedding,
        top_k: int = 10,
    ):
        """Initialize the community index querier."""
        self._vector_index_cls = vector_index
        self._embedding = embedding
        self._top_k = top_k

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Query the community index for communities relevant to the query.

        Reads from context:
            query: The user's question.

        Writes to context:
            community_matches: List of matched community report dicts.
        """
        query = context.get("query", "")
        if not query:
            context["community_matches"] = []
            return context

        embed_dim = self._embedding.get_embedding_dim()
        graph_name = huge_settings.graph_name

        query_embedding = self._embedding.get_texts_embeddings([query])[0]
        index = self._vector_index_cls.from_name(
            embed_dim, graph_name, "communities"
        )

        try:
            results = index.search(query_embedding, self._top_k, dis_threshold=2.0)
        except Exception as e:
            log.warning("Community index search failed: %s", e)
            results = []

        # Parse stored JSON reports back to dicts
        parsed_reports = []
        for result in results:
            try:
                report = json.loads(result)
                parsed_reports.append(report)
            except json.JSONDecodeError:
                log.warning("Failed to parse community report from index")

        context["community_matches"] = parsed_reports
        log.debug("Community index query returned %d matches", len(parsed_reports))
        return context
