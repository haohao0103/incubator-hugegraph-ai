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

"""Bridge Feishu documents into the HugeGraph-AI graph/index flows.

The Feishu connector (``hugegraph_llm.document.feishu_connector``) produces
:class:`~hugegraph_llm.document.document_loader.Document` objects with
provenance metadata, while the existing indexing and graph-extraction flows
(``BuildVectorIndexFlow`` / ``GraphExtractFlow``) consume plain
``texts: List[str]``.  This module closes that gap: it loads a Feishu Wiki
space, extracts the document text, and feeds it into the graph and vector-index
flows via the scheduler.

PoC scope: Wiki + Docx body text only, manual / scheduled full + incremental
import.  No event subscription, no bitable/sheets/chat.

Usage::

    from hugegraph_llm.flows.feishu_ingest import FeishuIngestService

    result = FeishuIngestService.ingest_wiki(
        space_id="<wiki_space_id>",
        app_id="<feishu_app_id>",
        app_secret="<feishu_app_secret>",
        graph_schema='{"vertexlabels": [...], "edgelabels": [...]}',
    )
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.config import prompt
from hugegraph_llm.document.document_loader import Document
from hugegraph_llm.document.feishu_connector import FeishuConnector
from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.operators.document_op.chunk_split import SPLIT_TYPE_DOCUMENT
from hugegraph_llm.operators.graph_op.entity_resolution import EntityResolution
from hugegraph_llm.utils.log import log

MODE_FULL = "full"
MODE_INCREMENTAL = "incremental"
VALID_MODES = (MODE_FULL, MODE_INCREMENTAL)

# Key-value metadata carried on each loaded Feishu Document (see FeishuConnector).
_PROVENANCE_KEYS = ("title", "doc_token", "node_token", "url", "updated_time")


@dataclass
class FeishuIngestResult:
    """Summary of a Feishu ingestion run.

    ``graph_result`` / ``vector_result`` hold the JSON strings returned by the
    corresponding flows, or ``None`` when that stage was skipped.
    """

    space_id: str
    mode: str
    doc_count: int
    text_count: int
    docs: List[Dict[str, Any]] = field(default_factory=list)
    graph_result: Optional[str] = None
    vector_result: Optional[str] = None
    resolution_result: Optional[Dict[str, Any]] = None
    commit_result: Optional[str] = None
    vertex_count: int = 0
    edge_count: int = 0


class FeishuIngestService:
    """Load a Feishu Wiki space and feed it into the HugeGraph-AI flows."""

    @staticmethod
    def docs_to_texts(docs: List[Document]) -> List[str]:
        """Return the plain-text list consumed by the indexing/extraction flows.

        Non-empty document content is passed through verbatim (including any
        Markdown produced by the connector).
        """
        return [doc.content for doc in docs if doc.content and doc.content.strip()]

    @staticmethod
    def _docs_summary(docs: List[Document]) -> List[Dict[str, Any]]:
        """Collect provenance metadata for reporting/audit.

        Note: the existing flows only consume ``texts``, so this metadata is not
        (yet) propagated into graph vertices/edges or vector records; it is
        returned here so callers can keep an ingestion audit trail.
        """
        summary: List[Dict[str, Any]] = []
        for doc in docs:
            entry: Dict[str, Any] = {"doc_id": doc.doc_id}
            for key in _PROVENANCE_KEYS:
                entry[key] = doc.metadata.get(key)
            summary.append(entry)
        return summary

    @staticmethod
    def _load_docs(
        connector: FeishuConnector,
        space_id: str,
        mode: str,
        since_timestamp: Optional[int],
    ) -> List[Document]:
        """Load Feishu documents for the given mode (full or incremental)."""
        if mode == MODE_FULL:
            return connector.load_wiki_space(space_id)
        if mode == MODE_INCREMENTAL:
            if since_timestamp is None:
                raise ValueError("mode='incremental' requires a `since_timestamp` (epoch seconds).")
            return connector.load_wiki_space_incremental(space_id, since_timestamp)
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    @classmethod
    def ingest_wiki(
        cls,
        space_id: str,
        app_id: str = "",
        app_secret: str = "",
        base_url: str = "https://open.feishu.cn",
        timeout: int = 30,
        mode: str = MODE_FULL,
        since_timestamp: Optional[int] = None,
        graph_schema: Optional[str] = None,
        example_prompt: Optional[str] = None,
        extract_type: str = "property_graph",
        split_type: str = SPLIT_TYPE_DOCUMENT,
        language: str = "zh",
        client_config: Any = None,
        build_graph: bool = True,
        build_index: bool = True,
        connector: Optional[FeishuConnector] = None,
    ) -> FeishuIngestResult:
        """Load a Feishu Wiki space and run the requested flows.

        Args:
            space_id: Feishu Wiki space id.
            app_id: Feishu self-built app id (ignored when ``connector`` is given).
            app_secret: Feishu self-built app secret (ignored when ``connector`` is given).
            base_url: Open platform domain (Feishu vs Lark international).
            timeout: HTTP timeout in seconds for the connector.
            mode: ``"full"`` (every Docx) or ``"incremental"`` (updated after ``since_timestamp``).
            since_timestamp: Unix epoch seconds; required when ``mode="incremental"``.
            graph_schema: Graph schema (JSON string or existing graph name) for ``GraphExtractFlow``.
            example_prompt: Optional graph-extraction prompt header (defaults to the configured prompt).
            extract_type: Extraction type forwarded to ``GraphExtractFlow``.
            split_type: Chunk granularity forwarded to ``GraphExtractFlow``.
            language: ``"zh"`` or ``"en"``.
            client_config: Optional request-scoped HugeGraph connection (see ``GraphExtractRequest``).
            build_graph: Run ``GraphExtractFlow`` after loading.
            build_index: Run ``BuildVectorIndexFlow`` after loading.
            connector: Pre-built ``FeishuConnector`` (useful for testing/injection).

        Returns:
            A summary of the run, including per-document provenance metadata.
        """
        connector = connector or FeishuConnector(
            app_id=app_id,
            app_secret=app_secret,
            base_url=base_url,
            timeout=timeout,
        )

        docs = cls._load_docs(connector, space_id, mode, since_timestamp)
        texts = cls.docs_to_texts(docs)
        result = FeishuIngestResult(
            space_id=space_id,
            mode=mode,
            doc_count=len(docs),
            text_count=len(texts),
            docs=cls._docs_summary(docs),
        )

        if not texts:
            log.warning(
                "No Feishu documents (or empty text) found for space %s; skipping flows.",
                space_id,
            )
            return result

        if not (build_graph or build_index):
            return result

        scheduler = SchedulerSingleton.get_instance()

        if build_graph:
            if not graph_schema:
                raise ValueError("build_graph=True requires a `graph_schema` (JSON string or existing graph name).")
            result.graph_result = scheduler.schedule_flow(
                FlowName.GRAPH_EXTRACT,
                graph_schema,
                texts,
                example_prompt or prompt.extract_graph_prompt,
                extract_type,
                language=language,
                split_type=split_type,
                client_config=client_config,
            )

        if build_index:
            result.vector_result = scheduler.schedule_flow(FlowName.BUILD_VECTOR_INDEX, texts)

        return result

    @classmethod
    def ingest_wiki_full(
        cls,
        space_id: str,
        app_id: str = "",
        app_secret: str = "",
        base_url: str = "https://open.feishu.cn",
        timeout: int = 30,
        mode: str = MODE_FULL,
        since_timestamp: Optional[int] = None,
        graph_schema: Optional[str] = None,
        auto_schema: bool = False,
        few_shot_schema: Optional[Any] = None,
        example_prompt: Optional[str] = None,
        extract_type: str = "property_graph",
        split_type: str = SPLIT_TYPE_DOCUMENT,
        language: str = "zh",
        client_config: Any = None,
        resolve: bool = False,
        resolver: Optional[EntityResolution] = None,
        commit: bool = False,
        build_index: bool = True,
        connector: Optional[FeishuConnector] = None,
    ) -> FeishuIngestResult:
        """Full KG pipeline: load → schema → extract → resolve → commit → index.

        Extends :meth:`ingest_wiki` with the steps needed to actually land a
        deduplicated graph into HugeGraph (rather than just returning extracted
        vertices/edges):

        1. load Feishu documents and extract ``texts``;
        2. resolve a schema (explicit ``graph_schema`` or LLM-inferred via
           ``auto_schema=True`` using ``BuildSchemaFlow``);
        3. extract vertices/edges via ``GraphExtractFlow``;
        4. optionally merge duplicate entities with ``resolver``
           (``EntityResolution.resolve_in_memory_pure``, pre-commit);
        5. optionally commit via ``ImportGraphDataFlow``;
        6. optionally build the vector index.

        Args:
            resolve: Run pre-commit entity resolution on extracted vertices.
            resolver: Pre-built ``EntityResolution`` (required when ``resolve=True``).
            commit: Commit the (deduplicated) graph to HugeGraph.
            auto_schema: Infer a schema draft from ``texts`` via ``BuildSchemaFlow``
                (ignored when ``graph_schema`` is provided).
            few_shot_schema: Optional few-shot schema for ``auto_schema``.
            See :meth:`ingest_wiki` for the remaining arguments.

        Returns:
            A summary including ``vertex_count``/``edge_count`` after resolution,
            and ``resolution_result``/``commit_result`` when those stages ran.
        """
        connector = connector or FeishuConnector(
            app_id=app_id,
            app_secret=app_secret,
            base_url=base_url,
            timeout=timeout,
        )

        docs = cls._load_docs(connector, space_id, mode, since_timestamp)
        texts = cls.docs_to_texts(docs)
        result = FeishuIngestResult(
            space_id=space_id,
            mode=mode,
            doc_count=len(docs),
            text_count=len(texts),
            docs=cls._docs_summary(docs),
        )

        if not texts:
            log.warning(
                "No Feishu documents (or empty text) found for space %s; skipping flows.",
                space_id,
            )
            return result

        scheduler = SchedulerSingleton.get_instance()

        # 1. Resolve schema (explicit wins over auto-inference).
        if graph_schema:
            schema = graph_schema
        elif auto_schema:
            schema = scheduler.schedule_flow(FlowName.BUILD_SCHEMA, texts, None, few_shot_schema)
            if not schema:
                raise ValueError("auto_schema=True produced an empty schema from BuildSchemaFlow.")
        else:
            raise ValueError("Provide a `graph_schema` or set `auto_schema=True`.")

        # 2. Extract vertices/edges.
        graph_result = scheduler.schedule_flow(
            FlowName.GRAPH_EXTRACT,
            schema,
            texts,
            example_prompt or prompt.extract_graph_prompt,
            extract_type,
            language=language,
            split_type=split_type,
            client_config=client_config,
        )
        result.graph_result = graph_result
        graph = json.loads(graph_result) if isinstance(graph_result, str) else graph_result
        vertices = list(graph.get("vertices", []))
        edges = list(graph.get("edges", []))

        # 3. Pre-commit entity resolution (dedupe vertices, repoint edges).
        if resolve:
            if resolver is None:
                raise ValueError("resolve=True requires a `resolver` (EntityResolution instance).")
            resolved = resolver.resolve_in_memory_pure(vertices, edges)
            vertices = resolved["vertices"]
            edges = resolved["edges"]
            result.resolution_result = resolved["resolution_result"]

        result.vertex_count = len(vertices)
        result.edge_count = len(edges)

        # 4. Commit to HugeGraph.
        if commit:
            data_str = json.dumps({"vertices": vertices, "edges": edges}, ensure_ascii=False)
            result.commit_result = scheduler.schedule_flow(FlowName.IMPORT_GRAPH_DATA, data_str, schema)

        # 5. Vector index.
        if build_index:
            result.vector_result = scheduler.schedule_flow(FlowName.BUILD_VECTOR_INDEX, texts)

        return result
