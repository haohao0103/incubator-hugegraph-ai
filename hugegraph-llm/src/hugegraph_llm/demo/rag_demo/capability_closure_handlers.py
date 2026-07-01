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

"""Handlers for the Capability Closure Gradio tab.

Exposes the following capabilities in the UI:
- Property Graph Extraction
- Incremental Index Flow
- Gremlin Validator + Self-Correction Loop
- Query Classifier
- Synonym Manager
- Chunk Similarity Edges

Note: Multimodal RAG handlers have been removed; they are now
exclusively served by Tab 11 (Multimodal GraphRAG).
"""

import json
import os
from typing import Any, Dict, List, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.agents.agent_loop import QueryClassifier
from hugegraph_llm.config import huge_settings
from hugegraph_llm.document.chunk_split import ChunkSplitter
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.graph_op.chunk_sim_edges import ChunkSimEdgeBuilder
from hugegraph_llm.operators.graph_op.incremental_utils import find_affected_communities
from hugegraph_llm.operators.graph_op.synonym_manager import SynonymManager
from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import Commit2Graph
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.operators.llm_op.gremlin_validator import GremlinRetryLoop
from hugegraph_llm.operators.llm_op.info_extract import InfoExtract
from hugegraph_llm.operators.llm_op.property_graph_extract import PropertyGraphExtract
from hugegraph_llm.utils.log import log


# ── Helpers ───────────────────────────────────────────────────


def _get_graph_client() -> Optional[PyHugeClient]:
    try:
        return PyHugeClient(
            url=huge_settings.graph_url,
            graph=huge_settings.graph_name,
            user=huge_settings.graph_user,
            pwd=huge_settings.graph_pwd,
            graphspace=huge_settings.graph_space,
        )
    except Exception as e:
        log.warning("Failed to create PyHugeClient: %s", e)
        return None


def _serialize(result: Any) -> str:
    """Serialize result to JSON string, handling dataclasses and enums."""
    try:
        return json.dumps(result, ensure_ascii=False, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {e}", "raw": str(result)}, ensure_ascii=False, indent=2)


def _get_llm():
    try:
        return LLMs().get_llm()
    except Exception as e:
        log.warning("Failed to get default LLM: %s", e)
        return None


def _get_embedding():
    try:
        return Embeddings().get_embedding()
    except Exception as e:
        log.warning("Failed to get default embedding: %s", e)
        return None


# ── 1. Property Graph Extraction ──────────────────────────────


def property_graph_extract(text: str, schema_text: str = "") -> str:
    """Extract property graph (vertices + edges) from text."""
    if not text or not text.strip():
        return _serialize({"error": "Empty input text"})
    try:
        llm = _get_llm()
        if not llm:
            return _serialize({"error": "No LLM configured"})

        schema = None
        if schema_text and schema_text.strip():
            try:
                schema = json.loads(schema_text)
            except json.JSONDecodeError as e:
                return _serialize({"error": f"Invalid schema JSON: {e}"})
        else:
            # Auto-fetch schema
            try:
                manager = SchemaManager(huge_settings.graph_name)
                ctx = manager.run({})
                schema = ctx.get("schema", ctx.get("simple_schema", {}))
            except Exception as e:
                log.warning("Could not auto-fetch schema: %s", e)
                return _serialize({"error": f"No schema provided and auto-fetch failed: {e}"})

        if not schema or "vertexlabels" not in schema or "edgelabels" not in schema:
            return _serialize({"error": "Schema must contain vertexlabels and edgelabels"})

        chunks = [text]
        context = {"schema": schema, "chunks": chunks, "vertices": [], "edges": []}
        extractor = PropertyGraphExtract(llm=llm)
        result = extractor.run(context)

        return _serialize({
            "vertices": result.get("vertices", []),
            "edges": result.get("edges", []),
            "vertex_count": len(result.get("vertices", [])),
            "edge_count": len(result.get("edges", [])),
            "llm_calls": result.get("call_count", 0),
        })
    except Exception as e:
        log.error("Property graph extraction failed: %s", e)
        return _serialize({"error": str(e)})


# ── 3. Incremental Index Flow ─────────────────────────────────


def incremental_index_flow(texts_text: str, graph_name: str = "") -> str:
    """Run a simplified incremental index flow on new document texts."""
    if not texts_text or not texts_text.strip():
        return _serialize({"error": "Empty input texts"})

    client = _get_graph_client()
    if not client:
        return _serialize({"error": "Cannot connect to HugeGraph"})

    llm = _get_llm()
    if not llm:
        return _serialize({"error": "No LLM configured"})

    try:
        texts = [t.strip() for t in texts_text.split("\n---\n") if t.strip()]
        if not texts:
            texts = [texts_text.strip()]

        # Step 1: Chunk split
        splitter = ChunkSplitter(split_type="paragraph", language="zh")
        all_chunks = []
        for text in texts:
            chunks = splitter.split(text)
            all_chunks.extend(chunks)

        # Step 2: Info extract
        extractor = InfoExtract(llm=llm)
        schema_manager = SchemaManager(graph_name or huge_settings.graph_name)
        schema_ctx = schema_manager.run({})
        schema = schema_ctx.get("schema", schema_ctx.get("simple_schema", {}))

        context = {
            "schema": schema,
            "chunks": all_chunks,
            "vertices": [],
            "edges": [],
        }
        result = extractor.run(context)
        vertices = result.get("vertices", [])
        edges = result.get("edges", [])

        # Step 3: Commit to graph
        committer = Commit2Graph()
        commit_data = {"schema": schema, "vertices": vertices, "edges": edges}
        committer.run(commit_data)

        # Step 4: Find affected communities
        new_vertex_ids = [v.get("id") for v in vertices if v.get("id")]
        affected = find_affected_communities(client, new_vertex_ids, hop=1)

        return _serialize({
            "texts_processed": len(texts),
            "chunks": len(all_chunks),
            "vertices_added": len(vertices),
            "edges_added": len(edges),
            "new_vertex_ids": new_vertex_ids[:20],
            "affected_communities": sorted(affected),
            "affected_community_count": len(affected),
        })
    except Exception as e:
        log.error("Incremental index flow failed: %s", e)
        return _serialize({"error": str(e)})


# ── 4. Gremlin Self-Correction ────────────────────────────────


def gremlin_self_correct(query: str, max_retries: int = 3, language: str = "cn") -> str:
    """Generate, validate, and execute a Gremlin query with self-correction."""
    if not query or not query.strip():
        return _serialize({"error": "Empty query"})

    client = _get_graph_client()
    if not client:
        return _serialize({"error": "Cannot connect to HugeGraph"})

    try:
        # Fetch schema
        schema_manager = SchemaManager(huge_settings.graph_name)
        schema_ctx = schema_manager.run({})
        schema = schema_ctx.get("simple_schema", schema_ctx.get("schema", {}))
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

        llm = _get_llm()
        if not llm:
            return _serialize({"error": "No LLM configured"})

        retry_loop = GremlinRetryLoop(
            llm=llm,
            graph_client=client,
            schema=schema_text,
            max_retries=max_retries,
            language=language,
        )
        result = retry_loop.generate_and_execute(query)
        return _serialize(result)
    except Exception as e:
        log.error("Gremlin self-correction failed: %s", e)
        return _serialize({"error": str(e)})


# ── 5. Query Classifier ───────────────────────────────────────


def query_classifier_demo(query: str, use_llm: bool = False) -> str:
    """Classify a query as simple or complex."""
    if not query or not query.strip():
        return _serialize({"error": "Empty query"})
    try:
        llm = _get_llm() if use_llm else None
        is_complex = QueryClassifier.classify(query, llm)
        return _serialize({
            "query": query,
            "is_complex": is_complex,
            "route_target": "agent" if is_complex else "fast_graph_only",
            "reason": "regex_matched_complex" if is_complex and not use_llm else ("llm_classified" if use_llm else "regex_simple"),
        })
    except Exception as e:
        log.error("Query classification failed: %s", e)
        return _serialize({"error": str(e)})


# ── 6. Synonym Manager ────────────────────────────────────────


def _load_synonym_manager() -> SynonymManager:
    return SynonymManager.from_saved()


def synonym_add(canonical: str, aliases_text: str, category: str = "general") -> str:
    """Add a synonym group."""
    if not canonical or not canonical.strip():
        return _serialize({"error": "Empty canonical term"})
    aliases = [a.strip() for a in aliases_text.replace("，", ",").split(",") if a.strip()]
    try:
        manager = _load_synonym_manager()
        group = manager.add_synonym(canonical.strip(), aliases, category=category)
        manager.save()
        return _serialize({
            "group_id": group.group_id,
            "canonical": group.canonical,
            "aliases": group.aliases,
            "category": group.category,
            "total_groups": manager.group_count,
        })
    except Exception as e:
        log.error("Add synonym failed: %s", e)
        return _serialize({"error": str(e)})


def synonym_expand(query: str) -> str:
    """Expand a query with synonyms."""
    if not query or not query.strip():
        return _serialize({"error": "Empty query"})
    try:
        manager = _load_synonym_manager()
        expanded = manager.expand_query(query)
        return _serialize({
            "original": query,
            "expanded": expanded,
            "total_groups": manager.group_count,
        })
    except Exception as e:
        log.error("Expand synonym failed: %s", e)
        return _serialize({"error": str(e)})


def synonym_list() -> str:
    """List all synonym groups."""
    try:
        manager = _load_synonym_manager()
        groups = [g.to_dict() for g in manager._groups.values()]
        return _serialize({"total_groups": manager.group_count, "groups": groups})
    except Exception as e:
        log.error("List synonyms failed: %s", e)
        return _serialize({"error": str(e)})


# ── 7. Chunk Similarity Edges ─────────────────────────────────


def chunk_sim_edges_build(chunk_label: str = "Chunk", top_k: int = 3, min_score: float = 0.5) -> str:
    """Build SIMILAR edges between Chunk vertices."""
    client = _get_graph_client()
    if not client:
        return _serialize({"error": "Cannot connect to HugeGraph"})

    embedding = _get_embedding()
    if not embedding:
        return _serialize({"error": "No embedding model configured"})

    try:
        # We don't have direct access to the vector index, so use embedding-only mode.
        # The builder will compute embeddings and search via graph fallback.
        builder = ChunkSimEdgeBuilder(
            embedding=embedding,
            graph_client=client,
            top_k=top_k,
            min_score=min_score,
        )
        count = builder.build_all(chunk_label=chunk_label)
        return _serialize({
            "edges_added": count,
            "chunk_label": chunk_label,
            "top_k": top_k,
            "min_score": min_score,
        })
    except Exception as e:
        log.error("Chunk sim edges build failed: %s", e)
        return _serialize({"error": str(e)})
