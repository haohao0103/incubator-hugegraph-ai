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
Handler functions for the GraphRAG Enhancement demo tab.

Wires PPR Retriever, Cascade Propagation, Identity Edge Builder,
Dual Keyword Extract, Community Summary Generator, HyDE, Gleaning,
Claim Extract, Provenance Answer, and BM25 Search into UI-callable
functions that return dicts suitable for Gradio components.
"""

import json
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


# ── PPR Retriever ──────────────────────────────────────────────

def ppr_retriever_demo(
    query: str,
    alpha: float = 0.15,
    max_depth: int = 2,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Demonstrate PPR (Personalized PageRank) retrieval.

    Args:
        query: Search query.
        alpha: PPR teleport probability (0.0-1.0).
        max_depth: Maximum traversal depth.
        top_k: Number of top results to return.

    Returns:
        Dict with seed entities, PPR scores, and retrieved context.
    """
    if not query or not query.strip():
        return {"seed_entities": [], "ppr_scores": {}, "context": "",
                "total_entities_reached": 0, "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.graph_op.ppr_retriever import PPRRetriever
        from hugegraph_llm.utils.hugegraph_utils import get_hg_client

        client = get_hg_client()
        retriever = PPRRetriever(client=client, alpha=alpha)

        result = retriever.retrieve(query=query, max_depth=max_depth, top_k=top_k)

        seed_entities = result.get("seed_entities", [])
        ppr_scores = result.get("ppr_scores", {})
        context = result.get("context", "")

        # Format scores for display
        score_display = {}
        for k, v in ppr_scores.items():
            score_display[str(k)] = round(float(v), 6)

        return {
            "seed_entities": seed_entities[:20],
            "ppr_scores": score_display,
            "context": context[:2000] if context else "",
            "total_entities_reached": len(ppr_scores),
            "alpha": alpha,
            "max_depth": max_depth,
            "error": None,
        }
    except Exception as e:
        log.error("PPR retriever demo error: %s", e)
        return {"seed_entities": [], "ppr_scores": {}, "context": "",
                "total_entities_reached": 0, "error": f"PPR retrieval failed: {str(e)}"}


# ── Cascade Propagation ───────────────────────────────────────

def cascade_propagation_demo(
    query: str,
    ppr_alpha: float = 0.15,
    entity_threshold: float = 0.01,
    chunk_top_k: int = 10,
) -> Dict[str, Any]:
    """Demonstrate three-layer cascade propagation (Entity→Relation→Chunk).

    Args:
        query: Search query to seed the propagation.
        ppr_alpha: PPR teleport probability.
        entity_threshold: Minimum entity score threshold.
        chunk_top_k: Number of top chunks to return.

    Returns:
        Dict with per-layer scores and propagation trace.
    """
    if not query or not query.strip():
        return {"entity_scores": {}, "relation_scores": {}, "chunk_scores": {},
                "propagation_trace": [], "context": "", "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.graph_op.cascade_propagation import (
            CascadeMatrixBuilder, CascadePropagator, CascadeConfig,
        )
        from hugegraph_llm.utils.hugegraph_utils import get_hg_client

        client = get_hg_client()
        builder = CascadeMatrixBuilder(graph_client=client)
        propagator = CascadePropagator()

        # Build matrices from local data
        e2r, r2c, e_map, r_map, c_map = builder.build(
            entities={}, relations=[], chunks={}
        )

        config = CascadeConfig(ppr_alpha=ppr_alpha)
        result = propagator.propagate(
            seed_scores={"query": 1.0},
            e2r=e2r, r2c=r2c,
            e_map=e_map, r_map=r_map, c_map=c_map,
            config=config,
        )

        # Format scores
        entity_scores = {str(k): round(float(v), 6) for k, v in result.entity_scores.items()}
        relation_scores = {str(k): round(float(v), 6) for k, v in result.relation_scores.items()}
        chunk_scores = {str(k): round(float(v), 6) for k, v in result.chunk_scores.items()}

        trace = [
            {"step": 1, "name": "Seed Entity Identification", "entities": len(result.seed_entities)},
            {"step": 2, "name": "PPR on Entity Layer", "scores": len(entity_scores)},
            {"step": 3, "name": "Entity→Relation Propagation", "relations": len(relation_scores)},
            {"step": 4, "name": "Relation→Chunk Propagation", "chunks": len(chunk_scores)},
        ]

        return {
            "entity_scores": entity_scores,
            "relation_scores": relation_scores,
            "chunk_scores": chunk_scores,
            "propagation_trace": trace,
            "context": "",
            "total_entities": len(e_map),
            "total_relations": len(r_map),
            "total_chunks": len(c_map),
            "error": None,
        }
    except Exception as e:
        log.error("Cascade propagation demo error: %s", e)
        return {"entity_scores": {}, "relation_scores": {}, "chunk_scores": {},
                "propagation_trace": [], "context": "",
                "error": f"Cascade propagation failed: {str(e)}"}


# ── Identity Edge Builder ─────────────────────────────────────

def identity_edge_demo(
    entity_names: str,
    similarity_threshold: float = 0.9,
    top_k_neighbors: int = 5,
) -> Dict[str, Any]:
    """Demonstrate identity edge building (same_as edges for similar entities).

    Args:
        entity_names: Comma-separated entity names.
        similarity_threshold: Minimum embedding similarity to create same_as edge.
        top_k_neighbors: Max same_as edges per entity.

    Returns:
        Dict with entity pairs, similarity scores, and merge suggestions.
    """
    if not entity_names or not entity_names.strip():
        return {"entity_pairs": [], "merge_suggestions": [], "total_entities": 0,
                "edges_created": 0, "error": "Please enter entity names."}

    try:
        from hugegraph_llm.operators.graph_op.identity_edge_builder import (
            IdentityEdgeBuilder, IdentityEdgeConfig,
        )

        names = [n.strip() for n in entity_names.split(",") if n.strip()]
        if not names:
            return {"entity_pairs": [], "merge_suggestions": [], "total_entities": 0,
                    "edges_created": 0, "error": "No valid entity names."}

        # Use default embedding (will attempt to load from settings)
        config = IdentityEdgeConfig(
            similarity_threshold=similarity_threshold,
            top_k_neighbors=top_k_neighbors,
        )
        builder = IdentityEdgeBuilder(embedding_fn=None, config=config)
        entity_ids = names
        entity_texts = {n: n for n in names}

        result = builder.build(entity_ids, entity_texts)

        # Format pairs for display
        pairs = []
        for src, tgt, score in result.entity_pairs:
            pairs.append({
                "source": src, "target": tgt,
                "similarity": round(float(score), 4),
                "action": "same_as" if float(score) >= similarity_threshold else "skip",
            })

        # Merge suggestions (groups of similar entities)
        merge_groups = []
        seen = set()
        for src, tgt, score in result.entity_pairs:
            if float(score) >= similarity_threshold and src not in seen and tgt not in seen:
                merge_groups.append({"canonical": src, "aliases": [tgt], "similarity": round(float(score), 4)})
                seen.add(src)
                seen.add(tgt)

        return {
            "entity_pairs": pairs[:30],
            "merge_suggestions": merge_groups[:10],
            "total_entities": len(names),
            "edges_created": result.total_edges,
            "threshold": similarity_threshold,
            "error": None,
        }
    except Exception as e:
        log.error("Identity edge demo error: %s", e)
        return {"entity_pairs": [], "merge_suggestions": [], "total_entities": 0,
                "edges_created": 0, "error": f"Identity edge builder failed: {str(e)}"}


# ── Dual Keyword Extract ──────────────────────────────────────

def dual_keyword_demo(
    query: str,
    extraction_method: str = "heuristic",
    language: str = "en",
) -> Dict[str, Any]:
    """Demonstrate dual-level keyword extraction (hl_keywords + ll_keywords).

    Args:
        query: Search query to extract keywords from.
        extraction_method: "heuristic" or "llm".
        language: "en" or "zh".

    Returns:
        Dict with hl_keywords, ll_keywords, and extraction metadata.
    """
    if not query or not query.strip():
        return {"hl_keywords": [], "ll_keywords": [], "all_keywords": [],
                "extraction_method": extraction_method, "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.llm_op.dual_keyword_extract import (
            DualKeywordExtract, DualKeywordConfig, DualKeywordResult,
        )

        config = DualKeywordConfig(language=language)
        # Only use heuristic for demo (LLM requires live API)
        use_llm = extraction_method == "llm"
        llm = None
        if use_llm:
            try:
                from hugegraph_llm.models.init_llm import LLMFactory
                llm = LLMFactory.get_llm()
            except Exception:
                log.warning("LLM not available, falling back to heuristic")
                use_llm = False

        extractor = DualKeywordExtract(llm=llm, config=config)
        result = extractor.extract(query)

        return {
            "hl_keywords": list(result.hl_keywords),
            "ll_keywords": list(result.ll_keywords),
            "all_keywords": list(result.hl_keywords) + list(result.ll_keywords),
            "hl_count": len(result.hl_keywords),
            "ll_count": len(result.ll_keywords),
            "extraction_method": result.extraction_method,
            "has_keywords": result.has_keywords,
            "error": None,
        }
    except Exception as e:
        log.error("Dual keyword demo error: %s", e)
        return {"hl_keywords": [], "ll_keywords": [], "all_keywords": [],
                "extraction_method": extraction_method,
                "error": f"Dual keyword extract failed: {str(e)}"}


# ── Community Summary Generator ────────────────────────────────

def community_summary_demo(
    algorithm: str = "louvain",
    max_levels: int = 2,
    summary_method: str = "heuristic",
) -> Dict[str, Any]:
    """Demonstrate community detection + summary generation.

    Args:
        algorithm: Community detection algorithm (louvain/wcc).
        max_levels: Number of hierarchical levels.
        summary_method: "heuristic" or "llm" for summary generation.

    Returns:
        Dict with community summaries and statistics.
    """
    try:
        from hugegraph_llm.operators.graph_rag_enhancements.community_summary_generator import (
            CommunitySummaryGenerator, CommunitySummaryConfig,
        )
        from hugegraph_llm.operators.graph_op.community_detect import CommunityDetect
        from hugegraph_llm.utils.hugegraph_utils import get_hg_client

        client = get_hg_client()

        # Step 1: Run community detection
        detector = CommunityDetect(client=client)
        communities = detector.run(algorithm=algorithm, max_levels=max_levels)

        # Step 2: Generate summaries
        use_llm = summary_method == "llm"
        llm = None
        if use_llm:
            try:
                from hugegraph_llm.models.init_llm import LLMFactory
                llm = LLMFactory.get_llm()
            except Exception:
                log.warning("LLM not available, falling back to heuristic")

        config = CommunitySummaryConfig(use_llm_summary=use_llm)
        generator = CommunitySummaryGenerator(llm=llm, config=config)
        reports = generator.generate(communities)

        # Format for display
        display_reports = []
        for report in reports[:20]:
            display_reports.append({
                "community_id": report.community_id if hasattr(report, 'community_id') else "?",
                "title": report.title if hasattr(report, 'title') else "Untitled",
                "summary": report.summary[:200] if hasattr(report, 'summary') else "",
                "rank": round(float(report.rank), 4) if hasattr(report, 'rank') else 0,
                "entity_count": report.entity_count if hasattr(report, 'entity_count') else 0,
            })

        return {
            "community_reports": display_reports,
            "total_communities": len(communities) if isinstance(communities, list) else 0,
            "total_reports": len(reports),
            "algorithm": algorithm,
            "summary_method": summary_method if use_llm else "heuristic",
            "error": None,
        }
    except Exception as e:
        log.error("Community summary demo error: %s", e)
        return {"community_reports": [], "total_communities": 0,
                "total_reports": 0, "error": f"Community summary failed: {str(e)}"}


# ── HyDE Generate ──────────────────────────────────────────────

def hyde_demo(query: str) -> Dict[str, Any]:
    """Demonstrate HyDE (Hypothetical Document Embedding) query enhancement.

    Args:
        query: Original user query.

    Returns:
        Dict with hypothetical answer and embedding comparison.
    """
    if not query or not query.strip():
        return {"original_query": "", "hypothetical_answer": "", "enhanced_query": "",
                "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.llm_op.hyde_generate import HyDEGenerate
        from hugegraph_llm.models.init_llm import LLMFactory

        llm = LLMFactory.get_llm()
        hyde = HyDEGenerate(llm=llm)

        result = hyde.run({"query": query})
        hypo_answer = result.get("hyde_answer", "")
        enhanced = result.get("enhanced_query", query)

        return {
            "original_query": query,
            "hypothetical_answer": hypo_answer,
            "enhanced_query": enhanced,
            "error": None,
        }
    except Exception as e:
        log.error("HyDE demo error: %s", e)
        return {"original_query": query, "hypothetical_answer": "",
                "enhanced_query": query, "error": f"HyDE generation failed: {str(e)}"}


# ── Gleaning Extractor ─────────────────────────────────────────

def gleaning_demo(
    query: str,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Demonstrate gleaning (follow-up question extraction).

    Args:
        query: Initial query.
        max_rounds: Maximum gleaning rounds.

    Returns:
        Dict with extracted follow-up questions and progressive answers.
    """
    if not query or not query.strip():
        return {"follow_up_questions": [], "progressive_answers": [],
                "total_rounds": 0, "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.graph_rag_enhancements.gleaning_extractor import GleaningExtractor
        from hugegraph_llm.models.init_llm import LLMFactory

        llm = LLMFactory.get_llm()
        gleaning = GleaningExtractor(llm=llm, max_rounds=max_rounds)

        result = gleaning.run({"query": query})

        follow_ups = result.get("follow_up_questions", [])
        answers = result.get("progressive_answers", [])

        return {
            "follow_up_questions": follow_ups,
            "progressive_answers": answers,
            "total_rounds": len(follow_ups),
            "max_rounds": max_rounds,
            "error": None,
        }
    except Exception as e:
        log.error("Gleaning demo error: %s", e)
        return {"follow_up_questions": [], "progressive_answers": [],
                "total_rounds": 0, "error": f"Gleaning extraction failed: {str(e)}"}


# ── Provenance Answer ──────────────────────────────────────────

def provenance_demo(query: str) -> Dict[str, Any]:
    """Demonstrate provenance-traced answer generation.

    Args:
        query: Search query.

    Returns:
        Dict with answer and source provenance trace.
    """
    if not query or not query.strip():
        return {"answer": "", "provenance": [], "source_count": 0,
                "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.llm_op.provenance_answer import ProvenanceAnswer
        from hugegraph_llm.models.init_llm import LLMFactory

        llm = LLMFactory.get_llm()
        provenance = ProvenanceAnswer(llm=llm)

        result = provenance.run({"query": query})

        answer = result.get("answer", "")
        sources = result.get("provenance", [])

        # Format provenance for display
        display_sources = []
        for src in sources[:20]:
            if isinstance(src, dict):
                display_sources.append(src)
            else:
                display_sources.append({"source": str(src)})

        return {
            "answer": answer,
            "provenance": display_sources,
            "source_count": len(sources),
            "error": None,
        }
    except Exception as e:
        log.error("Provenance demo error: %s", e)
        return {"answer": "", "provenance": [], "source_count": 0,
                "error": f"Provenance answer failed: {str(e)}"}


# ── BM25 Search ────────────────────────────────────────────────

def bm25_demo(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Demonstand BM25 full-text keyword search.

    Args:
        query: Search query.
        top_k: Number of results.

    Returns:
        Dict with BM25 search results and scores.
    """
    if not query or not query.strip():
        return {"results": [], "scores": {}, "total_matches": 0,
                "error": "Please enter a query."}

    try:
        from hugegraph_llm.operators.index_op.bm25_index_query import BM25IndexQuery
        from hugegraph_llm.utils.graph_index_utils import get_vector_index_class

        bm25 = BM25IndexQuery()
        result = bm25.run({"query": query, "top_k": top_k})

        results = result.get("bm25_results", [])
        scores = {str(r): round(float(s), 4) for r, s in result.get("bm25_scores", {}).items()}

        return {
            "results": results[:top_k],
            "scores": scores,
            "total_matches": len(results),
            "error": None,
        }
    except Exception as e:
        log.error("BM25 demo error: %s", e)
        return {"results": [], "scores": {}, "total_matches": 0,
                "error": f"BM25 search failed: {str(e)}"}
