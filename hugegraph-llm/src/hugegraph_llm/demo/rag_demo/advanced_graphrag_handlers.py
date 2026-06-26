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
Handler functions for the Advanced GraphRAG demo tab.

Wires the DRIFT search, Schema validation, Community reports,
Entity Resolution, and Incremental Index flows into UI-callable
functions that return dicts suitable for Gradio components.
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.utils.log import log


# ── DRIFT Search ──────────────────────────────────────────────

def drift_search_answer(query: str, communities_top_k: int = 5,
                         language: str = "cn") -> Dict[str, Any]:
    """Execute DRIFT multi-hop search and return detailed pipeline trace.

    Returns a dict with:
      - answer: Final synthesized answer
      - pipeline: List of 5 step results (hyde, community_match, primer, local, reduce)
      - metadata: call_count, communities_used, depth_reached
    """
    if not query or not query.strip():
        return {
            "answer": "",
            "pipeline": [],
            "metadata": {"call_count": 0, "communities_used": 0, "depth_reached": 0},
            "error": "Please enter a query.",
        }

    scheduler = SchedulerSingleton.get_instance()
    try:
        result = scheduler.schedule_flow(
            FlowName.DRIFT_SEARCH,
            query=query,
            communities_top_k=communities_top_k,
            language=language,
        )
        answer = result.get("drift_answer", "")
        findings = result.get("drift_findings", [])
        primer = result.get("drift_primer", {})

        # Build pipeline trace for visualization
        pipeline = [
            {
                "step": 1,
                "name": "HyDE (Hypothetical Document Embedding)",
                "status": "completed",
                "detail": "Generated hypothetical answer to enrich query embedding.",
            },
            {
                "step": 2,
                "name": "Community Matching",
                "status": "completed",
                "detail": f"Matched {result.get('drift_communities_used', 0)} relevant communities via vector similarity.",
            },
            {
                "step": 3,
                "name": "Primer Analysis",
                "status": "completed",
                "detail": f"Initial answer generated. Follow-up queries: {len(primer.get('follow_up_queries', []))}",
            },
            {
                "step": 4,
                "name": "Parallel Local Search",
                "status": "completed",
                "detail": f"Executed {len(findings)} local searches across follow-up queries.",
            },
            {
                "step": 5,
                "name": "Reduce (Synthesis)",
                "status": "completed",
                "detail": "Synthesized all findings into final answer.",
            },
        ]

        metadata = {
            "call_count": result.get("call_count", 0),
            "communities_used": result.get("drift_communities_used", 0),
            "depth_reached": result.get("drift_depth_reached", 0),
            "findings_count": len(findings),
        }

        return {
            "answer": answer,
            "pipeline": pipeline,
            "metadata": metadata,
            "findings": findings[:5],  # Top 5 for display
            "primer": primer,
            "error": None,
        }

    except Exception as e:
        log.error("DRIFT search handler error: %s", e)
        return {
            "answer": "",
            "pipeline": [],
            "metadata": {},
            "findings": [],
            "primer": {},
            "error": f"DRIFT search failed: {str(e)}",
        }


# ── Schema Validation ───────────────────────────────────────

def schema_validate(schema_json: str) -> Dict[str, Any]:
    """Validate a graph schema definition.

    Args:
        schema_json: JSON string defining the schema (entities, relations, etc.)

    Returns:
        Dict with valid, errors, warnings, suggestions.
    """
    import json

    if not schema_json or not schema_json.strip():
        return {
            "valid": False,
            "errors": ["Please enter schema JSON."],
            "warnings": [],
            "suggestions": [],
            "entity_count": 0,
            "relation_count": 0,
        }

    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Invalid JSON: {str(e)}"],
            "warnings": [],
            "suggestions": [],
            "entity_count": 0,
            "relation_count": 0,
        }

    scheduler = SchedulerSingleton.get_instance()
    try:
        result = scheduler.schedule_flow(
            FlowName.SCHEMA_VALIDATION,
            schema=schema,
        )
        return {
            "valid": result.get("valid", False),
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
            "suggestions": result.get("suggestions", []),
            "entity_count": result.get("entity_count", 0),
            "relation_count": result.get("relation_count", 0),
        }
    except Exception as e:
        log.error("Schema validation handler error: %s", e)
        return {
            "valid": False,
            "errors": [f"Schema validation failed: {str(e)}"],
            "warnings": [],
            "suggestions": [],
            "entity_count": 0,
            "relation_count": 0,
        }


# ── Entity Resolution ────────────────────────────────────────

def entity_resolve(entities_text: str, strategy: str = "hybrid") -> Dict[str, Any]:
    """Run entity resolution on a list of entity names.

    Args:
        entities_text: Newline-separated entity names.
        strategy: Resolution strategy (exact_match, embedding, llm_verify, hybrid).

    Returns:
        Dict with groups, total_entities, resolved_count, unresolved_count.
    """
    if not entities_text or not entities_text.strip():
        return {
            "groups": [],
            "total_entities": 0,
            "resolved_count": 0,
            "unresolved_count": 0,
            "error": "Please enter entity names (one per line).",
        }

    entity_names = [e.strip() for e in entities_text.strip().split("\n") if e.strip()]
    if not entity_names:
        return {
            "groups": [],
            "total_entities": 0,
            "resolved_count": 0,
            "unresolved_count": 0,
            "error": "No valid entity names found.",
        }

    # Use the entity resolution operator directly (no Flow needed for simple demo)
    try:
        from hugegraph_llm.operators.graph_op.entity_resolution import EntityResolution

        er = EntityResolution(client=None, strategy=strategy)
        candidates = []
        for name in entity_names:
            candidates.append({"name": name, "label": "Entity", "properties": {"name": name}})

        result = er.run({
            "vertices_info": candidates,
            "candidate": candidates,
        })

        groups = result.get("groups", [])
        resolved = result.get("resolved_count", len(entity_names))
        unresolved = result.get("unresolved_count", 0)

        # Format groups for display
        display_groups = []
        for i, group in enumerate(groups[:20]):
            names = [item.get("name", "?") for item in group]
            display_groups.append({
                "group_id": i + 1,
                "canonical": names[0] if names else "?",
                "members": names,
                "size": len(names),
            })

        return {
            "groups": display_groups,
            "total_entities": len(entity_names),
            "resolved_count": resolved,
            "unresolved_count": unresolved,
            "strategy": strategy,
            "error": None,
        }
    except Exception as e:
        log.error("Entity resolution handler error: %s", e)
        return {
            "groups": [],
            "total_entities": len(entity_names),
            "resolved_count": 0,
            "unresolved_count": 0,
            "strategy": strategy,
            "error": f"Entity resolution failed: {str(e)}",
        }


# ── Community Reports Viewer ────────────────────────────────

def get_community_reports(limit: int = 10) -> Dict[str, Any]:
    """Retrieve generated community reports for visualization.

    Returns:
        Dict with reports list and summary stats.
    """
    try:
        from hugegraph_llm.utils.vector_index_utils import get_vector_index_class
        from hugegraph_llm.config.index_config import IndexConfig
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings

        # Try to load community reports from the index
        vector_index_cls = get_vector_index_class(IndexConfig().cur_vector_index)
        embedding = Embeddings().get_embedding()

        reports = []
        # Attempt to search community index
        try:
            idx = vector_index_cls.from_name("community")
            if hasattr(idx, 'data') and idx.data:
                for item in idx.data[:limit]:
                    if isinstance(item, dict):
                        reports.append(item)
                    elif isinstance(item, str):
                        import json
                        try:
                            reports.append(json.loads(item))
                        except (json.JSONDecodeError, TypeError):
                            reports.append({"summary": item, "title": "Untitled"})
        except Exception:
            log.debug("No community index found, returning empty reports")

        return {
            "reports": reports,
            "total_reports": len(reports),
            "error": None if reports else "No community reports found. Run community detection first.",
        }
    except Exception as e:
        log.error("Community reports handler error: %s", e)
        return {
            "reports": [],
            "total_reports": 0,
            "error": f"Failed to load reports: {str(e)}",
        }


# ── Incremental Index Status ─────────────────────────────────

def incremental_index_status() -> Dict[str, Any]:
    """Check the current incremental index status.

    Returns:
        Dict with index info, document count, last_update.
    """
    try:
        from hugegraph_llm.utils.graph_index_utils import get_graph_index_info

        info = get_graph_index_info()
        return {
            "vertex_count": info.get("vertex_count", 0),
            "edge_count": info.get("edge_count", 0),
            "index_exists": info.get("index_exists", False),
            "index_type": info.get("index_type", "none"),
            "last_indexed": info.get("last_indexed", "never"),
            "error": None,
        }
    except Exception as e:
        log.error("Incremental index status handler error: %s", e)
        return {
            "vertex_count": 0,
            "edge_count": 0,
            "index_exists": False,
            "index_type": "none",
            "last_indexed": "never",
            "error": f"Failed to get index status: {str(e)}",
        }


# ── RRF Fusion Demo ─────────────────────────────────────────

def rrf_demo(query: str = "", top_k: int = 5) -> Dict[str, Any]:
    """Demonstrate RRF fusion on a query.

    Shows how multiple retrieval channels are merged using
    Reciprocal Rank Fusion.

    Returns:
        Dict with per-channel results and fused results.
    """
    if not query or not query.strip():
        return {
            "vector_results": [],
            "graph_results": [],
            "keyword_results": [],
            "fused_results": [],
            "fused_scores": {},
            "error": "Please enter a query.",
        }

    try:
        from hugegraph_llm.models.embeddings.init_embedding import Embeddings
        from hugegraph_llm.utils.vector_index_utils import get_vector_index_class
        from hugegraph_llm.config.index_config import IndexConfig
        from hugegraph_llm.operators.graph_op.rrf_fusion import ReciprocalRankFusion

        embedding = Embeddings().get_embedding()
        vector_index_cls = get_vector_index_class(IndexConfig().cur_vector_index)

        # Simulate multi-channel search
        vector_results = []
        graph_results = []
        keyword_results = []

        try:
            query_vec = embedding.get_texts_embeddings([query])[0]
            idx = vector_index_cls.from_name("")
            raw = idx.search(query_vec, top_k * 2)
            if isinstance(raw, list):
                vector_results = [str(r) for r in raw[:top_k * 2]]
        except Exception:
            vector_results = [f"(simulated vector result {i+1})" for i in range(3)]

        # Simulate graph results
        try:
            from hugegraph_llm.utils.hugegraph_utils import get_hg_client
            client = get_hg_client()
            resp = client.gremlin(
                f'g.V().hasLabel("Entity").limit({top_k}).valueMap("name")'
            ).exec()
            if isinstance(resp, dict):
                data = resp.get("data", [])
                graph_results = [str(d) for d in data[:top_k]]
        except Exception:
            graph_results = [f"(simulated graph result {i+1})" for i in range(3)]

        # Fuse with RRF
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse([
            ("vector", vector_results),
            ("graph", graph_results),
        ])

        return {
            "vector_results": vector_results[:top_k],
            "graph_results": graph_results[:top_k],
            "keyword_results": keyword_results[:top_k],
            "fused_results": fused.top_k(top_k),
            "fused_scores": {str(k): round(v, 4) for k, v in fused.scores.items()},
            "error": None,
        }
    except Exception as e:
        log.error("RRF demo handler error: %s", e)
        return {
            "vector_results": [],
            "graph_results": [],
            "keyword_results": [],
            "fused_results": [],
            "fused_scores": {},
            "error": f"RRF demo failed: {str(e)}",
        }


# ── Token Budget Demo ────────────────────────────────────────

def token_budget_demo(query: str = "", max_tokens: int = 2000) -> Dict[str, Any]:
    """Demonstrate Token Budget allocation for a query.

    Shows how context tokens are distributed across entity,
    relation, and community categories.

    Returns:
        Dict with budget summary and per-category breakdown.
    """
    if not query or not query.strip():
        return {
            "context": "",
            "summary": {},
            "error": "Please enter a query.",
        }

    try:
        from hugegraph_llm.operators.graph_op.token_budget import (
            TokenBudget, TokenBudgetConfig, _estimate_tokens,
        )

        config = TokenBudgetConfig(max_total_tokens=max_tokens)
        budget = TokenBudget(config)

        # Simulate filling the budget with search results
        # (In production, this would use actual retrieval results)
        sample_entities = [
            ("Entity: Apache HugeGraph", 8),
            ("Entity: Graph Database", 6),
            ("Entity: Gremlin Traversal Language", 10),
            ("Entity: TinkerPop Framework", 6),
            ("Entity: Knowledge Graph", 5),
        ]
        sample_relations = [
            ("(HugeGraph)-[implements]->(TinkerPop)", 7),
            ("(HugeGraph)-[supports]->(Gremlin)", 5),
            ("(Gremlin)-[is_a]->(Traversal Language)", 8),
        ]
        sample_communities = [
            ("Community: Graph Database Ecosystem", 15),
            ("Community: Apache Software Foundation Projects", 12),
        ]

        accepted, rejected = 0, 0
        for text, est in sample_entities * 5:  # Simulate more
            if budget.add("entity", text, est):
                accepted += 1
            else:
                rejected += 1
        for text, est in sample_relations * 3:
            if budget.add("relation", text, est):
                accepted += 1
            else:
                rejected += 1
        for text, est in sample_communities * 2:
            if budget.add("community", text, est):
                accepted += 1
            else:
                rejected += 1

        summary = budget.summary()
        context = budget.build_context()

        return {
            "context": context[:max_tokens],
            "summary": summary,
            "accepted_entries": accepted,
            "rejected_entries": rejected,
            "error": None,
        }
    except Exception as e:
        log.error("Token budget demo handler error: %s", e)
        return {
            "context": "",
            "summary": {},
            "accepted_entries": 0,
            "rejected_entries": 0,
            "error": f"Token budget demo failed: {str(e)}",
        }
