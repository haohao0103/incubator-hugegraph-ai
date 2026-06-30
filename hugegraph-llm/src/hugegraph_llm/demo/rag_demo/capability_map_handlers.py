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

"""Handlers for the Capability Map Gradio tab.

Exposes missing utilities from the current branch that are not yet
available in other Gradio tabs:
- Fetch graph data summary
- Get graph schema
- Validate Gremlin query
- Incremental index community utilities
"""

import json
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd
from pyhugegraph.client import PyHugeClient

from hugegraph_llm.config import huge_settings, prompt
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.graph_op.incremental_utils import (
    find_affected_communities,
    persist_community_assignments,
)
from hugegraph_llm.operators.hugegraph_op.fetch_graph_data import FetchGraphData
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.operators.llm_op.gremlin_validator import GremlinValidator
from hugegraph_llm.utils.log import log


CAPABILITY_MATRIX = [
    # Category, Capability, Exposed Tab, Status, Priority, Notes
    ("Index", "Vector Index Build", "Tab 1", "Exposed", "-", "Core indexing"),
    ("Index", "Semantic Vertex-ID Index", "Tab 1", "Exposed", "-", "Update Vid Embedding"),
    ("Index", "Gremlin Example Index", "Tab 3", "Exposed", "-", "Text2Gremlin templates"),
    ("Index", "Community Index", "Tab 4", "Partial", "P1", "Build via community detection"),
    ("Index", "BM25 Keyword Index", "Tab 8", "Exposed", "-", "Optional plugin"),
    ("Index", "Incremental Index Update", "Tab 10", "Exposed", "-", "Simplified incremental flow"),
    ("Graph", "Community Detection", "Tab 4/7/8", "Exposed", "-", "Louvain/WCC"),
    ("Graph", "Entity Resolution", "Tab 7", "Exposed", "-", "4 strategies"),
    ("Graph", "PPR Retriever", "Tab 8", "Exposed", "-", "Personalized PageRank"),
    ("Graph", "Cascade Propagation", "Tab 8", "Exposed", "-", "Entity→Relation→Chunk"),
    ("Graph", "Identity Edge Builder", "Tab 8", "Exposed", "-", "same_as edges"),
    ("Graph", "RRF Fusion", "Tab 7", "Exposed", "-", "3-channel fusion"),
    ("Graph", "Token Budget", "Tab 7", "Exposed", "-", "Context budget control"),
    ("Graph", "Schema Validation", "Tab 7", "Exposed", "-", "Schema constraint check"),
    ("Graph", "Synonym Manager", "Tab 10", "Exposed", "-", "Graph quality enhancement"),
    ("Graph", "Chunk Similarity Edges", "Tab 10", "Exposed", "-", "Chunk-level graph edges"),
    ("LLM", "Info Extract (Triples)", "Tab 1", "Exposed", "-", "Standard KG extraction"),
    ("LLM", "Property Graph Extract", "Tab 10", "Exposed", "-", "Rich property extraction"),
    ("LLM", "Keyword Extract", "Tab 2", "Exposed", "-", "Query keywords"),
    ("LLM", "Dual Keyword Extract", "Tab 8", "Exposed", "-", "hl/ll keywords"),
    ("LLM", "Answer Synthesize", "Tab 2", "Exposed", "-", "Final answer generation"),
    ("LLM", "Gremlin Generate", "Tab 3", "Exposed", "-", "Text2Gremlin"),
    ("LLM", "Gremlin Validator", "Tab 10", "Exposed", "-", "Self-correcting Text2Gremlin"),
    ("LLM", "HyDE Generate", "Tab 8", "Exposed", "-", "Hypothetical answer embedding"),
    ("LLM", "DRIFT Search", "Tab 7", "Exposed", "-", "Multi-hop reasoning"),
    ("LLM", "Gleaning Extract", "Tab 8", "Exposed", "-", "Follow-up extraction"),
    ("LLM", "Provenance Answer", "Tab 8", "Exposed", "-", "Source tracing"),
    ("LLM", "Community Report", "Tab 4/7/8", "Exposed", "-", "Community summary"),
    ("LLM", "Global Search", "Tab 4", "Exposed", "-", "Cross-community QA"),
    ("LLM", "Schema Build", "Tab 1", "Exposed", "-", "Auto schema generation"),
    ("LLM", "Prompt Generate", "Tab 1", "Exposed", "-", "Few-shot prompt generation"),
    ("LLM", "Coref Resolution", "None", "Missing", "P1", "Coreference resolution"),
    ("LLM", "Claim Extract", "None", "Missing", "P1", "Claim-based extraction"),
    ("LLM", "Disambiguate Data", "None", "Missing", "P1", "Word sense disambiguation"),
    ("HugeGraph", "Commit to Graph", "Tab 1", "Exposed", "-", "Import graph data"),
    ("HugeGraph", "Fetch Graph Data", "None", "Missing", "P1", "Graph summary/vertices/edges"),
    ("HugeGraph", "Schema Manager", "None", "Missing", "P1", "Get schema from HugeGraph"),
    ("HugeGraph", "Provenance Manager", "Tab 8", "Exposed", "-", "Source tracing"),
    ("Document", "Chunk Split", "Tab 1", "Partial", "P2", "Implicit in indexing"),
    ("Document", "Word Extract", "None", "Missing", "P2", "Keyword extraction"),
    ("Document", "TextRank Word Extract", "None", "Missing", "P2", "TextRank keywords"),
    ("Multimodal", "Multimodal KG Builder", "Tab 10", "Exposed", "-", "Image/PDF → KG"),
    ("Multimodal", "Multimodal Retriever", "Tab 10", "Exposed", "-", "Image-text retrieval"),
    ("Multimodal", "PDF/Image Extractor", "Tab 10", "Exposed", "-", "VLM-based extraction"),
    ("Multimodal", "VLM Descriptor", "Tab 10", "Exposed", "-", "Visual description"),
    ("Agent", "ReAct Agent Loop", "Tab 4", "Exposed", "-", "Multi-step reasoning"),
    ("Agent", "Tool Registry", "Tab 4", "Exposed", "-", "Agent tools"),
    ("Agent", "MCP Adapter", "None", "Missing", "P1", "MCP server integration"),
    ("Agent", "Query Classifier", "Tab 10", "Exposed", "-", "Intent routing"),
    ("Agent", "Agent Memory", "None", "Missing", "P0", "Branch: feature/agent-memory-collection"),
    ("Flow", "E2E RAG Pipeline", "None", "Missing", "P1", "End-to-end RAG"),
    ("Flow", "Incremental Index Flow", "None", "Missing", "P0", "Partially via utils"),
    ("Branch", "Code Graph + MCP", "None", "Missing", "P0", "Branch: poc/0614-codegraph-hugegraph-mcp"),
    ("Branch", "Skills Graph / Code Review", "None", "Missing", "P2", "Branch: poc/0618-skills-graph-code-review-wiki"),
    ("Branch", "Supply Chain Agent Router", "None", "Missing", "P1", "Branch: poc/0615-supply-chain-agent-router"),
]


def get_capability_matrix() -> pd.DataFrame:
    """Return the capability gap matrix as a DataFrame."""
    df = pd.DataFrame(
        CAPABILITY_MATRIX,
        columns=["Category", "Capability", "UI Tab", "Status", "Priority", "Notes"],
    )
    # Add a status sort key
    status_order = {"Missing": 0, "Partial": 1, "Exposed": 2}
    df["_sort"] = df["Status"].map(status_order)
    df = df.sort_values(["_sort", "Category", "Priority"]).drop("_sort", axis=1)
    return df.reset_index(drop=True)


def _get_graph_client() -> Optional[PyHugeClient]:
    """Create a PyHugeClient from current settings."""
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


def fetch_graph_summary(v_limit: int = 100, e_limit: int = 50) -> Dict[str, Any]:
    """Fetch a brief summary of the current graph."""
    client = _get_graph_client()
    if client is None:
        return {"error": "Cannot connect to HugeGraph. Check graph config."}
    try:
        fetcher = FetchGraphData(client, v_limit=v_limit, e_limit=e_limit)
        summary = fetcher.run({})
        return summary
    except Exception as e:
        log.error("Fetch graph summary failed: %s", e)
        return {"error": str(e)}
    finally:
        try:
            client.close()
        except Exception:
            pass


def get_graph_schema() -> Dict[str, Any]:
    """Get the schema of the current graph."""
    try:
        manager = SchemaManager(huge_settings.graph_name)
        context = manager.run({})
        schema = context.get("simple_schema", context.get("schema", {}))
        return {"schema": schema, "graph_name": huge_settings.graph_name}
    except Exception as e:
        log.error("Get graph schema failed: %s", e)
        return {"error": str(e)}


def validate_gremlin(gremlin: str, schema_text: str = "", language: str = "en") -> Dict[str, Any]:
    """Validate a Gremlin query using LLM-driven validation."""
    if not gremlin or not gremlin.strip():
        return {"valid": False, "issues": ["Empty Gremlin query"], "fixed_query": ""}
    try:
        llm = LLMs().get_text2gql_llm()
        validator = GremlinValidator(llm=llm, language=language)
        if not schema_text.strip():
            # Try to get schema automatically
            try:
                manager = SchemaManager(huge_settings.graph_name)
                context = manager.run({})
                schema = context.get("simple_schema", context.get("schema", {}))
                schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
            except Exception as e:
                log.warning("Could not auto-fetch schema for validation: %s", e)
                schema_text = "No schema provided."
        return validator.validate(gremlin, schema_text)
    except Exception as e:
        log.error("Gremlin validation failed: %s", e)
        return {"valid": False, "issues": [str(e)], "fixed_query": ""}


def incremental_index_tool(
    action: str,
    vertex_ids_text: str = "",
    community_text: str = "",
    hop: int = 1,
) -> Dict[str, Any]:
    """Run incremental index utilities."""
    client = _get_graph_client()
    if client is None:
        return {"error": "Cannot connect to HugeGraph. Check graph config."}
    try:
        if action == "find_affected":
            vertex_ids = [v.strip() for v in vertex_ids_text.split(",") if v.strip()]
            if not vertex_ids:
                return {"error": "Please provide vertex IDs (comma-separated)."}
            affected = find_affected_communities(client, vertex_ids, hop=hop)
            return {
                "action": "find_affected_communities",
                "new_vertices": len(vertex_ids),
                "affected_communities": sorted(affected),
                "affected_count": len(affected),
            }
        if action == "persist_communities":
            try:
                communities = json.loads(community_text) if community_text.strip() else []
            except json.JSONDecodeError as e:
                return {"error": f"Invalid community JSON: {e}"}
            if not isinstance(communities, list):
                return {"error": "community_text must be a JSON list."}
            result = persist_community_assignments(client, communities)
            return {"action": "persist_community_assignments", **result}
        return {"error": f"Unknown action: {action}"}
    except Exception as e:
        log.error("Incremental index tool failed: %s", e)
        return {"error": str(e)}
    finally:
        try:
            client.close()
        except Exception:
            pass


# UI wrapper functions

def ui_fetch_graph_summary(v_limit: int, e_limit: int) -> str:
    result = fetch_graph_summary(int(v_limit), int(e_limit))
    return json.dumps(result, ensure_ascii=False, indent=2)


def ui_get_graph_schema() -> str:
    result = get_graph_schema()
    return json.dumps(result, ensure_ascii=False, indent=2)


def ui_validate_gremlin(gremlin: str, language: str) -> str:
    result = validate_gremlin(gremlin, language=language)
    return json.dumps(result, ensure_ascii=False, indent=2)


def ui_incremental_tool(action: str, vertex_ids: str, community_text: str, hop: int) -> str:
    result = incremental_index_tool(action, vertex_ids, community_text, hop)
    return json.dumps(result, ensure_ascii=False, indent=2)
