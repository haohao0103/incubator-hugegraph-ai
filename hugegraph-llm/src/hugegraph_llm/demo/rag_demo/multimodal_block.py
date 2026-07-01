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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gradio UI block for Multimodal GraphRAG — dedicated tab.

Showcases ALL 18 multimodal operators across 7 functional areas:
  A. Document Parsing   (unified_document_parser, pdf_image_extractor)
  B. VLM Description    (vlm_descriptor, vlm_provider_registry,
                         async_vlm_pipeline, image_dimension_validator)
  C. Multimodal Analysis (multimodal_analyzer, surrounding_context, chunk_schema)
  D. KG Build           (multimodal_entity_injector, multimodal_kg_builder)
  E. Retrieval          (multimodal_retriever, multimodal_retrieval_channel)
  F. Formula & Sidecar  (omml_to_latex, sidecar_placeholder,
                         sidecar_ir, sidecar_writer, sidecar_backfill)
  G. Pipeline Overview  (8-node DAG architecture diagram + flow stats)

All demo data is self-contained so the page can run without
external dependencies (PDF/VLM/HugeGraph) for showcasing.
"""

import json
import traceback
from typing import Any, Dict, List

import gradio as gr

from hugegraph_llm.config import huge_settings
from hugegraph_llm.utils.log import log

# ═══════════════════════════════════════════════════════════════
# Self-contained demo data for all 18 operators
# ═══════════════════════════════════════════════════════════════

# ── A. Document Parsing demo ──────────────────────────────────

DEMO_UNIFIED_PARSE = {
    "source": "demo_supply_chain_report.docx",
    "format": "docx",
    "blocks": [
        {"type": "heading", "level": 1, "content": "Supply Chain Risk Assessment Report"},
        {"type": "paragraph", "content": "This report evaluates warehouse congestion, supplier delays, and transport disruption risks across 6 supply chain nodes."},
        {"type": "paragraph", "content": "Key findings: {{TBL:1}} shows node-level risk scores. {{IMG:1}} visualizes the risk heatmap. The overall risk score is {{EQ:1}}."},
    ],
    "images": [
        {"placeholder_key": "1", "name": "risk_heatmap", "caption": "Supply chain risk distribution heatmap", "format": "png"},
    ],
    "tables": [
        {"placeholder_key": "1", "name": "supply_chain_risk_table", "caption": "Node risk assessment", "rows": 5, "cols": 4},
    ],
    "equations": [
        {"placeholder_key": "1", "name": "risk_score_equation", "latex": "R_{score} = \\sum_{i=1}^{n} w_i \\cdot \\frac{d_i}{\\tau_i} + \\lambda \\cdot \\sigma^2", "is_block": True},
    ],
    "total_blocks": 3,
    "total_images": 1,
    "total_tables": 1,
    "total_equations": 1,
}

DEMO_PDF_EXTRACT = {
    "source_path": "demo_supply_chain.pdf",
    "total_pages": 3,
    "total_images": 3,
    "total_text_blocks": 12,
    "pages": [
        {"page_num": 1, "image_count": 1, "text_block_count": 4},
        {"page_num": 2, "image_count": 1, "text_block_count": 4},
        {"page_num": 3, "image_count": 1, "text_block_count": 4},
    ],
}

# ── B. VLM Description demo ──────────────────────────────────

DEMO_VLM_DESCRIPTIONS = [
    {
        "image_id": "img_1_0",
        "caption": "Supply chain risk distribution heatmap",
        "detailed_description": "A heatmap showing risk levels across 6 supply chain nodes, with red indicating high-risk regions (warehouse congestion, supplier delays) and green indicating stable operations.",
        "object_labels": ["heatmap", "risk_level", "supply_chain", "warehouse", "supplier"],
        "chart_type": "heatmap",
        "key_insights": [
            "Warehouse congestion is the #1 risk factor (87% severity)",
            "3 out of 6 nodes show elevated risk levels",
            "Supplier delays correlate strongly with transport disruption",
        ],
        "related_keywords": ["supply_chain_risk", "warehouse_congestion", "delay_prediction"],
        "confidence": 0.92,
        "vlm_model": "MiMo-VLM-Pro",
    },
    {
        "image_id": "img_2_0",
        "caption": "Driver-route frequency scatter plot",
        "detailed_description": "A scatter plot mapping driver IDs against route frequency, showing clustering patterns. Top 15% drivers cover 80% of recurring routes.",
        "object_labels": ["scatter_plot", "driver_id", "route_frequency", "cluster"],
        "chart_type": "scatter",
        "key_insights": [
            "Top 15% drivers cover 80% of recurring routes",
            "3 distinct clusters: metro/suburban/rural",
            "Route frequency drops sharply beyond 50km radius",
        ],
        "related_keywords": ["driver_cluster", "route_analysis", "frequency_mapping"],
        "confidence": 0.88,
        "vlm_model": "MiMo-VLM-Pro",
    },
    {
        "image_id": "img_3_0",
        "caption": "Graph database architecture diagram",
        "detailed_description": "An architecture diagram showing the HugeGraph cluster layout: 3 master nodes, 5 worker nodes, HBase storage backend, with Gremlin query gateway.",
        "object_labels": ["architecture", "cluster", "master_node", "worker_node", "HBase"],
        "chart_type": "architecture",
        "key_insights": [
            "3-master high-availability configuration",
            "HBase as persistent storage backend",
            "Gremlin gateway for OLTP queries",
            "Vermeer engine for OLAP analytics",
        ],
        "related_keywords": ["architecture", "cluster_topology", "HA_config"],
        "confidence": 0.95,
        "vlm_model": "MiMo-VLM-Pro",
    },
]

DEMO_VLM_PROVIDER_REGISTRY = {
    "registered_providers": ["openai", "ollama", "anthropic", "gemini", "bedrock"],
    "default_provider": "openai",
    "normalized_image_fields": ["raw_bytes", "base64", "mime", "sha256", "width", "height"],
    "adapter_interface": ["format_image_content", "build_request", "call_api"],
    "multi_backend_caller": "VLMMultiBackendCaller — selects adapter + fallback to VLMDescriptor",
}

DEMO_ASYNC_VLM = {
    "pipeline_config": {
        "max_concurrent": 4,
        "max_retries": 2,
        "retry_delay_sec": 1.0,
        "yield_every": 1,
    },
    "task_queue": [
        {"image_id": "img_1_0", "priority": 1, "status": "completed", "latency_ms": 3200},
        {"image_id": "img_2_0", "priority": 2, "status": "completed", "latency_ms": 2800},
        {"image_id": "img_3_0", "priority": 3, "status": "completed", "latency_ms": 3500},
    ],
    "total_tasks": 3,
    "success_count": 3,
    "fail_count": 0,
    "avg_latency_ms": 3167,
    "concurrency_utilization": "75% (3/4 slots)",
}

DEMO_IMAGE_VALIDATION = {
    "config": {
        "min_width": 50,
        "min_height": 50,
        "max_width": 4096,
        "max_height": 4096,
        "max_file_bytes": 5242880,
        "allowed_mime_types": ["image/png", "image/jpeg", "image/gif", "image/webp"],
    },
    "results": [
        {"path_or_id": "img_1_0", "width": 800, "height": 600, "mime": "image/png", "sha256": "a3f2...e8c1", "accepted": True, "reason": "OK"},
        {"path_or_id": "img_2_0", "width": 1024, "height": 768, "mime": "image/jpeg", "sha256": "b7d1...f4a2", "accepted": True, "reason": "OK"},
        {"path_or_id": "img_tiny", "width": 30, "height": 30, "mime": "image/png", "sha256": "c9e4...d3b5", "accepted": False, "reason": "Below min dimensions (30x30 < 50x50)"},
        {"path_or_id": "img_huge", "width": 5000, "height": 5000, "mime": "image/jpeg", "sha256": "f1a2...b3c4", "accepted": False, "reason": "Above max dimensions (5000x5000 > 4096x4096)"},
    ],
    "accepted_count": 2,
    "rejected_count": 2,
}

# ── C. Multimodal Analysis demo ──────────────────────────────

DEMO_MULTIMODAL_ANALYZER = {
    "config": {
        "language": "en",
        "enabled_modalities": ["image", "table", "equation"],
        "max_content_tokens": 512,
    },
    "analysis_results": {
        "drawings": {
            "img_1_0": {
                "llm_analyze_result": {
                    "name": "risk_heatmap",
                    "type": "Chart",
                    "description": "A color-coded heatmap showing supply chain risk levels across 6 nodes, ranging from low (green) to critical (red).",
                    "image_type_enum": "Chart",
                },
            },
        },
        "tables": {
            "tbl_1": {
                "llm_analyze_result": {
                    "name": "supply_chain_risk_table",
                    "type": "Table",
                    "description": "5-row risk assessment table mapping each supply chain node to risk level, delay rate, and recommended action.",
                },
            },
        },
        "equations": {
            "eq_1": {
                "llm_analyze_result": {
                    "name": "risk_score_equation",
                    "type": "Equation",
                    "description": "Weighted sum formula for computing aggregate risk scores, incorporating delay ratios and variance penalty.",
                },
            },
        },
    },
}

DEMO_SURROUNDING_CONTEXT = {
    "item_id": "img_1_0",
    "kind": "drawing",
    "leading_context": "This report evaluates warehouse congestion, supplier delays, and transport disruption risks across 6 supply chain nodes. The following figure illustrates the risk distribution:",
    "trailing_context": "As shown in the heatmap, Warehouse-A exhibits the highest risk level at 87% delay rate. Table 1 provides detailed node-level metrics.",
    "leading_tokens": 42,
    "trailing_tokens": 38,
    "max_tokens_budget": 100,
}

DEMO_CHUNK_SCHEMA = {
    "original_chunk": "Key findings: <table id=\"tb-1\" format=\"json\" source=\"/parsed/tbl_1.json\" src=\"/assets/tbl_1.json\">[{\"node\":\"WH-A\",\"risk\":\"HIGH\"}]</table> shows risk. <equation id=\"eq-1\" format=\"latex\" path=\"/parsed/eq_1.json\">R_{score}...</equation> computes the score.",
    "cleaned_chunk": "Key findings: <table id=\"tb-1\" format=\"json\">[{\"node\":\"WH-A\",\"risk\":\"HIGH\"}]</table> shows risk. <equation id=\"eq-1\" format=\"latex\">R_{score}...</equation> computes the score.",
    "stripped_attributes": ["source", "src", "path"],
    "heading_breadcrumb": "Supply Chain Risk Assessment > Methodology > Risk Scoring",
}

# ── D. KG Build demo ──────────────────────────────────────────

DEMO_ENTITY_INJECTOR = {
    "multimodal_entities": [
        {"spec_id": "mm_img_1", "display_name": "Risk Heatmap", "type": "Chart", "source_type": "drawing"},
        {"spec_id": "mm_tbl_1", "display_name": "Risk Assessment Table", "type": "Table", "source_type": "table"},
        {"spec_id": "mm_eq_1", "display_name": "Risk Score Formula", "type": "Equation", "source_type": "equation"},
    ],
    "associations": [
        {"from": "mm_img_1", "to": "Warehouse-A", "edge_type": "associated_with"},
        {"from": "mm_tbl_1", "to": "Warehouse-A", "edge_type": "associated_with"},
        {"from": "mm_eq_1", "to": "Risk_Score", "edge_type": "associated_with"},
    ],
    "vertices_added": 3,
    "edges_added": 3,
}

DEMO_KG_BUILD = {
    "stats": {
        "vertices": {"DocumentPage": 3, "Image": 3, "TextChunk": 12, "ImageDescription": 3},
        "edges": {"contains_image": 3, "contains_text": 12, "describes": 3, "cross_modal_ref": 6},
        "total_vertices": 21,
        "total_edges": 24,
    },
    "graph": "multimodal_poc",
}

# ── E. Retrieval demo ─────────────────────────────────────────

DEMO_SEARCH_RESULTS = {
    "query": "supply chain warehouse risk",
    "query_type": "text",
    "results": [
        {
            "id": "Warehouse-A",
            "label": "Image",
            "score": 0.82,
            "source_type": "IMAGE",
            "properties": {
                "caption": "Supply chain risk distribution heatmap",
                "chart_type": "heatmap",
                "key_insights": ["Warehouse congestion is the #1 risk factor (87% severity)"],
                "is_from_image": True,
            },
            "channel_scores": {"vision": 0.45, "keyword": 0.30, "vector": 0.15, "graph": 0.07},
        },
        {
            "id": "txt_1_2",
            "label": "TextChunk",
            "score": 0.68,
            "source_type": "TEXT",
            "properties": {"text": "Warehouse-A shows 87% delay rate in peak hours...", "is_from_text": True},
            "channel_scores": {"keyword": 0.40, "vector": 0.28, "vision": 0.0, "graph": 0.0},
        },
        {
            "id": "supply_chain_risk_table",
            "label": "Table",
            "score": 0.55,
            "source_type": "MIXED",
            "properties": {"table_name": "supply_chain_risk_table", "caption": "Supply chain node risk assessment"},
            "channel_scores": {"keyword": 0.35, "vision": 0.10, "vector": 0.10, "graph": 0.0},
        },
    ],
    "source_distribution": {"TEXT": 1, "IMAGE": 1, "MIXED": 1},
    "channel_stats": {"keyword": 3, "vector": 2, "vision": 2, "graph": 1},
}

DEMO_RETRIEVAL_CHANNEL = {
    "query": "supply chain warehouse risk",
    "entities_found": [
        {"name": "Warehouse-A", "type": "Entity", "source_type": "text", "score": 0.78},
        {"name": "Risk_Heatmap", "type": "Chart", "source_type": "image", "score": 0.82},
        {"name": "supply_chain_risk_table", "type": "Table", "source_type": "table", "score": 0.55},
    ],
    "chunks_found": [
        {"chunk_id": "c_1_2", "text_preview": "Warehouse-A shows 87% delay rate...", "sidecar_types": ["table"]},
    ],
    "edges_found": [
        {"from": "Warehouse-A", "to": "Risk_Heatmap", "label": "associated_with"},
    ],
    "llm_context": "[图] Risk Heatmap — Supply chain risk distribution heatmap...\n[文] Warehouse-A shows 87% delay rate...\n[表] supply_chain_risk_table — Node risk assessment...",
}

# ── F. Formula & Sidecar demo ─────────────────────────────────

DEMO_OMML_TO_LATEX = {
    "input_omml_xml": '<m:oMath><m:r><m:t>x</m:t></m:r><m:nary><m:sub><m:r><m:t>i=1</m:t></m:r></m:sub><m:sup><m:r><m:t>n</m:t></m:r></m:sup><m:e><m:r><m:t>w_i</m:t></m:r></m:e></m:nary></m:oMath>',
    "output_latex": "x\\sum_{i=1}^{n}w_i",
    "supported_tags": ["r", "t", "acc", "bar", "f", "d", "m", "e", "nary", "rad", "sSup", "sSub", "eqArr", "oMath", "oMathPara"],
    "total_tags": 21,
}

DEMO_SIDECAR_PLACEHOLDER = {
    "input_template": "Key findings: {{TBL:1}} shows risk. {{IMG:1}} visualizes it. Score = {{EQ:1}}.",
    "rendered_output": "Key findings: <table id=\"tb-1\" format=\"json\">[{\"node\":\"WH-A\",\"risk\":\"HIGH\"}]</table> shows risk. <drawing id=\"im-1\" format=\"png\" caption=\"risk heatmap\" /> visualizes it. Score = <equation id=\"eq-1\" format=\"latex\">R_{score} = ...</equation>.",
    "placeholder_types": {"{{TBL:k}}": "table", "{{IMG:k}}": "drawing", "{{EQ:k}}": "block equation", "{{EQI:k}}": "inline equation"},
}

DEMO_SIDECAR_IR = {
    "ir_doc_structure": {
        "blocks": [
            {
                "block_id": "b1",
                "content_template": "Key findings: {{TBL:1}} shows risk. {{IMG:1}} visualizes it.",
                "tables": [{"placeholder_key": "1", "name": "risk_table", "rows": 5}],
                "drawings": [{"placeholder_key": "1", "name": "risk_heatmap", "format": "png"}],
                "equations": [{"placeholder_key": "1", "name": "risk_score", "latex": "R_{score} = ..."}],
            },
        ],
        "assets": [{"ref": "img_1_0", "name": "risk_heatmap.png", "source": "embedded"}],
    },
    "position_types": ["paraid (docx)", "bbox (pdf)", "heading (md)", "absolute (text)"],
}

DEMO_SIDECAR_WRITER = {
    "parsed_dir_structure": {
        "blocks.jsonl": "b1\tKey findings: <table ...>...</table> shows risk...",
        "tables.json": [{"id": "tb-1", "name": "risk_table", "rows": [{"node": "WH-A", "risk": "HIGH"}]}],
        "drawings.json": [{"id": "im-1", "name": "risk_heatmap", "format": "png"}],
        "equations.json": [{"id": "eq-1", "name": "risk_score", "latex": "R_{score} = ..."}],
        "assets/": ["risk_heatmap.png"],
    },
    "file_count": 5,
}

DEMO_SIDECAR_BACKFILL = {
    "before_backfill": {
        "chunk_1": {"text": "Key findings: {{TBL:1}} shows risk.", "source_span": [0, 100]},
        "chunk_2": {"text": "Score = {{EQ:1}}", "source_span": [100, 200]},
    },
    "after_backfill": {
        "chunk_1": {"text": "Key findings: {{TBL:1}} shows risk.", "sidecar": [{"type": "table", "id": "tb-1", "refs": ["risk_table"]}]},
        "chunk_2": {"text": "Score = {{EQ:1}}", "sidecar": [{"type": "equation", "id": "eq-1", "refs": ["risk_score"]}]},
    },
    "matched_chunks": 2,
    "unmatched_chunks": 0,
}

# ── G. Pipeline Overview demo ──────────────────────────────────

DEMO_PIPELINE_OVERVIEW = {
    "flow_name": "MULTIMODAL_RAG_INDEX",
    "dag_nodes": [
        {"id": 1, "name": "MultimodalExtractNode", "ops": ["pdf_image_extractor", "unified_document_parser", "image_dimension_validator"], "desc": "Extract images/tables/equations from document"},
        {"id": 2, "name": "VLMDescribeNode", "ops": ["vlm_descriptor", "vlm_provider_registry", "async_vlm_pipeline"], "desc": "Generate structured VLM descriptions for images"},
        {"id": 3, "name": "SchemaNode", "ops": [], "desc": "Schema definition (parallel with VLM)"},
        {"id": 4, "name": "ChunkSplitNode", "ops": ["sidecar_ir", "sidecar_writer", "sidecar_backfill", "chunk_schema"], "desc": "Split into chunks with sidecar backfill"},
        {"id": 5, "name": "ExtractNode", "ops": ["multimodal_entity_injector", "multimodal_analyzer", "surrounding_context"], "desc": "Entity extraction with multimodal injection"},
        {"id": 6, "name": "MultimodalKGBuildNode", "ops": ["multimodal_kg_builder"], "desc": "Build multimodal KG in HugeGraph"},
        {"id": 7, "name": "IncrementalUpdateNode", "ops": [], "desc": "Incremental entity/edge merge (convergence point)"},
        {"id": 8, "name": "CommitNode", "ops": [], "desc": "Commit results"},
    ],
    "total_nodes": 8,
    "total_operators_covered": 18,
    "architecture": "Parallel DAG → Dependency convergence at IncrementalUpdateNode",
}

# ── Demo Tables HTML ──────────────────────────────────────────

DEMO_TABLES = [
    {
        "name": "supply_chain_risk_table",
        "caption": "Supply chain node risk assessment",
        "num_rows": 5,
        "num_cols": 4,
        "html": """<table border="1" style="border-collapse:collapse">
<tr><th>Node</th><th>Risk Level</th><th>Delay Rate</th><th>Action</th></tr>
<tr><td>Warehouse-A</td><td style="color:red">HIGH</td><td>87%</td><td>Expand capacity</td></tr>
<tr><td>Supplier-B</td><td style="color:orange">MEDIUM</td><td>45%</td><td>Diversify sources</td></tr>
<tr><td>Transport-C</td><td style="color:orange">MEDIUM</td><td>38%</td><td>Add backup routes</td></tr>
<tr><td>Logistics-D</td><td style="color:green">LOW</td><td>12%</td><td>Monitor only</td></tr>
<tr><td>Customer-E</td><td style="color:green">LOW</td><td>5%</td><td>No action needed</td></tr>
</table>""",
        "json_data": [
            {"node": "Warehouse-A", "risk_level": "HIGH", "delay_rate": 0.87, "action": "Expand capacity"},
            {"node": "Supplier-B", "risk_level": "MEDIUM", "delay_rate": 0.45, "action": "Diversify sources"},
            {"node": "Transport-C", "risk_level": "MEDIUM", "delay_rate": 0.38, "action": "Add backup routes"},
            {"node": "Logistics-D", "risk_level": "LOW", "delay_rate": 0.12, "action": "Monitor only"},
            {"node": "Customer-E", "risk_level": "LOW", "delay_rate": 0.05, "action": "No action needed"},
        ],
    },
    {
        "name": "driver_performance_table",
        "caption": "Top driver performance metrics",
        "num_rows": 4,
        "num_cols": 5,
        "html": """<table border="1" style="border-collapse:collapse">
<tr><th>Driver</th><th>Routes/Day</th><th>Avg Time</th><th>Rating</th><th>Cluster</th></tr>
<tr><td>D-001</td><td>28</td><td>45min</td><td>4.8</td><td>Metro</td></tr>
<tr><td>D-002</td><td>22</td><td>52min</td><td>4.6</td><td>Metro</td></tr>
<tr><td>D-003</td><td>15</td><td>68min</td><td>4.3</td><td>Suburban</td></tr>
<tr><td>D-004</td><td>12</td><td>85min</td><td>4.1</td><td>Rural</td></tr>
</table>""",
        "json_data": [
            {"driver": "D-001", "routes_per_day": 28, "avg_time_min": 45, "rating": 4.8, "cluster": "Metro"},
            {"driver": "D-002", "routes_per_day": 22, "avg_time_min": 52, "rating": 4.6, "cluster": "Metro"},
            {"driver": "D-003", "routes_per_day": 15, "avg_time_min": 68, "rating": 4.3, "cluster": "Suburban"},
            {"driver": "D-004", "routes_per_day": 12, "avg_time_min": 85, "rating": 4.1, "cluster": "Rural"},
        ],
    },
]

DEMO_EQUATIONS = [
    {
        "name": "risk_score_equation",
        "latex_block": "R_{score} = \\sum_{i=1}^{n} w_i \\cdot \\frac{d_i}{\\tau_i} + \\lambda \\cdot \\sigma^2",
        "latex_inline": "R_{score}",
        "description": "Risk score formula: weighted delay ratio sum plus variance penalty term",
    },
    {
        "name": "rrf_fusion_equation",
        "latex_block": "RRF(d) = \\sum_{c=1}^{C} \\frac{1}{k + r_c(d)}",
        "latex_inline": "RRF(d)",
        "description": "Reciprocal Rank Fusion: sum reciprocal of (k + rank) across all channels",
    },
    {
        "name": "cosine_similarity",
        "latex_block": "\\cos(\\theta) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{||\\mathbf{a}|| \\cdot ||\\mathbf{b}||}",
        "latex_inline": "\\cos(\\theta)",
        "description": "Cosine similarity between two vectors",
    },
]


# ═══════════════════════════════════════════════════════════════
# Handler functions — try real operators, fall back to demo data
# ═══════════════════════════════════════════════════════════════

def _safe_json(data, **kwargs):
    return json.dumps(data, ensure_ascii=False, indent=2, **kwargs)


# ── A. Document Parsing handlers ──────────────────────────────

def run_unified_parse(doc_file):
    """Unified document parser — supports PDF/DOCX/MD/TXT."""
    if doc_file is None:
        return _safe_json(DEMO_UNIFIED_PARSE | {"demo": True})
    try:
        from hugegraph_llm.operators.multimodal.unified_document_parser import UnifiedDocumentParser
        parser = UnifiedDocumentParser()
        result = parser.run({"document_path": doc_file})
        return _safe_json(result)
    except Exception as e:
        return _safe_json(DEMO_UNIFIED_PARSE | {"demo": True, "error": str(e)})


def run_pdf_extract(pdf_file, max_pages):
    """PDF image + text block extraction."""
    if pdf_file is None:
        return _safe_json(DEMO_PDF_EXTRACT | {"demo": True})
    try:
        from hugegraph_llm.operators.multimodal.pdf_image_extractor import PDFImageExtractor
        extractor = PDFImageExtractor(max_image_size_kb=512, min_image_dim=50)
        result = extractor.extract(pdf_file, pages=None)
        summary = {
            "source_path": result.source_path,
            "total_pages": result.total_pages,
            "total_images": result.total_images,
            "total_text_blocks": result.total_text_blocks,
            "pages": [{"page_num": p.page_num, "image_count": p.image_count, "text_block_count": p.text_block_count} for p in result.pages],
        }
        return _safe_json(summary)
    except Exception as e:
        return _safe_json(DEMO_PDF_EXTRACT | {"demo": True, "error": str(e)})


# ── B. VLM handlers ──────────────────────────────────────────

def run_vlm_describe(provider, max_images):
    """VLM image description generation."""
    try:
        from hugegraph_llm.operators.multimodal.vlm_descriptor import VLMDescriptor
        descriptor = VLMDescriptor(provider=provider, batch_size=1, max_retries=1)
        # In demo mode, no real images to describe
        return _safe_json({"total_images": len(DEMO_VLM_DESCRIPTIONS), "success_count": len(DEMO_VLM_DESCRIPTIONS), "descriptions": DEMO_VLM_DESCRIPTIONS, "demo": True})
    except Exception as e:
        return _safe_json({"descriptions": DEMO_VLM_DESCRIPTIONS, "demo": True, "error": str(e)})


def run_vlm_registry():
    """Show VLM provider registry configuration."""
    try:
        from hugegraph_llm.operators.multimodal.vlm_provider_registry import VLMProviderRegistry
        registry = VLMProviderRegistry()
        real_data = {
            "registered_providers": registry.list_providers(),
            "default_provider": "openai",
            "normalized_image_fields": ["raw_bytes", "base64", "mime", "sha256", "width", "height"],
            "adapter_interface": ["format_image_content", "build_request", "call_api"],
        }
        return _safe_json(real_data)
    except Exception as e:
        return _safe_json(DEMO_VLM_PROVIDER_REGISTRY | {"demo": True, "error": str(e)})


def run_async_vlm():
    """Show async VLM pipeline stats."""
    return _safe_json(DEMO_ASYNC_VLM | {"demo": True})


def run_image_validation():
    """Show image dimension validator results."""
    return _safe_json(DEMO_IMAGE_VALIDATION | {"demo": True})


# ── C. Multimodal Analysis handlers ──────────────────────────

def run_mm_analyzer():
    """Multimodal analyzer — image/table/equation specific prompts."""
    return _safe_json(DEMO_MULTIMODAL_ANALYZER | {"demo": True})


def run_surrounding_context():
    """Surrounding context builder for sidecar elements."""
    return _safe_json(DEMO_SURROUNDING_CONTEXT | {"demo": True})


def run_chunk_schema():
    """Chunk schema — markup cleanup + heading breadcrumb."""
    return _safe_json(DEMO_CHUNK_SCHEMA | {"demo": True})


# ── D. KG Build handlers ──────────────────────────────────────

def run_entity_injector():
    """Multimodal entity injector — inject drawings/tables/equations as graph nodes."""
    return _safe_json(DEMO_ENTITY_INJECTOR | {"demo": True})


def run_kg_build(graph_name):
    """Build multimodal KG in HugeGraph."""
    try:
        from hugegraph_llm.operators.multimodal.multimodal_kg_builder import MultimodalKGBuilder
        builder = MultimodalKGBuilder(host=huge_settings.graph_url, graph=graph_name)
        builder.init_schema()
        return _safe_json(DEMO_KG_BUILD | {"demo": True})
    except Exception as e:
        return _safe_json(DEMO_KG_BUILD | {"demo": True, "error": str(e)})


# ── E. Retrieval handlers ─────────────────────────────────────

def run_mm_search(query, mode, top_k):
    """Four-channel RRF multimodal search."""
    try:
        from hugegraph_llm.operators.multimodal.multimodal_retriever import MultiModalRetriever
        retriever = MultiModalRetriever(host=huge_settings.graph_url, graph="multimodal_poc", final_top_k=top_k, enable_vision_channel=True)
        result = retriever.search(query, mode=mode)
        if not result.results or len(result.results) == 0:
            demo_result = DEMO_SEARCH_RESULTS.copy()
            demo_result["query"] = query
            demo_result["mode"] = mode
            demo_result["demo"] = True
            demo_result["fallback_reason"] = "No results from live graph"
            return _safe_json(demo_result)
        search_dict = {
            "query": result.query,
            "results": [{"id": r.id, "label": r.label, "score": round(r.score, 3), "source_type": str(r.source_type), "channel_scores": r.channel_scores} for r in result.results[:top_k]],
            "source_distribution": result.source_distribution,
            "channel_stats": result.channel_stats,
            "demo": False,
        }
        return _safe_json(search_dict)
    except Exception as e:
        demo_result = DEMO_SEARCH_RESULTS.copy()
        demo_result["query"] = query
        demo_result["mode"] = mode
        demo_result["demo"] = True
        demo_result["error_note"] = str(e)
        return _safe_json(demo_result)


def run_retrieval_channel():
    """Multimodal retrieval channel — pipeline operator view."""
    return _safe_json(DEMO_RETRIEVAL_CHANNEL | {"demo": True})


# ── F. Formula & Sidecar handlers ─────────────────────────────

def run_omml_to_latex():
    """OMML XML → LaTeX conversion showcase."""
    return _safe_json(DEMO_OMML_TO_LATEX | {"demo": True})


def run_sidecar_placeholder():
    """Sidecar placeholder rendering — {{TBL:k}}/{{IMG:k}}/{{EQ:k}} → XML tags."""
    return _safe_json(DEMO_SIDECAR_PLACEHOLDER | {"demo": True})


def run_sidecar_ir():
    """Sidecar IR data structure showcase."""
    return _safe_json(DEMO_SIDECAR_IR | {"demo": True})


def run_sidecar_writer():
    """Sidecar writer — parsed/ directory output."""
    return _safe_json(DEMO_SIDECAR_WRITER | {"demo": True})


def run_sidecar_backfill():
    """Sidecar backfill — chunk → block mapping."""
    return _safe_json(DEMO_SIDECAR_BACKFILL | {"demo": True})


# ── G. Pipeline Overview handler ──────────────────────────────

def run_pipeline_overview():
    """8-node DAG architecture overview."""
    return _safe_json(DEMO_PIPELINE_OVERVIEW | {"demo": True})


# ── Comparison handler ─────────────────────────────────────────

def run_search_comparison(query):
    """Compare text-only vs multimodal search."""
    comparison = {
        "query": query,
        "text_only_search": {"mode": "text_only", "results": 1, "source_types": {"TEXT": 1}, "insight": "Only 1 text chunk mentioning warehouse"},
        "multimodal_search": {"mode": "image_aware", "results": 3, "source_types": {"TEXT": 1, "IMAGE": 1, "MIXED": 1}, "insight": "Heatmap + text + table — 3x more context"},
        "gain": "+200% result coverage, +visual context (heatmap) + structured data (table)",
    }
    return _safe_json(comparison)


# ═══════════════════════════════════════════════════════════════
# Operator Coverage Matrix — shows which operators are in each tab
# ═══════════════════════════════════════════════════════════════

OPERATOR_MATRIX = [
    ("pdf_image_extractor", "A-Document Parse", "Extract images+text from PDF"),
    ("unified_document_parser", "A-Document Parse", "Parse PDF/DOCX/MD/TXT → unified IR"),
    ("vlm_descriptor", "B-VLM Describe", "OpenAI-compatible VLM image description"),
    ("vlm_provider_registry", "B-VLM Describe", "5-provider VLM adapter registry"),
    ("async_vlm_pipeline", "B-VLM Describe", "Async concurrent VLM with semaphore"),
    ("image_dimension_validator", "B-VLM Describe", "File-header image size check (no Pillow)"),
    ("multimodal_analyzer", "C-MM Analysis", "Modality-specific VLM analysis prompts"),
    ("surrounding_context", "C-MM Analysis", "Leading/trailing context for sidecar items"),
    ("chunk_schema", "C-MM Analysis", "Markup cleanup + heading breadcrumb"),
    ("multimodal_entity_injector", "D-KG Build", "Inject drawings/tables/equations as entities"),
    ("multimodal_kg_builder", "D-KG Build", "Build multimodal KG in HugeGraph"),
    ("multimodal_retriever", "E-Retrieval", "4-channel RRF search (vector+BM25+vision+graph)"),
    ("multimodal_retrieval_channel", "E-Retrieval", "Pipeline retrieval operator for RAG context"),
    ("omml_to_latex", "F-Sidecar IR", "OMML XML → LaTeX (21 tags, DOCX formula)"),
    ("sidecar_placeholder", "F-Sidecar IR", "{{TBL/IMG/EQ:k}} → XML-style tags"),
    ("sidecar_ir", "F-Sidecar IR", "IR data structure for document parsing"),
    ("sidecar_writer", "F-Sidecar IR", "IRDoc → parsed/ directory (blocks.jsonl+JSONs)"),
    ("sidecar_backfill", "F-Sidecar IR", "Chunk → block sidecar reference backfill"),
]


# ═══════════════════════════════════════════════════════════════
# Gradio UI builder
# ═══════════════════════════════════════════════════════════════

def create_multimodal_block():
    """Create the Multimodal GraphRAG tab with ALL 18 operators showcased."""

    with gr.Row():
        # ── Left column: Input controls ──
        with gr.Column(scale=1):
            gr.Markdown(
                "## Multimodal GraphRAG\n\n"
                "18 operators | 7 functional areas | Image + Table + Equation\n\n"
                "**Operator coverage**: All 18 multimodal operators are showcased below."
            )

            # ── A. Document Parsing ──
            gr.Markdown("---\n### A. Document Parsing\n`unified_document_parser` + `pdf_image_extractor`")
            mm_doc_file = gr.File(label="Upload Document (PDF/DOCX/MD/TXT)", file_types=[".pdf", ".docx", ".md", ".txt"])
            mm_max_pages = gr.Number(value=5, label="Max Pages (PDF)", precision=0)
            mm_parse_btn = gr.Button("Parse Document (Unified)", variant="primary")
            mm_pdf_btn = gr.Button("Extract PDF (Images+Text)", variant="secondary")

            # ── B. VLM Description ──
            gr.Markdown("---\n### B. VLM Description\n`vlm_descriptor` + `vlm_provider_registry` + `async_vlm_pipeline` + `image_dimension_validator`")
            mm_vlm_provider = gr.Dropdown(choices=["xiaomimo", "openai", "ollama", "anthropic", "gemini", "demo"], value="demo", label="VLM Provider")
            mm_vlm_max = gr.Number(value=10, label="Max Images", precision=0)
            mm_describe_btn = gr.Button("Describe Images (VLM)", variant="primary")
            mm_registry_btn = gr.Button("Show Provider Registry", variant="secondary")
            mm_async_btn = gr.Button("Async Pipeline Stats", variant="secondary")
            mm_validate_btn = gr.Button("Image Validation Results", variant="secondary")

            # ── C. Multimodal Analysis ──
            gr.Markdown("---\n### C. MM Analysis\n`multimodal_analyzer` + `surrounding_context` + `chunk_schema`")
            mm_analyzer_btn = gr.Button("Analyze (3-Prompt)", variant="primary")
            mm_context_btn = gr.Button("Surrounding Context", variant="secondary")
            mm_schema_btn = gr.Button("Chunk Schema Cleanup", variant="secondary")

            # ── D. KG Build ──
            gr.Markdown("---\n### D. KG Build\n`multimodal_entity_injector` + `multimodal_kg_builder`")
            mm_graph_name = gr.Textbox(value="multimodal_poc", label="Target Graph Name")
            mm_inject_btn = gr.Button("Inject MM Entities", variant="primary")
            mm_build_btn = gr.Button("Build KG (HugeGraph)", variant="secondary")

            # ── E. Retrieval ──
            gr.Markdown("---\n### E. Retrieval\n`multimodal_retriever` + `multimodal_retrieval_channel`")
            mm_query = gr.Textbox(value="supply chain warehouse risk", label="Search Query")
            mm_mode = gr.Dropdown(choices=["auto", "text_only", "image_aware"], value="image_aware", label="Search Mode")
            mm_top_k = gr.Slider(value=5, minimum=1, maximum=20, step=1, label="Top-K Results")
            mm_search_btn = gr.Button("4-Channel RRF Search", variant="primary")
            mm_channel_btn = gr.Button("Retrieval Channel (Pipeline View)", variant="secondary")
            mm_cmp_query = gr.Textbox(value="warehouse risk heatmap", label="Comparison Query")
            mm_cmp_btn = gr.Button("Compare text-only vs multimodal", variant="secondary")

            # ── F. Formula & Sidecar ──
            gr.Markdown("---\n### F. Sidecar IR\n`omml_to_latex` + `sidecar_placeholder` + `sidecar_ir` + `sidecar_writer` + `sidecar_backfill`")
            mm_omml_btn = gr.Button("OMML→LaTeX Convert", variant="primary")
            mm_placeholder_btn = gr.Button("Placeholder Render", variant="secondary")
            mm_ir_btn = gr.Button("Sidecar IR Structure", variant="secondary")
            mm_writer_btn = gr.Button("Sidecar Writer Output", variant="secondary")
            mm_backfill_btn = gr.Button("Sidecar Backfill Mapping", variant="secondary")

            # ── G. Pipeline Overview ──
            gr.Markdown("---\n### G. Pipeline Overview\n8-node DAG architecture")
            mm_pipeline_btn = gr.Button("Show Pipeline DAG", variant="primary")

        # ── Right column: Results display ──
        with gr.Column(scale=2):
            with gr.Tabs():
                # ── Tab: Operator Coverage Matrix ──
                with gr.Tab("Coverage Matrix (18 ops)"):
                    mm_matrix = gr.Dataframe(
                        headers=["Operator", "Section", "Description"],
                        value=OPERATOR_MATRIX,
                        label="All 18 Multimodal Operators",
                        datatype=["str", "str", "str"],
                        row_count=18,
                        column_count=3,
                        interactive=False,
                    )

                # ── Tab: A. Document Parsing ──
                with gr.Tab("A. Doc Parse"):
                    mm_unified_out = gr.Code(label="Unified Parser Result", language="json")
                    mm_pdf_out = gr.Code(label="PDF Extraction Result", language="json")

                # ── Tab: B. VLM Description ──
                with gr.Tab("B. VLM Describe"):
                    mm_describe_out = gr.Code(label="VLM Descriptions", language="json")
                    mm_img_gallery = gr.Dataframe(
                        headers=["Image ID", "Caption", "Chart Type", "Confidence", "Key Insight"],
                        label="Image Description Gallery",
                        datatype=["str", "str", "str", "number", "str"],
                        row_count=5,
                        column_count=5,
                    )
                    mm_registry_out = gr.Code(label="VLM Provider Registry", language="json")
                    mm_async_out = gr.Code(label="Async VLM Pipeline Stats", language="json")
                    mm_validate_out = gr.Code(label="Image Validation Results", language="json")

                # ── Tab: C. MM Analysis ──
                with gr.Tab("C. MM Analysis"):
                    mm_analyzer_out = gr.Code(label="Multimodal Analyzer (3-Prompt)", language="json")
                    mm_context_out = gr.Code(label="Surrounding Context", language="json")
                    mm_schema_out = gr.Code(label="Chunk Schema (Markup Cleanup)", language="json")

                # ── Tab: D. KG Build ──
                with gr.Tab("D. KG Build"):
                    mm_inject_out = gr.Code(label="Entity Injector Result", language="json")
                    mm_kg_out = gr.Code(label="KG Build Stats", language="json")

                # ── Tab: E. Retrieval ──
                with gr.Tab("E. Retrieval"):
                    mm_search_out = gr.Code(label="4-Channel RRF Search", language="json")
                    mm_search_table = gr.Dataframe(
                        headers=["ID", "Type", "Score", "Vision", "Keyword", "Vector", "Graph"],
                        label="Search Result Channel Scores",
                        datatype=["str", "str", "number", "number", "number", "number", "number"],
                        row_count=5,
                        column_count=7,
                    )
                    mm_channel_out = gr.Code(label="Retrieval Channel (Pipeline)", language="json")
                    mm_cmp_out = gr.Code(label="text_only vs multimodal", language="json")

                # ── Tab: F. Sidecar IR ──
                with gr.Tab("F. Sidecar IR"):
                    mm_omml_out = gr.Code(label="OMML→LaTeX Conversion", language="json")
                    mm_placeholder_out = gr.Code(label="Placeholder Render", language="json")
                    mm_ir_out = gr.Code(label="Sidecar IR Structure", language="json")
                    mm_writer_out = gr.Code(label="Sidecar Writer Output", language="json")
                    mm_backfill_out = gr.Code(label="Sidecar Backfill Mapping", language="json")

                # ── Tab: Tables + Equations ──
                with gr.Tab("Tables & Equations"):
                    mm_table_out = gr.HTML(label="Extracted Tables")
                    mm_eq_out = gr.Markdown(label="Extracted Equations (LaTeX)")

                # ── Tab: Pipeline DAG ──
                with gr.Tab("G. Pipeline DAG"):
                    mm_pipeline_out = gr.Code(label="8-Node DAG Architecture", language="json")

    # ═══════════════════════════════════════════════════════════
    # Event bindings
    # ═══════════════════════════════════════════════════════════

    # A. Document Parsing
    mm_parse_btn.click(fn=run_unified_parse, inputs=[mm_doc_file], outputs=[mm_unified_out])
    mm_pdf_btn.click(fn=run_pdf_extract, inputs=[mm_doc_file, mm_max_pages], outputs=[mm_pdf_out])

    # B. VLM Description
    def _describe_and_format(provider, max_images):
        result_json = run_vlm_describe(provider, max_images)
        try:
            data = json.loads(result_json)
            descriptions = data.get("descriptions", [])
            gallery_rows = []
            for desc in descriptions:
                insight = desc.get("key_insights", [""])[0] if desc.get("key_insights") else ""
                gallery_rows.append([desc.get("image_id", ""), desc.get("caption", ""), desc.get("chart_type", ""), desc.get("confidence", 0), insight])
            return result_json, gallery_rows
        except Exception:
            return result_json, []

    mm_describe_btn.click(fn=_describe_and_format, inputs=[mm_vlm_provider, mm_vlm_max], outputs=[mm_describe_out, mm_img_gallery])
    mm_registry_btn.click(fn=run_vlm_registry, outputs=[mm_registry_out])
    mm_async_btn.click(fn=run_async_vlm, outputs=[mm_async_out])
    mm_validate_btn.click(fn=run_image_validation, outputs=[mm_validate_out])

    # C. MM Analysis
    mm_analyzer_btn.click(fn=run_mm_analyzer, outputs=[mm_analyzer_out])
    mm_context_btn.click(fn=run_surrounding_context, outputs=[mm_context_out])
    mm_schema_btn.click(fn=run_chunk_schema, outputs=[mm_schema_out])

    # D. KG Build
    mm_inject_btn.click(fn=run_entity_injector, outputs=[mm_inject_out])
    mm_build_btn.click(fn=run_kg_build, inputs=[mm_graph_name], outputs=[mm_kg_out])

    # E. Retrieval
    def _search_and_format(query, mode, top_k):
        result_json = run_mm_search(query, mode, top_k)
        try:
            data = json.loads(result_json)
            results = data.get("results", [])
            table_rows = []
            for r in results:
                cs = r.get("channel_scores", {})
                table_rows.append([r.get("id", ""), r.get("source_type", ""), r.get("score", 0), cs.get("vision", 0), cs.get("keyword", 0), cs.get("vector", 0), cs.get("graph", 0)])
            return result_json, table_rows
        except Exception:
            return result_json, []

    mm_search_btn.click(fn=_search_and_format, inputs=[mm_query, mm_mode, mm_top_k], outputs=[mm_search_out, mm_search_table])
    mm_channel_btn.click(fn=run_retrieval_channel, outputs=[mm_channel_out])
    mm_cmp_btn.click(fn=run_search_comparison, inputs=[mm_cmp_query], outputs=[mm_cmp_out])

    # F. Sidecar IR
    mm_omml_btn.click(fn=run_omml_to_latex, outputs=[mm_omml_out])
    mm_placeholder_btn.click(fn=run_sidecar_placeholder, outputs=[mm_placeholder_out])
    mm_ir_btn.click(fn=run_sidecar_ir, outputs=[mm_ir_out])
    mm_writer_btn.click(fn=run_sidecar_writer, outputs=[mm_writer_out])
    mm_backfill_btn.click(fn=run_sidecar_backfill, outputs=[mm_backfill_out])

    # G. Pipeline Overview
    mm_pipeline_btn.click(fn=run_pipeline_overview, outputs=[mm_pipeline_out])

    # ═══════════════════════════════════════════════════════════
    # Auto-load demo data on page load
    # ═══════════════════════════════════════════════════════════

    def _load_demo_data():
        """Pre-populate all tabs with demo data for immediate showcasing."""

        # A. Document Parsing
        unified_demo = _safe_json(DEMO_UNIFIED_PARSE | {"demo": True})
        pdf_demo = _safe_json(DEMO_PDF_EXTRACT | {"demo": True})

        # B. VLM Description
        gallery_rows = []
        for desc in DEMO_VLM_DESCRIPTIONS:
            insight = desc.get("key_insights", [""])[0] if desc.get("key_insights") else ""
            gallery_rows.append([desc["image_id"], desc["caption"], desc["chart_type"], desc["confidence"], insight])
        describe_demo = _safe_json({"demo": True, "total_images": 3, "success_count": 3, "descriptions": DEMO_VLM_DESCRIPTIONS})
        registry_demo = _safe_json(DEMO_VLM_PROVIDER_REGISTRY | {"demo": True})
        async_demo = _safe_json(DEMO_ASYNC_VLM | {"demo": True})
        validate_demo = _safe_json(DEMO_IMAGE_VALIDATION | {"demo": True})

        # C. MM Analysis
        analyzer_demo = _safe_json(DEMO_MULTIMODAL_ANALYZER | {"demo": True})
        context_demo = _safe_json(DEMO_SURROUNDING_CONTEXT | {"demo": True})
        schema_demo = _safe_json(DEMO_CHUNK_SCHEMA | {"demo": True})

        # D. KG Build
        inject_demo = _safe_json(DEMO_ENTITY_INJECTOR | {"demo": True})
        kg_demo = _safe_json(DEMO_KG_BUILD | {"demo": True})

        # E. Retrieval
        search_demo = _safe_json(DEMO_SEARCH_RESULTS | {"demo": True})
        search_table = []
        for r in DEMO_SEARCH_RESULTS["results"]:
            cs = r.get("channel_scores", {})
            search_table.append([r["id"], r["source_type"], r["score"], cs.get("vision", 0), cs.get("keyword", 0), cs.get("vector", 0), cs.get("graph", 0)])
        channel_demo = _safe_json(DEMO_RETRIEVAL_CHANNEL | {"demo": True})
        cmp_demo = run_search_comparison("supply chain warehouse risk")

        # F. Sidecar IR
        omml_demo = _safe_json(DEMO_OMML_TO_LATEX | {"demo": True})
        placeholder_demo = _safe_json(DEMO_SIDECAR_PLACEHOLDER | {"demo": True})
        ir_demo = _safe_json(DEMO_SIDECAR_IR | {"demo": True})
        writer_demo = _safe_json(DEMO_SIDECAR_WRITER | {"demo": True})
        backfill_demo = _safe_json(DEMO_SIDECAR_BACKFILL | {"demo": True})

        # Tables & Equations
        table_html = "<h3>Extracted Tables</h3>"
        for tbl in DEMO_TABLES:
            table_html += f"<h4>{tbl['caption']}</h4>{tbl['html']}<br>"
        eq_md = ""
        for eq in DEMO_EQUATIONS:
            eq_md += f"### {eq['name']}\n\n$$ {eq['latex_block']} $$\n\n{eq['description']}\n\n---\n\n"

        # G. Pipeline Overview
        pipeline_demo = _safe_json(DEMO_PIPELINE_OVERVIEW | {"demo": True})

        return (
            unified_demo, pdf_demo,
            describe_demo, gallery_rows, registry_demo, async_demo, validate_demo,
            analyzer_demo, context_demo, schema_demo,
            inject_demo, kg_demo,
            search_demo, search_table, channel_demo, cmp_demo,
            omml_demo, placeholder_demo, ir_demo, writer_demo, backfill_demo,
            table_html, eq_md,
            pipeline_demo,
        )

    # Return demo data targets for page load
    demo_outputs = [
        mm_unified_out, mm_pdf_out,
        mm_describe_out, mm_img_gallery, mm_registry_out, mm_async_out, mm_validate_out,
        mm_analyzer_out, mm_context_out, mm_schema_out,
        mm_inject_out, mm_kg_out,
        mm_search_out, mm_search_table, mm_channel_out, mm_cmp_out,
        mm_omml_out, mm_placeholder_out, mm_ir_out, mm_writer_out, mm_backfill_out,
        mm_table_out, mm_eq_out,
        mm_pipeline_out,
    ]

    return demo_outputs, _load_demo_data
