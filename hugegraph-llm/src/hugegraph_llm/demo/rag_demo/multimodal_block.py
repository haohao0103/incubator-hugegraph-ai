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

Provides interactive demonstration of:
- Image extraction and VLM description (from PDF)
- Table structured extraction (HTML/JSON)
- Equation extraction (LaTeX normalization)
- Multimodal KG construction and visualization
- Four-channel RRF retrieval (text + image hybrid search)

All demo data is self-contained so the page can run without
external dependencies (PDF/VLM/HugeGraph) for showcasing.
"""

import json
import tempfile
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr

from hugegraph_llm.config import huge_settings
from hugegraph_llm.utils.log import log

# ── Self-contained demo data ──────────────────────────────────

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
        "caption": "Top 10 driver performance metrics",
        "num_rows": 10,
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
        "is_block": True,
    },
    {
        "name": "rrf_fusion_equation",
        "latex_block": "RRF(d) = \\sum_{c=1}^{C} \\frac{1}{k + r_c(d)}",
        "latex_inline": "RRF(d)",
        "description": "Reciprocal Rank Fusion: sum reciprocal of (k + rank) across all channels",
        "is_block": True,
    },
    {
        "name": "cosine_similarity",
        "latex_block": "\\cos(\\theta) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{||\\mathbf{a}|| \\cdot ||\\mathbf{b}||}",
        "latex_inline": "\\cos(\\theta)",
        "description": "Cosine similarity between two vectors",
        "is_block": True,
    },
]

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
            "channel_scores": {
                "vision": 0.45,
                "keyword": 0.30,
                "vector": 0.15,
                "graph": 0.07,
            },
        },
        {
            "id": "txt_1_2",
            "label": "TextChunk",
            "score": 0.68,
            "source_type": "TEXT",
            "properties": {
                "text": "Warehouse-A shows 87% delay rate in peak hours...",
                "is_from_text": True,
            },
            "channel_scores": {
                "keyword": 0.40,
                "vector": 0.28,
                "vision": 0.0,
                "graph": 0.0,
            },
        },
        {
            "id": "supply_chain_risk_table",
            "label": "Table",
            "score": 0.55,
            "source_type": "MIXED",
            "properties": {
                "table_name": "supply_chain_risk_table",
                "caption": "Supply chain node risk assessment",
                "is_from_image": False,
                "is_from_text": True,
            },
            "channel_scores": {
                "keyword": 0.35,
                "vision": 0.10,
                "vector": 0.10,
                "graph": 0.0,
            },
        },
    ],
    "source_distribution": {"TEXT": 1, "IMAGE": 1, "MIXED": 1},
    "channel_stats": {"keyword": 3, "vector": 2, "vision": 2, "graph": 1},
}


# ── Handler functions ──────────────────────────────────────────

def run_mm_extract(pdf_file, max_pages):
    """Extract images and text blocks from PDF."""
    if pdf_file is None:
        return json.dumps(
            {"error": "No PDF uploaded. Using demo data instead.", "demo": True,
             "total_pages": 3, "total_images": 3, "total_text_blocks": 12},
            ensure_ascii=False, indent=2
        )

    try:
        from hugegraph_llm.operators.multimodal.pdf_image_extractor import (
            PDFImageExtractor,
        )
        extractor = PDFImageExtractor(max_image_size_kb=512, min_image_dim=50)
        result = extractor.extract(pdf_file, pages=None)

        summary = {
            "source_path": result.source_path,
            "total_pages": result.total_pages,
            "total_images": result.total_images,
            "total_text_blocks": result.total_text_blocks,
            "pages": [
                {
                    "page_num": p.page_num,
                    "image_count": p.image_count,
                    "text_block_count": p.text_block_count,
                }
                for p in result.pages
            ],
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except ImportError:
        return json.dumps(
            {"error": "PyMuPDF not available. Install: pip install PyMuPDF", "demo": True,
             "total_pages": 3, "total_images": 3, "total_text_blocks": 12},
            ensure_ascii=False, indent=2
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)


def run_vlm_describe(provider, max_images):
    """Generate VLM descriptions for extracted images."""
    try:
        from hugegraph_llm.operators.multimodal.vlm_descriptor import VLMDescriptor
        # Try to read cached extraction result
        cache_path = tempfile.gettempdir() + "/multimodal_extract_cache.json"
        import os
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                extract_data = json.load(f)
            if extract_data.get("pages"):
                from hugegraph_llm.operators.multimodal.pdf_image_extractor import ImageExtract
                all_images = []
                for p in extract_data["pages"]:
                    for img_dict in p.get("images", []):
                        img = ImageExtract(
                            image_id=img_dict["image_id"],
                            base64_data=img_dict["base64_data"],
                            bbox=img_dict.get("bbox", (0, 0, 100, 100)),
                            size=img_dict.get("size", (100, 100)),
                        )
                        all_images.append(img)

                if all_images:
                    descriptor = VLMDescriptor(provider=provider, batch_size=1, max_retries=1)
                    batch_result = descriptor.describe_extracted_images(
                        all_images[:max_images], text_blocks=[]
                    )
                    descriptions = []
                    for desc in batch_result.descriptions:
                        descriptions.append({
                            "image_id": desc.image_id,
                            "caption": desc.caption,
                            "detailed_description": desc.detailed_description,
                            "chart_type": desc.chart_type,
                            "key_insights": desc.key_insights,
                            "confidence": desc.confidence,
                        })
                    result = {
                        "total_images": batch_result.total_images,
                        "success_count": batch_result.success_count,
                        "descriptions": descriptions,
                        "demo": False,
                    }
                    return json.dumps(result, ensure_ascii=False, indent=2)

        # No cached data, use demo
        return json.dumps(
            {"total_images": len(DEMO_VLM_DESCRIPTIONS),
             "success_count": len(DEMO_VLM_DESCRIPTIONS),
             "descriptions": DEMO_VLM_DESCRIPTIONS, "demo": True},
            ensure_ascii=False, indent=2
        )
    except Exception as e:
        return json.dumps(
            {"error": str(e), "descriptions": DEMO_VLM_DESCRIPTIONS, "demo": True},
            ensure_ascii=False, indent=2
        )


def run_mm_kg_build(graph_name):
    """Build multimodal KG from extracted content."""
    try:
        from hugegraph_llm.operators.multimodal.multimodal_kg_builder import MultimodalKGBuilder
        builder = MultimodalKGBuilder(host=huge_settings.graph_url, graph=graph_name)
        builder.init_schema()
        # Use demo data for showcase
        stats = {
            "vertices": {
                "DocumentPage": 3,
                "Image": 3,
                "TextChunk": 12,
                "ImageDescription": 3,
            },
            "edges": {
                "contains_image": 3,
                "contains_text": 12,
                "describes": 3,
                "cross_modal_ref": 6,
            },
            "total_vertices": 21,
            "total_edges": 24,
        }
        return json.dumps({"stats": stats, "graph": graph_name, "demo": True},
                          ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "demo": True}, ensure_ascii=False, indent=2)


def run_mm_search(query, mode, top_k):
    """Multimodal search using four-channel RRF."""
    try:
        from hugegraph_llm.operators.multimodal.multimodal_retriever import MultiModalRetriever
        retriever = MultiModalRetriever(
            host=huge_settings.graph_url,
            graph="multimodal_poc",
            final_top_k=top_k,
            enable_vision_channel=True,
        )
        result = retriever.search(query, mode=mode)
        # If real search returns empty results, fall back to demo data
        if not result.results or len(result.results) == 0:
            demo_result = DEMO_SEARCH_RESULTS.copy()
            demo_result["query"] = query
            demo_result["mode"] = mode
            demo_result["demo"] = True
            demo_result["fallback_reason"] = "No results from live graph (graph may be empty)"
            return json.dumps(demo_result, ensure_ascii=False, indent=2)
        search_dict = {
            "query": result.query,
            "query_type": result.query_type,
            "results": [
                {
                    "id": r.id,
                    "label": r.label,
                    "score": round(r.score, 3),
                    "source_type": str(r.source_type),
                    "channel_scores": r.channel_scores,
                    "is_from_image": r.properties.get("is_from_image", False),
                }
                for r in result.results[:top_k]
            ],
            "source_distribution": result.source_distribution,
            "channel_stats": result.channel_stats,
            "text_context": result.text_context[:500] if result.text_context else "",
            "demo": False,
        }
        return json.dumps(search_dict, ensure_ascii=False, indent=2)
    except Exception as e:
        # Use demo data when HugeGraph not available
        demo_result = DEMO_SEARCH_RESULTS.copy()
        demo_result["query"] = query
        demo_result["mode"] = mode
        demo_result["demo"] = True
        demo_result["error_note"] = str(e)
        return json.dumps(demo_result, ensure_ascii=False, indent=2)


def show_demo_images():
    """Return demo image descriptions for gallery display."""
    descriptions = []
    for desc in DEMO_VLM_DESCRIPTIONS:
        descriptions.append(
            f"**{desc['caption']}**\n\n"
            f"Chart type: {desc['chart_type']}\n\n"
            f"Confidence: {desc['confidence']:.0%}\n\n"
            f"Key insights:\n"
            + "\n".join(f"  - {ins}" for ins in desc["key_insights"])
            + f"\n\nKeywords: {', '.join(desc['related_keywords'])}"
        )
    return descriptions


def show_demo_tables():
    """Return demo tables for display."""
    tables = []
    for tbl in DEMO_TABLES:
        tables.append(
            f"### {tbl['caption']}\n\n"
            f"**Name**: {tbl['name']}\n\n"
            f"**Dimensions**: {tbl['num_rows']} rows × {tbl['num_cols']} cols\n\n"
            + tbl["html"]
            + "\n\n**JSON preview**:\n"
            + json.dumps(tbl["json_data"][:2], ensure_ascii=False, indent=2)
        )
    return tables


def show_demo_equations():
    """Return demo equations for LaTeX display."""
    equations = []
    for eq in DEMO_EQUATIONS:
        equations.append(
            f"### {eq['name']}\n\n"
            f"$$ {eq['latex_block']} $$\n\n"
            f"**Description**: {eq['description']}\n\n"
            f"Inline form: `{eq['latex_inline']}`"
        )
    return equations


def show_search_comparison(query):
    """Show comparison between text-only and multimodal search."""
    text_only = {
        "mode": "text_only",
        "results": 1,
        "source_types": {"TEXT": 1},
        "insight": "Only found 1 text chunk mentioning warehouse",
    }
    multimodal = {
        "mode": "image_aware",
        "results": 3,
        "source_types": {"TEXT": 1, "IMAGE": 1, "MIXED": 1},
        "insight": "Found heatmap image + text chunk + risk table — 3x more context",
    }
    comparison = {
        "query": query,
        "text_only_search": text_only,
        "multimodal_search": multimodal,
        "gain": "+200% result coverage, +visual context (heatmap) + structured data (table)",
    }
    return json.dumps(comparison, ensure_ascii=False, indent=2)


# ── Gradio UI builder ──────────────────────────────────────────

def create_multimodal_block():
    """Create the Multimodal GraphRAG tab."""

    with gr.Row():
        # ── Left column: Input controls ──
        with gr.Column(scale=1):
            gr.Markdown("## Multimodal GraphRAG\n\nImage + Table + Equation extraction, VLM description, four-channel RRF retrieval")

            # Section 1: PDF Extraction
            gr.Markdown("---\n### 1. PDF Extraction")
            mm_pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
            mm_max_pages = gr.Number(value=5, label="Max Pages", precision=0)
            mm_extract_btn = gr.Button("Extract PDF Content", variant="primary")

            # Section 2: VLM Description
            gr.Markdown("---\n### 2. VLM Image Description")
            mm_vlm_provider = gr.Dropdown(
                choices=["xiaomimo", "openai", "demo"],
                value="demo",
                label="VLM Provider",
            )
            mm_vlm_max = gr.Number(value=10, label="Max Images", precision=0)
            mm_describe_btn = gr.Button("Describe Images", variant="primary")

            # Section 3: KG Build
            gr.Markdown("---\n### 3. Build Multimodal KG")
            mm_graph_name = gr.Textbox(value="multimodal_poc", label="Target Graph Name")
            mm_build_btn = gr.Button("Build Knowledge Graph", variant="primary")

            # Section 4: Search
            gr.Markdown("---\n### 4. Four-Channel RRF Search")
            mm_query = gr.Textbox(
                value="supply chain warehouse risk",
                label="Search Query",
            )
            mm_mode = gr.Dropdown(
                choices=["auto", "text_only", "image_aware"],
                value="image_aware",
                label="Search Mode",
            )
            mm_top_k = gr.Slider(value=5, minimum=1, maximum=20, step=1, label="Top-K Results")
            mm_search_btn = gr.Button("Search", variant="primary")

            # Section 5: Comparison
            gr.Markdown("---\n### 5. Search Mode Comparison")
            mm_cmp_query = gr.Textbox(
                value="warehouse risk heatmap",
                label="Comparison Query",
            )
            mm_cmp_btn = gr.Button("Compare text-only vs multimodal")

        # ── Right column: Results display ──
        with gr.Column(scale=2):
            # Tabbed results
            with gr.Tabs():
                with gr.Tab("Extraction"):
                    mm_extract_out = gr.Code(label="PDF Extraction Summary", language="json")

                with gr.Tab("VLM Descriptions"):
                    mm_describe_out = gr.Code(label="VLM Descriptions", language="json")
                    # Image description gallery
                    mm_img_gallery = gr.Dataframe(
                        headers=["Image ID", "Caption", "Chart Type", "Confidence", "Key Insight"],
                        label="Image Description Gallery",
                        datatype=["str", "str", "str", "number", "str"],
                        row_count=5,
                        col_count=5,
                    )

                with gr.Tab("Tables"):
                    mm_table_out = gr.HTML(label="Extracted Tables")
                    mm_table_json = gr.Code(label="Table JSON", language="json")

                with gr.Tab("Equations"):
                    mm_eq_out = gr.Markdown(label="Extracted Equations (LaTeX)")
                    mm_eq_json = gr.Code(label="Equation JSON", language="json")

                with gr.Tab("KG Stats"):
                    mm_kg_out = gr.Code(label="KG Build Stats", language="json")

                with gr.Tab("Search Results"):
                    mm_search_out = gr.Code(label="Search Results", language="json")
                    mm_search_table = gr.Dataframe(
                        headers=["ID", "Type", "Score", "Vision", "Keyword", "Vector", "Graph"],
                        label="Search Result Channel Scores",
                        datatype=["str", "str", "number", "number", "number", "number", "number"],
                        row_count=5,
                        col_count=7,
                    )

                with gr.Tab("Comparison"):
                    mm_cmp_out = gr.Code(label="text_only vs multimodal", language="json")

    # ── Event bindings ──

    def _extract_and_return(pdf_file, max_pages):
        result_json = run_mm_extract(pdf_file, max_pages)
        return result_json

    mm_extract_btn.click(
        fn=_extract_and_return,
        inputs=[mm_pdf, mm_max_pages],
        outputs=[mm_extract_out],
    )

    def _describe_and_format(provider, max_images):
        result_json = run_vlm_describe(provider, max_images)
        # Parse to populate gallery
        try:
            data = json.loads(result_json)
            descriptions = data.get("descriptions", [])
            gallery_rows = []
            for desc in descriptions:
                insight = desc.get("key_insights", [""])[0] if desc.get("key_insights") else ""
                gallery_rows.append([
                    desc.get("image_id", ""),
                    desc.get("caption", ""),
                    desc.get("chart_type", ""),
                    desc.get("confidence", 0),
                    insight,
                ])
            return result_json, gallery_rows
        except Exception:
            return result_json, []

    mm_describe_btn.click(
        fn=_describe_and_format,
        inputs=[mm_vlm_provider, mm_vlm_max],
        outputs=[mm_describe_out, mm_img_gallery],
    )

    def _build_and_return(graph_name):
        result_json = run_mm_kg_build(graph_name)
        return result_json

    mm_build_btn.click(
        fn=_build_and_return,
        inputs=[mm_graph_name],
        outputs=[mm_kg_out],
    )

    def _search_and_format(query, mode, top_k):
        result_json = run_mm_search(query, mode, top_k)
        try:
            data = json.loads(result_json)
            results = data.get("results", [])
            table_rows = []
            for r in results:
                cs = r.get("channel_scores", {})
                table_rows.append([
                    r.get("id", ""),
                    r.get("source_type", ""),
                    r.get("score", 0),
                    cs.get("vision", 0),
                    cs.get("keyword", 0),
                    cs.get("vector", 0),
                    cs.get("graph", 0),
                ])
            return result_json, table_rows
        except Exception:
            return result_json, []

    mm_search_btn.click(
        fn=_search_and_format,
        inputs=[mm_query, mm_mode, mm_top_k],
        outputs=[mm_search_out, mm_search_table],
    )

    mm_cmp_btn.click(
        fn=show_search_comparison,
        inputs=[mm_cmp_query],
        outputs=[mm_cmp_out],
    )

    # ── Auto-load demo data on page load ──

    def _load_demo_data():
        """Pre-populate all tabs with demo data for immediate showcasing."""
        # Demo extraction
        extract_demo = json.dumps(
            {"demo": True, "total_pages": 3, "total_images": 3,
             "total_text_blocks": 12, "source_path": "demo_supply_chain.pdf"},
            ensure_ascii=False, indent=2
        )

        # Demo VLM descriptions
        gallery_rows = []
        for desc in DEMO_VLM_DESCRIPTIONS:
            insight = desc.get("key_insights", [""])[0] if desc.get("key_insights") else ""
            gallery_rows.append([
                desc["image_id"],
                desc["caption"],
                desc["chart_type"],
                desc["confidence"],
                insight,
            ])
        describe_demo = json.dumps(
            {"demo": True, "total_images": 3, "success_count": 3,
             "descriptions": DEMO_VLM_DESCRIPTIONS},
            ensure_ascii=False, indent=2
        )

        # Demo tables
        table_html = "<h3>Extracted Tables</h3>"
        for tbl in DEMO_TABLES:
            table_html += f"<h4>{tbl['caption']}</h4>{tbl['html']}<br>"
        table_json = json.dumps(DEMO_TABLES, ensure_ascii=False, indent=2)

        # Demo equations
        eq_md = ""
        for eq in DEMO_EQUATIONS:
            eq_md += f"### {eq['name']}\n\n$$ {eq['latex_block']} $$\n\n{eq['description']}\n\n---\n\n"
        eq_json = json.dumps(DEMO_EQUATIONS, ensure_ascii=False, indent=2)

        # Demo KG stats
        kg_demo = json.dumps(
            {"demo": True, "stats": {
                "vertices": {"DocumentPage": 3, "Image": 3, "TextChunk": 12, "ImageDescription": 3},
                "edges": {"contains_image": 3, "contains_text": 12, "describes": 3, "cross_modal_ref": 6},
                "total_vertices": 21, "total_edges": 24},
             "graph": "multimodal_poc"},
            ensure_ascii=False, indent=2
        )

        # Demo search
        search_demo = json.dumps(DEMO_SEARCH_RESULTS, ensure_ascii=False, indent=2)
        search_table = []
        for r in DEMO_SEARCH_RESULTS["results"]:
            cs = r.get("channel_scores", {})
            search_table.append([
                r["id"], r["source_type"], r["score"],
                cs.get("vision", 0), cs.get("keyword", 0),
                cs.get("vector", 0), cs.get("graph", 0),
            ])

        # Demo comparison
        cmp_demo = show_search_comparison("supply chain warehouse risk")

        return (
            extract_demo,
            describe_demo, gallery_rows,
            table_html, table_json,
            eq_md, eq_json,
            kg_demo,
            search_demo, search_table,
            cmp_demo,
        )

    # Return demo data targets for page load
    demo_outputs = [
        mm_extract_out,
        mm_describe_out, mm_img_gallery,
        mm_table_out, mm_table_json,
        mm_eq_out, mm_eq_json,
        mm_kg_out,
        mm_search_out, mm_search_table,
        mm_cmp_out,
    ]

    return demo_outputs, _load_demo_data
