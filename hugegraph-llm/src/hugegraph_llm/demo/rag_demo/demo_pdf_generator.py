"""
Demo PDF generator for Multimodal GraphRAG walkthrough.

Creates a 3-page "Supply Chain Risk Assessment Report" PDF containing:
  - Page 1: Title + Executive Summary + Risk heatmap image (matplotlib)
  - Page 2: Detailed analysis + Data table (reportlab) + Risk score formula
  - Page 3: Recommendations + Supply chain network diagram + Cumulative formula

This PDF exercises ALL 18 multimodal operators:
  A. Document Parsing   — pdf_image_extractor extracts images/text
  B. VLM Description    — vlm_descriptor describes heatmap + network diagram
  C. MM Analysis        — multimodal_analyzer analyzes image/table/equation
  D. KG Build           — multimodal_entity_injector + multimodal_kg_builder
  E. Retrieval          — multimodal_retriever 4-channel search
  F. Formula & Sidecar  — omml_to_latex + sidecar IR pipeline
"""

import io
import math
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Chinese font setup ──────────────────────────────────────────

# Try common CJK font paths (macOS)
_CJK_FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

_CJK_FONT_NAME = None

def _register_cjk_font():
    global _CJK_FONT_NAME
    for path in _CJK_FONT_PATHS:
        if os.path.exists(path):
            try:
                name = "CJKMain"
                pdfmetrics.registerFont(TTFont(name, path))
                _CJK_FONT_NAME = name
                return
            except Exception:
                continue
    # Fallback: no CJK font, use Helvetica (English only)
    _CJK_FONT_NAME = "Helvetica"


def _generate_risk_heatmap() -> bytes:
    """Generate a supply chain risk heatmap PNG (bytes)."""
    nodes = [
        "Warehouse-A", "Warehouse-B", "Warehouse-C",
        "Supplier-X", "Supplier-Y", "Transport-Z",
    ]
    risk_dims = ["Congestion", "Delay", "Disruption", "Quality", "Cost"]
    # Risk scores matrix (higher = worse)
    data = np.array([
        [0.82, 0.45, 0.67, 0.33, 0.71],  # Warehouse-A
        [0.55, 0.78, 0.42, 0.61, 0.38],  # Warehouse-B
        [0.91, 0.63, 0.85, 0.47, 0.92],  # Warehouse-C (highest risk)
        [0.37, 0.89, 0.31, 0.72, 0.44],  # Supplier-X
        [0.48, 0.56, 0.93, 0.29, 0.65],  # Supplier-Y
        [0.73, 0.41, 0.58, 0.85, 0.49],  # Transport-Z
    ])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(risk_dims)))
    ax.set_xticklabels(risk_dims, fontsize=9)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=9)
    ax.set_title("Supply Chain Risk Heatmap", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(nodes)):
        for j in range(len(risk_dims)):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Risk Score", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _generate_network_diagram() -> bytes:
    """Generate a supply chain network topology PNG (bytes)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Node positions (layered layout)
    positions = {
        "Supplier-X": (1, 4),
        "Supplier-Y": (1, 2),
        "Warehouse-A": (3, 5),
        "Warehouse-B": (3, 3),
        "Warehouse-C": (3, 1),
        "Transport-Z": (5, 3),
        "Customer": (7, 3),
    }
    # Risk colors (red = high, green = low)
    risk_scores = {
        "Supplier-X": 0.55, "Supplier-Y": 0.52,
        "Warehouse-A": 0.64, "Warehouse-B": 0.55, "Warehouse-C": 0.86,
        "Transport-Z": 0.51, "Customer": 0.20,
    }
    edges = [
        ("Supplier-X", "Warehouse-A"),
        ("Supplier-X", "Warehouse-B"),
        ("Supplier-Y", "Warehouse-B"),
        ("Supplier-Y", "Warehouse-C"),
        ("Warehouse-A", "Transport-Z"),
        ("Warehouse-B", "Transport-Z"),
        ("Warehouse-C", "Transport-Z"),
        ("Transport-Z", "Customer"),
    ]

    for node, (x, y) in positions.items():
        risk = risk_scores[node]
        r = min(risk * 2, 1.0)
        node_color = (r, 0.2, 1 - r, 0.7)  # red-green gradient
        ax.scatter(x, y, s=300, c=[node_color], edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(node, (x, y), fontsize=8, ha="center", va="bottom",
                     xytext=(0, 15), textcoords="offset points", fontweight="bold")

    for src, dst in edges:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        # Edge risk = avg of endpoints
        avg_risk = (risk_scores[src] + risk_scores[dst]) / 2
        edge_color = (min(avg_risk * 1.5, 1.0), 0.3, 1 - min(avg_risk * 1.5, 1.0), 0.5)
        ax.plot([sx, dx], [sy, dy], color=edge_color, linewidth=2, zorder=3)

    ax.set_title("Supply Chain Network & Risk Topology", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_demo_pdf(output_path: str = None) -> str:
    """Generate the 3-page Supply Chain Risk Assessment Report PDF.

    Args:
        output_path: Where to save the PDF. If None, saves to a temp file.

    Returns:
        The absolute path to the generated PDF file.
    """
    _register_cjk_font()

    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "supply_chain_risk_report.pdf")

    # Generate images first
    heatmap_png = _generate_risk_heatmap()
    network_png = _generate_network_diagram()

    # Save images to temp files for reportlab
    heatmap_path = os.path.join(tempfile.gettempdir(), "_heatmap.png")
    network_path = os.path.join(tempfile.gettempdir(), "_network.png")
    with open(heatmap_path, "wb") as f:
        f.write(heatmap_png)
    with open(network_path, "wb") as f:
        f.write(network_png)

    # Build PDF
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    font_name = _CJK_FONT_NAME or "Helvetica"
    title_style = ParagraphStyle(
        "DemoTitle", parent=styles["Title"], fontName=font_name, fontSize=18,
    )
    heading_style = ParagraphStyle(
        "DemoHeading", parent=styles["Heading2"], fontName=font_name, fontSize=14,
    )
    body_style = ParagraphStyle(
        "DemoBody", parent=styles["Normal"], fontName=font_name, fontSize=10, leading=14,
    )
    formula_style = ParagraphStyle(
        "DemoFormula", parent=styles["Normal"], fontName="Courier", fontSize=11,
        textColor=colors.darkblue, backColor=colors.Color(0.95, 0.95, 1.0),
        leftIndent=20, rightIndent=20, spaceBefore=10, spaceAfter=10,
    )

    story = []

    # ═══ Page 1: Title + Executive Summary + Heatmap ═══
    story.append(Paragraph("Supply Chain Risk Assessment Report", title_style))
    story.append(Paragraph("供应链风控评估报告", heading_style))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "<b>Executive Summary / 执行摘要</b>: This report evaluates warehouse congestion, "
        "supplier delays, and transport disruption risks across 6 supply chain nodes. "
        "Warehouse-C exhibits the highest composite risk score (0.86), driven by severe "
        "disruption vulnerability (0.85) and cost overrun probability (0.92). "
        "The risk heatmap below visualizes the multidimensional risk distribution.",
        body_style,
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("<b>Figure 1: Risk Heatmap</b> — Node-level risk scores across 5 dimensions.", body_style))
    story.append(RLImage(heatmap_path, width=160 * mm, height=100 * mm))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "Key observation: Warehouse-C shows critical risk levels across congestion (0.91), "
        "disruption (0.85), and cost (0.92). Immediate mitigation recommended.",
        body_style,
    ))
    story.append(Paragraph(
        "Supplier-Y faces the highest disruption risk (0.93) due to geographic concentration. "
        "Transport-Z has quality concerns (0.85) requiring inspection protocol updates.",
        body_style,
    ))

    # ═══ Page 2: Detailed Analysis + Table + Formula ═══
    story.append(PageBreak())
    story.append(Paragraph("Detailed Risk Analysis / 详细风险分析", heading_style))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Table 1 below presents the composite risk scores for each supply chain node. "
        "The overall risk score is computed using the weighted formula shown after the table.",
        body_style,
    ))
    story.append(Spacer(1, 3 * mm))

    # Risk data table
    table_data = [
        ["Node", "Congestion", "Delay", "Disruption", "Quality", "Cost", "Composite R"],
        ["Warehouse-A", "0.82", "0.45", "0.67", "0.33", "0.71", "0.64"],
        ["Warehouse-B", "0.55", "0.78", "0.42", "0.61", "0.38", "0.55"],
        ["Warehouse-C", "0.91", "0.63", "0.85", "0.47", "0.92", "0.86"],
        ["Supplier-X", "0.37", "0.89", "0.31", "0.72", "0.44", "0.55"],
        ["Supplier-Y", "0.48", "0.56", "0.93", "0.29", "0.65", "0.52"],
        ["Transport-Z", "0.73", "0.41", "0.58", "0.85", "0.49", "0.51"],
    ]
    t = Table(table_data, colWidths=[70, 55, 50, 60, 55, 50, 70])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.6)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), font_name),
        ("FONTNAME", (0, 1), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        # Highlight highest risk row
        ("BACKGROUND", (0, 3), (-1, 3), colors.Color(1.0, 0.85, 0.85)),
    ]))
    story.append(Paragraph("<b>Table 1: Supply Chain Node Risk Assessment</b>", body_style))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    # Risk score formula
    story.append(Paragraph("<b>Equation 1: Composite Risk Score Formula</b>", body_style))
    story.append(Paragraph(
        "R_score = sum(w_i * d_i / tau_i) + lambda * sigma^2",
        formula_style,
    ))
    story.append(Paragraph(
        "Where: w_i = dimension weight, d_i = observed delay, tau_i = tolerance threshold, "
        "lambda = volatility coefficient, sigma^2 = variance of historical disruptions.",
        body_style,
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Warehouse-C achieves the highest R_score = 0.86 because its disruption "
        "vulnerability (d_i/tau_i ratio = 0.85/0.50 = 1.70) far exceeds the tolerance "
        "threshold, combined with high cost variance (sigma^2 = 0.82).",
        body_style,
    ))

    # ═══ Page 3: Recommendations + Network Diagram + Second Formula ═══
    story.append(PageBreak())
    story.append(Paragraph("Recommendations & Network Topology / 建议与网络拓扑", heading_style))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Based on the risk assessment, we recommend the following mitigation strategies:",
        body_style,
    ))
    story.append(Paragraph(
        "1. <b>Warehouse-C</b>: Implement dual-supplier redundancy to reduce disruption "
        "dependency from 0.85 to target 0.45. Expected cost: 2.3M, ROI: 18 months.",
        body_style,
    ))
    story.append(Paragraph(
        "2. <b>Supplier-Y</b>: Diversify geographic sourcing (add 2 backup suppliers "
        "in different regions) to lower disruption probability from 0.93 to 0.55.",
        body_style,
    ))
    story.append(Paragraph(
        "3. <b>Transport-Z</b>: Deploy automated quality inspection gates at 3 critical "
        "junctions to address the 0.85 quality risk score.",
        body_style,
    ))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("<b>Figure 2: Supply Chain Network Topology</b>", body_style))
    story.append(RLImage(network_path, width=150 * mm, height=110 * mm))
    story.append(Spacer(1, 5 * mm))

    # Cumulative risk propagation formula
    story.append(Paragraph("<b>Equation 2: Cascading Risk Propagation Model</b>", body_style))
    story.append(Paragraph(
        "P_cascade = P_source * prod_{k=1}^{n-1} (1 - R_k) * (1 + alpha * C_k)",
        formula_style,
    ))
    story.append(Paragraph(
        "Where: P_source = initial disruption probability, R_k = resilience factor at node k, "
        "C_k = coupling strength between nodes k and k+1, alpha = amplification coefficient.",
        body_style,
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The network topology reveals that Warehouse-C is a critical cascade amplifier: "
        "it receives inputs from Supplier-Y (disruption=0.93) and feeds Transport-Z "
        "(quality=0.85), creating a 3-node cascade chain with P_cascade = 0.93 * "
        "(1-0.55)*(1+0.3*0.68) * (1-0.51)*(1+0.3*0.85) = 0.31, which is 3x the "
        "acceptable threshold of 0.10.",
        body_style,
    ))

    # Build PDF
    doc.build(story)

    # Clean up temp images
    os.unlink(heatmap_path)
    os.unlink(network_path)

    return output_path


# ── Pre-generated PDF path for Gradio demo ────────────────────────

DEMO_PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "assets", "demo_supply_chain_risk_report.pdf"
)


def ensure_demo_pdf() -> str:
    """Generate the demo PDF if it doesn't exist, return its path."""
    # Try project assets dir first
    project_root = Path(__file__).parent.parent.parent.parent.parent
    assets_dir = project_root / "assets"
    pdf_path = assets_dir / "demo_supply_chain_risk_report.pdf"

    if pdf_path.exists():
        return str(pdf_path)

    # Create assets dir if needed
    assets_dir.mkdir(parents=True, exist_ok=True)
    return generate_demo_pdf(str(pdf_path))


# ── Demo questions for walkthrough ─────────────────────────────────

DEMO_QUESTIONS = [
    {
        "id": "Q1",
        "question": "哪些节点风险评分最高？What nodes have the highest risk scores?",
        "target_modality": "text",
        "expected_hit": "Warehouse-C (R=0.86), Warehouse-A (R=0.64)",
        "activated_ops": [
            "pdf_image_extractor", "unified_document_parser",
            "multimodal_entity_injector", "multimodal_kg_builder",
            "multimodal_retriever", "multimodal_retrieval_channel",
        ],
    },
    {
        "id": "Q2",
        "question": "热力图展示了什么风险分布？What risk distribution does the heatmap show?",
        "target_modality": "image",
        "expected_hit": "Warehouse-C highest (congestion 0.91, cost 0.92)",
        "activated_ops": [
            "pdf_image_extractor", "vlm_descriptor", "vlm_provider_registry",
            "async_vlm_pipeline", "image_dimension_validator",
            "multimodal_analyzer", "surrounding_context",
            "multimodal_retriever", "multimodal_retrieval_channel",
        ],
    },
    {
        "id": "Q3",
        "question": "风险评分公式R_score怎么计算？How is the R_score formula computed?",
        "target_modality": "equation",
        "expected_hit": "R_score = sum(w_i*d_i/tau_i) + lambda*sigma^2",
        "activated_ops": [
            "omml_to_latex", "sidecar_placeholder", "sidecar_ir",
            "sidecar_writer", "sidecar_backfill",
            "chunk_schema",
            "multimodal_retriever", "multimodal_retrieval_channel",
        ],
    },
    {
        "id": "Q4",
        "question": "仓库拥堵和供应商延迟有什么关联？How is warehouse congestion linked to supplier delay?",
        "target_modality": "graph",
        "expected_hit": "Supplier-Y → Warehouse-C (disruption 0.93 → congestion 0.91)",
        "activated_ops": [
            "multimodal_entity_injector", "multimodal_kg_builder",
            "multimodal_retriever", "multimodal_retrieval_channel",
        ],
    },
    {
        "id": "Q5",
        "question": "Warehouse-C的完整风险评估是什么？What is the full risk assessment for Warehouse-C?",
        "target_modality": "mixed",
        "expected_hit": "Text: R=0.86, highest node. Image: heatmap shows red cluster. Formula: d_i/tau_i=1.70",
        "activated_ops": [
            # ALL 18 operators activated for mixed query
            "pdf_image_extractor", "unified_document_parser",
            "vlm_descriptor", "vlm_provider_registry",
            "async_vlm_pipeline", "image_dimension_validator",
            "multimodal_analyzer", "surrounding_context", "chunk_schema",
            "multimodal_entity_injector", "multimodal_kg_builder",
            "multimodal_retriever", "multimodal_retrieval_channel",
            "omml_to_latex", "sidecar_placeholder", "sidecar_ir",
            "sidecar_writer", "sidecar_backfill",
        ],
    },
]
