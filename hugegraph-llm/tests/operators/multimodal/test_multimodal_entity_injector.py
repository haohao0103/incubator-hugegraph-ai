"""Tests for multimodal_entity_injector.py — LightRAG-style multimodal entity injection."""

import time
from unittest.mock import patch

import pytest

from hugegraph_llm.operators.multimodal.multimodal_entity_injector import (
    CHART_TYPE_ENUM,
    CHART_TYPE_FALLBACK,
    MULTIMODAL_TYPE_ENUM,
    MULTIMODAL_TYPE_FALLBACK,
    MultimodalAssociationSpec,
    MultimodalEntityInjector,
    MultimodalEntitySpec,
    build_association_description,
    classify_multimodal_type,
    inject_multimodal_entities,
    parse_mm_display_name,
)


# ---------------------------------------------------------------------------
# parse_mm_display_name
# ---------------------------------------------------------------------------

class TestParseMMDisplayName:
    def test_image_name_tag(self):
        content = "[Image Name]crispr_cas9_workflow\n[Image Type]Flowchart\n..."
        assert parse_mm_display_name(content, "fallback") == "crispr_cas9_workflow"

    def test_table_name_tag(self):
        content = "[Table Name]q4_revenue_by_region\n..."
        assert parse_mm_display_name(content, "fallback") == "q4_revenue_by_region"

    def test_equation_name_tag(self):
        content = "[Equation Name]bayes_theorem_posterior\n..."
        assert parse_mm_display_name(content, "fallback") == "bayes_theorem_posterior"

    def test_no_tag_returns_fallback(self):
        content = "Just plain text without any multimodal name tag."
        assert parse_mm_display_name(content, "img_0_0") == "img_0_0"

    def test_empty_content_returns_fallback(self):
        assert parse_mm_display_name("", "fallback_id") == "fallback_id"

    def test_empty_name_after_tag_returns_fallback(self):
        content = "[Image Name]\n..."
        assert parse_mm_display_name(content, "fallback") == "fallback"

    def test_multiline_content_first_match(self):
        content = "[Image Name]first_match\nSome text\n[Table Name]second_match\n..."
        # Only the first match is used
        assert parse_mm_display_name(content, "fb") == "first_match"


# ---------------------------------------------------------------------------
# classify_multimodal_type
# ---------------------------------------------------------------------------

class TestClassifyMultimodalType:
    def test_bar_maps_to_chart(self):
        assert classify_multimodal_type(vlm_chart_type="bar") == "Chart"

    def test_line_maps_to_chart(self):
        assert classify_multimodal_type(vlm_chart_type="line") == "Chart"

    def test_pie_maps_to_chart(self):
        assert classify_multimodal_type(vlm_chart_type="pie") == "Chart"

    def test_scatter_maps_to_chart(self):
        assert classify_multimodal_type(vlm_chart_type="scatter") == "Chart"

    def test_table_maps_to_table(self):
        assert classify_multimodal_type(vlm_chart_type="table") == "Table"

    def test_flowchart_maps_to_flowchart(self):
        assert classify_multimodal_type(vlm_chart_type="flowchart") == "Flowchart"

    def test_photo_maps_to_photo(self):
        assert classify_multimodal_type(vlm_chart_type="photo") == "Photo"

    def test_screenshot_maps_to_screenshot(self):
        assert classify_multimodal_type(vlm_chart_type="screenshot") == "Screenshot"

    def test_other_maps_to_other(self):
        assert classify_multimodal_type(vlm_chart_type="other") == "Other"

    def test_empty_chart_type_returns_fallback(self):
        assert classify_multimodal_type(vlm_chart_type="") == MULTIMODAL_TYPE_FALLBACK

    def test_unknown_chart_type_returns_other(self):
        assert classify_multimodal_type(vlm_chart_type="unknown_type") == "Other"


# ---------------------------------------------------------------------------
# build_association_description
# ---------------------------------------------------------------------------

class TestBuildAssociationDescription:
    def test_with_heading(self):
        desc = build_association_description(
            tgt_entity="Alice",
            mm_type="drawing",
            mm_display_name="crispr_workflow",
            heading_label="Introduction",
            file_path="paper.pdf",
        )
        assert "Alice" in desc
        assert "drawing" in desc
        assert "crispr_workflow" in desc
        assert "in section Introduction of document" in desc
        assert "paper.pdf" in desc

    def test_without_heading(self):
        desc = build_association_description(
            tgt_entity="Revenue",
            mm_type="table",
            mm_display_name="q4_results",
            heading_label="",
            file_path="report.pdf",
        )
        assert "of document" in desc
        assert "in section" not in desc

    def test_empty_file_path(self):
        desc = build_association_description(
            tgt_entity="EntityA",
            mm_type="equation",
            mm_display_name="euler_identity",
            heading_label="",
            file_path="",
        )
        assert desc.endswith("\"\"")


# ---------------------------------------------------------------------------
# MultimodalEntityInjector.run()
# ---------------------------------------------------------------------------

class TestMultimodalEntityInjector:
    def test_empty_multimodal_items(self):
        injector = MultimodalEntityInjector()
        context = {"vertices": [], "edges": [], "multimodal_items": [], "existing_entities": []}
        result = injector.run(context)
        assert result["vertices"] == []
        assert result["edges"] == []
        assert result.get("multimodal_entities", []) == []

    def test_no_multimodal_items_key(self):
        injector = MultimodalEntityInjector()
        context = {"vertices": [], "edges": []}
        result = injector.run(context)
        assert result["vertices"] == []
        assert result["edges"] == []

    def test_single_drawing_with_one_entity(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {
                    "type": "drawing",
                    "id": "img_0_0",
                    "content": "[Image Name]bar_chart\n[Image Type]Chart\nA revenue chart.",
                    "source_id": "chunk_0",
                    "file_path": "report.pdf",
                    "heading": "Results",
                    "vlm_chart_type": "bar",
                }
            ],
            "existing_entities": ["Revenue", "Company"],
        }
        result = injector.run(context)

        # Should have 1 multimodal vertex + 2 association edges
        assert len(result["vertices"]) == 1
        assert len(result["edges"]) == 2

        # Vertex check
        vertex = result["vertices"][0]
        assert vertex["entity_name"] == "img_0_0"
        assert vertex["entity_type"] == "drawing"
        assert "bar_chart" in vertex["description"]

        # Edge check
        edge_targets = [e["tgt_id"] for e in result["edges"]]
        assert "Revenue" in edge_targets
        assert "Company" in edge_targets
        assert "img_0_0" not in edge_targets  # 不与自身关联

    def test_table_with_no_existing_entities(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {
                    "type": "table",
                    "id": "tbl_1_0",
                    "content": "[Table Name]financial_summary\n...",
                    "source_id": "chunk_1",
                    "file_path": "finance.pdf",
                    "heading": "",
                    "vlm_chart_type": "table",
                }
            ],
            "existing_entities": [],
        }
        result = injector.run(context)

        # Vertex only, no edges (no existing entities to associate with)
        assert len(result["vertices"]) == 1
        assert len(result["edges"]) == 0

    def test_equation_with_entities(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {
                    "type": "equation",
                    "id": "eq_2_0",
                    "content": "[Equation Name]bayes_theorem\nE = mc^2",
                    "source_id": "chunk_2",
                    "file_path": "physics.pdf",
                    "heading": "Theory",
                    "vlm_chart_type": "",
                }
            ],
            "existing_entities": ["Energy", "Mass"],
        }
        result = injector.run(context)

        assert len(result["vertices"]) == 1
        assert len(result["edges"]) == 2

        # Verify association description contains heading
        for edge in result["edges"]:
            if edge["tgt_id"] == "Energy":
                assert "in section Theory of document" in edge["description"]

    def test_skip_item_with_no_id(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {"type": "drawing", "id": "", "content": "..."},
                {"type": "drawing", "id": "img_valid", "content": "..."},
            ],
            "existing_entities": ["EntityA"],
        }
        result = injector.run(context)
        # Only 1 vertex (the valid item), edges only from valid item
        assert len(result["vertices"]) == 1
        assert len(result["edges"]) == 1

    def test_multiple_items_multiple_entities(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {"type": "drawing", "id": "img_0", "content": "...", "source_id": "c0", "file_path": "f.pdf", "heading": "S1", "vlm_chart_type": "bar"},
                {"type": "table", "id": "tbl_0", "content": "...", "source_id": "c0", "file_path": "f.pdf", "heading": "S1", "vlm_chart_type": "table"},
                {"type": "equation", "id": "eq_0", "content": "...", "source_id": "c0", "file_path": "f.pdf", "heading": "", "vlm_chart_type": ""},
            ],
            "existing_entities": ["A", "B"],
        }
        result = injector.run(context)

        # 3 vertices + 6 edges (2 entities * 3 items)
        assert len(result["vertices"]) == 3
        assert len(result["edges"]) == 6

    def test_context_initialized_if_missing(self):
        injector = MultimodalEntityInjector()
        context = {"multimodal_items": [{"type": "drawing", "id": "img_0", "content": "..."}]}
        result = injector.run(context)
        assert "vertices" in result
        assert "edges" in result
        assert isinstance(result["vertices"], list)
        assert isinstance(result["edges"], list)

    def test_result_contains_mm_entities_and_associations(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {"type": "drawing", "id": "img_0", "content": "...", "source_id": "c0", "file_path": "f.pdf", "heading": "", "vlm_chart_type": ""},
            ],
            "existing_entities": ["EntityA"],
        }
        result = injector.run(context)
        assert "multimodal_entities" in result
        assert "multimodal_associations" in result
        assert isinstance(result["multimodal_entities"], list)
        assert isinstance(result["multimodal_associations"], list)
        assert len(result["multimodal_entities"]) == 1
        assert len(result["multimodal_associations"]) == 1

    def test_entity_spec_fields(self):
        injector = MultimodalEntityInjector()
        context = {
            "multimodal_items": [
                {
                    "type": "drawing", "id": "img_0",
                    "content": "[Image Name]test_name\n...",
                    "source_id": "c0", "file_path": "f.pdf",
                    "heading": "", "vlm_chart_type": "bar",
                },
            ],
            "existing_entities": [],
        }
        result = injector.run(context)
        spec = result["multimodal_entities"][0]
        assert isinstance(spec, MultimodalEntitySpec)
        assert spec.entity_name == "img_0"
        assert spec.display_name == "test_name"
        assert spec.entity_type == "Chart"  # "bar" → "Chart"

    def test_timestamp_is_current_time(self):
        fixed_time = 1700000000
        with patch("hugegraph_llm.operators.multimodal.multimodal_entity_injector.time") as mock_time:
            mock_time.time.return_value = fixed_time
            mock_time.int = int  # keep int() working

            injector = MultimodalEntityInjector()
            context = {
                "multimodal_items": [
                    {"type": "drawing", "id": "img_0", "content": "...", "source_id": "c0"},
                ],
                "existing_entities": [],
            }
            result = injector.run(context)
            assert result["vertices"][0]["timestamp"] == fixed_time


# ---------------------------------------------------------------------------
# inject_multimodal_entities convenience function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    def test_convenience_calls_injector_run(self):
        context = {
            "multimodal_items": [
                {"type": "drawing", "id": "img_0", "content": "...", "source_id": "c0"},
            ],
            "existing_entities": ["E1"],
        }
        result = inject_multimodal_entities(context)
        assert len(result["vertices"]) == 1
        assert len(result["edges"]) == 1


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    def test_multimodal_type_enum_contains_expected_values(self):
        expected = ["Photo", "Chart", "Table", "Flowchart", "Other"]
        for val in expected:
            assert val in MULTIMODAL_TYPE_ENUM

    def test_chart_type_enum_contains_expected_values(self):
        expected = ["bar", "line", "pie", "scatter", "table", "flowchart", "other"]
        for val in expected:
            assert val in CHART_TYPE_ENUM

    def test_fallback_values(self):
        assert MULTIMODAL_TYPE_FALLBACK == "Other"
        assert CHART_TYPE_FALLBACK == "other"
