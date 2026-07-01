# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.

"""Tests for MultimodalAnalyzer and SurroundingContextEnricher operators."""

import json
import pytest

from hugegraph_llm.operators.multimodal.multimodal_analyzer import (
    MultimodalAnalyzer,
    MultimodalAnalyzerConfig,
    MULTIMODAL_PROMPTS,
    IMAGE_TYPE_ENUM,
    IMAGE_TYPE_FALLBACK,
    table_content_format_label,
    _modality_to_prompt_key,
    _parse_json_response,
)
from hugegraph_llm.operators.multimodal.surrounding_context import (
    SurroundingContextEnricher,
    build_surrounding,
    _find_target_span,
    _remove_table_tags,
    _strip_internal_markers,
    _atomize,
    _estimate_tokens,
    DEFAULT_SURROUNDING_MAX_TOKENS,
)


# ============================================================================
# MultimodalAnalyzer Tests
# ============================================================================

class TestMultimodalAnalyzer:
    def test_prompt_keys_exist(self):
        assert "image_analysis" in MULTIMODAL_PROMPTS
        assert "table_analysis" in MULTIMODAL_PROMPTS
        assert "equation_analysis" in MULTIMODAL_PROMPTS

    def test_image_type_enum(self):
        assert "Photo" in IMAGE_TYPE_ENUM
        assert "Chart" in IMAGE_TYPE_ENUM
        assert "Other" in IMAGE_TYPE_ENUM
        assert IMAGE_TYPE_FALLBACK == "Other"

    def test_table_format_label_json(self):
        label = table_content_format_label("json")
        assert "JSON" in label
        assert "2-D array" in label

    def test_table_format_label_html(self):
        label = table_content_format_label("html")
        assert "HTML" in label
        assert "rowspan" in label

    def test_table_format_label_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown table format"):
            table_content_format_label("csv")

    def test_modality_to_prompt_key(self):
        assert _modality_to_prompt_key("drawings") == "image_analysis"
        assert _modality_to_prompt_key("tables") == "table_analysis"
        assert _modality_to_prompt_key("equations") == "equation_analysis"
        assert _modality_to_prompt_key("other") == ""

    def test_parse_json_response_direct(self):
        result = _parse_json_response('{"name": "test", "type": "Photo"}')
        assert result["name"] == "test"
        assert result["type"] == "Photo"

    def test_parse_json_response_markdown_wrapped(self):
        response = '```json\n{"name": "test", "description": "desc"}\n```'
        result = _parse_json_response(response)
        assert result["name"] == "test"

    def test_parse_json_response_brace_extract(self):
        response = 'Here is the result: {"name": "test"} end'
        result = _parse_json_response(response)
        assert result["name"] == "test"

    def test_parse_json_response_invalid_returns_none(self):
        result = _parse_json_response("not json at all")
        assert result is None

    def test_analyzer_run_with_llm(self):
        """Test run() with mock LLM function."""
        mock_response = json.dumps({
            "name": "revenue_chart",
            "type": "Chart",
            "description": "A bar chart showing quarterly revenue growth.",
        })

        def mock_llm(prompt: str) -> str:
            return mock_response

        analyzer = MultimodalAnalyzer(llm_func=mock_llm)
        context = {
            "multimodal_sidecars": {
                "drawings": {
                    "im-hash-0001": {
                        "id": "im-hash-0001",
                        "blockid": "block1",
                        "content": "",
                        "caption": "Figure 1",
                        "format": "png",
                    },
                },
                "tables": {},
                "equations": {},
            },
            "blocks_content_by_id": {},
        }

        result = analyzer.run(context)
        assert "multimodal_analysis_results" in result
        assert "drawings" in result["multimodal_analysis_results"]
        item = result["multimodal_analysis_results"]["drawings"]["im-hash-0001"]
        assert "llm_analyze_result" in item
        assert item["llm_analyze_result"]["name"] == "revenue_chart"
        assert item["llm_analyze_result"]["type"] == "Chart"

    def test_analyzer_run_table_analysis(self):
        """Test table analysis with mock LLM."""
        mock_response = json.dumps({
            "name": "quarterly_revenue",
            "description": "Revenue by quarter showing growth trend.",
        })

        def mock_llm(prompt: str) -> str:
            return mock_response

        analyzer = MultimodalAnalyzer(llm_func=mock_llm)
        context = {
            "multimodal_sidecars": {
                "drawings": {},
                "tables": {
                    "tb-hash-0001": {
                        "id": "tb-hash-0001",
                        "blockid": "block1",
                        "content": '[["Q1", "$1.2M"], ["Q2", "$2.3M"]]',
                        "caption": "Revenue table",
                        "format": "json",
                    },
                },
                "equations": {},
            },
            "blocks_content_by_id": {},
        }

        result = analyzer.run(context)
        tables = result["multimodal_analysis_results"]["tables"]
        assert "tb-hash-0001" in tables
        assert tables["tb-hash-0001"]["llm_analyze_result"]["name"] == "quarterly_revenue"

    def test_analyzer_run_equation_analysis(self):
        """Test equation analysis with mock LLM."""
        mock_response = json.dumps({
            "name": "bayes_theorem_posterior",
            "equation": "P(A|B) = \\frac{P(B|A)P(A)}{P(B)}",
            "description": "Bayes' theorem for posterior probability.",
        })

        def mock_llm(prompt: str) -> str:
            return mock_response

        analyzer = MultimodalAnalyzer(llm_func=mock_llm)
        context = {
            "multimodal_sidecars": {
                "drawings": {},
                "tables": {},
                "equations": {
                    "eq-hash-0001": {
                        "id": "eq-hash-0001",
                        "blockid": "block1",
                        "content": "P(A|B) = P(B|A)P(A)/P(B)",
                        "caption": "Bayes' theorem",
                        "format": "latex",
                    },
                },
            },
            "blocks_content_by_id": {},
        }

        result = analyzer.run(context)
        eqs = result["multimodal_analysis_results"]["equations"]
        assert "eq-hash-0001" in eqs
        assert "bayes_theorem_posterior" in eqs["eq-hash-0001"]["llm_analyze_result"]["name"]

    def test_analyzer_image_type_fallback(self):
        """Invalid image type should fallback to 'Other'."""
        mock_response = json.dumps({
            "name": "test_img",
            "type": "InvalidType",
            "description": "A test image.",
        })

        def mock_llm(prompt: str) -> str:
            return mock_response

        analyzer = MultimodalAnalyzer(llm_func=mock_llm)
        context = {
            "multimodal_sidecars": {
                "drawings": {
                    "im-0001": {"id": "im-0001", "blockid": "b1", "content": "", "caption": ""},
                },
                "tables": {},
                "equations": {},
            },
            "blocks_content_by_id": {},
        }

        result = analyzer.run(context)
        item = result["multimodal_analysis_results"]["drawings"]["im-0001"]
        assert item["llm_analyze_result"]["type"] == "Other"

    def test_analyzer_config_enabled_modalities(self):
        config = MultimodalAnalyzerConfig(
            language="Chinese",
            enabled_modalities={"tables"},
        )
        analyzer = MultimodalAnalyzer(config=config, llm_func=lambda x: '{"name":"t","description":"d"}')
        context = {
            "multimodal_sidecars": {
                "drawings": {"im-0001": {"id": "im-0001", "blockid": "b1", "content": "", "caption": ""}},
                "tables": {},
                "equations": {},
            },
            "blocks_content_by_id": {},
        }
        result = analyzer.run(context)
        # Only tables enabled, drawings should be empty
        assert result["multimodal_analysis_results"]["drawings"] == {}

    def test_analyzer_no_llm_skips(self):
        """Without LLM function, analysis is skipped."""
        analyzer = MultimodalAnalyzer()
        context = {
            "multimodal_sidecars": {
                "drawings": {"im-0001": {"id": "im-0001", "content": ""}},
                "tables": {},
                "equations": {},
            },
            "blocks_content_by_id": {},
        }
        result = analyzer.run(context)
        assert result["multimodal_analysis_results"]["drawings"] == {}

    def test_analyzer_skip_already_analyzed(self):
        """Items with llm_analyze_result already should be preserved."""
        analyzer = MultimodalAnalyzer(llm_func=lambda x: '{"name":"new"}')
        context = {
            "multimodal_sidecars": {
                "drawings": {
                    "im-0001": {
                        "id": "im-0001",
                        "llm_analyze_result": {"name": "old_name"},
                    },
                },
                "tables": {},
                "equations": {},
            },
            "blocks_content_by_id": {},
        }
        result = analyzer.run(context)
        assert result["multimodal_analysis_results"]["drawings"]["im-0001"]["llm_analyze_result"]["name"] == "old_name"


# ============================================================================
# SurroundingContextEnricher Tests
# ============================================================================

class TestSurroundingContextEnricher:
    def test_find_target_span_drawing(self):
        block = '<drawing id="im-abc-0001" format="png" caption="Fig 1" path="assets/fig.png" src="" />'
        span = _find_target_span("drawings", "im-abc-0001", block)
        assert span is not None
        assert span[0] == 0
        assert span[1] == len(block)

    def test_find_target_span_table(self):
        block = '<table id="tb-abc-0001" format="json">[["a","b"]]</table>'
        span = _find_target_span("tables", "tb-abc-0001", block)
        assert span is not None

    def test_find_target_span_equation(self):
        block = '<equation id="eq-abc-0001" format="latex">E=mc^2</equation>'
        span = _find_target_span("equations", "eq-abc-0001", block)
        assert span is not None

    def test_find_target_span_not_found(self):
        span = _find_target_span("drawings", "im-notexist", "some text")
        assert span is None

    def test_remove_table_tags(self):
        text = "Before <table id=\"tb-1\" format=\"json\">data</table> After"
        result = _remove_table_tags(text)
        assert "Before" in result
        assert "After" in result
        assert "<table" not in result

    def test_strip_internal_markers(self):
        text = '<drawing id="im-1" path="fig.png" src="" />'
        result = _strip_internal_markers(text)
        assert "id=" not in result
        assert "path=" not in result
        assert "src=" not in result

    def test_atomize_text_only(self):
        atoms = _atomize("Hello world")
        assert len(atoms) == 1
        assert atoms[0][0] == "text"

    def test_atomize_mixed(self):
        text = "Before <drawing id=\"im-1\" format=\"png\" /> After"
        atoms = _atomize(text)
        assert len(atoms) == 3
        assert atoms[0][0] == "text"
        assert atoms[1][0] == "drawing"
        assert atoms[2][0] == "text"

    def test_estimate_tokens_empty(self):
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_text(self):
        tokens = _estimate_tokens("Hello world this is a test")
        assert tokens > 0

    def test_build_surrounding_basic(self):
        block = "Intro text <drawing id=\"im-1\" format=\"png\" /> conclusion text"
        span = _find_target_span("drawings", "im-1", block)
        assert span is not None

        result = build_surrounding(
            kind="drawings",
            block_content=block,
            target_start=span[0],
            target_end=span[1],
            max_tokens=2000,
        )
        assert "leading" in result
        assert "trailing" in result
        # Leading should contain "Intro text"
        assert "Intro" in result["leading"]

    def test_enricher_run(self):
        """Test SurroundingContextEnricher.run() operator protocol."""
        block_content = (
            "The system uses <drawing id=\"im-hash-0001\" format=\"png\" "
            "caption=\"Fig 1\" path=\"fig.png\" src=\"\" /> "
            "to illustrate the architecture."
        )
        context = {
            "multimodal_sidecars": {
                "drawings": {
                    "im-hash-0001": {
                        "id": "im-hash-0001",
                        "blockid": "block1",
                        "content": "",
                        "caption": "Fig 1",
                    },
                },
                "tables": {},
                "equations": {},
            },
            "blocks_content_by_id": {
                "block1": block_content,
            },
        }

        enricher = SurroundingContextEnricher(max_tokens=2000)
        result = enricher.run(context)

        assert "surrounding_enrichment_counts" in result
        assert result["surrounding_enrichment_counts"]["drawings"] == 1
        item = result["multimodal_sidecars"]["drawings"]["im-hash-0001"]
        assert "surrounding" in item
        assert "leading" in item["surrounding"]
        assert "trailing" in item["surrounding"]

    def test_enricher_empty_sidecars(self):
        context = {
            "multimodal_sidecars": {},
            "blocks_content_by_id": {},
        }
        enricher = SurroundingContextEnricher()
        result = enricher.run(context)
        assert result["surrounding_enrichment_counts"] == {
            "drawings": 0, "tables": 0, "equations": 0,
        }

    def test_build_surrounding_tables_removes_sibling_tables(self):
        """For tables kind, sibling table tags should be removed."""
        block = (
            "See <table id=\"tb-other\" format=\"json\">other data</table> "
            "and <table id=\"tb-abc-0001\" format=\"json\">target data</table> "
            "then more text."
        )
        span = _find_target_span("tables", "tb-abc-0001", block)
        assert span is not None

        result = build_surrounding(
            kind="tables",
            block_content=block,
            target_start=span[0],
            target_end=span[1],
            max_tokens=2000,
        )
        # Leading should NOT contain the sibling table
        assert "tb-other" not in result["leading"]
