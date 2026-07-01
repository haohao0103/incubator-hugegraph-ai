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

"""Tests for multimodal pipeline integration nodes and flow."""

import json
import pytest
from unittest.mock import MagicMock, patch

from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.nodes.multimodal_node.multimodal_extract_node import MultimodalExtractNode
from hugegraph_llm.nodes.multimodal_node.vlm_describe_node import VLMDescribeNode
from hugegraph_llm.nodes.multimodal_node.multimodal_kg_build_node import MultimodalKGBuildNode


# ── MultimodalExtractNode Tests ──────────────────────────────────

class TestMultimodalExtractNode:
    """Tests for PDF extraction node in pipeline."""

    def test_node_init_no_pdf(self):
        """Node should pass through when no PDF is provided."""
        node = MultimodalExtractNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        # No pdf_file_path set
        status = node.node_init()
        assert not status.isErr()
        assert node.extractor is None  # No extractor created

    def test_operator_schedule_no_pdf(self):
        """When no PDF, operator_schedule should pass through data."""
        node = MultimodalExtractNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        data_json = {"vertices": [{"name": "entity1"}]}
        result = node.operator_schedule(data_json)
        assert "vertices" in result
        assert "multimodal_extracted" not in result

    @patch("hugegraph_llm.nodes.multimodal_node.multimodal_extract_node.PDFImageExtractor")
    def test_node_init_with_pdf(self, mock_extractor_cls):
        """Node should create extractor when PDF path is provided."""
        node = MultimodalExtractNode()
        node.wk_input = WkFlowInput()
        node.wk_input.pdf_file_path = "/tmp/test.pdf"
        node.wk_input.pdf_max_pages = 3
        node.context = WkFlowState()

        status = node.node_init()
        assert not status.isErr()
        mock_extractor_cls.assert_called_once()

    @patch("hugegraph_llm.nodes.multimodal_node.multimodal_extract_node.PDFImageExtractor")
    def test_operator_schedule_with_pdf(self, mock_extractor_cls):
        """When PDF is provided, extraction should populate state fields."""
        mock_extractor = MagicMock()
        mock_result = MagicMock()
        mock_result.source_path = "/tmp/test.pdf"
        mock_result.total_pages = 2
        mock_result.total_images = 3
        mock_result.total_text_blocks = 5
        mock_page = MagicMock()
        mock_page.page_num = 1
        mock_page.page_size = (595, 842)
        mock_page.image_count = 2
        mock_page.text_block_count = 3
        mock_page.images = []
        mock_page.text_blocks = []
        mock_result.pages = [mock_page]
        mock_extractor.extract.return_value = mock_result
        mock_extractor_cls.return_value = mock_extractor

        node = MultimodalExtractNode()
        node.wk_input = WkFlowInput()
        node.wk_input.pdf_file_path = "/tmp/test.pdf"
        node.wk_input.pdf_max_pages = 5
        node.context = WkFlowState()
        node.extractor = mock_extractor

        data_json = {}
        result = node.operator_schedule(data_json)

        assert result.get("multimodal_extracted") is True
        assert result.get("total_images") == 3
        assert result.get("total_text_blocks") == 5
        assert "pdf_extraction_result" in result

    @patch("hugegraph_llm.nodes.multimodal_node.multimodal_extract_node.PDFImageExtractor")
    def test_operator_schedule_extraction_error(self, mock_extractor_cls):
        """Extraction error should be captured, not crash pipeline."""
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("PDF parsing failed")
        mock_extractor_cls.return_value = mock_extractor

        node = MultimodalExtractNode()
        node.wk_input = WkFlowInput()
        node.wk_input.pdf_file_path = "/tmp/bad.pdf"
        node.context = WkFlowState()
        node.extractor = mock_extractor

        data_json = {}
        result = node.operator_schedule(data_json)
        assert "multimodal_extract_error" in result


# ── VLMDescribeNode Tests ────────────────────────────────────────

class TestVLMDescribeNode:
    """Tests for VLM description node in pipeline."""

    def test_operator_schedule_no_extraction(self):
        """When no extraction result, VLM should be skipped."""
        node = VLMDescribeNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        data_json = {"vertices": []}

        result = node.operator_schedule(data_json)
        assert "vlm_descriptions" not in result or result.get("vlm_describe_skipped") is True

    def test_operator_schedule_no_images(self):
        """When extraction has no images, VLM should skip."""
        node = VLMDescribeNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        data_json = {"pdf_extraction_result": {"pages": [{"images": []}]}}

        result = node.operator_schedule(data_json)
        assert result.get("vlm_describe_skipped") is True

    @patch("hugegraph_llm.nodes.multimodal_node.vlm_describe_node.VLMDescriptor")
    def test_operator_schedule_with_images(self, mock_descriptor_cls):
        """VLM should produce descriptions for extracted images."""
        mock_descriptor = MagicMock()
        mock_batch_result = MagicMock()
        mock_desc = MagicMock()
        mock_desc.image_id = "img_1_0"
        mock_desc.caption = "Test caption"
        mock_desc.detailed_description = "Detailed description"
        mock_desc.object_labels = ["chart"]
        mock_desc.chart_type = "bar"
        mock_desc.key_insights = ["Insight 1"]
        mock_desc.related_keywords = ["keyword1"]
        mock_desc.confidence = 0.9
        mock_desc.vlm_model = "test-vlm"
        mock_batch_result.descriptions = [mock_desc]
        mock_batch_result.success_count = 1
        mock_batch_result.total_images = 1
        mock_batch_result.success_rate = 1.0
        mock_descriptor.describe_extracted_images.return_value = mock_batch_result
        mock_descriptor_cls.return_value = mock_descriptor

        node = VLMDescribeNode()
        node.wk_input = WkFlowInput()
        node.wk_input.vlm_provider = "test"
        node.wk_input.vlm_max_images = 10
        node.context = WkFlowState()
        node.descriptor = mock_descriptor

        data_json = {
            "pdf_extraction_result": {
                "pages": [
                    {
                        "images": [
                            {"image_id": "img_1_0", "base64_data": "fake_base64",
                             "bbox": (0, 0, 100, 100), "size": (100, 100)}
                        ]
                    }
                ]
            }
        }
        result = node.operator_schedule(data_json)

        assert "vlm_descriptions" in result
        assert len(result["vlm_descriptions"]) == 1
        assert result["vlm_descriptions"][0]["caption"] == "Test caption"

    @patch("hugegraph_llm.nodes.multimodal_node.vlm_describe_node.VLMDescriptor")
    def test_operator_schedule_vlm_error(self, mock_descriptor_cls):
        """VLM error should be captured with fallback empty descriptions."""
        mock_descriptor = MagicMock()
        mock_descriptor.describe_extracted_images.side_effect = RuntimeError("VLM API error")
        mock_descriptor_cls.return_value = mock_descriptor

        node = VLMDescribeNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        node.descriptor = mock_descriptor

        data_json = {
            "pdf_extraction_result": {
                "pages": [{"images": [{"image_id": "img_1_0", "base64_data": "fake"}]}]
            }
        }
        result = node.operator_schedule(data_json)

        assert "vlm_describe_error" in result
        assert result["vlm_descriptions"] == []


# ── MultimodalKGBuildNode Tests ────────────────────────────────

class TestMultimodalKGBuildNode:
    """Tests for multimodal KG build node in pipeline."""

    def test_operator_schedule_no_extraction(self):
        """When no extraction result, KG build should be skipped."""
        node = MultimodalKGBuildNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        data_json = {}

        result = node.operator_schedule(data_json)
        assert "multimodal_kg_built" not in result

    @patch("hugegraph_llm.nodes.multimodal_node.multimodal_kg_build_node.MultimodalKGBuilder")
    def test_operator_schedule_with_extraction(self, mock_builder_cls):
        """KG build should create vertices and edges from extraction data."""
        mock_builder = MagicMock()
        mock_stats = MagicMock()
        mock_stats.summary.return_value = {
            "vertices": {"Image": 3, "TextChunk": 5},
            "edges": {"contains_image": 3, "describes": 3},
            "total_vertices": 8,
            "total_edges": 6,
        }
        mock_builder.build.return_value = mock_stats
        mock_builder_cls.return_value = mock_builder

        node = MultimodalKGBuildNode()
        node.wk_input = WkFlowInput()
        node.wk_input.multimodal_kg_name = "test_kg"
        node.context = WkFlowState()
        node.builder = mock_builder

        data_json = {
            "pdf_extraction_result": {
                "source_path": "/tmp/test.pdf",
                "total_pages": 1,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size": (595, 842),
                        "images": [
                            {"image_id": "img_1_0", "base64_data": "fake",
                             "bbox": (0, 0, 100, 100), "size": (100, 100)}
                        ],
                        "text_blocks": [
                            {"block_id": "txt_1_0", "text": "Sample text",
                             "bbox": (0, 0, 100, 50), "is_heading": False}
                        ],
                    }
                ],
            },
        }
        result = node.operator_schedule(data_json)

        assert result.get("multimodal_kg_built") is True
        assert "multimodal_kg_stats" in result

    @patch("hugegraph_llm.nodes.multimodal_node.multimodal_kg_build_node.MultimodalKGBuilder")
    def test_operator_schedule_build_error(self, mock_builder_cls):
        """KG build error should be captured, not crash pipeline."""
        mock_builder = MagicMock()
        mock_builder.init_schema.side_effect = RuntimeError("HugeGraph connection failed")
        mock_builder_cls.return_value = mock_builder

        node = MultimodalKGBuildNode()
        node.wk_input = WkFlowInput()
        node.context = WkFlowState()
        node.builder = mock_builder

        data_json = {
            "pdf_extraction_result": {
                "source_path": "/tmp/test.pdf",
                "total_pages": 1,
                "pages": [{"page_num": 1, "images": [], "text_blocks": []}],
            },
        }
        result = node.operator_schedule(data_json)

        assert "multimodal_kg_build_error" in result


# ── WkFlowInput Multimodal Fields Tests ──────────────────────

class TestWkFlowInputMultimodalFields:
    """Tests for multimodal fields added to WkFlowInput."""

    def test_multimodal_fields_exist(self):
        """New multimodal fields should be accessible."""
        input_obj = WkFlowInput()
        assert hasattr(input_obj, "pdf_file_path")
        assert hasattr(input_obj, "pdf_max_pages")
        assert hasattr(input_obj, "vlm_provider")
        assert hasattr(input_obj, "vlm_max_images")
        assert hasattr(input_obj, "multimodal_kg_name")
        assert hasattr(input_obj, "multimodal_mode")

    def test_multimodal_fields_default_none(self):
        """All multimodal fields should default to None."""
        input_obj = WkFlowInput()
        assert input_obj.pdf_file_path is None
        assert input_obj.pdf_max_pages is None
        assert input_obj.vlm_provider is None
        assert input_obj.vlm_max_images is None
        assert input_obj.multimodal_kg_name is None
        assert input_obj.multimodal_mode is None

    def test_multimodal_fields_reset(self):
        """reset() should clear all multimodal fields."""
        input_obj = WkFlowInput()
        input_obj.pdf_file_path = "/tmp/test.pdf"
        input_obj.vlm_provider = "xiaomimo"
        input_obj.multimodal_kg_name = "test_kg"

        from pycgraph import CStatus
        input_obj.reset(CStatus())

        assert input_obj.pdf_file_path is None
        assert input_obj.vlm_provider is None
        assert input_obj.multimodal_kg_name is None


# ── WkFlowState Multimodal Fields Tests ──────────────────────

class TestWkFlowStateMultimodalFields:
    """Tests for multimodal fields added to WkFlowState."""

    def test_multimodal_state_fields_exist(self):
        """New state fields should be accessible."""
        state = WkFlowState()
        assert hasattr(state, "pdf_extraction_result")
        assert hasattr(state, "vlm_descriptions")
        assert hasattr(state, "multimodal_kg_built")
        assert hasattr(state, "multimodal_kg_stats")
        assert hasattr(state, "multimodal_search_result")

    def test_multimodal_state_fields_default_none(self):
        """All state fields should default to None."""
        state = WkFlowState()
        assert state.pdf_extraction_result is None
        assert state.vlm_descriptions is None
        assert state.multimodal_kg_built is None
        assert state.multimodal_kg_stats is None
        assert state.multimodal_search_result is None

    def test_multimodal_state_fields_setup_reset(self):
        """setup() should clear all multimodal state fields."""
        state = WkFlowState()
        state.pdf_extraction_result = {"test": True}
        state.vlm_descriptions = [{"caption": "test"}]
        state.multimodal_kg_built = True
        state.multimodal_kg_stats = {"total": 10}
        state.multimodal_search_result = {"results": []}

        state.setup()

        assert state.pdf_extraction_result is None
        assert state.vlm_descriptions is None
        assert state.multimodal_kg_built is None
        assert state.multimodal_kg_stats is None
        assert state.multimodal_search_result is None

    def test_assign_from_json_multimodal(self):
        """assign_from_json should work for multimodal fields."""
        state = WkFlowState()
        state.assign_from_json({
            "multimodal_extracted": True,
            "total_images": 3,
            "vlm_descriptions": [{"caption": "test"}],
        })
        assert state.multimodal_extracted is True
        assert state.total_images == 3
        assert state.vlm_descriptions == [{"caption": "test"}]


# ── FlowName Multimodal Tests ──────────────────────────────────

class TestFlowNameMultimodal:
    """Tests for FlowName enum multimodal entries."""

    def test_multimodal_flow_names_exist(self):
        """FlowName enum should have multimodal INDEX entry."""
        from hugegraph_llm.flows import FlowName
        assert hasattr(FlowName, "MULTIMODAL_RAG_INDEX")
        # MULTIMODAL_RAG_SEARCH removed — search uses MultiModalRetriever directly

    def test_multimodal_flow_names_values(self):
        """FlowName values should match expected strings."""
        from hugegraph_llm.flows import FlowName
        assert FlowName.MULTIMODAL_RAG_INDEX.value == "multimodal_rag_index"


# ── Multimodal Block Demo Data Tests ────────────────────────────

class TestMultimodalBlockDemoData:
    """Tests for self-contained demo data in multimodal_block.py."""

    def test_demo_vlm_descriptions_structure(self):
        """Demo VLM descriptions should have required fields."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import DEMO_VLM_DESCRIPTIONS
        for desc in DEMO_VLM_DESCRIPTIONS:
            assert "image_id" in desc
            assert "caption" in desc
            assert "chart_type" in desc
            assert "key_insights" in desc
            assert "confidence" in desc
            assert "related_keywords" in desc

    def test_demo_tables_structure(self):
        """Demo tables should have HTML and JSON data."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import DEMO_TABLES
        for tbl in DEMO_TABLES:
            assert "name" in tbl
            assert "html" in tbl
            assert "json_data" in tbl
            assert "caption" in tbl

    def test_demo_equations_structure(self):
        """Demo equations should have LaTeX content."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import DEMO_EQUATIONS
        for eq in DEMO_EQUATIONS:
            assert "name" in eq
            assert "latex_block" in eq
            assert "description" in eq

    def test_demo_search_results_structure(self):
        """Demo search results should have channel scores."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import DEMO_SEARCH_RESULTS
        assert "results" in DEMO_SEARCH_RESULTS
        assert "source_distribution" in DEMO_SEARCH_RESULTS
        for r in DEMO_SEARCH_RESULTS["results"]:
            assert "channel_scores" in r
            assert "source_type" in r

    def test_show_search_comparison(self):
        """Comparison should return valid JSON."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import show_search_comparison
        result = show_search_comparison("test query")
        data = json.loads(result)
        assert "text_only_search" in data
        assert "multimodal_search" in data
        assert "gain" in data

    def test_show_demo_images(self):
        """Image gallery data should be formatted correctly."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import show_demo_images
        descriptions = show_demo_images()
        assert len(descriptions) == 3
        assert "Chart type" in descriptions[0]

    def test_show_demo_tables(self):
        """Table display data should contain HTML."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import show_demo_tables
        tables = show_demo_tables()
        assert len(tables) == 2
        assert "<table" in tables[0]

    def test_show_demo_equations(self):
        """Equation display should contain LaTeX notation."""
        from hugegraph_llm.demo.rag_demo.multimodal_block import show_demo_equations
        equations = show_demo_equations()
        assert len(equations) == 3
        assert "$$" in equations[0]
