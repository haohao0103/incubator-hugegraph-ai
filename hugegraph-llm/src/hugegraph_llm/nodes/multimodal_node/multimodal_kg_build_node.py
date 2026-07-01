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

"""Multimodal KG build node — builds knowledge graph from extracted multimodal content."""

from typing import Any, Dict, Optional

from hugegraph_llm.config import huge_settings
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.multimodal.multimodal_kg_builder import (
    MultimodalKGBuilder,
)
from hugegraph_llm.operators.multimodal.pdf_image_extractor import (
    PDFExtractionResult,
    PageResult,
    ImageExtract,
    TextBlockExtract,
)
from hugegraph_llm.operators.multimodal.vlm_descriptor import (
    BatchDescribeResult,
    ImageDescription,
)
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class MultimodalKGBuildNode(BaseNode):
    """
    Node for building a multimodal knowledge graph from extracted
    PDF content and VLM descriptions.

    Uses MultimodalKGBuilder to create vertices and edges:
    - DocumentPage, Image, TextChunk, ImageDescription vertices
    - contains_image, contains_text, describes, cross_modal_ref edges
    """

    builder: Optional[MultimodalKGBuilder] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        graph_name = getattr(self.wk_input, "multimodal_kg_name", "multimodal_poc") or "multimodal_poc"
        host = huge_settings.graph_url

        self.builder = MultimodalKGBuilder(host=host, graph=graph_name)
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        extraction_dict = data_json.get("pdf_extraction_result")
        descriptions_list = data_json.get("vlm_descriptions", [])

        if extraction_dict is None:
            # No extraction data, skip KG build
            return data_json

        # Reconstruct PDFExtractionResult from serialized dict
        pages = []
        for page_dict in extraction_dict.get("pages", []):
            images = []
            for img_dict in page_dict.get("images", []):
                img = ImageExtract(
                    image_id=img_dict["image_id"],
                    base64_data=img_dict["base64_data"],
                    bbox=img_dict.get("bbox", (0, 0, 100, 100)),
                    size=img_dict.get("size", (100, 100)),
                    page_num=page_dict.get("page_num", 1),
                )
                images.append(img)

            text_blocks = []
            for tb_dict in page_dict.get("text_blocks", []):
                tb = TextBlockExtract(
                    block_id=tb_dict["block_id"],
                    text=tb_dict["text"],
                    bbox=tb_dict.get("bbox", (0, 0, 100, 100)),
                    is_heading=tb_dict.get("is_heading", False),
                    page_num=page_dict.get("page_num", 1),
                )
                text_blocks.append(tb)

            page = PageResult(
                page_num=page_dict["page_num"],
                page_size=page_dict.get("page_size", (0, 0)),
                images=images,
                text_blocks=text_blocks,
            )
            pages.append(page)

        extraction_result = PDFExtractionResult(
            source_path=extraction_dict.get("source_path", ""),
            total_pages=extraction_dict.get("total_pages", 0),
            pages=pages,
        )

        # Reconstruct BatchDescribeResult if descriptions available
        describe_result = None
        if descriptions_list:
            image_descriptions = []
            for desc_dict in descriptions_list:
                desc = ImageDescription(
                    image_id=desc_dict["image_id"],
                    caption=desc_dict.get("caption", ""),
                    detailed_description=desc_dict.get("detailed_description", ""),
                    object_labels=desc_dict.get("object_labels", []),
                    chart_type=desc_dict.get("chart_type", "other"),
                    key_insights=desc_dict.get("key_insights", []),
                    related_keywords=desc_dict.get("related_keywords", []),
                    confidence=desc_dict.get("confidence", 0.5),
                    vlm_model=desc_dict.get("vlm_model", ""),
                    generation_time_ms=0,
                )
                image_descriptions.append(desc)

            describe_result = BatchDescribeResult(
                descriptions=image_descriptions,
                total_images=len(image_descriptions),
                success_count=data_json.get("vlm_success_count", len(image_descriptions)),
                fail_count=0,
                success_rate=data_json.get("vlm_success_rate", 1.0),
                total_time_ms=0,
            )

        try:
            # Initialize schema first
            self.builder.init_schema()
            # Build the multimodal KG
            stats = self.builder.build(
                extraction_result,
                describe_result,
                document_name=extraction_dict.get("source_path", "document"),
            )

            data_json["multimodal_kg_stats"] = stats.summary()
            data_json["multimodal_kg_built"] = True
        except Exception as e:
            data_json["multimodal_kg_build_error"] = str(e)

        return data_json
