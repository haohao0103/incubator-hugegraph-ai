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

"""Multimodal PDF extraction node — extracts images and text blocks from PDF."""

from typing import Any, Dict, Optional

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.multimodal.pdf_image_extractor import (
    PDFImageExtractor,
)
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class MultimodalExtractNode(BaseNode):
    """
    Node for extracting images and text blocks from PDF documents.

    Uses PDFImageExtractor (PyMuPDF-based) to extract structured
    content: image base64 data, bounding boxes, text blocks with
    headings, and page metadata.
    """

    extractor: Optional[PDFImageExtractor] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        pdf_path = getattr(self.wk_input, "pdf_file_path", None)
        max_pages = getattr(self.wk_input, "pdf_max_pages", 5) or 5
        max_image_size_kb = getattr(self.wk_input, "pdf_max_image_size_kb", 512) or 512

        if pdf_path is None:
            # No PDF to extract, pass through
            return super().node_init()

        self.extractor = PDFImageExtractor(
            max_image_size_kb=max_image_size_kb,
            min_image_dim=50,
        )
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        pdf_path = getattr(self.wk_input, "pdf_file_path", None)
        max_pages = getattr(self.wk_input, "pdf_max_pages", 5) or 5

        if pdf_path is None or self.extractor is None:
            # No PDF, just pass through existing data
            return data_json

        try:
            result = self.extractor.extract(pdf_path, pages=None)
            # Serialize extraction result to dict for WkFlowState
            extraction_dict = {
                "pdf_extraction_result": {
                    "source_path": result.source_path,
                    "total_pages": result.total_pages,
                    "pages": [
                        {
                            "page_num": p.page_num,
                            "page_size": p.page_size,
                            "image_count": p.image_count,
                            "text_block_count": p.text_block_count,
                            "images": [
                                {
                                    "image_id": img.image_id,
                                    "base64_data": img.base64_data,
                                    "bbox": img.bbox,
                                    "size": img.size,
                                }
                                for img in p.images
                            ],
                            "text_blocks": [
                                {
                                    "block_id": tb.block_id,
                                    "text": tb.text,
                                    "bbox": tb.bbox,
                                    "is_heading": tb.is_heading,
                                }
                                for tb in p.text_blocks
                            ],
                        }
                        for p in result.pages
                    ],
                },
                "multimodal_extracted": True,
                "total_images": result.total_images,
                "total_text_blocks": result.total_text_blocks,
            }
            # Merge with existing data
            data_json.update(extraction_dict)
        except Exception as e:
            data_json["multimodal_extract_error"] = str(e)

        return data_json
