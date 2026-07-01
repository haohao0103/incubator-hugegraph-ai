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

"""VLM description node — generates structured descriptions for extracted images."""

from typing import Any, Dict, Optional

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.multimodal.pdf_image_extractor import ImageExtract
from hugegraph_llm.operators.multimodal.vlm_descriptor import VLMDescriptor
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class VLMDescribeNode(BaseNode):
    """
    Node for generating VLM (Vision-Language Model) descriptions
    for images extracted from PDF documents.

    Uses VLMDescriptor to produce structured descriptions:
    caption, detailed_description, object_labels, chart_type,
    key_insights, related_keywords.
    """

    descriptor: Optional[VLMDescriptor] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        vlm_provider = getattr(self.wk_input, "vlm_provider", "xiaomimo") or "xiaomimo"
        vlm_max_images = getattr(self.wk_input, "vlm_max_images", 10) or 10

        self.descriptor = VLMDescriptor(
            provider=vlm_provider,
            batch_size=min(vlm_max_images, 3),
            max_retries=2,
        )
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        extraction_result = data_json.get("pdf_extraction_result")
        if extraction_result is None:
            # No extraction result, skip VLM
            return data_json

        vlm_max_images = getattr(self.wk_input, "vlm_max_images", 10) or 10

        # Reconstruct ImageExtract objects from serialized data
        all_images = []
        for page in extraction_result.get("pages", []):
            for img_dict in page.get("images", []):
                img = ImageExtract(
                    image_id=img_dict["image_id"],
                    base64_data=img_dict["base64_data"],
                    bbox=img_dict.get("bbox", (0, 0, 100, 100)),
                    size=img_dict.get("size", (100, 100)),
                    page_num=page.get("page_num", 1),
                )
                all_images.append(img)

        if not all_images:
            data_json["vlm_describe_skipped"] = True
            return data_json

        # Limit images
        images_to_process = all_images[:vlm_max_images]

        try:
            batch_result = self.descriptor.describe_extracted_images(
                images_to_process, text_blocks=[]
            )
            # Serialize descriptions to dict
            descriptions = []
            for desc in batch_result.descriptions:
                descriptions.append({
                    "image_id": desc.image_id,
                    "caption": desc.caption,
                    "detailed_description": desc.detailed_description,
                    "object_labels": desc.object_labels,
                    "chart_type": desc.chart_type,
                    "key_insights": desc.key_insights,
                    "related_keywords": desc.related_keywords,
                    "confidence": desc.confidence,
                    "vlm_model": desc.vlm_model,
                })

            data_json["vlm_descriptions"] = descriptions
            data_json["vlm_success_count"] = batch_result.success_count
            data_json["vlm_total_images"] = batch_result.total_images
            data_json["vlm_success_rate"] = batch_result.success_rate
        except Exception as e:
            data_json["vlm_describe_error"] = str(e)
            # Provide fallback empty descriptions so pipeline can continue
            data_json["vlm_descriptions"] = []

        return data_json
