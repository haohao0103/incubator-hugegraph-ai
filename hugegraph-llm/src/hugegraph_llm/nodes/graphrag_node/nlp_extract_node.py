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

"""NLP hybrid extraction node for low-cost entity/relation extraction."""

from typing import Any, Dict, Optional

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.graphrag_op.nlp_hybrid_extract import ExtractMode, HybridExtractor
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class NLPExtractNode(BaseNode):
    """
    Node for NLP-based hybrid entity/relation extraction.

    Supports three extraction modes:
    - NLP_ONLY: Pure NLP extraction (lowest cost)
    - LLM_ONLY: Traditional LLM extraction (highest quality)
    - HYBRID: NLP coarse extraction + LLM refinement (balanced)

    The extraction mode is determined by WkFlowInput.extract_type.
    """

    extractor: Optional[HybridExtractor] = None
    context: Optional[WkFlowState] = None
    wk_input: Optional[WkFlowInput] = None

    def node_init(self):
        extract_mode_str = getattr(self.wk_input, "extract_mode", "hybrid")
        language = getattr(self.wk_input, "language", "en")

        try:
            extract_mode = ExtractMode(extract_mode_str)
        except ValueError:
            extract_mode = ExtractMode.HYBRID

        # Only initialize LLM if not NLP_ONLY mode
        llm = None
        if extract_mode != ExtractMode.NLP_ONLY:
            try:
                from hugegraph_llm.config import llm_settings
                from hugegraph_llm.models.llms.init_llm import get_chat_llm

                llm = get_chat_llm(llm_settings)
            except Exception:  # pylint: disable=broad-except
                # LLM unavailable, fall back to NLP_ONLY
                extract_mode = ExtractMode.NLP_ONLY

        self.extractor = HybridExtractor(
            llm=llm,
            extract_mode=extract_mode,
            language=language,
        )
        return super().node_init()

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        return self.extractor.run(data_json)
