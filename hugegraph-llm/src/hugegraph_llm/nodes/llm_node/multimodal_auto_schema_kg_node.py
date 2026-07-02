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

"""Multimodal AutoSchemaKG node for incremental indexing flows.

Infers a HugeGraph schema from text + VLM image/table descriptions when no
explicit schema is provided. This node is intended for multimodal indexing
flows where PDF pages and images contribute to the schema.
"""

import json
from typing import Any, Dict, List, Optional

from pycgraph import CStatus

from hugegraph_llm.models.llms.init_llm import get_extract_llm
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.llm_op.auto_schema_kg import MultimodalAutoSchemaKGOperator
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState
from hugegraph_llm.utils.log import log


class MultimodalAutoSchemaKGNode(BaseNode):
    """Workflow node that infers a HugeGraph schema from multimodal input.

    Behavior:
    * If the workflow context already contains a ``schema`` dict, the node is a
      no-op and passes it through.
    * If ``wk_input.schema`` is a JSON string, it is parsed and placed into the
      context as a dict.
    * Otherwise, the node builds a document from ``wk_input.texts`` plus any
      available ``data_json["pdf_extraction_result"]`` / ``data_json["vlm_descriptions"]``
      / ``data_json["text_blocks"]`` and feeds it to ``MultimodalAutoSchemaKGOperator``.
    """

    context: WkFlowState = None
    wk_input: WkFlowInput = None
    operator: Optional[MultimodalAutoSchemaKGOperator] = None

    def __init__(self, llm: Any = None, instructions: str = ""):
        super().__init__()
        self._llm = llm
        self._instructions = instructions

    def node_init(self):
        llm = self._llm or get_extract_llm()
        self.operator = MultimodalAutoSchemaKGOperator(
            llm=llm,
            schema_commit_client=None,
            review_callback=None,
            allow_commit=False,
            instructions=self._instructions,
        )
        return super().node_init()

    def operator_schedule(self, data_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if data_json is None:
            data_json = {}

        existing_schema = data_json.get("schema")
        if isinstance(existing_schema, dict) and existing_schema.get("vertexlabels"):
            log.info("MultimodalAutoSchemaKGNode: reusing schema already in context")
            return data_json

        if self.wk_input.schema is not None:
            raw_schema = self.wk_input.schema.strip()
            if raw_schema.startswith("{"):
                try:
                    schema = json.loads(raw_schema)
                    data_json["schema"] = schema
                    log.info("MultimodalAutoSchemaKGNode: using user-provided JSON schema")
                    return data_json
                except json.JSONDecodeError as exc:
                    log.error("Invalid JSON schema in wk_input.schema: %s", exc)
                    return CStatus(-1, f"Invalid JSON schema in wk_input.schema: {exc}")

        # Prefer explicit document text, but also merge any multimodal sources.
        texts = self.wk_input.texts
        if isinstance(texts, str):
            document = texts
        elif isinstance(texts, list):
            document = "\n\n".join(str(t) for t in texts if t)
        else:
            document = ""

        pdf_extraction_result = data_json.get("pdf_extraction_result")
        vlm_descriptions = data_json.get("vlm_descriptions")
        text_blocks = data_json.get("text_blocks")

        if not document and not pdf_extraction_result and not vlm_descriptions and not text_blocks:
            return CStatus(-1, "MultimodalAutoSchemaKGNode requires text or multimodal input")

        log.info("MultimodalAutoSchemaKGNode: inferring schema from text + %d multimodal descriptions",
                 len(vlm_descriptions) if vlm_descriptions else 0)
        result = self.operator.run(
            document=document or None,
            pdf_extraction_result=pdf_extraction_result,
            vlm_descriptions=vlm_descriptions,
            text_blocks=text_blocks,
        )
        if not result.review.approved:
            return CStatus(-1, f"MultimodalAutoSchemaKG schema inference rejected: {result.review.reason}")
        data_json["schema"] = result.draft.to_schema_dict()
        data_json["schema_draft"] = {
            "human_readable": result.draft.to_human_readable(),
            "raw_llm_response": result.draft.raw_llm_response,
        }
        return data_json
