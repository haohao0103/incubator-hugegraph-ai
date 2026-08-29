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

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status

from hugegraph_llm.api.models.feishu_ingest_requests import FeishuIngestRequest
from hugegraph_llm.api.models.feishu_ingest_responses import FeishuIngestResponse, FeishuIngestResultModel
from hugegraph_llm.flows.feishu_ingest import FeishuIngestService
from hugegraph_llm.utils.log import log


def _parse_json(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a flow result JSON string, or return ``None`` when absent."""
    if value is None:
        return None
    return json.loads(value)


class FeishuIngestApi:
    @staticmethod
    def ingest_sync(req: FeishuIngestRequest) -> FeishuIngestResponse:
        try:
            result = FeishuIngestService.ingest_wiki(
                space_id=req.space_id,
                app_id=req.app_id,
                app_secret=req.app_secret,
                base_url=req.base_url,
                timeout=req.timeout,
                mode=req.mode,
                since_timestamp=req.since_timestamp,
                graph_schema=req.graph_schema,
                example_prompt=req.example_prompt,
                extract_type=req.extract_type,
                split_type=req.split_type,
                language=req.language,
                client_config=req.client_config,
                build_graph=req.build_graph,
                build_index=req.build_index,
            )
            return FeishuIngestResponse(
                result=FeishuIngestResultModel(
                    space_id=result.space_id,
                    mode=result.mode,
                    doc_count=result.doc_count,
                    text_count=result.text_count,
                    docs=result.docs,
                    graph_result=_parse_json(result.graph_result),
                    vector_result=_parse_json(result.vector_result),
                )
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error("Error in feishu_ingest_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during Feishu ingestion.",
            ) from e


def feishu_ingest_http_api(router: APIRouter):
    @router.post(
        "/feishu/ingest",
        status_code=status.HTTP_200_OK,
        response_model=FeishuIngestResponse,
    )
    def feishu_ingest_api(req: FeishuIngestRequest):
        return FeishuIngestApi.ingest_sync(req)
