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

from fastapi import APIRouter, HTTPException, status

from hugegraph_llm.api.models.text2sql_requests import Text2SQLRequest
from hugegraph_llm.api.models.text2sql_responses import Text2SQLResponse
from hugegraph_llm.text2sql.pipeline import Text2SQLPipeline
from hugegraph_llm.utils.log import log


class Text2SQLApi:
    @staticmethod
    def answer(req: Text2SQLRequest, pipeline: Text2SQLPipeline) -> Text2SQLResponse:
        try:
            result = pipeline.answer(req.question)
            return Text2SQLResponse(
                question=req.question,
                sql=result.sql,
                resolved_terms=result.resolved_terms,
                tables=result.tables,
                metrics=result.linked_metrics,
            )
        except Exception as e:
            log.error("Error in text2sql_api: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during Text2SQL generation.",
            ) from e


def text2sql_http_api(router: APIRouter, pipeline: Text2SQLPipeline):
    @router.post("/text2sql", status_code=status.HTTP_200_OK, response_model=Text2SQLResponse)
    def text2sql_api(req: Text2SQLRequest):
        return Text2SQLApi.answer(req, pipeline)
