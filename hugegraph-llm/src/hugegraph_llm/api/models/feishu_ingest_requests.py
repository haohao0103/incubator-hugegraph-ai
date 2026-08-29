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

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hugegraph_llm.api.models.graph_extract_requests import (
    GraphExtractClientConfig,
    normalize_graph_schema,
    validate_schema_client_config,
)


class FeishuIngestRequest(BaseModel):
    """Request body for ingesting a Feishu Wiki space into the graph/index flows."""

    model_config = ConfigDict(populate_by_name=True)

    space_id: str = Field(..., description="Feishu Wiki space id.")
    app_id: str = Field("", description="Feishu self-built app id.")
    app_secret: str = Field("", description="Feishu self-built app secret.")
    base_url: str = Field("https://open.feishu.cn", description="Open platform domain (Feishu vs Lark).")
    timeout: int = Field(30, ge=1, description="Connector HTTP timeout in seconds.")
    mode: Literal["full", "incremental"] = Field("full", description="Import mode.")
    since_timestamp: Optional[int] = Field(None, description="Epoch seconds; required for incremental import.")
    graph_schema: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        alias="schema",
        description="Graph schema (JSON string/object) or existing graph name; required when build_graph=True.",
    )
    example_prompt: Optional[str] = Field(None, description="Optional graph extraction prompt header.")
    extract_type: Literal["property_graph"] = Field("property_graph", description="Extraction type.")
    language: Literal["zh", "en"] = Field("zh", description="Language for chunk splitting.")
    split_type: Literal["document", "paragraph", "sentence"] = Field("document", description="Chunk split granularity.")
    client_config: Optional[GraphExtractClientConfig] = Field(None, description="Request-scoped HugeGraph connection.")
    build_graph: bool = Field(True, description="Run GraphExtractFlow after loading.")
    build_index: bool = Field(True, description="Run BuildVectorIndexFlow after loading.")

    @field_validator("graph_schema")
    @classmethod
    def normalize_schema(cls, v):
        if v is None:
            return None
        return normalize_graph_schema(v)

    @model_validator(mode="after")
    def validate_request(self):
        if self.mode == "incremental" and self.since_timestamp is None:
            raise ValueError("since_timestamp is required when mode='incremental'.")
        if not (self.build_graph or self.build_index):
            raise ValueError("at least one of build_graph or build_index must be true.")
        if self.build_graph:
            if self.graph_schema is None:
                raise ValueError("graph_schema is required when build_graph=True.")
            validate_schema_client_config(self.graph_schema, self.client_config)
        return self
