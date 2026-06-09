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

"""
Schema validation flow — integrates SchemaValidator into the DAG pipeline.

Validates extracted entities and relations against the schema,
filtering out invalid items before they reach downstream consumers.

This flow is typically used as a post-processing step after entity
extraction (GraphExtractFlow) to ensure data quality before
committing to the graph.
"""

from typing import Any, Dict

from pycgraph import GPipeline

from hugegraph_llm.flows.common import BaseFlow
from hugegraph_llm.operators.graph_op.schema_validator import SchemaValidator
from hugegraph_llm.state.ai_state import WkFlowInput
from hugegraph_llm.utils.log import log


class SchemaValidationFlow(BaseFlow):
    """Flow for schema validation of extracted graph data.

    Pipeline steps:
    1. Validate entities against schema
    2. Validate relations against schema
    3. Return validated items and error report
    """

    def __init__(self, strict_mode: bool = False):
        """
        :param strict_mode: If True, unknown properties cause errors.
        """
        self._strict_mode = strict_mode
        self._validator = SchemaValidator(strict_mode=strict_mode)

    def build_flow(self, *args, **kwargs) -> GPipeline:
        pipeline = super().build_flow(*args, **kwargs)
        # Schema validation is operator-based, not DAG-based.
        # The pipeline is minimal; actual validation happens in run().
        return pipeline

    def prepare(self, wkflow_input: WkFlowInput, *args, **kwargs):
        super().prepare(wkflow_input, *args, **kwargs)

    def post_deal(self, pipeline) -> Dict:
        return super().post_deal(pipeline)

    def run(self, entities: list, relations: list = None,
            schema=None) -> Dict:
        """Run schema validation directly (operator protocol).

        :param entities: List of entity dicts.
        :param relations: List of relation dicts.
        :param schema: Optional custom SchemaDefinition.
        :return: Validation results dict.
        """
        if schema is not None:
            self._validator._schema = schema

        ctx = {
            "extracted_entities": entities or [],
            "extracted_relations": relations or [],
        }
        return self._validator.run(ctx)
