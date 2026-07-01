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

"""GraphRAG operators — EDC schema pipeline and guided extraction."""

import importlib


# Lazy imports to avoid circular dependency with the full HG-AI import chain.
# Each module is loaded only when accessed via __getattr__.

_MODULE_MAP = {
    "GraphRAGSchemaConfig": ".graphrag_schema_config",
    "SchemaMode": ".graphrag_schema_config",
    "CanonicalizeStrategy": ".graphrag_schema_config",
    "DefineTriggerPolicy": ".graphrag_schema_config",
    "KGSchemaDefineOperator": ".kg_schema_define",
    "KGSchemaCanonicalizeOperator": ".kg_schema_canonicalize",
    "GuidedExtractOperator": ".guided_extract",
    "GuidedEntity": ".guided_extract",
    "GuidedRelation": ".guided_extract",
    "GuidedExtractResponse": ".guided_extract",
    "GuidedResponseModelBuilder": ".guided_extract",
    "EDCPipelineOrchestrator": ".edc_pipeline",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name], __package__)
        attr = getattr(module, name)
        # Cache in module globals for faster subsequent access
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_MODULE_MAP.keys())
