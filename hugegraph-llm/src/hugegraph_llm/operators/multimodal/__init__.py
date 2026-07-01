# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HugeGraph-AI Multimodal operators for GraphRAG pipeline.

Adapted from LightRAG sidecar IR + placeholders + multimodal_context architecture.
"""

from hugegraph_llm.operators.multimodal.sidecar_ir import (
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRPosition,
    IRTable,
    AssetSpec,
)
from hugegraph_llm.operators.multimodal.sidecar_placeholder import (
    render_table_tag,
    render_drawing_tag,
    render_equation_tag,
    render_template,
)
from hugegraph_llm.operators.multimodal.sidecar_writer import write_sidecar
from hugegraph_llm.operators.multimodal.multimodal_analyzer import MultimodalAnalyzer
from hugegraph_llm.operators.multimodal.surrounding_context import (
    SurroundingContextEnricher,
    build_surrounding,
)
from hugegraph_llm.operators.multimodal.multimodal_entity_injector import (
    MultimodalEntityInjector,
)

__all__ = [
    "IRBlock",
    "IRDoc",
    "IRDrawing",
    "IREquation",
    "IRPosition",
    "IRTable",
    "AssetSpec",
    "render_table_tag",
    "render_drawing_tag",
    "render_equation_tag",
    "render_template",
    "write_sidecar",
    "MultimodalAnalyzer",
    "SurroundingContextEnricher",
    "build_surrounding",
    "MultimodalEntityInjector",
]
