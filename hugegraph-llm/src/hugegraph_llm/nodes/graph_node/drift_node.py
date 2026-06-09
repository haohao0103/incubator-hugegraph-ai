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

"""DRIFT search DAG node."""

from typing import Any, Dict, Optional

from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.llm_op.drift_search import DriftSearch
from hugegraph_llm.utils.log import log
from pycgraph import CStatus


class DriftSearchNode(BaseNode):
    """DAG node for DRIFT search pipeline.

    Orchestrates the 5-step DRIFT search:
    HyDE → Community Match → Primer → Parallel Local Search → Reduce
    """

    operator: DriftSearch

    def node_init(self) -> CStatus:
        """Initialize the DRIFT search operator."""
        try:
            max_depth = (
                self.wk_input.drift_max_depth
                if getattr(self.wk_input, "drift_max_depth", None) is not None
                else 2
            )
            communities_top_k = (
                self.wk_input.drift_communities_top_k
                if getattr(self.wk_input, "drift_communities_top_k", None) is not None
                else 5
            )
            language = (
                self.wk_input.language
                if getattr(self.wk_input, "language", None) is not None
                else "en"
            )

            embedding = None
            try:
                embedding = Embeddings().get_embedding()
            except Exception:
                log.debug("No embedding available for DriftSearchNode")

            self.operator = DriftSearch(
                embedding=embedding,
                max_local_depth=max_depth,
                communities_top_k=communities_top_k,
                language=language,
            )
            return super().node_init()
        except Exception as e:
            log.error("Failed to initialize DriftSearchNode: %s", e)
            return CStatus(-1, f"DriftSearchNode initialization failed: {e}")

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the DRIFT search pipeline."""
        try:
            result = self.operator.run(data_json)
            return result
        except Exception as e:
            log.error("DRIFT search failed: %s", e)
            data_json["drift_answer"] = f"DRIFT search failed: {e}"
            return data_json
