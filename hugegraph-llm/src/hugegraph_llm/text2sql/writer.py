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

"""Commit a :class:`SemanticModel` into HugeGraph via ``ImportGraphDataFlow``.

This closes the "seed_graph -> HugeGraph" wiring: the semantic-layer graph and
its constraint schema are serialized to the exact shapes ``Commit2Graph``
expects, then scheduled through the existing import flow.
"""

import json
from typing import Tuple

from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.text2sql.model import SemanticModel
from hugegraph_llm.text2sql.schema import Text2SQLSchema
from hugegraph_llm.text2sql.seed import seed_graph


def build_commit_payload(model: SemanticModel) -> Tuple[str, str]:
    """Return ``(data_str, schema_str)`` ready for ``ImportGraphDataFlow``.

    - ``data_str`` is a JSON string of ``{"vertices", "edges"}`` (from seed_graph);
      vertex ids are label-scoped (``"table:order"``), matching Commit2Graph's
      primary-key ``mapping_id`` so edges resolve to real server ids.
    - ``schema_str`` is the constraint schema (``propertykeys``/``vertexlabels``/
      ``edgelabels``) with ``usePrimaryKeyId`` semantics.
    """
    schema_str = json.dumps(Text2SQLSchema.to_hugegraph_dict(), ensure_ascii=False)
    data_str = json.dumps(seed_graph(model), ensure_ascii=False)
    return data_str, schema_str


def write_semantic_graph(model: SemanticModel, scheduler=None) -> str:
    """Create the schema and commit the semantic graph into HugeGraph.

    Args:
        model: The semantic model to persist.
        scheduler: Optional scheduler (defaults to ``SchedulerSingleton``); injectable for tests.

    Returns:
        The ``ImportGraphDataFlow`` post_deal result (JSON string).
    """
    scheduler = scheduler or SchedulerSingleton.get_instance()
    data_str, schema_str = build_commit_payload(model)
    return scheduler.schedule_flow(FlowName.IMPORT_GRAPH_DATA, data_str, schema_str)
