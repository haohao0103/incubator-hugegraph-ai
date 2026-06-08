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


from typing import Any, Dict, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.utils.log import log

# Default limits for graph data fetching (configurable via HugeGraphConfig)
DEFAULT_VERTEX_LIMIT = 10000
DEFAULT_EDGE_LIMIT = 200


class FetchGraphData:
    def __init__(self, graph: PyHugeClient, v_limit: int = DEFAULT_VERTEX_LIMIT,
                 e_limit: int = DEFAULT_EDGE_LIMIT):
        self.graph = graph
        self.v_limit = v_limit
        self.e_limit = e_limit

    def run(self, graph_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if graph_summary is None:
            graph_summary = {}

        keys = ["vertex_num", "edge_num", "vertices", "edges", "note"]

        groovy_code = f"""
        def res = [:];
        res.{keys[0]} = g.V().count().next();
        res.{keys[1]} = g.E().count().next();
        res.{keys[2]} = g.V().id().limit({self.v_limit}).toList();
        res.{keys[3]} = g.E().id().limit({self.e_limit}).toList();
        res.{keys[4]} = "Only ≤{self.v_limit} VIDs and ≤ {self.e_limit} EIDs for brief overview .";
        return res;
        """

        try:
            response = self.graph.gremlin().exec(groovy_code)
            result = response.get("data") if isinstance(response, dict) else None
            if isinstance(result, list) and len(result) > 0:
                if len(result) == 1 and isinstance(result[0], dict):
                    graph_summary.update({key: result[0].get(key) for key in keys})
                else:
                    graph_summary.update(
                        {
                            key: result[i].get(key) if i < len(result) and isinstance(result[i], dict) else None
                            for i, key in enumerate(keys)
                        }
                    )
        except Exception as e:
            log.error("FetchGraphData: Gremlin execution failed: %s", e)

        return graph_summary
