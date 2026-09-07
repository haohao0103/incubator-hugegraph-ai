# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text2SQL on top of the graph-native semantic layer.

This package is deliberately thin. Its former in-code semantic model,
retrieval, schema rendering, graph writer and seed loader were removed:
everything overlapping :mod:`hugegraph_llm.semantic_layer` defers to it.
What remains is the order-domain sample (:mod:`.orders`) and the prompt
adapter (:mod:`.pipeline`).
"""

from hugegraph_llm.text2sql.orders import (
    ORDER_METRICS,
    ORDER_QUERIES,
    ORDER_TABLES,
    ORDER_TERMS,
    OrderDomainConnector,
    ensure_order_domain,
    orders_projection,
)
from hugegraph_llm.text2sql.pipeline import Text2SQLPipeline, Text2SQLResult

__all__ = [
    "ORDER_METRICS",
    "ORDER_QUERIES",
    "ORDER_TABLES",
    "ORDER_TERMS",
    "OrderDomainConnector",
    "ensure_order_domain",
    "orders_projection",
    "Text2SQLPipeline",
    "Text2SQLResult",
]
