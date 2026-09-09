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

"""Agent memory built on Graphiti's temporal knowledge graph, HugeGraph-backed.

Graphiti models memory as a **bi-temporal** graph: every fact carries

* ``valid_at`` / ``invalid_at``  -- when the fact was true (valid time)
* ``created_at`` / ``expired_at`` -- when we learned and superseded it
  (transaction time)

and contradicting information **invalidates old edges instead of deleting
them**, which is what makes "what did we believe on date X?" answerable.
That is precisely the recall-over-time requirement this module serves.

Submodules:

* :mod:`.driver` -- HugeGraph REST client (schema, vertices, edges, gremlin)
* :mod:`.hugegraph_driver` -- Graphiti ``GraphDriver`` implementation
* :mod:`.embedder` -- local sentence-transformers embedder

Deliberate choices:

* **REST, not Gremlin, for the driver.** HugeGraph's Gremlin script engine
  is absent on some JDK configurations (see the JDK note in the semantic
  layer docs), so a REST + traversers driver keeps the memory path free of
  that dependency.
* **graphiti-core is an optional dependency.** It is *not* vendored: import
  it only when the extra is installed, so the rest of hugegraph-llm keeps
  working without it.
"""

from hugegraph_llm.memory.driver import HugeGraphClient

__all__ = ["HugeGraphClient"]


def __getattr__(name: str):
    """Lazily expose the Graphiti driver, which needs the optional extra."""
    if name in ("HugeGraphDriver", "HugeGraphGraphOperations", "HugeGraphSearchInterface"):
        from hugegraph_llm.memory import hugegraph_driver

        return getattr(hugegraph_driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
