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

"""HugeGraph tools for LangChain agents."""

import re
from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log


# Characters that could cause Gremlin injection if unescaped
_GREMLIN_DANGEROUS_CHARS = re.compile(r'[\"\'\\)\(;\[\]{}]')
_GREMLIN_SANITIZE_MAX_LEN = 256


def _sanitize_gremlin_value(value: str) -> str:
    """Sanitize a value that will be interpolated into a Gremlin query string.

    Removes dangerous characters and truncates to a safe length.
    This is a defense-in-depth measure; for production use,
    prefer parameterized queries where the backend supports them.
    """
    value = value.strip()[:_GREMLIN_SANITIZE_MAX_LEN]
    value = _GREMLIN_DANGEROUS_CHARS.sub("", value)
    if not value:
        raise ValueError("Sanitized value is empty — possible injection attempt")
    return value


class HugeGraphTool:
    """Base class for a HugeGraph agent tool."""

    name: str = ""
    description: str = ""

    def __init__(self, graph_client: Optional[Any] = None, **kwargs):
        self._graph_client = graph_client

    def run(self, query: str, **kwargs) -> str:
        raise NotImplementedError


class GremlinQueryTool(HugeGraphTool):
    """Execute a Gremlin query against HugeGraph.

    name = "gremlin_query"
    description = "Run a Gremlin traversal query. Returns JSON results."
    """

    name = "gremlin_query"
    description = (
        "Execute a Gremlin traversal query against the HugeGraph database. "
        "Use for graph structure exploration, path finding, and pattern matching."
    )

    def run(self, query: str, **kwargs) -> str:
        try:
            resp = self._graph_client.gremlin(query).exec()
            return str(resp)
        except Exception as e:
            log.error("GremlinQueryTool error: %s", e)
            raise


class VectorSearchTool(HugeGraphTool):
    """Search the vector index for similar texts.

    name = "vector_search"
    description = "Search vector index by text query. Returns similar documents."
    """

    name = "vector_search"
    description = (
        "Search the vector index for documents similar to the query text. "
        "Returns ranked results with similarity scores."
    )

    def __init__(self, graph_client=None, embedding=None, vector_index=None, **kwargs):
        super().__init__(graph_client=graph_client, **kwargs)
        self._embedding = embedding
        self._vector_index = vector_index

    def run(self, query: str, top_k: int = 5, **kwargs) -> str:
        if not self._embedding or not self._vector_index:
            return "Vector search not configured."
        try:
            vec = self._embedding.get_texts_embeddings([query])[0]
            results = self._vector_index.search(vec, top_k)
            return str(results)
        except Exception as e:
            log.error("VectorSearchTool error: %s", e)
            raise


class EntitySearchTool(HugeGraphTool):
    """Search for entities in the graph by name or property.

    name = "entity_search"
    description = "Find entities matching a name pattern."
    """

    name = "entity_search"
    description = (
        "Search for entities in the graph by name or property value. "
        "Returns entity vertices with their properties."
    )

    def run(self, query: str, label: str = "Entity", **kwargs) -> str:
        try:
            safe_query = _sanitize_gremlin_value(query)
            safe_label = _sanitize_gremlin_value(label)
            gremlin = (
                f'g.V().hasLabel("{safe_label}")'
                f'.has("name", containing("{safe_query}"))'
                f'.limit(10).valueMap()'
            )
            resp = self._graph_client.gremlin(gremlin).exec()
            return str(resp)
        except Exception as e:
            log.error("EntitySearchTool error: %s", e)
            raise


class CommunityInfoTool(HugeGraphTool):
    """Get information about a community in the graph.

    name = "community_info"
    description = "Get community details by ID."
    """

    name = "community_info"
    description = (
        "Retrieve information about a specific community in the graph, "
        "including member entities and community summary/report."
    )

    def run(self, query: str, **kwargs) -> str:
        try:
            community_id = _sanitize_gremlin_value(query.strip())
            gremlin = (
                f'g.V().hasLabel("Community")'
                f'.has("community_id", "{community_id}")'
                f'.valueMap()'
            )
            resp = self._graph_client.gremlin(gremlin).exec()
            return str(resp)
        except Exception as e:
            log.error("CommunityInfoTool error: %s", e)
            raise


class SchemaInfoTool(HugeGraphTool):
    """Get the graph schema (vertex/edge labels).

    name = "schema_info"
    description = "List vertex labels, edge labels, and property keys."
    """

    name = "schema_info"
    description = (
        "Retrieve the graph schema including vertex labels, edge labels, "
        "and their property definitions."
    )

    def run(self, query: str = "", **kwargs) -> str:
        try:
            schema = {}
            vl_resp = self._graph_client.gremlin(
                'g.V().label().dedup()'
            ).exec()
            el_resp = self._graph_client.gremlin(
                'g.E().label().dedup()'
            ).exec()
            schema["vertex_labels"] = str(vl_resp) if vl_resp else "[]"
            schema["edge_labels"] = str(el_resp) if el_resp else "[]"
            return str(schema)
        except Exception as e:
            log.error("SchemaInfoTool error: %s", e)
            raise


class PathFindTool(HugeGraphTool):
    """Find paths between two entities.

    name = "path_find"
    description = "Find shortest path between entities. Format: 'entity1|entity2'"
    """

    name = "path_find"
    description = (
        "Find the shortest path between two entities in the graph. "
        "Format query as 'entity_name1|entity_name2'. "
        "Returns the path with intermediate nodes and edges."
    )

    def run(self, query: str, max_depth: int = 5, **kwargs) -> str:
        try:
            parts = query.split("|")
            if len(parts) != 2:
                return "Format: 'entity1|entity2'"
            src, dst = [_sanitize_gremlin_value(p) for p in parts]
            gremlin = (
                f'g.V().has("name", "{src}").'
                f'repeat(out().simplePath()).'
                f'until(has("name", "{dst}") or loops().is(gt({max_depth}))).'
                f'has("name", "{dst}").'
                f'path().limit(3)'
            )
            resp = self._graph_client.gremlin(gremlin).exec()
            return str(resp)
        except Exception as e:
            log.error("PathFindTool error: %s", e)
            raise


class NeighborExploreTool(HugeGraphTool):
    """Explore neighbors of an entity.

    name = "neighbor_explore"
    description = "Explore neighbors of an entity by name."
    """

    name = "neighbor_explore"
    description = (
        "Explore the neighbors (connected entities) of a given entity. "
        "Returns the entity's adjacent vertices and the edges connecting them."
    )

    def run(self, query: str, depth: int = 1, **kwargs) -> str:
        try:
            safe_query = _sanitize_gremlin_value(query)
            gremlin = (
                f'g.V().has("name", "{safe_query}").'
                f'repeat(out()).times({depth}).'
                f'dedup().limit(20).valueMap()'
            )
            resp = self._graph_client.gremlin(gremlin).exec()
            return str(resp)
        except Exception as e:
            log.error("NeighborExploreTool error: %s", e)
            raise


# Tool registry
_TOOL_CLASSES = [
    GremlinQueryTool,
    EntitySearchTool,
    CommunityInfoTool,
    SchemaInfoTool,
    PathFindTool,
    NeighborExploreTool,
    VectorSearchTool,
]


def create_hugegraph_tools(
    graph_client: Optional[Any] = None,
    embedding: Optional[Any] = None,
    vector_index: Optional[Any] = None,
) -> List[HugeGraphTool]:
    """Create the standard set of HugeGraph tools for LangChain agents.

    :param graph_client: HugeGraph Python client.
    :param embedding: Embedding model.
    :param vector_index: Vector index.
    :return: List of instantiated tool objects.
    """
    tools = []
    for cls in _TOOL_CLASSES:
        try:
            if cls is VectorSearchTool:
                tools.append(cls(
                    graph_client=graph_client,
                    embedding=embedding,
                    vector_index=vector_index,
                ))
            else:
                tools.append(cls(graph_client=graph_client))
        except Exception as e:
            log.warning("Failed to create tool %s: %s", cls.name, e)
    return tools
