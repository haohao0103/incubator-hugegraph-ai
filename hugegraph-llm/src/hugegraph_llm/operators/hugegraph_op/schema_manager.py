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
import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from pyhugegraph.client import PyHugeClient
from requests.exceptions import RequestException

from hugegraph_llm.config import huge_settings
from hugegraph_llm.enums.property_cardinality import PropertyCardinality
from hugegraph_llm.enums.property_data_type import PropertyDataType
from hugegraph_llm.operators.hugegraph_op.retry_utils import retry_on_connection_error
from hugegraph_llm.utils.log import log

# Schema kinds (matches the HugeGraph REST schema path segments)
PROPERTY_KEYS = "propertykeys"
VERTEX_LABELS = "vertexlabels"
EDGE_LABELS = "edgelabels"
INDEX_LABELS = "indexlabels"

# Vertex id strategies supported by HugeGraph
ID_STRATEGY_PRIMARY_KEY = "PRIMARY_KEY"
ID_STRATEGY_CUSTOMIZE_STRING = "CUSTOMIZE_STRING"
ID_STRATEGY_CUSTOMIZE_NUMBER = "CUSTOMIZE_NUMBER"
ID_STRATEGY_AUTOMATIC = "AUTOMATIC"
ID_STRATEGY_AUTO = "AUTO"


class SchemaManager:
    def __init__(
        self,
        graph_name: str,
        *,
        connection: Optional[Dict[str, Any]] = None,
        client: Optional[PyHugeClient] = None,
    ):
        self.graph_name = graph_name
        if client is not None:
            # Reuse an existing client (e.g. from Commit2Graph) so the whole
            # pipeline shares one connection instead of opening a new one.
            self.client = client
        else:
            # Apply a request-scoped connection as a complete unit (omitted fields stay as
            # given) so it cannot fall back to global huge_settings per-field.
            if connection is not None:
                url = connection.get("url")
                user = connection.get("user")
                pwd = connection.get("pwd")
                graphspace = connection.get("graphspace")
            else:
                url = huge_settings.graph_url
                user = huge_settings.graph_user
                pwd = huge_settings.graph_pwd
                graphspace = huge_settings.graph_space
            self.client = PyHugeClient(
                url=url,
                graph=self.graph_name,
                user=user,
                pwd=pwd,
                graphspace=graphspace,
            )
        self.schema = self.client.schema()
        self._schema_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def encode_vertex_id(vid: Any) -> str:
        """Encode a CUSTOMIZE_STRING vertex id for use in a REST path segment.

        HugeGraph 1.7.x requires customize-string ids in the REST path to be
        JSON string literals (wrapped in double quotes), otherwise the server
        rejects them with "must be formatted as Number/String/UUID". A bare
        ``Apple`` or an id containing CJK / spaces is rejected, so we wrap the
        raw id in double quotes and percent-encode the whole literal.
        """
        return quote(json.dumps(str(vid), ensure_ascii=False), safe="")

    # -- idempotent schema management ----------------------------------------

    @retry_on_connection_error()
    def exists(self, kind: str, name: str) -> bool:
        """Check whether a propertykey/vertexlabel/edgelabel/indexlabel exists.

        Idempotent create pattern (generalized from the MS-GraphRAG HugeGraph
        provider): always probe the server before POSTing a create so repeated
        runs are no-ops instead of 500 "already exists" errors.
        """
        getters = {
            PROPERTY_KEYS: self.schema.getPropertyKey,
            VERTEX_LABELS: self.schema.getVertexLabel,
            EDGE_LABELS: self.schema.getEdgeLabel,
            INDEX_LABELS: self.schema.getIndexLabel,
        }
        getter = getters.get(kind)
        if getter is None:
            raise ValueError(f"Unknown schema kind: {kind}")
        try:
            return getter(name) is not None
        except Exception as exc:  # noqa: BLE001
            # Network / server hiccups must not break the pipeline; treat as
            # "unknown" and let the downstream create attempt surface errors.
            log.warning("exists(%s, %s) check failed: %s", kind, name, exc)
            return False

    def create_property_key(
        self, name: str, data_type: str, cardinality: str = "SINGLE"
    ) -> bool:
        """Idempotently create a property key. Returns True if newly created."""
        if self.exists(PROPERTY_KEYS, name):
            log.info("PropertyKey '%s' already exists, skip", name)
            return False
        try:
            dtype = PropertyDataType(data_type)
            card = PropertyCardinality(cardinality)
        except ValueError:
            log.critical(
                "Invalid data type %s / cardinality %s for property %s, skip",
                data_type,
                cardinality,
                name,
            )
            return False
        builder = self.schema.propertyKey(name)
        self._apply_data_type(builder, dtype)
        self._apply_cardinality(builder, card)
        builder.ifNotExist().create()
        return True

    def create_vertex_label(
        self,
        name: str,
        properties: list,
        id_strategy: str = ID_STRATEGY_PRIMARY_KEY,
        primary_keys: Optional[list] = None,
        nullable_keys: Optional[list] = None,
    ) -> bool:
        """Idempotently create a vertex label.

        ``nullable_keys`` defaults to ALL properties (generalized from the
        MS-GraphRAG provider) so sparse upserts are allowed instead of 400
        "property cannot be null" errors when a row only sets a subset.
        """
        if self.exists(VERTEX_LABELS, name):
            log.info("VertexLabel '%s' already exists, skip", name)
            return False
        pks = list(primary_keys or [])
        if id_strategy == ID_STRATEGY_PRIMARY_KEY:
            if not pks:
                log.error(
                    "VertexLabel '%s' uses PRIMARY_KEY but no primary_keys given, skip",
                    name,
                )
                return False
            builder = self.schema.vertexLabel(name).usePrimaryKeyId().primaryKeys(*pks)
        elif id_strategy == ID_STRATEGY_CUSTOMIZE_STRING:
            builder = self.schema.vertexLabel(name).useCustomizeStringId()
        elif id_strategy == ID_STRATEGY_CUSTOMIZE_NUMBER:
            builder = self.schema.vertexLabel(name).useCustomizeNumberId()
        elif id_strategy in (ID_STRATEGY_AUTOMATIC, ID_STRATEGY_AUTO):
            builder = self.schema.vertexLabel(name).useAutomaticId()
        else:
            log.error("Unknown id_strategy %s for VertexLabel '%s', skip", id_strategy, name)
            return False
        builder.properties(*properties)
        builder.nullableKeys(*(nullable_keys if nullable_keys is not None else properties))
        builder.ifNotExist().create()
        return True

    def create_edge_label(
        self, name: str, source_label: str, target_label: str, properties: list
    ) -> bool:
        """Idempotently create an edge label (all properties nullable by default)."""
        if self.exists(EDGE_LABELS, name):
            log.info("EdgeLabel '%s' already exists, skip", name)
            return False
        builder = self.schema.edgeLabel(name)
        builder.sourceLabel(source_label)
        builder.targetLabel(target_label)
        builder.properties(*properties)
        builder.nullableKeys(*properties)
        builder.ifNotExist().create()
        return True

    def create_index_label(
        self,
        name: str,
        base_label: str,
        field: str,
        index_type: str = "SECONDARY",
        on: str = "vertex",
    ) -> bool:
        """Idempotently create an index label (SECONDARY/RANGE/SEARCH/SHARD/UNIQUE)."""
        if index_type not in ("SECONDARY", "RANGE", "SEARCH", "SHARD", "UNIQUE"):
            log.error("Unknown index_type %s for IndexLabel '%s', skip", index_type, name)
            return False
        if self.exists(INDEX_LABELS, name):
            log.info("IndexLabel '%s' already exists, skip", name)
            return False
        builder = self.schema.indexLabel(name)
        if on == "edge":
            builder.onE(base_label)
        else:
            builder.onV(base_label)
        builder.by(field)
        if index_type == "SECONDARY":
            builder.secondary()
        elif index_type == "RANGE":
            builder.range()
        elif index_type == "SEARCH":
            builder.search()
        elif index_type == "SHARD":
            builder.shard()
        elif index_type == "UNIQUE":  # pragma: no branch - earlier checks make the False path unreachable
            builder.unique()
        builder.ifNotExist().create()
        return True

    def ensure_schema(self, schema: Dict[str, Any]) -> Dict[str, int]:
        """Idempotently create every object in a schema dict.

        Mirrors the old ``Commit2Graph.init_schema_if_need`` flow but exposes
        explicit exists-probe idempotency, nullable-everything vertex/edge
        labels, and the ``name`` secondary index required for ``.has('name')``
        filtering. Returns a summary dict of what was newly created.

        The SERVER-side primary keys are authoritative for already-existing
        labels (a label may have been created earlier with a different id
        strategy), which is why the ``name``-index decision reads them first.
        """
        summary = {
            "property_keys": 0,
            "vertex_labels": 0,
            "edge_labels": 0,
            "index_labels": 0,
        }

        for prop in schema.get("propertykeys", []):
            if self.create_property_key(
                prop["name"], prop["data_type"], prop.get("cardinality", "SINGLE")
            ):
                summary["property_keys"] += 1

        server_pk: Dict[str, set] = {}
        try:
            server_schema = self.schema.getSchema()
            for vl in (server_schema or {}).get("vertexlabels", []):
                server_pk[vl["name"]] = set(vl.get("primary_keys", []))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch server schema for PK check: %s", exc)

        for vertex in schema.get("vertexlabels", []):
            vertex_label = vertex["name"]
            properties = vertex["properties"]
            if self.create_vertex_label(
                vertex_label,
                properties,
                id_strategy=vertex.get("id_strategy", ID_STRATEGY_PRIMARY_KEY),
                primary_keys=vertex.get("primary_keys"),
                nullable_keys=vertex.get("nullable_keys"),
            ):
                summary["vertex_labels"] += 1

            # Secondary index on `name` is required for `.has('name', ...)`
            # filtering (HugeGraph rejects property filters on non-indexed
            # keys). Skip when `name` is already a primary key (auto-indexed,
            # and HugeGraph forbids a secondary index on a PK property).
            if "name" in properties:
                pk = server_pk.get(vertex_label)
                if pk is None:
                    pk = set(vertex.get("primary_keys") or [])
                if "name" not in pk:
                    try:
                        if self.create_index_label(
                            f"{vertex_label}ByName", vertex_label, "name", "SECONDARY"
                        ):
                            summary["index_labels"] += 1
                    except Exception as exc:  # noqa: BLE001
                        # Benign: index already exists, or name is (still) a PK.
                        if "No need to build index" not in str(exc):
                            raise

        for edge in schema.get("edgelabels", []):
            if self.create_edge_label(
                edge["name"],
                edge["source_label"],
                edge["target_label"],
                edge["properties"],
            ):
                summary["edge_labels"] += 1

        return summary

    # -- internal property builders ------------------------------------------

    def _apply_data_type(self, builder, data_type: PropertyDataType) -> None:
        if data_type == PropertyDataType.BOOLEAN:
            log.error("Boolean type is not supported")
        elif data_type == PropertyDataType.BYTE:
            log.warning("Byte type is not supported, use int instead")
            builder.asInt()
        elif data_type == PropertyDataType.INT:
            builder.asInt()
        elif data_type == PropertyDataType.LONG:
            builder.asLong()
        elif data_type == PropertyDataType.FLOAT:
            log.warning("Float type is not supported, use double instead")
            builder.asDouble()
        elif data_type == PropertyDataType.DOUBLE:
            builder.asDouble()
        elif data_type == PropertyDataType.TEXT:
            builder.asText()
        elif data_type == PropertyDataType.BLOB:
            log.warning("Blob type is not supported, use text instead")
            builder.asText()
        elif data_type == PropertyDataType.DATE:
            builder.asDate()
        elif data_type == PropertyDataType.UUID:
            log.warning("UUID type is not supported, use text instead")
            builder.asText()
        else:  # pragma: no cover - Enum conversion already rejects unknown types
            log.error("Unknown data type %s for property_key %s", data_type, builder)

    def _apply_cardinality(self, builder, cardinality: PropertyCardinality) -> None:
        if cardinality == PropertyCardinality.SINGLE:
            builder.valueSingle()
        elif cardinality == PropertyCardinality.LIST:
            builder.valueList()
        elif cardinality == PropertyCardinality.SET:
            builder.valueSet()
        else:  # pragma: no cover - Enum conversion already rejects unknown cardinalities
            log.error("Unknown cardinality %s for property_key %s", cardinality, builder)

    @retry_on_connection_error()
    def list_indexes(self, base_label: Optional[str] = None) -> List[Dict[str, Any]]:
        """List index label metadata (generalized from neo4j's index introspection).

        Optionally filters to indexes on a given base vertex label. Returns
        ``[{name, base_type, base_value, fields, index_type}, ...]``; network
        failures degrade to an empty list.
        """
        try:
            indexes = self.schema.getIndexLabels() or []
        except RequestException as exc:
            log.warning("list_indexes failed for graph '%s': %s", self.graph_name, exc)
            return []
        result: List[Dict[str, Any]] = []
        for idx in indexes:
            info = self._index_info(idx)
            if base_label is None or info["base_value"] == base_label:
                result.append(info)
        return result

    @retry_on_connection_error()
    def get_index_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for one index label, or None when absent/unreachable."""
        try:
            idx = self.schema.getIndexLabel(name)
        except RequestException as exc:
            log.warning("get_index_info(%s) failed: %s", name, exc)
            return None
        if idx is None:
            return None
        return self._index_info(idx)

    @staticmethod
    def _index_info(idx: Any) -> Dict[str, Any]:
        return {
            "name": idx.name,
            "base_type": idx.baseType,
            "base_value": idx.baseValue,
            "fields": list(idx.fields or []),
            "index_type": idx.indexType,
        }

    @retry_on_connection_error()
    def get_schema_cached(self, refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Lazy schema fetch with caching (generalized from NebulaGraphStore).

        The full schema is fetched once and reused; ``refresh=True`` forces a
        re-fetch. Returns ``None`` on repeated connection failures.
        """
        if refresh or self._schema_cache is None:
            try:
                self._schema_cache = self.schema.getSchema()
            except RequestException as exc:
                log.warning("get_schema_cached failed for graph '%s': %s", self.graph_name, exc)
                return None
        return self._schema_cache

    def invalidate_schema_cache(self) -> None:
        """Drop the cached schema so the next call re-fetches it."""
        self._schema_cache = None

    def build_schema_description(self, refresh: bool = False) -> str:
        """LLM-friendly schema text (generalized from NebulaGraphStore.refresh_schema).

        Formats nodes, edges and relationships into a compact description
        suitable for Text2Gremlin / prompt building::

            Node properties: ...
            Edge properties: ...
            Relationships: ...
        """
        schema = self.get_schema_cached(refresh=refresh)
        if not schema:
            return ""
        nodes = [
            f"{vl['name']}({', '.join(vl.get('properties', []))})"
            for vl in schema.get("vertexlabels", [])
        ]
        edges = [
            f"{el['name']}({', '.join(el.get('properties', []))})"
            for el in schema.get("edgelabels", [])
        ]
        rels = [
            f"{el['source_label']}-[{el['name']}]->{el['target_label']}"
            for el in schema.get("edgelabels", [])
        ]
        return (
            f"Node properties: {nodes}\n"
            f"Edge properties: {edges}\n"
            f"Relationships: {rels}"
        )

    def infer_relationships(
        self, sample: bool = True, refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Infer source/target labels for every edge label.

        Base information comes from the declared schema; when ``sample`` is
        True, one real edge is sampled per edge label to confirm the endpoints
        (mirrors NebulaGraphStore's sample-edge trick). Sampling failures
        degrade to ``sample: None``.
        """
        schema = self.get_schema_cached(refresh=refresh)
        relationships: List[Dict[str, Any]] = []
        for edge in (schema or {}).get("edgelabels", []):
            rel: Dict[str, Any] = {
                "edge": edge["name"],
                "source_label": edge.get("source_label"),
                "target_label": edge.get("target_label"),
            }
            if sample:
                rel["sample"] = self._sample_edge(edge["name"])
            relationships.append(rel)
        return relationships

    def _sample_edge(self, edge_label: str) -> Optional[Dict[str, Any]]:
        """Sample one real edge of the given label to confirm endpoints."""
        try:
            resp = self.client.gremlin().exec(
                f"g.E().hasLabel('{edge_label}').limit(1)"
            )
            # pyhugegraph gremlin exec returns {"data": [...], "meta": {...}}
            edges = resp.get("data") if isinstance(resp, dict) else (resp or [])
            if not edges:
                return None
            first = edges[0]
            return {
                "outV": str(first.get("outV")),
                "inV": str(first.get("inV")),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("sample edge %r failed: %s", edge_label, exc)
            return None

    def probe_capabilities(self) -> Dict[str, bool]:
        """Fail-fast capability probe (generalized from neo4j-graphrag-python).

        Verifies at construction time that the graph is reachable and the
        schema is readable, so callers fail early instead of discovering a
        broken connection mid-pipeline. Returns a dict of boolean probes.
        """
        caps: Dict[str, bool] = {
            "graph_reachable": False,
            "schema_readable": False,
        }
        try:
            schema = self.schema.getSchema()
            caps["graph_reachable"] = True
            caps["schema_readable"] = bool(schema)
        except RequestException as exc:
            log.warning("Capability probe failed for graph '%s': %s", self.graph_name, exc)
        return caps

    def simple_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        mini_schema = {}  # type: ignore

        # Add necessary vertexlabels items (3)
        if "vertexlabels" in schema:
            mini_schema["vertexlabels"] = []
            for vertex in schema["vertexlabels"]:
                new_vertex = {key: vertex[key] for key in ["id", "name", "properties"] if key in vertex}
                mini_schema["vertexlabels"].append(new_vertex)

        # Add necessary edgelabels items (4)
        if "edgelabels" in schema:
            mini_schema["edgelabels"] = []
            for edge in schema["edgelabels"]:
                new_edge = {
                    key: edge[key] for key in ["name", "source_label", "target_label", "properties"] if key in edge
                }
                mini_schema["edgelabels"].append(new_edge)

        return mini_schema

    def run(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if context is None:
            context = {}
        try:
            schema = self.schema.getSchema()
        except RequestException as e:
            raise ValueError(f"Failed to connect to HugeGraph to get schema '{self.graph_name}': {e}") from e
        if not schema["vertexlabels"] and not schema["edgelabels"]:
            raise ValueError(f"Cannot get {self.graph_name}'s schema from HugeGraph!")

        context.update({"schema": schema})
        # TODO: enhance the logic here
        context["simple_schema"] = self.simple_schema(schema)
        return context
