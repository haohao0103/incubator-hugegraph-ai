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
"""HugeGraph-backed table provider with namespace (delta) isolation.

Generalized from the MS-GraphRAG HugeGraph provider's ``TableProvider``:
a provider owns a single :class:`~hugegraph_llm.operators.hugegraph_op.schema_manager.SchemaManager`
connection and serves :class:`~hugegraph_llm.operators.hugegraph_op.kg_table.KgTable`
instances. An optional ``namespace`` prefixes table names so incremental
updates can write to a ``delta_<table>`` label set that is physically
isolated from the main tables (mirrors the parquet child-dir convention of
MS-GraphRAG's update pipeline).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.config import huge_settings
from hugegraph_llm.operators.hugegraph_op.kg_table import KgTable
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager


class KgTableProvider:
    """Serves :class:`KgTable` instances over one shared HugeGraph client.

    The provider lazily builds a :class:`SchemaManager` (either from an
    injected client, an explicit connection dict, or the global
    ``huge_settings``) so every table shares a single connection.
    """

    def __init__(
        self,
        client: Optional[PyHugeClient] = None,
        *,
        graph_name: Optional[str] = None,
        connection: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        label_prefix: str = "",
    ) -> None:
        if client is not None:
            self._schema_manager = SchemaManager(graph_name or "", client=client)
        else:
            self._schema_manager = SchemaManager(
                graph_name or huge_settings.graph_name, connection=connection
            )
        self._client = self._schema_manager.client
        self._namespace = namespace or ""
        self._label_prefix = label_prefix
        self._tables: Dict[str, KgTable] = {}

    @property
    def client(self) -> PyHugeClient:
        """The shared HugeGraph client."""
        return self._client

    @property
    def namespace(self) -> str:
        """The namespace prefix applied to table names."""
        return self._namespace

    def _namespaced(self, table_name: str) -> str:
        if self._namespace:
            return f"{self._namespace}_{table_name}"
        return table_name

    def table(self, name: str, schema: Optional[Dict[str, str]] = None, **kwargs: Any) -> KgTable:
        """Get (or lazily create) a table, cached per namespaced name.

        Extra kwargs (e.g. ``page_size``, ``label_prefix``) are forwarded to
        :class:`KgTable` on first creation; a per-call ``label_prefix`` wins
        over the provider-wide one.
        """
        ns_name = self._namespaced(name)
        if ns_name not in self._tables:
            table_kwargs = dict(kwargs)
            table_kwargs.setdefault("label_prefix", self._label_prefix)
            self._tables[ns_name] = KgTable(
                self._client, ns_name, schema=schema, **table_kwargs
            )
        return self._tables[ns_name]

    def child(self, name: str) -> "KgTableProvider":
        """Create a namespaced child provider for delta isolation.

        The child shares the same client but prefixes table names with
        ``<name>_`` so delta/previous table sets are physically isolated
        (e.g. ``child("delta").table("entities")`` reads/writes
        ``delta_entities``).
        """
        child_ns = f"{self._namespace}_{name}" if self._namespace else name
        return KgTableProvider(
            self._client,
            graph_name=self._schema_manager.graph_name,
            namespace=child_ns,
            label_prefix=self._label_prefix,
        )

    def namespaced_tables(self) -> Dict[str, KgTable]:
        """All tables opened so far (namespaced name -> table)."""
        return dict(self._tables)
