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

"""Shared helpers that build a :class:`SchemaGraph` from a SchemaMetadata dict.

Used by the HTTP API (reload-style validation / eval), the offline eval
harness and the tests — kept free of FastAPI / pipeline globals so tools can
import it without booting the API layer.
"""

from typing import List, Optional

from .schema_graph.builder import SchemaGraphBuilder
from .schema_graph.model import Column, SchemaGraph, Table, Term


def build_schema_from_meta(meta: dict) -> SchemaGraph:
    """Populate a SchemaGraphBuilder from a metadata dict."""
    b = SchemaGraphBuilder()
    for t in meta.get("tables", []):
        b.add_table(
            Table(
                name=t["name"],
                database=t.get("database", ""),
                comment=t.get("comment", ""),
                is_fact=bool(t.get("is_fact", False)),
            )
        )
    for c in meta.get("columns", []):
        b.add_column(
            Column(
                name=c["name"],
                table=c["table"],
                data_type=c.get("data_type", ""),
                comment=c.get("comment", ""),
            )
        )
    for fk in meta.get("foreign_keys", []):
        if len(fk) == 2:
            b.add_foreign_key(fk[0], fk[1])
    for lg in meta.get("lineage", []):
        if len(lg) == 2:
            b.add_lineage(lg[0], lg[1])
    if meta.get("query_logs"):
        b.add_query_logs(meta["query_logs"])
    for tm in meta.get("terms", []):
        b.add_term(Term(name=tm["name"], comment=tm.get("comment", "")))
    for tb in meta.get("term_bindings", []):
        if len(tb) == 2:
            b.bind_term(tb[0], tb[1])
    return b.build()
