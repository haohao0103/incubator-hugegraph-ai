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

"""Migrate an existing HugeGraph metadata graph into the semantic layer.

This is the connector that matters when no warehouse catalog is reachable:
the organisation's metadata already lives in a HugeGraph KG (tables, fields,
metrics and their bindings), so the semantic layer is bootstrapped from the
KG instead of from a live warehouse.

Source model (as observed on the ``kg_rag`` graph)::

    Table  {name, comment, row_count}
    Field  {name="table.column", comment, type}
    Metric {name, definition, formula?, aliases?}   -- business metrics
    Query  {name="metric:GMV"}                      -- metric references
    edge hasColumn          Table  -> Field
    edge computedFromField  Metric -> Field
    edge lineage            Table  -> Table   (may be absent)
    edge synonym            Metric <-> Metric

Two quirks of real graphs that this connector handles explicitly rather
than assuming away:

1. **Labels are not trustworthy.** In ``kg_rag`` the ``Metric`` label carries
   two different things -- real business metrics (they have ``definition``)
   and bare field references such as ``dim_user.register_at`` (they do not).
   The ``Query`` label likewise holds ``metric:GMV`` entries, not queries.
   Classification is therefore driven by *shape* (which properties are
   present), never by label alone.
2. **Relationships may be missing entirely.** ``kg_rag`` has no ``lineage``
   edges and all ``formula`` values are empty, so ``LINEAGE`` and
   ``HAS_EXPRESSION`` simply end up empty. That is reported in the summary
   instead of being silently treated as success.

Foreign keys are not declared anywhere in the source, so they are inferred
from shared ``*_id`` / ``id`` column names (conservative, same heuristic as
``nl2sql.hugegraph_schema_source``). Inferred FKs are marked ``proven=false``
so downstream join generation never presents them as declared integrity.
"""

from typing import Any, Dict, List, Optional, Tuple

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.semantic_layer.connectors.base import SourceConnector, make_id
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.semantic_layer.paging import iter_edges, iter_vertices
from hugegraph_llm.semantic_layer.schema_def import coerce_value
from hugegraph_llm.utils.log import log

__all__ = ["HugeGraphSourceConnector", "SourceGraphMapping"]

#: Default source labels/properties, matching the ``kg_rag`` graph.
DEFAULT_MAPPING: Dict[str, Any] = {
    "table_label": "Table",
    "field_label": "Field",
    "metric_label": "Metric",
    "query_label": "Query",
    "has_column_edge": "hasColumn",
    "computed_from_field_edge": "computedFromField",
    "lineage_edge": "lineage",
    "synonym_edge": "synonym",
    "table_name_prop": "name",
    "table_comment_prop": "comment",
    "table_rowcount_prop": "row_count",
    "field_name_prop": "name",
    "field_comment_prop": "comment",
    "field_type_prop": "type",
    "metric_name_prop": "name",
    "metric_definition_prop": "definition",
    "metric_formula_prop": "formula",
    "metric_aliases_prop": "aliases",
}


class SourceGraphMapping:
    """Label/property mapping for the source graph."""

    def __init__(self, **overrides: Any) -> None:
        merged = dict(DEFAULT_MAPPING)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        self._m = merged

    def __getattr__(self, item: str) -> Any:
        try:
            return self._m[item]
        except KeyError as exc:
            raise AttributeError(f"unknown mapping key: {item}") from exc


def _split_source_id(raw: Any) -> str:
    """Strip HugeGraph's ``<numeric>:<label>`` prefix from an id.

    Source vertices may be returned with a compound id (``3:orders.id``);
    for matching we always want the bare logical id.
    """
    text = "" if raw is None else str(raw)
    return text.split(":", 1)[1] if ":" in text and not text.startswith("http") else text


class HugeGraphSourceConnector(SourceConnector):
    """Reads metadata from a HugeGraph graph and loads it as a semantic layer."""

    name = "hugegraph_source"

    def __init__(
        self,
        *,
        source_client: PyHugeClient,
        target_client: PyHugeClient,
        mapping: Optional[SourceGraphMapping] = None,
        infer_foreign_keys: bool = True,
        batch_size: int = 200,
    ) -> None:
        super().__init__()
        self.source = source_client
        self.target = target_client
        self.mapping = mapping or SourceGraphMapping()
        self.infer_foreign_keys = infer_foreign_keys
        self.batch_size = batch_size
        # populated during transform(), consumed by load()
        self._vertices: List[Tuple[str, Dict[str, Any]]] = []
        self._edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self.stats: Dict[str, int] = {}

    # -- extract ------------------------------------------------------------

    def extract(self) -> List[Dict[str, Any]]:
        """Page through every relevant source label (vertices + edges).

        Returns a single list of raw vertex records; edges are kept on the
        instance because they are meaningless without the vertex id map.
        """
        m = self.mapping
        pull: Dict[str, Any] = {}
        for key in ("table", "field", "metric", "query"):
            label = getattr(m, f"{key}_label")
            pull[f"{key}_vertices"] = self._page_vertices(label)
        for key in ("has_column", "computed_from_field", "lineage", "synonym"):
            label = getattr(m, f"{key}_edge")
            pull[f"{key}_edges"] = self._page_edges(label)

        self._raw_edges = pull  # consumed by transform()
        log.info(
            "semantic layer extract: %s tables, %s fields, %s metrics, %s queries; "
            "edges: hasColumn=%s computedFromField=%s lineage=%s synonym=%s",
            len(pull["table_vertices"]), len(pull["field_vertices"]),
            len(pull["metric_vertices"]), len(pull["query_vertices"]),
            len(pull["has_column_edges"]), len(pull["computed_from_field_edges"]),
            len(pull["lineage_edges"]), len(pull["synonym_edges"]),
        )
        # Flatten: transform() works off `pull`, but the contract wants a list.
        return [
            {"kind": key, "records": records}
            for key, records in pull.items()
        ]

    def _page_vertices(self, label: str) -> List[Dict[str, Any]]:
        # Uses the Gremlin pager: the REST pager sends an empty `&page`
        # parameter on the first call, which HugeGraph answers with 500.
        return [
            {"id": vid, "properties": props}
            for vid, props in iter_vertices(self.source, label)
        ]

    def _page_edges(self, label: str) -> List[Dict[str, Any]]:
        return [
            {"outV": out_v, "inV": in_v, "properties": props}
            for out_v, in_v, props in iter_edges(self.source, label)
        ]

    # -- transform ----------------------------------------------------------

    def transform(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map source records onto semantic layer vertices and edges.

        Classification is property-shape driven (see module docstring), so a
        mislabelled source graph still produces a sane semantic layer.
        """
        pull = {r["kind"]: r["records"] for r in raw}
        m = self.mapping
        vertices: Dict[str, Dict[str, Any]] = {}
        edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
        stats = {
            "tables": 0, "columns": 0, "metrics": 0, "terms": 0,
            "has_column": 0, "term_maps": 0, "lineage": 0,
            "synonym": 0, "foreign_keys": 0, "skipped_metric_like": 0,
        }

        # -- source id -> logical name, needed to resolve edge endpoints
        table_name_by_id: Dict[str, str] = {}
        field_name_by_id: Dict[str, str] = {}
        metric_name_by_id: Dict[str, str] = {}

        # ---- Tables ----
        for v in pull["table_vertices"]:
            name = (v.get("properties") or {}).get(m.table_name_prop)
            if not name:
                continue
            table_name_by_id[_split_source_id(v["id"])] = name
            vid = make_id("table", name)
            vertices[vid] = {
                "label": VertexLabel.TABLE.value,
                "properties": self._clean({
                    "name": name,
                    "comment": (v.get("properties") or {}).get(m.table_comment_prop, ""),
                    "row_count": (v.get("properties") or {}).get(m.table_rowcount_prop, 0),
                }),
            }
            stats["tables"] += 1

        # ---- Columns (Field vertices, name is "table.column") ----
        column_by_full_name: Dict[str, str] = {}
        for v in pull["field_vertices"]:
            props = v.get("properties") or {}
            full = props.get(m.field_name_prop)
            if not full or "." not in str(full):
                continue
            table_name, column_name = str(full).split(".", 1)
            field_name_by_id[_split_source_id(v["id"])] = str(full)
            vid = make_id("column", full)
            column_by_full_name[str(full)] = vid
            vertices[vid] = {
                "label": VertexLabel.COLUMN.value,
                "properties": self._clean({
                    "name": column_name,
                    "table": table_name,
                    "data_type": props.get(m.field_type_prop, "") or "",
                    "comment": props.get(m.field_comment_prop, "") or "",
                }),
            }
            stats["columns"] += 1

        # ---- Metrics / business terms ----
        # Shape-driven: a Metric with `definition` is a business concept; one
        # that only carries a dotted name is a field reference (skip it, the
        # Field already exists). A `Query` named `metric:X` is a metric.
        for v in pull["metric_vertices"]:
            props = v.get("properties") or {}
            name = props.get(m.metric_name_prop)
            if not name:
                continue
            sid = _split_source_id(v["id"])
            if m.metric_definition_prop not in props:
                stats["skipped_metric_like"] += 1
                continue  # field reference, not a business concept
            metric_name_by_id[sid] = name
            vid = make_id("term", name)
            vertices[vid] = {
                "label": VertexLabel.BUSINESS_TERM.value,
                "properties": self._clean({
                    "name": name,
                    "description": props.get(m.metric_definition_prop, "") or "",
                    "aliases": self._split_aliases(props.get(m.metric_aliases_prop)),
                }),
            }
            stats["terms"] += 1

        for v in pull["query_vertices"]:
            props = v.get("properties") or {}
            name = props.get("name")
            if not name:
                continue
            if str(name).startswith("metric:"):
                metric_name = str(name).split(":", 1)[1]
                vid = make_id("metric", metric_name)
                vertices[vid] = {
                    "label": VertexLabel.METRIC.value,
                    "properties": self._clean({"name": metric_name}),
                }
                stats["metrics"] += 1

        # ---- Edges ----
        for e in pull["has_column_edges"]:
            table = table_name_by_id.get(_split_source_id(e["outV"]))
            field = field_name_by_id.get(_split_source_id(e["inV"]))
            if table and field:
                edges.append((
                    EdgeLabel.HAS_COLUMN.value,
                    make_id("table", table),
                    make_id("column", field),
                    {},
                ))
                stats["has_column"] += 1

        for e in pull["computed_from_field_edges"]:
            term = metric_name_by_id.get(_split_source_id(e["outV"]))
            field = field_name_by_id.get(_split_source_id(e["inV"]))
            if term and field:
                edges.append((
                    EdgeLabel.TERM_MAPS.value,
                    make_id("term", term),
                    make_id("column", field),
                    {},
                ))
                stats["term_maps"] += 1

        for e in pull["lineage_edges"]:
            up = table_name_by_id.get(_split_source_id(e["outV"]))
            down = table_name_by_id.get(_split_source_id(e["inV"]))
            if up and down and up != down:
                edges.append((
                    EdgeLabel.LINEAGE.value,
                    make_id("table", up),
                    make_id("table", down),
                    {},
                ))
                stats["lineage"] += 1

        for e in pull["synonym_edges"]:
            left = metric_name_by_id.get(_split_source_id(e["outV"]))
            right = metric_name_by_id.get(_split_source_id(e["inV"]))
            if left and right and left != right:
                edges.append((
                    EdgeLabel.SYNONYM.value,
                    make_id("term", left),
                    make_id("term", right),
                    {},
                ))
                stats["synonym"] += 1

        # ---- Foreign keys: not declared, so infer from shared *_id names ----
        if self.infer_foreign_keys:
            edges.extend(self._infer_foreign_keys(vertices, stats))

        self._vertices = list(vertices.items())
        self._edges = [e for e in edges if self._endpoints_exist(e, vertices)]
        self.stats = stats
        log.info("semantic layer transform: %s", stats)
        return [{"vertices": self._vertices, "edges": self._edges}]

    def _infer_foreign_keys(
        self, vertices: Dict[str, Dict[str, Any]], stats: Dict[str, int]
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Infer weak FKs from shared ``id`` / ``*_id`` column names.

        Marked ``proven=false``: these are heuristics, and downstream join
        generation must never render them as declared referential integrity.
        """
        by_name: Dict[str, List[Tuple[str, str]]] = {}
        for vid, node in vertices.items():
            if node["label"] != VertexLabel.COLUMN.value:
                continue
            col = node["properties"].get("name", "")
            if col == "id" or col.endswith("_" + "id"):
                by_name.setdefault(col, []).append(
                    (vid, node["properties"].get("table", ""))
                )
        out: List[Tuple[str, str, str, Dict[str, Any]]] = []
        seen = set()
        for fulls in by_name.values():
            for i in range(len(fulls)):
                for j in range(i + 1, len(fulls)):
                    a, a_tbl = fulls[i]
                    b, b_tbl = fulls[j]
                    if a_tbl == b_tbl:
                        continue
                    key = tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((
                        EdgeLabel.REFERENCES.value, a, b, {"proven": False}
                    ))
                    stats["foreign_keys"] += 1
        return out

    @staticmethod
    def _endpoints_exist(
        edge: Tuple[str, str, str, Dict[str, Any]],
        vertices: Dict[str, Dict[str, Any]],
    ) -> bool:
        _label, out_v, in_v, _props = edge
        if out_v not in vertices or in_v not in vertices:
            log.warning("semantic layer: dropping edge with missing endpoint: %s", edge)
            return False
        return True

    # -- load ---------------------------------------------------------------

    def load(self, records: List[Dict[str, Any]]) -> int:
        """Upsert vertices then edges. Returns vertices + edges written."""
        payload = records[0] if records else {"vertices": [], "edges": []}
        written = 0
        for vid, node in payload.get("vertices", []):
            try:
                self.target.graph().addVertex(
                    node["label"], node["properties"], id=vid
                )
                written += 1
            except Exception as exc:  # noqa: BLE001 - surface, don't abort batch
                log.warning("semantic layer: vertex %s failed: %s", vid, exc)

        for label, out_v, in_v, props in payload.get("edges", []):
            try:
                self.target.graph().addEdge(label, out_v, in_v, props or {})
                written += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "semantic layer: edge %s %s->%s failed: %s", label, out_v, in_v, exc
                )
        log.info("semantic layer load: wrote %s objects", written)
        return written

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _clean(props: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce to declared types, dropping unknown/None entries.

        HugeGraph rejects undeclared property keys and mis-typed values, so
        both are removed here rather than surfacing as a 400 mid-batch.
        """
        cleaned = {}
        for key, value in props.items():
            coerced = coerce_value(key, value)
            if coerced is None:
                continue
            cleaned[key] = coerced
        return cleaned

    @staticmethod
    def _split_aliases(raw: Any) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, (list, tuple, set)):
            return [str(a).strip() for a in raw if str(a).strip()]
        return [a.strip() for a in str(raw).split(";") if a.strip()]
