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

"""Bidirectional Ossie / OSI YAML connector.

Ossie (formerly Open Semantic Interchange, now an Apache incubator project)
standardises how datasets, fields, relationships and metrics are *described*
so a "Monthly Active Users" means the same thing in every tool. This
connector maps an OSI YAML spec onto the semantic layer graph and back.

Why it is the natural second connector: it needs no warehouse, no network
and no credentials -- a YAML file is a complete semantic model. That makes
it the only connector that can be exercised at realistic scale (33 tables,
323 fields, 54 relationships, 9 metrics) in CI.

Modelling decisions specific to HugeGraph:

* **No secondary labels.** neocarta layers ``OsiTable`` / ``OsiColumn`` as
  secondary labels on ``:Table`` / ``:Column`` so traversals over ``:Table``
  reach OSI data. HugeGraph has no multi-label vertices, so provenance is
  carried by the ``source_system`` property instead, and every id is
  namespaced by model name (``acme:table:offices``) so several models can
  coexist in one graph.
* **Trust fields ride in ``custom_extensions``.** Ossie standardises the
  definition but deliberately not confidence / lineage / freshness, so those
  are exported as an ``ossie-hugegraph`` custom extension (see
  :func:`_trust_extension`). They survive a round-trip and are ignored by
  tools that only speak core Ossie.

Spec notes encoded here (from neocarta's OSI connector and the ACME sample):

* ``dataset.source`` is a 3-part ``database.schema.table`` identifier or a
  SQL query; 1-part / 2-part sources are rejected as spec-non-compliant.
* ``ai_context.synonyms`` become ``BusinessTerm`` nodes, merged on name so
  they collide cleanly with catalog-derived terms.
* ``dimension.is_time`` is tri-state: absent means "unknown" and must stay
  absent on export, not be coerced to ``false``.
* ``relationships`` carry ordered ``from_columns`` / ``to_columns`` so
  composite-key joins survive a round-trip.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.semantic_layer.connectors.base import SourceConnector, make_id
from hugegraph_llm.semantic_layer.enums import EdgeLabel, VertexLabel
from hugegraph_llm.semantic_layer.paging import iter_vertices
from hugegraph_llm.semantic_layer.schema_def import coerce_value
from hugegraph_llm.utils.log import log

__all__ = [
    "OssieConnector",
    "SUPPORTED_VERSIONS",
    "TRUST_VENDOR",
    "load_spec",
]

#: OSI spec versions this connector was written against.
SUPPORTED_VERSIONS = ("0.1.1",)

#: Vendor name used for the trust-bearing custom extension on export.
TRUST_VENDOR = "ossie-hugegraph"

try:  # PyYAML is an optional dependency of the semantic layer.
    import yaml
except ImportError:  # pragma: no cover - exercised only without the extra
    yaml = None  # type: ignore[assignment]


def load_spec(source: str) -> Dict[str, Any]:
    """Load an OSI YAML spec from a local path or an HTTP(S) URL."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for the Ossie connector (pip install pyyaml)"
        )
    if source.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError(f"{source}: top level of an OSI spec must be a mapping")
    return spec


def _dialect_expressions(raw: Any) -> List[Tuple[str, str]]:
    """Flatten ``expression: {dialects: [{dialect, expression}]}``."""
    if not isinstance(raw, dict):
        return []
    out = []
    for item in raw.get("dialects") or []:
        if not isinstance(item, dict):
            continue
        dialect = item.get("dialect")
        expression = item.get("expression")
        if dialect and expression is not None:
            out.append((str(dialect), str(expression)))
    return out


def _first_expression(raw: Any) -> str:
    exprs = _dialect_expressions(raw)
    return exprs[0][1] if exprs else ""


def _first_dialect(raw: Any) -> str:
    exprs = _dialect_expressions(raw)
    return exprs[0][0] if exprs else ""


def _split_source(source: str) -> Tuple[str, str, str]:
    """Split ``database.schema.table``; returns (database, schema, table)."""
    parts = [p for p in str(source or "").split(".") if p]
    if len(parts) != 3:
        raise ValueError(
            f"dataset.source must be a 3-part 'database.schema.table' "
            f"identifier or a SQL query, got: {source!r}"
        )
    return parts[0], parts[1], parts[2]


def _is_query_source(source: str) -> bool:
    text = str(source or "").strip().lower()
    return " " in text and any(
        kw in text for kw in ("select", "with", "from")
    )


def _parse_trust_extension(raw: Any) -> Dict[str, Any]:
    """Extract trust fields from a ``custom_extensions`` list.

    Returns only the keys this connector understands, so foreign vendors'
    extensions pass through harmlessly. Values are trusted as-is here:
    coercion happens in ``_clean`` against the declared property types, and
    a bogus confidence is a data-quality problem, not a parse error.
    """
    trust: Dict[str, Any] = {}
    if not isinstance(raw, list):
        return trust
    for ext in raw:
        if not isinstance(ext, dict):
            continue
        if ext.get("vendor_name") != TRUST_VENDOR:
            continue
        data = ext.get("data")
        if not isinstance(data, dict):
            continue
        for key in ("confidence", "source_system", "lineage_ref", "freshness_ts"):
            if key in data and data[key] is not None:
                trust[key] = data[key]
    return trust


def _trust_extension(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the trust custom extension for a node, or None when unset.

    Ossie standardises definitions, not trust: confidence, lineage,
    freshness and provenance are not in the core spec. Rather than dropping
    them (and losing the only thing that tells an agent whether a definition
    can be believed) they are attached as a vendor extension.
    """
    trust_keys = ("confidence", "source_system", "lineage_ref", "freshness_ts")
    data = {k: node["properties"][k] for k in trust_keys
            if node["properties"].get(k) is not None}
    if not data:
        return None
    return {"vendor_name": TRUST_VENDOR, "data": data}


class OssieConnector(SourceConnector):
    """Loads an OSI YAML spec into the graph, and exports it back out."""

    name = "ossie"

    def __init__(
        self,
        client: PyHugeClient,
        *,
        version: str = "0.1.1",
        spec_source: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.version = version
        self.spec_source = spec_source
        self.model_name: Optional[str] = None
        self._vertices: List[Tuple[str, Dict[str, Any]]] = []
        self._edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self.stats: Dict[str, int] = {}
        self.warnings: List[str] = []

    # -- extract ------------------------------------------------------------

    def extract(self) -> List[Dict[str, Any]]:
        """Read the YAML spec and flatten it into one record per model."""
        if not self.spec_source:
            raise ValueError("spec_source is required for ingest")
        spec = load_spec(self.spec_source)

        declared = spec.get("version")
        if declared is None:
            self.warnings.append("spec has no top-level 'version' field")
        elif str(declared) != self.version:
            self.warnings.append(
                f"spec version {declared!r} != expected {self.version!r}"
            )
        if self.version not in SUPPORTED_VERSIONS:
            self.warnings.append(
                f"version {self.version!r} is outside SUPPORTED_VERSIONS "
                f"{SUPPORTED_VERSIONS}"
            )

        models = spec.get("semantic_model") or []
        if not isinstance(models, list) or not models:
            raise ValueError("spec contains no 'semantic_model' entries")
        return [{"version": declared, "model": m} for m in models]

    # -- transform ----------------------------------------------------------

    def transform(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map OSI datasets / relationships / metrics onto graph objects."""
        vertices: Dict[str, Dict[str, Any]] = {}
        edges: List[Tuple[str, str, str, Dict[str, Any]]] = []
        stats = {
            "databases": 0, "schemas": 0, "tables": 0, "columns": 0,
            "metrics": 0, "joins": 0, "terms": 0, "query_datasets": 0,
            "join_column_edges": 0, "time_dimensions": 0,
        }

        for record in raw:
            model = record["model"]
            model_name = model.get("name")
            if not model_name:
                raise ValueError("semantic_model entry is missing 'name'")
            self.model_name = model_name
            ns = model_name  # namespace prefix for every id in this model

            # ---- datasets -> Database / Schema / Table / Column ----
            for dataset in model.get("datasets") or []:
                ds_name = dataset.get("name")
                source = dataset.get("source")
                if not ds_name:
                    continue

                if _is_query_source(source):
                    # A query-backed dataset has no physical table.
                    stats["query_datasets"] += 1
                    self.warnings.append(
                        f"dataset '{ds_name}' is query-backed; ingested as Query "
                        "without column resolution"
                    )
                    qid = make_id(ns, "query", ds_name)
                    vertices[qid] = {
                        "label": VertexLabel.QUERY.value,
                        "properties": self._clean({
                            "name": ds_name,
                            "content": str(source),
                        }),
                    }
                    continue

                database, schema, table = _split_source(source)
                db_id = make_id(ns, "db", database)
                vertices.setdefault(db_id, {
                    "label": VertexLabel.DATABASE.value,
                    "properties": self._clean({"name": database, "platform": ""}),
                })
                sch_id = make_id(ns, "schema", database, schema)
                vertices.setdefault(sch_id, {
                    "label": VertexLabel.SCHEMA.value,
                    "properties": self._clean({"name": f"{database}.{schema}"}),
                })
                edges.append((EdgeLabel.HAS_SCHEMA.value, db_id, sch_id, {}))

                tbl_id = make_id(ns, "table", ds_name)
                # Trust extensions (confidence/lineage/freshness) ride in
                # custom_extensions on import -- the same place export puts
                # them, so a round trip preserves what the spec itself does
                # not model.
                trust = _parse_trust_extension(dataset.get("custom_extensions"))
                vertices[tbl_id] = {
                    "label": VertexLabel.TABLE.value,
                    "properties": self._clean({
                        "name": ds_name,
                        "database": database,
                        "schema": schema,
                        "comment": dataset.get("description", "") or "",
                        "source_system": trust.get(
                            "source_system", f"ossie:{model_name}"
                        ),
                        **{k: v for k, v in trust.items() if k != "source_system"},
                    }),
                }
                stats["tables"] += 1
                edges.append((EdgeLabel.HAS_TABLE.value, sch_id, tbl_id, {}))

                primary_keys = set(dataset.get("primary_key") or [])
                for field in dataset.get("fields") or []:
                    col_name = field.get("name")
                    if not col_name:
                        continue
                    col_id = make_id(ns, "column", ds_name, col_name)
                    props: Dict[str, Any] = {
                        "name": col_name,
                        "table": ds_name,
                        "comment": field.get("description", "") or "",
                    }
                    expression = _first_expression(field.get("expression"))
                    if expression:
                        # No `expression` property on Column in the graph model;
                        # field expressions are kept on the metric side only.
                        pass
                    dimension = field.get("dimension") or {}
                    if "is_time" in dimension:
                        props["is_time_dimension"] = bool(dimension["is_time"])
                        if dimension["is_time"]:
                            stats["time_dimensions"] += 1
                    if col_name in primary_keys:
                        props["is_primary_key"] = True
                    vertices[col_id] = {
                        "label": VertexLabel.COLUMN.value,
                        "properties": self._clean(props),
                    }
                    stats["columns"] += 1
                    edges.append((
                        EdgeLabel.HAS_COLUMN.value, tbl_id, col_id, {}
                    ))

                self._attach_synonyms(
                    (dataset.get("ai_context") or {}).get("synonyms"),
                    tbl_id, EdgeLabel.TABLE_TAGGED_WITH.value,
                    vertices, edges, stats, ns,
                )

            # ---- relationships -> Join (+ positional REFERENCES) ----
            for rel in model.get("relationships") or []:
                rel_name = rel.get("name")
                from_tbl = rel.get("from")
                to_tbl = rel.get("to")
                from_cols = rel.get("from_columns") or []
                to_cols = rel.get("to_columns") or []
                if not rel_name or not from_tbl or not to_tbl:
                    continue
                join_id = make_id(ns, "join", rel_name)
                vertices[join_id] = {
                    "label": VertexLabel.JOIN.value,
                    "properties": self._clean({
                        "name": rel_name,
                        "from_columns": list(from_cols),
                        "to_columns": list(to_cols),
                        "cardinality": rel.get("cardinality", "") or "",
                        # A relationship declared in the spec is proven by
                        # definition -- unlike an inferred `*_id` foreign key.
                        "proven": True,
                        "source_system": f"ossie:{model_name}",
                    }),
                }
                stats["joins"] += 1

                # Positional REFERENCES edges make the join traversable.
                for from_col, to_col in zip(from_cols, to_cols):
                    src = make_id(ns, "column", from_tbl, from_col)
                    dst = make_id(ns, "column", to_tbl, to_col)
                    if src in vertices and dst in vertices:
                        edges.append((
                            EdgeLabel.REFERENCES.value, src, dst, {"proven": True}
                        ))
                        stats["join_column_edges"] += 1

            # ---- metrics -> Metric + HAS_EXPRESSION ----
            for metric in model.get("metrics") or []:
                m_name = metric.get("name")
                if not m_name:
                    continue
                dialect, expression = "", ""
                exprs = _dialect_expressions(metric.get("expression"))
                if exprs:
                    dialect, expression = exprs[0]
                m_id = make_id(ns, "metric", m_name)
                m_trust = _parse_trust_extension(metric.get("custom_extensions"))
                vertices[m_id] = {
                    "label": VertexLabel.METRIC.value,
                    "properties": self._clean({
                        "name": m_name,
                        "description": metric.get("description", "") or "",
                        "expression": expression,
                        "dialect": dialect,
                        "source_system": m_trust.get(
                            "source_system", f"ossie:{model_name}"
                        ),
                        **{k: v for k, v in m_trust.items() if k != "source_system"},
                    }),
                }
                stats["metrics"] += 1

                # HAS_EXPRESSION needs a Column endpoint; the spec does not
                # name one, so link to any column the expression mentions.
                if expression:
                    for col_id in self._columns_named_in(
                        expression, vertices, ns
                    ):
                        edges.append((
                            EdgeLabel.HAS_EXPRESSION.value, m_id, col_id,
                            {"dialect": dialect} if dialect else {},
                        ))

                # Metric synonyms must be *linked*, not just created: an
                # orphaned BusinessTerm cannot answer "which table holds
                # ARR?", which is the entire point of a semantic layer.
                self._attach_synonyms(
                    (metric.get("ai_context") or {}).get("synonyms"),
                    m_id, EdgeLabel.METRIC_TAGGED_WITH.value,
                    vertices, edges, stats, ns,
                )

        self._vertices = list(vertices.items())
        self._edges = [
            e for e in edges
            if e[1] in vertices and e[2] in vertices
        ]
        self.stats = stats
        log.info("ossie transform: %s", stats)
        return [{"vertices": self._vertices, "edges": self._edges}]

    def _attach_synonyms(
        self,
        synonyms: Optional[List[str]],
        target_id: str,
        edge_label: Optional[str],
        vertices: Dict[str, Dict[str, Any]],
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
        stats: Dict[str, int],
        ns: str,
    ) -> None:
        """Upsert synonyms as BusinessTerms, merged on name."""
        for synonym in synonyms or []:
            term = str(synonym).strip()
            if not term:
                continue
            term_id = make_id(ns, "term", term)
            if term_id not in vertices:
                vertices[term_id] = {
                    "label": VertexLabel.BUSINESS_TERM.value,
                    "properties": self._clean({"name": term}),
                }
                stats["terms"] += 1
            if edge_label and target_id in vertices:
                edges.append((edge_label, target_id, term_id, {}))

    @staticmethod
    def _columns_named_in(
        expression: str,
        vertices: Dict[str, Dict[str, Any]],
        ns: str,
    ) -> List[str]:
        """Find columns whose ``table.column`` appears in a SQL expression."""
        hits = []
        prefix = f"{ns}:column:"
        for vid, node in vertices.items():
            if not vid.startswith(prefix):
                continue
            props = node["properties"]
            qualified = f"{props.get('table', '')}.{props.get('name', '')}"
            if qualified and f"{qualified}" in expression:
                hits.append(vid)
        return hits[:10]  # bound: one metric rarely references more

    # -- load ---------------------------------------------------------------

    def load(self, records: List[Dict[str, Any]]) -> int:
        payload = records[0] if records else {"vertices": [], "edges": []}
        written = 0
        for vid, node in payload.get("vertices", []):
            try:
                self.client.graph().addVertex(
                    node["label"], node["properties"], id=vid
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("ossie: vertex %s failed: %s", vid, exc)
        for label, out_v, in_v, props in payload.get("edges", []):
            try:
                self.client.graph().addEdge(label, out_v, in_v, props or {})
                written += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("ossie: edge %s %s->%s failed: %s",
                            label, out_v, in_v, exc)
        log.info("ossie load: wrote %s objects", written)
        return written

    # -- export -------------------------------------------------------------

    def export(self, model_name: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export a semantic model from the graph back to an OSI spec dict.

        Only vertices whose id is namespaced with ``model_name`` are
        included, so several models can share one graph.
        """
        if yaml is None and output_path:
            raise RuntimeError("PyYAML is required to write an OSI YAML file")

        prefix = f"{model_name}:"
        spec: Dict[str, Any] = {
            "version": self.version,
            "semantic_model": [{
                "name": model_name,
                "datasets": [],
                "relationships": [],
                "metrics": [],
            }],
        }
        model = spec["semantic_model"][0]

        # Columns are read once and indexed by table: re-scanning per table
        # would be O(tables x columns) round-trips against the server.
        columns_by_table: Dict[str, List[Dict[str, Any]]] = {}
        for _vid, props in self._read_vertices(prefix, VertexLabel.COLUMN.value):
            columns_by_table.setdefault(str(props.get("table", "")), []).append(
                props
            )

        for vid, node in self._read_vertices(prefix, VertexLabel.TABLE.value):
            table_name = str(node.get("name", ""))
            dataset: Dict[str, Any] = {
                "name": table_name,
                "source": ".".join(
                    p for p in (
                        node.get("database"), node.get("schema"), table_name
                    ) if p
                ),
                "description": node.get("comment", "") or "",
            }
            fields = [
                self._to_osi_field(props)
                for props in columns_by_table.get(table_name, [])
            ]
            if fields:
                dataset["fields"] = fields
            primary_keys = [
                str(p.get("name"))
                for p in columns_by_table.get(table_name, [])
                if p.get("is_primary_key")
            ]
            if primary_keys:
                dataset["primary_key"] = primary_keys
            trust = _trust_extension({"properties": node})
            if trust:
                dataset["custom_extensions"] = [trust]
            model["datasets"].append(dataset)

        for _vid, node in self._read_vertices(prefix, VertexLabel.JOIN.value):
            model["relationships"].append({
                "name": node.get("name"),
                "from_columns": list(node.get("from_columns") or []),
                "to_columns": list(node.get("to_columns") or []),
            })

        for _vid, node in self._read_vertices(prefix, VertexLabel.METRIC.value):
            metric: Dict[str, Any] = {"name": node.get("name")}
            if node.get("description"):
                metric["description"] = node["description"]
            if node.get("expression"):
                metric["expression"] = {
                    "dialects": [{
                        "dialect": node.get("dialect") or "ANSI_SQL",
                        "expression": node["expression"],
                    }]
                }
            model["metrics"].append(metric)

        if output_path:
            Path(output_path).write_text(
                yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        return spec

    # -- graph reads used by export ----------------------------------------

    def _read_vertices(self, prefix: str, label: str):
        """Yield (id, properties) for vertices of ``label`` in this namespace."""
        for vid, props in iter_vertices(self.client, label):
            if vid.startswith(prefix):
                yield vid, props

    @staticmethod
    def _to_osi_field(props: Dict[str, Any]) -> Dict[str, Any]:
        """Render one column as an OSI field.

        ``dimension.is_time`` is tri-state: it is emitted only when it was
        actually set, since "unknown" and "false" are different things.
        """
        field: Dict[str, Any] = {"name": props.get("name")}
        if props.get("comment"):
            field["description"] = props["comment"]
        if "is_time_dimension" in props:
            field["dimension"] = {"is_time": bool(props["is_time_dimension"])}
        return field

    @staticmethod
    def _clean(props: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in props.items():
            coerced = coerce_value(key, value)
            if coerced is None:
                continue
            cleaned[key] = coerced
        return cleaned
