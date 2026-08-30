"""Metadata-KG governance rule engine (KgRuleEngine).

Targeted at the NL2SQL anchor case: the HugeGraph knowledge graph built by
hugegraph-ai (Table / Field / Metric + hasColumn / computedFrom /
computedFromField / dependsOn, see ``api.unified_convert.KG_SCHEMA``) is the
semantic context an NL2SQL layer consumes. Garbage in the graph (dangling
edges, orphan fields, duplicated primary keys, inconsistent metric
definitions) directly produces wrong SQL. This engine runs a declarative rule
set over the graph and reports violations plus deterministic derivations, so
the KG feeding NL2SQL stays accurate, complete and auditable.

Rules are grouped by what they protect:

* A - structural integrity: no dangling objects the SQL layer could reference
* B - metric semantic consistency: definitions and formulas must resolve
* C - deterministic derivation: transitive closure / bottom-table inference
* D - disambiguation: same-name metrics across domains, missing domain tags

The rule *checks* are pure functions over a plain graph-data dict so they are
trivially unit-testable; the engine adds a thin loader from a live
HugeGraph client.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from hugegraph_llm.config import huge_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_FIELD_TYPES = {
    "BOOLEAN",
    "BYTE",
    "DATE",
    "DATETIME",
    "DOUBLE",
    "FLOAT",
    "INT",
    "LONG",
    "OBJECT",
    "STRING",
    "UUID",
}

# edge label -> (source label, target label) as declared in KG_SCHEMA
EDGE_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "hasColumn": ("Table", "Field"),
    "computedFrom": ("Metric", "Table"),
    "computedFromField": ("Metric", "Field"),
    "dependsOn": ("Metric", "Metric"),
}

NODE_LABELS = ("Table", "Field", "Metric")
EDGE_LABELS = tuple(EDGE_ENDPOINTS.keys())

# ---------------------------------------------------------------------------
# graph_data cache
# ---------------------------------------------------------------------------
# Every KgSqlVoter / KgMetricAuthority / KgLineageApi construction calls
# load_graph() -> a full label scan (measured ~1.8s on the dev slice). A short
# TTL cache keeps a single NL2SQL request from re-dumping the same graph
# several times. Keyed by graph name; stale-safe because the writes that change
# these labels (ingest, seed) are infrequent and the TTL is short.
_GRAPH_CACHE: Dict[str, Tuple[float, "GraphData"]] = {}
_GRAPH_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE_TTL = float(os.environ.get("KG_GRAPH_CACHE_TTL", "5"))


@dataclass
class KgViolation:
    """One rule breach, carrying enough context to act on it."""

    rule_id: str
    level: str  # "error" | "warning"
    target: str  # e.g. "Field: order.amount"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "level": self.level,
            "target": self.target,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class KgRuleReport:
    """Aggregated outcome of running the rule set on one graph snapshot."""

    violations: List[KgViolation] = field(default_factory=list)
    derived: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.level == "warning")

    def by_rule(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for v in self.violations:
            counts[v.rule_id] += 1
        return dict(counts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "derived": self.derived,
            "stats": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "total": len(self.violations),
                "by_rule": self.by_rule(),
            },
        }


# ---------------------------------------------------------------------------
# Rule helpers over the plain graph-data dict
# ---------------------------------------------------------------------------

GraphData = Dict[str, Any]  # {"vertices": {label: [dict]}, "edges": {label: [(src, dst)]}}


def _field_names(vertices: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    return {v.get("name") for v in vertices.get("Field", []) if v.get("name")}


# ---------------------------------------------------------------------------
# Rule checks (each returns List[KgViolation])
# ---------------------------------------------------------------------------


def check_a1_orphan_field(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """A1: every Field must have at least one incoming hasColumn edge."""
    out: List[KgViolation] = []
    owned = {dst for src, dst in edges.get("hasColumn", [])}
    for f in vertices.get("Field", []):
        name = f.get("name")
        if name and name not in owned:
            out.append(
                KgViolation(
                    rule_id="A1",
                    level="error",
                    target=f"Field: {name}",
                    message="field has no owning hasColumn edge",
                    details={"field": name},
                )
            )
    return out


def check_a2_dangling_edges(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """A2: every edge endpoint must exist as a vertex of the declared label."""
    out: List[KgViolation] = []
    all_vertices = {
        label: {v.get("name") for v in vs if v.get("name")}
        for label, vs in vertices.items()
    }
    for label, pairs in edges.items():
        src_label, dst_label = EDGE_ENDPOINTS.get(label, (None, None))
        src_ok = all_vertices.get(src_label, set())
        dst_ok = all_vertices.get(dst_label, set())
        for src, dst in pairs:
            if src_label and src not in src_ok:
                out.append(
                    KgViolation(
                        rule_id="A2",
                        level="error",
                        target=f"{label}: {src}->{dst}",
                        message=f"edge source {src!r} is not a {src_label} vertex",
                        details={"edge_label": label, "src": src, "dst": dst},
                    )
                )
            if dst_label and dst not in dst_ok:
                out.append(
                    KgViolation(
                        rule_id="A2",
                        level="error",
                        target=f"{label}: {src}->{dst}",
                        message=f"edge target {dst!r} is not a {dst_label} vertex",
                        details={"edge_label": label, "src": src, "dst": dst},
                    )
                )
    return out


def check_a3_duplicate_pk(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """A3: name is the primary key - duplicates within one label are violations."""
    out: List[KgViolation] = []
    for label, vs in vertices.items():
        seen: Dict[str, int] = defaultdict(int)
        for v in vs:
            name = v.get("name")
            if name:
                seen[name] += 1
        for name, count in seen.items():
            if count > 1:
                out.append(
                    KgViolation(
                        rule_id="A3",
                        level="error",
                        target=f"{label}: {name}",
                        message=f"duplicate primary key ({count} occurrences)",
                        details={"label": label, "name": name, "count": count},
                    )
                )
    return out


def check_a4_invalid_field_type(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """A4: Field.type must be a known HugeGraph data type."""
    out: List[KgViolation] = []
    for f in vertices.get("Field", []):
        name, ftype = f.get("name"), f.get("type")
        if name and ftype and ftype.upper() not in VALID_FIELD_TYPES:
            out.append(
                KgViolation(
                    rule_id="A4",
                    level="error",
                    target=f"Field: {name}",
                    message=f"unknown field type {ftype!r}",
                    details={"field": name, "type": ftype},
                )
            )
    return out


def check_a5_empty_table(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """A5: a Table with no outgoing hasColumn edge holds no columns."""
    out: List[KgViolation] = []
    with_fields = {src for src, dst in edges.get("hasColumn", [])}
    for t in vertices.get("Table", []):
        name = t.get("name")
        if name and name not in with_fields:
            out.append(
                KgViolation(
                    rule_id="A5",
                    level="warning",
                    target=f"Table: {name}",
                    message="table has no columns (no hasColumn edge)",
                    details={"table": name},
                )
            )
    return out


def check_b1_metric_no_source(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """B1: every Metric must have at least one computedFrom/computedFromField edge."""
    out: List[KgViolation] = []
    sourced = set()
    for label in ("computedFrom", "computedFromField"):
        for src, dst in edges.get(label, []):
            sourced.add(src)
    for m in vertices.get("Metric", []):
        name = m.get("name")
        if name and name not in sourced:
            out.append(
                KgViolation(
                    rule_id="B1",
                    level="error",
                    target=f"Metric: {name}",
                    message="metric has no computedFrom/computedFromField edge",
                    details={"metric": name},
                )
            )
    return out


def _referenced_fields(formula: Optional[str]) -> List[str]:
    """Extract ``table.field`` / bare ``field`` references from a formula."""
    if not formula:
        return []
    refs: List[str] = []
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", str(formula)):
        refs.append(token)
    return refs


def check_b2_formula_dangling_ref(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """B2: every ``table.field`` reference in a Metric.formula must resolve."""
    out: List[KgViolation] = []
    known = _field_names(vertices)
    for m in vertices.get("Metric", []):
        name = m.get("name")
        refs = _referenced_fields(m.get("formula"))
        for ref in refs:
            if ref not in known:
                out.append(
                    KgViolation(
                        rule_id="B2",
                        level="error",
                        target=f"Metric: {name}",
                        message=f"formula references unknown field {ref!r}",
                        details={"metric": name, "ref": ref, "formula": m.get("formula")},
                    )
                )
    return out


def check_b3_dependency_cycle(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """B3: dependsOn must be acyclic (DFS back-edge detection, incl. self loops)."""
    out: List[KgViolation] = []
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edges.get("dependsOn", []):
        adj[src].append(dst)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = defaultdict(int)
    stack: List[str] = []
    cycle_reported: Set[Tuple[str, str]] = set()

    def dfs(node: str) -> None:
        color[node] = GRAY
        stack.append(node)
        for nxt in adj[node]:
            if color[nxt] == GRAY:
                key = tuple(sorted([node, nxt]))
                if key not in cycle_reported:
                    cycle_reported.add(key)
                    out.append(
                        KgViolation(
                            rule_id="B3",
                            level="error",
                            target=f"dependsOn: {node}->{nxt}",
                            message="dependency cycle detected",
                            details={"edge": (node, nxt), "path": list(stack) + [nxt]},
                        )
                    )
            elif color[nxt] == WHITE:
                dfs(nxt)
        stack.pop()
        color[node] = BLACK

    for node in list(adj.keys()):
        if color[node] == WHITE:
            dfs(node)
    return out


def derive_c1_dependency_closure(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
    """C1: transitive closure over dependsOn - full lineage per metric."""
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edges.get("dependsOn", []):
        adj[src].append(dst)
    closure: List[Dict[str, Any]] = []
    for metric in sorted(adj.keys()):
        reached: Set[str] = set()
        frontier = list(adj[metric])
        while frontier:
            cur = frontier.pop()
            if cur in reached:
                continue
            reached.add(cur)
            frontier.extend(adj.get(cur, []))
        if reached:  # pragma: no branch - frontier starts non-empty for every adj key
            closure.append(
                {"metric": metric, "upstream": sorted(reached), "depth": len(reached)}
            )
    return closure


def derive_c2_bottom_tables(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
    """C2: Metric->Field + Field->Table implies the metric is computed from the table."""
    field_of_table: Dict[str, str] = {
        dst: src for src, dst in edges.get("hasColumn", [])
    }
    derived: List[Dict[str, Any]] = []
    metric_fields: Dict[str, Set[str]] = defaultdict(set)
    for src, dst in edges.get("computedFromField", []):
        metric_fields[src].add(dst)
    for metric, fields in sorted(metric_fields.items()):
        tables = sorted({field_of_table[f] for f in fields if f in field_of_table})
        if tables:
            derived.append({"metric": metric, "tables": tables})
    return derived


def check_d1_metric_conflict(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """D1: same metric name with different definitions across domains."""
    out: List[KgViolation] = []
    defs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in vertices.get("Metric", []):
        name = m.get("name")
        if name:
            defs[name].append(m)
    for name, items in defs.items():
        distinct_defs = {m.get("definition") for m in items}
        if len(distinct_defs) > 1:
            out.append(
                KgViolation(
                    rule_id="D1",
                    level="error",
                    target=f"Metric: {name}",
                    message="same metric name has conflicting definitions",
                    details={
                        "metric": name,
                        "definitions": sorted(str(d) for d in distinct_defs),
                        "count": len(items),
                    },
                )
            )
    return out


def check_d2_missing_domain(vertices: Dict[str, List[Dict[str, Any]]], edges: Dict[str, List[Tuple[str, str]]]) -> List[KgViolation]:
    """D2: vertices should carry a domain tag for NL2SQL disambiguation."""
    out: List[KgViolation] = []
    for label in ("Table", "Field", "Metric"):
        for v in vertices.get(label, []):
            name = v.get("name")
            if name and not v.get("domain"):
                out.append(
                    KgViolation(
                        rule_id="D2",
                        level="warning",
                        target=f"{label}: {name}",
                        message="missing domain tag",
                        details={"label": label, "name": name},
                    )
                )
    return out


# rule_id -> (check, derived_fn_or_None, level, description)
_RULE_SPECS: Dict[str, Tuple[Callable, Optional[Callable], str, str]] = {
    "A1": (check_a1_orphan_field, None, "error", "field without owning table"),
    "A2": (check_a2_dangling_edges, None, "error", "edge endpoint not a vertex"),
    "A3": (check_a3_duplicate_pk, None, "error", "duplicate primary key"),
    "A4": (check_a4_invalid_field_type, None, "error", "unknown field type"),
    "A5": (check_a5_empty_table, None, "warning", "table without columns"),
    "B1": (check_b1_metric_no_source, None, "error", "metric without source"),
    "B2": (check_b2_formula_dangling_ref, None, "error", "formula dangling reference"),
    "B3": (check_b3_dependency_cycle, None, "error", "dependency cycle"),
    "C1": (lambda v, e: [], derive_c1_dependency_closure, "info", "dependency closure"),
    "C2": (lambda v, e: [], derive_c2_bottom_tables, "info", "bottom-table inference"),
    "D1": (check_d1_metric_conflict, None, "error", "metric definition conflict"),
    "D2": (check_d2_missing_domain, None, "warning", "missing domain tag"),
}

RULE_IDS = tuple(_RULE_SPECS.keys())


def run_rules(data: GraphData) -> KgRuleReport:
    """Run the full rule set over a graph snapshot (pure function)."""
    report = KgRuleReport()
    vertices = data.get("vertices", {})
    edges = data.get("edges", {})
    for rule_id, (check, derive, _, _) in _RULE_SPECS.items():
        report.violations.extend(check(vertices, edges))
        if derive is not None:
            report.derived[rule_id] = derive(vertices, edges)
    return report


# ---------------------------------------------------------------------------
# Engine: live-HugeGraph loader + orchestration
# ---------------------------------------------------------------------------


class KgRuleEngine:
    """Load a graph snapshot from a live HugeGraph and run the rule set.

    The loader fetches vertices by the three KG labels and edges by the four
    declared labels through the Gremlin API, then delegates to :func:`run_rules`.
    """

    def __init__(self, client: Any, graph_name: Optional[str] = None) -> None:
        self._client = client
        self._graph_name = graph_name

    def load_graph(self, force_refresh: bool = False) -> GraphData:
        """Pull Table/Field/Metric vertices + KG edges from the live graph.

        Results are cached per graph name for ``_GRAPH_CACHE_TTL`` seconds
        (see module constants) so repeated constructions in one request reuse
        the same snapshot; pass ``force_refresh=True`` to bypass the cache.
        """
        key = self._graph_name or huge_settings.graph_name
        now = time.time()
        with _GRAPH_CACHE_LOCK:
            hit = _GRAPH_CACHE.get(key)
            if not force_refresh and hit is not None and now - hit[0] < _GRAPH_CACHE_TTL:
                return hit[1]

        vertices: Dict[str, List[Dict[str, Any]]] = {}
        for label in NODE_LABELS:
            vertices[label] = self._fetch_vertices(label)
        edges: Dict[str, List[Tuple[str, str]]] = {}
        for label in EDGE_LABELS:
            edges[label] = self._fetch_edges(label)
        data: GraphData = {"vertices": vertices, "edges": edges}

        with _GRAPH_CACHE_LOCK:
            _GRAPH_CACHE[key] = (time.time(), data)
        return data

    @classmethod
    def invalidate_graph_cache(cls, graph_name: Optional[str] = None) -> None:
        """Drop the cached snapshot for one graph (all graphs when omitted)."""
        with _GRAPH_CACHE_LOCK:
            if graph_name is None:
                _GRAPH_CACHE.clear()
            else:
                _GRAPH_CACHE.pop(graph_name, None)

    def _fetch_vertices(self, label: str) -> List[Dict[str, Any]]:
        resp = self._client.gremlin().exec(
            f"g.V().hasLabel('{label}').elementMap()"
        )
        rows = resp.get("data") if isinstance(resp, dict) else (resp or [])
        out: List[Dict[str, Any]] = []
        for row in rows or []:
            props = {k: v for k, v in row.items() if k not in ("id", "label")}
            out.append(props)
        return out

    def _fetch_edges(self, label: str) -> List[Tuple[str, str]]:
        resp = self._client.gremlin().exec(
            f"g.E().hasLabel('{label}').elementMap()"
        )
        rows = resp.get("data") if isinstance(resp, dict) else (resp or [])
        out: List[Tuple[str, str]] = []
        for row in rows or []:
            src = self._endpoint_name(row.get("OUT") or row.get("outV"))
            dst = self._endpoint_name(row.get("IN") or row.get("inV"))
            if src is not None and dst is not None:
                out.append((src, dst))
        return out

    @staticmethod
    def _endpoint_name(ep: Any) -> Optional[str]:
        """Extract the logical vertex name from an edge endpoint.

        elementMap returns OUT/IN as nested ``{'id': '<n>:<name>', 'label': ...}``;
        older clients may return a plain string. The numeric id prefix is
        stripped so endpoints align with vertex ``name`` values.
        """
        if ep is None:
            return None
        raw = ep.get("id") if isinstance(ep, dict) else ep
        if raw is None:
            return None
        text = str(raw)
        if ":" in text:
            prefix, _, rest = text.partition(":")
            if prefix.isdigit():
                return rest
        return text

    def run(self, data: Optional[GraphData] = None) -> KgRuleReport:
        """Run rules; loads from the live graph when ``data`` is not supplied."""
        snapshot = data if data is not None else self.load_graph()
        return run_rules(snapshot)
