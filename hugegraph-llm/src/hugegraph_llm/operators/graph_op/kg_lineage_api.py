"""Lineage API over the metadata KG (NL2SQL explainability & join inference).

The metadata graph (Table / Field / Metric + hasColumn / computedFrom /
computedFromField / dependsOn) is a *lineage graph*: a metric is computed from
fields, fields belong to tables, and metrics depend on other metrics. The same
traversal that answers "where does this number come from?" also answers the
questions an NL2SQL layer needs at generation time:

* **explainability** -- when the LLM picks a metric, show the human where its
  number is defined, so a wrong SUM/COUNT is caught before execution;
* **join inference** -- the lineage exposes which tables a metric spans, so the
  FROM/JOIN clause can be grounded in the graph instead of guessed.

The API is a pure, deterministic traversal over the :data:`GraphData` shape
shared by every KG-RAG module (no LLM call), so it is trivially testable and
cheap to run on every NL2SQL request.

Typical use::

    api = KgLineageApi(client)
    print(api.explain("order_total"))      # human-readable lineage
    print(api.upstream("order.amount"))    # where this field comes from
    print(api.tables_of_metric("order_total"))  # { 'order' }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.operators.graph_op.kg_rule_engine import (
    KgRuleEngine,
    GraphData,
)

# edge label -> declared (source label, target label)
_EDGE_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "hasColumn": ("Table", "Field"),
    "computedFrom": ("Metric", "Table"),
    "computedFromField": ("Metric", "Field"),
    "dependsOn": ("Metric", "Metric"),
}


@dataclass
class LineageNode:
    """One node on a lineage path."""

    kind: str  # "Table" | "Field" | "Metric"
    name: str
    via: str = ""  # edge label that reached this node (empty for the root)


@dataclass
class LineageResult:
    """Upstream or downstream lineage of one target node."""

    target: str
    direction: str  # "upstream" | "downstream"
    nodes: List[LineageNode] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [n.name for n in self.nodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "direction": self.direction,
            "nodes": [
                {"kind": n.kind, "name": n.name, "via": n.via} for n in self.nodes
            ],
        }


class KgLineageApi:
    """Traverse the metadata graph to expose metric/field/table lineage."""

    def __init__(
        self,
        client: Optional[Any] = None,
        graph_name: Optional[str] = None,
        data: Optional[GraphData] = None,
    ) -> None:
        self._client = client
        self._graph_name = graph_name
        self._data = data if data is not None else None

    # -- graph access --------------------------------------------------------

    def _graph(self) -> GraphData:
        if self._data is None:
            if self._client is None:
                return {"vertices": {}, "edges": {}}
            self._data = KgRuleEngine(self._client, self._graph_name).load_graph()
        return self._data

    # -- adjacency -----------------------------------------------------------

    def _adjacency(self) -> Dict[str, List[Tuple[str, str, str, bool]]]:
        """node -> [(edge_label, neighbour, neighbour_kind, is_forward)].

        ``is_forward`` is True when the entry follows the edge's declared
        source->target direction, False when it is the reverse. The orientation
        is what decides whether a neighbour is upstream or downstream.
        """
        data = self._graph()
        name_kind = {
            v.get("name"): label
            for label, vs in data.get("vertices", {}).items()
            for v in vs
            if v.get("name")
        }
        adj: Dict[str, List[Tuple[str, str, str, bool]]] = {}
        for elabel, (src_label, dst_label) in _EDGE_ENDPOINTS.items():
            for src, dst in data.get("edges", {}).get(elabel, []):
                adj.setdefault(src, []).append((elabel, dst, name_kind.get(dst, dst_label), True))
                adj.setdefault(dst, []).append((elabel, src, name_kind.get(src, src_label), False))
        return adj

    def _kind(self, name: str) -> Optional[str]:
        data = self._graph()
        for label, vs in data.get("vertices", {}).items():
            for v in vs:
                if v.get("name") == name:
                    return label
        return None

    @staticmethod
    def _keep_up(elabel: str, is_forward: bool) -> bool:
        """Is ``neighbour`` an *upstream* (built-from) node of the current node?"""
        if elabel == "dependsOn":
            return is_forward          # node depends on neighbour -> neighbour upstream
        if elabel in ("computedFrom", "computedFromField"):
            return is_forward          # node computed from neighbour -> upstream
        if elabel == "hasColumn":
            return not is_forward      # node is the field, neighbour is its table
        return False

    @staticmethod
    def _keep_down(elabel: str, is_forward: bool) -> bool:
        """Is ``neighbour`` a *downstream* (depends-on / derived-from) node?"""
        if elabel == "dependsOn":
            return not is_forward      # neighbour depends on node -> downstream
        if elabel in ("computedFrom", "computedFromField"):
            return not is_forward      # neighbour metric uses node -> downstream
        if elabel == "hasColumn":
            return is_forward          # node is table, neighbour is its field
        return False

    # -- traversal -----------------------------------------------------------

    def _traverse(self, target: str, direction: str, max_depth: int = 12) -> LineageResult:
        result = LineageResult(target=target, direction=direction)
        if not target:
            return result
        adj = self._adjacency()
        if target not in adj:
            return result
        keep = self._keep_up if direction == "upstream" else self._keep_down

        visited: Set[str] = {target}
        queue: List[Tuple[str, int]] = [(target, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for elabel, neigh, neigh_kind, is_forward in adj.get(node, []):
                if not keep(elabel, is_forward) or neigh in visited:
                    continue
                visited.add(neigh)
                result.nodes.append(LineageNode(kind=neigh_kind, name=neigh, via=elabel))
                queue.append((neigh, depth + 1))
        return result

    # -- public API ----------------------------------------------------------

    def upstream(self, target: str, max_depth: int = 12) -> LineageResult:
        """Return what ``target`` is *built from* (tables, fields, metrics)."""
        return self._traverse(target, "upstream", max_depth)

    def downstream(self, target: str, max_depth: int = 12) -> LineageResult:
        """Return what *depends on / is derived from* ``target``."""
        return self._traverse(target, "downstream", max_depth)

    def explain(self, target: str) -> str:
        """Human-readable lineage for an NL2SQL explanation block."""
        kind = self._kind(target)
        if not kind:
            return f"血缘图中不存在 {target!r}，无法解释其来源。"
        up = self.upstream(target)
        down = self.downstream(target)
        lines = [f"【{kind}】{target} 的血缘："]
        if up.nodes:
            lines.append("  上游来源（构建自）：")
            for n in up.nodes:
                lines.append(f"    - {n.kind} {n.name}（via {n.via}）")
        else:
            lines.append("  上游来源：无（基础表/字段）")
        if down.nodes:
            lines.append("  下游消费（被依赖）：")
            for n in down.nodes:
                lines.append(f"    - {n.kind} {n.name}（via {n.via}）")
        else:
            lines.append("  下游消费：无")
        return "\n".join(lines)

    def tables_of_metric(self, metric: str) -> Set[str]:
        """Tables a metric can be computed from (for grounding the FROM/JOIN)."""
        out: Set[str] = set()
        data = self._graph()
        for src, dst in data.get("edges", {}).get("computedFrom", []):
            if src == metric:
                out.add(dst)
        field_owner = {dst: src for src, dst in data.get("edges", {}).get("hasColumn", [])}
        for src, dst in data.get("edges", {}).get("computedFromField", []):
            if src == metric and dst in field_owner:
                out.add(field_owner[dst])
        return out
