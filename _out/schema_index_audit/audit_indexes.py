"""Read-only audit: what indexes exist on the kg_rag metadata KG, and what
the hot Gremlin lookups actually cost.

Answers:
1. current property keys + index labels on the live graph (REST schema API);
2. vertex counts per business label (Table/Field/Metric/Query) -- graph scale;
3. profile() of the hot lookup patterns the NL2SQL pipeline issues
   (label+name point lookups, Query domain scans) to quantify scan cost.

Read-only: never creates/drops anything.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, List

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hugegraph-llm", "src")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.disable(logging.CRITICAL)

from hugegraph_llm.config import huge_settings  # noqa: E402
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager  # noqa: E402
from pyhugegraph.client import PyHugeClient  # noqa: E402

GRAPH = os.environ.get("KG_E2E_GRAPH", "kg_rag")


def _client():
    return PyHugeClient(
        url=huge_settings.graph_url,
        graph=GRAPH,
        user=huge_settings.graph_user,
        pwd=huge_settings.graph_pwd,
        graphspace=huge_settings.graph_space,
    )


def _exec(client, q: str) -> List[Any]:
    resp = client.gremlin().exec(q)
    return resp.get("data") if isinstance(resp, dict) else (resp or [])


def _attr(obj: Any, *names: str) -> Any:
    """Read the first present attribute/property from a schema data object."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        props = getattr(obj, "properties", None)
        if isinstance(props, dict) and n in props:
            return props[n]
    return None


def main() -> int:
    try:
        client = _client()
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: cannot build client: {exc}")
        return 0

    try:
        _exec(client, "g.V().limit(1).count()")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: gremlin endpoint unreachable: {exc}")
        return 0

    mgr = SchemaManager(GRAPH, client=client)
    print(f"=== graph={GRAPH} ===\n")

    # 1) property keys
    print("--- property keys ---")
    pks = mgr.schema.getPropertyKeys() or []
    for pk in sorted(pks, key=lambda x: str(_attr(x, "name"))):
        print(f"  {_attr(pk, 'name'):<24} type={_attr(pk, 'data_type')} "
              f"card={_attr(pk, 'cardinality')} status={_attr(pk, 'status')}")
    if not pks:
        print("  (none)")

    # 2) index labels
    print("\n--- index labels ---")
    indexes = mgr.list_indexes()
    if indexes:
        for idx in indexes:
            print(
                f"  {idx['name']:<28} base={idx['base_type']}:{idx['base_value']} "
                f"type={idx['index_type']} fields={idx['fields']}"
            )
    else:
        print("  (none -- every has(label,key,value) lookup is a full scan)")

    # 3) vertex labels present
    print("\n--- vertex labels ---")
    vls = mgr.schema.getVertexLabels() or []
    for vl in sorted(vls, key=lambda x: str(_attr(x, "name"))):
        props = _attr(vl, "properties") or []
        names = ",".join(str(p) for p in props)
        print(f"  {_attr(vl, 'name'):<20} id_strategy={_attr(vl, 'id_strategy', 'idStrategy')} "
              f"pk={_attr(vl, 'primary_keys', 'primaryKeys')} properties=[{names}]")

    # 3b) confirm whether non-PK property queries are rejected without index
    print("\n--- probe: non-indexed property query (expect reject) ---")
    for label, key, val in (("Table", "comment", "订单表"), ("Query", "question", "各城市订单总额")):
        try:
            n = _exec(client, f"g.V().has('{label}','{key}','{val}').count()")
            print(f"  has('{label}','{key}',...) -> OK count={n[0] if n else 0}")
        except Exception as exc:
            print(f"  has('{label}','{key}',...) -> REJECTED: {str(exc)[:140]}")

    # 4) business label vertex counts
    print("\n--- vertex counts (business labels) ---")
    for label in ("Table", "Field", "Metric", "Query"):
        try:
            n = _exec(client, f"g.V().hasLabel('{label}').count()")
            print(f"  {label:<8} {n[0] if n else 0}")
        except Exception as exc:
            print(f"  {label:<8} ERR {exc}")

    # 5) profile of the hot lookups (whether an index is used or a full scan)
    print("\n--- profile: hot lookup patterns ---")
    probes = [
        ("point: has('Table','name','order')",
         "g.V().has('Table','name','order').count()"),
        ("point: has('Field','name','order.amount')",
         "g.V().has('Field','name','order.amount').count()"),
        ("point: has('Metric','name','order_total')",
         "g.V().has('Metric','name','order_total').count()"),
        ("point: has('Query','domain','demo_golden')",
         "g.V().has('Query','domain','demo_golden').count()"),
        ("label scan: hasLabel('Table').elementMap()",
         "g.V().hasLabel('Table').count()"),
    ]
    for label, q in probes:
        try:
            rows = _exec(client, q + ".profile()")
            print(f"\n  [{label}]")
            print(f"    query: {q}")
            for r in rows:
                print("    " + str(r)[:500])
        except Exception as exc:
            print(f"\n  [{label}] ERR: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
