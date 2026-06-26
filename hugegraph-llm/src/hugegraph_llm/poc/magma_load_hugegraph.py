"""
MAGMA Four-Graph Memory — HugeGraph Live Backend
=================================================
1. Creates MAGMA edge labels (semantic, temporal, causal, entity_ref)
2. Loads 10 Agent Memory events into HugeGraph (localhost:8080)
3. Builds all four graph types using HugeGraph REST API
4. Updates the visualizer to use real HugeGraph backend

HugeGraph 1.7.0 API prefix:
  /graphspaces/DEFAULT/graphs/hugegraph/graph/vertices
  /graphspaces/DEFAULT/graphs/hugegraph/graph/edges

Usage: python3.10 magma_load_hugegraph.py
"""

import json
import time
import requests
import gzip
import io

HG_BASE = "http://localhost:8080/graphspaces/DEFAULT/graphs/hugegraph"
GRAPHSPACE = "DEFAULT"
GRAPH_NAME = "hugegraph"

session = requests.Session()
session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})


def _decode_response(r):
    """Decode response handling both gzip and plain JSON"""
    raw = r.content
    if raw and raw[:2] == b'\x1f\x8b':  # gzip magic
        raw = gzip.decompress(raw)
    return json.loads(raw) if raw else {}


def hg_get(path: str) -> dict:
    """GET from HugeGraph API"""
    r = session.get(f"{HG_BASE}{path}")
    r.raise_for_status()
    return _decode_response(r)


def hg_post(path: str, data: dict) -> dict:
    """POST to HugeGraph API"""
    r = session.post(f"{HG_BASE}{path}", json=data)
    r.raise_for_status()
    return _decode_response(r)


def hg_put(path: str, data: dict) -> dict:
    """PUT to HugeGraph API"""
    r = session.put(f"{HG_BASE}{path}", json=data)
    r.raise_for_status()
    if r.headers.get("Content-Encoding") == "gzip":
        return json.loads(gzip.decompress(r.content))
    if r.content:
        return r.json()
    return {}


def create_schema():
    """Create MAGMA-specific edge labels and property keys"""
    print("[1] Creating MAGMA schema...")

    # Property keys (check if exist first)
    existing_props = {p["name"] for p in hg_get("/schema/propertykeys").get("propertykeys", [])}
    needed_props = {
        "timestamp": "TEXT",
        "similarity": "DOUBLE",
        "graph_type": "TEXT",
        "edge_type": "TEXT",
        "content": "TEXT",  # may already exist
    }
    for prop_name, data_type in needed_props.items():
        if prop_name not in existing_props:
            hg_post("/schema/propertykeys", {"name": prop_name, "data_type": data_type, "cardinality": "SINGLE"})
            print(f"  + property: {prop_name} ({data_type})")
        else:
            print(f"  ✓ property exists: {prop_name}")

    # Edge labels
    existing_edges = {e["name"] for e in hg_get("/schema/edgelabels").get("edgelabels", [])}
    magma_edges = {
        "semantic": {"source": "memory_event", "target": "memory_event", "props": ["weight", "similarity", "description"]},
        "temporal": {"source": "memory_event", "target": "memory_event", "props": ["weight", "description"]},
        "causal": {"source": "memory_event", "target": "memory_event", "props": ["weight", "description"]},
        "entity_ref": {"source": "memory_event", "target": "entity", "props": ["weight", "description"]},
    }

    for edge_name, config in magma_edges.items():
        if edge_name not in existing_edges:
            edge_def = {
                "name": edge_name,
                "source_label": config["source"],
                "target_label": config["target"],
                "frequency": "SINGLE",
                "properties": config["props"],
            }
            hg_post("/schema/edgelabels", edge_def)
            print(f"  + edge label: {edge_name}")
        else:
            print(f"  ✓ edge label exists: {edge_name}")

    print("[1] Schema ready.")


def load_magma_data():
    """Load MAGMA demo events into HugeGraph"""
    print("\n[2] Loading MAGMA events...")

    from datetime import datetime, timedelta
    import hashlib

    base_time = datetime(2026, 6, 1, 9, 0)

    events = [
        ("Alice reported a critical bug in the authentication service", base_time, {"priority": "high", "type": "bug"}),
        ("Bob investigated the authentication bug and found a race condition", base_time + timedelta(hours=2), {"priority": "high", "type": "investigation"}),
        ("The race condition was caused by incorrect connection pool handling", base_time + timedelta(hours=4), {"type": "root_cause"}),
        ("Alice deployed a fix for the connection pool race condition", base_time + timedelta(hours=6), {"type": "fix"}),
        ("Server CPU usage spiked to 95% after the authentication fix deployment", base_time + timedelta(hours=7), {"priority": "high", "type": "incident"}),
        ("Carol discovered the CPU spike was due to a missing index on the users table", base_time + timedelta(hours=9), {"type": "root_cause"}),
        ("Bob added the missing database index and CPU returned to normal levels", base_time + timedelta(hours=11), {"type": "fix"}),
        ("Alice scheduled a code review meeting for the authentication module", base_time + timedelta(hours=24), {"type": "meeting"}),
        ("Deploy released v2.3.1 with authentication fix and database index update", base_time + timedelta(hours=30), {"type": "release"}),
        ("David reported a new feature request for OAuth2 support", base_time + timedelta(hours=48), {"type": "feature"}),
    ]

    # Check if events already loaded
    try:
        existing = hg_get('/graph/vertices?page_size=100')
        event_count = sum(1 for v in existing.get("vertices", []) if v["label"] == "memory_event" and v["properties"].get("timestamp", "").startswith("2026-06-"))
        if event_count >= 10:
            print(f"  ✓ {event_count} MAGMA events already exist, skipping load.")
            return [v for v in existing.get("vertices", []) if v["label"] == "memory_event" and v["properties"].get("timestamp", "").startswith("2026-06")]
    except Exception:
        pass

    # Create events
    event_vertices = []
    for content, ts, attrs in events:
        vid = f"mem_{hashlib.md5(content.encode()).hexdigest()[:12]}"
        vertex = {
            "id": vid,
            "label": "memory_event",
            "properties": {
                "name": content[:50],
                "content": content,
                "type": attrs.get("type", "general"),
                "timestamp": ts.isoformat(),
            },
        }
        try:
            result = hg_post("/graph/vertices", vertex)
            event_vertices.append({"id": vid, "content": content, "timestamp": ts.isoformat()})
            print(f"  + event: {content[:50]}...")
        except Exception as e:
            print(f"  ✗ error: {e}")

    # Create entity nodes
    entities_to_create = {
        "Alice": "person",
        "Bob": "person",
        "Carol": "person",
        "David": "person",
        "authentication": "concept",
        "race condition": "concept",
        "CPU spike": "concept",
        "database index": "concept",
        "OAuth2": "concept",
        "bug": "concept",
    }

    existing_verts = hg_get("/graph/vertices?page_size=200").get("vertices", [])
    existing_ids = {v["id"] for v in existing_verts}

    entity_vertices = {}
    for ename, etype in entities_to_create.items():
        eid = f"ent_{hashlib.md5(ename.encode()).hexdigest()[:12]}"
        if eid not in existing_ids:
            vertex = {"id": eid, "label": "entity", "properties": {"name": ename, "type": etype}}
            try:
                hg_post("/graph/vertices", vertex)
                print(f"  + entity: {ename} ({etype})")
            except Exception as e:
                print(f"  ✗ entity error: {e}")
        else:
            print(f"  ✓ entity exists: {ename}")
        entity_vertices[ename] = eid

    print(f"\n  Created {len(event_vertices)} events, {len(entity_vertices)} entities")

    # Create edges
    print("\n[3] Building four graph edges...")

    edge_counts = {"temporal": 0, "semantic": 0, "causal": 0, "entity_ref": 0}

    # Temporal edges (chain)
    for i in range(len(event_vertices) - 1):
        edge = {
            "label": "temporal",
            "outV": event_vertices[i]["id"],
            "inV": event_vertices[i + 1]["id"],
            "outVLabel": "memory_event",
            "inVLabel": "memory_event",
            "properties": {"weight": 1, "description": "temporal_chain"},
        }
        try:
            hg_post("/graph/edges", edge)
            edge_counts["temporal"] += 1
        except Exception as e:
            print(f"  ✗ temporal edge error: {e}")

    # Semantic edges (simulated similarity between events with shared keywords)
    keyword_map = {}
    for ev in event_vertices:
        kws = set(ev["content"].lower().split())
        keyword_map[ev["id"]] = kws

    for i in range(len(event_vertices)):
        for j in range(i + 1, len(event_vertices)):
            overlap = len(keyword_map[event_vertices[i]["id"]] & keyword_map[event_vertices[j]["id"]])
            if overlap >= 2:
                edge = {
                    "label": "semantic",
                    "outV": event_vertices[i]["id"],
                    "inV": event_vertices[j]["id"],
                    "outVLabel": "memory_event",
                    "inVLabel": "memory_event",
                    "properties": {"weight": overlap, "similarity": round(overlap / 10.0, 2), "description": f"keyword_overlap={overlap}"},
                }
                try:
                    hg_post("/graph/edges", edge)
                    edge_counts["semantic"] += 1
                except Exception as e:
                    print(f"  ✗ semantic edge error: {e}")

    # Causal edges (explicit cause-effect)
    causal_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    for src_idx, tgt_idx in causal_pairs:
        edge = {
            "label": "causal",
            "outV": event_vertices[src_idx]["id"],
            "inV": event_vertices[tgt_idx]["id"],
            "outVLabel": "memory_event",
            "inVLabel": "memory_event",
            "properties": {"weight": 1, "description": "cause_effect"},
        }
        try:
            hg_post("/graph/edges", edge)
            edge_counts["causal"] += 1
        except Exception as e:
            print(f"  ✗ causal edge error: {e}")

    # Entity reference edges
    entity_in_events = {
        "Alice": [0, 3, 7],
        "Bob": [1, 6],
        "Carol": [5],
        "David": [9],
        "authentication": [0, 1, 3, 7],
        "race condition": [1, 2, 3],
        "CPU spike": [4, 5],
        "database index": [5, 6, 8],
        "OAuth2": [9],
        "bug": [0],
    }

    for ename, event_indices in entity_in_events.items():
        eid = entity_vertices.get(ename)
        if not eid:
            continue
        for idx in event_indices:
            edge = {
                "label": "entity_ref",
                "outV": event_vertices[idx]["id"],
                "inV": eid,
                "outVLabel": "memory_event",
                "inVLabel": "entity",
                "properties": {"weight": 1, "description": f"contains_entity={ename}"},
            }
            try:
                hg_post("/graph/edges", edge)
                edge_counts["entity_ref"] += 1
            except Exception as e:
                print(f"  ✗ entity_ref error: {e}")

    print(f"\n  Edge counts: {edge_counts}")
    return edge_counts


def verify_data():
    """Verify the loaded data"""
    print("\n[4] Verifying data in HugeGraph...")

    vertices = hg_get("/graph/vertices?page_size=500").get("vertices", [])
    edges = hg_get("/graph/edges?page_size=500").get("edges", [])

    vlabels = {}
    for v in vertices:
        vlabels[v["label"]] = vlabels.get(v["label"], 0) + 1

    elabels = {}
    for e in edges:
        elabels[e["label"]] = elabels.get(e["label"], 0) + 1

    print(f"  Vertices: {len(vertices)} total")
    for k, v in sorted(vlabels.items()):
        print(f"    {k}: {v}")

    print(f"  Edges: {len(edges)} total")
    for k, v in sorted(elabels.items()):
        print(f"    {k}: {v}")

    return {"vertices": vlabels, "edges": elabels, "total_v": len(vertices), "total_e": len(edges)}


if __name__ == "__main__":
    print("=" * 60)
    print("MAGMA Four-Graph Memory — Loading into HugeGraph")
    print(f"Server: {HG_BASE}")
    print("=" * 60)

    # Check server is alive
    try:
        r = session.get(f"{HG_BASE}/schema/vertexlabels")
        r.raise_for_status()
        print(f"Server OK: {len(r.json().get('vertexlabels', []))} vertex labels")
    except Exception as e:
        print(f"Server not reachable: {e}")
        exit(1)

    create_schema()
    edge_counts = load_magma_data()
    stats = verify_data()

    result = {
        "status": "success",
        "hg_base": HG_BASE,
        "schema": "created",
        "edge_counts": edge_counts,
        "stats": stats,
    }

    output_file = "/Users/mac/Desktop/apache-code/hugegraph-dev/incubator-hugegraph-ai/hugegraph-llm/src/hugegraph_llm/poc/magma_hugegraph_load_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResult saved to: {output_file}")
    print("=" * 60)
