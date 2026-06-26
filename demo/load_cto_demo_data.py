#!/usr/bin/env python3
"""
CTO Demo Data Loader — 将业界数据集导入 HugeGraph

数据集:
  1. OpenFlights: 全球机场 + 航线网络 (交通/物流场景)
  2. Wiki-Vote: Wikipedia 管理员投票网络 (社交网络分析场景)

Usage:
  python load_cto_demo_data.py [--graph GRAPH_NAME] [--skip-schema]
"""

import csv
import json
import sys
import os
import time
import argparse

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
from collections import defaultdict

# ─── Config ─────────────────────────────────────────────────
HG_BASE = "http://localhost:8080"
GRAPH_NAME = "cto_demo"
DEMO_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'demo_data')

# ─── HugeGraph REST helpers ─────────────────────────────────

def hg_request(method, path, json_data=None):
    """Send request to HugeGraph REST API."""
    url = f"{HG_BASE}/graphs/{GRAPH_NAME}/{path.lstrip('/')}" if not path.startswith("http") else path
    headers = {"Content-Type": "application/json"}
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=json_data, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, headers=headers, json=json_data, timeout=30)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unknown method: {method}")

        if resp.status_code >= 400:
            print(f"  ⚠ {method} {path} → {resp.status_code}: {resp.text[:200]}")
            return None
        if resp.status_code == 204:
            return None
        return resp.json() if resp.text else None
    except Exception as e:
        print(f"  ✗ {method} {path} → {e}")
        return None


def create_graph():
    """Create the graph space."""
    # Check if already exists
    resp = requests.get(f"{HG_BASE}/graphs/{GRAPH_NAME}", timeout=10)
    if resp.status_code == 200:
        print(f"  Graph '{GRAPH_NAME}' already exists, clearing...")
        hg_request("DELETE", f"{HG_BASE}/graphs/{GRAPH_NAME}/clear")
        time.sleep(2)
        return True

    payload = {
        "graph_name": GRAPH_NAME,
        "backend": "rocksdb",
    }
    resp = requests.post(f"{HG_BASE}/graphs", json=payload, timeout=10)
    if resp.status_code in (201, 202):
        print(f"  ✓ Created graph '{GRAPH_NAME}'")
        time.sleep(1)
        return True
    elif resp.status_code == 409:  # Already exists
        print(f"  Graph '{GRAPH_NAME}' already exists")
        return True
    else:
        print(f"  ✗ Failed to create graph: {resp.status_code} {resp.text}")
        return False


def create_schema():
    """Create vertex labels, edge labels, and property keys."""
    print("\n── Creating Schema ──")

    # Property keys
    pk_types = {
        "name": "TEXT",
        "city": "TEXT",
        "country": "TEXT",
        "iata": "TEXT",
        "icao": "TEXT",
        "latitude": "DOUBLE",
        "longitude": "DOUBLE",
        "altitude": "INT",
        "timezone": "TEXT",
        "airline_name": "TEXT",
        "airline_iata": "TEXT",
        "airline_country": "TEXT",
        "route_equipment": "TEXT",
        "stops": "INT",
        "vote_count": "INT",
        "degree": "INT",
        "betweenness": "DOUBLE",
        "pagerank": "DOUBLE",
        "community": "INT",
        "type": "TEXT",
        "description": "TEXT",
        "dataset": "TEXT",
    }

    for pk_name, pk_type in pk_types.items():
        payload = {
            "name": pk_name,
            "data_type": pk_type,
            "cardinality": "SINGLE",
        }
        resp = hg_request("POST", "schema/propertykeys", payload)
        if resp is not None:
            print(f"  ✓ PropertyKey: {pk_name} ({pk_type})")
        else:
            # Try to get existing
            existing = hg_request("GET", f"schema/propertykeys/{pk_name}")
            if existing:
                print(f"  - PropertyKey '{pk_name}' already exists")

    # Vertex labels
    vertex_labels = [
        # OpenFlights
        ("airport", "id", ["name", "city", "country", "iata", "icao",
         "latitude", "longitude", "altitude", "timezone", "dataset"]),
        ("airline", "id", ["airline_name", "airline_iata", "airline_country", "dataset"]),
        # Wiki-Vote
        ("user", "id", ["vote_count", "degree", "betweenness", "pagerank",
         "community", "dataset"]),
    ]

    for vl_name, id_strategy, properties in vertex_labels:
        payload = {
            "name": vl_name,
            "id_strategy": "PRIMARY_KEY",
            "primary_keys": ["id"],
            "properties": properties,
            "nullable_keys": properties,
        }
        resp = hg_request("POST", "schema/vertexlabels", payload)
        if resp:
            print(f"  ✓ VertexLabel: {vl_name} ({len(properties)} props)")
        else:
            print(f"  - VertexLabel '{vl_name}' may already exist")

    # Edge labels
    edge_labels = [
        ("route", "airport", "airport", ["route_equipment", "stops", "dataset"]),
        ("vote", "user", "user", ["dataset"]),
    ]

    for el_name, src, tgt, properties in edge_labels:
        payload = {
            "name": el_name,
            "source_label": src,
            "target_label": tgt,
            "properties": properties,
            "nullable_keys": properties,
        }
        resp = hg_request("POST", "schema/edgelabels", payload)
        if resp:
            print(f"  ✓ EdgeLabel: {el_name} ({src}→{tgt})")
        else:
            print(f"  - EdgeLabel '{el_name}' may already exist")

    print("  Schema creation complete.")


def load_openflights():
    """Load OpenFlights airports + routes + airlines data."""
    print("\n── Loading OpenFlights Dataset ──")

    # Parse airports
    airports = []
    with open(os.path.join(DEMO_DATA_DIR, "airports.dat"), "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) >= 8:
                try:
                    airports.append({
                        "id": parts[0],
                        "label": "airport",
                        "type": "vertex",
                        "properties": {
                            "name": parts[1],
                            "city": parts[2],
                            "country": parts[3],
                            "iata": parts[4] if parts[4] != "\\N" else "",
                            "icao": parts[5] if parts[5] != "\\N" else "",
                            "latitude": float(parts[6]) if parts[6] and parts[6] != "\\N" else 0.0,
                            "longitude": float(parts[7]) if parts[7] and parts[7] != "\\N" else 0.0,
                            "altitude": int(float(parts[8])) if len(parts) > 8 and parts[8] and parts[8] != "\\N" else 0,
                            "timezone": parts[9] if len(parts) > 9 else "",
                            "dataset": "openflights",
                        }
                    })
                except (ValueError, IndexError):
                    continue

    print(f"  Parsed {len(airports)} airports")

    # Parse airlines  
    airlines = []
    with open(os.path.join(DEMO_DATA_DIR, "airlines.dat"), "r", encoding="utf-8") as f:
        airline_ids = set()
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) >= 3 and parts[0] not in airline_ids:
                airline_ids.add(parts[0])
                airlines.append({
                    "id": f"AL_{parts[0]}",
                    "label": "airline",
                    "type": "vertex",
                    "properties": {
                        "airline_name": parts[1],
                        "airline_iata": parts[2] if parts[2] != "\\N" else "",
                        "airline_country": parts[3] if len(parts) > 3 and parts[3] != "\\N" else "",
                        "dataset": "openflights",
                    }
                })

    print(f"  Parsed {len(airlines)} airlines")

    # Batch insert airports
    batch_size = 500
    total_airports = 0
    for i in range(0, len(airports), batch_size):
        batch = airports[i:i+batch_size]
        resp = hg_request("POST", "graph/vertices/batch", batch)
        if resp is not None:
            total_airports += len(batch)
        else:
            print(f"  ⚠ Batch insert airports failed at {i}")

    print(f"  ✓ Loaded {total_airports} airports")

    # Batch insert airlines
    total_airlines = 0
    for i in range(0, len(airlines), batch_size):
        batch = airlines[i:i+batch_size]
        resp = hg_request("POST", "graph/vertices/batch", batch)
        if resp is not None:
            total_airlines += len(batch)
    print(f"  ✓ Loaded {total_airlines} airlines")

    # Parse and load routes
    routes = []
    airport_ids = {a["id"] for a in airports}
    with open(os.path.join(DEMO_DATA_DIR, "routes.dat"), "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) >= 5:
                src_id = parts[3] if parts[3] != "\\N" else ""
                tgt_id = parts[5] if parts[5] != "\\N" else ""
                if src_id in airport_ids and tgt_id in airport_ids:
                    routes.append({
                        "label": "route",
                        "type": "edge",
                        "outV": src_id,
                        "inV": tgt_id,
                        "outVLabel": "airport",
                        "inVLabel": "airport",
                        "properties": {
                            "route_equipment": parts[8] if len(parts) > 8 and parts[8] != "\\N" else "",
                            "stops": int(parts[7]) if len(parts) > 7 and parts[7] and parts[7] != "\\N" else 0,
                            "dataset": "openflights",
                        }
                    })

    print(f"  Parsed {len(routes)} routes")

    total_routes = 0
    for i in range(0, len(routes), batch_size):
        batch = routes[i:i+batch_size]
        resp = hg_request("POST", "graph/edges/batch", batch)
        if resp is not None:
            total_routes += len(batch)

    print(f"  ✓ Loaded {total_routes} routes")


def load_wiki_vote():
    """Load Wikipedia voting network (subset for demo: top nodes)."""
    print("\n── Loading Wiki-Vote Dataset ──")

    # Parse edges, count degrees
    edges = []
    node_degrees = defaultdict(int)
    node_set = set()

    with open(os.path.join(DEMO_DATA_DIR, "wiki-vote.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                src, tgt = parts[0], parts[1]
                edges.append((src, tgt))
                node_degrees[src] += 1
                node_degrees[tgt] += 1
                node_set.add(src)
                node_set.add(tgt)

    print(f"  Full dataset: {len(node_set)} users, {len(edges)} votes")

    # Take top 2000 nodes by degree for interactive demo
    top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:2000]
    top_node_ids = {n[0] for n in top_nodes}

    # Filter edges where both nodes are in top set
    filtered_edges = [(s, t) for s, t in edges if s in top_node_ids and t in top_node_ids]
    filtered_nodes = set()
    for s, t in filtered_edges:
        filtered_nodes.add(s)
        filtered_nodes.add(t)

    print(f"  Demo subset: {len(filtered_nodes)} users, {len(filtered_edges)} votes")

    # Compute simple metrics
    out_degrees = defaultdict(int)
    in_degrees = defaultdict(int)
    for s, t in filtered_edges:
        out_degrees[s] += 1
        in_degrees[t] += 1

    # Load users
    users = []
    for node_id in filtered_nodes:
        users.append({
            "id": node_id,
            "label": "user",
            "type": "vertex",
            "properties": {
                "degree": node_degrees[node_id],
                "vote_count": out_degrees.get(node_id, 0),
                "pagerank": 0.0,
                "community": 0,
                "dataset": "wiki-vote",
            }
        })

    batch_size = 500
    total_users = 0
    for i in range(0, len(users), batch_size):
        batch = users[i:i+batch_size]
        resp = hg_request("POST", "graph/vertices/batch", batch)
        if resp is not None:
            total_users += len(batch)
    print(f"  ✓ Loaded {total_users} users")

    # Load votes
    votes = []
    for src, tgt in filtered_edges:
        votes.append({
            "label": "vote",
            "type": "edge",
            "outV": src,
            "inV": tgt,
            "outVLabel": "user",
            "inVLabel": "user",
            "properties": {
                "dataset": "wiki-vote",
            }
        })

    total_votes = 0
    for i in range(0, len(votes), batch_size):
        batch = votes[i:i+batch_size]
        resp = hg_request("POST", "graph/edges/batch", batch)
        if resp is not None:
            total_votes += len(batch)
    print(f"  ✓ Loaded {total_votes} votes")


def verify():
    """Verify data load."""
    print("\n── Verification ──")
    counts = {}
    for label in ["airport", "airline", "user"]:
        resp = hg_request("GET", f"graph/vertices?label={label}&limit=1")
        if resp:
            counts[label] = len(resp.get("vertices", []))
            # Get total count via count API
            cresp = requests.get(
                f"{HG_BASE}/graphs/{GRAPH_NAME}/graph/vertices/count?label={label}",
                timeout=10
            )
            if cresp.status_code == 200:
                counts[label] = cresp.json().get("count", 0)

    for label in ["route", "vote"]:
        cresp = requests.get(
            f"{HG_BASE}/graphs/{GRAPH_NAME}/graph/edges/count?label={label}",
            timeout=10
        )
        if cresp.status_code == 200:
            counts[label] = cresp.json().get("count", 0)

    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Run a demo Gremlin query
    print("\n  Test Gremlin: 3-hop from PEK (Beijing)...")
    gremlin = "g.V().has('airport','iata','PEK').repeat(out('route')).times(3).path().limit(5)"
    payload = {"gremlin": gremlin}
    resp = requests.post(
        f"{HG_BASE}/graphs/{GRAPH_NAME}/gremlin",
        json=payload,
        timeout=30
    )
    if resp.status_code == 200:
        result = resp.json()
        paths = len(result.get("data", []))
        print(f"  ✓ Found {paths} 3-hop paths from PEK")
    else:
        print(f"  ⚠ Gremlin query failed: {resp.status_code}")

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=GRAPH_NAME, help="Graph name")
    parser.add_argument("--skip-schema", action="store_true", help="Skip schema creation")
    args = parser.parse_args()

    global GRAPH_NAME
    GRAPH_NAME = args.graph

    print("=" * 60)
    print("  HugeGraph CTO Demo — Data Loader")
    print("=" * 60)

    # Step 1: Create graph
    print("\n── Graph Setup ──")
    if not create_graph():
        print("  ✗ Failed to setup graph, aborting.")
        return

    # Step 2: Schema
    if not args.skip_schema:
        create_schema()
    else:
        print("\n  ⏭  Skipping schema creation (--skip-schema)")

    # Step 3: Load data
    load_openflights()
    load_wiki_vote()

    # Step 4: Verify
    counts = verify()

    print(f"\n{'=' * 60}")
    print(f"  ✓ Demo data loaded successfully!")
    print(f"  Total vertices: {sum(counts.get(l, 0) for l in ['airport', 'airline', 'user'])}")
    print(f"  Total edges: {sum(counts.get(l, 0) for l in ['route', 'vote'])}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
