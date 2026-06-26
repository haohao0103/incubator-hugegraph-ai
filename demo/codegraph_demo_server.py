#!/usr/bin/env python3
"""
CodeGraph CTO Demo Server — 代码知识图谱交互式后端

API:
  GET  /api/stats          — 图统计
  GET  /api/nodes          — 节点列表 (支持 type/file 过滤)
  GET  /api/node/<id>      — 节点详情 + 代码片段
  GET  /api/graph          — 完整图数据 (供 D3.js 可视化)
  GET  /api/neighbors/<id> — 节点的邻居图
  GET  /api/traverse       — 多跳遍历 (?source=X&hops=N&direction=out|in|both)
  GET  /api/impact/<id>    — 影响分析 (反向调用链)
  GET  /api/search         — 搜索 (?q=keyword&type=name|code)
  GET  /api/callers/<id>   — 谁调用了这个函数
  GET  /api/callees/<id>   — 这个函数调用了谁
  GET  /api/hubs           — 中心节点 (Hub 检测)
"""

import json
import os
import sys
from collections import Counter, defaultdict
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# ─── Load pre-parsed data ──────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "codegraph_parsed.json")
with open(DATA_FILE, "r") as f:
    DATA = json.load(f)

nodes_list = DATA["nodes"]
edges_list = DATA["edges"]
stats = DATA.get("stats", {})

# Build index
node_index: dict = {n["id"]: n for n in nodes_list}

# Filter edges: keep only edges where both ends are in node_index
valid_edges = [e for e in edges_list
               if e["source"] in node_index and e["target"] in node_index]
print(f"  Filtered edges: {len(edges_list)} → {len(valid_edges)} "
      f"(removed {len(edges_list) - len(valid_edges)} referencing builtins/external)")

# Build adjacency: {node_id: {in: [edge, ...], out: [edge, ...]}}
adj = defaultdict(lambda: {"in": [], "out": []})
for e in valid_edges:
    adj[e["target"]]["in"].append(e)
    adj[e["source"]]["out"].append(e)

# File -> nodes index
file_nodes: dict = defaultdict(list)
for n in nodes_list:
    file_nodes[n["file_path"]].append(n)

# Compute hub score (in_degree + out_degree) from valid edges only
hub_score = defaultdict(int)
for e in valid_edges:
    hub_score[e["source"]] += 1
    hub_score[e["target"]] += 1
top_hubs = sorted(hub_score.items(), key=lambda x: x[1], reverse=True)

# ─── API Routes ────────────────────────────────────────────

@app.route("/api/stats")
def get_stats():
    """Overall graph statistics."""
    return jsonify({
        "total_nodes": len(nodes_list),
        "total_edges": len(valid_edges),
        "node_types": stats.get("node_types", {}),
        "edge_types": stats.get("edge_types", {}),
        "parsed_files": stats.get("parsed_files", 0),
        "hub_count": len([h for h in top_hubs if h[1] >= 10]),
        "top_hubs": [
            {"id": hid, "name": node_index[hid]["name"] if hid in node_index else "?",
             "score": score}
            for hid, score in top_hubs[:10]
        ],
    })


@app.route("/api/nodes")
def get_nodes():
    """List nodes with optional filters."""
    node_type = request.args.get("type")
    file_path = request.args.get("file")
    limit = int(request.args.get("limit", 200))

    result = nodes_list
    if node_type:
        result = [n for n in result if n["node_type"] == node_type]
    if file_path:
        result = [n for n in result if file_path in n["file_path"]]

    return jsonify(result[:limit])


@app.route("/api/node/<node_id>")
def get_node(node_id):
    """Get a single node with code preview and stats."""
    node = node_index.get(node_id)
    if not node:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404

    in_edges = adj[node_id]["in"]
    out_edges = adj[node_id]["out"]

    return jsonify({
        "node": node,
        "in_degree": len(in_edges),
        "out_degree": len(out_edges),
        "in_edges": [
            {"source": e["source"], "type": e["edge_type"],
             "source_name": node_index.get(e["source"], {}).get("name", "?")}
            for e in in_edges[:30]
        ],
        "out_edges": [
            {"target": e["target"], "type": e["edge_type"],
             "target_name": node_index.get(e["target"], {}).get("name", "?")}
            for e in out_edges[:30]
        ],
        "hub_score": hub_score.get(node_id, 0),
    })


@app.route("/api/graph")
def get_graph():
    """Get full or filtered graph data for visualization."""
    node_type = request.args.get("type")
    edge_type = request.args.get("edge_type")
    max_nodes = int(request.args.get("max_nodes", 500))

    nodes = nodes_list
    edges = list(valid_edges)

    if node_type:
        nodes = [n for n in nodes if n["node_type"] == node_type]
        node_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    if edge_type:
        edges = [e for e in edges if e["edge_type"] == edge_type]
        # Re-filter nodes to match edges
        connected = set()
        for e in edges:
            connected.add(e["source"])
            connected.add(e["target"])
        nodes = [n for n in nodes if n["id"] in connected]

    # Limit to top N by hub score
    if len(nodes) > max_nodes:
        top_ids = {hid for hid, _ in top_hubs[:max_nodes]}
        nodes = [n for n in nodes if n["id"] in top_ids][:max_nodes]
        node_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    return jsonify({
        "nodes": nodes,
        "edges": edges,
    })


@app.route("/api/neighbors/<node_id>")
def get_neighbors(node_id):
    """Get ego network for a node (1-hop neighborhood)."""
    if node_id not in node_index:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404

    related = set()
    related.add(node_id)
    for e in adj[node_id]["in"]:
        related.add(e["source"])
    for e in adj[node_id]["out"]:
        related.add(e["target"])

    sub_nodes = [n for n in nodes_list if n["id"] in related]
    sub_ids = {n["id"] for n in sub_nodes}
    sub_edges = [e for e in valid_edges
                 if e["source"] in sub_ids and e["target"] in sub_ids]

    return jsonify({"nodes": sub_nodes, "edges": sub_edges, "center": node_id})


@app.route("/api/traverse")
def traverse():
    """Multi-hop traversal from a source node."""
    source_id = request.args.get("source")
    hops = int(request.args.get("hops", 2))
    direction = request.args.get("direction", "out")  # out | in | both

    if source_id not in node_index:
        return jsonify({"error": f"Source '{source_id}' not found"}), 404

    visited = {source_id}
    frontier = {source_id}
    paths = [[source_id]]

    for _ in range(hops):
        new_frontier = set()
        new_paths = []
        for node in frontier:
            neighbors = []
            if direction in ("out", "both"):
                for e in adj[node]["out"]:
                    neighbors.append(e["target"])
            if direction in ("in", "both"):
                for e in adj[node]["in"]:
                    neighbors.append(e["source"])

            for nb in neighbors:
                if nb not in visited:
                    new_frontier.add(nb)
                    new_paths.append([node, nb])

        visited.update(new_frontier)
        frontier = new_frontier
        paths.extend(new_paths)

    sub_nodes = [n for n in nodes_list if n["id"] in visited]
    sub_ids = {n["id"] for n in sub_nodes}
    sub_edges = [e for e in valid_edges
                 if e["source"] in sub_ids and e["target"] in sub_ids]

    return jsonify({
        "nodes": sub_nodes,
        "edges": sub_edges,
        "source": source_id,
        "hops": hops,
        "direction": direction,
        "reachable_count": len(visited) - 1,
    })


@app.route("/api/impact/<node_id>")
def impact_analysis(node_id):
    """Impact analysis: what depends on this node? (reverse call chain)."""
    if node_id not in node_index:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404

    # BFS reverse: follow "in" edges recursively
    visited = {node_id}
    frontier = {node_id}
    levels = {node_id: 0}

    for depth in range(1, 6):  # max 5 levels
        new_frontier = set()
        for node in frontier:
            for e in adj[node]["in"]:
                if e["source"] not in visited:
                    visited.add(e["source"])
                    levels[e["source"]] = depth
                    new_frontier.add(e["source"])
        if not new_frontier:
            break
        frontier = new_frontier

    sub_nodes = [n for n in nodes_list if n["id"] in visited]
    sub_ids = {n["id"] for n in sub_nodes}
    sub_edges = [e for e in valid_edges
                 if e["source"] in sub_ids and e["target"] in sub_ids]

    # Count by level
    level_counts = Counter(levels.values())

    return jsonify({
        "nodes": sub_nodes,
        "edges": sub_edges,
        "target": node_id,
        "impact_depth": max(levels.values()),
        "impacted_count": len(visited) - 1,
        "level_counts": dict(level_counts),
        "levels": {k: v for k, v in levels.items() if k in node_index},
    })


@app.route("/api/search")
def search():
    """Search nodes by name keyword."""
    q = request.args.get("q", "").lower().strip()
    search_type = request.args.get("type", "name")  # name | code
    limit = int(request.args.get("limit", 50))

    if not q:
        return jsonify([])

    results = []
    for n in nodes_list:
        if search_type == "name":
            if q in n["name"].lower():
                results.append(n)
        elif search_type == "code":
            if q in n.get("source_code", "").lower():
                results.append(n)
        if len(results) >= limit:
            break

    return jsonify(results)


@app.route("/api/callers/<node_id>")
def get_callers(node_id):
    """Who calls this function?"""
    if node_id not in node_index:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404

    callers_list = []
    for e in adj[node_id]["in"]:
        if e["edge_type"] == "calls":
            caller = node_index.get(e["source"], {})
            callers_list.append({
                "id": e["source"],
                "name": caller.get("name", "?"),
                "node_type": caller.get("node_type", "?"),
                "file_path": caller.get("file_path", "?"),
            })

    return jsonify({
        "target": node_index[node_id],
        "callers": callers_list,
        "total": len(callers_list),
    })


@app.route("/api/callees/<node_id>")
def get_callees(node_id):
    """Who does this function call?"""
    if node_id not in node_index:
        return jsonify({"error": f"Node '{node_id}' not found"}), 404

    callees_list = []
    for e in adj[node_id]["out"]:
        if e["edge_type"] == "calls":
            callee = node_index.get(e["target"], {})
            callees_list.append({
                "id": e["target"],
                "name": callee.get("name", "?"),
                "node_type": callee.get("node_type", "?"),
                "file_path": callee.get("file_path", "?"),
            })

    return jsonify({
        "source": node_index[node_id],
        "callees": callees_list,
        "total": len(callees_list),
    })


@app.route("/api/hubs")
def get_hubs():
    """Get hub nodes sorted by connectivity."""
    limit = int(request.args.get("limit", 20))
    hubs = []
    for hid, score in top_hubs[:limit]:
        n = node_index.get(hid, {})
        hubs.append({
            "id": hid,
            "name": n.get("name", "?"),
            "node_type": n.get("node_type", "?"),
            "file_path": n.get("file_path", "?"),
            "hub_score": score,
        })
    return jsonify(hubs)


@app.route("/")
def index():
    """Serve the demo frontend."""
    return send_from_directory(
        os.path.dirname(__file__), "codegraph_demo.html"
    )


def main():
    port = int(os.environ.get("PORT", 5100))
    print(f"\n🚀 CodeGraph CTO Demo Server")
    print(f"   http://localhost:{port}")
    print(f"   Nodes: {len(nodes_list)}  |  Edges: {len(valid_edges)}")
    print(f"   Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
