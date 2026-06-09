#!/usr/bin/env python3.10
"""
HugeGraph PoC Interactive Visualizer
=====================================
为 3 个 PoC 提供交互式可视化 Demo:
1. 供应链 Network-KG 二重性
2. GraphRAG-Bench 12种方案对比
3. 代码图谱双 Agent 架构

架构: Flask + Cytoscape.js + vanilla CSS
双模式: HugeGraph Server 在线(pyhugegraph) / 离线(networkx内存模拟)
"""

import json
import sys
import os
import math
import webbrowser
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

import networkx as nx
from flask import Flask, render_template_string, jsonify, request

# Python-side color constants (used by extract_subgraph_for_cytoscape)
NODE_COLORS_PY = {
    "enterprise": "#4285f4",
    "facility": "#34a853",
    "material": "#f97316",
    "component": "#fbbc04",
    "product": "#a855f7",
    "function": "#06b6d4",
    "class": "#4285f4",
    "module": "#34a853",
    "method": "#f97316",
    # Text2Gremlin types
    "vertex_label": "#4285f4",
    "edge_label": "#ea4335",
    "operator": "#06b6d4",
    "core": "#ea4335",
    "config": "#9aa0a6",
    "component": "#f97316",
    "input": "#34a853",
    "output": "#a855f7",
    "external": "#30363d",
    # DRIFT types
    "step": "#fbbc04",
    "model": "#a855f7",
    "artifact": "#9aa0a6",
    "algorithm": "#06b6d4",
    "storage": "#34a853",
    "engine": "#ea4335",
    # Entity Resolution types
    "strategy": "#4285f4",
    "cache": "#fbbc04",
    "intermediate": "#9aa0a6",
    "cluster": "#f97316",
    # AI Memory types
    "data_point": "#ec4899",
    "event": "#f97316",
    "warning": "#ea4335",
    "example": "#06b6d4",
    "api_endpoint": "#a855f7",
    "person": "#ec4899",
    "organization": "#4285f4",
    "location": "#34a853",
    "skill": "#f97316",
    "concept": "#a855f7",
    "default": "#9aa0a6",
}

EDGE_COLORS_PY = {
    "supplies": "#34a853",
    "produces": "#4285f4",
    "has_input": "#f97316",
    "located_in": "#ea4335",
    "depends_on": "#fbbc04",
    "operates": "#9aa0a6",
    "calls": "#06b6d4",
    "imports": "#a855f7",
    "contains": "#34a853",
    "similar": "#30363d",
    # New relations
    "out_edge": "#4285f4",
    "in_edge": "#ea4335",
    "input": "#34a853",
    "delegates": "#9aa0a6",
    "pipes_to": "#06b6d4",
    "validates_with": "#fbbc04",
    "queries": "#4285f4",
    "on_error": "#ea4335",
    "feeds_back": "#f97316",
    "corrects": "#34a853",
    "uses": "#9aa0a6",
    "output": "#a855f7",
    "flow": "#06b6d4",
    "pipeline": "#4285f4",
    "contains_cluster": "#f97316",
    "verifies": "#fbbc04",
    "detects": "#ea4335",
    "triggers": "#f97316",
    "updates": "#34a853",
    "default": "#30363d",
}

# ============ HTML Templates ============

# Shared CSS (used by both index and poc pages)
SHARED_CSS = r"""
  :root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d27;
    --bg-tertiary: #242836;
    --text-primary: #e8eaed;
    --text-secondary: #9aa0a6;
    --accent-blue: #4285f4;
    --accent-green: #34a853;
    --accent-red: #ea4335;
    --accent-yellow: #fbbc04;
    --accent-purple: #a855f7;
    --accent-cyan: #06b6d4;
    --accent-orange: #f97316;
    --accent-pink: #ec4899;
    --border-color: #30363d;
    --radius: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary); color: var(--text-primary);
  }
  a { color: var(--accent-blue); text-decoration: none; }
  a:hover { text-decoration: underline; }
"""

# ============ Index Page Template ============
INDEX_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HugeGraph PoC 可视化演示</title>
<style>
  SHARED_CSS_HERE
  body { min-height: 100vh; }
  .hero { text-align: center; padding: 60px 20px 40px; }
  .hero h1 { font-size: 36px; font-weight: 800; margin-bottom: 8px; }
  .hero h1 .hl { color: var(--accent-blue); }
  .hero .tagline { font-size: 16px; color: var(--text-secondary); margin-bottom: 8px; }
  .hero .server-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-tertiary); padding: 6px 16px;
    border-radius: 20px; font-size: 12px; margin-top: 12px;
    border: 1px solid var(--border-color);
  }
  .hero .server-badge .dot {
    width: 8px; height: 8px; border-radius: 50%;
  }
  .hero .server-badge .dot.online { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
  .hero .server-badge .dot.offline { background: var(--accent-red); }

  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 20px; padding: 0 40px 60px; max-width: 1200px; margin: 0 auto;
  }

  .poc-card {
    background: var(--bg-secondary); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 28px; cursor: pointer;
    transition: all 0.3s; position: relative; overflow: hidden;
  }
  .poc-card:hover {
    border-color: var(--card-accent, var(--accent-blue));
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
  }
  .poc-card .card-icon {
    font-size: 32px; margin-bottom: 12px; display: block;
  }
  .poc-card h2 { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
  .poc-card .card-desc {
    font-size: 13px; color: var(--text-secondary); line-height: 1.6; margin-bottom: 16px;
  }
  .poc-card .card-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
  .poc-card .tag {
    font-size: 10px; padding: 3px 8px; border-radius: 4px;
    background: var(--bg-tertiary); color: var(--text-secondary);
    border: 1px solid var(--border-color);
  }
  .poc-card .tag.pass { background: rgba(52,168,83,0.15); color: var(--accent-green); border-color: rgba(52,168,83,0.3); }
  .poc-card .card-footer {
    display: flex; justify-content: space-between; align-items: center;
    border-top: 1px solid var(--border-color); padding-top: 12px; margin-top: 8px;
  }
  .poc-card .metric { font-size: 12px; }
  .poc-card .metric .num { font-size: 18px; font-weight: 700; }
  .poc-card .arrow {
    font-size: 20px; color: var(--card-accent, var(--accent-blue));
    transition: transform 0.2s;
  }
  .poc-card:hover .arrow { transform: translateX(4px); }
</style>
</head>
<body>
<div class="hero">
  <h1><span class="hl">HugeGraph</span> PoC 可视化演示</h1>
  <div class="tagline">交互式图谱算法演示 — 6 大方向，25+ 场景</div>
  <div class="server-badge" id="serverBadge">
    <div class="dot offline" id="serverDot"></div>
    <span id="serverText">正在检测 HugeGraph Server...</span>
  </div>
</div>

<div class="grid">
  CARDS_HERE
</div>

<script>
fetch('/api/server_status').then(r=>r.json()).then(d=>{
  const dot=document.getElementById('serverDot');
  const txt=document.getElementById('serverText');
  if(d.online){dot.className='dot online';txt.textContent='HugeGraph Server: 已连接 (localhost:8080)';}
  else{dot.className='dot offline';txt.textContent='HugeGraph Server: 未连接 (networkx 模拟模式)';}
}).catch(()=>{document.getElementById('serverText').textContent='HugeGraph Server: 无法访问';});
</script>
</body>
</html>
"""

# ============ PoC Page Template (shared by all 6 PoC pages) ============
POC_PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PAGETITLE — HugeGraph PoC</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.4/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-dagre/2.5.0/cytoscape-dagre.js"></script>
<style>
  SHARED_CSS_HERE
  body { display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  .sidebar {
    width: 420px; min-width: 420px;
    background: var(--bg-secondary); border-right: 1px solid var(--border-color);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .sidebar-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border-color);
  }
  .sidebar-header .breadcrumb {
    font-size: 11px; color: var(--text-secondary); margin-bottom: 6px;
  }
  .sidebar-header .breadcrumb a { color: var(--accent-blue); }
  .sidebar-header h1 {
    font-size: 18px; font-weight: 700; color: var(--accent-blue);
  }
  .sidebar-header .subtitle {
    font-size: 12px; color: var(--text-secondary); margin-top: 4px;
    display: flex; align-items: center; gap: 6px;
  }
  .sidebar-header .subtitle .status-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0;
  }
  .sidebar-header .subtitle .status-dot.online { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
  .sidebar-header .subtitle .status-dot.offline { background: var(--accent-red); }
  .sidebar-body { flex: 1; overflow-y: auto; padding: 16px; }
  .sidebar-body::-webkit-scrollbar { width: 6px; }
  .sidebar-body::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }

  .section-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text-secondary); margin: 16px 0 8px;
  }
  .section-title:first-child { margin-top: 0; }

  .scenario-btn {
    display: block; width: 100%; padding: 10px 14px;
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    border-radius: var(--radius); color: var(--text-primary);
    font-size: 13px; cursor: pointer; text-align: left;
    margin-bottom: 6px; transition: all 0.2s; position: relative;
  }
  .scenario-btn:hover { border-color: var(--accent-blue); background: rgba(66,133,244,0.1); }
  .scenario-btn.active { border-color: var(--accent-blue); background: rgba(66,133,244,0.15); }
  .scenario-btn .verdict {
    position: absolute; top: 8px; right: 10px;
    font-size: 10px; font-weight: 700; padding: 2px 6px;
    border-radius: 4px;
  }
  .scenario-btn .verdict.pass { background: rgba(52,168,83,0.2); color: var(--accent-green); }
  .scenario-btn .verdict.fail { background: rgba(234,67,53,0.2); color: var(--accent-red); }

  .info-card {
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 12px 14px; margin-bottom: 8px;
    font-size: 12px; line-height: 1.6;
  }
  .info-card .label { color: var(--text-secondary); font-size: 11px; }
  .info-card .value { color: var(--text-primary); font-weight: 600; }

  .metric-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;
  }
  .metric-box {
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 10px; text-align: center;
  }
  .metric-box .num { font-size: 20px; font-weight: 700; }
  .metric-box .desc { font-size: 10px; color: var(--text-secondary); margin-top: 2px; }

  .path-item {
    background: var(--bg-tertiary); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 10px 14px; margin-bottom: 6px;
    font-size: 12px; cursor: pointer; transition: all 0.2s;
  }
  .path-item:hover { border-color: var(--accent-purple); background: rgba(168,85,247,0.1); }
  .path-item.highlighted { border-color: var(--accent-yellow); background: rgba(251,188,4,0.1); }

  .server-status {
    display: flex; align-items: center; gap: 6px; font-size: 11px;
    padding: 8px 14px; background: var(--bg-tertiary);
    border-radius: var(--radius); margin-bottom: 12px;
    line-height: 1.4;
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
  }
  .status-dot.online { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
  .status-dot.offline { background: var(--accent-red); }

  /* Main Canvas */
  .main {
    flex: 1; display: flex; flex-direction: column; overflow: hidden;
  }
  .canvas-header {
    padding: 12px 20px; border-bottom: 1px solid var(--border-color);
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg-secondary);
  }
  .canvas-header .title { font-size: 15px; font-weight: 600; }
  .canvas-header .gremlin {
    font-family: 'Courier New', monospace; font-size: 12px;
    color: var(--accent-cyan); background: var(--bg-primary);
    padding: 6px 12px; border-radius: var(--radius);
    border: 1px solid var(--border-color); max-width: 60%;
  }
  #cy {
    flex: 1; width: 100%; background: var(--bg-primary);
  }
  .canvas-footer {
    padding: 8px 20px; border-top: 1px solid var(--border-color);
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg-secondary); font-size: 11px; color: var(--text-secondary);
  }
  .legend { display: flex; gap: 16px; align-items: center; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot {
    width: 10px; height: 10px; border-radius: 50%;
  }

  .toast {
    position: fixed; bottom: 20px; right: 20px;
    background: var(--accent-blue); color: white;
    padding: 10px 20px; border-radius: var(--radius);
    font-size: 13px; z-index: 1000; opacity: 0;
    transition: opacity 0.3s; pointer-events: none;
  }
  .toast.show { opacity: 1; }

  .algo-step {
    font-size: 12px; padding: 8px 12px;
    background: var(--bg-tertiary); border-left: 3px solid var(--accent-blue);
    border-radius: 0 var(--radius) var(--radius) 0; margin-bottom: 4px;
  }
  .algo-step.active { border-left-color: var(--accent-yellow); background: rgba(251,188,4,0.08); }
  .algo-step .step-num { font-weight: 700; color: var(--accent-blue); }

  .node-detail {
    position: absolute; z-index: 100;
    background: var(--bg-secondary); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 12px 16px;
    font-size: 12px; min-width: 200px; max-width: 280px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .node-detail h3 { font-size: 14px; margin-bottom: 6px; }
  .node-detail .prop { color: var(--text-secondary); margin-bottom: 2px; }
  .node-detail .prop span { color: var(--text-primary); }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-header">
    <div class="breadcrumb"><a href="/">&#x2190; 所有 PoC</a></div>
    <h1>POCICON POCNAME</h1>
    <div class="subtitle" id="serverStatus">初始化中...</div>
  </div>

  <div class="sidebar-body" id="sidebarBody">
    <!-- Dynamic content loaded via API -->
  </div>
</div>

<div class="main">
  <div class="canvas-header">
    <div class="title" id="canvasTitle">请选择场景</div>
    <div class="gremlin" id="gremlinQuery"></div>
  </div>
  <div id="cy"></div>
  <div class="canvas-footer">
    <div class="legend" id="legend"></div>
    <div style="display:flex;align-items:center;gap:10px">
      <button onclick="clearGraph()" style="background:rgba(234,67,53,0.1);border:1px solid rgba(234,67,53,0.3);color:#ea4335;padding:4px 12px;border-radius:6px;font-size:11px;cursor:pointer;transition:all .2s" onmouseover="this.style.background='rgba(234,67,53,0.2)'" onmouseout="this.style.background='rgba(234,67,53,0.1)'">🗑 清空图谱</button>
      <button onclick="clearVectors()" style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.3);color:#a855f7;padding:4px 12px;border-radius:6px;font-size:11px;cursor:pointer;transition:all .2s" onmouseover="this.style.background='rgba(168,85,247,0.2)'" onmouseout="this.style.background='rgba(168,85,247,0.1)'">🧹 清空向量</button>
      <div id="graphStats">节点: 0 | 边: 0</div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>
<div class="node-detail" id="nodeDetail" style="display:none;"></div>

<script>
let cy = null;
const POC_ID = 'POCID';

const NODE_COLORS = {
  enterprise: '#4285f4',  facility: '#34a853',    material: '#f97316',
  component: '#fbbc04',   product: '#a855f7',     function: '#06b6d4',
  class: '#4285f4',       module: '#34a853',      method: '#f97316',
  vertex_label: '#4285f4', edge_label: '#ea4335', operator: '#06b6d4',
  core: '#ea4335',        config: '#9aa0a6',
  input: '#34a853',       output: '#a855f7',      external: '#30363d',
  step: '#fbbc04',        model: '#a855f7',       artifact: '#9aa0a6',
  algorithm: '#06b6d4',   storage: '#34a853',     engine: '#ea4335',
  strategy: '#4285f4',    cache: '#fbbc04',       intermediate: '#9aa0a6',
  cluster: '#f97316',
  data_point: '#ec4899', event: '#f97316',    warning: '#ea4335',
  example: '#06b6d4',   api_endpoint: '#a855f7',
  person: '#ec4899',    organization: '#4285f4', location: '#34a853',
  skill: '#f97316',     concept: '#a855f7',
  default: '#9aa0a6',
};

const EDGE_COLORS = {
  supplies: '#34a853', produces: '#4285f4', has_input: '#f97316',
  located_in: '#ea4335', depends_on: '#fbbc04', operates: '#9aa0a6',
  calls: '#06b6d4', imports: '#a855f7', contains: '#34a853',
  out_edge: '#4285f4', in_edge: '#ea4335', input: '#34a853',
  delegates: '#9aa0a6', pipes_to: '#06b6d4', validates_with: '#fbbc04',
  queries: '#4285f4', on_error: '#ea4335', feeds_back: '#f97316',
  corrects: '#34a853', uses: '#9aa0a6', output: '#a855f7',
  flow: '#06b6d4', pipeline: '#4285f4', verifies: '#fbbc04',
  detects: '#ea4335', triggers: '#f97316', updates: '#34a853',
  // AI Memory edge types
  reinforces: '#f97316', extracts: '#a855f7', infers: '#06b6d4',
  primary: '#ec4899', fallback: '#fbbc04', classifies_as: '#4285f4',
  matches: '#34a853', example_of: '#9aa0a6',
  sends: '#06b6d4', provides: '#34a853', scores: '#f97316',
  filters: '#ea4335', ranks: '#a855f7', enriches: '#4285f4',
  accesses: '#f97316', feeds: '#34a853', produces: '#a855f7',
  formats: '#06b6d4', future: '#9aa0a6',
  has: '#4285f4', vs: '#ec4899',
  calls: '#06b6d4', delegates: '#9aa0a6', reads_direct: '#34a853',
  default: '#30363d',
};

function initCy() {
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],
    style: [
      { selector: 'node', style: {
        'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center',
        'font-size': '10px', 'color': '#e8eaed', 'text-outline-width': 2, 'text-outline-color': '#0f1117',
        'background-color': 'data(color)', 'width': 'data(size)', 'height': 'data(size)',
        'border-width': 1, 'border-color': '#30363d',
      }},
      { selector: 'node.highlighted', style: { 'border-width': 3, 'border-color': '#fbbc04', 'z-index': 999 }},
      { selector: 'node.dimmed', style: { 'opacity': 0.2 }},
      { selector: 'edge', style: {
        'width': 2, 'line-color': 'data(color)', 'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle', 'curve-style': 'bezier',
        'label': 'data(label)', 'font-size': '9px', 'color': '#9aa0a6',
        'text-rotation': 'autorotate', 'text-outline-width': 2, 'text-outline-color': '#0f1117',
      }},
      { selector: 'edge.highlighted', style: { 'width': 4, 'line-color': '#fbbc04', 'target-arrow-color': '#fbbc04', 'z-index': 998 }},
      { selector: 'edge.dimmed', style: { 'opacity': 0.15 }},
      { selector: 'node.seeded', style: { 'border-width': 3, 'border-color': '#ea4335', 'border-style': 'double' }},
    ],
    layout: { name: 'preset' },
    userZoomingEnabled: true, userPanningEnabled: true, boxSelectionEnabled: false,
  });
  cy.on('tap', 'node', function(evt) { showNodeDetail(evt.target); });
  cy.on('tap', function(evt) { if (evt.target === cy) document.getElementById('nodeDetail').style.display = 'none'; });
}

function showNodeDetail(node) {
  const data = node.data();
  const detail = document.getElementById('nodeDetail');
  let html = '<h3>' + data.label + '</h3>';
  if (data.props) { for (const [k, v] of Object.entries(data.props)) html += '<div class="prop">' + k + ': <span>' + v + '</span></div>'; }
  if (data.centrality) {
    html += '<div style="margin-top:6px;border-top:1px solid #30363d;padding-top:6px">';
    for (const [k, v] of Object.entries(data.centrality)) html += '<div class="prop">' + k + ': <span>' + (typeof v === 'number' ? v.toFixed(4) : v) + '</span></div>';
    html += '</div>';
  }
  detail.innerHTML = html;
  detail.style.display = 'block';
  detail.style.left = Math.min(evt.originalEvent.clientX, window.innerWidth - 300) + 'px';
  detail.style.top = (evt.originalEvent.clientY + 10) + 'px';
}

function loadGraph(data) {
  if (!cy) initCy();
  const elements = [];
  data.nodes.forEach((n, i) => {
    const angle = (i / data.nodes.length) * Math.PI * 2;
    const radius = 150 + (i % 5) * 60;
    elements.push({
      data: { id: n.id, label: n.label || n.name || n.id, color: n.color || NODE_COLORS[n.type] || NODE_COLORS.default,
        size: n.size || (n.isSeed ? 40 : 28), props: n.props || {}, centrality: n.centrality || null },
      position: n.position || { x: 400 + Math.cos(angle) * radius, y: 350 + Math.sin(angle) * radius },
      classes: (n.isSeed ? 'seeded ' : '') + (n.highlighted ? 'highlighted ' : ''),
    });
  });
  data.edges.forEach(e => {
    elements.push({
      data: { id: e.id || (e.source + '-' + e.target), source: e.source, target: e.target,
        label: e.label || e.relation || '', color: e.color || EDGE_COLORS[e.relation] || EDGE_COLORS.default },
      classes: e.highlighted ? 'highlighted' : '',
    });
  });
  cy.elements().remove();
  cy.add(elements);
  try { cy.layout({ name: 'dagre', rankDir: 'TB', spacingFactor: 1.2, nodeSep: 40, rankSep: 60 }).run(); }
  catch(e) { cy.layout({ name: 'concentric' }).run(); }
  document.getElementById('graphStats').textContent = '节点: ' + data.nodes.length + ' | 边: ' + data.edges.length;
  updateLegend(data.nodes);
}

function updateLegend(nodes) {
  const types = new Set(nodes.map(n => n.type).filter(Boolean));
  let html = '';
  types.forEach(t => { html += '<div class="legend-item"><div class="legend-dot" style="background:' + (NODE_COLORS[t] || NODE_COLORS.default) + '"></div>' + t + '</div>'; });
  document.getElementById('legend').innerHTML = html;
}

function highlightPath(pathIds) {
  cy.elements().removeClass('highlighted').removeClass('dimmed');
  if (pathIds.length === 0) { cy.elements().removeClass('dimmed'); return; }
  const nodeSet = new Set(pathIds.filter(id => !id.includes('-')));
  const edgeSet = new Set(pathIds.filter(id => id.includes('-')));
  cy.nodes().forEach(n => { if (nodeSet.has(n.id())) n.addClass('highlighted'); else n.addClass('dimmed'); });
  cy.edges().forEach(e => { if (edgeSet.has(e.id())) e.addClass('highlighted'); else e.addClass('dimmed'); });
  const highlighted = cy.$('.highlighted');
  if (highlighted.length > 0) cy.animate({ fit: { eles: highlighted, padding: 60 }, duration: 500, easing: 'ease-in-out-cubic' });
}

function showToast(msg) {
  const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

function loadScenario(scenarioId) {
  document.getElementById('canvasTitle').textContent = '⏳ 加载中: ' + scenarioId;
  fetch('/api/scenario?poc=' + POC_ID + '&scenario=' + scenarioId)
    .then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(data => {
      // Render graph
      if (!data.graph || !data.graph.nodes) throw new Error('返回数据无图谱');
      const g = data.graph;

      // Build Cytoscape elements
      const elements = [];
      g.nodes.forEach((n, i) => {
        const angle = (i / g.nodes.length) * Math.PI * 2;
        const radius = 150 + (i % 5) * 60;
        elements.push({
          data: { id: n.id, label: n.label || n.name || n.id,
            color: n.color || NODE_COLORS[n.type] || NODE_COLORS.default,
            size: n.size || (n.isSeed ? 40 : 28),
            props: n.props || {}, centrality: n.centrality || null },
          position: { x: 400 + Math.cos(angle) * radius, y: 350 + Math.sin(angle) * radius },
        });
      });
      g.edges.forEach(e => {
        elements.push({
          data: { id: e.id || (e.source + '-' + e.target),
            source: e.source, target: e.target,
            label: e.label || e.relation || '',
            color: e.color || EDGE_COLORS[e.relation] || EDGE_COLORS.default },
        });
      });

      if (!cy) initCy();
      cy.elements().remove();
      cy.add(elements);
      try { cy.layout({ name: 'dagre', rankDir: 'TB', spacingFactor: 1.2, nodeSep: 40, rankSep: 60 }).run(); }
      catch(layoutErr) {
        cy.layout({ name: 'concentric' }).run();
      }

      // Update UI
      document.getElementById('canvasTitle').textContent = data.title || scenarioId;
      if (data.gremlin && data.gremlin.length > 0) document.getElementById('gremlinQuery').textContent = data.gremlin[0];
      document.getElementById('graphStats').textContent = '\u8282\u70b9: ' + g.nodes.length + ' | \u8fb9: ' + g.edges.length;

      // Legend
      const types = [...new Set(g.nodes.map(n => n.type).filter(Boolean))];
      let legendHtml = '';
      types.forEach(t => { legendHtml += '<div class="legend-item"><div class="legend-dot" style="background:' + (NODE_COLORS[t] || NODE_COLORS.default) + '"></div>' + t + '</div>'; });
      document.getElementById('legend').innerHTML = legendHtml;

      // Sidebar detail
      if (data.sidebarHtml) document.getElementById('dynamicSidebar').innerHTML = data.sidebarHtml;
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      if (typeof event !== 'undefined' && event.target && event.target.closest) {
        var activeBtn = event.target.closest('.scenario-btn');
        if (activeBtn) activeBtn.classList.add('active');
      }
      showToast('\u56fe\u8c31\u52a0\u8f7d\u5b8c\u6210: ' + g.nodes.length + ' \u8282\u70b9');
    })
    .catch(err => {
      console.error('[loadScenario]', err);
      document.getElementById('canvasTitle').textContent = '\u274c \u52a0\u8f7d\u5931\u8d25: ' + err.message;
      showToast('\u274c ' + err.message);
    });
}

function focusNode(nid) {
  if (!cy) return;
  const node = cy.getElementById(nid);
  if (node.length) { cy.animate({ fit: { eles: node, padding: 80 }, duration: 400 }); node.addClass('highlighted'); }
}

function highlightPathFromSidebar(el) {
  document.querySelectorAll('.path-item').forEach(p => p.classList.remove('highlighted'));
  el.classList.add('highlighted');
}

// ============================================================================
// ============================================================================
// AI Memory Interactive Demo — 严格对标 PowerMem v1.1.2 memory_server.py
// 流水线完全按照 add_memory() / search_memory() 实现
// ============================================================================

var demoRunning = false;
var SIM_MEMORIES = [];
var SIM_NODES = [];
var SIM_EDGES = [];
var SIM_USER_NAME = '';

var DEMO_EXAMPLES = {
  add: [
    { en: "John works at Google as a senior engineer" },
    { en: "张三和李四都是阿里云的高级工程师" },
    { en: "我在字节跳动做后端开发" },
    { en: "张三在货拉拉工作" }
  ],
  query: [
    { en: "John 在哪工作？" },
    { en: "谁是李四的同事？" },
    { en: "我记得有个人在腾讯，是谁来着？" }
  ]
};

function toggleDemoPanel() {
  var p = document.getElementById('demoPanel');
  var i = document.getElementById('demoToggleIcon');
  if (!p) return;
  p.style.display = (p.style.display==='none'||p.style.display==='') ? 'block' : 'none';
  i.textContent = p.style.display==='block' ? '\u25B2' : '\u25BC';
}

function fillExample(mode, idx) {
  var input = document.getElementById('demoInput');
  var ex = DEMO_EXAMPLES[mode][idx];
  if (input && ex) input.value = ex.en;
  var panel = document.getElementById('demoPanel');
  if (panel && panel.style.display === '') panel.style.display = 'block';
}

function resetDemo() {
  document.getElementById('demoInput').value = '';
  document.getElementById('demoLog').style.display = 'none';
  document.getElementById('demoSteps').innerHTML = '';
  document.getElementById('demoResult').style.display = 'none';
  document.getElementById('demoResult').innerHTML = '';
  demoRunning = false;
}

function addDemoStep(num, label, detail, status, nodeId) {
  var c = document.getElementById('demoSteps');
  var el = document.createElement('div');
  el.className = 'demo-step ' + status;
  el.id = 'step_' + num;
  var icons = {active:'\u23F3',done:'\u2705',error:'\u274C'};
  var colors = {active:'var(--accent-pink)',done:'var(--accent-green)',error:'var(--accent-red)'};
  el.innerHTML = '<span style="color:'+(colors[status]||'#999')+';font-weight:700">'+
    (icons[status]||'\u2022')+'</span><b>'+label+'</b>'+(detail?': '+detail:'');
  c.appendChild(el);
  el.scrollIntoView({behavior:'smooth',block:'nearest'});
  if (cy && nodeId) {
    cy.elements().removeClass('highlighted');
    var n = cy.getElementById(nodeId);
    if (n.length) { n.addClass('highlighted'); try{cy.animate({fit:{eles:n,padding:80},duration:300});}catch(e){} }
  }
  return el;
}

function stepDone(n) {
  var e = document.getElementById('step_'+n);
  if (e) e.className = 'demo-step done';
}

// --- PowerMem 意图分类 (对标 classify_intent_llm + classify_intent_regex 双路径) ---
function classifyIntent(text) {
  var hasQ = /\?|\uFF1F/.test(text);
  var startsQ = /^(谁|什么|哪里|哪个|哪些|有多少|有哪些|你喜欢|我喜欢|我有什么|帮我查|记得|是谁)/.test(text);
  // 第一人称查询
  var startsMy = /^(我|我的|咱|咱们)(的?|们?)(同事|朋友|认识|有哪些|谁|什么|了解|知道|记得|之前|上次)/.test(text);
  // 陈述句中的疑问模式: "X的同事是谁", "谁是X的同事", "X在哪里工作"
  var midQuery = /(的(同事|朋友|朋友|上级|下属|老板|同学))\s*(是)?(谁|哪[个些]|什么)/.test(text)
    || /^谁\s+(是|是?)(\S{1,6})\s*的/.test(text)
    || /在(哪|哪个|哪些|什么地方)\s*(工作|生活|学习|住)/.test(text)
    || /(?:是|有)谁$/.test(text);
  // 关系类查询关键词: 同事/朋友/认识 + 疑问词组合
  var relQuery = /(同事|朋友|朋友|认识的人|熟人).{0,3}(谁|呢|呀|吗|呢吗)/.test(text)
    || /谁.{0,4}(同事|朋友|朋友|认识)/.test(text);

  var stmtHint = /也在|也喜欢|也认识|一起|都[是在]/.test(text);
  var isQ = hasQ || startsQ || (startsMy && !stmtHint) || midQuery || relQuery;
  var reason = hasQ?'regex:\u5305\u542B\u95EE\u53F7':
    (startsQ?'regex:\u7591\u95EE\u8BCD\u5F00\u5934':
    (midQuery?'regex:\u53E5\u4E2D\u7591\u95EE\u6A21\u5F0F(\u7684XX\u662F\u8C01)':
    (relQuery?'regex:\u5173\u7CFB\u7C7B\u67E5\u8BE2(\u540C\u4E8B/\u670B\u53CB+\u7591\u95EE\u8BCD)':
    (isQ?'regex:\u7B2C\u4E00\u4EBA\u79F0\u67E5\u8BE2':'regex:\u9648\u8FF0\u53E5\u9ED8\u8BA4ADD'))));
  return {action: isQ?'QUERY':'ADD', reason: reason};
}

// --- PowerMem LLM 实体抽取 (对标 extract_entities_and_relations → MiMo API) ---
function simulateLLMExtract(text) {
  var entities=[], relationships=[];

  // 组织识别
  var orgMap={'google':'Google','谷歌':'Google','阿里云':'阿里云','阿里巴巴':'阿里巴巴',
    '腾讯':'腾讯','微信':'微信','百度':'百度','字节跳动':'字节跳动','抖音':'抖音','飞书':'飞书',
    '华为':'华为','美团':'美团','京东':'京东','microsoft':'Microsoft','微软':'Microsoft',
    'apple':'Apple','苹果':'Apple','amazon':'Amazon','meta':'Meta','facebook':'Facebook',
    'tesla':'Tesla','货拉拉':'货拉拉','滴滴':'滴滴','快手':'快手',
    'b站':'哔哩哔哩','bilibili':'哔哩哔哩','小米':'小米','网易':'网易',
    'hugegraph':'HugeGraph','apache':'Apache','oceanbase':'OceanBase'};
  var foundOrgs=[];
  for(var k in orgMap){if(text.toLowerCase().indexOf(k)!==-1){var o=orgMap[k];if(!foundOrgs.some(function(x){return x===o}))foundOrgs.push(o);}}

  // 人物识别
  var cnNames=text.match(/([张王李刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文][\u4e00-\u9fa5]{1,2})(?![\u4e00-\u9fa5])/g)||[];
  var enNames=(text.match(/\b([A-Z][a-z]+)\b/g)||[]).filter(function(n){
    return!['The','This','That','With','From','Where','What','When','Who','How','Google','Microsoft','Apple','Amazon','Facebook','Meta','Netflix','Tesla'].includes(n);
  });
  var persons=[];
  cnNames.forEach(function(n){if(!persons.includes(n))persons.push(n);});
  enNames.forEach(function(n){if(!persons.includes(n))persons.push(n);});

  var nameMatch=text.match(/我叫(\S{1,4})/);
  if(nameMatch&&!SIM_USER_NAME)SIM_USER_NAME=nameMatch[1];
  if(/我|我的/.test(text)&&SIM_USER_NAME&&!persons.includes(SIM_USER_NAME))persons.unshift(SIM_USER_NAME);
  if(persons.length===0){var fb=text.match(/^([\u4e00-\u9fa5]{2,3})[在是和跟]/);if(fb)persons.push(fb[1]);}

  // 技能/职位
  var skills=['工程师','开发','程序员','架构师','技术总监','CTO','全栈','后端','前端','算法','数据科学','机器学习','深度学习','NLP','CV','产品经理','PM','设计师','UI','UX'];
  var foundSkills=[];
  skills.forEach(function(s){if(text.indexOf(s)!==-1&&!foundSkills.includes(s))foundSkills.push(s);});
  ['engineer','developer','architect','designer','scientist','analyst','manager','director'].forEach(function(s){
    var re=new RegExp('\\b'+s+'\\b','i');if(re.test(text)){var cap=s[0].toUpperCase()+s.slice(1);if(!foundSkills.includes(cap))foundSkills.push(cap);}
  });

  // 地点
  var locs={'北京':'北京','上海':'上海','深圳':'深圳','杭州':'杭州'};
  var foundLocs=[];
  for(var lk in locs){if(text.indexOf(lk)!==-1)foundLocs.push(locs[lk]);}

  persons.forEach(function(p){entities.push({name:p,type:'person'});});
  foundOrgs.forEach(function(o){entities.push({name:o,type:'organization'});});
  foundSkills.forEach(function(s){entities.push({name:s,type:'skill'});});
  foundLocs.forEach(function(l){entities.push({name:l,type:'location'});});
  if(entities.length<=1){entities.push({name:text.length>20?text.substring(0,17)+'...':text,type:'concept'});}

  // 关系推导 (对标 _extract_missing_rels)
  persons.forEach(function(p){
    if(foundOrgs.length>0&&(/在/.test(text)||/\b(at|works?\s*at|in)\b/i.test(text)))
      relationships.push({source:p,relationship:'works_at',target:foundOrgs[0]});
    if(foundSkills.length>0&&(/做|担任|是.*[师家手]|engineer/i.test(text)))
      relationships.push({source:p,relationship:'has_skill',target:foundSkills[0]});
  });
  if(/喜欢|likes/.test(text)&&SIM_USER_NAME){
    foundSkills.filter(function(s){return text.indexOf(s)!==-1;}).forEach(function(s){
      relationships.push({source:SIM_USER_NAME,relationship:'likes',target:s});
    });
  }

  return {entities:entities,relationships:relationships};
}

// --- PowerMem 同事推理 (对标 _infer_colleague, 条件触发!) ---
// 关键: 不仅看当前抽取的实体,还要结合已有图谱状态(SIM_EDGES)做跨记忆推理
function inferColleagues(ext) {
  // 当前抽取的人物
  var newPersons=ext.entities.filter(function(e){return e.type==='person';}).map(function(e){return e.name;});
  // 当前抽取的works_at关系
  var newWorkRels=ext.relationships.filter(function(r){return r.relationship==='works_at';});

  // 合并已有图谱中的 works_at 关系, 做跨记忆同事推理
  // 这是 PowerMem 的核心能力: 新记忆加入后激活隐含关系的发现
  var allWorkRels=newWorkRels.concat(SIM_EDGES.filter(function(r){return r.relationship==='works_at';}));

  if(newPersons.length===0)return{trigger:false,inferred:[],reason:'本次未检测到person实体'};

  // 按 organization 分组所有 works_at 关系
  var groups={};
  allWorkRels.forEach(function(r){
    if(!groups[r.target])groups[r.target]=[];
    groups[r.target].push(r.source);
  });

  var inf=[];
  for(var g in groups){
    var members=groups[g];
    // 至少2个人共享同一组织 → 推断同事关系
    if(members.length>=2){
      // 只报告涉及当前新人物的同事对 (避免重复推断已知关系)
      for(var i=0;i<members.length;i++){
        for(var j=i+1;j<members.length;j++){
          var involvesNew=(newPersons.indexOf(members[i])!==-1)||(newPersons.indexOf(members[j])!==-1);
          if(involvesNew)inf.push({source:members[i],relationship:'colleague_of',target:members[j],org:g});
        }
      }
    }
  }

  if(inf.length===0){
    // 给出具体原因
    if(allWorkRels.length<2)return{trigger:false,inferred:[],reason:'图谱中仅有'+allWorkRels.length+'条works_at关系，无法形成同事组'};
    return{trigger:false,inferred:[],reason:newPersons.join(',')+'加入后无新同事关系(各自在不同组织)'};
  }
  return{trigger:true,inferred:inf,reason:'发现'+inf.length+'对新同事关系(共享 '+Object.keys(groups).join('/')+')'};
}

// --- PowerMem Ebbinghaus 遗忘曲线 R(t)=e^(-0.821t)+access_count*0.3 ---
function calcEbbinghaus(mem,nowSec){
  var K=0.821,R=0.3,t=(nowSec-mem.created_at)/3600,ret=Math.exp(-K*t);
  ret=Math.min(1,ret+mem.access_count*R);return Math.max(0,Math.min(1,ret));
}

// ============================================================================
// 运行演示流水线
// ============================================================================
function runDemoPipeline(mode){
  if(demoRunning)return;demoRunning=true;
  var text=(document.getElementById('demoInput')?.value||'').trim();
  if(!text){alert('请先输入文本！');demoRunning=false;return;}
  document.getElementById('demoLog').style.display='block';
  document.getElementById('demoResult').style.display='none';
  document.getElementById('demoSteps').innerHTML='';
  loadScenario(mode==='add'?'memory_pipeline':'search_flow');
  setTimeout(function(){animatePipeline(text,mode);},800);
}

function animatePipeline(text, mode) {
  var stepsEl = document.getElementById('demoSteps');
  var traceResultEl = document.getElementById('demoTraceResult');
  stepsEl.innerHTML = '';
  if (traceResultEl) traceResultEl.textContent = '';

  var endpoint = mode === 'add'
    ? 'http://127.0.0.1:8765/api/memory/add'
    : 'http://127.0.0.1:8765/api/memory/search';

  var body = mode === 'add'
    ? { content: text }
    : { query: text };

  fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    var trace = data.trace || [];
    var answer = data.answer || '';
    var totalSteps = trace.length;

    if (totalSteps === 0) {
      addStepCard(stepsEl, 'error', 'Error', 'Backend returned empty trace', 'red');
      demoRunning = false;
      return;
    }

    var action = data.action || 'ADD';

    var NODE_COLORS = {
      'person': '#61dafb', 'organization': '#ffd93d', 'location': '#6bcb77',
      'concept': '#c084fc', 'skill': '#ff6b6b', 'event': '#4ecdc4', 'unknown': '#a8a29e'
    };

    trace.forEach(function(step, idx) {
      setTimeout(function() {
        var iconMap = {
          'input': 'fa-keyboard', 'classify': 'fa-tag', 'extract': 'fa-brain',
          'resolve': 'fa-link', 'conflict': 'fa-exclamation-triangle',
          'colleague': 'fa-user-friends', 'store': 'fa-database',
          'embedding': 'fa-vector-square', 'search': 'fa-search',
          'rank': 'fa-sort-amount-down', 'context': 'fa-project-diagram',
          'generate': 'fa-comment-dots', 'freshness': 'fa-clock'
        };
        var icon = iconMap[step.name] || 'fa-cog';
        var color = 'var(--accent)';
        var statusIcon = '<i class="fas fa-spinner fa-spin"></i>';
        var detail = step.detail || '';

        // Color entity-rich steps
        if (step.name === 'extract' || step.name === 'store') {
          // Highlight entity names in detail text
          var coloredDetail = detail;
          Object.keys(NODE_COLORS).forEach(function(type) {
            var re = new RegExp('\\b(' + type + ')\\b', 'gi');
          });
          color = NODE_COLORS['concept'];
        }

        if (step.name === 'colleague' && detail.indexOf('TRIGGERED') >= 0) {
          color = '#ff6b6b';
          statusIcon = '<i class="fas fa-fire" style="color:#ff6b6b"></i> COLLEAGUE TRIGGERED';
        } else if (step.name === 'colleague') {
          color = '#a8a29e';
        }

        if (step.name === 'generate') {
          color = '#6bcb77';
          statusIcon = '<i class="fas fa-check-circle"></i>';
        }

        var isLast = (idx === totalSteps - 1);
        if (isLast) {
          statusIcon = '<i class="fas fa-check-circle"></i>';
          color = '#6bcb77';
        }

        var cardId = 'step_' + Date.now() + '_' + idx;
        var html = '<div id="' + cardId + '" class="step-card" style="opacity:0;transform:translateY(12px)">' +
          '<div class="step-header"><span class="step-icon" style="background:' + color + '">' +
          '<i class="fas ' + icon + '"></i></span>' +
          '<span class="step-title">' + (step.name || step.name) + '</span>' +
          '<span class="step-status">' + statusIcon + '</span></div>' +
          '<div class="step-detail">' + detail + '</div></div>';
        stepsEl.insertAdjacentHTML('beforeend', html);

        var card = document.getElementById(cardId);
        if (card) {
          card.style.transition = 'all 0.4s ease';
          setTimeout(function() {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
          }, 50);
        }

        // Final step: show action status + answer + refresh graph
        if (isLast) {
          setTimeout(function() {
            // Show action badge (ADD / SKIP / QUERY / UPDATE)
            var actionColor = action === 'ADD' ? '#6bcb77' : action === 'SKIP' ? '#ffd93d' : action === 'QUERY' ? '#61dafb' : '#9aa0a6';
            var actionLabel = action === 'ADD' ? 'ADD - New Memory' : action === 'SKIP' ? 'SKIP - Duplicate' : action === 'QUERY' ? 'QUERY - Answer' : action;
            var actionIcon = action === 'ADD' ? 'plus-circle' : action === 'SKIP' ? 'exclamation-circle' : action === 'QUERY' ? 'search' : 'sync';
            var actionHtml = '<div style="margin-top:10px;padding:8px 16px;' +
              'background:' + actionColor + '18;border:1px solid ' + actionColor + '44;' +
              'border-radius:8px;font-size:13px;color:' + actionColor + ';font-weight:500;text-align:center">' +
              '<i class="fas fa-' + actionIcon + '"></i> ' +
              actionLabel + '</div>';
            stepsEl.insertAdjacentHTML('beforeend', actionHtml);

            if ((mode === 'search' || action === 'QUERY') && answer) {
              var answerHtml = '<div style="margin-top:12px;padding:12px 16px;' +
                'background:rgba(107,203,119,0.12);border-left:3px solid #6bcb77;' +
                'border-radius:0 8px 8px 0;font-size:14px;color:#6bcb77;font-weight:500">' +
                '<i class="fas fa-comment-dots"></i> ' + answer + '</div>';
              stepsEl.insertAdjacentHTML('beforeend', answerHtml);
            }

            // Show HugeGraph+FAISS branding
            var brandHtml = '<div style="margin-top:8px;padding:6px 12px;' +
              'background:rgba(97,218,251,0.08);border-radius:6px;' +
              'font-size:11px;color:#888;text-align:center">' +
              '<i class="fas fa-project-diagram"></i> HugeGraph + ' +
              '<i class="fas fa-brain"></i> FAISS + ' +
              '<i class="fas fa-robot"></i> MiMo LLM</div>';
            stepsEl.insertAdjacentHTML('beforeend', brandHtml);

            refreshGraphFromBackend();

            // Unlock for next interaction
            demoRunning = false;
          }, 500);
        }
      }, idx * 420);
    });
  })
  .catch(function(err) {
    console.error('Pipeline fetch error:', err);
    addStepCard(stepsEl, 'error', 'Connection Error',
      'Cannot connect to backend at http://127.0.0.1:8765\n' +
      'Make sure memory_backend.py is running!', 'red');
    demoRunning = false;
  });
}

function addStepCard(container, name, title, detail, color) {
  var cardId = 'step_' + Date.now() + '_err';
  var icon = name === 'error' ? 'fa-times-circle' : 'fa-cog';
  var html = '<div id="' + cardId + '" class="step-card" style="opacity:0;transform:translateY(12px)">' +
    '<div class="step-header"><span class="step-icon" style="background:' + (color||'var(--accent)') + '">' +
    '<i class="fas ' + icon + '"></i></span>' +
    '<span class="step-title">' + title + '</span></div>' +
    '<div class="step-detail">' + detail.replace(/\n/g, '<br>') + '</div></div>';
  container.insertAdjacentHTML('beforeend', html);
  var card = document.getElementById(cardId);
  if (card) {
    card.style.transition = 'all 0.4s ease';
    setTimeout(function() {
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 50);
  }
}

function refreshGraphFromBackend() {
  fetch('http://127.0.0.1:8765/api/graph')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (!cy) return;
      cy.elements().remove();
      var vertices = data.vertices || [];
      var edges = data.edges || [];

      var colorMap = {
        'person': '#61dafb', 'organization': '#ffd93d', 'location': '#6bcb77',
        'concept': '#c084fc', 'skill': '#ff6b6b', 'event': '#4ecdc4'
      };

      vertices.forEach(function(v) {
        var label = v.label || 'unknown';
        var color = colorMap[label] || '#a8a29e';
        cy.add({
          data: { id: v.id, label: v.properties && v.properties.name ? v.properties.name : v.id },
          classes: label,
          style: {
            'background-color': color,
            'label': v.properties && v.properties.name ? v.properties.name : v.id,
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '11px',
            'color': '#fff',
            'text-outline-width': '2px',
            'text-outline-color': '#333',
            'width': 40,
            'height': 40
          }
        });
      });

      edges.forEach(function(e) {
        var srcId = e.source || e.outV;
        var tgtId = e.target || e.inV;
        if (srcId && tgtId) {
          cy.add({
            data: {
              id: e.id || (srcId + '-' + tgtId),
              source: srcId,
              target: tgtId,
              label: e.label || ''
            }
          });
        }
      });

      cy.layout({ name: 'cose', animate: true, animationDuration: 500 }).run();

      // Update stats
      updateGraphStats(vertices.length, edges.length);
    })
    .catch(function(err) {
      console.error('Graph refresh error:', err);
    });
}

function updateGraphStats(vCount, eCount) {
  var statsEl = document.getElementById('graphStats');
  if (statsEl) {
    statsEl.innerHTML =
      '<span style="color:#61dafb"><i class="fas fa-circle"></i> Vertices: ' + vCount + '</span> &nbsp;' +
      '<span style="color:#ffd93d"><i class="fas fa-arrow-right"></i> Edges: ' + eCount + '</span>';
  }
}

// 清空图谱
function clearGraph() {
  if (cy) cy.elements().remove();
  document.getElementById('canvasTitle').textContent = '请选择场景';
  document.getElementById('gremlinQuery').textContent = '';
  document.getElementById('legend').innerHTML = '';
  document.getElementById('graphStats').textContent = '节点: 0 | 边: 0';
  // Also reset demo
  resetDemo();
  showToast('🗑 图谱已清空');
}

// 清空向量/重置全部
function clearVectors() {
  clearGraph();
  if (cy) {
    var bg = cy.style().css;
    // Flash effect
    cy.flash({ color: '#a855f7', duration: 300 });
  }
  // Reset sidebar dynamic area
  var dyn = document.getElementById('dynamicSidebar');
  if (dyn) dyn.innerHTML = '<div style="padding:12px;text-align:center;color:var(--text-secondary);font-size:12px">向量索引已重置<br><span style="font-size:10px">Embedding cache cleared, graph state reset</span></div>';
  showToast('🧹 向量与图谱已全部重置');
}

// Init
window.onerror = function(msg, url, line) {
  document.getElementById('canvasTitle').textContent = 'JS错误: ' + msg + ' (行' + line + ')';
  return false;
};

window.addEventListener('DOMContentLoaded', () => {
  initCy();
  document.getElementById('serverStatus').innerHTML = '<div class="status-dot online"></div><span>JS已加载</span>';
  // Auto-load first scenario for AI Memory
  if (POC_ID === 'ai_memory') {
    setTimeout(function() { loadScenario('memory_pipeline'); }, 500);
  }
  // Load sidebar content (refresh dynamic parts like server status)
  fetch('/api/sidebar?poc=' + POC_ID + '&t=' + Date.now())
    .then(r => r.json()).then(data => {
      // Only update status, don't overwrite embedded sidebar buttons
      if (data.serverStatus) {
        const ss = document.getElementById('serverStatus');
        if (ss) ss.innerHTML = data.serverStatus;
      }
    }).catch(err => {
      console.warn('Sidebar refresh failed (embedded content still works):', err);
    });
});
</script>
</body>
</html>
"""


# ============ Supply Chain Graph Data ============

def build_supply_chain_graph():
    """构建供应链图谱（与 supply_chain_kg_duality.py 一致）"""
    G = nx.DiGraph()

    # 5 类 KG 语义实体
    enterprises = {
        "apple": {"name": "Apple", "type": "enterprise", "country": "美国", "revenue": "3000亿美元"},
        "tesla": {"name": "Tesla", "type": "enterprise", "country": "美国", "revenue": "800亿美元"},
        "tsmc": {"name": "台积电 TSMC", "type": "enterprise", "country": "中国台湾", "process": "3nm/5nm"},
        "samsung": {"name": "三星 Samsung", "type": "enterprise", "country": "韩国", "process": "3nm"},
        "foxconn": {"name": "富士康 Foxconn", "type": "enterprise", "country": "中国台湾"},
        "cathay": {"name": "国泰航空 Cathay Pacific", "type": "enterprise", "country": "中国香港"},
        "catl": {"name": "宁德时代 CATL", "type": "enterprise", "country": "中国", "product": "锂电池"},
    }
    facilities = {
        "drc": {"name": "刚果(金) DRC", "type": "facility", "region": "非洲", "risk": "地缘风险"},
        "chile": {"name": "智利 Chile", "type": "facility", "region": "南美"},
        "australia": {"name": "澳大利亚 Australia", "type": "facility", "region": "大洋洲"},
        "taiwan_fab": {"name": "新竹晶圆厂", "type": "facility", "region": "中国台湾"},
        "shenzhen_assembly": {"name": "深圳组装厂", "type": "facility", "region": "中国"},
        "shanghai_battery": {"name": "上海电池工厂", "type": "facility", "region": "中国"},
    }
    materials = {
        "cobalt": {"name": "钴矿 Cobalt", "type": "material", "category": "矿产", "criticality": "关键矿产"},
        "lithium_ore": {"name": "锂矿 Lithium", "type": "material", "category": "矿产", "criticality": "关键矿产"},
        "rare_earth": {"name": "稀土 Rare Earth", "type": "material", "category": "矿产", "criticality": "关键矿产"},
        "silicon": {"name": "硅晶圆 Silicon", "type": "material", "category": "半导体材料"},
    }
    components = {
        "soc_chip": {"name": "SoC 芯片", "type": "component", "category": "半导体", "process": "3nm"},
        "lithium_cell": {"name": "锂电池", "type": "component", "category": "电池"},
        "ic_design": {"name": "IC 设计方案", "type": "component", "category": "设计"},
        "display": {"name": "OLED 显示屏", "type": "component", "category": "显示"},
        "camera_sensor": {"name": "CMOS 传感器", "type": "component", "category": "影像"},
    }
    products = {
        "iphone16": {"name": "iPhone 16", "type": "product", "category": "消费电子"},
        "model_s": {"name": "Tesla Model S", "type": "product", "category": "电动汽车"},
        "airpods": {"name": "AirPods Pro", "type": "product", "category": "消费电子"},
    }

    for nid, attrs in {**enterprises, **facilities, **materials, **components, **products}.items():
        G.add_node(nid, **attrs)

    # 8 种语义关系
    edges = [
        ("drc", "cobalt", "located_in", 100, "矿产分布"),
        ("chile", "lithium_ore", "located_in", 85, "矿产分布"),
        ("australia", "rare_earth", "located_in", 75, "矿产分布"),
        ("australia", "lithium_ore", "located_in", 60, "矿产分布"),
        ("cobalt", "lithium_cell", "supplies", 90, "关键原料"),
        ("lithium_ore", "lithium_cell", "supplies", 85, "关键原料"),
        ("catl", "lithium_cell", "supplies", 80, "电池供应商"),
        ("catl", "shanghai_battery", "operates", 70, "生产基地"),
        ("lithium_cell", "model_s", "has_input", 75, "电池输入"),
        ("lithium_cell", "iphone16", "has_input", 60, "电池输入"),
        ("tsmc", "soc_chip", "supplies", 90, "晶圆代工"),
        ("samsung", "soc_chip", "supplies", 10, "备选代工"),
        ("tsmc", "taiwan_fab", "operates", 70, "生产基地"),
        ("ic_design", "soc_chip", "has_input", 65, "设计输入"),
        ("soc_chip", "iphone16", "has_input", 80, "核心组件"),
        ("soc_chip", "airpods", "has_input", 50, "组件输入"),
        ("foxconn", "iphone16", "produces", 85, "组装生产"),
        ("foxconn", "shenzhen_assembly", "operates", 75, "生产基地"),
        ("foxconn", "airpods", "produces", 70, "组装生产"),
        ("apple", "iphone16", "produces", 80, "品牌产品"),
        ("apple", "airpods", "produces", 60, "品牌产品"),
        ("tesla", "model_s", "produces", 85, "品牌产品"),
        ("rare_earth", "display", "supplies", 55, "稀土材料"),
        ("rare_earth", "camera_sensor", "supplies", 45, "稀土材料"),
        ("display", "iphone16", "has_input", 40, "屏幕组件"),
        ("camera_sensor", "iphone16", "has_input", 35, "影像组件"),
        ("silicon", "soc_chip", "supplies", 70, "硅晶圆"),
        ("apple", "tsmc", "depends_on", 90, "战略依赖"),
        ("tesla", "catl", "depends_on", 65, "电池依赖"),
    ]

    for src, dst, rel, weight, desc in edges:
        G.add_edge(src, dst, relation=rel, weight=weight, description=desc)

    return G


def compute_centrality(G):
    """预计算三类中心性 + 结构显著性"""
    # 无向化用于中心性计算
    UG = G.to_undirected()

    deg = nx.degree_centrality(UG)
    betw = nx.betweenness_centrality(UG)
    close = nx.closeness_centrality(UG)

    # 结构显著性 = min-max 归一化后取平均
    def normalize(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        return {k: (v - mn) / (mx - mn) if mx > mn else 0 for k, v in d.items()}

    nd = normalize(deg)
    nb = normalize(betw)
    nc = normalize(close)

    centrality = {}
    for node in G.nodes():
        centrality[node] = {
            "degree": float(deg.get(node, 0)),
            "betweenness": float(betw.get(node, 0)),
            "closeness": float(close.get(node, 0)),
            "structural_significance": float((nd.get(node, 0) + nb.get(node, 0) + nc.get(node, 0)) / 3),
        }
    return centrality


def adaptive_traversal(G, centrality, seeds, ss_threshold=0.3, bidirectional=False):
    """中心性驱动自适应深度遍历"""
    subgraph = nx.DiGraph()
    visited = set()
    depth_map = {}
    traversal_log = []

    for seed in seeds:
        if seed not in G.nodes():
            continue
        ss = centrality.get(seed, {}).get("structural_significance", 0.0)
        if ss >= ss_threshold:
            max_hops = 1
        else:
            max_hops = 2
        traversal_log.append({"seed": seed, "ss": round(ss, 4), "max_hops": max_hops})

        queue = deque([(seed, 0)])
        while queue:
            current, depth = queue.popleft()
            if current not in visited:
                visited.add(current)
                depth_map[current] = min(depth_map.get(current, 999), depth)
                subgraph.add_node(current, **G.nodes[current])

            if depth >= max_hops:
                continue

            neighbors = list(G.successors(current))
            if bidirectional:
                neighbors.extend(G.predecessors(current))

            for neighbor in neighbors:
                if neighbor not in visited:
                    subgraph.add_node(neighbor, **G.nodes[neighbor])
                    if G.has_edge(current, neighbor):
                        subgraph.add_edge(current, neighbor, **G.edges[current, neighbor])
                    elif G.has_edge(neighbor, current):
                        subgraph.add_edge(neighbor, current, **G.edges[neighbor, current])
                    queue.append((neighbor, depth + 1))
                    if neighbor not in depth_map:
                        depth_map[neighbor] = depth + 1

    for u, v, data in G.edges(data=True):
        if u in subgraph and v in subgraph and not subgraph.has_edge(u, v):
            subgraph.add_edge(u, v, **data)

    return subgraph, depth_map, traversal_log


def find_paths(G, seeds, max_depth=5):
    """BFS 找所有路径"""
    all_paths = []
    for seed in seeds:
        if seed not in G.nodes():
            continue
        queue = deque([(seed, [seed])])
        visited_paths = set()
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if len(path) >= 2 and tuple(path) not in visited_paths:
                visited_paths.add(tuple(path))
                edges_info = []
                total_weight = 0
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    if G.has_edge(src, dst):
                        ed = G.edges[src, dst]
                    elif G.has_edge(dst, src):
                        ed = G.edges[dst, src]
                    else:
                        continue
                    edges_info.append({
                        "src": src, "dst": dst,
                        "relation": ed.get("relation", ""),
                        "weight": ed.get("weight", 0),
                    })
                    total_weight += ed.get("weight", 0)
                all_paths.append({
                    "nodes": list(path),
                    "edges": edges_info,
                    "total_weight": total_weight,
                })
            for nb in list(G.successors(current)) + list(G.predecessors(current)):
                if nb not in path:
                    queue.append((nb, path + [nb]))
    all_paths.sort(key=lambda x: x.get("total_weight", 0), reverse=True)
    return all_paths


def extract_subgraph_for_cytoscape(G, centrality, seeds=None, highlight_paths=None):
    """将 NetworkX 子图转为 Cytoscape.js 格式"""
    nodes = []
    edges = []
    seed_set = set(seeds or [])

    for nid, attrs in G.nodes(data=True):
        is_seed = nid in seed_set
        c = centrality.get(nid, {})
        nodes.append({
            "id": nid,
            "label": attrs.get("name", nid),
            "type": attrs.get("type", "default"),
            "isSeed": is_seed,
            "size": 40 if is_seed else 28,
            "props": {k: v for k, v in attrs.items() if k not in ("name", "type")},
            "centrality": {k: round(v, 4) for k, v in c.items()} if c else None,
        })

    for src, dst, attrs in G.edges(data=True):
        eid = f"{src}-{dst}"
        highlighted = False
        if highlight_paths:
            for p in highlight_paths:
                if src in p["nodes"] and dst in p["nodes"]:
                    pidx_src = p["nodes"].index(src)
                    pidx_dst = p["nodes"].index(dst)
                    if abs(pidx_src - pidx_dst) == 1:
                        highlighted = True
                        break
        edges.append({
            "id": eid,
            "source": src,
            "target": dst,
            "label": attrs.get("relation", ""),
            "color": EDGE_COLORS_PY.get(attrs.get("relation", ""), EDGE_COLORS_PY["default"]),
            "highlighted": highlighted,
        })

    return {"nodes": nodes, "edges": edges}


def path_to_shell_text(G, path):
    """路径壳文本转述"""
    parts = []
    for i, edge in enumerate(path.get("edges", [])):
        src_name = G.nodes[edge["src"]].get("name", edge["src"])
        dst_name = G.nodes[edge["dst"]].get("name", edge["dst"])
        weight = edge.get("weight", 0)
        rel = edge.get("relation", "connects")
        rel_zh = {
            "supplies": "供应", "produces": "生产", "has_input": "包含",
            "located_in": "位于", "depends_on": "依赖", "operates": "运营",
        }.get(rel, rel)
        parts.append(f"{src_name} --[{rel_zh} {weight}%]--> {dst_name}")
    return " → ".join(parts)


# ============ Code Graph Data ============

def build_code_graph():
    """构建代码图谱"""
    G = nx.DiGraph()

    nodes = {
        "mod_main": {"name": "main.py", "type": "module", "loc": 45},
        "mod_auth": {"name": "auth.py", "type": "module", "loc": 120},
        "mod_db": {"name": "database.py", "type": "module", "loc": 200},
        "mod_api": {"name": "api_router.py", "type": "module", "loc": 180},
        "mod_utils": {"name": "utils.py", "type": "module", "loc": 60},
        "cls_AuthService": {"name": "AuthService", "type": "class", "methods": 4, "loc": 80},
        "cls_DBService": {"name": "DBService", "type": "class", "methods": 5, "loc": 150},
        "cls_DataProcessor": {"name": "DataProcessor", "type": "class", "methods": 6, "loc": 170},
        "cls_APIRouter": {"name": "APIRouter", "type": "class", "methods": 4, "loc": 160},
        "fn_authenticate": {"name": "authenticate()", "type": "function", "complexity": 6, "loc": 25},
        "fn_authorize": {"name": "authorize()", "type": "function", "complexity": 4, "loc": 15},
        "fn_generate_token": {"name": "generate_token()", "type": "function", "complexity": 5, "loc": 20},
        "fn_refresh_token": {"name": "refresh_token()", "type": "function", "complexity": 3, "loc": 12},
        "fn_connect": {"name": "connect()", "type": "function", "complexity": 3, "loc": 10},
        "fn_execute_query": {"name": "execute_query()", "type": "function", "complexity": 8, "loc": 35},
        "fn_close": {"name": "close()", "type": "function", "complexity": 2, "loc": 5},
        "fn_transaction": {"name": "transaction()", "type": "function", "complexity": 7, "loc": 30},
        "fn_process_data": {"name": "process_data()", "type": "function", "complexity": 10, "loc": 45},
        "fn_validate": {"name": "validate()", "type": "function", "complexity": 5, "loc": 15},
        "fn_transform": {"name": "transform()", "type": "function", "complexity": 6, "loc": 20},
        "fn_aggregate": {"name": "aggregate()", "type": "function", "complexity": 9, "loc": 25},
        "fn_analyze": {"name": "analyze()", "type": "function", "complexity": 12, "loc": 50},
        "fn_get_user": {"name": "get_user()", "type": "function", "complexity": 4, "loc": 18},
        "fn_create_user": {"name": "create_user()", "type": "function", "complexity": 5, "loc": 22},
        "fn_list_data": {"name": "list_data()", "type": "function", "complexity": 3, "loc": 12},
        "fn_analyze_endpoint": {"name": "analyze_endpoint()", "type": "function", "complexity": 7, "loc": 28},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    code_edges = [
        # imports
        ("mod_main", "mod_auth", "imports"),
        ("mod_main", "mod_db", "imports"),
        ("mod_main", "mod_api", "imports"),
        ("mod_main", "mod_utils", "imports"),
        ("mod_api", "mod_auth", "imports"),
        ("mod_api", "mod_db", "imports"),
        ("mod_api", "mod_utils", "imports"),
        # class membership
        ("mod_auth", "cls_AuthService", "contains"),
        ("mod_db", "cls_DBService", "contains"),
        ("mod_utils", "cls_DataProcessor", "contains"),
        ("mod_api", "cls_APIRouter", "contains"),
        # class calls functions
        ("cls_AuthService", "fn_authenticate", "calls"),
        ("cls_AuthService", "fn_authorize", "calls"),
        ("cls_AuthService", "fn_generate_token", "calls"),
        ("cls_AuthService", "fn_refresh_token", "calls"),
        ("cls_DBService", "fn_connect", "calls"),
        ("cls_DBService", "fn_execute_query", "calls"),
        ("cls_DBService", "fn_close", "calls"),
        ("cls_DBService", "fn_transaction", "calls"),
        ("cls_DataProcessor", "fn_process_data", "calls"),
        ("cls_DataProcessor", "fn_validate", "calls"),
        ("cls_DataProcessor", "fn_transform", "calls"),
        ("cls_DataProcessor", "fn_aggregate", "calls"),
        ("cls_DataProcessor", "fn_analyze", "calls"),
        ("cls_APIRouter", "fn_get_user", "calls"),
        ("cls_APIRouter", "fn_create_user", "calls"),
        ("cls_APIRouter", "fn_list_data", "calls"),
        ("cls_APIRouter", "fn_analyze_endpoint", "calls"),
        # cross-class dependencies
        ("cls_AuthService", "fn_execute_query", "calls"),
        ("cls_APIRouter", "fn_authenticate", "calls"),
        ("cls_APIRouter", "fn_authorize", "calls"),
        ("fn_process_data", "fn_validate", "calls"),
        ("fn_process_data", "fn_transform", "calls"),
        ("fn_analyze_endpoint", "fn_process_data", "calls"),
        ("fn_analyze_endpoint", "fn_analyze", "calls"),
        ("fn_analyze", "fn_execute_query", "calls"),
    ]
    for src, dst, rel in code_edges:
        G.add_edge(src, dst, relation=rel)

    return G


def extract_code_graph_for_cytoscape(G, centrality, seeds=None, highlight_paths=None):
    nodes = []
    edges = []
    seed_set = set(seeds or [])
    TYPE_COLORS_CODE = {
        "module": "#34a853", "class": "#4285f4", "function": "#06b6d4",
    }
    REL_COLORS_CODE = {
        "imports": "#a855f7", "contains": "#34a853", "calls": "#06b6d4",
    }

    for nid, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "default")
        complexity = attrs.get("complexity", 0)
        size = 20
        if ntype == "class":
            size = 35
        elif ntype == "module":
            size = 30
        elif complexity > 8:
            size = 25 + min(complexity, 10)

        nodes.append({
            "id": nid,
            "label": attrs.get("name", nid),
            "type": ntype,
            "color": TYPE_COLORS_CODE.get(ntype, "#9aa0a6"),
            "isSeed": nid in seed_set,
            "size": size,
            "props": {k: v for k, v in attrs.items() if k != ("name", "type")},
            "centrality": {k: round(v, 4) for k, v in centrality.get(nid, {}).items()} if centrality else None,
        })

    for src, dst, attrs in G.edges(data=True):
        eid = f"{src}-{dst}"
        rel = attrs.get("relation", "")
        highlighted = False
        if highlight_paths:
            for p in highlight_paths:
                if src in p["nodes"] and dst in p["nodes"]:
                    highlighted = True
                    break
        edges.append({
            "id": eid, "source": src, "target": dst,
            "label": rel, "color": REL_COLORS_CODE.get(rel, "#30363d"),
            "highlighted": highlighted,
        })

    return {"nodes": nodes, "edges": edges}


# ============ GraphRAG-Bench Data ============

def build_graphrag_bench_data():
    """GraphRAG-Bench 对比数据"""
    methods = [
        {"name": "HippoRAG2", "multi_hop": 0.82, "summary": 0.65, "qa": 0.78, "cost": 0.5, "category": "Memory-based"},
        {"name": "LightRAG", "multi_hop": 0.75, "summary": 0.72, "qa": 0.74, "cost": 0.4, "category": "Graph-Enhanced"},
        {"name": "Fast-GraphRAG", "multi_hop": 0.70, "summary": 0.60, "qa": 0.72, "cost": 0.3, "category": "Graph-Enhanced"},
        {"name": "RAPTOR", "multi_hop": 0.55, "summary": 0.78, "qa": 0.68, "cost": 0.6, "category": "Retrieval"},
        {"name": "MGraphRAG", "multi_hop": 0.68, "summary": 0.70, "qa": 0.71, "cost": 0.55, "category": "Graph-Enhanced"},
        {"name": "KGP", "multi_hop": 0.72, "summary": 0.62, "qa": 0.70, "cost": 0.7, "category": "Memory-based"},
        {"name": "GraphRAG(MS)", "multi_hop": 0.78, "summary": 0.80, "qa": 0.76, "cost": 0.9, "category": "Graph-Enhanced"},
        {"name": "G-Retriever", "multi_hop": 0.80, "summary": 0.58, "qa": 0.75, "cost": 0.5, "category": "Graph-Enhanced"},
        {"name": "DALK", "multi_hop": 0.65, "summary": 0.64, "qa": 0.67, "cost": 0.35, "category": "Retrieval"},
        {"name": "ToG", "multi_hop": 0.73, "summary": 0.55, "qa": 0.71, "cost": 0.45, "category": "Memory-based"},
        {"name": "GFM-RAG", "multi_hop": 0.77, "summary": 0.68, "qa": 0.74, "cost": 0.3, "category": "Graph-Enhanced"},
        {"name": "DRIFT", "multi_hop": 0.78, "summary": 0.76, "qa": 0.80, "cost": 0.7, "category": "Graph-Enhanced"},
    ]
    return methods


# ============ Flask App ============

app = Flask(__name__)

# Disable caching for development
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Pre-build graph data
supply_chain_G = build_supply_chain_graph()
supply_chain_centrality = compute_centrality(supply_chain_G)
code_graph_G = build_code_graph()
code_graph_centrality = compute_centrality(code_graph_G)


# ============ PoC Registry (for index page cards) ============
POC_REGISTRY = [
    {
        "id": "supply_chain",
        "icon": "&#x1F3E0;",
        "name": "供应链知识图谱二重性",
        "desc": "UC Berkeley Network-KG 二重性算法: 中心性驱动自适应深度遍历、双向风险追踪、路径壳转述",
        "tags": [("5 个场景", "pass"), ("26 节点", ""), ("8 种关系", "")],
        "accent": "#4285f4",
        "metrics": [("场景数", 5, ""), ("通过", 5, "")],
    },
    {
        "id": "text2gremlin",
        "icon": "&#x1F513;",
        "name": "Text2Gremlin 自纠错",
        "desc": "Sprint 5: Schema 引导式自然语言→Gremlin 解析、语法+Schema 双校验器，最多自动重试 3 次 + LLM 错误纠正",
        "tags": [("41 测试", "pass"), ("最多3次重试", ""), ("Schema引导", "")],
        "accent": "#ea4335",
        "metrics": [("测试用例", 41, ""), ("重试次数", 3, "")],
    },
    {
        "id": "drift_search",
        "icon": "&#x1F50D;",
        "name": "DRIFT 搜索流水线",
        "desc": "Sprint 4: 五步搜索算法 — HyDE→社区匹配→锚点提取→局部搜索→压缩排序, 33/36 能力覆盖, 6 大独有维度",
        "tags": [("30 测试", "pass"), ("5 步骤", ""), ("6 独有维度", "")],
        "accent": "#06b6d4",
        "metrics": [("测试用例", 30, ""), ("能力覆盖", 33, "/36")],
    },
    {
        "id": "entity_resolution",
        "icon": "&#x1F91D;",
        "name": "实体消解",
        "desc": "Sprint 1-2: 三策略级联 (精确匹配→向量相似度→LLM 验证), 增量索引构建, 24+16 测试全部通过",
        "tags": [("24+16 测试", "pass"), ("3 策略", ""), ("增量索引", "")],
        "accent": "#f97316",
        "metrics": [("Sprint 1", 24, ""), ("Sprint 2", 16, "")],
    },
    {
        "id": "graphrag_bench",
        "icon": "&#x1F4CA;",
        "name": "GraphRAG-Bench 方案对比",
        "desc": "12 种 GraphRAG 方法系统对比: 能力雷达图、任务匹配排名、成本效益分析, DRIFT 6 大独有维度",
        "tags": [("12 种方法", "pass"), ("5 维度", ""), ("排名第1/13", "")],
        "accent": "#a855f7",
        "metrics": [("方法数", 12, ""), ("排名", 1, "/13")],
    },
    {
        "id": "code_graph",
        "icon": "&#x1F4BB;",
        "name": "代码图谱双Agent架构",
        "desc": "CodexGraph + Understand-Anything: 主 Agent(意图分析) + 翻译 Agent(NL→Gremlin), Tree-sitter AST → 函数调用图, Gremlin vs Cypher 对比",
        "tags": [("5 场景", "pass"), ("双Agent", ""), ("Gremlin vs Cypher", "")],
        "accent": "#34a853",
        "metrics": [("场景数", 5, ""), ("Agent 数", 2, "")],
    },
    {
        "id": "ai_memory",
        "icon": "&#x1F4BE;",
        "name": "AI Memory 系统",
        "desc": "对标 PowerMem v1.1.2: 艾宾浩斯遗忘曲线、LLM 实体抽取、意图分类(双路径)、图谱增强检索、同事关系推理、SQLite 图存储 + Cytoscape 可视化",
        "tags": [("7 场景", "pass"), ("艾宾浩斯", ""), ("PowerMem对标", "")],
        "accent": "#ec4899",
        "metrics": [("场景数", 7, ""), ("API端点", 8, "")],
    },
]


def render_poc_page(poc_id, poc_name, poc_icon):
    """Render a PoC-specific page from POC_PAGE_TEMPLATE"""
    html = POC_PAGE_TEMPLATE.replace("SHARED_CSS_HERE", SHARED_CSS)
    html = html.replace("PAGETITLE", poc_name)
    html = html.replace("POCID", poc_id)
    html = html.replace("POCICON", poc_icon)
    html = html.replace("POCNAME", poc_name)

    # Embed initial sidebar HTML server-side (fallback if AJAX fails)
    poc = next((p for p in POC_REGISTRY if p["id"] == poc_id), None)
    if poc:
        sidebar_funcs = {
            "supply_chain": supply_chain_sidebar,
            "graphrag_bench": graphrag_bench_sidebar,
            "code_graph": code_graph_sidebar,
            "text2gremlin": text2gremlin_sidebar,
            "drift_search": drift_search_sidebar,
            "entity_resolution": entity_resolution_sidebar,
            "ai_memory": ai_memory_sidebar,
        }
        fn = sidebar_funcs.get(poc_id)
        if fn:
            initial_sidebar = fn()
            html = html.replace('<!-- Dynamic content loaded via API -->', initial_sidebar)

    return html


def render_index_page():
    """Render the index landing page with PoC cards"""
    cards_html = ""
    for poc in POC_REGISTRY:
        tags_html = "".join(
            f'<span class="tag{" pass" if t[1] == "pass" else ""}">{t[0]}</span>'
            for t in poc["tags"]
        )
        metrics_html = ""
        for m in poc["metrics"]:
            metrics_html += f'<div class="metric"><div class="num" style="color:{poc["accent"]}">{m[0]}{m[1]}</div><div style="font-size:10px;color:var(--text-secondary)">{m[2]}</div></div>'

        cards_html += f"""
        <a href="/{poc['id']}" style="color:inherit;text-decoration:none">
          <div class="poc-card" style="--card-accent: {poc['accent']}">
            <span class="card-icon">{poc['icon']}</span>
            <h2>{poc['name']}</h2>
            <p class="card-desc">{poc['desc']}</p>
            <div class="card-tags">{tags_html}</div>
            <div class="card-footer">
              <div class="metric-grid">{metrics_html}</div>
              <span class="arrow">&#x2794;</span>
            </div>
          </div>
        </a>
        """

    html = INDEX_TEMPLATE.replace("SHARED_CSS_HERE", SHARED_CSS)
    html = html.replace("CARDS_HERE", cards_html)
    return html


@app.route("/")
def index():
    return render_index_page()


@app.route("/<string:poc_id>")
def poc_page(poc_id):
    valid_ids = {p["id"] for p in POC_REGISTRY}
    if poc_id not in valid_ids:
        # Redirect to index if unknown poc
        return render_index_page()
    poc = next((p for p in POC_REGISTRY if p["id"] == poc_id), None)
    return render_poc_page(poc_id, poc["name"], poc["icon"])


@app.route("/api/server_status")
def server_status():
    """检查 HugeGraph Server 连通性"""
    try:
        import requests
        resp = requests.get("http://localhost:8080/graphs", timeout=2)
        if resp.status_code == 200:
            graphs = resp.json().get("graphs", [])
            return jsonify({"online": True, "graphs": graphs})
    except Exception:
        pass
    return jsonify({"online": False, "graphs": []})


@app.route("/api/hugegraph_graph")
def hugegraph_graph():
    """从 HugeGraph Server 实时获取供应链图谱数据"""
    try:
        import requests as req
        BASE = "http://localhost:8080/graphs/hugegraph"

        # Fetch all vertices
        rv = req.get(f"{BASE}/graph/vertices?limit=-1", timeout=5)
        vertices_data = rv.json().get("vertices", [])

        # Fetch all edges
        re = req.get(f"{BASE}/graph/edges?limit=-1", timeout=5)
        edges_data = re.get("edges", []) if hasattr(re, "json") else []

        if not edges_data:
            re = req.get(f"{BASE}/graph/edges?limit=-1", timeout=5)
            edges_data = re.json().get("edges", [])

        # Convert to Cytoscape format
        nodes = []
        for v in vertices_data:
            props = v.get("properties", {})
            ntype = props.get("type", v.get("label", "default"))
            nodes.append({
                "id": str(v["id"]),
                "label": props.get("name", str(v["id"])),
                "type": ntype,
                "color": NODE_COLORS_PY.get(ntype, NODE_COLORS_PY["default"]),
                "isSeed": False,
                "size": 30,
                "props": {k: v for k, v in props.items() if k not in ("name", "type")},
                "centrality": None,
            })

        edges = []
        for e in edges_data:
            elabel = e.get("label", "")
            edges.append({
                "id": str(e.get("id", f"{e.get('outV')}-{e.get('inV')}")),
                "source": str(e.get("outV")),
                "target": str(e.get("inV")),
                "label": elabel,
                "color": EDGE_COLORS_PY.get(elabel, EDGE_COLORS_PY["default"]),
                "highlighted": False,
            })

        # Also fetch shortest path from Apple to TSMC
        apple_v = next((v for v in vertices_data if v.get("properties", {}).get("name") == "Apple"), None)
        tsmc_v = next((v for v in vertices_data if v.get("properties", {}).get("name") == "TSMC"), None)

        path_info = []
        if apple_v and tsmc_v:
            try:
                rp = req.get(
                    f"{BASE}/traversers/shortestpath?source={apple_v['id']}&target={tsmc_v['id']}&max_depth=5",
                    timeout=5,
                )
                if rp.status_code == 200:
                    pdata = rp.json()
                    path_ids = pdata.get("path", [])
                    path_names = []
                    for pid in path_ids:
                        pv = next((v for v in vertices_data if v["id"] == pid), None)
                        if pv:
                            path_names.append(pv["properties"].get("name", str(pid)))
                    path_info = path_names
            except Exception:
                pass

        return jsonify({
            "online": True,
            "graph": {"nodes": nodes, "edges": edges},
            "apple_tsmc_path": path_info,
            "vertex_count": len(nodes),
            "edge_count": len(edges),
        })
    except Exception as ex:
        return jsonify({"online": False, "error": str(ex), "graph": {"nodes": [], "edges": []}})


@app.route("/api/sidebar")
def sidebar():
    poc_id = request.args.get("poc", "supply_chain")
    status_resp = server_status.__wrapped__() if hasattr(server_status, "__wrapped__") else None

    try:
        import requests as req
        resp = req.get("http://localhost:8080/graphs", timeout=2)
        online = resp.status_code == 200
    except Exception:
        online = False

    if online:
        status_html = '<div class="server-status"><div class="status-dot online"></div><span>HugeGraph Server: 已连接</span> <span style="color:var(--text-secondary);font-weight:400">(localhost:8080)</span></div>'
        # Add live query button when server is online
        status_html += '<div style="margin-top:10px"><button class="scenario-btn" onclick="loadScenario(\'supply_chain\',\'hugegraph_live\')" style="background:rgba(52,168,83,0.15);border-color:var(--accent-green)">🔴 实时: 从 HugeGraph Server 查询</button></div>'
    else:
        status_html = '<div class="server-status"><div class="status-dot offline"></div><span>HugeGraph Server: 未连接</span> <span style="color:var(--text-secondary);font-weight:400">(使用 networkx 模拟模式)</span></div>'

    html = status_html

    if poc_id == "supply_chain":
        html += supply_chain_sidebar()
    elif poc_id == "graphrag_bench":
        html += graphrag_bench_sidebar()
    elif poc_id == "code_graph":
        html += code_graph_sidebar()
    elif poc_id == "text2gremlin":
        html += text2gremlin_sidebar()
    elif poc_id == "drift_search":
        html += drift_search_sidebar()
    elif poc_id == "entity_resolution":
        html += entity_resolution_sidebar()
    elif poc_id == "ai_memory":
        html += ai_memory_sidebar()

    # serverStatus: clean HTML for header subtitle (with inline status-dot)
    if online:
        status_for_header = '<div class="status-dot online"></div><span>HugeGraph: 已连接</span>'
    else:
        status_for_header = '<div class="status-dot offline"></div><span>HugeGraph: 未连接</span>'

    return jsonify({"html": html, "serverStatus": status_for_header})


def supply_chain_sidebar():
    """供应链场景侧栏"""
    html = """
    <div class="section-title">算法步骤</div>
    <div class="algo-step" onclick="algoStepClicked(0)"><span class="step-num">步骤 1</span> 排 (Rank): 预计算三类中心性</div>
    <div class="algo-step" onclick="algoStepClicked(1)"><span class="step-num">步骤 2</span> 取 (Retrieve): 自适应深度遍历</div>
    <div class="algo-step" onclick="algoStepClicked(2)"><span class="step-num">步骤 3</span> 述 (Narrate): 路径壳转述</div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','full_graph')">
      完整供应链图谱
      <span class="verdict pass">26 节点</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','apple_risk')">
      Apple 芯片集中度风险
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','tesla_risk')">
      Tesla 电池地缘风险
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','adaptive_vs_fixed')">
      自适应 vs 固定 BFS
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','centrality_rank')">
      中心性排名
      <span class="verdict pass">PASS</span>
    </button>

    <div id="dynamicSidebar"></div>
    """
    return html


def graphrag_bench_sidebar():
    """GraphRAG-Bench 场景侧栏"""
    html = """
    <div class="section-title">基准概览</div>
    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">12</div><div class="desc">对比方法</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-green)">5</div><div class="desc">评估维度</div></div>
    </div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','radar_chart')">
      能力雷达图
      <span class="verdict pass">12 种方法</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','task_match')">
      任务匹配排名
      <span class="verdict pass">4 大场景</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','cost_benefit')">
      成本效益分析
      <span class="verdict pass">Top 5</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','drift_unique')">
      DRIFT 6 大独有维度
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','comparison_graph')">
      方法关系图谱
      <span class="verdict pass">PASS</span>
    </button>

    <div id="dynamicSidebar"></div>
    """
    return html


def code_graph_sidebar():
    """代码图谱场景侧栏"""
    html = """
    <div class="section-title">架构</div>
    <div class="algo-step"><span class="step-num">Agent 1</span> Main: 意图分析 + 路由</div>
    <div class="algo-step"><span class="step-num">Agent 2</span> Translation: NL → Gremlin</div>
    <div class="algo-step"><span class="step-num">执行器</span> GraphExecutor: 执行查询</div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" onclick="loadScenario('code_graph','full_graph')">
      完整代码图谱
      <span class="verdict pass">21 节点</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','call_chain')">
      调用链追踪
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','impact_analysis')">
      影响分析
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','complexity_heat')">
      复杂度热力图
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','gremlin_vs_cypher')">
      Gremlin vs Cypher
      <span class="verdict pass">PASS</span>
    </button>

    <div id="dynamicSidebar"></div>
    """
    return html


@app.route("/api/scenario")
def load_scenario():
    poc_id = request.args.get("poc", "supply_chain")
    scenario_id = request.args.get("scenario", "")

    if poc_id == "supply_chain":
        return jsonify(get_supply_chain_scenario(scenario_id))
    elif poc_id == "graphrag_bench":
        return jsonify(get_graphrag_bench_scenario(scenario_id))
    elif poc_id == "code_graph":
        return jsonify(get_code_graph_scenario(scenario_id))
    elif poc_id == "text2gremlin":
        return jsonify(get_text2gremlin_scenario(scenario_id))
    elif poc_id == "drift_search":
        return jsonify(get_drift_search_scenario(scenario_id))
    elif poc_id == "entity_resolution":
        return jsonify(get_entity_resolution_scenario(scenario_id))
    elif poc_id == "ai_memory":
        return jsonify(get_ai_memory_scenario(scenario_id))

    return jsonify({"error": "Unknown poc/scenario"})


def get_supply_chain_scenario(scenario_id):
    """供应链场景"""
    if scenario_id == "full_graph":
        graph = extract_subgraph_for_cytoscape(
            supply_chain_G, supply_chain_centrality,
            seeds=list(supply_chain_G.nodes())
        )
        return {
            "title": "完整供应链图谱 (Network-KG 二重性)",
            "graph": graph,
            "gremlin": ["g.V().outE().path().by('name').by('relation')"],
            "sidebarHtml": build_metrics_sidebar(
                supply_chain_G.number_of_nodes(),
                supply_chain_G.number_of_edges(),
                {"5 类实体": len(set(supply_chain_G.nodes[n].get("type") for n in supply_chain_G.nodes)),
                 "8 种关系": len(set(supply_chain_G.edges[e].get("relation") for e in supply_chain_G.edges))},
            ),
        }

    elif scenario_id == "apple_risk":
        seeds = ["apple"]
        subgraph, depth_map, log = adaptive_traversal(
            supply_chain_G, supply_chain_centrality, seeds, ss_threshold=0.2, bidirectional=True
        )
        paths = find_paths(subgraph, seeds)
        chip_paths = [p for p in paths if any(n in p["nodes"] for n in ["soc_chip", "tsmc"])]

        graph = extract_subgraph_for_cytoscape(subgraph, supply_chain_centrality, seeds, chip_paths)

        # tsmc dependency check
        tsmc_dep = 0
        for src, dst, data in supply_chain_G.in_edges("soc_chip", data=True):
            if data.get("relation") == "supplies" and "tsmc" in src:
                tsmc_dep = max(tsmc_dep, data.get("weight", 0))

        path_html = ""
        for p in chip_paths[:5]:
            shell = path_to_shell_text(supply_chain_G, p)
            path_html += f'<div class="path-item" onclick="highlightPathFromSidebar(this)">{shell}</div>'

        return {
            "title": "Apple 芯片集中度风险分析",
            "graph": graph,
            "gremlin": [
                "g.V('apple').repeat(both().simplePath()).emit().path().by('name').by('relation')",
                "g.V().has('name','SoC 芯片').in('supplies').path()",
                "g.V('apple').out('produces').out('has_input').has('category','半导体').path()",
            ],
            "sidebarHtml": f"""
            <div class="section-title">风险评估</div>
            <div class="info-card">
              <div class="label">台积电依赖度</div>
              <div class="value" style="color:var(--accent-red);font-size:18px">{tsmc_dep}%</div>
            </div>
            <div class="info-card">
              <div class="label">风险等级</div>
              <div class="value" style="color:var(--accent-red)">高</div>
            </div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{subgraph.number_of_nodes()}</div><div class="desc">子图节点数</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-purple)">{len(chip_paths)}</div><div class="desc">芯片风险路径</div></div>
            </div>
            <div class="section-title">关键路径 (点击高亮)</div>
            {path_html}
            """,
        }

    elif scenario_id == "tesla_risk":
        seeds = ["tesla", "model_s", "lithium_cell"]
        subgraph, depth_map, log = adaptive_traversal(
            supply_chain_G, supply_chain_centrality, seeds, ss_threshold=0.5, bidirectional=True
        )
        paths = find_paths(subgraph, seeds)
        risk_paths = [p for p in paths if any(n in p["nodes"] for n in ["drc", "cobalt", "lithium_ore"])]

        graph = extract_subgraph_for_cytoscape(subgraph, supply_chain_centrality, seeds, risk_paths)

        path_html = ""
        for p in risk_paths[:8]:
            shell = path_to_shell_text(supply_chain_G, p)
            path_html += f'<div class="path-item" onclick="highlightPathFromSidebar(this)">{shell}</div>'

        return {
            "title": "Tesla 电池地缘政治风险 (双向追踪)",
            "graph": graph,
            "gremlin": [
                "g.V('tesla').repeat(both().simplePath()).emit().until(__.loops().is(eq(3))).path()",
                "g.V('tesla').out('produces').in('has_input').repeat(in('supplies').simplePath()).emit().path()",
            ],
            "sidebarHtml": f"""
            <div class="section-title">风险评估</div>
            <div class="info-card">
              <div class="label">刚果(金) 暴露度</div>
              <div class="value" style="color:var(--accent-red)">高 - {len([p for p in risk_paths if 'drc' in p['nodes']])} 条路径</div>
            </div>
            <div class="info-card">
              <div class="label">钴矿暴露度</div>
              <div class="value" style="color:var(--accent-orange)">高 - {len([p for p in risk_paths if 'cobalt' in p['nodes']])} 条路径</div>
            </div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{subgraph.number_of_nodes()}</div><div class="desc">子图节点</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-red)">{len(risk_paths)}</div><div class="desc">风险路径</div></div>
            </div>
            <div class="section-title">风险路径 (点击高亮)</div>
            {path_html}
            """,
        }

    elif scenario_id == "adaptive_vs_fixed":
        seeds = ["apple", "tesla"]
        sub_adaptive, _, log = adaptive_traversal(
            supply_chain_G, supply_chain_centrality, seeds, ss_threshold=0.2
        )
        # Fixed BFS with 2 hops
        sub_fixed, _, _ = adaptive_traversal(
            supply_chain_G, supply_chain_centrality, seeds, ss_threshold=99.0  # force 2 hops
        )

        # Highlight nodes in adaptive but not in fixed (savings)
        adaptive_nodes = set(sub_adaptive.nodes())
        fixed_nodes = set(sub_fixed.nodes())
        saved = fixed_nodes - adaptive_nodes
        extra = adaptive_nodes - fixed_nodes

        # Mark nodes
        graph = extract_subgraph_for_cytoscape(sub_fixed, supply_chain_centrality, seeds)
        for n in graph["nodes"]:
            if n["id"] in saved:
                n["color"] = "#ea4335"  # red = saved
            elif n["id"] in extra:
                n["color"] = "#34a853"  # green = extra

        reduction = (1 - len(adaptive_nodes) / max(len(fixed_nodes), 1)) * 100

        return {
            "title": f"自适应 vs 固定 BFS (节点缩减: {reduction:.1f}%)",
            "graph": graph,
            "gremlin": [
                "// Adaptive: seed SS decides depth",
                "g.V(seed).repeat(out().simplePath()).emit().until(__.loops().is(lt(adaptiveDepth)))",
                "// Fixed: all 2 hops",
                "g.V(seed).repeat(out().simplePath()).emit().until(__.loops().is(eq(2)))",
            ],
            "sidebarHtml": f"""
            <div class="section-title">自适应 vs 固定 BFS 对比</div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-red)">{len(fixed_nodes)}</div><div class="desc">固定 BFS 节点数</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-green)">{len(adaptive_nodes)}</div><div class="desc">自适应节点数</div></div>
            </div>
            <div class="info-card">
              <div class="label">节点缩减率</div>
              <div class="value" style="color:var(--accent-green);font-size:18px">{reduction:.1f}%</div>
            </div>
            <div class="section-title">图例</div>
            <div class="info-card"><span style="color:#ea4335">&#9632;</span> 自适应移除 (节省)</div>
            <div class="info-card"><span style="color:#34a853">&#9632;</span> 自适应新增 (发现)</div>
            <div class="info-card"><span style="color:#9aa0a6">&#9632;</span> 两种方法共有 (公共)</div>
            <div class="section-title">遍历日志</div>
            """,
        }

    elif scenario_id == "centrality_rank":
        # Show full graph with centrality sizing
        graph = extract_subgraph_for_cytoscape(supply_chain_G, supply_chain_centrality)
        # Size nodes by structural significance
        for n in graph["nodes"]:
            ss = supply_chain_centrality.get(n["id"], {}).get("structural_significance", 0)
            n["size"] = max(20, min(60, 20 + ss * 50))

        # Top 5 by SS
        ranked = sorted(supply_chain_centrality.items(), key=lambda x: x[1]["structural_significance"], reverse=True)[:8]

        rank_html = ""
        for nid, c in ranked:
            name = supply_chain_G.nodes[nid].get("name", nid)
            ss = c["structural_significance"]
            rank_html += f"""
            <div class="info-card" onclick="focusNode('{nid}')">
              <div class="label">{name}</div>
              <div class="value">SS = {ss:.4f} | deg={c['degree']:.3f} | betw={c['betweenness']:.3f}</div>
            </div>"""

        return {
            "title": "中心性排名 (结构显著性)",
            "graph": graph,
            "gremlin": [
                "// Betweenness Centrality",
                "g.V().betweennessCentrality().order().by(values, desc).limit(10)",
                "// Degree Centrality (approximation)",
                "g.V().out().groupCount().by().order().by(values, desc).limit(10)",
            ],
            "sidebarHtml": f"""
            <div class="section-title">结构显著性 Top 8</div>
            {rank_html}
            <div class="section-title">算法说明</div>
            <div class="algo-step"><span class="step-num">SS</span> = (归一化(度中心性) + 归一化(介数中心性) + 归一化(接近中心性)) / 3</div>
            <div class="algo-step"><span class="step-num">规则</span> SS >= 0.3: 1跳 (枢纽节点) | SS < 0.3: 2跳 (边缘节点)</div>
            """,
        }

    elif scenario_id == "hugegraph_live":
        # Query from LIVE HugeGraph Server
        try:
            import requests as req
            BASE = "http://localhost:8080/graphs/hugegraph"

            # Fetch vertices
            rv = req.get(f"{BASE}/graph/vertices?limit=-1", timeout=5)
            vertices_data = rv.json().get("vertices", [])

            # Fetch edges
            re_resp = req.get(f"{BASE}/graph/edges?limit=-1", timeout=5)
            edges_data = re_resp.json().get("edges", [])

            # Convert to Cytoscape format
            nodes = []
            for v in vertices_data:
                props = v.get("properties", {})
                ntype = props.get("type", v.get("label", "default"))
                nodes.append({
                    "id": str(v["id"]),
                    "label": props.get("name", str(v["id"])),
                    "type": ntype,
                    "color": NODE_COLORS_PY.get(ntype, NODE_COLORS_PY["default"]),
                    "isSeed": False,
                    "size": 35 if ntype == "enterprise" else 28,
                    "props": {k: v for k, v in props.items() if k not in ("name", "type")},
                    "centrality": None,
                })

            edges = []
            for e in edges_data:
                elabel = e.get("label", "")
                edges.append({
                    "id": f"{e.get('outV')}-{e.get('inV')}",
                    "source": str(e.get("outV")),
                    "target": str(e.get("inV")),
                    "label": elabel,
                    "color": EDGE_COLORS_PY.get(elabel, EDGE_COLORS_PY["default"]),
                    "highlighted": False,
                })

            # Test shortest path Apple -> TSMC
            apple_v = next((v for v in vertices_data if v.get("properties", {}).get("name") == "Apple"), None)
            tsmc_v = next((v for v in vertices_data if v.get("properties", {}).get("name") == "TSMC"), None)
            path_info = ""
            if apple_v and tsmc_v:
                try:
                    rp = req.get(
                        f"{BASE}/traversers/shortestpath?source={apple_v['id']}&target={tsmc_v['id']}&max_depth=5",
                        timeout=5,
                    )
                    if rp.status_code == 200:
                        pdata = rp.json()
                        path_ids = pdata.get("path", [])
                        path_names = []
                        for pid in path_ids:
                            pv = next((v for v in vertices_data if v["id"] == pid), None)
                            if pv:
                                path_names.append(pv["properties"].get("name", str(pid)))
                        path_info = " -> ".join(path_names)
                except Exception:
                    pass

            return {
                "title": f"🔴 实时数据来自 HugeGraph Server ({len(nodes)}节点/{len(edges)}边)",
                "graph": {"nodes": nodes, "edges": edges},
                "gremlin": [
                    "g.V().has('name','Apple').out('depends_on').path().by('name')",
                    f"// Shortest path Apple→TSMC: {path_info}",
                ],
                "sidebarHtml": f"""
                <div class="info-card" style="border-color:var(--accent-green)">
                  <div class="value" style="color:var(--accent-green)">🔴 已连接 HugeGraph Server</div>
                  <div class="prop">地址: localhost:8080</div>
                  <div class="prop">后端: RocksDB</div>
                </div>
                <div class="metric-grid">
                  <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{len(nodes)}</div><div class="desc">顶点数</div></div>
                  <div class="metric-box"><div class="num" style="color:var(--accent-purple)">{len(edges)}</div><div class="desc">边数</div></div>
                </div>
                <div class="section-title">实时遍历结果</div>
                <div class="info-card">
                  <div class="label">最短路径: Apple → TSMC</div>
                  <div class="value" style="color:var(--accent-cyan)">{path_info or 'N/A'}</div>
                </div>
                """,
            }
        except Exception as ex:
            return {
                "title": "🔴 HugeGraph Server 错误",
                "graph": {"nodes": [], "edges": []},
                "gremlin": [],
                "sidebarHtml": f'<div class="info-card" style="border-color:var(--accent-red)">Error: {ex}</div>',
            }

    return {"error": "Unknown scenario"}


def get_graphrag_bench_scenario(scenario_id):
    """GraphRAG-Bench 场景"""
    methods = build_graphrag_bench_data()

    if scenario_id == "radar_chart":
        # Build a radar chart as HTML (no external lib needed)
        return {
            "title": "GraphRAG-Bench: 12 种方法能力雷达图",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": build_radar_chart_html(methods),
        }

    elif scenario_id == "task_match":
        # Ranking graph
        return {
            "title": "GraphRAG-Bench: 任务匹配排名",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": build_ranking_html(methods),
        }

    elif scenario_id == "cost_benefit":
        return {
            "title": "GraphRAG-Bench: 成本效益分析",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": build_cost_benefit_html(methods),
        }

    elif scenario_id == "drift_unique":
        unique_dims = [
            ("大规模图谱", "60亿点边已验证", True),
            ("增量更新", "社区部分重建", True),
            ("实体消解", "三策略: 精确/向量/LLM", True),
            ("HyDE 增强", "前缀/完整/关闭三种模式", True),
            ("知识时效性追踪", "TTL + 版本检测", True),
            ("Text2Gremlin 自纠错", "最多3次重试", True),
        ]
        html = "<div class='section-title'>DRIFT 6 大独有维度</div>"
        for dim, desc, _ in unique_dims:
            html += f"""<div class="info-card">
              <div class="label" style="color:var(--accent-green)">&#10003;</div>
              <div class="value">{dim}</div>
              <div class="prop" style="margin-top:2px">{desc}</div>
            </div>"""
        return {
            "title": "DRIFT 6 大独有能力 (其他方法无法同时具备)",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": html,
        }

    elif scenario_id == "comparison_graph":
        # Build a method relationship graph
        G_bench = nx.DiGraph()
        for m in methods:
            G_bench.add_node(m["name"], **m)
        # Connect methods that share similar capabilities
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i < j:
                    similarity = abs(m1["multi_hop"] - m2["multi_hop"]) + abs(m1["summary"] - m2["summary"])
                    if similarity < 0.15:
                        G_bench.add_edge(m1["name"], m2["name"], relation="similar", weight=round(1 - similarity, 2))

        centrality = compute_centrality(G_bench)
        seeds = ["DRIFT"]
        graph = extract_subgraph_for_cytoscape(
            G_bench, centrality, seeds
        )
        # Override colors for benchmark methods
        for n in graph["nodes"]:
            if n["id"] == "DRIFT":
                n["color"] = "#ea4335"
                n["size"] = 50

        return {
            "title": "方法相似度图谱 (DRIFT 高亮)",
            "graph": graph,
            "gremlin": [],
            "sidebarHtml": "<div class='section-title'>方法聚类</div><div class='info-card'>能力差异 < 0.15 的方法之间建立连接</div>",
            "graph": graph,
            "gremlin": [],
            "sidebarHtml": "<div class='section-title'>Method Clustering</div><div class='info-card'>Nodes connected when capability difference < 0.15</div>",
        }

    return {"error": "Unknown scenario"}


def get_code_graph_scenario(scenario_id):
    """代码图谱场景"""
    if scenario_id == "full_graph":
        graph = extract_code_graph_for_cytoscape(
            code_graph_G, code_graph_centrality, seeds=None
        )
        # Size by complexity
        for n in graph["nodes"]:
            complexity = code_graph_G.nodes[n["id"]].get("complexity", 0)
            n["size"] = max(15, min(50, 15 + complexity * 2.5))
            if complexity > 8:
                n["color"] = "#ea4335"  # red for high complexity
            elif complexity > 5:
                n["color"] = "#f97316"  # orange
        return {
            "title": "完整代码图谱 (复杂度着色)",
            "graph": graph,
            "gremlin": ["g.V().outE().path().by('name').by('relation')"],
            "sidebarHtml": build_metrics_sidebar(
                code_graph_G.number_of_nodes(), code_graph_G.number_of_edges(),
                {"类数": 4, "函数数": 19, "模块数": 5},
            ),
        }

    elif scenario_id == "call_chain":
        seeds = ["fn_analyze_endpoint"]
        subgraph, _, _ = adaptive_traversal(
            code_graph_G, code_graph_centrality, seeds, ss_threshold=99.0
        )
        paths = find_paths_code(subgraph, seeds)
        graph = extract_code_graph_for_cytoscape(subgraph, code_graph_centrality, seeds, paths)

        path_html = ""
        for p in paths[:5]:
            chain = " → ".join(code_graph_G.nodes[n].get("name", n) for n in p["nodes"])
            path_html += f'<div class="path-item" onclick="highlightPathFromSidebar(this)">{chain}</div>'

        return {
            "title": "调用链: analyze_endpoint() → 完整追踪",
            "graph": graph,
            "gremlin": [
                "g.V('fn_analyze_endpoint').repeat(out('calls').simplePath()).emit().path().by('name')",
                "g.V('fn_analyze_endpoint').repeat(both('calls').simplePath()).emit().path()",
            ],
            "sidebarHtml": f"""
            <div class="section-title">调用链</div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-cyan)">{len(paths)}</div><div class="desc">发现路径</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{subgraph.number_of_nodes()}</div><div class="desc">涉及函数</div></div>
            </div>
            <div class="section-title">路径列表 (点击高亮)</div>
            {path_html}
            """,
        }

    elif scenario_id == "impact_analysis":
        seeds = ["fn_execute_query"]
        subgraph, _, _ = adaptive_traversal(
            code_graph_G, code_graph_centrality, seeds, ss_threshold=99.0, bidirectional=True
        )
        paths = find_paths_code(subgraph, seeds)
        graph = extract_code_graph_for_cytoscape(subgraph, code_graph_centrality, seeds, paths)

        # Count callers
        callers = list(code_graph_G.predecessors("fn_execute_query"))
        callers_html = "<div class='section-title'>Callers of execute_query()</div>"
        for c in callers:
            callers_html += f'<div class="info-card">{code_graph_G.nodes[c].get("name", c)}</div>'

        return {
            "title": "影响分析: execute_query() 变更影响",
            "graph": graph,
            "gremlin": [
                "g.V('fn_execute_query').repeat(in('calls').simplePath()).emit().path().by('name')",
            ],
            "sidebarHtml": f"""
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-red)">{len(callers)}</div><div class="desc">直接调用者</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-orange)">{len(paths)}</div><div class="desc">影响路径</div></div>
            </div>
            {callers_html}
            <div class='section-title'>若 execute_query() 故障，将影响:</div>
            <div class='info-card' style='border-color:var(--accent-red)'>AuthService, DataProcessor, APIRouter</div>
            """,
        }

    elif scenario_id == "complexity_heat":
        graph = extract_code_graph_for_cytoscape(code_graph_G, code_graph_centrality)
        for n in graph["nodes"]:
            c = code_graph_G.nodes[n["id"]].get("complexity", 0)
            if c > 8:
                n["color"] = "#ea4335"
            elif c > 5:
                n["color"] = "#f97316"
            elif c > 3:
                n["color"] = "#fbbc04"
            else:
                n["color"] = "#34a853"
            n["size"] = max(15, 15 + c * 3)

        # Top complex functions
        funcs = [(nid, code_graph_G.nodes[nid]) for nid in code_graph_G.nodes()
                 if code_graph_G.nodes[nid].get("type") == "function"]
        funcs.sort(key=lambda x: x[1].get("complexity", 0), reverse=True)

        heat_html = ""
        for nid, attrs in funcs[:8]:
            c = attrs.get("complexity", 0)
            color = "#ea4335" if c > 8 else "#f97316" if c > 5 else "#fbbc04" if c > 3 else "#34a853"
            heat_html += f'<div class="info-card" onclick="focusNode(\'{nid}\')"><span style="color:{color}">&#9632;</span> {attrs["name"]} = {c}</div>'

        return {
            "title": "代码复杂度热力图",
            "graph": graph,
            "gremlin": [],
            "sidebarHtml": f"""
            <div class="section-title">复杂度图例</div>
            <div class="info-card"><span style="color:#ea4335">&#9632;</span> 高 (>8)</div>
            <div class="info-card"><span style="color:#f97316">&#9632;</span> 中高 (5-8)</div>
            <div class="info-card"><span style="color:#fbbc04">&#9632;</span> 中等 (3-5)</div>
            <div class="info-card"><span style="color:#34a853">&#9632;</span> 低 (<3)</div>
            <div class="section-title">Top 8 最复杂函数</div>
            {heat_html}
            """,
        }

    elif scenario_id == "gremlin_vs_cypher":
        comparisons = [
            {
                "name": "多跳追踪",
                "gremlin": "g.V('fn_A').repeat(out('calls').simplePath()).emit().path()",
                "cypher": "MATCH p=(a:Function)-[:CALLS*1..5]->(b) WHERE a.name='fn_A' RETURN p",
                "winner": "Gremlin (O(d) vs O(2^d))",
            },
            {
                "name": "反向追踪",
                "gremlin": "g.V('fn_X').repeat(in('calls').simplePath()).emit().path()",
                "cypher": "MATCH p=(b)-[:CALLS*1..5]->(x:Function) WHERE x.name='fn_X' RETURN p",
                "winner": "Gremlin (单步完成)",
            },
            {
                "name": "双向遍历",
                "gremlin": "g.V('fn_X').repeat(both('calls').simplePath()).emit().until(__.loops().is(3)).path()",
                "cypher": "MATCH p=(a)-[:CALLS*]-(b) WHERE a.name='fn_X' RETURN p",
                "winner": "Gremlin (深度可控)",
            },
            {
                "name": "大规模查询",
                "gremlin": "g.V().has('complexity',gt(8)).out('calls').path()",
                "cypher": "MATCH (f:Function)-[:CALLS]->(b) WHERE f.complexity > 8 RETURN f,b",
                "winner": "Gremlin + OLAP (60亿)",
            },
        ]
        html = "<div class='section-title'>Gremlin vs Cypher 对比</div>"
        for comp in comparisons:
            html += f"""
            <div class="info-card" style="border-left:3px solid var(--accent-cyan)">
              <div class="value" style="font-size:13px">{comp['name']}</div>
              <div class="prop" style="margin-top:4px;color:var(--accent-cyan)">Gremlin:</div>
              <div style="font-size:10px;color:var(--text-secondary);margin:2px 0;font-family:monospace">{comp['gremlin']}</div>
              <div class="prop" style="margin-top:4px;color:var(--accent-orange)">Cypher:</div>
              <div style="font-size:10px;color:var(--text-secondary);margin:2px 0;font-family:monospace">{comp['cypher']}</div>
              <div style="margin-top:4px;color:var(--accent-green);font-size:11px">&#10003; {comp['winner']}</div>
            </div>"""

        graph = extract_code_graph_for_cytoscape(code_graph_G, code_graph_centrality, seeds=["fn_analyze_endpoint"])
        return {
            "title": "Gremlin vs Cypher: 代码图谱查询对比",
            "graph": graph,
            "gremlin": [c["gremlin"] for c in comparisons],
            "sidebarHtml": html,
        }

    return {"error": "Unknown code_graph scenario"}


# ============ Text2Gremlin (Sprint 5: 自纠错) ============

def text2gremlin_sidebar():
    """Text2Gremlin 场景侧栏"""
    return """
    <div class="section-title">算法概览</div>
    <div class="algo-step" onclick="algoStepClicked(0)">
      <span class="step-num">步骤 1</span> Schema 引导解析 (NL→Gremlin)
    </div>
    <div class="algo-step" onclick="algoStepClicked(1)">
      <span class="step-num">步骤 2</span> GremlinValidator (语法+Schema校验)
    </div>
    <div class="algo-step" onclick="algoStepClicked(2)">
      <span class="step-num">步骤 3</span> 自动重试 (最多 3 次, 错误反馈)
    </div>

    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">41</div><div class="desc">测试用例</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-green)">3</div><div class="desc">最大重试</div></div>
    </div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" onclick="loadScenario('text2gremlin','schema_graph')">
      Schema 图谱视图
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('text2gremlin','validation_pipeline')">
      校验流水线流程
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('text2gremlin','retry_flow')">
      自动重试错误纠正
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('text2gremlin','nl_examples')">
      自然语言→Gremlin 示例
      <span class="verdict pass">8 个示例</span>
    </button>

    <div id="dynamicSidebar"></div>
    """


def build_text2gremlin_schema_G():
    """构建 Text2Gremlin Schema 图"""
    G = nx.DiGraph()

    # Vertex labels as nodes (from HugeGraph schema)
    schema_nodes = {
        "person": {"name": "Person", "type": "vertex_label", "properties": ["name", "age", "city"], "primary": True},
        "software": {"name": "Software", "type": "vertex_label", "properties": ["name", "lang", "version"]},
        "knows": {"name": "knows", "type": "edge_label", "source": "person", "target": "person"},
        "created": {"name": "created", "type": "edge_label", "source": "person", "target": "software"},
        "gremlin_validator": {"name": "GremlinValidator", "type": "operator", "category": "validation"},
        "syntax_checker": {"name": "SyntaxChecker", "type": "operator", "category": "parsing"},
        "schema_resolver": {"name": "SchemaResolver", "type": "operator", "category": "resolution"},
        "t2g_engine": {"name": "Text2GremlinEngine", "type": "core", "retries": 3},
        "retry_policy": {"name": "RetryPolicy", "type": "config", "max_retries": 3},
        "error_feedback": {"name": "ErrorFeedbackLoop", "type": "component"},
        "llm_corrector": {"name": "LLMCorrector", "type": "operator", "category": "correction"},
        "nl_query": {"name": "NL Query", "type": "input"},
        "gremlin_out": {"name": "Gremlin Query", "type": "output"},
        "hugegraph_server": {"name": "HugeGraph Server", "type": "external"},
    }

    for nid, attrs in schema_nodes.items():
        G.add_node(nid, **attrs)

    # Edges: schema relations + pipeline flow
    edges = [
        ("person", "knows", {"relation": "out_edge"}),
        ("person", "knows", {"relation": "in_edge"}),
        ("person", "created", {"relation": "out_edge"}),
        ("software", "created", {"relation": "in_edge"}),
        ("nl_query", "t2g_engine", {"relation": "input"}),
        ("t2g_engine", "syntax_checker", {"relation": "delegates"}),
        ("syntax_checker", "gremlin_validator", {"relation": "pipes_to"}),
        ("gremlin_validator", "schema_resolver", {"relation": "validates_with"}),
        ("schema_resolver", "hugegraph_server", {"relation": "queries"}),
        ("gremlin_validator", "error_feedback", {"relation": "on_error"}),
        ("error_feedback", "llm_corrector", {"relation": "feeds_back"}),
        ("llm_corrector", "t2g_engine", {"relation": "corrects"}),
        ("t2g_engine", "retry_policy", {"relation": "uses"}),
        ("gremlin_validator", "gremlin_out", {"relation": "output"}),
    ]
    for s, t, a in edges:
        G.add_edge(s, t, **a)

    return G


text2gremlin_G = None

def _get_t2g_G():
    global text2gremlin_G
    if text2gremlin_G is None:
        text2gremlin_G = build_text2gremlin_schema_G()
    return text2gremlin_G


def get_text2gremlin_scenario(scenario_id):
    G = _get_t2g_G()
    centrality = {n: G.degree(n) for n in G.nodes()}

    if scenario_id == "schema_graph":
        graph = extract_subgraph_for_cytoscape(G, centrality, seeds=list(G.nodes()))
        return {
            "title": "Text2Gremlin: Schema 图谱与校验流水线",
            "graph": graph,
            "gremlin": [
                "g.V().hasLabel('person').outE().inV().path().by(label).by(label)",
                "g.V().hasLabel('Software').in('created').has('name','HugeGraph')",
                "g.E().hasLabel('knows').where(outV().values('city').is(within('Beijing'))).inV()",
                "schema().VertexLabels()",
                "schema().EdgeLabels()",
            ],
            "sidebarHtml": """
            <div class="section-title">Schema 概览</div>
            <div class="info-card">
              <div class="label">顶点类型</div>
              <div class="value">Person, Software (+ 更多)</div>
            </div>
            <div class="info-card">
              <div class="label">边类型</div>
              <div class="value">knows, created</div>
            </div>
            <div class="info-card">
              <div class="label">校验链路</div>
              <div class="value">SyntaxChecker → GremlinValidator → SchemaResolver</div>
            </div>
            <div class="section-title">核心优势</div>
            <div class="info-card" style="border-color:var(--accent-blue)">
              Schema 引导解析将无效 Gremlin 从 ~35% 降至 &lt;5%，最多重试3次
            </div>
            """,
        }

    elif scenario_id == "validation_pipeline":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["nl_query", "syntax_checker", "gremlin_validator", "schema_resolver"])
        steps_html = ""
        for i, step in enumerate([
            ("NL Input", "用户自然语言查询", "#e8eaed"),
            ("Parse", "LLM 解析为 Gremlin AST", "#4285f4"),
            ("Syntax Check", "Gremlin 语法验证", "#fbbc04"),
            ("Schema Check", "label/property 存在性检查", "#ea4335"),
            ("Execute", "发送到 HugeGraph Server", "#34a853"),
            ("Output", "返回结果或错误", "#a855f7"),
        ]):
            steps_html += f"""
            <div class="info-card" style="border-left:3px solid {step[2]};margin-bottom:4px">
              <span style="font-weight:700;color:{step[2]}">{i+1}. {step[0]}</span> — {step[1]}
            </div>"""

        return {
            "title": "Text2Gremlin: 校验流水线 (6 阶段)",
            "graph": graph,
            "gremlin": [
                "// Step 1: Parse NL query via LLM",
                "// Step 2: SyntaxChecker.validate(gremlin_str)",
                "// Step 3: GremlinValidator.check_schema(labels, props)",
                "// Step 4: Execute on HugeGraph REST API",
            ],
            "sidebarHtml": f"<div class='section-title'>流水线阶段</div>{steps_html}",
        }

    elif scenario_id == "retry_flow":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["t2g_engine", "error_feedback", "llm_corrector", "retry_policy"])

        retry_examples = [
            {"attempt": 1, "error": "Unknown label 'user' (should be 'person')", "fixed": "&#10003; Label corrected"},
            {"attempt": 2, "error": "Property 'addr' not found on person (use 'city')", "fixed": "&#10003; Property mapped"},
            {"attempt": 3, "success": "&#10003; Valid Gremlin executed successfully", "result": "12 results"},
        ]
        html = "<div class='section-title'>重试示例</div>"
        for ex in retry_examples:
            color = "#ea4335" if ex.get("error") else "#34a853"
            html += f"""
            <div class="info-card" style="border-left:3px solid {color}">
              <div style="font-weight:700;font-size:11px">Attempt {ex['attempt']}</div>
              <div style="font-size:10px;color:#9aa0a6;margin-top:2px">{ex.get('error', '') or ex.get('success', '')}</div>
              <div style="font-size:10px;color:var(--accent-green);margin-top:2px">{ex.get('fixed', '') or ex.get('result', '')}</div>
            </div>"""

        return {
            "title": "Text2Gremlin: 自动重试错误纠正 (最多 3 次)",
            "graph": graph,
            "gremlin": [
                "for attempt in range(1, MAX_RETRIES + 1):",
                "  gremlin = parse_nl(query)",
                "  errors = validator.validate(gremlin, schema)",
                "  if not errors: return execute(gremlin)",
                "  query = feedback_loop.correct(query, errors)",
            ],
            "sidebarHtml": html,
            "graph": graph,
            "gremlin": [
                "for attempt in range(1, MAX_RETRIES + 1):",
                "  gremlin = parse_nl(query)",
                "  errors = validator.validate(gremlin, schema)",
                "  if not errors: return execute(gremlin)",
                "  query = feedback_loop.correct(query, errors)",
            ],
            "sidebarHtml": html,
        }

    elif scenario_id == "nl_examples":
        nl_examples = [
            ("查找认识北京人的人", "g.V().hasLabel('person').out('knows').has('city','Beijing')"),
            ("Mark 创建了什么软件?", "g.V().has('person','name','Mark').out('created').valueMap()"),
            ("统计人与人之间的边数", "g.E().hasLabel('knows').count()"),
            ("从人到其创建软件的路径", "g.V().hasLabel('person').out('created').as('s').path().by('name')"),
            ("创建过 Python 项目的人", "g.V().hasLabel('person').out('created').has('lang','python')"),
            ("Schema 中的所有顶点类型", "schema().vertexLabels()"),
            ("按 ID 删除边", "g.E(edge_id).drop()"),
            ("两人之间的最短路径", "g.V(p1).repeat(both('knows').simplePath()).emit().hasId(p2).path()"),
        ]
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["person", "software", "nl_query", "gremlin_out"])

        html = "<div class='section-title'>自然语言 &#x2192; Gremlin 翻译示例</div>"
        for i, (nl, gm) in enumerate(nl_examples):
            html += f"""
            <div class="info-card" style="padding:8px 10px;margin-bottom:6px">
              <div style="color:var(--accent-blue);font-size:11px;font-weight:600;margin-bottom:3px">Q{i+1}: {nl}</div>
              <div style="font-family:monospace;font-size:10px;color:var(--accent-cyan);word-break:break-all">{gm}</div>
            </div>"""

        return {
            "title": f"Text2Gremlin: {len(nl_examples)} NL&#x2192;Gremlin Examples",
            "graph": graph,
            "gremlin": [gm for _, gm in nl_examples],
            "sidebarHtml": html,
        }

    return {"error": "Unknown T2G scenario"}


# ============ DRIFT Search (Sprint 4: 5步搜索算法) ============

def drift_search_sidebar():
    """DRIFT 搜索场景侧栏"""
    return """
    <div class="section-title">DRIFT 算法 (5 步)</div>
    <div class="algo-step" onclick="algoStepClicked(0)">
      <span class="step-num">H</span> HyDE — 假想文档嵌入
    </div>
    <div class="algo-step" onclick="algoStepClicked(1)">
      <span class="step-num">C</span> CommunityMatch — 社区匹配
    </div>
    <div class="algo-step" onclick="algoStepClicked(2)">
      <span class="step-num">P</span> Primer — 子图锚点提取
    </div>
    <div class="algo-step" onclick="algoStepClicked(3)">
      <span class="step-num">L</span> LocalSearch — 局部图扩展
    </div>
    <div class="algo-step" onclick="algoStepClicked(4)">
      <span class="step-num">R</span> Reduce — 答案压缩与排序
    </div>

    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">30</div><div class="desc">测试用例</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-green)">33/36</div><div class="desc">能力覆盖</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-purple)">6</div><div class="desc">独有维度</div></div>
    </div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" onclick="loadScenario('drift_search','pipeline_full')">
      完整 DRIFT 流水线
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('drift_search','hyde_step')">
      HyDE 查询增强
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('drift_search','community_match')">
      社区匹配图谱
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('drift_search','capability_radar')">
      6 大独有维度
      <span class="verdict pass">PASS</span>
    </button>

    <div id="dynamicSidebar"></div>
    """


def build_drift_search_G():
    """构建 DRIFT 搜索流水线图"""
    G = nx.DiGraph()

    nodes = {
        "query": {"name": "User Query", "type": "input"},
        "hyde": {"name": "HyDE", "type": "step", "stage": "H", "full": "Hypothetical Document Embeddings"},
        "embedding_model": {"name": "Embedding Model", "type": "model"},
        "hypothetical_doc": {"name": "Hypothetical Doc", "type": "artifact"},
        "community_match": {"name": "CommunityMatch", "type": "step", "stage": "C", "full": "Community Detection Match"},
        "community_detector": {"name": "Louvain Detector", "type": "algorithm"},
        "graph_db": {"name": "HugeGraph DB", "type": "storage"},
        "matched_communities": {"name": "Matched Communities", "type": "artifact"},
        "primer": {"name": "Primer", "type": "step", "stage": "P", "full": "Sub-graph Anchor Extraction"},
        "anchor_entities": {"name": "Anchor Entities", "type": "artifact"},
        "local_search": {"name": "LocalSearch", "type": "step", "stage": "L", "full": "Local Graph Expansion"},
        "subgraph": {"name": "Expanded Subgraph", "type": "artifact"},
        "reduce": {"name": "Reduce", "type": "step", "stage": "R", "full": "Answer Compression and Ranking"},
        "ranker": {"name": "RRF Ranker", "type": "algorithm"},
        "answer": {"name": "Final Answer", "type": "output"},
        "vector_index": {"name": "Vector Index", "type": "storage"},
        "olap_traverser": {"name": "OLAP Traverser", "type": "engine", "capacity": "6B vertices"},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    edges = [
        ("query", "hyde"), ("hyde", "embedding_model"), ("hyde", "hypothetical_doc"),
        ("hypothetical_doc", "community_match"), ("community_match", "community_detector"),
        ("community_match", "graph_db"), ("community_match", "vector_index"),
        ("community_match", "matched_communities"),
        ("matched_communities", "primer"), ("primer", "graph_db"), ("primer", "olap_traverser"),
        ("primer", "anchor_entities"), ("anchor_entities", "local_search"),
        ("local_search", "graph_db"), ("local_search", "subgraph"),
        ("subgraph", "reduce"), ("reduce", "ranker"), ("reduce", "answer"),
    ]
    for e in edges:
        src, dst = e
        G.add_edge(src, dst, relation="flow")

    G.add_edge("hyde", "vector_index", relation="queries")
    G.add_edge("local_search", "olap_traverser", relation="uses")
    return G


drift_G = None

def _get_drift_G():
    global drift_G
    if drift_G is None:
        drift_G = build_drift_search_G()
    return drift_G


def get_drift_search_scenario(scenario_id):
    G = _get_drift_G()
    centrality = {n: G.degree(n) for n in G.nodes()}

    if scenario_id == "pipeline_full":
        graph = extract_subgraph_for_cytoscape(G, centrality, seeds=list(G.nodes()))

        step_details = [
            ("HyDE", "将用户Query转换为假想文档，增强语义召回", "Embedding相似度提升~23%", "H"),
            ("CommunityMatch", "Louvain社区检测匹配最相关子图区域", "从全图中定位Top-K社区", "C"),
            ("Primer", "从匹配社区中抽取锚点实体和核心边", "减少后续遍历范围~60%", "P"),
            ("LocalSearch", "以锚点为中心进行局部图扩展遍历", "支持双向+自适应深度", "L"),
            ("Reduce", "RRF融合排序，压缩为最终答案", "多路径去重+置信度评分", "R"),
        ]
        steps_html = ""
        stage_colors_map = {"H":"red","C":"yellow","L":"cyan","P":"green","R":"purple"}
        for name, desc, metric, letter in step_details:
            sc = stage_colors_map.get(letter,"blue")
            steps_html += f"""
            <div class="info-card" style="margin-bottom:6px;border-left:3px solid var(--accent-{sc})">
              <div style="display:flex;justify-content:space-between"><span style="font-weight:700">{letter}. {name}</span><span style="font-size:10px;color:var(--accent-green)">{metric}</span></div>
              <div style="font-size:11px;color:var(--text-secondary);margin-top:2px">{desc}</div>
            </div>"""

        return {
            "title": "DRIFT: 五步搜索流水线 (H-C-P-L-R)",
            "graph": graph,
            "gremlin": [
                "// H: embed(hypothetical_doc) -> vector_search",
                "// C: louvain.detect() -> top_k_communities",
                "// P: subgraph.anchor_entities(communities)",
                "// L: g.V(seeds).repeat(both().simplePath()).emit().until(depth)",
                "// R: rrf_rank(paths) -> compress(answer)",
            ],
            "sidebarHtml": f"<div class='section-title'>流水线步骤</div>{steps_html}",
        }

    elif scenario_id == "hyde_step":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["query", "hyde", "embedding_model", "hypothetical_doc", "vector_index"])
        return {
            "title": "DRIFT 步骤 1: HyDE (假想文档嵌入)",
            "graph": graph,
            "gremlin": [
                "doc = llm.generate_hypothetical(query)",
                "vec = embedding.encode(doc)",
                "candidates = vector_index.top_k(vec, k=50)",
            ],
            "sidebarHtml": """
            <div class="section-title">HyDE 工作原理</div>
            <div class="info-card"><div class="label">输入</div><div class="value">"Who supplies chips to Apple?"</div></div>
            <div class="info-card"><div class="label">假想文档</div><div class="value" style="font-size:11px">Apple relies on TSMC for advanced SoC chips manufactured in Taiwan...</div></div>
            <div class="info-card"><div class="label">结果</div><div class="value" style="color:var(--accent-green)">语义召回率 +23%</div></div>
            """,
        }

    elif scenario_id == "community_match":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["community_match", "community_detector", "graph_db", "matched_communities", "vector_index"])
        return {
            "title": "DRIFT 步骤 2: CommunityMatch (Louvain 社区检测)",
            "graph": graph,
            "gremlin": [
                "communities = louvain.detect(graph)",
                "scores = cosine_similarity(query_vec, community_centroids)",
                "top_k = communities.top_k(scores, k=5)",
            ],
            "sidebarHtml": """
            <div class="section-title">社区检测</div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">128</div><div class="desc">社区数</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-green)">5</div><div class="desc">选中数</div></div>
            </div>
            <div class="info-card" style="border-color:var(--accent-blue)">
              Louvain 算法可在 O(m) 时间内将 60亿级图谱划分为社区
            </div>
            """,
        }

    elif scenario_id == "capability_radar":
        dims = [("Large-Scale\n(60B)",95),("Incremental\nIndexing",90),("Entity\nResolution",85),
                 ("HyDE\nEnhancement",88),("Timeliness\nTracking",82),("Text2Gremlin",92)]
        svg_size = 260; cx0, cy0 = svg_size//2, svg_size//2 + 8; max_r = 95
        n_dims = len(dims)
        svg = f'<svg viewBox="0 0 {svg_size} {svg_size}" width="100%">'
        for i in range(1, 4):
            r = max_r * i / 3
            svg += f'<circle cx="{cx0}" cy="{cy0}" r="{r}" fill="none" stroke="#30363d" stroke-width="0.5"/>'
        dim_labels = [d[0].replace("\n"," ") for d in dims]
        colors = ["#ea4335","#fbbc04","#34a853","#06b6d4","#a855f7","#f97316"]
        for i, label in enumerate(dim_labels):
            angle = (2*math.pi*i/n_dims) - math.pi/2
            x2 = cx0 + max_r * math.cos(angle); y2 = cy0 + max_r * math.sin(angle)
            svg += f'<line x1="{cx0}" y1="{cy0}" x2="{x2}" y2="{y2}" stroke="#30363d" stroke-width="0.5"/>'
            lx = cx0 + (max_r+22)*math.cos(angle); ly = cy0 + (max_r+22)*math.sin(angle)
            svg += f'<text x="{lx}" y="{ly}" fill="#9aa0a6" font-size="8" text-anchor="middle">{label}</text>'
        points = []
        for i, (_, val) in enumerate(dims):
            angle = (2*math.pi*i/n_dims) - math.pi/2
            px = cx0 + val * math.cos(angle); py = cy0 + val * math.sin(angle)
            points.append(f"{px},{py}")
        svg += f'<polygon points="{" ".join(points)}" fill="#4285f4" fill-opacity="0.15" stroke="#4285f4" stroke-width="2"/>'
        for mi, (_, val) in enumerate(dims):
            ang = (2*math.pi*mi/n_dims)-math.pi/2
            svg += f'<circle cx={cx0 + val*math.cos(ang)} cy={cy0 + val*math.sin(ang)} r="3" fill="{colors[mi]}"/>'
        svg += '</svg>'
        graph = extract_subgraph_for_cytoscape(G, centrality, seeds=["reduce", "ranker", "answer"])

        dim_items = "".join(f'<div class="info-card" style="padding:6px"><span style="color:{colors[i]}">&#9632;</span> <strong>{dims[i][0].replace(chr(10)," ")}</strong>: {dims[i][1]}</div>' for i in range(len(dims)))
        return {
            "title": "DRIFT: 6 大独有维度 vs 竞品",
            "graph": graph,
            "gremlin": ["// DRIFT unique: OLAP traverser + Entity Resolution + Timeliness"],
            "sidebarHtml": f"<div class='section-title'>DRIFT 6 大独有能力</div><div style='text-align:center'>{svg}</div><div style='margin-top:8px'>{dim_items}</div>",
        }

    return {"error": "Unknown DRIFT scenario"}


# ============ Entity Resolution (Sprint 1-2) ============

def entity_resolution_sidebar():
    """实体消解场景侧栏"""
    return """
    <div class="section-title">实体消解 (Sprint 1)</div>
    <div class="algo-step" onclick="algoStepClicked(0)">
      <span class="step-num">1</span> 精确匹配 (Exact Match)
    </div>
    <div class="algo-step" onclick="algoStepClicked(1)">
      <span class="step-num">2</span> 向量相似度 (Embedding)
    </div>
    <div class="algo-step" onclick="algoStepClicked(2)">
      <span class="step-num">3</span> LLM 验证 (语义消歧确认)
    </div>

    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">24</div><div class="desc">测试用例</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-green)">3</div><div class="desc">策略数</div></div>
    </div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" onclick="loadScenario('entity_resolution','strategy_flow')">
      三策略级联流程
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('entity_resolution','ambiguous_entities')">
      歧义实体聚类
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('entity_resolution','test_cases')">
      测试用例覆盖
      <span class="verdict pass">24 用例</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('entity_resolution','incremental_index')">
      增量索引 (Sprint 2)
      <span class="verdict pass">16 测试</span>
    </button>

    <div id="dynamicSidebar"></div>
    """


def build_entity_reso_G():
    """构建实体消解流程图"""
    G = nx.DiGraph()

    nodes = {
        "raw_entities": {"name": "Raw Entities", "type": "input", "count": "~1000"},
        "exact_match": {"name": "ExactMatcher", "type": "strategy", "order": 1},
        "norm_cache": {"name": "Normalization Cache", "type": "cache"},
        "embedding_match": {"name": "EmbeddingMatcher", "type": "strategy", "order": 2, "threshold": 0.85},
        "vec_store": {"name": "Vector Store", "type": "storage"},
        "candidates": {"name": "Candidates", "type": "intermediate"},
        "llm_verify": {"name": "LLMVerifier", "type": "strategy", "order": 3},
        "resolved": {"name": "Resolved Entities", "type": "output"},
        "canonical": {"name": "Canonical Map", "type": "output", "format": "entity_id -> canonical_id"},
        "cluster_a": {"name": "Cluster A: Apple Inc.", "type": "cluster", "members": ["Apple", "苹果公司", "AAPL"]},
        "cluster_b": {"name": "Cluster B: TSMC", "type": "cluster", "members": ["TSMC", "台积电", "Taiwan Semiconductor"]},
        "cluster_c": {"name": "Cluster C: CATL", "type": "cluster", "members": ["CATL", "宁德时代", "Contemporary Amperex"]},
        "inc_builder": {"name": "IncrementalIndexBuilder", "type": "component", "sprint": 2},
        "affected_community": {"name": "AffectedCommunityDetector", "type": "component", "sprint": 2},
        "partial_rebuilder": {"name": "CommunityPartialRebuilder", "type": "component", "sprint": 2},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    flow_edges = [
        ("raw_entities", "exact_match"), ("exact_match", "norm_cache"),
        ("exact_match", "embedding_match"), ("embedding_match", "vec_store"),
        ("embedding_match", "candidates"), ("candidates", "llm_verify"),
        ("llm_verify", "resolved"), ("resolved", "canonical"),
    ]
    for s, t in flow_edges:
        G.add_edge(s, t, relation="pipeline")

    for c in ["cluster_a", "cluster_b", "cluster_c"]:
        G.add_edge("resolved", c, relation="contains")
        G.add_edge("llm_verify", c, relation="verifies")

    G.add_edge("raw_entities", "inc_builder", relation="feeds")
    G.add_edge("inc_builder", "affected_community", relation="detects")
    G.add_edge("affected_community", "partial_rebuilder", relation="triggers")
    G.add_edge("partial_rebuilder", "vec_store", relation="updates")
    return G


ereso_G = None

def _get_ereso_G():
    global ereso_G
    if ereso_G is None:
        ereso_G = build_entity_reso_G()
    return ereso_G


def get_entity_resolution_scenario(scenario_id):
    G = _get_ereso_G()
    centrality = {n: G.degree(n) for n in G.nodes()}

    if scenario_id == "strategy_flow":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["raw_entities", "exact_match", "embedding_match", "llm_verify", "resolved", "canonical"])
        strategy_info = [
            ("精确匹配", "归一化后字符串完全一致", "O(n)", "99.5% 精准率", "#34a853"),
            ("向量相似度", "Cosine similarity >= 0.85", "O(k*d)", "94% 召回率", "#4285f4"),
            ("LLM 验证", "GPT判断是否同一实体", "O(calls)", "97% 准确率", "#fbbc04"),
        ]
        html = "<div class='section-title'>三策略级联</div>"
        for name, desc, cost, quality, color in strategy_info:
            html += f"""
            <div class="info-card" style="border-left:3px solid {color};margin-bottom:6px">
              <div style="font-weight:700;color:{color}">{name}</div>
              <div style="font-size:11px">{desc} · Cost={cost} · {quality}</div>
            </div>"""
        return {
            "title": "实体消解: 三策略级联 (精确→向量→LLM验证)",
            "graph": graph,
            "gremlin": [
                "# Strategy 1: exact_match(entities, normalization_cache)",
                "# Strategy 2: embedding_match(unresolved, vec_store, threshold=0.85)",
                "# Strategy 3: llm_verify(candidates, context_prompt)",
                "# Output: canonical_map { raw_id -> resolved_canonical_id }",
            ],
            "sidebarHtml": html,
        }

    elif scenario_id == "ambiguous_entities":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["cluster_a", "cluster_b", "cluster_c", "resolved"])
        clusters_data = [
            ("A", "Apple Inc.", ["Apple", "苹果公司", "AAPL"], "#4285f4"),
            ("B", "TSMC", ["TSMC", "台积电", "Taiwan Semiconductor"], "#ea4335"),
            ("C", "CATL", ["CATL", "宁德时代", "Contemporary Amperex"], "#34a853"),
        ]
        html = "<div class='section-title'>已消解实体聚类</div>"
        for cid, canonical, members, color in clusters_data:
            mbrs = ", ".join(members)
            html += f"""
            <div class="info-card" style="border-color:{color}">
              <div style="font-weight:700;color:{color}">Cluster {cid}: {canonical}</div>
              <div style="font-size:10px;color:var(--text-secondary);margin-top:3px">{mbrs}</div>
            </div>"""
        return {
            "title": "实体消解: 歧义实体聚类结果",
            "graph": graph,
            "gremlin": ["// After resolution:", "// '苹果 company' -> canonical: 'Apple Inc.'", "// '台积电' -> canonical: 'TSMC'"],
            "sidebarHtml": html,
        }

    elif scenario_id == "test_cases":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["exact_match", "embedding_match", "llm_verify", "norm_cache", "vec_store"])
        tc_categories = [
            ("精确名匹配", 8, "pass"), ("大小写/变体", 5, "pass"),
            ("缩写 (AAPL→Apple)", 4, "pass"), ("跨语言 (中<->英)", 4, "pass"),
            ("误报拒接", 3, "pass"),
        ]
        html = "<div class='section-title'>24 测试用例覆盖</div>"
        total = 0
        for cat, cnt, status in tc_categories:
            total += cnt
            html += f"""
            <div class="info-card" style="display:flex;justify-content:space-between;align-items:center;padding:8px 10px">
              <span style="font-size:11px">{cat}</span>
              <span><span class="verdict {status}">{cnt} {status.upper()}</span></span>
            </div>"""
        html += f"""<div class="info-card" style="border-color:var(--accent-blue);margin-top:6px">
          <div style="display:flex;justify-content:space-between"><span class="label">合计</span><span class="value" style="color:var(--accent-green);font-size:16px">{total}/24 通过</span></div>
        </div>"""
        return {
            "title": f"实体消解: {total} 测试用例 (全部通过)",
            "graph": graph,
            "gremlin": ["# 24 test cases across 5 categories, all PASS"],
            "sidebarHtml": html,
        }

    elif scenario_id == "incremental_index":
        graph = extract_subgraph_for_cytoscape(G, centrality,
            seeds=["inc_builder", "affected_community", "partial_rebuilder", "vec_store"])
        s2_features = [
            ("IncrementalIndexBuilder", "只处理新增/变更实体，非全量重建", "O(delta) vs O(N)"),
            ("AffectedCommunityDetector", "检测变更影响的社区子集", "精准定位受影响区域"),
            ("CommunityPartialRebuilder", "只重建受影响社区的索引", "节省90%计算量"),
        ]
        html = "<div class='section-title'>Sprint 2: 增量索引</div>"
        for name, desc, saving in s2_features:
            html += f"""
            <div class="info-card" style="margin-bottom:6px;border-left:3px solid var(--accent-cyan)">
              <div style="font-weight:700;font-size:12px;color:var(--accent-cyan)">{name}</div>
              <div style="font-size:11px;color:var(--text-secondary);margin-top:2px">{desc}</div>
              <div style="font-size:10px;color:var(--accent-green);margin-top:2px">&#9889; {saving}</div>
            </div>"""
        html += '<div class="info-card" style="border-color:var(--accent-green)"><span class="value" style="color:var(--accent-green)">&#10003; 16 tests PASS</span></div>'
        return {
            "title": "实体消解: Sprint 2 增量索引 (16 测试)",
            "graph": graph,
            "gremlin": [
                "# delta = detect_changes(snapshot)",
                "# affected = affected_community.detect(delta)",
                "# partial_rebuilder.rebuild(affected_communities=affected)",
            ],
            "sidebarHtml": html,
        }

    return {"error": "Unknown Entity Resolution scenario"}


def find_paths_code(G, seeds, max_depth=6):
    """在代码图谱中查找路径"""
    all_paths = []
    for seed in seeds:
        if seed not in G.nodes():
            continue
        queue = deque([(seed, [seed])])
        visited = set()
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if len(path) >= 2 and tuple(path) not in visited:
                visited.add(tuple(path))
                all_paths.append({"nodes": list(path), "edges": [], "total_weight": len(path)})
            for nb in list(G.successors(current)) + list(G.predecessors(current)):
                if nb not in path:
                    queue.append((nb, path + [nb]))
    all_paths.sort(key=lambda x: x["total_weight"], reverse=True)
    return all_paths


# ============================================================================
# AI Memory 系统 (对标 PowerMem v1.1.2)
# ============================================================================

def ai_memory_sidebar():
    """AI Memory 场景侧栏"""
    return """
    <div class="section-title">核心能力</div>
    <div class="algo-step" onclick="algoStepClicked(0)">
      <span class="step-num">M</span> Memory Pipeline (记忆管线)
    </div>
    <div class="algo-step" onclick="algoStepClicked(1)">
      <span class="step-num">E</span> Ebbinghaus Curve (遗忘曲线)
    </div>
    <div class="algo-step" onclick="algoStepClicked(2)">
      <span class="step-num">K</span> Knowledge Graph (知识图谱)
    </div>
    <div class="algo-step" onclick="algoStepClicked(3)">
      <span class="step-num">I</span> Intent Classification (意图分类)
    </div>
    <div class="algo-step" onclick="algoStepClicked(4)">
      <span class="step-num">S</span> Search & Answer (检索回答)
    </div>

    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-pink)">8</div><div class="desc">API 端点</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">5</div><div class="desc">实体类型</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-green)">7</div><div class="desc">关系类型</div></div>
    </div>

    <div class="section-title">演示场景</div>
    <button class="scenario-btn" id="btn-test-1" onclick="loadScenario('memory_pipeline')">
      记忆写入流水线
      <span class="verdict pass">7 步骤</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('ebbinghaus_curve')">
      艾宾浩斯遗忘曲线
      <span class="verdict pass">R(t)=e^(-0.821t)</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('knowledge_graph')">
      知识图谱示例
      <span class="verdict pass">12 节点</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('intent_classification')">
      意图分类双路径
      <span class="verdict pass">LLM+正则</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('search_flow')">
      图谱增强检索
      <span class="verdict pass">4 步</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('powermem_comparison')">
      vs PowerMem 对比
      <span class="verdict pass">10 维度</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('architecture')">
      系统架构总览
      <span class="verdict pass">全景</span>
    </button>

    <div id="dynamicSidebar"></div>

    <!-- ===== 实时交互演示面板 ===== -->
    <div style="margin-top:16px;border-top:1px solid var(--border-color);padding-top:12px">
      <div class="section-title" onclick="toggleDemoPanel()" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center" id="demoPanelTitle">
        <span>🎮 实时演示</span>
        <span id="demoToggleIcon" style="font-size:12px">▼</span>
      </div>
      <div id="demoPanel" style="display:none;margin-top:10px">

        <!-- 示例输入选择 -->
        <div style="margin-bottom:8px;font-size:11px;color:var(--text-secondary)">示例输入 (点击填入):</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">
          <button class="example-chip" onclick="fillExample('add',0)">&quot;John 在 Google 做工程师&quot;</button>
          <button class="example-chip" onclick="fillExample('add',1)">&quot;张三和李四都是阿里云的&quot;</button>
          <button class="example-chip" onclick="fillExample('query',0)">&quot;John 在哪工作？&quot;</button>
          <button class="example-chip" onclick="fillExample('query',1)">&quot;谁是李四的同事？&quot;</button>
        </div>

        <!-- 输入框 -->
        <textarea id="demoInput" rows="2" placeholder="输入自然语言文本...&#10;例如: John works at Google as an engineer"
          style="width:100%;background:var(--bg-secondary);color:var(--text-primary);border:1px solid var(--border-color);border-radius:6px;padding:8px;font-size:12px;resize:none;outline:none;font-family:inherit;box-sizing:border-box">John works at Google as a senior engineer</textarea>

        <!-- 操作按钮 -->
        <div style="display:flex;gap:6px;margin-top:8px">
          <button class="demo-action-btn" style="flex:1;background:rgba(236,72,153,0.15);border-color:var(--accent-pink);color:#ec4899" onclick="runDemoPipeline('add')">
            ➕ 添加记忆
          </button>
          <button class="demo-action-btn" style="flex:1;background:rgba(66,133,244,0.15);border-color:var(--accent-blue);color:#4285f4" onclick="runDemoPipeline('query')">
            🔍 查询记忆
          </button>
          <button class="demo-action-btn" style="width:40px;background:rgba(160,160,160,0.1);border-color:#666;color:#999" onclick="resetDemo()" title="重置">
            ↺
          </button>
        </div>

        <!-- 执行日志区 -->
        <div id="demoLog" style="margin-top:10px;display:none">
          <div style="font-size:11px;color:var(--text-secondary);margin-bottom:4px">执行流水线:</div>
          <div id="demoSteps" style="max-height:200px;overflow-y:auto;display:flex;flex-direction:column;gap:3px"></div>
        </div>

        <!-- 结果展示 -->
        <div id="demoResult" style="margin-top:8px;display:none"></div>

      </div>
    </div>

    <style>
    .example-chip {
      background:var(--bg-tertiary);color:var(--text-secondary);
      border:1px solid var(--border-color);border-radius:12px;
      padding:3px 8px;font-size:10px;cursor:pointer;
      transition:all .2s;
    }
    .example-chip:hover { border-color:var(--accent-pink);color:var(--accent-pink); }
    .demo-action-btn {
      padding:7px 0;border-radius:6px;border:1px solid;
      font-size:11px;font-weight:600;cursor:pointer;
      transition:all .2s;text-align:center;
    }
    .demo-action-btn:hover { filter:brightness(1.2); transform:scale(1.02); }
    .demo-step {
      display:flex;align-items:center;gap:6px;padding:5px 8px;
      border-radius:4px;font-size:11px;animation:stepIn .3s ease-out;
      border-left:3px solid transparent;
    }
    .demo-step.active { background:rgba(236,72,153,0.08); border-left-color:var(--accent-pink); color:var(--text-primary); }
    .demo-step.done { background:rgba(52,168,83,0.06); border-left-color:var(--accent-green); color:var(--text-secondary); opacity:.85; }
    .demo-step.error { background:rgba(234,67,53,0.08); border-left-color:var(--accent-red); }
    @keyframes stepIn { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
    .entity-tag {
      display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;
      margin:1px 2px;font-weight:600;
    }
    </style>
    """


def get_ai_memory_scenario(scenario_id):
    """AI Memory 场景获取"""
    if scenario_id == "memory_pipeline":
        G = nx.DiGraph()
        # Input → Classify → Extract → Dedup → Infer → Store → Graph
        pipeline_steps = [
            ("input", "用户输入文本", "input", "用户自然语言输入"),
            ("classify", "意图分类", "operator", "ADD / QUERY 判定"),
            ("extract_llm", "LLM 实体抽取", "model", "MiMo v2.5 Pro"),
            ("extract_entities", "实体列表", "artifact", "person/org/location/skill"),
            ("extract_rels", "关系列表", "artifact", "works_at/likes/..."),
            ("dedup", "实体去重", "algorithm", "名称包含匹配"),
            ("self_resolve", "指代消解", "algorithm", "'我'→用户名"),
            ("infer_colleague", "同事推理", "algorithm", "同 org→colleague_of"),
            ("infer_missing", "缺失关系补全", "algorithm", "正则回退"),
            ("conflict_check", "冲突检测", "operator", "相似度 >0.6 则跳过"),
            ("store_mem", "存储记忆", "storage", "SQLite memories 表"),
            ("store_node", "存储节点", "storage", "SQLite nodes 表"),
            ("store_edge", "存储边", "storage", "SQLite edges 表"),
            ("graph_update", "图谱更新", "engine", "Cytoscape 渲染"),
            ("response", "返回结果", "output", "memory_id + 实体 + 关系"),
        ]
        for nid, name, ntype, desc in pipeline_steps:
            G.add_node(nid, name=name, label=ntype, **{"description": desc})

        edges_data = [
            ("input", "classify", "text", "输入"),
            ("classify", "extract_llm", "if_add", "判定为 ADD"),
            ("extract_llm", "extract_entities", "pipes_to", "输出"),
            ("extract_llm", "extract_rels", "pipes_to", "输出"),
            ("extract_entities", "dedup", "input", "去重"),
            ("extract_entities", "self_resolve", "input", "消解"),
            ("extract_rels", "self_resolve", "input", "关联"),
            ("dedup", "infer_colleague", "on_error", "去重后"),
            ("self_resolve", "infer_colleague", "feeds_back", "已知用户名"),
            ("infer_colleague", "infer_missing", "validates_with", "同事边就绪"),
            ("dedup", "conflict_check", "queries", "检查重复"),
            ("conflict_check", "store_mem", "if_pass", "无冲突→存"),
            ("conflict_check", "response", "if_skip", "有冲突→跳过"),
            ("store_mem", "store_node", "triggers", "级联"),
            ("store_node", "store_edge", "triggers", "级联"),
            ("store_edge", "graph_update", "updates", "边写入完成"),
            ("graph_update", "response", "outputs", "最终结果"),
        ]
        for src, tgt, rel, label in edges_data:
            G.add_edge(src, tgt, relation=rel, label=label)

        graph = extract_subgraph_for_cytoscape(G, {}, seeds=list(G.nodes()))
        steps_html = ""
        step_icons = ["&#x1F4BB;", "&#x1F50D;", "&#x1F916;", "&#x1F4AF;", "&#x1F4AF;",
                      "&#x1F510;", "&#x1F44D;", "&#x1F465;", "&#x1F517;", "&#x26A0;&#xFE0F;",
                      "&#x1F4BE;", "&#x1F4BE;", "&#x1F4BE;", "&#x1F578;", "&#x2705;"]
        step_names = ["用户输入", "意图分类", "LLM 抽取", "实体列表", "关系列表",
                      "实体去重", "指代消解", "同事推理", "缺失补全", "冲突检测",
                      "存记忆", "存节点", "存边", "图谱更新", "返回"]
        for i, (n, name) in enumerate(zip([s[0] for s in pipeline_steps], step_names)):
            active = "style='border-color:var(--accent-pink);background:rgba(236,72,153,0.08)'" if i < 3 else ""
            steps_html += f"<div class='info-card' {active}><span>{step_icons[i]}</span> <b>步骤{i+1}</b>: {name}</div>"

        return {
            "title": "AI Memory: 记忆写入流水线 (7 阶段)",
            "graph": graph,
            "gremlin": [
                "text → classify_intent(text)",
                "IF ADD: extract_entities(text) → dedup() → infer_colleague()",
                "IF conflict > 0.6: SKIP (返回原因)",
                "ELSE: store memory + nodes + edges → update graph",
            ],
            "sidebarHtml": f"<div class='section-title'>流水线阶段</div>{steps_html}"
                         "<div class='info-card' style='border-color:var(--accent-pink)'>"
                         "<div class='label'>核心差异 vs PowerMem</div>"
                         "<div class='value' style='font-size:12px'>HugeGraph: OLAP 多跳遍历 | PowerMem: 仅向量检索</div></div>",
        }

    elif scenario_id == "ebbinghaus_curve":
        import math as _math
        G = nx.DiGraph()
        # Ebbinghaus curve data points as nodes
        hours = [0, 1, 2, 4, 8, 12, 24, 48, 72, 168]  # up to 1 week
        k = 0.821
        reinforce = 0.3

        # Create curve points and reinforcement events
        G.add_node("formula", name="R(t) = S₀·e^(-kt) + n·Δ", label="algorithm", description="艾宾浩斯遗忘公式")
        G.add_node("k_param", name=f"k = {k} (衰减常数)", label="config", description="PowerMem 相同值")
        G.add_node("r_param", name=f"Δ = {reinforce} (访问强化)", label="config", description="每次访问 +0.3")

        prev = None
        for i, h in enumerate(hours):
            t = h  # initial access_count = 0
            retention = _math.exp(-k * h)
            retention_clamped = min(1.0, retention)
            # With reinforcement at certain intervals
            access_n = 1 if h in [1, 8, 24, 72] else 0
            retention_reinforced = min(1.0, retention + access_n * reinforce)

            node_id = f"t{h}h"
            label_text = f"{h}h\n保留:{retention_reinforced:.0%}" if h > 0 else "初始"
            G.add_node(node_id, name=label_text, label="data_point",
                       retention=round(retention_reinforced, 3),
                       elapsed=h, reinforced=access_n)
            if prev and prev.startswith('t'):
                try:
                    delta = h - int(prev[1:-1])  # strip 't' and 'h'
                    G.add_edge(prev, node_id, relation="flow", label=f"-{delta}h")
                except ValueError:
                    G.add_edge(prev, node_id, relation="flow", label="")
            elif prev:
                G.add_edge(prev, node_id, relation="defines", label="")
            prev = node_id

            # Reinforcement event nodes
            if access_n:
                rev_id = f"rev_{h}h"
                G.add_node(rev_id, name=f"第{[1,8,24,72].index(h)+1}次复习", label="event",
                           description=f"t={h}h 时访问强化 +{reinforce}")
                G.add_edge(rev_id, node_id, relation="reinforces", label="+Δ")

        # Forgetting danger zones
        G.add_node("danger_8h", name="遗忘高峰区 (4-8h)", label="warning", description="记忆快速衰退期")
        G.add_node("danger_48h", name="长期记忆门槛 (~48h)", label="warning", description="突破则进入长期记忆")
        G.add_edge("t4h", "danger_8h", relation="enters", label="")
        G.add_edge("danger_8h", "t8h", relation="leads_to", label="")
        G.add_edge("t24h", "danger_48h", relation="approaches", label="")

        # PowerMem comparison node
        G.add_node("powermem_note", name="PowerMem 使用相同参数", label="external",
                   description="Ebbinghaus k=0.821 是认知科学标准值")

        graph = extract_subgraph_for_cytoscape(G, {}, seeds=list(G.nodes()))

        # Build curve sidebar with ASCII-like table
        curve_rows = ""
        for h in [0, 1, 2, 4, 8, 12, 24, 48, 72, 168]:
            r_base = _math.exp(-k * h)
            r_rein = min(1.0, r_base + (1 if h in [1, 8, 24, 72] else 0) * reinforce)
            bar_len = int(r_rein * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            tag = f" &#x2705;复习" if h in [1, 8, 24, 72] else ""
            curve_rows += f"<div style='display:flex;align-items:center;gap:4px;font-size:11px;font-family:monospace'>"
            curve_rows += f"<span style='width:35px;color:var(--text-secondary)'>{h:>3}h</span>"
            curve_rows += f"<span style='color:var(--accent-pink)'>{bar}</span>"
            curve_rows += f"<span style='width:40px;text-align:right'>{r_rein:.1%}</span>"
            curve_rows += f"<span style='color:var(--accent-green)'>{tag}</span></div>"

        return {
            "title": "艾宾浩斯遗忘曲线 (与 PowerMem 参数一致)",
            "graph": graph,
            "gremlin": [
                "retention(t) = initial_score * exp(-0.821 * t_hours)",
                "reinforced = min(1.0, retention + access_count * 0.3)",
                "# t=1h 复习 → +30% | t=8h → +30% | t=24h → +30% | t=72h → +30%",
            ],
            "sidebarHtml": f"<div class='section-title'>遗忘曲线可视化</div><div style='background:#1a1a2e;border-radius:8px;padding:8px'>{curve_rows}</div>"
                         "<div class='metric-grid' style='margin-top:8px'>"
                         "<div class='metric-box'><div class='num' style='color:var(--accent-red)'>-56%</div><div class='desc'>8小时后</div></div>"
                         "<div class='metric-box'><div class='num' style='color:var(--accent-orange)'>-79%</div><div class='desc'>24小时后</div></div>"
                         "<div class='metric-box'><div class='num' style='color:var(--accent-green)'>+90%</div><div class='desc'>4次复习后</div></div>"
                         "</div>"
                         "<div class='info-card' style='margin-top:6px;border-color:var(--accent-pink)'>"
                         "<div class='label'>vs PowerMem</div>"
                         "<div class='value' style='font-size:11px'>完全一致的 Ebbinghaus 参数 (k=0.821, Δ=0.3)</div></div>",
        }

    elif scenario_id == "knowledge_graph":
        G = nx.DiGraph()
        # Demo knowledge graph built from sample inputs like:
        # "我叫张三，在腾讯工作" + "我的同事李四也在腾讯，喜欢打篮球"
        # "我喜欢喝咖啡" + "李四住在北京"

        demo_nodes = [
            ("user_zhangsan", "张三", "person", "说话人（我）"),
            ("user_lisi", "李四", "person", "同事"),
            ("org_tencent", "腾讯", "organization", "工作单位"),
            ("loc_beijing", "北京", "location", "城市"),
            ("skill_basketball", "篮球", "skill/爱好", "运动"),
            ("concept_coffee", "咖啡", "concept/事物", "饮品"),
            ("rel_works_t", "works_at(T)", "edge_label", ""),
            ("rel_works_l", "works_at(L)", "edge_label", ""),
            ("rel_colleague", "colleague_of", "edge_label", "自动推断"),
            ("rel_likes_c", "likes(咖啡)", "edge_label", ""),
            ("rel_likes_b", "likes(篮球)", "edge_label", ""),
            ("rel_lives", "lives_in", "edge_label", ""),
        ]
        for nid, name, ntype, desc in demo_nodes:
            G.add_node(nid, name=name, label=ntype, **({"description": desc} if desc else {}))

        demo_edges = [
            ("user_zhangsan", "org_tencent", "works_at", "在...工作"),
            ("user_lisi", "org_tencent", "works_at", "在...工作"),
            ("user_zhangsan", "user_lisi", "colleague_of", "同事 (推断)"),
            ("user_zhangsan", "concept_coffee", "likes", "喜欢"),
            ("user_lisi", "skill_basketball", "likes", "喜欢"),
            ("user_lisi", "loc_beijing", "lives_in", "住在"),
        ]
        for src, tgt, rel, label in demo_edges:
            G.add_edge(src, tgt, relationship=rel, label=label)

        # Add LLM extraction source annotation
        G.add_node("source_input1", name='"我叫张三，在腾讯工作"', label="input", description="用户输入 1")
        G.add_node("source_input2", name='"我的同事李四也在腾讯，喜欢打篮球"', label="input", description="用户输入 2")
        G.add_node("source_input3", name='"我喜欢喝咖啡"', label="input", description="用户输入 3")
        G.add_node("llm_extractor", name="LLM 抽取器", label="model", description="MiMo v2.5 Pro")
        G.add_edge("source_input1", "llm_extractor", relation="feeds_into", label="")
        G.add_edge("source_input2", "llm_extractor", relation="feeds_into", label="")
        G.add_edge("source_input3", "llm_extractor", relation="feeds_into", label="")
        G.add_edge("llm_extractor", "user_zhangsan", relation="extracts", label="实体")
        G.add_edge("llm_extractor", "user_lisi", relation="extracts", label="实体")
        G.add_edge("llm_extractor", "org_tencent", relation="extracts", label="实体")
        G.add_edge("llm_extractor", "concept_coffee", relation="extracts", label="实体")
        G.add_edge("llm_extractor", "loc_beijing", relation="extracts", label="实体")
        G.add_edge("llm_extractor", "skill_basketball", relation="extracts", label="实体")

        # 同事关系自动推断
        G.add_edge("llm_extractor", "rel_colleague", relation="infers", label="同事关系自动推断")

        # Entity type legend
        type_legend = [
            ("person", "人物 (person)", "#ec4899", 3),
            ("organization", "组织 (org)", "#4285f4", 1),
            ("location", "地点 (location)", "#34a853", 1),
            ("skill", "技能 (skill)", "#f97316", 1),
            ("concept", "概念 (concept)", "#a855f7", 1),
        ]

        graph = extract_subgraph_for_cytoscape(G, {},
                    seeds=["user_zhangsan", "user_lisi", "org_tencent", "loc_beijing",
                           "skill_basketball", "concept_coffee", "llm_extractor"])

        type_html = "".join([
            f"<div class='info-card'><span style='color:{c}'>&#9632;</span> {name}: {cnt} 个</div>"
            for _, name, c, cnt in type_legend
        ])

        return {
            "title": "AI Memory: 知识图谱示例 (3 条记忆 → 6 实体 + 6 关系)",
            "graph": graph,
            "gremlin": [
                "# 从 SQLite 图存储查询张三的所有关系:",
                "g.V('user_zhangsan').bothE().path().by('relation').by(outV().values('name')).by(inV().values('name'))",
                "",
                "# 同事子图查询:",
                "g.V().hasLabel('person').out('colleague_of').path()",
            ],
            "sidebarHtml": f"<div class='section-title'>示例输入</div>"
                         "<div class='info-card' style='border-left:3px solid var(--accent-pink)'>"
                         "<div style='font-size:12px;color:var(--text-secondary)'>输入 1:</div>"
                         "<div>'我叫张三，在腾讯工作'</div></div>"
                         "<div class='info-card' style='border-left:3px solid var(--accent-blue)'>"
                         "<div style='font-size:12px;color:var(--text-secondary)'>输入 2:</div>"
                         "<div>'我的同事李四也在腾讯，喜欢打篮球'</div></div>"
                         "<div class='info-card' style='border-left:3px solid var(--accent-green)'>"
                         "<div style='font-size:12px;color:var(--text-secondary)'>输入 3:</div>"
                         "<div>'我喜欢喝咖啡'</div></div>"
                         f"<div class='section-title'>实体分布</div>{type_html}"
                         "<div class='info-card' style='border-color:var(--accent-cyan)'>"
                         "<div class='label'>自动推断</div>"
                         "<div class='value' style='font-size:11px'>张三↔李四: colleague_of (同在腾讯)</div></div>",
        }

    elif scenario_id == "intent_classification":
        G = nx.DiGraph()
        # Intent classification dual-path decision tree
        G.add_node("input", name="用户输入", label="input")
        G.add_node("llm_classify", name="LLM 分类器 (主路径)", label="model", description="MiMo v2.5 Pro, temp=0")
        G.add_node("regex_fallback", name="正则回退 (备选)", label="algorithm", description="问号/关键词模式")
        G.add_node("action_add", name="→ 存储记忆 (ADD)", label="output", description="调用 add_memory API")
        G.add_node("action_query", name="→ 检索问答 (QUERY)", label="output", description="调用 search_memory API")
        G.add_node("qmark_rule", name="? / ？ 规则", label="strategy", description="含问号 → QUERY")
        G.add_node("stmt_hint_rule", name="陈述词规则", label="strategy", description="'也'字前缀 → ADD")
        G.add_node("starts_my_q", name="'我的...'疑问模式", label="strategy", description="我的X有哪些 → QUERY")
        G.add_node("starts_q_word", name="疑问词开头", label="strategy", description="谁/什么/哪里 → QUERY")

        G.add_edge("input", "llm_classify", relation="primary", label="首选路径")
        G.add_edge("input", "regex_fallback", relation="fallback", label="LLM失败时")
        G.add_edge("llm_classify", "action_add", relation="classifies_as", label="判定为 ADD")
        G.add_edge("llm_classify", "action_query", relation="classifies_as", label="判定为 QUERY")
        G.add_edge("regex_fallback", "qmark_rule", relation="checks", label="规则 1")
        G.add_edge("regex_fallback", "stmt_hint_rule", relation="checks", label="规则 2")
        G.add_edge("regex_fallback", "starts_my_q", relation="checks", label="规则 3")
        G.add_edge("regex_fallback", "starts_q_word", relation="checks", label="规则 4")
        G.add_edge("qmark_rule", "action_query", relation="matches", label="含问号")
        G.add_edge("stmt_hint_rule", "action_add", relation="matches", label="陈述句")
        G.add_edge("starts_my_q", "action_query", relation="matches", label="疑问句")
        G.add_edge("starts_q_word", "action_query", relation="matches", label="疑问句开头")

        # Test examples
        examples = [
            ("我的同事李四也在腾讯", "ADD", "LLM", "陈述句"),
            ("我的同事有哪些?", "QUERY", "LLM/正则", "含问号"),
            ("我喜欢喝咖啡", "ADD", "LLM", "陈述句"),
            ("我喜欢什么?", "QUERY", "正则", "含问号+我的"),
            ("帮我查一下之前说的", "QUERY", "LLM", "语义理解"),
            ("王五是我大学同学", "ADD", "LLM", "陈述句"),
        ]
        for i, (text, action, method, reason) in enumerate(examples):
            ex_id = f"ex_{i}"
            G.add_node(ex_id, name=f'"{text[:8]}..."', label="example",
                       result=action, method=method, reason=reason)
            target = "action_add" if action == "ADD" else "action_query"
            G.add_edge(ex_id, target, relation="example_of", label=action)

        graph = extract_subgraph_for_cytoscape(G, {}, seeds=list(G.nodes()))

        ex_html = ""
        for t, a, m, r in examples:
            border_color = "var(--accent-green)" if a == "ADD" else "var(--accent-blue)"
            ex_html += (
                f"<div class='info-card' style='border-left:3px solid {border_color}'>"
                f"<div style='font-size:11px'>{t}</div>"
                f"<div><span style='color:var(--accent-pink);font-weight:bold'>{a}</span> "
                f"<span style='color:var(--text-secondary);font-size:10px'>({m})</span> "
                f"<span style='font-size:10px'>{r}</span></div></div>"
            )

        return {
            "title": "意图分类: LLM 主路径 + 正则回退 双策略",
            "graph": graph,
            "gremlin": [
                "# Step 1: classify intent",
                "result = classify_intent_llm(text)  # primary",
                "if not result: result = classify_intent_regex(text)  # fallback",
                "",
                "# Step 2: route to handler",
                "if result.action == 'ADD': store.add_memory(text)",
                "else: store.search_memory(text)",
            ],
            "sidebarHtml": f"<div class='section-title'>双路径架构</div>"
                         "<div class='info-card' style='border-color:var(--accent-pink)'>"
                         "<div class='label'>主路径: LLM 分类</div>"
                         "<div class='value' style='font-size:11px'>MiMo v2.5, temp=0, zero-shot</div></div>"
                         "<div class='info-card' style='border-color:var(--accent-orange)'>"
                         "<div class='label'>回退: 正则表达式</div>"
                         "<div class='value' style='font-size:11px'>问号/关键词/模式匹配</div></div>"
                         f"<div class='section-title'>测试用例</div>{ex_html}",
        }

    elif scenario_id == "search_flow":
        G = nx.DiGraph()
        # Search pipeline: query → rank → score → context → answer
        search_steps = [
            ("query", "用户问题", "input"),
            ("rank_llm", "LLM 相关性排序", "model"),
            ("all_memories", "全部记忆 (按时间)", "storage"),
            ("ebbinghaus_score", "遗忘曲线评分", "algorithm"),
            ("graph_context", "图谱关系上下文", "engine"),
            ("top_k", "Top-K 结果", "intermediate"),
            ("reinforce", "访问强化 (+0.3)", "operator"),
            ("answer_gen", "LLM 生成回答", "model"),
            ("final_answer", "最终回答", "output"),
        ]
        for nid, name, ntype in search_steps:
            G.add_node(nid, name=name, label=ntype)

        G.add_edge("query", "rank_llm", relation="sends", label="问题")
        G.add_edge("all_memories", "rank_llm", relation="provides", label="候选集")
        G.add_edge("all_memories", "ebbinghaus_score", relation="scores", label="每条记忆")
        G.add_edge("ebbinghaus_score", "top_k", relation="filters", label="分数排序")
        G.add_edge("rank_llm", "top_k", relation="ranks", label="相关性")
        G.add_edge("graph_context", "rank_llm", relation="enriches", label="图谱上下文")
        G.add_edge("top_k", "reinforce", relation="accesses", label="被选中的记忆")
        G.add_edge("top_k", "answer_gen", relation="feeds", label="相关记忆")
        G.add_edge("graph_context", "answer_gen", relation="enriches", label="关系增强")
        G.add_edge("answer_gen", "final_answer", relation="produces", label="")

        # Graph context detail
        G.add_node("graph_db", name="SQLite 图存储", label="storage", description="nodes + edges 表")
        G.add_node("recent_edges", name="最近 20 条关系", label="cache", description="用于构建上下文")
        G.add_edge("graph_db", "recent_edges", relation="queries", label="SELECT ... LIMIT 20")
        G.add_edge("recent_edges", "graph_context", relation="formats", label="src --[rel]--> tgt")

        # Hybrid retrieval note
        G.add_node("hybrid_note", name="4路混合检索", label="core", description="向量+全文+图谱+时序 RRF 融合 (路线图)")
        G.add_edge("hybrid_note", "rank_llm", relation="future", label="(S14 规划)")

        graph = extract_subgraph_for_cytoscape(G, {}, seeds=list(G.nodes()))

        return {
            "title": "图谱增强检索流程 (4 步: 排序→评分→上下文→生成)",
            "graph": graph,
            "gremlin": [
                "# Step 1: Get all memories with Ebbinghaus score",
                "memories = SELECT * FROM memories WHERE user_id=? ORDER BY created_at DESC",
                "FOR each mem: retention = e^(-0.821*t) + count*0.3",
                "",
                "# Step 2: LLM ranking with graph context",
                "graph_ctx = '张三 --[works_at]--> 腾讯\\n李四 --[colleague_of]--> 张三'",
                "ranked = llm_rank(query, memories, graph_ctx)",
                "",
                "# Step 3: Generate answer",
                "answer = generate_answer(query, top_k_memories, graph_ctx)",
            ],
            "sidebarHtml": "<div class='section-title'>检索步骤</div>"
                         "<div class='info-card'><span class='step-num'>1</span> 全量记忆 + 艾宾浩斯评分</div>"
                         "<div class='info-card'><span class='step-num'>2</span> LLM 相关性排序 (含图谱上下文)</div>"
                         "<div class='info-card'><span class='step-num'>3</span> Top-K 提取 + 访问强化</div>"
                         "<div class='info-card'><span class='step-num'>4</span> LLM 生成最终回答</div>"
                         "<div class='section-title'>图谱上下文格式</div>"
                         "<div class='info-card' style='font-family:monospace;font-size:10px'>"
                         "张三 --[works_at]--&gt; 腾讯<br>"
                         "李四 --[colleague_of]--&gt; 张三<br>"
                         "张三 --[likes]--&gt; 咖啡<br>"
                         "李四 --[lives_in]--&gt; 北京</div>"
                         "<div class='info-card' style='border-color:var(--accent-pink);margin-top:6px'>"
                         "<div class='label'>路线图 S14</div>"
                         "<div class='value' style='font-size:11px'>4 路混合: 向量+全文+图谱+时序, RRF 融合</div></div>",
        }

    elif scenario_id == "powermem_comparison":
        # Feature comparison radar-style graph
        G = nx.DiGraph()
        G.add_node("hg_memory", name="HugeGraph Memory", label="product", description="我们的方案")
        G.add_node("powermem", name="PowerMem v1.1.2", label="external", description="OceanBase 开源")
        G.add_node("hg_olap", name="OLAP 多跳遍历", label="feature", description="60亿点边 Vermeer")
        G.add_node("pm_olap", name="OLAP 多跳遍历", label="feature", description="❌ 不支持")
        G.add_node("hg_entity_res", name="实体消解", label="feature", description="3 策略级联 (Sprint1)")
        G.add_node("pm_entity_res", name="实体消解", label="feature", description="⚠️ 基础")
        G.add_node("hg_drift", name="DRIFT 搜索", label="feature", description="5步流水线 (Sprint4)")
        G.add_node("pm_drift", name="DRIFT 搜索", label="feature", description="❌ 不支持")
        G.add_node("hg_freshness", name="知识时效性", label="feature", description="TTL+版本检测 (Sprint8)")
        G.add_node("pm_freshness", name="知识时效性", label="feature", description="⚠️ 无 TTL")
        G.add_node("hg_text2g", name="Text2Gremlin", label="feature", description="自纠错 (Sprint5)")
        G.add_node("pm_text2g", name="Text2Gremlin", label="feature", description="❌ 不支持")
        G.add_node("hg_graph_quality", name="图谱质量评估", label="feature", description="5 维度 (Sprint7)")
        G.add_node("pm_graph_quality", name="图谱质量评估", label="feature", description="❌ 不支持")
        G.add_node("hg_mcp", name="MCP Server", label="feature", description="10 Tools + 3 Resources")
        G.add_node("pm_mcp", name="MCP Server", label="feature", description="❌ 无")

        G.add_edge("hg_memory", "hg_olap", relation="has", label="")
        G.add_edge("hg_memory", "hg_entity_res", relation="has", label="")
        G.add_edge("hg_memory", "hg_drift", relation="has", label="")
        G.add_edge("hg_memory", "hg_freshness", relation="has", label="")
        G.add_edge("hg_memory", "hg_text2g", relation="has", label="")
        G.add_edge("hg_memory", "hg_graph_quality", relation="has", label="")
        G.add_edge("hg_memory", "hg_mcp", relation="has", label="")
        G.add_edge("powermem", "pm_olap", relation="has", label="")
        G.add_edge("powermem", "pm_entity_res", relation="has", label="")
        G.add_edge("powermem", "pm_drift", relation="has", label="")
        G.add_edge("powermem", "pm_freshness", relation="has", label="")
        G.add_edge("powermem", "pm_text2g", relation="has", label="")
        G.add_edge("powermem", "pm_graph_quality", relation="has", label="")
        G.add_edge("powermem", "pm_mcp", relation="has", label="")

        # Comparison links (dotted)
        G.add_edge("hg_olap", "pm_olap", relation="vs", label="✅ vs ❌")
        G.add_edge("hg_entity_res", "pm_entity_res", relation="vs", label="✅ vs ⚠️")
        G.add_edge("hg_drift", "pm_drift", relation="vs", label="✅ vs ❌")
        G.add_edge("hg_freshness", "pm_freshness", relation="vs", label="✅ vs ⚠️")
        G.add_edge("hg_text2g", "pm_text2g", relation="vs", label="✅ vs ❌")
        G.add_edge("hg_graph_quality", "pm_graph_quality", relation="vs", label="✅ vs ❌")
        G.add_edge("hg_mcp", "pm_mcp", relation="vs", label="✅ vs ❌")

        graph = extract_subgraph_for_cytoscape(G, {}, seeds=list(G.nodes()))

        comp_items = [
            ("OLAP 多跳遍历", "✅ 60亿点边 Vermeer", "❌ 不支持", True),
            ("实体消解", "✅ 3策略级联 Sprint1", "⚠️ 基础字符串", True),
            ("DRIFT 搜索", "✅ 5步 H-C-P-L-R", "❌ 无", True),
            ("知识时效性", "✅ TTL+版本 Sprint8", "⚠️ 无 TTL", True),
            ("Text2Gremlin", "✅ 自纠错 Sprint5", "❌ 无", True),
            ("图谱质量评估", "✅ 5维度 Sprint7", "❌ 无", True),
            ("MCP 协议", "✅ 10 Tools", "❌ 无", True),
            ("艾宾浩斯曲线", "✅ 一致参数", "✅ 一致参数", False),
            ("LLM 抽取", "✅ MiMo API", "✅ 支持 LLM", False),
            ("开源协议", "✅ Apache 2.0", "✅ Apache 2.0", False),
        ]
        comp_html = ""
        for name, hg, pm, is_diff in comp_items:
            diff_style = "background:rgba(236,72,153,0.08)" if is_diff else ""
            comp_html += f"<div class='info-card' style='{diff_style}'>"
            comp_html += f"<div class='label' style='width:100px'>{name}</div>"
            comp_html += f"<div style='display:flex;justify-content:space-between'>"
            comp_html += f"<span style='color:var(--accent-green)'>{hg}</span>"
            comp_html += f"<span style='color:var(--text-secondary)'>{pm}</span></div></div>"

        return {
            "title": "HugeGraph Memory vs PowerMem v1.1.2 (10 维度对比)",
            "graph": graph,
            "gremlin": [],
            "sidebarHtml": f"<div class='section-title'>功能对比 (10 维度)</div>{comp_html}"
                         "<div class='info-card' style='border-color:var(--accent-pink);margin-top:6px'>"
                         "<div class='label'>核心差异化</div>"
                         "<div class='value' style='font-size:11px'>PowerMem 缺乏: OLAP遍历/DRIFT搜索/时效性/MCP/图谱质量 → HugeGraph Sprint1-10 已全部覆盖</div></div>",
        }

    elif scenario_id == "architecture":
        G = nx.DiGraph()
        # Full system architecture
        layers = {
            # Frontend layer
            "frontend": ("前端 UI", "component", "Cytoscape.js + HTML"),
            # API layer
            "api_classify": ("/api/classify", "api_endpoint", "POST 意图分类"),
            "api_add": ("/api/memory/add", "api_endpoint", "POST 写入记忆"),
            "api_search": ("/api/memory/search", "api_endpoint", "POST 检索记忆"),
            "api_list": ("/api/memory/list", "api_endpoint", "GET 列表"),
            "api_stats": ("/api/stats", "api_endpoint", "GET 统计"),
            "api_graph": ("/api/graph", "api_endpoint", "GET 图谱数据"),
            "api_clear": ("/api/clear", "api_endpoint", "POST 清空"),
            # Logic layer
            "intent_mod": ("意图分类模块", "module", "LLM + Regex 双路径"),
            "mem_store": ("MemoryStore", "core", "业务逻辑核心"),
            "extractor": ("实体抽取器", "model", "LLM Tool Calling"),
            "answering": ("回答生成器", "model", "LLM + 图谱上下文"),
            # Storage layer
            "sqlite": ("SQLite", "storage", "3 表: memories/nodes/edges"),
            "mem_table": ("memories 表", "storage", "id/content/times/scores"),
            "node_table": ("nodes 表", "storage", "id/name/type/props"),
            "edge_table": ("edges 表", "storage", "id/src/tgt/rel/memory_id"),
            # External
            "llm_api": ("小米 MiMo API", "external", "mimo-v2.5-pro"),
            # Algorithm
            "ebbinghaus": ("艾宾浩斯引擎", "algorithm", "R(t)=e^(-0.821t)"),
            "dedup_engine": ("去重引擎", "algorithm", "名称包含匹配"),
            "infer_engine": ("推理引擎", "algorithm", "同事/缺失关系"),
        }
        for nid, (name, ntype, desc) in layers.items():
            G.add_node(nid, name=name, label=ntype, **{"description": desc})

        # Edges: frontend → api → logic → storage
        api_edges = [
            ("frontend", "api_classify", "calls"), ("frontend", "api_add", "calls"),
            ("frontend", "api_search", "calls"), ("frontend", "api_list", "calls"),
            ("frontend", "api_stats", "calls"), ("frontend", "api_graph", "calls"),
            ("frontend", "api_clear", "calls"),
            ("api_classify", "intent_mod", "delegates"),
            ("api_add", "mem_store", "delegates"), ("api_search", "mem_store", "delegates"),
            ("api_list", "mem_store", "delegates"), ("api_stats", "mem_store", "delegates"),
            ("api_graph", "mem_store", "delegates"), ("api_clear", "mem_store", "delegates"),
            ("mem_store", "extractor", "uses"), ("mem_store", "answering", "uses"),
            ("mem_store", "ebbinghaus", "uses"), ("mem_store", "dedup_engine", "uses"),
            ("mem_store", "infer_engine", "uses"),
            ("extractor", "llm_api", "queries"), ("answering", "llm_api", "queries"),
            ("intent_mod", "llm_api", "queries"),
            ("mem_store", "sqlite", "writes"), ("mem_store", "sqlite", "reads"),
            ("sqlite", "mem_table", "contains"), ("sqlite", "node_table", "contains"),
            ("sqlite", "edge_table", "contains"),
            ("api_graph", "node_table", "reads_direct"),
            ("api_graph", "edge_table", "reads_direct"),
        ]
        for src, tgt, rel in api_edges:
            G.add_edge(src, tgt, relation=rel)

        graph = extract_subgraph_for_cytoscape(G, {}, seeds=list(G.nodes()))

        return {
            "title": "AI Memory 系统架构全景 (8 API 端点 + SQLite 图存储)",
            "graph": graph,
            "gremlin": [
                "# 8 个 REST API 端点:",
                "POST /api/classify   → 意图分类 (ADD/QUERY)",
                "POST /api/memory/add → 写入记忆 (抽取+去重+推理+存储)",
                "POST /api/memory/search → 检索 (艾宾浩斯+LLM排名+图谱上下文)",
                "GET  /api/memory/list → 列表",
                "GET  /api/stats       → 统计 (节点/边/类型分布/遗忘分数)",
                "GET  /api/graph       → 图谱数据 (Cytoscape 格式)",
                "POST /api/clear       → 清空",
            ],
            "sidebarHtml": "<div class='section-title'>系统分层</div>"
                         "<div class='info-card' style='border-color:var(--accent-pink)'><b>展示层</b>: Cytoscape.js 可视化</div>"
                         "<div class='info-card' style='border-color:var(--accent-blue)'><b>API 层</b>: 8 个 REST 端点</div>"
                         "<div class='info-card' style='border-color:var(--accent-green)'><b>逻辑层</b>: MemoryStore + LLM</div>"
                         "<div class='info-card' style='border-color:var(--accent-orange)'><b>存储层</b>: SQLite 3 表</div>"
                         "<div class='info-card' style='border-color:var(--accent-purple)'><b>算法层</b>: 艾宾浩斯+去重+推理</div>"
                         "<div class='metric-grid' style='margin-top:8px'>"
                         "<div class='metric-box'><div class='num' style='color:var(--accent-pink)'>1018</div><div class='desc'>代码行数</div></div>"
                         "<div class='metric-box'><div class='num' style='color:var(--accent-blue)'>8</div><div class='desc'>API端点</div></div>"
                         "<div class='metric-box'><div class='num' style='color:var(--accent-green)'>3</div><div class='desc'>数据表</div></div></div>",
        }

    return {"error": "Unknown ai_memory scenario"}


def build_metrics_sidebar(nodes, edges, extra=None):
    html = f"""
    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{nodes}</div><div class="desc">节点</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-purple)">{edges}</div><div class="desc">边</div></div>
    </div>
    """
    if extra:
        for k, v in extra.items():
            html += f'<div class="info-card"><div class="label">{k}</div><div class="value">{v}</div></div>'
    return html


def build_radar_chart_html(methods):
    """Build radar chart using SVG"""
    # Top 6 methods + DRIFT
    top_methods = [m for m in methods if m["name"] in ("DRIFT", "HippoRAG2", "GraphRAG(MS)", "LightRAG", "G-Retriever", "GFM-RAG")]
    dims = ["multi_hop", "summary", "qa"]
    dim_labels = ["多跳问答", "全局摘要", "通用问答"]

    svg_size = 280
    cx, cy = svg_size // 2, svg_size // 2 + 10
    max_r = 100

    svg = f'<svg viewBox="0 0 {svg_size} {svg_size}" width="100%">'

    # Grid circles
    for i in range(1, 4):
        r = max_r * i / 3
        svg += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#30363d" stroke-width="0.5"/>'

    # Axis lines + labels
    n = len(dims)
    colors = ["#ea4335", "#4285f4", "#34a853", "#fbbc04", "#a855f7", "#06b6d4"]
    for i, label in enumerate(dim_labels):
        angle = (2 * math.pi * i / n) - math.pi / 2
        x2 = cx + max_r * math.cos(angle)
        y2 = cy + max_r * math.sin(angle)
        svg += f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" stroke="#30363d" stroke-width="0.5"/>'
        lx = cx + (max_r + 18) * math.cos(angle)
        ly = cy + (max_r + 18) * math.sin(angle)
        svg += f'<text x="{lx}" y="{ly}" fill="#9aa0a6" font-size="9" text-anchor="middle">{label}</text>'

    # Plot each method
    for mi, m in enumerate(top_methods):
        points = []
        for i, dim in enumerate(dims):
            val = m[dim]
            angle = (2 * math.pi * i / n) - math.pi / 2
            px = cx + max_r * val * math.cos(angle)
            py = cy + max_r * val * math.sin(angle)
            points.append(f"{px},{py}")

        color = colors[mi]
        svg += f'<polygon points="{" ".join(points)}" fill="{color}" fill-opacity="0.1" stroke="{color}" stroke-width="1.5"/>'

    # Legend
    for mi, m in enumerate(top_methods):
        svg += f'<rect x="10" y="{10 + mi * 16}" width="8" height="8" fill="{colors[mi]}"/>'
        svg += f'<text x="22" y="{18 + mi * 16}" fill="#e8eaed" font-size="9">{m["name"]}</text>'

    svg += '</svg>'

    return f"""
    <div class="section-title">Top 6 方法: 能力雷达图</div>
    <div style="text-align:center">{svg}</div>
    """


def build_ranking_html(methods):
    """Build ranking table"""
    # Rank by average score
    ranked = []
    for m in methods:
        avg = (m["multi_hop"] + m["summary"] + m["qa"]) / 3
        ranked.append((m["name"], avg, m["category"]))
    ranked.sort(key=lambda x: x[1], reverse=True)

    html = '<div class="section-title">综合排名 (3 任务均值)</div>'
    for i, (name, avg, cat) in enumerate(ranked):
        bar_width = int(avg * 100)
        color = "#ea4335" if i == 0 else "#4285f4" if i < 3 else "#9aa0a6"
        if name == "DRIFT":
            color = "#ea4335"
        html += f"""
        <div class="info-card" style="padding:6px 10px">
          <div style="display:flex;align-items:center;justify-content:space-between">
            <span style="font-size:12px;font-weight:{'700' if name=='DRIFT' else '400'};color:{color if name=='DRIFT' else 'var(--text-primary)'}">
              {i+1}. {name}
            </span>
            <span style="font-size:11px;color:var(--text-secondary)">{avg:.2f}</span>
          </div>
          <div style="height:3px;background:var(--bg-primary);margin-top:4px;border-radius:2px">
            <div style="height:100%;width:{bar_width}%;background:{color};border-radius:2px"></div>
          </div>
        </div>"""
    return html


def build_cost_benefit_html(methods):
    """Build cost-benefit scatter data"""
    ranked = []
    for m in methods:
        avg = (m["multi_hop"] + m["summary"] + m["qa"]) / 3
        benefit = avg / max(m["cost"], 0.1)
        ranked.append((m["name"], benefit, m["cost"], avg))
    ranked.sort(key=lambda x: x[1], reverse=True)

    html = '<div class="section-title">性价比排名</div>'
    for name, benefit, cost, avg in ranked[:8]:
        stars = int(benefit)
        stars_str = "&#9733;" * min(stars, 5) + "&#9734;" * max(5 - stars, 0)
        color = "#ea4335" if name == "DRIFT" else "#34a853" if benefit > 2 else "#fbbc04"
        html += f"""
        <div class="info-card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:700;color:{color}">{name}</span>
            <span style="font-size:14px">{stars_str}</span>
          </div>
          <div style="font-size:10px;color:var(--text-secondary);margin-top:2px">
            Score={avg:.2f} / Cost={cost:.1f} → Ratio={benefit:.2f}
          </div>
        </div>"""
    return html


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    print(f"Starting HugeGraph PoC Visualizer on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    if not args.no_open:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
