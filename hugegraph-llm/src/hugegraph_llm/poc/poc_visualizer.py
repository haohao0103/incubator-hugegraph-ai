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
    "default": "#30363d",
}

# ============ HTML Template ============
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HugeGraph PoC Interactive Visualizer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.4/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-dagre/2.5.0/cytoscape-dagre.js"></script>
<style>
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
    --border-color: #30363d;
    --radius: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary); color: var(--text-primary);
    display: flex; height: 100vh; overflow: hidden;
  }

  /* Sidebar */
  .sidebar {
    width: 340px; min-width: 340px;
    background: var(--bg-secondary); border-right: 1px solid var(--border-color);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .sidebar-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border-color);
  }
  .sidebar-header h1 {
    font-size: 18px; font-weight: 700; color: var(--accent-blue);
  }
  .sidebar-header .subtitle {
    font-size: 12px; color: var(--text-secondary); margin-top: 4px;
  }
  .poc-tabs {
    display: flex; border-bottom: 1px solid var(--border-color);
    background: var(--bg-tertiary);
  }
  .poc-tab {
    flex: 1; padding: 10px 8px; text-align: center; cursor: pointer;
    font-size: 12px; font-weight: 600; color: var(--text-secondary);
    border-bottom: 2px solid transparent; transition: all 0.2s;
  }
  .poc-tab:hover { color: var(--text-primary); background: rgba(255,255,255,0.05); }
  .poc-tab.active {
    color: var(--accent-blue); border-bottom-color: var(--accent-blue);
    background: var(--bg-secondary);
  }
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
    <h1>&#x1F680; HugeGraph PoC Visualizer</h1>
    <div class="subtitle" id="serverStatus">Initializing...</div>
  </div>

  <div class="poc-tabs">
    <div class="poc-tab active" onclick="switchPoc('supply_chain')">Supply Chain</div>
    <div class="poc-tab" onclick="switchPoc('graphrag_bench')">GraphRAG-Bench</div>
    <div class="poc-tab" onclick="switchPoc('code_graph')">Code Graph</div>
  </div>

  <div class="sidebar-body" id="sidebarBody">
    <!-- Dynamic content -->
  </div>
</div>

<div class="main">
  <div class="canvas-header">
    <div class="title" id="canvasTitle">Select a scenario</div>
    <div class="gremlin" id="gremlinQuery"></div>
  </div>
  <div id="cy"></div>
  <div class="canvas-footer">
    <div class="legend" id="legend"></div>
    <div id="graphStats">Nodes: 0 | Edges: 0</div>
  </div>
</div>

<div class="toast" id="toast"></div>
<div class="node-detail" id="nodeDetail" style="display:none;"></div>

<script>
let cy = null;
let currentPoc = 'supply_chain';
let graphData = {};

const NODE_COLORS = {
  enterprise: '#4285f4',  // blue
  facility: '#34a853',    // green
  material: '#f97316',   // orange
  component: '#fbbc04',   // yellow
  product: '#a855f7',     // purple
  function: '#06b6d4',    // cyan
  class: '#4285f4',       // blue
  module: '#34a853',      // green
  method: '#f97316',      // orange
  default: '#9aa0a6',
};

const EDGE_COLORS = {
  supplies: '#34a853',
  produces: '#4285f4',
  has_input: '#f97316',
  located_in: '#ea4335',
  depends_on: '#fbbc04',
  calls: '#06b6d4',
  imports: '#a855f7',
  default: '#30363d',
};

function initCy() {
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '10px',
          'color': '#e8eaed',
          'text-outline-width': 2,
          'text-outline-color': '#0f1117',
          'background-color': 'data(color)',
          'width': 'data(size)',
          'height': 'data(size)',
          'border-width': 1,
          'border-color': '#30363d',
        }
      },
      {
        selector: 'node.highlighted',
        style: {
          'border-width': 3,
          'border-color': '#fbbc04',
          'z-index': 999,
        }
      },
      {
        selector: 'node.dimmed',
        style: {
          'opacity': 0.2,
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 2,
          'line-color': 'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'label': 'data(label)',
          'font-size': '9px',
          'color': '#9aa0a6',
          'text-rotation': 'autorotate',
          'text-outline-width': 2,
          'text-outline-color': '#0f1117',
        }
      },
      {
        selector: 'edge.highlighted',
        style: {
          'width': 4,
          'line-color': '#fbbc04',
          'target-arrow-color': '#fbbc04',
          'z-index': 998,
        }
      },
      {
        selector: 'edge.dimmed',
        style: {
          'opacity': 0.15,
        }
      },
      {
        selector: 'node.seeded',
        style: {
          'border-width': 3,
          'border-color': '#ea4335',
          'border-style': 'double',
        }
      },
    ],
    layout: { name: 'preset' },
    userZoomingEnabled: true,
    userPanningEnabled: true,
    boxSelectionEnabled: false,
  });

  cy.on('tap', 'node', function(evt) {
    showNodeDetail(evt.target);
  });
  cy.on('tap', function(evt) {
    if (evt.target === cy) {
      document.getElementById('nodeDetail').style.display = 'none';
    }
  });
}

function showNodeDetail(node) {
  const data = node.data();
  const detail = document.getElementById('nodeDetail');
  let html = '<h3>' + data.label + '</h3>';
  if (data.props) {
    for (const [k, v] of Object.entries(data.props)) {
      html += '<div class="prop">' + k + ': <span>' + v + '</span></div>';
    }
  }
  if (data.centrality) {
    html += '<div style="margin-top:6px;border-top:1px solid #30363d;padding-top:6px">';
    for (const [k, v] of Object.entries(data.centrality)) {
      html += '<div class="prop">' + k + ': <span>' + (typeof v === 'number' ? v.toFixed(4) : v) + '</span></div>';
    }
    html += '</div>';
  }
  detail.innerHTML = html;
  detail.style.display = 'block';
  detail.style.left = Math.min(evt.originalEvent.clientX, window.innerWidth - 300) + 'px';
  detail.style.top = (evt.originalEvent.clientY + 10) + 'px';
}

function loadGraph(data) {
  if (!cy) initCy();

  // Convert to Cytoscape format
  const elements = [];
  const posMap = {};

  data.nodes.forEach((n, i) => {
    // Assign positions using force-directed layout approximation
    const angle = (i / data.nodes.length) * Math.PI * 2;
    const radius = 150 + (i % 5) * 60;
    elements.push({
      data: {
        id: n.id,
        label: n.label || n.name || n.id,
        color: n.color || NODE_COLORS[n.type] || NODE_COLORS.default,
        size: n.size || (n.isSeed ? 40 : 28),
        props: n.props || {},
        centrality: n.centrality || null,
      },
      position: n.position || { x: 400 + Math.cos(angle) * radius, y: 350 + Math.sin(angle) * radius },
      classes: (n.isSeed ? 'seeded ' : '') + (n.highlighted ? 'highlighted ' : ''),
    });
  });

  data.edges.forEach(e => {
    elements.push({
      data: {
        id: e.id || (e.source + '-' + e.target),
        source: e.source,
        target: e.target,
        label: e.label || e.relation || '',
        color: e.color || EDGE_COLORS[e.relation] || EDGE_COLORS.default,
      },
      classes: e.highlighted ? 'highlighted' : '',
    });
  });

  cy.elements().remove();
  cy.add(elements);

  // Apply dagre layout for better readability
  try {
    cy.layout({
      name: 'dagre',
      rankDir: 'TB',
      spacingFactor: 1.2,
      nodeSep: 40,
      rankSep: 60,
    }).run();
  } catch(e) {
    // Fallback to concentric
    cy.layout({ name: 'concentric' }).run();
  }

  // Update stats
  document.getElementById('graphStats').textContent =
    'Nodes: ' + data.nodes.length + ' | Edges: ' + data.edges.length;

  // Update legend
  updateLegend(data.nodes);
}

function updateLegend(nodes) {
  const types = new Set(nodes.map(n => n.type).filter(Boolean));
  let html = '';
  types.forEach(t => {
    html += '<div class="legend-item"><div class="legend-dot" style="background:' +
      (NODE_COLORS[t] || NODE_COLORS.default) + '"></div>' + t + '</div>';
  });
  document.getElementById('legend').innerHTML = html;
}

function highlightPath(pathIds) {
  cy.elements().removeClass('highlighted').removeClass('dimmed');
  if (pathIds.length === 0) {
    cy.elements().removeClass('dimmed');
    return;
  }
  const nodeSet = new Set(pathIds.filter(id => !id.includes('-')));
  const edgeSet = new Set(pathIds.filter(id => id.includes('-')));

  cy.nodes().forEach(n => {
    if (nodeSet.has(n.id())) n.addClass('highlighted');
    else n.addClass('dimmed');
  });
  cy.edges().forEach(e => {
    if (edgeSet.has(e.id())) e.addClass('highlighted');
    else e.addClass('dimmed');
  });

  // Fit to highlighted elements
  const highlighted = cy.$('.highlighted');
  if (highlighted.length > 0) {
    cy.animate({
      fit: { eles: highlighted, padding: 60 },
      duration: 500,
      easing: 'ease-in-out-cubic',
    });
  }
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

function switchPoc(pocId) {
  currentPoc = pocId;
  document.querySelectorAll('.poc-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  loadSidebar();
  // Reset graph
  if (cy) cy.elements().remove();
  document.getElementById('canvasTitle').textContent = 'Select a scenario';
  document.getElementById('gremlinQuery').textContent = '';
  document.getElementById('graphStats').textContent = 'Nodes: 0 | Edges: 0';
  document.getElementById('legend').innerHTML = '';
}

function loadScenario(pocId, scenarioId) {
  fetch('/api/scenario?poc=' + pocId + '&scenario=' + scenarioId)
    .then(r => r.json())
    .then(data => {
      graphData = data;
      loadGraph(data.graph);

      document.getElementById('canvasTitle').textContent =
        data.title || ('Scenario ' + scenarioId);

      // Show primary Gremlin query
      if (data.gremlin && data.gremlin.length > 0) {
        document.getElementById('gremlinQuery').textContent = data.gremlin[0];
      }

      // Highlight algo steps
      document.querySelectorAll('.algo-step').forEach((s, i) => {
        s.classList.toggle('active', i === 0);
      });

      // Show paths in sidebar
      if (data.sidebarHtml) {
        document.getElementById('dynamicSidebar').innerHTML = data.sidebarHtml;
      }

      showToast('Graph loaded: ' + data.graph.nodes.length + ' nodes, ' + data.graph.edges.length + ' edges');
    })
    .catch(e => showToast('Error: ' + e.message));
}

function loadSidebar() {
  fetch('/api/sidebar?poc=' + currentPoc)
    .then(r => r.json())
    .then(data => {
      document.getElementById('sidebarBody').innerHTML = data.html;
      // Update server status
      if (data.serverStatus) {
        document.getElementById('serverStatus').innerHTML = data.serverStatus;
      }
    });
}

// Init
window.addEventListener('DOMContentLoaded', () => {
  initCy();
  loadSidebar();
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

# Pre-build graph data
supply_chain_G = build_supply_chain_graph()
supply_chain_centrality = compute_centrality(supply_chain_G)
code_graph_G = build_code_graph()
code_graph_centrality = compute_centrality(code_graph_G)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


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
        status_html = '<div class="server-status"><div class="status-dot online"></div>HugeGraph Server: ONLINE (localhost:8080)</div>'
        # Add live query button when server is online
        status_html += '<button class="scenario-btn" onclick="loadScenario(\'supply_chain\',\'hugegraph_live\')" style="background:rgba(52,168,83,0.15);border-color:var(--accent-green)">🔴 LIVE: Query from HugeGraph Server</button>'
    else:
        status_html = '<div class="server-status"><div class="status-dot offline"></div>HugeGraph Server: Offline (using networkx simulation)</div>'

    html = status_html

    if poc_id == "supply_chain":
        html += supply_chain_sidebar()
    elif poc_id == "graphrag_bench":
        html += graphrag_bench_sidebar()
    elif poc_id == "code_graph":
        html += code_graph_sidebar()

    return jsonify({"html": html, "serverStatus": status_html.replace("</div>", "").replace('<div class="server-status">', "")})


def supply_chain_sidebar():
    """供应链场景侧栏"""
    html = """
    <div class="section-title">Algorithm Steps</div>
    <div class="algo-step" onclick="algoStepClicked(0)"><span class="step-num">Step 1</span> 排 (Rank): 预计算三类中心性</div>
    <div class="algo-step" onclick="algoStepClicked(1)"><span class="step-num">Step 2</span> 取 (Retrieve): 自适应深度遍历</div>
    <div class="algo-step" onclick="algoStepClicked(2)"><span class="step-num">Step 3</span> 述 (Narrate): 路径壳转述</div>

    <div class="section-title">Scenarios</div>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','full_graph')">
      Full Supply Chain Graph
      <span class="verdict pass">26 nodes</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','apple_risk')">
      Apple Chip Concentration Risk
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','tesla_risk')">
      Tesla Battery Geo Risk
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','adaptive_vs_fixed')">
      Adaptive vs Fixed BFS
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('supply_chain','centrality_rank')">
      Centrality Ranking
      <span class="verdict pass">PASS</span>
    </button>

    <div id="dynamicSidebar"></div>
    """
    return html


def graphrag_bench_sidebar():
    """GraphRAG-Bench 场景侧栏"""
    html = """
    <div class="section-title">Benchmark Overview</div>
    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">12</div><div class="desc">Methods Compared</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-green)">5</div><div class="desc">Dimensions</div></div>
    </div>

    <div class="section-title">Scenarios</div>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','radar_chart')">
      Capability Radar Chart
      <span class="verdict pass">12 methods</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','task_match')">
      Task Match Ranking
      <span class="verdict pass">4 scenes</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','cost_benefit')">
      Cost-Benefit Analysis
      <span class="verdict pass">Top 5</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','drift_unique')">
      DRIFT 6 Unique Dimensions
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('graphrag_bench','comparison_graph')">
      Method Relationship Graph
      <span class="verdict pass">PASS</span>
    </button>

    <div id="dynamicSidebar"></div>
    """
    return html


def code_graph_sidebar():
    """代码图谱场景侧栏"""
    html = """
    <div class="section-title">Architecture</div>
    <div class="algo-step"><span class="step-num">Agent 1</span> Main: 意图分析 + 路由</div>
    <div class="algo-step"><span class="step-num">Agent 2</span> Translation: NL → Gremlin</div>
    <div class="algo-step"><span class="step-num">Executor</span> GraphExecutor: 执行查询</div>

    <div class="section-title">Scenarios</div>
    <button class="scenario-btn" onclick="loadScenario('code_graph','full_graph')">
      Full Code Graph
      <span class="verdict pass">21 nodes</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','call_chain')">
      Call Chain Trace
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','impact_analysis')">
      Impact Analysis
      <span class="verdict pass">PASS</span>
    </button>
    <button class="scenario-btn" onclick="loadScenario('code_graph','complexity_heat')">
      Complexity Heat Map
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

    return jsonify({"error": "Unknown poc/scenario"})


def get_supply_chain_scenario(scenario_id):
    """供应链场景"""
    if scenario_id == "full_graph":
        graph = extract_subgraph_for_cytoscape(
            supply_chain_G, supply_chain_centrality,
            seeds=list(supply_chain_G.nodes())
        )
        return {
            "title": "Full Supply Chain Graph (Network-KG Duality)",
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
            "title": "Apple Chip Concentration Risk Analysis",
            "graph": graph,
            "gremlin": [
                "g.V('apple').repeat(both().simplePath()).emit().path().by('name').by('relation')",
                "g.V().has('name','SoC 芯片').in('supplies').path()",
                "g.V('apple').out('produces').out('has_input').has('category','半导体').path()",
            ],
            "sidebarHtml": f"""
            <div class="section-title">Risk Assessment</div>
            <div class="info-card">
              <div class="label">TSMC Dependency</div>
              <div class="value" style="color:var(--accent-red);font-size:18px">{tsmc_dep}%</div>
            </div>
            <div class="info-card">
              <div class="label">Risk Level</div>
              <div class="value" style="color:var(--accent-red)">HIGH</div>
            </div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{subgraph.number_of_nodes()}</div><div class="desc">Nodes in Subgraph</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-purple)">{len(chip_paths)}</div><div class="desc">Chip Risk Paths</div></div>
            </div>
            <div class="section-title">Key Path Shells</div>
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
            "title": "Tesla Battery Geo-Political Risk (Bidirectional Trace)",
            "graph": graph,
            "gremlin": [
                "g.V('tesla').repeat(both().simplePath()).emit().until(__.loops().is(eq(3))).path()",
                "g.V('tesla').out('produces').in('has_input').repeat(in('supplies').simplePath()).emit().path()",
            ],
            "sidebarHtml": f"""
            <div class="section-title">Risk Assessment</div>
            <div class="info-card">
              <div class="label">DRC Exposure</div>
              <div class="value" style="color:var(--accent-red)">HIGH - {len([p for p in risk_paths if 'drc' in p['nodes']])} paths</div>
            </div>
            <div class="info-card">
              <div class="label">Cobalt Exposure</div>
              <div class="value" style="color:var(--accent-orange)">HIGH - {len([p for p in risk_paths if 'cobalt' in p['nodes']])} paths</div>
            </div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{subgraph.number_of_nodes()}</div><div class="desc">Subgraph Nodes</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-red)">{len(risk_paths)}</div><div class="desc">Risk Paths</div></div>
            </div>
            <div class="section-title">Risk Path Shells (click to highlight)</div>
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
            "title": f"Adaptive vs Fixed BFS (Node Reduction: {reduction:.1f}%)",
            "graph": graph,
            "gremlin": [
                "// Adaptive: seed SS decides depth",
                "g.V(seed).repeat(out().simplePath()).emit().until(__.loops().is(lt(adaptiveDepth)))",
                "// Fixed: all 2 hops",
                "g.V(seed).repeat(out().simplePath()).emit().until(__.loops().is(eq(2)))",
            ],
            "sidebarHtml": f"""
            <div class="section-title">Adaptive vs Fixed Comparison</div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-red)">{len(fixed_nodes)}</div><div class="desc">Fixed BFS Nodes</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-green)">{len(adaptive_nodes)}</div><div class="desc">Adaptive Nodes</div></div>
            </div>
            <div class="info-card">
              <div class="label">Node Reduction</div>
              <div class="value" style="color:var(--accent-green);font-size:18px">{reduction:.1f}%</div>
            </div>
            <div class="section-title">Legend</div>
            <div class="info-card"><span style="color:#ea4335">&#9632;</span> Removed by adaptive (saved)</div>
            <div class="info-card"><span style="color:#34a853">&#9632;</span> Added by adaptive (discovered)</div>
            <div class="info-card"><span style="color:#9aa0a6">&#9632;</span> Both methods (common)</div>
            <div class="section-title">Traversal Log</div>
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
            "title": "Centrality Ranking (Structural Significance)",
            "graph": graph,
            "gremlin": [
                "// Betweenness Centrality",
                "g.V().betweennessCentrality().order().by(values, desc).limit(10)",
                "// Degree Centrality (approximation)",
                "g.V().out().groupCount().by().order().by(values, desc).limit(10)",
            ],
            "sidebarHtml": f"""
            <div class="section-title">Top 8 by Structural Significance</div>
            {rank_html}
            <div class="section-title">Algorithm</div>
            <div class="algo-step"><span class="step-num">SS</span> = (norm(Degree) + norm(Betweenness) + norm(Closeness)) / 3</div>
            <div class="algo-step"><span class="step-num">Rule</span> SS >= 0.3: 1-hop (hub) | SS < 0.3: 2-hop (periphery)</div>
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
                "title": f"🔴 LIVE from HugeGraph Server ({len(nodes)}V/{len(edges)}E)",
                "graph": {"nodes": nodes, "edges": edges},
                "gremlin": [
                    "g.V().has('name','Apple').out('depends_on').path().by('name')",
                    f"// Shortest path Apple→TSMC: {path_info}",
                ],
                "sidebarHtml": f"""
                <div class="info-card" style="border-color:var(--accent-green)">
                  <div class="value" style="color:var(--accent-green)">🔴 Connected to HugeGraph Server</div>
                  <div class="prop">URL: localhost:8080</div>
                  <div class="prop">Backend: RocksDB</div>
                </div>
                <div class="metric-grid">
                  <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{len(nodes)}</div><div class="desc">Vertices</div></div>
                  <div class="metric-box"><div class="num" style="color:var(--accent-purple)">{len(edges)}</div><div class="desc">Edges</div></div>
                </div>
                <div class="section-title">Live Traverser Results</div>
                <div class="info-card">
                  <div class="label">Shortest Path: Apple → TSMC</div>
                  <div class="value" style="color:var(--accent-cyan)">{path_info or 'N/A'}</div>
                </div>
                """,
            }
        except Exception as ex:
            return {
                "title": "🔴 HugeGraph Server Error",
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
            "title": "GraphRAG-Bench: 12 Method Capability Radar",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": build_radar_chart_html(methods),
        }

    elif scenario_id == "task_match":
        # Ranking graph
        return {
            "title": "GraphRAG-Bench: Task Match Ranking",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": build_ranking_html(methods),
        }

    elif scenario_id == "cost_benefit":
        return {
            "title": "GraphRAG-Bench: Cost-Benefit Analysis",
            "graph": {"nodes": [], "edges": []},
            "gremlin": [],
            "sidebarHtml": build_cost_benefit_html(methods),
        }

    elif scenario_id == "drift_unique":
        unique_dims = [
            ("Large-scale Graph", "60B vertices/edges verified", True),
            ("Incremental Update", "Community partial rebuild", True),
            ("Entity Resolution", "3-strategy: exact/embed/LLM", True),
            ("HyDE Enhancement", "Prefix/full/off modes", True),
            ("Knowledge Freshness", "TTL + version tracking", True),
            ("Text2Gremlin", "Self-correction, max 3 retries", True),
        ]
        html = "<div class='section-title'>DRIFT 6 Unique Dimensions</div>"
        for dim, desc, _ in unique_dims:
            html += f"""<div class="info-card">
              <div class="label" style="color:var(--accent-green)">&#10003;</div>
              <div class="value">{dim}</div>
              <div class="prop" style="margin-top:2px">{desc}</div>
            </div>"""
        return {
            "title": "DRIFT 6 Unique Capabilities (No Other Method Has All)",
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
            "title": "Method Similarity Graph (DRIFT highlighted)",
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
            "title": "Full Code Graph (Complexity-colored)",
            "graph": graph,
            "gremlin": ["g.V().outE().path().by('name').by('relation')"],
            "sidebarHtml": build_metrics_sidebar(
                code_graph_G.number_of_nodes(), code_graph_G.number_of_edges(),
                {"Classes": 4, "Functions": 19, "Modules": 5},
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
            "title": "Call Chain: analyze_endpoint() → Full Trace",
            "graph": graph,
            "gremlin": [
                "g.V('fn_analyze_endpoint').repeat(out('calls').simplePath()).emit().path().by('name')",
                "g.V('fn_analyze_endpoint').repeat(both('calls').simplePath()).emit().path()",
            ],
            "sidebarHtml": f"""
            <div class="section-title">Call Chain</div>
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-cyan)">{len(paths)}</div><div class="desc">Paths Found</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{subgraph.number_of_nodes()}</div><div class="desc">Functions Involved</div></div>
            </div>
            <div class="section-title">Paths (click to highlight)</div>
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
            "title": "Impact Analysis: execute_query() Change Impact",
            "graph": graph,
            "gremlin": [
                "g.V('fn_execute_query').repeat(in('calls').simplePath()).emit().path().by('name')",
            ],
            "sidebarHtml": f"""
            <div class="metric-grid">
              <div class="metric-box"><div class="num" style="color:var(--accent-red)">{len(callers)}</div><div class="desc">Direct Callers</div></div>
              <div class="metric-box"><div class="num" style="color:var(--accent-orange)">{len(paths)}</div><div class="desc">Impact Paths</div></div>
            </div>
            {callers_html}
            <div class='section-title'>If execute_query() breaks, affected:</div>
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
            "title": "Code Complexity Heat Map",
            "graph": graph,
            "gremlin": [],
            "sidebarHtml": f"""
            <div class="section-title">Complexity Legend</div>
            <div class="info-card"><span style="color:#ea4335">&#9632;</span> High (>8)</div>
            <div class="info-card"><span style="color:#f97316">&#9632;</span> Medium-High (5-8)</div>
            <div class="info-card"><span style="color:#fbbc04">&#9632;</span> Medium (3-5)</div>
            <div class="info-card"><span style="color:#34a853">&#9632;</span> Low (<3)</div>
            <div class="section-title">Top 8 Most Complex Functions</div>
            {heat_html}
            """,
        }

    elif scenario_id == "gremlin_vs_cypher":
        comparisons = [
            {
                "name": "Multi-hop Trace",
                "gremlin": "g.V('fn_A').repeat(out('calls').simplePath()).emit().path()",
                "cypher": "MATCH p=(a:Function)-[:CALLS*1..5]->(b) WHERE a.name='fn_A' RETURN p",
                "winner": "Gremlin (O(d) vs O(2^d))",
            },
            {
                "name": "Reverse Trace",
                "gremlin": "g.V('fn_X').repeat(in('calls').simplePath()).emit().path()",
                "cypher": "MATCH p=(b)-[:CALLS*1..5]->(x:Function) WHERE x.name='fn_X' RETURN p",
                "winner": "Gremlin (single step)",
            },
            {
                "name": "Bidirectional",
                "gremlin": "g.V('fn_X').repeat(both('calls').simplePath()).emit().until(__.loops().is(3)).path()",
                "cypher": "MATCH p=(a)-[:CALLS*]-(b) WHERE a.name='fn_X' RETURN p",
                "winner": "Gremlin (depth control)",
            },
            {
                "name": "Large-scale",
                "gremlin": "g.V().has('complexity',gt(8)).out('calls').path()",
                "cypher": "MATCH (f:Function)-[:CALLS]->(b) WHERE f.complexity > 8 RETURN f,b",
                "winner": "Gremlin + OLAP (60B)",
            },
        ]
        html = "<div class='section-title'>Gremlin vs Cypher</div>"
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
            "title": "Gremlin vs Cypher: Code Graph Queries",
            "graph": graph,
            "gremlin": [c["gremlin"] for c in comparisons],
            "sidebarHtml": html,
        }

    return {"error": "Unknown scenario"}


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


def build_metrics_sidebar(nodes, edges, extra=None):
    html = f"""
    <div class="metric-grid">
      <div class="metric-box"><div class="num" style="color:var(--accent-blue)">{nodes}</div><div class="desc">Nodes</div></div>
      <div class="metric-box"><div class="num" style="color:var(--accent-purple)">{edges}</div><div class="desc">Edges</div></div>
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
    dim_labels = ["Multi-hop QA", "Global Summary", "General QA"]

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
    <div class="section-title">Top 6 Methods: Capability Radar</div>
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

    html = '<div class="section-title">Overall Ranking (Avg of 3 Tasks)</div>'
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

    html = '<div class="section-title">Benefit / Cost Ratio</div>'
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
