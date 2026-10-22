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

"""Self-contained interactive visualisation for the skill graph.

Produces a single HTML file (no external CDN, works offline) with a
force-directed layout of the graph. An optional retrieval result highlights
*why* each skill was pulled in (seed vs. prerequisite vs. composition).
"""

from __future__ import annotations

import html as _html
import json
from typing import Any, Dict, List, Optional

from hugegraph_llm.engines.skill_graph.schema import EdgeType, SkillGraph

_EDGE_COLORS = {
    EdgeType.DEPENDS_ON.value: "#2c3e50",
    EdgeType.COMPOSES_WITH.value: "#8e44ad",
    EdgeType.SIMILAR_TO.value: "#f39c12",
    EdgeType.BELONGS_TO.value: "#95a5a6",
    EdgeType.CONFLICTS_WITH.value: "#c0392b",
}

_REASON_COLORS = {
    "seed": "#e74c3c",
    "prerequisite": "#27ae60",
    "composition": "#8e44ad",
    "similar": "#f39c12",
}

# Human-friendly (Chinese) labels for the beginner-facing view.
_EDGE_LABELS = {
    EdgeType.DEPENDS_ON.value: "依赖（A 需要 B 作为前置）",
    EdgeType.COMPOSES_WITH.value: "组合（A 常与 B 一起用）",
    EdgeType.SIMILAR_TO.value: "相似（A 与 B 可互相替代）",
    EdgeType.BELONGS_TO.value: "属于（技能归属于类别）",
    EdgeType.CONFLICTS_WITH.value: "冲突（A 与 B 互斥）",
}

_REASON_LABELS = {
    "seed": "直接命中",
    "prerequisite": "前置依赖",
    "composition": "组合搭配",
    "similar": "相似替代",
}


def _payload(
    graph: SkillGraph,
    result: Optional[Any] = None,
    title: str = "Skill Graph",
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    for n in graph.nodes.values():
        nodes.append(
            {
                "id": n.id,
                "label": n.name,
                "type": n.node_type.value,
                "category": n.category,
                "description": n.description,
                "tags": n.tags,
                "tokens": n.token_estimate(),
            }
        )
    edges: List[Dict[str, Any]] = []
    for e in graph.edges:
        edges.append(
            {
                "source": e.source,
                "target": e.target,
                "type": e.edge_type.value,
                "weight": round(e.weight, 3),
                "provenance": e.provenance,
            }
        )

    highlight: Optional[Dict[str, Any]] = None
    value: Optional[Dict[str, Any]] = None
    if result is not None:
        highlight = {}
        for s in getattr(result, "skills", []):
            highlight[s.id] = {"reason": s.reason, "depth": s.depth}
        highlight["__meta__"] = {
            "query": getattr(result, "query", ""),
            "seeds": [s.id for s in getattr(result, "seeds", [])],
            "total_tokens": getattr(result, "total_tokens", 0),
        }

        # "Keyword match" = the seed matches only; the graph additionally pulls
        # in prerequisites / compositions the task needs. The delta is the value
        # the skill graph adds: it completes the toolkit the task needs.
        seeds = list(getattr(result, "seeds", []))
        skills = list(getattr(result, "skills", []))
        added = [s for s in skills if s.reason != "seed"]
        value = {
            "keyword_skills": len(seeds),
            "keyword_tokens": sum(getattr(s, "token_estimate", 0) for s in seeds),
            "added_skills": len(added),
            "delivered_skills": len(skills),
            "delivered_tokens": getattr(result, "total_tokens", 0),
            "added": [
                {"id": s.id, "name": s.name, "reason": s.reason}
                for s in added
            ],
        }

    return {
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "highlight": highlight,
        "value": value,
        "colors": {"edges": _EDGE_COLORS, "reasons": _REASON_COLORS},
        "labels": {"edges": _EDGE_LABELS, "reasons": _REASON_LABELS},
        "stats": graph.stats(),
    }


def render_html(
    graph: SkillGraph,
    result: Optional[Any] = None,
    title: str = "Skill Graph",
) -> str:
    """Render the skill graph as a standalone interactive HTML document."""
    data = json.dumps(_payload(graph, result, title), ensure_ascii=False)
    # Guard against a literal "</script>" breaking out of the script tag.
    data = data.replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", _html.escape(title)).replace("__DATA__", data)


def write_html(
    graph: SkillGraph,
    path: str,
    result: Optional[Any] = None,
    title: str = "Skill Graph",
) -> str:
    """Write :func:`render_html` output to ``path`` and return the path."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(graph, result=result, title=title))
    return path


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1620; --panel:#16202b; --fg:#e8eef5; --muted:#8ba0b3; --accent:#7fd1a8; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  body { display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  #hero { padding:14px 20px 12px; border-bottom:1px solid #22303d; background:linear-gradient(180deg,#182533,#121b25); }
  #hero h1 { font-size:17px; margin:0 0 4px; }
  #hero p { margin:0; font-size:12.5px; line-height:1.6; color:#a9bccd; max-width:1000px; }
  #hero .kbd { color:var(--accent); font-weight:600; }
  #app { display:flex; flex:1; min-height:0; }
  #stage { flex:1; position:relative; }
  canvas { display:block; width:100%; height:100%; cursor:grab; }
  #sidebar { width:360px; background:var(--panel); border-left:1px solid #22303d; padding:16px; overflow-y:auto; }
  h2 { font-size:12px; text-transform:none; letter-spacing:.02em; color:var(--muted); margin:18px 0 8px; }
  h2:first-child { margin-top:0; }
  .stats { font-size:12px; color:var(--muted); }
  .legend { font-size:12px; line-height:1.9; }
  .swatch { display:inline-block; width:22px; height:0; border-top-width:3px; border-top-style:solid; vertical-align:middle; margin-right:8px; }
  .dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:8px; vertical-align:middle; }
  #search { width:100%; padding:8px 10px; border-radius:8px; border:1px solid #2a3a4a; background:#0d151d; color:var(--fg); font-size:13px; }
  #detail { font-size:13px; line-height:1.6; }
  #detail .name { font-weight:600; font-size:14px; }
  #detail .desc { color:var(--muted); margin:6px 0; }
  .tag { display:inline-block; font-size:11px; background:#20303f; color:#bcd2e4; border-radius:999px; padding:1px 8px; margin:2px 4px 2px 0; }
  #querywrap { font-size:12.5px; background:#122029; border:1px solid #22303d; border-radius:8px; padding:12px; margin-top:6px; }
  #querywrap b { color:var(--accent); }
  .row { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .pill { font-size:11px; border-radius:999px; padding:1px 7px; color:#04121b; font-weight:600; }
  .cmp { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
  .cmp .card { background:#0d151d; border:1px solid #22303d; border-radius:8px; padding:8px 10px; }
  .cmp .card.graph { border-color:#2f6b4f; background:#101f19; }
  .cmp .card .k { font-size:11px; color:var(--muted); }
  .cmp .card .v { font-size:19px; font-weight:700; margin-top:2px; }
  .cmp .card .t { font-size:11px; color:var(--muted); }
  .delta { margin-top:10px; font-size:12px; line-height:1.65; }
  .delta .hl { color:var(--accent); font-weight:600; }
  .added { margin-top:8px; display:flex; flex-wrap:wrap; gap:5px; }
  .added .chip { font-size:11px; border-radius:6px; padding:2px 7px; background:#16281f; color:#bfe6cf; border:1px solid #2f6b4f; }
  .added .chip .rl { color:#8fb8a1; }
</style>
</head>
<body>
<header id="hero">
  <h1 id="title"></h1>
  <p>技能图谱把零散的「技能」用关系连成一张网。当 AI 接到任务时，它不再凭关键词瞎找，
     而是沿这张网找到真正需要的技能 —— 包括你<span class="kbd">没说出口、却必须先具备的前置技能</span>。
     蓝点＝技能，橙点＝类别；连线颜色代表技能之间的关系，点任意节点可查看详情。</p>
</header>
<div id="app">
  <div id="stage"><canvas id="cv"></canvas></div>
  <div id="sidebar">
    <h2>这张图有多大</h2>
    <div class="stats" id="stats"></div>
    <div id="querywrap"></div>
    <h2>图例 · 节点</h2>
    <div class="legend">
      <div><span class="dot" style="background:#4c9be8"></span>技能（一个可复用的能力）</div>
      <div><span class="dot" style="background:#e6a23c"></span>类别（技能的分组）</div>
    </div>
    <h2>图例 · 关系（连线）</h2>
    <div class="legend" id="edgelist"></div>
    <h2>颜色 = 为什么被选中</h2>
    <div class="legend" id="reasonlist"></div>
    <h2>筛选</h2>
    <input id="search" placeholder="输入技能名筛选…"/>
    <h2>选中详情</h2>
    <div id="detail">点击图中任意节点查看详情。</div>
  </div>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let W = 0, H = 0, DPR = window.devicePixelRatio || 1;
function resize(){
  const r = cv.parentElement.getBoundingClientRect();
  W = r.width; H = r.height;
  cv.width = W * DPR; cv.height = H * DPR;
  ctx.setTransform(DPR,0,0,DPR,0,0);
}
window.addEventListener('resize', resize);

const R = 7;
const nodes = DATA.nodes.map((n,i) => ({
  ...n,
  x: (window.innerWidth-330)/2 + Math.cos(i*2.4)*180*Math.random(),
  y: window.innerHeight/2 + Math.sin(i*2.4)*180*Math.random(),
  vx:0, vy:0
}));
const index = {}; nodes.forEach((n,i)=>index[n.id]=i);
const links = DATA.edges.map(e => ({ s:index[e.source], t:index[e.target], type:e.type, weight:e.weight }))
                        .filter(l => l.s!==undefined && l.t!==undefined);
const deg = {}; links.forEach(l => { deg[l.s]=(deg[l.s]||0)+1; deg[l.t]=(deg[l.t]||0)+1; });

const reasons = {};
let seeds = new Set();
if (DATA.highlight){
  for (const k in DATA.highlight){ if (k!=='__meta__') reasons[k]=DATA.highlight[k]; }
  seeds = new Set(DATA.highlight.__meta__.seeds || []);
}
const dimmed = !!DATA.highlight;

function nodeColor(n){ return n.type === 'category' ? '#e6a23c' : '#4c9be8'; }
function ringColor(id){ const r = reasons[id]; return r ? DATA.colors.reasons[r.reason] : null; }
function inResult(id){ return !!reasons[id] || seeds.has(id); }
let enabled = () => true;

function step(){
  for (let i=0;i<nodes.length;i++){
    const a = nodes[i];
    for (let j=i+1;j<nodes.length;j++){
      const b = nodes[j];
      let dx=b.x-a.x, dy=b.y-a.y; let d2=dx*dx+dy*dy; if(d2<1)d2=1;
      const d=Math.sqrt(d2); const f=2600/d2;
      const fx=f*dx/d, fy=f*dy/d;
      a.vx-=fx; a.vy-=fy; b.vx+=fx; b.vy+=fy;
    }
  }
  links.forEach(l => {
    const a=nodes[l.s], b=nodes[l.t];
    let dx=b.x-a.x, dy=b.y-a.y; let d=Math.sqrt(dx*dx+dy*dy)||1;
    const k = (l.type==='belongs_to') ? 0.010 : 0.03;
    const f = k*(d-110);
    const fx=f*dx/d, fy=f*dy/d;
    a.vx+=fx; a.vy+=fy; b.vx-=fx; b.vy-=fy;
  });
  nodes.forEach(n => {
    n.vx += (W/2 - n.x)*0.0016; n.vy += (H/2 - n.y)*0.0016;
    if (n!==dragged){ n.x+=n.vx; n.y+=n.vy; }
    n.vx*=0.86; n.vy*=0.86;
    n.x=Math.max(R+4, Math.min(W-R-4, n.x));
    n.y=Math.max(R+4, Math.min(H-R-4, n.y));
  });
}

function draw(){
  ctx.clearRect(0,0,W,H);
  links.forEach(l => {
    const a=nodes[l.s], b=nodes[l.t];
    ctx.strokeStyle = DATA.colors.edges[l.type] || '#555';
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = l.type==='depends_on' ? 2 : 1.4;
    ctx.setLineDash(l.type==='similar_to' ? [3,3] : (l.type==='belongs_to' ? [6,4] : []));
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  });
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  nodes.forEach(n => {
    const r = R + Math.min(6,(deg[index[n.id]]||0));
    ctx.globalAlpha = (dimmed && !inResult(n.id)) ? 0.15 : (enabled(n)?1:0.2);
    ctx.beginPath(); ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle = nodeColor(n); ctx.fill();
    const rc = ringColor(n.id);
    if (rc){ ctx.lineWidth=3; ctx.strokeStyle=rc; ctx.stroke(); }
    if (n.type !== 'category'){
      ctx.globalAlpha = (dimmed && !inResult(n.id)) ? 0.2 : 0.95;
      ctx.fillStyle = '#dce7f1';
      ctx.font='11px -apple-system,Segoe UI,sans-serif';
      ctx.textAlign='center';
      ctx.fillText(n.label, n.x, n.y - r - 5);
    }
  });
  ctx.globalAlpha = 1;
}

function loop(){ step(); draw(); requestAnimationFrame(loop); }

let dragged=null, down=null;
function pos(ev){ const r=cv.getBoundingClientRect(); return {x:ev.clientX-r.left, y:ev.clientY-r.top}; }
function pick(p){
  let best=null, bd=1e9;
  nodes.forEach(n => { const d=(n.x-p.x)**2+(n.y-p.y)**2; if(d<bd){bd=d;best=n;} });
  return (best && bd < 400) ? best : null;
}
cv.addEventListener('mousedown', ev => { const p=pos(ev); dragged=pick(p); down=dragged; });
cv.addEventListener('mousemove', ev => {
  const p=pos(ev);
  if (dragged){ dragged.x=p.x; dragged.y=p.y; dragged.vx=dragged.vy=0; }
  else { cv.style.cursor = pick(p)?'pointer':'grab'; }
});
window.addEventListener('mouseup', () => { if (dragged && down===dragged) select(dragged); dragged=null; down=null; });

function select(n){
  const d = document.getElementById('detail');
  const tags = (n.tags||[]).map(t=>`<span class="tag">${t}</span>`).join('');
  const rc = reasons[n.id];
  const rl = (DATA.labels && DATA.labels.reasons) || {};
  const typeCn = n.type === 'category' ? '类别' : '技能';
  const badge = rc ? `<span class="pill" style="background:${DATA.colors.reasons[rc.reason]}">${rl[rc.reason] || rc.reason} · 第 ${rc.depth} 层</span>` : '';
  d.innerHTML = `<div class="row"><span class="name">${n.label}</span>${badge}</div>
    <div class="tag" style="background:#22303d">${typeCn}</div>
    <div class="desc">${n.description||''}</div>
    <div class="stats">类别：${n.category||'—'} · 约 ${n.tokens} tokens</div>
    <div>${tags}</div>`;
}

document.getElementById('search').addEventListener('input', ev => {
  const q = ev.target.value.trim().toLowerCase();
  enabled = (n) => !q || n.label.toLowerCase().includes(q);
});

const REASON_LABELS = (DATA.labels && DATA.labels.reasons) || {};
const EDGE_LABELS = (DATA.labels && DATA.labels.edges) || {};

document.getElementById('edgelist').innerHTML = Object.keys(DATA.colors.edges).map(t =>
  `<div><span class="swatch" style="border-top-color:${DATA.colors.edges[t]}"></span>${EDGE_LABELS[t] || t}</div>`).join('');

document.getElementById('reasonlist').innerHTML = Object.keys(DATA.colors.reasons).map(r =>
  `<div><span class="dot" style="background:${DATA.colors.reasons[r]}"></span>${REASON_LABELS[r] || r}</div>`).join('');

document.getElementById('title').textContent = DATA.title;
const s = DATA.stats;
document.getElementById('stats').innerHTML =
  `<b>${s.skills}</b> 个技能 · <b>${s.categories}</b> 个类别 · <b>${s.edges}</b> 条关系 · 依赖环 <b>${s.cycles}</b>`;

if (DATA.highlight){
  const m = DATA.highlight.__meta__;
  const v = DATA.value;
  let html = `<h2>本次任务</h2><div id="querywrap"><div>“${m.query}”</div>`;
  if (v){
    html += `<div class="cmp">
        <div class="card"><div class="k">① 只看关键词命中</div>
          <div class="v">${v.keyword_skills} 个</div><div class="t">~${v.keyword_tokens} tokens</div></div>
        <div class="card graph"><div class="k">② 图谱补齐的前置技能</div>
          <div class="v">+${v.added_skills} 个</div><div class="t">任务才能真正跑通</div></div>
      </div>`;
    if (v.added_skills > 0){
      html += `<div class="delta">扁平检索只给你 <b>${v.keyword_skills}</b> 个“看起来相关”的技能就停下了；
        技能图谱沿依赖关系又自动补齐了 <span class="hl">${v.added_skills} 个必须先具备的技能</span>，
        最终交付 <b>${v.delivered_skills}</b> 个技能（约 ${v.delivered_tokens} tokens）：</div>
        <div class="added">${v.added.map(a =>
          `<span class="chip">${a.name} <span class="rl">· ${REASON_LABELS[a.reason] || a.reason}</span></span>`).join('')}</div>`;
    } else {
      html += `<div class="delta">该数据集只声明了技能本身、没有显式依赖关系，所以图谱结果与关键词命中一致（共 <b>${v.delivered_skills}</b> 个技能）。
        “图谱补齐前置技能”的效果请看另一个演示：内置依赖语料。</div>`;
    }
  }
  html += `<div style="margin-top:10px;color:#8ba0b3">圈的颜色代表被选中的原因（见左侧图例）。</div></div>`;
  document.getElementById('querywrap').innerHTML = html;
}

resize(); loop();
</script>
</body>
</html>
"""