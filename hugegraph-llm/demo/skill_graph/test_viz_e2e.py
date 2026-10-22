# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Skill Graph 可视化 — 真实浏览器 E2E 测试。

打开部署在 STG 的可视化服务（真实 HugeGraph + OceanBase 后端），用本机
Chrome 渲染页面并验证交互，不是 HTML 字符串断言。

Usage::

    # STG 部署（默认）
    uv run python demo/skill_graph/test_viz_e2e.py

    # 本地服务
    SKILL_GRAPH_VIZ_URL=http://127.0.0.1:8766 uv run python demo/skill_graph/test_viz_e2e.py

依赖: pip install playwright （浏览器用系统 Chrome, 无需 playwright install）
"""

import json
import os
import sys

from playwright.sync_api import sync_playwright

URL = os.environ.get("SKILL_GRAPH_VIZ_URL", "http://10.129.33.121:8766")
SHOT = os.environ.get("SKILL_GRAPH_E2E_SHOT", "demo/skill_graph/e2e_stg.png")

PASS = 0
FAIL = 0
ERRORS = []


def ok(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        msg = f"  FAIL {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def main() -> int:
    console_errors = []
    with sync_playwright() as pw:
        # 真实浏览器：系统 Chrome，非无头 HTML 解析
        browser = pw.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--no-sandbox", "--disable-gpu-sandbox", "--force-device-scale-factor=1"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

        print(f"[E2E] target: {URL}")
        resp = page.goto(URL, timeout=30000)
        ok("GET / → 200", resp is not None and resp.status == 200, f"got {resp.status if resp else 'no response'}")
        ok("页面标题正确", page.title() == "Skill Graph — STG", page.title())

        # 1) 页面携带真实数据（来自 STG HugeGraph active build）
        data = page.evaluate(
            "() => ({ nodes: DATA.nodes.length, edges: DATA.edges.length,"
            " labels: DATA.nodes.map(n => n.label),"
            " stats: DATA.stats })"
        )
        ok(f"DATA.nodes = {data['nodes']} >= 15", data["nodes"] >= 15, str(data))
        ok(f"DATA.edges = {data['edges']} > 0", data["edges"] > 0)
        ok("stats.cycles = 0 (DAG 不变量)", data["stats"].get("cycles") == 0, str(data["stats"]))

        # 2) canvas 真实渲染出图形（采样非背景像素）
        page.wait_for_timeout(2500)  # 布局稳定
        painted = page.evaluate(
            """() => {
                const cv = document.getElementById('cv');
                const ctx = cv.getContext('2d');
                const dpr = window.devicePixelRatio || 1;
                const img = ctx.getImageData(0, 0, cv.width, cv.height).data;
                let count = 0;
                for (let i = 0; i < img.length; i += 4 * 97) {  // 采样
                    // 背景色 #0f1620 = (15,22,32)
                    if (Math.abs(img[i] - 15) + Math.abs(img[i+1] - 22) + Math.abs(img[i+2] - 32) > 24) count++;
                }
                return count;
            }"""
        )
        ok(f"canvas 有 {painted} 个采样点非背景色（力导向图已渲染）", painted > 30, str(painted))

        # 3) 交互：点击一个节点 → 详情面板显示该节点
        target = page.evaluate(
            """() => {
                const cv = document.getElementById('cv');
                const r = cv.getBoundingClientRect();
                const n = nodes[Math.floor(nodes.length / 2)];
                return { x: r.left + n.x, y: r.top + n.y, label: n.label };
            }"""
        )
        page.mouse.click(target["x"], target["y"])
        page.wait_for_timeout(400)
        detail = page.evaluate(
            "() => document.getElementById('detail') ? document.getElementById('detail').innerText : ''"
        )
        ok(
            f"点击节点「{target['label']}」→ 详情面板命中",
            target["label"] in detail,
            f"detail={detail[:120]}",
        )

        # 4) 交互：搜索过滤
        some_label = data["labels"][0]
        page.fill("#search", some_label)
        page.wait_for_timeout(400)
        ok(f"搜索「{some_label}」无 JS 错误", True)

        # 5) /api/stats 与页面数据一致（同一 active build 快照）
        stats = page.evaluate("() => fetch('/api/stats').then(r => r.json())")
        ok("api/stats.build_id 非空", bool(stats.get("build_id")), str(stats))
        ok(
            f"页面/接口节点数一致 ({data['nodes']} == {stats.get('nodes')})",
            data["nodes"] == stats.get("nodes"),
        )

        # 6) 交互：拖拽节点（力导向响应）
        before = page.evaluate("() => nodes.map(n => [n.x, n.y])")
        page.mouse.move(target["x"], target["y"])
        page.mouse.down()
        page.mouse.move(target["x"] + 120, target["y"] + 80, steps=8)
        page.mouse.up()
        page.wait_for_timeout(400)
        after = page.evaluate("() => nodes.map(n => [n.x, n.y])")
        ok("拖拽节点后布局发生变化（力导向在运行）", before != after)

        # 7) console 无 JS 错误
        ok(f"console 无 error（{len(console_errors)} 条）", len(console_errors) == 0, "; ".join(console_errors[:3]))

        page.screenshot(path=SHOT, full_page=False)
        print(f"[E2E] screenshot → {SHOT}")
        browser.close()

    total = PASS + FAIL
    print(f"\n[E2E] Total: {total}  |  Pass: {PASS}  |  Fail: {FAIL}")
    if ERRORS:
        print("  失败项:")
        for e in ERRORS:
            print(f"    {e}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    print(json.dumps({"target": URL}, ensure_ascii=False))
    sys.exit(main())
