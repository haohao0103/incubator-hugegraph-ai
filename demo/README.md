# CodeGraph 交互式 Demo

本目录包含 CodeGraph 代码知识图谱的可视化演示系统，可直接用于向 CTO/团队展示代码图谱的构建、查询与分析能力。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `codegraph_demo_server.py` | Flask 后端，提供图数据 API |
| `codegraph_demo.html` | 主可视化页面（力导向图 + 交互面板） |
| `codegraph_build_demo.html` | 图构建流程 6 步演示 |
| `codegraph_parsed.json` | 预解析的业界标准 requests 库代码图（321 节点 / 1633 边） |

---

## 快速启动

```bash
# 进入仓库根目录后
python demo/codegraph_demo_server.py
```

服务启动后：

- 代码图可视化：http://localhost:5100
- 图构建演示：http://localhost:5100/build

---

## 重新生成图数据

如果想解析其他项目，修改 `codegraph_demo_server.py` 中的 `GRAPH_TARGET_DIR` 变量，或运行：

```bash
python -c "
import json, sys
sys.path.insert(0, 'hugegraph-llm/src')
from hugegraph_llm.poc.codegraph_hugegraph_mcp import PythonCodeParser, find_python_files

parser = PythonCodeParser()
for fp in find_python_files('/path/to/your/code', max_files=100):
    try:
        parser.parse_file(fp)
    except Exception as e:
        print(f'skip {fp}: {e}')

nodes = [{'id': n.id, 'name': n.name, 'node_type': n.node_type,
          'file_path': n.file_path, 'line_start': n.line_start,
          'source_code': (n.source_code or '')[:300]} for n in parser.nodes]
edges = [{'source': e.source_id, 'target': e.target_id, 'edge_type': e.edge_type} for e in parser.edges]

with open('demo/codegraph_parsed.json', 'w') as f:
    json.dump({'nodes': nodes, 'edges': edges}, f, ensure_ascii=False, indent=1)
print(f'nodes={len(nodes)}, edges={len(edges)}')
"
```

---

## API 列表

| 接口 | 说明 |
|------|------|
| `GET /api/stats` | 全局统计、Top Hubs |
| `GET /api/graph?node_type=&edge_type=` | 返回完整图或按类型过滤 |
| `GET /api/search?q=...&limit=...` | BM25 代码搜索 |
| `GET /api/neighbors/<id>` | 1-hop 邻居子图 |
| `GET /api/hubs?limit=...` | Top hub 节点 |
| `GET /api/impact/<id>` | 修改影响范围分析 |
| `GET /api/traverse?source=...&hops=...&direction=...` | 多跳可达遍历 |
| `GET /api/callers/<id>` | 调用当前节点的函数 |
| `GET /api/callees/<id>` | 当前节点调用的函数 |

---

## 交互说明

- **拖拽节点**：调整布局
- **单击节点**：右侧面板显示详情、代码片段、度数
- **右键节点**：影响分析 / 多跳遍历 / 调用方 / 被调用方
- **搜索框**：输入函数名/类名回车，聚焦并高亮
- **顶部模式按钮**：切换 calls / contains / imports / inherits / all
