# HugeGraph CodeGraph PoC — 完全复现指南

> 本指南面向希望复现 **CodeGraph 代码知识图谱 PoC** 及其配套评测、交互式 Demo 的同学/同事。  
> 拉取仓库后，只需修改 `hugegraph-llm/.env` 中的 LLM 配置即可跑通绝大多数流程。

---

## 1. 代码与分支

```bash
git clone git@github.com:haohao0103/incubator-hugegraph-ai.git
cd incubator-hugegraph-ai
git checkout poc/0614-codegraph-hugegraph-mcp
```

> 当前所有 PoC、Demo、Benchmark、数据集均在该分支上。

---

## 2. 环境要求

| 组件 | 版本/说明 |
|------|----------|
| Python | 3.10.x 或 3.11.x（`hugegraph-llm` 限制 `<3.12`） |
| HugeGraph Server | 1.7.0（REST `8080` / Gremlin `8182`） |
| OS | macOS / Linux（本仓库开发于 macOS） |
| LLM | 任意 OpenAI 兼容接口（默认配置为小米 MiMo v2.5 Pro） |

---

## 3. 安装依赖

### 3.1 创建虚拟环境（推荐）

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3.2 安装 workspace 包

```bash
# 安装 hugegraph-llm 及其依赖
pip install -e ./hugegraph-llm

# 如需向量数据库后端
pip install -e "./hugegraph-llm[vectordb]"
```

### 3.3 CodeGraph Demo 额外依赖

```bash
pip install flask jieba rank_bm25
```

> `rank_bm25` 用于 CodeGraph 的 BM25 代码搜索；`jieba` 用于中文/英文分词；`flask` 用于 Demo 后端。

---

## 4. 启动 HugeGraph

如果你本地已有 HugeGraph 1.7.0：

```bash
bash /usr/local/hugegraph-server/bin/start-hugegraph.sh
```

验证服务：

```bash
curl http://127.0.0.1:8080/versions
```

> CodeGraph PoC 的 SQLite-only 路径**不需要** HugeGraph；只有 `HugeGraphCodeGraph` 和 GraphRAG-Bench 需要连接 HugeGraph。

---

## 5. 配置 LLM

### 5.1 复制环境变量模板

```bash
cp hugegraph-llm/.env.example hugegraph-llm/.env
```

### 5.2 修改 `.env`

打开 `hugegraph-llm/.env`，把 `OPENAI_*_API_KEY` 替换为你的真实 key。例如使用 OpenAI：

```ini
OPENAI_CHAT_API_BASE=https://api.openai.com/v1
OPENAI_CHAT_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_CHAT_LANGUAGE_MODEL=gpt-4.1-mini

OPENAI_EXTRACT_API_BASE=https://api.openai.com/v1
OPENAI_EXTRACT_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_EXTRACT_LANGUAGE_MODEL=gpt-4.1-mini

OPENAI_TEXT2GQL_API_BASE=https://api.openai.com/v1
OPENAI_TEXT2GQL_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_TEXT2GQL_LANGUAGE_MODEL=gpt-4.1-mini

OPENAI_AGENT_API_BASE=https://api.openai.com/v1
OPENAI_AGENT_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_AGENT_LANGUAGE_MODEL=gpt-4.1-mini

OPENAI_EMBEDDING_API_BASE=https://api.openai.com/v1
OPENAI_EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

使用小米 MiMo（默认）：

```ini
OPENAI_CHAT_API_BASE=https://api.xiaomimimo.com/v1
OPENAI_CHAT_API_KEY=your_mimo_key
OPENAI_CHAT_LANGUAGE_MODEL=mimo-v2.5-pro
# ... 其余 CHAT/EXTRACT/TEXT2GQL/AGENT/EMBEDDING 同样设置
```

### 5.3  HugeGraph 密码

```ini
GRAPH_URL=http://127.0.0.1:8080
GRAPH_NAME=hugegraph
GRAPH_USER=admin
GRAPH_PWD=xxx        # 改为你本地 HugeGraph 的 admin 密码
```

---

## 6. CodeGraph PoC 复现

### 6.1 运行单测（验证 PoC 正确性）

```bash
cd hugegraph-llm
pytest tests/poc/test_codegraph_hugegraph_mcp.py -v --cov=src/hugegraph_llm/poc/codegraph_hugegraph_mcp --cov-report=term-missing
```

预期结果：

```text
108 passed
97% statement coverage
```

### 6.2 运行 PoC（SQLite + HugeGraph 双后端）

```bash
cd ..
python hugegraph-llm/src/hugegraph_llm/poc/codegraph_hugegraph_mcp.py
```

运行后会在当前目录生成：

- `codegraph_hugegraph_mcp_result.json`：解析统计与查询结果
- 若 HugeGraph 可用，会自动写入图数据到 `GRAPH_NAME`

### 6.3 重新解析任意代码库

修改脚本中 `target_dir` 或调用 `PythonCodeParser`：

```python
from hugegraph_llm.poc.codegraph_hugegraph_mcp import PythonCodeParser, find_python_files

parser = PythonCodeParser()
for fp in find_python_files("/path/to/your/project", max_files=100):
    try:
        parser.parse_file(fp)
    except Exception as e:
        print(f"skip {fp}: {e}")

print(f"nodes={len(parser.nodes)}, edges={len(parser.edges)}")
```

---

## 7. 交互式 CodeGraph Demo

### 7.1 启动后端

```bash
python demo/codegraph_demo_server.py
```

### 7.2 打开页面

- **代码图可视化**：http://localhost:5100
- **图构建流程演示**：http://localhost:5100/build

### 7.3 主要交互

| 操作 | 效果 |
|------|------|
| 拖拽节点 | 自由布局 |
| 单击节点 | 右侧显示代码、元数据、度数 |
| 右键菜单 | 影响分析 / 多跳遍历 / 调用方 / 被调用方 |
| 搜索框 | 输入函数名回车，聚焦到对应节点 |
| 模式切换 | calls / contains / imports / inherits / all |

---

## 8. Benchmark 复现

### 8.1 代码图谱覆盖率 + BM25 搜索基准

```bash
python benchmark/run_codegraph_benchmark.py
```

依赖：

```bash
pip install requests flask django rank_bm25 jieba
```

输出：

- `benchmark/benchmark_result.json`
- 包含 requests / flask / django 三项目的节点/边统计、BM25 MRR/NDCG、调用图内部率、结构查询延迟

### 8.2 PyCG 调用图准确率评估

```bash
python benchmark/eval_pycg.py
```

输出：

- `benchmark/pycg_eval_result.json`

### 8.3 查看 CTO 仪表盘

```bash
cd benchmark
python3 -m http.server 5200 --directory .
```

访问：http://localhost:5200/cto_dashboard.html

---

## 9. GraphRAG-Bench 评测复现

> 需要 HugeGraph 已启动且 `.env` 中 LLM 配置正确。

### 9.1 原始 GraphRAG-Bench 评测

```bash
cd hugegraph-llm
python src/hugegraph_llm/poc/benchmark_llm_generation.py
```

运行后生成：

- `hugegraph-llm/src/hugegraph_llm/poc/benchmark_llm_generation_result.json`
- 并更新 `hugegraph-llm/docs/daily_research/2026-06-26-graphrag-bench-evaluation.md`

### 9.2 P0-Improved 全管道评测（推荐）

使用真实 embedding、FAISS + BM25 + HugeGraph 三通道检索、RRF 融合：

```bash
cd hugegraph-llm
python src/hugegraph_llm/poc/graphrag_bench_p0_improved.py
```

输出：

- `hugegraph-llm/poc_results/graphrag_bench_p0_improved_result.json`

> 注意：完整跑完约需 15-30 分钟，取决于 LLM 延迟。

---

## 10. MemSim 中文记忆基准复现

> 无需 HugeGraph，只需 `.env` 中 LLM 配置正确。

```bash
cd hugegraph-llm
python src/hugegraph_llm/poc/benchmark_memsim.py
```

输出：

- `hugegraph-llm/src/hugegraph_llm/poc/benchmark_memsim_result.json`

默认每个 split 采样 20 题（共 120 题），可通过 `.env` 中的 `MEMSIM_SAMPLE_PER_SPLIT` 调整。

---

## 11. 数据集说明

本仓库已内置以下数据集，可直接复现：

| 路径 | 来源 | 用途 |
|------|------|------|
| `hugegraph-llm/benchmark_data/GraphRAG-Bench/` | [GraphRAG-Bench](https://github.com/.../GraphRAG-Bench) (ICLR'26) | GraphRAG 端到端评测 |
| `hugegraph-llm/benchmark_data/MemSim/memdaily.json` | [MemSim](https://github.com/nuster1128/MemSim) | 中文记忆基准评测 |
| `benchmark_data/PyCG/` | [PyCG](https://github.com/vitsalis/PyCG) | 调用图准确率评测真值 |
| `demo_data/airports.dat` | OpenFlights | 交通/物流网络演示 |
| `demo_data/airlines.dat` | OpenFlights | 航空公司元数据 |
| `demo_data/routes.dat` | OpenFlights | 航线网络 |
| `demo_data/wiki-vote.txt` | SNAP | 社交网络演示 |
| `demo_data/amazon0302.txt` | SNAP | 产品共购网络演示 |

---

## 12. 常见问题

### Q1: `ImportError: cannot import name 'PythonCodeParser'`

确保 `hugegraph-llm/src` 在 `PYTHONPATH` 中：

```bash
export PYTHONPATH="$PWD/hugegraph-llm/src:$PYTHONPATH"
```

或已执行 `pip install -e ./hugegraph-llm`。

### Q2: HugeGraph 连接失败

- 检查 HugeGraph 是否启动：`curl http://127.0.0.1:8080/versions`
- 检查 `.env` 中 `GRAPH_URL`、`GRAPH_USER`、`GRAPH_PWD`
- CodeGraph 支持 SQLite-only 降级，HugeGraph 不可用时仍可用大部分功能

### Q3: Demo 端口被占用

```bash
# 杀掉旧进程
pkill -f codegraph_demo_server
pkill -f "http.server 5200"
```

### Q4: jieba 分词导致 BM25 搜不到

BM25 使用精确 token 匹配。查询词应使用完整函数名，例如 `process_order` 而不是 `order`。

### Q5: `.env` 被 git 忽略无法提交

这是预期行为——`.env` 包含敏感信息，仓库中只提交 `.env.example`。  
复现时本地创建 `hugegraph-llm/.env` 即可。

---

## 13. 目录结构速查

```
incubator-hugegraph-ai/
├── hugegraph-llm/
│   ├── .env.example                  # LLM / HugeGraph 配置模板
│   ├── BENCHMARK.md                  # GraphRAG-Bench / MemSim 评测说明
│   ├── src/hugegraph_llm/poc/
│   │   ├── codegraph_hugegraph_mcp.py          # CodeGraph PoC 主文件
│   │   ├── codegraph_hugegraph_mcp_result.json # PoC 运行结果
│   │   ├── benchmark_llm_generation.py         # GraphRAG-Bench 原始评测
│   │   ├── graphrag_bench_p0_improved.py       # GraphRAG-Bench P0 改进版
│   │   ├── benchmark_memsim.py                 # MemSim 中文记忆基准
│   │   └── benchmark_memsim_result.json        # MemSim 评测结果
│   ├── tests/poc/
│   │   └── test_codegraph_hugegraph_mcp.py     # 108 个单元测试
│   ├── benchmark_data/             # 评测数据集
│   │   ├── GraphRAG-Bench/         # GraphRAG-Bench (ICLR'26)
│   │   └── MemSim/                 # MemDaily 中文记忆数据集
│   └── docs/daily_research/
│       └── 2026-06-26-graphrag-bench-evaluation.md
├── demo/
│   ├── codegraph_demo_server.py      # Flask 后端
│   ├── codegraph_demo.html           # 代码图可视化
│   ├── codegraph_build_demo.html     # 图构建流程演示
│   └── codegraph_parsed.json         # 预解析图数据
├── benchmark/
│   ├── run_codegraph_benchmark.py    # 代码图谱覆盖率 + BM25 基准
│   ├── eval_pycg.py                  # PyCG 调用图准确率
│   ├── cto_dashboard.html            # 交互式仪表盘
│   ├── benchmark_result.json         # 实测结果
│   └── pycg_eval_result.json         # PyCG 评估结果
├── demo_data/                        # 演示数据集
├── docs/
│   └── CODEGRAPH_BENCHMARK_AND_COMPETITOR_ANALYSIS.md
└── REPRODUCE.md                      # 本文件
```

---

## 14. 最小可运行验证

```bash
# 1. 环境
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ./hugegraph-llm
pip install flask jieba rank_bm25

# 2. 配置
cp hugegraph-llm/.env.example hugegraph-llm/.env
# 编辑 hugegraph-llm/.env，填入你的 OPENAI_*_API_KEY

# 3. 测试（SQLite-only，无需 HugeGraph）
cd hugegraph-llm
pytest tests/poc/test_codegraph_hugegraph_mcp.py -q

# 4. Demo
python ../demo/codegraph_demo_server.py
# 浏览器打开 http://localhost:5100
```

完成以上步骤即代表复现成功。
