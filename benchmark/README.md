# CodeGraph Benchmark Suite

本目录包含 CodeGraph 代码知识图谱的全套评测脚本和 CTO 仪表盘，用于量化评估代码图谱构建质量、搜索效果、结构查询性能和调用图准确性。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `run_codegraph_benchmark.py` | 代码图谱覆盖率 + BM25 搜索 + 结构查询性能评测 |
| `eval_pycg.py` | 使用 PyCG 基准评测调用图准确率 |
| `cto_dashboard.html` | 交互式 CTO 仪表盘（融合所有评测数据） |
| `benchmark_result.json` | `run_codegraph_benchmark.py` 的实测结果 |
| `pycg_eval_result.json` | `eval_pycg.py` 的实测结果 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install requests flask django rank_bm25 jieba
```

### 2. 运行代码图谱基准

```bash
python benchmark/run_codegraph_benchmark.py
```

输出文件：

- `benchmark/benchmark_result.json`

评测维度：

| 维度 | 说明 |
|------|------|
| 节点/边统计 | 解析 requests / flask / django 的 function / class / module 数量 |
| BM25 搜索 | 自然语言查询的 MRR、NDCG@10、Recall@10 |
| 结构查询 | 1-hop 邻接、2-hop 遍历、hub 分析、影响范围分析延迟 |
| 调用图拓扑 | 内部调用率、跨模块调用率、Top hubs |

### 3. 运行 PyCG 调用图准确率评估

```bash
python benchmark/eval_pycg.py
```

输出文件：

- `benchmark/pycg_eval_result.json`

> `benchmark_data/PyCG/` 已包含在仓库中，无需额外下载。

### 4. 启动 CTO 仪表盘

```bash
cd benchmark
python3 -m http.server 5200 --directory .
```

访问：http://localhost:5200/cto_dashboard.html

---

## 评测指标说明

### BM25 搜索质量

- **MRR** (Mean Reciprocal Rank)：第一个相关结果的倒数排名的平均值
- **NDCG@10**：归一化折损累计收益，衡量排序质量
- **Recall@10**：前 10 个结果中命中目标的比例

### 调用图准确率（PyCG）

- **Precision**：预测的调用边中正确的比例
- **Recall**：真实调用边中被预测出的比例
- **F1**：Precision 和 Recall 的调和平均

### 结构查询延迟

- 1-hop 邻接查询
- 2-hop 遍历
- Hub 检测
- 影响范围分析（爆炸半径）

---

## 复现说明

完整复现步骤请参考仓库根目录的 `REPRODUCE.md`。
