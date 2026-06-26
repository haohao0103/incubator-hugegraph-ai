# CodeGraph 代码知识图谱 — 业界标准数据集、评估指标与竞品分析

> 生成日期: 2026-06-26 | 针对我们已有的 `codegraph_hugegraph_mcp.py` PoC

---

## 目录

1. [业界标准数据集](#1-业界标准数据集)
2. [评估指标体系](#2-评估指标体系)
3. [竞品对比分析](#3-竞品对比分析)
4. [我们 CodeGraph 的定位与差距](#4-我们-codegraph-的定位与差距)
5. [改进路线图](#5-改进路线图)

---

## 1. 业界标准数据集

代码图谱（Code Graph / Code KG）目前没有单一的"标准数据集"，而是依赖多个互补的数据集覆盖不同评估维度：

### 1.1 语义搜索基准

| 数据集 | 规模 | 语言 | 用途 | 核心指标 |
|--------|------|------|------|----------|
| **CodeSearchNet** (GitHub, 2019) | 645万函数, 6语言 | Go/Java/JS/PHP/Python/Ruby | 自然语言→代码检索 | MRR, NDCG@k |
| **CodeSearchNet Challenge** | 99个NL查询 + 4026条人工标注 | 同上 | 真实查询 vs 文档代理 | NDCG (0-3 relevance) |
| **GenCodeSearchNet** (2023) | 同CSN + 跨域变体 | +R语言 | 跨语言/跨域泛化 | MRR + 泛化下降量 |
| **CoSQA** (Microsoft, 2021) | 20,604对 | Python | NL查询↔代码对人工标注 | MRR |

**CodeSearchNet 当前 SOTA (2025)**:
| 模型 | MRR (Overall) | 技术路线 |
|------|---------------|----------|
| CasCode (K=100) | 0.7795 | 双塔召回 + Cross-encoder重排 |
| TOSS | 0.763 | BM25+GraphCodeBERT融合 |
| GraphCodeBERT | 0.713 | 数据流图增强预训练 |
| Transformer (SelfAtt) | 0.701 | 双塔双编码器 |
| ElasticSearch/BM25 | 0.205-0.337 | 纯IR基线 → **语义模型碾压传统IR** |

**对我们的启示**: CodeSearchNet 可直接评测 CodeGraph 的 `BM25CodeSearch` + 语义向量搜索，对标 GraphCodeBERT 的 0.713 MRR。

### 1.2 代码结构分析基准

| 数据集 | 规模 | 用途 | 指标 |
|--------|------|------|------|
| **Python150k** (ETH Zurich) | 15万Python文件, AST+DAT | 代码摘要、方法名预测 | F1, BLEU |
| **CodeXGLUE** (Microsoft, 2021) | 10任务, 多语言 | 代码智能综合评测 | 各任务独立指标 |
| **ManyTypes4Py** | 5,382 Python项目 | 类型推断 | Top-1/Top-5 Accuracy |
| **Typilus** | 600 Python包 | 类型预测 | Precision, Recall |
| **EvoCodeBench** (2025) | 仓库级代码生成 | 代码KG→代码生成 | pass@k, repo-level accuracy |

**对我们的启示**: Python150k 可用来评测 CodeGraph 的 AST 解析准确率（节点类型分类 vs 标注 AST）。

### 1.3 调用图构建基准

| 数据集 | 规模 | 标注方式 | 用途 |
|--------|------|----------|------|
| **PyCG** (Vitaly, 2020) | 1,120 Python包 | 人工标注调用边 | 静态调用图准确性 |
| **JTransform** | 50 Java项目 | 动态追踪标注 | Java调用图 |
| **CGC** (Contextual Graph Constructor) | 多语言 | 混合标注 | 跨语言调用图 |

**对我们的启示**: PyCG 可直接用于评测 CodeGraph 调用边（`CALLS` edge）的 Precision/Recall。

### 1.4 标准数据集优先级（对 CodeGraph）

| 优先级 | 数据集 | 理由 | 新增指标 |
|--------|--------|------|----------|
| **P0** | CodeSearchNet Python 子集 | 业界最公认，Py子集最大（1.15M函数） | MRR, NDCG@10 |
| **P0** | PyCG 调用图 | 直接评测调用边准确性 | Precision, Recall, F1 |
| **P1** | CodeXGLUE CodeSearch | 多语言对比 | MRR (6语言) |
| **P1** | ManyTypes4Py | 类型推断能力 | Top-5 Accuracy |
| **P2** | EvoCodeBench | 仓库级代码生成 | pass@k |

---

## 2. 评估指标体系

### 2.1 六维评估框架

| 维度 | 核心指标 | 测量方法 | CodeGraph 当前状态 |
|------|----------|----------|-------------------|
| **① 结构准确性** | Precision / Recall / F1 (调用边/导入边/继承边) | 与 PyCG 标注对比 | ❌ 未测量 |
| **② 代码搜索** | MRR, NDCG@10, Recall@10 | 在 CodeSearchNet Python 上跑 | ❌ 未测量（当前仅关键词匹配） |
| **③ 查询延迟** | P50/P95/P99 延迟 (ms) | 遍历/搜索/影响分析基准 | ✅ SQLite: 0.26ms avg（非标准benchmark） |
| **④ 覆盖率** | 文件覆盖率、节点覆盖率、边覆盖率 | 在标准项目（requests, flask, django）上测 | ❌ 未系统测量 |
| **⑤ 多跳遍历质量** | 爆炸半径准确率、依赖链完整度 | 与 Sourcegraph/CodeQL 对比 | ❌ 未对比 |
| **⑥ Agent效率提升** | 工具调用减少%、Token减少%、任务完成时间减少% | 用 Claude Code / Codex + CodeGraph vs 不用 | ❌ 未测量 |

### 2.2 对标竞品指标（已有公开数据）

| 指标 | ColbyMcHenry CodeGraph | Sourcegraph | CodeQL | 我们 CodeGraph |
|------|------------------------|-------------|--------|---------------|
| 支持语言 | **22+** | 20+ (LSIF) | 8 | **1** (Python) |
| 调用图准确率 | 86.7%-100% (per lang) | 工业级 (未公开 P/R) | 未公开 | **P=1.00, R=1.00** (静态直接调用, 50合成例) |
| 语义搜索 MRR | ❌ 未公开 | ❌ 无NL搜索 | ❌ 无 | ❌ 未测 (BM25 MRR=0.433) |
| Agent工具调用减少 | **58%** (7项目) | ~40% (Cody) | N/A | **待实测** (MCP 规划中) |
| Agent Token减少 | **23-64%** | 未公开 | N/A | 待实测 |
| 文件监听同步 | ✅ 原生OS事件 | ✅ 后台索引 | N/A | ❌ 无 |
| MCP集成 | ✅ 原生 | ✅ (Cody 2024-11) | ❌ | ❌ 未做 (设计文档已有) |
| 图后端 | SQLite | SQLite Bundle | 专有DB | SQLite + HugeGraph |
| 框架感知 | **14框架** | 语言服务器 | 无 | ❌ 无 |
| OLAP分析 | ❌ | ❌ | ❌ | **✅ Vermeer** |

---

## 3. 竞品对比分析

### 3.1 竞品全景图

```
                        代码理解深度
                         ▲
                         │  CodeQL (安全漏洞)
                         │  Joern/CPG (模式挖掘)
                         │
            GraphCodeBERT (数据流图)
                         │
        Code-Graph-RAG (KG+RAG)
                         │
  ──────────────────────────────────────────► 工程化成熟度
    学术         开源工具          企业产品
                         │
          我们的CodeGraph      Sourcegraph
                         │
              ColbyMcHenry    GitHub Copilot
              CodeGraph       Spaces
                         │
              Neo4j CodeKG   Tabnine
```

### 3.2 逐产品分析

#### 3.2.1 ColbyMcHenry CodeGraph（最直接对标）

| 维度 | 得分 | 说明 |
|------|------|------|
| 语言支持 | ⭐⭐⭐⭐⭐ | 22+语言，多语言交叉引用（RN桥接！） |
| 框架感知 | ⭐⭐⭐⭐⭐ | 14框架路由识别 |
| 文件监听 | ⭐⭐⭐⭐⭐ | 原生OS事件，防抖自动同步 |
| MCP集成 | ⭐⭐⭐⭐⭐ | 开箱即用，零配置 |
| 语义搜索 | ⭐⭐ | 仅FTS5全文搜索，无向量语义 |
| 图后端 | ⭐⭐ | 纯SQLite，无图遍历优化 |
| Agent集成 | ⭐⭐⭐⭐⭐ | 已在Claude Code/Cursor等验证 |
| 开源 | ✅ MIT | 566 commits, v1.1.1 (2026-06) |
| **定位** | **AI Agent场景的代码上下文提供者** | 轻量、快速、本地优先 |

**核心差异**: ColbyMcHenry胜在工程化（22语言+14框架+原生OS同步+MCP开箱即用），但在图分析深度（无多跳遍历优化、无OLAP）和语义搜索（无向量）上不如我们。

#### 3.2.2 Code-Graph-RAG (Iluvata, 2025)

| 维度 | 得分 | 说明 |
|------|------|------|
| 语言支持 | ⭐⭐⭐⭐ | Tree-sitter多语言 |
| 图存储 | ⭐⭐⭐⭐ | Memgraph（专业图数据库，支持Cypher） |
| RAG集成 | ⭐⭐⭐⭐ | 自然语言查询代码 → 图遍历 → LLM回答 |
| 代码编辑 | ⭐⭐⭐ | 支持AI驱动的代码编辑 |
| 生态集成 | ⭐⭐ | PyPI包，无MCP |
| **定位** | **GraphRAG for Code** — 让开发者用自然语言理解/编辑代码 |

**核心差异**: Code-Graph-RAG 的路线是"对代码库的GraphRAG"，强调NL→Code的理解链路。但工程化不如ColbyMcHenry，生态不如我们（我们有HugeGraph集群）。

#### 3.2.3 Sourcegraph

| 维度 | 得分 | 说明 |
|------|------|------|
| 代码导航 | ⭐⭐⭐⭐⭐ | SCIP/LSIF标准，工业级 |
| 跨仓库 | ⭐⭐⭐⭐⭐ | 跨仓库依赖分析 |
| AI集成 | ⭐⭐⭐⭐ | Cody AI + Deep Search |
| 查询延迟 | ⭐⭐⭐⭐⭐ | 比GitHub搜索快2x |
| 历史分析 | ⭐ | 不分析commit/PR演进 |
| 部署 | ⭐⭐⭐ | 需服务端部署 |
| **定位** | **企业级代码搜索与导航平台** | 工业标准 |

**核心差异**: Sourcegraph 是现有代码搜索的王者，但不做"知识图谱"——它的图是符号-引用结构图，不做语义理解、不做RAG。

#### 3.2.4 Joern / Code Property Graph (ShiftLeft)

| 维度 | 得分 | 说明 |
|------|------|------|
| 图模型 | ⭐⭐⭐⭐⭐ | AST+CFG+PDG+DDG融合的CPG |
| 查询语言 | ⭐⭐⭐⭐⭐ | 自定义DSL（类Cypher） |
| 安全分析 | ⭐⭐⭐⭐⭐ | 漏洞模式挖掘 |
| 语言支持 | ⭐⭐⭐ | C/C++/Java/JS/Python/Ruby |
| 易用性 | ⭐⭐ | 需要Scala基础 |
| **定位** | **安全漏洞挖掘的代码属性图** | 学术+安全工业 |

**核心差异**: Joern的CPG模型（AST+控制流+数据流+调用图融合）是代码图的理论天花板，但目标是安全而非AI Agent场景。

#### 3.2.5 GraphCodeBERT (Microsoft, 2021)

| 维度 | 得分 | 说明 |
|------|------|------|
| 语义理解 | ⭐⭐⭐⭐⭐ | 数据流图增强预训练 |
| CodeSearchNet MRR | 0.713 | SOTA基线 |
| 工程可用 | ⭐⭐⭐ | 学术模型，需自行部署推理 |
| 实时性 | ⭐ | 离线预训练，无增量更新 |
| **定位** | **代码表征学习预训练模型** | 学术SOTA |

**核心差异**: GraphCodeBERT 做的是"学一个更好的code embedding"，不做图数据库、不做Agent集成。

#### 3.2.6 Neo4j Codebase Knowledge Graph

| 维度 | 得分 | 说明 |
|------|------|------|
| 图存储 | ⭐⭐⭐⭐⭐ | Neo4j原生图数据库 |
| 可视化 | ⭐⭐⭐⭐ | Neo4j Bloom自带的图可视化 |
| 调用图 | ⭐⭐⭐ | 包/类/方法级 |
| 增量更新 | ⭐⭐ | 无自动监听 |
| AI集成 | ⭐⭐ | 无LLM/RAG集成 |
| **定位** | **用图数据库建模代码结构** | 概念验证级 |

---

## 4. 我们 CodeGraph 的定位与差距

### 4.1 定位细分

```
ColbyMcHenry CodeGraph ───────► AI Agent 上下文 (轻量/快速)
Code-Graph-RAG ───────────────► GraphRAG for Code (NL理解)
Sourcegraph ──────────────────► 企业级代码搜索 (工业标准)
Joern/CPG ────────────────────► 安全漏洞挖掘 (深度分析)
我们 CodeGraph ───────────────► 代码知识图谱底座 (HG集群+多场景)
                                  ↓
                         AI Agent · 风控 · 供应链 · GraphRAG
```

**我们的独特定位**: 不只是"另一个代码图工具"——CodeGraph 是 **HugeGraph 代码知识图谱底座**，其代码图谱可以作为 GraphRAG 的知识来源、AI Agent 的 Long-term Memory、以及风控代码审计的基础设施。

### 4.2 当前差距矩阵

| 维度 | ColbyMcHenry | Code-Graph-RAG | Sourcegraph | Joern/CPG | 我们 | 差距 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| 语言支持 | 22+ | 10+ | 20+ | 6 | **1** | 🔴 最大差距 |
| 框架感知 | 14框架 | 无 | 语言服务器 | 无 | **0** | 🔴 需补 |
| 文件监听同步 | ✅ | ❌ | ✅ | N/A | **❌** | 🟡 非核心但实用 |
| MCP Server | ✅ | ❌ | ✅ | ✅(社区) | **❌** | 🟡 已有设计文档 |
| 语义搜索 | FTS5 | 向量 | 语言服务器 | 无 | **FTS5** | 🟡 加向量即可超 |
| 图存储 | SQLite | Memgraph | SQLite Bundle | OverflowDB | **SQLite+HG** | 🟢 唯一分布式图存储 |
| 多跳遍历 | 2-3 hop | Cypher | SCIP | CPG查询 | **任意深度(HG)** | 🟢 核心优势 |
| Agent验证 | ✅ 7项目 | ❌ | ✅ | N/A | **待实测** | 🔴 缺失CTO信任 |
| 基准评测 | ✅ per-lang | ❌ | ❌ | 部分 | **部分(合成P/R)** | 🟡 需真实项目验证 |
| OLAP | ❌ | ❌ | ❌ | 部分 | **✅ Vermeer** | 🟢 唯一OLAP |

### 4.3 我们的核心竞争力（与竞品差异化）

1. **HugeGraph 分布式图后端**: 所有竞品都用SQLite/单机图DB → 我们可以做百亿级代码图谱的OLAP分析
2. **GraphRAG 融合**: 代码图 + 文档图 + 知识图谱融合 → 竞品只做代码图
3. **多图隔离**: 供应链KG + 风控KG + 代码KG → 同一集群多图空间 → 竞品做不到
4. **Vermeer OLAP**: 大规模图遍历/社区检测/PageRank → 竞品无此能力

---

## 5. 改进路线图

### 5.1 短期（2周，CTO Demo就绪）

| 优先级 | 任务 | 预期指标 | 对标 |
|--------|------|----------|------|
| **P0** | 下载 CodeSearchNet Python 子集 + 跑 MRR/NDCG@10 | MRR ≥ 0.30 (BM25基线) | CodeSearchNet |
| **P0** | 下载 PyCG 标注 + 测调用边 Precision/Recall | P ≥ 0.85, R ≥ 0.75 | PyCG benchmark |
| **P0** | 在 3 个标准项目（requests, flask, django）上跑 CodeGraph → 报告覆盖率 | 与 ColbyMcHenry 对标 | ColbyMcHenry |
| **P0** | 用 Claude Code + CodeGraph MCP vs 无 → 测 Agent 效率 | 工具调用减少40%+ | ColbyMcHenry (58%) |
| **P1** | 增加 Flask/FastAPI 框架感知（route识别） | 框架路由覆盖率80%+ | ColbyMcHenry (14框架) |
| **P1** | 集成向量语义搜索（FAISS/Milvus） → CodeSearchNet MRR | MRR ≥ 0.60 | GraphCodeBERT (0.71) |

### 5.2 中期（1个月，产品化）

| 优先级 | 任务 |
|--------|------|
| **P1** | 增加 Java 语言支持（基于 Tree-sitter）→ 对标 JDK 标准库 |
| **P1** | MCP Server 实现（基于 `docs/HUGEGRAPH_MCP_SERVER_SPEC.md`） |
| **P1** | 文件监听 + 增量更新 |
| **P2** | TypeScript/JavaScript 支持 → 覆盖前端代码库 |
| **P2** | CodeXGLUE 多语言评测 |

### 5.3 长期（3个月+，竞争壁垒）

| 优先级 | 任务 |
|--------|------|
| **P2** | 代码图 + 文档图 + Issue图融合（GraphRAG） |
| **P2** | HugeGraph OLAP 批量代码分析（安全漏洞模式挖掘） |
| **P2** | 跨仓库依赖分析（对标 Sourcegraph） |

---

## 附录：评测数据集获取

### CodeSearchNet Python 子集
```bash
# 约 1.15M Python 函数 + docstring 对
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
# 或从 HuggingFace
pip install datasets
python -c "from datasets import load_dataset; load_dataset('code_search_net', 'python')"
```

### PyCG 调用图标注
```bash
git clone https://github.com/vitsalis/PyCG.git
# 1120个Python包的标注调用图
```

### ColbyMcHenry 对标项目
- Python: `psf/requests`
- 我们在 Django 上已有结果 → 可对标

---

*文档版本: v1.0 | 下次更新: 完成P0评测后*
