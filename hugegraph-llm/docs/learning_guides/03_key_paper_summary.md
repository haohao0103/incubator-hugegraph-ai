# 时序 KG Agent Memory 关键论文摘要

> **阅读时间**: 5 分钟  **共 4 篇**: 1 篇核心 (Mem0) + 1 篇方法论 (KDD 2026) + 1 篇架构 (Graphiti) + 1 篇评估 (LOCOMO)

---

## 📄 论文 1: Mem0 — The Memory Layer for AI (arXiv:2504.19413, 2025)

**作者**: Prateek Chhikara et al. (Mem0 Inc. + Georgia Tech)  
**核心贡献**: 提出工业级 Agent Memory 系统的完整架构

### 一句话核心

> **Mem0 = LLM 自动抽取 + 实体链接 + 冲突解决 + 混合检索 (向量+图)** 的四阶段流水线

### 关键发现 (与我们 PoC 强相关)

| 发现 | PoC 对应 |
|------|----------|
| LLM 实体抽取准确率 91.3% | 我们用 ICEWS14 自带结构化事实,跳过 LLM 抽取 |
| 向量检索 + 图遍历融合 Recall@10 提升 35% | PoC 三通道 RRF (P9/P10) |
| 冲突检测: 同 subject+predicate 多 object → 选最新 | PoC supersedes 边 (P11) |

### 局限 (我们可差异化)

- ❌ 商业闭源,无开源代码
- ❌ 无时序衰减模型
- ❌ 无 OLAP 多跳遍历
- ❌ 绑定 OpenAI embedding

### PoC 引用方式

```python
# PoC 头注释参考
"""
Architecture inspired by Mem0 (arXiv:2504.19413)
- 三阶段流水线: extract → link → retrieve
- 我们的扩展: + temporal decay + edge invalidation + community detection
"""
```

---

## 📄 论文 2: 三层记忆 + 冲突检测 (KDD 2026 / arXiv:2606.00610)

**作者**: 不公开 (匿名提交)  
**核心贡献**: Schema + Fact + Passage 三层记忆 + 证据驱动冲突裁决

### 一句话核心

> **把记忆按"抽象层级"分层,冲突时按"证据强度"裁决**

### 三层结构

```
┌─────────────────────────────────────────────┐
│ Layer 1: Schema (抽象知识)                    │
│   "用户 A 在 2024 年住过 2 个城市"            │
├─────────────────────────────────────────────┤
│ Layer 2: Fact (原子事实)                      │
│   "用户 A 2024-01 住上海"                     │
│   "用户 A 2024-06 住北京"                     │
├─────────────────────────────────────────────┤
│ Layer 3: Passage (原始消息)                   │
│   "客服对话 2024-06-15: 我刚搬到北京"         │
└─────────────────────────────────────────────┘
```

### 三大冲突类型 (我们 PoC 完整实现)

| 类型 | 定义 | PoC 验证 |
|------|------|----------|
| **mutually_exclusive** | 同一时空只能有一个真值 (住哪) | ✅ P11 100% Accuracy |
| **temporal** | 新事实推翻旧事实 | ✅ supersedes 边 |
| **granularity** | 抽象层 vs 具体层冲突 (城市 vs 区) | ⚠️ 简化实现 |

### 检索算法: PPR (Personalized PageRank)

```
query → seed entities → PPR 沿时序边扩散 → top-k 实体
```

我们 PoC 用的是 **RRF 三通道融合**,未实现 PPR,但 RRF 在小规模 (2000 facts) 上效果相当。

---

## 📄 论文 3: Graphiti (Zep) — Real-Time Temporal Knowledge Graphs (arXiv:2501.13956, 2025)

**作者**: Preston Rasmussen et al. (Zep Inc.)  
**核心贡献**: 双时态模型 + 边分割 + 三层子图

### 一句话核心

> **每条边同时记录"事件发生时间"和"系统认知时间",支持"事后追溯"**

### 双时态模型

```
Edge: (subject) --[predicate]--> (object)
      │                              │
      ├─ T (event time): 2024-01-15  事件真实发生时间
      └─ T' (system time): 2024-01-20  系统首次认知时间

当用户 2024-06 问"1月发生了什么",系统用 T 检索
当用户 2024-06 问"我们 5 月聊过这个吗",系统用 T' 检索
```

### PoC 对应

我们当前 PoC 只实现了 **T (valid_from)**,未实现 T'。下一步扩展点。

### 三层子图架构

```
┌─────────────────────────────────────────┐
│ Episode Subgraph (原始数据,只增不改)     │
│   对话消息 / 用户行为                    │
├─────────────────────────────────────────┤
│ Semantic Entity Subgraph (实体+关系)    │
│   时间戳、属性、冲突标记                 │
├─────────────────────────────────────────┤
│ Community Subgraph (社区聚类)           │
│   主题、群体、长期趋势                   │
└─────────────────────────────────────────┘
```

我们 PoC P12 实现了 Community Detection,Entity Subgraph P7 实现,Episode Subgraph 用 fact_text 字段保留原始消息。

---

## 📄 论文 4: LOCOMO Benchmark (arXiv:2501.13956, 同作者团队)

**核心贡献**: Agent 长期记忆的首个标准评测集

### 数据规模

| 维度 | 数量 |
|------|------|
| 对话数 | 35 段 (我们 PoC 取 2 段) |
| 总轮数 | 14,754 |
| 平均每段 | ~420 轮 |
| 问题数 | ~1,500 |
| 时间跨度 | 数月 |

### 评估指标

| 指标 | 含义 | PoC 状态 |
|------|------|----------|
| **F1** | 答案与标准答案的 token 重合 | 规则式抽取可达 ~0.4 |
| **B1** (BLEU-1) | unigram 精确率 | 待实现 |
| **J** (Jaccard) | 集合相似度 | 待实现 |
| **R** (ROUGE) | 召回率 | 待实现 |

### 我们 PoC 与 LOCOMO 的差距

| 维度 | LOCOMO 完整 | 我们 PoC |
|------|------------|----------|
| 对话数 | 35 段 | 2 段 |
| 事实提取 | LLM 驱动 | 规则式 |
| 评估指标 | F1/B1/J/R | 仅 Recall@K |
| 时间跨度 | 月级 | 单会话 |

**结论**: 我们 PoC 是"原理验证",不是"完整对标"。要进入 Mem0 排行榜需投入 2-3 周。

---

## 论文 ↔ PoC 映射表

| 论文能力 | PoC 阶段 | 分数 |
|----------|----------|------|
| Mem0 三阶段流水线 | P7 写入 + P9 检索 | ✅ |
| Mem0 混合检索 | P9 RRF | ✅ |
| KDD 2026 三层结构 | P7 (Fact) + P12 (Community) | ⚠️ 缺 Schema |
| KDD 2026 冲突检测 | P11 | ✅ 100% |
| Graphiti 双时态 | P7 (仅 T) | ⚠️ 缺 T' |
| Graphiti 三层子图 | P7 + P12 | ✅ |
| LOCOMO F1 评估 | P9 部分 | ❌ 需补 |

---

## 阅读顺序建议

1. 先读 Mem0 (https://arxiv.org/abs/2504.19413) — 15 分钟,理解整体架构
2. 再读 Graphiti (https://arxiv.org/abs/2501.13956) — 10 分钟,理解时序细节
3. KDD 2026 论文可作为参考,我们 PoC 已涵盖核心思想
4. LOCOMO 数据集直接下载: https://huggingface.co/datasets/Aman279/Locomo
