# GraphRAG 源码级对标分析

> 日期: 2026-06-30 | 对标对象: LightRAG, MS-GraphRAG, HippoRAG2, Fast-GraphRAG
> 目标: 通用GraphRAG框架对标, 源码级差距识别与补齐方案

---

## 1. 检索架构源码级对比

### 1.1 核心数据流对比

| 框架 | 检索通道 | 向量→图关系 | 图遍历方式 | BM25? | 融合机制 |
|------|---------|-----------|-----------|-------|---------|
| **HugeGraph (我们)** | **3通道并行** | **独立并行, RRF融合** | k_neighbor BFS (depth=2) | ✅ BM25Simple | RRF `Σ 1/(K+rank)` |
| **LightRAG** | **2通道串行** | Vector种子→1-hop图扩展 | igraph邻居遍历 | ❌ | Context拼接(LLM融合) |
| **MS-GraphRAG** | **1通道串行** | Vector→实体→1-hop邻居 | 实体邻居+社区 | ❌ | Token budget比例分配 |
| **HippoRAG2** | **2通道串行** | Vector种子→PPR扩散 | igraph PPR (PRPACK) | ❌ | Vector→PPR→LLM Rerank |
| **Fast-GraphRAG** | **2通道串行** | Vector分数→PPR种子 | igraph PPR | ❌ | 稀疏矩阵级联传播 |

### 1.2 关键发现: BM25是独有设计

**4个竞品全部没有BM25独立通道**。这不是巧合,而是设计哲学选择:

- LightRAG: 用LLM提取hl_keywords/ll_keywords→向量匹配代替关键词搜索
- MS-GraphRAG: 纯向量实体定位+社区摘要, 无关键词搜索
- HippoRAG2: Vector+PPR, 无BM25
- Fast-GraphRAG: Vector→PPR→Entity→Relation→Chunk级联, 无BM25

**我们的BM25评估**: P0-v5中 BM25永远返回10条(avg_bm25_hits=10.0), 但:
- BM25的检索质量取决于tokenize精度 — 我们用的是 `re.findall(r'[a-zA-Z]{2,}')`, 对长词/短语/专有名词很差
- BM25没有和向量检索做overlap分析 — 不知道BM25独立召回了多少Vector没找到的chunk
- BM25在RRF中和Vector等权(1/(K+rank)) — 没有通道权重调节

---

## 2. 源码级关键模块拆解

### 2.1 我们的代码 (poc_graphrag_bench_p0_v5.py)

```python
# 三通道 RRF fusion (行 592-605)
def rrf_fusion(vector_results, bm25_results, graph_boost_ids, k=60):
    scores = {}
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    # Graph boost: 1.5x weight (不是独立通道, 而是boost已有chunk)
    for doc_id in graph_boost_ids:
        if doc_id in scores:
            scores[doc_id] *= 1.5
        else:
            scores[doc_id] = 1.5 / (k + 1)
```

**问题1**: Graph通道不是真正的独立检索通道 — 它只是对Vector/BM25已召回的chunk做boost, 不是独立召回新chunk

**问题2**: BM25和Vector在RRF中完全等权, 没有通道权重

**问题3**: Entity name matching精度极差 (Medical graph_hits=0)

### 2.2 LightRAG 源码关键路径

**检索入口**: `operate.py` → `aquery()`/`query()`

```python
# 双层检索: LLM提取关键词 → 向量匹配 → 图1-hop扩展
# 低级检索: ll_keywords → entities_vdb → 实体1-hop邻居
# 高级检索: hl_keywords → relationships_vdb → 关系1-hop邻居
# 融合: Round-Robin交替拼接(不是RRF), 最终LLM综合
```

**关键优势**:
- 3个独立向量库: `entities_vdb`, `relationships_vdb`, `chunks_vdb`
- 实体和关系分别索引, 检索更精准
- 双层关键词分解(LLM从query提取hl/ll_keywords)

**我们缺失**: 没有实体/关系分离向量索引, 所有东西都在一个FAISS中

### 2.3 Fast-GraphRAG 源码关键路径

**检索入口**: `_state_manager.py` → `get_context()` (行 185-261)

```python
# Vector Scoring → PPR Diffusion → Entity→Relation→Chunk 级联传播
# Step 1: Embedding encode(named+generic+query) → HNSWLib score_all
# Step 2: vdb_scores → PPR reset_prob → igraph.personalized_pagerank()
# Step 3: entity_scores.dot(e2r) → relation_scores
# Step 4: relation_scores.dot(r2c) → chunk_scores
# Ranking: Threshold(0.005) → TopK(64 relations) → TopK(8 chunks)
```

**关键优势**:
- PPR扩散替代BFS遍历 — 从种子节点向全图扩散分数, 发现远距离关联
- 稀疏矩阵全链路(csr_matrix) — 大规模图高效计算
- Entity identity edges(相似度>0.9→"is"边) — 实体去重+PPR互传分数
- Entity→Relation→Chunk三级传播 — 分数沿图结构自然流动

**我们缺失**: PPR已有(`ppr_retriever.py`), 但级联传播和identity edges没有

### 2.4 MS-GraphRAG 源码关键路径

**检索入口**: 
- Local Search: `local_search/mix_context_builder.py`
- Global Search: `global_search/community_context.py`

```python
# Local Search: 
#   Vector定位实体 → 实体1-hop邻居 → 相关社区 → text_units
#   Token budget: 按10%实体/10%关系/30%社区/50%文本比例分配
#
# Global Search:
#   Leiden社区分层 → Map-Reduce over community reports
#   Community reports是LLM在索引阶段预生成的摘要
#
# DRIFT Search:
#   HyDE查询扩展 → 多轮Local Search迭代
```

**关键优势**:
- **社区摘要**: Leiden聚类 + LLM生成社区报告 (索引阶段预计算)
- **Global Search**: 处理"整体性问题"(如"数据集的5个主题")
- **Token budget**: 比例分配而非简单TopK截断
- **DRIFT**: HyDE + 多轮迭代搜索

**我们缺失**: 社区检测+摘要(已有`community_detector.py`但未集成), Global Search, Token budget

### 2.5 HippoRAG2 源码关键路径

**检索入口**: `main.py` → `query()`

```python
# Vector(query→fact/passage) → LLM Rerank(DSPy) → PPR(igraph PRPACK)
# OpenIE自动三元组提取 (not LLM NER)
# Synonymy edges (embedding相似度>threshold → 同义边)
# 查询编码: query_to_fact + query_to_passage (双策略)
```

**关键优势**:
- OpenIE提取(更鲁棒, 不依赖LLM prompt)
- Synonymy edges(实体融合在图结构中)
- LLM Rerank(DSPy框架, 可训练的过滤策略)
- 双编码策略(fact+passage)

**我们缺失**: OpenIE提取, Synonymy edges, LLM Rerank

---

## 3. BM25通道战略决策

### 3.1 竞品为什么不用BM25

| 原因 | 说明 |
|------|------|
| **图+向量已足够** | PPR/k_neighbor从图结构中召回远距离关联, Vector召回语义相似, 两者互补覆盖了BM25能召回的大部分内容 |
| **BM25精度依赖tokenizer** | BM25对英文简单tokenize可以, 但对中文/多语言/专有名词需要专门tokenizer, 复杂度高 |
| **RRF权重难调** | BM25和Vector等权融合, 在大多数场景Vector质量更高, BM25引入噪声 |
| **图数据库不支持全文** | HugeGraph/Neo4j的全文检索需要外部引擎, 增加了架构复杂度 |

### 3.2 BM25要不要保留?

**结论: 保留但降级为可选插件, 不作为核心架构通道**

理由:
1. **4个竞品全没有BM25** — 这是行业共识, 不需要独立BM25通道
2. **我们的BM25质量差** — `re.findall(r'[a-zA-Z]{2,}')` tokenize太粗糙, 医学专有名词匹配率极低
3. **架构简化** — 核心架构应该是 **Vector→PPR→级联传播**, 这才是对标竞品的主流设计
4. **BM25仍有价值** — 对精确关键词匹配(代码标识符、药物名、缩写)有Vector无法替代的能力
5. **作为可选插件** — 类似Fast-GraphRAG的`backend_factory`, BM25可以配置为`fulltext_backend=bm25|none`

**调整方案**: 
- 核心架构改为 **Vector→PPR(Graph)→级联传播** (对标Fast-GraphRAG/HippoRAG2)
- BM25作为 **可选增强插件** (配置项, 默认关闭)
- 融合从三通道RRF改为 **Vector→PPR级联 + 可选BM25 boost**

---

## 4. 源码级差距清单与补齐方案

### P0级差距 (必须补齐, 否则无法对标)

| # | 差距模块 | 来源竞品 | 当前状态 | 补齐方案 |
|---|---------|---------|---------|---------|
| **P0-1** | **PPR扩散检索** | HippoRAG2/Fast-GraphRAG | ✅ 已有`ppr_retriever.py` (push-style) | 集成到主检索管道: Vector→PPR→RRF, 替代k_neighbor BFS |
| **P0-2** | **Entity→Relation→Chunk级联传播** | Fast-GraphRAG | ❌ 无 | 实现稀疏矩阵传播: entity_scores.dot(e2r)→relation→chunk, 借鉴Fast-GraphRAG `_state_manager.py` 行296-309 |
| **P0-3** | **Entity identity/dedup edges** | Fast-GraphRAG/HippoRAG2 | ❌ 无 | 实体embedding相似度>0.9→自动创建"same_as"边, 借鉴Fast-GraphRAG `_state_manager.py` 行131-174 |
| **P0-4** | **双层关键词分解** | LightRAG | ❌ 单层heuristic提取 | LLM从query提取hl_keywords/ll_keywords, 借鉴LightRAG `operate.py` |
| **P0-5** | **社区摘要** | MS-GraphRAG | 部分(`community_detector.py`有但未集成) | Leiden聚类+LLM生成社区报告+Global Search模式 |

### P1级差距 (重要增强)

| # | 差距模块 | 来源竞品 | 当前状态 | 补齐方案 |
|---|---------|---------|---------|---------|
| **P1-1** | **HNSWLib向量存储** | Fast-GraphRAG | FAISS | HNSWLib更高效(增量更新+并发搜索), 借鉴Fast-GraphRAG `_vdb_hnswlib.py` |
| **P1-2** | **RankingPolicy(Threshold/TopK/Elbow)** | Fast-GraphRAG | 简单TopK截断 | 实现三级过滤: entity threshold→relation TopK→chunk TopK, 借鉴 `_policies/_ranking.py` |
| **P1-3** | **DRIFT迭代搜索** | MS-GraphRAG | ❌ 无 | HyDE查询扩展+多轮Local Search, 借鉴MS-GraphRAG `drift_search.py` |
| **P1-4** | **Gleaning多轮提取** | Fast-GraphRAG | ❌ 单轮提取 | 最多N轮LLM继续提取实体/关系, 借鉴Fast-GraphRAG `_information_extraction.py` 行76-119 |
| **P1-5** | **Token Budget分配** | MS-GraphRAG | ❌ 简单TopK截断 | 比例分配: 10%实体+10%关系+30%社区+50%文本, 借鉴MS-GraphRAG `context_builder.py` |

### P2级差距 (差异化优势)

| # | 差距模块 | 来源竞品 | 当前状态 | 说明 |
|---|---------|---------|---------|------|
| **P2-1** | **Gremlin/Text2Gremlin查询** | 无竞品 | ✅ 已有 | 独有优势, 结构化图查询能力 |
| **P2-2** | **OLAP多跳Vermeer** | 无竞品 | ✅ 已有 | 独有优势, 大规模图计算 |
| **P2-3** | **60亿点边运维验证** | 无竞品 | ✅ 已有 | 独有优势, 3年生产验证 |
| **P2-4** | **BM25可选插件** | 无竞品有BM25 | ✅ 已有但粗糙 | 独有设计, 精确关键词匹配 |

---

## 5. 补齐路线图

### Week 1: 核心检索管道重构 (P0-1 + P0-2 + P0-3)

**目标**: 将三通道RRF改为 **Vector→PPR级联传播**

```
当前: Vector ──┐
       BM25  ──┤→ RRF融合 → TopK → LLM
       Graph ──┘

目标: Vector种子 → PPR扩散 → Entity分数 → Relation分数 → Chunk分数
       └── identity edges(去重+互传) ──┘
       └── [可选] BM25 boost ──────────┘
```

**具体实现**:
1. 将`ppr_retriever.py`集成到主检索管道 (已有Push-style PPR, 需对接HugeGraph k_neighbor获取子图)
2. 构建 e2r/r2c 稀疏映射矩阵 (借鉴Fast-GraphRAG `_state_manager.py` 行373-385)
3. 实现entity identity edges (embedding sim>0.9 → "same_as"边)

### Week 2: 双层关键词+社区摘要 (P0-4 + P0-5)

**目标**: LightRAG级别的关键词提取 + MS-GraphRAG级别的社区摘要

1. 实现query→{hl_keywords, ll_keywords}分解 (借鉴LightRAG)
2. 实体/关系分离向量索引 (借鉴LightRAG的3 VDB设计)
3. Leiden社区聚类 + LLM社区报告生成 (已有`community_detector.py`)
4. Global Search模式 (Map-Reduce over community reports)

### Week 3: RankingPolicy+DRIFT+评估拉齐 (P1级)

1. RankingPolicy三级过滤 (Threshold→TopK→Elbow)
2. DRIFT迭代搜索 (HyDE + 多轮)
3. 重新跑GraphRAG-Bench评测, 对标竞品ACC
4. BM25降级为可选插件 (默认关闭)

---

## 6. 竞品源码位置索引

| 竞品 | 本地路径 | 关键文件 |
|------|---------|---------|
| LightRAG | `/Users/mac/Desktop/apache-code/hugegraph-dev/lightrag/` | `operate.py`(检索入口), `lightrag.py`(KG构建), `kg/shared_storage.py`(3 VDB) |
| MS-GraphRAG | `/Users/mac/Desktop/apache-code/hugegraph-dev/ms-graphrag/` | `query/local_search/`, `query/global_search/`, `query/drift_search/`, `index/create_final_entities.py` |
| HippoRAG2 | `/Users/mac/Desktop/apache-code/hugegraph-dev/hipporag2/` | `main.py`(检索入口), `models/`, `openie/`(三元组提取) |
| Fast-GraphRAG | `/Users/mac/Desktop/apache-code/hugegraph-dev/fast-graphrag/` | `_state_manager.py`(检索核心), `_gdb_igraph.py`(PPR), `_vdb_hnswlib.py`(向量), `_policies/_ranking.py`(Ranking) |
| HugeGraph | `/Users/mac/Desktop/apache-code/hugegraph-dev/incubator-hugegraph-ai/hugegraph-llm/` | `ppr_retriever.py`(PPR), `rrf_fusion.py`(RRF), `community_detector.py`(社区) |

---

## 7. 总结

**BM25决策**: 保留为可选插件, 不作为核心架构通道。4个竞品全无BM25, 行业共识是Vector+图级联足够覆盖关键词检索场景。核心架构改为 **Vector→PPR→级联传播**。

**最大差距**: 不是BM25, 而是:
1. PPR级联传播(替代当前粗陋的k_neighbor BFS+1.5x boost)
2. Entity identity edges(实体去重+分数互传)
3. 双层关键词分解(hl/ll)
4. 社区摘要+Global Search

**我们的独有优势**: Gremlin查询, OLAP Vermeer, 60亿点边运维验证 — 这些竞品全没有。
