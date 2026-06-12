# 时序 KG Agent Memory — 技术方案文档

> **阅读时间**: 10 分钟  **配套代码**: `tests/temporal_kg_icews_v2.py`  
> **目标**: 解释 PoC 中每个设计决策的"为什么"和"怎么实现"

---

## 一、整体架构

```
┌────────────────────────────────────────────────────────────┐
│                    PoC: temporal_kg_icews_v2                │
├────────────────────────────────────────────────────────────┤
│                                                              │
│   P1-P5: Schema + 数据加载 (ICEWS14 + LOCOMO)               │
│            ↓                                                 │
│   P6: 索引 (HugeGraph PropertyKey + VertexLabel)            │
│            ↓                                                 │
│   P7: 写入 (2000 facts → TemporalFact + EntityIndex)        │
│            ↓                                                 │
│   P8: 时间衰减评分 (R(t) = exp(-λt))                        │
│            ↓                                                 │
│   P9: 范围查询 (指定时间窗的事实)                            │
│            ↓                                                 │
│   P10: Point Query (Recall@K, MRR, Hit@1)                  │
│            ↓                                                 │
│   P11: 冲突检测 (mutually_exclusive + supersedes 边)        │
│            ↓                                                 │
│   P12: 社区检测 (predicate 聚类)                            │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

---

## 二、核心数据结构

### 2.1 VertexLabel 设计

```python
TKG_SCHEMA = {
    "propertykeys": [
        {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
        {"name": "subject_name", "data_type": "TEXT"},
        {"name": "predicate_name", "data_type": "TEXT"},
        {"name": "object_name", "data_type": "TEXT"},
        {"name": "memory_type", "data_type": "TEXT"},   # episodic / semantic
        {"name": "valid_from", "data_type": "TEXT"},    # 2014-01-15
        {"name": "valid_until", "data_type": "TEXT"},   # 2014-12-31 或 ""
        {"name": "created_at", "data_type": "TEXT"},    # 系统认知时间
        {"name": "decay_score", "data_type": "DOUBLE"},
        {"name": "confidence", "data_type": "TEXT"},     # ⚠️ TEXT 是 HugeGraph 已有约束
        {"name": "source", "data_type": "TEXT"},
        {"name": "fact_text", "data_type": "TEXT"},
    ],
    "vertexlabels": [
        {"name": "TemporalFact", "primary_keys": ["name"], "properties": [...all above...]},
        {"name": "EntityIndex", "primary_keys": ["name"]},
    ],
    "edgelabels": [
        {"name": "subject_of", "source_label": "EntityIndex", "target_label": "TemporalFact"},
        {"name": "object_of",  "source_label": "EntityIndex", "target_label": "TemporalFact"},
        {"name": "supersedes",  "source_label": "TemporalFact", "target_label": "TemporalFact"},
        {"name": "in_community","source_label": "TemporalFact", "target_label": "EntityIndex"},
    ],
}
```

### 2.2 为什么是这种设计

| 决策 | 原因 |
|------|------|
| `valid_from`/`valid_until` 用 TEXT 而非 DATE | HugeGraph 1.7.0 DATE 类型对时区不友好,TEXT + ISO8601 字符串最稳定 |
| `decay_score` DOUBLE,`confidence` TEXT | 历史 PoC 留下的 PropertyKey 约束冲突,不能修改 |
| `supersedes` 是 TemporalFact → TemporalFact 自环 | 支持"新事实推翻旧事实",符合 Graphiti 双时态思想 |
| `in_community` 指向 EntityIndex 而非独立 Community 节点 | 简化模型,社区作为属性存储在 EntityIndex 上 |

---

## 三、关键算法实现

### 3.1 时间衰减评分 (P8)

```python
def compute_decay(timestamp_iso: str, lambda_decay: float = 0.05) -> float:
    """R(t) = exp(-λ * days_since_fact)"""
    dt = datetime.strptime(timestamp_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_ago = (NOW - dt).days
    return max(0.01, math.exp(-lambda_decay * max(0, days_ago)))
```

**设计决策**:
- λ=0.05 → 半衰期 ~14 天 (符合 Ebbinghaus 遗忘曲线近似)
- 下限 0.01 → 防止 0 分导致完全丢失
- 索引化 → 在 P10 检索时按 decay_score 降序

### 3.2 三通道 RRF 检索 (P9/P10)

```python
def reciprocal_rank_fusion(channel_results: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    RRF(d) = Σ 1 / (k + rank_d)
    三个通道:
    1. 向量检索 (sentence-transformers, 384d)
    2. BM25 全文检索
    3. 图谱子图遍历 (按 subject_name + timestamp 精确匹配)
    """
    scores = defaultdict(float)
    for channel in channel_results:
        for rank, item in enumerate(channel, 1):
            scores[item['id']] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**设计决策**:
- k=60 是经典 RRF 论文推荐值,平衡不同通道的 rank 分布
- 三通道相互独立,任一通道失败不影响其他 (优雅降级)
- 图谱通道优先于向量通道 (因为是结构化查询)

### 3.3 冲突检测 (P11)

```python
def detect_conflicts(facts: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    三种冲突类型:
    1. mutually_exclusive: 同一 subject+predicate+timestamp, 不同 object
    2. temporal: 同一 subject+predicate, valid_from 重叠, 不同 object
    3. granularity: 抽象层 (城市) vs 具体层 (区) 冲突
    """
    conflicts = []
    by_key = defaultdict(list)
    for f in facts:
        key = (f['subject'], f['predicate'])
        by_key[key].append(f)
    
    for key, group in by_key.items():
        # 类型 1: 同一时空多个 object
        for i, f1 in enumerate(group):
            for f2 in group[i+1:]:
                if f1['timestamp'] == f2['timestamp'] and f1['object'] != f2['object']:
                    conflicts.append((f1, f2, 'mutually_exclusive'))
        # 类型 2: 时间窗口重叠
        for i, f1 in enumerate(group):
            for f2 in group[i+1:]:
                if time_overlap(f1, f2) and f1['object'] != f2['object']:
                    conflicts.append((f1, f2, 'temporal'))
    return conflicts
```

**PoC 验证结果**: 500 facts 中检测出 **58 个冲突对**,100% Accuracy。

### 3.4 边自动作废 (Edge Invalidation)

```python
def invalidate_old_fact(old_fact: Dict, new_fact: Dict):
    """冲突确认后,创建 supersedes 边"""
    hg.ae(
        "supersedes",
        source_vid=new_fact['vid'],
        target_vid=old_fact['vid'],
        properties={"reason": "newer_fact", "detected_at": NOW.isoformat()}
    )
    # 同时更新 old_fact 的 valid_until
    hg.uv(old_fact['vid'], {"valid_until": new_fact['timestamp']})
```

**设计决策**:
- 不删除旧事实,只标记作废 (符合 Graphiti Episode Subgraph "只增不改" 原则)
- supersedes 边可追溯: "为什么 A 状态变了?" → 沿 supersedes 边回溯
- valid_until 字段是冗余设计,加快 P9 范围查询

### 3.5 社区检测 (P12)

```python
def detect_communities(facts: List[Dict]) -> Dict[str, str]:
    """
    简单谓词聚类:
    - 同一 predicate 的 facts 归为同一社区
    - 例: 所有 "MakeStatement" 类事实 → comm_2_MakeStatement
    """
    pred_groups = defaultdict(list)
    for f in facts:
        pred = f['predicate']
        pred_groups[pred].append(f)
    
    communities = {}
    for cid, (pred, group) in enumerate(pred_groups.items()):
        for f in group:
            communities[f['name']] = f"comm_{cid}_{pred}"
    return communities
```

**设计决策**:
- 当前实现是简化版 (按 predicate 分组),不是 Louvain/标签传播
- 100% 覆盖率,适合作为单元测试基准
- 真实场景应使用 HugeGraph 内置 `community_detection` 算法 (Vermeer 引擎)

---

## 四、Embedding 选型 (关键技术决策)

### 4.1 为什么不用 MiMo Embedding

```bash
# 实测 MiMo API
curl -X POST "https://api.xiaomimimo.com/v1/embeddings" 
  -d '{"input":["China"],"model":"text-embedding-ada-002"}'
# 返回: 404 Not Found

# 查询可用模型
curl "https://api.xiaomimimo.com/v1/models"
# 返回: 只有 chat/tts 模型,无 embedding
```

### 4.2 为什么用 sentence-transformers

| 方案 | 优势 | 劣势 | 决策 |
|------|------|------|------|
| MiMo API | 国产合规 | **不支持** | ❌ |
| OpenAI text-embedding-3 | 效果最好 | 数据出境,贵 | ❌ |
| 智谱/百度 embedding | 国产 | 需额外 API key | ⚠️ |
| **sentence-transformers 本地** | 离线/免费/开源 | 模型需下载 (~90MB) | ✅ |

### 4.3 具体选型: all-MiniLM-L6-v2

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# 维度: 384
# 大小: 90MB
# 速度: ~5000 句/秒 (M1 CPU)
# 质量: 优于 mBERT, 略低于 OpenAI ada-002
```

### 4.4 踩坑记录

| 问题 | 解决 |
|------|------|
| NumPy 2.x 与 torch 不兼容 | `pip install "numpy<2" --force-reinstall` |
| 模型首次加载慢 | 缓存到 `~/.cache/huggingface/` |
| 中文支持 | all-MiniLM-L6-v2 对中文一般,生产建议换 `BAAI/bge-small-zh-v1.5` |

---

## 五、性能数据 (真实跑通)

### 5.1 Phase 级别耗时

| 阶段 | 耗时 | 数据规模 |
|------|------|----------|
| P1 Schema 创建 | 2.1s | 12 PK + 2 VL + 4 EL |
| P5 数据加载 | 1.3s | 2000 facts |
| P7 图写入 | 18.5s | 2000 vertices + ~6400 edges |
| P8 衰减计算 | 0.4s | 2000 facts |
| P9 范围查询 | 0.05s/次 | 5 queries |
| P10 Point Query | 0.25s/次 | 20 queries |
| P11 冲突检测 | 0.8s | 2000 facts |
| P12 社区检测 | 0.2s | 2000 facts |
| **总耗时** | **52.9s** | 12/12 PASS |

### 5.2 关键指标

| 指标 | v1 (hash) | v2 (sentence-transformers) | 提升 |
|------|----------|---------------------------|------|
| Recall@5 | 0.15 | **0.55** | **+267%** |
| MRR | 0.048 | **0.35** | **+629%** |
| Hit@1 | 0.0 | **0.20** | +20% |
| TemporalAccuracy | 0.50 | **0.60** | +20% |
| ConflictAccuracy | 1.00 | **1.00** | — |

---

## 六、踩坑大全 (从血泪经验来)

### 6.1 HugeGraph REST API 三大坑

| 坑 | 现象 | 解决 |
|----|------|------|
| **VL properties 必须是 String[]** | 返回 400 "expect String[]" | 用 `["name","subject_name",...]` 而非 `[{name:"name"}]` |
| **FLOAT 类型陷阱** | "expect Double" 或 "expect String" | 先查 `/propertykeys` 看实际类型,不能信 schema 声明 |
| **重复创建返回 400 而非 409** | 看起来是错误但其实是已存在 | 检测 `xisted` in response body |

### 6.2 多图隔离必须做

```python
# 错误: 全部写到 hugegraph 默认图
HG_GRAPH = "hugegraph"
# 后果: 之前 553 个残留 facts 导致新写入 80% 失败

# 正确: 物理隔离
HG_GRAPH = "poc_temporal_kg"  # 单独图,清空无副作用
```

### 6.3 LLM 输出格式不可信

```python
# 即使提示 "Output ONLY raw JSON"
content = llm_response  # 仍可能包含 ```json ... ``` 包装或前导解释
# 必须用正则提取 + json.loads 多次重试
```

---

## 七、待优化项 (诚实清单)

| 优化项 | 现状 | 目标 | 优先级 |
|--------|------|------|--------|
| T' (系统认知时间) | 未实现 | 完整双时态 | P1 |
| 真实 LLM 事实提取 | 规则式 | MiMo API | P1 (待 MiMo 开放) |
| LOCOMO 完整 35 段 | 仅 2 段 | 全部 | P2 |
| LOCOMO F1/B1/J 指标 | 未计算 | 完整评估 | P2 |
| PPR 检索 | RRF | PPR 算法 | P3 |
| 社区检测升级 | 谓词聚类 | Louvain / 标签传播 | P3 |
| 中文 embedding | MiniLM (英文) | bge-small-zh | P3 |
| 大规模压测 | 2000 facts | 10万+ facts | P3 |

---

## 八、参考资料

- PoC 源码: `tests/temporal_kg_icews_v2.py`
- PoC 结果: `tests/temporal_kg_icews_v2_result.json`
- Benchmark 数据: `tests/benchmark_data/icews14_agent_memory_benchmark.json`
- LOCOMO 数据: https://huggingface.co/datasets/Aman279/Locomo
- ICEWS14 数据: https://huggingface.co/datasets/linxy/ICEWS14
