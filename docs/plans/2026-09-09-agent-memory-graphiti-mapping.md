# Agent Memory：Graphiti 时序模型 → HugeGraph schema 映射

> 日期：2026-09-09
> 目的：把 Graphiti 已验证的双时间线记忆模型落到 HugeGraph，为「时序图 / 实体关系网络 / 状态图」三项需求提供存储与查询地基。
> 代码：`hugegraph-llm/src/hugegraph_llm/memory/`（分支 `feat/agent-memory-graphiti`）
> 源起：`/Users/mac/Desktop/apache-code/hg-memory-zep`（fork `haohao0103/h-zep`）已完成的 Graphiti → HugeGraph 驱动。

---

## 0. 结论摘要

1. **Graphiti → HugeGraph 驱动已完成并验证**：50 个测试在 graphiti **0.29.2 与 0.30.2 两个版本上均全绿**，可直接作为 memory 模块底座。
2. **双时间线模型正是需求①所需**：`valid_at/invalid_at`（事实时间）+ `created_at/expired_at`（事务时间），且**矛盾信息打 `expired_at` 而非删除**——历史状态回溯因此成立。
3. **HugeGraph Server 不需要开发**（单节点路径）：时序是属性 + RANGE 索引，AS-OF 是双边界过滤；原生 temporal 引擎分支（`feature/temporal-graph-support`）当前**只有 Skeleton、无后端实现**，且只覆盖 valid time，暂不可用。
4. **需求③状态图是唯一空白**：Graphiti 也没有状态建模，需新建 `State` 顶点 + `TRANSITION` 边。

---

## 1. 资产来源与验证记录

| 项 | 值 |
|---|---|
| 来源 | `/Users/mac/Desktop/apache-code/hg-memory-zep`，fork `haohao0103/h-zep`，HEAD `d4fa08e` |
| 核心文件 | `hugegraph_driver.py`（568 行，`GraphDriver` 子类）、`driver.py`（247 行 REST 客户端）、`embedder.py` |
| 测试 | 50 个用例，离线运行（fake client + fake embedder） |
| 验证结果 | graphiti 0.29.2：50 passed；graphiti 0.30.2：50 passed |
| 端到端（源项目文档） | 3 段 episode（2024-03 → 2025-06 → 2025-09），10 Entity + 3 Episodic + 6 RELATES_TO；`@2024-10-01` 查询正确返回历史雇主 Acme，排除尚未生效的 Globex |

**导入方式**：不 vendored 源码（源项目内含 422 文件的 `graphiti-source/` 子树），改为在 `pyproject.toml` 声明可选依赖：

```toml
[project.optional-dependencies]
memory = ["graphiti-core>=0.29,<0.31"]
```

---

## 2. Graphiti 时序模型 → HugeGraph 字段映射

### 2.1 顶点

| Graphiti 概念 | HugeGraph 顶点标签 | 关键属性 | 说明 |
|---|---|---|---|
| `EntityNode` | `Entity` | `uuid`, `name`, `summary`, `created_at` | 用户/对象/概念等实体 |
| `EpisodicNode` | `Episode` | `uuid`, `content`, `source`, `created_at`, `valid_at` | 原始对话/事件，记忆溯源入口 |

### 2.2 边（时序核心）

| Graphiti 概念 | HugeGraph 边标签 | 时序字段 | 语义 |
|---|---|---|---|
| `EntityEdge` | `RELATES_TO` | `valid_at`, `invalid_at`, `created_at`, `expired_at` | 实体间事实，**双时间线** |
| `EpisodicEdge` | `MENTIONS` | `created_at` | episode → 实体，溯源用 |

### 2.3 双时间线字段语义（关键）

| 字段 | 时间轴 | 含义 | 未设置时 |
|---|---|---|---|
| `valid_at` | Valid Time | 事实**开始为真**的时刻 | 立即生效 |
| `invalid_at` | Valid Time | 事实**停止为真**的时刻 | 仍有效 |
| `created_at` | Transaction Time | 我们**何时得知**该事实 | 必填 |
| `expired_at` | Transaction Time | 我们**何时判定其失效** | 仍被相信 |

**两个时间轴的分工**：
- Valid Time 回答「**事实本身**在何时成立」——用户 2024 年在 Acme 工作，`valid_at=2024-03`，`invalid_at=2025-06`
- Transaction Time 回答「**我们何时知道**」——2025-09 才被告知跳槽，则 `created_at=2025-09`

**失效而非删除**（Graphiti 的核心设计，也是回溯能力的原因）：
新信息与旧事实矛盾时，给旧边打 `expired_at`，**不删除它**。因此「2024-10-01 时我们认为张明在哪工作」这类 AS-OF 查询可精确回答。

### 2.4 AS-OF 查询的实现（HugeGraph 单节点）

Graphiti 的语义是「取 `valid_at ≤ t < invalid_at` 且 `expired_at` 为空/晚于 t 的边」。落到 HugeGraph：

```groovy
// 双边界过滤：半开区间 [valid_at, invalid_at)
g.E().hasLabel('RELATES_TO')
 .has('valid_at', lte(t))
 .has('invalid_at', gt(t))
```

**关键工程约定——开放区间用哨兵值而非 null**：
未失效的边其 `invalid_at` 必须写入 `Long.MAX_VALUE` 而不是留空。原因：HugeGraph 的 `has()` 过滤要求属性有索引，而 null 值不进索引，会导致「仍有效的边」被漏查（我们语义层 `Column.is_time_dimension` 的三态设计同理）。配合 RANGE 索引：

| 索引 | 类型 | 字段 |
|---|---|---|
| `relatesByValidFrom` | RANGE | `RELATES_TO.valid_at` |
| `relatesByInvalidAt` | RANGE | `RELATES_TO.invalid_at` |
| `entityByName` | SECONDARY | `Entity.name` |

---

## 3. 三项需求的落点

| 需求 | 状态 | 实现 |
|---|---|---|
| ① 时序图（演化/事件追踪/历史回溯） | ✅ 模型与驱动就绪 | 双时间线字段 + AS-OF 双边界过滤；Server 侧可用 `EdgeLabel` TTL 做记忆自动衰减 |
| ② 实体关系网络 | ✅ 模型就绪 | `Entity`/`Episode` 顶点 + `RELATES_TO`/`MENTIONS` 边；复用已有图遍历与预算能力 |
| ③ 状态图（当前状态/流转） | ❌ **空白，需新建** | 见 §4 |

---

## 4. 状态图设计（需求③，唯一需新建部分）

Graphiti 无状态建模，需自行设计——但可复用我们已有的路径求解：

```
State 顶点      {name, entity_uuid, valid_from, valid_to}
TRANSITION 边   {from_state, to_state, valid_at, trigger_event}
```

- **当前状态识别**：`has('valid_to', Long.MAX_VALUE)` 的最新 `State`——一次索引查询，非全量扫描
- **状态流转分析**：`TRANSITION` 链上的路径查询，**直接复用 `semantic_layer/join_path.py` 的 BFS/Dijkstra**
- **与时序边的关系**：`State` 是 `Entity` 的子类型节点，`TRANSITION` 与 `RELATES_TO` 并行存在，互不干扰

---

## 5. 架构决策：REST vs Gremlin

|  | 本 memory 驱动 | 既有 semantic_layer |
|---|---|---|
| HugeGraph 访问方式 | **REST + traversers** | **Gremlin** |
| JDK 约束 | 无 | 锁死 JDK 11（JDK 17/21 下 Gremlin script engine 不存在） |

**当前决策**：memory 模块统一用 REST（不受 JDK 约束，生产更稳）；semantic_layer 暂不动其 Gremlin 实现（已有 376 个测试验证，且当前环境 Gremlin 可用）。
**后续待定**：是否将 semantic_layer 的 Gremlin 调用迁移到 REST，以消除 JDK 版本红线。需实测两套访问层在同一 HugeGraph 上的性能与兼容性后再定。

---

## 6. 已知限制

1. **向量检索为进程内余弦扫描**——HugeGraph 1.7 无向量索引（与我们语义层结论一致）。PoC 规模（数百至数千节点）可接受，上量需外置向量库。
2. **原生 temporal 引擎不可用**：`feature/temporal-graph-support` 分支仅有 `TemporalBackendStoreSkeleton`，无任何后端实现，且文档要求「不支持则报 not-supported、禁止静默退化」；另外它第一阶段只做 Valid Time，事务时间仅预留——**语义上还不如 Graphiti 模型完整**。等 HStore 集群就绪后再评估。
3. **embedder 依赖 sentence-transformers**，离线环境需预置模型；已设为可选。

---

## 7. 落地进展

| 步骤 | 状态 | 结果 |
|---|---|---|
| 双时间线 schema 落 HugeGraph | ✅ | `memory/schema.py`；RANGE 索引 + `OPEN` 哨兵 |
| 时序原语（as_of / between / 失效 / 演化） | ✅ | `memory/temporal.py` |
| 状态图（当前状态 / 流转路径） | ✅ | `add_state` / `close_state` / `current_state` / `transition_path` |
| 真实服务器冒烟 | ✅ | 见 §8 |

## 8. 真实服务器冒烟结果（单节点 HugeGraph 1.7.0，图 `memory_smoke`）

```
expired by f2: ['f1']                        ← 矛盾事实被失效而非删除
as_of 2024-06: ['works at Acme']             ← 历史状态回溯 ✅
as_of 2025-06: ['works at Globex']
history of works_at: [Acme → Globex]         ← 演化追踪 ✅
current state: suspended                      ← 当前状态识别 ✅
transition path: [active → suspended]         ← 状态流转分析 ✅
total facts retained: 2                       ← 旧事实保留，未删除
```

三项需求中 ①② 已由 Graphiti 模型直接满足，③ 由新增的 state 建模满足。

**冒烟暴露并修复的两个 REST 编码 bug**（驱动自带 50 个离线测试用 fake client，**查不出**这两个问题）：

1. `_json_str` 把所有值字符串化——Zep 模型（全 TEXT 属性、日期为 ISO 字符串）的遗留设计。对 LONG 类型的 `created_at` 发送 `"1704067200000"` 会被服务端拒绝：`actual type String`。改为数值/布尔透传，仅 list/dict 做 JSON 编码。
2. `update_edge` 给边 id 加了引号。边 id 形如 `Su1>1>1>>Sc1`，加引号后服务端报 `Invalid format of edge id`；而**顶点 id 恰恰需要引号**——两种 id 的引用规则相反。已修正并加测试锁定。

两处均已补测试（`tests/memory/test_driver_quoting.py`），且在 graphiti 0.29.2 / 0.30.2 上均 82/82 通过。

## 9. 检索层：站在通用 RAG 栈上（非语义层）

**重要更正**：最初提议"与 semantic_layer 合并检索能力"是错的——那混淆了**领域功能**（语义层=数仓元数据，服务 Text2SQL）与**工程能力**（检索栈）。memory 需要的是 RAG 检索流程，其归属是通用检索栈，不是语义层。

### 复用关系

| 组件 | 来源 | 是否领域耦合 |
|---|---|---|
| `KGRetriever` / `RetrieverResultItem` | `operators/graph_op/kg_retriever_base.py`（已有） | 否，通用 ✅ |
| `ReciprocalRankFusion` | `operators/graph_op/rrf_fusion.py`（已有） | 否，通用 ✅ |
| `SchemaRetriever` / `GraphStructureRetriever` 等 | `kg_multi_retrieval.py`（已有） | **是**（`NODE_LABELS = Table/Field/Metric`）❌ |

**结论**：memory 复用前两个通用件，**不用** `SchemaRetriever` 系列——它是数仓 schema linking 专用（检索单元是 Table/Field/Metric 顶点），而 memory 的检索单元是 `RELATES_TO` **边**（fact），领域不匹配。

### memory 自研部分（仅时序维度）

`memory/retrieval.py`：

1. **时序门控（filter，非 score）**——只保留 `as_of(t)` 成立的 fact。在 T 时不为真的事实，无论多匹配都不该出现。
2. **时间衰减（score）**——半衰期 30 天的指数衰减，基于"事实为真持续了多久"而非"我们何时得知"。

**关键顺序：门控先于打分。** 若先打分再过滤，过期事实会占用 rank 预算把当前事实挤出 `top_k`（`test_gating_precedes_scoring` 锁定此语义）。

### 真实服务器冒烟（`memory_retrieval_smoke`）

```
as_of T0+10d:  0.794 works at Acme (current=False)   ← 仍为真，但后来被推翻
               0.794 prefers dark mode (current=True)
as_of T60:     0.500 works at Globex
               0.250 prefers dark mode               ← 60 天衰减到 0.25
top_k=1:       [works at Globex]                     ← 门控生效
附加通道:      channels=['temporal','extra_0'] → 顺序变为 [f3, f2]
```

双时间线语义被这个输出精确验证：T0+10d 时 Acme **仍为真**（valid 区间内）但 `is_current=False`（T30 被 Globex 取代）——两个时间轴独立工作。

### 测试暴露的两个问题（均已修）

1. **通道未命名**：`extra_channels` 传入裸 list，与 RRF 的 `(channel, items)` 元组混用导致解包失败。统一为具名通道 `extra_N`。
2. **我自己写错的测试期望**：原以为 `include_superseded=True` 能返回 T60 时的 Acme——但 Acme 在 T60 **事实本身已不成立**（valid 区间 T0–T30 已过），任何模式都不该返回。`include_superseded` 只控制"已被推翻的信念"，不控制"不为真的事实"。已改为能区分两轴的场景（valid 到 T60 但 T30 被 supersede）。

**101 个测试通过**（50 驱动 + 21 时序 + 19 检索 + 11 编码）。

## 10. 下一步

1. 向量通道接入（`MultiRecallConfig` 里 `vector` 已预留但未挂载；memory 目前只有时序+附加通道）
2. 与 `semantic_layer` 的**并列关系**确认：两者是同一通用检索栈在不同领域的应用，无需互相依赖
3. REST vs Gremlin 统一决策（待实测数据）
