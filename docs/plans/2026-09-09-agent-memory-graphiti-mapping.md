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

## 7. 下一步

1. `State` + `TRANSITION` 建模落进 schema_def，实现当前状态识别与流转路径（需求③）
2. 双时间线字段 + RANGE 索引 + 哨兵值约定落地，跑 AS-OF 查询冒烟
3. 与 `semantic_layer` 的能力合并点评估：预算器、BM25/RRF 融合、MCP 工具层
