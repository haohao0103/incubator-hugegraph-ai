# Skill Graph 业界对标与提升计划

> **状态：** 进行中
> **日期：** 2026-10-22
> **范围：** `hugegraph-llm/src/hugegraph_llm/engines/skill_graph/`
> **技术全流程文档：** [`hugegraph-llm/demo/skill_graph/README.md`](../../../hugegraph-llm/demo/skill_graph/README.md)
> **说明：** 本文所有“我们的现状”均以当前代码为准；对标数据来自下列公开论文与仓库的 README/架构文档。

---

## 1. 目标

回答三个问题，并给出可执行计划：

1. 与业界代表性 Skill Graph 框架相比，我们**真正缺什么**；
2. 哪些差距**值得补**，哪些是对方定位不同、我们不该跟；
3. 按什么顺序补，**每项的验收标准**是什么。

**核心判断：检索算法我们不落后，落后的是“证明它有用”和“让它越用越准”。**

---

## 2. 对标对象

| 框架 | 定位 | 公开材料 |
|---|---|---|
| **Graph-of-Skills (GoS)** | EMNLP 2026；依赖感知的结构化检索；SkillsBench + ALFWorld；2000 技能规模验证 | [GitHub](https://github.com/davidliuk/graph-of-skills) · [arXiv:2604.05333](https://arxiv.org/abs/2604.05333) |
| **SkillDAG** | 自演化类型化技能图；Agent 推理时可改图；执行证据回流 | [GitHub](https://github.com/Ericbai06/SkillDAG) · [arXiv:2606.03056](https://arxiv.org/abs/2606.03056) |
| **SkillNet** | 技能全生命周期基础设施；50 万+ 公共技能库；发现/创建/评估/分析/路由 | [GitHub](https://github.com/zjunlp/SkillNet) · [arXiv:2603.04448](https://arxiv.org/abs/2603.04448) |

---

## 3. 能力矩阵

| 维度 | GoS | SkillDAG | SkillNet | 我们 |
|---|---|---|---|---|
| 图模型 | 类型边 + 依赖 | 类型边 + `specializes` + 冲突 5 子类 | `compose_with` / `similar_to` | 5 类边（[schema.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/schema.py)） |
| 建边方式 | 确定性 I/O 契约边 + LLM 校验 | LLM 冷启动 + **执行证据在线改图** | LLM 场景分析 | LLM 双视图 + 相似度 |
| 检索 | 混合种子 + 反向 PPR + budget | 向量 + BFS + 冲突区 | BM25 + 向量 + 图扩展 + LLM Wiki 路由 | 混合种子 + 反向 PPR + 预算装箱（[retriever.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/retriever.py)） |
| 集合校验 | 部分（依赖闭包） | **check / expand / repair-set** | 路由时约束 | 仅建图时冲突剪枝 |
| 在线演化 | 无 | **propose / commit，add / remove / retype** | 无 | 无 |
| 证据 / provenance | `deterministic_io` / `llm_validated` | reason + evidence + task-id | 引用原文行号 | 有 provenance 字段，无证据链 |
| 评测 | SkillsBench + ALFWorld + 消融 + 2000 规模 | SkillsBench + ALFWorld | SkillNet-Gym 基准 | 示例查询 + 单测 |
| 技能生命周期 | 无 | 无 | 创建 / 评估 / 安全 / 成本 | 无 |
| 分发集成 | Claude Code MCP 插件 | CLI（Agent 可调用） | PyPI SDK + MCP + 浏览器 UI | CLI + 演示服务 |
| 生产存储 | JSON | JSON | 本地索引快照 | **HugeGraph + OceanBase + 原子发布/回滚** |
| 增量构建 | 未强调 | 在线增量 | 分析快照 | **内容 diff 增量** |

---

## 4. 我们的真实位置

### 4.1 领先（可作为差异化护城河）

- **服务即真相**：HugeGraph 存图结构，OceanBase 存向量与正文，生产多副本不依赖本地 JSON（[service_loader.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/service_loader.py)）。
- **版本化 + 原子指针切换**：发布/回滚不影响在途查询（[manifest.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/manifest.py)）。
- **查询下沉**：只召回种子 + 有界扩展，复杂度 O(候选数)，不是“全图进内存”（[hg_store.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/hg_store.py)）。
- **增量构建成本**只与新技能数相关（[graph_builder.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/graph_builder.py)）。

### 4.2 持平

- 反向 PPR、混合种子、RRF、预算化 hydration、5 类带类型边、LLM 建边。
- 检索核心配方与 GoS 同级；GoS 的消融结论（反向遍历 > 图传播本身）我们已按同一方向实现。

### 4.3 落后

见下一节差距明细。

---

## 5. 差距明细

| # | 差距 | 对标证据 | 我们的现状 |
|---|---|---|---|
| **G1** | **没有可信评测** | GoS 有主结果、组件消融、预算对齐实验、2000 技能扩展；SkillDAG 有 SkillsBench/ALFWorld 指标与预期结果文件 | 只有 3 个手工查询，无法证明相对扁平检索的增益 |
| **G2** | **没有在线演化闭环** | SkillDAG 的 `propose-edge` / `edit-edge`，执行一次任务即可留下证据 | 图建完即固化，LLM 误判无法被运行结果纠正 |
| **G3** | **没有集合校验/修复** | SkillDAG 的 `check-set` / `expand-set` / `repair-set` | 冲突只在建图时剪枝，查询结果不校验“这套技能能否一起用” |
| **G4** | **边类型不完备** | SkillDAG 有 `specializes`，`conflicts_with` 分 5 个子类：资源互斥、状态污染、协议冲突、产物覆盖、条件触发 | 5 类边，无 `specializes`，冲突无子类，无法指导修复 |
| **G5** | **边缺证据** | GoS 区分 `deterministic_io` / `llm_validated`；SkillDAG 每条边带 reason/evidence/task-id | 有 provenance 字段，但没有“为什么连这条边”的可审计证据 |
| **G6** | **没有技能质量门禁** | SkillNet 评估安全性、完整性、可执行性、可维护性、成本 | 默认 SKILL.md 都是好技能，坏文档直接进图 |
| **G7** | **没有规模验证** | GoS 2000 技能；在线传播 p50 ~1.5ms；构建成本 $0.29/200、$2.36/2000 | 最大只验证 34 节点；文档中“万级无压力”未实测 |
| **G8** | **没有生态入口** | GoS 提供 Claude Code MCP 插件；SkillNet 提供 PyPI SDK + MCP + UI | 只有 CLI 和 demo Flask，Agent 接入成本高 |

### 5.1 可借鉴的关键实证

GoS 的受控实验（200 技能、87 查询、预算对齐，只改变图操作）：

| 检索操作 | R_avail | R_full | C（bundle 完整率） | D（依赖对共同召回） |
|---|---:|---:|---:|---:|
| 反向 PPR（GoS） | **0.575** | **0.502** | **0.421** | **0.654** |
| 仅正向 PPR | 0.490 | 0.428 | 0.311 | 0.362 |
| 无图（扁平基线） | 0.542 | 0.473 | 0.368 | 0.481 |
| 一跳依赖 | 0.542 | 0.473 | 0.368 | 0.481 |

结论：**反向遍历比“有没有图”更关键**；正向传播甚至不如不用图。我们的实现方向正确，但必须用同类实验证明。

其他可用事实：

- 构建成本：200 技能 $0.29，2000 技能 $2.36；在线传播 p50 ~1.5ms。
- 文档质量是主要瓶颈：描述删掉一半，召回下降 0.146；I/O 字段删掉一半，仅下降 0.024。
- 规模扩展：2000 技能时 GoS 奖励 31.3，向量检索 23.8，全量暴露 26.7；全量 token 从 1.93M 涨到 5.84M，检索方案基本恒定。

---

## 6. 提升计划

> 执行顺序：**P0 是前置门槛，未完成前不做 P1/P2**。所有目标值先作为待校准目标，首轮实测后回填。

### P0：先补可信度

#### P0-1 建立检索级评测与消融基线

- [ ] 准备评测集：SkillsBench 查询集（87 条）+ 50 条中文开放式任务。
- [ ] 实现 retrieval-only harness：种子检索 → 图扩展 → 预算装箱，不跑 Agent。
- [ ] 指标：`R_avail`、`R_full`、`C`、`D`、交付 token、dropped token、p50/p95 延迟。
- [ ] 四组消融：完整方案、仅正向 PPR、无图、一跳依赖；可选去词法通道。
- [ ] 固化构建产物与嵌入缓存，保证实验可复现、LLM 成本可控。
- [ ] 输出 `reports/` 下的 JSON + Markdown 报告。

**验收标准**

- 一条命令复现全部指标；
- 能回答“相对无图基线，C/D/token 各提升多少”；
- 首轮目标：C ≥ 0.60、D ≥ 0.65（对齐 GoS 同量级），p95 检索 < 100ms（2k 技能）。

**代码落点**

- 新增 `hugegraph-llm/demo/skill_graph/eval/`；
- 复用 [retriever.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/retriever.py) 的 `retrieve_from_seeds` 做受控实验。

#### P0-2 冲突感知的集合校验/修复

- [ ] `Graph.check_set(ids)`：返回缺失前置、冲突对、冗余替代。
- [ ] `Graph.repair_set(ids)`：给出最小修复建议（删谁 / 换谁 / 补谁）。
- [ ] `RetrievalResult` 附带 `warnings`，不再静默返回互斥组合。
- [ ] CLI 增加 `check-set` / `repair-set`。
- [ ] 单测覆盖：缺失依赖、冲突、环、冗余四类。

**验收标准**

- 交付集合中不存在未声明的冲突对；
- `repair_set` 输出的集合能通过 `check_set`；
- 现有图无需重建即可使用。

**代码落点**

- [schema.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/schema.py)（图不变量与集合校验）；
- [cli.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/cli.py)。

#### P0-3 边证据与状态字段

- [ ] `SkillEdge` 增加 `evidence`（原文片段 / 触发任务 / 执行结果）与 `status`（proposed / committed）。
- [ ] 建边与诱导阶段写入证据；无证据的 LLM 边默认 `proposed`。
- [ ] manifest 哈希包含证据，发布可追溯。
- [ ] 可视化支持查看边证据。

**验收标准**

- 每条诱导边都能回答“为什么连”；
- 检索只消费 `committed` 边；
- 发布/回滚后证据不丢失。

---

### P1：让图越用越准

#### P1-4 在线演化协议（对标 SkillDAG）

- [ ] `propose_edge` / `remove_edge` / `retype_edge` 先 dry-run，返回影响面。
- [ ] 提交前强制校验：环检测、冲突剪枝、对称闭包。
- [ ] 提交必须带 `reason` 与 `task_id`。
- [ ] append-only 审计日志，支持按任务/构建回滚。
- [ ] 检索只使用 `committed` 边。

**验收标准**

- 任意改图后 DAG 不变量成立；
- 可回滚到改图前状态；
- 审计日志完整、可查询。

**代码落点**

- 新增 `evolution.py`；
- 与 [manifest.py](../../../hugegraph-llm/src/hugegraph_llm/engines/skill_graph/manifest.py) 的版本机制打通。

#### P1-5 边类型补齐

- [ ] 新增 `specializes`（A 是 B 的细化），用于替代关系与路由降级。
- [ ] `conflicts_with` 增加 5 个子类：`resource_exclusion`、`state_contamination`、`protocol_conflict`、`artifact_clobber`、`condition_triggered`。
- [ ] 同步诱导 prompt、schema、HugeGraph edge label、可视化图例、检索 walk 规则。
- [ ] 旧图迁移：缺失子类默认 `unknown`。

**验收标准**

- 新类型端到端可用（建边 → 存储 → 检索 → 可视化）；
- 旧数据可无损加载；
- 冲突子类能指导 `repair_set` 给出不同修复策略。

#### P1-6 确定性 I/O 契约边（对标 GoS）

- [ ] 从 SKILL.md 解析 `inputs` / `outputs`。
- [ ] 输出-输入匹配生成 `depends_on`，provenance=`deterministic_io`。
- [ ] LLM 只做校验，不作为唯一来源。
- [ ] 明确确定性边与 LLM 边的冲突优先级。

**验收标准**

- 确定性边可复现、零 LLM 成本；
- 在评测集上单独报告确定性边的覆盖与精度。

#### P1-7 规模与成本压测

- [ ] 生成 2k / 10k 技能合成图。
- [ ] 测量构建成本与耗时、查询 p50/p95、内存占用、HG/OB 往返延迟。
- [ ] 用实测数据修正技术文档中未经验证的规模结论。

**验收标准**

- 输出“规模-延迟-成本”曲线；
- 2k 技能 p95 检索 < 100ms；
- 构建成本不高于 GoS 同规模量级。

---

### P2：补生态与治理

#### P2-8 技能质量门禁（对标 SkillNet）

- [ ] 入库检查：描述完整性、I/O 可执行性、正文长度/成本、冲突声明。
- [ ] 质量分写入节点，低分技能默认不进候选。
- [ ] 输出低质量技能报告。

**验收标准**

- 坏 SKILL.md 无法污染图；
- 评测中能量化质量分对召回的影响。

#### P2-9 SDK / MCP / 服务化

- [ ] 提供 Python SDK。
- [ ] 提供 MCP Server，暴露 `search` / `check-set` / `get-dependencies`。
- [ ] 服务端补 API Key、租户隔离、限流、审计。

**验收标准**

- Claude Code / Cursor 可直接接入；
- 不依赖 demo 页面即可完成检索与集合校验。

#### P2-10 文档质量反馈

- [ ] 描述质量纳入建边置信度，低质量描述降低边权。
- [ ] 复现 GoS 的“描述删半 → 召回下降”实验，验证敏感性。

**验收标准**

- 质量分与召回/边精度有可量化相关性；
- 质量门禁可通过配置关闭。

---

## 7. 验收与度量

| 指标 | 定义 | 首轮目标（待实测校准） |
|---|---|---|
| `R_avail` | 可用技能集合上的召回率 | ≥ 0.55 |
| `R_full` | 全量技能集合上的召回率 | ≥ 0.48 |
| `C` | 交付 bundle 与完整依赖闭包的完全一致率 | ≥ 0.60 |
| `D` | 依赖对共同召回率 | ≥ 0.65 |
| 冲突违规 | 交付集合中未声明的冲突对数 | 0 |
| 预算遵守 | 交付 token ≤ budget 的比例 | 100% |
| p95 延迟 | 2k 技能图上的检索 p95 | < 100ms |
| 可追溯性 | 诱导边带证据的比例 | 100%（P0 后） |

---

## 8. 代码落点汇总

| 任务 | 主要文件 |
|---|---|
| P0-1 评测 | 新增 `demo/skill_graph/eval/`，复用 `retriever.py` |
| P0-2 集合校验 | `schema.py`、`retriever.py`、`cli.py` |
| P0-3 证据 | `schema.py`、`edge_inducer.py`、`manifest.py`、`viz.py` |
| P1-4 在线演化 | 新增 `evolution.py`、`manifest.py` |
| P1-5 边类型 | `schema.py`、`edge_inducer.py`、`graph_builder.py`、`hg_store.py`、`viz.py` |
| P1-6 确定性边 | `skill_loader.py`、`edge_inducer.py`、`schema.py` |
| P1-7 压测 | 新增 `demo/skill_graph/bench/` |
| P2-8 质量门禁 | 新增 `quality.py`、`graph_builder.py` |
| P2-9 SDK/MCP | 新增 `sdk/`、`mcp_server.py`、`demo/skill_graph/server.py` |
| P2-10 文档质量 | `edge_inducer.py`、`schema.py` |

---

## 9. 风险与不做清单

### 风险

- **评测依赖外部基准**：SkillsBench 需要 Docker/Harbor，先用 retrieval-only 子集，避免被环境阻塞；
- **LLM 成本**：评测与消融复用同一构建产物与嵌入缓存，禁止每个实验重建图；
- **在线演化破坏图**：必须 propose/commit 两阶段 + 不变量校验 + 审计日志 + 可回滚；
- **新增边类型影响 HugeGraph schema**：schema 创建保持幂等，旧图默认值迁移后再切换；
- **指标被“调参”污染**：评测集与消融配置冻结后再调优，报告必须包含基线。

### 不做

- **不追 50 万公共技能库**：那是 SkillNet 的定位，不是我们的；
- **不从零造 benchmark**：直接对齐 SkillsBench/ALFWorld 子集，保证可比；
- **不在 P0 完成前堆功能**：没有评测，P1/P2 无法证明价值；
- **不把“图”当成固定工作流的替代品**：固定流程用模板更划算，边界见技术文档 §1.1。