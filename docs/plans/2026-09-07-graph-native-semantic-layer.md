# HugeGraph 图原生语义层落地方案

> 版本：v1.0
> 日期：2026-09-07
> 依据：`neocarta`（neo4j-labs，本地 `neocarta/`，HEAD `ba6bf79`）源码实证 + 本工作区 HugeGraph 1.7.0 实测结论（`docs/HG_SEMANTICA_INTEGRATION.md`）+ `incubator-hugegraph-ai/hugegraph-llm` 现有 `nl2sql/` 资产
> 定位：这是一份**可执行**的落地方案，不做框架选型综述。所有结论均来自本工作区真实代码与运行记录。
>
> **路径基准**：本文写于多仓库并列的工作区，两类路径并存——
> - `neocarta/`、`docs/HG_SEMANTICA_INTEGRATION.md` 指**工作区根**，不在本仓库内；
> - `hugegraph-llm/src/hugegraph_llm/...` 指**本仓库根**，即本文所在的 `incubator-hugegraph-ai`。
>
> **状态（2026-09-07）**：M0 建模层、M1 连接器、M2 检索与裁剪、M3 MCP 工具层已实现并实测（见 §5.5–§5.9），代码在 `feat/graph-native-semantic-layer` 分支；M4 评测体系、M5 Ossie 完善尚未开始。

---

## 0. TL;DR

1. **不要从零开始。** 本工作区 `incubator-hugegraph-ai/hugegraph-llm/src/hugegraph_llm/nl2sql/` 已经是一个"图增强 Text2SQL 语义层"的半成品：`schema_graph`（元模型）、`join_path`（Dijkstra + Steiner + proven 判定）、`linking`（schema linking）、`engine`（networkx/Vermeer 双引擎）、`vector_store`（外置向量）、`hugegraph_schema_source`（从 live HG 拉 schema）、`evaluation/`。**复用率预估 > 60%**，自研工作量集中在"检索裁剪 + MCP 化 + 约束/信任层"。
2. **neocarta 值得抄的是架构，不是数据。** 它的连接器契约、MCP 工具设计、TableContext 装配、OSI 双向映射都是好样板；但文章引用的"token 降 20–30%、多表 JOIN 准确率 +10pp"**在本仓库中无可复现依据**——`neocarta/eval/README.md` 原文写明 "not yet complete and should not be used as a reference"。这两个数字不能进决策材料。
3. **真正的移植障碍不是 Cypher→Gremlin 语法，而是 HugeGraph 缺两类索引。** HugeGraph 1.7.0 只有 secondary / range / shard / unique，`IndexType.java` 中 `SEARCH`（全文）标注 "not supported now"，**向量索引完全不存在**；Gremlin 侧也没有 Cypher 的 `OPTIONAL MATCH` / `COLLECT{}` 子查询。因此检索必须改成**外置 ANN + Gremlin 取行 + Python 侧装配**的三段式，而不是"把 Cypher 改写成 Gremlin"。
4. **不建议引入 Cube / MetricFlow 做约束层再双写回 HugeGraph。** 双写一致性成本高于收益。正确做法是 **HugeGraph 作为唯一 SoT**，用 SHACL/SKOS 做约束（复用 Semantica ontology 模块），Ossie 只作为**导出格式**而非建模入口。
5. **文章漏掉的关键一层是"信任层"。** Ossie 不标准化 `confidence / lineage / freshness / provenance`（"定义能迁移，信任不能迁移"）。本工作区恰恰在这点上有积累（`JoinStep.proven`、Semantica PROV-O），应作为自研差异化重点，并在 Ossie 导出时挂在 `custom_extensions` 下。

---

## 1. 证据基线

| 项 | 结论 | 来源 |
|---|---|---|
| HugeGraph 版本 | 1.7.0（TinkerPop 3.5.1，JDK 11 锁死） | `incubator-hugegraph/pom.xml:90`；`docs/HG_SEMANTICA_INTEGRATION.md:14` |
| Gremlin 通道 | 可用。`POST /gremlin` + `aliases:{"g":"__g_DEFAULT-hugegraph"}` + 响应 gzip | `docs/HG_SEMANTICA_INTEGRATION.md:25-28` |
| 索引能力 | secondary / range(int,float,long,double) / shard / unique；**search（全文）not supported；无向量** | `hugegraph-core/.../type/define/IndexType.java:22-41` |
| ID 策略 | 采用 `CUSTOMIZE_STRING`，顶点 id == 业务逻辑 id | `docs/HG_SEMANTICA_INTEGRATION.md:127-138` |
| neocarta 状态 | Neo4j Labs 实验项目，含 OSI 双向连接器，eval 未完工 | `neocarta/README.md:7-12`；`neocarta/eval/README.md:3` |
| 本地 KG 后端 | `HugeGraphStore` 适配器已跑通端到端（`ALL CHECKS PASSED`） | `docs/HG_SEMANTICA_INTEGRATION.md:208-215` |

---

## 2. 现状盘点

### 2.1 已具备，可直接复用

| 资产 | 位置 | 用途 |
|---|---|---|
| Schema Graph 元模型 | `nl2sql/schema_graph/model.py` | Table/Column/Term 三类节点；`BELONGS_TO / FOREIGN_KEY / LINEAGE / CO_OCCUR / TERM_MAPS` 五类边；`EDGE_JOIN_WEIGHT` 权重表 |
| JOIN 路径发现 | `nl2sql/join_path/path_finder.py` | `shortest_path`（Dijkstra）、`connect`（Steiner 2-approx）、`domains`（社区发现）、`JoinStep.proven` 判定 |
| 图计算引擎抽象 | `nl2sql/engine/{base,local,vermeer}.py` | `GraphEngine` 接口：networkx 本地 / Vermeer `weighted_sssp` 分布式 |
| Schema 召回 | `nl2sql/linking/schema_linker.py` | 术语 → 表/列 绑定 |
| 向量检索 | `nl2sql/vector_store.py` | `SchemaVectorStore` 抽象 + Numpy / Milvus / OceanBase 三种实现，**已绕开 HG 无向量索引的问题** |
| HugeGraph 取 schema | `nl2sql/hugegraph_schema_source.py` | 从 live KG 拉 Table/Field/Metric/Query，可配 label 映射、`*_id` 弱外键推断、query log 共现挖掘、本地缓存 |
| 评测 | `nl2sql/evaluation/evaluator.py` | 已有的 NL2SQL 评测骨架 |
| MCP Server | `nl2sql/../servers/mcp_server.py` | 已有 `hg_schema / gremlin_query / vertex_search / edge_search / k_neighbor / text2gremlin / rag_query / olap_query`，且已实现 `_bm25_search` + `_rrf_fusion` |
| 本体/约束 | `semantica/semantica/ontology/` | SHACL 校验、OWL 生成、SKOS 词表（约束层候选） |
| 图存储适配 | `semantica/semantica/graph_store/hugegraph_store.py` | 已验证的 HG CRUD / Gremlin / shortest_path |

### 2.2 需改造

| 项 | 现状 | 改造方向 |
|---|---|---|
| 元模型 | 只有 Table/Column/Term，无 Database/Schema/Metric 表达式/Join 一等公民 | 扩为 §5 的元模型，兼容 Ossie 概念 |
| 召回 | 单一向量召回为主 | 改为多路召回 + 图内扩展 + token 预算裁剪（§6） |
| 上下文装配 | 无 token 预算概念 | 新增 `TableContext` 装配器 + token 预算器 |
| MCP 语义层工具 | 现有 MCP 是"图查询工具"，不是"语义层上下文工具" | 新增语义层 tool 组（§6.4） |
| 写入校验 | 无 | 接 SHACL，写入即校验（§8） |

### 2.3 缺失，需自研

1. **Connector 契约**（extract/transform/load/ingest + 幂等 ID + 可选依赖懒加载）——照搬 neocarta `connectors/_base.py` 的 `SourceConnectorProtocol`。
2. **子图裁剪与 token 预算器**——neocarta 也没有显式预算器，这是本方案的增量。
3. **信任层**（confidence / lineage / freshness / provenance）——Ossie 不管，业界空白。
4. **Ossie 双向映射**——可参考 `neocarta/connectors/osi/`，但需按 HugeGraph schema 重做。
5. **可复现评测基线**——必须自建（§11）。

---

## 3. 对文章建议的逐条评估

| # | 文章建议 | 结论 | 依据与修正 |
|---|---|---|---|
| 1 | neocarta 是最直接可参考的图原生语义层实现路径 | **采纳（架构层）** | 连接器契约、MCP 工具分级注册、TableContext 装配、OSI 双向连接器，四项直接照搬 |
| 2 | 把 neocarta 的 Cypher 改写为 Gremlin，用 `repeat().until().path()` 实现 JOIN 最短路径与子图裁剪 | **部分采纳，需修正** | 语法可移植，但两点不成立：① HugeGraph 无向量/全文索引，`CALL db.index.vector.queryNodes` 无对应物；② 无 `OPTIONAL MATCH`/`COLLECT{}`，表上下文无法在查询内装配。且 TinkerPop `shortestPath()` 是**无权**的，带权最短路必须留在 `GraphEngine`（networkx Dijkstra / Vermeer weighted_sssp）——这与现有 `join_path` 抽象天然契合 |
| 3 | token 降 20–30%、简单查询最高 10×、多表 JOIN 准确率 +10pp | **不采纳为决策依据** | `neocarta/eval/README.md:3` 原文："The evaluation suite is not yet complete and should not be used as a reference." 仓库内无基线数据、无数据集、无脚本产出 |
| 4 | 用 Cube 或 MetricFlow 的 YAML 建模做约束层，再同步写入 HugeGraph | **不采纳** | 双写一致性 + 两套语义（YAML vs 图）维护成本过高；本地已有 Term/Metric 建模与 `hugegraph_schema_source`。改为 HugeGraph 单一 SoT + SHACL 约束 + Ossie 仅导出 |
| 5 | 对接 Apache Ossie 避免"又一个孤岛语义层" | **采纳，但后置到 M5** | Ossie 2026-07 才进孵化器，spec 未稳定；neocarta 的 OSI 连接器可作字段映射参考（`OsiTable/OsiColumn/Metric/Join/Expression/Aspect`） |
| 6 | 约束层 / 检索层 / 生成层三层框架 | **采纳** | 与现有代码可直接映射，见 §4 |
| 7 | （文章未提）信任层 | **新增** | Ossie 不标准化 confidence/lineage/freshness；本地已有 `proven` 语义与 Semantica PROV-O，是最有壁垒的一层 |
| 8 | （文章未提）本地已有 `nl2sql/` 资产 | **新增** | 复用率 > 60%，这是本方案相对"照搬 neocarta"的最大成本节省 |

---

## 4. 目标架构

```
┌─ L4 消费层 ────────────────────────────────────────────────┐
│  MCP Server（语义层 tool 组）  ·  REST API  ·  CLI          │
└────────────────────────────────────────────────────────────┘
┌─ L3 生成层 ────────────────────────────────────────────────┐
│  裁剪后 TableContext → Prompt 装配 → SQL/Gremlin 生成      │
│  JOIN 约束注入：仅 proven join 渲染 ON 子句                │
│  执行反馈 → Query 节点（回流 L1）                          │
└────────────────────────────────────────────────────────────┘
┌─ L2 检索层 ────────────────────────────────────────────────┐
│  ① 候选生成：外置 ANN（Numpy/Milvus）+ 术语精确匹配        │
│              + BM25（mcp_server 已有）+ RRF 融合           │
│  ② 图内扩展：Gremlin 二跳子图（≤2 hop）                    │
│  ③ 裁剪装配：Steiner 连通 + token 预算器 → TableContext    │
└────────────────────────────────────────────────────────────┘
┌─ L1 建模层 ────────────────────────────────────────────────┐
│  Connector 契约（extract/transform/load/ingest）           │
│  SHACL/SKOS 约束校验  ·  幂等 ID  ·  版本快照              │
└────────────────────────────────────────────────────────────┘
┌─ L0 存储层 ────────────────────────────────────────────────┐
│  HugeGraph 1.7.0（CUSTOMIZE_STRING id / secondary+range）  │
│  外置向量库（Numpy → Milvus）                              │
└────────────────────────────────────────────────────────────┘
```

三层框架与现有代码的映射：

| 层 | 现有资产 | 本次增量 |
|---|---|---|
| 约束层 | `schema_graph/model.py`、`semantica/ontology/`（SHACL/SKOS） | 元模型扩展 + 写入即校验 + 版本快照 |
| 检索层 | `linking/`、`vector_store.py`、`engine/`、`hugegraph_schema_source.py` | 多路召回 + 图内扩展 + token 预算器 |
| 生成层 | `pipeline.py`、`join_path/path_finder.py`、`llm_gateway.py` | TableContext prompt 化 + proven join 强制注入 |
| 治理层 | `observability.py`、Semantica PROV-O | confidence / lineage / freshness 一等公民 |

---

## 5. 元模型：HugeGraph schema 落地

### 5.1 顶点标签

| 标签 | id 策略 | id 样例 | 关键属性 |
|---|---|---|---|
| `Database` | CUSTOMIZE_STRING | `db:dw` | name, platform, service |
| `Schema` | CUSTOMIZE_STRING | `schema:dw.public` | name, description |
| `Table` | CUSTOMIZE_STRING | `table:dw.orders` | name, database, schema, comment, row_count, is_fact, freshness_ts, confidence, owner |
| `Column` | CUSTOMIZE_STRING | `column:dw.orders.amount` | name, table, data_type, comment, is_primary_key, is_foreign_key, nullable, is_time_dimension, sample_values(LIST, ≤5), confidence |
| `BusinessTerm` | CUSTOMIZE_STRING | `term:月活` | name, description, aliases(LIST), category, synonyms(LIST), source |
| `Metric` | CUSTOMIZE_STRING | `metric:MAU` | name, description, expression, dialect, grain, unit, owner, confidence |
| `Join` | CUSTOMIZE_STRING | `join:orders.customer_id->customers.id` | from_columns(LIST), to_columns(LIST), cardinality, proven(BOOLEAN), source |
| `Query` | CUSTOMIZE_STRING | `query:<content hash>` | content, exec_count, last_seen_ts, schema_refs(LIST) |
| `Domain` | CUSTOMIZE_STRING | `domain:3` | name, description, resolution |

设计取舍（相对 neocarta 的简化）：
- **不建 `Value` 节点**。neocarta 用 `Column -HAS_VALUE-> Value` 存样本值，在 60 亿点边规模下会爆炸；改为 `Column.sample_values` 列表属性，前 5 个样本值内联，且可关闭（`value_sample_limit=0`）。
- **不建 `Glossary`/`Category` 节点**（neocarta 为兼容 Dataplex 引入），用 `BusinessTerm.category` 属性替代，减少一跳遍历。

### 5.2 边标签（**M0 必须一次性建全**）

| 边 | 起 → 止 | 属性 | join cost | 说明 |
|---|---|---|---|---|
| `HAS_SCHEMA` | Database → Schema | — | — | 容器 |
| `HAS_TABLE` | Schema → Table | — | — | 容器 |
| `HAS_COLUMN` | Table → Column | — | 0.5 | 结构归属 |
| `REFERENCES` | Column → Column | proven | 1.0 | 声明外键，最强 join 证据 |
| `LINEAGE` | Table → Table | job_id | 1.5 | 血缘，上游→下游 |
| `CO_OCCUR` | Table ↔ Table | weight | 3.0 | query log 共现挖掘 |
| `TABLE_TAGGED_WITH` | Table → BusinessTerm | — | ∞ | 术语挂载，不可 join 穿越 |
| `COLUMN_TAGGED_WITH` | Column → BusinessTerm | — | ∞ | 术语挂载，不可 join 穿越 |
| `TERM_MAPS` | BusinessTerm → Column | — | ∞ | 术语→列绑定（等价于 neocarta `DEFINES`） |
| `HAS_EXPRESSION` | Metric → Column | dialect | — | 指标计算依赖 |
| `SYNONYM` | BusinessTerm ↔ BusinessTerm | — | — | 同义术语 |
| `USES_TABLE` / `USES_COLUMN` | Query → Table / Column | use_count | — | 反馈回流边 |

`EDGE_JOIN_WEIGHT` 直接沿用 `nl2sql/schema_graph/model.py:81-87` 的现成取值，保证与现有 `path_finder` 行为一致。

**红线（M0 实测确认）**：HugeGraph 边标签首次创建即锁定 `source_label → target_label`（见 `docs/HG_SEMANTICA_INTEGRATION.md:202-204`），**一个标签只能承载一种端点组合**。因此 neocarta 的单一 `TAGGED_WITH` 在 HugeGraph 上无法同时表示 Table→BusinessTerm 与 Column→BusinessTerm，必须拆成 `TABLE_TAGGED_WITH` / `COLUMN_TAGGED_WITH` 两个标签。M0 必须把上表全部 13 个边标签一次建齐，否则后期要删 label 重建并迁移已有边。

### 5.3 索引（M0 建齐）

| 索引 | 类型 | 字段 |
|---|---|---|
| `table_by_name` | secondary | Table.name |
| `column_by_name` | secondary | Column.name |
| `term_by_name` | secondary | BusinessTerm.name |
| `metric_by_name` | secondary | Metric.name |
| `table_by_row_count` | range_long | Table.row_count |
| `table_by_freshness` | range_long | Table.freshness_ts |
| `cooccur_by_weight` | range_double | CO_OCCUR.weight |

任何被 `.has()` 过滤的属性都必须有索引，否则抛 `NoIndexException`（`docs/HG_SEMANTICA_INTEGRATION.md:34`）。**不要依赖 `search` 全文索引**，源码标注 not supported。

### 5.4 属性类型

属性键类型一旦创建不可变（`docs/HG_SEMANTICA_INTEGRATION.md:98-102`）。M0 定死并在写入侧做 coerce（复用 `HugeGraphStore._coerce_props_to_schema()` 的思路）：

```
TEXT: name/description/comment/expression/dialect/owner/source/category/job_id/content
LONG: row_count/freshness_ts/exec_count/last_seen_ts
DOUBLE: confidence/weight
BOOLEAN: is_primary_key/is_foreign_key/nullable/is_time_dimension/is_fact/proven
LIST(TEXT): aliases/synonyms/sample_values/schema_refs/from_columns/to_columns
```

**BOOLEAN 前置修复（已做）**：`SchemaManager._apply_data_type` 原本对 `BOOLEAN` 只打 `log.error` 而不调用任何 `builder.asXxx()`，导致属性键无类型、创建必然失败——但 `PropertyDataType` 枚举与 pyhugegraph 的 `asBool()` 都支持该类型，属明确的漏分支。已在 `feat/graph-native-semantic-layer` 分支补上 `builder.asBool()`，实测 `is_primary_key` 以 `BOOLEAN` 落库成功。

### 5.5 M0 落地状态（2026-09-07 实测）

代码位于 `incubator-hugegraph-ai`（分支 `feat/graph-native-semantic-layer`）：

```
hugegraph-llm/src/hugegraph_llm/semantic_layer/
├── enums.py                 # 顶点/边标签 + EDGE_JOIN_COST
├── schema_def.py            # 属性键/标签/索引声明 + coerce + 自校验
├── bootstrap.py             # 幂等建图 + verify 报告
└── connectors/base.py       # M1 连接器契约骨架
hugegraph-llm/src/tests/semantic_layer/     # 29 个单测
```

复用 `operators/hugegraph_op/schema_manager.py` 的幂等 `ensure_schema`（probe-then-create），未自写 REST 客户端。

对运行中 HugeGraph（8081，新建测试图 `semantic_layer_m0_test`）连跑两次的结果：

| 检查项 | 结果 |
|---|---|
| 顶点标签 | 9/9 全部创建 |
| 边标签 | 13/13 全部创建（含拆分后的两个 TAGGED） |
| 索引 | 12 个（9 个自动 `{label}ByName` + 3 个显式 RANGE，含边索引 `cooccurByWeight`） |
| 第二次运行 `created` | `{"property_keys":0,"vertex_labels":0,"edge_labels":0,"index_labels":0}` —— **幂等** |
| `verify().ok` | `True`（无缺失标签、无端点错配、无缺失索引） |
| 回归 | 既有 `test_schema_manager.py` 73 个测试仍全通过 |

### 5.6 M1 落地：以已有 HugeGraph KG 为输入端（2026-09-07 实测）

**前提变更**：当前环境**没有可连接的 warehouse**，原计划的 warehouse catalog 连接器失去输入端。但语义层的元数据来源不必是实时数仓——组织已有的 HugeGraph KG（`kg_rag`）里已经存着表、字段、指标及其绑定，语义层应当从 KG 引导，而不是等数仓接进来。

`kg_rag` 实测规模：12 表 / 80 字段 / 41 个 Metric 标签顶点 / 4 个 Query 标签顶点，73 条 `hasColumn`、11 条 `computedFromField`、4 条 `synonym`，且带中文注释（`地区表`、`所在城市`）。

**实测暴露的两个数据陷阱（已在连接器中处理）**：

1. **标签名不可信**。`Metric` 标签下混装两类对象：11 个带 `definition` 的是真业务指标（营收、客单价、毛利额…），30 个只有 `id/name` 且形如 `dim_user.register_at` 的是字段引用；`Query` 标签下装的其实是 `metric:GMV` 这类指标引用，根本没有 `schema_refs`。因此分类**按属性形状判定，绝不按标签名**。
2. **关系可能整类缺失**。`kg_rag` 无 `lineage` 边，且 11 个指标的 `formula` **全为空字符串** → `LINEAGE` 与 `HAS_EXPRESSION` 为空。这在 summary 里如实报 0，不静默当成成功。

**两个能力缺口（诚实记录，非 bug）**：

| 缺口 | 后果 | 应对 |
|---|---|---|
| 指标无 `formula`/`expression` | `HAS_EXPRESSION` 建不起来，指标无法自动展开为可计算表达式 | 需人工补录或后续从 SQL 日志反推；当前术语只能做"术语→列"绑定（`TERM_MAPS`） |
| 无真实 query log | `CO_OCCUR` 无法挖掘；`lineage` 也为 0 | JOIN 路径目前**只能靠 `*_id` 弱外键推断**（实测推断出 17 条，全部标记 `proven=false`） |

推断出的外键一律 `proven=false`，下游生成层不得当作声明式完整性渲染（沿用 `JoinStep.to_sql()` 的既有语义：unproven 输出注释而非 `ON` 子句）。

**连接器**：`semantic_layer/connectors/hugegraph_source.py`（`HugeGraphSourceConnector`），走 §2.3 的 `SourceConnector` 契约。迁移结果：

| 项 | 数量 |
|---|---|
| Table / Column | 12 / 73 |
| BusinessTerm / Metric | 11 / 4 |
| `HAS_COLUMN` / `TERM_MAPS` / `REFERENCES` / `SYNONYM` | 73 / 11 / 17 / 1 |
| `LINEAGE` | 0（源图无血缘） |
| 落库对象总数 | 202 |

中文语义绑定完整保留，例：`动销 → ads_daily_sales.sell_through_rate`、`结算 → payments.settlement_amount`、`履约 → fulfillment_hours`。

**顺带修掉的客户端坑**：`pyhugegraph` 的 `getEdgeByPage`/`getVertexByPage` 必须**用关键字参数调用**——位置传参会把 `page` 填进 `direction`，触发 `NotFoundError("Direction can not be empty")`。已在连接器内固定为关键字调用并加注释。

**后续连接器**：

| 连接器 | 状态 | 说明 |
|---|---|---|
| Ossie / OSI YAML | **已完成**（见 §5.7） | 唯一可离线全量验证的连接器，33 表规模往返无损 |
| warehouse catalog | **后置** | 等有数仓可连时再做，接口已预留 |

### 5.7 M1 落地：Ossie 双向连接器（2026-09-07 实测）

`semantic_layer/connectors/ossie.py`，用 neocarta 自带 `datasets/osi/acme_semantic_model.yaml`（33 表 / 323 字段 / 54 关系 / 9 指标 / 51 同义词）离线验证，**无需任何外部连接**——这是唯一能做规模化、可进 CI 的连接器。

| 项 | 结果 |
|---|---|
| 导入 | 970 对象（33 Table / 323 Column / 54 Join / 51 BusinessTerm / 9 Metric） |
| 边 | 323 `HAS_COLUMN` + 54 `REFERENCES` + 16 `HAS_EXPRESSION` + 51 `TABLE_TAGGED_WITH` |
| 导出往返 | **零差异**：33 dataset / 323 字段 / 33 PK / 54 relationship（列顺序一致）/ 9 metric（表达式全等）/ 55 个 `is_time` 三态 |
| trust 外挂 | `source_system` 以 `custom_extensions[ossie-hugegraph]` 形式保留 |

**相对 neocarta 的 HugeGraph 适配**：HugeGraph 无多标签顶点，neocarta 的 `OsiTable`/`OsiColumn` 次级标签做法不可行；改用 `source_system` 属性 + **id 按模型名命名空间**（`acme_corp_model:table:offices`），使多个模型可共存于一张图。

**实测踩到的三个坑**（均已修，且都是"小样本跑通、大数据量才炸"的类型）：

1. **pyhugegraph 分页 bug**：`getVertexByPage`/`getEdgeByPage` 在 `page=None` 时拼出空参数 `?label=X&page&limit=500`，HugeGraph 1.7 返回 **500**。12 表的 `kg_rag` 侥幸跑通，323 列的 ACME 直接失败。已在 `semantic_layer/paging.py` 改用 Gremlin `range()` 分页绕过。
2. **`elementMap()` 返回不能 unwrap**：我最初把单元素列表当标量拆包，导致 LIST 属性 `from_columns: ["opportunity_id"]` 变成字符串，导出时 `list(...)` 将其炸成 `["o","p","p",...]`。已改为原样返回（SINGLE→标量、LIST→列表）。
3. **属性未声明**：`source_system` 未列入 `Metric` 标签声明，`lineage_ref` 全局缺失 → 9 个 Metric 顶点写入全部失败（`bootstrap.verify()` 原本只查标签存在，查不出属性缺失）。已补齐 trust 四件套，并**增强 `verify()` 检查标签声明属性是否齐全**。

前两个坑的共同教训：小样本验证会掩盖分页与类型问题，M4 评测必须用 33 表规模而非 12 表。

### 5.8 M2 落地：检索与裁剪（2026-09-07 实测）

模块位于 `semantic_layer/`：

| 模块 | 职责 |
|---|---|
| `context.py` | `TableContext` / `ColumnContext`，字段名兼容 neocarta 契约；三级渲染 `full` / `keys_only` / `name_only`；CJK 感知的 token 估算 |
| `budget.py` | token 预算器：超预算即降级，绝不超限；stale 表排最后但不丢弃 |
| `bm25.py` | 进程内 BM25（HG 无全文索引，`mcp_server._bm25_search` 是 `NotImplementedError` 占位） |
| `fusion.py` | **n 路** RRF（现有 `_rrf_fusion` 只支持两路，而本层是三路召回） |
| `readers.py` | 全部 Gremlin 收敛于此；输出 `SemanticProjection` 纯 Python 结构 |
| `retrieval.py` | 编排：召回 → 扩展 → 连通 → 装配 → 预算 |

**对原方案的两处修正**：

1. **加载整个 join projection，而非服务端遍历。** ACME 33 表 323 列 ≈ 400 节点，分页拉取只花 0.1s，换来的是扩展逻辑变成可测的 Python BFS。方案原文设想的 Gremlin 服务端 2 跳遍历推迟到投影超过约 1 万节点时再做，改动点集中在 `GremlinSemanticReader.expand`。
2. **连通性用 BFS 保证，Steiner 留作可选。** `GraphEngine.steiner_join_tree` 已具备，但 BFS 从最强种子出发已能给出"单一连通分量"这一核心保证，且无外部依赖。

**33 表实测**（`semantic_m2b`，内存后端）：

| 指标 | 结果 |
|---|---|
| 投影加载 | 0.1s（33 表 / 323 列 / 51 术语 / 54 引用） |
| 单次检索 | **2ms** |
| token 削减 | 检索 12 表 887 tokens vs 全量 33 表 2194 tokens，**省 60%** |
| 预算阶梯 | 4000→12 表全 full；800→10 full + 1 keys_only；500→7 full + 3 name_only；60→1 表 keys_only。**任何档位都不超预算** |
| 术语多跳 | `ARR`/`MRR`→`subscriptions`、`MQLs`→`leads`、`CSAT`→`support_tickets`、`FTEs`→`employees` |

**实测暴露的四个 bug**（都已修，前三个会静默产生错误结果）：

1. **外键邻接是单向的**。`orders.customer_id → customers.id` 只存一个方向，从 `customers` 出发 BFS 找不到 `orders`，导致可 join 的表被当成不可连通而丢弃——恰好是"幻觉 JOIN"的反面错误。已改为对称邻接。
2. **`_expand` 读 `self.config` 而非传入的 `cfg`**，导致调用方覆盖 `hops` 静默失效。已改为显式传参。
3. **CJK token 被丢弃**。`_maybe_split_camel` 的正则只匹配 ASCII，中文 token 全部落空，中文元数据做不了字面召回。已加 CJK 分支。
4. **指标同义词是孤儿**。ACME 的 51 个术语里只有 13 个连到表，指标侧的 38 个没有任何边（我原先传 `edge_label=None`），`FTEs` 这类缩写查不出任何表。已新增 `METRIC_TAGGED_WITH` 边，并打通 **术语 → 指标 → HAS_EXPRESSION → 列 → 表** 四跳路径——这正是"图原生"相对扁平 schema 文件的价值所在。

**降级策略的设计取舍**：`keys_only` 会丢弃既非主键又无外键的普通列。这是有意的——该级别的目标只是让模型能写出 `ON` 子句，渲染一堆裸列名会让它误以为表只有这些列（`num_columns` 会如实反映所见列数）。

### 5.9 M3 落地：MCP 工具层（2026-09-07 实测）

两个新模块：`join_path.py`（JOIN 路径求解）与 `mcp_tools.py`（工具层）。

**关键设计：传输层无关。** 本环境**未安装 `mcp` SDK**，且它不在项目依赖里——若在模块顶部 `import mcp`，整个语义层会变得不可导入。因此工具层是"纯函数 + JSON Schema 描述"，挂载到 `HugeGraphMCPServer`（或任何 MCP SDK server）只是薄适配层，写在 SDK 真实存在的地方。`mcp_tools.py` 本身**零 MCP 依赖**，测试无需 SDK。

**能力探测（沿用 neocarta 模式）**：启动时不注册所有工具让 Agent 自己试错，而是先探测图谱实际能力，只注册能跑通的工具。上下文工具**按能力命名**，所以 Agent 光看工具列表就知道自己拿到的是哪种召回：

| 能力组合 | 注册的上下文工具 |
|---|---|
| 向量 + 术语 | `get_context_by_term_hybrid_search` |
| 仅向量 | `get_context_by_table_hybrid_search` |
| 仅 BM25 | `get_context_by_table_full_text_search` |
| 都没有 | `get_context_by_table_lookup` |

常驻工具：`list_schemas`、`list_tables_by_schema`、`get_table_columns`、`get_full_metadata_schema`（标记 expensive）、`get_join_path`；有术语时追加 `search_business_terms`。

**ACME 33 表实测**：

```
能力探测: tables=33, columns=323, business_terms=yes, vector_index=no, lexical_index=yes
注册工具: get_context_by_table_full_text_search, list_schemas, list_tables_by_schema,
         get_join_path, get_table_columns, get_full_metadata_schema, search_business_terms
```

- **术语召回**（命中真实词表时很准）：`active employees`→employees、`monthly recurring revenue`→subscriptions、`headcount`→employees、`ARR`→subscriptions、`FTEs`→employees、`CSAT`→support_tickets
- **`get_join_path`**：`performance_reviews→employees` 1 跳、`employees→departments` 1 跳、`campaigns→job_titles` **2 跳经 employees**（全部 `proven=True`）；对不存在的表 `employees→accounts` 如实返回 NOT FOUND 并提示"不要臆造 JOIN"
- **截断诚实**：`get_full_metadata_schema(max_tokens=1200)` 返回 17/33 表并置 `truncated=True`
- **性能**：单次检索 2–3ms

**必须诚实记录的召回缺口**：`vector_index=no` 意味着当前处于最弱召回档。用**模型真实词表**提问时很准，但**改写说法**会明显退化——例："monthly active users" 在该模型里并非术语（真实术语是 `active employees` / `monthly recurring revenue`），只能靠 BM25 猜，结果偏到 teams/products。这个缺口已由工具名暴露给 Agent，接上 Milvus 后即进入 `..._term_hybrid_search` 档位。这也是 §11 评测必须自建的核心理由之一。

### 5.10 M4 落地：评测体系（2026-09-07 实测）

模块 `semantic_layer/evaluation/`：`dataset.py`（数据集 + 从 gold SQL 自动抽取 gold 表）、`metrics.py`（四项指标）、`runner.py`（编排 + baseline）、`cli.py`（CI 入口）。数据集 `semantic_layer/resources/acme_eval.jsonl`（45 例：20 术语 / 15 schema / 10 改写，其中 20 例多表）。

**设计要点**：

- **gold 表从 gold SQL 自动抽取**，不手工标注。手工标注会和 SQL 漂移（标注 2 张表而 SQL join 了 3 张），且会安静地奖励"少召回"。
- **每例标注 `source`**（`term` / `schema` / `free`）。区分"模型认识的词汇"与"改写说法"至关重要——合并统计会掩盖真实覆盖面，这是 benchmark 自我高估的典型方式。
- **baseline 是全量 schema 注入**（无语义层的做法），而不是另一个检索系统。这才是"语义层是否值回票价"的诚实对照。
- **`not_measured` 字段显式列出测不了的东西**：`execution_accuracy`（需数仓跑 gold SQL）、`end_to_end_sql_correctness`（取决于生成模型）。宁可留空也不填无法复现的数字。

**CLI（可进 CI）**：

```bash
python -m hugegraph_llm.semantic_layer.evaluation.cli \
    --dataset .../resources/acme_eval.jsonl --graph semantic_m2b \
    --output report.json --fail-under "recall@5=0.90"
```

已验证三种门禁行为：达标 exit 0、不达标 exit 1 并打印实际值、非法指标名报错并列出可选项。

**45 例实测结果（ACME 33 表）**：

| 指标 | 值 |
|---|---|
| recall@5 | 0.963 |
| all gold tables found | 1.000 |
| retrieved set joinable | **1.000** |
| gold set joinable | 1.000 |
| business-term recall fired | 0.356 |
| mean tokens（检索） | 841 |
| mean tokens（全量） | 2194 |
| **token saving** | **61.7%** |
| 多表子集（20 例） | recall@5 0.917，joinable 1.000 |

**评测暴露的真问题：precision@5 只有 0.293。**

召回很高、joinable 100%、success 1.000 看着漂亮，但 P@5 0.293 说明**检索了约 12 张表才覆盖约 3.5 张 gold 表**——扩展过于激进。这正是"只看召回"会漏掉的问题：success 指标在过度召回下会被平凡满足（gold 表碰巧都在里面），而模型实际拿到的是一堆噪声表。

配置扫描（同一数据集）：

| 配置 | P@5 | R@5 | allGold | tokens | saving |
|---|---|---|---|---|---|
| 默认 top8/hop2/max12 | 0.293 | **0.963** | **1.000** | 841 | 61.7% |
| hop0/max12 | 0.416 | 0.822 | 0.800 | 265 | 87.9% |
| hop1/max6 | 0.319 | 0.833 | 0.844 | 328 | 85.1% |
| **top5/hop1/max8** | 0.289 | 0.941 | 0.978 | **565** | **74.2%** |
| hop1/max4 | 0.424 | 0.793 | 0.733 | 224 | 89.8% |

**当前保留默认（top8/hop2/max12）的理由**：召回优先于 token 成本。表缺失会直接导致查询无法作答，而多余的表只会多花 token——并且 token 侧还有预算器兜底（可降级），表缺失则无补救。这是"宁可多召回再裁剪"的取舍，不是疏忽。

若部署场景 token 吃紧，`top5/hop1/max8` 是更优点：少 2.2pp 召回换 33% token。但**该结论只在 ACME 一个数据集上验证过**，改默认前应在第二个数据集上复核，避免对单数据集过拟合。

**待办**：precision 是下一个要攻的指标（0.29 偏低）。方向是给扩展加分数阈值（`hop_decay` 之外再设绝对下限），而非继续调 `max_tables`。

---

## 6. 检索层：三段式召回与子图裁剪

### 6.1 ① 候选生成（外置 ANN + 术语 + BM25 + RRF）

HugeGraph 无向量索引，向量只能外置。复用 `nl2sql/vector_store.py` 的 `SchemaVectorStore` 抽象（Numpy 起步，Milvus 生产），向量库里只存 `(node_id, vector)`，召回后拿 `node_id` 回图做遍历。

```python
# 伪代码：候选生成
vec_hits   = store.search(embed(q), top_k=20)              # Milvus/Numpy
term_hits  = hg.gremlin(TERM_LOOKUP, {"names": extract_terms(q)})   # secondary index 精确命中
bm_hits    = bm25_search(q, corpus=table_corpus)           # 复用 mcp_server._bm25_search
seeds      = rrf_fusion([vec_hits, term_hits, bm_hits])    # 复用 mcp_server._rrf_fusion
```

术语桥接检索（neocarta 的 `business_term_hybrid` 核心，一段 Gremlin 即可实现）：

```groovy
// Gremlin：术语 → 挂载的表/列
g.V().has('BusinessTerm', 'name', within($names))
 .union(both('TABLE_TAGGED_WITH', 'COLUMN_TAGGED_WITH'), out('TERM_MAPS').in('HAS_COLUMN'))
 .dedup().limit(50)
 .project('id', 'label', 'name')
 .by(id()).by(label()).by(values('name'))
```

### 6.2 ② 图内扩展（≤2 跳）

表级连通性是"跨表 JOIN"的关键。注意表与表之间通过 Column 桥接：

```groovy
// Gremlin：从种子表做 2 跳扩展，simplePath 防环
g.V().has('Table', 'name', within($seeds))
 .repeat(
    union(
      out('HAS_COLUMN').both('REFERENCES').in('HAS_COLUMN'),  // FK 桥接
      both('LINEAGE'),                                        // 血缘
      both('CO_OCCUR')                                        // 共现
    ).simplePath()
 )
 .times(2).emit()
 .dedup().limit(30)
 .project('name', 'row_count', 'comment')
 .by(values('name'))
 .by(coalesce(values('row_count'), constant(0)))
 .by(coalesce(values('comment'), constant('')))
```

### 6.3 ③ 裁剪装配（Steiner + token 预算）

- **连通性保证**：调用现有 `JoinPathFinder.connect(tables)`（`nl2sql/join_path/path_finder.py:136`）求 Steiner 树，保证返回的表集合**一定可 join**，避免出现 LLM 无法连接的孤立表。
- **带权最优路径留在引擎层**：HugeGraph 的 `shortestPath()` 是无权的，带权 Dijkstra 走 `LocalEngine`（networkx）或 `Vermeer`（`weighted_sssp`）。这是本方案对文章"用 Gremlin 实现 JOIN 最短路径"的**明确修正**。
- **token 预算器**（新增组件，neocarta 无）：

```python
def fit_context(tables: list[TableContext], max_tokens: int) -> list[TableContext]:
    """按 (is_seed, score, row_count) 排序，贪心装填；超预算则降级列集合。"""
    ordered = sorted(tables, key=lambda t: (not t.is_seed, -t.score, -t.row_count))
    out, used = [], 0
    for t in ordered:
        for level in ("full", "keys_only", "name_only"):   # 三级降级
            cost = estimate_tokens(t.render(level))
            if used + cost <= max_tokens:
                out.append(t.render(level)); used += cost; break
        else:
            break
    return out
```

降级规则：`full`（全部列 + 描述 + 样本值）→ `keys_only`（仅 PK/FK/命中列 + 表名）→ `name_only`（仅表名）。

### 6.4 表上下文装配（替代 Cypher 的 `OPTIONAL MATCH` + `COLLECT{}`）

neocarta 在 Cypher 里用 `COLLECT{}` 子查询一次装配出 TableContext（`neocarta/_mcp/cypher/vector_search.py:59-74`）。HugeGraph 无此能力，改为**Gremlin 取行 + Python 侧聚合**：

```groovy
// Gremlin：单表的列 + 外键引用（一次取平铺行）
g.V().has('Table', 'name', $t).out('HAS_COLUMN')
 .project('column_name', 'data_type', 'key_type', 'refs', 'samples')
 .by(values('name'))
 .by(coalesce(values('data_type'), constant('')))
 .by(choose(values('is_primary_key'), constant('primary'),
        choose(values('is_foreign_key'), constant('foreign'), constant(''))))
 .by(both('REFERENCES').in('HAS_COLUMN').values('name').fold())
 .by(coalesce(values('sample_values'), constant([])))
```

Python 侧按 `table_name` 聚合成 `TableContext`（字段命名对齐 neocarta：`table_name / table_description / database_name / schema_name / columns / num_columns / primary_key`），好处是与 neocarta 的 MCP 输出契约兼容，将来可平滑切换。

### 6.5 MCP 工具组

对齐 neocarta 的工具命名与**索引探测降级**机制（`neocarta/_mcp/README.md:125-133`）：启动时探测向量库/术语节点是否就绪，按优先级注册唯一工具，避免 Agent 面对一堆不可用工具：

| 优先级 | 工具 | 前置条件 |
|---|---|---|
| 1 | `get_context_by_term_hybrid_search` | 向量库就绪 + BusinessTerm 节点存在 |
| 2 | `get_context_by_table_hybrid_search` | 向量库就绪 + BM25 语料就绪 |
| 3 | `get_context_by_table_vector_search` | 仅向量库 |
| 3 | `get_context_by_table_full_text_search` | 仅 BM25 |
| — | `list_schemas` / `list_tables_by_schema` | 常驻注册 |
| — | `get_join_path(left, right)` | 常驻注册（**neocarta 没有，本方案的增量**） |
| — | `get_full_metadata_schema` | 常驻注册，标注 expensive，仅调试用 |

`get_join_path` 是本方案相对 neocarta 的增强：直接把 proven join 路径交给 Agent，从根上消除"模型臆造 JOIN"。

---

## 7. 生成层

直接复用 `JoinStep.to_sql()`（`nl2sql/join_path/path_finder.py:68-76`）的既有语义：

- `proven=True` → 渲染 `a.id = b.id`
- `proven=False` → 渲染 `/* unproven join: a <-> b */`，**禁止臆造**

注入 prompt 时把 JOIN 约束作为硬约束单独成段，而不是混在 schema 描述里：

```
## 可用 JOIN（必须严格使用，不得自行推导）
orders.customer_id = customers.id          [FOREIGN_KEY, proven]
orders.product_id  = products.id           [FOREIGN_KEY, proven]
/* unproven: orders <-> shipments [CO_OCCUR]，如需关联请向用户确认 */
```

反馈回流：执行成功的 SQL 写入 `Query` 节点 + `USES_TABLE/USES_COLUMN` 边，异步重算 `CO_OCCUR.weight`（周级批处理），形成"使用越多、召回越准"的正循环——这补上了文章提到的 MetricFlow 局限（"新查询和反馈无法即时进入 YAML"），图结构的优势正在于此。

---

## 8. 约束层与信任层

### 8.1 约束层（SHACL 即校验）

复用 `semantica/semantica/ontology/`（SHACL 校验 + SKOS 词表）。写入即校验，核心约束：

- `Metric` 必须至少一条 `HAS_EXPRESSION` 到 `Column`
- `Column` 的 `is_foreign_key=true` 必须存在 `REFERENCES` 边
- `Table` 必须属于唯一 `Schema`
- `BusinessTerm` 的 `name` 唯一（幂等 MERGE 键）
- `confidence` ∈ [0,1]

不通过则拒绝写入并给出 SHACL ValidationReport，而不是"先写后修"。

### 8.2 信任层（本方案的差异化）

每个 `Table / Column / Metric / BusinessTerm` 携带：

| 字段 | 含义 | 来源 |
|---|---|---|
| `confidence` | 0–1，人工确认=1.0，推断=0.5–0.8，LLM 抽取=抽取置信度 | 连接器 / 人工 |
| `lineage_ref` | 上游系统标识 | 血缘采集 |
| `freshness_ts` | 元数据最后验证时间 | 调度任务 |
| `source_system` | 来源系统 | 连接器 |
| `provenance_uri` | PROV-O 记录 URI | Semantica provenance |

**消费规则**：`confidence < 0.5` 或 `freshness_ts` 超期的节点，在 MCP 返回中降级（排在后面并打 `stale` 标记），绝不静默剔除——让 Agent 知道"这个定义可能过时"，而不是假装它不存在。

---

## 9. 互操作层：Ossie

策略：**先建模对齐，后格式对接**。M5 再做，避免被未稳定的 spec 绑住。

参考 `neocarta/connectors/osi/` 的字段映射：

| Ossie 概念 | 本方案图元素 |
|---|---|
| dataset | `Table`（+ `OsiTable` 兼容标签） |
| field | `Column` |
| relationship | `Join`（保留 `from_columns`/`to_columns` 顺序以支持复合键） |
| metric | `Metric` + `HAS_EXPRESSION`（dialect-specific expression） |
| ai_context / custom extensions | `Aspect` 子类型节点 |

**trust 字段外挂**：`confidence / lineage / freshness / provenance` 在导出时写入 `custom_extensions` 段（对齐 neocarta `OsiCustomExtensions` 的做法），导入时再解析回来。这样既符合 spec，又不丢失信任信息。

---

## 10. 里程碑

假设 2–3 人，按人周估算：

| 阶段 | 交付物 | 工期 | 验收标准 |
|---|---|---|---|
| **M0 元模型与图 schema** | §5 全部顶点/边/索引定义脚本 + 类型 coerce + 一次性建全 | 2 周 → **已完成** | 干净 server 上一键建图；重复执行幂等（实测通过，见 §5.5） |
| **M1 连接器契约** | `SourceConnector`（extract/transform/load/ingest）+ **HugeGraph 源连接器**（替代 warehouse 连接器，见 §5.6）+ 幂等 ID | 3 周 → **已完成** | kg_rag 12 表/73 列/11 术语全量导入，202 对象落库（见 §5.6） |
| **M2 检索与裁剪** | 多路召回 + RRF + 图内 2 跳扩展 + Steiner + token 预算器 | 3 周 → **已完成** | 33 表实测 P95 2ms；预算永不超限（见 §5.8） |
| **M3 MCP Server** | 语义层 tool 组 + 启动探测降级 + `get_join_path` | 2 周 → **已完成** | 33 表实测能力探测 + 7 工具；见 §5.9 |
| **M4 评测体系** | 数据集 + 四项指标 + baseline 对比（与 M2/M3 并行） | 2 周 → **已完成** | 45 例数据集 + CLI + CI 门禁；见 §5.10 |
| **M5 Ossie 双向** | 导入/导出 + trust 外挂 `custom_extensions` | 2 周 | ACME 样例往返无损 |
| **M6 反馈回流** | Query 节点写入 + CO_OCCUR 权重周级重算 | 2 周 | 权重随查询分布变化，召回 Top-5 提升 |

建议排期：M0 → M1 → (M2 ∥ M4) → M3 → M5 → M6，总计约 14 周。

---

## 11. 评测体系（补 neocarta 缺口）

`neocarta/eval/` 是空壳，因此**必须自建基线，且结论只能出自自建评测**。

| 指标 | 定义 | Baseline 对照 |
|---|---|---|
| **召回** | 表级 P@5 / R@5（gold SQL 中的表是否被召回） | BM25 单路召回 |
| **token 成本** | 注入 prompt 的 schema token 数 | `get_full_metadata_schema` 全量注入 |
| **JOIN 准确率** | 生成 SQL 的 JOIN 落在 proven join 集合内的比例 | 不注入 join hint |
| **端到端准确率** | 执行结果集与 gold 一致的比例（可执行率单列） | 现有 `evaluation/evaluator.py` |

数据集：以 `nl2sql/resources/example_warehouse.json` 为起点，补齐 ACME 33 表规模（neocarta 提供了 `datasets/osi/acme_semantic_model.yaml` 可直接转成图），标注 ≥150 条 NL→SQL 对，其中 ≥50 条为多表（≥3 表）JOIN，专门验证 JOIN 收益。

**红线**：任何对外引用的收益数字，必须附本评测脚本的产出，不得引用 neocarta README 的二手数据。

---

## 12. 风险与红线

| 风险 | 影响 | 应对 |
|---|---|---|
| JDK 版本 | JDK 17/21 → `gremlin-groovy not available`，Gremlin 通道整体失效 | 锁 JDK 11，写进 Dockerfile 与 CI |
| Gremlin 调用细节 | 必须带 `aliases`；响应 gzip；ID 数字/字符串双形态 | 统一封装客户端，禁止业务代码裸写 HTTP |
| 属性过滤无索引 | `NoIndexException` | M0 建齐索引；新增 `.has()` 字段须同步加索引（代码评审检查项） |
| 边标签端点锁定 | 后期加端点组合需删 label 重建并迁移边 | M0 一次建全（§5.2）；**实测确认一个标签只能一种端点组合**，故 `TAGGED_WITH` 拆为两个 |
| 属性键类型不可变 | 类型推断错误需重建 | 定死类型 + 写入侧 coerce |
| 无向量/全文索引 | 语义检索能力受限 | 外置 Milvus + BM25；HG 只做结构与精确匹配 |
| `search` 索引标注 not supported | 若误用会静默失败 | 明令禁用 |
| 元数据规模 | 60 亿点边业务图 vs 元数据图 | **元数据图独立部署**（独立的 `hugegraph` 实例或独立 graph space），不与业务图混布 |
| Ossie spec 不稳定 | 早期绑定导致返工 | M5 后置，trust 走 `custom_extensions` |

---

## 13. 关键决策记录（ADR 摘要）

| ID | 决策 | 备选 | 理由 |
|---|---|---|---|
| ADR-1 | HugeGraph 作为语义层唯一 SoT | 引入 Cube/MetricFlow 双写 | 避免双写一致性成本；本地已有建模能力 |
| ADR-2 | 向量外置（Milvus），图内不做 ANN | 等 HG 支持向量索引 | HG 1.7.0 无向量索引，且元数据图规模小，外置更简单 |
| ADR-3 | 带权最短路留在 `GraphEngine` | 用 Gremlin `shortestPath()` | TinkerPop `shortestPath` 无权；现有 Dijkstra/Steiner 已实现且可切 Vermeer |
| ADR-4 | 表上下文在 Python 侧装配 | 用 Cypher 式 `COLLECT` 子查询 | HugeGraph 不支持 `COLLECT{}`/`OPTIONAL MATCH` |
| ADR-5 | 样本值内联为列属性 | 建 `Value` 节点（neocarta 做法） | 避免节点爆炸；保持输出契约兼容即可 |
| ADR-6 | 自建评测，不引用 neocarta 收益数据 | 直接引用 README 数据 | neocarta eval 明确 WIP，无可复现基线 |
| ADR-7 | Ossie 后置到 M5，trust 走 custom_extensions | 早期对齐 spec | spec 未稳；trust 是 Ossie 不管的差异化能力 |
| ADR-8 | 元数据图独立部署 | 与业务图同实例 | 隔离规模与故障域，元数据图是毫秒级查询路径 |
