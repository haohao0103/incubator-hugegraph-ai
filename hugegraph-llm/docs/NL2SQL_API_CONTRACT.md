# NL2SQL 语义层 HTTP 接口对接文档（给上层平台）

> 版本：v1（2026-08-30）｜对接对象：上层 NL2SQL 平台
> 角色定位：本服务是**语义层 / 知识底座**，**不生成 SQL**。平台把自然语言问题发来，
> 我们返回「生成 SQL 所需的确切数据」：相关表/字段/指标（带分数）、可证明的 JOIN 路径、
> 收窄后的 schema 上下文。平台拿这些拼 prompt 喂自己的 SQL 生成模型。

---

## 1. 服务信息

| 项 | 值 |
|---|---|
| Base URL | 按部署环境提供（本机示例 `http://127.0.0.1:8910`） |
| 协议 | HTTP POST + JSON |
| 字符集 | UTF-8 |
| 健康检查 | `GET /healthz`（如挂载） |

### 启动方式（供运维参考）

```bash
# 仅挂 NL2SQL router 的极简服务（示例端口 8910）
PYTHONPATH=hugegraph-llm/src \
  /path/to/hg-llm/python scripts/nl2sql_hg_server.py

# 或使用完整 hugegraph-llm app（NL2SQL router 已挂载）
# NL2SQL_HG_GRAPH=kg_rag NL2SQL_HG_URL=http://127.0.0.1:8081  # 生产默认从 HugeGraph 建 schema
```

---

## 2. Schema 初始化（二选一，服务启动后调用一次）

### 2.1 直灌元数据：`POST /nl2sql/reload`

平台侧已有数据字典/元数据时使用。请求体 `SchemaMetadata`：

```json
{
  "metadata": {
    "tables": [
      {"name": "orders", "database": "dw", "comment": "订单表", "is_fact": true}
    ],
    "columns": [
      {"name": "order_id", "table": "dw.orders", "data_type": "bigint", "comment": "订单编号"},
      {"name": "pay_amount", "table": "dw.payments", "data_type": "decimal", "comment": "支付金额"}
    ],
    "foreign_keys": [["dw.orders.user_id", "dw.users.user_id"]],
    "lineage": [["dw.orders", "dw.ads_daily_sales"]],
    "query_logs": [["dw.orders", "dw.payments"]],
    "terms": [
      {"name": "支付总额", "comment": "支付总额（口径：支付表 pay_amount 汇总）"}
    ],
    "term_bindings": [["支付总额", "dw.payments.pay_amount"]]
  }
}
```

字段说明（**最小可用集 = tables + columns**，其余按以下优先级）：
- `terms / term_bindings`：强烈建议（指标口径统一是核心能力；不填则指标相关问法命中率下降）。
- `foreign_keys`：可选。不填自动按 `*_id` 同名列推断弱外键，JOIN 强证据略少但可用。
- `lineage`：可选但建议。血缘是差异化能力；拿不到也没关系，JOIN 路径仍由 FK + 共现支撑。
- `query_logs`：**可不提供**。共现边是"经验性 JOIN 提示"，仅作弱证据；缺省时 JOIN 路径
  完全由 foreign_keys / lineage 支撑，主链路不受影响。**无需人工整理**——如果平台侧有
  历史 SQL 日志（数仓审计/调度系统一般都有），交给我们，我们写脚本自动解析 SQL、
  提取同查表对生成共现，工作量不在你们侧。连 SQL 日志都没有就整个跳过共现。`

> **强烈建议**：平台侧把**线上任务 SQL 脚本（文本）**一并导出给我们。用
> `scripts/sql_metadata_miner.py`（sqlglot 解析）可从 SQL 自动产出：
> `lineage`（INSERT/CTAS 目标表 → 血缘）、`foreign_keys`（JOIN/WHERE 等值条件 →
> 证明级 JOIN 键，比 `*_id` 推断强）、`query_logs`（同查表对加权共现）。
> 这样平台只需手给 tables/columns/terms 三样，其余全自动。

### 2.2 从 HugeGraph 拉取：`POST /nl2sql/load_hugegraph`

元数据已灌入 HugeGraph 图（如 `kg_rag`，含 Table/Field/Metric/Query 顶点与
hasColumn/computedFrom/computedFromField/dependsOn 边）时使用：

```json
{
  "url": "http://127.0.0.1:8081",
  "graph": "kg_rag",
  "infer_foreign_keys": true,
  "use_embedding": false
}
```

> `use_embedding: true` 开启 P2 向量语义召回（总额≈金额 类同义匹配），需服务侧配置好
> embedding 后端（本地 bge-small-zh-v1.5，512 维，离线可用）。

---

## 3. 查询接口（平台侧核心调用面）

### 3.1 L1 表/字段检索：`POST /nl2sql/link`

问题 → 最相关的表与字段（带 PPR 分数）。

```json
{"question": "支付总额是多少", "top_k": 10}
```

响应：

```json
{
  "question": "支付总额是多少",
  "items": [
    {"node_type": "column", "name": "pay_amount", "score": 0.142,
     "table": "dw.payments",
     "properties": {"table": "dw.payments", "data_type": "decimal",
                    "comment": "支付金额", "is_primary_key": false, "is_foreign_key": false}},
    {"node_type": "table", "name": "payments", "score": 0.131, "table": "",
     "properties": {"database": "dw", "comment": "支付表", "row_count": 0, "is_fact": true}}
  ]
}
```

平台用法：取 `items` 前 3-5 个作为候选 schema 元素。

### 3.2 L2 JOIN 路径：`POST /nl2sql/join_path`

两表间最短 JOIN 路径，带可证明的 ON 子句。

```json
{"source": "dw.orders", "target": "dw.payments"}
```

响应：

```json
{
  "source": "dw.orders",
  "target": "dw.payments",
  "path": {
    "tables": ["dw.orders", "dw.payments"],
    "total_cost": 1.0,
    "all_proven": true,
    "steps": [
      {"left_table": "dw.orders", "right_table": "dw.payments",
       "left_column": "dw.orders.order_id", "right_column": "dw.payments.order_id",
       "edge_type": "foreign_key", "cost": 1.0, "proven": true,
       "on_clause": "dw.orders.order_id = dw.payments.order_id"}
    ]
  }
}
```

平台用法：跨表问题时，把 `on_clause` 直接拼进 FROM/JOIN。`proven: true` = 声明外键（强证据）；
`proven: false` = 共现推断（弱证据，需要业务确认）。

### 3.3 L3 表聚类：`POST /nl2sql/communities`

把库切成业务域（louvain），用于导航/推荐。

```json
{"resolution": 1.0, "algorithm": "louvain"}
```

响应：

```json
{
  "domains": {"dw.orders": 0, "dw.payments": 0, "dw.users": 1}
}
```

### 3.4 收窄 schema 上下文：`POST /nl2sql/schema_context`

问题 → 收窄后的 schema 文本（可直接进 LLM prompt 的 SuperSonic 式视图）。

```json
{"question": "支付总额是多少", "top_k": 10, "include_joins": true}
```

响应（结构示意）：

```json
{
  "question": "支付总额是多少",
  "schema": "... 收窄后的 CREATE TABLE 风格文本（只含相关表/字段） ...",
  "joins": ["dw.orders.order_id = dw.payments.order_id"]
}
```

平台用法：**直接拼进 prompt 的 schema 段**，省 token 且减少无关表干扰。

### 3.5 全链路：`POST /nl2sql/run`

link + join + schema_context 一次返回（平台想少调接口时用）。

---

## 4. 平台侧推荐调用流程

```
问题 → /nl2sql/link（拿相关表字段）
     → （跨表）→ /nl2sql/join_path 或 /nl2sql/schema_context(include_joins=true)
     → 拼 prompt：schema_context 文本 + 问题 → 你们的 SQL 生成模型 → SQL
```

## 5. 质量与参数建议

- `top_k`：link/schema_context 建议 **3**（我们压测的最优召回点；更大值提升召回、略降精度）。
- 语义匹配：P0 词法（中文口径词/字段注释子串）+ P2 向量（同义匹配：总额≈金额）双通道；
  字段注释质量直接决定命中率——**注释要写业务口径，不要只写英文名**。
- 指标口径：`terms` 里给足口径定义，`/nl2sql/link` 会优先命中指标绑定列。
- 降级：embedding 后端不可用时自动降级纯词法，不影响接口可用性。

## 6. 明天联调清单

1. 平台侧提供 2-3 张真实表元数据 → 我们 `reload` 建 schema（或确认走 `kg_rag` 图）。
2. 平台侧给 5-10 条真实问题 → 我们跑 `link`/`schema_context` 人工验收命中质量。
3. 对齐 prompt 模板：确认 schema_context 输出格式是否直接可拼（不够可加字段）。
4. 约定 Base URL 与鉴权（内网直连 / token）。
