# 统一导入 / 查询 API 设计 Spec（hugegraph-ai GraphRAG）

> 状态：设计说明（spec），非实现。所有 flow 名来自当前仓库 `hugegraph-llm/src/hugegraph_llm/flows/`。

## 1. 设计原则

- **内部多通道，外部单入口**：用户只面对 **1 个导入入口 + 1 个查询框**，后端按来源 / 意图路由。
- **同一张知识图谱**：三类数据（库表字段 comment / 指标 / 飞书文档）全部汇入固定图名 `kg-rag`，内部用顶点 / 边 label 分型；**绝不与 60 亿点边关系图混淆**。
- **查询默认混合**：用户不需选"精确 / 语义 / 混合"，系统自动路由；仅高级模式暴露开关。

## 2. 图模型（同一张 KG 内的顶点 / 边 label）

- 顶点 label：`Table` `Field` `Metric` `Document` `Chunk` `Entity`
- 边 label：
  - `hasColumn`  : `Table` → `Field`
  - `computedFrom`: `Metric` → `Table` / `Field`
  - `dependsOn`  : `Metric` → `Metric`
  - `mentions`   : `Document` / `Chunk` → `Entity` / `Table` / `Field`
  - `synonym_edge`: 任意 → 任意（实体消歧 `EntityResolutionFlow` 产出）
- 公共属性：`domain`（业务域隔离）、`source`（来源系统）、`name`

## 3. 统一导入 API

```
POST /api/v1/ingest
{
  "source_type": "feishu" | "catalog_csv" | "jdbc" | "metric_json" | "auto",
  "payload": { ... },            # 按 source_type 填对应字段
  "domain": "risk_control",      # 可选，业务域隔离
  "options": { "entity_resolution": true, "build_vector_index": true }
}
→ 202 { "task_id": "...", "graph": "kg-rag", "status": "queued" }
```

### 来源识别（配置点 2）
- 默认用户在下拉里选 `source_type`（最稳，避免误判）。
- `auto` 时按内容嗅探：`catalog_csv`/`jdbc` 列名含 `table/field/comment`；`metric_json` 含 `metric_name/formula`；纯文本 → 文档。

### 路由伪代码（配置点 1 + 2）

```python
def ingest(req):
    t = req.source_type
    if t == "auto":
        t = detect_by_schema(req.payload)

    if t == "feishu":
        FeishuIngestFlow(graph=KG, domain=req.domain).run(req.payload)
    elif t in ("catalog_csv", "jdbc"):
        ImportGraphDataFlow(mode="catalog", graph=KG, domain=req.domain).run(req.payload)
    elif t == "metric_json":
        ImportGraphDataFlow(mode="metric", graph=KG, domain=req.domain).run(req.payload)

    # 三条路径的共同后缀（配置点：实体消歧 + 向量）
    if req.options.entity_resolution:
        EntityResolutionFlow(graph=KG).run()
    if req.options.build_vector_index:
        BuildVectorIndexFlow(graph=KG).run()
```

> 关键：三条路径都写同一个 `graph=KG`（配置点 1 的 `HUGEGRAPH_GRAPH=kg-rag`），保证进同一张图；实体消歧跨类型连边。

## 4. 统一查询 API

```
POST /api/v1/query
{
  "question": "去年大促退款率怎么算？相关文档有哪些？",
  "mode": "auto" | "precise" | "semantic" | "hybrid",   # 默认 auto
  "domain": "risk_control",                              # 可选过滤
  "top_k": 5
}
→ 200 { "answer": "...", "citations": [...], "route": "graphrag", "subgraph": {...} }
```

### 路由伪代码（配置点 3）

```python
def query(req):
    if req.mode != "auto":
        return run_mode(req.mode, req)          # 高级模式：precise/semantic/hybrid 直连

    # auto：先判断 query 是否命中图中已有结构化节点
    hits = gremlin_match(req.question,
                         labels=[Table, Field, Metric],
                         domain=req.domain)
    if hits and is_structured_query(req.question):
        return Text2GremlinFlow(req.question, domain=req.domain).run()   # 精确
    return RAGGraphVectorFlow(req.question, domain=req.domain).run()     # 语义混合（图 + 向量）
```

### 路由判据
- `gremlin_match`：检查 query 是否命中图中已有结构化节点名（表 / 字段 / 指标）。
- 命中且问法像检索（"有哪些字段 / 依赖哪些表 / 口径是什么"）→ **精确**（`Text2GremlinFlow`）。
- 否则 → **语义混合**（`RAGGraphVectorFlow`，图遍历 + 向量召回，LLM 带引用作答）。

## 5. 四个配置点落地（部署期设一次，非写代码）

1. **图名统一**：环境变量 `HUGEGRAPH_GRAPH=kg-rag`，所有 ingest / query flow 读取它
   （`FeishuIngestFlow` / `ImportGraphDataFlow` / `EntityResolutionFlow` / 各 query flow）。
2. **来源识别**：`/api/v1/ingest` 的 `source_type` 下拉默认非 `auto`；`auto` 用列名嗅探兜底。
3. **查询路由**：`/api/v1/query` 的 `mode` 默认 `auto`（混合兜底）；前端仅对"高级模式"暴露
   `precise` / `semantic` / `hybrid` 开关。
4. **domain 隔离**：所有顶点写 `domain` 属性；ingest / query 都带 `domain` 过滤。
   多业务域共用一张图，不另开图。

## 6. 红线

- KG 图（`kg-rag`）与 60 亿点边**关系图物理隔离**（不同 graph 配置），导入只触 KG。
- 只导元数据 / comment / 定义 / 文档，**不导行级数据**（行级数据属于关系图）。
