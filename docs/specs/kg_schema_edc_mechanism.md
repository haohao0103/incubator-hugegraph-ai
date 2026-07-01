# HugeGraph-AI 知识图谱 Schema 定义机制

> **定位声明**: 本文只描述**知识图谱(KG)**的schema定义机制。关系图谱的schema是业务人员人工定义的，与本文描述的EDC系统无关。两者是不同数据层、不同命名空间、不同集群，不混合管理。

---

## 一句话总结

> 知识图谱的schema不需要人工预先定义完整，而是随文本处理自动进化。核心问题是LLM对同一概念会起不同名字（嫌疑人/嫌疑犯/suspect），EDC三阶段自动合并，防止类型爆炸。同时支持人工预定义核心类型和审批低置信度合并，确保schema可控。

---

## 两种模式

| 模式 | 适用场景 | Schema怎么来 | 是否需要EDC |
|------|---------|-------------|------------|
| **EVOLVING**（默认） | 开放域——客服对话、新闻分析、元数据检索 | LLM自由提取 → EDC三阶段自动合并同义类型 → schema逐步进化 | ✅ 需要（Extract→Define→Canonicalize） |
| **GUIDED**（可选） | 封闭域——风控增量、代码图谱、金融监管 | 人工定义允许的vertex/edge类型 → Pydantic约束LLM只输出这些类型 → schema一步到位 | ❌ 不需要（Guided直接约束） |

**不支持裸schema-free**——因为LLM叫同一概念6个名字（嫌疑人/嫌疑犯/suspect/犯罪人/涉案人员/offender），直接写入就炸了，6个vertex type其实只有1个概念。

---

## EVOLVING 模式：EDC三阶段详解

### 核心问题：Type Explosion

LLM从不同文本里提取同一概念，会用不同的词：

| 第一批文本 | 第二批文本 | 第三批文本 |
|-----------|-----------|-----------|
| "字段" | "列" | "column" |
| "表" | "数据表" | "datatable" |
| "嫌疑人" | "嫌疑犯" | "suspect" |

如果直接写入KG，每个文本批次都发明新的type名 → type explosion → schema失控。

### Phase 1 — Extract：LLM自由提取

LLM从文本中提取实体和关系，**不受schema约束**，产出任意类型名。

**示例**：处理货拉拉数仓文档

| 实体 | LLM给的type |
|------|-----------|
| 司机宽表 | "表" |
| 物理车型 | "字段" |
| 实际车型 | "字段" |
| 订单宽表 | "表" |

这是schema-free中间状态——LLM用了自己的语言，不是预定义的术语。如果停在这里，"字段"和"列"就是两个不同的vertex type，检索时无法合并召回。

### Phase 2 — Define：为新类型生成语义定义

对**不在 `known_type_registry` 中的新类型**，调用LLM生成定义：

```json
{
  "字段": {
    "description": "数据库表中的列，包含字段名、数据类型和业务含义描述",
    "properties": [
      {"name": "字段名", "type": "string", "cardinality": "single", "required": true},
      {"name": "数据类型", "type": "string", "cardinality": "single", "required": true},
      {"name": "业务含义", "type": "string", "cardinality": "optional", "required": false},
      {"name": "所属表", "type": "string", "cardinality": "single", "required": true}
    ],
    "parent_types": ["Entity"],
    "distinguishing_features": " narrower than 'attribute', refers specifically to a database column"
  },
  "表": {
    "description": "数据库中的数据表，存储特定业务域的结构化数据",
    "properties": [
      {"name": "表名", "type": "string", "cardinality": "single", "required": true},
      {"name": "表描述", "type": "string", "cardinality": "optional", "required": false},
      {"name": "数据域", "type": "string", "cardinality": "optional", "required": false}
    ],
    "parent_types": ["Entity"],
    "distinguishing_features": "broader than 'view', contains persistent stored data"
  }
}
```

**关键机制**：
- 定义存入 `known_type_registry`，下次遇到"字段"直接跳过Define
- 如果人工在 `manual_type_definitions` 中预定义了"表"和"字段"，LLM调用直接跳过——**零LLM开销**
- 第一次运行：所有类型都是新的，Define开销大
- 稳定运行：registry已累积，新类型占比<10%，Define≈0额外调用

**触发策略**（3种可选）：

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `NEW_TYPES_ONLY`（默认） | 只对不在registry的新类型触发Define | 日常运行 |
| `ALWAYS` | 每次运行都重新定义所有类型 | 初始bootstrapping |
| `THRESHOLD` | 新类型占比超过阈值才触发 | 控制成本 |

### Phase 3 — Canonicalize：合并同义类型

计算embedding相似度，发现LLM在不同文本中叫同一概念的不同名字，合并为统一canonical type。

**阈值分层**：

| 相似度区间 | 决策 | 标记 |
|-----------|------|------|
| ≥ 0.85 | 强制合并 → canonical_type统一 | `forced` |
| 0.70 ~ 0.85 | 建议合并 → 需人工审批 | `suggested` |
| < 0.70 | 保持独立类型 | `unchanged` |

**示例**：

| raw_type | canonical_type | 相似度 | 决策 |
|----------|---------------|--------|------|
| "字段" | field | — (首次定义) | 首次进入registry |
| "列" | field | 0.89 | forced |
| "column" | field | 0.86 | forced |
| "属性" | field | 0.78 | suggested (需人工审批) |
| "指标" | metric | 0.35 | unchanged (独立类型) |

**存储双标签**：
- `label=field`（canonicalized统一类型名，用于检索）
- `raw_type="字段"/"列"/"column"`（LLM原始输出，保留语义丰富度）

**检索效果**：用户问"实际车型" → 匹配到canonical_type=`field` → 同时命中"物理车型"、"实际车型"两个字段的实体 → 返回司机宽表正确答案

**3种Canonicalize策略**：

| 策略 | 原理 | 精度 | 成本 |
|------|------|------|------|
| `EMBEDDING_SIM`（默认） | embedding向量相似度 | 中 | 低（预计算embedding） |
| `EXACT_MATCH` | 仅大小写不敏感字符串匹配 | 低 | 零 |
| `LLM_CLASSIFY` | LLM分类每个类型 | 高 | 高（每次分类1个LLM调用） |

---

## 人工干预能力

EDC**不排斥人工**，提供3个入口：

| 入口 | 作用 | 代码配置 | 适用场景 |
|------|------|---------|---------|
| **预种子registry** | 启动前人工定义核心类型，LLM只补充长尾 | `manual_type_definitions={"表": {...}, "字段": {...}}` | 首次部署、核心领域类型明确 |
| **审批suggested映射** | 0.70~0.85相似度的合并需人工确认 | `require_human_approval_below=0.70` | 谨慎场景，不允许自动合并低置信度映射 |
| **覆盖Define结果** | 人工定义优先，跳过LLM调用 | `allow_manual_override=True` | LLM定义不满意时人工修正 |

**实操流程**：
1. 第一次跑EDC，全部交给LLM → 检查registry结果
2. 修改不满意的定义 → 作为 `manual_type_definitions` 预种子
3. 后续运行该类型跳过LLM → 稳定运行人工干预≈0

---

## Guided 模式（可选增强）

如果场景的vertex type是**人工已知的**（比如元数据检索只有"表"、"字段"、"业务术语"3个核心type），可以直接用Guided模式：

Pydantic ResponseModel约束LLM——**只允许输出这3个label**，不允许自由发明"列"、"数据表"等。Extract一步到位，不需要Define和Canonicalize。

**Guided适用场景**：
- ✅ 风控增量：person/device/ip/account/organization/location 6个核心类型
- ✅ 代码图谱：class/function/module/file/import 5个核心类型
- ✅ 元数据检索：表/字段/业务术语 3个核心类型

**Guided不适用场景**：
- ❌ 客服对话：用户会问任何领域的问题
- ❌ 新闻分析：事件类型无法穷举
- ❌ 开放域问答：概念空间无限

---

## 代码模块索引

| 模块 | 文件 | 功能 |
|------|------|------|
| SchemaConfig | `graphrag_schema_config.py` | 配置类：SchemaMode、CanonicalizeStrategy、DefineTriggerPolicy、人工干预参数 |
| Define Operator | `kg_schema_define.py` | EDC Phase 2：LLM为新类型生成语义定义，人工override跳过LLM |
| Canonicalize Operator | `kg_schema_canonicalize.py` | EDC Phase 3：3种策略合并同义类型，阈值分层forced/suggested/unchanged |
| Guided Extract | `guided_extract.py` | Pydantic V2 ResponseModel约束提取，只允许预定义label |
| EDC Pipeline | `edc_pipeline.py` | 编排三阶段：Config初始化→Extract后处理→Define→Canonicalize |

**测试**：44/44 passed，覆盖config验证、define/canonicalize/guided operators、pipeline集成、跨运行registry累积。

---

## 货拉拉元数据检索场景映射

货拉拉文章中的4个痛点，EDC如何解决：

| 痛点 | 根因 | EDC对应阶段 | 解决方式 |
|------|------|-------------|---------|
| "实际车型"检索不到"物理车型" | 同义词向量匹配失效 | **Canonicalize** | "实际车型"↔"物理车型" embedding=0.88 → forced合并为`vehicle_model` |
| "司机卸货位置"无法匹配"经度+纬度" | 业务口径不同表述 | **Define** | 给"卸货位置"生成定义"司机完成卸货的地理位置坐标(含经度、纬度属性)" |
| 字段缺少comment就胡乱回答 | 知识营养不良 | **Define** | LLM为每个type生成description+properties，补齐缺失的业务语义 |
| "手工梳理+LLM抽取"混合模式 | 需要人工兜底 | **人工预定义** | known_type_registry支持人工预种子，LLM只补充未定义的 |

---

## 架构决策记录

| 决策 | 理由 | 日期 |
|------|------|------|
| EDC Evolving为默认，Guided为可选增强 | Canonicalize是唯一能防止type explosion的机制 | 2026-07-01 |
| 不支持裸schema-free | LLM自由提取产生type noise → 同概念多个type名 | 2026-07-01 |
| Extract改造：加性后处理 | 保留现有lowercase+strip规范化，双输出(normalized key + raw_type) | 2026-07-01 |
| Canonicalize：预计算embedding查表 | 已知类型数量有限(~50)，一次初始化足够 | 2026-07-01 |
| Define：仅新类型触发LLM调用 | known_type_registry缓存，稳定运行≈0额外调用 | 2026-07-01 |
| 人工干预：3个入口 | 预种子registry、审批suggested、override Define | 2026-07-01 |
| Canonicalize目标：KG自身registry去重 | 不是桥接外部图谱，是KG内部合并同义类型 | 2026-07-01 |

---

*文档版本: v1.0 | 最后更新: 2026-07-01 | 作者: HugeGraph-AI GraphRAG Team*
