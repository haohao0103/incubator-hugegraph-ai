# 知识图谱构建（实体/关系抽取）能力源码级比对

比对对象：`hugegraph-ai` 抽取链路 vs `microsoft/graphrag` / `neo4j-graphrag` / `langchain-experimental` / `KGGen`

> 所有结论均来自**本地实读源码**，附文件路径与行号，可逐条复核。未逐一验证的部分已显式标注。

---

## 0. 结论摘要

**hugegraph-ai 与主流框架不存在"能力代差"，但存在三处实质短板，且全部集中在"抽取产物的语义丰度"上，而非工程能力。**

| 判定 | 结论 |
|---|---|
| **工程/治理能力** | hugegraph-ai **领先**（严格 schema 契约、确定性顶点 ID、自动本体发现、实体消歧、增量、删除、多引擎社区检测） |
| **抽取产物语义丰度** | hugegraph-ai **落后**（实体/关系无自由文本描述、无关系强度） |
| **抽取召回机制** | hugegraph-ai **落后**（gleaning 多轮补全未进生产主链） |

真正的差距只有三条（按影响排序）：

1. **实体/关系无 `description` 自由文本** —— 生产抽取只产出 schema 内定义的属性，不产出 LLM 生成的描述文本
2. **无 `relationship_strength` 关系强度** —— 图是无权图，无法做加权 PPR / 关系排序
3. **Gleaning 未进生产主链** —— 只存在于 `graph_rag_enhancements` 增强层，主链 `extract_info` 是单轮抽取

---

## 1. 证据来源（版本与路径）

| 对象 | 版本/位置 | 核心证据文件 |
|---|---|---|
| **hugegraph-ai** | 本仓库（`incubator-hugegraph-ai/hugegraph-llm`） | `operators/llm_op/property_graph_extract.py`（403 行）<br>`operators/llm_op/info_extract.py`（202 行）<br>`config/prompt_config.py:49`（EN 抽取 prompt）<br>`nodes/llm_node/extract_info.py`（生产主链）<br>`operators/llm_op/auto_schema_kg.py`（1123 行） |
| **microsoft/graphrag** | 本地 `ms-graphrag/packages/graphrag` | `prompts/index/extract_graph.py`（129 行，GRAPH_EXTRACTION_PROMPT）<br>`index/operations/extract_graph/graph_extractor.py` |
| **neo4j-graphrag** | 1.19.0（pip 下载解包） | `components/entity_relation_extractor.py`<br>`experimental/pipeline/kg_builder.py` |
| **langchain-experimental** | 0.4.2（pip 下载解包） | `langchain_experimental/graph_transformers/llm.py` |
| **KGGen** | 0.4.0（pip 下载解包） | `kg_gen/kg_gen.py`<br>`kg_gen/steps/_1_get_entities.py` `_2_get_relations.py` `_3_cluster_graph.py` |

---

## 2. 根本差异：两种抽取范式

这是理解全部差距的起点——**hugegraph-ai 和其他框架抽取的不是同一种东西。**

### 2.1 hugegraph-ai：Schema 契约驱动的「顶点/边 JSON」

`property_graph_extract.py` 使用的 prompt（`config/prompt_config.py:49-88`）要求 LLM 输出：

```
{"vertices":[{"id":"1:Sarah","label":"person","properties":{"name":"Sarah","age":30, ...}}],
 "edges":[{"label":"roommate","outV":"1:Sarah","outVLabel":"person",
           "inV":"1:James","inVLabel":"person","properties":{"date":"2010"}}]}
```

关键约束（prompt 原文）：
- **确定性顶点 ID 规则**：`id = "{vertexLabelID}:{primary_key}"`，且 `vertexLabelID` 必须取自 schema 的 `vertexlabels[].id`，**"Never invent it from the label text"**
- **严格 schema 契约**："Do not extract labels or properties that are absent from the schema" / "Do not translate schema field names" / "Preserve property data types"
- **端点必须自洽**："Only output an edge if both endpoint vertices are also present in vertices"

**即：LLM 只能填充 schema 已定义的槽位，不能创造语义。**

### 2.2 ms-graphrag：LLM 自由描述的「实体/关系元组」

`prompts/index/extract_graph.py:6-129`（GRAPH_EXTRACTION_PROMPT）要求输出两类记录：

```
("entity"<|><entity_name><|><entity_type><|><entity_description>)
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
```

关键要素：
- `entity_type`：来自配置的 `[{entity_types}]` 白名单
- **`entity_description`**：LLM 自由生成的实体描述文本
- **`relationship_description`**：关系描述文本
- **`relationship_strength`**：**数值型关系强度**（`graph_extractor.py:22` 提示为 numeric score）

**即：LLM 在产出结构的同时产出语义文本与权重。**

### 2.3 范式差异的后果

| 后果 | 说明 |
|---|---|
| 描述文本缺失 | hugegraph-ai 顶点只有 `properties`（schema 预定义字段），**没有 LLM 生成的自然语言描述**，社区摘要与实体向量检索缺少高质量输入 |
| 无权图 | hugegraph-ai 边只有 schema 属性，**没有 strength**，PPR/排序只能用无权图 |
| 但：噪声可控 | hugegraph-ai **不可能**抽取出 `Acme Corp` / `Acme Corporation` 这类 schema 外的重复节点——schema 契约天然免疫 |

---

## 3. 能力矩阵（12 维度，源码级判定）

图例：✅ 生产链路已实现 ｜ ⚠️ 有但弱/在增强层 ｜ ❌ 无

| # | 维度 | hugegraph-ai | ms-graphrag | neo4j-graphrag | LangChain | KGGen |
|---|---|---|---|---|---|---|
| 1 | 实体抽取 | ✅ schema 顶点 | ✅ 含 description | ✅ 含 properties | ✅ 含 type/properties | ✅ 独立步骤 `_1_get_entities` |
| 2 | 关系抽取 | ✅ schema 边 | ✅ 含 description+strength | ✅ 含 properties | ✅ 三元组 | ✅ 独立步骤 `_2_get_relations` |
| 3 | **实体描述文本** | ❌ **仅 schema 属性** | ✅ LLM description | ✅ properties | ✅ node_properties 可选 | ❌ |
| 4 | **关系强度** | ❌ **无权** | ✅ numeric strength | ❌ | ❌ | ❌ |
| 5 | Schema 约束 | ✅✅ **严格契约+ID规则** | ⚠️ entity_types 白名单 | ✅ GraphSchema | ✅ allowed_nodes/relationships | ❌ 无约束 |
| 6 | 确定性实体 ID | ✅✅ PRIMARY_KEY 幂等 | ❌ 字符串 title | ❌ | ❌ | ❌ |
| 7 | **Gleaning 多轮补全** | ⚠️ **仅增强层，未进主链** | ✅ 生产（max_gleanings 循环） | ❌ | ❌ | ❌ |
| 8 | 实体消歧/合并 | ✅ entity_resolution + synonym + identity_edge + 模糊 id 匹配 | ⚠️ 靠 description 隐式合并 | ❌ | ❌ | ✅✅ DSPy 聚类（核心卖点） |
| 9 | 自动本体/schema 发现 | ✅✅ auto_schema_kg（1123 行） | ❌ 需配 entity_types | ❌ 需提供 schema | ❌ 需提供白名单 | ❌ |
| 10 | 社区检测 | ✅✅ 多引擎多算法（Vermeer/Computer/Leiden/Louvain/LPA/WCC） | ✅ Leiden（graspologic） | ❌ | ❌ | ❌ |
| 11 | 社区摘要 | ✅ CommunityReportGenerate（LLM） | ✅ community_reports | ❌ | ❌ | ❌ |
| 12 | 增量/删除 | ✅✅ incremental_merge + doc_deletion | ⚠️ update 索引 | ❌ | ❌ | ❌ |

---

## 4. 逐框架源码细节

### 4.1 microsoft/graphrag —— 抽取主循环（`graph_extractor.py`）

```python
# graphrag/index/operations/extract_graph/graph_extractor.py
43:    _max_gleanings: int
101:        if self._max_gleanings > 0:
102:            for i in range(self._max_gleanings):   # 多轮补全循环
...
148:                entity_description = clean_str(record_attributes[3])
152:                    "description": entity_description,
159:                edge_description = clean_str(record_attributes[3])
168:                    "description": edge_description,
```

配合 prompt 的两个专用指令（`extract_graph.py:128-129`）：
- `CONTINUE_PROMPT`："MANY entities and relationships were missed in the last extraction… Add them below using the same format"
- `LOOP_PROMPT`：Y/N 判断是否还有遗漏

**这是 ms-graphrag 实体召回率的主要来源**——首轮漏抽的实体靠后续轮次补齐。

### 4.2 neo4j-graphrag —— JSON 容错（`entity_relation_extractor.py`）

```python
55:def balance_curly_braces(json_string: str) -> str:   # 括号平衡修复
104:def fix_invalid_json(raw_json: str) -> str:          # 损坏 JSON 修复
46:class OnError(enum.Enum):                            # IGNORE / RAISE 策略
120:        create_lexical_graph (bool)                 # chunk 也入图（词法图）
```

**JSON 容错做得比 hugegraph-ai 细**：LLM 输出残缺 JSON 时先修复再解析。

对照 hugegraph-ai（`property_graph_extract.py:170-222`）：
```python
text = re.sub(r"```\w*\n?", "", text)                 # 剥离 markdown fence
json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)   # 正则抓 JSON
...
except json.JSONDecodeError:
    log.critical("Invalid property graph JSON! ...")  # 解析失败 → 直接丢弃
```
**判定**：hugegraph-ai 能剥离 fence 并正则提取，但**不做损坏 JSON 修复**，解析失败即丢弃该 chunk（次要差距）。

### 4.3 LangChain LLMGraphTransformer —— 白名单约束

```python
# langchain_experimental/graph_transformers/llm.py
324:    node_properties: Union[bool, List[str]] = False,
326:    relationship_properties: Union[bool, List[str]] = False,
```
支持 `allowed_nodes` / `allowed_relationships` / `node_properties` 白名单，用 `JsonOutputParser` 解析。

**与 hugegraph-ai 的 schema 约束思路一致，但粒度更粗**（只约束类型名与属性名，不约束 ID 规则与数据类型）。

### 4.4 KGGen —— DSPy 驱动的实体聚类（`_3_cluster_graph.py`）

```python
33:def get_extract_cluster_sig(items: set[str]) -> dspy.Signature:      # 抽取聚类
50:def get_validate_cluster_sig(items: set[str]) -> dspy.Signature:     # 校验聚类
69:def get_check_existing_clusters_sig(...)                             # 匹配已有聚类
217:def cluster_items(
279:def cluster_graph(graph: Graph, context: str = "") -> Graph:
```

分三步的 pipeline：`_1_get_entities` → `_2_get_relations` → `_3_cluster_graph`。

**核心差异化**：用 DSPy Signature 做 **LLM 聚类的"抽取→校验→匹配已有"三段式**，并分批处理（`_process_batch`）。这是目前开源里**最完整的 LLM 驱动实体消歧**实现。

对照 hugegraph-ai 的消歧（`graph_op/` 下四个算子）：
- `entity_resolution.py`、`synonym_manager.py`、`identity_edge_builder.py`、`incremental_merge.py`
- `property_graph_extract.py:223` `_add_fuzzy_vertex_ids`（模糊匹配顶点 id）

**判定**：hugegraph-ai 走的是**规则/字符串相似度**路线（确定性、快、可解释），KGGen 走 **LLM 语义聚类**路线（更准、慢、需 LLM 调用）。**KGGen 在"语义等价但字面不同"的场景更强；hugegraph-ai 在规模化成本上更强。**

---

## 5. 三大差距 · 根因 · 影响

### 差距 1：实体/关系无 `description`（影响最大）

| 项 | 内容 |
|---|---|
| **证据** | `config/prompt_config.py:49-88` 的 Output Contract 只定义 `properties`，无 description 字段 |
| **根因** | 设计取向：schema 契约优先，LLM 只填槽位，不生成自由文本 |
| **影响** | ① 社区摘要（`CommunityReportGenerate`）输入为结构化属性，语义密度低于 LLM 描述<br>② 实体向量化缺少高质量文本，实体级语义检索偏弱<br>③ 关系缺少语义解释，Global Search 可解释性下降 |

### 差距 2：无 `relationship_strength`

| 项 | 内容 |
|---|---|
| **证据** | 边 schema 只有 `properties`，prompt 无 strength 要求；ms-graphrag 在 `graph_extractor.py:22` 明确要求 numeric score |
| **根因** | 同上，schema 契约未定义强度字段 |
| **影响** | 图为无权图，PPR / 实体排序 / 关系剪枝只能用拓扑结构，无法按语义强度加权 |

### 差距 3：Gleaning 未进生产主链

| 项 | 内容 |
|---|---|
| **证据** | 生产主链 `nodes/llm_node/extract_info.py:33-52` 只按 `extract_type` 选择 `InfoExtract` 或 `PropertyGraphExtract`，**无 gleaning 调用**；`GleaningExtractor` 仅被 `demo/rag_demo/graphrag_enhancement_handlers.py:424-428` 引用 |
| **根因** | gleaning 实现放在 `graph_rag_enhancements` 增强层，未并入主链 |
| **影响** | 单轮抽取的实体召回率低于 ms-graphrag 的多轮补全（长文档/密集实体场景更明显） |

> 注：`GleaningExtractor`（`graph_rag_enhancements/gleaning_extractor.py`，配置 `max_rounds=1`）实现对标的是 **LightRAG** 的 gleaning，而非 ms-graphrag 的 Y/N 循环式。即便启用，轮次策略也不同。

---

## 6. hugegraph-ai 的领先项（明确优势）

1. **确定性顶点 ID + PRIMARY_KEY 幂等** —— `prompt` 强制 `id = "{vertexLabelID}:{primary_key}"`，重复灌入天然去重。其他四个框架均用实体名字符串，无此保证。
2. **严格 schema 契约** —— 类型/属性/数据类型/端点自洽全约束，LLM 无法创造 schema 外节点。这是**多厂商知识"不串味"**的硬保障，强于所有对比框架。
3. **自动本体发现** —— `auto_schema_kg.py`（1123 行）：单文档 → LLM 推断 schema draft → 人工审核 → 落 HugeGraph。其余框架均需预先提供 schema/types。
4. **完整图治理算子** —— `entity_resolution` / `synonym_manager` / `identity_edge_builder` / `doc_deletion` / `incremental_merge` / `incremental_utils`，覆盖图谱全生命周期。
5. **多引擎社区检测** —— `graph_op/community_detect.py`：Vermeer（Go 内存）/ HugeGraph-Computer（OLAP）/ 本地 Leiden-Louvain 三级降级，支持 louvain / lpa / wcc / pagerank / clustering_coefficient；实测 Vermeer 比本地快 6.4–11x。
6. **增量与删除** —— `incremental_index_flow` + `doc_deletion`，生产可用；ms-graphrag 仅支持索引 update。

---

## 7. 补齐建议（按性价比排序）

### P0 —— 不改架构即可补（建议优先做）

1. **给顶点/边增加 `description` 类属性**
   - 做法：在 schema 生成阶段（`auto_schema_kg` 或手工 schema）为每个 vertexLabel 增加可选 `description` property，并在抽取 prompt 中要求 LLM 填充
   - 收益：直接补上差距 1，社区摘要/实体向量质量提升
   - 风险：低（schema 契约不变，只是多一个属性槽位）

2. **给边增加 `strength` 属性**
   - 做法：同上，edgelabel 增加 `strength` INT/DOUBLE property，prompt 要求 1–10 打分
   - 收益：补上差距 2，PPR/关系排序可加权
   - 风险：低

### P1 —— 需要主链改造

3. **Gleaning 并入生产主链**
   - 做法：把 `GleaningExtractor` 的循环逻辑并入 `extract_info.py`，或新增 `extract_type="property_graph_gleaning"`
   - 建议：采用 ms-graphrag 的 **Y/N 循环判定**（比固定轮次更省 token），而非 LightRAG 的固定轮次
   - 收益：补上差距 3，实体召回率提升

### P2 —— 可选增强

4. ✅ **JSON 容错**（已落地）：新增 `balance_curly_braces` + `_repair_json`（无第三方依赖，覆盖尾随逗号与 token 截断导致的对象/数组未闭合）。仅在 `json.loads` 失败时触发，正常路径零开销。
5. ✅ **语义消歧增强**（经核实**已具备**，无需新增）：`entity_resolution.py`（1121 行）已内置四级策略 `exact_match` / `embedding` / `llm_verify` / `hybrid`，LLM 消歧通过 `strategy` 参数可选启用。原建议（借鉴 KGGen 三段式）是在未读全该文件时作出的误判，**此处更正**。

---

## 7.1 落地记录（2026-08-31）

| 项 | 状态 | 实现 |
|---|---|---|
| **P0-1 顶点 `description`** | ✅ | `auto_schema_kg._normalize_vertex_labels` 自动预留 `description`（nullable）；`_as_schema_draft` 同步注册 TEXT propertykey；EN/CN prompt 加"语义增强"段（条件式：schema 未声明则不填） |
| **P0-2 边 `strength`** | ✅ | `auto_schema_kg._normalize_edge_labels` 自动预留 `strength`（nullable）；`_as_schema_draft` 强制其 data_type 为 **INT**（避免被通用回填误标成 TEXT）；prompt 要求 1–10 整数 |
| **P1 Gleaning 入主链** | ✅ | `PropertyGraphExtract` 新增 `gleaning` / `max_gleanings`；`GLEANING_CONTINUE_PROMPT` + `GLEANING_LOOP_PROMPT`（Y/N 门控，对标 ms-graphrag）；`extract_info.py` 经 `EXTRACT_GLEANING` / `EXTRACT_MAX_GLEANINGS` 接入主链 |
| **P2-1 JSON 容错** | ✅ | 见上 |
| **P2-2 LLM 消歧** | — | 已具备，无需改动 |

**兼容性设计**：三处改动均默认不生效——`description`/`strength` 依赖 schema 是否声明（未声明则被 `filter_item` 静默丢弃），gleaning 默认 `False`。既有 schema 与调用方行为完全不变。

**回归验证**：`src/tests/operators/` 基线 `26 failed, 1401 passed`，改动后**数字完全一致**（26 个失败为 duckdb 等环境依赖缺失导致的固有失败，已通过 stash 对比确认非本次引入）。

---

## 8. 一句话总结

**hugegraph-ai 的抽取是"schema 优先的确定性结构化抽取"，工程与治理能力全面领先；但在"LLM 生成的语义文本与关系强度"这一层刻意留白，导致社区摘要质量、实体语义检索、加权图排序弱于 ms-graphrag。补齐方式是给 schema 加 `description` / `strength` 两个属性槽位并把 gleaning 并入主链——不需要推翻现有架构。**
