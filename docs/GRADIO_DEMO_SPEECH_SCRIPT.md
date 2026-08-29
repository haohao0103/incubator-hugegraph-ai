# HugeGraph-AI Gradio Demo 完整演讲脚本

> **适用版本**: 9-Tab 架构 (commit `deb4141` + `c4926ee`)
> **演讲时长**: 约 25-30 分钟
> **演示前提**: HugeGraph 服务运行中 (`localhost:8080`)，LLM API 可用 (MiMo v2.5 Pro)
>
> **关键规则**: 每个按钮点击前必须确认输入框已填入数据！本脚本的「预填数据」部分已为每个操作准备了可直接复制粘贴的示例数据。

---

## 📋 总览：9 个标签页的定位与串联关系

```
用户意图流向：

  [原始文档] ──→ Tab1 BuildRAGIndex ──→ [向量索引 + 图数据] ──┐
                                                              │
  [无Schema?] ──→ Tab2 SchemaStudio ──→ [Schema JSON] ───────┤
                                                              ├─→ Tab5 RagQA (问答)
  [需要高级检索?] ──→ Tab3 GraphRAG Core ──→ [增强检索链路] ──┤
                                                              │
  [复杂推理?] ──→→ Tab4 AgentGlobalSearch ──→ [Agent多步推理] ─┘

  [自然语言查图?] ──→ Tab6 Text2Gremlin ──→ [Gremlin查询]

  [PDF/图片/表格?] ──→ Tab7 Multimodal ──→ [多模态知识图谱]

  [运维管理] ←── → Tab8 AdminOps (独立)

  [能力全景] ←── → Tab9 CapabilityMap (参考)
```

**核心数据流（必走路径）**:
```
Tab2(建Schema) → Tab1(导入+抽取+写入HugeGraph) → Tab3(配置检索增强) → Tab5(问答验证)
```

---

# ══════════════════════════════════════════════════════════
# TAB 1: Build RAG Index（索引构建入口）
# ══════════════════════════════════════════════════════════

## 📍 定位
整个 Demo 的 **数据入口**。用户在这里上传文档、配置 Schema、执行"文档→向量索引→图谱抽取→写入 HugeGraph"的完整 pipeline。

## 🎯 核心交互区域（4 列布局）

### 左列：文档输入（双 Tab 切换）

| 组件 | 类型 | 说明 | 预填数据 |
|------|------|------|----------|
| **text** 子Tab | TextBox (20行) | 粘贴纯文本 | 见下方预填 |
| **file** 子Tab | File (多选) | 上传 TXT/DOCX/PDF | 上传任意文档 |

**⚠️ 注意**: text 和 file 是互斥 Tab——选 file 会清空 text，选 text 会清空 file。

### 中列：Schema 配置

| 组件 | 类型 | 说明 | 预填数据 |
|------|------|------|----------|
| **Graph Schema** | Code (JSON) | HugeGraph Schema 定义 | `hugegraph` (自动从图实例拉取) 或粘贴 JSON |

**预填 Schema 示例**（供应链风控场景）：
```json
{
  "vertexlabels": [
    {"name": "Supplier", "id": 1, "properties": {"name": "text", "risk_score": "double"}},
    {"name": "Warehouse", "id": 2, "properties": {"name": "text", "location": "text"}},
    {"name": "Transport", "id": 3, "properties": {"name": "text"}}
  ],
  "edgelabels": [
    {"name": "supplies_to", "source_label": "Supplier", "target_label": "Warehouse"},
    {"name": "distributes_to", "source_label": "Warehouse", "target_label": "Transport"}
  ]
}
```

### 右列：Prompt 模板

| 组件 | 类型 | 说明 |
|------|------|------|
| **Graph Extract Prompt Header** | Code (Markdown) | LLM 抽取图谱时用的 system prompt |

**💡 提示**: 如果没有自定义 prompt，保持默认即可。Prompt Generator（下方折叠区）可以自动生成。

## 🔘 操作按钮行（从左到右）

### Row 1: 信息查看 & 清理（折叠 Accordion）

| 按钮 | 功能 | 需要输入？ | 点击后输出 |
|------|------|-----------|-----------|
| **Get Vector Index Info** | 查看当前向量索引状态 | ❌ 不需要 | Output Info (JSON) — chunk 数量、维度等 |
| **Get Graph Index Info** | 查看 Vid Embedding 索引状态 | ❌ 不需要 | Output Info (JSON) — vertex 数量、embedding 维度 |
| **Clear Chunks Vector Index** | 清除文档 chunk 向量索引 | ⚠️ 危险操作 | 无输出（静默清理） |
| **Clear Graph Vid Vector Index** | 清除顶点向量索引 | ⚠️ 危险操作 | 无输出（静默清理） |
| **Clear Graph Data** | 清除 HugeGraph 中所有图数据 | 🔴 极危险 | 无输出（静默清理） |

**🎤 演讲台词**:
> "这五个按钮是运维用的。左边两个是**只读**——随时可以点，会显示当前索引状态。右边三个带 'Clear' 的是**写操作**——点之前要想清楚，因为数据删了就没了。我们一般只在重新构建索引前才清。"

### Row 2: 核心 Pipeline（最重要的 4 个按钮）

| 按钮 | 功能 | 前置条件 | 预填数据 |
|------|------|---------|----------|
| **① Import into Vector** | 文档分块 + Embedding + 写入 Milvus/Faiss | ✅ text/file 有内容 | text Tab 里粘贴文本或 file 上传文件 |
| **② Extract Graph Data (1)** | LLM 从文本中抽取三元组 | ✅ Schema 已配 + Prompt 有值 | 同上 |
| **③ Load into GraphDB (2)** | 把抽取的三元组写入 HugeGraph | ✅ ①和②都完成（Output Info 有数据） | 自动读取 Output Info |
| **④ Update Vid Embedding** | 为图中所有顶点建立/更新向量索引 | ✅ ③完成（图里有数据） | ❌ 不需要输入 |

**⚠️ 执行顺序 = 按钮上的数字！①→②→③→④，不能乱。**

**🎤 演讲台词**:
> "这是整个 pipeline 的核心四步。注意看按钮上的数字标注——这就是执行顺序。
>
> **第一步 Import into Vector**：把你的原始文档切成小块，用 Embedding 模型编码成向量，存到向量数据库里。这是传统 RAG 的基础。
>
> **第二步 Extract Graph Data**：调用 LLM，根据你配的 Schema，从同样的文档里抽取出实体和关系——也就是三元组。
>
> **第三步 Load into GraphDB**：把第二步抽出来的三元组，通过 HugeGraph REST API 写入图数据库。这一步完成后，你的 HugeGraph 里就有可查询的知识图谱了。
>
> **第四步 Update Vid Embedding**：这是今天我们刚优化完的功能——给图中每个顶点做 embedding，存到 `graph_vids` 向量索引里。这样后续查询才能用语义搜索找到对应的实体。
>
> **关键点**：①和②可以并行（都只需要原始文档），但③必须在①②之后，④必须在③之后。"

### 预填数据（直接可用）

**Text 输入框预填文本**（供应链风控场景，约 200 字）：
```
Supplier-Y is a critical component provider located in Shenzhen. They supply electronic parts to Warehouse-C in Beijing with a disruption risk of 0.93.

Warehouse-C is a major distribution hub serving Northern China. It experiences severe congestion during peak hours (congestion risk: 0.91) and high operational costs (cost risk: 0.92). Warehouse-C distributes goods to Transport-Z which covers the last-mile delivery route.

Transport-Z is a logistics carrier operating in urban areas. Their quality risk score is 0.85 due to occasional delivery delays.

Key relationships:
- Supplier-Y --[supplies_to]--> Warehouse-C
- Warehouse-C --[distributes_to]--> Transport-Z
- Warehouse-C has high congestion and cost risks requiring attention.
```

## 📐 折叠区域

### Auto-Generate Schema? (折叠态)
提示用户去 **Tab 2 Schema Studio** 使用 AutoSchemaKG 自动生成 Schema。

### Graph Extraction Prompt Generator (折叠态)

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Expected scenario/direction | TextBox | `supply chain risk analysis` |
| Few-shot example Dropdown | Dropdown | 选择第一个示例 |
| View example details | Accordion (折叠) | 展开看示例详情 |
| **🚀 Auto-generate** | Button (primary) | 点后自动生成 prompt → 写入右侧 Prompt Header |

**🎤 演讲台词**:
> "如果你不知道怎么写 Graph Extract Prompt，可以用这个 Prompt Generator。选择一个场景方向、挑一个 few-shot 示例，点击自动生成。它会调用 LLM 基于示例风格为你定制一个高质量的抽取 prompt。

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 传什么 |
|----------|---------|--------|
| **Tab 2** → Tab 1 | Schema Studio 生成的 Schema JSON | 粘贴到 Graph Schema 框 |
| Tab 1 → **Tab 5** | ①~④ 全部执行完后 | 去 RagQA 验证问答效果 |
| Tab 1 → **Tab 8** | 随时可以去 | 用 Gremlin Console 验证数据是否写入成功 |

---

# ══════════════════════════════════════════════════════════
# TAB 2: Schema Studio（Schema 构建工作室）
# ══════════════════════════════════════════════════════════

## 📍 定位
**Schema 的"零到一"和"一到 N"**。Section A 用 LLM 从文档推断初始 Schema，Section B 做 Schema 演进（生产级）。

## 🎯 Section A: AutoSchemaKG（LLM 推断 Schema）

### 顶部控制栏

| 组件 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| Mode Radio | Radio | `batch` (推荐) | Single=单文档推断; Batch=多文档合并+冲突检测 |
| Instructions (optional) | TextBox | 空 | 给 LLM 的额外指令，如 `focus on supply-chain entities, include timestamp properties` |

### 文档输入区

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Load Single-doc Example | Dropdown | 选择 `"Supply Chain Risk"` |
| Load Multi-doc Example | Dropdown | 选择 `"Supply Chain Risk (multi-doc)"` (batch模式) |
| Document(s) TextBox | TextBox (10行) | 选了 dropdown 后自动填充 |

**预填 Instructions**:
```
focus on supply-chain entities, include risk_score as double property, add timestamp properties
```

### 操作按钮（右列）

| 按钮 | 功能 | 需要什么 |
|------|------|----------|
| **Generate Schema Draft** (primary) | 调 LLM 生成 Schema 草案 | Document(s) 不为空 |
| **Approve & Commit to HugeGraph** (secondary) | 将 Schema 提交到 HugeGraph（创建 vertex/edge label） | Schema JSON 不为空 |
| **Reset** (stop, sm) | 重置所有输出 | - |

### 输出区

| 组件 | 内容 |
|------|------|
| **Schema Preview (Markdown)** | 人可读的 Schema 摘要 |
| **Schema JSON (editable)** | 完整 Schema JSON，可编辑后提交 |

### 🌉 Export to EDC Guided Mode（桥接按钮）

| 组件 | 内容 |
|------|------|
| **Export Schema to EDC Guided Mode** Button | 把当前 Schema 的 label 列表提取出来，格式化为 EDC Pipeline 的约束配置 |
| **Guided Mode Config (JSON)** | 输出：`{"mode":"GUIDED","allowed_vertex_labels":[...],"allowed_edge_labels":[...]}` |

**🎤 演讲台词**:
> "这个桥接按钮是今天整合的一个亮点。你在 Section A 用 LLM 生成了初始 Schema 后，点一下这个按钮，它会把 Schema 里的 vertex_labels 和 edge_labels 提取出来，格式化成 EDC Pipeline 的 Guided Mode 配置。然后你把这个配置粘贴到下面的 Section B，就能在已有 Schema 的约束下继续演进——不会产生重复或冲突的类型。

### Suggested Review Questions (折叠)
展开后显示 5 个审查问题，帮助评估 Schema 质量。

---

## 🎯 Section B: EDC Pipeline（Schema 演进）

**EDC = Extract → Define → Canonicalize**

| 阶段 | 做什么 | 类比 |
|------|--------|------|
| **Extract** | 从新文档中发现潜在类型 | 探矿 |
| **Define** | 决定哪些类型纳入/排除/合并 | 选矿 |
| **Canonicalize** | 标准化属性名、类型约束 | 冶炼 |

**🎤 演讲台词**:
> "Section B 是生产级功能。当你已经有一个初始 Schema（比如从 Section A 来的），随着不断有新文档进入系统，你需要让 Schema 逐步演化而不是每次推倒重来。EDC Pipeline 就是干这个的——它有三个阶段：
>
> **Extract** 阶段扫描新文档，找出所有可能的新实体类型和新关系类型；
> **Define** 阶段让你审核这些发现——保留有用的、合并重复的、拒绝不相关的；
> **Canonicalize** 阶段确保最终 Schema 符合 HugeGraph 的命名规范和约束要求。
>
> 如果你用的是 Guided Mode（从 Section A 导出过来的），Extract 就只会关注 allowed_vertex_lists 里的类型，不会跑偏。"

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 传什么 |
|----------|---------|--------|
| Tab 2 (Section A) → **Tab 1** | Approve & Commit 后生成的 Schema JSON | 复制到 Tab 1 的 Graph Schema 框 |
| Tab 2 (Export) → **Tab 2 (Section B)** | Guided Mode Config | 粘贴到 Section B 的 Guided 配置框 |
| **Tab 1** → Tab 2 | 发现 Schema 不满足需求 | 回来用 AutoSchemaKG 重新生成 |

---

# ══════════════════════════════════════════════════════════
# TAB 3: GraphRAG Core（检索能力核心）
# ══════════════════════════════════════════════════════════

## 📍 定位
**GraphRAG 检索引擎的完整工具箱**。按检索 pipeline 的阶段组织，对标 4 大竞品的核心算法。

## 🏗️ 六段式架构概览

| Section | 名称 | 对标竞品 | 核心问题 |
|---------|------|---------|----------|
| **A** | 图检索引擎 | Fast-GraphRAG, HippoRAG2 | 怎么从图的某个起点开始高效遍历？ |
| **B** | 检索增强 | LightRAG | 怎样提升查询的召回率和精度？ |
| **C** | 推理精炼 | MS-GraphRAG | 多跳推理怎么做？上下文预算怎么控制？ |
| **D** | 可信输出 | 企业级需求 | 答案怎么溯源？实体怎么消重？ |
| **E** | 关键词降级 | HugeGraph 独有 | 向量搜不到精确名称怎么办？ |
| **F** | 块图增强 | 内部创新 | Chunk 之间如何建立 SIMILAR 边？ |

---

## A. 图检索引擎（默认展开 ✅）

### A1: PPR Retriever（个性化 PageRank）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `Warehouse-C supply chain risk` |
| Alpha (Teleport概率) | Slider | `0.15` (默认) |
| Max Depth | Slider | `2` (默认) |
| Top-K Results | Slider | `10` (默认) |
| **Run PPR Search** | Button (primary) | 👈 点击 |

**输出**:
- **PPR检索结果** (JSON): `{total_entities_reached, alpha, seed_entities}`
- **PPR Scores** (JSON, 折叠): 每个 entity 的 PPR 分数排名
- **Retrieved Context** (TextBox): 拿到的文本上下文

**🎤 演讲台词**:
> "PPR 是 Personalized PageRank——个性化 PageRank 算法。它的原理是：从一个或多个种子节点出发，模拟随机游走，按照概率跳转到邻居节点或跳回原点（teleport）。Alpha 就是 teleport 的概率，越小越倾向于在局部深度探索。
>
> 在我们的场景里，用户的查询先被匹配到一个或多个种子实体，然后 PPR 从这些种子出发，找到图中最相关的 Top-K 实体。这是 LightRAG 和 HippoRAG2 都在用的核心算法，区别是我们支持实时计算，不需要像 MS-GraphRAG 那样预建社区索引。"

### A2: Cascade Propagation（三层传播）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `supplier disruption impact on warehouse` |
| PPR Alpha | Slider | `0.15` |
| Entity Threshold | Slider | `0.01` |
| Chunk Top-K | Slider | `10` |
| **Run Cascade** | Button (secondary) | 👈 点击 |

**输出**:
- **传播步骤 Trace** (JSON): Entity Layer → Relation Layer → Chunk Layer 三层传播过程
- **Entity/Relation/Chunk Layer Scores** (JSON 各一个): 每层的得分分布

**🎤 演讲台词**:
> "Cascade Propagation 是三层传播模型——从实体层开始，传播到关系层，再传播到 Chunk 层。每一层都有独立的阈值过滤，保证只有高相关性内容才会被传递下去。这解决了单纯 PPR 只能拿到实体、拿不到具体文档片段的问题。"

### A3: Identity Edge Builder（实体消解）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 实体名称(逗号分隔) | TextBox (2行) | `Apple Inc, Apple, AAPL, Microsoft Corp, MSFT, 微软` |
| Similarity Threshold | Slider | `0.9` (默认) |
| Top-K Neighbors | Slider | `5` (默认) |
| **Build Identity Edges** | Button (secondary) | 👈 点击 |

**输出**:
- **相似实体对** (JSON): `[{"a": "Apple Inc", "b": "Apple", "sim": 0.94}, ...]`
- **合并建议** (折叠): 建议合并的分组

**🎤 演讲台词**:
> "Identity Edge Builder 解决的是实体消解问题——同一个真实世界实体在文档里可能有不同的名字，比如 'Apple Inc'、'Apple'、'AAPL' 其实都是苹果公司。这个算子通过 embedding 相似度和图结构邻居来识别这些别名关系，然后建立 same_as 边。阈值设为 0.9 表示非常确信才建立 identity 边。"

---

## B. 检索增强（默认折叠 ▶️ 点击展开）

### B1: Dual Keyword（双层关键词提取）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `What is the treatment for diabetes type 2?` |
| 提取方式 Method | Dropdown | `heuristic` (启发式，快) / `llm` (精准，慢) |
| 语言 Language | Dropdown | `en` 或 `zh` |
| **Extract Keywords** | Button (secondary) | 👈 点击 |

**输出**:
- **hl_keywords** (JSON): 高层概念关键词，如 `["diabetes", "treatment", "blood sugar"]`
- **ll_keywords** (JSON): 低层实体关键词，如 `["metformin", "insulin", "GLP-1"]`

**🎤 演讲台词**:
> "Dual Keyword 是 LightRAG 的核心创新之一——它把关键词分成两层。高层关键词（hl）是抽象概念，用来做粗粒度的图遍历；低层关键词（ll）是具体实体，用来做精确定位。你可以选 heuristic 模式（基于规则的快速提取）或 LLM 模式（更准但更慢）。"

### B2: HyDE（假设性文档嵌入）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 原始查询 Original Query | TextBox (2行) | `How does PPR improve graph retrieval accuracy?` |
| **Generate HyDE** | Button (secondary) | 👈 点击 |

**输出**:
- **假设性答案 Hypothetical Answer** (TextBox): LLM 生成的假想答案段落
- **增强查询 Enhanced Query** (TextBox): 原查询 + HyDE 答案的组合

**🎤 演讲台词**:
> "HyDE 全称 Hypothetical Document Embedding——假设性文档嵌入。它的思路很巧妙：与其直接用短查询去做向量搜索（信息量太少），不如先让 LLM 生成一个'假想的答案文档'，然后用这个长文档去做搜索。实验证明这对提升召回率非常有效。"

### B3: RRF Multi-Channel Fusion（倒数秩融合）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Query | TextBox (1行) | `supply chain risk management` |
| Top-K Results | Slider | `5` |
| **Run RRF Fusion** | Button (secondary) | 👈 点击 |

**输出**:
- **Per-Channel Results** (JSON): vector/graph/keyword 三路各自结果
- **RRF Fused Results** (JSON): 融合后的最终排序

**🎤 演讲台词**:
> "RRF 是 Reciprocal Rank Fusion——倒数秩融合。当我们有多个检索通道（向量通道、图通道、关键词通道），每个通道返回自己的排序结果时，RRF 提供了一种无需训练参数的融合方法。公式很简单：每条结果的分数 = 1/(k + rank)。k 通常取 60。最后按总分重新排序。这是我们三通道融合的核心算法。"

---

## C. 推理与精炼（默认折叠 ▶️）

### C1: DRIFT Multi-Hop Reasoning Search

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Query | TextBox (2行) | `Which suppliers have the highest risk of disrupting warehouse operations?` |
| Communities Top-K | Slider | `5` |
| Language | Dropdown | `cn` / `en` |
| **Run DRIFT Search** | Button (primary) | 👈 点击 |

**输出**:
- **Final Answer** (Markdown): DRIFT 的最终答案
- **Pipeline Trace (5 Steps)** (JSON, 展开): 5 步迭代过程
- **Metadata** (JSON): 执行元数据
- **Top Findings** (JSON): 关键发现列表

**🎤 演讲台词**:
> "DRIFT 是我们实现的多跳推理搜索引擎，对标 MS-GraphRAG 的社区搜索但更轻量。它的工作方式是 5 步迭代循环：
> 1. 从查询中识别种子社区
> 2. 并行扩展多个候选社区
> 3. 对每个社区做局部搜索
> 4. 合并去重
> 5. 用 LLM 综合成最终答案
>
> 它的优势是不需要预建社区索引，实时计算即可。Top-K 参数控制考虑多少个社区，Language 参数控制答案语言。"

### C2: Gleaning（追问提取）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `Warehouse-C risk mitigation strategies` |
| 最大追问轮数 Max Rounds | Slider | `3` |
| **Run Gleaning** | Button (secondary) | 👈 点击 |

**输出**:
- **追问列表 Follow-up Questions** (JSON): LLM 自动生成的追问
- **渐进答案 Progressive Answers** (JSON): 逐轮深化的答案

**🎤 演讲台词**:
> "Gleaning 是追问机制——第一轮回答可能比较浅，Gleaning 会自动生成追问问题，然后在图上找更多信息来深化答案。最多追问 3 轮（可调），每轮都会比上一轮更有深度。"

### C3: Token Budget Control

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Query | TextBox (1行) | `warehouse supplier risk analysis` |
| Max Tokens | Slider | `2000` |
| **Simulate Budget** | Button (secondary) | 👈 点击 |

**输出**:
- **Budget Summary** (JSON): 三级分配（query/entity/relation/chunk 各多少 token）
- **Generated Context** (TextBox): 截断后的实际上下文

**🎤 演讲台词**:
> "Token Budget Control 解决的是上下文窗口限制问题。当你的图很大、相关信息很多时，不能无限制地把所有内容塞给 LLM。Token Budget 做三级分配——查询本身占一些、实体描述占一些、关系路径占一些、chunk 文本占一些。超出的部分按重要性截断。"

---

## D. 可信输出（默认折叠 ▶️）

### D1: Provenance Answer（溯源回答）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `What is the risk score of Warehouse-C?` |
| **Run Provenance Answer** | Button (secondary) | 👈 点击 |

**输出**:
- **溯源答案** (TextBox): 带引用来源的答案
- **溯源路径 Source Provenance** (JSON): 每个事实的来源追踪

### D2: Entity Resolution（实体去重）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Entity Names (one per line) | TextBox (6行) | `Apple Inc.\nApple (fruit)\nAAPL\nMicrosoft Corporation\nMSFT\n微软中国` |
| Resolution Strategy | Dropdown | `hybrid` (推荐) |
| **Resolve Entities** | Button (secondary) | 👈 点击 |

**输出**:
- **Resolution Groups** (JSON): 分组结果
- **Summary** (Markdown): 统计摘要

### D3: Schema Constraint Validation

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Schema JSON | Code (已有预填) | Person/Company + works_at 关系 |
| **Validate Schema** | Button (secondary) | 👈 点击 |

**输出**:
- **Validation Result** (JSON): valid/error count/warnings
- **Status** (Markdown): 通过/失败总结

---

## E. BM25 关键词降级（默认折叠 ▶️）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `张三 货拉拉 订单 #12345` (中文精确匹配场景) |
| Top-K | Slider | `10` |
| **Run BM25 Search** | Button (secondary) | 👈 点击 |

**输出**:
- **BM25检索结果** (JSON): 精确关键词命中的结果

**🎤 演讲台词**:
> "BM25 是我们的秘密武器——四大竞品都没有。为什么需要它？因为向量搜索对精确匹配很不友好。比如用户问 '张三的公司'，如果 '张三' 这个词在 embedding 空间中被泛化了，可能搜不到。BM25 做的是传统的倒排索引精确匹配，专门捕捉专有名词、代码标识符这类向量搜不到的东西。它是降级插件——主路是向量+图，BM25 兜底。"

---

## F. 块图增强（默认折叠 ▶️）

### F1: Chunk Similarity Edges（KNN 建 SIMILAR 边）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Chunk Vertex Label | TextBox | `Chunk` (默认) |
| KNN Top-K | Slider | `3` |
| Min Similarity | Slider | `0.5` |
| **Build SIMILAR Edges** | Button (secondary) | 👈 点击 |

**输出**:
- **Chunk Similarity Result** (JSON): 建了多少条 SIMILAR 边

### F2: Property Graph Extract（独立抽取算子）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Input Text | TextBox (4行) | `Alice works at Google as a software engineer in Mountain View.` |
| Schema JSON (optional) | Code | 留空 = 自由抽取 |
| **Extract Property Graph** | Button (secondary) | 👈 点击 |

**输出**:
- **Property Graph Result** (JSON): 抽取出的属性图

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 传什么 |
|----------|---------|--------|
| **Tab 1** (③ Load into GraphDB 完成) → Tab 3 | 图中有数据后 | 配置并使用各种检索算法 |
| Tab 3 → **Tab 5** | 选好检索策略后 | 去 RagQA 验证效果 |
| Tab 3 (DRIFT) → **Tab 4** | DRIFT 搜不到？ | 用 Agent 做更深推理 |

---

# ══════════════════════════════════════════════════════════
# TAB 4: Agent Global Search（Agent 多步推理）
# ══════════════════════════════════════════════════════════

## 📍 定位
**Phase 3 Agentic RAG**。当固定 pipeline 无法回答时，LLM 驱动的 Agent 自主决定调用哪个工具、走哪条路径。

## 🎯 四个子模块

### 1. Agent 多步推理（ReAct 模式）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `分析 Warehouse-C 的完整风险传导路径：从供应商中断到最终配送延迟，涉及哪些中间节点？每种风险的可控性如何？` |
| 最大步数 Max Steps | Slider | `10` (默认) |
| **🚀 运行 Agent** | Button (primary) | 👈 点击 |

**输出**:
- **Agent 答案** (TextBox, 8行): Agent 的最终回答
- **推理步骤 Trace** (JSON, 折叠): ReAct 的 Thought-Action-Observation 循环

**🎤 演讲台词**:
> "Agent 模块是我们的 Phase 3 能力——Agentic RAG。它使用 ReAct 模式（Reasoning + Acting）：Agent 先思考（Thought）该用什么工具，然后执行（Action），观察结果（Observation），再决定下一步。最大步数限制防止无限循环。
>
> 和 Tab 3 的 DRIFT 不同，DRIFT 是固定 5 步流程，而 Agent 的路径是动态的——它可能先查图、再做向量搜索、再调 Text2Gremlin、再综合判断。适合那种你事先不知道该用什么方法的复杂问题。"

### 2. 全局搜索 Global Search（社区摘要问答）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 全局查询 Global Query | TextBox (2行) | `这份供应链文档的主要主题有哪些？各主题之间的关系是什么？` |
| **🌍 全局搜索** | Button (secondary) | 👈 点击 |

**输出**:
- **全局答案** (TextBox, 8行): 基于社区摘要的回答

**前置条件**: 必须先执行过 **Community Detection（下方的🏗️模块）**

**🎤 演讲台词**:
> "Global Search 是跨文档的主题级问答。它不是针对某个具体实体的，而是回答'这些文档整体讲了什么'这类宏观问题。底层依赖社区索引——先把图划分成若干社区，每个社区生成一段摘要，然后查询时匹配合适的社区摘要来回答。"

### 3. 社区检测 Community Detection

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 算法 Algorithm | Dropdown | `louvain` (推荐) / `wcc` |
| 层级 Levels | Slider | `2` (默认) |
| **🔧 构建社区索引** | Button (secondary) | 👈 点击（只需一次）|

**输出**:
- **状态 Status** (TextBox): 社区数量、报告数量、索引状态

**🎤 演讲台词**:
> "Community Detection 是离线操作，通常只需要执行一次。Louvain 算法是目前最常用的社区检测算法——它通过 modularity 最大化来发现图中的紧密连接组。Levels=2 表示构建两层社区结构，支持不同粒度的全局搜索。"

### 4. 图搜索操作 Graph RAG Search

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 操作模式 Mode | Dropdown | `graph_traverse` (默认) |
| 查询文本 Query | TextBox | `warehouse congestion` |
| 顶点ID列表 Vertex IDs | TextBox | `Supplier-Y, Warehouse-C` (graph_traverse 模式需要) |
| 遍历深度 Max Depth | Slider | `2` |
| 最大结果数 Max Items | Slider | `10` |
| 关键词列表 Keywords | TextBox | `supply, warehouse` (semantic_id_lookup 模式需要) |
| **🔍 执行图搜索** | Button (secondary) | 👈 点击 |

**四种模式说明**:

| 模式 | 需要的输入 | 做什么 |
|------|-----------|--------|
| `graph_traverse` | Vertex IDs | 从指定顶点出发 N-hop 遍历 |
| `semantic_id_lookup` | Keywords | 语义 ID 查找（精确+模糊） |
| `text2gremlin` | Query | 自然语言转 Gremlin 再执行 |
| `schema_lookup` | (自动) | 查询 Schema 结构 |

**输出**:
- **搜索结果** (JSON): 取决于模式

### 5. 查询分类 Query Classifier

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| 查询 Query | TextBox (2行) | `Warehouse-C 的风险评分是多少？` |
| 使用 LLM 精细分类 | Checkbox | `False` (默认用规则) |
| **分类查询 Classify** | Button (secondary) | 👈 点击 |

**输出**:
- **分类结果** (Code JSON): `{"type": "simple"/"complex", "confidence": 0.85, "route": "graph_search"/"agent"}`

**🎤 演讲台词**:
> "Query Classifier 是路由器——它判断一个查询是简单的（直接图搜索就能答）还是复杂的（需要 Agent 推理），然后分发到不同的处理路径。用 LLM 分类更准但有延迟；规则分类更快但在边界 case 可能不准。"

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 传什么 |
|----------|---------|--------|
| **Tab 3** (DRIFT 深度不够) → Tab 4 | 复杂多跳问题 | Agent 自主规划路径 |
| **Tab 5** (答案不满意) → Tab 4 | 需要 Agent 补充推理 | 同上 |
| Tab 4 (text2gremlin 模式) → **Tab 6** | 想单独用 Text2Gremlin | 跳过去 |

---

# ══════════════════════════════════════════════════════════
# TAB 5: Rag QA（问答验证）
# ══════════════════════════════════════════════════════════

## 📍 定位
**端到端问答验证台**。Tab 1 建好索引后的最终验证场所。

## 🎯 区域 1: HugeGraph RAG Query（单问题问答）

### 左列：问题 & 答案展示

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| **Question** | TextBox (3行) | `Which supplier has the highest disruption risk and how does it affect Warehouse-C?` |
| Basic LLM Answer | Markdown | （输出，不可编辑） |
| Vector-only Answer | Markdown | （输出，不可编辑） |
| Graph-only Answer | Markdown | （输出，不可编辑） |
| Graph-Vector Answer | Markdown | （输出，不可编辑） |
| Query Prompt | TextBox (7行) | 保持默认（answer prompt） |
| Keywords Extraction Prompt | TextBox (7行) | 保持默认 |

### 右列：模式开关 & 参数

| 组件 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **Basic LLM Answer** | Radio (True/False) | `False` | 纯 LLM 回答（不走 RAG） |
| **Vector-only Answer** | Radio (True/False) | `False` | 仅向量检索 RAG |
| **Graph-only Answer** | Radio (True/False) | `✅ True`（默认开启） | 仅图检索 RAG |
| **Graph-Vector Answer** | Radio (True/False) | `False` | 向量+图融合 RAG |
| Rerank method | Dropdown | `reranker` | reranker / bleu |
| Template Num | Number | `-1` | Text2Gremlin 参考模板数（-1=禁用） |
| **Graph Ratio** | Slider | `0.6` | 图 vs 向量结果混合比例（仅 Graph-Vector 时可调） |
| Near neighbor first | Checkbox | `False` | 优先近邻 |
| Custom Related Information | Text | 空 | 额外上下文（可选） |
| **Answer Question** | Button (primary) | 👈 **核心按钮** | 流式输出 4 种答案对比 |

**🎤 演讲台词**:
> "这里是最终验证的地方。你问一个问题，系统同时跑四种模式给你看：
>
> **Basic LLM Answer**：完全不走 RAG，就是裸 LLM 回答。作为 baseline 对比。
> **Vector-only Answer**：传统 RAG——向量检索相关 chunk，喂给 LLM。
> **Graph-only Answer**：图 RAG——从图里搜子图，喂给 LLM。**这是我们 HugeGraph 的核心差异化能力。**
> **Graph-Vector Answer**：双向融合——向量和图的结果按比例混合。
>
> 你可以通过四个 Radio 按钮选择开启哪些模式。默认只开 Graph-only。点击 **Answer Question** 后，四种答案会流式输出，方便你横向对比效果。
>
> **实用技巧**：如果你想对比哪种模式更好，把四个全打开，同一个问题跑一次，看哪个答案更准确。"

## 🎯 区域 2: Batch Back-testing（批量回测）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Questions File | File (.xlsx/.csv) | 上传测试题文件 |
| Download Template File | File (预填) | questions_template.xlsx — 下载填好再上传 |
| Max Lines To Show | Number | `1` |
| **Generate Answer (Batch)** | Button (primary) | 👈 批量生成答案 |
| Questions & Answers (Preview) | DataFrame | 显示前 N 行结果 |
| Download Answered File | File | 下载完整答案 xlsx |

**🎤 演讲台词**:
> "批量回测功能让你一次性验证大量问题的回答质量。下载模板，填入问题和期望答案，上传后点击批量生成。系统会用当前配置的所有模式逐一回答，最后输出一个完整的对比 Excel。这在做评测或者回归测试时非常有用。"

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 什么时候来 |
|----------|---------|-----------|
| **Tab 1** (①~④全部完成) → Tab 5 | 索引建好了 | **第一时间来验证** |
| **Tab 3** (调整了检索策略) → Tab 5 | 改了参数 | 重新验证效果 |
| Tab 5 (效果不好) → **Tab 3** | 需要调参 | 回去换策略 |
| Tab 5 (效果不好) → **Tab 4** | 问题太复杂 | 让 Agent 试试 |

---

# ══════════════════════════════════════════════════════════
# TAB 6: Text2Gremlin（自然语言转 Gremlin）
# ══════════════════════════════════════════════════════════

## 📍 定位
**非开发者也能查图**。用自然语言描述你想查什么，自动翻译成 Gremlin 查询语句并执行。

## 🎯 区域 1: 构建模板向量索引（可选）

| 组件 | 类型 | 说明 |
|------|------|------|
| Upload Text-Gremlin Pairs File | File | CSV (`query,gremlin` 格式) or JSON |
| Result Message | TextBox | 构建结果 |
| **Build Example Vector Index** | Button (primary) | 构建相似度匹配索引 |

**预填文件**: 系统自带 `text2gremlin.csv`（已有几组 query-gremlin 对）

**🎤 演讲台词**:
> "Text2Gremlin 的工作原理分为两阶段。第一阶段是**离线构建模板库**——你提供一组 (自然语言问题, Gremlin语句) 对，系统把它们 encode 成向量索引。第二阶段是**在线查询**——用户输入自然语言，系统在模板库中找到最相似的几个模板，把它们的 Gremlin 语句拿出来，要么直接执行，要么作为参考让 LLM 改写后再执行。
>
> 如果你不上传文件，系统自带有几个示例对，可以直接用。"

## 🎯 区域 2: 自然语言转 Gremlin（核心）

### 左列：输入 & 输出

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| **Nature Language Query** | TextBox | `Find all suppliers that have risk score above 0.9` |
| **Similar Template (TopN)** | Code (JS) | （输出）匹配到的模板 |
| **Gremlin With Template** | TextBox | （输出）基于模板改写的 Gremlin |
| **Gremlin Without Template** | TextBox | （输出）纯 LLM 生成的 Gremlin |
| **Query With Template Output** | Code (JSON) | （输出）Template 方式的执行结果 |
| **Query Without Template Output** | Code (JSON) | （输出）非 Template 方式的执行结果 |

### 右列：参数

| 组件 | 类型 | 默认值 |
|------|------|--------|
| Number of refer examples | Slider | `2` |
| **Schema** | TextBox | `hugegraph` 或 JSON |
| **Prompt** | TextBox (20行) | Gremlin 生成 Prompt（保持默认） |
| **Text2Gremlin** | Button (primary) | 👈 点击 |

**🎤 演讲台词**:
> "输入你的自然语言问题，点击 Text2Gremlin。系统会做两件事：
>
> **With Template**（左边的输出）：先在模板库里找相似的 Gremlin，找到后基于这个模板改写。优点是生成的 Gremlin 更规范、更容易执行正确。
> **Without Template**（右边输出）：直接让 LLM 从头生成 Gremlin。优点是灵活性更高，但可能语法有问题。
>
> 两边都会实际执行 Gremlin 并返回结果，所以你能直接对比哪种方式更好。"

## 🎯 区域 3: Gremlin Self-Correction（自纠错）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Natural Language Query | TextBox (2行) | `List all warehouses sorted by their risk scores in descending order` |
| Max Retries | Slider | `3` (默认) |
| Language | Dropdown | `cn` / `en` |
| **Generate & Validate Gremlin** | Button (secondary) | 👈 点击 |

**输出**:
- **Gremlin Retry Result** (Code JSON): 包含尝试历史、最终成功/失败状态

**🎤 演讲台词**:
> "Self-Correction 是 Text2Gremlin 的增强版——它不只生成 Gremlin，还会尝试验证执行。如果执行报错（语法错误、属性不存在等），它会把错误信息反馈给 LLM 让它修正，最多重试 3 次。这大大提高了成功率。"

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 什么时候来 |
|----------|---------|-----------|
| **Tab 1** (③完成) → Tab 6 | 图里有数据后 | 验证数据能否用自然语言查到 |
| Tab 4 (Graph RAG Search) → **Tab 6** | text2gremlin 模式 | 单独深入使用 |
| Tab 8 (Gremlin Console) → **Tab 6** | 手写 Gremlin 太累？ | 用自然语言代替手写 |

---

# ══════════════════════════════════════════════════════════
# TAB 7: Multimodal（多模态 GraphRAG）
# ══════════════════════════════════════════════════════════

## 📍 定位
**Phase 4 多模态能力**。处理含图片、表格、公式的 PDF 文档，构建多模态知识图谱。

## 🏗️ 七大功能区 + Walkthrough Demo

### 左列：操作按钮（按功能区分组）

#### A. Document Parsing（文档解析）

| 按钮 | 功能 | 需要输入？ |
|------|------|-----------|
| **Parse Document (Unified)** | 解析 PDF/DOCX/MD/TXT | 可以上传文件，不上传则返回 demo 数据 |
| **Extract PDF (Images+Text)** | 逐页提取图片和文本 | 同上 |

**预填**: 不上传文件 = 返回内置 demo 数据（3页供应链风控 PDF），**可以直接点击** ✅

**Demo 数据内容**:
- 来源: `demo_supply_chain_report.docx`
- 包含: 3 个 paragraph blocks + 1 张热力图 + 1 张风险表 + 1 个风险公式
- 输出格式: unified IR (blocks + images + tables + equations)

#### B. VLM Description（视觉语言模型描述）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| VLM Provider | Dropdown | `demo` (用预设数据) / `xiaomimo` / `openai` 等 |
| Max Images | Number | `10` |
| **Describe Images (VLM)** | Button (primary) | 👈 直接点（demo模式） |
| **Show Provider Registry** | Button (secondary) | 👈 展示注册的 VLM 提供商 |
| **Async Pipeline Stats** | Button (secondary) | 👈 异步并发统计 |
| **Image Validation Results** | Button (secondary) | 👈 图片尺寸校验结果 |

**🎤 演讲台词**:
> "VLM Description 区域展示了 4 个 operator：
> **VLM Descriptor** — 调用视觉语言模型（如 MiMo-VLM-Pro）为每张图片生成结构化描述，包括对象标签、图表类型、关键洞察。
> **Provider Registry** — 我们适配了 5 家 VLM 商（OpenAI/Ollama/Anthropic/Gemini/Bedrock），统一接口调用。
> **Async VLM Pipeline** — 异步并发处理，信号量控制最大并发数（默认 4），避免 API 限流。
> **Image Dimension Validator** — 在发送给 VLM 之前先检查图片尺寸（读文件头，不用 Pillow），过滤不合格的图片节省 token。"

#### C. MM Analysis（多模态分析）

| 按钮 | 功能 |
|------|------|
| **Analyze (3-Prompt)** | 对图片/表格/公式分别用专门的 prompt 分析 |
| **Surrounding Context** | 提取 sidecar 元素的上下文（前后文） |
| **Chunk Schema Cleanup** | 清理 chunk 中的 markup 标签 |

**全部可直接点击** ✅ (返回 demo 数据)

#### D. KG Build（知识图谱构建）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Target Graph Name | TextBox | `multimodal_poc` (默认) |
| **Inject MM Entities** | Button (primary) | 👈 注入图片/表格/公式作为特殊实体 |
| **Build KG (HugeGraph)** | Button (secondary) | 👈 写入 HugeGraph |

**Demo 输出**:
- 3 个 multimodal entities (Risk Heatmap Chart / Risk Assessment Table / Risk Score Formula)
- 3 条 associated_with 边
- 最终图: 21 vertices, 24 edges

#### E. Retrieval（四通道检索）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| Search Query | TextBox | `supply chain warehouse risk` (已预填) |
| Search Mode | Dropdown | `image_aware` (默认) |
| Top-K Results | Slider | `5` |
| **4-Channel RRF Search** | Button (primary) | 👈 **核心按钮** |
| Comparison Query | TextBox | `warehouse risk heatmap` |
| **Compare text-only vs multimodal** | Button (secondary) | 👈 对比增益 |

**四通道**: Vision(视觉) + Keyword(关键词) + Vector(向量) + Graph(图) → RRF 融合

**Demo 搜索结果** (3 条):
| ID | Type | Score | 主要来源 |
|----|------|-------|---------|
| Warehouse-A | Image | 0.82 | vision: 0.45 |
| txt_1_2 | TextChunk | 0.68 | keyword: 0.40 |
| supply_chain_risk_table | Table(MIXED) | 0.55 | keyword: 0.35 |

**对比结果**: Multimodal 比 text-only 多覆盖 **+200%** 结果（3 vs 1），且增加了视觉和结构化数据上下文。

#### F. Sidecar IR（Sidecar 中间表示）

| 按钮 | 功能 | 输出示例 |
|------|------|----------|
| **OMML→LaTeX Convert** | DOCX 公式 XML 转 LaTeX | `x\sum_{i=1}^{n}w_i` |
| **Placeholder Render** | `{{TBL:k}}/{{IMG:k}}/{{EQ:k}}` → XML 标签 | `<table id="tb-1"...>` |
| **Sidecar IR Structure** | Sidecar IR 数据结构定义 | blocks + assets |
| **Sidecar Writer Output** | parsed/ 目录输出结构 | blocks.jsonl + tables.json ... |
| **Sidecar Backfill Mapping** | chunk → block 反向映射 | 匹配率 100% (2/2) |

**全部可直接点击** ✅

#### G. Pipeline Overview（管线总览）

| 按钮 | 功能 | 输出 |
|------|------|------|
| **Show Pipeline DAG** | 8 节点 DAG 架构图 | JSON 格式的 pipeline 定义 |

**8 节点 DAG**:
```
MultimodalExtractNode → VLMDescribeNode ─┬→ ChunkSplitNode → ExtractNode → MultimodalKGBuildNode → IncrementalUpdateNode → CommitNode
                                      └→ SchemaNode (并行)
```
覆盖 **18 个 operators**。

### 右列：输出展示（Tab 切换）

| Tab 名 | 内容 |
|--------|------|
| Coverage Matrix (18 ops) | 18 行表格：operator 名 / 所属 section / 一句话描述 |
| A. Doc Parse | Unified Parser + PDF Extraction 的 JSON 输出 |
| B. VLM Describe | VLM 描述 + Image Gallery 表格 + Registry/Async/Validate |
| C. MM Analysis | 3-Prompt 分析 + Surrounding Context + Chunk Schema |
| D. KG Build | Entity Injector + KG Build Stats |
| E. Retrieval | 搜索结果 + Channel Scores 表格 + Channel Pipeline + 对比 |
| F. Sidecar IR | OMML/LaTeX/Placeholder/IR/Writer/Backfill 共 6 个输出 |
| Tables & Equations | 渲染后的 HTML 表格 + LaTeX 公式 |
| G. Pipeline DAG | DAG 架构 JSON |
| **H. Walkthrough 🎯** | **6 步引导式 Demo（重点！）** |

### H. Walkthrough Demo（重点推荐 ✅）

**这是整个 Tab 7 最精彩的部分——6 步引导式 Demo，覆盖全部 18 个 operators。**

#### 预设操作流程：

**Step 0: 生成 Demo PDF**
| 按钮 | 功能 |
|------|------|
| **📄 Generate Demo PDF** | 生成一份 3 页供应链风控 PDF（含热力图+表格+公式+网络拓扑图） |

**Step 1~6: 逐步执行**
| 按钮 | 覆盖 Operators | 预计耗时 |
|------|---------------|---------|
| **1️⃣ Parse PDF** | pdf_image_extractor + unified_document_parser + image_dimension_validator | ~3s |
| **2️⃣ VLM Describe** | vlm_descriptor + vlm_provider_registry + async_vlm_pipeline | ~5s |
| **3️⃣ MM Analysis** | multimodal_analyzer + surrounding_context + chunk_schema | ~2s |
| **4️⃣ Formula & Sidecar** | omml_to_latex + sidecar_placeholder + sidecar_ir + sidecar_writer + sidecar_backfill | ~2s |
| **5️⃣ KG Build** | multimodal_entity_injector + multimodal_kg_builder | ~3s |
| **6️⃣ Search** | multimodal_retriever + multimodal_retrieval_channel | ~2s |

#### 5 个预设问题（Step 6 的快捷按钮）：

| 按钮 | 问题 | 目标模态 | 覆盖 Ops |
|------|------|---------|----------|
| **Q1: 📝 哪些节点风险最高？** | 纯文本查询 | text | 6 ops |
| **Q2: 🖼 热力图展示什么？** | 图片查询 | image | 9 ops |
| **Q3: 📐 R_score 公式是什么？** | 公式查询 | equation | 8 ops |
| **Q4: 🔗 仓库和供应商关联？** | 图谱查询 | graph | 4 ops |
| **Q5: 🎯 Warehouse-C 完整评估？** | 混合查询 | **mixed (ALL 18 ops)** | **18 ops** |

还有 **Custom Query** 输入框 + **Search** 按钮，支持任意问题。

**🎤 演讲台词（Walkthrough 部分——这是高潮）**:
> "现在我给大家演示完整的 6 步 Walkthrough。这是我们多模态能力的集中展示，覆盖了全部 18 个 operators。
>
> **首先点 Generate Demo PDF**——这会生成一份 3 页的供应链风控报告 PDF，里面有热力图、数据表、风险公式和网络拓扑图。
>
> **Step 1: Parse PDF** —— 解析器会逐页提取图片和文本块，识别出 3 张图、12 个文本块。
>
> **Step 2: VLM Describe** —— 调用 MiMo-VLM-Pro 为每张图片生成结构化描述。热力图被识别为 heatmap 类型，关键洞察是'仓库拥堵是第 1 风险因素'。
>
> **Step 3: MM Analysis** —— 对图片、表格、公式分别用专门的分析 prompt。表格被识别为 5 行风险评估表，公式被识别为加权求和风险评分公式。
>
> **Step 4: Formula & Sidecar** —— DOCX 里的 OMML 公式被转成 LaTeX，{{TBL:1}} 这类占位符被渲染成实际的 XML 引用，chunk 里回填了 sidecar 引用关系。
>
> **Step 5: KG Build** —— 图片作为 Chart 节点、表格作为 Table 节点、公式作为 Equation 节点注入图库，建立 associated_with 边。最终 21 个点、24 条边。
>
> **Step 6: Search** —— 最后用四通道 RRF 搜索。大家看 Q5 这个按钮——它触发的查询会用到全部 18 个 operators，是终极展示。
>
> 大家可以试试点 Q2（图片问题）或 Q3（公式问题），看看不同模态的查询是怎么利用对应的分析结果的。"

---

## ↔️ 与其他 Tab 的连接

| 从哪里来 | 到哪里去 | 什么时候来 |
|----------|---------|-----------|
| **独立使用** | Tab 7 | 有 PDF/图片文档时直接来 |
| Tab 7 (KG Build 完成) → **Tab 5** | 多模态图谱建好了 | 去 RagQA 验证 |
| Tab 7 → **Tab 8** | 验证数据 | 用 Gremlin 查多模态实体 |

---

# ══════════════════════════════════════════════════════════
# TAB 8: Admin & Ops（管理与运维）
# ══════════════════════════════════════════════════════════

## 📍 定位
**运维工具台**。不涉及算法，纯粹的管理功能。

## 🎯 Section A: Graph Tools（图数据库工具）

### A1: Gremlin Query Console

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| **Gremlin query** | TextBox (8行) | `g.V().limit(10)` (默认) |
| **Output** | Code (JSON) | 查询结果 |
| **Run Gremlin Query** | Button (primary) | 👈 执行 |

**常用 Gremlin 预填查询**（依次演示）：
```gremlin
-- 1. 查看前 10 个顶点
g.V().limit(10)

-- 2. 查看所有边类型
g.E().label().groupCount()

-- 3. 查询 Warehouse 相关的顶点
g.V().has('name', containingText('Warehouse')).valueMap()

-- 4. 统计各类顶点数量
g.V().label().groupCount()

-- 5. 2-hop 邻居
g.V().has('name','Warehouse-C').bothE().otherV().path().by('name').limit(20)
```

**🎤 演讲台词**:
> "Gremlin Console 是最常用的运维工具。你可以直接写 Gremlin 查询来验证数据是否正确写入了 HugeGraph。
>
> 我推荐演示 5 个递进查询：
> 1. `g.V().limit(10)` —— 最基本的，确认图不为空
> 2. `g.E().label().groupCount()` —— 看有哪些类型的边
> 3. 带 has 条件的查询 —— 精确查找某个实体及其属性
> 4. groupCount —— 快速了解图的整体规模
> 5. bothE.path —— 看 2-hop 拓扑，验证关系是否正确"

### A2: Data Backup

| 按钮 | 功能 | 说明 |
|------|------|------|
| **Backup Graph Data Now** | 立即备份数据 | 备份到磁盘 |
| *(BETA) Init HugeGraph Test Data* | 初始化测试数据 | ⚠️ BETA 功能，慎用 |

> "备份功能每天凌晨 1 点自动执行（APScheduler 定时任务）。这里的手动按钮用于临时备份。Init Test Data 是 Beta 功能——它会在图里插入一组预定义的测试数据，适合首次演示时快速初始化环境。"

## 🎯 Section B: Admin Log Viewer（日志查看器，需密码）

| 组件 | 类型 | 预填数据 |
|------|------|----------|
| **Enter Password** | TextBox (password) | 输入 admin_token |
| **Submit** | Button | 验证密码 |
| **LLM Server Log** | Code (20行, auto-refresh 60s) | 日志内容（密码正确后显示） |
| **Clear / Refresh** | Buttons (密码验证后可见) | 清空/刷新日志 |

> "日志查看器是受保护的——需要输入管理员密码（在 config 中配置的 admin_token）。密码正确后会显示 llm-server.log 的最近 125 行，并且每 60 秒自动刷新。可以看到每次请求的 LLM 调用记录、token 消耗、错误信息等。"

---

# ══════════════════════════════════════════════════════════
# TAB 9: Capability Map（能力全景图）
# ══════════════════════════════════════════════════════════

## 📍 定位
**参考面板**。展示 HugeGraph-AI 的完整能力矩阵，不需要交互操作，供查阅。

> "Tab 9 是静态的能力地图——它列出我们支持的每一个 operator、每个 GraphRAG 发展阶段的覆盖情况、以及和竞品的能力对比。这是一个'速查表'，当你忘记某个功能在哪里的时候可以来这里找。
>
> 不需要特别操作，浏览即可。"

---

# ══════════════════════════════════════════════════════════
# 🎬 完整演示流程（推荐顺序）
# ══════════════════════════════════════════════════════════

## 场景：供应链风控知识图谱端到端演示

### Phase 0: 环境准备（2 分钟）

1. 打开 **Tab 8 AdminOps**
   - 点 **Run Gremlin Query**（保持默认 `g.V().limit(10)`）
   - 确认图服务连通 ✅
   - 如果图为空，点 **(BETA) Init HugeGraph Test Data** 初始化测试数据

### Phase 1: Schema 构建（3 分钟）

2. 切换到 **Tab 2 Schema Studio**
   - **Section A**: Mode 选 `batch` → Load Multi-doc Example 选 `"Supply Chain Risk (multi-doc)"` 
   - Instructions 填入: `focus on supply-chain entities, include risk_score as double property`
   - 点 **Generate Schema Draft** → 观察 Schema Preview 和 Schema JSON
   - 点 **Approve & Commit to HugeGraph** → 创建 vertex/edge labels
   - 点 **Export Schema to EDC Guided Mode** → 观察输出的 GUIDED 配置

3. （可选）**Section B**: 展开看 EDC Pipeline 的界面（不需要真的跑）

### Phase 2: 数据导入（5 分钟）

4. 切换到 **Tab 1 Build RAGIndex**
   - 把 Tab 2 生成的 Schema JSON 复制到 **Graph Schema** 框（或填 `hugegraph` 自动拉取）
   - **text** Tab: 粘贴预填的供应链风控文本（见上方 Tab 1 预填数据）
   - 点 **① Import into Vector** → 等 Output Info 显示 chunk 数量
   - 点 **② Extract Graph Data (1)** → 等 Output Info 显示三元组数量
   - 点 **③ Load into GraphDB (2)** → 等写入完成
   - 点 **④ Update Vid Embedding** → 等向量索引更新完成

5. 回 **Tab 8** 点 **Run Gremlin Query** → 用 `g.V().label().groupCount()` 验证数据写入

### Phase 3: 检索增强配置（5 分钟）

6. 切换到 **Tab 3 GraphRAG Core**
   - **A1 PPR**: 填入 `Warehouse-C supply chain risk` → **Run PPR Search**
   - **A2 Cascade**: 填入 `supplier disruption impact` → **Run Cascade**
   - **E BM25**: 填入 `Warehouse-C Supplier-Y` → **Run BM25 Search**
   - （可选展开 B/C/D/E/F 各演示一个）

### Phase 4: 问答验证（3 分钟）

7. 切换到 **Tab 5 Rag QA**
   - Question 填入: `Which supplier has the highest disruption risk and how does it affect Warehouse-C?`
   - 开启 **Graph-only Answer** (Radio=True)
   - 点 **Answer Question** → 观察流式输出
   - （可选）开启全部 4 种模式对比

### Phase 5: 高级能力展示（5 分钟）

8. **Tab 4 Agent**: 填入复杂推理问题 → **运行 Agent**
9. **Tab 6 Text2Gremlin**: 填入自然语言查询 → **Text2Gremlin** → 观察两路输出
10. **Tab 7 Multimodal**:
    - 切到 **H. Walkthrough** Tab
    - 点 **📄 Generate Demo PDF**
    - 依次点 **1️⃣→2️⃣→3️⃣→4️⃣→5️⃣→6️⃣**
    - 最后点 **Q5: 🎯 Warehouse-C 完整评估？** (ALL 18 ops)

### Phase 6: 收尾（1 分钟）

11. **Tab 9 Capability Map**: 浏览能力全景图
12. 总结回顾数据流: `Tab2 → Tab1 → Tab3 → Tab5`（主线）+ `Tab7`（多模态扩展）

---

## ⚠️ 常见报错及预防措施

| 报错 | 原因 | 预防方法 |
|------|------|----------|
| `Please provide original text` | Prompt Generator 的三个输入有空 | 填满 text + scenario + 选 example |
| `No valid schema JSON yet` | Export to EDC 时还没生成 Schema | 先点 Generate Schema Draft |
| `Please select at least one generate mode` | Rag QA 四个 Radio 全 False | 至少开一个（建议 Graph-only） |
| `Please enter a query` | 某个查询框为空就点了按钮 | 所有查询框必须填入数据 |
| `Admin token is not configured securely` | 日志查看器未配置 admin_token | 先在 config 里设置 |
| `Graph connection refused` | HugeGraph 未启动 | 确保 localhost:8080 可访问 |
| `LLM API error` | MiMo API 不可用 | 检查 LLM 配置和网络 |
| `File not found` | Batch 回测没上传文件 | 先下载模板填写再上传 |

---

## 📊 快速参考卡（可打印随身）

```
┌──────────────────────────────────────────────────────────────┐
│           HugeGraph-AI Gradio Demo Quick Reference            │
├──────────────────────────────────────────────────────────────┤
│ Tab 1: BuildRAGIndex     →  ①Import ②Extract ③Load ④VidEmb │
│ Tab 2: SchemaStudio      →  A:AutoSchema  B:EDC  Export桥接  │
│ Tab 3: GraphRAGCore      →  A~F 六段式检索(PPR/Cascade/BM25..)│
│ Tab 4: AgentGlobalSearch →  Agent+Global+Community+GRSearch  │
│ Tab 5: RagQA             →  4种模式单问题+批量回测              │
│ Tab 6: Text2Gremlin      →  NL→Gremlin+SelfCorrection       │
│ Tab 7: Multimodal        →  18 operators + 6步Walkthrough     │
│ Tab 8: AdminOps          →  Gremlin Console+Backup+Log Viewer │
│ Tab 9: CapabilityMap     →  静态能力矩阵（只读）              │
├──────────────────────────────────────────────────────────────┤
│ 主线流程:  Tab2(Schema) → Tab1(Import+Extract+Load) → Tab5(QA)│
│ 多模态:   Tab7(Walkthrough 6步) → Tab5(QA验证)               │
│ 运维:     Tab8(Gremlin 验证) ← 随时可切                     │
└──────────────────────────────────────────────────────────────┘
```
