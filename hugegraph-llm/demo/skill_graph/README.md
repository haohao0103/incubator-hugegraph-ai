# Skill Graph 构建与检索完整流程文档

> 模块位置：`hugegraph-llm/src/hugegraph_llm/engines/skill_graph/`
> 分支：`feature/skill-graph`
> 对标业界方案：**Graph-of-Skills（GoS）**、**SkillDAG**、**SkillNet**

---

## 1. 是什么：一句话与一张图

**一句话**：把扁平的技能库（一堆互不关联的 `SKILL.md`）组织成一张**带类型边的图**，Agent 检索技能时不再只按"文本相似"返回零散条目，而是**自动带出任务所需的完整前置依赖链，并严格控制 token 预算**。

```
                        ┌─────────────────────────────────────────────┐
                        │              离线（L1）一次构建              │
                        │                                             │
  SKILL.md 目录 ──────► │ 加载解析 → 嵌入 → LLM/相似度建边 → 图规整   │
  (anthropics/skills)   │            │                                │
                        │            ▼                                │
                        │   ┌───────────── 权威存储 ─────────────┐   │
                        │   │ HugeGraph（图结构: 节点+边）        │   │
                        │   │ OceanBase（向量+正文+全文索引）     │   │
                        │   └─────────────────────────────────────┘   │
                        │        └── skill_graph.json（构建产物，      │
                        │            仅供离线开发/CI，在线不依赖）     │
                        └─────────────────────────────────────────────┘
                                        │
                        ┌─────────────────────────────────────────────┐
                        │              在线（L2）每次查询               │
                        │                                             │
  用户任务/Query ─────► │ ① 从服务组装图(HG结构 + OB向量/正文)         │
                        │ ② 混合种子检索(BM25+语义, RRF融合)           │
                        │ ③ 反向 Personalized PageRank(沿依赖回传)     │
                        │ ④ 预算化 Hydration(种子优先, 贪心装箱)       │
                        │            │                                │
                        │            ▼                                │
                        │   依赖完备、预算内的技能上下文 → LLM/Agent    │
                        └─────────────────────────────────────────────┘
```

**生产架构 = 服务即真相（services as the source of truth）**：

- **HugeGraph** 是图结构的权威（skill/category 顶点 + 5 类边）；
- **OceanBase** 是内容与向量的权威（embedding + body 全文 + HNSW/ik 双索引）；
- **`skill_graph.json` 只是构建产物**——单机开发、CI、离线演示用；在线查询路径完全不依赖它（见 §5.6 `--from-service`）。多个进程/副本共享的是两个服务，不存在多真相源漂移问题。

---

## 2. L0 本体层：节点与边的类型系统

定义于 [schema.py](../../src/hugegraph_llm/engines/skill_graph/schema.py)。

### 2.1 节点（2 种）

| 类型 | 说明 | ID 规则 |
|---|---|---|
| `skill` | 一个原子/复合能力，对应一个 `SKILL.md` | `skill::<kebab-name>` |
| `category` | 分类枢纽节点（`belongs_to` 的目标） | `category::<kebab-name>` |

节点字段：`name` / `description` / `category` / `tags` / `node_type` / `source_path` / `body`（正文全文）/ `embedding`（向量）/ `metadata`。

- **`index_text`**（属性）= name + description + tags 拼接，是嵌入与 BM25 的索引文本 —— **短文本**，不嵌入正文。
- **`token_estimate()`** = `max(1, ceil(len(body or index_text) / 4))`，即技能被"装进上下文"的粗略 token 成本（~4 字符/token）。
- **`body` 全文始终保留在节点上**（JSON 落盘 + OB `content` 列），Agent 最终拿到的就是这个。

### 2.2 边（5 种，业界标准关系集）

| 边类型 | 语义 | 方向 | 检索时的作用 |
|---|---|---|---|
| `depends_on` | A 需要 B 作为前置条件 | 有向，**DAG 骨干** | 检索时**反向遍历**，自动补齐前置链 |
| `composes_with` | A 与 B 常组合使用 | 对称（双向物化） | 扩展候选组合 |
| `similar_to` | A 与 B 近似/可替代 | 对称 | 备选扩展（默认关闭） |
| `belongs_to` | A 属于类别 B | 有向 skill→category | 分类导航 |
| `conflicts_with` | A 与 B 互斥 | 对称 | **仅用于剪枝**，从不作为正向扩展依据 |

**用大白话再说一遍**（每种边配一个例子，技能库视角）：

- **`depends_on`（依赖）**："要做 A，得先有 B"。例：`rag-pipeline depends_on vector-index` —— 没有向量索引，RAG 流水线根本跑不起来。**这是最值钱的边**：检索"搭一个 RAG"时它保证 vector-index、text-embedding 这些查询里没提的前置被自动带上。
- **`composes_with`（组合）**："A 和 B 经常搭伙干活，谁也不欠谁"。例：`docx composes_with brand-guidelines` —— 写 Word 文档时顺手套用品牌规范，但两者可独立使用。
- **`similar_to`（相似/可替代）**："A 和 B 干的是差不多的事，选一个就行"。例：两个不同作者写的"格式转换"技能。用途是备选：首选技能超预算时，给替代品。
- **`belongs_to`（分类）**："A 是 B 这个分类底下的"。例：`pptx belongs_to documents`。纯组织结构，不参与依赖推理。
- **`conflicts_with`（互斥）**："A 和 B 不能一起用"。例：两个操作同一个 Git 分支的自动化技能。**它从来不扩展结果，只负责把矛盾剪掉** —— 一对技能已有正向关系又声明互斥时，删掉互斥边（以正向关系为准）。

每条边带 `weight`（相似度/置信度）与 `provenance`（`explicit` 声明 / `similarity` 相似度 / `llm` LLM 判定 / `heuristic`）—— 边的来源可追溯。

### 2.3 图的不变量（构建时强制）

1. **无自环、无重复边**（`(source, target, edge_type)` 三元组去重）。
2. **依赖无环**：`depends_on` 构成 DAG。检测到环时，删环内**权重最小**的边断环，并记入 `BuildReport.cycles_broken`。
3. **冲突一致性**：若同一对技能同时存在正向关系（depends_on/composes_with/similar_to）与 `conflicts_with`，则删冲突边（SkillDAG 的"冲突只用于剪枝"原则）。
4. **对称边物化**：`composes_with`/`similar_to`/`conflicts_with` 单向声明后自动补反向边。

---

## 3. 数据源：SKILL.md 格式

加载器 [skill_loader.py](../../src/hugegraph_llm/engines/skill_graph/skill_loader.py) 递归扫描目录下所有 `SKILL.md`，解析 YAML front-matter + 正文：

```markdown
---
name: rag-pipeline
description: Compose retrieval and generation into an end-to-end RAG flow
metadata:
  category: rag
  tags: [retrieval, generation, pipeline]
relations:            # 显式声明（可选；业界数据集通常没有）
  depends_on: [vector-index, text-embedding]
  composes_with: [evaluation]
---

（正文 body，Agent 实际消费的完整内容）
```

规则：
- `name` 缺省取父目录名（Anthropic 技能约定）；`metadata.category` 自动物化出 `category` 节点和 `belongs_to` 边。
- `relations` 支持别名：`dependencies`/`requires`/`prerequisites`→depends_on，`compose_with`→composes_with，`similar`/`alternative_to`→similar_to。
- **业界公开数据集**（`anthropics/skills`，19 个技能，由 `tests/download_skills_dataset.py` 下载）**不带任何 relations** —— 所以必须靠 L1 的边诱导层建边（见 §4.3）。
- 解析完全确定性、不调 LLM；PyYAML 不可用时退化为内置的迷你 YAML 解析器。

### 3.1 快速接入：用户只需要写"两行"数据

上面的完整格式看起来吓人，但**用户实际只需要提供一个 `SKILL.md`，其中必填的只有 `name` 和 `description`**，其余全部可选：

```markdown
---
name: pdf-form-filling
description: Detect, fill and flatten interactive AcroForm fields in PDF documents
---

（正文：怎么用这个技能，写给 Agent 看的操作说明）
```

- 没有 `metadata.category`？节点就不挂分类，检索照常工作。
- 没有 `relations`？**根本不需要** —— `--edge-mode llm` 会自动判断它和其他技能是依赖、组合还是相似关系（业界数据集 `anthropics/skills` 就是这么构建的，19 个技能全部零声明）。
- 正文可以任意长，写得越清楚 Agent 用得越好，但格式随意。

**三种接入深度，按需选择**：

| 深度 | 用户写什么 | 边怎么来 | 适合谁 |
|---|---|---|---|
| **零门槛** | 只写 name + description + 正文 | 全部由 LLM 自动诱导（`--edge-mode llm`） | 大多数用户；Anthropic 数据集就是这个形态 |
| **半托管** | 零门槛基础上，个别**确定**的关系写进 `relations` | 显式边直接采信，其余 LLM 补全 | 知道某几条硬依赖的作者 |
| **全显式** | 写全 `relations` + category | 可用 `--edge-mode similarity` 纯离线构建，零 LLM 成本 | 内部受控语料、CI 环境 |

> 显式边（`provenance=explicit`）是作者的声明，**永远不会被构建器删掉**；LLM/相似度边是推断，技能内容变化时会自动重推。

**新技能进来时的完整接入动作只有一条命令**（增量构建，见 §4.8）：

```bash
python -m hugegraph_llm.engines.skill_graph.cli build \
  --skills-dir <新技能所在目录> \
  --base demo/skill_graph/skill_graph.json \
  --out demo/skill_graph/skill_graph.json \
  --embedder project --edge-mode llm
```

---

## 4. L1 离线构建层：六步流水线

构建器 [graph_builder.py](../../src/hugegraph_llm/engines/skill_graph/graph_builder.py)，入口 `GraphBuilder.build(graph) -> (graph, report)`：

```
(1) 嵌入每个技能 ──► (2) 丢弃悬空边 ──► (3) 边诱导 [LLM 或 相似度]
                                              │
(6) 落盘 JSON ◄── (5) DAG 断环 ◄── (4a) 对称闭包 + (4b) 冲突剪枝
        │
        ├──► (7) write_to_hugegraph  —— 图存储
        └──► (8) write_to_oceanbase  —— 向量+全文存储
```

### 4.1 第 1 步：嵌入（embedder.py）

对每个技能的 `index_text`（name+description+tags，短文本）计算向量。嵌入器三档解析（`resolve_embedder()`）：

1. **`HttpEmbedder`**：`OPENAI_EMBEDDING_URL` 设置时启用。适配非标准端点路径（如网关的 `/embedding` 而非 `/v1/embeddings`）+ 自定义请求头（`OPENAI_DEFAULT_HEADERS`），支持批量 + 失败时逐条回退。生产 demo 用它（1536 维）。
2. 项目 `Embeddings` 后端（OpenAI/Ollama/LiteLLM）。
3. **`HashingEmbedder`**：特征哈希（词 + bigram + 字符三元组，支持中文），确定性、零依赖、零网络 —— 单测与离线场景的默认值。

### 4.2 第 2 步：清边

显式声明的边若端点缺失（引用了不存在的技能）直接丢弃，计入 `dangling_edges`。

### 4.3 第 3 步：边诱导（核心，二选一）

#### 模式 A：`--edge-mode llm`（业界标准，SkillDAG 配方）

[edge_inducer.py](../../src/hugegraph_llm/engines/skill_graph/edge_inducer.py) 的 `LLMEdgeInducer`，四阶段：

1. **双视图生成**：对每个技能调一次 LLM（`needs` prompt），让它用一句话回答"**用好这个技能需要什么前置能力**"，得到第二视图文本 `e_needs`（截断 400 字符后嵌入）。第一视图 `e_self` 就是第 1 步的技能嵌入。
   > 为什么双视图：`e_self` 只能发现"像不像"（相似），`e_needs` 才能发现"谁需要谁"（依赖）。这是 SkillDAG 冷启动建图的关键设计。
2. **候选对召回**（便宜的召回层，不是最终判定）：任意技能对 `(A,B)` 打分 =
   `max(cos(A.self, B.self), cos(A.needs, B.self), cos(A.self, B.needs))`，
   超过阈值（真实嵌入建议 ≥0.5）进入候选，每技能最多 top-k（默认 5），全局上限 `max_pairs`（默认 200，成本护栏）。
3. **LLM 分类**（每对一次调用）：给分类器 `classify` prompt（A/B 的 name+description），要求返回 JSON `{"type": "depends_on|composes_with|similar_to|conflicts_with|none"}`。解析容错：先取 ```json 围栏，再取裸 `{}`。非法输出一律按 `none` 丢弃。
4. **依赖自动定向**：`depends_on` 不接受 LLM 的方向顺序，而由几何证据重定向 —— 比较 `cos(A.needs, B.self)` vs `cos(B.needs, A.self)`，**谁的需求更像对方的本体，谁就是依赖方**（边从依赖方指向提供方）。

每条加入的边 `provenance="llm"`，weight=候选相似度。

#### 模式 B：`--edge-mode similarity`（默认，确定性）

纯嵌入余弦诱导 `similar_to` 边：每技能提名 top-k 相似邻居，**只有互提名的对**（mutual kNN）才建边 —— 避免单向枢纽噪声。真实数据集下此模式只会产出"读起来像"的边，因此业界数据集构建请用模式 A。

### 4.4 第 4 步：图规整

- **对称闭包**：对称边补反向（`symmetric_edges_added`）。
- **冲突剪枝**：`prune_conflicts()` 删与正向关系矛盾的 `conflicts_with`。

### 4.5 第 5 步：DAG 断环

循环检测（DFS 三色标记）→ 找到环 → 删环内权重最小的 `depends_on` 边 → 重复直到无环。保证检索时依赖遍历**确定性**（不会绕环）。

### 4.6 第 6-8 步：落盘（三种存储，一份数据三份视图）

| 存储 | 入口 | 内容 | 用途 |
|---|---|---|---|
| **HugeGraph**（图结构权威） | `write_to_hugegraph()` | skill/category 顶点 + 5 类边 | 图查询/可视化/多跳分析；**在线查询路径的结构来源**（`--from-service`）；graphspace 支持（1.7+） |
| **OceanBase**（内容/向量权威） | `write_to_oceanbase()` | 每技能一行：embedding + surface_text(索引文本) + content(body 全文) | **向量检索（HNSW）+ 全文检索（ik 分词）双通道**；**在线查询路径的 embedding/body 来源** |
| **JSON**（构建产物） | `graph.save(path)` | 节点（含 body 全文 + 向量）+ 全部边 | 单机开发/CI/离线演示的可快照产物；生产在线路径**不依赖**它 |

**HugeGraph 写入细节**（graphspace `dataleap` / 图 `skillgraph`）：
- 顶点用 `CUSTOMIZE_STRING` ID，直接复用稳定 ID（`skill::pptx`）→ Gremlin 可 `g.V('skill::rag-pipeline')` 精确定位。
- Schema（property key / vertex label / edge label）按需幂等创建。
- `--clear` 通过 Gremlin `g.V().hasLabel('skill').drop()` 清旧数据。

**OceanBase 写入细节**（表 `skill_graph_skill_semantic`，独占前缀隔离）：
- `embedding VECTOR(1536)` + **HNSW 向量索引**（VSAG 库，COSINE 距离，M=16, EF_CONSTRUCTION=200, EF_SEARCH=64）。
- `surface_text` 全文索引，**ik 分词器 smart 模式**（支持中文）。
- `content longtext` 存 body 全文（长技能如 claude-api ~105KB，TEXT 65535 不够必须 longtext）。
- Upsert（`ON DUPLICATE KEY UPDATE`）幂等，重建不产生脏数据。

### 4.7 构建产物示例（真实运行结果）

| 数据集 | 节点 | 边构成 |
|---|---|---|
| builtin RAG 技能（10 技能+5 类别） | 15 | 12 depends_on + 10 belongs_to + 60 composes_with，0 环 |
| anthropics/skills（19 技能） | 19 | 70 composes_with + 2 similar_to（LLM 判定文档类技能为组合关系，符合直觉），0 环 |

`BuildReport`（JSON）逐项给出：`skills/categories/explicit_edges/similarity_edges/llm_edges/llm_induction{needs_generated, candidate_pairs, classified_pairs, by_type}/symmetric_edges_added/pruned_conflicts/cycles_broken/dangling_edges/embedding_dim`。

### 4.8 增量构建：不必每次全量

全量构建在技能库变大后有两个成本痛点：LLM 建边是 O(全部技能) 的调用（19 个技能 ≈ 19 次 needs + 上百次分类），重新嵌入也要重算所有节点。**增量构建**（`--base`）只处理新增/变化的技能：

```bash
python -m hugegraph_llm.engines.skill_graph.cli build \
  --skills-dir <技能目录> \
  --base demo/skill_graph/skill_graph.json \   # 已有图：作为基底
  --out demo/skill_graph/skill_graph.json \    # 覆盖写回
  --embedder project --edge-mode llm
```

`--skills-dir` 指向**新增技能所在目录**（或全量目录，引擎自动 diff），与 base 比对后：

| 变化类型 | 处理 |
|---|---|
| **新技能** | 加入图 → 重嵌入 → 调 LLM 生成 `e_needs` → 与全图候选打分分类建边 |
| **内容变化**（body/description/tags 任一改变） | 更新节点 → **删除其全部旧诱导边**（similarity/llm，双向）→ 重新嵌入 + 重新诱导；其显式边也按新 front-matter 重新声明 |
| **未变化** | **零成本**：嵌入、`e_needs` 视图、已有边全部复用，不调一次 LLM |

正确性保证：

- **旧-旧技能对不重新分类**：候选只从新/变更技能出发（对全图打分），上次判过的关系不重复花钱。
- **`e_needs` 复用**：needs 文本持久化在节点的 `metadata['needs']`（JSON 里），旧技能直接取用。
- **图不变量全图重跑**：对称闭包、冲突剪枝、DAG 断环是纯本地计算，每次增量后对整图重新执行，合并结果永远合法。
- **嵌入维度校验**：base 图与新 embedder 维度不一致（如换嵌入模型）直接报错，防止混合维度破坏余弦计算 —— 此时必须去掉 `--base` 全量重建。
- **历史边保留**：增量不删未涉及技能的旧边。与全量重建相比，增量图的边集是全量的**超集**（实测 19 技能分两批增量：56 边 vs 全量 44 边，多出的 12 条是第一批时期建立的相似边，新技能加入后依然有效，予以保留）。要得到与全量完全一致的图，去掉 `--base` 重建即可。

输出中 `incremental.rebuilt_skills` 列出本次真正重建的技能；无变化时为空数组（幂等重跑零成本）。**限制**：从目录中删除技能不会把它从图中移除（`provenance=explicit` 的边是作者声明，目录变化不自动清理）；需要"删干净"时用全量重建。

实测成本（LLM 模式）：19 技能全量构建 ≈ 19 次 needs + ~140 次分类；此后每加 1 个技能的增量构建 = **1 次 needs + ≤ top-k 次分类**，与图的总规模无关。

---

## 5. L2 在线检索层：三步检索（Graph-of-Skills 配方）

检索器 [retriever.py](../../src/hugegraph_llm/engines/skill_graph/retriever.py)，入口 `SkillRetriever.retrieve(query) -> RetrievalResult`。纯 Python、确定性，查询时不需要外部图/向量库（JSON 内存图即可跑；OB 通道用于服务化场景）。

### 5.1 第 ① 步：混合种子检索（retrieve_seeds）

查询同时跑两个通道，**不信任单通道**：

- **词法通道 BM25**：对全体技能的 `index_text` 建倒排（k1=1.5, b=0.75），query 分词后打分排序。
- **语义通道**：query 嵌入（与构建同嵌入器），与每个技能向量算余弦排序。

两份排名用 **RRF（Reciprocal Rank Fusion，k=60）** 融合：

```
score(d) = Σ_channel  1 / (60 + rank_channel(d) + 1)
```

RRF 只用**排名**不用原始分值 → 两通道分值分布不同（BM25 无界、余弦 [-1,1]）也能稳健融合；且一个通道完全 miss 时另一通道仍能命中。取 top-k（默认 5）为**种子**，即"用户任务直接相关的技能"。

### 5.2 第 ② 步：反向 Personalized PageRank（_reverse_ppr）

从种子集合沿**依赖反方向**（技能 → 它的 depends_on 目标，含 composes_with）传播概率质量：

- 个性化向量：种子均分 1.0，非种子为 0。
- 阻尼系数 0.85，迭代 60 轮收敛；悬挂节点（无出边的技能）质量按个性化向量回流。
- 传的是**边的 weight**（归一化），所以 LLM 给的相似度置信度会传导为"前置的重要性"。

**为什么是"反向"**：用户的种子是"要做成的事"，而做成它需要的东西在 depends_on 的**上游**。正向 PPR 会把质量推向"依赖我的人"（下游消费者，任务无关）；反向 PPR 让**整条前置链点亮**并带梯度 —— 离种子越近、被越多种子需要的前置，分数越高。这是 GoS"依赖感知"的核心。

同时做一次 **BFS 依赖遍历**（_walk_dependencies）：沿 depends_on（+composes_with，+可选 similar_to）逐层展开，记录每个新节点的 `reason`（prerequisite/composition/similar）与 `depth`（离种子几跳）。

### 5.3 第 ③ 步：预算化 Hydration（retrieve 装箱）

把候选装进 token 预算（默认 4000），**贪心、种子绝对优先**：

```
候选顺序 = [种子(按RRF融合分降序)] + [图扩展节点(按PPR质量降序)]
for 候选:
    cost = token_estimate (正文长度/4)
    if cost > 剩余预算 且 结果非空: 跳过并累计 dropped_tokens
    else: 装入，扣预算
```

- **种子先于一切扩展节点消费预算**：用户直接匹配的技能永远不被后到的小技能挤掉。
- 第一条**永远装入**（预算再小结果也不为空）。
- 超预算的技能计入 `dropped_tokens` —— 这正是相对扁平技能库省下 token 的量化证据。

### 5.4 输出：RetrievalResult

每个技能带**完整可解释性**：`reason`（seed/prerequisite/composition/similar）、`depth`、`score`、`token_estimate`、`body`。两种渲染：

- `to_context()`：Markdown 技能上下文块（`## name _(reason=..., depth=...)_` + 描述 + 正文），直接拼进 Agent prompt。
- `to_dict()`：结构化 JSON（`--no-body` 可省正文）。

### 5.5 真实检索演示（端到端验证结果）

**查询 1**（builtin RAG 图）：`"build an end-to-end RAG pipeline"`，top-k=3，预算 4000：

| 技能 | reason | depth |
|---|---|---|
| rag-pipeline / data-ingestion / knowledge-graph-build | seed（RRF 融合命中） | 0 |
| **text-embedding / vector-index / prompt-engineering / reranking** | **prerequisite**（自动补齐！查询里一个字没提） | 1 |
| text-chunking / cypher-query / evaluation | composition | 1 |

无关节点（如 legacy-search）不被带入 —— 图的剪枝价值。

**查询 2**（anthropics 业界图）：`"create a professional PowerPoint presentation from a Word document"`，top-k=3，预算 12000：

- 种子：docx(1486t) + pptx(4955t) + canvas-design(2892t) —— 语义通道对 pptx 相似度 0.83，词法通道命中 docx。
- 组合扩展：pdf、brand-guidelines、internal-comms（PPR 降序填充）。
- 总计 11965/12000 tokens，dropped 57262 tokens —— **预算裁掉了 5.7 万 token 的无关技能正文**。

**查询 3**（OB 通道，中文）：`"做一个检索增强生成 RAG 流水线"`
- OB 向量检索（HNSW）：rag-pipeline **0.849**、data-ingestion 0.747、cypher-query 0.735。
- OB 全文检索（ik）：`"powerpoint presentation"` → pptx 命中。

### 5.6 生产模式：`--from-service`（服务即真相，无本地文件）

`query` / `stats` / `viz` 都支持 `--from-service`：图从**两个服务**组装（[service_loader.py](../../src/hugegraph_llm/engines/skill_graph/service_loader.py)），全程不碰任何本地 JSON：

```bash
python -m hugegraph_llm.engines.skill_graph.cli query \
  "create a professional PowerPoint presentation from a Word document" \
  --from-service --top-k 3 --token-budget 12000
# HugeGraph: g.V().hasLabel('skill','category') + g.E() 拉全部结构
# OceanBase: SELECT node_id, embedding, content 补齐向量与正文
```

- 组装结果与 JSON 路径**逐字节一致**（实测 §5.5 查询 2：seeds docx/pptx/canvas-design + 组合扩展 pdf/brand-guidelines/internal-comms，11965/12000 tokens）——证明 JSON 只是同一份服务数据的快照，不是独立真相。
- HG 有而 OB 没有的节点（如 category 枢纽）自动降级到词法通道，不报错。
- `--from-service` 下 embedder 自动切到 `project`（必须与构建时同款，否则语义通道失真）。
- **规模注记**：种子检索当前在内存图上跑（BM25 + 余弦），万级技能内无压力；再往上应把种子通道换成 OB 的 HNSW/ik 原生检索（`ob_store.vector_search/fulltext_search` 已具备，语义通道已有生产实测：§5.5 查询 3），PPR/依赖闭包仍在内存图上做（GoS 标准设计——图算法天然适合内存，服务端只负责权威存储与检索）。

---

## 6. L3 集成层：CLI 与配置

CLI [cli.py](../../src/hugegraph_llm/engines/skill_graph/cli.py)，`python -m hugegraph_llm.engines.skill_graph.cli <子命令>`（也挂载为项目 skill 子命令）。

### 6.1 构建

```bash
python -m hugegraph_llm.engines.skill_graph.cli build \
  --skills-dir demo/skill_graph/skills/skills \      # SKILL.md 目录
  --out demo/skill_graph/skill_graph.json \          # 权威 JSON 落盘
  --embedder project \                               # 生产嵌入(OPENAI_EMBEDDING_URL)
  --edge-mode llm \                                  # LLM 驱动建边(业界标准)
  --similarity-threshold 0.5 --max-similar 5 --max-pairs 200 \
  --write-hugegraph \                                # → HugeGraph
  --hugegraph-url http://10.129.33.121:8080 \
  --graphspace dataleap --graph-name skillgraph --clear \
  --write-oceanbase                                  # → OceanBase
```

输出 JSON：`stats`（图统计）+ `report`（构建六步计数）+ `hugegraph`/`oceanbase`（写入计数）。

### 6.2 检索 / 统计 / 可视化

```bash
# 生产模式：从 HugeGraph + OceanBase 组装图（服务即真相，无本地文件）
... query "create a professional PowerPoint presentation from a Word document" \
    --from-service --top-k 3 --token-budget 12000

# 离线开发：从 JSON 构建产物加载（CI/单机调试用）
... query "build an end-to-end RAG pipeline" \
    --graph demo/skill_graph/builtin_graph.json --embedder project --top-k 3
#   --json 结构化输出   --no-body 只出元信息   --token-budget N   --include-similar

# 图统计（两种来源都支持：--from-service 或 --graph）
... stats --from-service

# 中文交互式可视化（检索结果高亮）
... viz --from-service \
    --query "create a professional PowerPoint presentation" \
    --token-budget 12000 --out demo/skill_graph/skill_graph.html --title "..."
```

### 6.3 环境配置（.env，构建/检索共享）

```ini
# 嵌入（HttpEmbedder 适配非标准路径 + 自定义头）
OPENAI_EMBEDDING_URL=http://<gateway>/embedding
OPENAI_DEFAULT_HEADERS=hller=<your-id>
# Chat LLM（LLM 建边；enable_thinking=false 让网关推理模型非流式快速返回）
OPENAI_CHAT_API_BASE=http://<gateway>/v1
OPENAI_CHAT_LANGUAGE_MODEL=qwen3.6-plus
OPENAI_CHAT_DISABLE_THINKING=true
# OceanBase（向量+全文底座；表用 skill_graph_ 前缀隔离）
OCEANBASE_HOST=...aliyuncs.com   OCEANBASE_PORT=3306
OCEANBASE_USER=...  OCEANBASE_PASSWORD=...  OCEANBASE_DATABASE=dataleap_graph_rag
OCEANBASE_TABLE=skill_graph_skill_semantic
# HugeGraph（图底座，1.7+ graphspace）
SKILL_GRAPH_HG_URL=http://10.129.33.121:8080
SKILL_GRAPH_HG_GRAPHSPACE=dataleap
SKILL_GRAPH_HG_GRAPH=skillgraph
```

### 6.4 图查询（HugeGraph Gremlin，服务侧消费）

```groovy
// rag-pipeline 的直接前置
g.V('skill::rag-pipeline').out('depends_on').values('node_name')
// → [reranking, vector-index, text-embedding, prompt-engineering]

// docx 的组合技能（对称边已物化，both 可用）
g.V('skill::docx').both('composes_with').values('node_name')
```

> 注意：`node_name` 未建二级索引时，服务端会拒绝 `has('node_name',...)` 查询；请按稳定 ID 查询（`g.V('skill::...')`）。

---

## 7. 关键设计决策（为什么这么设计）

| 决策 | 依据 |
|---|---|
| 5 种带类型边，而非单一"相似"边 | SkillNet 关系集 + SkillDAG：`conflicts_with` 只用于剪枝；`depends_on` 构成 DAG 骨干 |
| 建边用 LLM 分类器而非纯余弦 | 纯相似度只能产出 similar_to（"读起来像"），无法产出 depends_on（"谁需要谁"）；SkillDAG 双视图 + LLM 分类是业界冷启动标准做法 |
| `e_needs` 第二视图由 LLM 生成 | 技能文本只描述"它做什么"，不描述"用它需要什么"；缺失的视图只能补造 |
| 依赖方向由 `cos(needs, self)` 几何证据定向 | LLM 对方向的判断不稳定；嵌入几何提供可复现、可测试的定向 |
| RRF 融合而非加权重排 | 跨分值分布稳健，免调参；某通道 miss 不影响另一通道（GoS 做法） |
| 反向 PPR 而非正向 | 种子是"目标"，前置在 depends_on 上游；反向传播才点亮前置链 |
| 预算化 hydration 且种子绝对优先 | GoS：控制上下文窗口；种子是用户意图的锚点，必须保住 |
| body 全文同时存 JSON + OB content 列 | 元数据/索引文本（surface_text）供检索，正文（content/body）供 Agent 最终消费，缺一不可 |
| 服务即真相：HG=结构权威、OB=内容权威、JSON=构建产物 | 生产上多进程/多副本共享的必须是服务而非本地文件（无并发、无漂移）；`--from-service` 与 JSON 路径实测结果一致，JSON 只是快照 |
| 增量构建按内容 diff、needs 视图持久化复用 | 技能库持续增长时全量重构建成本 O(n) LLM 调用不可接受；持久化 `e_needs` 使增量成本只与新技能数相关（SkillDAG 部署形态） |

## 8. 模块文件速查

| 文件 | 层 | 职责 |
|---|---|---|
| `schema.py` | L0 | 节点/边类型、环检测、冲突剪枝、JSON 持久化 |
| `skill_loader.py` | L1 | SKILL.md → 节点+显式边（确定性，零 LLM） |
| `embedder.py` | L1/L2 | HashingEmbedder / HttpEmbedder / resolve_embedder / tokenize |
| `edge_inducer.py` | L1 | LLM 双视图候选召回 + 分类 + 依赖定向（SkillDAG） |
| `llm.py` | L1 | GatewayChatLLM（适配网关推理模型）+ resolve_chat_llm |
| `graph_builder.py` | L1 | 六步构建编排 + write_to_hugegraph |
| `ob_store.py` | L1/L2 | OceanBase 表 schema（HNSW+ik）/ upsert / 向量与全文查询 |
| `service_loader.py` | L2 | 生产模式图组装：HG 拉结构 + OB 补向量/正文（`--from-service`） |
| `retriever.py` | L2 | BM25+RRF 种子、反向 PPR、预算化 hydration |
| `viz.py` | L3 | 中文交互式 HTML 可视化（检索高亮） |
| `cli.py` | L3 | build / query / stats / viz 子命令 |
| `tests/unit/test_skill_graph.py` | L3 | 21 个单测覆盖全部层（增量构建、服务加载 Gremlin 装配、OB 向量解析） |

测试：`python -m pytest tests/unit/test_skill_graph.py -q` → 21 passed。