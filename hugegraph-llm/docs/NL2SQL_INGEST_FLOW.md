# NL2SQL 写入链路梳理（平台侧数据 → 我们的语义层）

> 版本：v1（2026-08-30）｜路径：`hugegraph-llm/src/hugegraph_llm/nl2sql/`
> 本文回答三个问题：平台写入数据走哪个接口？请求长什么样？进来之后每层类/方法干什么、落到哪。

---

## 0. 总览

**生产主链路 = 先写 HugeGraph 图（持久化），再从图读（不允许内存图）**：

```
平台结构化元数据(SchemaMetadata JSON)
 └─[写] scripts/ingest_metadata_to_hg.py::ingest() ──> HugeGraph kg_rag 图
                                                        (Table/Field/Metric/Query 顶点
                                                         + hasColumn/computedFromField 边)
 └─[读] hugegraph_schema_source.py::build_schema_from_hugegraph() ──> SchemaGraph
 └─[入] nl2sql_api.py::nl2sql_load_hugegraph() / get_pipeline() ──> NL2SQLPipeline
 └─[查] /nl2sql/link | /join_path | /schema_context

飞书文档 ── RAG 文档导入(GraphRAGIndexFlow) ──> chunk→向量库, 实体→图（并行线）
```

> `/nl2sql/reload`（内存直灌）仅用于本地快速验证，**生产不允许走内存图**。

---

## 1. 生产写入：`scripts/ingest_metadata_to_hg.py`（持久化到 HugeGraph）

**入口函数**：`ingest(meta: dict, url: str, graph: str, clear: bool = False) -> dict`
（CLI：`python scripts/ingest_metadata_to_hg.py --meta <SchemaMetadata.json> --url http://127.0.0.1:8081 --graph kg_rag [--clear]`）

**输入**：SchemaMetadata JSON（同 reload 的 `tables/columns/foreign_keys/lineage/query_logs/terms/term_bindings`）。

**写入内容（与读取 loader 严丝合缝）**：
- Table 顶点：label=Table，name=**裸表名**（"dw.orders"→"orders"），comment（PRIMARY_KEY id）
- Field 顶点：label=Field，name="**table.column**"（"orders.order_id"），comment，type
- Metric 顶点：label=Metric，name=指标名，formula=表达式，definition=口径
- Query 顶点：label=Query，schema_refs="t1;t2;..."（共现，loader 读取）
- hasColumn 边：Table→Field（结构完整性）
- computedFromField 边：Metric→Field（**指标→列绑定，loader 读取**）

**内部实现要点**：
- 底层走 HugeGraph REST（`urllib` + `ProxyHandler({})` 绕代理、自动解 gzip、batch 批量接口）
- PRIMARY_KEY 顶点 id 是 `<label_id>:<name>`（如 "1:orders"），**写顶点后回读真实 id 再建边**
- 清图：按 label 回读 id 逐个 DELETE（REST 不支持 label 级批量删除；`/gremlin` 未绑 `g`）

**验证**：`scripts/e2e_ingest_load.py`（corpus → 灌图 → loader 读回），
12表/73列/11指标/101边 与输入完全一致（PASS），11 个指标绑定 + 17 个 `*_id` FK 推断全部正确。

## 2. 生产读取：`POST /nl2sql/load_hugegraph`（从图读）

**HTTP 层**：`nl2sql_api.py::nl2sql_load_hugegraph(req: HgLoadRequest)`（约 L376）

```
HgLoadRequest {url=http://127.0.0.1:8081, graph=kg_rag, infer_foreign_keys=true, use_embedding=false}
```

**处理流程**：

```
nl2sql_load_hugegraph(req)
 └─ build_schema_from_hugegraph(url, graph, infer_foreign_keys)
     （hugegraph_schema_source.py）
     ├─ urllib 请求 HG REST（ProxyHandler({}) 绕代理，自动解 gzip，page 分页+1000页守卫）
     ├─ 拉顶点：Table / Field(name 按首个"."拆 table.column) / Metric / Query(schema_refs)
     ├─ 拉边：hasColumn(Table→Field) / computedFrom/computedFromField(Metric→Field)
     │        / dependsOn / schema_refs(分号分隔表名)→CO_OCCUR
     ├─ 弱FK启发式：无声明FK时按 *_id 同名列推断
     └─ 返回 SchemaGraph（与 build_schema 产物同构）
 └─ embedder = _make_embedder() if use_embedding else None
 └─ NL2SQLPipeline(schema, engine=_make_engine(schema), embedder=embedder)
```

**落点**：替换进程内缓存 `_PIPELINE`；或起服务时设 `NL2SQL_HG_GRAPH=kg_rag`（`get_pipeline()` 首建时自动从图拉）。

---

## 3. ~~入口 1：POST /nl2sql/reload~~（仅本地验证，生产不用）

---

## 4. 入口 3：飞书文档（RAG 线，与 NL2SQL 并行）

文档是另一条管线（hugegraph-llm 既有能力），存法不同：

```
飞书文档 → 导出文本/markdown
 └─ GraphRAGIndexFlow（flows/graphrag_index_flow.py，文档导入总流程）
     ├─ 文档解析/切块（nodes/document_node/chunk_split.py）→ 每块文本
     ├─ 每块 embed → 向量库（indices/vector_index/*，Faiss/Milvus/Qdrant/OceanBase）
     │     chunk 记录 = 向量 + chunk原文 + 元数据
     ├─ LLM 抽取实体关系（GraphExtractFlow / nodes/graph_node/*）→ HugeGraph 图
     └─ 实体向量/社区索引
 └─ 查询：POST /rag（问答）、/rag/graph（图召回）、/graph/extract（抽取）
```

**落点**：chunk → 向量库；实体关系 → HugeGraph 图。与 NL2SQL 的 SchemaGraph 互不相干，
但共用同一套 HugeGraph + 向量库底座。

---

## 5. 写入后，查询时怎么用（link 内部，供理解）

```
/link(question)
 └─ SchemaLinker.link()（linking/schema_linker.py）
     ├─ P0 词法种子：_seed_nodes() —— term名/定义子串、表/字段名/comment 子串、token重叠
     ├─ P2 向量种子（有 embedder 时）：_ensure_vector_index() 懒构建
     │     └─ 首次调用时把每个节点 surface 文本(表名+comment) embed →
     │        NumpySchemaVectorStore.upsert()（nl2sql/vector_store.py，可换 Milvus/OceanBase）
     │     _vector_seeds()：question embed → store.search() top-k → 语义种子
     ├─ 种子合并 → LocalEngine/VermeerEngine.personalized_pagerank(seeds, alpha=0.85)
     │     （engine/local.py::personalized_pagerank → nx.pagerank，权重=边weight）
     └─ 按 PPR 得分排序 → 返回 top 表/字段（带 score）
```

**要点**：向量写入是**懒加载**（首次 link 才 embed 全节点并 upsert 向量库），不是 reload
时发生。图侧 PPR 每问一次实时算（小 schema 毫秒级；大 schema 切 Vermeer）。

---

## 6. 明天联调最小闭环

1. 平台给 `SchemaMetadata` JSON（tables/columns/terms 三样必给，其余可选）→ 我们 `reload`
2. 起服务：`PYTHONPATH=hugegraph-llm/src python scripts/nl2sql_hg_server.py`
3. 平台调 `/nl2sql/link` 验命中、`/nl2sql/schema_context` 验 prompt 可用性
4. 要语义匹配再开 `NL2SQL_EMBEDDING=1`（本地 bge 离线 embedding）
5. SQL 日志给到后：`python scripts/sql_metadata_miner.py --dir <sql目录> --out meta_extra.json`
   合并进 reload 的 foreign_keys/lineage/query_logs
