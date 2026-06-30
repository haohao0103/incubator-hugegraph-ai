# Gradio 交互式页面能力覆盖度分析

> 分析时间：2026-06-30
> 当前分支：`feature/graphrag-baseline`
> 分析范围：hugegraph-llm Gradio UI 8 个 Tab vs. 项目全部能力

## 1. 结论：尚未完全展示

当前 Gradio 页面 **未完全展示** HugeGraph-AI 的全部能力。粗略统计：

- 当前分支可用能力模块：约 **60+**
- Gradio UI 已覆盖：约 **35-40**
- 缺失/未充分暴露：约 **20+**
- 其他分支（Agent Memory、Code Graph、Skills Graph、供应链）尚未合并到当前分支，无法通过 UI 展示

## 2. 当前 Gradio UI 9 个 Tab 能力清单

| Tab | 名称 | 已展示能力 |
|-----|------|------------|
| 1 | Build RAG Index | 文本/文件上传、向量索引构建、图数据抽取、图数据导入、Schema 生成、Graph Extract Prompt 生成、Vid Embedding 更新、索引信息/清理 |
| 2 | RAG & User Functions | 单问题 RAG 回答（4 种模式）、批量问答评测、ReRank、Graph Ratio |
| 3 | Text2gremlin | 模板索引构建、自然语言转 Gremlin、Gremlin 执行 |
| 4 | Agent & Global Search | ReAct Agent、Global Search、社区检测、Graph RAG Search（4 种模式） |
| 5 | Graph Tools | Gremlin 执行、图数据备份、初始化测试数据 |
| 6 | Admin Tools | 日志查看、日志清理 |
| 7 | Advanced GraphRAG | DRIFT 多跳、Schema 校验、实体消解、社区报告、RRF 融合、Token Budget |
| 8 | GraphRAG Enhancement | PPR、Cascade 传播、Identity Edge、Dual Keyword、Community Summary、HyDE、Gleaning、Provenance、BM25 |
| 9 | Capability Map | 能力矩阵、覆盖率概览、缺失工具快速演示（Fetch Graph Summary、Get Schema、Validate Gremlin、Incremental Index） |

## 3. 项目能力全景（按类别）

### 3.1 索引与检索（Index & Retrieval）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 向量索引构建 | `index_op/build_vector_index.py` | Tab 1 | 已展示 |
| 语义索引（Vertex ID） | `index_op/build_semantic_index.py` | Tab 1 | 已展示（Update Vid Embedding） |
| Gremlin 示例索引 | `index_op/build_gremlin_example_index.py` | Tab 3 | 已展示 |
| 社区索引构建 | `index_op/build_community_index.py` | Tab 4（社区检测） | 部分展示 |
| BM25 索引查询 | `index_op/bm25_index_query.py` | Tab 8 | 已展示 |
| 语义 ID 查询 | `index_op/semantic_id_query.py` | Tab 4 Graph RAG Search | 部分展示 |
| 向量索引查询 | `index_op/vector_index_query.py` | Tab 2 | 已展示 |
| 增量索引更新 | `graph_op/incremental_utils.py`, `flows/incremental_index_flow.py` | Tab 7（仅状态） | **未充分展示** |

### 3.2 图操作（Graph Operators）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 社区检测 | `graph_op/community_detect.py` | Tab 4/7/8 | 已展示 |
| 实体消解 | `graph_op/entity_resolution.py` | Tab 7 | 已展示 |
| PPR 检索 | `graph_op/ppr_retriever.py` | Tab 8 | 已展示 |
| Cascade 传播 | `graph_op/cascade_propagation.py` | Tab 8 | 已展示 |
| Identity Edge 构建 | `graph_op/identity_edge_builder.py` | Tab 8 | 已展示 |
| RRF 融合 | `graph_op/rrf_fusion.py` | Tab 7 | 已展示 |
| Token Budget | `graph_op/token_budget.py` | Tab 7 | 已展示 |
| Schema 校验 | `graph_op/schema_validator.py` | Tab 7 | 已展示 |
| 同义词管理 | `graph_op/synonym_manager.py` | 无 | **缺失** |
| Chunk 相似边 | `graph_op/chunk_sim_edges.py` | 无 | **缺失** |
| 增量图更新 | `graph_op/incremental_utils.py` | 无 | **缺失** |

### 3.3 LLM 操作（LLM Operators）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 信息抽取（三元组） | `llm_op/info_extract.py` | Tab 1 | 已展示 |
| 属性图抽取 | `llm_op/property_graph_extract.py` | 无 | **缺失** |
| 关键词抽取 | `llm_op/keyword_extract.py` | Tab 2 | 已展示 |
| 双层关键词抽取 | `llm_op/dual_keyword_extract.py` | Tab 8 | 已展示 |
| 答案合成 | `llm_op/answer_synthesize.py` | Tab 2 | 已展示 |
| Gremlin 生成 | `llm_op/gremlin_generate.py` | Tab 3 | 已展示 |
| Gremlin 校验 | `llm_op/gremlin_validator.py` | 无 | **缺失** |
| HyDE 生成 | `llm_op/hyde_generate.py` | Tab 8 | 已展示 |
| DRIFT 搜索 | `llm_op/drift_search.py` | Tab 7 | 已展示 |
| Gleaning 追问 | `llm_op/gleaning_extractor.py` | Tab 8 | 已展示 |
| Provenance 回答 | `llm_op/provenance_answer.py` | Tab 8 | 已展示 |
| 社区报告 | `llm_op/community_report.py` | Tab 4/7/8 | 已展示 |
| 全局搜索 | `llm_op/global_search.py` | Tab 4 | 已展示 |
| Schema 构建 | `llm_op/schema_build.py` | Tab 1 | 已展示 |
| Prompt 生成 | `llm_op/prompt_generate.py` | Tab 1 | 已展示 |
| 共指消解 | `llm_op/coref_resolution.py` | 无 | **缺失** |
| Claim 抽取 | `llm_op/claim_extract.py` | 无 | **缺失** |
| 歧义消解 | `llm_op/disambiguate_data.py` | 无 | **缺失** |
| 非结构化数据 utils | `llm_op/unstructured_data_utils.py` | 无 | **缺失** |

### 3.4 HugeGraph 操作（HugeGraph Operators）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 提交到 HugeGraph | `hugegraph_op/commit_to_hugegraph.py` | Tab 1（导入） | 已展示 |
| 获取图数据 | `hugegraph_op/fetch_graph_data.py` | 无 | **缺失** |
| Schema 管理 | `hugegraph_op/schema_manager.py` | 无 | **缺失** |
| 溯源管理 | `hugegraph_op/provenance_manager.py` | Tab 8 | 已展示 |

### 3.5 文档操作（Document Operators）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 文本分块 | `document_op/chunk_split.py` | Tab 1（隐式） | 部分展示 |
| 关键词提取 | `document_op/word_extract.py` | 无 | **缺失** |
| TextRank 关键词 | `document_op/textrank_word_extract.py` | 无 | **缺失** |

### 3.6 多模态（Multimodal）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 多模态 KG 构建 | `multimodal/multimodal_kg_builder.py` | 无 | **缺失** |
| 多模态检索 | `multimodal/multimodal_retriever.py` | 无 | **缺失** |
| PDF/图片提取 | `multimodal/pdf_image_extractor.py` | 无 | **缺失** |
| VLM 描述 | `multimodal/vlm_descriptor.py` | 无 | **缺失** |

### 3.7 Agent 与工具（Agents & Tools）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| ReAct Agent Loop | `agents/agent_loop.py` | Tab 4 | 已展示 |
| 工具注册 | `agents/tool_registry.py` | Tab 4 | 已展示 |
| MCP 适配器 | `agents/mcp_adapter.py` | 无 | **缺失** |
| Agent 记忆 | `agents/memory/`（空） | 无 | **缺失** |
| 查询分类器 | `nodes/query_classifier_node.py` | 无 | **缺失** |
| 工具执行节点 | `nodes/tool_execution_node.py` | 无 | **缺失** |

### 3.8 RAG 流程与节点（Flows & Nodes）

| 能力 | 代码位置 | UI 状态 | 备注 |
|------|---------|---------|------|
| 端到端 RAG 流程 | `rag_op/e2e_rag_pipeline.py` | 无 | **缺失** |
| 实体消解流程 | `flows/entity_resolution_flow.py` | Tab 7 | 已展示 |
| 溯源流程 | `flows/provenance_flow.py` | Tab 8 | 已展示 |
| 增量索引流程 | `flows/incremental_index_flow.py` | 无 | **缺失** |

## 4. 其他分支未合并能力（无法在 feature/graphrag-baseline UI 中展示）

| 能力 | 分支 | 主要产物 | 对 Gradio 的影响 |
|------|------|----------|-----------------|
| Agent 记忆（MAGMA / MemGraphRAG） | `feature/agent-memory-collection` | 12 个文件，16k+ 行，含可视化 Demo | **缺失** |
| Code Graph + MCP | `poc/0614-codegraph-hugegraph-mcp` | tree-sitter 多语言解析、MCP 工具、语义代码搜索 | **缺失** |
| Skills Graph / Code-Review-Graph / LLM Wiki | `poc/0618-skills-graph-code-review-wiki` | 22 skills + 34 边、Code Review 图谱 | **缺失** |
| 供应链 Agent 路由 | `poc/0615-supply-chain-agent-router` | 意图分类 + 四通道检索 + RRF 融合 | **缺失** |
| Agentic RAG E2E | `poc/0612-agentic-rag-e2e` | 91.3% accuracy | **缺失** |
| GraphRAG-Bench 适配 | `poc/0618-graphrag-bench-adaptation` | 评测框架 | **缺失** |

## 5. 缺口优先级矩阵

| 优先级 | 缺口 | 原因 | 实现难度 |
|--------|------|------|----------|
| P0 | 多模态 RAG | 当前分支代码已存在，完全未展示 | 中 |
| P0 | 属性图抽取 | 当前分支代码已存在，与 Tab 1 互补 | 低 |
| P0 | 增量索引/图更新 | 当前分支代码已存在，生产必需 | 中 |
| P0 | Agent 记忆 | 工作记忆明确，有完整分支和 Demo | 高（需合并分支） |
| P0 | Code Graph + MCP | 工作记忆明确，有完整分支和 Demo | 高（需合并分支） |
| P1 | Query Classifier / MCP 适配器 | Agent 能力增强 | 中 |
| P1 | Gremlin 校验 / Claim 抽取 / 共指消解 | LLM 操作增强 | 中 |
| P1 | 同义词管理 / Chunk 相似边 | 图质量增强 | 中 |
| P1 | 供应链 Agent 路由 | 有 PoC 分支 | 高（需合并分支） |
| P2 | Skills Graph / Code-Review-Graph | 有 PoC 分支 | 高（需合并分支） |
| P2 | 文档关键词提取 | 文档处理增强 | 低 |
| P2 | 图数据获取 / Schema 管理 | 图谱管理增强 | 中 |

## 6. 关键结论

1. **当前 8 个 Tab 已覆盖 GraphRAG 核心能力**：索引构建、检索、问答、Text2Gremlin、Agent、Global Search、Advanced GraphRAG、GraphRAG Enhancement 等。
2. **已新增 Tab 9 Capability Map**：用于可视化展示能力矩阵和部分缺失工具的快捷演示（Fetch Graph Summary、Get Schema、Validate Gremlin、Incremental Index）。
3. **GraphRAG 内部仍有未暴露能力**：多模态、属性图抽取、增量索引完整流程、Gremlin 校验独立入口、Claim/共指消解、同义词管理等。
4. **最大缺口来自其他分支**：Agent Memory、Code Graph、Skills Graph、供应链 Agent 是工作记忆中明确的高投入 PoC，但尚未合并到 `feature/graphrag-baseline`，因此无法在 Gradio 中展示。
5. **建议**：若要让 Gradio 真正“完全展示”能力，需要：
   - 在当前分支内补齐 P0/P1 缺失的 UI 入口；
   - 将 `feature/agent-memory-collection`、`poc/0614-codegraph-hugegraph-mcp`、`poc/0618-skills-graph-code-review-wiki`、`poc/0615-supply-chain-agent-router` 等分支评估合并或 cherry-pick 到 `feature/graphrag-baseline`；
   - 在 Gradio 中新增对应 Tab（如 Tab 10 Agent Memory、Tab 11 Code Graph、Tab 12 Skills Graph、Tab 13 Supply Chain）。

## 7. 下一步建议

1. **短期（当前分支）**：
   - ✅ 已完成：新增 Capability Map Tab，展示能力矩阵并提供缺失工具快捷入口。
   - 待补齐：多模态 RAG、属性图抽取、增量索引完整流程、Query Classifier 等 UI 入口。
2. **中期（分支合并）**：合并 Agent Memory 和 Code Graph 分支，新增对应 Tab。
3. **长期**：建立 Gradio 能力矩阵自动化检查机制，确保新能力开发时同步添加 UI 入口。
