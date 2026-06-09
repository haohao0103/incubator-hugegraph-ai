# HugeGraph Memory 生产级开发路线图

> **版本**: v1.0-draft
> **日期**: 2026-06-08
> **目标**: 将现有 PoC 级 Memory 代码（78 tests）升级为对标 PowerMem v1.1.2 的生产级 AI 记忆系统
> **工期预估**: 14 个 Sprint（约 14 周），3 条并行路线

---

## 一、现状诊断：我们有什么 vs 差什么

### 1.1 现有代码资产（可复用）

| 模块 | 文件 | 测试 | 生产级评估 |
|------|------|------|-----------|
| MemoryGraph ABC | `agents/memory/base.py` | 78 | 接口过简（仅4方法），需扩展为 MemoryGraph 完整接口 |
| HugeGraphMemory | `agents/memory/hugegraph_memory.py` | 30 | **Demo级**: 硬编码Gremlin、字符串拼接注入风险、无事务、无连接池 |
| GraphRAGMemoryAgent | `agents/memory/graphrag_memory_agent.py` | 48 | **Demo级**: 未实际调用Sprint1-10 operators，chat()仅做了简单LLM生成 |
| KnowledgeFreshnessTracker | `operators/graph_op/knowledge_freshness.py` | 20 | **中等**: TTL+陈旧度评分可用，但非Ebbinghaus衰减加权 |
| GraphQualityAssessor | `operators/graph_op/graph_quality_assessor.py` | 19 | **可用**: 5维度评估，可集成到记忆质量管线 |
| ContextAwareQA | `operators/llm_op/context_aware_qa.py` | 32 | **可用**: 上下文管理+查询重写+证据溯源 |
| MultiGranularityRetriever | `operators/index_op/multi_granularity_retrieve.py` | 23 | **可用**: 双层检索，可集成到记忆检索管线 |
| DRIFT Search | `operators/llm_op/drift_search.py` | 30 | **可用**: 5步搜索算法 |
| EntityResolution | `operators/graph_op/entity_resolution.py` | 24 | **可用**: 三策略消解，PowerMem无此能力 |
| Text2Gremlin | `operators/llm_op/text2gremlin*.py` | 41 | **可用**: 自纠错，PowerMem无此能力 |
| MCP Server | `mcp/hugegraph_mcp_server.py` | - | **可用**: 10 tools + 3 resources，但**缺少memory相关tools** |

### 1.2 完全缺失模块（需从零建设）

| 模块 | PowerMem 对标 | 优先级 |
|------|--------------|--------|
| 异步接口 AsyncMemory | `AsyncMemory` 类 (1850行) | P0 |
| HTTP API Server | FastAPI, 15+ 端点, Swagger, API Key | P0 |
| MCP Memory Tools | add/search/update/delete memory tools | P0 |
| 记忆提取管线 | 事实提取→相似搜索→冲突决策→ADD/UPDATE/DELETE | P0 |
| Ebbinghaus 遗忘曲线 | R(t)=e^(-0.821t), 访问强化+0.3, 4生命周期 | P1 |
| 混合检索融合 | 4路(向量+全文+稀疏+图)→RRF→Reranker | P1 |
| CLI 工具 | pmem 命令行 (CRUD/配置/备份/迁移/REPL) | P1 |
| Dashboard | 实时统计/分析/健康监控 | P1 |
| Provider 工厂系统 | LLM/Embedding/Rerank 多Provider注册 | P1 |
| Benchmark 评测 | LOCOMO/AppWorld 基准测试 | P2 |
| 多SDK语言 | Go/Java/TypeScript SDK | P3 |

### 1.3 硬伤清单（必须立即修复）

1. **Gremlin 字符串拼接注入**: `_check_existing_entity()` 等方法直接f-string拼接用户输入
2. **MemoryConfig 未挂载全局配置**: 独立内联类，参数硬编码
3. **BM25 排序无持久化索引**: 每次search临时import并构建BM25
4. **无错误重试/事务/连接池**: 直接调用client.gremlin()，失败即抛异常
5. **LLM prompt 硬编码**: 无模板管理，无custom prompt支持
6. **无认证/鉴权**: 完全裸接口
7. **无幂等性保证**: add()重复调用会产生重复数据
8. **无Snowflake ID**: 无全局唯一ID生成器

---

## 二、目标架构

### 2.1 分层架构（对标 PowerMem 5层）

```
┌─────────────────────────────────────────────────────────────┐
│ 集成层 (Integration)                                        │
│  Python SDK (sync)  │  Async SDK  │  HTTP Server  │  MCP Server  │  CLI  │
├─────────────────────────────────────────────────────────────┤
│ 应用层 (Application)                                         │
│  Memory (中央编排)  │  UserMemory (画像)  │  IntelligenceManager │
├─────────────────────────────────────────────────────────────┤
│ 编排层 (Orchestration)                                       │
│  MemoryPipeline (提取→去重→决策→执行)  │  MemorySearchPipeline  │  EbbinghausPlugin  │
├─────────────────────────────────────────────────────────────┤
│ 存储层 (Storage)                                            │
│  MemoryGraph (HugeGraph实现)  │  VectorStore (外部/内置)  │  StoreAdapter  │
├─────────────────────────────────────────────────────────────┤
│ 基础设施层 (Infrastructure)                                  │
│  LLMProvider  │  EmbeddingProvider  │  Config  │  ID生成  │  日志/审计  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 HugeGraph 差异化定位

```
PowerMem 图搜索: 浅层多跳 (BFS/DFS) → 作为混合检索的一路
HugeGraph 图搜索:
  ├── OLAP Traverser: 60亿点边大规模遍历 (PowerMem/Mem0 均无)
  ├── DRIFT 搜索: HyDE→社区→Primer→Local→Reduce 5步 (Sprint 4)
  ├── 多粒度检索: 社区级+实体级+chunk级双层 (Sprint 6)
  ├── 实体消解: exact/embedding/llm_verify 三策略 (Sprint 1)
  ├── 图谱质量: 5维度评估 (Sprint 7)
  ├── 知识时效: TTL/版本/陈旧度 (Sprint 8)
  ├── 证据溯源: answer_support + 引用格式化 (Sprint 9)
  └── Text2Gremlin: NL→Gremlin自纠错 (Sprint 5)
```

### 2.3 核心接口定义

#### MemoryGraph 完整接口（对标 PowerMem MemoryGraph）

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class MemoryGraph(ABC):
    """Graph storage backend for AI Memory system.

    Must be implemented by any graph database backend.
    Reference: PowerMem MemoryGraph interface (6 core methods).
    """

    @abstractmethod
    def add_node(
        self,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Add a memory node to the graph."""

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a directed edge between two memory nodes."""

    @abstractmethod
    def traverse(
        self,
        start_id: str,
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-hop traversal from a starting node."""

    @abstractmethod
    def get_neighbors(
        self,
        memory_id: str,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get directly connected neighbors of a memory node."""

    @abstractmethod
    def search_graph(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search the graph for memories matching a query."""

    @abstractmethod
    def delete_node(self, memory_id: str) -> bool:
        """Delete a memory node and all its edges."""
```

#### Memory 中央编排器接口（对标 PowerMem Memory）

```python
class Memory:
    """Central orchestrator for AI memory system.

    Coordinates: storage backends, embedding, LLM, intelligence plugins.
    Reference: PowerMem Memory class (13-step init, dual-mode add).
    """

    # === 核心 CRUD ===
    def add(self, messages, infer: bool = True, user_id: str = None,
            agent_id: str = None, run_id: str = None,
            metadata: Dict = None, filters: Dict = None) -> Dict[str, Any]: ...
    def search(self, query, limit: int = 100, user_id: str = None,
               agent_id: str = None, run_id: str = None,
               filters: Dict = None, threshold: float = None) -> Dict[str, Any]: ...
    def get(self, memory_id, user_id: str = None, agent_id: str = None) -> Optional[Dict]: ...
    def update(self, memory_id, content, user_id: str = None,
               agent_id: str = None, metadata: Dict = None) -> Dict[str, Any]: ...
    def delete(self, memory_id, user_id: str = None, agent_id: str = None) -> bool: ...
    def delete_all(self, user_id: str = None, agent_id: str = None, run_id: str = None) -> bool: ...
    def get_all(self, user_id: str = None, agent_id: str = None,
                run_id: str = None, sort_by: str = None) -> Dict[str, List[Dict]]: ...

    # === 高级操作 ===
    def optimize(self, strategy: str = "semantic", threshold: float = 0.95) -> Dict: ...
    def stats(self, user_id: str = None) -> Dict: ...
    def export_memories(self, format: str = "json", user_id: str = None) -> str: ...
    def import_memories(self, data: str, format: str = "json", infer: bool = True) -> Dict: ...
```

#### Ebbinghaus 算法参数（对标 PowerMem EbbinghausPlugin）

```python
EBBINGHAUS_CONFIG = {
    "base_retention_1h": 0.44,      # 1小时后保留率 44%
    "decay_constant": 0.821,          # lambda = -ln(0.44)
    "min_retention": 0.2,             # 最低保留率 20%
    "initial_retention": 1.0,         # 新记忆初始值
    "reinforcement_factor": 0.3,       # 访问强化 +0.3
    "working_threshold": 0.3,         # <0.3 = WORKING
    "short_term_threshold": 0.6,       # 0.3-0.6 = SHORT_TERM
    "long_term_threshold": 0.8,         # 0.6-0.8 = SHORT_TERM, >=0.8 = LONG_TERM
}
# R(t) = exp(-0.821 * hours_elapsed), floor = 0.2
# R_new = min(R_current + 0.3, 1.0) when accessed
```

---

## 三、Sprint 分解（14 周）

### Phase 0: 基础设施（1-2 周） — 为生产级代码打地基

| Sprint | 内容 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **S11: 基础设施重构** | 1. Snowflake ID 生成器<br>2. MemoryConfig 挂载 HugeGraphConfig<br>3. Provider 工厂 (LLM/Embedding)<br>4. Gremlin 参数化查询 (防注入)<br>5. 连接池 + 重试机制<br>6. Prompt 模板管理 | `infra/id_generator.py`<br>`infra/provider_factory.py`<br>`infra/gremlin_safe.py`<br>`infra/prompt_templates.py`<br>`config/` 新增配置 | 1. 所有 Gremlin 查询使用参数化<br>2. LLM/Embedding 可通过配置切换<br>3. 旧测试全部通过 + 新增 20 tests |

### Phase 1: 存储层重写（2-3 周） — 将 Demo 级 Memory 升级为生产级

| Sprint | 内容 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **S12: MemoryGraph 接口+实现** | 1. 重写 `base.py` 扩展为 6 方法接口<br>2. 重写 `hugegraph_memory.py`: 参数化Gremlin、事务、连接池<br>3. Snowflake ID 集成<br>4. 向量索引集成（HugeGraph vector 或外部 Milvus）<br>5. 全文索引集成 | `agents/memory/base.py` (v2)<br>`agents/memory/hugegraph_memory.py` (v2)<br>`agents/memory/vector_store.py` | 1. 所有 Gremlin 参数化<br>2. add_node 返回 Snowflake ID<br>3. traverse 支持 max_hops+filters<br>4. 50 tests 通过 |
| **S13: 混合检索引擎** | 1. HybridSearchEngine: 4路并行检索<br>2. RRF (Reciprocal Rank Fusion) 融合<br>3. Ebbinghaus 加权排序<br>4. 可选 Reranker 接口<br>5. 复用 MultiGranularityRetriever + DRIFT | `operators/memory_op/hybrid_search.py`<br>`operators/memory_op/rrf_fusion.py`<br>`operators/memory_op/ebbinghaus_plugin.py` | 1. 4路检索可独立开关<br>2. RRF 融合 k=60<br>3. Ebbinghaus 衰减公式验证<br>4. 40 tests |

### Phase 2: 编排层（3-4 周） — 核心 Memory 系统上线

| Sprint | 内容 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **S14: 记忆提取管线** | 1. FactExtractor: LLM 事实提取 (对标 FACT_RETRIEVAL_PROMPT)<br>2. ConflictResolver: 相似搜索→冲突分析<br>3. ActionDecider: ADD/UPDATE/DELETE/NONE 决策<br>4. 去重: content hash + 语义相似度<br>5. Prompt 模板 (支持 custom_prompt) | `operators/memory_op/fact_extractor.py`<br>`operators/memory_op/conflict_resolver.py`<br>`operators/memory_op/action_decider.py`<br>`operators/memory_op/deduplicator.py`<br>`prompts/memory_prompts.py` | 1. infer=True 时走完整管线<br>2. 冲突检测准确率 > 85%<br>3. 决策覆盖 ADD/UPDATE/DELETE/NONE<br>4. 40 tests |
| **S15: Memory 中央编排器** | 1. Memory 类 (sync): 双模式 add (simple/intelligent)<br>2. search/get/update/delete/get_all/stats<br>3. IntelligenceManager 集成 (Ebbinghaus+Quality)<br>4. 审计日志 AuditLogger<br>5. 多租户隔离 (user_id/agent_id/run_id) | `agents/memory/memory.py` (v3)<br>`agents/memory/intelligence_manager.py`<br>`agents/memory/audit_logger.py` | 1. Memory.add(infer=True) 走完整管线<br>2. Memory.add(infer=False) 走简单模式<br>3. 多租户完全隔离<br>4. 50 tests |
| **S16: AsyncMemory + 事务** | 1. AsyncMemory 类 (全量异步)<br>2. 事务支持: add 时 graph+vector 原子写入<br>3. 批量操作: batch_add/batch_delete<br>4. 并发安全: 分布式锁 (HugeGraph不支持原生锁时的方案) | `agents/memory/async_memory.py`<br>`agents/memory/transaction.py` | 1. AsyncMemory 与 Memory 接口一致<br>2. 事务回滚测试<br>3. 并发 add 无数据丢失<br>4. 30 tests |

### Phase 3: 集成层（3-4 周） — API 生态建设

| Sprint | 内容 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **S17: HTTP API Server** | 1. FastAPI server: 15+ 端点<br>2. API Key 认证<br>3. CORS + Rate Limiting (slowapi)<br>4. Swagger/ReDoc 文档<br>5. Prometheus 指标<br>6. Docker 部署 | `server/main.py`<br>`server/api/v1/memories.py`<br>`server/api/v1/system.py`<br>`server/middleware/auth.py`<br>`server/middleware/rate_limit.py`<br>`Dockerfile` | 1. POST/GET/PUT/DELETE /memories 全通<br>2. API Key 认证生效<br>3. /docs Swagger 可访问<br>4. Docker 构建成功<br>5. 30 tests |
| **S18: MCP Memory Server** | 1. MCP server (SSE/stdio/StreamableHTTP)<br>2. 7 tools: add/search/get/update/delete/list/delete_all<br>3. Claude Desktop / Cursor / Copilot 集成<br>4. 复用现有 MCP server 框架 | `mcp/memory_mcp_server.py`<br>`mcp/memory_tools.py` | 1. SSE 模式启动正常<br>2. stdio 模式启动正常<br>3. Claude Desktop 可调用<br>4. 20 tests |
| **S19: CLI + Dashboard** | 1. CLI (hgmem): add/search/list/stats/backup/restore/config/shell<br>2. Web Dashboard: 记忆统计/分布/质量/健康<br>3. 备份/恢复/迁移工具<br>4. pip install 支持 | `cli/main.py`<br>`cli/commands/`<br>`server/dashboard/`<br>`setup.py` 更新 | 1. hgmem memory add "..." 正常<br>2. Dashboard 可视化显示<br>3. 备份→恢复流程验证<br>4. 20 tests |

### Phase 4: GraphRAG 融合 + 差异化（2 周） — 发挥 HugeGraph 独特优势

| Sprint | 内容 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **S20: GraphRAG Memory 融合** | 1. 重写 GraphRAGMemoryAgent: 实际调用 Sprint1-10 operators<br>2. Memory.search() 融合 DRIFT 搜索<br>3. 实体消解 (Sprint 1) 集成到记忆去重<br>4. 图谱质量 (Sprint 7) 集成到 optimize()<br>5. 知识时效 (Sprint 8) + Ebbinghaus 双引擎<br>6. 证据溯源 (Sprint 9) 集成到 search 响应 | `agents/memory/graphrag_memory_agent.py` (v2)<br>`agents/memory/graphrag_fusion.py` | 1. chat() 实际调用 DRIFT/MultiGranularity<br>2. 搜索结果含 evidence traces<br>3. OLAP traverser 可选路径<br>4. 40 tests |
| **S21: OLAP Traverser 集成** | 1. OLAP traverser 作为图遍历后端<br>2. 大规模多跳 (>3 hops) 查询支持<br>3. OLAP 结果与 OLTP 结果融合<br>4. 自动路由: 小规模用OLTP Gremlin，大规模用OLAP | `agents/memory/olap_traverser.py`<br>`agents/memory/traversal_router.py` | 1. >1000节点遍历走OLAP<br>2. <100节点遍历走OLTP<br>3. 结果格式统一<br>4. 20 tests |

### Phase 5: 质量保障 + 发布（1-2 周）

| Sprint | 内容 | 交付物 | 验收标准 |
|--------|------|--------|---------|
| **S22: 测试+Benchmark+文档** | 1. 集成测试套件 (HugeGraph Server 启动)<br>2. LOCOMO 基准测试脚本<br>3. 压力测试 (并发/延迟/吞吐)<br>4. 完整 API 文档 + 用户指南<br>5. CHANGELOG + 发布说明 | `tests/integration/`<br>`benchmarks/locomo/`<br>`benchmarks/stress/`<br>`docs/MEMORY_API.md`<br>`docs/MEMORY_DEPLOYMENT.md` | 1. 集成测试全部通过<br>2. LOCOMO 对比数据<br>3. 压测报告 (p50/p95/p99)<br>4. 文档可独立使用 |

---

## 四、目录结构规划

```
hugegraph-llm/src/hugegraph_llm/
├── agents/memory/                          # 存储层 + 编排层
│   ├── __init__.py
│   ├── base.py                             # MemoryGraph ABC (6 methods)
│   ├── memory.py                           # Memory 中央编排器 (sync)
│   ├── async_memory.py                     # AsyncMemory (async)
│   ├── hugegraph_memory.py                 # HugeGraph MemoryGraph 实现
│   ├── vector_store.py                     # 向量存储适配 (HugeGraph vector / Milvus)
│   ├── fulltext_store.py                   # 全文存储适配
│   ├── graphrag_memory_agent.py            # GraphRAG + Memory 融合 (v2)
│   ├── graphrag_fusion.py                  # GraphRAG 融合逻辑
│   ├── olap_traverser.py                   # OLAP 大规模遍历
│   ├── traversal_router.py                 # OLTP/OLAP 自动路由
│   ├── intelligence_manager.py             # Intelligence 编排
│   ├── audit_logger.py                     # 审计日志
│   └── transaction.py                      # 事务管理
│
├── operators/memory_op/                     # 记忆操作算子
│   ├── __init__.py
│   ├── fact_extractor.py                   # LLM 事实提取
│   ├── conflict_resolver.py                # 冲突分析与解决
│   ├── action_decider.py                   # ADD/UPDATE/DELETE 决策
│   ├── deduplicator.py                     # 去重 (hash + semantic)
│   ├── hybrid_search.py                    # 4路混合检索引擎
│   ├── rrf_fusion.py                       # RRF 融合排序
│   ├── ebbinghaus_plugin.py                # Ebbinghaus 遗忘曲线
│   └── memory_quality.py                   # 记忆质量评估
│
├── infra/                                  # 基础设施
│   ├── __init__.py
│   ├── id_generator.py                     # Snowflake ID
│   ├── provider_factory.py                 # LLM/Embedding/Rerank 工厂
│   ├── gremlin_safe.py                     # 参数化 Gremlin 查询
│   ├── prompt_templates.py                 # Prompt 模板管理
│   └── retry.py                            # 重试机制
│
├── server/                                 # HTTP API Server
│   ├── __init__.py
│   ├── main.py                             # FastAPI app
│   ├── models/                             # Pydantic models
│   │   ├── request.py
│   │   └── response.py
│   ├── api/v1/
│   │   ├── memories.py                     # /api/v1/memories 端点
│   │   └── system.py                       # /api/v1/system 端点
│   ├── middleware/
│   │   ├── auth.py                         # API Key 认证
│   │   └── rate_limit.py                   # 限流
│   └── dashboard/                          # Web Dashboard (静态)
│
├── mcp/
│   ├── hugegraph_mcp_server.py             # 现有: 图查询 MCP (保留)
│   ├── memory_mcp_server.py                # 新增: 记忆 MCP
│   └── memory_tools.py                     # MCP tools 定义
│
├── cli/                                    # 命令行工具
│   ├── __init__.py
│   ├── main.py                             # hgmem 入口
│   └── commands/
│       ├── memory.py                       # add/search/list/delete
│       ├── config.py                       # config init/test
│       └── manage.py                       # backup/restore/migrate
│
├── prompts/
│   ├── memory_prompts.py                   # 记忆提取/决策 prompt
│   └── graphrag_prompts.py                 # GraphRAG prompt (已有)
│
└── config/
    └── hugegraph_config.py                  # 新增 memory 相关配置
```

---

## 五、配置体系扩展

在 `HugeGraphConfig` 中新增的配置项：

```python
# === Memory 系统配置 ===

# 图存储
memory_graph_enabled: bool = True
memory_graph_backend: str = "hugegraph"  # hugegraph / neo4j / neptune

# 向量存储
memory_vector_enabled: bool = True
memory_vector_backend: str = "hugegraph"  # hugegraph / milvus / pgvector

# 全文存储
memory_fulltext_enabled: bool = True

# Ebbinghaus 遗忘曲线
memory_ebbinghaus_enabled: bool = True
memory_ebbinghaus_initial_retention: float = 1.0
memory_ebbinghaus_decay_rate: float = 0.1
memory_ebbinghaus_reinforcement_factor: float = 0.3

# 记忆提取管线
memory_infer_enabled: bool = True
memory_custom_fact_prompt: Optional[str] = None
memory_custom_update_prompt: Optional[str] = None
memory_dedup_threshold: float = 0.95
memory_fallback_to_simple_add: bool = True

# HTTP Server
memory_server_host: str = "0.0.0.0"
memory_server_port: int = 8001  # 避免与图MCP的8848冲突
memory_server_auth_enabled: bool = False
memory_server_api_keys: str = ""
memory_server_rate_limit_per_min: int = 100

# MCP Server
memory_mcp_enabled: bool = True
memory_mcp_port: int = 8849
memory_mcp_transport: str = "sse"  # sse / stdio / streamable-http

# OLAP Traverser
memory_olap_enabled: bool = False
memory_olap_threshold: int = 1000  # 节点数阈值，超过走OLAP
```

---

## 六、里程碑与验收标准

| 里程碑 | 时间 | 验收标准 | 对标 PowerMem |
|--------|------|---------|--------------|
| **M1: 基础设施就绪** | Week 2 | Gremlin 参数化、Provider 工厂、ID 生成、配置体系 | - |
| **M2: 存储层生产级** | Week 5 | MemoryGraph 6方法完整实现、混合检索4路、Ebbinghaus | GraphStoreBase + 混合检索 |
| **M3: Memory API 可用** | Week 9 | sync + async Memory 类、完整提取管线、CLI 基础命令 | Memory + AsyncMemory |
| **M4: 集成层完成** | Week 12 | HTTP Server + MCP Server + CLI + Dashboard | HTTP + MCP + CLI + Dashboard |
| **M5: 差异化上线** | Week 14 | GraphRAG融合、OLAP遍历、LOCOMO评测 | 无对标（独有能力） |

---

## 七、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| HugeGraph Server 不支持原生全文索引 | 混合检索缺一路 | 引入外部 Elasticsearch / 使用 HugeGraph 的属性索引 |
| OLAP Traverser API 变更 | 集成失败 | 抽象 TraverserAdapter 接口，隔离依赖 |
| LLM 提取质量不稳定 | 记忆质量差 | Ebbinghaus 自动淘汰低质量记忆 + 人工审核接口 |
| 性能瓶颈 (每次add/search都调LLM) | 延迟过高 | simple模式不走LLM / 异步管线 / 缓存策略 |
| PowerMem 快速迭代到GA | 市场窗口关闭 | 聚焦差异化 (OLAP + GraphRAG深度) 而非全面追平 |

---

## 八、成功指标

### 功能指标
- [ ] Memory API 完整覆盖 PowerMem Memory 类的全部方法
- [ ] HTTP Server 15+ 端点全部可用
- [ ] MCP Server 7 tools 在 Claude Desktop 可调用
- [ ] CLI 支持完整记忆生命周期管理

### 质量指标
- [ ] 单元测试 >= 400 cases (当前 78 + 新增 322+)
- [ ] 集成测试 >= 50 cases
- [ ] 代码覆盖率 >= 80%
- [ ] Gremlin 查询 100% 参数化 (0 注入风险)

### 性能指标
- [ ] add() (infer=False) p95 < 100ms
- [ ] add() (infer=True) p95 < 3s
- [ ] search() p95 < 500ms
- [ ] HTTP Server 吞吐 >= 100 req/s (单 worker)

### 差异化指标
- [ ] OLAP Traverser 支持 > 1亿节点遍历
- [ ] GraphRAG 融合: DRIFT 搜索准确率 > 基线 20%
- [ ] LOCOMO benchmark: 准确率 >= PowerMem 的 80%

---

## 九、执行纪律

### 每日工作流
1. 读取本文档确认当前 Sprint 任务
2. 按 Sprint 内容逐项实现
3. 每个模块完成后立即编写测试
4. 全部测试通过后 git commit
5. 每个 Sprint 完成后运行全量测试确认无回归

### 代码规范
- 所有 Gremlin 查询必须使用 `gremlin_safe.py` 参数化接口
- 所有 LLM 调用必须使用 `prompt_templates.py` 模板管理
- 所有配置必须通过 `HugeGraphConfig` 统一管理
- 所有新文件必须包含 Apache License 头
- 所有公开方法必须包含 docstring + type hints
- 每个新增 operator 必须有 >= 80% 测试覆盖

### 提交规范
```
[Memory] S12: 重写 HugeGraphMemory 参数化Gremlin查询
[Memory] S14: 实现事实提取管线 FactExtractor
[Memory] S17: 新增 HTTP API Server /api/v1/memories 端点
```
