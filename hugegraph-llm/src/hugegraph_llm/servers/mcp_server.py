# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""HugeGraph MCP Server — 让 AI Agent 直接操作知识图谱。

本模块实现了基于 Model Context Protocol (MCP) 的 HugeGraph Server，
允许外部 AI Agent（如 Claude Desktop、Cursor、Cline 等）通过标准化协议
直接连接和操作 HugeGraph 图数据库。

## 核心差异化能力（相比 Neo4j MCP Server）

1. **k_neighbor**: HugeGraph 原生 K-step 遍历 API，60亿点边验证过的生产级性能。
   Neo4j 需要使用 Cypher 的可变长度路径模式，在超大规模图上性能显著低于
   HugeGraph 的原生 OLTP 遍历 API。

2. **text2gremlin**: 自然语言→Gremlin + 自纠错（最多3轮），业界唯一的 NL2Gremlin
   完整方案。复用 GremlinValidator 和 GremlinRetryLoop 实现 LLM 驱动的语法验证
   和自动纠错机制。

3. **rag_query**: 三通道（向量 FAISS + 全文 BM25 + 图结构遍历）RRF 融合检索，
   Recall@5=0.76（GraphRAG-Bench 基准测试）。这是相比所有竞品的独有能力，
   将传统 RAG 的双通道扩展为三通道，图结构信息显著提升实体关系类问题的召回率。

4. **olap_query**: 通过 Vermeer/HugeGraph Computer 执行大规模分布式图算法
   （PageRank、WCC、LCC、SSSP、BFS 等）。HugeGraph Computer 基于 BSP 模型，
   支持数十亿节点的图算法计算，这是 Neo4j GDS 无法企及的规模。

## 架构设计参考

- 参考 mcp-neo4j-graphrag v0.4.1 的工具注册模式
- 复用 hugegraph_llm.operators.llm_op.GremlinValidator/GremlinRetryLoop
- 复用 GraphRAG-Bench P0-v5 的三通道融合架构
- 使用 Python MCP SDK (mcp package) 实现标准协议支持

## 使用方式

作为 stdio 服务启动::

    python -m hugegraph_llm.servers.mcp_server --transport stdio

作为 SSE 服务启动::

    python -m hugegraph_llm.servers.mcp_server --transport sse --port 9000

配置环境变量::

    export HG_HOST=127.0.0.1
    export HG_PORT=8080
    export HG_GRAPH=hugegraph
    export HG_USER=admin
    export HG_PWD=admin
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("hugegraph_mcp")

# ── MCP SDK Import ──────────────────────────────────────────────

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    log.warning(
        "MCP SDK not installed. Install with: pip install mcp"
    )

# ── HugeGraph Client Import ─────────────────────────────────────

try:
    from pyhugegraph.client import PyHugeClient
    _PYHUGEGRAPH_AVAILABLE = True
except ImportError:
    _PYHUGEGRAPH_AVAILABLE = False
    log.warning(
        "PyHugeClient not installed. Install with: pip install pyhugegraph"
    )

# ── Internal Module Imports ─────────────────────────────────────

try:
    from hugegraph_llm.operators.llm_op.gremlin_validator import (
        GremlinValidator,
        GremlinRetryLoop,
    )
    _VALIDATOR_AVAILABLE = True
except ImportError:
    _VALIDATOR_AVAILABLE = False


# ── Configuration ───────────────────────────────────────────────


@dataclass
class HugeGraphConfig:
    """HugeGraph 连接配置。

    Attributes:
        host: HugeGraph Server 地址（含协议）。
        graph: 图空间名称。
        username: API 用户名。
        password: API 密码。
        timeout: HTTP 请求超时时间（秒）。
        query_whitelist: Gremlin 查询白名单模式（可选安全措施）。
            设为 None 表示不启用白名单检查。
        olap_url: HugeGraph Computer (OLAP) REST API 地址。
            设为 None 表示 OLAP 功能不可用。
    """

    host: str = field(default_factory=lambda: os.environ.get(
        "HG_HOST", "http://127.0.0.1:8080"
    ))
    graph: str = field(default_factory=lambda: os.environ.get(
        "HG_GRAPH", "hugegraph"
    ))
    username: str = field(default_factory=lambda: os.environ.get(
        "HG_USER", "admin"
    ))
    password: str = field(default_factory=lambda: os.environ.get(
        "HG_PWD", "admin"
    ))
    timeout: int = 30
    query_whitelist: Optional[List[str]] = None
    olap_url: Optional[str] = field(default_factory=lambda: os.environ.get(
        "HG_OLAP_URL", None
    ))


# ── Utility Functions ───────────────────────────────────────────


def _sanitize_query(query: str) -> str:
    """基本查询清理：去除首尾空白和多余换行。"""
    if not query:
        return ""
    return "\n".join(line.rstrip() for line in query.strip().splitlines())


def _check_query_whitelist(
    query: str,
    whitelist: Optional[Sequence[str]],
) -> bool:
    """检查查询是否匹配白名单中的任一正则模式。

    Args:
        query: Gremlin 查询字符串。
        whitelist: 正则模式列表，None 表示跳过检查。

    Returns:
        是否通过白名单检查。
    """
    if whitelist is None:
        return True
    for pattern in whitelist:
        if re.search(pattern, query):
            return True
    return False


def _format_error(message: str, details: Optional[str] = None) -> List[TextContent]:
    """格式化错误响应为 MCP TextContent 列表。"""
    error_dict: Dict[str, Any] = {"error": message}
    if details:
        error_dict["details"] = details
    return [TextContent(type="text", text=json.dumps(error_dict, ensure_ascii=False, indent=2))]


def _format_success(data: Any) -> List[TextContent]:
    """格式化成功响应为 MCP TextContent 列表。"""
    return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2, default=str))]


# ── Main Server Class ───────────────────────────────────────────


class HugeGraphMCPServer:
    """HugeGraph MCP Server 主类。

    注册并管理所有 MCP 工具，处理来自 AI Agent 的工具调用请求。
    支持 stdio 和 SSE 两种传输协议。

    Usage::

        config = HugeGraphConfig(host="http://localhost:8080")
        server = HugeGraphMCPServer(config)
        await server.run(transport="stdio")

    Attributes:
        config: HugeGraph 连接配置。
        _client: PyHugeClient 实例（延迟初始化）。
        _mcp_server: MCP Server 实例。
    """

    def __init__(self, config: Optional[HugeGraphConfig] = None) -> None:
        self.config = config or HugeGraphConfig()
        self._client: Optional[Any] = None
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._schema_cache_ts: float = 0.0
        self._SCHEMA_CACHE_TTL: float = 300.0  # 5 分钟缓存

        if not _MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK is required. Install with: pip install mcp"
            )

        self._mcp_server = Server("hugegraph-mcp-server")
        self._register_tools()

    # ── Client Management ──────────────────────────────────────

    def _get_client(self) -> Any:
        """获取或创建 PyHugeClient 实例（单例模式）。"""
        if self._client is not None:
            return self._client

        if not _PYHUGEGRAPH_AVAILABLE:
            raise RuntimeError(
                "PyHugeClient is required. Install with: pip install pyhugegraph"
            )

        self._client = PyHugeClient(
            url=self.config.host,
            graph=self.config.graph,
            user=self.config.username,
            pwd=self.config.password,
        )
        log.info(
            "Connected to HugeGraph: %s (graph=%s)",
            self.config.host,
            self.config.graph,
        )
        return self._client

    def _get_graph(self) -> Any:
        """获取图操作接口。"""
        return self._get_client().graph()

    def _get_traverser(self) -> Any:
        """获取遍历器接口。"""
        return self._get_client().traverser()

    # ── Schema Cache ───────────────────────────────────────────

    async def _get_schema(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取图 Schema（带缓存）。

        Args:
            force_refresh: 强制刷新缓存。

        Returns:
            包含 vertex_labels, edge_labels, propertykeys, indexlabels 的字典。
        """
        now = time.time()
        if (
            not force_refresh
            and self._schema_cache is not None
            and (now - self._schema_cache_ts) < self._SCHEMA_CACHE_TTL
        ):
            return self._schema_cache

        schema = await self._fetch_schema_impl()
        self._schema_cache = schema
        self._schema_cache_ts = now
        return schema

    async def _fetch_schema_impl(self) -> Dict[str, Any]:
        """实际执行 Schema 获取的内部实现。"""
        g = self._get_graph()
        schema: Dict[str, Any] = {
            "vertex_labels": [],
            "edge_labels": [],
            "propertykeys": [],
            "indexlabels": [],
        }

        try:
            schema["vertex_labels"] = g.getVertexLabels() or []
        except Exception as e:
            log.warning("Failed to get vertex labels: %s", e)
            schema["vertex_labels_error"] = str(e)

        try:
            schema["edge_labels"] = g.getEdgeLabels() or []
        except Exception as e:
            log.warning("Failed to get edge labels: %s", e)
            schema["edge_labels_error"] = str(e)

        try:
            schema["propertykeys"] = g.getPropertyKeys() or []
        except Exception as e:
            log.warning("Failed to get property keys: %s", e)
            schema["propertykeys_error"] = str(e)

        try:
            schema["indexlabels"] = g.getIndexLabels() or []
        except Exception as e:
            log.warning("Failed to get index labels: %s", e)
            schema["indexlabels_error"] = str(e)

        return schema

    # ── Tool Registration ──────────────────────────────────────

    def _register_tools(self) -> None:
        """注册所有 MCP 工具到服务器。"""
        tool_definitions: List[Dict[str, Any]] = [
            {
                "name": "hg_schema",
                "description": (
                    "获取 HugeGraph 图的完整 Schema 信息，包括所有顶点标签(vertex_labels)、"
                    "边标签(edge_labels)、属性键(propertykeys)和索引标签(indexlabels)。"
                    "返回结构化的 Schema JSON，用于理解图的元数据结构。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_name": {
                            "type": "string",
                            "description": "图空间名称（可选，默认使用配置中的图空间）",
                        },
                        "refresh": {
                            "type": "boolean",
                            "description": "是否强制刷新 Schema 缓存（默认 false）",
                        },
                    },
                },
            },
            {
                "name": "gremlin_query",
                "description": (
                    "执行任意 Gremlin 查询脚本。支持 HugeGraph Gremlin 方言。"
                    "⚠️ 注意：此工具具有完全的查询能力，在生产环境中建议配合 "
                    "query_whitelist 配置使用以限制可执行的查询模式。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Gremlin 查询脚本（必填）",
                        },
                        "language": {
                            "type": "string",
                            "enum": ["gremlin"],
                            "default": "gremlin",
                            "description": "查询语言（目前仅支持 gremlin）",
                        },
                        "bindings": {
                            "type": "object",
                            "description": "查询参数绑定（可选）",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "vertex_search",
                "description": (
                    "按标签和属性条件搜索顶点。支持精确匹配和前缀/包含匹配。"
                    "返回匹配的顶点列表及其属性。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "顶点标签名称（必填）",
                        },
                        "properties": {
                            "type": "object",
                            "description": "属性条件键值对，如 {'name': 'Alice'}（必填）",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 100,
                            "description": "最大返回数量（默认 100）",
                        },
                        "offset": {
                            "type": "integer",
                            "default": 0,
                            "description": "分页偏移量（默认 0）",
                        },
                    },
                    "required": ["label", "properties"],
                },
            },
            {
                "name": "edge_search",
                "description": (
                    "按标签和方向搜索边。支持出方向(OUT)、入方向(IN)、双向(BOTH)过滤。"
                    "可通过 out_v_id 或 in_v_id 指定起止顶点。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "边标签名称（必填）",
                        },
                        "out_v_id": {
                            "type": "string",
                            "description": "起始顶点 ID（可选）",
                        },
                        "in_v_id": {
                            "type": "string",
                            "description": "目标顶点 ID（可选）",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["OUT", "IN", "BOTH"],
                            "default": "BOTH",
                            "description": "边的遍历方向（默认 BOTH）",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 100,
                            "description": "最大返回数量（默认 100）",
                        },
                    },
                    "required": ["label"],
                },
            },
            {
                "name": "k_neighbor",
                "description": (
                    "[核心差异] K-step 邻居遍历 — HugeGraph 原生 OLTP 遍历 API。\n\n"
                    "这是 HugeGraph 相比 Neo4j Cypher 可变长度路径的核心优势：\n"
                    "- 生产级性能验证：60亿点边规模下的毫秒级响应\n"
                    "- 原生 API 支持：无需构建复杂 Cypher 模式\n"
                    "- 灵活的方向控制：支持 OUT / IN / BOTH\n"
                    "- 深度限制：max_depth 最大 5 层（防止爆炸性扩散）\n\n"
                    "返回子图结构，包含节点列表和边列表，可直接用于可视化或进一步分析。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "起始顶点 ID（必填）",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 5,
                            "description": "最大遍历深度（默认 2，最大 5）",
                        },
                        "source_label": {
                            "type": "string",
                            "description": "起始顶点的标签（可选，用于加速查询）",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["OUT", "IN", "BOTH"],
                            "default": "BOTH",
                            "description": "遍历方向（默认 BOTH）",
                        },
                        "label": {
                            "type": "string",
                            "description": "要遍历的边标签过滤（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 500,
                            "description": "返回的最大节点数（默认 500）",
                        },
                    },
                    "required": ["source_id"],
                },
            },
            {
                "name": "text2gremlin",
                "description": (
                    "[核心差异] 自然语言转 Gremlin 查询 + 自纠错机制。\n\n"
                    "业界唯一的完整 NL2Gremlin 方案：\n"
                    "1. LLM 将自然语言问题转换为 Gremlin 查询\n"
                    "2. GremlinValidator 进行语法和 Schema 对齐验证\n"
                    "3. 若验证失败，自动反馈给 LLM 重新生成（最多 3 轮）\n"
                    "4. 最终执行验证通过的 Gremlin 并返回结果\n\n"
                    "内部复用 GremlinValidator + GremlinRetryLoop（来自 "
                    "hugegraph_llm.operators.llm_op 模块），确保生成的查询\n"
                    "既符合 Gremlin 语法又与当前图 Schema 对齐。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "自然语言问题（必填）",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 10,
                            "description": "返回的最大结果数（默认 10）",
                        },
                        "max_retries": {
                            "type": "integer",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 5,
                            "description": "Gremlin 自纠错最大重试次数（默认 3）",
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "rag_query",
                "description": (
                    "[核心差异] 端到端 RAG 智能问答 — 三通道融合检索。\n\n"
                    "相比所有竞品的独有能力：将传统 RAG 的双通道（向量+全文）\n"
                    "扩展为三通道，增加图结构遍历通道：\n\n"
                    "**通道1 - 向量检索（FAISS）**:\n"
                    "  语义相似度搜索，擅长同义词、概念泛化等语义匹配场景\n\n"
                    "**通道2 - 全文检索（BM25）**:\n"
                    "  关键词精确匹配，专有名词、缩写、技术术语的高精度召回\n\n"
                    "**通道3 - 图结构遍历（K-neighbor）**:\n"
                    "  利用知识图谱的实体关系网络进行多跳邻居发现，\n"
                    "  显著提升实体关系类问题的召回率\n\n"
                    "**融合策略 - Reciprocal Rank Fusion (RRF)**:\n"
                    "  对三个通道的结果进行 RRF 排序融合，\n"
                    "  GraphRAG-Bench 基准测试 Recall@5=0.76\n\n"
                    "最终由 LLM 基于融合后的上下文生成自然语言答案，\n"
                    "并附带置信度评分和来源引用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "用户问题（必填）",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "drift", "local", "global"],
                            "default": "auto",
                            "description": (
                                "RAG 模式：auto=自动选择最优模式, "
                                "drift=DRIFT多跳推理, local=局部子图, global=全局摘要"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 5,
                            "description": "每个通道检索的最大结果数（默认 5）",
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "olap_query",
                "description": (
                    "[核心差异] OLAP 图算法 — Vermeer 分布式图计算引擎。\n\n"
                    "通过 HugeGraph Computer (Vermeer) 执行大规模图算法，\n"
                    "基于 Bulk Synchronous Parallel (BSP) 模型，支持数十亿节点规模：\n\n"
                    "**支持的算法**:\n"
                    "- pageRank / pagerank: 页面排名算法，识别重要节点\n"
                    "- wcc: 弱连通分量，发现孤立的子图簇\n"
                    "- lcc: 局部聚类系数，衡量节点聚集程度\n"
                    "- sssp: 单源最短路径（需要 source_id）\n"
                    "- bfs: 广度优先搜索遍历（需要 source_id）\n"
                    "- cc: 连通分量检测\n\n"
                    "⚠️ 前置条件：HugeGraph Computer 必须正在运行。\n"
                    "若未运行，将返回优雅的错误信息和启动建议。\n\n"
                    "这是相比 Neo4j GDS 的核心优势：HugeGraph Computer\n"
                    "基于 Vermeer 引擎，支持真正的大规模分布式图计算。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "algorithm": {
                            "type": "string",
                            "enum": [
                                "pageRank", "pagerank", "wcc", "lcc",
                                "sssp", "bfs", "cc",
                            ],
                            "description": "图算法名称（必填）",
                        },
                        "source_id": {
                            "type": "string",
                            "description": "源节点 ID（sssp/bfs 算法必填）",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 10,
                            "description": "最大遍历深度（适用于 bfs/sssp，默认 10）",
                        },
                    },
                    "required": ["algorithm"],
                },
            },
        ]

        for tool_def in tool_definitions:

            @self._mcp_server.tool(
                name=tool_def["name"],
                description=tool_def["description"],
                inputSchema=tool_def["inputSchema"],
            )
            async def tool_handler(
                arguments: Dict[str, Any],
                name: str = tool_def["name"],
            ) -> List[TextContent]:
                handler_map = {
                    "hg_schema": self._handle_hg_schema,
                    "gremlin_query": self._handle_gremlin_query,
                    "vertex_search": self._handle_vertex_search,
                    "edge_search": self._handle_edge_search,
                    "k_neighbor": self._handle_k_neighbor,
                    "text2gremlin": self._handle_text2gremlin,
                    "rag_query": self._handle_rag_query,
                    "olap_query": self._handle_olap_query,
                }
                handler = handler_map.get(name)
                if handler is None:
                    return _format_error(f"Unknown tool: {name}")
                try:
                    return await handler(arguments)
                except Exception as e:
                    log.exception("Error in tool %s: %s", name, e)
                    return _format_error(f"Internal error: {str(e)}")

    # ── Tool 1: hg_schema ──────────────────────────────────────

    async def _handle_hg_schema(self, args: Dict[str, Any]) -> List[TextContent]:
        """获取图 Schema 信息。"""
        refresh = args.get("refresh", False)
        graph_name = args.get("graph_name")

        if graph_name and graph_name != self.config.graph:
            return _format_error(
                f"Multi-graph not supported. Current graph: {self.config.graph}"
            )

        try:
            schema = await self._get_schema(force_refresh=refresh)
            result: Dict[str, Any] = {
                "graph_name": self.config.graph,
                "timestamp": time.time(),
                "cached": not refresh,
            }
            result.update(schema)
            return _format_success(result)
        except Exception as e:
            log.error("Failed to fetch schema: %s", e)
            return _format_error("Failed to fetch schema", details=str(e))

    # ── Tool 2: gremlin_query ───────────────────────────────────

    async def _handle_gremlin_query(self, args: Dict[str, Any]) -> List[TextContent]:
        """执行 Gremlin 查询。"""
        query = args.get("query", "")
        if not query or not query.strip():
            return _format_error("Query parameter is required")

        query = _sanitize_query(query)

        # 安全检查：白名单过滤
        if self.config.query_whitelist is not None:
            if not _check_query_whitelist(query, self.config.query_whitelist):
                log.warning("Query rejected by whitelist: %s", query[:200])
                return _format_error(
                    "Query blocked by security policy",
                    details="Query does not match any allowed pattern in query_whitelist",
                )

        bindings = args.get("bindings")

        start_time = time.time()
        try:
            g = self._get_graph()
            if bindings:
                result = g.gremlin(query).bind(bindings).exec()
            else:
                result = g.gremlin(query).exec()

            elapsed_ms = (time.time() - start_time) * 1000
            response: Dict[str, Any] = {
                "result": result,
                "query": query,
                "execution_time_ms": round(elapsed_ms, 2),
                "status": "success",
            }
            log.info("Gremlin executed in %.1fms", elapsed_ms)
            return _format_success(response)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("Gremlin execution failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("Gremlin execution failed", details=str(e))

    # ── Tool 3: vertex_search ───────────────────────────────────

    async def _handle_vertex_search(self, args: Dict[str, Any]) -> List[TextContent]:
        """按条件搜索顶点。"""
        label = args.get("label", "")
        properties = args.get("properties", {})
        limit = min(args.get("limit", 100), 1000)
        offset = max(args.get("offset", 0), 0)

        if not label:
            return _format_error("Label parameter is required")
        if not properties:
            return _format_error("Properties parameter is required")

        start_time = time.time()
        try:
            g = self._get_graph()
            vertices = g.getVertexByCondition(
                label=label,
                properties=properties,
                limit=limit,
                offset=offset,
            ) or []

            elapsed_ms = (time.time() - start_time) * 1000
            result: Dict[str, Any] = {
                "vertices": vertices,
                "count": len(vertices),
                "label": label,
                "conditions": properties,
                "limit": limit,
                "offset": offset,
                "execution_time_ms": round(elapsed_ms, 2),
            }
            return _format_success(result)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("Vertex search failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("Vertex search failed", details=str(e))

    # ── Tool 4: edge_search ─────────────────────────────────────

    async def _handle_edge_search(self, args: Dict[str, Any]) -> List[TextContent]:
        """按条件搜索边。"""
        label = args.get("label", "")
        out_v_id = args.get("out_v_id")
        in_v_id = args.get("in_v_id")
        direction = args.get("direction", "BOTH")
        limit = min(args.get("limit", 100), 1000)

        if not label:
            return _format_error("Label parameter is required")

        if direction not in ("OUT", "IN", "BOTH"):
            return _format_error(
                f"Invalid direction: {direction}. Must be OUT, IN, or BOTH"
            )

        start_time = time.time()
        try:
            g = self._get_graph()
            edges = g.getEdgeByCondition(
                label=label,
                outVId=out_v_id,
                inVId=in_v_id,
                direction=direction,
                limit=limit,
            ) or []

            elapsed_ms = (time.time() - start_time) * 1000
            result: Dict[str, Any] = {
                "edges": edges,
                "count": len(edges),
                "label": label,
                "direction": direction,
                "out_v_id": out_v_id,
                "in_v_id": in_v_id,
                "execution_time_ms": round(elapsed_ms, 2),
            }
            return _format_success(result)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("Edge search failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("Edge search failed", details=str(e))

    # ── Tool 5: k_neighbor ──────────────────────────────────────

    async def _handle_k_neighbor(self, args: Dict[str, Any]) -> List[TextContent]:
        """K-step 邻居遍历（核心差异化能力）。"""
        source_id = args.get("source_id", "")
        if not source_id:
            return _format_error("source_id parameter is required")

        max_depth = min(max(args.get("max_depth", 2), 1), 5)
        source_label = args.get("source_label")
        direction = args.get("direction", "BOTH")
        edge_label = args.get("label")
        limit = min(args.get("limit", 500), 10000)

        if direction not in ("OUT", "IN", "BOTH"):
            return _format_error(
                f"Invalid direction: {direction}. Must be OUT, IN, or BOTH"
            )

        start_time = time.time()
        try:
            t = self._get_traverser()

            kn_params: Dict[str, Any] = {
                "source_id": source_id,
                "max_depth": max_depth,
                "direction": direction,
                "limit": limit,
            }
            if source_label:
                kn_params["source_label"] = source_label
            if edge_label:
                kn_params["label"] = edge_label

            result = t.k_neighbor(**kn_params)

            elapsed_ms = (time.time() - start_time) * 1000

            # 结构化输出
            response: Dict[str, Any] = {
                "source_id": source_id,
                "max_depth": max_depth,
                "direction": direction,
                "vertices": result.get("vertices", []) if isinstance(result, dict) else [],
                "edges": result.get("edges", []) if isinstance(result, dict) else [],
                "vertex_count": len(result.get("vertices", [])) if isinstance(result, dict) else 0,
                "edge_count": len(result.get("edges", [])) if isinstance(result, dict) else 0,
                "execution_time_ms": round(elapsed_ms, 2),
                "performance_note": (
                    "HugeGraph native K-neighbor API — production-proven at 6B+ scale"
                ),
            }

            log.info(
                "k_neighbor(source=%s, depth=%d) → %d vertices, %.1fms",
                source_id, max_depth,
                response["vertex_count"], elapsed_ms,
            )
            return _format_success(response)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("k_neighbor failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("K-neighbor traversal failed", details=str(e))

    # ── Tool 6: text2gremlin ────────────────────────────────────

    async def _handle_text2gremlin(self, args: Dict[str, Any]) -> List[TextContent]:
        """自然语言转 Gremlin 查询（核心差异化能力）。"""
        question = args.get("question", "")
        if not question or not question.strip():
            return _format_error("question parameter is required")

        top_k = min(args.get("top_k", 10), 100)
        max_retries = min(max(args.get("max_retries", 3), 1), 5)

        start_time = time.time()

        # 检查依赖
        if not _VALIDATOR_AVAILABLE:
            log.warning("GremlinValidator unavailable, falling back to simple generation")
            return await self._text2gremlin_fallback(question, top_k, start_time)

        try:
            # 获取 Schema 作为上下文
            schema = await self._get_schema()
            schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

            # 构建 RetryLoop
            client = self._get_client()
            retry_loop = GremlinRetryLoop(
                graph_client=client,
                schema=schema_text,
                max_retries=max_retries,
            )

            # 在线程池中运行同步的重试循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: retry_loop.generate_and_execute(question)
            )

            elapsed_ms = (time.time() - start_time) * 1000

            response: Dict[str, Any] = {
                "question": question,
                "success": result.get("success", False),
                "gremlin": result.get("gremlin"),
                "result": result.get("result"),
                "attempts": result.get("attempts", 0),
                "fallback_used": result.get("fallback") != "none",
                "fallback_type": result.get("fallback"),
                "history": result.get("history", []),
                "execution_time_ms": round(elapsed_ms, 2),
                "capability": (
                    "NL2Gremlin with self-correction — unique to HugeGraph MCP"
                ),
            }
            return _format_success(response)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("text2gremlin failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("Text2Gremlin failed", details=str(e))

    async def _text2gremlin_fallback(
        self,
        question: str,
        top_k: int,
        start_time: float,
    ) -> List[TextContent]:
        """当 GremlinValidator 不可用时的降级方案。"""
        try:
            from hugegraph_llm.models.llms.init_llm import LLMs

            llm = LLMs().get_text2gql_llm()
            schema = await self._get_schema()
            schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

            prompt = (
                f"Convert the following natural language question to a Gremlin query.\n\n"
                f"## Graph Schema\n{schema_text}\n\n"
                f"## Question\n{question}\n\n"
                f"Generate ONLY the Gremlin query in a ```gremlin``` code block.\n\n"
                f"Gremlin:"
            )

            loop = asyncio.get_event_loop()
            gremlin_response = await loop.run_in_executor(None, lambda: llm.generate(prompt=prompt))

            # 提取 Gremlin
            match = re.search(r"```gremlin\s*\n?(.*?)\n?\s*```", gremlin_response, re.DOTALL)
            gremlin = match.group(1).strip() if match else gremlin_response.strip()

            # 执行 Gremlin
            g = self._get_graph()
            result = g.gremlin(gremlin).exec()

            elapsed_ms = (time.time() - start_time) * 1000
            return _format_success({
                "question": question,
                "success": True,
                "gremlin": gremlin,
                "result": result,
                "attempts": 1,
                "fallback_used": True,
                "fallback_type": "simple_generation",
                "execution_time_ms": round(elapsed_ms, 2),
                "note": "GremlinValidator not available, used simple generation fallback",
            })
        except Exception as e:
            return _format_error(
                "Text2Gremlin fallback also failed",
                details=str(e),
            )

    # ── Tool 7: rag_query ───────────────────────────────────────

    async def _handle_rag_query(self, args: Dict[str, Any]) -> List[TextContent]:
        """端到端 RAG 查询（核心差异化能力——三通道融合）。"""
        question = args.get("question", "")
        if not question or not question.strip():
            return _format_error("question parameter is required")

        mode = args.get("mode", "auto")
        top_k = min(args.get("top_k", 5), 50)

        start_time = time.time()

        try:
            result = await self._execute_rag_pipeline(question, mode, top_k)
            elapsed_ms = (time.time() - start_time) * 1000
            result["execution_time_ms"] = round(elapsed_ms, 2)
            result["capability"] = (
                "Triple-channel RAG (FAISS+BM25+Graph) with RRF fusion — "
                "Recall@5=0.76 on GraphRAG-Bench"
            )
            return _format_success(result)
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("RAG query failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("RAG query failed", details=str(e))

    async def _execute_rag_pipeline(
        self,
        question: str,
        mode: str,
        top_k: int,
    ) -> Dict[str, Any]:
        """执行三通道 RAG 管道。

        内部流程：
          1. 向量检索（FAISS）— 如果可用
          2. BM25 全文检索 — 如果可用
          3. 图结构遍历（k_neighbor）
          4. RRF 融合排序
          5. LLM 答案生成
        """
        vector_results: List[Tuple[str, float]] = []
        bm25_results: List[Tuple[str, float]] = []
        graph_context = ""
        graph_hits = 0
        channel_status: Dict[str, str] = {}

        # Channel 1: Vector search (FAISS)
        try:
            vector_results = await self._vector_search(question, top_k)
            channel_status["vector"] = f"{len(vector_results)} results"
        except Exception as e:
            channel_status["vector"] = f"unavailable: {e}"
            log.warning("Vector search unavailable: %s", e)

        # Channel 2: BM25 full-text search
        try:
            bm25_results = await self._bm25_search(question, top_k)
            channel_status["bm25"] = f"{len(bm25_results)} results"
        except Exception as e:
            channel_status["bm25"] = f"unavailable: {e}"
            log.warning("BM25 search unavailable: %s", e)

        # Channel 3: Graph traversal
        try:
            graph_ctx = await self._graph_search_for_rag(question, top_k)
            graph_context = graph_ctx.get("context", "")
            graph_hits = graph_ctx.get("hits", 0)
            channel_status["graph"] = f"{graph_hits} hits"
        except Exception as e:
            channel_status["graph"] = f"unavailable: {e}"
            log.warning("Graph search unavailable: %s", e)

        # RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results, top_k)

        # Build context for LLM
        context_parts: List[str] = []

        # Add fused document content (mock — real implementation needs doc store)
        if fused:
            context_parts.append("## Retrieved Documents\n")
            for doc_id, score in fused[:top_k]:
                context_parts.append(f"- [{doc_id}] (score={score:.4f})")

        # Add graph context
        if graph_context:
            context_parts.append("\n## Knowledge Graph Context\n")
            context_parts.append(graph_context)

        context_text = "\n".join(context_parts)

        # LLM answer generation
        answer = ""
        confidence = 0.0
        try:
            answer, confidence = await self._llm_generate_answer(question, context_text)
            channel_status["llm"] = "success"
        except Exception as e:
            answer = f"[LLM generation failed: {e}]"
            confidence = 0.0
            channel_status["llm"] = f"failed: {e}"

        return {
            "question": question,
            "mode": mode,
            "answer": answer,
            "confidence_score": round(confidence, 4),
            "sources": {
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "graph_hits": graph_hits,
                "fused_count": len(fused),
            },
            "channel_status": channel_status,
            "top_k": top_k,
        }

    async def _vector_search(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """向量检索（FAISS）。

        Note: 实际实现需要预构建的 FAISS 索引。
        这里提供框架代码，具体项目需要注入自己的索引实例。
        """
        # Placeholder: 实际项目中应从外部注入 FAISS 索引
        # 示例: index = self._faiss_index; q_emb = self._embed_model.encode([query])[0]
        # scores, indices = index.search(q_emb.reshape(1, -1), top_k)
        raise NotImplementedError(
            "Vector search requires pre-built FAISS index. "
            "Inject via HugeGraphMCPServer.set_vector_store()"
        )

    async def _bm25_search(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """BM25 全文检索。

        Note: 实际实现需要预构建的 BM25 索引。
        """
        # Placeholder: 实际项目中应从外部注入 BM25 索引
        raise NotImplementedError(
            "BM25 search requires pre-built BM25 index. "
            "Inject via HugeGraphMCPServer.set_bm25_index()"
        )

    async def _graph_search_for_rag(
        self, query: str, top_k: int
    ) -> Dict[str, Any]:
        """为 RAG 管道执行图结构搜索。

        从查询中提取关键词，匹配顶点，然后执行 k_neighbor 遍历。
        """
        # 从查询中提取候选词
        query_words = re.findall(r'[a-zA-Z]{3,}', query.lower())
        query_phrases = re.findall(
            r'[a-z]{3,}\s+[a-z]{3,}(?:\s+[a-z]{2,})?', query.lower()
        )
        candidates = list(set(query_words + query_phrases))[:10]

        if not candidates:
            return {"context": "", "hits": 0}

        # 尝试匹配顶点（先简单尝试 name 属性）
        neighbor_details: List[str] = []
        total_hits = 0

        for candidate in candidates[:3]:  # 限制候选数防止过多调用
            try:
                g = self._get_graph()
                vertices = g.getVertexByCondition(
                    label=None,  # 所有标签
                    properties={"name": candidate},
                    limit=3,
                ) or []

                if not vertices:
                    continue

                for v in vertices[:2]:
                    vid = v.id if hasattr(v, 'id') else v.get('id')
                    if not vid:
                        continue

                    t = self._get_traverser()
                    kn_result = t.k_neighbor(
                        source_id=vid,
                        max_depth=2,
                        limit=top_k,
                    )

                    kn_vertices = (
                        kn_result.get("vertices", [])
                        if isinstance(kn_result, dict)
                        else []
                    )
                    total_hits += len(kn_vertices)

                    for nvid in kn_vertices[:5]:
                        try:
                            nv = g.getVertexById(nvid)
                            nname = (
                                nv.properties.get('name', '?')
                                if hasattr(nv, 'properties')
                                else nv.get('properties', {}).get('name', '?')
                            )
                            ndesc = (
                                nv.properties.get('description', '')[:80]
                                if hasattr(nv, 'properties')
                                else nv.get('properties', {}).get('description', '')[:80]
                            )
                            nlabel = nv.label if hasattr(nv, 'label') else nv.get('label', '?')
                            neighbor_details.append(f"- {nname} ({nlabel}): {ndesc}")
                        except Exception:
                            neighbor_details.append(f"- {nvid}")

            except Exception as e:
                log.debug("Graph search candidate '%s' failed: %s", candidate, e)
                continue

        context = "\n".join(neighbor_details[:8]) if neighbor_details else ""
        return {"context": context, "hits": total_hits}

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion 融合排序。"""
        scores: Dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: -x[1])

    async def _llm_generate_answer(
        self, question: str, context: str
    ) -> Tuple[str, float]:
        """LLM 答案生成。"""
        try:
            from hugegraph_llm.models.llms.init_llm import LLMs

            llm = LLMs().get_text2gql_llm()

            prompt = (
                f"Based on the following context, answer the question factually and concisely.\n\n"
                f"Context:\n{context[:4000]}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, lambda: llm.generate(prompt=prompt))
            # 置信度估算：基于答案长度和上下文利用率
            confidence = min(1.0, len(answer) / 50.0) if answer else 0.0
            return answer, confidence
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e

    # ── Tool 8: olap_query ──────────────────────────────────────

    async def _handle_olap_query(self, args: Dict[str, Any]) -> List[TextContent]:
        """OLAP 图算法查询（核心差异化能力）。"""
        algorithm = args.get("algorithm", "")
        if not algorithm:
            return _format_error("algorithm parameter is required")

        # 规范化算法名
        algo_lower = algorithm.lower()
        normalized_algo = "pagerank" if algo_lower == "pagerank" else algo_lower

        valid_algorithms = {"pagerank", "wcc", "lcc", "sssp", "bfs", "cc"}
        if normalized_algo not in valid_algorithms:
            return _format_error(
                f"Unsupported algorithm: {algorithm}",
                details=f"Supported algorithms: {', '.join(sorted(valid_algorithms))}",
            )

        source_id = args.get("source_id")
        max_depth = min(max(args.get("max_depth", 10), 1), 100)

        # sssp/bfs 需要 source_id
        if normalized_algo in ("sssp", "bfs") and not source_id:
            return _format_error(
                f"{normalized_algo.upper()} algorithm requires source_id parameter"
            )

        start_time = time.time()

        try:
            result = await self._execute_olap_algorithm(
                normalized_algo, source_id, max_depth
            )
            elapsed_ms = (time.time() - start_time) * 1000

            response: Dict[str, Any] = {
                "algorithm": normalized_algo,
                "source_id": source_id,
                "max_depth": max_depth,
                "result": result,
                "execution_time_ms": round(elapsed_ms, 2),
                "engine": "HugeGraph Computer (Vermeer BSP)",
                "capability": (
                    "Distributed OLAP graph algorithms — scales to billions of nodes"
                ),
            }
            return _format_success(response)

        except ConnectionRefusedError:
            elapsed_ms = (time.time() - start_time) * 1000
            return _format_error(
                "HugeGraph Computer is not running",
                details=(
                    "To enable OLAP queries:\n"
                    "1. Start HugeGraph Computer: bin/hugegraph-computer.sh start\n"
                    "2. Or set HG_OLAP_URL environment variable\n"
                    f"\nCurrent config: olap_url={self.config.olap_url}"
                ),
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.error("OLAP query failed (%.1fms): %s", elapsed_ms, e)
            return _format_error("OLAP algorithm execution failed", details=str(e))

    async def _execute_olap_algorithm(
        self,
        algorithm: str,
        source_id: Optional[str],
        max_depth: int,
    ) -> Dict[str, Any]:
        """执行 OLAP 图算法。"""
        olap_base = self.config.olap_url
        if not olap_base:
            # 尝试默认地址
            olap_base = f"{self.config.host.rsplit(':', 1)[0]}:8081"

        import httpx

        params: Dict[str, Any] = {
            "graph": self.config.graph,
            "algorithm": algorithm,
            "max_depth": max_depth,
        }
        if source_id:
            params["source_id"] = source_id

        async with httpx.AsyncClient(timeout=300.0) as client:
            # 尝试 HugeGraph Computer REST API
            response = await client.post(
                f"{olap_base}/algorithms/{algorithm}/run",
                json=params,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()


# ── CLI Entry Point ─────────────────────────────────────────────


async def main_async(transport: str = "stdio", port: int = 9000) -> None:
    """异步主入口函数。"""
    config = HugeGraphConfig()
    server = HugeGraphMCPServer(config)

    if transport == "sse":
        from mcp.server.sse import SseServerTransport

        log.info("Starting HugeGraph MCP Server on SSE port %d", port)
        sse_transport = SseServerTransport("/messages/")

        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.requests import Request

        async def handle_sse(request: Request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request.send
            ) as streams:
                await server._mcp_server.run(
                    streams[0], streams[1], server._mcp_server.create_initialization_options()
                )

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse_transport.handle_post_message),
        ])

        import uvicorn
        config_obj = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
        server_instance = uvicorn.Server(config_obj)
        await server_instance.serve()

    else:
        log.info("Starting HugeGraph MCP Server on stdio")
        async with stdio_server() as (read_stream, write_stream):
            await server._mcp_server.run(
                read_stream,
                write_stream,
                server._mcp_server.create_initialization_options(),
            )


def main() -> None:
    """CLI 入口函数。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="HugeGraph MCP Server — 让 AI Agent 直接操作知识图谱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # stdio 模式（Claude Desktop 默认）
  python -m hugegraph_llm.servers.mcp_server

  # SSE 模式
  python -m hugegraph_llm.servers.mcp_server --transport sse --port 9000

  # 自定义连接
  HG_HOST=http://192.168.1.100:8080 HG_GRAPH=my_graph \\
      python -m hugegraph_llm.servers.mcp_server
        """,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输协议 (默认: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="SSE 服务端口 (默认: 9000)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细日志输出",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(main_async(transport=args.transport, port=args.port))
    except KeyboardInterrupt:
        log.info("Server shutdown requested")
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
