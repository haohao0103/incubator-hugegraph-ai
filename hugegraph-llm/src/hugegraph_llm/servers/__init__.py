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

"""HugeGraph MCP Server 模块。

提供基于 Model Context Protocol 的 HugeGraph Server 实现，
允许外部 AI Agent 通过标准化协议连接和操作图数据库。

主要组件:
    - HugeGraphMCPServer: 主服务器类
    - HugeGraphConfig: 连接配置数据类

使用方式:
    from hugegraph_llm.servers.mcp_server import HugeGraphMCPServer, HugeGraphConfig

    config = HugeGraphConfig(host="http://localhost:8080")
    server = HugeGraphMCPServer(config)
    await server.run(transport="stdio")
"""

from hugegraph_llm.servers.mcp_server import (
    HugeGraphConfig,
    HugeGraphMCPServer,
)

__all__ = [
    "HugeGraphConfig",
    "HugeGraphMCPServer",
]
