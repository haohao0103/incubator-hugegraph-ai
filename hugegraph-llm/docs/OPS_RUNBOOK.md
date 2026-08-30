# NL2SQL 语义层 —— 运维手册（OPS Runbook）

> 版本：v1（2026-08-30）｜配套：docs/NL2SQL_API_CONTRACT.md（接口）、docs/NL2SQL_INGEST_FLOW.md（写入）、docs/PRODUCTION_READINESS.md（就绪度）

---

## 1. 部署

```bash
# 依赖：Python 3.10 venv + hugegraph-llm（源码安装）+ 本地 bge 模型缓存
cd incubator-hugegraph-ai
PYTHONPATH=hugegraph-llm/src python -m uvicorn \
    hugegraph_llm.demo.rag_demo.app:app --host 0.0.0.0 --port 8910
```

生产建议：容器化 + 环境分离（dev/staging/prod 各一套 HugeGraph 图 + 配置），
`Dockerfile` / helm chart 见部署计划（P1 未完成项）。

## 2. 环境变量（集中配置）

| 变量 | 默认 | 说明 |
|---|---|---|
| `NL2SQL_HG_GRAPH` | 未设 | 设了则从该 HugeGraph 图构建 schema（生产必设，如 kg_rag） |
| `NL2SQL_HG_URL` | http://127.0.0.1:8081 | HugeGraph REST 地址 |
| `NL2SQL_EMBEDDING` | 未设 | =1 启用 P2 向量语义（需 embedding 后端，本地 bge 离线） |
| `NL2SQL_KEYWORD_LLM` | 未设 | =1 启用 LLM 关键词提取 |
| `NL2SQL_LLM_TIMEOUT` | 60 | run 端点 LLM 调用超时（秒） |
| `NL2SQL_HG_RETRIES` | 3 | 图读取失败重试次数 |
| `NL2SQL_DEFAULT_TOP_K` | 10 | link/schema_context 默认 top_k |
| `NL2SQL_MIN_SCORE` | 未设 | 全局拒答阈值（低于即 out_of_kb） |
| `VERMEER_MASTER` | http://127.0.0.1:6688 | Vermeer 集群地址（可达则引擎自动切换） |

## 3. 日志

- 审计日志（request-id 串联）：`[nl2sql][<req_id>] <METHOD> <path> <ms> -> <status>`
- 平台调用方传 `X-Request-Id` 头可回传关联；服务端生成则响应头带回。
- 常规日志走 hugegraph-llm 的 log 模块（控制台 + 文件，见部署日志配置）。

## 4. 降级矩阵（healthz 会报告）

| 依赖挂了 | 行为 | healthz 状态 |
|---|---|---|
| Vermeer 集群 | 引擎自动回退 LocalEngine | degraded: vermeer_unreachable_using_local |
| embedding 后端 | P2 自动关闭，纯词法 | degraded: embedder_failed_lexical_only |
| HugeGraph 图源 | 启动构建失败 → 503 NL2SQL_DEPENDENCY_UNAVAILABLE | degraded: hugegraph_unreachable |
| LLM（仅 run） | 503 NL2SQL_LLM_UNAVAILABLE / 超时 504 | 不适用（查询端点不受影响） |
| 全部正常 | — | ok |

**查询端点（link/join_path/communities/schema_context）不依赖 LLM**，LLM 挂了不影响检索。

## 5. 恢复

1. **图数据异常**：重灌（`ingest_metadata_to_hg.py --clear`，kg_rag 专用图安全）；或增量补灌（PRIMARY_KEY 幂等）。
2. **pipeline 状态异常**：重启服务即可（内存 pipeline 从图重建）。
3. **embedding 不可用**：确认模型缓存存在（HF_HUB_OFFLINE=1）；临时降级词法（接口自动）。

## 6. 监控

- `GET /nl2sql/metrics`：prometheus 文本（endpoint 维度 calls/errors/latency/out_of_kb）
- 建议接入：QPS、p50/p99 延迟、out_of_kb 率（拒答率异常升高=知识库缺口信号）、降级计数。

## 7. SLO 建议（待真实评测后定）

| 指标 | 目标（建议） | 备注 |
|---|---|---|
| link 命中率（recall@3） | ≥ 0.85 | 真实评测集定基线 |
| p95 延迟 | < 500ms | 检索链路（不含 LLM）；run 含 LLM 另计 |
| out_of_kb 率 | 可观测即可 | 升高=知识库需补 |
| 可用性 | 99.5% | 依赖 HugeGraph/向量库的 SLA |

## 8. 多环境

- dev/staging/prod 各一套 `NL2SQL_HG_GRAPH` 图（图名带后缀隔离），配置独立 env。
- 数据同步：dev 用样例，staging 用脱敏子集，prod 全量。
- 权限：prod 加 token 鉴权 + 表级权限（等数仓 DBA 权限清单，P0 未完成项）。

## 9. 增量同步（schema 变更）

```bash
# 全量首次 / 重建：
python scripts/ingest_metadata_to_hg.py --meta <meta.json> --clear

# 增量（只写图中没有的表/列/指标，PRIMARY_KEY 幂等，可反复执行）：
python scripts/ingest_metadata_to_hg.py --meta <meta.json> --diff
```

调度示例（cron，每 30 分钟）：
```
*/30 * * * * cd <repo> && PYTHONPATH=hugegraph-llm/src \
  python nl2sql_tools/ingest_metadata_to_hg.py --meta <meta.json> --diff \
  >> _out/hg_ingest/logs/sync.log 2>&1
```
增量语义：只新增，不删除（删除走 --clear 全量重建）。

## 10. 敏感字段与租户权限

- 敏感检测：列名/注释含 phone/手机/身份证/password/token/银行卡 等 → schema_context 输出标 `[SENSITIVE]`；元数据列可显式 `sensitive: true/false` 覆盖启发式。
- 租户权限（可选）：配置文件路径设 `NL2SQL_PERMISSIONS=<rules.json>`，格式：
  ```json
  {"tenant_a": {"dw.users": ["user_id", "city"]}, "tenant_b": ["dw.users"]}
  ```
  请求带 `tenant` 字段即按权限过滤列；未配置规则 = 全放行（向后兼容）。
- 执行结果脱敏：`permissions.mask_value()` 提供通用打码，接 SqlExecutor 结果时按敏感列调用。
