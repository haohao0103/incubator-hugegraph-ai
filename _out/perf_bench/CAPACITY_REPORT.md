# NL2SQL 语义层容量报告（2026-08-30）

> 数据来源：`_out/perf_bench/logs/bench_kg_scale.log`（KG 链路组件基准）
> 与 `_out/perf_bench/logs/bench_semantic_scale.log`（语义层 PPR 基准）。
> 环境：hg-e2e venv (py3.10) / LocalEngine（进程内 networkx）/ 合成数据。
> 压测脚本：`_out/perf_bench/bench_kg_scale.py`、`_out/perf_bench/bench_semantic_scale.py`。

## 1. KG 链路组件（KgSchemaLinker / KgSqlValidator / KgSqlVoter）

| scale | vertices | link(ms) | validator build(ms) | vote-shared(ms) | vote-standalone(ms) |
|---|---|---|---|---|---|
| 500 表 | 3,505 | 9.1 | 4.3 | 10.1 | 14.2 |
| 1,000 表 | 7,010 | 18.0 | 10.8 | 18.2 | 29.9 |
| 2,000 表 | 14,020 | 36.5 | 22.3 | 38.0 | 60.1 |
| 5,000 表 | 35,050 | 94.6 | 66.7 | 99.7 | 189.9 |

**读法**：link 每 1000 表 ≈ +18ms，线性；5000 表 <100ms。validator 复用
（vote-shared）比每投票重建（vote-standalone）省 ~40-50% 时间——生产必须
走共享 validator。

## 2. 语义层 PPR（SchemaLinker + LocalEngine，词法 P0）

| tables | columns | link(ms) |
|---|---|---|
| 500 | 4,500 | 34.8 |
| 1,000 | 9,000 | 67.1 |
| 2,000 | 18,000 | 170.9 |
| 5,000 | 45,000 | 458.9 |

**读法**：PPR 全图传播代价明显高于 KG linking（5000 表 459ms vs 95ms），
近线性但系数大。

## 3. 容量结论与引擎切换建议

| 规模 | 延迟 | 建议引擎 |
|---|---|---|
| ≤1,000 表 | <70ms | LocalEngine（进程内，零运维） |
| 1,000–2,000 表 | 70–170ms | LocalEngine，若 QPS 高上 Vermeer |
| 2,000–5,000 表 | 170–460ms | **VermeerEngine（集群）**——延迟敏感场景必须 |
| >5,000 表 | >460ms | VermeerEngine（PPR/最短路/社区全上集群） |

- 当前生产图 `kg_rag` 12 表（真实负载 ~5ms 级，healthz 显示 engine=vermeer，
  已在集群上跑）。
- 实测 API 层缓存（LRU 60s）把重复问题延迟从 ~4.3s（含首构建）压到 **5ms**；
  高频问题建议配合结果缓存。
- 容量红线：单机 LocalEngine 建议上限 **≈2000 表**（<200ms 预算）；
  超限自动切 Vermeer（`_make_engine` 已实现 healthcheck 自动选择）。
