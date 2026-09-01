# 场景一 PoC：HugeGraph 语义中间层支撑 Text2SQL

> 定位：验证 HugeGraph 知识图谱作为「语义层 / 知识底座」支撑上层平台 Text2SQL 的核心能力闭环。
> 这是生产链路（平台侧 NL2SQL）的技术验证，不是独立玩具。

- 执行时间：2026-08-31 19:39 ~ 20:01（Run 3 基准 + L3 沿图传播升级 + 真实 MiMo 复验，key 取自 `hugegraph-llm/.env`）
- 真实链路：HugeGraph 1.7（`kg_rag`，`127.0.0.1:8081`）+ semantica hg-backend `GraphStore` + 小米 MiMo `mimo-v2.5-pro`（OpenAI 兼容，真实调用）
- 脚本：`/Users/mac/Desktop/apache-code/hg-rag-hmsgraphrag/incubator-hugegraph-ai/hugegraph-llm/nl2sql_tools/semantica_nl2sql_poc.py`
- 结果 JSON：`/Users/mac/Desktop/apache-code/hg-rag-hmsgraphrag/incubator-hugegraph-ai/_out/semantica_poc/semantica_poc.json`
- 运行日志：`/Users/mac/Desktop/apache-code/hg-rag-hmsgraphrag/incubator-hugegraph-ai/_out/semantica_poc/logs/semantica_poc.log`

---

## 1. 三层架构

| 层 | 职责 | 实现 |
|----|------|------|
| **L1 混合检索** | 在业务术语本体上做词法召回 + 图遍历扩展 | `lexical_match`（双向中文/英文词法，3/2/1 加权）+ `expand_context`（term→field→table→join 两跳闭包→口径） |
| **L2 提示组装** | 召回上下文 → prompt → LLM 生成 SQL | `assemble_prompt`（术语/表/字段/JOIN/口径/历史纠错六段式）+ `call_mimo`（temperature=0） |
| **L3 纠错决策记录** | 用户修正 SQL 时落带 provenance 的纠错节点 | `record_correction` → `CorrectionDecision` 节点 + `pocAppliesTo*` 边挂回术语/口径/字段；召回时**沿语义边图传播**（`pocSynonym`/`pocComputedFrom*`/`pocTermField`/`pocMetricField`/`pocHasColumn`/`pocHasCaliber`，2 跳），对全部可达节点召回纠错并按纠错 id 去重 |

本体规模（演示）：4 表 / 13 字段 / 6 术语 / 2 指标 / 2 口径，含 join 路径、同义词、指标公式链（`term_aov = gmv / paid_orders`）。

**与 Vanna.ai 的核心差异化（L3）**：Vanna 只存 `(question, sql)` 二元组，不存「为什么错、挂到哪个口径、哪个字段」；本 PoC 的纠错决策带 provenance（挂 `term_gmv` / `cal_gmv_paid` / `fld_users_city`），纠错知识可沿图传播到**表述不同但语义相同**的新问题，而不是只对同字面问题生效。

---

## 2. 演示结果

### Q1 纠错闭环（真实生效 ✅）

| 阶段 | SQL 关键点 | 说明 |
|------|-----------|------|
| PhaseA 首轮 | `DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m-01')` 过滤上月 | 召回已带 GMV 口径 + paid 过滤 + users 表，但时间表达为 MySQL 方言、无 `< 月初` 上界 |
| PhaseB 纠错 | `date_trunc('month', now()) - interval '1 month'` 动态上月，补 `o.created_at < date_trunc('month', now())` | 落 `corr_a39537b1`，provenance 挂 `['term_gmv', 'cal_gmv_paid', 'fld_users_city']` |
| PhaseC 重问 | 召回历史纠错 **2 条**：① 种子直接命中（挂 `term_gmv`）② **经传播命中**（挂 `cal_gmv_paid`，不在词法种子内，靠 `metric_gmv --pocHasCaliber-->` 传播可达）；生成 SQL 与正确 SQL 结构一致（动态上月 + paid 口径 + JOIN users） | **纠错闭环 + 沿图传播真实生效** |

### L3 沿图传播验证（✅，2026-08-31 升级）

传播统计（Q1 提问）：`seed=5` → 沿语义边 2 跳 `propagated=4`（`cal_gmv_paid`/`fld_orders_amount`/`fld_orders_status`/`tbl_orders`）→ `reached=9`。演示纠错挂非种子节点 `cal_gmv_paid`，仅靠传播即被召回（`PhaseC: 传播额外召回 ...: True`）。多端点重复（一条纠错挂 3 个端点）已按纠错 id 去重。

### Q2 join 两跳闭包（✅）

「客单价最高的品类」→ 检索从 `orders` 沿 `pocJoinPath` 扩两跳得到 `order_items → products`，上下文全量列出 13 个字段；生成 SQL 用 CTE 两跳 join：

```sql
WITH order_category AS (
    SELECT DISTINCT oi.order_id, p.category
    FROM order_items oi JOIN products p ON oi.product_id = p.id
)
SELECT oc.category, SUM(o.amount) / COUNT(DISTINCT o.id) AS aov
FROM order_category oc JOIN orders o ON oc.order_id = o.id
WHERE o.status = 'paid' GROUP BY oc.category ORDER BY aov DESC
```

无两跳闭包时 LLM 只能瞎编 `products.category` / `order_items.*` 这类不在上下文中的列。

---

## 3. HG 规范沉淀（本次踩坑，写进后端规范）

| # | 规范 | 踩坑现象 |
|---|------|---------|
| 1 | **逻辑 id 禁止含冒号**，一律下划线（`tbl_orders`） | semantica `create_relationship` 把 `<id>:<suffix>` 误判为 HG 复合 id（`<numeric>:<label>`），边标签 body 的 source/target label 被解析成错误值，HG 返回 400 |
| 2 | **边标签严格按 (源标签,目标标签) 对一一命名**（`pocHasColumn`/`pocTermField`/`pocJoinPath`/`pocAppliesToTerm`…） | 复用已有边标签且端点对不一致时，store 走「删旧标签→重建」路径：异步删除有竞态，且**连带删掉该标签下其它边**（本次误删 e2e 的 hasColumn 73 条边，已用 `restore_kg_rag.py` 恢复） |
| 3 | **标签冲突必须加前缀隔离**（`PoCTable`/`PoCField`/`PoCMetric`） | kg_rag 已有 e2e 建的 `Metric` 标签（PRIMARY_KEY id 策略），同名 CUSTOMIZE_STRING 写入报 `Can't customize vertex id when id strategy is 'PRIMARY_KEY'` |
| 4 | **Gremlin 对未定义 label 做 hasLabel/inE 会抛 Undefined（vertex/edge）label，整条查询作废** | reset 必须先 GET `/schema/vertexlabels` 再 drop；fetch_corrections 必须先查现有边标签再 inE（首轮无纠错时 `pocAppliesTo*` 尚不存在） |
| 5 | **中文检索必须双向匹配**：节点 token-in-question（强，3 分）+ question-token-in-node（弱，2 分） | 中文无天然分隔符，把整个问题当一个 token 子串匹配必失败（Run 1 空召回，MiMo 只能返回"无法生成"） |
| 6 | **join 做两跳闭包 + 上下文内所有表字段全量列出** | 单跳无法回答跨表问题（AOV 需 products.category）；字段不全 LLM 会瞎编列名 |
| 7 | **Gremlin 通道只跑 Gremlin**；semantica 高位 Cypher API（DecisionRecorder/DecisionQuery/ContextRetriever）在本后端不可用 | 直接调用必崩，PoC 只用 `create_node`/`create_relationship`/`execute_query(gremlin)` |

---

## 4. 运行方式

```bash
cd /Users/mac/Desktop/apache-code/hg-rag-hmsgraphrag/incubator-hugegraph-ai
PYTHONPATH=/Users/mac/Desktop/apache-code/hugegraph-dev/semantica-hg-backend \
OPENAI_CHAT_API_BASE=https://api.xiaomimimo.com/v1 \
OPENAI_CHAT_API_KEY=<MiMo key> \
OPENAI_CHAT_LANGUAGE_MODEL=mimo-v2.5-pro \
/Users/mac/.venvs/semantica/bin/python3.12 \
    hugegraph-llm/nl2sql_tools/semantica_nl2sql_poc.py
```

- 前置：HugeGraph 1.7 本地实例运行中（REST `:8081`，图 `kg_rag` 存在）。
- 未设 `OPENAI_CHAT_API_KEY` 时 L2 返回占位 SQL，L1/L3 演示不中断。
- 日志追加写 `/Users/mac/Desktop/apache-code/hg-rag-hmsgraphrag/incubator-hugegraph-ai/_out/semantica_poc/logs/semantica_poc.log`，结果覆盖写 `semantica_poc.json`。
- MiMo key 通过环境变量注入，不落盘。

---

## 5. 与目标定位的对应

| 平台侧 Text2SQL 所需能力 | PoC 验证点 |
|--------------------------|-----------|
| Schema Linking（表字段检索） | L1 词法召回 + 图扩展，Q1 命中 `users.city` / `orders.amount` |
| JOIN 路径发现 | `pocJoinPath` 两跳闭包，Q2 自动带出 `order_items→products` |
| 业务术语映射 | `BusinessTerm` 节点 + `pocTermField` 边（GMV→amount、支付订单→status filter） |
| 指标口径统一 | `Caliber` 节点 + `pocHasCaliber`，prompt 强制口径约束（paid 才计入 GMV） |
| 纠错沉淀（知识积累） | L3 `CorrectionDecision` + provenance，跨表述复用 |

## 6. 下一步候选

1. ✅ 已做：L3 纠错沿图传播（同义词/指标链/口径可达也召回）——2026-08-31 升级验证；
2. ✅ 已做：HG 规范（第 3 节）回写进 semantica hg-backend 代码注释/文档；
3. 接入真实数仓 schema（替换演示本体），跑平台侧真实问题集；
4. 纠错召回的 prompt 注入做相关性排序（当前按传播顺序平铺，多条时可能稀释重点）。
