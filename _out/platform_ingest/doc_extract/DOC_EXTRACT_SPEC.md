# 飞书文档 → 指标口径抽取层（接口设计 v1）

> 定位：结构化库表字段指标数据是**主数据**（`ingest_adapter` 直接灌库）；
> 飞书文档是**口径权威补充源**——平台给的指标 JSON 可能缺 `formula`/`definition`
> （只有指标名），而口径定义写在飞书文档里。本层负责从文档抽出口径并 merge 回主数据。

## 1. 数据流

```
飞书文档（拉取/导出为文本）
  │  split_doc_blocks（空行/标题分块）
  ▼
extract_metrics_from_text（LLM 抽取，glm-5.3，可注入 generate）
  │  validate_glossary（结构校验）
  ▼
MetricsGlossary（口径 JSON，与 metric_json payload 同构）
  │  merge_glossary（权威口径回填 platform metrics）
  ▼
metric_payload → ingest_adapter.ingest_platform → KG（Metric 顶点 + computedFrom 边）
```

## 2. MetricsGlossary 结构定义（接口契约）

```jsonc
// 与 unified_convert.convert_metric_to_graph 的 metric payload 完全同构，
// 因此 glossary 可以直接 merge 进灌库 payload，无需二次转换。
{
  "metrics": [
    {
      "name": "order_total",              // 指标英文名（必填，唯一）
      "definition": "订单总额",            // 中文口径描述（可空）
      "formula": "SUM(order.amount)",     // 口径公式（可空，权威来源=文档）
      "source_tables": ["order"],         // 来源表
      "source_fields": ["order.amount"],  // 来源字段（表.字段）
      "depends_on": []                    // 依赖指标
    }
  ]
}
```

校验规则（`validate_glossary`）：`metrics` 必须是数组；每项必须含 `name`（非空、
不重复）且包含全部 6 个 key（允许空值）。

## 3. 抽取接口

| 函数 | 签名 | 说明 |
|---|---|---|
| `split_doc_blocks` | `(text: str) -> List[str]` | 按空行/标题切成自包含块 |
| `extract_metrics_from_text` | `(text, generate=None, max_blocks=20) -> List[dict]` | 逐块 LLM 抽取；`generate` 可注入（默认 glm-5.3）；单块失败跳过不中断 |
| `validate_glossary` | `(glossary) -> (ok, issues)` | 契约校验 |
| `merge_glossary` | `(metric_payload, glossary) -> (merged, stats)` | **权威口径回填**：definition/formula 以文档为准（冲突计数）；source 字段只填空 |

LLM 输出容错：支持裸 JSON 数组或 ```json fenced block；解析失败返回空（降级）。

## 4. merge 语义（口径权威性）

- `definition` / `formula`：**文档优先**（飞书是口径权威来源），与平台已有值不一致时
  以文档覆盖并计入 `conflicts`；
- `source_tables` / `source_fields` / `depends_on`：仅在平台为空时填充；
- 只 merge 平台里**已存在**的指标（glossary 独有指标默认不新增，避免文档噪声进 KG，
  可通过参数放开）。

## 5. 端到端样例

```bash
# 主数据（平台给的结构化 JSON）→ kg_platform
python _out/platform_ingest/ingest_adapter.py \
  --catalog sample_data/platform_catalog.json \
  --metrics sample_data/platform_metrics.json \
  --graph kg_platform --domain platform --reset

# 飞书文档（模拟）→ 口径抽取 → merge → 二次灌库 → 中文问题可命中
python _out/platform_ingest/run_platform_pipeline.py --graph kg_platform
```

## 术语抽取（v2 增补）

除指标口径外，文档还产出**术语实体**（同义词层自动扩展）：

- `extract_terms_from_text(text)`：抽取 `中文名（english_name）` / `中文名(english_name)` 成对写法 → TermNode（canonical=英文，aliases=[中文]）；
- `glossary_to_terms(glossary)`：从已抽取指标的定义首词（"客单价：平均每单成交金额…"）回建术语别名——与口径抽取共用同一信号；
- 产物与 `KgTermGraph.from_jargon_map({别名: 规范名})` 直接兼容，merge 进术语图后，查询理解的同义词扩展自动覆盖文档新说法（无需改 definition）。

管线建议：文档 → `extract_metrics_from_text` + `extract_terms_from_text` → 口径 merge 回灌库 payload（现有）+ 术语 merge 进术语图（新增）→ 检索的别名路即可命中文档里出现过的说法。
