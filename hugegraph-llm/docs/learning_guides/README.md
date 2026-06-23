# HugeGraph-AI PoC 知识交付物体系 — 索引

> **你的使用手册**: 每个 PoC 方向配 5 份文档，读完就能独立评估该方向值不值得做

---

## 一、已有文档（Temporal KG Agent Memory 方向）

| 顺序 | 文件 | 阅读时间 | 解决什么问题 |
|------|------|---------|-------------|
| **第 1 步** | [01_temporal_kg_overview.md](01_temporal_kg_overview.md) | 10 min | 这个方向是什么、解决什么痛点、HugeGraph 差异化在哪 |
| **第 2 步** | [02_hugegraph_competitive_analysis.md](02_hugegraph_competitive_analysis.md) | 5 min | 与 Mem0 / Graphiti / Neo4j / PowerMem 的硬指标对比 |
| **第 3 步** | [03_key_paper_summary.md](03_key_paper_summary.md) | 5 min | 4 篇核心论文（Mem0 / KDD2026 / Graphiti / LOCOMO）精华 |
| **第 4 步** | [04_technical_approach.md](04_technical_approach.md) | 10 min | 我们怎么实现的、关键算法、踩过的坑 |
| **第 5 步** | [05_results_interpretation.md](05_results_interpretation.md) | 5 min | 每个数字什么意思、与业界差距多大、哪里还烂 |

**总阅读时间**: 35 分钟  **读完效果**: 能独立判断"这个方向要不要继续投资源"

---

## 二、你的学习路径（按场景）

### 场景 A: "我只想知道这个方向靠不靠谱"（5 分钟）
```
读 01_temporal_kg_overview.md 的"核心痛点"+"与 HugeGraph 核心差异化"两节
→ 翻到 05_results_interpretation.md 的"结论"一节
→ 看总分 89/100 和未达成清单
→ 决策：继续 / 暂停 / 止损
```

### 场景 B: "我要说服别人 HugeGraph 做这个有优势"（10 分钟）
```
读 02_hugegraph_competitive_analysis.md 全篇
→ 重点抄"三大核心差异化"表格
→ 附 05_results_interpretation.md 的"优势项目"表格
```

### 场景 C: "我要看代码实现细节"（15 分钟）
```
读 04_technical_approach.md 的"关键算法实现"+"踩坑大全"
→ 对照 tests/temporal_kg_icews_v2.py 源码逐行看
→ 不懂的查 03_key_paper_summary.md 的论文引用
```

### 场景 D: "我要评估这个 PoC 质量"（10 分钟）
```
读 05_results_interpretation.md 全篇
→ 重点看"与业界基线对比"+"未达成目标"两节
→ 对照 poc-eval 评分卡（见下方"评估手册"）
```

---

## 三、评估手册：你如何独立打分

### 3.1 三步验证法（你自己就能做）

**Step 1: 红线合规检查（2 分钟）**
```bash
# 运行 poc-eval skill
python /Users/mac/.workbuddy/skills/poc-eval/scripts/eval_poc.py tests/temporal_kg_icews_v2.py
# 看总分和决策标记
```

**Step 2: 断网/停服验证（3 分钟）**
```bash
# 正常跑一遍
python tests/temporal_kg_icews_v2.py
# 停掉 HugeGraph
bash /usr/local/hugegraph-server/bin/stop-hugegraph.sh
# 再跑一遍 → 必须 FAIL（如果还能 PASS，说明有内存降级，立即打回）
python tests/temporal_kg_icews_v2.py
# 恢复 HugeGraph
bash /usr/local/hugegraph-server/bin/start-hugegraph.sh
```

**Step 3: 指标真实性抽查（5 分钟）**
```bash
# 看 result.json 里的数字
cat tests/temporal_kg_icews_v2_result.json
# 重点查: Recall@K / MRR / Latency / Accuracy
# 如果看到 "mock" / "fake" / "simulated" → 立即打回
```

### 3.2 评分卡（你自己填）

| 维度 | 满分 | 你自己给分 | 依据 |
|------|------|-----------|------|
| **真实可用性** | 20 | __ | 是否调了真实 HG REST？断网后是否 FAIL？ |
| **代码质量** | 20 | __ | 是否 <500 行？有无硬编码？ |
| **可复现性** | 20 | __ | 是否用标准数据集？指标是否量化？ |
| **业务价值** | 20 | __ | 是否覆盖 HugeGraph 核心场景？ |
| **可演进性** | 20 | __ | 能否拆成独立算子？ |
| **总分** | 100 | __ | |

**决策规则**:
- 80-100: 🟢 方法论验证通过 → 工程化排期
- 60-79: 🟡 方法论有价值 → 补充数据/指标后再评
- 40-59: 🟠 方法论存疑 → 对比已有方案后再决策
- <40: 🔴 方法论不可行 → 止损

### 3.3 必问我的 5 个问题（每次 PoC 交付时）

你直接复制这段发给我，我逐项回答：

```
[ ] 是否做了断网/停服验证？结果？
[ ] result.json 中是否有 mock / fake / simulated 字段？
[ ] 标准数据集来源？（ HuggingFace 链接 / 论文引用）
[ ] 量化指标与 Mem0 / Graphiti / 其他论文的对比？
[ ] 未达成目标有几条？预计多久修复？
```

---

## 四、待补齐方向（5 份文档缺口）

| 方向 | 10分钟指南 | 竞品对比 | 论文摘要 | 技术方案 | 结果解读 | 状态 |
|------|-----------|---------|---------|---------|---------|------|
| Agentic RAG | ⚠️ 简版 | ⚠️ 简版 | ⚠️ 简版 | ⚠️ 简版 | ⚠️ 简版 | **唯一有草稿** |
| **Temporal KG** | ✅ | ✅ | ✅ | ✅ | ✅ | **本次补齐** |
| L0→L3 Memory | ❌ | ❌ | ❌ | ❌ | ❌ | 未启动 |
| GraphRAG-Bench | ❌ | ❌ | ❌ | ❌ | ❌ | 未启动 |
| 供应链 KG 二重性 | ❌ | ❌ | ❌ | ❌ | ❌ | 未启动 |
| Code Graph | ❌ | ❌ | ❌ | ❌ | ❌ | 未启动 |

**下一步**: 你指定方向，我按同样 5 份模板补齐。

---

## 五、快捷命令

```bash
# 评估任意 PoC
python /Users/mac/.workbuddy/skills/poc-eval/scripts/eval_poc.py tests/xxx.py

# 跑全量评分卡
bash /Users/mac/.workbuddy/skills/poc-eval/scripts/weekly_scorecard.sh

# 查看最新 result.json
cat tests/xxx_result.json | python -m json.tool

# 查看今日工作日志
cat /Users/mac/Desktop/apache-code/hugegraph-dev/.workbuddy/memory/$(date +%Y-%m-%d).md
```

---

## 六、关键文件路径速查

```
hugegraph-llm/
├── docs/learning_guides/          ← 学习文档（你在这里）
│   ├── README.md                  ← 本文件（索引）
│   ├── 01_temporal_kg_overview.md
│   ├── 02_hugegraph_competitive_analysis.md
│   ├── 03_key_paper_summary.md
│   ├── 04_technical_approach.md
│   └── 05_results_interpretation.md
├── tests/
│   ├── temporal_kg_icews_v2.py    ← PoC 源码
│   ├── temporal_kg_icews_v2_result.json
│   └── benchmark_data/
│       └── icews14_agent_memory_benchmark.json
└── .workbuddy/memory/
    └── 2026-06-12.md              ← 今日工作日志
```
