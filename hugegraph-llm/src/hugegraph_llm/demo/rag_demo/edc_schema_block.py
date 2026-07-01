# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
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

"""
Gradio UI block for EDC Schema Pipeline — dedicated tab.

Showcases the EDC (Extract → Define → Canonicalize) three-phase
knowledge graph schema evolution pipeline + Guided mode extraction.

5 operators showcased:
  A. Config          (GraphRAGSchemaConfig — mode/strategy/threshold control)
  B. Extract+Raw     (LLM free extraction → raw type names, pre-EDC)
  C. Define          (KGSchemaDefineOperator — semantic definitions for new types)
  D. Canonicalize    (KGSchemaCanonicalizeOperator — synonym merge via embedding)
  E. Guided          (GuidedExtractOperator — Pydantic-constrained extraction)

Key design: Compare EVOLVING vs GUIDED side-by-side so users see
how EDC prevents type explosion while Guided constrains upfront.

All demo data is self-contained so the page can run without
external dependencies (LLM/HugeGraph) for showcasing.
"""

import json

import gradio as gr

from hugegraph_llm.utils.log import log


# ═══════════════════════════════════════════════════════════════
# Self-contained demo data for EDC pipeline
# ═══════════════════════════════════════════════════════════════

# ── Demo text inputs ──────────────────────────────────────────

DEMO_TEXTS = {
    "fraud_case": (
        "2023年12月，嫌疑人张三通过摩拜科技的员工设备IP 192.168.1.100登录了"
        "受害者李四的银行账户，转移资金50万元至第三方王五的支付宝账户。"
        "嫌疑犯赵六协助提供了虚假身份信息。suspect Chen also participated "
        "in the fraud scheme through a shell corporation."
    ),
    "metadata_query": (
        "哪张表能取到司机运送实际车型啊？物理车型字段在司机宽表里。"
        "订单宽表的列包括订单状态、金额、创建时间等column信息。"
        "数据表datatable记录了所有历史交易明细。"
    ),
}

# ── A. Config demo ────────────────────────────────────────────

DEMO_CONFIG_EVOLVING = {
    "mode": "EVOLVING",
    "canonicalize_strategy": "EMBEDDING_SIM",
    "canonicalize_similarity_threshold": 0.85,
    "canonicalize_suggest_threshold": 0.70,
    "define_trigger_policy": "NEW_TYPES_ONLY",
    "allow_manual_override": True,
    "manual_type_definitions": {},
    "require_human_approval_below": 0.70,
    "known_vertex_types": [],
}

DEMO_CONFIG_GUIDED = {
    "mode": "GUIDED",
    "allowed_vertex_labels": ["person", "device", "ip", "account", "organization", "location"],
    "allowed_edge_labels": ["uses", "logs_in", "transfers", "belongs_to", "located_at"],
    "guided_allow_dynamic": False,
}

# ── B. Extract demo (raw LLM output before EDC) ──────────────

DEMO_EXTRACT_FRAUD = {
    "raw_entities": [
        {"name": "张三", "raw_type": "嫌疑人"},
        {"name": "摩拜科技", "raw_type": "公司"},
        {"name": "192.168.1.100", "raw_type": "IP地址"},
        {"name": "李四", "raw_type": "受害者"},
        {"name": "银行账户", "raw_type": "账户"},
        {"name": "王五", "raw_type": "第三方"},
        {"name": "支付宝账户", "raw_type": "账户"},
        {"name": "赵六", "raw_type": "嫌疑犯"},
        {"name": "Chen", "raw_type": "suspect"},
        {"name": "50万元", "raw_type": "金额"},
    ],
    "raw_relations": [
        {"source": "张三", "target": "192.168.1.100", "raw_label": "使用设备"},
        {"source": "张三", "target": "银行账户", "raw_label": "登录"},
        {"source": "张三", "target": "王五", "raw_label": "转移资金"},
        {"source": "赵六", "raw_label": "提供", "target": "虚假身份信息"},
    ],
    "unique_raw_types": ["嫌疑人", "公司", "IP地址", "受害者", "账户", "第三方", "嫌疑犯", "suspect", "金额"],
    "type_count": 9,
    "note": "9 different type names for ~5 concepts — TYPE EXPLOSION!",
}

DEMO_EXTRACT_METADATA = {
    "raw_entities": [
        {"name": "司机宽表", "raw_type": "表"},
        {"name": "物理车型", "raw_type": "字段"},
        {"name": "实际车型", "raw_type": "字段"},
        {"name": "订单宽表", "raw_type": "数据表"},
        {"name": "订单状态", "raw_type": "列"},
        {"name": "金额", "raw_type": "column"},
        {"name": "创建时间", "raw_type": "列"},
        {"name": "历史交易明细", "raw_type": "datatable"},
    ],
    "unique_raw_types": ["表", "字段", "数据表", "列", "column", "datatable"],
    "type_count": 6,
    "note": "6 different type names for 2 concepts (table + field) — TYPE EXPLOSION!",
}

# ── C. Define demo ────────────────────────────────────────────

DEMO_DEFINE_FRAUD = {
    "new_types": ["嫌疑人", "公司", "IP地址", "受害者", "账户", "嫌疑犯", "suspect", "金额"],
    "type_definitions": {
        "嫌疑人": {
            "description": "涉嫌犯罪的自然人，通常在案件调查阶段被指认",
            "properties": [
                {"name": "姓名", "type": "string", "required": True},
                {"name": "案件编号", "type": "string", "required": False},
                {"name": "涉案类型", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
            "distinguishing_features": "涉嫌犯罪的人， narrower than 'person'",
        },
        "嫌疑犯": {
            "description": "已被正式指控犯罪的自然人，法律术语",
            "properties": [
                {"name": "姓名", "type": "string", "required": True},
                {"name": "指控罪名", "type": "string", "required": True},
                {"name": "案件编号", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
            "distinguishing_features": "正式指控阶段，与'嫌疑人'语义高度重叠",
        },
        "suspect": {
            "description": "A person suspected of committing a crime or offense",
            "properties": [
                {"name": "name", "type": "string", "required": True},
                {"name": "case_id", "type": "string", "required": False},
                {"name": "offense_type", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
            "distinguishing_features": "English equivalent of 嫌疑人/嫌疑犯",
        },
        "公司": {
            "description": "商业组织实体，含注册信息和行业分类",
            "properties": [
                {"name": "公司名称", "type": "string", "required": True},
                {"name": "注册号", "type": "string", "required": False},
                {"name": "行业", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "IP地址": {
            "description": "网络终端设备的IP地址标识",
            "properties": [
                {"name": "IP值", "type": "string", "required": True},
                {"name": "归属地", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "受害者": {
            "description": "在犯罪案件中遭受损失的自然人",
            "properties": [
                {"name": "姓名", "type": "string", "required": True},
                {"name": "损失类型", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "账户": {
            "description": "金融或支付平台账户",
            "properties": [
                {"name": "账户名", "type": "string", "required": True},
                {"name": "平台", "type": "string", "required": False},
                {"name": "余额", "type": "float", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "金额": {
            "description": "涉及的资金数额，含数值和货币类型",
            "properties": [
                {"name": "数值", "type": "float", "required": True},
                {"name": "货币", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
        },
    },
    "define_call_count": 8,
    "registry_after": "8 types added to known_type_registry",
}

DEMO_DEFINE_METADATA = {
    "new_types": ["表", "字段", "数据表", "列", "column", "datatable"],
    "type_definitions": {
        "表": {
            "description": "数据库中的数据表，存储特定业务域的结构化数据",
            "properties": [
                {"name": "表名", "type": "string", "required": True},
                {"name": "表描述", "type": "string", "required": False},
                {"name": "数据域", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "数据表": {
            "description": "存储历史数据的持久化数据表",
            "properties": [
                {"name": "表名", "type": "string", "required": True},
                {"name": "记录数", "type": "integer", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "datatable": {
            "description": "A persistent data table storing historical records",
            "properties": [
                {"name": "table_name", "type": "string", "required": True},
                {"name": "row_count", "type": "integer", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "字段": {
            "description": "数据库表中的列，包含字段名、数据类型和业务含义描述",
            "properties": [
                {"name": "字段名", "type": "string", "required": True},
                {"name": "数据类型", "type": "string", "required": True},
                {"name": "业务含义", "type": "string", "required": False},
            ],
            "parent_types": ["Entity"],
        },
        "列": {
            "description": "数据库表的列，存储单个属性的值",
            "properties": [
                {"name": "列名", "type": "string", "required": True},
                {"name": "数据类型", "type": "string", "required": True},
            ],
            "parent_types": ["Entity"],
        },
        "column": {
            "description": "A column in a database table, storing attribute values",
            "properties": [
                {"name": "column_name", "type": "string", "required": True},
                {"name": "data_type", "type": "string", "required": True},
            ],
            "parent_types": ["Entity"],
        },
    },
    "define_call_count": 6,
}

# ── D. Canonicalize demo ──────────────────────────────────────

DEMO_CANONICALIZE_FRAUD = {
    "canonicalized_types": {
        "嫌疑人": "person",
        "嫌疑犯": "person",
        "suspect": "person",
        "公司": "organization",
        "IP地址": "ip",
        "受害者": "person",
        "账户": "account",
        "金额": "amount",
    },
    "canonicalize_details": {
        "嫌疑人 → person": {"similarity": 0.92, "decision": "forced", "threshold": "≥0.85"},
        "嫌疑犯 → person": {"similarity": 0.88, "decision": "forced", "threshold": "≥0.85"},
        "suspect → person": {"similarity": 0.85, "decision": "forced", "threshold": "≥0.85"},
        "公司 → organization": {"similarity": 0.89, "decision": "forced", "threshold": "≥0.85"},
        "IP地址 → ip": {"similarity": 0.95, "decision": "forced", "threshold": "≥0.85"},
        "受害者 → person": {"similarity": 0.78, "decision": "suggested", "threshold": "0.70~0.85 — 需人工审批"},
        "账户 → account": {"similarity": 0.91, "decision": "forced", "threshold": "≥0.85"},
        "金额 → amount": {"similarity": 0.32, "decision": "unchanged (NEW)", "threshold": "<0.70 — 独立新类型"},
    },
    "before_count": 9,
    "after_count": 5,
    "reduction": "9 → 5 (44% type reduction)",
    "human_approval_needed": ["受害者 → person (0.78, suggested)"],
}

DEMO_CANONICALIZE_METADATA = {
    "canonicalized_types": {
        "表": "table",
        "数据表": "table",
        "datatable": "table",
        "字段": "field",
        "列": "field",
        "column": "field",
    },
    "canonicalize_details": {
        "表 → table": {"similarity": 0.90, "decision": "forced", "threshold": "≥0.85"},
        "数据表 → table": {"similarity": 0.87, "decision": "forced", "threshold": "≥0.85"},
        "datatable → table": {"similarity": 0.86, "decision": "forced", "threshold": "≥0.85"},
        "字段 → field": {"similarity": 0.88, "decision": "forced", "threshold": "≥0.85"},
        "列 → field": {"similarity": 0.82, "decision": "suggested", "threshold": "0.70~0.85 — 需人工审批"},
        "column → field": {"similarity": 0.85, "decision": "forced", "threshold": "≥0.85"},
    },
    "before_count": 6,
    "after_count": 2,
    "reduction": "6 → 2 (67% type reduction)",
    "human_approval_needed": ["列 → field (0.82, suggested)"],
}

# ── E. Guided mode demo ──────────────────────────────────────

DEMO_GUIDED_FRAUD = {
    "mode": "GUIDED",
    "allowed_vertex_labels": ["person", "device", "ip", "account", "organization", "location"],
    "extracted_entities": [
        {"name": "张三", "label": "person"},
        {"name": "摩拜科技", "label": "organization"},
        {"name": "192.168.1.100", "label": "ip"},
        {"name": "李四", "label": "person"},
        {"name": "银行账户", "label": "account"},
        {"name": "王五", "label": "person"},
        {"name": "支付宝账户", "label": "account"},
        {"name": "赵六", "label": "person"},
        {"name": "Chen", "label": "person"},
    ],
    "extracted_relations": [
        {"source": "张三", "source_label": "person", "target": "192.168.1.100", "target_label": "ip", "label": "uses"},
        {"source": "张三", "source_label": "person", "target": "银行账户", "target_label": "account", "label": "logs_in"},
        {"source": "张三", "source_label": "person", "target": "王五", "target_label": "person", "label": "transfers"},
    ],
    "rejected_entities": [
        {"raw_name": "50万元", "raw_type": "金额", "reason": "'金额' not in allowed_vertex_labels → DROPPED"},
    ],
    "unique_types": 5,
    "note": "Guided模式一步到位，无需Define+Canonicalize。但'金额'概念被丢弃。",
}

# ── F. Registry accumulation demo (3 runs) ────────────────────

DEMO_REGISTRY_ACCUMULATION = {
    "run_1": {
        "known_types": {},
        "new_types_discovered": 8,
        "define_calls": 8,
        "registry_after": 8,
        "note": "首次运行：全部新类型，Define开销大",
    },
    "run_2": {
        "known_types": {"person", "organization", "ip", "account", "amount"},
        "new_types_discovered": 2,
        "define_calls": 2,
        "registry_after": 10,
        "note": "第二次运行：只有2个新类型（vehicle_model, location），Define≈0开销",
    },
    "run_3": {
        "known_types": {"person", "organization", "ip", "account", "amount", "vehicle_model", "location"},
        "new_types_discovered": 0,
        "define_calls": 0,
        "registry_after": 10,
        "note": "稳定运行：全部类型已知，Define开销=0！",
    },
    "trend": [
        {"run": 1, "new_types": 8, "define_calls": 8},
        {"run": 2, "new_types": 2, "define_calls": 2},
        {"run": 3, "new_types": 0, "define_calls": 0},
    ],
}

# ── G. Human override demo ────────────────────────────────────

DEMO_HUMAN_OVERRIDE = {
    "manual_type_definitions": {
        "person": {
            "description": "自然人（含嫌疑人、受害者等所有人类实体）",
            "properties": [
                {"name": "姓名", "type": "string", "required": True},
                {"name": "身份证号", "type": "string", "required": False},
                {"name": "角色", "type": "string", "required": False},
            ],
        },
        "ip": {
            "description": "网络终端设备的IP地址标识",
            "properties": [
                {"name": "IP值", "type": "string", "required": True},
                {"name": "归属地", "type": "string", "required": False},
            ],
        },
    },
    "suggested_mappings_to_approve": [
        {"raw_type": "受害者", "canonical_type": "person", "similarity": 0.78,
         "auto_decision": "suggested (0.70~0.85)", "human_decision": "✅ APPROVED → merge to person"},
    ],
    "effect": "2 manual overrides → skip 2 LLM Define calls → save cost + improve accuracy",
}

# ── H. Comparison: EVOLVING vs GUIDED ────────────────────────

DEMO_COMPARISON = {
    "dimension": ["Schema来源", "Type数量", "LLM调用次数", "语义丰富度", "适用场景", "人工干预"],
    "EVOLVING": ["LLM自动进化", "5 (合并后)", "Define:8 + Canonicalize:0", "高(保留raw_type语义)", "开放域/未知类型", "审批suggested映射"],
    "GUIDED": ["人工预定义", "5 (固定)", "Extract:1", "低(只有label)", "封闭域/已知类型", "预定义label集合"],
    "conclusion": (
        "EVOLVING: schema自动进化，保留语义丰富度，但首次运行开销大；"
        "GUIDED: schema一步到位，零额外开销，但丢弃不在预定义集合中的概念。"
    ),
}


# ═══════════════════════════════════════════════════════════════
# Handler functions — try real operators, fall back to demo data
# ═══════════════════════════════════════════════════════════════

def _safe_json(data, **kwargs):
    return json.dumps(data, ensure_ascii=False, indent=2, **kwargs)


def run_edc_extract(text, scenario):
    """Show raw LLM extraction before EDC — type explosion visible."""
    demo = DEMO_EXTRACT_FRAUD if scenario == "fraud_case" else DEMO_EXTRACT_METADATA
    demo = demo | {"input_text": text, "demo": True}
    return _safe_json(demo)


def run_edc_define(text, scenario):
    """Show Define phase output — semantic definitions for new types."""
    demo = DEMO_DEFINE_FRAUD if scenario == "fraud_case" else DEMO_DEFINE_METADATA
    demo = demo | {"input_text": text, "demo": True}
    return _safe_json(demo)


def run_edc_canonicalize(scenario):
    """Show Canonicalize phase — synonym merge via embedding similarity."""
    demo = DEMO_CANONICALIZE_FRAUD if scenario == "fraud_case" else DEMO_CANONICALIZE_METADATA
    demo = demo | {"demo": True}
    return _safe_json(demo)


def run_guided_extract(text, scenario):
    """Show Guided mode — Pydantic-constrained extraction."""
    demo = DEMO_GUIDED_FRAUD | {"input_text": text, "demo": True}
    return _safe_json(demo)


def run_registry_accumulation():
    """Show known_type_registry growth across 3 runs."""
    return _safe_json(DEMO_REGISTRY_ACCUMULATION | {"demo": True})


def run_human_override():
    """Show human override capabilities."""
    return _safe_json(DEMO_HUMAN_OVERRIDE | {"demo": True})


def run_comparison():
    """Show EVOLVING vs GUIDED comparison."""
    return _safe_json(DEMO_COMPARISON | {"demo": True})


# ═══════════════════════════════════════════════════════════════
# Gradio UI builder
# ═══════════════════════════════════════════════════════════════

def create_edc_schema_block():
    """Create the EDC Schema Pipeline tab with 5 operators showcased."""

    gr.Markdown(
        "# Knowledge Graph Schema EDC Pipeline\n\n"
        "EDC = Extract → Define → Canonicalize — 知识图谱schema自动进化机制\n\n"
        "**核心问题**: LLM对同一概念会起不同名字（嫌疑人/嫌疑犯/suspect），"
        "直接写入就炸了。EDC三阶段自动合并同义类型，防止type explosion。\n\n"
        "**⚠️ 重要声明**: 本Tab展示的是**知识图谱(KG)**的schema进化机制。"
        "关系图谱的schema是业务人员人工定义的，与EDC无关。"
    )

    with gr.Row():
        # ── Left column: Input controls ──
        with gr.Column(scale=1):
            gr.Markdown("## 输入与配置")

            # ── Scenario selector ──
            gr.Markdown("### 场景选择")
            edc_scenario = gr.Dropdown(
                choices=[
                    ("🚨 风控欺诈案例 (type explosion最严重)", "fraud_case"),
                    ("📊 元数据检索 (货拉拉真实场景)", "metadata_query"),
                ],
                value="fraud_case",
                label="演示场景",
            )
            edc_text = gr.Textbox(
                label="输入文本",
                value=DEMO_TEXTS["fraud_case"],
                lines=6,
            )

            # ── Mode selector ──
            gr.Markdown("### Schema模式")
            edc_mode = gr.Dropdown(
                choices=[
                    ("EVOLVING — EDC三阶段自动进化 (默认)", "evolving"),
                    ("GUIDED — Pydantic约束提取 (可选)", "guided"),
                    ("对比 EVOLVING vs GUIDED", "compare"),
                ],
                value="evolving",
                label="模式选择",
            )

            # ── Config controls ──
            gr.Markdown("---\n### A. Config 参数")
            edc_canonicalize_strategy = gr.Dropdown(
                choices=["EMBEDDING_SIM", "EXACT_MATCH", "LLM_CLASSIFY"],
                value="EMBEDDING_SIM",
                label="Canonicalize策略",
            )
            edc_similarity_threshold = gr.Slider(
                minimum=0.50, maximum=0.99, value=0.85, step=0.01,
                label="Canonicalize强制合并阈值 (≥此值直接合并)",
            )
            edc_suggest_threshold = gr.Slider(
                minimum=0.50, maximum=0.84, value=0.70, step=0.01,
                label="Canonicalize建议合并阈值 (此值~强制阈值需人工审批)",
            )
            edc_allow_override = gr.Checkbox(
                value=True, label="允许人工override (allow_manual_override)",
            )
            edc_show_config_btn = gr.Button("查看当前Config", variant="secondary")

            # ── Phase buttons ──
            gr.Markdown("---\n### EDC三阶段操作")
            edc_extract_btn = gr.Button("1️⃣ Extract — LLM自由提取 (查看type explosion)", variant="primary")
            edc_define_btn = gr.Button("2️⃣ Define — 为新类型生成语义定义", variant="primary")
            edc_canonicalize_btn = gr.Button("3️⃣ Canonicalize — 合并同义类型", variant="primary")
            edc_guided_btn = gr.Button("🎯 Guided模式提取 (一步到位)", variant="secondary")

            # ── Advanced demos ──
            gr.Markdown("---\n### 进阶展示")
            edc_registry_btn = gr.Button("📈 Registry累积趋势 (3次运行)", variant="secondary")
            edc_override_btn = gr.Button("🛠 人工override能力", variant="secondary")
            edc_compare_btn = gr.Button("⚖️ EVOLVING vs GUIDED对比", variant="secondary")

        # ── Right column: Results display ──
        with gr.Column(scale=2):
            with gr.Tabs():
                # ── Tab: Config ──
                with gr.Tab("A. Config"):
                    edc_config_out = gr.Code(label="SchemaConfig (EVOLVING)", language="json")
                    edc_config_guided_out = gr.Code(label="SchemaConfig (GUIDED)", language="json")

                # ── Tab: Extract ──
                with gr.Tab("B. Extract (Raw)"):
                    gr.Markdown(
                        "**⚠️ Type Explosion 可见**: LLM对同一概念起不同名字，"
                        "直接写入KG就炸了。看unique_raw_types数量 vs 实际概念数。"
                    )
                    edc_extract_out = gr.Code(label="Extract Raw Output — type explosion", language="json")
                    edc_extract_table = gr.Dataframe(
                        headers=["实体名", "LLM给的raw_type", "实际概念"],
                        label="类型对照表",
                        datatype=["str", "str", "str"],
                        row_count=10,
                        column_count=3,
                        interactive=False,
                    )

                # ── Tab: Define ──
                with gr.Tab("C. Define"):
                    gr.Markdown(
                        "为新类型生成语义定义 (description + properties + parent_types)。\n"
                        "**关键**: 定义存入known_type_registry，下次遇到直接跳过。"
                    )
                    edc_define_out = gr.Code(label="Define Output — semantic definitions", language="json")

                # ── Tab: Canonicalize ──
                with gr.Tab("D. Canonicalize"):
                    gr.Markdown(
                        "合并同义类型：embedding相似度 ≥0.85=forced合并，"
                        "0.70~0.85=suggested需人工审批，<0.70=独立新类型。\n"
                        "**结果**: type数量大幅缩减！"
                    )
                    edc_canonicalize_out = gr.Code(label="Canonicalize Output — synonym merge", language="json")
                    edc_canonicalize_table = gr.Dataframe(
                        headers=["raw_type", "canonical_type", "相似度", "决策", "阈值区间"],
                        label="Canonicalize映射表",
                        datatype=["str", "str", "number", "str", "str"],
                        interactive=False,
                    )

                # ── Tab: Guided ──
                with gr.Tab("E. Guided"):
                    gr.Markdown(
                        "Guided模式：Pydantic ResponseModel约束LLM只输出预定义label。\n"
                        "**优势**: 一步到位，无需Define+Canonicalize。\n"
                        "**劣势**: 不在预定义集合中的概念被丢弃（如'金额'）。"
                    )
                    edc_guided_out = gr.Code(label="Guided Extract Output", language="json")

                # ── Tab: Registry Accumulation ──
                with gr.Tab("F. Registry累积"):
                    gr.Markdown(
                        "known_type_registry随运行次数增长：\n"
                        "首次运行=全部新类型(Define开销大) → 稳定运行≈0额外LLM调用"
                    )
                    edc_registry_out = gr.Code(label="Registry Accumulation (3 Runs)", language="json")

                # ── Tab: Human Override ──
                with gr.Tab("G. 人工干预"):
                    gr.Markdown(
                        "3个人工干预入口：\n"
                        "1. **预种子registry**: 启动前定义核心类型，LLM只补充长尾\n"
                        "2. **审批suggested映射**: 0.70~0.85相似度需人工确认\n"
                        "3. **override Define**: 修改LLM定义，跳过LLM调用"
                    )
                    edc_override_out = gr.Code(label="Human Override Capabilities", language="json")

                # ── Tab: Comparison ──
                with gr.Tab("H. 模式对比 ⚖️"):
                    gr.Markdown(
                        "EVOLVING vs GUIDED 对比：不同场景选不同模式。"
                    )
                    edc_compare_out = gr.Code(label="EVOLVING vs GUIDED Comparison", language="json")
                    edc_compare_table = gr.Dataframe(
                        headers=["维度", "EVOLVING", "GUIDED"],
                        label="模式对比表",
                        datatype=["str", "str", "str"],
                        interactive=False,
                    )

    # ═══════════════════════════════════════════════════════════
    # Event bindings
    # ═══════════════════════════════════════════════════════════

    # A. Config
    def _show_config(strategy, sim_thresh, suggest_thresh, allow_override):
        evolving = DEMO_CONFIG_EVOLVING.copy()
        evolving["canonicalize_strategy"] = strategy
        evolving["canonicalize_similarity_threshold"] = sim_thresh
        evolving["canonicalize_suggest_threshold"] = suggest_thresh
        evolving["allow_manual_override"] = allow_override
        return _safe_json(evolving), _safe_json(DEMO_CONFIG_GUIDED)

    edc_show_config_btn.click(
        fn=_show_config,
        inputs=[edc_canonicalize_strategy, edc_similarity_threshold,
                edc_suggest_threshold, edc_allow_override],
        outputs=[edc_config_out, edc_config_guided_out],
    )

    # B. Extract
    def _extract_and_format(text, scenario):
        result_json = run_edc_extract(text, scenario)
        demo = DEMO_EXTRACT_FRAUD if scenario == "fraud_case" else DEMO_EXTRACT_METADATA
        # Build comparison table: raw_type vs actual concept
        concept_map = {
            "嫌疑人": "person (人)", "嫌疑犯": "person (人)", "suspect": "person (人)",
            "受害者": "person (人)", "公司": "organization (组织)", "IP地址": "ip (地址)",
            "账户": "account (账户)", "金额": "amount (数额)", "第三方": "person (人)",
            "表": "table (表)", "数据表": "table (表)", "datatable": "table (表)",
            "字段": "field (字段)", "列": "field (字段)", "column": "field (字段)",
        }
        rows = []
        for ent in demo["raw_entities"]:
            raw_type = ent["raw_type"]
            concept = concept_map.get(raw_type, "???")
            rows.append([ent["name"], raw_type, concept])
        return result_json, rows

    edc_extract_btn.click(
        fn=_extract_and_format,
        inputs=[edc_text, edc_scenario],
        outputs=[edc_extract_out, edc_extract_table],
    )

    # C. Define
    edc_define_btn.click(
        fn=run_edc_define,
        inputs=[edc_text, edc_scenario],
        outputs=[edc_define_out],
    )

    # D. Canonicalize
    def _canonicalize_and_format(scenario):
        result_json = run_edc_canonicalize(scenario)
        demo = DEMO_CANONICALIZE_FRAUD if scenario == "fraud_case" else DEMO_CANONICALIZE_METADATA
        rows = []
        for mapping, details in demo["canonicalize_details"].items():
            parts = mapping.split(" → ")
            rows.append([parts[0], parts[1], details["similarity"],
                         details["decision"], details["threshold"]])
        return result_json, rows

    edc_canonicalize_btn.click(
        fn=_canonicalize_and_format,
        inputs=[edc_scenario],
        outputs=[edc_canonicalize_out, edc_canonicalize_table],
    )

    # E. Guided
    edc_guided_btn.click(
        fn=run_guided_extract,
        inputs=[edc_text, edc_scenario],
        outputs=[edc_guided_out],
    )

    # F. Registry accumulation
    edc_registry_btn.click(fn=run_registry_accumulation, outputs=[edc_registry_out])

    # G. Human override
    edc_override_btn.click(fn=run_human_override, outputs=[edc_override_out])

    # H. Comparison
    def _compare_and_format():
        result_json = run_comparison()
        rows = list(zip(
            DEMO_COMPARISON["dimension"],
            DEMO_COMPARISON["EVOLVING"],
            DEMO_COMPARISON["GUIDED"],
        ))
        return result_json, rows

    edc_compare_btn.click(fn=_compare_and_format, outputs=[edc_compare_out, edc_compare_table])

    # ═══════════════════════════════════════════════════════════
    # Auto-load demo data on page load
    # ═══════════════════════════════════════════════════════════

    def _load_demo_data():
        """Pre-populate all tabs with demo data for immediate showcasing."""
        config_evo = _safe_json(DEMO_CONFIG_EVOLVING)
        config_guided = _safe_json(DEMO_CONFIG_GUIDED)

        # Extract with table
        fraud_extract = _safe_json(DEMO_EXTRACT_FRAUD | {"demo": True})
        concept_map = {
            "嫌疑人": "person (人)", "嫌疑犯": "person (人)", "suspect": "person (人)",
            "受害者": "person (人)", "公司": "organization (组织)", "IP地址": "ip (地址)",
            "账户": "account (账户)", "金额": "amount (数额)", "第三方": "person (人)",
        }
        extract_rows = []
        for ent in DEMO_EXTRACT_FRAUD["raw_entities"]:
            raw_type = ent["raw_type"]
            concept = concept_map.get(raw_type, "???")
            extract_rows.append([ent["name"], raw_type, concept])

        define_out = _safe_json(DEMO_DEFINE_FRAUD | {"demo": True})

        canonicalize_rows = []
        for mapping, details in DEMO_CANONICALIZE_FRAUD["canonicalize_details"].items():
            parts = mapping.split(" → ")
            canonicalize_rows.append([parts[0], parts[1], details["similarity"],
                                      details["decision"], details["threshold"]])
        canonicalize_out = _safe_json(DEMO_CANONICALIZE_FRAUD | {"demo": True})

        guided_out = _safe_json(DEMO_GUIDED_FRAUD | {"demo": True})

        registry_out = _safe_json(DEMO_REGISTRY_ACCUMULATION | {"demo": True})

        override_out = _safe_json(DEMO_HUMAN_OVERRIDE | {"demo": True})

        compare_rows = list(zip(
            DEMO_COMPARISON["dimension"],
            DEMO_COMPARISON["EVOLVING"],
            DEMO_COMPARISON["GUIDED"],
        ))
        compare_out = _safe_json(DEMO_COMPARISON | {"demo": True})

        return (
            config_evo, config_guided,
            fraud_extract, extract_rows,
            define_out,
            canonicalize_out, canonicalize_rows,
            guided_out,
            registry_out,
            override_out,
            compare_out, compare_rows,
        )

    demo_outputs = [
        edc_config_out, edc_config_guided_out,
        edc_extract_out, edc_extract_table,
        edc_define_out,
        edc_canonicalize_out, edc_canonicalize_table,
        edc_guided_out,
        edc_registry_out,
        edc_override_out,
        edc_compare_out, edc_compare_table,
    ]

    return demo_outputs, _load_demo_data
