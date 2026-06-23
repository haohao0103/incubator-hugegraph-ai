#!/usr/bin/env python3
"""
HugeGraph Schema 自动校验脚本 — poc-redline v1.2 配套工具
运行方式:
    python tests/hugegraph_schema_validator.py tests/xxx_e2e_validation.py
    python tests/hugegraph_schema_validator.py --schema-json '{...}'

校验规则来源: docs/HUGEGRAPH_SCHEMA_DESIGN_GUIDE.md
"""
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── 命名规范 ────────────────────────────────────────────────

VERTEX_LABEL_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')  # PascalCase
EDGE_LABEL_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')       # snake_case lowercase
PROPERTY_KEY_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')     # snake_case lowercase

# ── 反模式黑名单 ────────────────────────────────────────────

ANTIPATTERN_VERTEX_LABELS = {"Node", "node", "Entity", "entity", "Item", "item", "Vertex", "vertex"}
ANTIPATTERN_EDGE_LABELS = {"relation", "edge", "link", "connects", "relates", "has_relation"}

# ── 推荐属性名标准化 ────────────────────────────────────────

RECOMMENDED_PROPERTY_NAMES = {
    "name", "description", "created_at", "updated_at", "valid_from", "valid_until",
    "confidence", "weight", "source", "category", "country", "revenue_usd",
    "market_share_pct", "stock_code", "entity_type",
}


# ═════════════════════════════════════════════════════════════
# Schema 解析
# ═════════════════════════════════════════════════════════════

def extract_schema_from_python(filepath: str) -> Optional[Dict[str, Any]]:
    """从 Python 文件中提取 Schema 字典（如 TKG_SCHEMA, TECH_KG_SCHEMA）."""
    import ast
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    # 找 Schema 变量赋值: XXX_SCHEMA = { ... }
    match = re.search(r'([A-Z_]*SCHEMA)\s*=\s*(\{.*?\})(?=\s*\n[A-Z]|\s*$)', src, re.DOTALL)
    if not match:
        return None

    schema_str = match.group(2)
    try:
        # 使用 ast.literal_eval 解析 Python 字典（支持 True/False/None）
        return ast.literal_eval(schema_str)
    except (SyntaxError, ValueError) as e:
        print(f"  ⚠️  解析 Schema 失败: {e}")
        return None


def parse_schema(schema: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """解析 Schema 字典，返回 (property_keys, vertex_labels, edge_labels)."""
    pks = schema.get("propertykeys", [])
    vls = schema.get("vertexlabels", [])
    els = schema.get("edgelabels", [])
    return pks, vls, els


# ═════════════════════════════════════════════════════════════
# 校验规则
# ═════════════════════════════════════════════════════════════

def check_vertex_label_naming(vls: List[Dict]) -> List[Tuple[str, str]]:
    """检查 VertexLabel 命名规范: PascalCase."""
    errors = []
    for vl in vls:
        name = vl.get("name", "")
        if not name:
            errors.append(("VertexLabel", "空名称"))
            continue
        if not VERTEX_LABEL_PATTERN.match(name):
            errors.append(("VertexLabel", f"'{name}' 不符合 PascalCase"))
        if name in ANTIPATTERN_VERTEX_LABELS:
            errors.append(("VertexLabel", f"'{name}' 是反模式万能标签"))
    return errors


def check_edge_label_naming(els: List[Dict]) -> List[Tuple[str, str]]:
    """检查 EdgeLabel 命名规范: snake_case 动词/动宾短语."""
    errors = []
    for el in els:
        name = el.get("name", "")
        if not name:
            errors.append(("EdgeLabel", "空名称"))
            continue
        if not EDGE_LABEL_PATTERN.match(name):
            errors.append(("EdgeLabel", f"'{name}' 不符合 snake_case"))
        if name in ANTIPATTERN_EDGE_LABELS:
            errors.append(("EdgeLabel", f"'{name}' 是反模式无向语义标签"))
    return errors


def check_property_key_naming(pks: List[Dict]) -> List[Tuple[str, str]]:
    """检查 PropertyKey 命名规范: snake_case."""
    errors = []
    for pk in pks:
        name = pk.get("name", "")
        if not name:
            errors.append(("PropertyKey", "空名称"))
            continue
        if not PROPERTY_KEY_PATTERN.match(name):
            errors.append(("PropertyKey", f"'{name}' 不符合 snake_case"))
    return errors


def check_primary_keys(vls: List[Dict], pks: List[Dict]) -> List[Tuple[str, str]]:
    """检查 VertexLabel 主键设置."""
    errors = []
    pk_names = {p.get("name") for p in pks}
    for vl in vls:
        name = vl.get("name", "")
        primary = vl.get("primary_keys", [])
        if not primary:
            errors.append(("PrimaryKey", f"'{name}' 未设置 primary_keys"))
        else:
            for pk in primary:
                if pk not in pk_names:
                    errors.append(("PrimaryKey", f"'{name}' 的主键 '{pk}' 未在 propertykeys 中定义"))
    return errors


def check_edge_connectivity(els: List[Dict], vl_names: set) -> List[Tuple[str, str]]:
    """检查 EdgeLabel 的 source_label/target_label 是否指向存在的 VertexLabel."""
    errors = []
    for el in els:
        name = el.get("name", "")
        sl = el.get("source_label", "")
        tl = el.get("target_label", "")
        if sl not in vl_names:
            errors.append(("EdgeConnectivity", f"'{name}' source_label='{sl}' 不存在"))
        if tl not in vl_names:
            errors.append(("EdgeConnectivity", f"'{name}' target_label='{tl}' 不存在"))
    return errors


def check_label_count(vls: List[Dict], els: List[Dict]) -> List[Tuple[str, str]]:
    """检查标签数量是否在合理范围（避免爆炸）."""
    errors = []
    if len(vls) > 50:
        errors.append(("LabelCount", f"VertexLabel 数量 {len(vls)} > 50，存在爆炸风险"))
    if len(els) > 30:
        errors.append(("LabelCount", f"EdgeLabel 数量 {len(els)} > 30，存在爆炸风险"))
    return errors


def check_universal_label(vls: List[Dict]) -> List[Tuple[str, str]]:
    """检查是否存在万能标签（反模式）."""
    errors = []
    for vl in vls:
        name = vl.get("name", "")
        props = vl.get("properties", [])
        # 如果标签名是反模式，且有一个 "type"/"entity_type"/"category" 属性 → 万能标签
        if name in ANTIPATTERN_VERTEX_LABELS:
            if any(p in props for p in ("type", "entity_type", "category", "kind")):
                errors.append(("UniversalLabel",
                    f"'{name}' 是万能标签，靠属性区分类型（建议拆分为多个 VertexLabel）"))
    return errors


# ═════════════════════════════════════════════════════════════
# 报告输出
# ═════════════════════════════════════════════════════════════

def validate_schema(schema: Dict[str, Any], source_name: str = "") -> Dict[str, Any]:
    """运行全部校验规则，返回报告."""
    pks, vls, els = parse_schema(schema)
    vl_names = {v.get("name", "") for v in vls}

    all_errors = []
    all_errors.extend(check_vertex_label_naming(vls))
    all_errors.extend(check_edge_label_naming(els))
    all_errors.extend(check_property_key_naming(pks))
    all_errors.extend(check_primary_keys(vls, pks))
    all_errors.extend(check_edge_connectivity(els, vl_names))
    all_errors.extend(check_label_count(vls, els))
    all_errors.extend(check_universal_label(vls))

    report = {
        "source": source_name,
        "property_keys": len(pks),
        "vertex_labels": len(vls),
        "edge_labels": len(els),
        "errors": [{"category": cat, "message": msg} for cat, msg in all_errors],
        "error_count": len(all_errors),
        "valid": len(all_errors) == 0,
    }
    return report


def print_report(report: Dict[str, Any]):
    print(f"\n{'='*60}")
    print(f"HugeGraph Schema 校验报告")
    if report.get("source"):
        print(f"来源: {report['source']}")
    print(f"{'='*60}")
    print(f"  PropertyKeys:   {report['property_keys']}")
    print(f"  VertexLabels:   {report['vertex_labels']}")
    print(f"  EdgeLabels:     {report['edge_labels']}")
    print(f"  错误数:         {report['error_count']}")
    print(f"{'='*60}")

    if report["errors"]:
        for err in report["errors"]:
            print(f"  ❌ [{err['category']}] {err['message']}")
    else:
        print(f"  ✅ 全部通过，Schema 设计合规")

    print(f"{'='*60}")
    if report["valid"]:
        print("🟢 OVERALL: VALID — 符合 HUGEGRAPH_SCHEMA_DESIGN_GUIDE.md")
    else:
        print("🔴 OVERALL: INVALID — 存在 Schema 设计问题，需修正")
    print(f"{'='*60}\n")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HugeGraph Schema 自动校验")
    parser.add_argument("file", nargs="?", help="PoC Python 文件路径（自动提取 Schema）")
    parser.add_argument("--schema-json", help="直接传入 Schema JSON 字符串")
    args = parser.parse_args()

    schema = None
    source_name = ""

    if args.schema_json:
        schema = json.loads(args.schema_json)
        source_name = "--schema-json"
    elif args.file:
        if not os.path.exists(args.file):
            print(f"❌ FILE_NOT_FOUND: {args.file}")
            sys.exit(1)
        schema = extract_schema_from_python(args.file)
        source_name = args.file
        if schema is None:
            print(f"  ⚠️  未在 {args.file} 中检测到 Schema 定义（如 XXX_SCHEMA = {{...}}）")
            print(f"  如果 Schema 是通过 REST 动态创建的，请使用 --schema-json 传入")
            sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    report = validate_schema(schema, source_name)
    print_report(report)
    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
