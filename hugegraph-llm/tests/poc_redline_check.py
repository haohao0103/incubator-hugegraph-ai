#!/usr/bin/env python3
"""
poc-redline v1.2 自检脚本 — 运行方式:
    python tests/poc_redline_check.py tests/xxx_e2e_validation.py

P0 铁律（违反即视为生产事故）：
  1. 真实 HugeGraph: 代码中出现 localhost:8080 / 127.0.0.1:8080
  2. 真实 LLM: 代码中出现 api.xiaomimimo.com 或真实 API 调用
  3. 零模拟: 无 setTimeout/fake_data/mock_data/hardcoded_result
  4. 标准数据集: 非自造数据（需人工复核，脚本给出 WARNING）
  5. 量化指标: 代码中有 Accuracy/Recall/F1/ROUGE/Latency
  6. 结果文件: 同目录存在 *_result.json
  7. Schema 合规: 遵循 HUGEGRAPH_SCHEMA_DESIGN_GUIDE.md 规范
"""
import sys, os, re, argparse

CHECKS = [
    ("P0-1 真实 HugeGraph", r"localhost:8080|127\.0\.0\.1:8080"),
    ("P0-2 真实 LLM",       r"api\.xiaomimimo\.com|openai\.com|api_key|api_key"),
    ("P0-5 量化指标",       r"Accuracy|Recall@K?|F1|ROUGE|Latency|Precision|BLEU|NDCG|MRR|EM"),
]

FORBIDDEN = [
    ("P0-3 零模拟(setTimeout)", r"setTimeout|setInterval"),
    ("P0-3 零模拟(fake/mock)",  r"fake_data|mock_data|mocked|hardcoded_result|mock_llm|fake_embedding|hash_embedding"),
]

# Schema 反模式检查（违反 = FAIL，必须修正）
SCHEMA_ANTIPATTERNS = [
    ("SCHEMA-1 万能标签",      r'"name"\s*:\s*"Node"|"name"\s*:\s*"node"'),
    ("SCHEMA-2 标签爆炸",      r'vertexlabels.*\n.*"name":\s*"\w+_\w+_\w+_\w+'),  # 4段以上下划线
    ("SCHEMA-3 无向语义边",    r'"name"\s*:\s*"relation"|"name"\s*:\s*"edge"|"name"\s*:\s*"link"'),
]

# Schema 推荐检查（WARNING 级别，建议修正）
SCHEMA_RECOMMENDATIONS = [
    ("SCHEMA-R1 属性名不一致", r'revenue_usd.*revenue[^_]|revenue[^_].*revenue_usd'),
    ("SCHEMA-R2 未建索引",     r'index_type|indexlabels', True),  # True = 期望存在
]

DATASET_HINTS = [
    ("HuggingFace datasets", r"load_dataset|datasets\.load|huggingface"),
    ("标准 benchmark",      r"GraphRAG-Bench|HotpotQA|MuSiQue|MS MARCO|Natural Questions|SQuAD|CMRC|DuReader|2WikiMultiHopQA|ICEWS|FB15k|WN18RR"),
]

def check_file(filepath):
    if not os.path.exists(filepath):
        print(f"❌ FILE_NOT_FOUND: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    result_file = os.path.join(dirname, basename.replace(".py", "_result.json"))

    print(f"\n{'='*60}")
    print(f"poc-redline v1.1 自检报告")
    print(f"目标文件: {filepath}")
    print(f"{'='*60}\n")

    report = {"file": filepath, "checks": [], "overall": "PASS"}
    all_pass = True

    # 正向检查
    for name, pattern in CHECKS:
        found = bool(re.search(pattern, src, re.IGNORECASE))
        status = "✅ PASS" if found else "❌ FAIL"
        if not found:
            all_pass = False
            report["overall"] = "FAIL"
        report["checks"].append({"item": name, "status": "PASS" if found else "FAIL"})
        print(f"  {status}  {name}")

    # 禁止项检查
    for name, pattern in FORBIDDEN:
        found = bool(re.search(pattern, src, re.IGNORECASE))
        status = "❌ FAIL" if found else "✅ PASS"
        if found:
            all_pass = False
            report["overall"] = "FAIL"
        report["checks"].append({"item": name, "status": "FAIL" if found else "PASS"})
        print(f"  {status}  {name}")

    # 数据集检查（WARNING 级别，需人工复核）
    dataset_found = False
    for name, pattern in DATASET_HINTS:
        if re.search(pattern, src, re.IGNORECASE):
            dataset_found = True
            break
    status = "✅ PASS" if dataset_found else "⚠️  WARNING (未检测到标准数据集引用，需人工复核)"
    if not dataset_found:
        report["overall"] = "WARN"
    report["checks"].append({"item": "P0-4 标准数据集", "status": "PASS" if dataset_found else "WARN"})
    print(f"  {status}  P0-4 标准数据集")

    # Schema 反模式检查（FAIL 级别）
    schema_has_definition = bool(re.search(r'vertexlabels|edgelabels|propertykeys', src, re.IGNORECASE))
    if schema_has_definition:
        for name, pattern in SCHEMA_ANTIPATTERNS:
            found = bool(re.search(pattern, src, re.IGNORECASE))
            status = "❌ FAIL" if found else "✅ PASS"
            if found:
                all_pass = False
                report["overall"] = "FAIL"
            report["checks"].append({"item": name, "status": "FAIL" if found else "PASS"})
            print(f"  {status}  {name}")
    else:
        print(f"  ⚪ SKIP  SCHEMA-* (代码中未检测到 Schema 定义，跳过 Schema 检查)")

    # Schema 推荐检查（WARNING 级别）
    if schema_has_definition:
        for name, pattern in SCHEMA_RECOMMENDATIONS:
            found = bool(re.search(pattern, src, re.IGNORECASE))
            status = "⚠️  WARNING" if not found else "✅ PASS"
            if not found:
                if report["overall"] == "PASS":
                    report["overall"] = "WARN"
            report["checks"].append({"item": name, "status": "PASS" if found else "WARN"})
            print(f"  {status}  {name}")

    # 结果文件检查
    result_exists = os.path.exists(result_file)
    status = "✅ PASS" if result_exists else "⚠️  WARNING (结果文件不存在)"
    if not result_exists:
        if report["overall"] == "PASS":
            report["overall"] = "WARN"
    report["checks"].append({"item": "P0-6 结果文件", "status": "PASS" if result_exists else "WARN"})
    print(f"  {status}  P0-6 结果文件 ({os.path.basename(result_file)})")

    print(f"\n{'='*60}")
    if all_pass and report["overall"] == "PASS":
        print("🟢 OVERALL: PASS — 符合 poc-redline v1.1 P0 铁律")
    elif report["overall"] == "WARN":
        print("🟡 OVERALL: WARNING — 基本合规，有建议项需人工复核")
    else:
        print("🔴 OVERALL: FAIL — 违反 poc-redline v1.1 P0 铁律，禁止交付")
    print(f"{'='*60}\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="poc-redline v1.1 自检脚本")
    parser.add_argument("file", help="PoC Python 文件路径")
    args = parser.parse_args()
    check_file(args.file)
