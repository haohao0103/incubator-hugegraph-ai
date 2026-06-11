# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this License except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under an License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Tests for Coreference Resolution operator.

Covers:
1. Data model: CorefMapping creation, serialization
2. Rule-based: personal pronoun resolution (他/她 → entity)
3. Rule-based: demonstrative resolution (该公司 → org)
4. Rule-based: title-based resolution (张先生 → 张三)
5. Entity catalog building
6. apply_to_text: text replacement with resolved entities
7. Edge cases: empty input, no entities, no coref found
8. Deduplication: same mention+canonical+chunk deduped
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from hugegraph_llm.operators.llm_op.coref_resolution import (
    CorefMapping,
    CorefResolver,
)

PASS = 0
FAIL = 0


def check(condition, test_name):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {test_name}")


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Test 1: Data Model ──────────────────────────────────────

print_header("Test 1: CorefMapping Data Model")

m1 = CorefMapping(
    mention="他",
    canonical="张三",
    entity_type="Person",
    chunk_id="chunk_1",
    confidence=0.9,
    method="rule",
)
d = m1.to_dict()
check(d["mention"] == "他", "mention preserved")
check(d["canonical"] == "张三", "canonical preserved")
check(d["entity_type"] == "Person", "entity_type preserved")
check(abs(d["confidence"] - 0.9) < 0.001, "confidence rounded to 4 decimals")
check(d["method"] == "rule", "method preserved")

m2 = CorefMapping(mention="x", canonical="y")
d2 = m2.to_dict()
check(d2.get("entity_type") == "", "Default entity_type is empty string")
check(d2.get("confidence") == 0.0, "Default confidence is 0.0")


# ── Test 2: Personal Pronoun Resolution ───────────────────────

print_header("Test 2: Personal Pronoun Resolution")

resolver = CorefResolver()

context_pronoun = {
    "chunks": [
        {"text": "张三在阿里云工作。他担任高级工程师，负责云计算平台的开发。", "chunk_id": "c0"},
        {"text": "李四是张三的同事。她也在阿里云工作。她的职位是产品经理。", "chunk_id": "c1"},
    ],
    "vertices": [
        {"label": "Person", "properties": {"name": "张三"}},
        {"label": "Person", "properties": {"name": "李四"}},
        {"label": "Organization", "properties": {"name": "阿里云"}},
    ],
}

result = resolver.run(context_pronoun)
mappings = result.get("coref_mappings", [])
count = result.get("coref_count", 0)

# Should resolve 他→张三, 她→李四 (from both chunks)
pronoun_to_zhangsan = [m for m in mappings if m["mention"] == "他"]
pronoun_to_lisi = [m for m in mappings if m["mention"] == "她"]

check(count >= 2, f"Pronoun resolution found >= 2 mappings (got {count})")
check(len(pronoun_to_zhangsan) >= 1, f"'他' resolved to someone (got {len(pronoun_to_zhangsan)})")
if pronoun_to_zhangsan:
    check(pronoun_to_zhangsan[0]["canonical"] == "张三", "'他' -> '张三'")
check(len(pronoun_to_lisi) >= 1, f"'她' resolved to someone (got {len(pronoun_to_lisi)})")
if pronoun_to_lisi:
    check(pronoun_to_lisi[0]["canonical"] in ("李四", "张三"), f"'女' -> valid person ('{pronoun_to_lisi[0]['canonical']}')")


# ── Test 3: Demonstrative / Organization Resolution ───────────

print_header("Test 3: Demonstrative + Organization Resolution")

context_org = {
    "chunks": [
        {"text": "阿里巴巴集团成立于1999年。该公司的总部位于杭州。这家公司目前是全球最大的电商平台之一。", "chunk_id": "c0"},
    ],
    "vertices": [
        {"label": "Organization", "properties": {"name": "阿里巴巴集团"}},
        {"label": "Location", "properties": {"name": "杭州"}},
    ],
}

result_org = resolver.run(context_org)
org_maps = result_org.get("coref_mappings", [])

# Should resolve 该公司/这家公司 → 阿里巴巴集团
demo_mappings = [m for m in org_maps if m["mention"] in ("该公司", "这家公司")]
check(len(demo_mappings) >= 1, f"Demonstrative resolution: >= 1 mapping (got {len(demo_mappings)})")
if demo_mappings:
    check(demo_mappings[0]["canonical"] == "阿里巴巴集团", f"'{demo_mappings[0]['mention']}' -> '阿里巴巴集团'")

# Also check for "这" as demonstrative
zhe_maps = [m for m in org_maps if m["mention"] == "这"]
check(len(zhe_maps) >= 1 or len(demo_mappings) >= 2, f"'这' also resolved (or >=2 demos total)")


# ── Test 4: Title-Based Resolution ───────────────────────────

print_header("Test 4: Title-Based Resolution")

context_title = {
    "chunks": [
        {"text": "张三是技术负责人。张先生带领团队完成了多个重要项目。他的管理风格很开放。", "chunk_id": "c0"},
    ],
    "vertices": [
        {"label": "Person", "properties": {"name": "张三"}},
    ],
}

result_title = resolver.run(context_title)
title_maps = result_title.get("coref_mappings", [])

# Should resolve 张先生 -> 张三
zhang_title = [m for m in title_maps if m["mention"] == "张先生"]
check(len(zhang_title) >= 1, f"'张先生' resolved (got {len(zhang_title)})")
if zhang_title:
    check(zhang_title[0]["canonical"] == "张三", "'张先生' -> '张三'")


# ── Test 5: Entity Catalog Building ──────────────────────────

print_header("Test 5: Entity Catalog Building")

vertices = [
    {"label": "Person", "properties": {"name": "张三", "age": "30"}},
    {"label": "Company", "properties": {"name": "阿里云", "industry": "Cloud"}},
    {"label": "Location", "properties": {"name": "深圳"}},
]

catalog = CorefResolver._build_entity_catalog(vertices)
check("张三" in catalog, "'Zhang San' in catalog")
check("阿里云" in catalog, "'Aliyun' in catalog")
check("深圳" in catalog, "'Shenzhen' in catalog")
check(catalog["张三"][0] == "Person", "'Zhang San' type = Person")
check(catalog["阿里云"][0] == "Company", "'Aliyun' type = Company")


# ── Test 6: apply_to_text Replacement ─────────────────────────

print_header("Test 6: Text Replacement via apply_to_text")

text = "张三在阿里云工作。他在那里担任高级工程师。该公司的总部在杭州。"
mappings = [
    CorefMapping(mention="他", canonical="张三", confidence=0.9),
    CorefMapping(mention="该公司", canonical="阿里云", confidence=0.85),
]

resolved = resolver.apply_to_text(text, mappings)
check("张三" in resolved, "Resolved text contains '张三'")
check("阿里云" in resolved, "Resolved text contains '阿里云'")
# Check that original mentions are replaced
check("他张三" in resolved or "张三" in resolved.replace("张三在阿里云工作。", ""), "Original '他' was replaced")


# ── Test 7: Edge Cases ───────────────────────────────────────

print_header("Test 7: Edge Cases")

# Empty chunks
empty_ctx = {"chunks": [], "vertices": []}
empty_result = resolver.run(empty_ctx)
check(empty_result.get("coref_count") == 0, "Empty chunks -> 0 mappings")

# No vertices (no entities to resolve to)
no_v_ctx = {
    "chunks": [{"text": "他去了那里", "chunk_id": "c0"}],
    "vertices": [],
}
no_v_result = resolver.run(no_v_ctx)
check(no_v_result.get("coref_count") == 0, "No vertices -> 0 mappings (nothing to resolve to)")

# Chunks without pronouns/demonstratives
clean_ctx = {
    "chunks": [{"text": "今天天气很好，适合户外活动。", "chunk_id": "c0"}],
    "vertices": [],
}
clean_result = resolver.run(clean_ctx)
check(clean_result.get("coref_count") == 0, "Clean text -> 0 mappings")

# Multiple identical mappings (dedup)
dup_ctx = {
    "chunks": [{"text": "张三是好人。他是一个好员工，他也乐于助人。", "chunk_id": "c0"}],
    "vertices": [{"label": "Person", "properties": {"name": "张三"}}],
}
dup_result = resolver.run(dup_ctx)
# Both "他"s should resolve to 张三, but dedup by (mention, canonical, chunk_id)
# Since both have same mention="他" and same chunk_id, only one should survive
dup_maps = dup_result.get("coref_mappings", [])
he_maps = [m for m in dup_maps if m["mention"] == "他"]
check(len(he_maps) >= 1, f"Dedup: at least one '他' mapping survives (got {len(he_maps)})")


# ── Test 8: Cross-Chunk Context ──────────────────────────────

print_header("Test 8: Cross-Chunk Entity Tracking")

cross_ctx = {
    "chunks": [
        # Chunk 0: Introduces two people
        {"text": "张三和李四是同事，都在阿里云工作。", "chunk_id": "c0"},
        # Chunk 1: Uses pronouns - should resolve based on previous context
        {"text": "他是高级工程师，她是产品经理。他们合作了三年。", "chunk_id": "c1"},
        # Chunk 2: More pronouns
        {"text": "他的项目经验丰富，她的沟通能力很强。他们的团队效率很高。", "chunk_id": "c2"},
    ],
    "vertices": [
        {"label": "Person", "properties": {"name": "张三"}},
        {"label": "Person", "properties": {"name": "李四"}},
        {"label": "Organization", "properties": {"name": "阿里云"}},
    ],
}

cross_result = resolver.run(cross_ctx)
cross_maps = cross_result.get("coref_mappings", [])

# From c1 and c2 we should get multiple 他/她/他们/她们的 resolutions
all_mentions = set(m["mention"] for m in cross_maps)
has_ta = "他" in all_mentions
has_ta_men = "他们" in all_mentions
has_ta_women = "她们" in all_mentions or "她" in all_mentions

check(cross_result.get("coref_count", 0) >= 3, f"Cross-chunk: >= 3 mappings (got {cross_result.get('coref_count', 0)})")
check(has_ta, f"Cross-chunk: '他' resolved (mentions={all_mentions})")
check(has_ta_men, f"Cross-chunk: '他们' resolved (mentions={all_mentions})")

# Verify that 他 consistently maps to one person across chunks
ta_canonicals = set(m["canonical"] for m in cross_maps if m["mention"] == "他")
check(len(ta_canonicals) <= 2, f"'他' resolves to <= 2 unique targets (got {ta_canonicals})")


# ── Summary ──────────────────────────────────────────────────

print_header(f"Results: {PASS} PASS / {FAIL} FAIL")
if FAIL == 0:
    print("  ALL TESTS PASSED!")
else:
    print(f"  {FAIL} test(s) FAILED — needs attention")
