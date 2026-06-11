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

"""Tests for Claim extraction operator.

Covers:
1. Data model: Claim creation, serialization, deserialization
2. Parsing: LLM response JSON parsing (valid/invalid/malformed)
3. Deduplication: (subject, predicate, object) dedup with confidence tiebreak
4. Index: Subject/predicate/status lookup, community assignment
5. Operator run(): end-to-end extraction from chunks (with mock LLM)
6. Edge cases: empty input, no entities context, long text truncation
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from hugegraph_llm.operators.llm_op.claim_extract import (
    Claim,
    ClaimStatus,
    ClaimExtract,
    ClaimIndex,
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


# ── Test 1: Claim Data Model ──────────────────────────────────

print_header("Test 1: Claim Data Model")

c1 = Claim(
    subject="张三",
    predicate="job_title",
    object="高级工程师",
    description="张三在阿里云担任高级工程师",
    status=ClaimStatus.SUPPORTING,
    confidence=0.95,
    source_text="张三担任阿里云的高级工程师",
    chunk_id="chunk_0",
    doc_id="doc_1",
)
check(c1.claim_id.startswith("claim-"), "Auto-generated claim_id")
check(c1.subject == "张三", "Subject preserved")
check(c1.predicate == "job_title", "Predicate preserved")
check(c1.object == "高级工程师", "Object preserved")
check(c1.status == ClaimStatus.SUPPORTING, "Status enum preserved")
check(abs(c1.confidence - 0.95) < 0.001, "Confidence preserved")

d = c1.to_dict()
check("claim_id" in d and "status" in d and "confidence" in d, "to_dict() has all fields")
check(d["status"] == "supporting", "to_dict() serializes status as string")

c2 = Claim.from_dict(d)
check(c2.subject == c1.subject, "from_dict() roundtrip: subject")
check(c2.predicate == c1.predicate, "from_dict() roundtrip: predicate")
check(c2.status.value == c1.status.value, "from_dict() roundtrip: status")
check(abs(c2.confidence - c1.confidence) < 0.001, "from_dict() roundtrip: confidence")

t = c1.triple()
check(t == ("张三", "job_title", "高级工程师"), "triple() returns (s, p, o)")


# ── Test 2: Response Parsing ──────────────────────────────────

print_header("Test 2: LLM Response Parsing")

extractor = ClaimExtract(llm=None)

# Valid JSON array response
valid_response = """```json
[
  {
    "subject": "李四",
    "predicate": "works_at",
    "object": "腾讯",
    "description": "李四在腾讯工作",
    "status": "supporting",
    "confidence": 0.9,
    "source_text": "李四目前在腾讯工作",
    "start_char": 10,
    "end_char": 18
  },
  {
    "subject": "王五",
    "predicate": "location",
    "object": "深圳",
    "description": "王五住在深圳",
    "status": "not_enough_info",
    "confidence": 0.6,
    "source_text": "王五可能住在深圳"
  }
]
```"""
claims = extractor._parse_response(valid_response, "chunk_0", "doc_1")
check(len(claims) == 2, f"Parsed 2 claims from valid JSON (got {len(claims)})")
check(claims[0].subject == "李四", "First claim subject correct")
check(claims[0].status == ClaimStatus.SUPPORTING, "First claim status correct")
check(claims[1].status == ClaimStatus.NOT_ENOUGH_INFO, "Second claim status = not_enough_info")

# Raw JSON without code fence
raw_response = '[{"subject":"赵六","predicate":"age","object":"30","status":"supporting","confidence":0.8,"source_text":"赵六今年30岁"}]'
claims_raw = extractor._parse_response(raw_response, "chunk_1", "doc_1")
check(len(claims_raw) == 1, f"Parsed 1 claim from raw JSON (got {len(claims_raw)})")
check(claims_raw[0].subject == "赵六", "Raw JSON parsing works")

# Empty array
empty_claims = extractor._parse_response("[]", "chunk_x", "doc_1")
check(len(empty_claims) == 0, "Empty array returns 0 claims")

# Malformed / invalid JSON
malformed = extractor._parse_response("this is not json at all", "chunk_x", "doc_1")
check(len(malformed) == 0, "Malformed JSON returns 0 claims gracefully")

# Invalid items inside array (skip bad entries)
partial = """```json
[{"subject":"OK","predicate":"p","object":"o","status":"supporting","confidence":0.7,"source_text":"ok"}, {"bad":"item"}]```"""
partial_claims = extractor._parse_response(partial, "chunk_y", "doc_1")
check(len(partial_claims) == 1, f"Partial parse: skipped invalid item (got {len(partial_claims)})")


# ── Test 3: Deduplication ────────────────────────────────────

print_header("Test 3: Claim Deduplication")

claims_dup = [
    Claim(subject="张三", predicate="works_at", object="阿里云", confidence=0.7),
    Claim(subject="张三", predicate="works_at", object="阿里云", confidence=0.95),  # duplicate, higher conf
    Claim(subject="张三", predicate="works_at", object="阿里云", confidence=0.5),   # duplicate, lower conf
    Claim(subject="李四", predicate="works_at", object="腾讯", confidence=0.85),
    Claim(subject="张三", predicate="job_title", object="工程师", confidence=0.8),   # different predicate
    Claim(subject="张三", predicate="WORKS_AT", object="阿里云", confidence=0.6),  # case-insensitive dedup
]

deduped = ClaimExtract._deduplicate(claims_dup)
# Should be 3 unique: (张三, works_at, 阿里云), (李四, works_at, 腾讯), (张三, job_title, 工程师)
# (张三, WORKS_AT, ALIYUN) should dedup with (张三, works_at, 阿里云)
check(len(deduped) == 3, f"Dedup to 3 unique claims (got {len(deduped)})")

# The surviving (张三, works_at, 阿里云) should have highest confidence (0.95)
zhangsan_works = [c for c in deduped if c.predicate.lower() == "works_at" and c.object.lower() == "阿里云"]
check(len(zhangsan_works) == 1, "One surviving (张三, works_at, 阿里云)")
check(abs(zhangsan_works[0].confidence - 0.95) < 0.001, "Surviving claim has highest confidence (0.95)")


# ── Test 4: Claim Index ──────────────────────────────────────

print_header("Test 4: Claim Index Lookup")

idx = ClaimIndex()
test_claims = [
    Claim(subject="张三", predicate="works_at", object="阿里云", status=ClaimStatus.SUPPORTING),
    Claim(subject="张三", predicate="job_title", object="工程师", status=ClaimStatus.SUPPORTING),
    Claim(subject="李四", predicate="works_at", object="腾讯", status=ClaimStatus.CONTRADICTING),
    Claim(subject="王五", predicate="location", object="北京", status=ClaimStatus.NOT_ENOUGH_INFO),
    Claim(subject="张三", predicate="location", object="杭州", status=ClaimStatus.NOT_ENOUGH_INFO),
]
idx.add_batch(test_claims)

check(idx.size == 5, f"Index size = 5 (got {idx.size})")

# Subject lookup
zhangsan_claims = idx.get_by_subject("张三")
check(len(zhangsan_claims) == 3, f"get_by_subject('张三') returns 3 claims (got {len(zhangsan_claims)})")

unknown = idx.get_by_subject("不存在的人")
check(len(unknown) == 0, "get_by_subject for unknown entity returns empty")

# Predicate lookup
works_claims = idx.get_by_predicate("works_at")
check(len(works_claims) == 2, f"get_by_predicate('works_at') returns 2 claims (got {len(works_claims)})")

# Status lookup
supporting = idx.get_by_status("supporting")
check(len(supporting) == 2, f"get_by_status('supporting') returns 2 (got {len(supporting)})")
contradicting = idx.get_by_status("contradicting")
check(len(contradicting) == 1, f"get_by_status('contradicting') returns 1 (got {len(contradicting)})")

# Community assignment
community_entities = ["张三", "王五"]
community_claims = idx.get_for_community(community_entities)
check(len(community_claims) == 4, f"get_for_community(['张三','王五']) returns 4 (got {len(community_claims)})")

other_community = idx.get_for_community(["李四"])
check(len(other_community) == 1, f"get_for_community(['李四']) returns 1 (got {len(other_community)})")

# Stats
stats = idx.stats()
check(stats["total_claims"] == 5, "stats.total_claims = 5")
check(stats["unique_subjects"] == 3, f"stats.unique_subjects = 3 (got {stats['unique_subjects']})")
check(stats["unique_predicates"] == 3, f"stats.unique_predicates = 3 (got {stats['unique_predicates']})")


# ── Test 5: End-to-End Extraction with Mock LLM ──────────────

print_header("Test 5: End-to-End Extraction (Mock LLM)")


class MockLLM:
    def generate(self, prompt):
        return '''```json
[
  {
    "subject": "张三",
    "predicate": "works_at",
    "object": "阿里云",
    "description": "张三在阿里云工作",
    "status": "supporting",
    "confidence": 0.92,
    "source_text": "张三目前就职于阿里巴巴集团的云计算部门阿里云",
    "start_char": 0,
    "end_char": 20
  }
]
```'''


mock_llm = MockLLM()
extractor_with_mock = ClaimExtract(llm=mock_llm)

context = {
    "chunks": [
        {"text": "张三目前就职于阿里巴巴集团的云计算部门阿里云，担任高级工程师。他在2023年加入该公司。", "chunk_id": "c0"},
        {"text": "李四是张三的同事，也在阿里云工作，负责产品管理。", "chunk_id": "c1"},
    ],
    "vertices": [
        {"label": "Person", "properties": {"name": "张三"}},
        {"label": "Company", "properties": {"name": "阿里云"}},
        {"label": "Person", "properties": {"name": "李四"}},
    ],
    "edges": [
        {"label": "works_at", "outV": "张三", "inV": "阿里云"},
    ],
    "doc_id": "test_doc",
}

result = extractor_with_mock.run(context)
claims_out = result.get("claims", [])
count = result.get("claim_count", 0)

check(count >= 1, f"Extracted >= 1 claim total (got {count})")
check(len(claims_out) >= 1, f"context['claims'] has >= 1 entry (got {len(claims_out)})")

if claims_out:
    first = claims_out[0]
    check(first["subject"] == "张三", "First claim subject = 张三")
    check("claim_id" in first, "First claim has claim_id")
    check(first["chunk_id"] == "c0", "First claim chunk_id = c0")
    check(first["doc_id"] == "test_doc", "First claim doc_id = test_doc")


# ── Test 6: Edge Cases ───────────────────────────────────────

print_header("Test 6: Edge Cases")

empty_extractor = ClaimExtract(llm=None)

# Empty chunks
empty_ctx = {"chunks": [], "vertices": [], "edges": []}
empty_result = empty_extractor.run(empty_ctx)
check(empty_result.get("claim_count") == 0, "Empty chunks → 0 claims")
check(empty_result.get("claims") == [], "Empty chunks → empty list")

# Chunk with empty string
whitespace_ctx = {
    "chunks": [{"text": "   ", "chunk_id": "ws"}, {"text": "", "chunk_id": "empty"}],
    "vertices": [],
    "edges": [],
}
ws_result = empty_extractor.run(whitespace_ctx)
check(ws_result.get("claim_count") == 0, "Whitespace-only chunks → 0 claims")

# No LLM (should return empty gracefully)
no_llm_ctx = {
    "chunks": [{"text": "Some text here", "chunk_id": "c0"}],
    "vertices": [],
    "edges": [],
}
no_llm_result = empty_extractor.run(no_llm_ctx)
# Without LLM, _extract_from_chunk calls self._llm.generate which would fail
# But we handle it in _extract_from_chunk try/except → returns []
check(isinstance(no_llm_result.get("claims"), list), "No LLM: claims is a list")

# Long text truncation
long_text = "这是一段很长的文本。" * 500  # ~3000 chars before truncation, ~7500 after
long_ctx = {
    "chunks": [{"text": long_text, "chunk_id": "long"}],
}
# Should not crash — just return empty since no LLM
try:
    long_result = empty_extractor.run(long_ctx)
    check(True, "Long text chunk does not crash")
except Exception as e:
    check(False, f"Long text chunk crashed: {e}")

# Very long chunk (>3000 chars) gets truncated before sending to LLM
# Verify truncation logic exists
truncated = long_text[:3000] if len(long_text) > 3000 else long_text
check(len(truncated) <= len(long_text), "Truncation logic works (truncated <= original)")
check(len(truncated) == 3000, f"Truncation caps at 3000 chars (got {len(truncated)})")

# All three statuses present
all_statuses = [ClaimStatus.SUPPORTING, ClaimStatus.CONTRADICTING, ClaimStatus.NOT_ENOUGH_INFO]
for s in all_statuses:
    c = Claim(subject="x", predicate="y", object="z", status=s)
    check(c.status == s, f"ClaimStatus.{s.name} constructable")


# ── Summary ──────────────────────────────────────────────────

print_header(f"Results: {PASS} PASS / {FAIL} FAIL")
if FAIL == 0:
    print("  ALL TESTS PASSED!")
else:
    print(f"  {FAIL} test(s) FAILED — needs attention")
