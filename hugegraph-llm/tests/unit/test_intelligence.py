# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
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

"""Tests for hugegraph_llm.engines.memory.intelligence."""

import pytest

from hugegraph_llm.engines.memory.intelligence import (
    ImportanceEvaluator,
    EbbinghausDecay,
    MemoryOptimizer,
    EntityExtractor,
)


class TestImportanceEvaluator:
    def test_empty_content(self):
        ev = ImportanceEvaluator()
        assert ev.score("") == 0.0

    def test_heuristic_score(self):
        ev = ImportanceEvaluator()
        score = ev.score("I work at Alibaba and my birthday is 1990-05-20.")
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # contains org + date

    def test_short_content_penalty(self):
        ev = ImportanceEvaluator()
        score = ev.score("hi")
        assert score < 0.5

    def test_llm_callback(self):
        ev = ImportanceEvaluator(llm_callback=lambda p: '{"score": 0.9}', use_llm=True)
        assert ev.score("any content") == 0.9

    def test_llm_callback_fallback(self):
        ev = ImportanceEvaluator(llm_callback=lambda p: "score is 0.75", use_llm=True)
        assert ev.score("any content") == 0.75


class TestEbbinghausDecay:
    def test_retention_decays(self):
        decay = EbbinghausDecay(k=0.1, reinforce=0.05)
        ret = decay.retention(1.0, elapsed_hours=24)
        assert 0.0 < ret < 1.0

    def test_reinforcement(self):
        decay = EbbinghausDecay(k=0.1, reinforce=0.05)
        ret = decay.retention(1.0, elapsed_hours=0, access_count=10)
        assert ret == 1.0  # capped at 1.0

    def test_time_to_rehearsal(self):
        decay = EbbinghausDecay(k=0.1, reinforce=0.0)
        hours = decay.time_to_rehearsal(0.8, threshold=0.3)
        assert hours > 0

    def test_already_below_threshold(self):
        decay = EbbinghausDecay(k=0.1, reinforce=0.0)
        assert decay.time_to_rehearsal(0.2, threshold=0.3) == 0.0


class TestMemoryOptimizer:
    def test_content_hash(self):
        opt = MemoryOptimizer()
        assert opt.content_hash("hello") == opt.content_hash("hello")
        assert opt.content_hash("hello") != opt.content_hash("world")

    def test_deduplicate_exact(self):
        opt = MemoryOptimizer()
        memories = [
            {"id": "1", "content": "hello"},
            {"id": "2", "content": "hello"},
            {"id": "3", "content": "world"},
        ]
        kept, dups = opt.deduplicate(memories, strategy="exact")
        assert len(kept) == 2
        assert len(dups) == 1

    def test_deduplicate_semantic(self):
        opt = MemoryOptimizer()
        memories = [
            {"id": "1", "content": "hello, world"},
            {"id": "2", "content": "hello world"},
        ]
        kept, dups = opt.deduplicate(memories, strategy="semantic")
        assert len(kept) == 1

    def test_deduplicate_unknown_strategy(self):
        opt = MemoryOptimizer()
        with pytest.raises(ValueError):
            opt.deduplicate([], strategy="unknown")

    def test_detect_conflict(self):
        opt = MemoryOptimizer()
        conflict = opt.detect_conflict("张三不喜欢咖啡", ["张三喜欢咖啡"])
        assert conflict is not None
        assert conflict["type"] == "contradiction"

    def test_no_conflict(self):
        opt = MemoryOptimizer()
        conflict = opt.detect_conflict("张三喜欢咖啡", ["张三喜欢茶"])
        assert conflict is None


class TestEntityExtractor:
    def test_extract_organizations(self):
        ex = EntityExtractor()
        entities = ex.extract("我在阿里巴巴集团工作")
        assert any(e["type"] == "organization" for e in entities)

    def test_extract_people(self):
        ex = EntityExtractor()
        entities = ex.extract("同事李四是上海人")
        assert any(e["type"] == "person" for e in entities)

    def test_extract_locations(self):
        ex = EntityExtractor()
        entities = ex.extract("我去过北京市")
        assert any(e["type"] == "location" for e in entities)

    def test_llm_callback(self):
        ex = EntityExtractor(llm_callback=lambda p: '{"entities": [{"name": "Alice", "type": "person"}]}')
        entities = ex.extract("Alice works here")
        assert any(e["name"] == "Alice" for e in entities)

    def test_merge_dedupes(self):
        ex = EntityExtractor()
        merged = ex._merge(
            [{"name": "Alice", "type": "person"}],
            [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}],
        )
        assert len(merged) == 2
