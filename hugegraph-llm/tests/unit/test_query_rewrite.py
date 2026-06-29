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

"""Tests for hugegraph_llm.engines.memory.query_rewrite."""

import pytest

from hugegraph_llm.engines.memory.query_rewrite import QueryRewriteEngine, rewrite_query


class TestQueryRewriteEngine:
    def test_empty_query(self):
        engine = QueryRewriteEngine()
        assert engine.rewrite("  ") == ""
        assert engine.variants("") == [""]

    def test_alias_expansion(self):
        engine = QueryRewriteEngine(aliases={"tx": "Tencent", "腾讯": "Tencent"})
        assert engine.rewrite("tx 在哪里上班？") == "Tencent 在哪里上班？"
        assert engine.rewrite("腾讯总部在哪里？") == "Tencent总部在哪里？"

    def test_pronoun_resolution_with_profile(self):
        engine = QueryRewriteEngine(user_profile="User is 张三 who works at Alibaba.")
        rewritten = engine.rewrite("他在哪里上班？")
        assert "他" not in rewritten
        assert "User is" in rewritten or "张三" in rewritten

    def test_pronoun_resolution_without_profile(self):
        engine = QueryRewriteEngine()
        assert engine.rewrite("他在哪里上班？") == "他在哪里上班？"

    def test_variants_and_keyword_query(self):
        engine = QueryRewriteEngine()
        result = engine.expand_query("张三在哪里工作？")
        assert result["original"] == "张三在哪里工作？"
        assert result["rewritten"] == "张三在哪里工作？"
        assert len(result["variants"]) >= 1
        assert "keyword_query" in result

    def test_factory_function(self):
        result = rewrite_query(
            "他在哪里上班？",
            aliases={"他": "张三"},
            user_profile="张三 works at Huawei.",
        )
        assert result["original"] == "他在哪里上班？"
        assert "张三" in result["rewritten"]

    def test_tokenization(self):
        engine = QueryRewriteEngine()
        tokens = engine._tokenize("张三在哪里上班？")
        assert "张三" in tokens
        assert "哪里" not in tokens
        assert "在" not in tokens


class TestQueryRewriteEdgeCases:
    def test_mixed_language_profile(self):
        engine = QueryRewriteEngine(user_profile="User works at ByteDance, 北京市")
        entities = engine._profile_entities
        assert any("ByteDance" in e for e in entities)
        assert any("北京" in e for e in entities)

    def test_only_particles(self):
        engine = QueryRewriteEngine()
        result = engine.expand_query("的")
        assert result["original"] == "的"
