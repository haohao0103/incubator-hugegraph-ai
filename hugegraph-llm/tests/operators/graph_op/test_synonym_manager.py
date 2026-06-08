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

"""Tests for synonym manager."""

import tempfile

import pytest

from hugegraph_llm.operators.graph_op.synonym_manager import (
    SynonymGroup,
    SynonymManager,
    SYNONYM_EDGE_LABEL,
)


class TestSynonymGroup:
    """Tests for SynonymGroup dataclass."""

    def test_all_terms(self):
        g = SynonymGroup("syn_1", "canonical", ["alias1", "alias2"])
        assert g.all_terms == ["canonical", "alias1", "alias2"]

    def test_to_dict_roundtrip(self):
        g = SynonymGroup(
            "syn_1",
            "physical car model",
            ["actual car model", "vehicle type"],
            category="business_term",
            metadata={"domain": "logistics"},
        )
        d = g.to_dict()
        assert d["canonical"] == "physical car model"
        assert d["aliases"] == ["actual car model", "vehicle type"]
        assert d["category"] == "business_term"

        restored = SynonymGroup.from_dict(d)
        assert restored.group_id == "syn_1"
        assert restored.canonical == "physical car model"
        assert restored.aliases == ["actual car model", "vehicle type"]


class TestSynonymManager:
    """Tests for SynonymManager."""

    def _make_manager(self) -> SynonymManager:
        sm = SynonymManager()
        sm.add_synonym("物理车型", ["实际车型", "使用车型"])
        sm.add_synonym("司机ID", ["driver_id", "司机编号"])
        sm.add_synonym("订单状态", ["order_status"])
        return sm

    def test_add_synonym(self):
        sm = SynonymManager()
        sm.add_synonym("物理车型", ["实际车型"])
        assert sm.group_count == 1

    def test_empty_manager(self):
        sm = SynonymManager()
        assert sm.group_count == 0
        assert sm.lookup("anything") is None
        assert sm.expand_query("anything") == "anything"

    def test_lookup_canonical(self):
        sm = self._make_manager()
        assert sm.lookup("物理车型") == "物理车型"
        assert sm.lookup("实际车型") == "物理车型"
        assert sm.lookup("使用车型") == "物理车型"

    def test_lookup_case_insensitive(self):
        sm = self._make_manager()
        assert sm.lookup("物理车型") == "物理车型"
        assert sm.lookup("司机id") == "司机ID"

    def test_lookup_not_found(self):
        sm = self._make_manager()
        assert sm.lookup("不存在") is None

    def test_get_group(self):
        sm = self._make_manager()
        group = sm.get_group("物理车型")
        assert group is not None
        assert group.canonical == "物理车型"
        assert "实际车型" in group.aliases
        assert group.category == "general"

    def test_get_group_not_found(self):
        sm = self._make_manager()
        assert sm.get_group("不存在") is None

    def test_add_alias(self):
        sm = self._make_manager()
        result = sm.add_alias("物理车型", "真实车型")
        assert result is True
        assert sm.lookup("真实车型") == "物理车型"

    def test_add_alias_nonexistent(self):
        sm = self._make_manager()
        result = sm.add_alias("不存在", "some alias")
        assert result is False

    def test_remove_group(self):
        sm = self._make_manager()
        assert sm.group_count == 3
        sm.remove_group("物理车型")
        assert sm.group_count == 2
        assert sm.lookup("实际车型") is None

    def test_remove_group_not_found(self):
        sm = self._make_manager()
        result = sm.remove_group("不存在")
        assert result is False

    def test_expand_query(self):
        sm = self._make_manager()
        expanded = sm.expand_query("实际车型的字段在哪")
        assert "物理车型" in expanded
        assert "使用车型" in expanded

    def test_expand_query_no_synonyms(self):
        sm = self._make_manager()
        result = sm.expand_query("不相关的查询")
        assert result == "不相关的查询"

    def test_expand_tokens(self):
        sm = self._make_manager()
        tokens = ["实际车型", "字段"]
        expanded = sm.expand_tokens(tokens)
        assert "物理车型" in expanded
        assert "使用车型" in expanded
        assert "字段" in expanded

    def test_expand_tokens_dedup(self):
        sm = self._make_manager()
        tokens = ["物理车型", "实际车型"]
        expanded = sm.expand_tokens(tokens)
        # No duplicates (both already in group)
        unique = list(set(t for t in expanded))
        assert len(expanded) == len(unique)

    def test_to_graph_edges(self):
        sm = self._make_manager()
        edges = sm.to_graph_edges()
        assert len(edges) == 5  # 2 aliases in group1 + 2 in group2 + 1 in group3
        assert all(e["label"] == SYNONYM_EDGE_LABEL for e in edges)
        canonicals = {e["source"] for e in edges}
        assert "物理车型" in canonicals
        assert "司机ID" in canonicals

    def test_import_from_edges(self):
        sm = SynonymManager()
        edges = [
            {"source": "物理车型", "target": "实际车型", "label": SYNONYM_EDGE_LABEL,
             "properties": {"category": "business_term"}},
            {"source": "物理车型", "target": "使用车型", "label": SYNONYM_EDGE_LABEL,
             "properties": {"category": "business_term"}},
        ]
        count = sm.import_from_edges(edges)
        assert count == 1  # one canonical, two aliases
        assert sm.group_count == 1
        assert sm.lookup("实际车型") == "物理车型"

    def test_import_skips_non_synonym_edges(self):
        sm = SynonymManager()
        edges = [
            {"source": "table1", "target": "table2", "label": "DERIVED_FROM"},
            {"source": "物理车型", "target": "实际车型", "label": SYNONYM_EDGE_LABEL},
        ]
        count = sm.import_from_edges(edges)
        assert count == 1

    def test_persistence(self):
        sm = self._make_manager()
        with tempfile.TemporaryDirectory() as tmpdir:
            import hugegraph_llm.operators.graph_op.synonym_manager as mod
            old_rp = mod.resource_path
            mod.resource_path = tmpdir
            try:
                sm.save("test_graph", "synonyms")
                loaded = SynonymManager.from_saved("test_graph", "synonyms")
                assert loaded.group_count == 3
                assert loaded.lookup("实际车型") == "物理车型"
                assert loaded.lookup("driver_id") == "司机ID"
            finally:
                mod.resource_path = old_rp

    def test_persistence_empty(self):
        loaded = SynonymManager.from_saved("nonexistent", "synonyms")
        assert loaded.group_count == 0

    def test_multiple_groups_same_alias_handling(self):
        """Adding a group whose canonical conflicts with existing alias should warn."""
        sm = SynonymManager()
        sm.add_synonym("termA", ["aliasX"])
        with pytest.raises(ValueError):
            sm.add_synonym("aliasX", ["aliasY"])

    def test_empty_canonical_raises(self):
        sm = SynonymManager()
        with pytest.raises(ValueError):
            sm.add_synonym("", ["alias"])

    def test_empty_aliases_allowed(self):
        sm = SynonymManager()
        g = sm.add_synonym("standalone", [])
        assert g.aliases == []
        assert sm.lookup("standalone") == "standalone"

    def test_category_metadata(self):
        sm = SynonymManager()
        g = sm.add_synonym(
            "物理车型",
            ["实际车型"],
            category="business_term",
            metadata={"domain": "logistics"},
        )
        assert g.category == "business_term"
        assert g.metadata["domain"] == "logistics"
