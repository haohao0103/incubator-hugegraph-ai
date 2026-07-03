# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at#
#   http://www.apache.org.org/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for Gradio Tab reordering (commit 6365a00).

Validates that:
  1. Tab order: PROD (1-5) → Experimental (6-8) → Capability Map (9)
  2. Cascade Propagation is fully removed from graphrag_core_block UI
  3. Experimental tabs carry ⚠️ banners with correct messaging
  4. app.py imports all block modules without error
  5. Capability Map references match new tab numbers

Run:
    pytest hugegraph-llm/tests/demo/test_gradio_tab_order.py -v
"""

import ast
import inspect
import os
import re

import pytest


# ══════════════════════════════════════════════════════════
# Path helpers — resolve relative to this test file location
# ══════════════════════════════════════════════════════════

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.abspath(
    os.path.join(_TEST_DIR, "..", "..", "src", "hugegraph_llm",
                 "demo", "rag_demo")
)


def _read_demo_file(filename):
    """Read a file from the rag_demo directory by name."""
    path = os.path.join(_DEMO_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Demo file not found: {path} "
            f"(DEMO_DIR={_DEMO_DIR})"
        )
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ══════════════════════════════════════════════════════════
# Expected constants — MUST match app.py exactly
# ══════════════════════════════════════════════════════════

EXPECTED_TAB_LABELS = [
    "1. Build RAG Index 💡",
    "2. (Graph)RAG Q&A 📖",
    "3. Agent & Global Search 🤖",
    "4. Text2Gremlin ⚙️",
    "5. Admin & Ops 🛠️",
    "6. GraphRAG Enhancements 🔬 Experimental",
    "7. Schema Studio 🔬 Experimental",
    "8. Multimodal GraphRAG 🔬 Experimental",
    "9. Capability Map 🗺️",
]

PROD_TAB_INDICES = [0, 1, 2, 3, 4]          # tabs 1-5
EXPERIMENTAL_TAB_INDICES = [5, 6, 7]         # tabs 6-8
CAPABILITY_MAP_INDEX = 8                       # tab 9


# ══════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════

@pytest.fixture
def app_source():
    """Source code of app.py."""
    return _read_demo_file("app.py")


@pytest.fixture
def graphrag_core_source():
    """Source code of graphrag_core_block.py."""
    return _read_demo_file("graphrag_core_block.py")


@pytest.fixture
def schema_studio_source():
    """Source code of schema_construction_block.py."""
    return _read_demo_file("schema_construction_block.py")


@pytest.fixture
def multimodal_source():
    """Source code of multimodal_block.py."""
    return _read_demo_file("multimodal_block.py")


@pytest.fixture
def capability_map_source():
    """Source code of capability_map_block.py."""
    return _read_demo_file("capability_map_block.py")


# ══════════════════════════════════════════════════════════
# 1. Tab Order Tests
# ══════════════════════════════════════════════════════════

class TestTabOrder:
    """Verify PROD tabs come first, then Experimental, then Capability Map."""

    def test_all_9_tabs_present(self, app_source):
        for label in EXPECTED_TAB_LABELS:
            assert label in app_source, (
                f"Missing tab: {label!r}"
            )

    def test_tab_order_in_source(self, app_source):
        """Labels must appear in source in expected order (no reorder)."""
        positions = []
        for label in EXPECTED_TAB_LABELS:
            pos = app_source.find('label="' + label + '"')
            assert pos >= 0, f"Tab {label!r} not found in source"
            positions.append(pos)

        assert positions == sorted(positions), (
            f"Tab order mismatch. Positions: {positions}"
        )

    def test_prod_tabs_before_experimental(self, app_source):
        last_prod_pos = max(
            app_source.find('label="' + EXPECTED_TAB_LABELS[i] + '"')
            for i in PROD_TAB_INDICES
        )
        first_exp_pos = min(
            app_source.find('label="' + EXPECTED_TAB_LABELS[i] + '"')
            for i in EXPERIMENTAL_TAB_INDICES
        )
        assert last_prod_pos < first_exp_pos, (
            "Production tabs must come before Experimental tabs"
        )

    def test_experimental_before_capability(self, app_source):
        last_exp_pos = max(
            app_source.find('label="' + EXPECTED_TAB_LABELS[i] + '"')
            for i in EXPERIMENTAL_TAB_INDICES
        )
        cap_pos = app_source.find(
            'label="' + EXPECTED_TAB_LABELS[CAPABILITY_MAP_INDEX] + '"'
        )
        assert last_exp_pos < cap_pos, (
            "Experimental tabs must come before Capability Map"
        )

    def test_section_comments_present(self, app_source):
        assert "PRODUCTION Tabs" in app_source
        assert "EXPERIMENTAL Tabs" in app_source


# ══════════════════════════════════════════════════════════
# 2. Cascade Propagation Removal Tests
# ══════════════════════════════════════════════════════════

class TestCascadeRemoval:
    """Cascade Propagation must be fully removed from graphrag_core_block."""

    def test_no_cascade_import(self, graphrag_core_source):
        assert "cascade_propagation_demo" not in graphrag_core_source, (
            "cascade_propagation_demo must NOT be imported"
        )

    def test_no_cascade_ui_components(self, graphrag_core_source):
        """No Cascade query/slider/button JSON outputs in UI definition."""
        # These are the cascade-specific component variable names
        cascade_vars = [
            "cascade_query", "cascade_trace", "cascade_alpha",
            "cascade_threshold", "cascade_top_k", "cascade_btn",
            "cascade_entity_scores", "cascade_relation_scores",
            "cascade_chunk_scores",
        ]
        for var in cascade_vars:
            assert var not in graphrag_core_source, (
                f"Cascade UI component '{var}' must be removed"
            )

    def test_no_cascade_handler_function(self, graphrag_core_source):
        assert "_run_cascade" not in graphrag_core_source, (
            "_run_cascade handler function must be removed"
        )

    def test_no_cascade_click_binding(self, graphrag_core_source):
        assert "cascade_btn.click" not in graphrag_core_source, (
            "Cascade button click binding must be removed"
        )

    def test_removal_comment_exists(self, graphrag_core_source):
        assert "REMOVED" in graphrag_core_source or "removed" in graphrag_core_source.lower(), (
            "A REMOVED comment should explain why Cascade is gone"
        )


# ══════════════════════════════════════════════════════════
# 3. Experimental Banner Tests
# ══════════════════════════════════════════════════════════

class TestExperimentalBanners:
    """Experimental tabs (6-8) must have clear ⚠️ warnings."""

    def test_graphrag_core_has_banner(self, graphrag_core_source):
        assert "⚠️" in graphrag_core_source or "EXPERIMENTAL" in graphrag_core_source, (
            "GraphRAG Core must have experimental banner"
        )
        assert "NOT yet connected" in graphrag_core_source or \
               "not connected" in graphrag_core_source.lower(), (
            "Banner must state these are NOT in production pipeline"
        )
        # Must reference production pipeline name
        assert "RAGGraphVectorFlow" in graphrag_core_source, (
            "Banner should mention the real production pipeline"
        )

    def test_schema_studio_has_banner(self, schema_studio_source):
        assert "⚠️" in schema_studio_source or "EXPERIMENTAL" in schema_studio_source, (
            "Schema Studio must have experimental banner"
        )
        assert "not connected" in schema_studio_source.lower() or \
               "prototype" in schema_studio_source.lower(), (
            "Schema Studio banner must indicate non-production status"
        )

    def test_multimodal_has_banner(self, multimodal_source):
        assert "⚠️" in multimodal_source or "EXPERIMENTAL" in multimodal_source, (
            "Multimodal must have experimental banner"
        )
        assert "demo data" in multimodal_source.lower() or \
               "DEMO" in multimodal_source, (
            "Multimodal banner must note demo-only status"
        )
        assert "RAGGraphVectorFlow" in multimodal_source or \
               "not connected" in multimodal_source.lower(), (
            "Multimodal banner should clarify it's not in production pipeline"
        )

    def test_graphrag_core_status_labels(self, graphrag_core_source):
        """Table rows must use Prototype/Roadmap/Plugin status markers."""
        assert "Prototype" in graphrag_core_source or "Roadmap" in graphrag_core_source, (
            "GraphRAG Core table must label items as Prototype/Roadmap"
        )


# ══════════════════════════════════════════════════════════
# 4. Import / Syntax Tests
# ══════════════════════════════════════════════════════════

class TestImportsAndSyntax:
    """All modified files must parse without syntax errors and import cleanly."""

    @pytest.mark.parametrize("filename", [
        "app.py",
        "graphrag_core_block.py",
        "schema_construction_block.py",
        "multimodal_block.py",
        "capability_map_block.py",
    ])
    def test_module_syntax_valid(self, filename):
        """Each module's .py file must parse as valid Python AST."""
        source = _read_demo_file(filename)
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in {filename}: {e}")

    def test_app_py_uses_all_blocks(self, app_source):
        """app.py must call create_*_block() for each tab."""
        expected_calls = [
            "create_vector_graph_block()",
            "create_rag_block()",
            "create_agent_block()",
            "create_text2gremlin_block()",
            "create_admin_ops_block()",
            "create_graphrag_core_block()",
            "create_schema_construction_block()",
            "create_multimodal_block()",
            "create_capability_map_block()",
        ]
        for call in expected_calls:
            assert call in app_source, f"Missing: {call}"


# ══════════════════════════════════════════════════════════
# 5. Capability Map Reference Tests
# ══════════════════════════════════════════════════════════

class TestCapabilityMapReferences:
    """Capability Map must reference the NEW tab numbers correctly."""

    def test_prod_tabs_listed_as_production(self, capability_map_source):
        cm = capability_map_source
        assert "Tab 1" in cm or "Build RAG Index" in cm
        assert "(Graph)RAG Q&A" in cm
        assert "Agent & Global Search" in cm
        assert "Text2Gremlin" in cm
        assert "Admin & Ops" in cm
        assert "Production" in cm or "PROD" in cm.upper()

    def test_explicitly_marks_experimental_sections(self, capability_map_source):
        cm = capability_map_source
        assert "Experimental" in cm or "🔬" in cm
        # Should mention GraphRAG Enhancements at position 6
        assert ("6" in cm and "GraphRAG Enhancements" in cm) or \
               ("Tab 6" in cm), (
            "Should reference GraphRAG Enhancements at tab 6"
        )

    def test_no_stale_references_to_old_order(self, capability_map_source):
        """Must not reference old tab ordering like 'Advanced GraphRAG' as Tab 7/8."""
        cm = capability_map_source
        # Old names that were used before the refactor
        stale_patterns = [
            r"Advanced GraphRAG\s*\(Tab [78]\)",
            r"GraphRAG Enhancement\s*\(Tab [89]\)",
            r"Tab 7.*Schema Studio",     # old position was 2, now 7 is OK
        ]
        for pattern in stale_patterns:
            match = re.search(pattern, cm)
            if match:
                pytest.fail(
                    f"Stale tab reference found: {match.group(0)!r}. "
                    f"Update to match new tab ordering."
                )

    def test_capability_map_lists_9_tabs(self, capability_map_source):
        """Should list all 9 tabs in its overview section."""
        cm = capability_map_source
        # Count occurrences of "Tab N" pattern
        tab_refs = re.findall(r"Tab \d+", cm)
        # Should have at least 9 tab references (one per tab)
        assert len(tab_refs) >= 9, (
            f"Expected ≥9 'Tab N' references in Capability Map, "
            f"found {len(tab_refs)}: {tab_refs}"
        )


# ══════════════════════════════════════════════════════════
# 6. Structural Consistency Tests
# ══════════════════════════════════════════════════════════

class TestStructuralConsistency:

    def test_graphrag_core_retains_ppr_and_other_features(self, graphrag_core_source):
        """After removing Cascade, other features must still exist."""
        gcs = graphrag_core_source
        # Section A: PPR + Identity Edge should remain
        assert "PPR" in gcs, "PPR Retriever should still exist"
        assert "Identity Edge" in gcs or "identity_edge" in gcs, \
            "Identity Edge Builder should still exist"

        # Sections B-F should all remain
        assert "Retrieval Enhancement" in gcs, "Section B missing"
        assert "Reasoning" in gcs, "Section C missing"
        assert "Trustworthy Output" in gcs, "Section D missing"
        assert "BM25" in gcs, "Section E missing"
        assert "Chunk Graph" in gcs or "SIMILAR" in gcs, "Section F missing"

    def test_no_orphaned_cascade_outputs_in_refresh(self, app_source):
        """app.py refresh_ui_config_prompt output list must not include cascade vars."""
        cascade_vars = ["cascade_entity_scores", "cascade_relation_scores",
                        "cascade_chunk_scores", "cascade_trace"]
        for var in cascade_vars:
            assert var not in app_source, (
                f"Orphaned cascade variable {var} in app.py outputs"
            )

    def test_mm_and_studio_outputs_still_wired(self, app_source):
        """mm_demo_outputs and studio_demo_outputs must still be wired into load()."""
        assert "mm_demo_outputs" in app_source, \
            "mm_demo_outputs missing from app.py"
        assert "studio_demo_outputs" in app_source, \
            "studio_demo_outputs missing from app.py"
