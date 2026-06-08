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

"""Tests for community detection using HugeGraph's own algorithms."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hugegraph_llm.operators.graph_op.community_detect import (
    ALGORITHM_LOUVAIN,
    ALGORITHM_WCC,
    COMMUNITY_ALGORITHMS,
    CommunityDetect,
)
from hugegraph_llm.operators.llm_op.community_report import (
    CommunityReport,
    CommunityReportGenerate,
)
from hugegraph_llm.operators.llm_op.global_search import GlobalSearch


# ── Test Data ─────────────────────────────────────────────────

def _make_test_vertices() -> list:
    return [
        {"id": "1:Alice", "label": "person", "props": {"name": "Alice"}},
        {"id": "2:Bob", "label": "person", "props": {"name": "Bob"}},
        {"id": "3:Carol", "label": "person", "props": {"name": "Carol"}},
        {"id": "4:Inc", "label": "company", "props": {"name": "Acme Inc"}},
        {"id": "5:Dave", "label": "person", "props": {"name": "Dave"}},
        {"id": "6:Eve", "label": "person", "props": {"name": "Eve"}},
        {"id": "7:Corp", "label": "company", "props": {"name": "Beta Corp"}},
    ]


def _make_test_edges() -> list:
    return [
        {"id": "e1", "label": "knows", "outV": "1:Alice", "inV": "2:Bob", "props": {}},
        {"id": "e2", "label": "knows", "outV": "2:Bob", "inV": "3:Carol", "props": {}},
        {"id": "e3", "label": "works_at", "outV": "1:Alice", "inV": "4:Inc", "props": {}},
        {"id": "e4", "label": "works_at", "outV": "2:Bob", "inV": "4:Inc", "props": {}},
        {"id": "e5", "label": "knows", "outV": "5:Dave", "inV": "6:Eve", "props": {}},
        {"id": "e6", "label": "works_at", "outV": "6:Eve", "inV": "7:Corp", "props": {}},
    ]


# ── CommunityDetect: Engine Resolution ────────────────────────


class TestEngineResolution:
    """Tests for engine auto-detection logic."""

    def test_engine_defaults_to_networkx_without_vermeer(self):
        """Without vermeer installed and no client, uses networkx."""
        detector = CommunityDetect(engine="auto")
        assert detector._resolved_engine == "networkx"

    def test_engine_force_networkx(self):
        """Explicit networkx engine."""
        detector = CommunityDetect(engine="networkx")
        assert detector._resolved_engine == "networkx"

    def test_algorithm_constants(self):
        """Test that algorithm constants are correct."""
        assert ALGORITHM_LOUVAIN == "louvain"
        assert ALGORITHM_WCC == "wcc"
        assert "louvain" in COMMUNITY_ALGORITHMS
        assert "wcc" in COMMUNITY_ALGORITHMS


# ── CommunityDetect: Result Parsing ────────────────────────────


class TestResultParsing:
    """Tests for parsing Vermeer and Computer result formats."""

    def test_parse_vermeer_dict_result(self):
        """Parse Vermeer result in {community_id: [vertex_ids]} format."""
        task_data = {
            "params": {
                "result": {
                    "0": ["1:Alice", "2:Bob", "3:Carol"],
                    "1": ["5:Dave", "6:Eve"],
                }
            }
        }
        communities = CommunityDetect._parse_vermeer_result(task_data)
        assert len(communities) == 2
        assert communities[0]["size"] == 3

    def test_parse_vermeer_list_result(self):
        """Parse Vermeer result in [{vertex_id, community_id}] format."""
        task_data = {
            "params": {
                "result": [
                    {"vertex_id": "1:Alice", "community_id": 0},
                    {"vertex_id": "2:Bob", "community_id": 0},
                    {"vertex_id": "5:Dave", "community_id": 1},
                ]
            }
        }
        communities = CommunityDetect._parse_vermeer_result(task_data)
        assert len(communities) == 2

    def test_parse_computer_dict_result(self):
        """Parse Computer result in {community: [vertices]} format."""
        data = {"result": {"0": ["1:Alice", "2:Bob"], "1": ["5:Dave"]}}
        communities = CommunityDetect._parse_computer_result(data)
        assert len(communities) == 2

    def test_parse_computer_list_result(self):
        """Parse Computer result in [{id, community}] format."""
        data = {
            "vertices": [
                {"id": "1:Alice", "community_id": 0},
                {"id": "2:Bob", "community_id": 0},
                {"id": "5:Dave", "community_id": 1},
            ]
        }
        communities = CommunityDetect._parse_computer_result(data)
        assert len(communities) == 2

    def test_parse_result_filters_small_communities(self):
        """Communities smaller than min size are filtered."""
        task_data = {
            "params": {"result": {"0": ["1:Alice"], "1": ["2:Bob", "3:Carol"]}}
        }
        communities = CommunityDetect._parse_vermeer_result(task_data)
        assert len(communities) == 1  # Only the size-2 community survives


# ── CommunityDetect: networkx Fallback ─────────────────────────


class TestNetworkxFallback:
    """Tests for the networkx fallback path."""

    def test_run_with_provided_data(self):
        """Full pipeline with provided vertex/edge data uses networkx Louvain."""
        detector = CommunityDetect(engine="networkx", min_community_size=2)
        result = detector.run({
            "vertices": _make_test_vertices(),
            "edges": _make_test_edges(),
        })
        assert result["community_count"] > 0
        assert result["engine_used"] == "networkx"
        for c in result["communities"]:
            assert "vertices" in c
            assert "size" in c
            assert c["size"] >= 2

    def test_run_with_empty_vertices(self):
        """Graceful handling of empty graph."""
        detector = CommunityDetect(engine="networkx")
        result = detector.run({"vertices": [], "edges": []})
        assert result["communities"] == []
        assert result["community_count"] == 0

    def test_run_enriches_communities(self):
        """Test that communities are enriched with vertex/edge details."""
        detector = CommunityDetect(engine="networkx")
        result = detector.run({
            "vertices": _make_test_vertices(),
            "edges": _make_test_edges(),
        })
        for c in result["communities"]:
            assert "vertex_details" in c
            assert "edge_details" in c
            assert "density" in c
            assert c["density"] >= 0.0


# ── CommunityReport Tests ─────────────────────────────────────


class TestCommunityReport:
    """Tests for the CommunityReport dataclass."""

    def test_to_dict(self):
        report = CommunityReport(
            community_id="L0_C0",
            level=0,
            title="Engineering Team",
            summary="A group of engineers working together.",
            key_entities=["Alice", "Bob", "Carol"],
            relationship_patterns=["coworker", "mentor"],
            importance_score=8.5,
        )
        d = report.to_dict()
        assert d["community_id"] == "L0_C0"
        assert d["importance_score"] == 8.5

    def test_to_text(self):
        report = CommunityReport(
            community_id="L0_C0", level=0,
            title="Test Community", summary="A test.",
            key_entities=["E1", "E2"],
            relationship_patterns=["pattern1"],
            importance_score=7.0,
        )
        text = report.to_text()
        assert "Test Community" in text
        assert "E1, E2" in text
        assert "importance: 7.0" in text


class TestCommunityReportGenerate:
    """Tests for the CommunityReportGenerate operator."""

    def test_fallback_report_without_llm(self):
        reporter = CommunityReportGenerate()
        comm = {
            "id": "L0_C0", "level": 0, "size": 3,
            "vertex_details": [
                {"id": "A", "label": "person", "props": {"name": "Alice"}},
                {"id": "B", "label": "person", "props": {"name": "Bob"}},
            ],
            "edge_details": [], "density": 0.5,
        }
        report = reporter._fallback_report(comm)
        assert "person" in report.title
        assert report.community_id == "L0_C0"

    def test_parse_valid_json(self):
        response = json.dumps({
            "title": "Test", "summary": "A test.",
            "key_entities": ["A"], "relationship_patterns": ["p"],
            "importance_score": 8.0,
        })
        parsed = CommunityReportGenerate._parse_response(response)
        assert parsed["title"] == "Test"

    def test_parse_markdown_code_block(self):
        response = '```json\n{"title": "MD", "summary": "s", "key_entities": [], "relationship_patterns": [], "importance_score": 7}\n```'
        parsed = CommunityReportGenerate._parse_response(response)
        assert parsed["title"] == "MD"

    def test_parse_invalid_json(self):
        parsed = CommunityReportGenerate._parse_response("not json at all")
        assert "title" in parsed
        assert parsed["importance_score"] == 3.0

    def test_run_empty_communities(self):
        reporter = CommunityReportGenerate()
        result = reporter.run({"communities": []})
        assert result["community_reports"] == []


# ── GlobalSearch Tests ────────────────────────────────────────


class TestGlobalSearch:
    """Tests for the GlobalSearch operator."""

    def _make_reports(self) -> list:
        return [
            {
                "community_id": "C0", "level": 0,
                "title": "Engineering",
                "summary": "Alice, Bob, Carol at Acme Inc.",
                "key_entities": ["Alice", "Bob"],
                "relationship_patterns": ["coworker"],
                "importance_score": 9.0,
            },
            {
                "community_id": "C1", "level": 0,
                "title": "External",
                "summary": "Dave and Eve at Beta Corp.",
                "key_entities": ["Dave", "Eve"],
                "relationship_patterns": ["coworker"],
                "importance_score": 5.0,
            },
        ]

    def test_no_reports_returns_message(self):
        searcher = GlobalSearch()
        result = searcher.run({"query": "test", "community_reports": []})
        assert "community reports have not been generated" in result["global_answer"].lower()

    def test_no_llm_returns_fallback(self):
        searcher = GlobalSearch()
        result = searcher.run({
            "query": "What is the structure?",
            "community_reports": self._make_reports(),
        })
        assert result["global_answer"] != ""

    def test_map_selects_top_by_importance(self):
        searcher = GlobalSearch(max_map_communities=1)
        result = searcher.run({
            "query": "test",
            "community_reports": self._make_reports(),
        })
        assert result["communities_used"] <= 1

    def test_parse_findings_format(self):
        response = (
            "Finding: Engineering team is highly connected.\n"
            "Score: 9.0\n\n"
            "Finding: External team is smaller.\n"
            "Score: 4.0\n"
        )
        findings = GlobalSearch._parse_findings(response, "C0", "Engineering")
        assert len(findings) == 2
        assert findings[0]["score"] == 9.0
        assert findings[0]["community_id"] == "C0"

    def test_reduce_fallback_no_llm(self):
        searcher = GlobalSearch()
        findings = [
            {"finding": "Key insight A", "score": 9.0, "community_title": "C1"},
            {"finding": "Key insight B", "score": 7.0, "community_title": "C2"},
        ]
        answer = searcher._reduce_phase("test query", findings)
        assert "Key insight A" in answer
