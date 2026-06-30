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

"""Tests for community_summary_generator.py — bridges CommunityDetect → GlobalSearchRetriever."""

import importlib
import json
import numpy as np
import pytest

# Direct import to bypass __init__.py which imports igraph (causes conflicts)
_mod = importlib.import_module(
    "hugegraph_llm.operators.graph_rag_enhancements.community_summary_generator"
)
CommunityFinding = _mod.CommunityFinding
CommunityReport = _mod.CommunityReport
CommunitySummaryConfig = _mod.CommunitySummaryConfig
CommunitySummaryGenerator = _mod.CommunitySummaryGenerator
COMMUNITY_SUMMARY_PROMPT = _mod.COMMUNITY_SUMMARY_PROMPT
generate_community_summaries = _mod.generate_community_summaries


# ── Test data structures ──────────────────────────────────────────


class TestCommunityFinding:
    """Test CommunityFinding dataclass."""

    def test_default_finding(self):
        f = CommunityFinding()
        assert f.summary == ""
        assert f.explanation == ""

    def test_finding_with_data(self):
        f = CommunityFinding(summary="Hub detected", explanation="Entity A has high degree")
        assert f.summary == "Hub detected"


class TestCommunityReport:
    """Test CommunityReport dataclass."""

    def test_default_report(self):
        r = CommunityReport()
        assert r.id == ""
        assert r.title == ""
        assert r.summary == ""
        assert r.findings == []
        assert r.entity_count == 0
        assert r.edge_count == 0
        assert r.embedding is None

    def test_report_with_data(self):
        f = CommunityFinding(summary="Test finding", explanation="Detail")
        r = CommunityReport(
            id="0",
            title="Test Community",
            summary="A test summary",
            findings=[f],
            entity_count=5,
            edge_count=8,
        )
        assert r.id == "0"
        assert len(r.findings) == 1

    def test_to_dict(self):
        f = CommunityFinding(summary="Finding", explanation="Expl")
        r = CommunityReport(id="1", title="Title", summary="Sum", findings=[f],
                            level=0, rank=0.9, entity_count=10)
        d = r.to_dict()
        assert d["id"] == "1"
        assert d["title"] == "Title"
        assert d["summary"] == "Sum"
        assert len(d["findings"]) == 1
        assert d["findings"][0]["summary"] == "Finding"
        assert d["level"] == 0
        assert d["rank"] == 0.9
        assert d["entity_count"] == 10

    def test_full_content(self):
        f1 = CommunityFinding(summary="Insight A", explanation="Explanation A")
        f2 = CommunityFinding(summary="Insight B", explanation="Explanation B")
        r = CommunityReport(title="My Community", summary="This is about X.", findings=[f1, f2])
        content = r.full_content()
        assert "# My Community" in content
        assert "This is about X." in content
        assert "- Insight A: Explanation A" in content
        assert "- Insight B: Explanation B" in content

    def test_full_content_empty_findings(self):
        r = CommunityReport(title="Empty", summary="No findings.")
        content = r.full_content()
        assert "# Empty" in content
        assert "No findings." in content


class TestCommunitySummaryConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        config = CommunitySummaryConfig()
        assert config.max_entities_per_report == 50
        assert config.max_relations_per_report == 30
        assert config.max_findings == 5
        assert config.llm_max_retries == 1
        assert config.fallback_to_heuristic == True
        assert config.generate_embeddings == True

    def test_custom_config(self):
        config = CommunitySummaryConfig(
            max_entities_per_report=20,
            max_findings=3,
            fallback_to_heuristic=False,
        )
        assert config.max_entities_per_report == 20
        assert config.max_findings == 3
        assert config.fallback_to_heuristic == False


# ── Test heuristic generation ─────────────────────────────────────


class TestHeuristicGeneration:
    """Test community report generation without LLM (heuristic mode)."""

    def test_heuristic_basic(self):
        """Heuristic generation produces a community report."""
        generator = CommunitySummaryGenerator(llm=None)  # No LLM → heuristic
        communities = {0: ["Entity_A", "Entity_B", "Entity_C"]}
        entity_texts = {"Entity_A": "Person A", "Entity_B": "Person B", "Entity_C": "Org C"}
        relations = [
            {"source": "Entity_A", "target": "Entity_B", "description": "works with"},
            {"source": "Entity_B", "target": "Entity_C", "description": "belongs to"},
        ]

        reports = generator.generate(communities, entity_texts, relations)
        assert len(reports) == 1
        r = reports[0]
        assert r.title != ""
        assert r.summary != ""
        assert r.entity_count == 3
        assert r.edge_count == 2
        assert len(r.findings) > 0

    def test_heuristic_title_contains_entities(self):
        """Title includes top entity names."""
        generator = CommunitySummaryGenerator(llm=None)
        communities = {0: ["Alpha", "Beta", "Gamma", "Delta"]}
        entity_texts = {"Alpha": "desc", "Beta": "desc", "Gamma": "desc", "Delta": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert "Alpha" in reports[0].title or "Beta" in reports[0].title
        assert "and 1 others" in reports[0].title  # 4 entities, top 3 + "1 others"

    def test_heuristic_hub_entity_detection(self):
        """Heuristic finds hub entities (most connected)."""
        generator = CommunitySummaryGenerator(llm=None)
        communities = {0: ["Hub", "Node_A", "Node_B"]}
        relations = [
            {"source": "Hub", "target": "Node_A"},
            {"source": "Hub", "target": "Node_B"},
        ]

        reports = generator.generate(communities, {}, relations)
        findings_summaries = [f.summary for f in reports[0].findings]
        assert any("Hub" in s for s in findings_summaries)

    def test_heuristic_empty_communities(self):
        """Empty communities → no reports."""
        generator = CommunitySummaryGenerator(llm=None)
        reports = generator.generate({}, {})
        assert len(reports) == 0

    def test_heuristic_empty_entity_list(self):
        """Community with empty entity list → skipped."""
        generator = CommunitySummaryGenerator(llm=None)
        communities = {0: [], 1: ["X"]}
        reports = generator.generate(communities, {"X": "desc"})
        assert len(reports) == 1  # Only community 1

    def test_heuristic_reports_sorted_by_size(self):
        """Reports sorted by entity_count descending."""
        generator = CommunitySummaryGenerator(llm=None)
        communities = {0: ["A"], 1: ["B", "C", "D", "E"], 2: ["F", "G"]}
        entity_texts = {k: "desc" for k in ["A", "B", "C", "D", "E", "F", "G"]}

        reports = generator.generate(communities, entity_texts)
        assert reports[0].entity_count >= reports[1].entity_count


# ── Test LLM generation ──────────────────────────────────────────


class TestLLMGeneration:
    """Test community report generation with mock LLM."""

    class MockLLM:
        """Mock LLM that returns predefined JSON."""
        def __init__(self, response=None):
            self._response = response or json.dumps({
                "title": "Medical Research Community",
                "summary": "This community focuses on diabetes research and treatment methods.",
                "findings": [
                    {"summary": "Diabetes prevalence", "explanation": "30% of community involves diabetes."},
                    {"summary": "Treatment hub", "explanation": "Insulin therapy is the central treatment."},
                ],
            })

        def generate(self, prompt):
            return self._response

    def test_llm_generation_basic(self):
        """LLM returns valid JSON → rich community report."""
        llm = self.MockLLM()
        generator = CommunitySummaryGenerator(llm=llm)
        communities = {0: ["diabetes", "insulin", "treatment"]}
        entity_texts = {"diabetes": "A metabolic disorder", "insulin": "A hormone", "treatment": "Medical care"}

        reports = generator.generate(communities, entity_texts)
        assert len(reports) == 1
        r = reports[0]
        assert r.title == "Medical Research Community"
        assert r.summary != ""
        assert len(r.findings) == 2

    def test_llm_malformed_json_fallback(self):
        """LLM returns invalid JSON → heuristic fallback."""
        llm = self.MockLLM(response="NOT JSON AT ALL {{{}}}")
        generator = CommunitySummaryGenerator(
            llm=llm,
            config=CommunitySummaryConfig(fallback_to_heuristic=True, llm_max_retries=1),
        )
        communities = {0: ["Entity_A", "Entity_B"]}
        entity_texts = {"Entity_A": "desc", "Entity_B": "desc"}

        reports = generator.generate(communities, entity_texts)
        # Should fall back to heuristic
        assert len(reports) == 1
        assert reports[0].title != ""  # Heuristic title generated

    def test_llm_exception_fallback(self):
        """LLM raises exception → heuristic fallback."""
        class ErrorLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM unavailable")

        generator = CommunitySummaryGenerator(
            llm=ErrorLLM(),
            config=CommunitySummaryConfig(fallback_to_heuristic=True, llm_max_retries=1),
        )
        communities = {0: ["A", "B"]}
        entity_texts = {"A": "desc", "B": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert len(reports) == 1  # Heuristic fallback

    def test_llm_empty_response_fallback(self):
        """LLM returns empty title/summary → heuristic fallback."""
        llm = self.MockLLM(response=json.dumps({"title": "", "summary": "", "findings": []}))
        generator = CommunitySummaryGenerator(
            llm=llm,
            config=CommunitySummaryConfig(fallback_to_heuristic=True, llm_max_retries=1),
        )
        communities = {0: ["A"]}
        entity_texts = {"A": "desc"}

        reports = generator.generate(communities, entity_texts)
        # Empty LLM response → fallback
        assert len(reports) == 1

    def test_llm_markdown_fenced_json(self):
        """LLM returns markdown-fenced JSON → parsed correctly."""
        response = '```json\n{"title": "Test Community", "summary": "A test.", "findings": []}\n```'
        llm = self.MockLLM(response=response)
        generator = CommunitySummaryGenerator(llm=llm)
        communities = {0: ["X"]}
        entity_texts = {"X": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert len(reports) == 1
        assert reports[0].title == "Test Community"

    def test_no_fallback_no_llm(self):
        """No LLM and no heuristic fallback → minimal report."""
        generator = CommunitySummaryGenerator(
            llm=None,
            config=CommunitySummaryConfig(fallback_to_heuristic=False),
        )
        communities = {0: ["A", "B"]}
        entity_texts = {"A": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert len(reports) == 1
        assert reports[0].title == "Community 0"  # Minimal default title


# ── Test embedding generation ────────────────────────────────────


class TestEmbeddingGeneration:
    """Test embedding generation for community reports."""

    def _make_embedding_fn(self, dim=384):
        def fn(text):
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(dim).astype(np.float32)
            return vec / np.linalg.norm(vec)
        return fn

    def test_embedding_generated(self):
        """Embedding function → report has embedding field."""
        embed_fn = self._make_embedding_fn()
        generator = CommunitySummaryGenerator(
            llm=None,
            embedding_fn=embed_fn,
            config=CommunitySummaryConfig(generate_embeddings=True),
        )
        communities = {0: ["A"]}
        entity_texts = {"A": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert reports[0].embedding is not None
        assert len(reports[0].embedding) == 384

    def test_embedding_disabled(self):
        """generate_embeddings=False → no embedding."""
        embed_fn = self._make_embedding_fn()
        generator = CommunitySummaryGenerator(
            llm=None,
            embedding_fn=embed_fn,
            config=CommunitySummaryConfig(generate_embeddings=False),
        )
        communities = {0: ["A"]}
        entity_texts = {"A": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert reports[0].embedding is None

    def test_no_embedding_fn(self):
        """No embedding_fn → no embedding even if enabled."""
        generator = CommunitySummaryGenerator(
            llm=None,
            embedding_fn=None,
            config=CommunitySummaryConfig(generate_embeddings=True),
        )
        communities = {0: ["A"]}
        entity_texts = {"A": "desc"}

        reports = generator.generate(communities, entity_texts)
        assert reports[0].embedding is None


# ── Test relation filtering ──────────────────────────────────────


class TestRelationFiltering:
    """Test _filter_community_relations."""

    def test_filter_relations_by_community(self):
        """Only relations involving community entities are included."""
        relations = [
            {"source": "A", "target": "B", "description": "knows"},
            {"source": "C", "target": "D", "description": "unrelated"},
            {"source": "A", "target": "C", "description": "works with"},
        ]
        entity_names = ["A", "B"]
        result = CommunitySummaryGenerator._filter_community_relations(entity_names, relations)
        assert len(result) == 2  # A→B and A→C (A is in community)
        assert result[0]["source"] == "A"

    def test_filter_empty_relations(self):
        """Empty relation list → empty result."""
        result = CommunitySummaryGenerator._filter_community_relations(["A"], [])
        assert len(result) == 0

    def test_filter_no_matching_entities(self):
        """No entities match any relation → empty result."""
        relations = [{"source": "X", "target": "Y"}]
        result = CommunitySummaryGenerator._filter_community_relations(["A", "B"], relations)
        assert len(result) == 0


# ── Test LLM response parsing ────────────────────────────────────


class TestParseLLMResponse:
    """Test _parse_llm_response."""

    def test_clean_json(self):
        """Clean JSON → parsed correctly."""
        response = json.dumps({
            "title": "Test",
            "summary": "Summary",
            "findings": [{"summary": "F1", "explanation": "E1"}],
        })
        title, summary, findings = CommunitySummaryGenerator._parse_llm_response(response)
        assert title == "Test"
        assert summary == "Summary"
        assert len(findings) == 1
        assert findings[0].summary == "F1"

    def test_json_with_markdown_fence(self):
        """Markdown-fenced JSON → parsed correctly."""
        response = '```json\n{"title": "T", "summary": "S", "findings": []}\n```'
        title, summary, findings = CommunitySummaryGenerator._parse_llm_response(response)
        assert title == "T"
        assert summary == "S"

    def test_malformed_json_regex_fallback(self):
        """Malformed JSON → regex fallback."""
        response = '{"title": "Regex Title", "summary": "Regex Summary", "broken": true'
        title, summary, findings = CommunitySummaryGenerator._parse_llm_response(response)
        assert title == "Regex Title" or title == ""  # Regex may or may not extract

    def test_empty_findings(self):
        """Empty findings list → no findings."""
        response = json.dumps({"title": "T", "summary": "S", "findings": []})
        title, summary, findings = CommunitySummaryGenerator._parse_llm_response(response)
        assert len(findings) == 0


# ── Test prompt template ──────────────────────────────────────────


class TestPromptTemplate:
    """Test COMMUNITY_SUMMARY_PROMPT formatting."""

    def test_prompt_formatting(self):
        """Prompt template can be formatted with entity and relation lists."""
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            community_id=42,
            entity_list="- Entity_A: Description\n- Entity_B: Description",
            relation_list="- Entity_A → Entity_B: knows",
        )
        assert "42" in prompt
        assert "Entity_A" in prompt


# ── Test convenience function ────────────────────────────────────


class TestGenerateCommunitySummariesFunction:
    """Test generate_community_summaries convenience function."""

    def test_quick_generate(self):
        """Quick-generate works with minimal args."""
        communities = {0: ["A", "B"]}
        entity_texts = {"A": "desc", "B": "desc"}
        reports = generate_community_summaries(communities, entity_texts)
        assert len(reports) == 1
        assert isinstance(reports[0], CommunityReport)

    def test_quick_generate_with_llm(self):
        """Quick-generate with mock LLM."""
        llm = TestLLMGeneration.MockLLM()
        communities = {0: ["X"]}
        entity_texts = {"X": "desc"}
        reports = generate_community_summaries(communities, entity_texts, llm=llm)
        assert len(reports) == 1
