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

"""Tests for the Skill Graph engine (L0-L3)."""

import importlib.util
import json
from pathlib import Path

import pytest

from hugegraph_llm.engines.skill_graph import (
    EdgeType,
    GraphBuilder,
    HashingEmbedder,
    NodeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
    SkillRetriever,
    load_skills,
    skill_id,
)
from hugegraph_llm.engines.skill_graph.cli import main as skill_cli_main
from hugegraph_llm.engines.skill_graph.schema import category_id

SKILLS = {
    "data-ingestion": {
        "description": "Load and normalize raw documents",
        "category": "data",
        "relations": [],
    },
    "text-chunking": {
        "description": "Split documents into chunks",
        "category": "data",
        "relations": [("depends_on", "data-ingestion")],
    },
    "vector-index": {
        "description": "Build and query an ANN index",
        "category": "retrieval",
        "relations": [("depends_on", "text-chunking")],
    },
    "rag-pipeline": {
        "description": "Compose retrieval and generation into a RAG flow",
        "category": "generation",
        "relations": [("depends_on", "vector-index"), ("depends_on", "text-chunking")],
    },
    "legacy-search": {
        "description": "Deprecated keyword-only search engine",
        "category": "retrieval",
        "relations": [],
    },
    "prompt-engineering": {
        "description": "Author robust LLM prompts",
        "category": "generation",
        "relations": [],
    },
}


def _write_corpus(root: Path, spec=SKILLS) -> Path:
    for name, meta in spec.items():
        # group relation targets by relation type so each YAML key appears once
        grouped: dict = {}
        for relation, target in meta.get("relations", []):
            grouped.setdefault(relation, []).append(target)
        if grouped:
            rel_lines = []
            for relation, targets in grouped.items():
                rel_lines.append(f"  {relation}:")
                rel_lines.extend(f"    - {t}" for t in targets)
            rel_block = "relations:\n" + "\n".join(rel_lines) + "\n"
        else:
            rel_block = ""
        content = (
            "---\n"
            f"name: {name}\n"
            f"description: {meta['description']}\n"
            "metadata:\n"
            f"  category: {meta['category']}\n"
            f"  tags: {name.replace('-', ',')}\n"
            f"{rel_block}"
            "---\n\n"
            f"# {name}\n\n{meta['description']}.\n"
        )
        target = root / name / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path / "skills")


# ── L0/L1: loading ───────────────────────────────────────────
def test_load_skills_nodes_categories_edges(corpus: Path):
    graph = load_skills(str(corpus))
    assert len(graph.skills()) == len(SKILLS)
    # one category node per distinct category
    assert len(graph.categories()) == 3
    # explicit depends_on edges: 1 + 1 + 2 = 4
    deps = [e for e in graph.edges if e.edge_type == EdgeType.DEPENDS_ON]
    assert len(deps) == 4
    # belongs_to edges materialised for every skill
    belongs = [e for e in graph.edges if e.edge_type == EdgeType.BELONGS_TO]
    assert len(belongs) == len(SKILLS)
    assert category_id("data") in graph.nodes
    assert graph.nodes[skill_id("rag-pipeline")].category == "generation"


def test_relation_aliases(tmp_path: Path):
    d = tmp_path / "a"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: x\nrequires:\n  - beta\n---\n\nbody\n",
        encoding="utf-8",
    )
    d2 = tmp_path / "b"
    d2.mkdir()
    (d2 / "SKILL.md").write_text("---\nname: beta\ndescription: y\n---\n\nbody\n", encoding="utf-8")
    graph = load_skills(str(tmp_path))
    assert any(
        e.edge_type == EdgeType.DEPENDS_ON
        and e.source == skill_id("alpha")
        and e.target == skill_id("beta")
        for e in graph.edges
    )


# ── L1: builder ──────────────────────────────────────────────
class _FakeEmbedder:
    """Maps specific skill ids to fixed vectors for deterministic similarity."""

    _VECTORS = {
        "skill::alpha": [1.0, 0.0, 0.0],
        "skill::beta": [0.99, 0.1, 0.0],
        "skill::gamma": [0.0, 0.0, 1.0],
    }

    def embed(self, text: str):
        for key, vec in self._VECTORS.items():
            if key.split("::")[1] in text:
                return vec
        return [0.0, 0.0, 0.0]


def _node(name: str) -> SkillNode:
    return SkillNode(id=skill_id(name), name=name, description=name, node_type=NodeType.SKILL)


def test_builder_embeds_and_induces_similarity():
    graph = SkillGraph(nodes=[_node("alpha"), _node("beta"), _node("gamma")])
    builder = GraphBuilder(embedder=_FakeEmbedder(), similarity_threshold=0.8, max_similar_per_node=3)
    graph, report = builder.build(graph)
    assert report.embedding_dim == 3
    sim = [e for e in graph.edges if e.edge_type == EdgeType.SIMILAR_TO]
    # alpha<->beta mutually similar; gamma is orthogonal -> no edge
    pairs = {(e.source, e.target) for e in sim}
    assert (skill_id("alpha"), skill_id("beta")) in pairs
    assert (skill_id("beta"), skill_id("alpha")) in pairs
    assert all(skill_id("gamma") not in p for p in pairs)
    # similarity edges are symmetric
    assert report.symmetric_edges_added >= 1


def test_builder_materialises_symmetric_composes_with():
    graph = SkillGraph(nodes=[_node("a"), _node("b")])
    graph.add_edge(SkillEdge(skill_id("a"), skill_id("b"), EdgeType.COMPOSES_WITH))
    builder = GraphBuilder(embedder=HashingEmbedder(dim=16))
    graph, report = builder.build(graph, induce_similarity=False)
    comp = [e for e in graph.edges if e.edge_type == EdgeType.COMPOSES_WITH]
    assert len(comp) == 2
    assert report.symmetric_edges_added == 1


def test_builder_breaks_dependency_cycles():
    a, b, c = _node("a"), _node("b"), _node("c")
    graph = SkillGraph(nodes=[a, b, c])
    graph.add_edge(SkillEdge(a.id, b.id, EdgeType.DEPENDS_ON, weight=1.0))
    graph.add_edge(SkillEdge(b.id, c.id, EdgeType.DEPENDS_ON, weight=1.0))
    graph.add_edge(SkillEdge(c.id, a.id, EdgeType.DEPENDS_ON, weight=0.1))
    assert graph.find_cycles()

    builder = GraphBuilder(embedder=HashingEmbedder(dim=16), enforce_dag=True)
    graph, report = builder.build(graph, induce_similarity=False)
    assert report.cycles_broken == 1
    assert graph.find_cycles() == []


def test_builder_prunes_contradictory_conflicts():
    graph = SkillGraph(nodes=[_node("a"), _node("b")])
    graph.add_edge(SkillEdge(skill_id("a"), skill_id("b"), EdgeType.DEPENDS_ON))
    graph.add_edge(SkillEdge(skill_id("a"), skill_id("b"), EdgeType.CONFLICTS_WITH))
    builder = GraphBuilder(embedder=HashingEmbedder(dim=16))
    graph, report = builder.build(graph, induce_similarity=False)
    # conflict edge + its symmetric counterpart are both pruned
    assert report.pruned_conflicts >= 1
    assert not [e for e in graph.edges if e.edge_type == EdgeType.CONFLICTS_WITH]


def test_graph_save_load_roundtrip(corpus: Path, tmp_path: Path):
    graph = load_skills(str(corpus))
    graph, _ = GraphBuilder(embedder=HashingEmbedder(dim=32)).build(graph)
    out = tmp_path / "graph.json"
    graph.save(str(out))
    reloaded = SkillGraph.load(str(out))
    assert reloaded.stats()["nodes"] == graph.stats()["nodes"]
    assert reloaded.stats()["edges"] == graph.stats()["edges"]
    assert reloaded.nodes[skill_id("rag-pipeline")].embedding == graph.nodes[skill_id("rag-pipeline")].embedding


# ── L2: retriever ────────────────────────────────────────────
def _built(corpus: Path) -> SkillGraph:
    graph = load_skills(str(corpus))
    graph, _ = GraphBuilder(embedder=HashingEmbedder(dim=64)).build(graph)
    return graph


def test_retriever_seed_and_dependency_closure(corpus: Path):
    retriever = SkillRetriever(_built(corpus), embedder=HashingEmbedder(dim=64))
    result = retriever.retrieve("rag pipeline", top_k_seeds=1, token_budget=100000)
    returned = {s.id for s in result.skills}
    assert skill_id("rag-pipeline") in returned
    # prerequisites must be pulled in (dependency-aware retrieval)
    assert skill_id("vector-index") in returned
    assert skill_id("text-chunking") in returned
    assert skill_id("data-ingestion") in returned
    # the deprecated engine is unrelated and should not be pulled in
    assert skill_id("legacy-search") not in returned
    reasons = {s.id: s.reason for s in result.skills}
    assert reasons[skill_id("rag-pipeline")] == "seed"
    assert reasons[skill_id("data-ingestion")] == "prerequisite"


def test_retriever_respects_token_budget(corpus: Path):
    retriever = SkillRetriever(_built(corpus), embedder=HashingEmbedder(dim=64))
    tight = retriever.retrieve("rag pipeline", top_k_seeds=1, token_budget=1)
    # at least the top seed survives even under a tiny budget
    assert len(tight.skills) == 1
    assert tight.dropped_tokens > 0

    full = retriever.retrieve("rag pipeline", top_k_seeds=1, token_budget=100000)
    assert len(full.skills) >= len(tight.skills)


def test_retrieval_result_serialisation(corpus: Path):
    retriever = SkillRetriever(_built(corpus), embedder=HashingEmbedder(dim=64))
    result = retriever.retrieve("vector index", top_k_seeds=2, token_budget=100000)
    payload = result.to_dict()
    json.dumps(payload)  # must be JSON-serialisable
    assert "skills" in payload
    assert result.to_context(include_body=True)


# ── L3: CLI ──────────────────────────────────────────────────
def test_cli_build_and_query(corpus: Path, tmp_path: Path, capsys):
    out = tmp_path / "sg.json"
    rc = skill_cli_main(
        ["build", "--skills-dir", str(corpus), "--out", str(out), "--no-similarity"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["skills"] == len(SKILLS)
    assert out.exists()

    rc = skill_cli_main(["query", "rag pipeline", "--graph", str(out), "--json", "--no-body"])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    names = {s["name"] for s in result["skills"]}
    assert "rag-pipeline" in names


def test_cli_stats_from_directory(corpus: Path, capsys):
    rc = skill_cli_main(["stats", "--skills-dir", str(corpus)])
    assert rc == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["skills"] == len(SKILLS)


# ── dataset downloader ───────────────────────────────────────
def test_download_builtin_corpus(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "download_skills_dataset.py"
    spec = importlib.util.spec_from_file_location("download_skills_dataset", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "dl"
    assert mod.main(["--source", "builtin", "--output", str(out)]) == 0
    graph = load_skills(str(out))
    assert len(graph.skills()) == len(mod._BUILTIN)
    # the corpus encodes a real dependency chain ending at data-ingestion
    assert skill_id("data-ingestion") in graph.nodes