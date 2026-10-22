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
    LLMEdgeInducer,
    NodeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
    SkillRetriever,
    load_skills,
    skill_id,
)
from hugegraph_llm.engines.skill_graph.cli import main as skill_cli_main
from hugegraph_llm.engines.skill_graph.hg_store import HugeGraphSkillStore
from hugegraph_llm.engines.skill_graph.ob_store import SemanticHit
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


# ── L1: typed edge induction (LLM, SkillDAG recipe) ──────────
class _FakeLLM:
    """Deterministic stand-in for a chat LLM (no network)."""

    def generate(self, prompt=None, *args, **kwargs):
        prompt = prompt or ""
        if "ONE SHORT SENTENCE" in prompt:
            return "a RAG flow requires a vector index and text chunking"
        # classification prompt: decide from the A/B names it contains
        if "rag-pipeline" in prompt and "vector-index" in prompt:
            return '{"type": "depends_on"}'
        if "text-chunking" in prompt and "data-ingestion" in prompt:
            return '```json\n{"type": "composes_with"}\n```'
        return '{"type": "none"}'


def test_llm_edge_inducer_adds_typed_edges():
    nodes = [_node("rag-pipeline"), _node("vector-index"), _node("text-chunking"), _node("data-ingestion")]
    graph = SkillGraph(nodes=nodes)
    inducer = LLMEdgeInducer(
        llm=_FakeLLM(),
        embedder=HashingEmbedder(dim=32),
        threshold=0.0,  # every pair becomes a candidate; the LLM decides the type
        candidate_top_k=4,
        max_pairs=20,
    )
    builder = GraphBuilder(embedder=HashingEmbedder(dim=32), edge_inducer=inducer)
    graph, report = builder.build(graph)

    assert report.llm_edges >= 1
    assert report.llm_induction["classified_pairs"] >= 1
    llm_edges = [e for e in graph.edges if e.provenance == "llm"]
    types = {e.edge_type for e in llm_edges}
    assert EdgeType.DEPENDS_ON in types
    # composes_with is symmetric -> materialised in both directions by the builder
    comp = [e for e in graph.edges if e.edge_type == EdgeType.COMPOSES_WITH]
    assert len(comp) == 2
    # no similarity edges were induced when a typed inducer is supplied
    assert report.similarity_edges == 0


def test_llm_edge_inducer_requires_generate():
    with pytest.raises(ValueError):
        LLMEdgeInducer(llm=object())


# ── L1: incremental build ────────────────────────────────────
class _CountingLLM(_FakeLLM):
    """FakeLLM that counts generate() calls, to prove incremental cheapness."""

    def __init__(self):
        super().__init__()
        self.needs_calls = 0

    def generate(self, prompt=None, *args, **kwargs):
        if prompt and "ONE SHORT SENTENCE" in prompt:
            self.needs_calls += 1
        return super().generate(prompt, *args, **kwargs)


def test_incremental_build_only_regenerates_needs_for_new_skills(tmp_path: Path):
    nodes = [_node("rag-pipeline"), _node("vector-index"), _node("text-chunking"), _node("data-ingestion")]
    graph = SkillGraph(nodes=nodes)
    llm = _CountingLLM()
    inducer = LLMEdgeInducer(llm=llm, embedder=HashingEmbedder(dim=32), threshold=0.0,
                             candidate_top_k=4, max_pairs=20)
    builder = GraphBuilder(embedder=HashingEmbedder(dim=32), edge_inducer=inducer)
    graph, report = builder.build(graph)
    full_needs_calls = llm.needs_calls
    assert full_needs_calls == 4  # one needs view per skill in a full build

    # persist, then add ONE new skill and build incrementally
    out = tmp_path / "base.json"
    graph.save(str(out))
    base = SkillGraph.load(str(out))
    new_node = _node("answer-generation")
    base.add_node(new_node)
    llm2 = _CountingLLM()
    inducer2 = LLMEdgeInducer(llm=llm2, embedder=HashingEmbedder(dim=32), threshold=0.0,
                              candidate_top_k=4, max_pairs=20)
    builder2 = GraphBuilder(embedder=HashingEmbedder(dim=32), edge_inducer=inducer2)
    base, report2 = builder2.build(base, new_skill_ids={new_node.id})

    # only the new skill costs an LLM needs call; old skills reuse persisted views
    assert llm2.needs_calls == 1
    assert report2.llm_induction["needs_generated"] == 1
    # old skills keep their previous llm edges; the new node is present
    assert new_node.id in base.nodes
    old_edges = [e for e in base.edges if e.provenance == "llm" and new_node.id not in (e.source, e.target)]
    assert old_edges, "incremental build must not drop previously induced edges"


def test_cli_incremental_build_new_and_updated(corpus: Path, tmp_path: Path, capsys):
    base = tmp_path / "base.json"
    merged = tmp_path / "merged.json"
    rc = skill_cli_main(["build", "--skills-dir", str(corpus), "--out", str(base), "--no-similarity"])
    assert rc == 0
    capsys.readouterr()

    # add a NEW skill and UPDATE an existing one
    new_dir = corpus / "answer-generation"
    new_dir.mkdir()
    (new_dir / "SKILL.md").write_text(
        "---\nname: answer-generation\ndescription: Compose LLM answers from retrieved chunks\n"
        "metadata:\n  category: generation\n---\n\n# answer-generation\n",
        encoding="utf-8",
    )
    updated = corpus / "rag-pipeline" / "SKILL.md"
    updated.write_text(updated.read_text(encoding="utf-8").replace(
        "Compose retrieval and generation into a RAG flow",
        "Compose retrieval and generation into a production RAG flow",
    ), encoding="utf-8")

    rc = skill_cli_main(["build", "--skills-dir", str(corpus), "--base", str(base),
                         "--out", str(merged), "--no-similarity"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    rebuilt = payload["incremental"]["rebuilt_skills"]
    assert rebuilt == [skill_id("answer-generation"), skill_id("rag-pipeline")]

    graph = SkillGraph.load(str(merged))
    assert len(graph.skills()) == len(SKILLS) + 1
    # unchanged skills keep their explicit edges; the updated one re-declares them
    deps = [e for e in graph.edges if e.edge_type == EdgeType.DEPENDS_ON]
    assert len(deps) == 4
    # rerun with no changes: nothing is rebuilt
    rc = skill_cli_main(["build", "--skills-dir", str(corpus), "--base", str(merged),
                         "--out", str(tmp_path / "again.json"), "--no-similarity"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["incremental"]["rebuilt_skills"] == []


def test_cli_incremental_rejects_clear_conflict(corpus: Path, tmp_path: Path):
    base = tmp_path / "base.json"
    skill_cli_main(["build", "--skills-dir", str(corpus), "--out", str(base), "--no-similarity"])
    with pytest.raises(SystemExit):
        skill_cli_main(["build", "--skills-dir", str(corpus), "--base", str(base), "--clear"])


def test_incremental_rejects_embedder_dimension_change(corpus: Path, tmp_path: Path):
    # build a base graph with a 32-d embedder
    graph = load_skills(str(corpus))
    graph, _ = GraphBuilder(embedder=HashingEmbedder(dim=32)).build(graph, induce_similarity=False)
    # sneak in a new skill and try to build incrementally with a 64-d embedder
    new_node = _node("latecomer")
    graph.add_node(new_node)
    with pytest.raises(ValueError, match="dimension mismatch"):
        GraphBuilder(embedder=HashingEmbedder(dim=64)).build(
            graph, induce_similarity=False, new_skill_ids={new_node.id}
        )


# ── service-backed loading (production source of truth) ──────
def test_graph_from_gremlin_data_roundtrip():
    from hugegraph_llm.engines.skill_graph.service_loader import graph_from_gremlin_data

    vertices = [
        {
            "id": "skill::pptx",
            "label": "skill",
            "type": "vertex",
            "properties": {
                "node_name": "pptx",
                "description": "PowerPoint decks",
                "category": "documents",
                "tags": "slides,presentation",
                "node_type": "skill",
                "source_path": "skills/pptx/SKILL.md",
            },
        },
        {
            "id": "category::documents",
            "label": "category",
            "type": "vertex",
            "properties": {"node_name": "documents", "node_type": "category"},
        },
        # unknown label: must be skipped, not fatal
        {"id": "skill::weird", "label": "legacy-widget", "type": "vertex", "properties": {}},
    ]
    edges = [
        {
            "id": "e1",
            "label": "composes_with",
            "type": "edge",
            "outV": "skill::pptx",
            "inV": "category::documents",
            "properties": {"weight": 0.9, "provenance": "llm"},
        },
        {"id": "e2", "label": "unknown_relation", "type": "edge", "outV": "skill::pptx", "inV": "category::documents"},
    ]
    graph = graph_from_gremlin_data(vertices, edges)
    assert set(graph.nodes) == {"skill::pptx", "category::documents"}
    assert graph.nodes["skill::pptx"].tags == ["slides", "presentation"]
    assert graph.nodes["category::documents"].node_type == NodeType.CATEGORY
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.edge_type == EdgeType.COMPOSES_WITH
    assert edge.weight == pytest.approx(0.9)
    assert edge.provenance == "llm"


def test_graph_from_gremlin_data_strips_build_prefix():
    from hugegraph_llm.engines.skill_graph.service_loader import graph_from_gremlin_data

    vertices = [
        {"id": "1a2b3c4d:skill::pptx", "label": "skill",
         "properties": {"node_name": "pptx", "node_type": "skill"}},
        {"id": "1a2b3c4d:category::documents", "label": "category",
         "properties": {"node_name": "documents", "node_type": "category"}},
    ]
    edges = [{
        "id": "e1", "label": "composes_with",
        "outV": "1a2b3c4d:skill::pptx", "inV": "1a2b3c4d:category::documents",
        "properties": {"weight": 0.9, "provenance": "llm"},
    }]
    graph = graph_from_gremlin_data(vertices, edges, id_prefix="1a2b3c4d:")
    assert set(graph.nodes) == {"skill::pptx", "category::documents"}
    assert graph.edges[0].source == "skill::pptx"
    assert graph.edges[0].target == "category::documents"


def test_parse_embedding_from_oceanbase_vector_strings():
    from hugegraph_llm.engines.skill_graph.service_loader import _parse_embedding

    assert _parse_embedding(None) is None
    assert _parse_embedding('[0.1,0.25,-0.3]') == pytest.approx([0.1, 0.25, -0.3])
    assert _parse_embedding([1, 2]) == [1.0, 2.0]
    assert _parse_embedding("0.5,0.6") == pytest.approx([0.5, 0.6])


class _FakePointer:
    def __init__(self, build_id: str):
        self.build_id = build_id


class _FakeHgStore:
    """Minimal stand-in for HugeGraphSkillStore's read surface."""

    def __init__(self, build_id: str):
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            url="http://hg:8080", graphspace=None, graph="skillgraph"
        )
        self.build_id = build_id
        self.reads = []

    def active_pointer(self):
        return _FakePointer(self.build_id)

    def read_build(self, build_id: str) -> SkillGraph:
        self.reads.append(build_id)
        return SkillGraph(nodes=[SkillNode(id="skill::pptx", name="pptx")])


def test_service_graph_cache_reloads_only_when_the_pointer_moves(monkeypatch):
    from hugegraph_llm.engines.skill_graph import service_loader

    store = _FakeHgStore("build-a")
    monkeypatch.setattr(service_loader, "_open_hg_store", lambda *a, **k: store)
    monkeypatch.setattr(
        service_loader, "hydrate_from_oceanbase", lambda graph, *a, **k: 0
    )
    cache = service_loader.ServiceGraphCache(max_entries=2)

    first = service_loader.load_graph_from_services(cache=cache)
    assert service_loader.load_graph_from_services(cache=cache) is first
    assert store.reads == ["build-a"]

    # publishing a new build invalidates by pointer, not by TTL
    store.build_id = "build-b"
    second = service_loader.load_graph_from_services(cache=cache)
    assert second is not first
    assert store.reads == ["build-a", "build-b"]

    # rolling back to A is served from the still-warm snapshot
    store.build_id = "build-a"
    assert service_loader.load_graph_from_services(cache=cache) is first
    assert store.reads == ["build-a", "build-b"]

    # explicit refresh bypasses the hit
    service_loader.load_graph_from_services(cache=cache, force_refresh=True)
    assert store.reads == ["build-a", "build-b", "build-a"]


def test_retriever_reuses_bm25_index_for_the_same_snapshot():
    graph = SkillGraph(nodes=[SkillNode(id="skill::rag", name="rag", description="retrieval")])
    other = SkillGraph(nodes=[SkillNode(id="skill::sql", name="sql", description="query")])
    embedder = HashingEmbedder(dim=32)

    first = SkillRetriever(graph, embedder=embedder)
    assert SkillRetriever(graph, embedder=embedder)._bm25 is first._bm25
    # a different snapshot must never share the index
    assert SkillRetriever(other, embedder=embedder)._bm25 is not first._bm25


def test_resource_names_reject_injection():
    from hugegraph_llm.engines.skill_graph.errors import BuildValidationError
    from hugegraph_llm.engines.skill_graph.manifest import (
        BuildManifest,
        build_scoped_table,
        new_build_id,
        short_build_id,
    )
    from hugegraph_llm.engines.skill_graph.ob_store import OceanBaseSkillStore

    build_id = new_build_id()
    assert build_scoped_table("skill_graph_skill_semantic", build_id).endswith(
        "__" + build_id.rsplit("-", 1)[-1]
    )

    with pytest.raises(BuildValidationError):
        short_build_id("20261020T101500Z-1a2b3c4d`; DROP TABLE x; --")
    with pytest.raises(BuildValidationError):
        BuildManifest(
            build_id="not-a-build-id",
            embedder_id="hash-32",
            embedder_dim=32,
            content_hash="deadbeef",
        ).validate()
    with pytest.raises(ValueError, match="invalid OceanBase table name"):
        OceanBaseSkillStore(
            host="127.0.0.1", user="root", database="test",
            table="t`; DROP TABLE x; --",
        )


# ── query-down retrieval (OceanBase seeds + bounded HugeGraph expansion) ──
class _BodyCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, tuple(params)))

    def fetchall(self):
        return [(nid, f"body:{nid}") for nid in self.calls[-1][1]]


class _BodyConnection:
    def __init__(self):
        self.cursor_obj = _BodyCursor()

    def cursor(self):
        return self.cursor_obj


def test_fetch_bodies_only_reads_requested_ids():
    from hugegraph_llm.engines.skill_graph.ob_store import OceanBaseSkillStore

    ob = object.__new__(OceanBaseSkillStore)
    ob.table = "skill_graph_semantic"
    ob._conn = _BodyConnection()

    bodies = ob.fetch_bodies(["skill::a", "skill::b", "skill::a"], batch_size=1)

    assert bodies == {"skill::a": "body:skill::a", "skill::b": "body:skill::b"}
    # de-duplicated and chunked: one IN(...) statement per id here
    assert len(ob._conn.cursor_obj.calls) == 2
    assert all("IN (%s)" in sql for sql, _ in ob._conn.cursor_obj.calls)


def test_hybrid_seed_search_fuses_channels_with_rrf(monkeypatch):
    from hugegraph_llm.engines.skill_graph.ob_store import OceanBaseSkillStore, SemanticHit

    def hit(nid):
        return SemanticHit(node_id=nid, title=nid, surface_text=nid, score=1.0, channel="x")

    ob = object.__new__(OceanBaseSkillStore)
    monkeypatch.setattr(ob, "fulltext_search", lambda q, top_k=50: [hit("a"), hit("b"), hit("c")][:top_k])
    monkeypatch.setattr(ob, "vector_search", lambda v, top_k=50: [hit("c"), hit("a")][:top_k])

    fused = ob.hybrid_seed_search("query", [0.1, 0.2], per_channel=3, limit=3)

    # a appears high in both rankings, so RRF must rank it first
    assert [h.node_id for h in fused] == ["a", "c", "b"]
    assert all(h.channel == "hybrid" for h in fused)
    assert ob.hybrid_seed_search("query", [0.1], per_channel=0) == []


def test_hybrid_seed_search_without_embedding_uses_fulltext_only(monkeypatch):
    from hugegraph_llm.engines.skill_graph.ob_store import OceanBaseSkillStore, SemanticHit

    ob = object.__new__(OceanBaseSkillStore)
    monkeypatch.setattr(
        ob,
        "fulltext_search",
        lambda q, top_k=50: [SemanticHit("a", "a", "a", 1.0, "fulltext")],
    )
    monkeypatch.setattr(
        ob, "vector_search", lambda v, top_k=50: pytest.fail("vector channel must not run")
    )

    assert [h.node_id for h in ob.hybrid_seed_search("q", None)] == ["a"]


class _GremlinGraphStore(HugeGraphSkillStore):
    """In-memory Gremlin double: answers the three query shapes used by expansion."""

    def __init__(self, vertices, edges):
        self.vertices = {v["id"]: v for v in vertices}
        self.edges = edges
        self.queries = []

    def _gremlin(self, query):
        self.queries.append(query)
        ids = [vid for vid in self.vertices if json.dumps(vid) in query]
        if ".bothE(" in query:
            data = [
                e
                for e in self.edges
                if json.dumps(e["outV"]) in query and json.dumps(e["inV"]) in query
            ]
            return {"data": data}
        if ".out(" in query:
            targets = {
                e["inV"]
                for e in self.edges
                if e["outV"] in ids and e["label"] in ("depends_on", "composes_with")
            }
            return {"data": [self.vertices[t] for t in targets if t in self.vertices]}
        return {"data": [self.vertices[i] for i in ids]}


def _versioned_graph(build_id="20261020T101500Z-1a2b3c4d"):
    prefix = "1a2b3c4d:"
    vertices = [
        {"id": prefix + "skill::viewer", "label": "skill",
         "properties": {"node_name": "viewer", "description": "render docs", "node_type": "skill"}},
        {"id": prefix + "skill::render", "label": "skill",
         "properties": {"node_name": "render", "description": "render output", "node_type": "skill"}},
        {"id": prefix + "skill::layout", "label": "skill",
         "properties": {"node_name": "layout", "description": "layout pages", "node_type": "skill"}},
        {"id": prefix + "skill::other", "label": "skill",
         "properties": {"node_name": "other", "description": "unrelated", "node_type": "skill"}},
    ]
    edges = [
        {"id": "e1", "label": "depends_on", "outV": prefix + "skill::viewer",
         "inV": prefix + "skill::render", "properties": {"weight": 1.0}},
        {"id": "e2", "label": "depends_on", "outV": prefix + "skill::render",
         "inV": prefix + "skill::layout", "properties": {"weight": 1.0}},
        {"id": "e3", "label": "belongs_to", "outV": prefix + "skill::viewer",
         "inV": prefix + "skill::other", "properties": {}},
    ]
    return build_id, _GremlinGraphStore(vertices, edges)


def test_expand_prerequisites_reads_only_the_seed_closure():
    build_id, store = _versioned_graph()

    graph = store.expand_prerequisites(["skill::viewer"], build_id, max_depth=2)

    assert set(graph.nodes) == {"skill::viewer", "skill::render", "skill::layout"}
    assert {e.target for e in graph.edges} == {"skill::render", "skill::layout"}
    # "other" is not a prerequisite, so neither it nor its edge is read
    assert "skill::other" not in graph.nodes
    # only dependency labels are traversed, never belongs_to
    out_queries = [q for q in store.queries if ".out(" in q]
    assert out_queries and all('"depends_on"' in q for q in out_queries)


def test_expand_prerequisites_honours_depth_and_node_caps():
    build_id, store = _versioned_graph()

    shallow = store.expand_prerequisites(["skill::viewer"], build_id, max_depth=1)
    assert set(shallow.nodes) == {"skill::viewer", "skill::render"}

    capped = store.expand_prerequisites(
        ["skill::viewer", "skill::other"], build_id, max_depth=5, max_nodes=2
    )
    assert len(capped.nodes) == 2


def test_expand_prerequisites_rejects_injected_edge_labels():
    build_id, store = _versioned_graph()

    with pytest.raises(ValueError, match="unknown edge type"):
        store.expand_prerequisites(
            ["skill::viewer"], build_id, edge_types=["depends_on') .drop(); g.V('"]
        )


class _FakeActivePointer:
    def __init__(self, build_id="20261020T101500Z-1a2b3c4d", table="skill_graph_semantic",
                 embedder_id="hashing-8", embedder_dim=8):
        self.build_id = build_id
        self.ob_table = table
        self.embedder_id = embedder_id
        self.embedder_dim = embedder_dim


class _FakeServiceHg:
    def __init__(self, pointer, graph):
        self.pointer = pointer
        self.graph = graph
        self.calls = []

    def active_pointer(self):
        return self.pointer

    def expand_prerequisites(self, seed_ids, build_id, **kwargs):
        self.calls.append((list(seed_ids), build_id, kwargs))
        return self.graph


class _FakeOb:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        _FakeOb.last = self

    def hybrid_seed_search(self, query, query_embedding=None, per_channel=50, limit=100):
        return [
            SemanticHit("skill::viewer", "viewer", "viewer", 0.9, "hybrid"),
            SemanticHit("skill::ghost", "ghost", "ghost", 0.5, "hybrid"),
        ]

    def fetch_bodies(self, node_ids, batch_size=200):
        return {nid: f"body:{nid}" for nid in node_ids}

    def close(self):
        self.closed = True


def _service_graph():
    return SkillGraph(
        nodes=[
            SkillNode(id="skill::viewer", name="viewer", description="render docs"),
            SkillNode(id="skill::render", name="render", description="render output"),
        ],
        edges=[SkillEdge(source="skill::viewer", target="skill::render",
                         edge_type=EdgeType.DEPENDS_ON)],
    )


def test_service_retriever_uses_bounded_query_path(monkeypatch):
    from hugegraph_llm.engines.skill_graph import ServiceSkillRetriever, ob_store

    pointer = _FakeActivePointer()
    hg = _FakeServiceHg(pointer, _service_graph())
    embedder = HashingEmbedder(dim=8)
    pointer.embedder_id = embedder.identity
    monkeypatch.setattr(ob_store, "OceanBaseSkillStore", _FakeOb)

    retriever = ServiceSkillRetriever(embedder=embedder, hg_store=hg)
    result = retriever.retrieve("how do I render a page", top_k_seeds=2)

    # the graph is never read in bulk: only the seed closure is requested
    assert hg.calls[0][0] == ["skill::viewer", "skill::ghost"]
    assert hg.calls[0][1] == pointer.build_id
    assert hg.calls[0][2]["max_depth"] == retriever.max_depth
    assert _FakeOb.last.kwargs["table"] == pointer.ob_table
    assert _FakeOb.last.closed is True

    assert [s.id for s in result.skills][:2] == ["skill::viewer", "skill::render"]
    assert result.skills[0].body == "body:skill::viewer"
    # the ghost seed is reported instead of silently dropped
    assert result.degraded is True
    assert "missing from HugeGraph" in result.warnings[0]


def test_service_retriever_fails_fast_on_missing_pointer_or_embedder_mismatch(monkeypatch):
    from hugegraph_llm.engines.skill_graph import ServiceSkillRetriever, ob_store
    from hugegraph_llm.engines.skill_graph.errors import (
        EmbeddingMismatchError,
        GraphReadError,
    )

    monkeypatch.setattr(ob_store, "OceanBaseSkillStore", _FakeOb)
    embedder = HashingEmbedder(dim=8)

    no_pointer = _FakeServiceHg(None, _service_graph())
    with pytest.raises(GraphReadError, match="no active skill-graph build pointer"):
        ServiceSkillRetriever(embedder=embedder, hg_store=no_pointer).retrieve("q")

    mismatched = _FakeServiceHg(_FakeActivePointer(embedder_id="other-embedder"),
                                _service_graph())
    with pytest.raises(EmbeddingMismatchError, match="indexed with embedder"):
        ServiceSkillRetriever(embedder=embedder, hg_store=mismatched).retrieve("q")


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