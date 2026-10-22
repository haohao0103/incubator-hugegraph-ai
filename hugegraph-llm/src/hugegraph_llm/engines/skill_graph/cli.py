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

"""CLI for the Skill Graph engine (the L3 integration layer).

Exposed both as a standalone module::

    python -m hugegraph_llm.engines.skill_graph.cli build --skills-dir ./skills
    python -m hugegraph_llm.engines.skill_graph.cli query "build a RAG pipeline" --graph ./skill_graph.json

and as a sub-command of the main ``hgmem`` CLI::

    hgmem skill build --skills-dir ./skills
    hgmem skill query "build a RAG pipeline" --graph ./skill_graph.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from hugegraph_llm.engines.skill_graph.embedder import HashingEmbedder, resolve_embedder
from hugegraph_llm.engines.skill_graph.errors import SkillGraphError
from hugegraph_llm.engines.skill_graph.graph_builder import GraphBuilder, write_to_hugegraph
from hugegraph_llm.engines.skill_graph.retriever import SkillRetriever
from hugegraph_llm.engines.skill_graph.schema import NodeType, SkillGraph, category_id

# The CLI writes machine-readable output to stdout; keep the engine's INFO logs
# from interleaving with it (the project root logger writes to stdout).
logging.getLogger("hugegraph_llm.engines.skill_graph").setLevel(logging.WARNING)


def _load_env() -> None:
    """Load the project ``.env`` into ``os.environ`` so LLM/embedding clients work.

    The engine reads its backend URLs, keys and headers straight from the
    environment (``OPENAI_EMBEDDING_URL`` / ``OPENAI_CHAT_API_BASE`` / ``...``),
    which the CLI must therefore populate before touching any client.
    """
    explicit = os.environ.get("HUGEGRAPH_LLM_ENV_PATH")
    candidates = [Path(explicit)] if explicit else []
    project_root = Path(__file__).resolve().parents[4]  # .../hugegraph-llm
    candidates.append(project_root / ".env")
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a project dependency
        return
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def _make_embedder(kind: str):
    """``hash`` = deterministic offline embedder; ``project`` = production backends.

    Explicitly choosing ``project`` implies strict mode: a production embedder
    that cannot be initialised must fail the command instead of silently
    switching to the 256-d hashing embedder (which would return an empty
    semantic channel against production vectors).
    """
    if kind == "project":
        return resolve_embedder(strict=True)
    return HashingEmbedder()


def _load_graph(args: argparse.Namespace, build_if_needed: bool = False) -> SkillGraph:
    """Load a graph from the backing services, a JSON file, or a skills directory."""
    if getattr(args, "from_service", False):
        from hugegraph_llm.engines.skill_graph.service_loader import load_graph_from_services

        # Service mode pairs the query embedder with the one used at build time;
        # the default 'hash' would silently break the semantic channel against
        # production-stored vectors, so switch to the production backend.
        if getattr(args, "embedder", "hash") == "hash":
            args.embedder = "project"
        return load_graph_from_services(
            hg_url=getattr(args, "hg_url", None),
            hg_user=getattr(args, "hg_user", "admin"),
            hg_pwd=getattr(args, "hg_pwd", "admin"),
            hg_graphspace=getattr(args, "hg_graphspace", None),
            hg_graph=getattr(args, "hg_graph", None),
        )
    graph_path = getattr(args, "graph", None)
    skills_dir = getattr(args, "skills_dir", None)
    if graph_path:
        graph = SkillGraph.load(graph_path)
        return graph
    if skills_dir:
        from hugegraph_llm.engines.skill_graph.skill_loader import load_skills

        builder = GraphBuilder(
            embedder=_make_embedder(getattr(args, "embedder", "hash")),
            similarity_threshold=getattr(args, "similarity_threshold", 0.72),
            max_similar_per_node=getattr(args, "max_similar", 5),
        )
        graph = load_skills(skills_dir)
        if build_if_needed:
            graph, _ = builder.build(graph)
        return graph
    raise SystemExit("error: provide --graph <json> or --skills-dir <dir>")


def _merge_into_base(base: SkillGraph, fresh: SkillGraph) -> tuple:
    """Sync ``fresh`` into ``base``; return ``(rebuild_ids, removed_ids)``.

    The skills directory is the desired state: skills that disappeared from it
    (and their edges) are removed, so a deleted ``SKILL.md`` cannot keep being
    served by an incremental build. New skills are added; changed skills are
    updated in place; unchanged skills keep their embeddings and ``needs``
    views, so an incremental build costs nothing for them. A changed skill's
    previous explicit out-edges are dropped and re-added from the fresh
    front-matter, so removing a ``relations:`` entry also removes the edge.
    Explicit edges of unchanged skills are simply re-asserted (dedup by key).
    """
    rebuild: set = set()
    removed: set = set()

    fresh_skill_ids = {
        node.id for node in fresh.nodes.values() if node.node_type == NodeType.SKILL
    }
    for node_id, node in list(base.nodes.items()):
        if node.node_type == NodeType.SKILL and node_id not in fresh_skill_ids:
            base.remove_node(node_id)  # also drops every edge touching it
            removed.add(node_id)

    for node in fresh.nodes.values():
        if node.node_type != NodeType.SKILL:
            continue
        existing = base.nodes.get(node.id)
        if existing is None:
            base.add_node(node)
            rebuild.add(node.id)
        elif (
            existing.body != node.body
            or existing.description != node.description
            or existing.tags != node.tags
        ):
            # drop stale explicit out-edges; fresh ones are re-added below
            base.edges = [
                e for e in base.edges if not (e.source == node.id and e.provenance == "explicit")
            ]
            base._edge_keys = {e.key() for e in base.edges}  # noqa: SLF001
            existing.name = node.name
            existing.description = node.description
            existing.category = node.category
            existing.tags = node.tags
            existing.body = node.body
            existing.source_path = node.source_path
            existing.metadata.update(node.metadata)
            rebuild.add(node.id)
    for node in fresh.nodes.values():
        if node.node_type == NodeType.CATEGORY and node.id not in base.nodes:
            base.add_node(node)
    for edge in fresh.edges:
        base.add_edge(edge)  # dedup by (source, target, type) key

    # Category hubs are derived from skills: drop those no longer referenced.
    used_categories = {
        category_id(node.category) for node in base.skills() if node.category
    }
    for category in list(base.categories()):
        if category.id not in used_categories:
            base.remove_node(category.id)

    return rebuild, removed


def cmd_build(args: argparse.Namespace) -> int:
    from hugegraph_llm.engines.skill_graph.skill_loader import load_skills

    embedder = _make_embedder(args.embedder)
    inducer = None
    if args.edge_mode == "llm":
        from hugegraph_llm.engines.skill_graph.edge_inducer import LLMEdgeInducer
        from hugegraph_llm.engines.skill_graph.llm import resolve_chat_llm

        llm = resolve_chat_llm()
        if llm is None:
            raise SystemExit("error: --edge-mode llm needs a chat LLM (set OPENAI_CHAT_API_BASE or LLM_*)")
        inducer = LLMEdgeInducer(
            llm=llm,
            embedder=embedder,
            threshold=args.similarity_threshold,
            candidate_top_k=args.max_similar,
            max_pairs=args.max_pairs,
        )
    builder = GraphBuilder(
        embedder=embedder,
        similarity_threshold=args.similarity_threshold,
        max_similar_per_node=args.max_similar,
        enforce_dag=not args.no_dag,
        edge_inducer=inducer,
    )

    if args.base and args.clear:
        raise SystemExit("error: --clear wipes the whole graph, which contradicts the incremental --base mode")

    if args.base:
        # Incremental build: load the previous graph, merge the fresh directory
        # into it, and only rebuild the skills that are new or have changed.
        base_path = Path(args.base)
        if not base_path.exists():
            raise SystemExit(f"error: --base not found: {args.base}")
        graph = SkillGraph.load(str(base_path))
        fresh = load_skills(args.skills_dir)
        new_skill_ids, removed_ids = _merge_into_base(graph, fresh)
        # Always run the build pass, even when nothing was added/changed:
        # removals still need the invariants re-run and a fresh manifest
        # (new content hash + build id) so the artifact stays publishable.
        graph, report = builder.build(
            graph,
            induce_similarity=not args.no_similarity,
            new_skill_ids=new_skill_ids,
        )
        payload = {
            "stats": graph.stats(),
            "report": report.to_dict(),
            "incremental": {
                "rebuilt_skills": sorted(new_skill_ids),
                "removed_skills": sorted(removed_ids),
            },
        }
    else:
        graph = load_skills(args.skills_dir)
        graph, report = builder.build(graph, induce_similarity=not args.no_similarity)
        payload = {"stats": graph.stats(), "report": report.to_dict()}

    if args.out:
        graph.save(args.out)
    if args.out:
        payload["out"] = args.out

    if args.write_hugegraph:
        payload["hugegraph"] = write_to_hugegraph(
            graph,
            graph_name=args.graph_name,
            url=args.hugegraph_url,
            user=args.hugegraph_user,
            pwd=args.hugegraph_pwd,
            graphspace=args.graphspace,
            clear=args.clear,
        )

    if args.write_oceanbase:
        from hugegraph_llm.engines.skill_graph.ob_store import write_to_oceanbase

        payload["oceanbase"] = write_to_oceanbase(
            graph,
            host=args.ob_host,
            port=args.ob_port,
            user=args.ob_user,
            password=args.ob_password,
            database=args.ob_database,
            table=args.ob_table,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    if getattr(args, "from_service", False):
        # Production path: seeds come from OceanBase and expansion from
        # HugeGraph; the full graph is never loaded into this process.
        if getattr(args, "embedder", "hash") == "hash":
            args.embedder = "project"
        from hugegraph_llm.engines.skill_graph.retriever import ServiceSkillRetriever

        retriever = ServiceSkillRetriever(
            embedder=_make_embedder(args.embedder),
            hg_url=args.hg_url,
            hg_user=args.hg_user,
            hg_pwd=args.hg_pwd,
            hg_graphspace=args.hg_graphspace,
            hg_graph=args.hg_graph,
            ob_host=getattr(args, "ob_host", None),
            ob_port=getattr(args, "ob_port", None),
            ob_user=getattr(args, "ob_user", None),
            ob_password=getattr(args, "ob_password", None),
            ob_database=getattr(args, "ob_database", None),
        )
        result = retriever.retrieve(
            args.query,
            top_k_seeds=args.top_k,
            token_budget=args.token_budget,
            max_skills=args.max_skills,
            include_similar=args.include_similar,
        )
    else:
        graph = _load_graph(args, build_if_needed=True)
        retriever = SkillRetriever(graph, embedder=_make_embedder(args.embedder))
        result = retriever.retrieve(
            args.query,
            top_k_seeds=args.top_k,
            token_budget=args.token_budget,
            max_skills=args.max_skills,
            include_similar=args.include_similar,
        )
    if args.json:
        print(json.dumps(result.to_dict(include_body=not args.no_body), ensure_ascii=False, indent=2))
    else:
        print(result.to_context(include_body=not args.no_body))
        print(
            f"\n[retrieved {len(result.skills)} skills, "
            f"~{result.total_tokens} tokens, seed-size={len(result.seeds)}]",
            file=sys.stderr,
        )
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    graph = _load_graph(args, build_if_needed=True)
    print(json.dumps(graph.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    from hugegraph_llm.engines.skill_graph.viz import write_html

    graph = _load_graph(args, build_if_needed=True)
    result = None
    if getattr(args, "query", None):
        retriever = SkillRetriever(graph, embedder=_make_embedder(args.embedder))
        result = retriever.retrieve(
            args.query,
            top_k_seeds=args.top_k,
            token_budget=args.token_budget,
        )
    out = write_html(graph, args.out, result=result, title=args.title)
    payload = {"out": out, "stats": graph.stats()}
    if result is not None:
        payload["highlighted"] = [s.id for s in result.skills]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    """Publish a built artifact under a new version and switch readers atomically."""
    from hugegraph_llm.engines.skill_graph.hg_store import HugeGraphConfig, publish_graph

    graph = SkillGraph.load(args.graph)
    config = HugeGraphConfig.from_env(
        url=args.hg_url,
        user=args.hg_user,
        pwd=args.hg_pwd,
        graphspace=args.hg_graphspace,
        graph=args.hg_graph,
    )
    payload = publish_graph(
        graph,
        hg_config=config,
        ob_host=args.ob_host,
        ob_port=args.ob_port,
        ob_user=args.ob_user,
        ob_password=args.ob_password,
        ob_database=args.ob_database,
        ob_table_base=args.ob_table,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Switch the active pointer back to a previous build (no data rewriting)."""
    from hugegraph_llm.engines.skill_graph.hg_store import HugeGraphConfig, rollback

    config = HugeGraphConfig.from_env(
        url=args.hg_url,
        user=args.hg_user,
        pwd=args.hg_pwd,
        graphspace=args.hg_graphspace,
        graph=args.hg_graph,
    )
    payload = rollback(args.build_id, hg_config=config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _skill_help(args: argparse.Namespace) -> int:
    print(
        "error: specify a skill sub-command (build | query | stats | viz | publish | rollback)",
        file=sys.stderr,
    )
    return 1


def _add_hg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hg-url", default=None, help="HugeGraph URL (or SKILL_GRAPH_HG_URL)")
    parser.add_argument("--hg-user", default="admin", help="HugeGraph user")
    parser.add_argument("--hg-pwd", default="admin", help="HugeGraph password")
    parser.add_argument("--hg-graphspace", default=None,
                        help="HugeGraph graphspace (or SKILL_GRAPH_HG_GRAPHSPACE)")
    parser.add_argument("--hg-graph", default=None, help="HugeGraph graph (or SKILL_GRAPH_HG_GRAPH)")


def _add_ob_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ob-host", default=None, help="OceanBase host (or OCEANBASE_HOST)")
    parser.add_argument("--ob-port", type=int, default=None, help="OceanBase port (or OCEANBASE_PORT)")
    parser.add_argument("--ob-user", default=None, help="OceanBase user (or OCEANBASE_USER)")
    parser.add_argument("--ob-password", default=None, help="OceanBase password (or OCEANBASE_PASSWORD)")
    parser.add_argument("--ob-database", default=None, help="OceanBase database (or OCEANBASE_DATABASE)")


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-service", action="store_true",
                        help="Load the graph from HugeGraph + OceanBase (PRODUCTION mode: the services "
                             "are the source of truth, no local JSON involved). Embedder switches to "
                             "'project' unless overridden")
    _add_hg_args(parser)
    parser.add_argument("--graph", default=None, help="Path to a built skill_graph.json (dev/CI artifact)")
    parser.add_argument("--skills-dir", default=None, help="Directory of SKILL.md files (built on the fly)")
    parser.add_argument("--embedder", choices=["hash", "project"], default="hash",
                        help="Embedding backend: 'hash' (offline, default) or 'project' (production)")
    parser.add_argument("--similarity-threshold", type=float, default=0.30)
    parser.add_argument("--max-similar", type=int, default=5)


def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``build`` / ``query`` / ``stats`` commands."""
    # build
    build = subparsers.add_parser("build", help="Build the skill graph from SKILL.md files")
    build.add_argument("--skills-dir", required=True, help="Directory containing SKILL.md files")
    build.add_argument("--out", default="skill_graph.json", help="Output JSON path")
    build.add_argument("--base", default=None,
                       help="Existing skill_graph.json to merge into (INCREMENTAL build: only "
                            "new/changed skills are re-embedded and re-induced; unchanged skills "
                            "reuse their persisted embeddings and 'needs' views, so LLM cost is "
                            "O(new skills) instead of O(all skills))")
    build.add_argument("--embedder", choices=["hash", "project"], default="hash",
                       help="Embedding backend: 'hash' (offline, default) or 'project' (production)")
    build.add_argument("--similarity-threshold", type=float, default=0.30, help="Cosine threshold for similar_to")
    build.add_argument("--max-similar", type=int, default=5, help="Max similar neighbours per skill")
    build.add_argument("--no-similarity", action="store_true", help="Skip similarity edge induction")
    build.add_argument("--edge-mode", choices=["similarity", "llm"], default="similarity",
                       help="Edge induction: 'similarity' (offline, default) or "
                            "'llm' (typed edges via the project chat LLM, SkillDAG recipe)")
    build.add_argument("--max-pairs", type=int, default=200,
                       help="Hard cap on LLM classification calls in --edge-mode llm")
    build.add_argument("--no-dag", action="store_true", help="Do not enforce the dependency DAG")
    build.add_argument("--write-hugegraph", action="store_true",
                       help="Also write the graph to HugeGraph (dev only: unversioned; use 'publish' "
                            "for an atomic, versioned production release)")
    build.add_argument("--graph-name", default="skillgraph", help="HugeGraph graph name")
    build.add_argument("--hugegraph-url", default="http://127.0.0.1:8080", help="HugeGraph REST URL")
    build.add_argument("--hugegraph-user", default="admin", help="HugeGraph user")
    build.add_argument("--hugegraph-pwd", default="admin", help="HugeGraph password")
    build.add_argument("--graphspace", default=None, help="HugeGraph graphspace (1.7+); omit for DEFAULT")
    build.add_argument("--clear", action="store_true", help="Clear existing skill/category vertices first")
    build.add_argument("--write-oceanbase", action="store_true",
                       help="Also write skill vectors + surface text to OceanBase (dev only: use "
                            "'publish' for an atomic, versioned production release)")
    build.add_argument("--ob-host", default=None, help="OceanBase host (or OCEANBASE_HOST)")
    build.add_argument("--ob-port", type=int, default=None, help="OceanBase port (or OCEANBASE_PORT)")
    build.add_argument("--ob-user", default=None, help="OceanBase user (or OCEANBASE_USER)")
    build.add_argument("--ob-password", default=None, help="OceanBase password (or OCEANBASE_PASSWORD)")
    build.add_argument("--ob-database", default=None, help="OceanBase database (or OCEANBASE_DATABASE)")
    build.add_argument("--ob-table", default=None,
                       help="OceanBase table; defaults to an isolated skill_graph_* table")
    build.set_defaults(func=cmd_build)

    # query
    query = subparsers.add_parser("query", help="Dependency-aware skill retrieval")
    query.add_argument("query", help="Query string")
    _add_source_args(query)
    _add_ob_args(query)
    query.add_argument("--top-k", type=int, default=5, help="Number of seeds")
    query.add_argument("--token-budget", type=int, default=4000, help="Hydration token budget")
    query.add_argument("--max-skills", type=int, default=None, help="Max skills to hydrate")
    query.add_argument("--include-similar", action="store_true", help="Also expand similar_to neighbours")
    query.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    query.add_argument("--no-body", action="store_true", help="Exclude skill bodies from output")
    query.set_defaults(func=cmd_query)

    # stats
    stats = subparsers.add_parser("stats", help="Print graph statistics")
    _add_source_args(stats)
    stats.set_defaults(func=cmd_stats)

    # viz
    viz = subparsers.add_parser("viz", help="Render an interactive HTML visualisation")
    _add_source_args(viz)
    viz.add_argument("--out", default="skill_graph.html", help="Output HTML path")
    viz.add_argument("--title", default="Skill Graph", help="Page title")
    viz.add_argument("--query", default=None, help="Optional query to highlight retrieved skills")
    viz.add_argument("--top-k", type=int, default=5, help="Number of seeds when --query is set")
    viz.add_argument("--token-budget", type=int, default=4000, help="Hydration token budget when --query is set")
    viz.set_defaults(func=cmd_viz)

    # publish
    publish = subparsers.add_parser(
        "publish",
        help="Atomically publish a built graph to HugeGraph + OceanBase under a new build id",
    )
    publish.add_argument("--graph", required=True,
                         help="Built skill_graph.json (must carry a build manifest)")
    _add_hg_args(publish)
    _add_ob_args(publish)
    publish.add_argument("--ob-table", default=None,
                         help="Base OceanBase table; the build publishes into <base>__<build>")
    publish.add_argument("--dry-run", action="store_true",
                         help="Validate and show the build id/table without writing anything")
    publish.set_defaults(func=cmd_publish)

    # rollback
    rollback = subparsers.add_parser(
        "rollback",
        help="Point readers back to a previous build (atomic pointer switch, no data rewrite)",
    )
    rollback.add_argument("--build-id", default=None,
                          help="Build to activate; defaults to the active build's previous_build")
    _add_hg_args(rollback)
    rollback.set_defaults(func=cmd_rollback)


def add_skill_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``skill`` sub-command onto an existing argparse subparsers."""
    skill = subparsers.add_parser("skill", help="Skill Graph: build / query / stats")
    sub = skill.add_subparsers(dest="skill_command", help="Skill Graph sub-commands")
    _register_commands(sub)
    skill.set_defaults(func=_skill_help)
    return skill


def main(argv: Optional[list] = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="skillgraph", description="Skill Graph engine CLI")
    subparsers = parser.add_subparsers(dest="command")
    _register_commands(subparsers)
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except SkillGraphError as exc:
        # Typed engine errors are actionable messages, not bugs: print them
        # without a traceback and exit non-zero so CI/pipelines stop.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())