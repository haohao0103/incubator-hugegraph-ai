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
import sys
from typing import Optional

from hugegraph_llm.engines.skill_graph.embedder import HashingEmbedder, resolve_embedder
from hugegraph_llm.engines.skill_graph.graph_builder import GraphBuilder, write_to_hugegraph
from hugegraph_llm.engines.skill_graph.retriever import SkillRetriever
from hugegraph_llm.engines.skill_graph.schema import SkillGraph

# The CLI writes machine-readable output to stdout; keep the engine's INFO logs
# from interleaving with it (the project root logger writes to stdout).
logging.getLogger("hugegraph_llm.engines.skill_graph").setLevel(logging.WARNING)


def _make_embedder(kind: str):
    """``hash`` = deterministic offline embedder; ``project`` = production backends."""
    if kind == "project":
        return resolve_embedder()
    return HashingEmbedder()


def _load_graph(args: argparse.Namespace, build_if_needed: bool = False) -> SkillGraph:
    """Load a graph from JSON, or build it from a skills directory."""
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


def cmd_build(args: argparse.Namespace) -> int:
    from hugegraph_llm.engines.skill_graph.skill_loader import load_skills

    graph = load_skills(args.skills_dir)
    builder = GraphBuilder(
        embedder=_make_embedder(args.embedder),
        similarity_threshold=args.similarity_threshold,
        max_similar_per_node=args.max_similar,
        enforce_dag=not args.no_dag,
    )
    graph, report = builder.build(graph, induce_similarity=not args.no_similarity)

    if args.out:
        graph.save(args.out)
    payload = {"stats": graph.stats(), "report": report.to_dict()}
    if args.out:
        payload["out"] = args.out

    if args.write_hugegraph:
        payload["hugegraph"] = write_to_hugegraph(
            graph,
            graph_name=args.graph_name,
            url=args.hugegraph_url,
            user=args.hugegraph_user,
            pwd=args.hugegraph_pwd,
            clear=args.clear,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
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
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    graph = _load_graph(args, build_if_needed=True)
    print(json.dumps(graph.stats(), ensure_ascii=False, indent=2))
    return 0


def _skill_help(args: argparse.Namespace) -> int:
    print("error: specify a skill sub-command (build | query | stats)", file=sys.stderr)
    return 1


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", default=None, help="Path to a built skill_graph.json")
    parser.add_argument("--skills-dir", default=None, help="Directory of SKILL.md files (built on the fly)")
    parser.add_argument("--embedder", choices=["hash", "project"], default="hash",
                        help="Embedding backend: 'hash' (offline, default) or 'project' (production)")
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    parser.add_argument("--max-similar", type=int, default=5)


def _register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``build`` / ``query`` / ``stats`` commands."""
    # build
    build = subparsers.add_parser("build", help="Build the skill graph from SKILL.md files")
    build.add_argument("--skills-dir", required=True, help="Directory containing SKILL.md files")
    build.add_argument("--out", default="skill_graph.json", help="Output JSON path")
    build.add_argument("--embedder", choices=["hash", "project"], default="hash",
                       help="Embedding backend: 'hash' (offline, default) or 'project' (production)")
    build.add_argument("--similarity-threshold", type=float, default=0.72, help="Cosine threshold for similar_to")
    build.add_argument("--max-similar", type=int, default=5, help="Max similar neighbours per skill")
    build.add_argument("--no-similarity", action="store_true", help="Skip similarity edge induction")
    build.add_argument("--no-dag", action="store_true", help="Do not enforce the dependency DAG")
    build.add_argument("--write-hugegraph", action="store_true", help="Also write the graph to HugeGraph")
    build.add_argument("--graph-name", default="skillgraph", help="HugeGraph graph name")
    build.add_argument("--hugegraph-url", default="http://127.0.0.1:8080", help="HugeGraph REST URL")
    build.add_argument("--hugegraph-user", default="admin", help="HugeGraph user")
    build.add_argument("--hugegraph-pwd", default="admin", help="HugeGraph password")
    build.add_argument("--clear", action="store_true", help="Clear existing skill/category vertices first")
    build.set_defaults(func=cmd_build)

    # query
    query = subparsers.add_parser("query", help="Dependency-aware skill retrieval")
    query.add_argument("query", help="Query string")
    _add_source_args(query)
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


def add_skill_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``skill`` sub-command onto an existing argparse subparsers."""
    skill = subparsers.add_parser("skill", help="Skill Graph: build / query / stats")
    sub = skill.add_subparsers(dest="skill_command", help="Skill Graph sub-commands")
    _register_commands(sub)
    skill.set_defaults(func=_skill_help)
    return skill


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="skillgraph", description="Skill Graph engine CLI")
    subparsers = parser.add_subparsers(dest="command")
    _register_commands(subparsers)
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())