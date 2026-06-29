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

"""
hgmem — command-line interface for HugeGraph-AI-Memory.

Mirrors the ergonomics of the Mem0 and PowerMem CLIs while exposing the
HugeGraph-backed memory pipeline: add, search, list, delete, server, locomo,
and audit commands.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure this workspace's source tree takes precedence over any venv-installed package.
_WORKSPACE_SRC = str(Path(__file__).resolve().parents[1])
if _WORKSPACE_SRC not in sys.path:
    sys.path.insert(0, _WORKSPACE_SRC)

from hugegraph_llm.poc.memory_backend import MemoryPipelineBackend, HugeGraphMemoryClient
from hugegraph_llm.utils.audit_log import get_audit_logger
from hugegraph_llm.utils.log import log


def _load_server_main():
    """Lazy load demo/memory_server.py which is not a package module."""
    import importlib.util
    server_path = Path(__file__).resolve().parents[2] / "demo" / "memory_server.py"
    spec = importlib.util.spec_from_file_location("memory_server", str(server_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memory_server"] = mod
    spec.loader.exec_module(mod)
    return mod.main


def _get_backend(graph_name: str) -> MemoryPipelineBackend:
    """Create a MemoryPipelineBackend for the given graph."""
    hg_client = HugeGraphMemoryClient(graph=graph_name)
    return MemoryPipelineBackend(hg_client=hg_client)


def cmd_add(args: argparse.Namespace) -> int:
    """Add a memory."""
    backend = _get_backend(args.graph_name)
    result = backend.add_memory(
        content=args.content,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("action") in ("ADD", "UPDATE") or "memory_id" in result else 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search memories."""
    backend = _get_backend(args.graph_name)
    result = backend.search_memory(
        query=args.query,
        user_id=args.user_id,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List memories for a user."""
    backend = _get_backend(args.graph_name)
    memories = backend.list_memories(user_id=args.user_id)
    print(json.dumps(memories, ensure_ascii=False, indent=2))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a memory by id."""
    backend = _get_backend(args.graph_name)
    result = backend.delete_memory(memory_id=args.memory_id, user_id=args.user_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 1


def cmd_update(args: argparse.Namespace) -> int:
    """Update a memory by id."""
    backend = _get_backend(args.graph_name)
    result = backend.update_memory(
        memory_id=args.memory_id,
        content=args.content,
        user_id=args.user_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Show memory backend stats."""
    backend = _get_backend(args.graph_name)
    stats = backend.get_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_server(args: argparse.Namespace) -> int:
    """Start the dashboard server."""
    server_main = _load_server_main()
    sys.argv = ["memory_server.py", "--port", str(args.port), "--graph-name", args.graph_name]
    server_main()
    return 0


def cmd_locomo(args: argparse.Namespace) -> int:
    """Run the LOCOMO benchmark."""
    # Import here to avoid heavy dependency on normal CLI usage.
    import importlib.util
    spec = importlib.util.spec_from_file_location("locomo", args.script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["locomo"] = mod
    spec.loader.exec_module(mod)

    result = mod.run_locomo(
        data_dir=args.data_dir,
        max_sessions=args.max_sessions,
        sample_qa=args.sample,
        batch_size=args.batch_size,
        graph_name=args.graph_name,
        fast_eval=args.fast_eval,
        workers=args.workers,
        extraction_workers=args.extraction_workers,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    metrics = result["metrics"]
    print("\n=== LOCOMO Benchmark Results ===")
    print(f"Sessions: {metrics['sessions']}")
    print(f"Questions: {metrics['total_questions']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Hit@5: {metrics['hit_at_5']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Avg latency: {metrics['avg_latency_ms']} ms")
    print(f"P95 latency: {metrics['p95_latency_ms']} ms")
    print(f"Token estimate: {metrics['token_estimate']}")
    print(f"Result written to {args.output}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Show audit log entries or stats."""
    logger = get_audit_logger()
    if args.stats:
        print(json.dumps(logger.get_stats(), ensure_ascii=False, indent=2))
        return 0
    events = logger.get_events(
        user_id=args.user_id,
        operation=args.operation,
        limit=args.limit,
        offset=args.offset,
    )
    print(json.dumps([e.to_dict() for e in events], ensure_ascii=False, indent=2))
    return 0


def cmd_rewrite(args: argparse.Namespace) -> int:
    """Rewrite a query for better retrieval."""
    backend = _get_backend(args.graph_name)
    result = backend.rewrite_query(query=args.query, user_id=args.user_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hgmem",
        description="HugeGraph-AI-Memory CLI",
    )
    parser.add_argument(
        "--graph-name",
        default=os.environ.get("HUGEGRAPH_GRAPH", "poc_memgraphrag"),
        help="HugeGraph graph name (default: poc_memgraphrag or HUGEGRAPH_GRAPH)",
    )
    parser.add_argument(
        "--user-id",
        default="demo_user",
        help="User/session identifier",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # add
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("content", help="Memory content to add")
    add_parser.add_argument("--agent-id", default=None, help="Agent ID")
    add_parser.add_argument("--run-id", default=None, help="Run ID")
    add_parser.set_defaults(func=cmd_add)

    # search
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.set_defaults(func=cmd_search)

    # list
    list_parser = subparsers.add_parser("list", help="List memories for a user")
    list_parser.set_defaults(func=cmd_list)

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete a memory by id")
    delete_parser.add_argument("memory_id", help="Memory ID")
    delete_parser.set_defaults(func=cmd_delete)

    # update
    update_parser = subparsers.add_parser("update", help="Update a memory by id")
    update_parser.add_argument("memory_id", help="Memory ID")
    update_parser.add_argument("content", help="New memory content")
    update_parser.set_defaults(func=cmd_update)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show memory backend stats")
    stats_parser.set_defaults(func=cmd_stats)

    # server
    server_parser = subparsers.add_parser("server", help="Start the dashboard server")
    server_parser.add_argument("--port", type=int, default=8765, help="Server port")
    server_parser.set_defaults(func=cmd_server)

    # locomo
    locomo_parser = subparsers.add_parser("locomo", help="Run the LOCOMO benchmark")
    locomo_parser.add_argument("--data-dir", default="./locomo_data", help="LOCOMO data directory")
    locomo_parser.add_argument("--max-sessions", type=int, default=None, help="Max sessions")
    locomo_parser.add_argument("--sample", type=int, default=None, help="Max QA per session")
    locomo_parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    locomo_parser.add_argument("--fast-eval", action="store_true", help="Retrieval-only fast eval")
    locomo_parser.add_argument("--workers", type=int, default=8, help="QA parallel workers")
    locomo_parser.add_argument("--extraction-workers", type=int, default=8, help="Extraction threads")
    locomo_parser.add_argument(
        "--script-path",
        default=os.path.join(os.path.dirname(__file__), "..", "tests", "locomo_benchmark.py"),
        help="Path to locomo_benchmark.py",
    )
    locomo_parser.add_argument("--output", default="locomo_result.json", help="Result output path")
    locomo_parser.set_defaults(func=cmd_locomo)

    # rewrite
    rewrite_parser = subparsers.add_parser("rewrite", help="Rewrite a query for retrieval")
    rewrite_parser.add_argument("query", help="Query string")
    rewrite_parser.set_defaults(func=cmd_rewrite)

    # audit
    audit_parser = subparsers.add_parser("audit", help="Show audit log")
    audit_parser.add_argument("--stats", action="store_true", help="Show statistics")
    audit_parser.add_argument("--operation", default=None, help="Filter by operation")
    audit_parser.add_argument("--limit", type=int, default=100, help="Limit")
    audit_parser.add_argument("--offset", type=int, default=0, help="Offset")
    audit_parser.set_defaults(func=cmd_audit)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the hgmem CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
