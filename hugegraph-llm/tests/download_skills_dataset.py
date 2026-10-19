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

"""Download a public skill corpus for the Skill Graph engine.

The Skill Graph loader consumes ``SKILL.md`` files (Anthropic skill convention:
YAML front-matter + markdown body). This script materialises such a corpus on
disk so ``hgmem skill build --skills-dir <dir>`` can run immediately.

Sources
    anthro   Public ``anthropics/skills`` repository (real ``SKILL.md`` files).
    builtin  A small, self-contained corpus emitted locally (offline fallback).

Examples
    python download_skills_dataset.py --source anthro --output ./skills
    python download_skills_dataset.py --source builtin --output ./skills --limit 10
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from typing import List, Optional

ANTHRO_ZIP = "https://codeload.github.com/anthropics/skills/zip/refs/heads/main"


# ── builtin offline corpus ───────────────────────────────────
# (name, description, category, tags, depends_on)
_BUILTIN: List[tuple] = [
    ("data-ingestion", "Load and normalize raw documents into the pipeline", "data",
     "data,ingestion,loader", []),
    ("text-chunking", "Split documents into retrieval-friendly chunks", "data",
     "chunking,splitter", ["data-ingestion"]),
    ("text-embedding", "Turn text into dense vectors for semantic search", "retrieval",
     "embedding,vectors", ["text-chunking"]),
    ("vector-index", "Build and query an approximate nearest-neighbour index", "retrieval",
     "faiss,ann,index", ["text-embedding"]),
    ("reranking", "Re-order candidate results with a cross-encoder", "retrieval",
     "rerank,cross-encoder", ["vector-index"]),
    ("prompt-engineering", "Author and iterate on robust LLM prompts", "generation",
     "prompt,llm", []),
    ("rag-pipeline", "Compose retrieval and generation into an end-to-end RAG flow", "generation",
     "rag,pipeline", ["vector-index", "reranking", "prompt-engineering"]),
    ("evaluation", "Measure retrieval and generation quality with benchmarks", "quality",
     "eval,benchmark", ["rag-pipeline"]),
    ("cypher-query", "Write graph queries over the knowledge graph", "graph",
     "graph,query", []),
    ("knowledge-graph-build", "Extract entities and relations into a knowledge graph", "graph",
     "kg,extraction", ["cypher-query", "data-ingestion"]),
]


def _builtin_corpus(limit: Optional[int] = None) -> dict:
    """Return ``{relative_path: content}`` for the offline corpus."""
    names = {n for (n, *_rest) in _BUILTIN}
    files: dict = {}
    for name, desc, category, tags, deps in _BUILTIN[: limit or len(_BUILTIN)]:
        deps = [d for d in deps if d in names]
        dep_block = ""
        if deps:
            dep_lines = "\n".join(f"    - {d}" for d in deps)
            dep_block = f"relations:\n  depends_on:\n{dep_lines}\n"
        content = (
            "---\n"
            f"name: {name}\n"
            f"description: {desc}\n"
            "metadata:\n"
            f"  category: {category}\n"
            f"  tags: {tags}\n"
            f"{dep_block}"
            "---\n\n"
            f"# {name}\n\n{desc}.\n\n"
            "## Steps\n\n"
            "1. Understand the inputs and expected outputs.\n"
            "2. Apply the skill to the current task.\n"
            "3. Validate the result before handing it off.\n"
        )
        files[f"{name}/SKILL.md"] = content
    return files


# ── remote corpora ───────────────────────────────────────────
def _download_anthropic(url: str = ANTHRO_ZIP) -> dict:
    """Download the anthropics/skills repo zip and extract ``SKILL.md`` files."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "hugegraph-ai-skillgraph"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
        payload = resp.read()
    files: dict = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for info in zf.namelist():
            base = info.split("/", 1)[-1]  # strip the top-level "repo-ref/" prefix
            if base.upper().endswith("SKILL.MD") and not info.endswith("/"):
                files[base] = zf.read(info).decode("utf-8", errors="replace")
    return files


def _write(files: dict, output: str) -> int:
    written = 0
    for rel, content in files.items():
        dest = os.path.join(output, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)
        written += 1
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download a public skill corpus")
    parser.add_argument("--source", choices=["anthro", "builtin"], default="builtin")
    parser.add_argument("--output", default="./skills", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max skills to write")
    parser.add_argument("--url", default=ANTHRO_ZIP, help="Override the source zip URL")
    args = parser.parse_args(argv)

    if args.source == "anthro":
        try:
            files = _download_anthropic(args.url)
            print(f"[download] fetched {len(files)} SKILL.md files from {args.url}", file=sys.stderr)
        except Exception as exc:  # network unavailable -> fall back gracefully
            print(f"[download] remote fetch failed ({exc}); falling back to builtin corpus", file=sys.stderr)
            files = _builtin_corpus(args.limit)
    else:
        files = _builtin_corpus(args.limit)

    if args.limit:
        items = list(files.items())[: args.limit]
        files = dict(items)

    if not files:
        print("[download] no SKILL.md files found", file=sys.stderr)
        return 1

    written = _write(files, args.output)
    print(f"[download] wrote {written} SKILL.md files to {args.output}")
    print(f"Next: hgmem skill build --skills-dir {args.output} --out skill_graph.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())