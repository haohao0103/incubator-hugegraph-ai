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

"""Deterministic loader: ``SKILL.md`` files to :class:`SkillGraph`.

The loader is intentionally *deterministic* and does not call any LLM. It
parses the YAML front-matter of every ``SKILL.md`` found under a directory,
extracts the declared relations (``depends_on`` / ``composes_with`` /
``similar_to`` / ``belongs_to`` / ``conflicts_with``) and materialises the
nodes and explicit edges.

Following Graph-of-Skills, edges are the *high-confidence* part of the graph,
so they are only taken from explicit declarations here. Heuristic/similarity
edges are induced separately by :mod:`graph_builder`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hugegraph_llm.engines.skill_graph.schema import (
    EdgeType,
    NodeType,
    SkillEdge,
    SkillGraph,
    SkillNode,
    category_id,
    skill_id,
)

log = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)$", re.DOTALL)

# Aliases accepted in front-matter `relations:` blocks, mapped to canonical edges.
_RELATION_ALIASES: Dict[str, EdgeType] = {
    "depends_on": EdgeType.DEPENDS_ON,
    "dependencies": EdgeType.DEPENDS_ON,
    "requires": EdgeType.DEPENDS_ON,
    "prerequisites": EdgeType.DEPENDS_ON,
    "composes_with": EdgeType.COMPOSES_WITH,
    "compose_with": EdgeType.COMPOSES_WITH,
    "similar_to": EdgeType.SIMILAR_TO,
    "similar": EdgeType.SIMILAR_TO,
    "alternative_to": EdgeType.SIMILAR_TO,
    "belongs_to": EdgeType.BELONGS_TO,
    "conflicts_with": EdgeType.CONFLICTS_WITH,
}


def _scalar(value: str) -> Any:
    """Coerce a YAML scalar string to a Python value (minimal parser helper)."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~", ""):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    return v


def _minimal_yaml(raw: str) -> Dict[str, Any]:
    """Best-effort parser for the small YAML subset used in front-matter.

    Handles nested mappings by indentation, inline/quoted scalars, folded (``>``)
    and literal (``|``) block scalars, and flat ``- item`` lists. Used only when
    PyYAML is not importable.
    """
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]
    lines = raw.splitlines()
    idx = 0
    n = len(lines)
    while idx < n:
        raw_line = lines[idx]
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            idx += 1
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if content.startswith("- "):
            if isinstance(container, list):
                container.append(_scalar(content[2:].strip()))
            idx += 1
            continue

        key, sep, value = content.partition(":")
        if not sep:
            idx += 1
            continue
        key = key.strip()
        value = value.strip()

        if value in (">", "|", ">-", "|-"):
            block_lines: List[str] = []
            idx += 1
            while idx < n:
                nxt = lines[idx]
                if not nxt.strip():
                    block_lines.append("")
                    idx += 1
                    continue
                if (len(nxt) - len(nxt.lstrip(" "))) <= indent:
                    break
                block_lines.append(nxt.strip())
                idx += 1
            if value.startswith("|"):
                container[key] = "\n".join(block_lines)
            else:
                container[key] = " ".join(line for line in block_lines if line)
            continue

        if value == "":
            look = idx + 1
            while look < n and (not lines[look].strip() or lines[look].lstrip().startswith("#")):
                look += 1
            new: Any = []
            if look < n and not lines[look].strip().startswith("- "):
                new = {}
            container[key] = new
            stack.append((indent, new))
            idx += 1
            continue

        container[key] = _scalar(value)
        idx += 1
    return root


def parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a markdown document into (front-matter dict, body)."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    raw, body = match.group(1), match.group(2)
    try:
        import yaml

        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            data = {}
    except ImportError:
        data = _minimal_yaml(raw)
    except Exception as exc:  # malformed YAML: fall back rather than crash
        log.warning("front-matter YAML parse failed, using minimal parser: %s", exc)
        data = _minimal_yaml(raw)
    return data, body


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[,\n]", value) if p.strip()]
    return [str(value)]


def _extract_tags(metadata: Dict[str, Any]) -> List[str]:
    raw = metadata.get("tags")
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return []


def load_skill_file(path: Path) -> Tuple[SkillNode, List[SkillEdge]]:
    """Parse a single ``SKILL.md`` into a node plus its explicit edges."""
    text = path.read_text(encoding="utf-8")
    data, body = parse_front_matter(text)

    # Fallback name: the parent directory (Anthropic skill convention).
    fallback_name = path.parent.name if path.name.upper() == "SKILL.MD" else path.stem
    name = str(data.get("name") or fallback_name).strip()
    description = str(data.get("description") or "").strip()
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    node = SkillNode(
        id=skill_id(name),
        name=name,
        description=description,
        category=str(metadata.get("category") or "").strip(),
        tags=_extract_tags(metadata),
        node_type=NodeType.SKILL,
        source_path=str(path),
        body=body.strip(),
        metadata={k: v for k, v in metadata.items() if k not in ("category", "tags")},
    )

    edges: List[SkillEdge] = []
    relations = data.get("relations") or {}
    if not isinstance(relations, dict):
        relations = {}
    # Also accept relation keys declared at the top level of the front-matter.
    for key, value in data.items():
        if key in _RELATION_ALIASES and key not in relations:
            relations[key] = value

    for key, targets in relations.items():
        edge_type = _RELATION_ALIASES.get(str(key).strip().lower())
        if edge_type is None:
            continue
        for target in _as_list(targets):
            tgt = category_id(target) if edge_type == EdgeType.BELONGS_TO else skill_id(target)
            if tgt == node.id:
                continue
            edges.append(
                SkillEdge(
                    source=node.id,
                    target=tgt,
                    edge_type=edge_type,
                    weight=1.0,
                    provenance="explicit",
                )
            )
    return node, edges


def load_skills(skills_dir: str) -> SkillGraph:
    """Load every ``SKILL.md`` under ``skills_dir`` into a :class:`SkillGraph`.

    Category nodes are materialised automatically for every distinct
    ``metadata.category``, together with the corresponding ``belongs_to`` edges.
    """
    root = Path(skills_dir)
    if not root.exists():
        raise FileNotFoundError(f"skills directory not found: {skills_dir}")

    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name.upper() == "SKILL.MD")
    nodes: List[SkillNode] = []
    edges: List[SkillEdge] = []
    for path in paths:
        node, node_edges = load_skill_file(path)
        nodes.append(node)
        edges.extend(node_edges)
    if not nodes:
        raise ValueError(f"no SKILL.md files found under {skills_dir}")

    # Materialise category nodes and belongs_to edges from metadata.category.
    categories: Dict[str, str] = {}
    for node in nodes:
        if node.category:
            categories[category_id(node.category)] = node.category
    cat_nodes = [
        SkillNode(
            id=cid,
            name=name,
            description=f"Skill category: {name}",
            node_type=NodeType.CATEGORY,
        )
        for cid, name in categories.items()
    ]
    for node in nodes:
        if node.category:
            edges.append(
                SkillEdge(
                    source=node.id,
                    target=category_id(node.category),
                    edge_type=EdgeType.BELONGS_TO,
                    weight=1.0,
                    provenance="explicit",
                )
            )

    graph = SkillGraph(nodes=nodes + cat_nodes)
    for edge in edges:
        graph.add_edge(edge)
    log.info("loaded %d skills and %d categories from %s", len(nodes), len(cat_nodes), skills_dir)
    return graph